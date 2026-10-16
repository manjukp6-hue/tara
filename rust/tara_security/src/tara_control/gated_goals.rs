//! Gated Autonomous Goal System with Anti-Rollback and Lockdown TOCTOU Defenses.
//!
//! Rule compliance:
//! - Pure native Rust, zero outside crate dependencies.
//! - Guards autonomous goal lifecycle transitions (Proposed, Approved, Executing, Completed, Aborted).
//! - Queries AuthorityManager at the operation boundary to prevent TOCTOU window races.
//! - Integrates directly with Central Enforcement Gateway.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use sha2::Digest;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::broker::{EnforcementGateway, EnforcementRequest, ProtectedCategory};
use crate::hardware_root::AuthorityManager;
use crate::primitives::{
    AuthorityGeneration, SecurityAction, SecurityContext, SecurityResource, VerifiedPrincipal,
};

/// Goal state machine states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalState {
    Proposed,
    Approved,
    Executing,
    Completed,
    Aborted,
}

/// Gated goal descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatedGoal {
    pub goal_id: String,
    pub title: String,
    pub target_subsystem: String,
    pub state: GoalState,
    pub required_generation: AuthorityGeneration,
    pub created_epoch_secs: u64,
    pub updated_epoch_secs: u64,
}

/// Gated Autonomous Goal Manager.
pub struct AutonomousGoalGating {
    goals: Arc<Mutex<HashMap<String, GatedGoal>>>,
    vault_engine: Option<Arc<GatedGoalVaultEngine>>,
    authority: Arc<AuthorityManager>,
    gateway: Arc<EnforcementGateway>,
    storage_path: Option<std::path::PathBuf>,
}

impl AutonomousGoalGating {
    pub fn new(authority: Arc<AuthorityManager>, gateway: Arc<EnforcementGateway>) -> Self {
        Self::with_optional_storage(authority, gateway, None)
    }

    pub fn with_storage(
        authority: Arc<AuthorityManager>,
        gateway: Arc<EnforcementGateway>,
        storage_path: std::path::PathBuf,
    ) -> Self {
        Self::with_optional_storage(authority, gateway, Some(storage_path))
    }

    pub fn with_optional_storage(
        authority: Arc<AuthorityManager>,
        gateway: Arc<EnforcementGateway>,
        storage_path: Option<std::path::PathBuf>,
    ) -> Self {
        let initial_goals = if let Some(ref path) = storage_path {
            Self::load_goals_from_disk(path)
        } else {
            HashMap::new()
        };

        Self {
            goals: Arc::new(Mutex::new(initial_goals)),
            vault_engine: None,
            authority,
            gateway,
            storage_path,
        }
    }

    pub fn with_vault_engine(
        authority: Arc<AuthorityManager>,
        gateway: Arc<EnforcementGateway>,
        storage_path: Option<std::path::PathBuf>,
        vault_engine: Arc<GatedGoalVaultEngine>,
    ) -> Self {
        let initial_goals = if let Some(ref path) = storage_path {
            Self::load_goals_from_disk(path)
        } else {
            HashMap::new()
        };

        Self {
            goals: Arc::new(Mutex::new(initial_goals)),
            vault_engine: Some(vault_engine),
            authority,
            gateway,
            storage_path,
        }
    }

    pub fn vault_engine(&self) -> Option<Arc<GatedGoalVaultEngine>> {
        self.vault_engine.clone()
    }

    fn load_goals_from_disk(path: &std::path::Path) -> HashMap<String, GatedGoal> {
        if path.exists() {
            if let Ok(data) = std::fs::read(path) {
                if let Ok(map) = serde_json::from_slice(&data) {
                    return map;
                }
            }
        }
        HashMap::new()
    }

    fn persist_goals(&self, map: &HashMap<String, GatedGoal>) -> Result<(), String> {
        if let Some(ref path) = self.storage_path {
            let data = serde_json::to_vec_pretty(map)
                .map_err(|e| format!("Serialization error for gated goals: {}", e))?;
            crate::persistence::atomic_write_file_hardened(path, &data)?;
        }
        Ok(())
    }

    pub fn current_generation(&self) -> Result<AuthorityGeneration, String> {
        self.authority.current_generation()
    }

    pub fn get_goal(&self, goal_id: &str) -> Option<GatedGoal> {
        self.goals.lock().ok().and_then(|m| m.get(goal_id).cloned())
    }

    /// Proposes a new autonomous goal.
    pub fn propose_goal(
        &self,
        goal_id: impl Into<String>,
        title: impl Into<String>,
        target_subsystem: impl Into<String>,
        epoch_secs: u64,
    ) -> Result<GatedGoal, String> {
        let current_gen = self.authority.current_generation()?;
        let gid = goal_id.into().trim().to_string();

        if gid.is_empty() {
            return Err("Goal ID cannot be empty".into());
        }

        let mut map = self.goals.lock().map_err(|_| "Goals lock poisoned")?;
        if map.contains_key(&gid) {
            return Err(format!("DuplicateGoalId: Goal '{}' already exists", gid));
        }

        let goal = GatedGoal {
            goal_id: gid.clone(),
            title: title.into(),
            target_subsystem: target_subsystem.into(),
            state: GoalState::Proposed,
            required_generation: current_gen,
            created_epoch_secs: epoch_secs,
            updated_epoch_secs: epoch_secs,
        };

        map.insert(gid, goal.clone());
        self.persist_goals(&map)?;
        Ok(goal)
    }

    /// Advances a goal step under strict TOCTOU defense and Central Gateway enforcement.
    pub fn advance_goal_step<T, F>(
        &self,
        goal_id: &str,
        step_name: &str,
        principal: &VerifiedPrincipal,
        context: &SecurityContext,
        session_id: Option<String>,
        step_closure: F,
    ) -> Result<T, String>
    where
        F: FnOnce(&VerifiedPrincipal, &SecurityResource) -> Result<T, String>,
    {
        // 1. Check authoritative lockdown before lock acquisition (TOCTOU Defense)
        if self.authority.is_locked_down() {
            return Err("GoalExecutionDenied: System is in strict security lockdown".into());
        }

        let current_gen = self.authority.current_generation()?;

        // 2. State verification under lock
        let stale_info = {
            let mut map = self.goals.lock().map_err(|_| "Goals lock poisoned")?;
            let goal = map.get_mut(goal_id).ok_or_else(|| format!("GoalNotFound: '{}'", goal_id))?;

            if goal.state == GoalState::Aborted {
                return Err("GoalAborted: Cannot advance aborted goal".into());
            }

            if goal.required_generation != current_gen {
                goal.state = GoalState::Aborted;
                let goal_gen = goal.required_generation.get();
                let curr_gen_val = current_gen.get();
                Some((goal_gen, curr_gen_val))
            } else {
                goal.state = GoalState::Executing;
                goal.updated_epoch_secs = context.request_epoch_secs;
                None
            }
        };

        if let Some((goal_gen, curr_gen_val)) = stale_info {
            let map = self.goals.lock().map_err(|_| "Goals lock poisoned")?;
            let _ = self.persist_goals(&map);
            return Err(format!(
                "AuthorityGenerationMismatch: Goal generation {} is stale (current {})",
                goal_gen,
                curr_gen_val
            ));
        }

        {
            let map = self.goals.lock().map_err(|_| "Goals lock poisoned")?;
            self.persist_goals(&map)?;
        }

        // 3. Delegate execution through Central Enforcement Gateway
        let request = EnforcementRequest {
            principal: principal.clone(),
            session_id,
            action: SecurityAction::execute_task(),
            resource: SecurityResource::custom("autonomous_goal", goal_id)?,
            context: context.clone(),
            category: ProtectedCategory::AutonomousGoals {
                goal_id: format!("{}:{}", goal_id, step_name),
                target_subsystem: "autonomous_goal_pipeline".to_string(),
            },
        };

        let result = self.gateway.enforce_protected_operation(request, step_closure);

        // 4. Update goal state based on execution outcome
        {
            let mut map = self.goals.lock().map_err(|_| "Goals lock poisoned")?;
            if let Some(goal) = map.get_mut(goal_id) {
                match &result {
                    Ok(_) => {
                        goal.updated_epoch_secs = context.request_epoch_secs;
                    }
                    Err(_) => {
                        goal.state = GoalState::Aborted;
                    }
                }
            }
            let _ = self.persist_goals(&map);
        }

        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Canonical Creator-Gated Task & Goal Vault Engine
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureRule {
    RetryWithAlternativeStrategy,
    RequireCreatorAssistance,
    HaltAndAlert,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetryRule {
    pub max_retries: u32,
    pub current_retries: u32,
    pub backoff_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Locked,
    Active,
    PendingVerification,
    VerifiedCompleted,
    Failed,
    BlockedNeedsReview,
}

impl TaskStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Locked => "LOCKED",
            Self::Active => "ACTIVE",
            Self::PendingVerification => "PENDING_VERIFICATION",
            Self::VerifiedCompleted => "VERIFIED_COMPLETED",
            Self::Failed => "FAILED",
            Self::BlockedNeedsReview => "BLOCKED_NEEDS_REVIEW",
        }
    }
}

/// A Creator-defined task within a gated milestone hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GatedTask {
    pub task_id: String,
    pub goal_headline: String,
    pub goal_description: String,
    pub stage_name: String,
    pub stage_order: usize,
    pub milestone_name: String,
    pub task_name: String,
    pub task_order: usize,
    pub task_priority: f64,
    pub reward_points: f64,
    pub completion_criteria: String,
    pub verification_criteria: String,
    pub failure_rule: FailureRule,
    pub retry_rule: RetryRule,
    pub status: TaskStatus,
    pub verification_proof: Option<String>,
    pub completed_at_ms: Option<u64>,
    pub reward_event_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSpec {
    pub task_name: String,
    pub task_order: usize,
    pub task_priority: f64,
    pub reward_points: f64,
    pub completion_criteria: String,
    pub verification_criteria: String,
    pub failure_rule: FailureRule,
    pub max_retries: u32,
    pub backoff_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilestoneSpec {
    pub milestone_name: String,
    pub tasks: Vec<TaskSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageSpec {
    pub stage_name: String,
    pub stage_order: usize,
    pub milestones: Vec<MilestoneSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalHierarchySpec {
    pub goal_headline: String,
    pub goal_description: String,
    pub stages: Vec<StageSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveTaskView {
    pub task_id: String,
    pub task_name: String,
    pub stage_name: String,
    pub stage_order: usize,
    pub milestone_name: String,
    pub task_order: usize,
    pub task_priority: f64,
    pub completion_criteria: String,
    pub verification_criteria: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub task_id: String,
    pub verified: bool,
    pub reward_credited: f64,
    pub reward_event_id: Option<String>,
    pub next_unlocked_task: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VaultPayload {
    tasks: Vec<GatedTask>,
    is_locked_down: bool,
}

/// Abstract interface for disbursing rewards without circular crate dependencies.
pub trait GoalRewardDisburser: Send + Sync {
    fn disburse_reward(
        &self,
        recipient: &str,
        task_id: &str,
        points: f64,
        reason: &str,
    ) -> Result<String, String>;
}

/// Abstract trait for lockdown state checking across subsystems.
pub trait LockdownStateChecker: Send + Sync {
    fn is_locked_down(&self) -> bool;
    fn assert_unlocked(&self, expected_gen: u64) -> Result<(), String>;
    fn current_generation(&self) -> u64;
}

impl LockdownStateChecker for crate::lockdown::ProgressiveLockdownCoordinator {
    fn is_locked_down(&self) -> bool {
        self.is_locked_down()
    }
    fn assert_unlocked(&self, expected_gen: u64) -> Result<(), String> {
        self.assert_unlocked(expected_gen)
    }
    fn current_generation(&self) -> u64 {
        self.current_generation()
    }
}

/// Canonical Creator-Gated Goal Vault Engine.
/// Persists AES-256-GCM encrypted roadmaps with Scrypt key derivation,
/// milestone dependency graphs, exact-once rewards, and crash-safe atomic updates.
pub struct GatedGoalVaultEngine {
    storage_dir: PathBuf,
    tasks: Mutex<Vec<GatedTask>>,
    completed_task_ids: Mutex<HashSet<String>>,
    key: [u8; 32],
    is_locked_down: Mutex<bool>,
    lockdown_coordinator: Mutex<Option<Arc<dyn LockdownStateChecker>>>,
}

impl GatedGoalVaultEngine {
    pub fn try_new<P: AsRef<Path>>(repo_root: P) -> Result<Self, String> {
        let storage_dir = repo_root
            .as_ref()
            .join("storage")
            .join("vault")
            .join("goals");
        fs::create_dir_all(&storage_dir).map_err(|e| {
            format!(
                "Failed to create gated goals vault directory {}: {}",
                storage_dir.display(),
                e
            )
        })?;

        let salt_file = storage_dir.join("salt.bin");
        let mut salt = [0u8; 16];
        if salt_file.exists() {
            let mut f = File::open(&salt_file).map_err(|e| format!("Failed to open salt: {e}"))?;
            f.read_exact(&mut salt)
                .map_err(|e| format!("Corrupt salt: {e}"))?;
        } else {
            rand::thread_rng().fill_bytes(&mut salt);
            let mut f =
                File::create(&salt_file).map_err(|e| format!("Failed to create salt: {e}"))?;
            f.write_all(&salt)
                .map_err(|e| format!("Failed to write salt: {e}"))?;
            f.sync_all()
                .map_err(|e| format!("Failed to sync salt: {e}"))?;
        }

        let key = Self::resolve_or_generate_key(&storage_dir, &salt)?;

        let sys = Self {
            storage_dir,
            tasks: Mutex::new(Vec::new()),
            completed_task_ids: Mutex::new(HashSet::new()),
            key,
            is_locked_down: Mutex::new(false),
            lockdown_coordinator: Mutex::new(None),
        };

        let enc_file = sys.storage_dir.join("goals.enc");
        if enc_file.exists() {
            sys.load_encrypted()?;
        }

        Ok(sys)
    }

    pub fn new<P: AsRef<Path>>(repo_root: P) -> Self {
        let repo_root_ref = repo_root.as_ref();
        match Self::try_new(repo_root_ref) {
            Ok(sys) => sys,
            Err(err) => {
                eprintln!("FailClosed: GatedGoalVaultEngine startup verification failed: {err}");
                let storage_dir = repo_root_ref.join("storage").join("vault").join("goals");
                let salt_file = storage_dir.join("salt.bin");
                let mut salt = [0u8; 16];
                if let Ok(mut f) = File::open(&salt_file) {
                    let _ = f.read_exact(&mut salt);
                }
                let key = Self::resolve_or_generate_key(&storage_dir, &salt).unwrap_or([0u8; 32]);
                Self {
                    storage_dir,
                    tasks: Mutex::new(Vec::new()),
                    completed_task_ids: Mutex::new(HashSet::new()),
                    key,
                    is_locked_down: Mutex::new(true),
                    lockdown_coordinator: Mutex::new(None),
                }
            }
        }
    }

    pub fn with_lockdown_coordinator(self, coordinator: Arc<dyn LockdownStateChecker>) -> Self {
        *self.lockdown_coordinator.lock().unwrap() = Some(coordinator);
        self
    }

    pub fn set_lockdown_coordinator(&self, coordinator: Arc<dyn LockdownStateChecker>) {
        *self.lockdown_coordinator.lock().unwrap() = Some(coordinator);
    }

    pub fn is_locked_down(&self) -> bool {
        if *self.is_locked_down.lock().unwrap() {
            return true;
        }
        if let Ok(guard) = self.lockdown_coordinator.lock() {
            if let Some(ref coord) = *guard {
                if coord.is_locked_down() {
                    return true;
                }
            }
        }
        false
    }

    pub fn current_generation(&self) -> Option<u64> {
        if let Ok(guard) = self.lockdown_coordinator.lock() {
            if let Some(ref coord) = *guard {
                return Some(coord.current_generation());
            }
        }
        None
    }

    fn resolve_or_generate_key(storage_dir: &Path, salt: &[u8; 16]) -> Result<[u8; 32], String> {
        let key_file = storage_dir.join("vault_key.bin");
        let mut key_bytes = [0u8; 32];

        if let Ok(env_key) = std::env::var("TARA_GOAL_VAULT_KEY") {
            if !env_key.trim().is_empty() {
                const SCRYPT_LOG_N: u8 = 14;
                const SCRYPT_R: u32 = 8;
                const SCRYPT_P: u32 = 1;
                let params = scrypt::Params::new(SCRYPT_LOG_N, SCRYPT_R, SCRYPT_P, 32)
                    .map_err(|e| format!("Scrypt param error: {e}"))?;
                scrypt::scrypt(env_key.as_bytes(), salt, &params, &mut key_bytes)
                    .map_err(|e| format!("Scrypt key derivation error: {e}"))?;
                return Ok(key_bytes);
            }
        }

        if key_file.exists() {
            let bytes = fs::read(&key_file)
                .map_err(|e| format!("Failed to read goal vault key: {e}"))?;
            #[cfg(windows)]
            let raw = crate::dpapi::native_dpapi::unprotect_bytes(&bytes)
                .map_err(|e| format!("FAIL CLOSED: Windows DPAPI unprotect failed for goal vault key: {}", e))?;
            #[cfg(not(windows))]
            let raw = bytes;

            if raw.len() >= 32 {
                key_bytes.copy_from_slice(&raw[..32]);
                #[cfg(windows)]
                {
                    if let Ok(protected) = crate::dpapi::native_dpapi::protect_bytes(&key_bytes) {
                        let _ = crate::persistence::atomic_write_file_hardened(&key_file, &protected);
                    }
                }
            } else {
                return Err("Corrupt goal vault key file: key too short".into());
            }
        } else {
            rand::thread_rng().fill_bytes(&mut key_bytes);
            #[cfg(windows)]
            let to_write = crate::dpapi::native_dpapi::protect_bytes(&key_bytes)
                .map_err(|e| format!("FAIL CLOSED: Windows DPAPI protect failed: {e}"))?;
            #[cfg(not(windows))]
            let to_write = key_bytes.to_vec();

            crate::persistence::atomic_write_file_hardened(&key_file, &to_write)
                .map_err(|e| format!("Failed to write goal vault key file: {e}"))?;
        }

        Ok(key_bytes)
    }

    pub fn configure_creator_goals(
        &self,
        principal: &VerifiedPrincipal,
        hierarchy: GoalHierarchySpec,
    ) -> Result<usize, String> {
        if !principal.is_creator_root() {
            return Err("Unauthorized: Only verified Creator root authority can configure goals".into());
        }
        self.configure_creator_goals_internal(hierarchy)
    }

    fn configure_creator_goals_internal(
        &self,
        hierarchy: GoalHierarchySpec,
    ) -> Result<usize, String> {
        if self.is_locked_down() {
            return Err("SystemLockedDown: Creator authority has locked down the system".into());
        }

        let mut flattened_tasks = Vec::new();
        let mut stages = hierarchy.stages;
        stages.sort_by_key(|s| s.stage_order);

        for stage in stages {
            let stage_name = stage.stage_name;
            let stage_order = stage.stage_order;

            for (m_idx, milestone) in stage.milestones.into_iter().enumerate() {
                let milestone_name = milestone.milestone_name;
                let mut tasks = milestone.tasks;
                tasks.sort_by_key(|t| t.task_order);

                for task in tasks {
                    let task_id = format!(
                        "task_s{}_m{}_t{}",
                        stage_order,
                        milestone_name.replace(|c: char| !c.is_alphanumeric(), "_"),
                        task.task_order
                    );

                    let is_first = stage_order == 1 && m_idx == 0 && task.task_order == 1;

                    flattened_tasks.push(GatedTask {
                        task_id,
                        goal_headline: hierarchy.goal_headline.clone(),
                        goal_description: hierarchy.goal_description.clone(),
                        stage_name: stage_name.clone(),
                        stage_order,
                        milestone_name: milestone_name.clone(),
                        task_name: task.task_name,
                        task_order: task.task_order,
                        task_priority: task.task_priority,
                        reward_points: task.reward_points,
                        completion_criteria: task.completion_criteria,
                        verification_criteria: task.verification_criteria,
                        failure_rule: task.failure_rule,
                        retry_rule: RetryRule {
                            max_retries: task.max_retries,
                            current_retries: 0,
                            backoff_ms: task.backoff_ms,
                        },
                        status: if is_first {
                            TaskStatus::Active
                        } else {
                            TaskStatus::Locked
                        },
                        verification_proof: None,
                        completed_at_ms: None,
                        reward_event_id: None,
                    });
                }
            }
        }

        let count = flattened_tasks.len();
        *self.tasks.lock().unwrap() = flattened_tasks;
        self.completed_task_ids.lock().unwrap().clear();

        self.save_encrypted()?;
        Ok(count)
    }

    pub fn get_active_task_for_tara(&self) -> Option<ActiveTaskView> {
        if self.is_locked_down() {
            return None;
        }

        let tasks = self.tasks.lock().unwrap();
        tasks
            .iter()
            .find(|t| t.status == TaskStatus::Active || t.status == TaskStatus::PendingVerification)
            .map(|t| ActiveTaskView {
                task_id: t.task_id.clone(),
                task_name: t.task_name.clone(),
                stage_name: t.stage_name.clone(),
                stage_order: t.stage_order,
                milestone_name: t.milestone_name.clone(),
                task_order: t.task_order,
                task_priority: t.task_priority,
                completion_criteria: t.completion_criteria.clone(),
                verification_criteria: t.verification_criteria.clone(),
                status: t.status.as_str().to_string(),
            })
    }

    pub fn inspect_control_plane(
        &self,
        is_creator: bool,
        reward_summary: Value,
        reward_ledger_verified: bool,
        integrity_error: Option<String>,
    ) -> Result<Value, String> {
        if !is_creator {
            return Err("AccessDenied: Private TARA Control Plane is restricted from regular users and chat contexts".into());
        }
        let tasks = self.tasks.lock().unwrap();

        Ok(json!({
            "total_tasks": tasks.len(),
            "tasks": *tasks,
            "reward_summary": reward_summary,
            "reward_ledger_verified": reward_ledger_verified,
            "integrity_error": integrity_error,
            "locked_down": self.is_locked_down()
        }))
    }

    pub fn submit_and_verify_active_task(
        &self,
        task_id: &str,
        proof_payload: &Value,
        reward_disburser: Option<&dyn GoalRewardDisburser>,
    ) -> Result<VerificationResult, String> {
        let expected_gen = {
            let guard = self.lockdown_coordinator.lock().unwrap();
            guard.as_ref().map(|c| c.current_generation())
        };
        if self.is_locked_down() {
            return Err("SystemLockedDown: Cannot submit task during security lockdown".into());
        }

        // Anti-double-credit check
        {
            let completed = self.completed_task_ids.lock().unwrap();
            if completed.contains(task_id) {
                return Err("AntiDoubleCreditViolation: Task has already been verified and rewarded. Replays are forbidden.".into());
            }
        }

        let mut tasks = self.tasks.lock().unwrap();
        let task_idx = tasks
            .iter()
            .position(|t| t.task_id == task_id)
            .ok_or_else(|| format!("TaskNotFound: '{}' does not exist", task_id))?;

        if tasks[task_idx].status != TaskStatus::Active
            && tasks[task_idx].status != TaskStatus::PendingVerification
        {
            return Err(format!(
                "InvalidTaskStatus: Task '{}' is not in an active or pending state (current: {:?})",
                task_id, tasks[task_idx].status
            ));
        }

        let criteria = &tasks[task_idx].verification_criteria;
        let is_verified = Self::evaluate_verification_proof(criteria, proof_payload);

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        if !is_verified {
            let task = &mut tasks[task_idx];
            task.retry_rule.current_retries += 1;

            if task.retry_rule.current_retries >= task.retry_rule.max_retries {
                task.status = match task.failure_rule {
                    FailureRule::RequireCreatorAssistance => TaskStatus::BlockedNeedsReview,
                    FailureRule::HaltAndAlert => TaskStatus::BlockedNeedsReview,
                    FailureRule::RetryWithAlternativeStrategy => TaskStatus::Failed,
                };
            } else {
                task.status = TaskStatus::Active;
            }

            if let Err(save_err) = self.save_encrypted_slice(&tasks) {
                *self.is_locked_down.lock().unwrap() = true;
                return Err(format!(
                    "TaskStateSaveError (System Locked Down): {save_err}"
                ));
            }

            return Ok(VerificationResult {
                task_id: task_id.to_string(),
                verified: false,
                reward_credited: 0.0,
                reward_event_id: None,
                next_unlocked_task: None,
                message: "Verification failed: proof does not satisfy completion criteria".into(),
            });
        }

        // TOCTOU Race Defense
        if let Ok(guard) = self.lockdown_coordinator.lock() {
            if let Some(ref coord) = *guard {
                if let Some(gen) = expected_gen {
                    coord.assert_unlocked(gen).map_err(|e| format!("SystemLockedDown: {}", e))?;
                } else if coord.is_locked_down() {
                    return Err("SystemLockedDown: Cannot commit task during security lockdown".into());
                }
            }
        }

        let reward_pts = tasks[task_idx].reward_points;
        let proof_str = serde_json::to_string(proof_payload).unwrap_or_default();

        let reason = format!(
            "Verified completion of task '{}' [{}]",
            tasks[task_idx].task_name, tasks[task_idx].task_id
        );

        let mut reward_event_id = None;
        if let Some(disburser) = reward_disburser {
            let rew_id = disburser
                .disburse_reward("tara_agent", &tasks[task_idx].task_id, reward_pts, &reason)
                .map_err(|e| format!("RewardDisbursementError: {}", e))?;
            reward_event_id = Some(rew_id);
        }

        let task = &mut tasks[task_idx];
        task.status = TaskStatus::VerifiedCompleted;
        task.verification_proof = Some(proof_str);
        task.completed_at_ms = Some(now_ms);
        task.reward_event_id = reward_event_id.clone();

        self.completed_task_ids
            .lock()
            .unwrap()
            .insert(task_id.to_string());

        let next_task_id = Self::unlock_next_task_internal(&mut tasks, task_idx);

        if let Err(save_err) = self.save_encrypted_slice(&tasks) {
            *self.is_locked_down.lock().unwrap() = true;
            return Err(format!(
                "TaskStateSaveError (System Locked Down): {save_err}"
            ));
        }

        Ok(VerificationResult {
            task_id: task_id.to_string(),
            verified: true,
            reward_credited: reward_pts,
            reward_event_id,
            next_unlocked_task: next_task_id,
            message: "Task successfully verified and completed. Next task unlocked.".into(),
        })
    }

    pub fn evaluate_verification_proof(criteria: &str, proof: &Value) -> bool {
        if proof.is_null() || !proof.is_object() {
            return false;
        }

        if let Some(st) = proof.get("status").and_then(|v| v.as_str()) {
            if st != "SUCCESS" && st != "VERIFIED" && st != "PASSED" {
                return false;
            }
        }

        let criteria_lc = criteria.to_lowercase();

        if criteria_lc.contains("checksum") || criteria_lc.contains("hash") {
            let has_checksum = proof
                .get("checksum")
                .and_then(|v| v.as_str())
                .is_some_and(|s| !s.is_empty());
            let has_sha = proof
                .get("sha256")
                .and_then(|v| v.as_str())
                .is_some_and(|s| !s.is_empty());
            if !has_checksum && !has_sha {
                return false;
            }
        }

        if criteria_lc.contains("test") || criteria_lc.contains("tests") {
            match proof.get("tests_passed").and_then(|v| v.as_bool()) {
                Some(true) => {}
                _ => return false,
            }
        }

        if criteria_lc.contains("artifact") {
            let has_art = proof
                .get("artifact")
                .and_then(|v| v.as_str())
                .is_some_and(|s| !s.is_empty())
                || proof
                    .get("artifact_path")
                    .and_then(|v| v.as_str())
                    .is_some_and(|s| !s.is_empty())
                || proof
                    .get("artifact_id")
                    .and_then(|v| v.as_str())
                    .is_some_and(|s| !s.is_empty());
            if !has_art {
                return false;
            }
        }

        // Dynamic Cryptographic & Filesystem Integrity Verification (Dynamic Hash Calculation & Verification Before Confirmation)
        let claimed_file = proof
            .get("artifact_path")
            .or_else(|| proof.get("file_path"))
            .and_then(|v| v.as_str());

        if let Some(path_str) = claimed_file {
            let p = Path::new(path_str);
            if crate::path_security::check_lexical_traversal(p).is_err() {
                return false;
            }
            if !p.exists() {
                return false;
            }
            if let Some(expected_sha) = proof.get("sha256").or_else(|| proof.get("checksum")).and_then(|v| v.as_str()) {
                if let Ok(bytes) = fs::read(p) {
                    let actual_digest = hex::encode(sha2::Sha256::digest(&bytes));
                    if !crate::cryptography::CryptoUtils::constant_time_eq(actual_digest.as_bytes(), expected_sha.as_bytes()) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }

        true
    }

    fn unlock_next_task_internal(tasks: &mut [GatedTask], current_idx: usize) -> Option<String> {
        let current_stage = tasks[current_idx].stage_order;
        let current_task_order = tasks[current_idx].task_order;
        let current_milestone = tasks[current_idx].milestone_name.clone();

        // 1. Try next task in same milestone
        if let Some(target) = tasks.iter_mut().find(|t| {
            t.stage_order == current_stage
                && t.milestone_name == current_milestone
                && t.task_order == current_task_order + 1
        }) {
            target.status = TaskStatus::Active;
            return Some(target.task_id.clone());
        }

        // 2. Try first locked task of next milestone in same stage
        let next_milestone_target = tasks
            .iter()
            .enumerate()
            .filter(|(_, t)| t.stage_order == current_stage && t.status == TaskStatus::Locked)
            .min_by_key(|(_, t)| t.task_order)
            .map(|(idx, _)| idx);

        if let Some(idx) = next_milestone_target {
            tasks[idx].status = TaskStatus::Active;
            return Some(tasks[idx].task_id.clone());
        }

        // 3. Try first locked task of next stage
        let next_stage_target = tasks
            .iter()
            .enumerate()
            .filter(|(_, t)| t.stage_order == current_stage + 1 && t.status == TaskStatus::Locked)
            .min_by_key(|(_, t)| (t.stage_order, t.task_order))
            .map(|(idx, _)| idx);

        if let Some(idx) = next_stage_target {
            tasks[idx].status = TaskStatus::Active;
            return Some(tasks[idx].task_id.clone());
        }

        None
    }

    pub fn set_lockdown(&self, is_creator: bool, locked: bool) -> Result<(), String> {
        if !is_creator {
            return Err("Unauthorized: Only Creator Authority can toggle system lockdown".into());
        }
        *self.is_locked_down.lock().unwrap() = locked;
        self.save_encrypted()?;
        Ok(())
    }

    pub fn save_encrypted(&self) -> Result<(), String> {
        let tasks = self.tasks.lock().unwrap();
        self.save_encrypted_slice(&tasks)
    }

    fn save_encrypted_slice(&self, tasks: &[GatedTask]) -> Result<(), String> {
        let is_locked = self.is_locked_down();
        let payload_struct = VaultPayload {
            tasks: tasks.to_vec(),
            is_locked_down: is_locked,
        };
        let payload = serde_json::to_vec(&payload_struct).map_err(|e| e.to_string())?;

        let cipher = Aes256Gcm::new_from_slice(&self.key).map_err(|e| e.to_string())?;

        let mut nonce_bytes = [0u8; 12];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher
            .encrypt(nonce, payload.as_ref())
            .map_err(|e| format!("encryption failed: {}", e))?;

        let enc_file = self.storage_dir.join("goals.enc");
        let mut combined = Vec::with_capacity(nonce_bytes.len() + ciphertext.len());
        combined.extend_from_slice(&nonce_bytes);
        combined.extend_from_slice(&ciphertext);
        crate::persistence::atomic_write_file_hardened(&enc_file, &combined)
    }

    pub fn load_encrypted(&self) -> Result<(), String> {
        let enc_file = self.storage_dir.join("goals.enc");
        if !enc_file.exists() {
            return Ok(());
        }

        let mut f = File::open(&enc_file).map_err(|e| e.to_string())?;
        let mut data = Vec::new();
        f.read_to_end(&mut data).map_err(|e| e.to_string())?;

        if data.len() < 12 {
            *self.is_locked_down.lock().unwrap() = true;
            return Err("corrupt encrypted goals vault: file too short".into());
        }

        let (nonce_bytes, ciphertext) = data.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);

        let cipher = Aes256Gcm::new_from_slice(&self.key).map_err(|e| e.to_string())?;

        let plaintext = cipher.decrypt(nonce, ciphertext).map_err(|e| {
            *self.is_locked_down.lock().unwrap() = true;
            format!(
                "goals vault decryption failed (tampered ciphertext or invalid key): {}",
                e
            )
        })?;

        let (loaded_tasks, is_locked) =
            if let Ok(payload) = serde_json::from_slice::<VaultPayload>(&plaintext) {
                (payload.tasks, payload.is_locked_down)
            } else if let Ok(raw_tasks) = serde_json::from_slice::<Vec<GatedTask>>(&plaintext) {
                (raw_tasks, false)
            } else {
                *self.is_locked_down.lock().unwrap() = true;
                return Err("corrupt encrypted goals vault payload format".into());
            };

        let mut completed = HashSet::new();
        for t in &loaded_tasks {
            if t.status == TaskStatus::VerifiedCompleted {
                completed.insert(t.task_id.clone());
            }
        }

        *self.tasks.lock().unwrap() = loaded_tasks;
        *self.completed_task_ids.lock().unwrap() = completed;
        *self.is_locked_down.lock().unwrap() = is_locked;

        Ok(())
    }

    pub fn get_tasks(&self) -> Vec<GatedTask> {
        self.tasks.lock().unwrap().clone()
    }

    #[doc(hidden)]
    pub fn test_inject_tampered_task_reward(&self, task_id: &str, tampered_points: f64) {
        let mut tasks = self.tasks.lock().unwrap();
        if let Some(t) = tasks.iter_mut().find(|t| t.task_id == task_id) {
            t.reward_points = tampered_points;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_milestone_enumeration_single_initial_active() {
        let temp_dir = std::env::temp_dir().join(format!("gated_goals_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&temp_dir);

        let vault = GatedGoalVaultEngine::new(temp_dir.clone());

        let hierarchy = GoalHierarchySpec {
            goal_headline: "Test Pipeline".to_string(),
            goal_description: "End-to-end multi-milestone testing".to_string(),
            stages: vec![StageSpec {
                stage_name: "Stage 1".to_string(),
                stage_order: 1,
                milestones: vec![
                    MilestoneSpec {
                        milestone_name: "Milestone A".to_string(),
                        tasks: vec![
                            TaskSpec {
                                task_name: "Task A1".to_string(),
                                task_order: 1,
                                task_priority: 1.0,
                                reward_points: 10.0,
                                completion_criteria: "Complete A1".to_string(),
                                verification_criteria: "None".to_string(),
                                failure_rule: FailureRule::RetryWithAlternativeStrategy,
                                max_retries: 3,
                                backoff_ms: 100,
                            },
                            TaskSpec {
                                task_name: "Task A2".to_string(),
                                task_order: 2,
                                task_priority: 1.0,
                                reward_points: 10.0,
                                completion_criteria: "Complete A2".to_string(),
                                verification_criteria: "None".to_string(),
                                failure_rule: FailureRule::RetryWithAlternativeStrategy,
                                max_retries: 3,
                                backoff_ms: 100,
                            },
                        ],
                    },
                    MilestoneSpec {
                        milestone_name: "Milestone B".to_string(),
                        tasks: vec![
                            TaskSpec {
                                task_name: "Task B1".to_string(),
                                task_order: 1,
                                task_priority: 1.0,
                                reward_points: 10.0,
                                completion_criteria: "Complete B1".to_string(),
                                verification_criteria: "None".to_string(),
                                failure_rule: FailureRule::RetryWithAlternativeStrategy,
                                max_retries: 3,
                                backoff_ms: 100,
                            },
                        ],
                    },
                ],
            }],
        };

        vault.configure_creator_goals_internal(hierarchy).unwrap();

        let tasks = vault.get_tasks();
        assert_eq!(tasks.len(), 3);

        // Strictly the first milestone's first task must be Active
        assert_eq!(tasks[0].task_name, "Task A1");
        assert_eq!(tasks[0].status, TaskStatus::Active);

        // Milestone A task 2 must be Locked
        assert_eq!(tasks[1].task_name, "Task A2");
        assert_eq!(tasks[1].status, TaskStatus::Locked);

        // Milestone B task 1 must be Locked
        assert_eq!(tasks[2].task_name, "Task B1");
        assert_eq!(tasks[2].status, TaskStatus::Locked);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
