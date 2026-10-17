//! Canonical Creator-Gated Task & Goal Server Adapter for TARA AI.
//!
//! Direct server adapter over the canonical `tara_security::tara_control::gated_goals::GatedGoalVaultEngine`.
//! Connects the server's `RewardSystem` via `GoalRewardDisburser` without circular crate dependencies.

use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

use crate::reward::{RewardCategory, RewardSystem};
use tara_security::lockdown::LockdownCoordinator;
use tara_security::primitives::{AuthenticatedPrincipal, VerifiedPrincipal};

// Re-export canonical types from tara_security
pub use tara_security::tara_control::gated_goals::{
    ActiveTaskView, FailureRule, GatedGoalVaultEngine, GatedTask, GoalHierarchySpec,
    GoalRewardDisburser, MilestoneSpec, RetryRule, StageSpec, TaskSpec, TaskStatus,
    VerificationResult,
};

/// Server adapter implementing `GoalRewardDisburser` for `RewardSystem`.
pub struct ServerRewardDisburser {
    reward_system: Arc<RewardSystem>,
}

impl ServerRewardDisburser {
    pub fn new(reward_system: Arc<RewardSystem>) -> Self {
        Self { reward_system }
    }
}

impl GoalRewardDisburser for ServerRewardDisburser {
    fn disburse_reward(
        &self,
        recipient: &str,
        _task_id: &str,
        points: f64,
        reason: &str,
    ) -> Result<String, String> {
        let rew_event = self
            .reward_system
            .record_reward(recipient, RewardCategory::TaskCompletion, points, reason)
            .map_err(|e| format!("RewardDisbursementError: {}", e))?;
        Ok(rew_event.event_id)
    }
}

/// Canonical Creator-Gated Goal System facade for TARA Server.
/// Delegates all storage, encryption, and milestone logic to `GatedGoalVaultEngine` in `tara_security`.
pub struct GatedGoalSystem {
    engine: GatedGoalVaultEngine,
    reward_system: Arc<RewardSystem>,
}

impl GatedGoalSystem {
    /// Try initializing the GatedGoalSystem.
    /// Strictly fails closed if the encrypted goals vault exists but is corrupted or tampered.
    pub fn try_new<P: AsRef<Path>>(
        repo_root: P,
        reward_system: Arc<RewardSystem>,
    ) -> Result<Self, String> {
        let engine = GatedGoalVaultEngine::try_new(repo_root)?;
        Ok(Self {
            engine,
            reward_system,
        })
    }

    /// Primary constructor. Fails closed with lockdown if encrypted vault is corrupted.
    pub fn new<P: AsRef<Path>>(repo_root: P, reward_system: Arc<RewardSystem>) -> Self {
        let engine = GatedGoalVaultEngine::new(repo_root);
        Self {
            engine,
            reward_system,
        }
    }

    /// Attaches the authoritative LockdownCoordinator to synchronize global lockdown state.
    pub fn with_lockdown_coordinator(mut self, coordinator: Arc<LockdownCoordinator>) -> Self {
        self.engine = self.engine.with_lockdown_coordinator(coordinator);
        self
    }

    /// Dynamically sets or updates the authoritative LockdownCoordinator reference.
    pub fn set_lockdown_coordinator(&mut self, coordinator: Arc<LockdownCoordinator>) {
        self.engine.set_lockdown_coordinator(coordinator);
    }

    /// Checks unified lockdown state across both internal vault guard and authoritative LockdownCoordinator.
    pub fn is_locked_down(&self) -> bool {
        self.engine.is_locked_down()
    }

    /// Strongly typed Creator-authorized injection of goal hierarchy.
    pub fn configure_creator_goals_with_authority(
        &self,
        principal: &AuthenticatedPrincipal,
        hierarchy: GoalHierarchySpec,
    ) -> Result<usize, String> {
        if !principal.is_creator() {
            return Err(
                "Unauthorized: Only verified Creator Authority can configure goal hierarchy".into(),
            );
        }
        let vp: VerifiedPrincipal = principal.try_into()?;
        self.engine.configure_creator_goals(&vp, hierarchy)
    }

    /// Creator-authorized injection of goal hierarchy with VerifiedPrincipal.
    pub fn configure_creator_goals(
        &self,
        principal: &VerifiedPrincipal,
        hierarchy: GoalHierarchySpec,
    ) -> Result<usize, String> {
        self.engine.configure_creator_goals(principal, hierarchy)
    }

    /// Surfaces only the single active task to TARA.
    pub fn get_active_task_for_tara(&self) -> Option<ActiveTaskView> {
        self.engine.get_active_task_for_tara()
    }

    /// Access control for Private Control Plane with AuthenticatedPrincipal.
    pub fn inspect_control_plane_with_authority(
        &self,
        principal: &AuthenticatedPrincipal,
    ) -> Result<Value, String> {
        let (integrity, err) = self.reward_system.verify_ledger_integrity();
        self.engine.inspect_control_plane(
            principal.is_creator(),
            self.reward_system.get_summary(),
            integrity,
            err,
        )
    }

    /// Access control for Private Control Plane.
    pub fn inspect_control_plane(&self, is_creator: bool) -> Result<Value, String> {
        let (integrity, err) = self.reward_system.verify_ledger_integrity();
        self.engine.inspect_control_plane(
            is_creator,
            self.reward_system.get_summary(),
            integrity,
            err,
        )
    }

    /// Independent Task Verification and Gating Engine.
    pub fn submit_and_verify_active_task(
        &self,
        task_id: &str,
        proof_payload: &Value,
    ) -> Result<VerificationResult, String> {
        if self.is_locked_down() {
            return Err(
                "LockdownActive: Task verification and reward disbursement blocked during security lockdown"
                    .into(),
            );
        }
        let disburser = ServerRewardDisburser::new(self.reward_system.clone());
        self.engine
            .submit_and_verify_active_task(task_id, proof_payload, Some(&disburser))
    }

    /// Creator-authorized lockdown freezing all task execution.
    pub fn set_lockdown(&self, is_creator: bool, locked: bool) -> Result<(), String> {
        self.engine.set_lockdown(is_creator, locked)
    }

    /// Save all tasks encrypted with AES-256-GCM.
    pub fn save_encrypted(&self) -> Result<(), String> {
        self.engine.save_encrypted()
    }

    /// Load tasks from AES-256-GCM encrypted file.
    /// Fails closed if ciphertext is modified or decryption fails.
    pub fn load_encrypted(&mut self) -> Result<(), String> {
        self.engine.load_encrypted()
    }

    /// Access reference to underlying canonical engine.
    pub fn engine(&self) -> &GatedGoalVaultEngine {
        &self.engine
    }

    /// SECURITY TEST ONLY: Direct tampering test helper to test detection.
    /// Strictly hidden from public docs and forbidden in production code.
    #[doc(hidden)]
    pub fn test_inject_tampered_task_reward(&self, task_id: &str, tampered_points: f64) {
        self.engine
            .test_inject_tampered_task_reward(task_id, tampered_points);
    }
}
