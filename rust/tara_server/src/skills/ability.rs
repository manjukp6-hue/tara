//! Unified Ability Orchestration Engine & Central Facade for TARA.
//!
//! Provides a single, clean entry point for all ability-related discovery,
//! acquisition, execution, practice, failure learning, self-training,
//! self-update, evaluation, and promotion/rollback.
//!
//! Architecture:
//! ```text
//! AbilityEngine
//!     │
//!     ├── CapabilityRegistry      (crate::control_plane::registry::CapabilityRegistry)
//!     ├── SkillEngine             (crate::skills::SkillEngine)
//!     ├── ToolRegistry            (crate::skills::tools_registry::ToolRegistry)
//!     ├── CapabilitySynthesizer   (crate::learning::CapabilitySynthesizer)
//!     ├── ExecutionGuard          (crate::rules::ExecutionGuard)
//!     ├── AutonomousOnlineLearner (crate::learning::AutonomousOnlineLearner)
//!     ├── ContinualLearningGovernor (crate::learning::ContinualLearningGovernor)
//!     ├── NativeSelfTrainer       (tara_engine::trainer::NativeSelfTrainer)
//!     ├── SelfUpdateController    (tara_engine::self_update::SelfUpdateController)
//!     ├── SkillsEvaluator         (tara_engine::skills_evaluator::SkillsEvaluator)
//!     └── SelfUpgradeEngine       (crate::learning::SelfUpgradeEngine)
//! ```
//!
//! Enforces:
//! 1. ZERO duplication of underlying engine logic (Existing Code First mandate).
//! 2. Pure Rust native execution (Zero Python workflow).
//! 3. Cryptographic integrity and access confirmation (Verification Before Confirmation & ExecutionGuard).
//! 4. Seamless coordination across all 11 core ability subsystems.

use serde_json::{json, Value};
use std::fs::{self, OpenOptions};
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use super::tools_registry::ToolRegistry;
use crate::brain::TrainingLifecycleState;
use crate::control_plane::registry::{Capability, CapabilityCategory, CapabilityRegistry};
use crate::learning::{
    AutonomousOnlineLearner, CapabilitySynthesizer, ContinualLearningGovernor, SelfUpgradeEngine,
};
use crate::rules::ExecutionGuard;
use crate::skills::SkillEngine;
use tara_engine::{SelfUpdateController, SkillsEvaluator};
use tara_security::primitives::AuthenticatedPrincipal;

/// Unified orchestrator coordinating TARA's existing capability and self-evolution systems.
pub struct AbilityEngine {
    repo_root: PathBuf,
    model_dir: PathBuf,
    pub(crate) capability_registry: Arc<CapabilityRegistry>,
    pub(crate) skill_engine: Arc<SkillEngine>,
    pub(crate) tool_registry: Arc<ToolRegistry>,
    pub(crate) capability_synthesizer: Arc<CapabilitySynthesizer>,
    pub(crate) execution_guard: Arc<ExecutionGuard>,
    pub(crate) training_gate: Arc<Mutex<TrainingLifecycleState>>,
    pub(crate) autonomous_learner: Arc<Mutex<AutonomousOnlineLearner>>,
    pub(crate) continual_governor: Arc<ContinualLearningGovernor>,
    pub(crate) self_update: Arc<SelfUpdateController>,
    pub(crate) skills_evaluator: Arc<SkillsEvaluator>,
    pub(crate) self_upgrade: Arc<SelfUpgradeEngine>,
}

/// Configuration components for constructing an `AbilityEngine` from shared subsystems.
pub struct AbilityComponents {
    pub repo_root: PathBuf,
    pub model_dir: PathBuf,
    pub capability_registry: Arc<CapabilityRegistry>,
    pub skill_engine: Arc<SkillEngine>,
    pub tool_registry: Arc<ToolRegistry>,
    pub capability_synthesizer: Arc<CapabilitySynthesizer>,
    pub execution_guard: Arc<ExecutionGuard>,
    pub training_gate: Arc<Mutex<TrainingLifecycleState>>,
    pub autonomous_learner: Arc<Mutex<AutonomousOnlineLearner>>,
    pub continual_governor: Arc<ContinualLearningGovernor>,
    pub self_update: Arc<SelfUpdateController>,
    pub skills_evaluator: Arc<SkillsEvaluator>,
    pub self_upgrade: Arc<SelfUpgradeEngine>,
}

impl AbilityEngine {
    /// Construct a new AbilityEngine from paths, initializing default underlying subsystems.
    pub fn new<P: AsRef<Path>>(repo_root: P, model_dir: &str) -> Self {
        let root = repo_root.as_ref().to_path_buf();
        let policy_path = root.join("TARA").join("RULES").join("compiled_policy.json");
        let loaded_policy =
            crate::rules::CompiledPolicy::from_json_file(&policy_path.to_string_lossy()).ok();
        let execution_guard = Arc::new(ExecutionGuard::new(loaded_policy));
        let training_gate = Arc::new(Mutex::new(TrainingLifecycleState::Idle));
        Self::new_with_shared(repo_root, model_dir, execution_guard, training_gate)
    }

    /// Construct an AbilityEngine with explicitly shared execution guard and training gate.
    pub fn new_with_shared<P: AsRef<Path>>(
        repo_root: P,
        model_dir: &str,
        execution_guard: Arc<ExecutionGuard>,
        training_gate: Arc<Mutex<TrainingLifecycleState>>,
    ) -> Self {
        let root = repo_root.as_ref().to_path_buf();
        let storage_root = root.join("storage");
        let skills_dir = storage_root.join("skills").to_string_lossy().to_string();
        let dynamic_skills_dir = root
            .join("TARA")
            .join("SKILLS")
            .to_string_lossy()
            .to_string();
        let _ = fs::create_dir_all(&skills_dir);
        let _ = fs::create_dir_all(&dynamic_skills_dir);

        let capability_registry = CapabilityRegistry::get_default();
        let skill_engine = Arc::new(SkillEngine::new(&skills_dir, &dynamic_skills_dir));
        let tool_registry = ToolRegistry::get_default(&root.to_string_lossy());
        let capability_synthesizer = Arc::new(CapabilitySynthesizer::new(&root));
        let autonomous_learner = Arc::new(Mutex::new(AutonomousOnlineLearner::new(
            &root.to_string_lossy(),
        )));
        let continual_governor = Arc::new(ContinualLearningGovernor::default());
        let self_update = Arc::new(SelfUpdateController::new(&root));
        let skills_evaluator = Arc::new(SkillsEvaluator::new(model_dir));
        let self_upgrade = Arc::new(SelfUpgradeEngine::new(&root));

        Self {
            repo_root: root,
            model_dir: PathBuf::from(model_dir),
            capability_registry,
            skill_engine,
            tool_registry,
            capability_synthesizer,
            execution_guard,
            training_gate,
            autonomous_learner,
            continual_governor,
            self_update,
            skills_evaluator,
            self_upgrade,
        }
    }

    /// Construct an AbilityEngine from a shared components bundle.
    pub fn from_components(components: AbilityComponents) -> Self {
        Self {
            repo_root: components.repo_root,
            model_dir: components.model_dir,
            capability_registry: components.capability_registry,
            skill_engine: components.skill_engine,
            tool_registry: components.tool_registry,
            capability_synthesizer: components.capability_synthesizer,
            execution_guard: components.execution_guard,
            training_gate: components.training_gate,
            autonomous_learner: components.autonomous_learner,
            continual_governor: components.continual_governor,
            self_update: components.self_update,
            skills_evaluator: components.skills_evaluator,
            self_upgrade: components.self_upgrade,
        }
    }

    /// 1. DISCOVER: Query all available capabilities, skills, and tools across all registries.
    pub fn discover(&self) -> Value {
        let capabilities = self.capability_registry.to_json();
        let skills = self.skill_engine.list_skills();
        let tools = self.tool_registry.list_tools();
        json!({
            "status": "DISCOVERED",
            "capabilities": capabilities,
            "skills_count": skills.len(),
            "skills": skills,
            "tools_count": tools.len(),
            "tools": tools,
        })
    }

    /// Check if a named ability/skill/tool is discoverable and available.
    pub fn is_available(&self, name: &str) -> bool {
        self.capability_registry.get_capability(name).is_some()
            || self.skill_engine.list_skills().contains(&name.to_string())
            || self.tool_registry.list_tools().contains(&name.to_string())
    }

    /// 2. ACQUIRE: Register or synthesize a new ability.
    pub fn acquire(
        &self,
        capability_name: &str,
        category: CapabilityCategory,
        description: &str,
        tool_code: Option<&str>,
        principal: &AuthenticatedPrincipal,
    ) -> Result<Value, String> {
        self.execution_guard
            .enforce_creator_principal(principal, "acquire_ability")?;

        if let Some(code) = tool_code {
            let storage_root = self.repo_root.join("storage");
            let kb = crate::knowledge::GlobalKnowledgeBase::new(&format!(
                "{}/knowledge",
                storage_root.display()
            ));
            let synth_res = self.capability_synthesizer.synthesize_tool_capability(
                &kb,
                capability_name,
                code,
                description,
                principal.is_creator(),
            )?;
            let cap = Capability::new(capability_name, category, description);
            let _ = self.capability_registry.upsert_capability(cap);
            Ok(json!({
                "status": "ACQUIRED_SYNTHESIZED",
                "capability": capability_name,
                "synthesis": synth_res,
            }))
        } else {
            let cap = Capability::new(capability_name, category, description);
            self.capability_registry
                .register_capability(cap)
                .map_err(|e| e.to_string())?;
            Ok(json!({
                "status": "ACQUIRED_REGISTERED",
                "capability": capability_name,
                "description": description,
            }))
        }
    }

    /// 3. IMPROVE SKILL: Update/improve an existing skill definition and persist it safely.
    pub fn improve_skill(
        &self,
        skill_name: &str,
        updated_definition: &Value,
        principal: &AuthenticatedPrincipal,
    ) -> Result<Value, String> {
        self.execution_guard
            .enforce_creator_principal(principal, "improve_skill")?;

        let result = self
            .skill_engine
            .add_dynamic_skill(skill_name, updated_definition.clone())?;

        Ok(json!({
            "status": "IMPROVED",
            "skill": skill_name,
            "detail": result,
        }))
    }

    /// 4. PRACTICE: Safely evaluate and test an ability with sample prompts/parameters.
    pub fn practice(
        &self,
        skill_name: &str,
        test_prompts: &[String],
        execution_params: Option<&Value>,
        principal: Option<&AuthenticatedPrincipal>,
    ) -> Result<Value, String> {
        let evaluation = self.skills_evaluator.evaluate(skill_name, test_prompts);

        let execution = if let Some(params) = execution_params {
            let p = principal
                .ok_or_else(|| "Practice execution requires authenticated principal".to_string())?;
            let act = crate::rules::AuthorizedAction::new(
                p.clone(),
                "execute_skill",
                skill_name,
                params.clone(),
            );
            let res = self.execution_guard.authorize_action(&act);
            if res.decision == "BLOCK" {
                return Err(format!("Skill execution blocked: {}", res.reason));
            }
            Some(self.skill_engine.execute_skill(skill_name, params.clone()))
        } else {
            None
        };

        Ok(json!({
            "status": "PRACTICED",
            "skill": skill_name,
            "neural_evaluation": evaluation,
            "test_execution": execution,
        }))
    }

    /// 5. LEARN FROM FAILURE: Record an experiential failure lesson and update continual learning anchors.
    pub fn learn_from_failure(
        &self,
        task_category: &str,
        trigger_pattern: &str,
        failed_strategy: &str,
        error_trace: &str,
    ) -> Result<Value, String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let lesson = json!({
            "lesson_id": format!("lesson_{}_{}", task_category, now),
            "task_category": task_category,
            "trigger_pattern": trigger_pattern,
            "strategy_used": failed_strategy,
            "outcome_score": 0.0,
            "root_cause": error_trace,
            "recommendation": format!("Avoid '{}', adapt alternative strategy", failed_strategy),
            "confidence_score": 0.5,
            "applied_count": 1,
            "timestamp_ms": now,
        });

        let store_dir = self.repo_root.join("storage").join("persistence");
        let _ = fs::create_dir_all(&store_dir);
        let lesson_file = store_dir.join("experiential_lessons.jsonl");

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&lesson_file)
            .map_err(|e| format!("Failed to open experiential lessons store: {e}"))?;

        let serialized_lesson = serde_json::to_string(&lesson).map_err(|e| e.to_string())?;
        writeln!(file, "{}", serialized_lesson)
            .map_err(|e| format!("Failed to append lesson: {e}"))?;
        file.sync_all()
            .map_err(|e| format!("Failed to sync experiential lessons store: {e}"))?;

        // Protect from degradation in ContinualLearningGovernor
        self.continual_governor.register_anchor(
            task_category,
            1.5, // baseline loss anchor
            0.0, // baseline accuracy for failed task
            1.0, // importance weight
        );

        Ok(json!({
            "status": "FAILURE_LEARNED",
            "lesson": lesson,
            "persisted_path": lesson_file.display().to_string(),
        }))
    }

    /// 6. REQUEST SELF-TRAINING: Run protocol-compliant isolated self-training cycle.
    pub fn request_self_training(
        &self,
        max_epochs: usize,
        dataset_dir: Option<&str>,
        curriculum_path: Option<&str>,
        principal: &AuthenticatedPrincipal,
    ) -> Result<Value, String> {
        self.execution_guard
            .enforce_creator_principal(principal, "self_training")?;

        // Shared process-wide training gate
        {
            let mut gate = self
                .training_gate
                .lock()
                .map_err(|e| format!("Training gate lock error: {e}"))?;
            if *gate != TrainingLifecycleState::Idle {
                return Err(format!(
                    "Training cycle rejected: training state is {:?}",
                    *gate
                ));
            }
            *gate = TrainingLifecycleState::Training;
        }

        let candidate_dir = self
            .repo_root
            .join("storage")
            .join("models")
            .join("tara_candidate_v1");
        let options = tara_engine::train_candidate::TrainingOptions {
            dataset_dir: dataset_dir.map(String::from),
            curriculum_path: curriculum_path.map(String::from),
            candidate_dir: Some(candidate_dir.to_string_lossy().to_string()),
            ..Default::default()
        };

        let result = tara_engine::train_candidate::run_controlled_training_with_options(
            self.model_dir.to_str().unwrap_or("storage/models/tara"),
            self.repo_root.to_str().unwrap_or("."),
            max_epochs,
            options,
        );

        // Always restore training gate state on completion or failure
        if let Ok(mut gate) = self.training_gate.lock() {
            *gate = TrainingLifecycleState::Idle;
        }

        let train_res = result.map_err(|e| e.to_string())?;

        Ok(json!({
            "status": "SELF_TRAINING_CYCLE_COMPLETED",
            "result": train_res,
        }))
    }

    /// Safely query current self-training status from the authoritative ledger.
    pub fn get_training_status(&self) -> Value {
        tara_engine::train_candidate::get_training_status(
            self.repo_root.to_str().unwrap_or("."),
            self.model_dir.to_str().unwrap_or("storage/models/tara"),
        )
    }

    /// 7. REQUEST SELF-UPDATE: Analyze vocabulary expansion requirements and stage candidate model.
    pub fn request_self_update(
        &self,
        dataset_path: &str,
        principal: &AuthenticatedPrincipal,
    ) -> Result<Value, String> {
        self.execution_guard
            .enforce_creator_principal(principal, "self_update")?;

        let model_dir_str = self.model_dir.to_string_lossy();
        let decision = self
            .self_update
            .analyze_vocabulary(&model_dir_str, dataset_path)
            .map_err(|e| e.to_string())?;

        Ok(json!({
            "status": "SELF_UPDATE_ANALYZED",
            "decision": decision,
        }))
    }

    /// 8. EVALUATE UPGRADE: Perform comprehensive subsystem health and compilation check.
    pub fn evaluate_upgrade(&self, subsystem: &str) -> Result<Value, String> {
        let inspect_res = self.self_upgrade.inspect_subsystem(subsystem)?;
        Ok(json!({
            "status": "UPGRADE_EVALUATION_COMPLETED",
            "inspection": inspect_res,
        }))
    }

    /// 9. PROMOTE OR ROLLBACK: Execute atomic upgrade promotion or restore prior snapshot.
    pub fn promote_or_rollback(
        &self,
        action: &str,
        upgrade_id: &str,
        subsystem: &str,
        affected_files: &[PathBuf],
        principal: &AuthenticatedPrincipal,
    ) -> Result<Value, String> {
        self.execution_guard
            .enforce_creator_principal(principal, "promote_or_rollback")?;

        let is_creator = principal.is_creator();
        match action.to_lowercase().as_str() {
            "promote" | "stage" => {
                let stage_res = self.self_upgrade.stage_upgrade(
                    subsystem,
                    upgrade_id,
                    affected_files,
                    is_creator,
                )?;
                Ok(json!({
                    "action": "PROMOTE_STAGE",
                    "status": "SUCCESS",
                    "result": stage_res,
                }))
            }
            "rollback" => {
                let rollback_res = self.self_upgrade.rollback_upgrade(upgrade_id, is_creator)?;
                Ok(json!({
                    "action": "ROLLBACK",
                    "status": "SUCCESS",
                    "result": rollback_res,
                }))
            }
            other => Err(format!(
                "Invalid action '{}'. Valid actions: 'promote' or 'rollback'",
                other
            )),
        }
    }

    /// Returns a reference to the underlying autonomous learner.
    pub fn autonomous_learner(&self) -> &Arc<Mutex<AutonomousOnlineLearner>> {
        &self.autonomous_learner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_ability_engine() -> AbilityEngine {
        let temp_dir =
            std::env::temp_dir().join(format!("tara_ability_test_{}", std::process::id()));
        let _ = fs::create_dir_all(&temp_dir);
        let model_dir = temp_dir.join("test_model");
        let _ = fs::create_dir_all(&model_dir);
        AbilityEngine::new(&temp_dir, &model_dir.to_string_lossy())
    }

    #[test]
    fn test_ability_engine_discover_and_availability() {
        let engine = create_test_ability_engine();
        let discovery = engine.discover();
        assert_eq!(discovery["status"], "DISCOVERED");
        assert!(discovery["skills_count"].as_u64().unwrap_or(0) > 0);
        assert!(discovery["tools_count"].as_u64().unwrap_or(0) > 0);

        // Built-in capabilities should be available
        assert!(engine.is_available("cognition.respond"));
        assert!(engine.is_available("multimedia.generate_image"));
        assert!(engine.is_available("file_inspector"));
    }

    #[test]
    fn test_ability_engine_acquire_authorization_gate() {
        let engine = create_test_ability_engine();
        let regular_user = AuthenticatedPrincipal::user("user_123", "sess_user");
        let creator = AuthenticatedPrincipal::mint_creator_for_test("sess_creator").unwrap();

        // Unauthorized non-creator must be blocked
        let res = engine.acquire(
            "custom.ability_1",
            CapabilityCategory::Custom,
            "Custom test ability",
            None,
            &regular_user,
        );
        assert!(
            res.is_err(),
            "Non-creator must be denied capability acquisition"
        );

        // Authorized creator can register
        let ok_res = engine.acquire(
            "custom.ability_1",
            CapabilityCategory::Custom,
            "Custom test ability",
            None,
            &creator,
        );
        assert!(
            ok_res.is_ok(),
            "Creator must be allowed capability registration"
        );
        assert!(engine.is_available("custom.ability_1"));
    }

    #[test]
    fn test_ability_engine_practice_and_failure_learning() {
        let engine = create_test_ability_engine();

        // Practice evaluation without execution
        let practice_res = engine.practice("multimedia.generate_image", &[], None, None);
        assert!(practice_res.is_ok());

        // Practice execution with creator principal
        let creator = AuthenticatedPrincipal::mint_creator_for_test("sess_creator").unwrap();
        let practice_exec = engine.practice(
            "multimedia.generate_image",
            &[],
            Some(&json!({})),
            Some(&creator),
        );
        assert!(practice_exec.is_ok());

        // Learn from failure
        let fail_res = engine.learn_from_failure(
            "CODING",
            "solve differential equation",
            "brute_force_grid",
            "Execution timeout exceeded",
        );
        assert!(fail_res.is_ok());
        let val = fail_res.unwrap();
        assert_eq!(val["status"], "FAILURE_LEARNED");
        assert_eq!(val["lesson"]["task_category"], "CODING");
        assert_eq!(val["lesson"]["outcome_score"], 0.0);
    }

    #[test]
    fn test_ability_engine_self_evolution_authorization_gates() {
        let engine = create_test_ability_engine();
        let regular_user = AuthenticatedPrincipal::user("user_123", "sess_user");

        // Self-training gate
        assert!(engine
            .request_self_training(1, None, None, &regular_user)
            .is_err());

        // Self-update gate
        assert!(engine
            .request_self_update("storage/datasets/non_existent.jsonl", &regular_user)
            .is_err());

        // Subsystem evaluation
        let eval_res = engine.evaluate_upgrade("MEMORY");
        assert!(eval_res.is_ok());

        // Promote/rollback gate
        assert!(engine
            .promote_or_rollback("rollback", "up_001", "MEMORY", &[], &regular_user)
            .is_err());
    }
}
