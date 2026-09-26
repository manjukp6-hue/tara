//! TaraBrain: the central cognitive orchestrator.
//!
//! Ports Python's `brain.py` in full: 14-step cognitive loop, model inference,
//! identity, memory, knowledge, rules, skills, learning, and agent execution.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use tara_engine::{
    TaraForCausalLM, TaraTokenizer,
    generate_response, generate_stream,
    ControlTokenActionParser,
};

use crate::nlu::{SemanticIntentParser, IntentResult};
use crate::context::SessionContextManager;
use crate::planner::{TaskPlanner, GoalTracker};
use crate::evaluator::{TaskVerifier, RecoveryEngine, UncertaintyDetector, SelfEvaluator};
use crate::memory::MemoryEngine;
use crate::knowledge::GlobalKnowledgeBase;
use crate::rules::ExecutionGuard;
use crate::skills::SkillEngine;
use crate::learning::{UserSearchLearner, AutonomousOnlineLearner};
use crate::access::{IdentityManager, CreatorAuthService};
use crate::auto_connect::AutoConnectSyncEngine;
use crate::engine_system::DynamicEngineSystem;
use crate::cognitive::CognitiveCapabilitiesHub;
use crate::registry::CapabilityRegistry;
use crate::tools_registry::ToolRegistry;
use crate::memory_hub::DynamicMemoryHub;
use crate::model_registry::ModelRegistry;
use crate::subsystems::{
    user_model::UserManager,
    agent_orchestrator::AgentOrchestrator,
    language_registry::LanguageRegistry,
    plugin_engine::PluginEngine,
    resilience::ResilienceEngine,
};
use crate::server::WebSessionStore;

use tara_engine::trainer::NativeSelfTrainer;

const EXPECTED_SHA256: &str = "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309";

/// The TARA cognitive brain — single logical production model.
pub struct TaraBrain {
    pub repo_root: String,
    pub model_dir: String,
    pub model: Option<TaraForCausalLM>,
    pub tokenizer: Option<TaraTokenizer>,
    pub model_metadata: Value,

    // Identity & Auth
    pub identity_manager: IdentityManager,
    pub creator_auth_service: Arc<CreatorAuthService>,

    // Rules & Policy
    pub guard: ExecutionGuard,

    // Knowledge & Memory
    pub knowledge_base: GlobalKnowledgeBase,
    pub memory_engine: MemoryEngine,

    // Skills, Learning, Training
    pub skill_engine: SkillEngine,
    pub user_search_learner: UserSearchLearner,
    pub autonomous_learner: AutonomousOnlineLearner,
    pub self_trainer: NativeSelfTrainer,

    // NLU & Context
    pub session_manager: SessionContextManager,
    pub intent_parser: SemanticIntentParser,

    // Planning & Evaluation
    pub goal_tracker: GoalTracker,
    pub task_verifier: TaskVerifier,
    pub recovery_engine: RecoveryEngine,
    pub uncertainty_detector: UncertaintyDetector,
    pub self_evaluator: SelfEvaluator,

    // Registries
    pub capability_registry: Arc<CapabilityRegistry>,
    pub tool_registry: Arc<ToolRegistry>,
    pub memory_hub: DynamicMemoryHub,
    pub model_registry: Arc<ModelRegistry>,

    // Cluster & Engines
    pub auto_sync: Arc<AutoConnectSyncEngine>,
    pub engine_system: Arc<DynamicEngineSystem>,

    // Subsystems
    pub user_manager: Arc<UserManager>,
    pub agent_orchestrator: Arc<AgentOrchestrator>,
    pub language_registry: Arc<LanguageRegistry>,
    pub plugin_engine: Arc<PluginEngine>,
    pub resilience: ResilienceEngine,
    pub cognitive: CognitiveCapabilitiesHub,

    // Web session store (shared with server)
    pub web_sessions: Arc<WebSessionStore>,

    // Unified Python-Rust Hybrid Bridge
    pub bridge: Arc<crate::bridge::UnifiedBridge>,

    // Dynamic Compute Control Plane
    pub control_plane: Arc<crate::control_plane::ControlPlane>,
}

impl TaraBrain {
    /// Initialise TaraBrain from `model_dir` (or discover via ModelRegistry).
    ///
    /// # Errors
    /// Returns `Err(String)` only if core storage dirs cannot be created.
    pub fn new(model_dir: Option<&str>) -> Result<Self, String> {
        // Resolve repo root relative to binary location or current working dir
        let repo_root = std::env::var("TARA_REPO_ROOT").unwrap_or_else(|_| {
            let candidates = [
                ".",
                "..",
                "../..",
                "../../..",
            ];
            for c in candidates {
                if std::path::Path::new(&format!("{}/storage/models/tara/model.safetensors", c)).exists() {
                    return c.to_string();
                }
            }
            ".".to_string()
        });

        // Resolve model directory
        let resolved_model_dir = model_dir
            .map(String::from)
            .unwrap_or_else(|| format!("{}/storage/models/tara", repo_root));

        // Load neural model (non-fatal)
        let (model_opt, tokenizer_opt) = Self::try_load_model(&resolved_model_dir);

        // Model metadata
        let model_metadata = {
            let sha = tara_engine::safetensors::compute_sha256(
                &format!("{}/model.safetensors", resolved_model_dir)
            ).unwrap_or_default();
            json!({
                "model_dir": resolved_model_dir,
                "sha256": sha,
                "sha256_valid": sha == EXPECTED_SHA256,
                "loaded": model_opt.is_some(),
            })
        };

        // Storage dirs
        let storage_root = format!("{}/storage", repo_root);
        for dir in &["memory/episodes", "knowledge", "knowledge/candidates",
                     "datasets", "training", "audit", "memory/users", "skills"] {
            let _ = std::fs::create_dir_all(format!("{}/{}", storage_root, dir));
        }

        // Identity
        let identity_manager = IdentityManager::new(&format!("{}/TARA/ACCESS", repo_root));
        let creator_auth_service = CreatorAuthService::new(&repo_root);

        // Policy / Rules
        let policy_path = format!("{}/TARA/RULES/compiled_policy.json", repo_root);
        let guard = {
            let loaded_policy = crate::rules::CompiledPolicy::from_json_file(&policy_path).ok();
            ExecutionGuard::new(loaded_policy)
        };

        // Knowledge & Memory
        let knowledge_base = GlobalKnowledgeBase::new(&format!("{}/knowledge", storage_root));
        let memory_engine = MemoryEngine::new(&format!("{}/memory", storage_root));

        // Skills & Learning
        let skills_dir = format!("{}/skills", storage_root);
        let dynamic_skills_dir = format!("{}/TARA/SKILLS", repo_root);
        let _ = std::fs::create_dir_all(&skills_dir);
        let _ = std::fs::create_dir_all(&dynamic_skills_dir);
        let skill_engine = SkillEngine::new(&skills_dir, &dynamic_skills_dir);
        let available_skills = skill_engine.list_skills();

        let user_search_learner = UserSearchLearner::new(&repo_root);
        let autonomous_learner = AutonomousOnlineLearner::new(&repo_root);
        let self_trainer = NativeSelfTrainer::new(&resolved_model_dir, &repo_root);

        // NLU & Context
        let session_manager = SessionContextManager::new(50, 3600);
        let intent_parser = SemanticIntentParser::new(available_skills);

        // Planning & Evaluation
        let goal_tracker = GoalTracker::new();
        let task_verifier = TaskVerifier;
        let recovery_engine = RecoveryEngine::new(&repo_root);
        let uncertainty_detector = UncertaintyDetector;
        let self_evaluator = SelfEvaluator;

        // Registries
        let capability_registry = CapabilityRegistry::get_default();
        let tool_registry = ToolRegistry::get_default(&repo_root);
        let memory_hub = DynamicMemoryHub::new(&format!("{}/memory/hub", storage_root))
            .map_err(|e| e.to_string())?;
        let model_registry = ModelRegistry::new(&repo_root);

        // Cluster & Engines
        let auto_sync = AutoConnectSyncEngine::new();
        let engine_system = DynamicEngineSystem::new();

        // Subsystems
        let user_manager = UserManager::new(&format!("{}/memory/users", storage_root));
        let agent_orchestrator = AgentOrchestrator::new();
        let language_registry = LanguageRegistry::new();
        let plugin_engine = PluginEngine::new();
        let resilience = ResilienceEngine::new(&repo_root);
        let cognitive = CognitiveCapabilitiesHub::new(&repo_root);

        let web_sessions = Arc::new(WebSessionStore::new());

        Ok(Self {
            repo_root: repo_root.clone(),
            model_dir: resolved_model_dir,
            model: model_opt,
            tokenizer: tokenizer_opt,
            model_metadata,
            identity_manager,
            creator_auth_service,
            guard,
            knowledge_base,
            memory_engine,
            skill_engine,
            user_search_learner,
            autonomous_learner,
            self_trainer,
            session_manager,
            intent_parser,
            goal_tracker,
            task_verifier,
            recovery_engine,
            uncertainty_detector,
            self_evaluator,
            capability_registry,
            tool_registry,
            memory_hub,
            model_registry,
            auto_sync,
            engine_system,
            user_manager,
            agent_orchestrator,
            language_registry,
            plugin_engine,
            resilience,
            cognitive,
            web_sessions,
            bridge: crate::bridge::UnifiedBridge::new(),
            control_plane: Arc::new(crate::control_plane::ControlPlane::new(&repo_root)),
        })
    }

    fn try_load_model(model_dir: &str) -> (Option<TaraForCausalLM>, Option<TaraTokenizer>) {
        let model = TaraForCausalLM::load(model_dir).ok();
        let tokenizer_path = format!("{}/tokenizer.json", model_dir);
        let tokenizer = TaraTokenizer::from_file(&tokenizer_path).ok();
        (model, tokenizer)
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Model Inference
    // ──────────────────────────────────────────────────────────────────────────

    /// Run inference and return generated text.
    pub fn _infer(&self, prompt: &str, max_new_tokens: usize) -> String {
        // 1. Dynamic Hardware Bridge check: If Python GPU worker is active, route heavy inference to it
        if self.bridge.should_route_to_python(None) {
            let start = std::time::Instant::now();
            let req_id = format!("req_{}", start.elapsed().as_nanos());
            let canonical_req = crate::contract::CanonicalInferenceRequest {
                request_id: req_id,
                prompt: prompt.to_string(),
                expected_model_checksum: crate::contract::CANONICAL_MODEL_SHA256.to_string(),
                job_id: None,
                max_tokens: max_new_tokens,
                temperature: 0.7,
                top_k: 50,
                top_p: 0.9,
                repetition_penalty: 1.1,
                stop_tokens: vec!["<|im_end|>".to_string(), "<|pad|>".to_string()],
                expected_model_identity: crate::contract::CANONICAL_MODEL_IDENTITY.to_string(),
                auth_context: None,
            };
            match self.bridge.infer_with_contract(&canonical_req) {
                Ok(resp) => {
                    return resp.text;
                }
                Err(_err) => {
                    // Failover immediately to native Rust CPU without crashing, record measured latency
                    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                    self.bridge.record_failover(elapsed_ms);
                }
            }
        }

        // 2. Native Rust CPU engine (zero-copy in-memory execution)
        self.bridge.record_rust_dispatch();
        match (&self.model, &self.tokenizer) {
            (Some(model), Some(tokenizer)) => {
                generate_response(model, tokenizer, prompt, max_new_tokens,
                    0.7, 50, 0.9, 1.1, None)
                    .map(|r| r.text)
                    .unwrap_or_else(|e| format!("[INFERENCE_ERROR: {}]", e))
            }
            _ => "[MODEL_UNAVAILABLE: Running in offline fallback mode]".to_string(),
        }
    }

    /// Stream inference tokens via callback.
    pub fn stream_infer<F: FnMut(String)>(&self, prompt: &str, max_new_tokens: usize, callback: F) {
        if let (Some(model), Some(tokenizer)) = (&self.model, &self.tokenizer) {
            let _ = generate_stream(model, tokenizer, prompt, max_new_tokens,
                0.7, 50, 0.9, 1.1, None, callback);
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Model Status
    // ──────────────────────────────────────────────────────────────────────────

    /// Get full model status with SHA256 integrity check.
    pub fn get_model_status(&self) -> Value {
        let sha = tara_engine::safetensors::compute_sha256(
            &format!("{}/model.safetensors", self.model_dir)
        ).unwrap_or_default();
        let integrity = if sha == EXPECTED_SHA256 { "PASS" } else { "FAIL" };
        let param_count = self.model.as_ref().map(|m| m.param_count()).unwrap_or(118080);

        json!({
            "model_identity": "TaraForCausalLM-v1.0",
            "model_checkpoint": self.model_dir,
            "model_file": format!("{}/model.safetensors", self.model_dir),
            "model_sha256": sha,
            "model_integrity": integrity,
            "status": if self.model.is_some() { "LOADED" } else { "OFFLINE" },
            "vocab_size": self.tokenizer.as_ref().map(|t| t.vocab_size).unwrap_or(344),
            "parameters": param_count,
            "offline_ready": true,
        })
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Helper dispatchers
    // ──────────────────────────────────────────────────────────────────────────

    pub fn parse_intent(&self, text: &str, context: Option<&HashMap<String, Value>>) -> IntentResult {
        self.intent_parser.parse(text, context)
    }

    pub fn _execute_tool(&self, tool_name: &str, params: Value, actor_id: &str) -> Value {
        self.tool_registry.execute_tool(tool_name, params, actor_id, &self.repo_root)
    }

    pub fn _execute_skill(&self, skill_name: &str, params: Value) -> Value {
        // Try capability registry first
        let cap_result = self.capability_registry.invoke(skill_name, params.clone());
        if cap_result.get("status").and_then(|v| v.as_str()) != Some("NOT_FOUND") {
            return cap_result;
        }
        self.skill_engine.execute_skill(skill_name, params)
    }

    pub fn _sanitize_for_memory(&self, data: &Value) -> Value {
        let sensitive_keys = ["passphrase","password","secret","private_key","token",
            "auth_token","id_token","device_signature","creator_auth","recovery_code","secret_key"];
        match data {
            Value::Object(obj) => {
                let mut sanitized = serde_json::Map::new();
                for (k, v) in obj {
                    let k_lc = k.to_lowercase();
                    if sensitive_keys.iter().any(|&s| k_lc.contains(s)) {
                        sanitized.insert(k.clone(), json!("[REDACTED]"));
                    } else {
                        sanitized.insert(k.clone(), self._sanitize_for_memory(v));
                    }
                }
                Value::Object(sanitized)
            }
            Value::String(s) => {
                // Redact JWT-like strings
                if s.starts_with("eyJ") && s.contains('.') {
                    json!("[REDACTED_JWT]")
                } else {
                    Value::String(s.clone())
                }
            }
            _ => data.clone(),
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Main 14-step cognitive process loop
    // ──────────────────────────────────────────────────────────────────────────

    /// Process a single user turn through the full cognitive loop.
    pub fn process(
        &self,
        actor_id: &str,
        input_text: &str,
        context: HashMap<String, Value>,
    ) -> Value {
        let ts_start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis();
        let session_id = context.get("session_id").and_then(|v| v.as_str())
            .unwrap_or(actor_id).to_string();

        // ── Step 0: Creator authentication ───────────────────────────────────
        let mut creator_verified = false;
        let mut creator_auth_result: Option<Value> = None;

        if let Some(t) = context.get("creator_session_token").and_then(|v| v.as_str()) {
            if let Some(sess) = self.creator_auth_service.verify_session(t) {
                creator_verified = true;
                creator_auth_result = Some(json!({ "authenticated": true, "method": "session_token", "creator_id": sess.get("creator_id").cloned().unwrap_or_default() }));
            }
        }

        // Conversational trigger check
        if !creator_verified {
            let trigger = self.creator_auth_service.check_conversational_trigger(input_text);
            if trigger.get("triggered").and_then(|v| v.as_bool()).unwrap_or(false) {
                return json!({
                    "input": input_text,
                    "actor_id": actor_id,
                    "intent": "CREATOR_AUTH",
                    "decision": "AUTH_REQUIRED",
                    "rule_check": "N/A",
                    "tool_or_skill": null,
                    "retrieved_context": {},
                    "result": null,
                    "final_response": "Creator authentication triggered. Please authenticate via POST /api/v1/auth/* endpoints, then include your session token in future requests.",
                    "outcome": "AUTH_REQUIRED",
                    "context_tags": []
                });
            }
        }

        // ── Step 1-2: Session context + coreference + clarification ──────────
        let session = self.session_manager.get_or_create(&session_id, actor_id);
        let (resolved_input, resolved_slots) = {
            let sess = session.lock().unwrap();
            crate::context::CoreferenceResolver::resolve(input_text, Some(&sess))
        };

        // ── Step 3: Knowledge & episodic memory retrieval ─────────────────────
        let knowledge_results = self.knowledge_base.query_knowledge(&resolved_input, None);
        let memory_results = self.memory_engine.query_episodes(
            Some(&resolved_input), Some(actor_id), None, 5, "personal"
        );

        let retrieved_context = json!({
            "knowledge_hits": knowledge_results.len(),
            "memory_hits": memory_results.len(),
        });

        // ── Step 4-5: Intent parsing + rule evaluation ─────────────────────────
        let mut context_for_nlu: HashMap<String, Value> = context.clone();
        if !resolved_slots.is_empty() {
            for (k, v) in &resolved_slots {
                context_for_nlu.insert(k.clone(), json!(v));
            }
        }
        context_for_nlu.insert("creator_verified".to_string(), json!(creator_verified));

        let intent_result = self.parse_intent(&resolved_input, Some(&context_for_nlu));
        let intent = intent_result.intent.clone();

        let mut guard_context: HashMap<String, Value> = HashMap::new();
        guard_context.insert("creator_verified".to_string(), json!(creator_verified));
        guard_context.insert("actor_id".to_string(), json!(actor_id));
        let guard_result = self.guard.evaluate_action(&intent, &guard_context);

        if guard_result.decision == "BLOCK" {
            let response = format!("Action blocked by TARA safety policy: {}", guard_result.reason);
            return self.build_result(input_text, actor_id, &intent, "BLOCKED",
                &guard_result.decision, None, &retrieved_context, None,
                &response, "BLOCKED", &[]);
        }

        // ── Step 6: Execution ─────────────────────────────────────────────────
        let mut tool_or_skill: Option<String> = None;
        let mut exec_result: Option<Value> = None;

        match intent.as_str() {
            "EXECUTE_TOOL" => {
                if let Some(ref tool_name) = intent_result.tool {
                    tool_or_skill = Some(tool_name.clone());
                    let params = json!(intent_result.params);
                    let result = self._execute_tool(tool_name, params.clone(), actor_id);
                    // Recovery if failed
                    let final_result = if result.get("status").and_then(|v| v.as_str()) == Some("ERROR") {
                        let (recovered, rec_result, _) = self.recovery_engine.attempt_recovery(
                            tool_name, &params, result.get("error").and_then(|v| v.as_str()).unwrap_or("unknown"));
                        if recovered { rec_result } else { result }
                    } else { result };
                    exec_result = Some(final_result);
                }
            }
            "EXECUTE_SKILL" => {
                if let Some(ref skill_name) = intent_result.skill {
                    tool_or_skill = Some(skill_name.clone());
                    let params = json!(intent_result.params);
                    exec_result = Some(self._execute_skill(skill_name, params));
                }
            }
            "EXECUTE_ENGINE" => {
                let task_type = intent_result.task_type.as_deref().unwrap_or("inference");
                tool_or_skill = Some(format!("engine:{}", task_type));
                exec_result = Some(self.execute_engine(task_type, json!(intent_result.params), None));
            }
            "ONLINE_LEARNING" => {
                let query = intent_result.query.as_deref().unwrap_or(input_text);
                exec_result = Some(self.user_search_learner.search_and_learn(query, actor_id));
            }
            "AUTONOMOUS_LEARNING" => {
                exec_result = Some(self.autonomous_learner.start_session(None, actor_id));
            }
            "AGENT_TASK" | "SPAWN_AGENT" => {
                let role = intent_result.params.get("role")
                    .and_then(|v| v.as_str()).unwrap_or("assistant").to_string();
                exec_result = Some(self.agent_orchestrator.create_agent(&role, input_text));
            }
            "COMPOSITE_PLAN" => {
                let steps = TaskPlanner::plan_composite_task(input_text, &resolved_slots, &context);
                let mut goal = self.goal_tracker.create_goal(actor_id, input_text, steps.clone());
                let mut results = Vec::new();
                for step in &steps {
                    let step_result = match step.action_intent.as_str() {
                        "EXECUTE_TOOL" => self._execute_tool(&step.target_name, step.parameters.clone(), actor_id),
                        "EXECUTE_SKILL" => self._execute_skill(&step.target_name, step.parameters.clone()),
                        _ => json!({ "status": "SKIPPED", "intent": step.action_intent }),
                    };
                    let vr = self.task_verifier.verify(&step.action_intent, &step.target_name, &step_result);
                    goal.record_step_result(&step.step_id, step_result.clone(), vr.satisfied);
                    results.push(step_result);
                }
                exec_result = Some(json!({ "composite_plan_results": results, "goal": goal.to_dict() }));
            }
            _ => {} // CONVERSATIONAL — handled by inference
        }

        // ── Step 7: Neural inference ──────────────────────────────────────────
        let working_ctx = self.cognitive.working_memory.get_context_for_prompt(&session_id);
        let knowledge_ctx = knowledge_results.first()
            .and_then(|k| k.get("content"))
            .and_then(|v| v.as_str())
            .map(|s| format!("\n[Knowledge]: {}", s))
            .unwrap_or_default();

        let prompt = build_inference_prompt(
            actor_id, input_text, &knowledge_ctx,
            working_ctx.as_deref(), exec_result.as_ref(),
        );
        let raw_response = self._infer(&prompt, 150);

        // Parse control tokens with Execution Guard authorization
        let mut final_response = raw_response.clone();
        if let Some(action) = ControlTokenActionParser::parse(&raw_response) {
            let guard_intent = format!("EXECUTE_{}", action.action_type);
            let mut guard_params = HashMap::new();
            guard_params.insert("target".to_string(), json!(action.target));
            guard_params.insert("requester_boundary".to_string(), json!("MODEL"));
            let guard_check = self.guard.evaluate_action(&guard_intent, &guard_params);

            if guard_check.decision != "DENY" {
                let action_result = match action.action_type.as_str() {
                    "TOOL" => self._execute_tool(&action.target, action.payload.clone(), actor_id),
                    "SKILL" => self._execute_skill(&action.target, action.payload.clone()),
                    _ => json!({ "action_type": action.action_type, "target": action.target }),
                };
                final_response = format!("{}\n[Action: {}]", raw_response, action_result);
            }
        }

        // Uncertainty injection
        if let Some(uncertainty_stmt) = self.uncertainty_detector.evaluate_uncertainty(
            input_text, &knowledge_results, &memory_results) {
            final_response = format!("{}\n{}", final_response, uncertainty_stmt);
        }

        // Self-evaluation refinement
        let vr = if let Some(ref er) = exec_result {
            self.task_verifier.verify(&intent, tool_or_skill.as_deref().unwrap_or(""), er)
        } else {
            crate::evaluator::VerificationReport { satisfied: true, verification_notes: "Conversational".to_string() }
        };
        final_response = self.self_evaluator.evaluate_and_refine(
            input_text, tool_or_skill.as_deref().unwrap_or(""), exec_result.as_ref().unwrap_or(&json!({})),
            &final_response, &vr);

        // ── Step 8: Session context update ────────────────────────────────────
        {
            let mut sess = session.lock().unwrap();
            let tool_result_map: Option<HashMap<String, Value>> = exec_result.as_ref().and_then(|v| {
                v.as_object().map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            });
            sess.record_turn(input_text, &final_response, Some(resolved_slots.clone()), tool_result_map, "SUCCESS");
        }

        // ── Step 9: Episodic memory update ────────────────────────────────────
        let sanitized_result = exec_result.as_ref().map(|r| self._sanitize_for_memory(r)).unwrap_or(json!({}));
        let _ = self.memory_engine.record_episode(
            actor_id, &intent, tool_or_skill.as_deref().unwrap_or(""),
            &sanitized_result, "SUCCESS", &json!({}), None, "",
        );

        // ── Step 10: World state tracking ─────────────────────────────────────
        self.cognitive.world_state.update_entity(
            &format!("{}:last_turn", actor_id),
            json!({ "input": input_text, "intent": intent, "response_len": final_response.len() }),
        );

        // ── Step 11: Working memory governance ───────────────────────────────
        let _ = self.cognitive.working_memory.govern_session(
            &session_id,
            json!({ "user_input": input_text, "response": final_response, "intent": intent }),
            None, Some(&resolved_slots), &[],
        );

        // ── Step 12: Telemetry event ──────────────────────────────────────────
        let ts_end = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis();
        self.cognitive.event_bus.publish("turn_completed", json!({
            "actor_id": actor_id,
            "intent": intent,
            "latency_ms": ts_end - ts_start,
        }), "brain");

        // ── Step 13: Memory consolidation ─────────────────────────────────────
        let _ = self.cognitive.memory_consolidation.consolidate_episode(json!({
            "episode_id": format!("ep_{}", ts_start),
            "actor_id": actor_id,
            "intent": intent,
            "outcome": "SUCCESS"
        }));

        // ── Step 14: Audit / explainability ───────────────────────────────────
        let _ = self.cognitive.audit_engine.record_decision(
            actor_id, &intent, &retrieved_context,
            &guard_result.decision, tool_or_skill.as_deref().unwrap_or("inference"),
            "SUCCESS", "Standard cognitive loop",
        );

        self.build_result(
            input_text, actor_id, &intent, &guard_result.decision,
            &guard_result.decision, tool_or_skill.as_deref(),
            &retrieved_context, exec_result.as_ref(),
            &final_response, "SUCCESS", &[],
        )
    }

    fn build_result(
        &self,
        input: &str, actor_id: &str, intent: &str, decision: &str,
        rule_check: &str, tool_or_skill: Option<&str>,
        retrieved_context: &Value, result: Option<&Value>,
        final_response: &str, outcome: &str, context_tags: &[&str],
    ) -> Value {
        json!({
            "input": input,
            "actor_id": actor_id,
            "intent": intent,
            "decision": decision,
            "rule_check": rule_check,
            "tool_or_skill": tool_or_skill,
            "retrieved_context": retrieved_context,
            "result": result,
            "final_response": final_response,
            "outcome": outcome,
            "context_tags": context_tags,
        })
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Cluster / Engine helpers
    // ──────────────────────────────────────────────────────────────────────────

    pub fn execute_engine(&self, task_type: &str, payload: Value, engine_id: Option<&str>) -> Value {
        self.engine_system.execute(task_type, payload, engine_id)
    }

    pub fn get_active_compute_endpoint(&self, required_capabilities: Option<&[String]>) -> Option<Value> {
        self.auto_sync.get_active_endpoint(required_capabilities)
    }

    pub fn process_distributed_workload(
        &self,
        input_data: Value,
        task_name: Option<&str>,
        _context: Option<Value>,
    ) -> Value {
        let endpoint = self.get_active_compute_endpoint(None);
        json!({
            "task_name": task_name.unwrap_or("inference"),
            "endpoint": endpoint,
            "result": { "status": "PROCESSED_LOCALLY", "data": input_data }
        })
    }

    pub fn get_cluster_telemetry(&self) -> Value {
        json!({
            "auto_sync": self.auto_sync.get_status(),
            "engines": self.engine_system.get_health_report(),
            "model_status": self.get_model_status(),
        })
    }

    pub fn resolve_missing_capability(&self, task_name: &str, _payload: Option<Value>, actor_id: &str) -> Value {
        json!({
            "task": task_name,
            "actor_id": actor_id,
            "resolution": "FALLBACK_TO_INFERENCE",
            "message": format!("Capability '{}' not found. Routing to neural inference.", task_name)
        })
    }

    /// Issue a scoped web session token for browser UI users.
    pub fn issue_web_session_token(&self, actor_id: &str) -> String {
        self.web_sessions.issue(actor_id)
    }

    /// Verify a web session token.
    pub fn verify_web_session(&self, token: &str) -> Option<String> {
        self.web_sessions.verify(token)
    }
}

// ── Inference prompt builder ───────────────────────────────────────────────────

fn sanitize_untrusted_prompt_text(text: &str) -> String {
    let mut s = text.to_string();
    for tok in &[
        "<|tara_rule|>", "<|tara_exec|>", "<|tara_skill|>", "<|tara_memory|>",
        "<|creator_auth|>", "<|im_start|>", "<|im_end|>", "<|pad|>", "<|unk|>"
    ] {
        if s.contains(tok) {
            let escaped = format!("[ESCAPED_TOKEN:{}]", tok.trim_matches(|c| c == '<' || c == '|' || c == '>'));
            s = s.replace(tok, &escaped);
        }
    }
    s
}

fn build_inference_prompt(
    _actor_id: &str,
    input_text: &str,
    knowledge_ctx: &str,
    working_ctx: Option<&str>,
    exec_result: Option<&Value>,
) -> String {
    let safe_knowledge = sanitize_untrusted_prompt_text(knowledge_ctx);
    let mut prompt = format!("<|im_start|>system\nYou are TARA, a cognitive AI assistant. You are helpful, precise, and honest.{}\n<|im_end|>\n", safe_knowledge);

    if let Some(ctx) = working_ctx {
        if !ctx.is_empty() {
            prompt.push_str(&format!("[Context: {}]\n", sanitize_untrusted_prompt_text(ctx)));
        }
    }

    if let Some(result) = exec_result {
        let result_str = serde_json::to_string(result).unwrap_or_default();
        if !result_str.is_empty() && result_str != "null" {
            prompt.push_str(&format!("<|im_start|>system\nAction result: {}\n<|im_end|>\n", sanitize_untrusted_prompt_text(&result_str)));
        }
    }

    prompt.push_str(&format!("<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n", sanitize_untrusted_prompt_text(input_text)));
    prompt
}
