//! Cognitive capabilities hub: world state, events, working memory, consolidation.

pub mod affect;
pub mod connection;
pub mod creativity;
pub mod curiosity;
#[path = "../../../tara_training_system/world_model/deliberative_search.rs"]
pub mod deliberative_search;
pub mod existential;
#[path = "../../../tara_training_system/autonomous_training/world_model/experiential.rs"]
pub mod experiential;
pub mod fast_weights;
#[path = "../../../tara_training_system/world_model/ontology.rs"]
pub mod ontology;
#[path = "../../../tara_training_system/world_model/reasoning.rs"]
pub mod reasoning;
pub mod self_model;
#[path = "../../../tara_training_system/world_model/world_state.rs"]
pub mod world_state;

pub use affect::{AffectiveHomeostasisEngine, ComputationalAffectState};
pub use connection::{ConnectionManager, UserConnectionProfile};
pub use creativity::{ConceptualFrame, CreativityEngine, EmergentBlend};
pub use curiosity::{
    CuriosityDriveEngine, EpistemicEntropyEvaluator, EpistemicState, ExplorationGoal,
};
pub use deliberative_search::{DeliberativeTreeSearchEngine, SearchNode};
pub use existential::{DialecticalReasoningResult, ExistentialReasoningEngine, MetacognitiveTopic};
pub use experiential::{
    ExperienceEpisode, ExperientialLearningEngine, ExperientialStage, LessonLearned,
};
pub use fast_weights::FastWeightPlasticityMatrix;
pub use ontology::{ConceptNode, OntologyEdge, OntologyEngine, OntologyRelation, SemanticPath};
pub use reasoning::{
    ConstraintSolver, MultiObjectiveOptimizer, ProbabilisticReasoningEngine,
    UncertaintyAwarePlanner,
};
pub use self_model::{CognitiveSelfState, EpistemicBoundary, SelfModelEngine};
pub use world_state::WorldStateTracker;

use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::sync::{Arc, Mutex};

// ─── Authoritative defaults for CognitiveCapabilitiesHub ─────────────────────
// Connected directly to subsystem definitions — single source of truth.

/// Maximum conversation turns held in working memory per session.
pub const DEFAULT_WORKING_MEMORY_TURNS: usize = 10;

/// UCB1 exploration constant for deliberative tree search (MCTS).
pub const DEFAULT_MCTS_EXPLORATION_C: f64 = deliberative_search::DEFAULT_C_PUCT;

/// Maximum number of tree-search nodes expanded per MCTS pass.
pub const DEFAULT_MCTS_MAX_NODES: usize = deliberative_search::DEFAULT_MAX_ITERATIONS;

/// Dimensionality of the fast-weights (Hebbian) associative memory matrix.
pub const DEFAULT_FAST_WEIGHT_DIM: usize = fast_weights::DEFAULT_DIM;

/// Per-step exponential decay factor for fast-weights plasticity.
pub const DEFAULT_FAST_WEIGHT_DECAY: f32 = fast_weights::DEFAULT_LAMBDA;

/// Hebbian learning rate for fast-weights write operations.
pub const DEFAULT_FAST_WEIGHT_LR: f32 = fast_weights::DEFAULT_ETA;

/// Number of most-recent turns assembled into the prompt context window.
pub const DEFAULT_CONTEXT_WINDOW_TURNS: usize = 3;
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the CognitiveCapabilitiesHub.
///
/// All numeric parameters live here so they can be driven from a configuration
/// file at startup — never embedded as magic literals in construction code.
/// The constants above are the single authoritative defaults; `Default::default()`
/// reads from them and nowhere else.
#[derive(Debug, Clone)]
pub struct CognitiveHubConfig {
    /// Maximum conversation turns retained in working memory per session.
    pub max_working_memory_turns: usize,
    /// UCB1 exploration constant for the deliberative tree search (MCTS).
    pub mcts_exploration_constant: f64,
    /// Maximum search-tree nodes before pruning in a single MCTS pass.
    pub mcts_max_nodes: usize,
    /// Dimensionality of the fast-weights associative memory matrix.
    pub fast_weight_dim: usize,
    /// Exponential decay factor for the fast-weights plasticity matrix.
    pub fast_weight_decay: f32,
    /// Hebbian learning rate for fast-weights write operations.
    pub fast_weight_learning_rate: f32,
    /// Number of recent turns included in the prompt context window.
    pub context_window_turns: usize,
}

impl Default for CognitiveHubConfig {
    /// Constructs configuration using dynamic runtime parameters, falling back to authoritative defaults.
    fn default() -> Self {
        Self {
            max_working_memory_turns: std::env::var("TARA_WORKING_MEMORY_TURNS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_WORKING_MEMORY_TURNS),
            mcts_exploration_constant: std::env::var("TARA_MCTS_EXPLORATION_C")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_MCTS_EXPLORATION_C),
            mcts_max_nodes: std::env::var("TARA_MCTS_MAX_NODES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_MCTS_MAX_NODES),
            fast_weight_dim: std::env::var("TARA_FAST_WEIGHT_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_FAST_WEIGHT_DIM),
            fast_weight_decay: std::env::var("TARA_FAST_WEIGHT_DECAY")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_FAST_WEIGHT_DECAY),
            fast_weight_learning_rate: std::env::var("TARA_FAST_WEIGHT_LR")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_FAST_WEIGHT_LR),
            context_window_turns: std::env::var("TARA_CONTEXT_WINDOW_TURNS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_CONTEXT_WINDOW_TURNS),
        }
    }
}

pub type EventSubscriber = Box<dyn Fn(Value) + Send + Sync>;
pub type SubscriberMap = Arc<Mutex<HashMap<String, Vec<EventSubscriber>>>>;

/// Event bus for intra-system pub/sub.
pub struct EventBus {
    subscribers: SubscriberMap,
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn publish(&self, event: &str, data: Value, _source: &str) {
        let subs = self.subscribers.lock().unwrap();
        if let Some(handlers) = subs.get(event) {
            for handler in handlers {
                handler(data.clone());
            }
        }
    }

    pub fn subscribe<F: Fn(Value) + Send + Sync + 'static>(&self, event: &str, handler: F) {
        self.subscribers
            .lock()
            .unwrap()
            .entry(event.to_string())
            .or_default()
            .push(Box::new(handler));
    }
}

/// Governs working memory across sessions.
pub struct WorkingMemoryGovernor {
    sessions: Arc<Mutex<HashMap<String, Vec<Value>>>>,
    max_turns: usize,
    /// Number of most-recent turns to include in prompt context assembly.
    context_window_turns: usize,
}

impl WorkingMemoryGovernor {
    pub fn new(max_turns: usize) -> Self {
        // Default context window is the full working memory size (no separate truncation).
        Self::with_context_window(max_turns, max_turns)
    }

    /// Create with an explicit context window size for prompt assembly.
    pub fn with_context_window(max_turns: usize, context_window_turns: usize) -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            max_turns,
            context_window_turns,
        }
    }

    pub fn govern_session(
        &self,
        session_id: &str,
        turn: Value,
        active_goal: Option<&str>,
        active_slots: Option<&HashMap<String, String>>,
        security_verdicts: &[Value],
    ) -> Value {
        let mut turn = turn;
        if !security_verdicts.is_empty() {
            if let Some(object) = turn.as_object_mut() {
                object.insert("security_verdicts".into(), json!(security_verdicts));
            } else {
                turn = json!({"turn": turn, "security_verdicts": security_verdicts});
            }
        }
        let mut sessions = self.sessions.lock().unwrap();
        let history = sessions.entry(session_id.to_string()).or_default();
        history.push(turn);
        if history.len() > self.max_turns {
            history.drain(0..history.len() - self.max_turns);
        }
        json!({
            "session_id": session_id,
            "turns_in_memory": history.len(),
            "active_goal": active_goal,
            "active_slots": active_slots.map(|s| json!(s)).unwrap_or(Value::Null)
        })
    }

    pub fn get_context_for_prompt(&self, session_id: &str) -> Option<String> {
        let sessions = self.sessions.lock().unwrap();
        if let Some(history) = sessions.get(session_id) {
            if history.is_empty() {
                return None;
            }
            let context: Vec<String> = history
                .iter()
                .rev()
                .take(self.context_window_turns)
                .map(|t| {
                    format!(
                        "User: {}\nTARA: {}",
                        t.get("user_input").and_then(|v| v.as_str()).unwrap_or(""),
                        t.get("response").and_then(|v| v.as_str()).unwrap_or("")
                    )
                })
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            Some(context.join("\n"))
        } else {
            None
        }
    }
}

/// Consolidates episodic memories into semantic knowledge.
pub struct MemoryConsolidationEngine {
    pub repo_root: String,
}

impl MemoryConsolidationEngine {
    pub fn new(repo_root: &str) -> Self {
        Self {
            repo_root: repo_root.to_string(),
        }
    }

    pub fn consolidate_episode(&self, episode: Value) -> Result<(), std::io::Error> {
        let dir = format!("{}/storage/memory/consolidated", self.repo_root);
        fs::create_dir_all(&dir)?;
        let id = episode
            .get("episode_id")
            .and_then(|v| v.as_str())
            .unwrap_or("ep");
        let path = format!("{}/{}.json", dir, id);
        fs::write(
            &path,
            serde_json::to_string_pretty(&episode).unwrap_or_default(),
        )?;
        Ok(())
    }
}

/// Prediction error loop: attempts actions and learns from mismatches.
pub struct PredictionErrorLoop {
    pub repo_root: String,
}

impl PredictionErrorLoop {
    pub fn new(repo_root: &str) -> Self {
        Self {
            repo_root: repo_root.to_string(),
        }
    }

    pub fn execute_and_learn_cycle<F>(&self, task_name: &str, action: F, expectation: &str) -> Value
    where
        F: FnOnce() -> Value,
    {
        let actual = action();
        let actual_str = actual.to_string();
        let match_score = if actual_str
            .to_lowercase()
            .contains(&expectation.to_lowercase())
        {
            1.0
        } else {
            0.0
        };
        json!({
            "task": task_name,
            "expectation": expectation,
            "match_score": match_score,
            "actual": actual
        })
    }
}

/// Audit and explainability engine.
pub struct AuditExplainabilityEngine {
    pub repo_root: String,
}

pub struct DecisionRecordParams<'a> {
    pub actor_id: &'a str,
    pub intent: &'a str,
    pub evidence: &'a Value,
    pub selected_strategy: &'a str,
    pub action: &'a str,
    pub outcome: &'a str,
    pub rationale: &'a str,
}

impl AuditExplainabilityEngine {
    pub fn new(repo_root: &str) -> Self {
        Self {
            repo_root: repo_root.to_string(),
        }
    }

    pub fn record_decision(&self, p: DecisionRecordParams<'_>) -> Result<(), std::io::Error> {
        let dir = format!("{}/storage/audit", self.repo_root);
        fs::create_dir_all(&dir)?;
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let record = json!({
            "ts": ts, "actor_id": p.actor_id, "intent": p.intent,
            "evidence": p.evidence, "strategy": p.selected_strategy,
            "action": p.action, "outcome": p.outcome, "rationale": p.rationale
        });
        let path = format!("{}/audit_{}.json", dir, ts);
        fs::write(
            &path,
            serde_json::to_string_pretty(&record).unwrap_or_default(),
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UnifiedAwarenessSnapshot {
    pub session_id: String,
    pub topic: String,
    pub system_healthy: bool,
    pub known_concepts_ratio: f64,
    pub epistemic_entropy: f64,
    pub affective_state: ComputationalAffectState,
    pub user_depth_preference: String,
    pub self_calibration_confidence: f64,
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IntuitionVerdict {
    pub mode: String, // "INTUITIVE_FAST_PATH" or "DELIBERATIVE_TREE_SEARCH"
    pub confidence: f32,
    pub retrieved_association_norm: f32,
    pub rationale: String,
}

/// The aggregated cognitive capabilities hub.
pub struct CognitiveCapabilitiesHub {
    pub world_state: WorldStateTracker,
    pub event_bus: EventBus,
    pub working_memory: WorkingMemoryGovernor,
    pub memory_consolidation: MemoryConsolidationEngine,
    pub prediction_error_loop: PredictionErrorLoop,
    pub audit_engine: AuditExplainabilityEngine,
    pub mcts: Arc<Mutex<DeliberativeTreeSearchEngine>>,
    pub fast_weights: Arc<Mutex<FastWeightPlasticityMatrix>>,
    pub curiosity: Arc<CuriosityDriveEngine>,
    pub experiential: Arc<ExperientialLearningEngine>,
    pub ontology: Arc<OntologyEngine>,
    pub constraint_solver: Arc<ConstraintSolver>,
    pub multi_objective: Arc<MultiObjectiveOptimizer>,
    pub probabilistic_reasoning: Arc<Mutex<ProbabilisticReasoningEngine>>,
    pub uncertainty_planner: Arc<UncertaintyAwarePlanner>,
    // Newly integrated canonical cognitive engines
    pub self_model: Arc<SelfModelEngine>,
    pub affect: Arc<AffectiveHomeostasisEngine>,
    pub creativity: Arc<CreativityEngine>,
    pub connection: Arc<ConnectionManager>,
    pub existential: Arc<ExistentialReasoningEngine>,
}

impl CognitiveCapabilitiesHub {
    /// Create a hub with default configuration parameters.
    pub fn new(repo_root: &str) -> Self {
        Self::with_config(repo_root, CognitiveHubConfig::default())
    }

    /// Create a hub with explicit configuration.
    ///
    /// All numeric parameters are driven from `config` — no magic literals embedded here.
    pub fn with_config(repo_root: &str, config: CognitiveHubConfig) -> Self {
        Self {
            world_state: WorldStateTracker::new(),
            event_bus: EventBus::new(),
            working_memory: WorkingMemoryGovernor::with_context_window(
                config.max_working_memory_turns,
                config.context_window_turns,
            ),
            memory_consolidation: MemoryConsolidationEngine::new(repo_root),
            prediction_error_loop: PredictionErrorLoop::new(repo_root),
            audit_engine: AuditExplainabilityEngine::new(repo_root),
            mcts: Arc::new(Mutex::new(DeliberativeTreeSearchEngine::new(
                config.mcts_exploration_constant,
                config.mcts_max_nodes,
            ))),
            fast_weights: Arc::new(Mutex::new(FastWeightPlasticityMatrix::new(
                config.fast_weight_dim,
                config.fast_weight_decay,
                config.fast_weight_learning_rate,
            ))),
            curiosity: Arc::new(CuriosityDriveEngine::new()),
            experiential: Arc::new(ExperientialLearningEngine::new(repo_root)),
            ontology: Arc::new(OntologyEngine::new()),
            constraint_solver: Arc::new(ConstraintSolver::new()),
            multi_objective: Arc::new(MultiObjectiveOptimizer::new()),
            probabilistic_reasoning: Arc::new(Mutex::new(ProbabilisticReasoningEngine::new())),
            uncertainty_planner: Arc::new(UncertaintyAwarePlanner::new()),
            self_model: Arc::new(SelfModelEngine::new(repo_root)),
            affect: Arc::new(AffectiveHomeostasisEngine::new()),
            creativity: Arc::new(CreativityEngine::new(repo_root)),
            connection: Arc::new(ConnectionManager::new(repo_root)),
            existential: Arc::new(ExistentialReasoningEngine::new()),
        }
    }

    /// Evaluates unified cognitive awareness:
    /// Aggregates environment/device health, epistemic uncertainty (entropy),
    /// homeostatic computational affect, user connection profile, and self-state.
    pub fn evaluate_unified_awareness(
        &self,
        session_id: &str,
        topic: &str,
    ) -> UnifiedAwarenessSnapshot {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let epistemic = self.curiosity.evaluate_epistemic_state(topic, &[]);
        let affect_st = self.affect.get_state();
        let user_prof = self.connection.get_or_create_profile(session_id);
        let self_st = self.self_model.introspect();

        let total = (epistemic.known_concepts_count + epistemic.unknown_concepts_count).max(1);
        let known_ratio = epistemic.known_concepts_count as f64 / total as f64;

        UnifiedAwarenessSnapshot {
            session_id: session_id.to_string(),
            topic: topic.to_string(),
            system_healthy: self_st.operational_status == "NOMINAL_ACTIVE",
            known_concepts_ratio: (known_ratio * 100.0).round() / 100.0,
            epistemic_entropy: (epistemic.epistemic_entropy * 100.0).round() / 100.0,
            affective_state: affect_st,
            user_depth_preference: user_prof.technical_depth,
            self_calibration_confidence: self_st.calibration_confidence,
            timestamp_ms: ts,
        }
    }

    /// Dual-Process Arbitration (System 1: Intuition vs System 2: Deliberative Tree Search).
    /// Tests fast-weight associative pattern memory. If associative recall norm/confidence
    /// exceeds threshold and epistemic tension is low, selects fast intuitive path;
    /// otherwise triggers deliberative tree search.
    pub fn arbitrate_intuition_vs_deliberation(
        &self,
        query_vec: &[f32],
        confidence_threshold: f32,
    ) -> IntuitionVerdict {
        let assoc = self
            .fast_weights
            .lock()
            .unwrap()
            .read_association(query_vec);
        let norm: f32 = assoc.iter().map(|x| x * x).sum::<f32>().sqrt();
        let affect_st = self.affect.get_state();

        // High epistemic tension forces deliberative path even if heuristic norm is moderate
        let effective_confidence = if affect_st.epistemic_tension > 0.6 {
            norm * 0.7
        } else {
            norm
        };

        if effective_confidence >= confidence_threshold {
            IntuitionVerdict {
                mode: "INTUITIVE_FAST_PATH".to_string(),
                confidence: (effective_confidence * 100.0).round() / 100.0,
                retrieved_association_norm: (norm * 100.0).round() / 100.0,
                rationale: "Associative fast-weights memory returned high confidence match; bypassing slow tree search".to_string(),
            }
        } else {
            IntuitionVerdict {
                mode: "DELIBERATIVE_TREE_SEARCH".to_string(),
                confidence: (effective_confidence * 100.0).round() / 100.0,
                retrieved_association_norm: (norm * 100.0).round() / 100.0,
                rationale: "Low associative familiarity or high epistemic tension detected; engaging deliberative tree search".to_string(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognitive_hub_awareness_and_intuition() {
        let tmp = std::env::temp_dir().join(format!("tara_hub_test_{}", uuid::Uuid::new_v4()));
        let hub = CognitiveCapabilitiesHub::new(tmp.to_str().unwrap());

        // 1. Evaluate unified awareness
        let awareness = hub.evaluate_unified_awareness("session_operator", "mathematics");
        assert_eq!(awareness.session_id, "session_operator");
        assert!(awareness.system_healthy);
        assert_eq!(awareness.user_depth_preference, "INTERMEDIATE");

        // 2. Test intuition arbitration when weights are empty (should choose Deliberative)
        let q = vec![1.0f32; 64];
        let verdict_empty = hub.arbitrate_intuition_vs_deliberation(&q, 0.5);
        assert_eq!(verdict_empty.mode, "DELIBERATIVE_TREE_SEARCH");

        // 3. Train fast-weights association (write pattern)
        {
            let mut fw = hub.fast_weights.lock().unwrap();
            fw.write_association(&q, &q);
        }

        // Test intuition arbitration with learned pattern (should choose Intuitive Fast Path)
        let verdict_learned = hub.arbitrate_intuition_vs_deliberation(&q, 0.5);
        assert_eq!(verdict_learned.mode, "INTUITIVE_FAST_PATH");

        let _ = fs::remove_dir_all(&tmp);
    }
}
