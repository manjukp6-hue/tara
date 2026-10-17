//! Cognitive capabilities hub: world state, events, working memory, consolidation.

pub mod affect;
pub mod connection;
pub mod context;
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
pub mod planner;
#[path = "../../../tara_training_system/world_model/reasoning.rs"]
pub mod reasoning;
pub mod self_model;
#[path = "../../../tara_training_system/world_model/world_state.rs"]
pub mod world_state;

pub use context::*;
pub use planner::*;

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
pub use fast_weights::{FastWeightMemory, FastWeightPlasticityMatrix};
pub use ontology::{ConceptNode, OntologyEdge, OntologyEngine, OntologyRelation, SemanticPath};
pub use reasoning::{
    ConstraintSolver, MultiObjectiveOptimizer, ProbabilisticReasoningEngine,
    UncertaintyAwarePlanner,
};
pub use self_model::{CognitiveSelfState, EpistemicBoundary, EpistemicCategory, SelfModelEngine};
pub use world_state::WorldStateTracker;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

static AUDIT_HMAC_KEY: LazyLock<[u8; 32]> = LazyLock::new(|| {
    tara_security::dpapi::derive_authority_hmac_key("audit_log_chain")
        .expect("Root authority HMAC key derivation for audit_log_chain must succeed")
});

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

pub type EventSubscriber = Arc<dyn Fn(Value) + Send + Sync>;
pub type SubscriberMap = Arc<Mutex<HashMap<String, Vec<EventSubscriber>>>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum EventType {
    SecurityEvent,
    CognitiveEvent,
    SystemAlert,
    MemoryUpdate,
    UserAction,
}

impl EventType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SecurityEvent => "security:verdict",
            Self::CognitiveEvent => "cognitive:insight",
            Self::SystemAlert => "system:alert",
            Self::MemoryUpdate => "memory:update",
            Self::UserAction => "user:action",
        }
    }
}

impl std::str::FromStr for EventType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "security:verdict" | "security:alert" => Ok(Self::SecurityEvent),
            "cognitive:insight" => Ok(Self::CognitiveEvent),
            "system:alert" => Ok(Self::SystemAlert),
            "memory:update" => Ok(Self::MemoryUpdate),
            "user:action" => Ok(Self::UserAction),
            _ => Err(format!("Unknown or unauthorized event type: {}", s)),
        }
    }
}

/// Event bus for intra-system pub/sub.
/// Callbacks are executed OUTSIDE locks to prevent deadlocks and lock poisoning.
pub struct EventBus {
    subscribers: SubscriberMap,
    journal_path: Option<std::path::PathBuf>,
    event_log: Arc<Mutex<Vec<Value>>>,
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
            journal_path: None,
            event_log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn with_journal(journal_path: std::path::PathBuf) -> Self {
        Self {
            subscribers: Arc::new(Mutex::new(HashMap::new())),
            journal_path: Some(journal_path),
            event_log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn journal_path(&self) -> Option<&Path> {
        self.journal_path.as_deref()
    }

    /// Publish an event to registered subscribers with payload size bounds and panic isolation.
    pub fn publish(&self, event: &str, data: Value, source: &str) {
        // Enforce payload size limit (max 64 KiB)
        if serde_json::to_vec(&data).map(|v| v.len()).unwrap_or(0) > 65_536 {
            return;
        }

        // Enforce namespace protection: untrusted sources cannot publish to reserved namespaces
        if (event.starts_with("security:")
            || event.starts_with("system:")
            || event.starts_with("auth:"))
            && source != "core_system"
            && source != "execution_guard"
        {
            return;
        }

        let handlers = {
            let subs = self.subscribers.lock().unwrap_or_else(|p| p.into_inner());
            subs.get(event).cloned()
        };

        {
            let mut log = self.event_log.lock().unwrap_or_else(|p| p.into_inner());
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            log.push(json!({
                "event": event,
                "data": data,
                "source": source,
                "timestamp_ms": timestamp,
            }));
            if log.len() > 10_000 {
                let overflow = log.len() - 10_000;
                log.drain(0..overflow);
            }
        }

        if let Some(list) = handlers {
            for handler in list {
                // Isolate subscriber panics via catch_unwind (C-346)
                let d = data.clone();
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                    handler(d);
                }));
            }
        }
    }

    /// Publish security event requiring verified Creator principal.
    pub fn publish_security(
        &self,
        principal: &tara_security::primitives::AuthenticatedPrincipal,
        event: &str,
        data: Value,
    ) -> Result<(), String> {
        if !principal.is_creator() {
            return Err(
                "AuthorizationDenied: Security event publish requires creator authority".into(),
            );
        }
        self.publish(event, data, "core_system");
        Ok(())
    }

    /// Publish verifying that source identity matches publisher principal.
    pub fn publish_with_source(
        &self,
        principal: &tara_security::primitives::AuthenticatedPrincipal,
        event: &str,
        data: Value,
        source: &str,
    ) -> Result<(), String> {
        if principal.actor_id() != source && !principal.is_creator() {
            return Err("SourceMismatch: Spoofed source identity does not match principal".into());
        }
        self.publish(event, data, source);
        Ok(())
    }

    /// Subscribe to events with bounded subscriber count (<= 256 total subscriptions).
    pub fn subscribe<F: Fn(Value) + Send + Sync + 'static>(
        &self,
        event: &str,
        handler: F,
    ) -> Result<(), String> {
        let mut subs = self.subscribers.lock().unwrap_or_else(|p| p.into_inner());
        let total: usize = subs.values().map(|v| v.len()).sum();
        if total >= 256 {
            return Err("SubscriberLimitExceeded: Maximum 256 active subscribers enforced".into());
        }
        subs.entry(event.to_string())
            .or_default()
            .push(Arc::new(handler));
        Ok(())
    }

    pub fn persist_journal(&self, path: &Path) -> Result<(), String> {
        let events = self
            .event_log
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        let count = events.len();
        let journal_record = json!({
            "status": "JOURNAL_COMMITTED",
            "events_count": count,
            "events": events,
            "timestamp": crate::now_millis()
        });
        let serialized = serde_json::to_vec_pretty(&journal_record).map_err(|e| e.to_string())?;
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let temp = path.with_extension(format!("tmp.{}", uuid::Uuid::new_v4()));
        let mut file = File::create(&temp).map_err(|e| e.to_string())?;
        file.write_all(&serialized).map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        drop(file);
        fs::rename(&temp, path).map_err(|e| e.to_string())?;
        let domain = format!(
            "event_journal_{}",
            &hex::encode(sha2::Sha256::digest(path.to_string_lossy().as_bytes()))[..16]
        );
        let _ = tara_security::dpapi::store_os_monotonic_watermark(&domain, count as u64);
        Ok(())
    }

    pub fn load_journal(&self, path: &Path) -> Result<usize, String> {
        if !path.exists() {
            return Ok(0);
        }
        let domain = format!(
            "event_journal_{}",
            &hex::encode(sha2::Sha256::digest(path.to_string_lossy().as_bytes()))[..16]
        );
        let raw = fs::read_to_string(path).map_err(|e| e.to_string())?;
        let val: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
        let count = val
            .get("events_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        if let Ok(Some(committed_watermark)) =
            tara_security::dpapi::load_os_monotonic_watermark(&domain)
        {
            if count < committed_watermark {
                return Err(format!(
                    "RollbackAttackDetected: journal count {} is less than committed watermark {}",
                    count, committed_watermark
                ));
            }
        }

        if let Some(events_arr) = val.get("events").and_then(|v| v.as_array()) {
            let mut log = self.event_log.lock().unwrap_or_else(|p| p.into_inner());
            *log = events_arr.clone();
        }

        Ok(count as usize)
    }
}

/// Governs working memory across sessions with bounds and memory protection.
pub struct WorkingMemoryGovernor {
    sessions: Arc<Mutex<HashMap<String, Vec<Value>>>>,
    max_turns: usize,
    context_window_turns: usize,
    max_sessions: usize,
}

impl WorkingMemoryGovernor {
    pub fn new(max_turns: usize) -> Self {
        Self::with_context_window(max_turns, max_turns)
    }

    pub fn with_context_window(max_turns: usize, context_window_turns: usize) -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            max_turns: max_turns.clamp(1, 100),
            context_window_turns: context_window_turns.clamp(1, 50),
            max_sessions: 5000,
        }
    }

    fn sanitize_session_id(session_id: &str) -> String {
        let trimmed = session_id.trim();
        if trimmed.is_empty() {
            return "default_session".to_string();
        }
        let clean: String = trimmed
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-' || *c == '.')
            .take(64)
            .collect();
        if clean.is_empty() {
            "default_session".to_string()
        } else {
            clean
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
        let clean_id = Self::sanitize_session_id(session_id);
        let mut turn = turn;
        if !security_verdicts.is_empty() {
            if let Some(object) = turn.as_object_mut() {
                object.insert("security_verdicts".into(), json!(security_verdicts));
            } else {
                turn = json!({"turn": turn, "security_verdicts": security_verdicts});
            }
        }

        let mut sessions = self.sessions.lock().unwrap_or_else(|p| p.into_inner());

        // Quota check
        if sessions.len() >= self.max_sessions && !sessions.contains_key(&clean_id) {
            // Evict an arbitrary existing session
            if let Some(evict_key) = sessions.keys().next().cloned() {
                sessions.remove(&evict_key);
            }
        }

        let history = sessions.entry(clean_id.clone()).or_default();
        history.push(turn);
        if history.len() > self.max_turns {
            history.drain(0..history.len() - self.max_turns);
        }

        json!({
            "heuristic_telemetry": true,
            "session_id": clean_id,
            "turns_in_memory": history.len(),
            "active_goal": active_goal,
            "active_slots": active_slots.map(|s| json!(s)).unwrap_or(Value::Null)
        })
    }

    pub fn get_context_for_prompt(&self, session_id: &str) -> Option<String> {
        let clean_id = Self::sanitize_session_id(session_id);
        let sessions = self.sessions.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(history) = sessions.get(&clean_id) {
            if history.is_empty() {
                return None;
            }
            let context: Vec<String> = history
                .iter()
                .rev()
                .take(self.context_window_turns)
                .map(|t| {
                    let u = t.get("user_input").and_then(|v| v.as_str()).unwrap_or("");
                    let r = t.get("response").and_then(|v| v.as_str()).unwrap_or("");
                    format!("[USER_MESSAGE]: {}\n[ASSISTANT_RESPONSE]: {}", u, r)
                })
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            Some(context.join("\n\n"))
        } else {
            None
        }
    }
}

/// Consolidates episodic memories into semantic knowledge with path traversal defense
/// and crash-safe transactional writes.
pub struct MemoryConsolidationEngine {
    repo_root: String,
}

impl MemoryConsolidationEngine {
    pub fn new(repo_root: &str) -> Self {
        Self {
            repo_root: repo_root.to_string(),
        }
    }

    pub fn repo_root(&self) -> &str {
        &self.repo_root
    }

    pub fn store_episode(
        &self,
        principal: &tara_security::primitives::AuthenticatedPrincipal,
        episode_id: &str,
        episode: Value,
    ) -> Result<(), String> {
        if !principal.is_authenticated() {
            return Err("Unauthorized: Authentication required to store episode".into());
        }
        if episode_id.is_empty()
            || episode_id.contains("..")
            || episode_id.contains('/')
            || episode_id.contains('\\')
            || episode_id.len() > 64
        {
            return Err("InvalidEpisodeId: Path traversal detected or malformed ID".into());
        }
        let serialized = serde_json::to_string_pretty(&episode).map_err(|e| e.to_string())?;
        if serialized.len() > 256 * 1024 {
            return Err("PayloadTooLarge: Episode payload exceeds 256 KiB limit".into());
        }
        let dir = Path::new(&self.repo_root)
            .join("storage")
            .join("memory")
            .join("consolidated");
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

        let final_path = dir.join(format!("{}.json", episode_id));
        let temp_path = dir.join(format!(".ep_{}.tmp", uuid::Uuid::new_v4()));

        let mut file = File::create(&temp_path).map_err(|e| e.to_string())?;
        file.write_all(serialized.as_bytes())
            .map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        drop(file);

        fs::rename(&temp_path, &final_path).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn prune(&self, max_episodes: usize) -> Result<usize, String> {
        let dir = Path::new(&self.repo_root)
            .join("storage")
            .join("memory")
            .join("consolidated");
        if !dir.exists() {
            return Ok(0);
        }
        let mut entries: Vec<std::path::PathBuf> = fs::read_dir(&dir)
            .map_err(|e| e.to_string())?
            .filter_map(|r| r.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
            .collect();

        let max_capped = max_episodes.min(5000);
        if entries.len() <= max_capped {
            return Ok(entries.len());
        }
        entries.sort();
        let to_remove = entries.len() - max_capped;
        for path in entries.iter().take(to_remove) {
            let _ = fs::remove_file(path);
        }
        Ok(max_capped)
    }

    pub fn consolidate_episode(&self, episode: Value) -> Result<(), std::io::Error> {
        let dir = Path::new(&self.repo_root)
            .join("storage")
            .join("memory")
            .join("consolidated");
        fs::create_dir_all(&dir)?;

        let raw_id = episode
            .get("episode_id")
            .and_then(|v| v.as_str())
            .unwrap_or("ep");

        // Strict path traversal sanitization: permit only alphanumeric, dash, underscore
        let clean_id: String = raw_id
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
            .take(64)
            .collect();

        let final_id = if clean_id.is_empty() {
            format!("ep_{}", uuid::Uuid::new_v4())
        } else {
            clean_id
        };

        let final_path = dir.join(format!("{}.json", final_id));
        let temp_path = dir.join(format!(".ep_{}.tmp", uuid::Uuid::new_v4()));

        let serialized = serde_json::to_string_pretty(&episode).unwrap_or_default();
        let mut file = File::create(&temp_path)?;
        file.write_all(serialized.as_bytes())?;
        file.sync_all()?;
        drop(file);

        fs::rename(&temp_path, &final_path)?;
        Ok(())
    }
}

/// Prediction error loop: evaluates internal expectation vs actual candidate outputs.
/// Strictly non-authoritative heuristic evaluation.
pub struct PredictionErrorLoop {
    repo_root: String,
}

impl PredictionErrorLoop {
    pub fn new(repo_root: &str) -> Self {
        Self {
            repo_root: repo_root.to_string(),
        }
    }

    pub fn new_with_rate(repo_root: &str, alpha: f64) -> Result<Self, String> {
        if !alpha.is_finite() || alpha < 0.0 || alpha > 1.0 {
            return Err("InvalidLearningRate: alpha must be in [0.0, 1.0]".into());
        }
        Ok(Self::new(repo_root))
    }

    pub fn repo_root(&self) -> &str {
        &self.repo_root
    }

    pub fn compute_error(&self, actual: f64, expected: f64) -> Result<f64, String> {
        if !actual.is_finite() || !expected.is_finite() {
            return Err("NonFinitePredictionError: actual and expected must be finite".into());
        }
        let loss = (actual - expected).abs().clamp(0.0, 10.0);
        Ok(loss)
    }

    pub fn execute<F, T>(
        &self,
        guard: &crate::rules::ExecutionGuard,
        principal: &tara_security::primitives::AuthenticatedPrincipal,
        action_name: &str,
        f: F,
    ) -> Result<T, String>
    where
        F: FnOnce() -> T,
    {
        guard.enforce_creator_principal(principal, action_name)?;
        Ok(f())
    }

    pub fn persist(&self, state: &Value) -> Result<(), String> {
        let dir = Path::new(&self.repo_root)
            .join("storage")
            .join("prediction");
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let final_path = dir.join("prediction_state.json");
        let temp_path = dir.join(format!(".pred_{}.tmp", uuid::Uuid::new_v4()));

        let serialized = serde_json::to_string_pretty(state).map_err(|e| e.to_string())?;
        let mut file = File::create(&temp_path).map_err(|e| e.to_string())?;
        file.write_all(serialized.as_bytes())
            .map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        drop(file);

        fs::rename(&temp_path, &final_path).map_err(|e| e.to_string())?;
        let domain = format!(
            "pred_watermark_{}",
            &hex::encode(sha2::Sha256::digest(self.repo_root.as_bytes()))[..16]
        );
        let _ = tara_security::dpapi::store_os_monotonic_watermark(&domain, 1);
        Ok(())
    }

    pub fn load(&self) -> Result<Value, String> {
        let path = Path::new(&self.repo_root)
            .join("storage")
            .join("prediction")
            .join("prediction_state.json");
        if !path.exists() {
            return Ok(json!({ "epoch": 0, "status": "INITIAL" }));
        }
        let domain = format!(
            "pred_watermark_{}",
            &hex::encode(sha2::Sha256::digest(self.repo_root.as_bytes()))[..16]
        );
        if let Ok(Some(committed_watermark)) =
            tara_security::dpapi::load_os_monotonic_watermark(&domain)
        {
            let raw = fs::read_to_string(&path).map_err(|e| e.to_string())?;
            let val: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
            let epoch = val.get("epoch").and_then(|v| v.as_u64()).unwrap_or(0);
            if epoch < committed_watermark {
                return Err(format!(
                    "RollbackAttackDetected: epoch {} is less than committed watermark {}",
                    epoch, committed_watermark
                ));
            }
            Ok(val)
        } else {
            let raw = fs::read_to_string(&path).map_err(|e| e.to_string())?;
            serde_json::from_str(&raw).map_err(|e| e.to_string())
        }
    }

    pub fn execute_and_learn_cycle<F>(&self, task_name: &str, action: F, expectation: &str) -> Value
    where
        F: FnOnce() -> Value,
    {
        let safe_task: String = task_name.chars().take(128).collect();
        let safe_expectation: String = expectation.chars().take(1024).collect();

        let actual = action();
        let actual_str = actual.to_string();
        let match_score = if actual_str
            .to_lowercase()
            .contains(&safe_expectation.to_lowercase())
        {
            1.0
        } else {
            0.0
        };

        json!({
            "heuristic_telemetry": true,
            "task": safe_task,
            "expectation": safe_expectation,
            "match_score": match_score,
            "actual": actual
        })
    }
}

/// Audit and explainability engine with collision-free filenames, secret redaction,
/// HMAC-SHA256 cryptographic signature chaining, and crash-safe transactional writes.
pub struct AuditExplainabilityEngine {
    repo_root: String,
    last_signature: Arc<Mutex<String>>,
    sequence_counter: Arc<Mutex<u64>>,
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
        let domain = format!(
            "audit_seq_{}",
            &hex::encode(Sha256::digest(repo_root.as_bytes()))[..16]
        );
        let mut initial_seq = tara_security::dpapi::load_os_monotonic_watermark(&domain)
            .ok()
            .flatten()
            .unwrap_or(0);
        let mut initial_sig = "AUDIT_GENESIS_ROOT".to_string();

        let dir = Path::new(repo_root).join("storage").join("audit");
        if dir.exists() {
            if let Ok(read_dir) = fs::read_dir(&dir) {
                let mut entries: Vec<std::path::PathBuf> = read_dir
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| {
                        p.is_file()
                            && p.file_name()
                                .and_then(|n| n.to_str())
                                .map(|n| n.starts_with("audit_") && n.ends_with(".json"))
                                .unwrap_or(false)
                    })
                    .collect();
                entries.sort();
                if let Some(last_path) = entries.last() {
                    if let Ok(raw) = fs::read_to_string(last_path) {
                        if let Ok(val) = serde_json::from_str::<Value>(&raw) {
                            if let Some(sig) = val.get("hmac_signature").and_then(|v| v.as_str()) {
                                if !sig.is_empty() {
                                    initial_sig = sig.to_string();
                                }
                            }
                            if let Some(seq) = val.get("seq").and_then(|v| v.as_u64()) {
                                initial_seq = initial_seq.max(seq);
                            }
                        }
                    }
                }
            }
        }

        Self {
            repo_root: repo_root.to_string(),
            last_signature: Arc::new(Mutex::new(initial_sig)),
            sequence_counter: Arc::new(Mutex::new(initial_seq)),
        }
    }

    pub fn last_signature(&self) -> String {
        self.last_signature
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    fn redact_secrets(input: &str) -> String {
        let mut s = input.to_string();
        for keyword in &["Bearer ", "token=", "key=", "password=", "secret="] {
            let mut search_from = 0;
            while let Some(rel_pos) = s[search_from..].find(keyword) {
                let start = search_from + rel_pos + keyword.len();
                let end = s[start..]
                    .find(|c: char| c.is_whitespace() || c == '&' || c == ';')
                    .map(|offset| start + offset)
                    .unwrap_or(s.len());
                if s[start..end] != *"[REDACTED]" {
                    s.replace_range(start..end, "[REDACTED]");
                }
                search_from = start + "[REDACTED]".len();
                if search_from >= s.len() {
                    break;
                }
            }
        }
        s
    }

    pub fn record_decision(&self, p: DecisionRecordParams<'_>) -> Result<(), std::io::Error> {
        let dir = Path::new(&self.repo_root).join("storage").join("audit");
        fs::create_dir_all(&dir)?;

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();

        let clean_rationale = Self::redact_secrets(p.rationale);
        let clean_action = Self::redact_secrets(p.action);

        let mut prev_sig_lock = self
            .last_signature
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let prev_sig = prev_sig_lock.clone();

        let domain = format!(
            "audit_seq_{}",
            &hex::encode(Sha256::digest(self.repo_root.as_bytes()))[..16]
        );
        let mut seq_guard = self
            .sequence_counter
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let next_seq = seq_guard.checked_add(1).ok_or_else(|| {
            std::io::Error::other("CounterExhausted: audit sequence reached u64::MAX")
        })?;
        *seq_guard = next_seq;
        let current_seq = next_seq;

        let record_content = json!({
            "seq": current_seq,
            "ts": ts,
            "actor_id": p.actor_id,
            "intent": p.intent,
            "evidence": p.evidence,
            "strategy": p.selected_strategy,
            "action": clean_action,
            "outcome": p.outcome,
            "rationale": clean_rationale,
            "prev_signature": prev_sig
        });

        let serialized_content = serde_json::to_vec(&record_content).unwrap_or_default();
        let mut signer = tara_security::cryptography::HmacSha256Signer::new(&*AUDIT_HMAC_KEY)
            .expect("HMAC accepts 32-byte key");
        signer.update(prev_sig.as_bytes());
        signer.update(b":");
        signer.update(&serialized_content);
        let hmac_sig = signer.finalize_hex();

        let final_record = json!({
            "seq": current_seq,
            "ts": ts,
            "actor_id": p.actor_id,
            "intent": p.intent,
            "evidence": p.evidence,
            "strategy": p.selected_strategy,
            "action": clean_action,
            "outcome": p.outcome,
            "rationale": clean_rationale,
            "prev_signature": prev_sig,
            "hmac_signature": hmac_sig
        });

        let serialized = serde_json::to_string_pretty(&final_record).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(serialized.as_bytes());
        let record_sha = hex::encode(hasher.finalize());

        let final_filename = format!(
            "audit_{}_{:012}_{}.json",
            ts,
            current_seq,
            uuid::Uuid::new_v4()
        );
        let final_path = dir.join(final_filename);
        let temp_path = dir.join(format!(".audit_{}.tmp", uuid::Uuid::new_v4()));

        let mut file = File::create(&temp_path)?;
        file.write_all(serialized.as_bytes())?;
        file.sync_all()?;
        drop(file);

        fs::rename(&temp_path, &final_path)?;
        tara_security::hardware_root::sync_directory_metadata(&final_path)
            .map_err(std::io::Error::other)?;
        tara_security::hardware_root::harden_file_permissions(&final_path)
            .map_err(std::io::Error::other)?;

        tara_security::dpapi::store_os_monotonic_watermark(&domain, current_seq)
            .map_err(std::io::Error::other)?;

        *prev_sig_lock = hmac_sig;
        drop(prev_sig_lock);

        // Ensure record hash is referenced
        let _ = record_sha;

        Ok(())
    }

    /// Verifies the cryptographic HMAC hash chain of audit records on disk.
    /// Detects record truncation, record omission, tampering, and genesis anchor mismatch.
    pub fn verify_audit_log_chain(&self) -> Result<usize, String> {
        let dir = Path::new(&self.repo_root).join("storage").join("audit");
        if !dir.exists() {
            return Ok(0);
        }
        let mut entries = Vec::new();
        for entry in fs::read_dir(&dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.is_file() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("audit_") && name.ends_with(".json") {
                        entries.push(path);
                    }
                }
            }
        }
        if entries.is_empty() {
            let current_head = self.last_signature();
            if current_head != "AUDIT_GENESIS_ROOT" {
                return Err(
                    "AuditLogTruncated: log directory is empty but head signature was recorded"
                        .to_string(),
                );
            }
            return Ok(0);
        }

        // Sort by file name / timestamp / sequence
        entries.sort();

        let mut expected_prev_sig = "AUDIT_GENESIS_ROOT".to_string();
        let mut verified_count = 0;
        let mut max_authenticated_sequence: u64 = 0;

        for path in &entries {
            let raw = fs::read_to_string(path)
                .map_err(|e| format!("Failed to read audit record {:?}: {}", path, e))?;
            let val: Value = serde_json::from_str(&raw)
                .map_err(|e| format!("Corrupt JSON in {:?}: {}", path, e))?;

            let prev_sig = val
                .get("prev_signature")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if prev_sig != expected_prev_sig {
                return Err(format!(
                    "AuditChainBroken: record in {:?} has prev_signature {} but expected {}",
                    path, prev_sig, expected_prev_sig
                ));
            }

            let file_hmac = val
                .get("hmac_signature")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            // Sequence number: parsed from JSON or fallback to verified_count + 1
            let record_seq = val
                .get("seq")
                .and_then(|v| v.as_u64())
                .unwrap_or(verified_count as u64 + 1);

            // Reconstruct content to verify HMAC (supporting both version with seq and legacy)
            let mut computed_hmac = String::new();
            if val.get("seq").is_some() {
                let content = json!({
                    "seq": record_seq,
                    "ts": val.get("ts"),
                    "actor_id": val.get("actor_id"),
                    "intent": val.get("intent"),
                    "evidence": val.get("evidence"),
                    "strategy": val.get("strategy"),
                    "action": val.get("action"),
                    "outcome": val.get("outcome"),
                    "rationale": val.get("rationale"),
                    "prev_signature": prev_sig
                });
                let serialized_content = serde_json::to_vec(&content).unwrap_or_default();
                let mut signer =
                    tara_security::cryptography::HmacSha256Signer::new(&*AUDIT_HMAC_KEY)
                        .expect("HMAC accepts 32-byte key");
                signer.update(prev_sig.as_bytes());
                signer.update(b":");
                signer.update(&serialized_content);
                computed_hmac = signer.finalize_hex();
            }

            if computed_hmac.is_empty()
                || !tara_security::cryptography::CryptoUtils::constant_time_eq(
                    file_hmac.as_bytes(),
                    computed_hmac.as_bytes(),
                )
            {
                // Fallback to legacy schema verification
                let content_legacy = json!({
                    "ts": val.get("ts"),
                    "actor_id": val.get("actor_id"),
                    "intent": val.get("intent"),
                    "evidence": val.get("evidence"),
                    "strategy": val.get("strategy"),
                    "action": val.get("action"),
                    "outcome": val.get("outcome"),
                    "rationale": val.get("rationale"),
                    "prev_signature": prev_sig
                });
                let serialized_content = serde_json::to_vec(&content_legacy).unwrap_or_default();
                let mut signer =
                    tara_security::cryptography::HmacSha256Signer::new(&*AUDIT_HMAC_KEY)
                        .expect("HMAC accepts 32-byte key");
                signer.update(prev_sig.as_bytes());
                signer.update(b":");
                signer.update(&serialized_content);
                let computed_legacy = signer.finalize_hex();

                if !tara_security::cryptography::CryptoUtils::constant_time_eq(
                    file_hmac.as_bytes(),
                    computed_legacy.as_bytes(),
                ) {
                    return Err(format!(
                        "AuditSignatureTampered: record in {:?} signature mismatch",
                        path
                    ));
                }
            }

            expected_prev_sig = file_hmac.to_string();
            verified_count += 1;
            max_authenticated_sequence = max_authenticated_sequence.max(record_seq);
        }

        let in_memory_head = self.last_signature();
        if in_memory_head != "AUDIT_GENESIS_ROOT" && expected_prev_sig != in_memory_head {
            return Err(format!(
                "AuditLogTruncated: disk chain head {} does not match recorded head signature {}",
                expected_prev_sig, in_memory_head
            ));
        }

        let domain = format!(
            "audit_seq_{}",
            &hex::encode(Sha256::digest(self.repo_root.as_bytes()))[..16]
        );
        if let Ok(Some(committed_watermark)) =
            tara_security::dpapi::load_os_monotonic_watermark(&domain)
        {
            if max_authenticated_sequence < committed_watermark {
                return Err(format!(
                    "AuditLogTruncated: highest authenticated sequence number {} is less than committed OS watermark {}",
                    max_authenticated_sequence, committed_watermark
                ));
            }
        }

        Ok(verified_count)
    }

    pub fn redact_evidence(&self, val: &Value) -> Value {
        match val {
            Value::String(s) => Value::String(Self::redact_secrets(s)),
            Value::Array(arr) => {
                Value::Array(arr.iter().map(|item| self.redact_evidence(item)).collect())
            }
            Value::Object(map) => {
                let mut new_map = serde_json::Map::new();
                for (k, v) in map {
                    let k_lower = k.to_lowercase();
                    if k_lower.contains("secret")
                        || k_lower.contains("token")
                        || k_lower.contains("password")
                        || k_lower.contains("key")
                    {
                        new_map.insert(k.clone(), Value::String("[REDACTED]".to_string()));
                    } else {
                        new_map.insert(k.clone(), self.redact_evidence(v));
                    }
                }
                Value::Object(new_map)
            }
            other => other.clone(),
        }
    }

    pub fn prune(&self, max_files: usize) -> Result<usize, String> {
        let dir = Path::new(&self.repo_root).join("storage").join("audit");
        if !dir.exists() {
            return Ok(0);
        }
        let mut entries: Vec<std::path::PathBuf> = fs::read_dir(&dir)
            .map_err(|e| e.to_string())?
            .filter_map(|r| r.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
            .collect();
        let max_capped = max_files.min(50000);
        if entries.len() <= max_capped {
            return Ok(entries.len());
        }
        entries.sort();
        let to_remove = entries.len() - max_capped;
        for path in entries.iter().take(to_remove) {
            let _ = fs::remove_file(path);
        }
        Ok(max_capped)
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
    pub is_heuristic_telemetry: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IntuitionVerdict {
    pub mode: String, // "INTUITIVE_FAST_PATH" or "DELIBERATIVE_TREE_SEARCH"
    pub confidence: f32,
    pub retrieved_association_norm: f32,
    pub rationale: String,
    pub is_heuristic_telemetry: bool,
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
    pub self_model: Arc<SelfModelEngine>,
    pub affect: Arc<AffectiveHomeostasisEngine>,
    pub creativity: Arc<CreativityEngine>,
    pub connection: Arc<ConnectionManager>,
    pub existential: Arc<ExistentialReasoningEngine>,
}

impl CognitiveCapabilitiesHub {
    pub fn new(repo_root: &str) -> Self {
        Self::with_config(repo_root, CognitiveHubConfig::default())
    }

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

    /// Evaluates unified cognitive awareness snapshot.
    pub fn evaluate_unified_awareness(
        &self,
        session_id: &str,
        topic: &str,
    ) -> UnifiedAwarenessSnapshot {
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
            is_heuristic_telemetry: true,
        }
    }

    /// Dual-Process Arbitration (System 1: Intuition vs System 2: Deliberative Tree Search).
    /// Heuristic arbitration only: selects candidate generation path.
    /// STRICT INVARIANT: Intuition fast path CANNOT bypass authorization checks or execution guards.
    pub fn arbitrate_intuition_vs_deliberation(
        &self,
        query_vec: &[f32],
        confidence_threshold: f32,
    ) -> IntuitionVerdict {
        let fw_guard = self.fast_weights.lock().unwrap_or_else(|p| p.into_inner());
        let assoc = fw_guard
            .read_association(query_vec)
            .unwrap_or_else(|_| vec![0.0f32; query_vec.len()]);
        drop(fw_guard);

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
                rationale: "Associative fast-weights memory returned high confidence match; selecting fast candidate generator".to_string(),
                is_heuristic_telemetry: true,
            }
        } else {
            IntuitionVerdict {
                mode: "DELIBERATIVE_TREE_SEARCH".to_string(),
                confidence: (effective_confidence * 100.0).round() / 100.0,
                retrieved_association_norm: (norm * 100.0).round() / 100.0,
                rationale: "Low associative familiarity or high epistemic tension detected; engaging deliberative tree search".to_string(),
                is_heuristic_telemetry: true,
            }
        }
    }

    pub fn check_lockdown(
        &self,
        coordinator: &tara_security::lockdown::LockdownCoordinator,
    ) -> Result<(), String> {
        if coordinator.is_in_lockdown() {
            return Err("LockdownActive: Operations blocked during security lockdown".into());
        }
        Ok(())
    }

    pub fn is_healthy(&self) -> bool {
        self.self_model.is_healthy()
    }

    pub fn dispatch(
        &self,
        principal: &tara_security::primitives::AuthenticatedPrincipal,
        request: &Value,
    ) -> Result<Value, String> {
        if !principal.is_authenticated() {
            return Err("AuthorizationDenied: Unauthenticated dispatch blocked".into());
        }
        if request.get("task").is_none() && request.get("action").is_none() {
            return Err("InvalidSchema: Request missing mandatory task or action field".into());
        }
        Ok(json!({
            "status": "DISPATCH_ACCEPTED",
            "principal": principal.actor_id(),
            "timestamp": crate::now_millis()
        }))
    }

    pub fn shutdown(&self) -> Result<(), String> {
        let _ = self.self_model.persist();
        Ok(())
    }
}

pub fn saturating_metric_add(current: u64, delta: u64) -> u64 {
    current.saturating_add(delta)
}

pub fn canonical_sort_json(val: &Value) -> Value {
    match val {
        Value::Object(map) => {
            let mut sorted = std::collections::BTreeMap::new();
            for (k, v) in map {
                sorted.insert(k.clone(), canonical_sort_json(v));
            }
            serde_json::to_value(sorted).unwrap_or(Value::Null)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(canonical_sort_json).collect()),
        other => other.clone(),
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
        assert!(awareness.is_heuristic_telemetry);

        // 2. Test intuition arbitration when weights are empty (should choose Deliberative)
        let q = vec![1.0f32; 64];
        let verdict_empty = hub.arbitrate_intuition_vs_deliberation(&q, 0.5);
        assert_eq!(verdict_empty.mode, "DELIBERATIVE_TREE_SEARCH");

        // 3. Train fast-weights association (write pattern)
        {
            let mut fw = hub.fast_weights.lock().unwrap();
            fw.write_association(&q, &q).unwrap();
        }

        // Test intuition arbitration with learned pattern (should choose Intuitive Fast Path)
        let verdict_learned = hub.arbitrate_intuition_vs_deliberation(&q, 0.5);
        assert_eq!(verdict_learned.mode, "INTUITIVE_FAST_PATH");

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_audit_engine_chain_recovery_on_startup() {
        let tmp = std::env::temp_dir().join(format!("tara_audit_test_{}", uuid::Uuid::new_v4()));
        let tmp_str = tmp.to_str().unwrap();

        // 1. Initial engine starts with genesis root
        let engine1 = AuditExplainabilityEngine::new(tmp_str);
        assert_eq!(engine1.last_signature(), "AUDIT_GENESIS_ROOT");

        // 2. Record decision in engine 1
        let dummy_evidence = json!({"ip": "127.0.0.1"});
        engine1
            .record_decision(DecisionRecordParams {
                actor_id: "admin",
                intent: "test_recovery",
                evidence: &dummy_evidence,
                selected_strategy: "allow",
                action: "grant",
                outcome: "success",
                rationale: "valid token=abc12345",
            })
            .unwrap();

        let sig1 = engine1.last_signature();
        assert_ne!(sig1, "AUDIT_GENESIS_ROOT");

        // 3. Second engine booted on same repo_root recovers last signature from disk
        let engine2 = AuditExplainabilityEngine::new(tmp_str);
        assert_eq!(engine2.last_signature(), sig1);

        // 4. Record decision in engine 2, chain must advance monotonically
        engine2
            .record_decision(DecisionRecordParams {
                actor_id: "admin",
                intent: "test_second_decision",
                evidence: &dummy_evidence,
                selected_strategy: "allow",
                action: "grant",
                outcome: "success",
                rationale: "routine check",
            })
            .unwrap();

        let sig2 = engine2.last_signature();
        assert_ne!(sig2, sig1);

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_redact_secrets_multi_match() {
        let input = "Request with Bearer my_secret_token_123 and token=abc456&key=xyz789; password=secret_pass and secret=top_secret_val";
        let redacted = AuditExplainabilityEngine::redact_secrets(input);
        assert!(!redacted.contains("my_secret_token_123"));
        assert!(!redacted.contains("abc456"));
        assert!(!redacted.contains("xyz789"));
        assert!(!redacted.contains("secret_pass"));
        assert!(!redacted.contains("top_secret_val"));
        assert!(redacted.contains("Bearer [REDACTED]"));
        assert!(redacted.contains("token=[REDACTED]"));
        assert!(redacted.contains("key=[REDACTED]"));
        assert!(redacted.contains("password=[REDACTED]"));
        assert!(redacted.contains("secret=[REDACTED]"));
    }
}
