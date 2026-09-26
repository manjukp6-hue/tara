//! Cognitive capabilities hub: world state, events, working memory, consolidation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::fs;
use serde_json::{json, Value};

/// Tracks world-state entities.
pub struct WorldStateTracker {
    pub entities: Arc<Mutex<HashMap<String, Value>>>,
}

impl WorldStateTracker {
    pub fn new() -> Self {
        Self { entities: Arc::new(Mutex::new(HashMap::new())) }
    }

    pub fn update_entity(&self, id: &str, attributes: Value) {
        self.entities.lock().unwrap().insert(id.to_string(), attributes);
    }

    pub fn get_entity(&self, id: &str) -> Option<Value> {
        self.entities.lock().unwrap().get(id).cloned()
    }

    pub fn snapshot(&self) -> Value {
        json!(self.entities.lock().unwrap().clone())
    }
}

/// Event bus for intra-system pub/sub.
pub struct EventBus {
    subscribers: Arc<Mutex<HashMap<String, Vec<Box<dyn Fn(Value) + Send + Sync>>>>>,
}

impl EventBus {
    pub fn new() -> Self {
        Self { subscribers: Arc::new(Mutex::new(HashMap::new())) }
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
        self.subscribers.lock().unwrap()
            .entry(event.to_string())
            .or_default()
            .push(Box::new(handler));
    }
}

/// Governs working memory across sessions.
pub struct WorkingMemoryGovernor {
    sessions: Arc<Mutex<HashMap<String, Vec<Value>>>>,
    max_turns: usize,
}

impl WorkingMemoryGovernor {
    pub fn new(max_turns: usize) -> Self {
        Self { sessions: Arc::new(Mutex::new(HashMap::new())), max_turns }
    }

    pub fn govern_session(
        &self,
        session_id: &str,
        turn: Value,
        active_goal: Option<&str>,
        active_slots: Option<&HashMap<String, String>>,
        security_verdicts: &[Value],
    ) -> Value {
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
            let context: Vec<String> = history.iter().rev().take(3).map(|t| {
                format!("User: {}\nTARA: {}",
                    t.get("user_input").and_then(|v| v.as_str()).unwrap_or(""),
                    t.get("response").and_then(|v| v.as_str()).unwrap_or(""))
            }).collect::<Vec<_>>().into_iter().rev().collect();
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
        Self { repo_root: repo_root.to_string() }
    }

    pub fn consolidate_episode(&self, episode: Value) -> Result<(), std::io::Error> {
        let dir = format!("{}/storage/memory/consolidated", self.repo_root);
        fs::create_dir_all(&dir)?;
        let id = episode.get("episode_id").and_then(|v| v.as_str()).unwrap_or("ep");
        let path = format!("{}/{}.json", dir, id);
        fs::write(&path, serde_json::to_string_pretty(&episode).unwrap_or_default())?;
        Ok(())
    }
}

/// Prediction error loop: attempts actions and learns from mismatches.
pub struct PredictionErrorLoop {
    pub repo_root: String,
}

impl PredictionErrorLoop {
    pub fn new(repo_root: &str) -> Self {
        Self { repo_root: repo_root.to_string() }
    }

    pub fn execute_and_learn_cycle<F>(&self, task_name: &str, action: F, expectation: &str) -> Value
    where F: FnOnce() -> Value
    {
        let actual = action();
        let actual_str = actual.to_string();
        let match_score = if actual_str.to_lowercase().contains(&expectation.to_lowercase()) { 1.0 } else { 0.0 };
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

impl AuditExplainabilityEngine {
    pub fn new(repo_root: &str) -> Self {
        Self { repo_root: repo_root.to_string() }
    }

    pub fn record_decision(
        &self,
        actor_id: &str,
        intent: &str,
        evidence: &Value,
        selected_strategy: &str,
        action: &str,
        outcome: &str,
        rationale: &str,
    ) -> Result<(), std::io::Error> {
        let dir = format!("{}/storage/audit", self.repo_root);
        fs::create_dir_all(&dir)?;
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis();
        let record = json!({
            "ts": ts, "actor_id": actor_id, "intent": intent,
            "evidence": evidence, "strategy": selected_strategy,
            "action": action, "outcome": outcome, "rationale": rationale
        });
        let path = format!("{}/audit_{}.json", dir, ts);
        fs::write(&path, serde_json::to_string_pretty(&record).unwrap_or_default())?;
        Ok(())
    }
}

/// The aggregated cognitive capabilities hub.
pub struct CognitiveCapabilitiesHub {
    pub world_state: WorldStateTracker,
    pub event_bus: EventBus,
    pub working_memory: WorkingMemoryGovernor,
    pub memory_consolidation: MemoryConsolidationEngine,
    pub prediction_error_loop: PredictionErrorLoop,
    pub audit_engine: AuditExplainabilityEngine,
}

impl CognitiveCapabilitiesHub {
    pub fn new(repo_root: &str) -> Self {
        Self {
            world_state: WorldStateTracker::new(),
            event_bus: EventBus::new(),
            working_memory: WorkingMemoryGovernor::new(10),
            memory_consolidation: MemoryConsolidationEngine::new(repo_root),
            prediction_error_loop: PredictionErrorLoop::new(repo_root),
            audit_engine: AuditExplainabilityEngine::new(repo_root),
        }
    }
}
