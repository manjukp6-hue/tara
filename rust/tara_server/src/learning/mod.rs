//! Learning subsystem: user search-based learning and autonomous online learning.

use serde_json::{json, Value};
use std::fs;

pub enum LearningMode {
    Supervised,
    Autonomous,
    Reinforcement,
}

/// Learns from user search results.
pub struct UserSearchLearner {
    pub repo_root: String,
}

impl UserSearchLearner {
    pub fn new(repo_root: &str) -> Self {
        Self { repo_root: repo_root.to_string() }
    }

    pub fn search_and_learn(&self, query: &str, user_id: &str) -> Value {
        // In production: integrate with a search API
        json!({
            "status": "SUCCESS",
            "action": "ONLINE_LEARNING",
            "query": query,
            "user_id": user_id,
            "sources_found": 0,
            "knowledge_updated": false,
            "message": format!("Online learning for query '{}' initiated. No external search configured.", query)
        })
    }
}

/// Autonomous online learner: self-directed learning sessions.
pub struct AutonomousOnlineLearner {
    pub repo_root: String,
}

impl AutonomousOnlineLearner {
    pub fn new(repo_root: &str) -> Self {
        Self { repo_root: repo_root.to_string() }
    }

    pub fn start_session(&self, mode: Option<LearningMode>, actor_id: &str) -> Value {
        let mode_str = match mode {
            Some(LearningMode::Supervised) => "Supervised",
            Some(LearningMode::Reinforcement) => "Reinforcement",
            _ => "Autonomous",
        };
        json!({
            "status": "SUCCESS",
            "action": "AUTONOMOUS_LEARNING",
            "mode": mode_str,
            "actor_id": actor_id,
            "message": format!("Autonomous {} learning session started.", mode_str)
        })
    }
}
