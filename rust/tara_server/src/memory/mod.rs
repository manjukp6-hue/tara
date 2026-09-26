//! Memory engine: episodic memory storage and retrieval.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use serde_json::{json, Value};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Manages episodic memory: records and queries interaction episodes.
pub struct MemoryEngine {
    pub memory_dir: String,
}

impl MemoryEngine {
    pub fn new(memory_dir: &str) -> Self {
        let _ = fs::create_dir_all(format!("{}/episodes", memory_dir));
        Self { memory_dir: memory_dir.to_string() }
    }

    /// Record a new episode to disk.
    pub fn record_episode(
        &self,
        actor_id: &str,
        intent: &str,
        action: &str,
        parameters: &Value,
        outcome: &str,
        observations: &Value,
        error: Option<&str>,
        reflection: &str,
    ) -> Result<String, MemoryError> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis();
        let episode_id = format!("ep_{}", ts);

        let episode = json!({
            "episode_id": episode_id,
            "actor_id": actor_id,
            "intent": intent,
            "action": action,
            "parameters": parameters,
            "outcome": outcome,
            "observations": observations,
            "error": error,
            "reflection": reflection,
            "timestamp": ts
        });

        let ep_dir = format!("{}/episodes/{}", self.memory_dir, actor_id);
        fs::create_dir_all(&ep_dir)?;
        let path = format!("{}/{}.json", ep_dir, episode_id);
        fs::write(&path, serde_json::to_string_pretty(&episode)?)?;

        Ok(episode_id)
    }

    /// Query episodes with optional filters.
    pub fn query_episodes(
        &self,
        query: Option<&str>,
        actor_id: Option<&str>,
        outcome_filter: Option<&str>,
        limit: usize,
        _scope: &str,
    ) -> Vec<Value> {
        let mut episodes = Vec::new();
        let ep_dir = format!("{}/episodes", self.memory_dir);

        if !Path::new(&ep_dir).exists() {
            return episodes;
        }

        let scan_dirs: Vec<String> = if let Some(aid) = actor_id {
            vec![format!("{}/{}", ep_dir, aid)]
        } else {
            fs::read_dir(&ep_dir).ok()
                .map(|rd| rd.flatten()
                    .filter(|e| e.path().is_dir())
                    .map(|e| e.path().to_string_lossy().to_string())
                    .collect())
                .unwrap_or_default()
        };

        for dir in scan_dirs {
            if let Ok(files) = fs::read_dir(&dir) {
                for entry in files.flatten() {
                    if entry.path().extension().and_then(|e| e.to_str()) == Some("json") {
                        if let Ok(raw) = fs::read_to_string(entry.path()) {
                            if let Ok(ep) = serde_json::from_str::<Value>(&raw) {
                                if let Some(q) = query {
                                    let text = ep.to_string().to_lowercase();
                                    if !text.contains(&q.to_lowercase()) {
                                        continue;
                                    }
                                }
                                if let Some(of) = outcome_filter {
                                    if ep.get("outcome").and_then(|v| v.as_str()) != Some(of) {
                                        continue;
                                    }
                                }
                                episodes.push(ep);
                                if episodes.len() >= limit {
                                    return episodes;
                                }
                            }
                        }
                    }
                }
            }
        }

        episodes
    }

    /// Return memory statistics.
    pub fn get_memory_stats(&self) -> Value {
        let ep_dir = format!("{}/episodes", self.memory_dir);
        let total = if let Ok(rd) = fs::read_dir(&ep_dir) {
            rd.flatten().flat_map(|e| {
                fs::read_dir(e.path()).ok().into_iter().flatten().flatten()
            }).filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("json"))
            .count()
        } else { 0 };
        json!({ "total_episodes": total, "memory_dir": self.memory_dir })
    }
}
