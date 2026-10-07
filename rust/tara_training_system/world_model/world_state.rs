//! TARA World State Tracker & Snapshot Engine (100% Native Rust).
//!
//! Maintains ground-truth reality state of entities and environments across cognitive turns.
//! Supports isolated candidate snapshotting, validation, and atomic production promotion.

use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Tracks world-state entities and environmental variables.
pub struct WorldStateTracker {
    pub entities: Arc<Mutex<HashMap<String, Value>>>,
}

impl Default for WorldStateTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl WorldStateTracker {
    pub fn new() -> Self {
        Self {
            entities: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn update_entity(&self, id: &str, attributes: Value) {
        self.entities
            .lock()
            .unwrap()
            .insert(id.to_string(), attributes);
    }

    pub fn get_entity(&self, id: &str) -> Option<Value> {
        self.entities.lock().unwrap().get(id).cloned()
    }

    pub fn snapshot(&self) -> Value {
        json!(self.entities.lock().unwrap().clone())
    }

    /// Save a versioned candidate snapshot to an isolated staging directory (Candidate before Promotion)
    pub fn save_candidate_snapshot(&self, target_dir: &Path) -> std::io::Result<PathBuf> {
        fs::create_dir_all(target_dir)?;
        let snap = self.snapshot();
        let target_file = target_dir.join("world_model_state.json");
        fs::write(&target_file, serde_json::to_string_pretty(&snap).map_err(std::io::Error::other)?)?;
        Ok(target_file)
    }

    /// Save a verified production snapshot (Only after promotion passes)
    pub fn save_production_snapshot(&self, target_file: &Path) -> std::io::Result<()> {
        if let Some(parent) = target_file.parent() {
            fs::create_dir_all(parent)?;
        }
        let snap = self.snapshot();
        fs::write(target_file, serde_json::to_string_pretty(&snap).map_err(std::io::Error::other)?)?;
        Ok(())
    }

    /// Load world state entities from a verified snapshot
    pub fn load_snapshot_from(&self, file_path: &Path) -> std::io::Result<usize> {
        if !file_path.exists() {
            return Ok(0);
        }
        let data = fs::read_to_string(file_path)?;
        let val: Value = serde_json::from_str(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut count = 0;
        if let Value::Object(map) = val {
            let mut lock = self.entities.lock().unwrap();
            for (k, v) in map {
                lock.insert(k, v);
                count += 1;
            }
        }
        Ok(count)
    }
}
