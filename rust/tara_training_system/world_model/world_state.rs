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
            .unwrap_or_else(|p| p.into_inner())
            .insert(id.to_string(), attributes);
    }

    pub fn get_entity(&self, id: &str) -> Option<Value> {
        self.entities
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(id)
            .cloned()
    }

    pub fn snapshot(&self) -> Value {
        json!(self.entities.lock().unwrap_or_else(|p| p.into_inner()).clone())
    }

    /// Save a versioned candidate snapshot to an isolated staging directory (Candidate before Promotion)
    pub fn save_candidate_snapshot(&self, target_dir: &Path) -> std::io::Result<PathBuf> {
        let snap = self.snapshot();
        let target_file = target_dir.join("world_model_state.json");
        write_json_atomically(&target_file, &snap)?;
        Ok(target_file)
    }

    /// Save a verified production snapshot (Only after promotion passes)
    pub fn save_production_snapshot(&self, target_file: &Path) -> std::io::Result<()> {
        let snap = self.snapshot();
        write_json_atomically(target_file, &snap)?;
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
            let mut lock = self.entities.lock().unwrap_or_else(|p| p.into_inner());
            for (k, v) in map {
                lock.insert(k, v);
                count += 1;
            }
        }
        Ok(count)
    }
}

fn write_json_atomically(target_file: &Path, val: &Value) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = target_file.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let serialized = serde_json::to_vec_pretty(val).map_err(std::io::Error::other)?;
    let tmp_file = target_file.with_extension(format!("json.tmp.{}", std::process::id()));
    let mut f = fs::File::create(&tmp_file)?;
    f.write_all(&serialized)?;
    f.sync_all()?;
    drop(f);
    fs::rename(&tmp_file, target_file)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_state_atomic_snapshot_lifecycle() {
        let tracker = WorldStateTracker::new();
        tracker.update_entity("user_location", json!({"lat": 12.9716, "lon": 77.5946}));
        tracker.update_entity("device_mode", json!({"mode": "online", "battery": 95}));

        let temp_dir = std::env::temp_dir().join(format!("tara_test_world_state_{}", std::process::id()));
        let saved_path = tracker.save_candidate_snapshot(&temp_dir).expect("snapshot save failed");
        assert!(saved_path.exists());

        let reader_tracker = WorldStateTracker::new();
        let loaded_count = reader_tracker.load_snapshot_from(&saved_path).expect("snapshot load failed");
        assert_eq!(loaded_count, 2);
        assert_eq!(
            reader_tracker.get_entity("device_mode").unwrap()["battery"],
            95
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
