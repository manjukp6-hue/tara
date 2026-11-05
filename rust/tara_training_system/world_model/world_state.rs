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
        json!(self
            .entities
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone())
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

    /// Projects all active ground-truth world-state entities into a continuous latent vector
    /// aligned with the Neural World Model's `latent_dim`.
    pub fn to_latent_vector(&self, target_dim: usize) -> Vec<f32> {
        let lock = self.entities.lock().unwrap_or_else(|p| p.into_inner());
        let mut vec = vec![0.0f32; target_dim];
        if lock.is_empty() {
            return vec;
        }

        // Sort keys for deterministic, canonical alignment
        let mut keys: Vec<&String> = lock.keys().collect();
        keys.sort();

        let num_entities = keys.len();
        for (idx, key) in keys.iter().enumerate() {
            let offset = (idx * 64) % target_dim;
            let val = &lock[*key];

            // 1. Continuous hash/embedding of the entity key
            let key_bytes = key.as_bytes();
            for (b_i, &b) in key_bytes.iter().enumerate() {
                let pos = (offset + b_i) % target_dim;
                let freq = (b as f32 * 0.05).sin();
                vec[pos] += freq / (num_entities as f32).sqrt();
            }

            // 2. Numerical reality features extraction
            if let Some(obj) = val.as_object() {
                for (attr_k, attr_v) in obj {
                    let attr_idx = (offset + attr_k.len() * 7) % target_dim;
                    if let Some(num) = attr_v.as_f64() {
                        let normalized = (num as f32).tanh();
                        vec[attr_idx] += normalized / (num_entities as f32).sqrt();
                    } else if let Some(b) = attr_v.as_bool() {
                        let flag = if b { 1.0f32 } else { -1.0f32 };
                        vec[attr_idx] += flag / (num_entities as f32).sqrt();
                    } else if let Some(s) = attr_v.as_str() {
                        let s_bytes = s.as_bytes();
                        let s_len = s_bytes.len().min(32);
                        for i in 0..s_len {
                            let p = (attr_idx + i) % target_dim;
                            vec[p] += ((s_bytes[i] as f32) * 0.01).cos() / (num_entities as f32).sqrt();
                        }
                    }
                }
            }
        }

        // Normalize to unit sphere for stable neural latent dynamics
        let norm = vec.iter().map(|&v| v * v).sum::<f32>().sqrt();
        if norm > 1e-6 {
            let inv = 1.0 / norm;
            for v in vec.iter_mut() {
                *v *= inv;
            }
        }
        vec
    }

    /// Closes the cognitive loop by updating reality state tracker entities from
    /// predicted Neural World Model dynamic transitions.
    pub fn update_from_latent_transition(
        &self,
        entity_id: &str,
        predicted_reward: f32,
        failure_probability: f32,
        viable: bool,
        confidence: f32,
        expected_value: f32,
    ) {
        let entry = json!({
            "predicted_reward": predicted_reward,
            "failure_probability": failure_probability,
            "viable": viable,
            "confidence": confidence,
            "expected_value": expected_value,
            "timestamp_ms": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64
        });
        self.update_entity(entity_id, entry);
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

        let temp_dir =
            std::env::temp_dir().join(format!("tara_test_world_state_{}", std::process::id()));
        let saved_path = tracker
            .save_candidate_snapshot(&temp_dir)
            .expect("snapshot save failed");
        assert!(saved_path.exists());

        let reader_tracker = WorldStateTracker::new();
        let loaded_count = reader_tracker
            .load_snapshot_from(&saved_path)
            .expect("snapshot load failed");
        assert_eq!(loaded_count, 2);
        assert_eq!(
            reader_tracker.get_entity("device_mode").unwrap()["battery"],
            95
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_world_state_latent_bridge_and_transition_updates() {
        let tracker = WorldStateTracker::new();
        tracker.update_entity("sandbox_status", json!({"active": true, "exit_code": 0, "latency_ms": 14.5}));
        tracker.update_entity("hardware_telemetry", json!({"cpu_percent": 42.0, "memory_mb": 512}));

        // 1. Verify continuous entity-to-latent projection
        let latent_dim = 2048;
        let s_t = tracker.to_latent_vector(latent_dim);
        assert_eq!(s_t.len(), latent_dim);
        let norm: f32 = s_t.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "latent vector must be normalized to unit sphere");

        // 2. Verify closed-loop dynamic transition update
        tracker.update_from_latent_transition("action_result:cargo_check", 0.95, 0.02, true, 0.98, 1.4);
        let updated = tracker.get_entity("action_result:cargo_check").expect("entity updated");
        assert_eq!(updated["viable"], true);
        assert!((updated["failure_probability"].as_f64().unwrap() - 0.02).abs() < 1e-4);
        assert!((updated["predicted_reward"].as_f64().unwrap() - 0.95).abs() < 1e-4);
    }
}
