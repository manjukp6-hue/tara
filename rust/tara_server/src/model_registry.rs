//! Model registry: version manifest management.

use std::collections::HashMap;
use std::fs;
use std::sync::{Arc, Mutex, OnceLock};
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct ModelVersionMetadata {
    pub version_id: String,
    pub artifact_location: String,
    pub parameter_count: u64,
    pub growth_type: String,
    pub parent_model_version: Option<String>,
    pub shard_count: usize,
    pub sha256: Option<String>,
    pub promoted_at: String,
    pub training_summary: Value,
}

pub struct ModelRegistry {
    manifest_path: String,
    versions: Mutex<Vec<ModelVersionMetadata>>,
    active_version: Mutex<Option<String>>,
}

impl ModelRegistry {
    pub fn new(repo_root: &str) -> Arc<Self> {
        let manifest_path = format!("{}/storage/models/versions_manifest.json", repo_root);
        let (versions, active) = Self::load_manifest(&manifest_path);
        Arc::new(Self {
            manifest_path,
            versions: Mutex::new(versions),
            active_version: Mutex::new(active),
        })
    }

    fn load_manifest(path: &str) -> (Vec<ModelVersionMetadata>, Option<String>) {
        if let Ok(raw) = fs::read_to_string(path) {
            if let Ok(v) = serde_json::from_str::<Value>(&raw) {
                let active = v.get("active_version").and_then(|v| v.as_str()).map(String::from);
                let versions = v.get("versions").and_then(|v| v.as_array())
                    .map(|arr| arr.iter().map(|e| ModelVersionMetadata {
                        version_id: e.get("version_id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                        artifact_location: e.get("artifact_location").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                        parameter_count: e.get("parameter_count").and_then(|v| v.as_u64()).unwrap_or(0),
                        growth_type: e.get("growth_type").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                        parent_model_version: e.get("parent_model_version").and_then(|v| v.as_str()).map(String::from),
                        shard_count: e.get("shard_count").and_then(|v| v.as_u64()).unwrap_or(1) as usize,
                        sha256: e.get("sha256").and_then(|v| v.as_str()).map(String::from),
                        promoted_at: e.get("promoted_at").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                        training_summary: e.get("training_summary").cloned().unwrap_or(json!({})),
                    }).collect())
                    .unwrap_or_default();
                return (versions, active);
            }
        }
        (Vec::new(), None)
    }

    pub fn get_active_version(&self) -> Option<String> {
        self.active_version.lock().unwrap().clone()
    }

    pub fn list_versions(&self) -> Vec<Value> {
        self.versions.lock().unwrap().iter().map(|v| json!({
            "version_id": v.version_id,
            "artifact_location": v.artifact_location,
            "parameter_count": v.parameter_count,
            "growth_type": v.growth_type,
            "sha256": v.sha256,
            "promoted_at": v.promoted_at,
        })).collect()
    }
}
