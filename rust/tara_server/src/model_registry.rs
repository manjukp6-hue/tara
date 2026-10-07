//! Model registry: version manifest management.

use serde_json::{json, Value};
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

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
                let active = v
                    .get("active_version")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let versions = match v.get("versions") {
                    Some(Value::Object(entries)) => entries
                        .iter()
                        .filter_map(|(key, entry)| Self::parse_version(key, entry))
                        .collect(),
                    Some(Value::Array(entries)) => entries
                        .iter()
                        .filter_map(|entry| Self::parse_version("", entry))
                        .collect(),
                    _ => Vec::new(),
                };
                return (versions, active);
            }
        }
        (Vec::new(), None)
    }

    fn parse_version(key: &str, entry: &Value) -> Option<ModelVersionMetadata> {
        let version_id = entry
            .get("version_id")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .or_else(|| (!key.is_empty()).then_some(key))?;
        Some(ModelVersionMetadata {
            version_id: version_id.to_string(),
            artifact_location: entry
                .get("artifact_location")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            parameter_count: entry
                .get("parameter_count")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            growth_type: entry
                .get("growth_type")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            parent_model_version: entry
                .get("parent_model_version")
                .and_then(Value::as_str)
                .map(String::from),
            shard_count: entry
                .get("shard_count")
                .and_then(Value::as_u64)
                .unwrap_or(1) as usize,
            sha256: entry
                .get("sha256")
                .or_else(|| entry.get("weights_sha256"))
                .and_then(Value::as_str)
                .map(String::from),
            promoted_at: entry
                .get("promoted_at")
                .or_else(|| entry.get("registered_at"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            training_summary: entry
                .get("training_summary")
                .or_else(|| entry.get("metrics"))
                .cloned()
                .unwrap_or(json!({})),
        })
    }

    pub fn get_active_version(&self) -> Option<String> {
        self.active_version.lock().unwrap().clone()
    }

    /// Returns the SHA-256 of the currently active model version weights,
    /// as recorded in the versions manifest at training registration time.
    /// Returns `None` if no active version is registered or it has no SHA recorded.
    pub fn get_active_version_sha(&self) -> Option<String> {
        let active_id = self.active_version.lock().unwrap().clone()?;
        let versions = self.versions.lock().unwrap();
        versions
            .iter()
            .find(|v| v.version_id == active_id)
            .and_then(|v| v.sha256.clone())
            .filter(|sha| !sha.is_empty())
    }

    /// Returns the artifact location of the currently active model version,
    /// or None if no active model version is registered.
    pub fn get_active_version_location(&self) -> Option<String> {
        let active_id = self.active_version.lock().unwrap().clone()?;
        let versions = self.versions.lock().unwrap();
        versions
            .iter()
            .find(|v| v.version_id == active_id)
            .map(|v| v.artifact_location.clone())
            .filter(|loc| !loc.is_empty())
    }

    pub fn list_versions(&self) -> Vec<Value> {
        self.versions
            .lock()
            .unwrap()
            .iter()
            .map(|v| {
                json!({
                    "version_id": v.version_id,
                    "artifact_location": v.artifact_location,
                    "parameter_count": v.parameter_count,
                    "growth_type": v.growth_type,
                    "parent_model_version": v.parent_model_version,
                    "sha256": v.sha256,
                    "promoted_at": v.promoted_at,
                    "shard_count": v.shard_count,
                    "training_summary": v.training_summary,
                })
            })
            .collect()
    }

    /// Persist a successful native training cycle as a new active model version.
    pub fn register_active_training(
        &self,
        model_dir: &str,
        training_summary: &Value,
    ) -> Result<String, String> {
        if training_summary.get("status").and_then(Value::as_str) != Some("COMPLETED") {
            return Err("only a completed training cycle can be registered".into());
        }
        let weights_path = Path::new(model_dir).join("model.safetensors");
        let sha256 = tara_engine::safetensors::compute_sha256(&weights_path.to_string_lossy())
            .map_err(|error| error.to_string())?;
        if sha256.len() != 64 {
            return Err("trained model checkpoint has an invalid SHA-256 digest".into());
        }
        let config_path = Path::new(model_dir).join("config.json");
        let config: Value =
            serde_json::from_slice(&fs::read(&config_path).map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())?;
        let parameter_count = tara_engine::safetensors::load_model_weights(model_dir)
            .map_err(|error| error.to_string())?
            .values()
            .map(|tensor| tensor.len() as u64)
            .sum::<u64>();
        if parameter_count == 0 {
            return Err("trained checkpoint contains no parameters".into());
        }

        let mut manifest: Value = match fs::read(&self.manifest_path) {
            Ok(raw) => serde_json::from_slice(&raw).map_err(|error| error.to_string())?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => json!({}),
            Err(error) => return Err(error.to_string()),
        };
        if !manifest.is_object() {
            return Err("model versions manifest root must be an object".into());
        }
        let active_before = manifest
            .get("active_version")
            .and_then(Value::as_str)
            .map(str::to_string);
        let versions = manifest
            .as_object_mut()
            .expect("manifest root checked")
            .entry("versions")
            .or_insert_with(|| json!({}));
        if versions.is_array() {
            let entries = versions.as_array().cloned().unwrap_or_default();
            let mut map = serde_json::Map::new();
            for entry in entries {
                if let Some(parsed) = Self::parse_version("", &entry) {
                    map.insert(parsed.version_id, entry);
                }
            }
            *versions = Value::Object(map);
        }
        let versions = versions
            .as_object_mut()
            .ok_or_else(|| "model versions must be an object or array".to_string())?;
        let stamp = crate::now_iso();
        let version_id = format!("TARA_TRAIN_{}", &sha256[..12]);
        let old_sha = training_summary
            .get("previous_model_sha256")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_string);
        if let Some(previous_id) = active_before.as_deref() {
            if let Some(previous) = versions.get_mut(previous_id).and_then(Value::as_object_mut) {
                if let Some(archive) = training_summary
                    .get("previous_artifact_location")
                    .and_then(Value::as_str)
                {
                    previous.insert("artifact_location".into(), json!(archive));
                }
                previous.insert("status".into(), json!("superseded"));
            }
        }
        let repo_root = Path::new(&self.manifest_path)
            .parent()
            .and_then(Path::parent)
            .and_then(Path::parent)
            .ok_or_else(|| "model manifest is outside the repository storage tree".to_string())?;
        let artifact_location = Path::new(model_dir)
            .strip_prefix(repo_root)
            .map(|path| path.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|_| model_dir.to_string());
        versions.insert(
            version_id.clone(),
            json!({
                "version_id": version_id,
                "artifact_location": artifact_location,
                "weights_sha256": sha256,
                "config": config,
                "parameter_count": parameter_count,
                "growth_type": "SupervisedTraining",
                "parent_model_version": active_before.clone(),
                "shard_count": 1,
                "tokenizer_vocab_size": config.get("vocab_size"),
                "compatible_capabilities": ["inference", "training"],
                "metrics": training_summary,
                "status": "active",
                "registered_at": stamp.clone()
            }),
        );
        let root = manifest.as_object_mut().expect("manifest root checked");
        if let (Some(previous_id), Some(old_sha)) = (active_before.as_deref(), old_sha) {
            root.entry("rollback_history")
                .or_insert_with(|| json!([]))
                .as_array_mut()
                .ok_or_else(|| "rollback_history must be an array".to_string())?
                .push(json!({
                    "version_id": previous_id,
                    "weights_sha256": old_sha,
                    "timestamp": stamp,
                    "reason": "Superseded by verified native self-training"
                }));
        }
        root.insert("active_version".into(), json!(version_id));
        Self::write_manifest_atomically(&self.manifest_path, &manifest)?;

        let metadata = Self::parse_version(&version_id, &manifest["versions"][&version_id])
            .ok_or_else(|| "persisted model version could not be parsed".to_string())?;
        let mut versions = self
            .versions
            .lock()
            .map_err(|_| "model version lock poisoned")?;
        if let (Some(previous_id), Some(archive)) = (
            active_before.as_deref(),
            training_summary
                .get("previous_artifact_location")
                .and_then(Value::as_str),
        ) {
            if let Some(previous) = versions
                .iter_mut()
                .find(|entry| entry.version_id == previous_id)
            {
                previous.artifact_location = archive.to_string();
            }
        }
        versions.retain(|entry| entry.version_id != version_id);
        versions.push(metadata);
        *self
            .active_version
            .lock()
            .map_err(|_| "active model version lock poisoned")? = Some(version_id.clone());
        Ok(version_id)
    }

    fn write_manifest_atomically(path: &str, manifest: &Value) -> Result<(), String> {
        let destination = Path::new(path);
        let parent = destination
            .parent()
            .ok_or_else(|| "model manifest has no parent directory".to_string())?;
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        let temp = destination.with_extension(format!("{}.tmp", std::process::id()));
        let backup = destination.with_extension(format!("{}.bak", std::process::id()));
        fs::write(
            &temp,
            serde_json::to_vec_pretty(manifest).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let had_destination = destination.exists();
        if had_destination {
            if backup.exists() {
                fs::remove_file(&backup).map_err(|error| error.to_string())?;
            }
            fs::rename(destination, &backup).map_err(|error| error.to_string())?;
        }
        if let Err(error) = fs::rename(&temp, destination) {
            if had_destination {
                let _ = fs::rename(&backup, destination);
            }
            let _ = fs::remove_file(&temp);
            return Err(error.to_string());
        }
        if had_destination {
            fs::remove_file(backup).map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::ModelRegistry;
    use serde_json::json;

    #[test]
    fn loads_the_persisted_object_version_manifest() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("tara_model_registry_{stamp}"));
        let model_dir = root.join("storage/models");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(
            model_dir.join("versions_manifest.json"),
            r#"{"active_version":"v2","versions":{"v1":{"version_id":"v1","artifact_location":"models/v1","weights_sha256":"old-hash","parameter_count":4,"registered_at":"t1"},"v2":{"artifact_location":"models/v2","sha256":"new-hash","parameter_count":8,"metrics":{"loss":0.5}}}}"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(root.to_str().unwrap());
        assert_eq!(registry.get_active_version().as_deref(), Some("v2"));
        let versions = registry.list_versions();
        assert_eq!(versions.len(), 2);
        assert!(versions
            .iter()
            .any(|version| { version["version_id"] == "v1" && version["sha256"] == "old-hash" }));
        assert!(versions.iter().any(|version| {
            version["version_id"] == "v2" && version["training_summary"]["loss"] == 0.5
        }));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn successful_training_registers_a_portable_active_version() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("tara_model_promotion_{stamp}"));
        let model_dir = root.join("storage/models/tara");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(
            root.join("storage/models/versions_manifest.json"),
            r#"{"active_version":"v1","versions":{"v1":{"version_id":"v1","artifact_location":"storage/models/tara","weights_sha256":"old-sha","parameter_count":4,"status":"active"}}}"#,
        )
        .unwrap();
        std::fs::write(
            model_dir.join("config.json"),
            r#"{"vocab_size":2,"hidden_size":2}"#,
        )
        .unwrap();
        tara_engine::safetensors::write_safetensors_with_shapes(
            &[("lm_head.weight".to_string(), vec![1.0f32, 2.0, 3.0, 4.0])]
                .into_iter()
                .collect(),
            &[("lm_head.weight".to_string(), vec![2, 2])]
                .into_iter()
                .collect(),
            &model_dir.join("model.safetensors").to_string_lossy(),
        )
        .unwrap();

        let registry = ModelRegistry::new(root.to_str().unwrap());
        let version_id = registry
            .register_active_training(
                model_dir.to_str().unwrap(),
                &json!({
                    "status": "COMPLETED",
                    "previous_model_sha256": "old-sha",
                    "previous_artifact_location": "storage/models/versions/old-sha.safetensors",
                    "output_path": model_dir.join("model.safetensors").to_string_lossy()
                }),
            )
            .unwrap();
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("storage/models/versions_manifest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(manifest["active_version"], version_id);
        assert_eq!(
            manifest["versions"]["v1"]["artifact_location"],
            "storage/models/versions/old-sha.safetensors"
        );
        assert_eq!(manifest["rollback_history"].as_array().unwrap().len(), 1);
        assert_eq!(
            manifest["versions"][&version_id]["artifact_location"],
            "storage/models/tara"
        );
        assert_eq!(registry.list_versions().len(), 2);
        std::fs::remove_dir_all(root).unwrap();
    }
}
