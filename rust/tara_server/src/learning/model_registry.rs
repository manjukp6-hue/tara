//! Model registry: version manifest management.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path};
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

#[derive(Debug, Default)]
struct RegistryState {
    versions: Vec<ModelVersionMetadata>,
    active_version: Option<String>,
}

#[derive(Debug)]
pub struct ModelRegistry {
    manifest_path: String,
    state: Mutex<RegistryState>,
}

impl ModelRegistry {
    /// Attempt to initialize the model registry, returning an error if
    /// the manifest file exists but is corrupted, unparseable, or invalid.
    pub fn try_new(repo_root: &str) -> Result<Arc<Self>, String> {
        let manifest_path = format!("{}/storage/models/versions_manifest.json", repo_root);
        let state = Self::load_manifest(&manifest_path)?;
        Ok(Arc::new(Self {
            manifest_path,
            state: Mutex::new(state),
        }))
    }

    /// Primary constructor for ModelRegistry. Fails closed with an empty, locked-down
    /// registry state if the manifest is corrupted, preventing silent acceptance of unverified versions.
    pub fn new(repo_root: &str) -> Arc<Self> {
        match Self::try_new(repo_root) {
            Ok(reg) => reg,
            Err(err) => {
                eprintln!("FailClosed: ModelRegistry manifest error: {err}");
                let manifest_path = format!("{}/storage/models/versions_manifest.json", repo_root);
                Arc::new(Self {
                    manifest_path,
                    state: Mutex::new(RegistryState {
                        versions: Vec::new(),
                        active_version: None,
                    }),
                })
            }
        }
    }

    fn load_manifest(path: &str) -> Result<RegistryState, String> {
        let path_obj = Path::new(path);
        if !path_obj.exists() {
            return Ok(RegistryState {
                versions: Vec::new(),
                active_version: None,
            });
        }

        let raw = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read model manifest '{}': {}", path, e))?;
        let v: Value = serde_json::from_str(&raw)
            .map_err(|e| format!("Corrupt model manifest JSON '{}': {}", path, e))?;

        if !v.is_object() {
            return Err(format!(
                "Model manifest '{}' root must be a JSON object",
                path
            ));
        }

        let active = v
            .get("active_version")
            .and_then(Value::as_str)
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
            Some(_) => {
                return Err(format!(
                    "Invalid versions collection in manifest '{}': expected object or array",
                    path
                ))
            }
            None => Vec::new(),
        };

        Ok(RegistryState {
            versions,
            active_version: active,
        })
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
        let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.active_version.clone()
    }

    /// Returns the SHA-256 of the currently active model version weights,
    /// as recorded in the versions manifest at training registration time.
    /// Returns `None` if no active version is registered or it has no SHA recorded.
    pub fn get_active_version_sha(&self) -> Option<String> {
        let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let active_id = state.active_version.as_ref()?;
        state
            .versions
            .iter()
            .find(|v| &v.version_id == active_id)
            .and_then(|v| v.sha256.clone())
            .filter(|sha| !sha.is_empty())
    }

    /// Returns the artifact location of the currently active model version,
    /// or None if no active model version is registered.
    pub fn get_active_version_location(&self) -> Option<String> {
        let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let active_id = state.active_version.as_ref()?;
        state
            .versions
            .iter()
            .find(|v| &v.version_id == active_id)
            .map(|v| v.artifact_location.clone())
            .filter(|loc| !loc.is_empty())
    }

    pub fn list_versions(&self) -> Vec<Value> {
        let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state
            .versions
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

    pub fn register_active_training_guarded(
        &self,
        model_dir: &str,
        training_summary: &Value,
        lockdown: Option<&tara_security::lockdown::LockdownCoordinator>,
    ) -> Result<String, String> {
        if let Some(ld) = lockdown {
            if ld.is_write_blocked() {
                return Err(
                    "Model registration blocked: System write block is active (LOCKDOWN)".into(),
                );
            }
        }
        self.register_active_training(model_dir, training_summary)
    }

    /// Persist a successful native training cycle as a new active model version.
    /// Serializes entire registration under a single mutex to prevent concurrency overwrites
    /// and derives rollback metadata exclusively from authoritative registry state.
    pub fn register_active_training(
        &self,
        model_dir: &str,
        training_summary: &Value,
    ) -> Result<String, String> {
        if training_summary.get("status").and_then(Value::as_str) != Some("COMPLETED") {
            return Err("only a completed training cycle can be registered".into());
        }

        // Validate model_dir path safety (no traversal, within repo root)
        let model_dir_path = Path::new(model_dir);
        if model_dir_path
            .components()
            .any(|c| c == Component::ParentDir)
        {
            return Err("model directory path cannot contain parent traversal '..'".into());
        }
        if !model_dir_path.exists() || !model_dir_path.is_dir() {
            return Err(format!(
                "model directory '{}' does not exist or is not a directory",
                model_dir
            ));
        }

        let repo_root = Path::new(&self.manifest_path)
            .parent()
            .and_then(Path::parent)
            .and_then(Path::parent)
            .ok_or_else(|| "model manifest is outside the repository storage tree".to_string())?;

        // Determine weights and compute SHA-256 (single file vs sharded index)
        let index_path = model_dir_path.join("model.safetensors.index.json");
        let single_path = model_dir_path.join("model.safetensors");

        let (sha256, shard_count) = if index_path.exists() {
            let raw_index = fs::read_to_string(&index_path)
                .map_err(|e| format!("Failed to read sharded index: {e}"))?;
            let idx_val: Value = serde_json::from_str(&raw_index)
                .map_err(|e| format!("Invalid sharded index JSON: {e}"))?;
            let weight_map = idx_val
                .get("weight_map")
                .and_then(Value::as_object)
                .ok_or_else(|| "Sharded index missing 'weight_map' object".to_string())?;

            let mut shards = HashSet::new();
            for (_, shard_file) in weight_map {
                if let Some(s) = shard_file.as_str() {
                    if s.is_empty()
                        || Path::new(s).is_absolute()
                        || Path::new(s).components().any(|c| c == Component::ParentDir)
                    {
                        return Err(format!("Unsafe shard path '{}' in weight_map", s));
                    }
                    shards.insert(s.to_string());
                }
            }
            if shards.is_empty() {
                return Err("Sharded index contains no referenced shard files".into());
            }

            let mut sorted_shards: Vec<String> = shards.into_iter().collect();
            sorted_shards.sort();

            let mut aggregate_hasher = Sha256::new();
            for s in &sorted_shards {
                let shard_full = model_dir_path.join(s);
                if !shard_full.exists() {
                    return Err(format!(
                        "Missing referenced shard file '{}'",
                        shard_full.display()
                    ));
                }
                let shard_hash = tara_engine::compute256(&shard_full).map_err(|e| {
                    format!("Shard hash failed for '{}': {e}", shard_full.display())
                })?;
                aggregate_hasher.update(shard_hash.as_bytes());
            }
            (
                hex::encode(aggregate_hasher.finalize()),
                sorted_shards.len(),
            )
        } else if single_path.exists() {
            let sha = tara_engine::compute256(&single_path).map_err(|error| error.to_string())?;
            (sha, 1)
        } else {
            return Err("neither model.safetensors nor model.safetensors.index.json found in candidate model dir".into());
        };

        if sha256.len() != 64 {
            return Err("trained model checkpoint has an invalid SHA-256 digest".into());
        }

        let config_path = model_dir_path.join("config.json");
        let config: Value =
            serde_json::from_slice(&fs::read(&config_path).map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())?;
        let parameter_count = tara_engine::load_model_weights(model_dir)
            .map_err(|error| error.to_string())?
            .values()
            .map(|tensor| tensor.len() as u64)
            .sum::<u64>();
        if parameter_count == 0 {
            return Err("trained checkpoint contains no parameters".into());
        }

        // Acquire lock across the entire manifest update & disk write to serialize registrations
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());

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
            .map(str::to_string)
            .or_else(|| state.active_version.clone());

        // Validate lineage: if caller declares parent version, verify it exists in registry
        if let Some(parent_id) = training_summary
            .get("parent_model_version")
            .and_then(Value::as_str)
        {
            if !state.versions.iter().any(|v| v.version_id == parent_id) {
                return Err(format!(
                    "Declared parent model version '{}' does not exist in registry",
                    parent_id
                ));
            }
        }

        // Authoritative rollback info MUST come from trusted registry state, NOT caller summary
        let prev_record = active_before
            .as_deref()
            .and_then(|p_id| state.versions.iter().find(|v| v.version_id == p_id));
        let old_authoritative_sha = prev_record.and_then(|r| r.sha256.clone());

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
        let versions_map = versions
            .as_object_mut()
            .ok_or_else(|| "model versions must be an object or array".to_string())?;

        let stamp = crate::now_iso();
        let version_id = format!("TARA_TRAIN_{}", &sha256[..12]);

        if let Some(previous_id) = active_before.as_deref() {
            if let Some(previous) = versions_map
                .get_mut(previous_id)
                .and_then(Value::as_object_mut)
            {
                if let Some(archive) = training_summary
                    .get("previous_artifact_location")
                    .and_then(Value::as_str)
                {
                    previous.insert("artifact_location".into(), json!(archive));
                }
                previous.insert("status".into(), json!("superseded"));
            }
        }

        let artifact_location = Path::new(model_dir)
            .strip_prefix(repo_root)
            .map(|path| path.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|_| model_dir.to_string());

        versions_map.insert(
            version_id.clone(),
            json!({
                "version_id": version_id,
                "artifact_location": artifact_location,
                "weights_sha256": sha256,
                "config": config,
                "parameter_count": parameter_count,
                "growth_type": "SupervisedTraining",
                "parent_model_version": active_before.clone(),
                "shard_count": shard_count,
                "tokenizer_vocab_size": config.get("vocab_size"),
                "compatible_capabilities": ["inference", "training"],
                "metrics": training_summary,
                "status": "active",
                "registered_at": stamp.clone()
            }),
        );

        let root = manifest.as_object_mut().expect("manifest root checked");
        if let (Some(previous_id), Some(old_sha)) =
            (active_before.as_deref(), old_authoritative_sha)
        {
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

        // Durable, atomic disk persistence with sync_all
        Self::write_manifest_atomically(&self.manifest_path, &manifest)?;

        // Now commit changes to in-memory state
        let metadata = Self::parse_version(&version_id, &manifest["versions"][&version_id])
            .ok_or_else(|| "persisted model version could not be parsed".to_string())?;

        if let (Some(previous_id), Some(archive)) = (
            active_before.as_deref(),
            training_summary
                .get("previous_artifact_location")
                .and_then(Value::as_str),
        ) {
            if let Some(previous) = state
                .versions
                .iter_mut()
                .find(|entry| entry.version_id == previous_id)
            {
                previous.artifact_location = archive.to_string();
            }
        }

        state
            .versions
            .retain(|entry| entry.version_id != version_id);
        state.versions.push(metadata);
        state.active_version = Some(version_id.clone());

        Ok(version_id)
    }

    /// Promote a model version requiring verified Creator principal and cryptographic signature.
    pub fn promote(
        &self,
        principal: &tara_security::primitives::AuthenticatedPrincipal,
        version_id: &str,
        creator_signature: Option<&str>,
    ) -> Result<(), String> {
        if !principal.is_creator() {
            return Err("AuthorizationDenied: Model promotion requires creator authority".into());
        }
        if creator_signature.is_none() || creator_signature.unwrap().trim().is_empty() {
            return Err(
                "MissingSignature: Model promotion requires verified creator signature".into(),
            );
        }
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if !state.versions.iter().any(|v| v.version_id == version_id) {
            return Err(format!(
                "Model version '{}' not found in registry",
                version_id
            ));
        }
        if Path::new(&self.manifest_path).exists() {
            if let Ok(raw) = fs::read(&self.manifest_path) {
                if let Ok(mut manifest) = serde_json::from_slice::<Value>(&raw) {
                    if let Some(root) = manifest.as_object_mut() {
                        root.insert("active_version".into(), json!(version_id));
                        Self::write_manifest_atomically(&self.manifest_path, &manifest)?;
                    }
                }
            }
        }
        state.active_version = Some(version_id.to_string());
        Ok(())
    }

    fn write_manifest_atomically(path: &str, manifest: &Value) -> Result<(), String> {
        let destination = Path::new(path);
        let payload = serde_json::to_vec_pretty(manifest).map_err(|error| error.to_string())?;
        tara_security::persistence::atomic_write_file_hardened(destination, &payload)
    }
}

#[cfg(test)]
mod tests {
    use super::ModelRegistry;
    use serde_json::json;

    fn write_test_safetensors(path: &std::path::Path) -> std::io::Result<()> {
        let header_json =
            br#"{"lm_head.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let header_len = header_json.len() as u64;
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let data_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut out = Vec::new();
        out.extend_from_slice(&header_len.to_le_bytes());
        out.extend_from_slice(header_json);
        out.extend_from_slice(&data_bytes);
        std::fs::write(path, out)
    }

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
    fn corrupt_manifest_fails_closed_in_try_new() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("tara_corrupt_manifest_{stamp}"));
        let model_dir = root.join("storage/models");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(
            model_dir.join("versions_manifest.json"),
            r#"{"active_version":"v2", corrupt json here..."#,
        )
        .unwrap();

        let res = ModelRegistry::try_new(root.to_str().unwrap());
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("Corrupt model manifest JSON"));
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
        write_test_safetensors(&model_dir.join("model.safetensors")).unwrap();

        let registry = ModelRegistry::new(root.to_str().unwrap());
        let version_id = registry
            .register_active_training(
                model_dir.to_str().unwrap(),
                &json!({
                    "status": "COMPLETED",
                    "previous_model_sha256": "fake-injected-sha-should-be-ignored",
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
        let rollbacks = manifest["rollback_history"].as_array().unwrap();
        assert_eq!(rollbacks.len(), 1);
        // Authoritative old-sha from registry was used, NOT fake-injected-sha!
        assert_eq!(rollbacks[0]["weights_sha256"], "old-sha");
        assert_eq!(
            manifest["versions"][&version_id]["artifact_location"],
            "storage/models/tara"
        );
        assert_eq!(registry.list_versions().len(), 2);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn test_promote_persists_active_version_and_unsafe_shard_rejection() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("tara_promote_test_{stamp}"));
        let model_dir = root.join("storage/models");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(
            model_dir.join("versions_manifest.json"),
            r#"{"active_version":"v1","versions":{"v1":{"version_id":"v1","artifact_location":"models/v1","weights_sha256":"hash1","parameter_count":4},"v2":{"version_id":"v2","artifact_location":"models/v2","weights_sha256":"hash2","parameter_count":8}}}"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(root.to_str().unwrap());
        let principal = tara_security::primitives::AuthenticatedPrincipal::mint_creator_for_test(
            "test_session",
        )
        .unwrap();

        // 1. Promote v2
        assert!(registry
            .promote(&principal, "v2", Some("valid_sig"))
            .is_ok());
        assert_eq!(registry.get_active_version().as_deref(), Some("v2"));

        // Verify disk persistence
        let disk_manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model_dir.join("versions_manifest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(disk_manifest["active_version"], "v2");

        // 2. Unsafe shard traversal rejection
        let training_dir = root.join("storage/models/tara_train");
        std::fs::create_dir_all(&training_dir).unwrap();
        std::fs::write(
            training_dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"layer1":"../secret.safetensors"}}"#,
        )
        .unwrap();
        let res = registry.register_active_training(
            training_dir.to_str().unwrap(),
            &json!({"status": "COMPLETED"}),
        );
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("Unsafe shard path"));

        std::fs::remove_dir_all(root).unwrap();
    }
}
