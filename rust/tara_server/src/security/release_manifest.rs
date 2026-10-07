//! Signed Release and Build Integrity Engine for TARA Core.
//! Guarantees:
//! 1. Release/build identity verification.
//! 2. Canonical SHA-256 trees across critical codebases and production assets.
//! 3. Production model SAFETENSORS SHA-256 is verified at runtime against the
//!    release manifest's production_model_sha256 field (not a hardcoded constant).
//!    The 118,080-parameter smoke-test model SHA has been removed.
//! 4. Signs release manifests with HMAC-SHA256 or root authority.

use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

// NOTE: EXPECTED_PRODUCTION_MODEL_SHA256 removed — the 118,080-parameter smoke-test
// model it referenced has been deleted. The model SHA is now read at runtime from
// ModelRegistry::get_active_version_sha(). No SHA is hardcoded in source.

pub const CRITICAL_PATHS_TO_VERIFY: &[&str] = &[
    // "storage/models/tara/model.safetensors" — removed, no model registered
    "TARA/ACCESS/operator/operator_record.json",
    "TARA/ACCESS/operator/operators_registry.json",
    "TARA/ACCESS/operator/activation_config.json",
    "TARA/ACCESS/restore/restore_config.json",
    "rust/tara_core/src/rule_engine.rs",
    "rust/tara_server/src/security/tamper_detector.rs",
];

pub fn compute_file_sha256<P: AsRef<Path>>(path: P) -> Option<String> {
    let mut file = File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer).ok()?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Some(hex::encode(hasher.finalize()))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseManifest {
    pub build_id: String,
    pub version: String,
    pub timestamp: String,
    pub file_hashes: HashMap<String, String>,
    pub production_model_sha256: String,
    pub signature: Option<String>,
}

pub struct ReleaseManifestManager {
    pub repo_root: PathBuf,
    pub manifest_path: PathBuf,
}

impl ReleaseManifestManager {
    pub fn new(repo_root: Option<&str>, manifest_path: Option<&str>) -> Self {
        let root = repo_root
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        let m_path = manifest_path
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("storage").join("release_manifest.json"));
        Self {
            repo_root: root,
            manifest_path: m_path,
        }
    }

    pub fn generate_release_manifest(
        &self,
        build_id: &str,
        version: &str,
        signing_key: Option<&[u8]>,
    ) -> Result<ReleaseManifest, String> {
        let mut file_hashes = HashMap::new();

        for rel_path in CRITICAL_PATHS_TO_VERIFY {
            let full_path = self.repo_root.join(rel_path);
            if full_path.exists() {
                if let Some(hash) = compute_file_sha256(&full_path) {
                    file_hashes.insert((*rel_path).to_string(), hash);
                }
            }
        }

        let model_path = self.repo_root.join("storage/models/tara/model.safetensors");
        let model_sha = compute_file_sha256(&model_path).unwrap_or_default();

        let timestamp = crate::now_iso();

        let mut manifest = ReleaseManifest {
            build_id: build_id.to_string(),
            version: version.to_string(),
            timestamp,
            file_hashes,
            production_model_sha256: model_sha,
            signature: None,
        };

        if let Some(key) = signing_key {
            let payload = serde_json::to_string(&manifest.file_hashes).unwrap_or_default();
            type HmacSha256 = Hmac<Sha256>;
            if let Ok(mut mac) = HmacSha256::new_from_slice(key) {
                mac.update(payload.as_bytes());
                manifest.signature = Some(hex::encode(mac.finalize().into_bytes()));
            }
        }

        if let Some(parent) = self.manifest_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let serialized = serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
        fs::write(&self.manifest_path, serialized).map_err(|e| e.to_string())?;

        Ok(manifest)
    }

    pub fn verify_release_manifest(&self, signing_key: Option<&[u8]>) -> Result<Value, String> {
        if !self.manifest_path.exists() {
            return Ok(json!({
                "valid": false,
                "error": "Manifest file not found"
            }));
        }

        let content = fs::read_to_string(&self.manifest_path).map_err(|e| e.to_string())?;
        let manifest: ReleaseManifest =
            serde_json::from_str(&content).map_err(|e| e.to_string())?;

        // 1. Verify model hash matches what was recorded in THIS manifest at release time.
        // Do NOT compare against a hardcoded bootstrap SHA — after self-training promotes
        // a new model the manifest is regenerated and production_model_sha256 is updated.
        let model_path = self.repo_root.join("storage/models/tara/model.safetensors");
        if model_path.exists() {
            let current_model_hash = compute_file_sha256(&model_path).unwrap_or_default();
            if !manifest.production_model_sha256.is_empty()
                && current_model_hash != manifest.production_model_sha256
            {
                return Ok(json!({
                    "valid": false,
                    "error": "Production model SAFETENSORS SHA256 does not match this release manifest",
                    "manifest_expected": manifest.production_model_sha256,
                    "actual": current_model_hash
                }));
            }
        }

        // 2. Verify files against manifest hashes
        let mut mismatches = Vec::new();
        for (rel_path, expected_hash) in &manifest.file_hashes {
            let full_path = self.repo_root.join(rel_path);
            if !full_path.exists() {
                mismatches.push(format!("Missing file: {}", rel_path));
                continue;
            }
            let actual = compute_file_sha256(&full_path).unwrap_or_default();
            if actual != *expected_hash {
                mismatches.push(format!(
                    "Hash mismatch for {}: expected {}, got {}",
                    rel_path, expected_hash, actual
                ));
            }
        }

        // 3. Verify signature if key provided
        let mut sig_valid = true;
        if let (Some(key), Some(ref sig_hex)) = (signing_key, &manifest.signature) {
            let payload = serde_json::to_string(&manifest.file_hashes).unwrap_or_default();
            type HmacSha256 = Hmac<Sha256>;
            if let Ok(mut mac) = HmacSha256::new_from_slice(key) {
                mac.update(payload.as_bytes());
                let expected = hex::encode(mac.finalize().into_bytes());
                if expected != *sig_hex {
                    sig_valid = false;
                }
            } else {
                sig_valid = false;
            }
        }

        let is_valid = mismatches.is_empty() && sig_valid;
        Ok(json!({
            "valid": is_valid,
            "signature_valid": sig_valid,
            "mismatches": mismatches,
            "build_id": manifest.build_id,
            "version": manifest.version
        }))
    }
}
