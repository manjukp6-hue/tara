//! Signed Release and Build Integrity Engine for TARA Core.
//! Guarantees:
//! 1. Release/build identity verification.
//! 2. Canonical SHA-256 trees across critical codebases and production assets.
//! 3. Production model SAFETENSORS SHA-256 is verified at runtime against the
//!    release manifest's production_model_sha256 field (not a hardcoded constant).
//! 4. Signs release manifests with HMAC-SHA256 covering all security-critical fields.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::persistence::atomic_write_file_hardened;

pub const CRITICAL_PATHS_TO_VERIFY: &[&str] = &[
    "TARA/ACCESS/operator/operator_record.json",
    "TARA/ACCESS/operator/operators_registry.json",
    "TARA/ACCESS/operator/activation_config.json",
    "TARA/ACCESS/restore/restore_config.json",
    "rust/tara_core/src/rule_engine.rs",
    "rust/tara_security/src/tara_control/tamper_detector.rs",
];

pub fn compute_file_sha256<P: AsRef<Path>>(path: P) -> Option<String> {
    crate::cryptography::CryptoUtils::compute256(path).ok()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseManifest {
    pub build_id: String,
    pub version: String,
    pub timestamp: String,
    pub file_hashes: HashMap<String, String>,
    #[serde(default)]
    pub file_tree: HashMap<String, String>,
    pub production_model_sha256: String,
    pub signature: Option<String>,
}

/// Computes canonical HMAC signing payload covering all security-critical fields:
/// build_id, version, timestamp, production_model_sha256, and sorted file hashes.
pub fn canonical_manifest_payload(
    build_id: &str,
    version: &str,
    timestamp: &str,
    production_model_sha256: &str,
    file_hashes: &HashMap<String, String>,
) -> String {
    let mut sorted_hashes: Vec<(&String, &String)> = file_hashes.iter().collect();
    sorted_hashes.sort_by_key(|(k, _)| *k);
    let hashes_json = serde_json::to_string(&sorted_hashes).unwrap_or_default();
    format!(
        "{}|{}|{}|{}|{}",
        build_id, version, timestamp, production_model_sha256, hashes_json
    )
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
        let model_sha = if model_path.exists() {
            compute_file_sha256(&model_path)
                .ok_or_else(|| "Failed to compute model SAFETENSORS SHA-256 (FailClosed)".to_string())?
        } else {
            String::new()
        };
        let now_epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let timestamp = format!("{now_epoch}");

        let mut manifest = ReleaseManifest {
            build_id: build_id.to_string(),
            version: version.to_string(),
            timestamp: timestamp.clone(),
            file_tree: file_hashes.clone(),
            file_hashes: file_hashes.clone(),
            production_model_sha256: model_sha.clone(),
            signature: None,
        };

        if let Some(key) = signing_key {
            let payload =
                canonical_manifest_payload(build_id, version, &timestamp, &model_sha, &file_hashes);
            if let Ok(sig_bytes) = crate::cryptography::CryptoUtils::hmac_sha256(key, payload.as_bytes()) {
                manifest.signature = Some(hex::encode(sig_bytes));
            }
        }

        let serialized = serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
        atomic_write_file_hardened(&self.manifest_path, serialized.as_bytes())?;

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

        // 1. Verify model hash matches what was recorded in THIS manifest at release time
        let model_path = self.repo_root.join("storage/models/tara/model.safetensors");
        if model_path.exists() {
            let current_model_hash = compute_file_sha256(&model_path)
                .ok_or_else(|| "Failed to compute production model SAFETENSORS SHA-256 for verification (FailClosed)".to_string())?;
            if !manifest.production_model_sha256.is_empty()
                && !crate::cryptography::CryptoUtils::constant_time_eq(
                    current_model_hash.as_bytes(),
                    manifest.production_model_sha256.as_bytes(),
                )
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
        let active_hashes = if !manifest.file_hashes.is_empty() {
            &manifest.file_hashes
        } else {
            &manifest.file_tree
        };

        let mut mismatches = Vec::new();
        for (rel_path, expected_hash) in active_hashes {
            if let Err(e) = crate::path_security::check_lexical_traversal(Path::new(rel_path)) {
                mismatches.push(format!("Illegal path traversal in manifest entry '{}': {e}", rel_path));
                continue;
            }
            let full_path = self.repo_root.join(rel_path);
            if !full_path.exists() {
                mismatches.push(format!("Missing file: {}", rel_path));
                continue;
            }
            let current_hash = match compute_file_sha256(&full_path) {
                Some(h) => h,
                None => {
                    mismatches.push(format!("Cannot read or hash file {}: integrity unverified", rel_path));
                    continue;
                }
            };
            if !crate::cryptography::CryptoUtils::constant_time_eq(
                current_hash.as_bytes(),
                expected_hash.as_bytes(),
            ) {
                mismatches.push(format!(
                    "Hash mismatch for {}: expected {}, got {}",
                    rel_path, expected_hash, current_hash
                ));
            }
        }

        // 3. Verify HMAC signature (anti-signature-stripping enforced)
        let effective_key_buf = if signing_key.is_none() {
            if let Ok(env_key) = std::env::var("TARA_RELEASE_SIGNING_KEY") {
                if !env_key.trim().is_empty() {
                    Some(env_key.into_bytes())
                } else {
                    None
                }
            } else {
                crate::dpapi::derive_authority_hmac_key("release_manifest_authority")
                    .ok()
                    .map(|k| k.to_vec())
            }
        } else {
            None
        };
        let active_key = signing_key.or(effective_key_buf.as_deref());

        let signature_valid = if let Some(ref sig_hex) = manifest.signature {
            if let Some(key) = active_key {
                let payload = canonical_manifest_payload(
                    &manifest.build_id,
                    &manifest.version,
                    &manifest.timestamp,
                    &manifest.production_model_sha256,
                    &manifest.file_hashes,
                );
                if let Ok(expected_bytes) = crate::cryptography::CryptoUtils::hmac_sha256(key, payload.as_bytes()) {
                    let expected_sig = hex::encode(expected_bytes);
                    Some(crate::cryptography::CryptoUtils::constant_time_eq(
                        expected_sig.as_bytes(),
                        sig_hex.as_bytes(),
                    ))
                } else {
                    Some(false)
                }
            } else {
                Some(false)
            }
        } else {
            // Anti-signature-stripping: An unsigned or signature-stripped manifest fails verification immediately
            mismatches.push("SignatureStrippedOrMissing: Release manifest must have a valid cryptographic signature".to_string());
            Some(false)
        };

        let sig_ok = signature_valid.unwrap_or(false) && manifest.signature.is_some();
        let is_valid = mismatches.is_empty() && sig_ok;

        Ok(json!({
            "valid": is_valid,
            "signature_valid": signature_valid.unwrap_or(false),
            "signature_verified": signature_valid,
            "mismatches": mismatches,
            "build_id": manifest.build_id,
            "version": manifest.version,
            "timestamp": manifest.timestamp
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_signing_and_tamper_detection() {
        let temp = std::env::temp_dir().join(format!("tara_sec_manifest_test_{}", std::process::id()));
        let _ = fs::create_dir_all(&temp);
        let mgr = ReleaseManifestManager::new(Some(temp.to_str().unwrap()), None);

        let key = b"secret_test_signing_key_32_bytes!";
        let manifest = mgr
            .generate_release_manifest("build_001", "1.0.0", Some(key))
            .unwrap();
        assert!(manifest.signature.is_some());

        // Verify succeeds
        let res = mgr.verify_release_manifest(Some(key)).unwrap();
        assert_eq!(res["valid"], true);
        assert_eq!(res["signature_valid"], true);

        // Tamper with build_id in manifest file
        let mut tampered = manifest.clone();
        tampered.build_id = "build_tampered".into();
        fs::write(
            &mgr.manifest_path,
            serde_json::to_string_pretty(&tampered).unwrap(),
        )
        .unwrap();

        // Signature must now fail!
        let tampered_res = mgr.verify_release_manifest(Some(key)).unwrap();
        assert_eq!(tampered_res["signature_valid"], false);
        assert_eq!(tampered_res["valid"], false);

        if temp.exists() {
            let _ = fs::remove_dir_all(&temp);
        }
    }

    #[test]
    fn test_manifest_signature_stripping_rejection() {
        let temp = std::env::temp_dir().join(format!("tara_sec_manifest_strip_test_{}", std::process::id()));
        let _ = fs::create_dir_all(&temp);
        let mgr = ReleaseManifestManager::new(Some(temp.to_str().unwrap()), None);

        let key = b"secret_test_signing_key_32_bytes!";
        let manifest = mgr
            .generate_release_manifest("build_002", "1.0.0", Some(key))
            .unwrap();
        assert!(manifest.signature.is_some());

        // Strip the signature from the manifest
        let mut stripped = manifest.clone();
        stripped.signature = None;
        fs::write(
            &mgr.manifest_path,
            serde_json::to_string_pretty(&stripped).unwrap(),
        )
        .unwrap();

        // Verification must fail closed even when no key is explicitly passed
        let stripped_res_no_key = mgr.verify_release_manifest(None).unwrap();
        assert_eq!(stripped_res_no_key["valid"], false);
        assert_eq!(stripped_res_no_key["signature_valid"], false);

        // Verification must also fail closed when key is passed
        let stripped_res_with_key = mgr.verify_release_manifest(Some(key)).unwrap();
        assert_eq!(stripped_res_with_key["valid"], false);
        assert_eq!(stripped_res_with_key["signature_valid"], false);

        if temp.exists() {
            let _ = fs::remove_dir_all(&temp);
        }
    }
}

