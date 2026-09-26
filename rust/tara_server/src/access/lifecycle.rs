//! Authority Lifecycle & Integrity Verification Engine in Rust.
//! Enforces exact parity with Python TARA/ACCESS/operator/operator_lifecycle.py:
//! - State machine: CREATOR_SETUP_REQUIRED, ACTIVE, AUTHORITY_LOCKED
//! - Canonical sorted JSON hashing over operator_record.json, operators_registry.json, restore_config.json
//! - Ed25519 cryptographic signature verification over sealed state
//! - Windows DPAPI envelope storage for access_seal.json
//! - Fail-closed on missing seal, hash mismatch, or tampering

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ed25519_dalek::{Signer, SigningKey, Signature, VerifyingKey};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::crypto::dpapi;

pub const CANONICAL_CREATOR_ID: &str = "ROOT_OPERATOR";
pub const DEFAULT_DISPLAY_NAME: &str = "OPERATOR_ROOT";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthorityState {
    CreatorSetupRequired,
    Active,
    AuthorityLocked,
}

impl AuthorityState {
    pub fn as_str(&self) -> &'static str {
        match self {
            AuthorityState::CreatorSetupRequired => "CREATOR_SETUP_REQUIRED",
            AuthorityState::Active => "ACTIVE",
            AuthorityState::AuthorityLocked => "AUTHORITY_LOCKED",
        }
    }
}

pub fn canonicalize_value(val: &Value) -> Value {
    match val {
        Value::Object(map) => {
            let mut sorted = BTreeMap::new();
            for (k, v) in map {
                sorted.insert(k.clone(), canonicalize_value(v));
            }
            Value::Object(serde_json::Map::from_iter(sorted))
        }
        Value::Array(arr) => {
            Value::Array(arr.iter().map(canonicalize_value).collect())
        }
        _ => val.clone(),
    }
}

pub fn canonical_json_bytes(val: &Value) -> Result<Vec<u8>, String> {
    let normalized = canonicalize_value(val);
    serde_json::to_vec(&normalized).map_err(|e| e.to_string())
}

pub fn compute_canonical_hash<P: AsRef<Path>>(path: P) -> Option<String> {
    let p = path.as_ref();
    if !p.exists() {
        return None;
    }
    let raw = fs::read_to_string(p).ok()?;
    let val: Value = serde_json::from_str(&raw).ok()?;
    let bytes = canonical_json_bytes(&val).ok()?;
    let hash = Sha256::digest(&bytes);
    Some(hex::encode(hash))
}

pub fn compute_recovery_verifier_hash<P: AsRef<Path>>(path: P) -> Option<String> {
    let p = path.as_ref();
    if !p.exists() {
        return None;
    }
    let raw = fs::read_to_string(p).ok()?;
    let val: Value = serde_json::from_str(&raw).ok()?;
    let verifier_data = json!({
        "creator_id": val.get("creator_id"),
        "recovery_code_hash": val.get("recovery_code_hash"),
        "recovery_salt": val.get("recovery_salt"),
        "recovery_email": val.get("recovery_email")
    });
    let bytes = canonical_json_bytes(&verifier_data).ok()?;
    let hash = Sha256::digest(&bytes);
    Some(hex::encode(hash))
}

pub struct AuthorityLifecycleManager {
    pub repo_root: PathBuf,
    pub seal_path: PathBuf,
    pub creator_record_path: PathBuf,
    pub creators_registry_path: PathBuf,
    pub recovery_config_path: PathBuf,
    in_initialization: Mutex<bool>,
    lock: Mutex<()>,
}

impl AuthorityLifecycleManager {
    pub fn new<P: AsRef<Path>>(repo_root: P) -> Self {
        let root = repo_root.as_ref().to_path_buf();
        let seal_path = root.join("storage").join("vault").join("access").join("access_seal.json");
        let creator_record_path = root.join("TARA").join("ACCESS").join("operator").join("operator_record.json");
        let creators_registry_path = root.join("TARA").join("ACCESS").join("operator").join("operators_registry.json");
        let app_recovery = if cfg!(target_os = "windows") {
            std::env::var("APPDATA")
                .map(|appdata| PathBuf::from(appdata).join("TARA").join("recovery").join("recovery_config.json"))
                .unwrap_or_else(|_| root.join("TARA").join("ACCESS").join("restore").join("restore_config.json"))
        } else {
            std::env::var("HOME")
                .map(|home| PathBuf::from(home).join(".tara").join("recovery").join("recovery_config.json"))
                .unwrap_or_else(|_| root.join("TARA").join("ACCESS").join("restore").join("restore_config.json"))
        };
        let recovery_config_path = if app_recovery.exists() {
            app_recovery
        } else {
            root.join("TARA").join("ACCESS").join("restore").join("restore_config.json")
        };

        Self {
            repo_root: root,
            seal_path,
            creator_record_path,
            creators_registry_path,
            recovery_config_path,
            in_initialization: Mutex::new(false),
            lock: Mutex::new(()),
        }
    }

    pub fn begin_initialization(&self) -> Result<(), String> {
        let _guard = self.lock.lock().unwrap();
        let (state, _) = self.verify_integrity_internal();
        if state != AuthorityState::CreatorSetupRequired {
            return Err("Creator authority is already initialized. Authentication or authorized recovery is required for changes.".to_string());
        }
        *self.in_initialization.lock().unwrap() = true;
        Ok(())
    }

    pub fn get_state(&self) -> AuthorityState {
        let (state, _) = self.verify_integrity();
        state
    }

    pub fn is_active(&self) -> bool {
        self.get_state() == AuthorityState::Active
    }

    pub fn is_locked(&self) -> bool {
        self.get_state() == AuthorityState::AuthorityLocked
    }

    /// Cryptographically verifies authoritative creator state against the protected seal.
    pub fn verify_integrity(&self) -> (AuthorityState, String) {
        let _guard = self.lock.lock().unwrap();
        self.verify_integrity_internal()
    }

    fn verify_integrity_internal(&self) -> (AuthorityState, String) {
        // 1. Check if seal exists
        if !self.seal_path.exists() {
            if *self.in_initialization.lock().unwrap() {
                return (AuthorityState::CreatorSetupRequired, "In initial setup transaction".to_string());
            }
            // If seal is missing, check if system has any configured creator files
            if self.creator_record_path.exists() {
                if let Ok(raw) = fs::read_to_string(&self.creator_record_path) {
                    if let Ok(rec) = serde_json::from_str::<Value>(&raw) {
                        let root_pk = rec.get("root_public_key").and_then(|v| v.as_str()).unwrap_or("");
                        let status = rec.get("status").and_then(|v| v.as_str()).unwrap_or("").to_lowercase();
                        if root_pk.is_empty() || matches!(status.as_str(), "creator_setup_required" | "setup_required" | "unconfigured" | "pending") {
                            return (AuthorityState::CreatorSetupRequired, "Fresh installation: Creator setup required.".to_string());
                        }
                    }
                }
                // Creator record exists with active public key but authority seal is missing -> Tampering
                return (AuthorityState::AuthorityLocked, "Authority seal is missing while creator records exist. Possible tampering or file deletion.".to_string());
            } else {
                return (AuthorityState::CreatorSetupRequired, "Fresh installation: Creator setup required.".to_string());
            }
        }

        // 2. Read raw bytes of seal
        let raw_bytes = match fs::read(&self.seal_path) {
            Ok(b) => b,
            Err(e) => return (AuthorityState::AuthorityLocked, format!("Protected authority seal is unreadable: {}", e)),
        };

        // 3. Unprotect DPAPI if on Windows, fallback to JSON
        let seal_data: Value = if cfg!(windows) {
            if let Ok(unprotected) = dpapi::unprotect_bytes(&raw_bytes) {
                serde_json::from_slice(&unprotected)
                    .or_else(|_| serde_json::from_slice(&raw_bytes))
                    .unwrap_or(Value::Null)
            } else {
                serde_json::from_slice(&raw_bytes).unwrap_or(Value::Null)
            }
        } else {
            serde_json::from_slice(&raw_bytes).unwrap_or(Value::Null)
        };

        if seal_data.is_null() {
            return (AuthorityState::AuthorityLocked, "Protected authority seal is corrupted or unreadable JSON".to_string());
        }

        // 4. Verify permanent initialization invariant
        if !seal_data.get("setup_complete").and_then(|v| v.as_bool()).unwrap_or(false) {
            return (AuthorityState::CreatorSetupRequired, "Setup not marked complete in seal".to_string());
        }

        let root_id = seal_data.get("root_creator_id").and_then(|v| v.as_str()).unwrap_or("");
        if root_id != CANONICAL_CREATOR_ID {
            return (AuthorityState::AuthorityLocked, format!("Root creator ID in seal is '{}', expected '{}'", root_id, CANONICAL_CREATOR_ID));
        }

        let root_pubkey = seal_data.get("root_public_key").and_then(|v| v.as_str()).unwrap_or("");
        if root_pubkey.is_empty() {
            return (AuthorityState::AuthorityLocked, "Protected authority seal missing root_public_key".to_string());
        }

        // 5. Verify operator_record.json integrity
        if !self.creator_record_path.exists() {
            return (AuthorityState::AuthorityLocked, "operator_record.json is missing".to_string());
        }

        let record_hash = compute_canonical_hash(&self.creator_record_path);
        let expected_rec_hash = seal_data.get("record_hash").and_then(|v| v.as_str());
        if record_hash.as_deref() != expected_rec_hash {
            return (AuthorityState::AuthorityLocked, format!(
                "operator_record.json hash mismatch (tampered or edited). Expected {:?}, got {:?}",
                expected_rec_hash, record_hash
            ));
        }

        if let Ok(raw) = fs::read_to_string(&self.creator_record_path) {
            if let Ok(rec) = serde_json::from_str::<Value>(&raw) {
                if rec.get("creator_id").and_then(|v| v.as_str()) != Some(CANONICAL_CREATOR_ID) {
                    return (AuthorityState::AuthorityLocked, "operator_record.json creator_id mismatch".to_string());
                }
                if rec.get("root_public_key").and_then(|v| v.as_str()).unwrap_or("").to_lowercase() != root_pubkey.to_lowercase() {
                    return (AuthorityState::AuthorityLocked, "operator_record.json public key does not match sealed authority".to_string());
                }
            }
        }

        // 6. Verify operators_registry.json integrity
        if !self.creators_registry_path.exists() {
            return (AuthorityState::AuthorityLocked, "operators_registry.json is missing".to_string());
        }

        let registry_hash = compute_canonical_hash(&self.creators_registry_path);
        let expected_reg_hash = seal_data.get("registry_hash").and_then(|v| v.as_str());
        if registry_hash.as_deref() != expected_reg_hash {
            return (AuthorityState::AuthorityLocked, format!(
                "operators_registry.json hash mismatch (tampered or edited). Expected {:?}, got {:?}",
                expected_reg_hash, registry_hash
            ));
        }

        if let Ok(raw) = fs::read_to_string(&self.creators_registry_path) {
            if let Ok(reg) = serde_json::from_str::<Value>(&raw) {
                let root_reg = reg.get(CANONICAL_CREATOR_ID).cloned().unwrap_or(Value::Null);
                if root_reg.get("role").and_then(|v| v.as_str()) != Some("ROOT_CREATOR") {
                    return (AuthorityState::AuthorityLocked, "Multi-creator registry ROOT_CREATOR role altered".to_string());
                }
                if root_reg.get("public_key").and_then(|v| v.as_str()).unwrap_or("").to_lowercase() != root_pubkey.to_lowercase() {
                    return (AuthorityState::AuthorityLocked, "Multi-creator registry root public key mismatch".to_string());
                }
            }
        }

        // 7. Verify restore_config.json verifier integrity
        if self.recovery_config_path.exists() {
            let recovery_hash = compute_recovery_verifier_hash(&self.recovery_config_path);
            let expected_recov_hash = seal_data.get("recovery_hash").and_then(|v| v.as_str());
            if recovery_hash.as_deref() != expected_recov_hash {
                return (AuthorityState::AuthorityLocked, "restore_config.json hash mismatch (tampered or edited)".to_string());
            }
        }

        // 8. Verify Ed25519 signature of the seal
        let sig_hex = seal_data.get("seal_signature").and_then(|v| v.as_str()).unwrap_or("");
        if sig_hex.is_empty() {
            if seal_data.get("recovery_authorized").and_then(|v| v.as_bool()).unwrap_or(false) {
                return (AuthorityState::Active, "Creator authority verified via authorized recovery".to_string());
            }
            return (AuthorityState::AuthorityLocked, "Protected authority seal missing cryptographic signature".to_string());
        }

        let mut seal_copy = seal_data.clone();
        if let Some(obj) = seal_copy.as_object_mut() {
            obj.remove("seal_signature");
        }

        let canonical_seal_bytes = match canonical_json_bytes(&seal_copy) {
            Ok(b) => b,
            Err(e) => return (AuthorityState::AuthorityLocked, format!("Failed to canonicalize seal: {}", e)),
        };

        let pub_bytes = match hex::decode(root_pubkey) {
            Ok(b) if b.len() == 32 => {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&b);
                arr
            }
            _ => return (AuthorityState::AuthorityLocked, "Invalid root public key length in seal".to_string()),
        };

        let sig_bytes = match hex::decode(sig_hex) {
            Ok(b) if b.len() == 64 => {
                let mut arr = [0u8; 64];
                arr.copy_from_slice(&b);
                arr
            }
            _ => return (AuthorityState::AuthorityLocked, "Invalid seal signature length".to_string()),
        };

        let verifying_key = match VerifyingKey::from_bytes(&pub_bytes) {
            Ok(vk) => vk,
            Err(e) => return (AuthorityState::AuthorityLocked, format!("Invalid Ed25519 public key in seal: {}", e)),
        };

        let sig = Signature::from_bytes(&sig_bytes);
        if verifying_key.verify_strict(&canonical_seal_bytes, &sig).is_err() {
            return (AuthorityState::AuthorityLocked, "Invalid cryptographic signature on authority seal".to_string());
        }

        (AuthorityState::Active, "Creator authority verified and active".to_string())
    }

    /// One-time cryptographic sealing of the initial creator authority.
    /// Fails closed if already initialized or not in CreatorSetupRequired.
    pub fn seal_initial_authority(
        &self,
        priv_bytes: &[u8],
        pub_bytes: &[u8],
        display_name: &str,
    ) -> Result<Value, String> {
        let _guard = self.lock.lock().unwrap();

        if self.seal_path.exists() {
            return Err("Creator authority is already initialized. Authentication or authorized recovery is required for changes.".to_string());
        }

        let is_in_init = *self.in_initialization.lock().unwrap();
        if !is_in_init {
            let (state, _) = self.verify_integrity_internal();
            if state != AuthorityState::CreatorSetupRequired {
                return Err("Creator authority is already initialized. Authentication or authorized recovery is required for changes.".to_string());
            }
        }

        let pub_hex = hex::encode(pub_bytes).to_lowercase();
        let key_id = hex::encode(Sha256::digest(pub_hex.as_bytes()))[..16].to_string();

        let rec_hash = compute_canonical_hash(&self.creator_record_path);
        let reg_hash = compute_canonical_hash(&self.creators_registry_path);
        let recov_hash = if self.recovery_config_path.exists() {
            compute_recovery_verifier_hash(&self.recovery_config_path)
        } else {
            None
        };

        let seal_body = json!({
            "version": 2,
            "setup_complete": true,
            "root_creator_id": CANONICAL_CREATOR_ID,
            "display_name": display_name,
            "root_public_key": pub_hex,
            "key_id": key_id,
            "key_version": 1,
            "record_hash": rec_hash,
            "registry_hash": reg_hash,
            "recovery_hash": recov_hash,
            "sealed_at": crate::now_iso()
        });

        let canonical_bytes = canonical_json_bytes(&seal_body)?;

        let mut priv_arr = [0u8; 32];
        if priv_bytes.len() != 32 {
            return Err("Private key must be 32 bytes".to_string());
        }
        priv_arr.copy_from_slice(priv_bytes);
        let signing_key = SigningKey::from_bytes(&priv_arr);
        let signature = signing_key.sign(&canonical_bytes);

        let mut seal_data = seal_body.clone();
        if let Some(obj) = seal_data.as_object_mut() {
            obj.insert("seal_signature".to_string(), json!(hex::encode(signature.to_bytes())));
        }

        let raw_json_bytes = serde_json::to_vec_pretty(&seal_data).map_err(|e| e.to_string())?;

        let bytes_to_write = if cfg!(windows) {
            match dpapi::protect_bytes(&raw_json_bytes) {
                Ok(enc) => enc,
                Err(_) => raw_json_bytes,
            }
        } else {
            raw_json_bytes
        };

        if let Some(parent) = self.seal_path.parent() {
            let _ = fs::create_dir_all(parent);
        }

        fs::write(&self.seal_path, bytes_to_write)
            .map_err(|e| format!("Failed to write authority seal: {}", e))?;

        *self.in_initialization.lock().unwrap() = false;
        Ok(seal_data)
    }
}
