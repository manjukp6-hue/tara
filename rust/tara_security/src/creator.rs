//! Creator Root Authority Lifecycle, Setup Engine, and Bootstrap Wizard.
//!
//! Enforces:
//! - State machine: CREATOR_SETUP_REQUIRED -> ACTIVE -> AUTHORITY_LOCKED.
//! - Canonical sorted JSON hashing over operator_record.json, operators_registry.json, restore_config.json.
//! - Ed25519 cryptographic signature verification over sealed state.
//! - Windows DPAPI envelope storage for access_seal.json (fails closed on DPAPI failure).
//! - Fail-closed on missing seal, hash mismatch, or tampering.
//! - Cold-start setup ceremony generating Ed25519 root authority keypair, Scrypt + AES-256-GCM + DPAPI private key storage,
//!   high-entropy recovery code, and authority state seal.
//! - Bootstrap wizard with terminal QR code rendering.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[cfg(windows)]
use crate::dpapi::native_dpapi;
use crate::dpapi::SecureKeyStorage;
use crate::persistence::atomic_write_file_hardened;
use crate::qr::QrCodeMatrix;
use crate::recovery::pbkdf2_hmac_sha256;
use crate::zero_trust::GoogleAuthService;

pub const CANONICAL_CREATOR_ID: &str = "ROOT_OPERATOR";
pub const DEFAULT_DISPLAY_NAME: &str = "OPERATOR_ROOT";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
        Value::Array(arr) => Value::Array(arr.iter().map(canonicalize_value).collect()),
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
        "verifier_version": val.get("verifier_version").unwrap_or(&json!(2)),
        "creator_id": val.get("creator_id"),
        "recovery_code_hash": val.get("recovery_code_hash"),
        "recovery_salt": val.get("recovery_salt"),
        "recovery_email": val.get("recovery_email"),
        "iterations": val.get("iterations")
    });
    let bytes = canonical_json_bytes(&verifier_data).ok()?;
    let hash = Sha256::digest(&bytes);
    Some(hex::encode(hash))
}

/// Authority Lifecycle Manager handling seals, state transitions, and tamper detection.
pub struct AuthorityLifecycleManager {
    repo_root: PathBuf,
    seal_path: PathBuf,
    creator_record_path: PathBuf,
    creators_registry_path: PathBuf,
    recovery_config_path: PathBuf,
    in_initialization: Mutex<bool>,
}

impl AuthorityLifecycleManager {
    pub fn new<P: AsRef<Path>>(repo_root: P) -> Self {
        let root = repo_root.as_ref().to_path_buf();
        let access_dir = root.join("storage").join("vault").join("access");
        Self {
            repo_root: root.clone(),
            seal_path: access_dir.join("access_seal.json"),
            creator_record_path: access_dir.join("operator_record.json"),
            creators_registry_path: access_dir.join("operators_registry.json"),
            recovery_config_path: access_dir.join("restore_config.json"),
            in_initialization: Mutex::new(false),
        }
    }

    pub fn repo_root(&self) -> &Path {
        &self.repo_root
    }

    pub fn set_in_initialization(&self, in_init: bool) {
        if let Ok(mut lock) = self.in_initialization.lock() {
            *lock = in_init;
        }
    }

    pub fn is_in_initialization(&self) -> bool {
        self.in_initialization.lock().map(|l| *l).unwrap_or(false)
    }

    pub fn verify_integrity(&self) -> (AuthorityState, String) {
        if let Ok(lock) = self.in_initialization.lock() {
            if *lock {
                return (
                    AuthorityState::CreatorSetupRequired,
                    "Initialization ceremony in progress".to_string(),
                );
            }
        }

        let watermark_path =
            crate::dpapi::get_os_isolated_watermark_path("creator_authority_initialized");
        let was_previously_initialized = crate::dpapi::load_monotonic_watermark(&watermark_path)
            .ok()
            .flatten()
            .map(|val| val > 0)
            .unwrap_or(false);

        if !self.creator_record_path.exists() && !self.seal_path.exists() {
            if was_previously_initialized {
                return (
                    AuthorityState::AuthorityLocked,
                    "Established authority state files were deleted or missing: fail closed into permanent lockdown".to_string(),
                );
            }
            return (
                AuthorityState::CreatorSetupRequired,
                "No creator authority records found (cold start)".to_string(),
            );
        }

        if !self.seal_path.exists() {
            return (
                AuthorityState::AuthorityLocked,
                "Authority seal missing: fail closed into lockdown".to_string(),
            );
        }

        // Read and unprotect seal envelope
        let seal_raw = match fs::read(&self.seal_path) {
            Ok(d) => d,
            Err(e) => {
                return (
                    AuthorityState::AuthorityLocked,
                    format!("Cannot read seal: {}", e),
                )
            }
        };

        #[cfg(windows)]
        let seal_bytes = match native_dpapi::unprotect_bytes(&seal_raw) {
            Ok(b) => b,
            Err(e) => {
                return (
                    AuthorityState::AuthorityLocked,
                    format!("DPAPI unprotect failed: {}", e),
                )
            }
        };
        #[cfg(not(windows))]
        let seal_bytes = seal_raw;

        let seal_val: Value = match serde_json::from_slice(&seal_bytes) {
            Ok(v) => v,
            Err(e) => {
                return (
                    AuthorityState::AuthorityLocked,
                    format!("Corrupted seal JSON: {}", e),
                )
            }
        };

        let operator_hash = compute_canonical_hash(&self.creator_record_path);
        let registry_hash = compute_canonical_hash(&self.creators_registry_path);
        let recovery_hash = compute_recovery_verifier_hash(&self.recovery_config_path);

        let sealed_op_hash = seal_val
            .get("operator_record_hash")
            .and_then(|v| v.as_str());
        let sealed_reg_hash = seal_val
            .get("operators_registry_hash")
            .and_then(|v| v.as_str());
        let sealed_rec_hash = seal_val.get("restore_config_hash").and_then(|v| v.as_str());

        if operator_hash.as_deref() != sealed_op_hash {
            return (
                AuthorityState::AuthorityLocked,
                "Operator record hash mismatch (tampered)".to_string(),
            );
        }
        if registry_hash.as_deref() != sealed_reg_hash {
            return (
                AuthorityState::AuthorityLocked,
                "Operators registry hash mismatch (tampered)".to_string(),
            );
        }
        if recovery_hash.as_deref() != sealed_rec_hash {
            return (
                AuthorityState::AuthorityLocked,
                "Recovery config hash mismatch (tampered)".to_string(),
            );
        }

        // Verify cryptographic Ed25519 signature
        if let (Some(pub_hex), Some(sig_hex)) = (
            seal_val.get("root_public_key").and_then(|v| v.as_str()),
            seal_val.get("seal_signature").and_then(|v| v.as_str()),
        ) {
            let pub_bytes = match hex::decode(pub_hex) {
                Ok(b) if b.len() == 32 => b,
                _ => {
                    return (
                        AuthorityState::AuthorityLocked,
                        "Invalid public key in seal".into(),
                    )
                }
            };
            let sig_bytes = match hex::decode(sig_hex) {
                Ok(b) if b.len() == 64 => b,
                _ => {
                    return (
                        AuthorityState::AuthorityLocked,
                        "Invalid signature in seal".into(),
                    )
                }
            };
            let mut pk_arr = [0u8; 32];
            pk_arr.copy_from_slice(&pub_bytes);
            let vk = match VerifyingKey::from_bytes(&pk_arr) {
                Ok(k) => k,
                Err(_) => {
                    return (
                        AuthorityState::AuthorityLocked,
                        "Invalid Ed25519 verifying key".into(),
                    )
                }
            };
            let sig = Signature::from_bytes(sig_bytes.as_slice().try_into().unwrap());

            let payload_to_verify = format!(
                "SEAL:2:{}:{}:{}",
                sealed_op_hash.unwrap_or(""),
                sealed_reg_hash.unwrap_or(""),
                sealed_rec_hash.unwrap_or("")
            );

            if vk
                .verify_strict(payload_to_verify.as_bytes(), &sig)
                .is_err()
            {
                return (
                    AuthorityState::AuthorityLocked,
                    "Cryptographic seal signature verification failed".into(),
                );
            }
        } else {
            return (
                AuthorityState::AuthorityLocked,
                "Missing root public key or seal signature".into(),
            );
        }

        (
            AuthorityState::Active,
            "Authority state verified and operational".to_string(),
        )
    }

    pub fn seal_authority_state(&self, signing_key: &SigningKey) -> Result<(), String> {
        let operator_hash = compute_canonical_hash(&self.creator_record_path)
            .ok_or("Cannot compute operator_record hash")?;
        let registry_hash = compute_canonical_hash(&self.creators_registry_path)
            .ok_or("Cannot compute operators_registry hash")?;
        let recovery_hash = compute_recovery_verifier_hash(&self.recovery_config_path)
            .ok_or("Cannot compute restore_config hash")?;

        let payload_to_sign = format!(
            "SEAL:2:{}:{}:{}",
            operator_hash, registry_hash, recovery_hash
        );
        let sig = signing_key.sign(payload_to_sign.as_bytes());
        let pub_hex = hex::encode(signing_key.verifying_key().as_bytes());

        let seal_record = json!({
            "version": 2,
            "root_public_key": pub_hex,
            "operator_record_hash": operator_hash,
            "operators_registry_hash": registry_hash,
            "restore_config_hash": recovery_hash,
            "seal_signature": hex::encode(sig.to_bytes()),
        });

        let seal_data = serde_json::to_vec_pretty(&seal_record).map_err(|e| e.to_string())?;

        #[cfg(windows)]
        let payload = native_dpapi::protect_bytes(&seal_data)?;
        #[cfg(not(windows))]
        let payload = seal_data;

        atomic_write_file_hardened(&self.seal_path, &payload)?;
        let watermark_path =
            crate::dpapi::get_os_isolated_watermark_path("creator_authority_initialized");
        let _ = crate::dpapi::store_monotonic_watermark(&watermark_path, 1);
        Ok(())
    }

    pub fn is_active(&self) -> bool {
        let (state, _) = self.verify_integrity();
        state == AuthorityState::Active
    }

    pub fn is_locked(&self) -> bool {
        let (state, _) = self.verify_integrity();
        state == AuthorityState::AuthorityLocked
    }

    pub fn begin_initialization(&self) -> Result<(), String> {
        self.set_in_initialization(true);
        Ok(())
    }

    pub fn seal_initial_authority(
        &self,
        signing_key_bytes: &[u8],
        _verifying_key_bytes: &[u8],
        _display_name: &str,
    ) -> Result<(), String> {
        if signing_key_bytes.len() != 32 {
            return Err("Invalid signing key length".into());
        }
        let arr: [u8; 32] = signing_key_bytes.try_into().unwrap();
        let sk = SigningKey::from_bytes(&arr);
        self.seal_authority_state(&sk)?;
        self.set_in_initialization(false);
        Ok(())
    }

    pub fn force_lockdown(&self) -> Result<(), String> {
        if self.seal_path.exists() {
            let _ = fs::remove_file(&self.seal_path);
        }
        Ok(())
    }

    /// Authoritatively retrieves the active root public key if authority integrity checks pass.
    pub fn get_root_public_key(&self) -> Option<String> {
        let (state, _) = self.verify_integrity();
        if state != AuthorityState::Active {
            return None;
        }
        if !self.creator_record_path.exists() {
            return None;
        }
        let data = fs::read(&self.creator_record_path).ok()?;
        let val: Value = serde_json::from_slice(&data).ok()?;
        val.get("root_public_key")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }
}

/// Revoked root key record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevokedKey {
    pub key_version: u64,
    pub public_key: String,
    pub revoked_at: String,
    pub reason: String,
}

/// Canonical operator/creator identity record stored on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatorIdentityRecord {
    pub creator_id: String,
    pub display_name: String,
    pub identity_version: u64,
    pub key_version: u64,
    pub root_public_key: Option<String>,
    pub status: String,
    pub recovery_enabled: bool,
    pub created_at: Option<String>,
    pub last_key_rotation: Option<String>,
    pub revoked_keys: Vec<RevokedKey>,
}

pub struct CreatorIdentity {
    record_path: PathBuf,
    record: CreatorIdentityRecord,
}

impl CreatorIdentity {
    pub fn new<P: AsRef<Path>>(record_path: Option<P>) -> Result<Self, String> {
        let p = match record_path {
            Some(path) => path.as_ref().to_path_buf(),
            None => PathBuf::from("storage/vault/access/operator_record.json"),
        };

        let mut identity = Self {
            record_path: p,
            record: CreatorIdentityRecord {
                creator_id: CANONICAL_CREATOR_ID.to_string(),
                display_name: DEFAULT_DISPLAY_NAME.to_string(),
                identity_version: 1,
                key_version: 1,
                root_public_key: None,
                status: "CREATOR_SETUP_REQUIRED".to_string(),
                recovery_enabled: false,
                created_at: None,
                last_key_rotation: None,
                revoked_keys: Vec::new(),
            },
        };

        if identity.record_path.exists() {
            identity.load()?;
        }

        Ok(identity)
    }

    pub fn record_path(&self) -> &Path {
        &self.record_path
    }

    pub fn record(&self) -> &CreatorIdentityRecord {
        &self.record
    }

    pub fn load(&mut self) -> Result<(), String> {
        let data = fs::read_to_string(&self.record_path)
            .map_err(|e| format!("Failed to read operator record: {}", e))?;
        let rec: CreatorIdentityRecord = serde_json::from_str(&data)
            .map_err(|e| format!("Corrupt operator record JSON: {}", e))?;
        if rec.creator_id != CANONICAL_CREATOR_ID {
            return Err(format!(
                "Operator record creator_id mismatch: expected '{}', found '{}'",
                CANONICAL_CREATOR_ID, rec.creator_id
            ));
        }
        self.record = rec;
        Ok(())
    }

    pub fn save(&self) -> Result<(), String> {
        let data = serde_json::to_string_pretty(&self.record).map_err(|e| e.to_string())?;
        atomic_write_file_hardened(&self.record_path, data.as_bytes())
    }

    pub fn is_initialized(&self) -> bool {
        self.record.root_public_key.is_some() && self.record.status.to_uppercase() == "ACTIVE"
    }

    pub fn initialize_root_creator(
        &mut self,
        public_key_hex: &str,
        display_name: Option<&str>,
    ) -> Result<CreatorIdentityRecord, String> {
        if self.is_initialized() {
            return Err("Creator authority is already initialized. Authentication or authorized recovery is required for changes.".into());
        }

        Self::validate_ed25519_pubkey(public_key_hex)?;

        let now_epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.record.creator_id = CANONICAL_CREATOR_ID.to_string();
        self.record.display_name = display_name.unwrap_or(DEFAULT_DISPLAY_NAME).to_string();
        self.record.root_public_key = Some(public_key_hex.to_lowercase());
        self.record.identity_version = 1;
        self.record.key_version = 1;
        self.record.status = "ACTIVE".to_string();
        self.record.created_at = Some(now_epoch.to_string());

        self.save()?;
        Ok(self.record.clone())
    }

    pub fn validate_ed25519_pubkey(public_key_hex: &str) -> Result<[u8; 32], String> {
        let bytes =
            hex::decode(public_key_hex).map_err(|e| format!("Invalid hex public key: {}", e))?;
        if bytes.len() != 32 {
            return Err("Public key must be exactly 32 bytes (64 hex chars)".into());
        }
        let arr: [u8; 32] = bytes[..32].try_into().unwrap();
        VerifyingKey::from_bytes(&arr)
            .map_err(|e| format!("Invalid Ed25519 public key curve point: {:?}", e))?;
        Ok(arr)
    }

    pub fn rotate_root_key(
        &mut self,
        new_public_key_hex: &str,
        reason: &str,
    ) -> Result<(), String> {
        Self::validate_ed25519_pubkey(new_public_key_hex)?;

        let old_key = self
            .record
            .root_public_key
            .clone()
            .ok_or_else(|| "No active key to rotate".to_string())?;

        let now_epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let revocation = RevokedKey {
            key_version: self.record.key_version,
            public_key: old_key,
            revoked_at: now_epoch.to_string(),
            reason: reason.to_string(),
        };

        self.record.revoked_keys.push(revocation);
        let next_ver = self
            .record
            .key_version
            .checked_add(1)
            .ok_or_else(|| "CounterExhausted: key version reached u64::MAX".to_string())?;
        self.record.key_version = next_ver;
        self.record.root_public_key = Some(new_public_key_hex.to_lowercase());
        self.record.last_key_rotation = Some(now_epoch.to_string());

        self.save()
    }

    pub fn update_display_name(&mut self, new_name: &str) -> Result<(), String> {
        let clean = new_name.trim();
        if clean.is_empty() {
            return Err("Display name cannot be empty".into());
        }
        self.record.display_name = clean.to_string();
        self.save()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatorSetupRequest {
    pub display_name: Option<String>,
    pub passphrase: String,
    pub google_email: Option<String>,
    pub device_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatorSetupResult {
    pub success: bool,
    pub creator_id: String,
    pub root_public_key: String,
    pub recovery_code: String,
    pub message: String,
}

/// Creator Setup Engine handling cold-start initialization of root authority.
pub struct CreatorSetupEngine {
    repo_root: String,
    lifecycle: Arc<AuthorityLifecycleManager>,
    key_storage: Arc<SecureKeyStorage>,
}

impl CreatorSetupEngine {
    pub fn new(repo_root: &str) -> Self {
        let root_path = PathBuf::from(repo_root);
        let lifecycle = Arc::new(AuthorityLifecycleManager::new(&root_path));
        let vault_dir = root_path.join("storage").join("vault").join("access");
        let key_storage = Arc::new(SecureKeyStorage::new(vault_dir));

        Self {
            repo_root: repo_root.to_string(),
            lifecycle,
            key_storage,
        }
    }

    pub fn lifecycle(&self) -> &Arc<AuthorityLifecycleManager> {
        &self.lifecycle
    }

    pub fn is_setup_required(&self) -> bool {
        let (state, _) = self.lifecycle.verify_integrity();
        state == AuthorityState::CreatorSetupRequired
    }

    pub fn setup_creator(
        &self,
        request: CreatorSetupRequest,
    ) -> Result<CreatorSetupResult, String> {
        self.initialize_creator_trust_root(request)
    }

    pub fn initialize_creator_trust_root(
        &self,
        request: CreatorSetupRequest,
    ) -> Result<CreatorSetupResult, String> {
        if !self.is_setup_required() {
            return Err("Authority already initialized; re-bootstrap prohibited".into());
        }

        if request.passphrase.trim().len() < 8 {
            return Err("Creator passphrase must be at least 8 characters".into());
        }

        self.lifecycle.set_in_initialization(true);

        let access_dir = PathBuf::from(&self.repo_root)
            .join("storage")
            .join("vault")
            .join("access");
        let _ = fs::create_dir_all(&access_dir);

        // 1. Generate root authority keypair
        let mut csprng = rand::rngs::OsRng;
        let signing_key = SigningKey::generate(&mut csprng);
        let verifying_key = signing_key.verifying_key();
        let pub_hex = hex::encode(verifying_key.as_bytes());

        // 2. Encrypt and store private key with Scrypt + AES-256-GCM + DPAPI
        self.key_storage.store_key(
            "creator_root_key",
            &signing_key.to_bytes(),
            &request.passphrase,
        )?;

        // 3. Generate high-entropy recovery code
        let mut rand_bytes = [0u8; 16];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut rand_bytes);
        let raw_hex = hex::encode_upper(rand_bytes);
        let recovery_code = format!(
            "TARA-{}-{}-{}-{}",
            &raw_hex[0..4],
            &raw_hex[4..8],
            &raw_hex[8..12],
            &raw_hex[12..16]
        );

        let mut salt = [0u8; 16];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut salt);
        let mut derived = [0u8; 32];
        pbkdf2_hmac_sha256(recovery_code.as_bytes(), &salt, 100_000, &mut derived);

        let recovery_config = json!({
            "verifier_version": 2,
            "creator_id": CANONICAL_CREATOR_ID,
            "recovery_code_hash": hex::encode(derived),
            "recovery_salt": hex::encode(salt),
            "iterations": 100_000,
            "recovery_email": request.google_email,
        });
        let rec_path = access_dir.join("restore_config.json");
        let rec_data = serde_json::to_string_pretty(&recovery_config).map_err(|e| e.to_string())?;
        atomic_write_file_hardened(&rec_path, rec_data.as_bytes())?;

        // 4. Write operator_record.json
        let now_epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let now_str = format!("{now_epoch}");
        let op_record = json!({
            "creator_id": CANONICAL_CREATOR_ID,
            "display_name": request.display_name.unwrap_or_else(|| DEFAULT_DISPLAY_NAME.to_string()),
            "identity_version": 1,
            "key_version": 1,
            "root_public_key": pub_hex,
            "status": "ACTIVE",
            "recovery_enabled": true,
            "created_at": now_str,
            "last_key_rotation": null,
            "revoked_keys": []
        });
        let op_path = access_dir.join("operator_record.json");
        let op_data = serde_json::to_string_pretty(&op_record).map_err(|e| e.to_string())?;
        atomic_write_file_hardened(&op_path, op_data.as_bytes())?;

        // 5. Write operators_registry.json
        let reg_record = json!({
            "operators": {
                CANONICAL_CREATOR_ID: op_record
            }
        });
        let reg_path = access_dir.join("operators_registry.json");
        let reg_data = serde_json::to_string_pretty(&reg_record).map_err(|e| e.to_string())?;
        atomic_write_file_hardened(&reg_path, reg_data.as_bytes())?;

        // 6. Seal authority state
        self.lifecycle.set_in_initialization(false);
        self.lifecycle.seal_authority_state(&signing_key)?;

        Ok(CreatorSetupResult {
            success: true,
            creator_id: CANONICAL_CREATOR_ID.to_string(),
            root_public_key: pub_hex,
            recovery_code,
            message: "Creator root authority ceremony completed successfully".to_string(),
        })
    }
}

pub struct QrPrompt;

impl QrPrompt {
    pub fn display_terminal_qr(content: &str) -> String {
        match QrCodeMatrix::encode(content) {
            Ok(qr) => qr.render_ascii(),
            Err(e) => format!("Failed to generate QR matrix: {}", e),
        }
    }

    pub fn format_fallback_secret(secret: &str) -> String {
        format!(
            "================ FALLBACK ENROLLMENT CODE ================\n\
             {}\n\
             ==========================================================",
            secret
        )
    }
}

pub struct BootstrapWizard {
    pub setup_engine: CreatorSetupEngine,
}

impl BootstrapWizard {
    pub fn new(repo_root: &str) -> Self {
        Self {
            setup_engine: CreatorSetupEngine::new(repo_root),
        }
    }

    pub fn execute_bootstrap(
        &self,
        request: CreatorSetupRequest,
    ) -> Result<CreatorSetupResult, String> {
        self.setup_engine.initialize_creator_trust_root(request)
    }

    pub fn render_bootstrap_prompt(&self, challenge_url: &str, fallback_code: &str) -> String {
        let mut out = String::new();
        out.push_str("╔══════════════════════════════════════════════════════════════╗\n");
        out.push_str("║             TARA CREATOR ROOT IDENTITY INITIALIZATION        ║\n");
        out.push_str("╚══════════════════════════════════════════════════════════════╝\n\n");
        out.push_str("Scan the QR code below using your trusted creator device:\n\n");
        out.push_str(&QrPrompt::display_terminal_qr(challenge_url));
        out.push_str("\n\n");
        out.push_str(&QrPrompt::format_fallback_secret(fallback_code));
        out.push('\n');
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Creator Authentication Service
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CreatorSessionRecord {
    pub actor_id: String,
    pub role: String,
    pub created_at: Instant,
    pub client_ip: String,
}

pub struct RateLimiter {
    attempts: HashMap<String, (u32, Instant, Option<Instant>)>,
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            attempts: HashMap::new(),
        }
    }

    pub fn check(&mut self, key: &str) -> (bool, u64) {
        if key.len() > 64 {
            return (false, 300);
        }
        let now = Instant::now();
        if let Some((count, last_failure, lockout)) = self.attempts.get_mut(key) {
            if let Some(lock_end) = lockout {
                if now < *lock_end {
                    let rem = (*lock_end - now).as_secs();
                    return (false, rem);
                } else {
                    *lockout = None;
                    *count = 0;
                }
            } else if now.duration_since(*last_failure) > Duration::from_secs(300) {
                *count = 0;
            }
        }
        (true, 0)
    }

    pub fn record_failure(&mut self, key: &str) {
        let key_bounded = if key.len() > 64 { &key[..64] } else { key };
        let now = Instant::now();
        if self.attempts.len() >= 10_000 {
            self.attempts.retain(|_, (_, last, lock)| {
                if let Some(end) = lock {
                    now < *end
                } else {
                    now.duration_since(*last) < Duration::from_secs(300)
                }
            });
            if self.attempts.len() >= 10_000 {
                // Only evict unlocked entries to preserve active security lockouts
                let unlocked_keys: Vec<String> = self
                    .attempts
                    .iter()
                    .filter(|(_, (_, _, lock))| lock.is_none())
                    .take(1000)
                    .map(|(k, _)| k.clone())
                    .collect();
                for k in unlocked_keys {
                    self.attempts.remove(&k);
                }
                // If table is saturated with active lockouts, fail closed: do not evict active lockouts
                if self.attempts.len() >= 10_000 && !self.attempts.contains_key(key_bounded) {
                    return;
                }
            }
        }
        let entry = self
            .attempts
            .entry(key_bounded.to_string())
            .or_insert((0, now, None));
        entry.0 += 1;
        entry.1 = now;
        if entry.0 >= 5 {
            entry.2 = Some(now + Duration::from_secs(300));
        }
    }

    pub fn record_success(&mut self, key: &str) {
        self.attempts.remove(key);
    }
}

/// Request parameters for initial creator setup ceremony.
#[derive(Debug, Clone)]
pub struct CreatorCeremonyRequest<'a> {
    pub google_id_token: &'a str,
    pub confirm_identity: bool,
    pub device_name: &'a str,
    pub device_public_key: Option<&'a str>,
    pub passphrase: Option<&'a str>,
    pub nonce: Option<&'a str>,
    pub client_ip: Option<&'a str>,
}

/// Creator authentication service: manages hardened creator sessions,
/// Scrypt + DPAPI keystore, RS256 JWKS Google auth, and sealed lifecycle state.
pub struct CreatorAuthService {
    pub repo_root: String,
    pub lifecycle: Arc<AuthorityLifecycleManager>,
    pub key_storage: Arc<SecureKeyStorage>,
    pub google_auth: Arc<GoogleAuthService>,
    pub recovery_manager: Arc<crate::recovery::RecoveryManager>,
    sessions: Mutex<HashMap<String, CreatorSessionRecord>>,
    qr_challenges: Mutex<HashMap<String, (String, Instant)>>,
    key_challenges: Mutex<HashMap<String, Instant>>,
    google_challenges: Mutex<HashMap<String, Instant>>,
    rate_limiter: Mutex<RateLimiter>,
    _process_lock_handle: Mutex<Option<fs::File>>,
}

impl CreatorAuthService {
    pub fn new(repo_root: &str) -> Arc<Self> {
        let root_path = PathBuf::from(repo_root);
        let lifecycle = Arc::new(AuthorityLifecycleManager::new(&root_path));
        let vault_dir = root_path.join("storage").join("vault").join("access");
        let key_storage = Arc::new(SecureKeyStorage::new(vault_dir));
        let google_auth = Arc::new(GoogleAuthService::new(None, None));
        let recovery_manager =
            Arc::new(crate::recovery::RecoveryManager::from_base_dir(&root_path));

        Arc::new(Self {
            repo_root: repo_root.to_string(),
            lifecycle,
            key_storage,
            google_auth,
            recovery_manager,
            sessions: Mutex::new(HashMap::new()),
            qr_challenges: Mutex::new(HashMap::new()),
            key_challenges: Mutex::new(HashMap::new()),
            google_challenges: Mutex::new(HashMap::new()),
            rate_limiter: Mutex::new(RateLimiter::new()),
            _process_lock_handle: Mutex::new(None),
        })
    }

    pub fn acquire_process_lock(&self) -> Result<PathBuf, String> {
        let lock_dir = Path::new(&self.repo_root)
            .join("storage")
            .join("vault")
            .join("access");
        fs::create_dir_all(&lock_dir).map_err(|e| {
            format!(
                "Failed to create auth lock dir {}: {}",
                lock_dir.display(),
                e
            )
        })?;
        let lock_path = lock_dir.join("tara_auth_service.lock");

        #[cfg(windows)]
        let file_res = {
            use std::os::windows::fs::OpenOptionsExt;
            fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .share_mode(0)
                .open(&lock_path)
        };
        #[cfg(not(windows))]
        let file_res = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path);

        let mut file = file_res.map_err(|e| {
            format!(
                "SingleDaemonLockContention: active OS process lock already held on {:?}: {}",
                lock_path, e
            )
        })?;

        let pid = std::process::id();
        file.set_len(0)
            .map_err(|e| format!("Failed to truncate auth process lock: {}", e))?;
        file.write_all(format!("{}\n", pid).as_bytes())
            .map_err(|e| format!("Failed to write PID to auth process lock: {}", e))?;
        file.sync_all().map_err(|e| e.to_string())?;
        crate::hardware_root::platform::harden_file_permissions(&lock_path)?;

        let mut handle_guard = match self._process_lock_handle.lock() {
            Ok(g) => g,
            Err(_) => {
                return Err("SecurityMutexPoisoned: process lock handle state corrupted".to_string())
            }
        };
        *handle_guard = Some(file);
        Ok(lock_path)
    }

    pub fn release_process_lock(&self) {
        if let Ok(mut handle) = self._process_lock_handle.lock() {
            *handle = None;
        }
    }

    pub fn create_key_challenge(&self) -> String {
        let mut nonce_bytes = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut nonce_bytes);
        let nonce = hex::encode(nonce_bytes);
        let mut ch = match self.key_challenges.lock() {
            Ok(c) => c,
            Err(_) => return String::new(),
        };
        ch.retain(|_, ts| ts.elapsed() < Duration::from_secs(300));
        ch.insert(nonce.clone(), Instant::now());
        nonce
    }

    /// Issue a single-use server-side challenge nonce for Google OAuth authentication (Anti-Replay defense).
    pub fn create_google_auth_nonce(&self) -> String {
        match self.google_auth.create_auth_nonce(300.0) {
            Ok(nonce) => {
                if let Ok(mut ch) = self.google_challenges.lock() {
                    ch.retain(|_, ts| ts.elapsed() < Duration::from_secs(300));
                    ch.insert(nonce.clone(), Instant::now());
                }
                nonce
            }
            Err(_) => String::new(),
        }
    }

    pub fn verify_session(&self, token: &str) -> Option<HashMap<String, String>> {
        if self.lifecycle.is_locked() {
            return None;
        }
        let mut sessions = match self.sessions.lock() {
            Ok(s) => s,
            Err(_) => return None,
        };
        sessions.retain(|_, s| s.created_at.elapsed() < Duration::from_secs(3600));
        if let Some(sess) = sessions.get(token) {
            let mut m = HashMap::new();
            m.insert("creator_id".to_string(), sess.actor_id.clone());
            m.insert("role".to_string(), sess.role.clone());
            m.insert("client_ip".to_string(), sess.client_ip.clone());
            return Some(m);
        }
        None
    }

    /// Mint an authenticated creator principal directly from a verified session token.
    pub fn mint_creator_principal(
        &self,
        token: &str,
    ) -> Result<crate::primitives::AuthenticatedPrincipal, String> {
        let sess = self
            .verify_session(token)
            .ok_or_else(|| "Session token is invalid or expired".to_string())?;
        let actor_id = sess
            .get("creator_id")
            .ok_or("Missing creator_id in verified session")?;
        let proof = crate::primitives::compute_session_proof_hex(actor_id, token);
        crate::primitives::AuthenticatedPrincipal::mint_creator(actor_id, token, &proof)
    }

    /// Mint a strongly-typed VerifiedPrincipal with Creator privileges strictly from a verified session token.
    pub fn mint_verified_creator_principal(
        &self,
        token: &str,
        generation: crate::primitives::AuthorityGeneration,
    ) -> Result<crate::primitives::VerifiedPrincipal, String> {
        let sess = self
            .verify_session(token)
            .ok_or_else(|| "Creator session token is invalid or expired".to_string())?;
        let actor_id = sess
            .get("creator_id")
            .ok_or("Missing creator_id in verified session")?;
        let pid = crate::primitives::PrincipalId::new(format!("actor_{}", actor_id))?;
        let proof = crate::primitives::compute_session_proof_hex(&pid.as_str(), token);
        let mut claims = HashMap::new();
        claims.insert(
            "session_id".to_string(),
            serde_json::Value::String(token.to_string()),
        );
        claims.insert(
            "session_proof".to_string(),
            serde_json::Value::String(proof),
        );
        if let Some(ip) = sess.get("client_ip") {
            claims.insert(
                "client_ip".to_string(),
                serde_json::Value::String(ip.clone()),
            );
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        crate::primitives::VerifiedPrincipal::construct_verified(
            pid,
            crate::primitives::PrincipalRole::creator(),
            crate::primitives::AssuranceLevel::CREATOR_ROOT,
            None,
            generation,
            now,
            claims,
        )
    }

    /// Mint initial bootstrap VerifiedPrincipal for system initialization ceremonies.
    pub fn mint_initialization_principal(
        &self,
        generation: crate::primitives::AuthorityGeneration,
    ) -> Result<crate::primitives::VerifiedPrincipal, String> {
        let (state, reason) = self.lifecycle.verify_integrity();
        if state != AuthorityState::CreatorSetupRequired || !self.lifecycle.is_in_initialization() {
            return Err(format!(
                "AuthorityMintDenied: mint_initialization_principal is strictly prohibited outside cold-start initialization ceremony (state: {:?}, reason: {})",
                state, reason
            ));
        }
        let pid = crate::primitives::PrincipalId::new("creator_root_policy_admin")?;
        let s_id = "bootstrap_policy_admin_session";
        let proof = crate::primitives::compute_session_proof_hex(&pid.as_str(), s_id);
        let mut claims = HashMap::new();
        claims.insert(
            "session_id".to_string(),
            serde_json::Value::String(s_id.to_string()),
        );
        claims.insert(
            "session_proof".to_string(),
            serde_json::Value::String(proof),
        );
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        crate::primitives::VerifiedPrincipal::construct_verified(
            pid,
            crate::primitives::PrincipalRole::creator(),
            crate::primitives::AssuranceLevel::CREATOR_ROOT,
            None,
            generation,
            now,
            claims,
        )
    }

    fn issue_session(&self, actor_id: &str, role: &str, client_ip: &str) -> String {
        let mut token_bytes = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut token_bytes);
        let token = hex::encode(token_bytes);
        if let Ok(mut sessions) = self.sessions.lock() {
            sessions.insert(
                token.clone(),
                CreatorSessionRecord {
                    actor_id: actor_id.to_string(),
                    role: role.to_string(),
                    created_at: Instant::now(),
                    client_ip: client_ip.to_string(),
                },
            );
        } else {
            return String::new();
        }
        token
    }

    pub fn check_rate_limit(&self, client_ip: &str) -> (bool, u64) {
        match self.rate_limiter.lock() {
            Ok(mut rl) => rl.check(client_ip),
            Err(_) => (false, 300),
        }
    }

    pub fn record_rate_limit_failure(&self, client_ip: &str) {
        if let Ok(mut rl) = self.rate_limiter.lock() {
            rl.record_failure(client_ip);
        }
    }

    pub fn record_rate_limit_success(&self, client_ip: &str) {
        if let Ok(mut rl) = self.rate_limiter.lock() {
            rl.record_success(client_ip);
        }
    }

    pub fn authenticate_google_token(
        &self,
        id_token: &str,
        nonce: Option<&str>,
        client_ip: &str,
    ) -> Value {
        let (state, reason) = self.lifecycle.verify_integrity();
        match state {
            AuthorityState::AuthorityLocked => {
                return json!({
                    "status": "AUTHORITY_LOCKED",
                    "error": format!("Creator authority is locked due to integrity mismatch: {}", reason)
                });
            }
            AuthorityState::CreatorSetupRequired => {
                return json!({
                    "status": "CREATOR_SETUP_REQUIRED",
                    "error": "Creator setup has not been performed yet. Real interactive setup is required."
                });
            }
            AuthorityState::Active => {}
        }

        let (allowed, rem) = self.check_rate_limit(client_ip);
        if !allowed {
            return json!({
                "status": "FAILED",
                "error": "Creator authentication failed. Rate limit exceeded.",
                "lockout_remaining": rem
            });
        }

        if id_token.trim().is_empty() {
            self.record_rate_limit_failure(client_ip);
            return json!({ "status": "FAILED", "error": "id_token is required" });
        }

        let req_nonce = match nonce {
            Some(n) if !n.trim().is_empty() => n.trim(),
            _ => {
                self.record_rate_limit_failure(client_ip);
                return json!({
                    "status": "FAILED",
                    "error": "Single-use challenge nonce is mandatory for Google OAuth login (Anti-Replay defense)"
                });
            }
        };

        let nonce_valid = {
            let mut ch = match self.google_challenges.lock() {
                Ok(c) => c,
                Err(_) => {
                    return json!({ "status": "FAILED", "error": "Internal security state error" })
                }
            };
            ch.retain(|_, ts| ts.elapsed() < Duration::from_secs(300));
            ch.remove(req_nonce).is_some()
        };
        if !nonce_valid {
            self.record_rate_limit_failure(client_ip);
            return json!({
                "status": "FAILED",
                "error": "Invalid, expired, or replayed Google OAuth challenge nonce"
            });
        }

        let verified_claims = match self.google_auth.verify_id_token(
            id_token,
            Some(req_nonce),
            true,
        ) {
            Ok(claims) => claims,
            Err(e) => {
                self.record_rate_limit_failure(client_ip);
                return json!({
                    "status": "FAILED",
                    "error": format!("Google JWT RS256 cryptographic verification failed: {}", e)
                });
            }
        };

        let token_email = verified_claims
            .get("email")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_lowercase();
        let token_sub = verified_claims
            .get("sub")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();

        let sealed_reg = Path::new(&self.repo_root)
            .join("storage")
            .join("vault")
            .join("access")
            .join("operators_registry.json");
        let sealed_rec = Path::new(&self.repo_root)
            .join("storage")
            .join("vault")
            .join("access")
            .join("operator_record.json");
        let legacy_reg = Path::new(&self.repo_root)
            .join("TARA")
            .join("ACCESS")
            .join("operator")
            .join("operators_registry.json");
        let legacy_rec = Path::new(&self.repo_root)
            .join("TARA")
            .join("ACCESS")
            .join("operator")
            .join("operator_record.json");

        let registry_path = if sealed_reg.exists() {
            sealed_reg
        } else {
            legacy_reg
        };
        let record_path = if sealed_rec.exists() {
            sealed_rec
        } else {
            legacy_rec
        };

        let mut matched_creator_id: Option<String> = None;
        let mut matched_role: String = "CREATOR".to_string();

        if let Ok(raw) = fs::read_to_string(&registry_path) {
            if let Ok(reg) = serde_json::from_str::<Value>(&raw) {
                if let Some(obj) = reg
                    .get("operators")
                    .and_then(|v| v.as_object())
                    .or_else(|| reg.as_object())
                {
                    for (cid, cval) in obj {
                        let auth_email = cval
                            .get("authorized_google_email")
                            .or_else(|| cval.get("google_email"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .trim()
                            .to_lowercase();

                        if !auth_email.is_empty() && auth_email == token_email {
                            let bound_sub = cval
                                .get("google_subject_id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            if bound_sub.is_empty() || bound_sub == token_sub {
                                matched_creator_id = Some(cid.clone());
                                matched_role = cval
                                    .get("role")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("CREATOR")
                                    .to_string();
                                break;
                            }
                        }
                    }
                }
            }
        }

        if matched_creator_id.is_none() {
            if let Ok(raw) = fs::read_to_string(&record_path) {
                if let Ok(rec) = serde_json::from_str::<Value>(&raw) {
                    let auth_email = rec
                        .get("authorized_google_email")
                        .or_else(|| rec.get("google_email"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim()
                        .to_lowercase();

                    if !auth_email.is_empty() && auth_email == token_email {
                        let cid = rec
                            .get("creator_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or(CANONICAL_CREATOR_ID);
                        matched_creator_id = Some(cid.to_string());
                        matched_role = "ROOT_CREATOR".to_string();
                    }
                }
            }
        }

        if let Some(creator_id) = matched_creator_id {
            self.record_rate_limit_success(client_ip);
            let session_token = self.issue_session(&creator_id, &matched_role, client_ip);
            json!({
                "status": "SUCCESS",
                "session_token": session_token,
                "creator_id": creator_id,
                "role": matched_role,
                "auth_method": "GOOGLE_OAUTH_VERIFIED"
            })
        } else {
            self.record_rate_limit_failure(client_ip);
            json!({
                "status": "FAILED",
                "error": "Creator authentication failed. Verified Google account is not enrolled as authorized creator."
            })
        }
    }

    pub fn authenticate_creator_key(
        &self,
        proof_signature_hex: Option<&str>,
        challenge_nonce: Option<&str>,
        claimed_creator_id: &str,
        passphrase: Option<&str>,
        client_ip: &str,
    ) -> Value {
        let (state, reason) = self.lifecycle.verify_integrity();
        match state {
            AuthorityState::AuthorityLocked => {
                return json!({
                    "status": "AUTHORITY_LOCKED",
                    "error": format!("Creator authority is locked due to integrity mismatch: {}", reason)
                });
            }
            AuthorityState::CreatorSetupRequired => {
                return json!({
                    "status": "CREATOR_SETUP_REQUIRED",
                    "error": "Creator setup has not been performed yet. Real interactive setup is required."
                });
            }
            AuthorityState::Active => {}
        }

        let (allowed, rem) = self.check_rate_limit(client_ip);
        if !allowed {
            return json!({
                "status": "FAILED",
                "error": "Creator authentication failed. Rate limit exceeded.",
                "lockout_remaining": rem
            });
        }

        if claimed_creator_id != CANONICAL_CREATOR_ID {
            self.record_rate_limit_failure(client_ip);
            return json!({ "status": "ERROR", "error": "Creator ID not recognised." });
        }

        let root_pubkey_hex = self.lifecycle.get_root_public_key();
        let mut auth_ok = false;

        // Path 1: Cryptographic Ed25519 signature proof over challenge nonce
        if let (Some(sig), Some(nonce), Some(ref root_pub)) =
            (proof_signature_hex, challenge_nonce, &root_pubkey_hex)
        {
            if !sig.is_empty() && !nonce.is_empty() {
                let mut ch = match self.key_challenges.lock() {
                    Ok(guard) => guard,
                    Err(_) => {
                        return json!({ "status": "ERROR", "error": "Security mutex poisoned, failing closed." });
                    }
                };
                ch.retain(|_, ts| ts.elapsed() < Duration::from_secs(300));
                let valid_nonce = ch.remove(nonce).is_some();
                drop(ch);

                if !valid_nonce {
                    self.record_rate_limit_failure(client_ip);
                    return json!({ "status": "ERROR", "error": "Invalid or expired challenge nonce." });
                }

                auth_ok =
                    crate::devices::DeviceCrypto::verify_signature(root_pub, nonce.as_bytes(), sig);
            }
        }

        // Path 2: Scrypt + AES-256-GCM + DPAPI keystore unlock
        if !auth_ok {
            if let Some(pp) = passphrase {
                let pp_clean = pp.trim();
                if !pp_clean.is_empty() {
                    let key_load_res = self
                        .key_storage
                        .load_private_key("operator_key", Some(pp_clean))
                        .or_else(|_| {
                            self.key_storage
                                .load_private_key("creator_root_key", Some(pp_clean))
                        });
                    match key_load_res {
                        Ok(mut decrypted_priv) if decrypted_priv.len() == 32 => {
                            use zeroize::Zeroize;
                            let mut priv_arr = [0u8; 32];
                            priv_arr.copy_from_slice(&decrypted_priv);
                            decrypted_priv.zeroize();
                            let signing_key = SigningKey::from_bytes(&priv_arr);
                            priv_arr.zeroize();
                            let derived_pub = hex::encode(signing_key.verifying_key().as_bytes());
                            if let Some(ref expected_pub) = root_pubkey_hex {
                                if constant_time_eq::constant_time_eq(
                                    derived_pub.as_bytes(),
                                    expected_pub.as_bytes(),
                                ) {
                                    auth_ok = true;
                                }
                            }
                        }
                        _ => {
                            auth_ok = false;
                        }
                    }
                }
            }
        }

        if auth_ok {
            self.record_rate_limit_success(client_ip);
            let session_token = self.issue_session(CANONICAL_CREATOR_ID, "ROOT_CREATOR", client_ip);
            json!({
                "status": "SUCCESS",
                "session_token": session_token,
                "creator_id": CANONICAL_CREATOR_ID,
                "role": "ROOT_CREATOR",
                "auth_method": "creator_key"
            })
        } else {
            self.record_rate_limit_failure(client_ip);
            json!({ "status": "ERROR", "error": "Creator key authentication failed." })
        }
    }

    pub fn authenticate_recovery(
        &self,
        recovery_code: &str,
        claimed_creator_id: &str,
        client_ip: &str,
    ) -> Value {
        // Step 1: Check persistent, durable lockout state (persists across restarts)
        if self.recovery_manager.is_locked_out() {
            let rem = self.recovery_manager.lockout_remaining_secs();
            return json!({
                "status": "LOCKED_OUT",
                "error": "Recovery service is persistently locked out due to prior failed attempts.",
                "lockout_remaining": rem
            });
        }

        // Step 2: Check volatile per-IP rate limit
        let (allowed, rem) = self.check_rate_limit(client_ip);
        if !allowed {
            return json!({
                "status": "FAILED",
                "error": "Recovery rate limit exceeded.",
                "lockout_remaining": rem
            });
        }

        if claimed_creator_id != CANONICAL_CREATOR_ID {
            let _ = self.recovery_manager.record_failure();
            self.record_rate_limit_failure(client_ip);
            return json!({ "status": "ERROR", "error": "Creator ID not recognised." });
        }

        let (state, reason) = self.lifecycle.verify_integrity();
        match state {
            AuthorityState::AuthorityLocked => {
                let _ = self.recovery_manager.record_failure();
                self.record_rate_limit_failure(client_ip);
                return json!({
                    "status": "AUTHORITY_LOCKED",
                    "error": format!("Authority is locked due to integrity mismatch: {}", reason)
                });
            }
            AuthorityState::CreatorSetupRequired => {
                let _ = self.recovery_manager.record_failure();
                self.record_rate_limit_failure(client_ip);
                return json!({
                    "status": "CREATOR_SETUP_REQUIRED",
                    "error": "Creator setup has not been performed yet. Real interactive setup is required."
                });
            }
            AuthorityState::Active => {}
        }

        let clean_code = recovery_code.trim();
        if clean_code.is_empty() {
            let _ = self.recovery_manager.record_failure();
            self.record_rate_limit_failure(client_ip);
            return json!({ "status": "ERROR", "error": "recovery_code is required" });
        }

        let vault_restore = PathBuf::from(&self.repo_root)
            .join("storage")
            .join("vault")
            .join("access")
            .join("restore_config.json");
        let app_recovery = if cfg!(target_os = "windows") {
            std::env::var("APPDATA")
                .map(|a| {
                    format!(
                        "{}/TARA/recovery/recovery_config.json",
                        a.replace('\\', "/")
                    )
                })
                .ok()
        } else {
            std::env::var("HOME")
                .map(|h| format!("{}/.tara/recovery/recovery_config.json", h))
                .ok()
        };
        let recovery_path = if vault_restore.exists() {
            vault_restore.to_string_lossy().to_string()
        } else if let Some(ref p) = app_recovery.filter(|p| std::path::Path::new(p).exists()) {
            p.clone()
        } else {
            format!("{}/TARA/ACCESS/restore/restore_config.json", self.repo_root)
        };

        let seal_path = PathBuf::from(&self.repo_root)
            .join("storage")
            .join("vault")
            .join("access")
            .join("access_seal.json");
        if let Ok(raw_seal) = fs::read_to_string(&seal_path) {
            if let Ok(seal_val) = serde_json::from_str::<Value>(&raw_seal) {
                if let Some(expected_rec_hash) =
                    seal_val.get("recovery_hash").and_then(|v| v.as_str())
                {
                    let computed_rec_hash = compute_recovery_verifier_hash(&recovery_path);
                    if computed_rec_hash.as_deref() != Some(expected_rec_hash) {
                        let _ = self.recovery_manager.record_failure();
                        self.record_rate_limit_failure(client_ip);
                        return json!({
                            "status": "AUTHORITY_LOCKED",
                            "error": "Recovery configuration hash mismatch against sealed authority."
                        });
                    }
                }
            }
        }

        if let Ok(raw) = fs::read_to_string(&recovery_path) {
            if let Ok(rec) = serde_json::from_str::<Value>(&raw) {
                let expected_hash = rec
                    .get("recovery_code_hash")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let salt_hex = rec
                    .get("recovery_salt")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let iterations = rec
                    .get("iterations")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(100_000) as u32;

                if !(50_000..=500_000).contains(&iterations) {
                    let _ = self.recovery_manager.record_failure();
                    self.record_rate_limit_failure(client_ip);
                    return json!({ "status": "ERROR", "error": "Invalid PBKDF2 iterations in recovery config." });
                }

                if let Ok(salt_bytes) = hex::decode(salt_hex) {
                    let mut is_valid = false;

                    let mut derived_pbkdf2 = [0u8; 32];
                    pbkdf2_hmac_sha256(
                        clean_code.as_bytes(),
                        &salt_bytes,
                        iterations,
                        &mut derived_pbkdf2,
                    );
                    let computed_pbkdf2 = hex::encode(derived_pbkdf2);

                    if constant_time_eq::constant_time_eq(
                        computed_pbkdf2.as_bytes(),
                        expected_hash.as_bytes(),
                    ) {
                        is_valid = true;
                    } else {
                        let mut derived_scrypt = [0u8; 32];
                        if crate::cryptography::CryptoUtils::derive_scrypt_key(
                            clean_code.as_bytes(),
                            &salt_bytes,
                            &mut derived_scrypt,
                        )
                        .is_ok()
                        {
                            let computed_scrypt = hex::encode(derived_scrypt);
                            if constant_time_eq::constant_time_eq(
                                computed_scrypt.as_bytes(),
                                expected_hash.as_bytes(),
                            ) {
                                is_valid = true;
                            }
                        }
                    }

                    if is_valid {
                        let _ = self.recovery_manager.record_success();
                        self.record_rate_limit_success(client_ip);
                        let session_token =
                            self.issue_session(CANONICAL_CREATOR_ID, "ROOT_CREATOR", client_ip);
                        return json!({
                            "status": "SUCCESS",
                            "session_token": session_token,
                            "creator_id": CANONICAL_CREATOR_ID,
                            "role": "ROOT_CREATOR",
                            "auth_method": "recovery"
                        });
                    }
                }
            }
        }

        let _ = self.recovery_manager.record_failure();
        self.record_rate_limit_failure(client_ip);
        json!({ "status": "ERROR", "error": "Invalid recovery code." })
    }

    pub fn create_qr_challenge(&self) -> Value {
        let mut ch_bytes = [0u8; 16];
        let mut nonce_bytes = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut ch_bytes);
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut nonce_bytes);
        let challenge_id = hex::encode(ch_bytes);
        let nonce = hex::encode(nonce_bytes);

        self.qr_challenges
            .lock()
            .unwrap()
            .insert(challenge_id.clone(), (nonce.clone(), Instant::now()));
        json!({ "challenge_id": challenge_id, "nonce": nonce, "expires_in": 300 })
    }

    pub fn get_qr_status(&self, challenge_id: &str) -> Value {
        let challenges = self.qr_challenges.lock().unwrap();
        if let Some((_, created_at)) = challenges.get(challenge_id) {
            if created_at.elapsed() < Duration::from_secs(300) {
                return json!({ "status": "PENDING", "challenge_id": challenge_id });
            }
        }
        json!({ "status": "EXPIRED", "challenge_id": challenge_id })
    }

    pub fn verify_qr_approval(
        &self,
        challenge_id: &str,
        device_id: &str,
        device_signature_hex: &str,
        identity_mgr: &crate::devices::IdentityManager,
        client_ip: &str,
    ) -> Value {
        let mut challenges = self.qr_challenges.lock().unwrap();
        let (nonce, created_at) = match challenges.remove(challenge_id) {
            Some(c) => c,
            None => return json!({ "status": "ERROR", "error": "Challenge expired or not found." }),
        };
        drop(challenges);

        if created_at.elapsed() >= Duration::from_secs(300) {
            return json!({ "status": "ERROR", "error": "Challenge expired." });
        }

        if device_id.trim().is_empty() || device_signature_hex.trim().is_empty() {
            return json!({
                "status": "ERROR",
                "error": "Device ID and cryptographic signature are required."
            });
        }

        let device = match identity_mgr.get_device(device_id) {
            Some(d) => d,
            None => {
                return json!({
                    "status": "ERROR",
                    "error": "Device not found in authorized device registry."
                });
            }
        };

        if device.creator_id != CANONICAL_CREATOR_ID {
            return json!({
                "status": "ERROR",
                "error": "Device is not registered to canonical creator."
            });
        }

        if device.status != "AUTHORIZED" {
            return json!({
                "status": "ERROR",
                "error": "Device is revoked or not authorized."
            });
        }

        let payload = format!("{}:{}", challenge_id, nonce);
        let sig_valid = crate::devices::DeviceCrypto::verify_signature(
            &device.device_public_key,
            payload.as_bytes(),
            device_signature_hex,
        ) || crate::devices::DeviceCrypto::verify_signature(
            &device.device_public_key,
            nonce.as_bytes(),
            device_signature_hex,
        );

        if !sig_valid {
            return json!({
                "status": "ERROR",
                "error": "Invalid cryptographic device signature."
            });
        }

        let session_token = self.issue_session(CANONICAL_CREATOR_ID, "ROOT_CREATOR", client_ip);
        json!({
            "status": "SUCCESS",
            "session_token": session_token,
            "creator_id": CANONICAL_CREATOR_ID,
            "role": "ROOT_CREATOR",
            "auth_method": "qr",
            "device_id": device_id
        })
    }

    pub fn check_conversational_trigger(&self, text: &str) -> Value {
        let text_lc = text.to_lowercase();
        let triggers = [
            "i am the creator",
            "creator mode",
            "enable creator",
            "authenticate creator",
            "verify creator",
            "switch to creator",
            "who is the creator",
            "creator access",
        ];

        let triggered = triggers.iter().any(|t| text_lc.contains(t));
        json!({
            "triggered": triggered,
            "matched_trigger": triggered.then_some("conversational_trigger"),
            "requires_auth": triggered
        })
    }

    pub fn handle_method_selection(
        &self,
        method: &str,
        _creator_id: &str,
        _client_ip: &str,
    ) -> Value {
        let instructions = match method {
            "google" => "Please provide your Google ID token via POST /api/v1/auth/google",
            "creator_key" => "Please provide proof_signature and challenge_nonce via POST /api/v1/auth/creator_key",
            "qr" => "Please scan the QR code returned by POST /api/v1/auth/qr_challenge",
            "recovery" => "Please provide your recovery_code via POST /api/v1/auth/recovery",
            _ => "Unknown method. Supported: google, creator_key, qr, recovery",
        };
        json!({ "status": "SUCCESS", "method": method, "instructions": instructions })
    }

    pub fn logout(&self, token: &str) -> Value {
        let removed = match self.sessions.lock() {
            Ok(mut s) => s.remove(token).is_some(),
            Err(_) => false,
        };
        json!({ "status": "SUCCESS", "logged_out": removed })
    }

    pub fn enroll_creator_google_identity(
        &self,
        creator_id: &str,
        google_email: &str,
        google_subject_id: Option<&str>,
        session_token: &str,
    ) -> Value {
        let sess = match self.verify_session(session_token) {
            Some(s) => s,
            None => {
                return json!({ "status": "ERROR", "error": "Invalid or expired session token." });
            }
        };

        if sess.get("role").map(|r| r.as_str()) != Some("ROOT_CREATOR") {
            return json!({ "status": "ERROR", "error": "ROOT_CREATOR authority required to enroll Google identities." });
        }

        let reg_path = format!(
            "{}/TARA/ACCESS/operator/operators_registry.json",
            self.repo_root
        );
        let mut reg_val = fs::read_to_string(&reg_path)
            .ok()
            .and_then(|r| serde_json::from_str::<Value>(&r).ok())
            .unwrap_or(json!({}));

        if let Some(obj) = reg_val.as_object_mut() {
            let entry = obj.entry(creator_id.to_string()).or_insert_with(|| {
                json!({
                    "creator_id": creator_id,
                    "role": "CREATOR",
                    "status": "active"
                })
            });
            if let Some(entry_obj) = entry.as_object_mut() {
                entry_obj.insert(
                    "authorized_google_email".to_string(),
                    json!(google_email.to_lowercase()),
                );
                if let Some(sub) = google_subject_id {
                    entry_obj.insert("google_subject_id".to_string(), json!(sub));
                }
            }
        }

        let _ = fs::create_dir_all(format!("{}/TARA/ACCESS/operator", self.repo_root));
        let _ = fs::write(
            &reg_path,
            serde_json::to_string_pretty(&reg_val).unwrap_or_default(),
        );

        json!({ "status": "SUCCESS", "google_email": google_email, "creator_id": creator_id })
    }

    pub fn setup_creator(
        &self,
        setup_req: &CreatorCeremonyRequest<'_>,
        identity_mgr: &crate::devices::IdentityManager,
        centralized_device_registry: Option<&crate::zero_trust::DeviceEnrollmentRegistry>,
    ) -> Value {
        let (state, _) = self.lifecycle.verify_integrity();
        if state != AuthorityState::CreatorSetupRequired {
            return json!({
                "status": "ERROR",
                "error": "Creator authority is already initialized. Authentication or authorized recovery is required for changes."
            });
        }

        if !setup_req.confirm_identity {
            return json!({ "status": "ERROR", "error": "Identity confirmation required." });
        }
        if setup_req.google_id_token.trim().is_empty() {
            return json!({ "status": "ERROR", "error": "google_id_token is required." });
        }

        // Cryptographically verify Google ID Token with RS256 JWKS and enforce nonce if provided (Anti-Replay)
        let require_nonce = setup_req.nonce.is_some();
        let verified_claims = match self.google_auth.verify_id_token(
            setup_req.google_id_token,
            setup_req.nonce,
            require_nonce,
        ) {
            Ok(claims) => claims,
            Err(e) => {
                return json!({
                    "status": "ERROR",
                    "error": format!("Google ID token cryptographic verification failed: {}", e)
                });
            }
        };

        let google_email = verified_claims
            .get("email")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_lowercase();
        if google_email.is_empty() {
            return json!({ "status": "ERROR", "error": "Google ID token missing verified email claim." });
        }
        let google_sub = verified_claims
            .get("sub")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();

        if let Err(e) = self.lifecycle.begin_initialization() {
            return json!({ "status": "ERROR", "error": e });
        }

        let now_epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let timestamp_str = format!("{now_epoch}");

        let device_val = if let Some(pubkey) = setup_req.device_public_key {
            match identity_mgr.register_device(
                pubkey,
                setup_req.device_name,
                "AUTHORIZED",
                &timestamp_str,
            ) {
                Ok(rec) => {
                    // Synchronize newly approved device immediately into centralized zero-trust registry
                    if let Some(registry) = centralized_device_registry {
                        if let Ok(dev_id) = crate::primitives::DeviceId::new(&rec.device_id) {
                            if let Ok(pub_bytes) = hex::decode(&rec.device_public_key) {
                                let zero_rec = crate::zero_trust::DeviceEnrollmentRecord {
                                    device_id: dev_id,
                                    enrolled_public_key: pub_bytes,
                                    bound_principal_id: crate::primitives::PrincipalId::new(
                                        "creator_root_operator",
                                    )
                                    .unwrap(),
                                    is_revoked: false,
                                };
                                let _ = registry.enroll_device(zero_rec);
                            }
                        }
                    }
                    Some(serde_json::to_value(&rec).unwrap_or(json!({})))
                }
                Err(e) => {
                    return json!({
                        "status": "ERROR",
                        "error": format!("Device registration failed: {}", e)
                    });
                }
            }
        } else {
            None
        };

        let mut rng = rand::thread_rng();
        let signing_key = SigningKey::generate(&mut rng);
        let verifying_key = signing_key.verifying_key();
        let pub_bytes = verifying_key.to_bytes();
        let pub_hex = hex::encode(pub_bytes);
        let priv_bytes = signing_key.to_bytes();

        let master_passphrase =
            if let Some(p) = setup_req.passphrase.filter(|p| !p.trim().is_empty()) {
                p.trim().to_string()
            } else {
                let mut pass_bytes = [0u8; 32];
                rand::RngCore::fill_bytes(&mut rng, &mut pass_bytes);
                hex::encode(pass_bytes)
            };

        let _ = self
            .key_storage
            .store_key("operator_key", &priv_bytes, &master_passphrase);
        let _ = self
            .key_storage
            .store_key("creator_root_key", &priv_bytes, &master_passphrase);

        let mut r_bytes = [0u8; 16];
        rand::RngCore::fill_bytes(&mut rng, &mut r_bytes);
        let recovery_code = hex::encode(r_bytes);

        let mut salt_bytes = [0u8; 16];
        rand::RngCore::fill_bytes(&mut rng, &mut salt_bytes);
        let mut rec_hash_bytes = [0u8; 32];
        pbkdf2_hmac_sha256(
            recovery_code.as_bytes(),
            &salt_bytes,
            100_000,
            &mut rec_hash_bytes,
        );

        let recovery_json = json!({
            "verifier_version": 2,
            "creator_id": CANONICAL_CREATOR_ID,
            "recovery_code_hash": hex::encode(rec_hash_bytes),
            "recovery_salt": hex::encode(salt_bytes),
            "recovery_email": google_email,
            "iterations": 100_000,
            "created_at": timestamp_str
        });

        let access_dir = PathBuf::from(&self.repo_root)
            .join("storage")
            .join("vault")
            .join("access");
        let _ = fs::create_dir_all(&access_dir);
        let rec_path = access_dir.join("restore_config.json");
        let _ = fs::write(
            &rec_path,
            serde_json::to_string_pretty(&recovery_json).unwrap_or_default(),
        );

        let record_json = json!({
            "creator_id": CANONICAL_CREATOR_ID,
            "display_name": DEFAULT_DISPLAY_NAME,
            "authorized_google_email": google_email,
            "google_subject_id": google_sub,
            "identity_version": 1,
            "key_version": 1,
            "root_public_key": pub_hex,
            "status": "ACTIVE",
            "recovery_enabled": true,
            "created_at": timestamp_str,
            "last_key_rotation": null,
            "revoked_keys": []
        });

        let op_path = access_dir.join("operator_record.json");
        let _ = fs::write(
            &op_path,
            serde_json::to_string_pretty(&record_json).unwrap_or_default(),
        );

        let registry_json = json!({
            "operators": {
                CANONICAL_CREATOR_ID: record_json
            }
        });
        let reg_path = access_dir.join("operators_registry.json");
        let _ = fs::write(
            &reg_path,
            serde_json::to_string_pretty(&registry_json).unwrap_or_default(),
        );

        let tara_access_dir = PathBuf::from(&self.repo_root).join("TARA").join("ACCESS");
        let _ = fs::create_dir_all(tara_access_dir.join("restore"));
        let _ = fs::create_dir_all(tara_access_dir.join("operator"));
        let _ = fs::write(
            tara_access_dir.join("restore").join("restore_config.json"),
            serde_json::to_string_pretty(&recovery_json).unwrap_or_default(),
        );
        let _ = fs::write(
            tara_access_dir
                .join("operator")
                .join("operator_record.json"),
            serde_json::to_string_pretty(&record_json).unwrap_or_default(),
        );
        let _ = fs::write(
            tara_access_dir
                .join("operator")
                .join("operators_registry.json"),
            serde_json::to_string_pretty(&registry_json).unwrap_or_default(),
        );

        let _ =
            self.lifecycle
                .seal_initial_authority(&priv_bytes, &pub_bytes, DEFAULT_DISPLAY_NAME);

        let effective_client_ip = setup_req.client_ip.unwrap_or("127.0.0.1");
        let session_token =
            self.issue_session(CANONICAL_CREATOR_ID, "ROOT_CREATOR", effective_client_ip);

        json!({
            "status": "SUCCESS",
            "authority_state": "ACTIVE",
            "creator_id": CANONICAL_CREATOR_ID,
            "display_name": DEFAULT_DISPLAY_NAME,
            "google_email": google_email,
            "recovery_code": recovery_code,
            "master_passphrase": master_passphrase,
            "device": device_val,
            "session": {
                "session_token": session_token,
                "creator_id": CANONICAL_CREATOR_ID,
                "role": "ROOT_CREATOR"
            }
        })
    }
}
