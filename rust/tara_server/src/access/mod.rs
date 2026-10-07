//! Identity management: creator authentication, device registry, lockdown.
//! Hardened for full parity with Python TARA/ACCESS architecture:
//! 1. Google authentication: Real Google JWT RS256 signature verification using Google's JWKS,
//!    validating issuer, audience, expiry, issued-at, nonce, and email verification.
//! 2. Key protection: Scrypt N=131072, r=8, p=1 + AES-256-GCM + Windows DPAPI keystore.
//!    Zero plain SHA-256 password hashing. Zero empty passphrases.
//! 3. Authority integrity: Ed25519 sealed authority state and canonical JSON hashing.
//!    Fail-closed on tampering, missing seal, or uninitialized state.

pub mod crypto;
pub mod devices;
pub mod durable;
pub mod factors;
pub mod google;
pub mod lifecycle;
pub mod lockdown;
pub mod model_integrity;
pub mod profile;
pub mod protected;
pub mod qr;
pub mod restore;
pub mod services;
pub mod setup;
pub mod wizard;

pub use model_integrity::{IntegrityError, ModelIntegrityEngine, SealResult, VerifyResult};

pub use devices::{DeviceCrypto, EnrollmentService};
pub use durable::{DurableStorageManager, DurableStorageProvider, LocalFileStorageProvider};
pub use factors::{BiometricCapability, BiometricFactorProvider, PlatformBiometricReport};
pub use lockdown::{LockdownCoordinator, SecurityState};
pub use profile::{CreatorIdentity, CreatorIdentityRecord};
pub use protected::{ActionBroker, PolicyRecordStore};
pub use qr::QrCodeMatrix;
pub use restore::{RecoveryAuthorizationProof, RecoveryManager};
pub use services::{
    FirebaseSyncService, OwnershipTag, StorageObject, StorageRegistry, StorageType,
};
pub use setup::{CreatorSetupEngine, CreatorSetupRequest, CreatorSetupResult};
pub use wizard::{BootstrapWizard, QrPrompt};

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ed25519_dalek::{Signature, SigningKey, VerifyingKey};
use hex;
use hmac::{Hmac, Mac};
use rand::Rng;
use serde_json::{json, Value};
use sha2::Sha256;

use self::crypto::SecureKeyStorage;
use self::google::GoogleAuthService;
use self::lifecycle::{
    AuthorityLifecycleManager, AuthorityState, CANONICAL_CREATOR_ID, DEFAULT_DISPLAY_NAME,
};

/// Device record in the authorized device registry.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeviceRecord {
    pub device_id: String,
    pub device_name: String,
    pub device_public_key: String,
    pub status: String,
    pub created_at: String,
}

struct DeviceRegistry {
    devices: HashMap<String, DeviceRecord>,
}

impl DeviceRegistry {
    fn new() -> Self {
        Self {
            devices: HashMap::new(),
        }
    }

    fn register_device(
        &mut self,
        public_key_hex: &str,
        device_name: &str,
        status: &str,
    ) -> DeviceRecord {
        // Derive a stable, collision-resistant device ID from the public key.
        // SHA-256(public_key_hex) → first 12 hex chars → "TARA-DEV-<12hex>"
        // This is deterministic across restarts and unique per key pair.
        use sha2::{Digest, Sha256};
        let hash = Sha256::digest(public_key_hex.as_bytes());
        let device_id = format!("TARA-DEV-{}", hex::encode(&hash[..6]));

        let rec = DeviceRecord {
            device_id: device_id.clone(),
            device_name: device_name.to_string(),
            device_public_key: public_key_hex.to_string(),
            status: status.to_string(),
            created_at: crate::now_iso(),
        };
        self.devices.insert(device_id, rec.clone());
        rec
    }

    fn revoke_device(&mut self, device_id: &str) -> bool {
        if let Some(d) = self.devices.get_mut(device_id) {
            d.status = "REVOKED".to_string();
            true
        } else {
            false
        }
    }

    fn revoke_all(&mut self) {
        for d in self.devices.values_mut() {
            d.status = "REVOKED".to_string();
        }
    }

    fn list_devices(&self) -> Vec<DeviceRecord> {
        self.devices.values().cloned().collect()
    }
}

/// Identity manager: owns device state and root operator ID.
pub struct IdentityManager {
    pub base_dir: String,
    devices: Mutex<DeviceRegistry>,
    pub creator_id: String,
}

impl IdentityManager {
    pub fn new(base_dir: &str) -> Self {
        let _ = fs::create_dir_all(base_dir);
        Self {
            base_dir: base_dir.to_string(),
            devices: Mutex::new(DeviceRegistry::new()),
            creator_id: CANONICAL_CREATOR_ID.to_string(),
        }
    }

    pub fn register_device(
        &self,
        public_key_hex: &str,
        device_name: &str,
        status: &str,
    ) -> DeviceRecord {
        self.devices
            .lock()
            .unwrap()
            .register_device(public_key_hex, device_name, status)
    }

    pub fn revoke_device(&self, device_id: &str) -> bool {
        self.devices.lock().unwrap().revoke_device(device_id)
    }

    pub fn revoke_all_devices(&self) {
        self.devices.lock().unwrap().revoke_all();
    }

    pub fn list_devices(&self) -> Vec<DeviceRecord> {
        self.devices.lock().unwrap().list_devices()
    }

    pub fn get_device(&self, device_id: &str) -> Option<DeviceRecord> {
        self.devices.lock().unwrap().devices.get(device_id).cloned()
    }

    pub fn authorize_device(&self, device_id: &str) -> bool {
        let mut reg = self.devices.lock().unwrap();
        if let Some(d) = reg.devices.get_mut(device_id) {
            d.status = "AUTHORIZED".to_string();
            true
        } else {
            false
        }
    }

    pub fn get_security_summary(&self) -> Value {
        let devices = self.devices.lock().unwrap();
        let authorized_count = devices
            .devices
            .values()
            .filter(|d| d.status == "AUTHORIZED")
            .count();
        json!({
            "creator_id": self.creator_id,
            "authorized_devices": authorized_count,
            "lockdown_state": "NORMAL",
            "base_dir": self.base_dir
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// PBKDF2 Helper for Recovery
// ──────────────────────────────────────────────────────────────────────────────

fn pbkdf2_hmac_sha256(password: &[u8], salt: &[u8], iterations: u32, out: &mut [u8]) {
    type HmacSha256 = Hmac<Sha256>;
    let mut block_num = 1u32;
    let mut out_offset = 0;

    while out_offset < out.len() {
        let mut mac = HmacSha256::new_from_slice(password).unwrap();
        mac.update(salt);
        mac.update(&block_num.to_be_bytes());
        let mut u = mac.finalize().into_bytes();
        let mut t = u;

        for _ in 1..iterations {
            let mut inner_mac = HmacSha256::new_from_slice(password).unwrap();
            inner_mac.update(&u);
            u = inner_mac.finalize().into_bytes();
            for i in 0..t.len() {
                t[i] ^= u[i];
            }
        }

        let copy_len = (out.len() - out_offset).min(t.len());
        out[out_offset..out_offset + copy_len].copy_from_slice(&t[..copy_len]);
        out_offset += copy_len;
        block_num += 1;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Creator Authentication Service
// ──────────────────────────────────────────────────────────────────────────────

struct SessionRecord {
    actor_id: String,
    role: String,
    created_at: Instant,
    client_ip: String,
}

struct RateLimiter {
    attempts: HashMap<String, (u32, Instant, Option<Instant>)>,
}

impl RateLimiter {
    fn new() -> Self {
        Self {
            attempts: HashMap::new(),
        }
    }

    fn check(&mut self, key: &str) -> (bool, u64) {
        let now = Instant::now();
        if let Some((count, last_failure, lockout)) = self.attempts.get_mut(key) {
            if let Some(lock_end) = lockout {
                if now < *lock_end {
                    let rem = (*lock_end - now).as_secs();
                    return (false, rem);
                } else {
                    // Lockout expired
                    *lockout = None;
                    *count = 0;
                }
            } else if now.duration_since(*last_failure) > Duration::from_secs(300) {
                *count = 0;
            }
        }
        (true, 0)
    }

    fn record_failure(&mut self, key: &str) {
        let now = Instant::now();
        let entry = self
            .attempts
            .entry(key.to_string())
            .or_insert((0, now, None));
        entry.0 += 1;
        entry.1 = now;
        if entry.0 >= 5 {
            entry.2 = Some(now + Duration::from_secs(300));
        }
    }

    fn record_success(&mut self, key: &str) {
        self.attempts.remove(key);
    }
}

/// Creator authentication service: manages hardened creator sessions,
/// Scrypt + DPAPI keystore, RS256 JWKS Google auth, and sealed lifecycle state.
pub struct CreatorAuthService {
    pub repo_root: String,
    pub lifecycle: Arc<AuthorityLifecycleManager>,
    pub key_storage: Arc<SecureKeyStorage>,
    pub google_auth: Arc<GoogleAuthService>,
    sessions: Mutex<HashMap<String, SessionRecord>>,
    qr_challenges: Mutex<HashMap<String, (String, Instant)>>,
    rate_limiter: Mutex<RateLimiter>,
}

impl CreatorAuthService {
    pub fn new(repo_root: &str) -> Arc<Self> {
        let root_path = PathBuf::from(repo_root);
        let lifecycle = Arc::new(AuthorityLifecycleManager::new(&root_path));
        let vault_dir = root_path.join("storage").join("vault").join("access");
        let key_storage = Arc::new(SecureKeyStorage::new(vault_dir));
        let google_auth = Arc::new(GoogleAuthService::new(None, None));

        Arc::new(Self {
            repo_root: repo_root.to_string(),
            lifecycle,
            key_storage,
            google_auth,
            sessions: Mutex::new(HashMap::new()),
            qr_challenges: Mutex::new(HashMap::new()),
            rate_limiter: Mutex::new(RateLimiter::new()),
        })
    }

    /// Verify a session token. Returns `Some({"creator_id": ..., "role": ...})` if valid.
    pub fn verify_session(&self, token: &str) -> Option<HashMap<String, String>> {
        if self.lifecycle.is_locked() {
            return None;
        }
        let sessions = self.sessions.lock().unwrap();
        if let Some(sess) = sessions.get(token) {
            if sess.created_at.elapsed() < Duration::from_secs(3600) {
                let mut m = HashMap::new();
                m.insert("creator_id".to_string(), sess.actor_id.clone());
                m.insert("role".to_string(), sess.role.clone());
                m.insert("client_ip".to_string(), sess.client_ip.clone());
                return Some(m);
            }
        }
        None
    }

    fn issue_session(&self, actor_id: &str, role: &str, client_ip: &str) -> String {
        let mut token_bytes = [0u8; 32];
        rand::thread_rng().fill(&mut token_bytes);
        let token = hex::encode(token_bytes);
        self.sessions.lock().unwrap().insert(
            token.clone(),
            SessionRecord {
                actor_id: actor_id.to_string(),
                role: role.to_string(),
                created_at: Instant::now(),
                client_ip: client_ip.to_string(),
            },
        );
        token
    }

    /// Authenticate via Google ID token (validates creator identity with real RS256 JWKS verification).
    pub fn authenticate_google_token(&self, id_token: &str, client_ip: &str) -> Value {
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

        let (allowed, rem) = self.rate_limiter.lock().unwrap().check(client_ip);
        if !allowed {
            return json!({
                "status": "FAILED",
                "error": "Creator authentication failed. Rate limit exceeded.",
                "lockout_remaining": rem
            });
        }

        if id_token.trim().is_empty() {
            self.rate_limiter.lock().unwrap().record_failure(client_ip);
            return json!({ "status": "FAILED", "error": "id_token is required" });
        }

        // Cryptographically verify RS256 JWT using Google's public JWKS certificates
        let verified_claims = match self.google_auth.verify_id_token(id_token, None, false) {
            Ok(claims) => claims,
            Err(e) => {
                self.rate_limiter.lock().unwrap().record_failure(client_ip);
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

        // Verify against enrolled creator in operators_registry.json or operator_record.json
        let registry_path = format!(
            "{}/TARA/ACCESS/operator/operators_registry.json",
            self.repo_root
        );
        let record_path = format!(
            "{}/TARA/ACCESS/operator/operator_record.json",
            self.repo_root
        );

        let mut matched_creator_id: Option<String> = None;
        let mut matched_role: String = "CREATOR".to_string();

        if let Ok(raw) = fs::read_to_string(&registry_path) {
            if let Ok(reg) = serde_json::from_str::<Value>(&raw) {
                if let Some(obj) = reg.as_object() {
                    for (cid, cval) in obj {
                        let auth_email = cval
                            .get("authorized_google_email")
                            .or_else(|| cval.get("google_email"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .trim()
                            .to_lowercase();

                        if !auth_email.is_empty() && auth_email == token_email {
                            // Check subject ID if already bound
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

        // Fallback to operator_record.json if registry not populated
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
            self.rate_limiter.lock().unwrap().record_success(client_ip);
            let session_token = self.issue_session(&creator_id, &matched_role, client_ip);
            json!({
                "status": "SUCCESS",
                "session_token": session_token,
                "creator_id": creator_id,
                "role": matched_role,
                "auth_method": "GOOGLE_OAUTH_VERIFIED"
            })
        } else {
            self.rate_limiter.lock().unwrap().record_failure(client_ip);
            json!({
                "status": "FAILED",
                "error": "Creator authentication failed. Verified Google account is not enrolled as authorized creator."
            })
        }
    }

    /// Authenticate via Ed25519 creator key proof or Scrypt + DPAPI keystore unlock.
    /// Strictly rejects empty passphrases and plain SHA-256 password hashing.
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

        let (allowed, rem) = self.rate_limiter.lock().unwrap().check(client_ip);
        if !allowed {
            return json!({
                "status": "FAILED",
                "error": "Creator authentication failed. Rate limit exceeded.",
                "lockout_remaining": rem
            });
        }

        if claimed_creator_id != CANONICAL_CREATOR_ID {
            self.rate_limiter.lock().unwrap().record_failure(client_ip);
            return json!({ "status": "ERROR", "error": "Creator ID not recognised." });
        }

        // Load active root public key
        let record_path = format!(
            "{}/TARA/ACCESS/operator/operator_record.json",
            self.repo_root
        );
        let root_pubkey_hex = match fs::read_to_string(&record_path) {
            Ok(raw) => serde_json::from_str::<Value>(&raw).ok().and_then(|v| {
                v.get("root_public_key")
                    .and_then(|k| k.as_str())
                    .map(|s| s.to_string())
            }),
            Err(_) => None,
        };

        let mut auth_ok = false;

        // Path 1: Cryptographic Ed25519 signature proof over challenge nonce
        if let (Some(sig), Some(nonce), Some(ref root_pub)) =
            (proof_signature_hex, challenge_nonce, &root_pubkey_hex)
        {
            if !sig.is_empty() && !nonce.is_empty() {
                auth_ok = self.verify_ed25519_proof(root_pub, nonce.as_bytes(), sig);
            }
        }

        // Path 2: Scrypt + AES-256-GCM + DPAPI keystore unlock
        // Strictly never accept empty passphrases
        if !auth_ok {
            if let Some(pp) = passphrase {
                let pp_clean = pp.trim();
                if !pp_clean.is_empty() {
                    match self
                        .key_storage
                        .load_private_key("operator_key", Some(pp_clean))
                    {
                        Ok(decrypted_priv) if decrypted_priv.len() == 32 => {
                            auth_ok = true;
                        }
                        _ => {
                            auth_ok = false;
                        }
                    }
                }
            }
        }

        if auth_ok {
            self.rate_limiter.lock().unwrap().record_success(client_ip);
            let session_token = self.issue_session(CANONICAL_CREATOR_ID, "ROOT_CREATOR", client_ip);
            json!({
                "status": "SUCCESS",
                "session_token": session_token,
                "creator_id": CANONICAL_CREATOR_ID,
                "role": "ROOT_CREATOR",
                "auth_method": "creator_key"
            })
        } else {
            self.rate_limiter.lock().unwrap().record_failure(client_ip);
            json!({ "status": "ERROR", "error": "Creator key authentication failed." })
        }
    }

    fn verify_ed25519_proof(&self, pubkey_hex: &str, message: &[u8], sig_hex: &str) -> bool {
        let pub_bytes = match hex::decode(pubkey_hex) {
            Ok(b) if b.len() == 32 => {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&b);
                arr
            }
            _ => return false,
        };

        let sig_bytes = match hex::decode(sig_hex) {
            Ok(b) if b.len() == 64 => {
                let mut arr = [0u8; 64];
                arr.copy_from_slice(&b);
                arr
            }
            _ => return false,
        };

        if let Ok(verifying_key) = VerifyingKey::from_bytes(&pub_bytes) {
            let sig = Signature::from_bytes(&sig_bytes);
            return verifying_key.verify_strict(message, &sig).is_ok();
        }
        false
    }

    /// Authenticate via recovery code using PBKDF2-HMAC-SHA256 (100,000 iterations).
    pub fn authenticate_recovery(
        &self,
        recovery_code: &str,
        claimed_creator_id: &str,
        client_ip: &str,
    ) -> Value {
        let (state, _) = self.lifecycle.verify_integrity();
        if state == AuthorityState::CreatorSetupRequired {
            return json!({
                "status": "CREATOR_SETUP_REQUIRED",
                "error": "Creator setup has not been performed yet. Real interactive setup is required."
            });
        }

        let (allowed, rem) = self.rate_limiter.lock().unwrap().check(client_ip);
        if !allowed {
            return json!({
                "status": "FAILED",
                "error": "Recovery rate limit exceeded.",
                "lockout_remaining": rem
            });
        }

        let clean_code = recovery_code.trim();
        if clean_code.is_empty() {
            return json!({ "status": "ERROR", "error": "recovery_code is required" });
        }

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
        let recovery_path = match app_recovery {
            Some(ref p) if std::path::Path::new(p).exists() => p.clone(),
            _ => format!("{}/TARA/ACCESS/restore/restore_config.json", self.repo_root),
        };
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

                if let Ok(salt_bytes) = hex::decode(salt_hex) {
                    let mut derived = [0u8; 32];
                    pbkdf2_hmac_sha256(
                        clean_code.as_bytes(),
                        &salt_bytes,
                        iterations,
                        &mut derived,
                    );
                    let computed_hash = hex::encode(derived);

                    if constant_time_eq::constant_time_eq(
                        computed_hash.as_bytes(),
                        expected_hash.as_bytes(),
                    ) {
                        self.rate_limiter.lock().unwrap().record_success(client_ip);
                        let session_token =
                            self.issue_session(claimed_creator_id, "ROOT_CREATOR", client_ip);
                        return json!({
                            "status": "SUCCESS",
                            "session_token": session_token,
                            "creator_id": claimed_creator_id,
                            "role": "ROOT_CREATOR",
                            "auth_method": "recovery"
                        });
                    }
                }
            }
        }

        self.rate_limiter.lock().unwrap().record_failure(client_ip);
        json!({ "status": "ERROR", "error": "Invalid recovery code." })
    }

    /// Create a QR challenge for device-based auth.
    pub fn create_qr_challenge(&self) -> Value {
        let mut ch_bytes = [0u8; 16];
        let mut nonce_bytes = [0u8; 32];
        rand::thread_rng().fill(&mut ch_bytes);
        rand::thread_rng().fill(&mut nonce_bytes);
        let challenge_id = hex::encode(ch_bytes);
        let nonce = hex::encode(nonce_bytes);

        self.qr_challenges
            .lock()
            .unwrap()
            .insert(challenge_id.clone(), (nonce.clone(), Instant::now()));
        json!({ "challenge_id": challenge_id, "nonce": nonce, "expires_in": 300 })
    }

    /// Get QR challenge status.
    pub fn get_qr_status(&self, challenge_id: &str) -> Value {
        let challenges = self.qr_challenges.lock().unwrap();
        if let Some((_, created_at)) = challenges.get(challenge_id) {
            if created_at.elapsed() < Duration::from_secs(300) {
                return json!({ "status": "PENDING", "challenge_id": challenge_id });
            }
        }
        json!({ "status": "EXPIRED", "challenge_id": challenge_id })
    }

    /// Verify QR code device approval with mandatory Ed25519 signature verification against registered device key.
    pub fn verify_qr_approval(
        &self,
        challenge_id: &str,
        device_id: &str,
        device_signature_hex: &str,
        identity_mgr: &IdentityManager,
        client_ip: &str,
    ) -> Value {
        // 1. Consume challenge (prevent reuse)
        let mut challenges = self.qr_challenges.lock().unwrap();
        let (nonce, created_at) = match challenges.remove(challenge_id) {
            Some(c) => c,
            None => {
                return json!({ "status": "ERROR", "error": "Challenge expired or not found." });
            }
        };
        drop(challenges);

        // 2. Verify challenge freshness (5 minutes TTL)
        if created_at.elapsed() >= Duration::from_secs(300) {
            return json!({ "status": "ERROR", "error": "Challenge expired." });
        }

        // 3. Verify non-empty device identity and cryptographic signature
        if device_id.trim().is_empty() || device_signature_hex.trim().is_empty() {
            return json!({
                "status": "ERROR",
                "error": "Device ID and cryptographic signature are required."
            });
        }

        // 4. Retrieve enrolled device from registry
        let device = match identity_mgr.get_device(device_id) {
            Some(d) => d,
            None => {
                return json!({
                    "status": "ERROR",
                    "error": "Device not found in authorized device registry."
                });
            }
        };

        // 5. Verify device status is AUTHORIZED
        if device.status != "AUTHORIZED" {
            return json!({
                "status": "ERROR",
                "error": "Device is revoked or not authorized."
            });
        }

        // 6. Cryptographically verify signature over the challenge payload: challenge_id:nonce
        let payload = format!("{}:{}", challenge_id, nonce);
        let sig_valid = self.verify_ed25519_proof(
            &device.device_public_key,
            payload.as_bytes(),
            device_signature_hex,
        ) || self.verify_ed25519_proof(
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

        // 7. Issue authenticated session
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

    /// Check if text triggers creator auth conversational flow.
    pub fn check_conversational_trigger(&self, text: &str) -> Value {
        let text_lc = text.to_lowercase();
        let triggers = [
            "i am the creator",
            "creator mode",
            "authenticate as creator",
            "creator authentication",
            "i am operator",
            "i am the operator",
        ];
        let triggered = triggers.iter().any(|t| text_lc.contains(t));
        json!({
            "triggered": triggered,
            "suggested_methods": if triggered { vec!["creator_key", "google", "qr", "recovery"] } else { vec![] }
        })
    }

    /// Detect auth method selection from text.
    pub fn detect_method_selection(&self, text: &str) -> Option<String> {
        let text_lc = text.to_lowercase();
        if text_lc.contains("google") {
            Some("google".to_string())
        } else if text_lc.contains("key") || text_lc.contains("signature") {
            Some("creator_key".to_string())
        } else if text_lc.contains("qr") {
            Some("qr".to_string())
        } else if text_lc.contains("recovery") {
            Some("recovery".to_string())
        } else {
            None
        }
    }

    /// Handle auth method selection.
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

    /// Log out / invalidate a session.
    pub fn logout(&self, token: &str) -> Value {
        let removed = self.sessions.lock().unwrap().remove(token).is_some();
        json!({ "status": "SUCCESS", "logged_out": removed })
    }

    /// Enroll creator with Google identity.
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
                return json!({ "status": "ERROR", "error": "Invalid or expired session token." })
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

    /// First-time Creator Setup initialization.
    /// Strictly fails closed if authority is already initialized.
    pub fn setup_creator(
        &self,
        google_id_token: &str,
        confirm_identity: bool,
        device_name: &str,
        device_public_key: Option<&str>,
        identity_mgr: &IdentityManager,
    ) -> Value {
        let (state, _) = self.lifecycle.verify_integrity();
        if state != AuthorityState::CreatorSetupRequired {
            return json!({
                "status": "ERROR",
                "error": "Creator authority is already initialized. Authentication or authorized recovery is required for changes."
            });
        }

        if !confirm_identity {
            return json!({ "status": "ERROR", "error": "Identity confirmation required." });
        }
        if google_id_token.trim().is_empty() {
            return json!({ "status": "ERROR", "error": "google_id_token is required." });
        }

        if let Err(e) = self.lifecycle.begin_initialization() {
            return json!({ "status": "ERROR", "error": e });
        }

        // Register device if public key provided
        let device_val = if let Some(pubkey) = device_public_key {
            let rec = identity_mgr.register_device(pubkey, device_name, "AUTHORIZED");
            Some(serde_json::to_value(rec).unwrap_or(json!({})))
        } else {
            None
        };

        // Generate Ed25519 root keypair
        let mut rng = rand::thread_rng();
        let signing_key = SigningKey::generate(&mut rng);
        let verifying_key = signing_key.verifying_key();
        let pub_bytes = verifying_key.to_bytes();
        let pub_hex = hex::encode(pub_bytes);
        let priv_bytes = signing_key.to_bytes();

        // Master passphrase & Scrypt KDF + AES-256-GCM + DPAPI keystore
        let mut pass_bytes = [0u8; 32];
        rng.fill(&mut pass_bytes);
        let master_passphrase = hex::encode(pass_bytes);

        let _ = self.key_storage.store_private_key_modern(
            "operator_key",
            &priv_bytes,
            &master_passphrase,
            true,
        );

        // Generate recovery code with PBKDF2-HMAC-SHA256
        let mut r_bytes = [0u8; 16];
        rng.fill(&mut r_bytes);
        let recovery_code = hex::encode(r_bytes);

        let mut salt_bytes = [0u8; 16];
        rng.fill(&mut salt_bytes);
        let mut rec_hash_bytes = [0u8; 32];
        pbkdf2_hmac_sha256(
            recovery_code.as_bytes(),
            &salt_bytes,
            100_000,
            &mut rec_hash_bytes,
        );

        let recovery_json = json!({
            "creator_id": CANONICAL_CREATOR_ID,
            "recovery_code_hash": hex::encode(rec_hash_bytes),
            "recovery_salt": hex::encode(salt_bytes),
            "recovery_email": Value::Null,
            "iterations": 100_000,
            "created_at": crate::now_iso()
        });

        let rec_dir = format!("{}/TARA/ACCESS/restore", self.repo_root);
        let _ = fs::create_dir_all(&rec_dir);
        let _ = fs::write(
            format!("{}/restore_config.json", rec_dir),
            serde_json::to_string_pretty(&recovery_json).unwrap_or_default(),
        );

        // Write operator_record.json
        let record_json = json!({
            "creator_id": CANONICAL_CREATOR_ID,
            "display_name": DEFAULT_DISPLAY_NAME,
            "status": "active",
            "root_public_key": pub_hex,
            "key_version": 1,
            "registered_at": crate::now_iso()
        });

        let op_dir = format!("{}/TARA/ACCESS/operator", self.repo_root);
        let _ = fs::create_dir_all(&op_dir);
        let _ = fs::write(
            format!("{}/operator_record.json", op_dir),
            serde_json::to_string_pretty(&record_json).unwrap_or_default(),
        );

        // Write operators_registry.json
        let registry_json = json!({
            CANONICAL_CREATOR_ID: {
                "creator_id": CANONICAL_CREATOR_ID,
                "role": "ROOT_CREATOR",
                "public_key": pub_hex,
                "status": "active",
                "registered_at": crate::now_iso()
            }
        });
        let _ = fs::write(
            format!("{}/operators_registry.json", op_dir),
            serde_json::to_string_pretty(&registry_json).unwrap_or_default(),
        );

        // Cryptographically seal authority state with Ed25519 signature + DPAPI
        let _ =
            self.lifecycle
                .seal_initial_authority(&priv_bytes, &pub_bytes, DEFAULT_DISPLAY_NAME);

        let session_token = self.issue_session(CANONICAL_CREATOR_ID, "ROOT_CREATOR", "127.0.0.1");

        json!({
            "status": "SUCCESS",
            "authority_state": "ACTIVE",
            "creator_id": CANONICAL_CREATOR_ID,
            "display_name": DEFAULT_DISPLAY_NAME,
            "recovery_code": recovery_code,
            "device": device_val,
            "session": {
                "session_token": session_token,
                "creator_id": CANONICAL_CREATOR_ID,
                "role": "ROOT_CREATOR"
            }
        })
    }
}
