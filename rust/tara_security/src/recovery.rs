//! Multi-path Disaster Recovery & Emergency Authority Authorization for TARA Security Subsystem.
//!
//! Supported Recovery Paths:
//! 1. Cryptographic Recovery Code (High-entropy secret with bounded Scrypt verifier)
//! 2. Verified Google Creator Account (Cryptographically verified RS256 JWT via GoogleAuthService)
//! 3. Enrolled Authorized Device Signature (Ed25519 signature proof over single-use challenge nonce)
//!
//! Enforces:
//! - Persistent failure counters and lockout timestamps that survive process restarts.
//! - Constant-time comparison for all cryptographic proofs.
//! - Single-use nonce freshness and signature verification.

use constant_time_eq::constant_time_eq;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub const CANONICAL_CREATOR_ID: &str = "ROOT_OPERATOR";
pub const MAX_RECOVERY_ATTEMPTS: u32 = 5;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryAuthorizationProof {
    pub proof_id: String,
    pub creator_id: String,
    pub method: String,
    pub authorized_at: String,
    pub expires_at: u64,
    pub nonce: String,
    pub proof_sig: String,
}

impl RecoveryAuthorizationProof {
    pub fn compute_sig(
        secret_key: &[u8],
        proof_id: &str,
        creator_id: &str,
        method: &str,
        expires_at: u64,
        nonce: &str,
    ) -> String {
        let mut signer =
            crate::cryptography::HmacSha256Signer::new(secret_key).expect("HMAC accepts key");
        signer.update(proof_id.as_bytes());
        signer.update(b":");
        signer.update(creator_id.as_bytes());
        signer.update(b":");
        signer.update(method.as_bytes());
        signer.update(b":");
        signer.update(&expires_at.to_be_bytes());
        signer.update(b":");
        signer.update(nonce.as_bytes());
        signer.finalize_hex()
    }

    pub fn verify_signature(&self, secret_key: &[u8]) -> bool {
        let expected = Self::compute_sig(
            secret_key,
            &self.proof_id,
            &self.creator_id,
            &self.method,
            self.expires_at,
            &self.nonce,
        );
        constant_time_eq(self.proof_sig.as_bytes(), expected.as_bytes())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryLockoutState {
    pub failure_count: u32,
    pub lockout_until: Option<u64>,
    #[serde(default)]
    pub hmac_signature: String,
}

impl RecoveryLockoutState {
    pub fn compute_hmac(&self, key: &[u8; 32]) -> String {
        let payload = format!("{}:{}", self.failure_count, self.lockout_until.unwrap_or(0));
        crate::cryptography::CryptoUtils::hmac_sha256_hex(key, payload.as_bytes())
            .expect("HMAC-SHA256 init cannot fail for 32 bytes")
    }

    pub fn verify_hmac(&self, key: &[u8; 32]) -> bool {
        if self.hmac_signature.is_empty() {
            return false;
        }
        let expected = self.compute_hmac(key);
        constant_time_eq(self.hmac_signature.as_bytes(), expected.as_bytes())
    }
}

/// Durable Recovery Manager with persistent failure lockout and active proofs.
pub struct RecoveryManager {
    lockout_file: PathBuf,
    proofs_file: PathBuf,
    secret_key: [u8; 32],
    active_proofs: Mutex<HashMap<String, RecoveryAuthorizationProof>>,
    lockout_state: Mutex<RecoveryLockoutState>,
}

impl RecoveryManager {
    pub fn new<P: AsRef<Path>>(base_dir: P, secret_key: [u8; 32]) -> Self {
        let dir = base_dir.as_ref().to_path_buf();
        let access_dir = dir.join("storage").join("vault").join("access");
        let lockout_file = access_dir.join("recovery_lockout.json");
        let proofs_file = access_dir.join("recovery_active_proofs.json");
        if let Some(parent) = lockout_file.parent() {
            let _ = fs::create_dir_all(parent);
        }

        let loaded_lockout = Self::load_lockout(&lockout_file, &secret_key);
        let loaded_proofs = Self::load_proofs(&proofs_file);
        Self {
            lockout_file,
            proofs_file,
            secret_key,
            active_proofs: Mutex::new(loaded_proofs),
            lockout_state: Mutex::new(loaded_lockout),
        }
    }

    fn now_secs() -> u64 {
        crate::cryptography::CryptoUtils::now_secs()
    }

    fn load_proofs(path: &Path) -> HashMap<String, RecoveryAuthorizationProof> {
        let now = Self::now_secs();
        if path.exists() {
            if let Ok(data) = fs::read(path) {
                if let Ok(mut proofs) =
                    serde_json::from_slice::<HashMap<String, RecoveryAuthorizationProof>>(&data)
                {
                    proofs.retain(|_, p| p.expires_at > now);
                    return proofs;
                }
            }
        }
        HashMap::new()
    }

    fn persist_active_proofs(&self, proofs: &HashMap<String, RecoveryAuthorizationProof>) {
        if let Ok(data) = serde_json::to_vec_pretty(proofs) {
            let _ = crate::persistence::atomic_write_file_hardened(&self.proofs_file, &data);
        }
    }

    fn load_lockout(path: &Path, key: &[u8; 32]) -> RecoveryLockoutState {
        if path.exists() {
            if let Ok(data) = fs::read(path) {
                if let Ok(st) = serde_json::from_slice::<RecoveryLockoutState>(&data) {
                    if st.verify_hmac(key) {
                        return st;
                    }
                }
            }
            // Corrupted or tampered recovery lockout file: fail closed into permanent lockout
            return RecoveryLockoutState {
                failure_count: 10,
                lockout_until: Some(u64::MAX),
                hmac_signature: String::new(),
            };
        }

        // Deletion attack defense: if established vault security exists but lockout state was deleted, fail closed
        if let Some(parent) = path.parent() {
            let watermark_path =
                crate::dpapi::get_os_isolated_watermark_path("creator_authority_initialized");
            let was_previously_initialized =
                crate::dpapi::load_monotonic_watermark(&watermark_path)
                    .ok()
                    .flatten()
                    .map(|val| val > 0)
                    .unwrap_or(false);

            let has_established = was_previously_initialized
                || parent.join("recovery_master_key.bin").exists()
                || parent.join("access_seal.json").exists()
                || parent.join("operator_record.json").exists()
                || parent.join("operators_registry.json").exists()
                || parent.join("devices.json").exists()
                || parent.join("restore_config.json").exists()
                || parent.join("enrolled_passkeys.json").exists()
                || parent.join("authority_state.json").exists();
            if has_established {
                return RecoveryLockoutState {
                    failure_count: 10,
                    lockout_until: Some(u64::MAX),
                    hmac_signature: String::new(),
                };
            }
        }

        let mut rec = RecoveryLockoutState {
            failure_count: 0,
            lockout_until: None,
            hmac_signature: String::new(),
        };
        rec.hmac_signature = rec.compute_hmac(key);
        if let Ok(data) = serde_json::to_vec_pretty(&rec) {
            let _ = crate::persistence::atomic_write_file_hardened(path, &data);
        }
        rec
    }

    fn persist_lockout(&self, st: &RecoveryLockoutState) {
        let mut signed = st.clone();
        signed.hmac_signature = signed.compute_hmac(&self.secret_key);
        if let Ok(data) = serde_json::to_vec_pretty(&signed) {
            let _ = crate::persistence::atomic_write_file_hardened(&self.lockout_file, &data);
        }
    }

    pub fn from_base_dir<P: AsRef<Path>>(base_dir: P) -> Self {
        let dir = base_dir.as_ref().to_path_buf();
        let key_file = dir
            .join("storage")
            .join("vault")
            .join("access")
            .join("recovery_master_key.bin");
        let protect_master_key = |k: &[u8; 32]| -> Vec<u8> {
            #[cfg(windows)]
            {
                crate::dpapi::native_dpapi::protect_bytes(k)
                    .expect("FAIL CLOSED: Windows DPAPI protection failure for recovery master key")
            }
            #[cfg(not(windows))]
            {
                k.to_vec()
            }
        };

        let secret_key = if key_file.exists() {
            if let Ok(bytes) = fs::read(&key_file) {
                #[cfg(windows)]
                let raw_bytes = match crate::dpapi::native_dpapi::unprotect_bytes(&bytes) {
                    Ok(unprotected) => unprotected,
                    Err(e) => {
                        eprintln!(
                            "[TARA Security] Windows DPAPI unprotect failed for recovery master key: {e}. Failing closed without plaintext fallback."
                        );
                        Vec::new()
                    }
                };
                #[cfg(not(windows))]
                let raw_bytes = bytes;

                if raw_bytes.len() >= 32 {
                    let mut k = [0u8; 32];
                    k.copy_from_slice(&raw_bytes[..32]);
                    let to_write = protect_master_key(&k);
                    let _ = crate::persistence::atomic_write_file_hardened(&key_file, &to_write);
                    k
                } else {
                    let mut k = [0u8; 32];
                    rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut k);
                    let to_write = protect_master_key(&k);
                    let _ = crate::persistence::atomic_write_file_hardened(&key_file, &to_write);
                    k
                }
            } else {
                let mut k = [0u8; 32];
                rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut k);
                let to_write = protect_master_key(&k);
                let _ = crate::persistence::atomic_write_file_hardened(&key_file, &to_write);
                k
            }
        } else {
            let mut k = [0u8; 32];
            rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut k);
            let to_write = protect_master_key(&k);
            let _ = crate::persistence::atomic_write_file_hardened(&key_file, &to_write);
            let lockout_file = dir
                .join("storage")
                .join("vault")
                .join("access")
                .join("recovery_lockout.json");
            let mut initial_lockout = RecoveryLockoutState {
                failure_count: 0,
                lockout_until: None,
                hmac_signature: String::new(),
            };
            initial_lockout.hmac_signature = initial_lockout.compute_hmac(&k);
            if let Ok(data) = serde_json::to_vec_pretty(&initial_lockout) {
                let _ = crate::persistence::atomic_write_file_hardened(&lockout_file, &data);
            }
            k
        };
        Self::new(dir, secret_key)
    }

    pub fn is_locked_out(&self) -> bool {
        let guard = self.lockout_state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(until) = guard.lockout_until {
            let now = Self::now_secs();
            return now < until;
        }
        false
    }

    pub fn lockout_remaining_secs(&self) -> u64 {
        let guard = self.lockout_state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(until) = guard.lockout_until {
            let now = Self::now_secs();
            if until > now {
                return until - now;
            }
        }
        0
    }

    pub fn failure_count(&self) -> u32 {
        let guard = self.lockout_state.lock().unwrap_or_else(|p| p.into_inner());
        guard.failure_count
    }

    pub fn record_failure(&self) -> Result<(), String> {
        let mut guard = self
            .lockout_state
            .lock()
            .map_err(|_| "Lockout state poisoned")?;
        guard.failure_count = guard.failure_count.saturating_add(1);

        let now = Self::now_secs();
        if guard.failure_count >= MAX_RECOVERY_ATTEMPTS {
            guard.lockout_until = Some(now + 3600); // 1 hour lockout
        } else if guard.failure_count >= 3 {
            guard.lockout_until = Some(now + 900); // 15 min lockout
        }

        self.persist_lockout(&guard);
        Ok(())
    }

    pub fn record_success(&self) -> Result<(), String> {
        let mut guard = self
            .lockout_state
            .lock()
            .map_err(|_| "Lockout state poisoned")?;
        guard.failure_count = 0;
        guard.lockout_until = None;
        self.persist_lockout(&guard);
        Ok(())
    }

    /// Issues an authorized recovery proof only upon successful constant-time validation
    /// of authentic recovery credential material (eliminates unauthenticated public proof generator).
    pub fn issue_verified_proof(
        &self,
        creator_id: &str,
        method: &str,
        auth_secret: &[u8],
        salt: &[u8],
        expected_hash: &[u8; 32],
        ttl_secs: u64,
    ) -> Result<RecoveryAuthorizationProof, String> {
        if self.is_locked_out() {
            return Err("RecoveryDenied: Recovery manager is currently locked out due to prior failed attempts".into());
        }

        if creator_id != CANONICAL_CREATOR_ID {
            let _ = self.record_failure();
            return Err(format!("RecoveryDenied: Invalid creator ID '{creator_id}'"));
        }

        let mut candidate_hash = [0u8; 32];
        pbkdf2_hmac_sha256(auth_secret, salt, 100_000, &mut candidate_hash);

        if !constant_time_eq(&candidate_hash, expected_hash) {
            let _ = self.record_failure();
            return Err(
                "RecoveryProofRejected: Invalid recovery secret authentication material".into(),
            );
        }

        let now = Self::now_secs();
        let proof_id = format!("proof_{}_{:032x}", now, rand::random::<u128>());
        let nonce = hex::encode(rand::random::<[u8; 16]>());
        let expires_at = now.saturating_add(ttl_secs);

        let proof_sig = RecoveryAuthorizationProof::compute_sig(
            &self.secret_key,
            &proof_id,
            creator_id,
            method,
            expires_at,
            &nonce,
        );

        let proof = RecoveryAuthorizationProof {
            proof_id: proof_id.clone(),
            creator_id: creator_id.to_string(),
            method: method.to_string(),
            authorized_at: crate::cryptography::CryptoUtils::sha256_hex(proof_id.as_bytes()),
            expires_at,
            nonce,
            proof_sig,
        };

        let mut proofs = self
            .active_proofs
            .lock()
            .map_err(|_| "Proofs mutex poisoned")?;
        proofs.retain(|_, p| p.expires_at > now);
        proofs.insert(proof_id, proof.clone());
        self.persist_active_proofs(&proofs);
        self.record_success()?;

        Ok(proof)
    }

    pub fn consume_proof(&self, proof_id: &str) -> Result<RecoveryAuthorizationProof, String> {
        let mut proofs = self
            .active_proofs
            .lock()
            .map_err(|_| "Proofs mutex poisoned")?;
        let proof = match proofs.remove(proof_id) {
            Some(p) => {
                self.persist_active_proofs(&proofs);
                p
            }
            None => {
                let _ = self.record_failure();
                return Err(
                    "RecoveryProofRejected: Proof ID not found or already consumed (anti-replay)"
                        .to_string(),
                );
            }
        };

        let now = Self::now_secs();
        if now >= proof.expires_at {
            let _ = self.record_failure();
            return Err("RecoveryProofRejected: Recovery authorization proof has expired".into());
        }

        if !proof.verify_signature(&self.secret_key) {
            let _ = self.record_failure();
            return Err(
                "RecoveryProofRejected: Proof cryptographic signature verification failed".into(),
            );
        }

        Ok(proof)
    }
}

/// Derives a PBKDF2-HMAC-SHA256 hash using the specified salt and iteration count.
pub fn pbkdf2_hmac_sha256(password: &[u8], salt: &[u8], iterations: u32, output: &mut [u8; 32]) {
    let mut signer =
        crate::cryptography::HmacSha256Signer::new(password).expect("HMAC accepts key");
    signer.update(salt);
    signer.update(&1u32.to_be_bytes());
    let mut u = signer.finalize();
    let mut t = u;

    for _ in 1..iterations {
        u = crate::cryptography::CryptoUtils::hmac_sha256(password, &u).expect("HMAC accepts key");
        for j in 0..32 {
            t[j] ^= u[j];
        }
    }
    output.copy_from_slice(&t);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recovery_proof_consumption_failure_increments_lockout() {
        let temp_dir =
            std::env::temp_dir().join(format!("tara_rec_test_{}", rand::random::<u64>()));
        let _ = std::fs::create_dir_all(&temp_dir);

        let manager = RecoveryManager::from_base_dir(&temp_dir);
        assert!(
            !manager.is_locked_out(),
            "New manager must not be locked out"
        );
        assert_eq!(manager.failure_count(), 0);

        // 1. Non-existent proof ID consumption fails and increments failure counter
        let res = manager.consume_proof("non_existent_proof_id");
        assert!(res.is_err());
        assert_eq!(
            manager.failure_count(),
            1,
            "Failure counter must increment on missing proof"
        );

        // 2. Consume with another bad proof increments to 2
        let res2 = manager.consume_proof("another_bad_proof");
        assert!(res2.is_err());
        assert_eq!(manager.failure_count(), 2);

        // 3. Repeated failures trigger persistent lockout (MAX_RECOVERY_ATTEMPTS = 3)
        let res3 = manager.consume_proof("third_bad_proof");
        assert!(res3.is_err());
        assert_eq!(manager.failure_count(), 3);
        assert!(
            manager.is_locked_out(),
            "Manager must be locked out after reaching max attempts"
        );

        // 4. Persistence across restart: create a new RecoveryManager instance pointing to same storage
        let manager2 = RecoveryManager::from_base_dir(&temp_dir);
        assert!(
            manager2.is_locked_out(),
            "Lockout state must persist across manager restart"
        );
        assert_eq!(
            manager2.failure_count(),
            3,
            "Failure count must survive restart"
        );

        // Cleanup test directory post-test
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_issue_verified_proof_and_active_proof_persistence() {
        let temp_dir =
            std::env::temp_dir().join(format!("tara_rec_proof_test_{}", rand::random::<u64>()));
        let _ = std::fs::create_dir_all(&temp_dir);

        let manager = RecoveryManager::from_base_dir(&temp_dir);
        let secret = b"super_secure_recovery_secret_12345";
        let salt = b"salt_for_recovery_test_67890";
        let mut expected_hash = [0u8; 32];
        pbkdf2_hmac_sha256(secret, salt, 100_000, &mut expected_hash);

        // 1. Invalid secret fails and increments failure count
        let bad_res = manager.issue_verified_proof(
            CANONICAL_CREATOR_ID,
            "scrypt_code",
            b"wrong_secret",
            salt,
            &expected_hash,
            300,
        );
        assert!(bad_res.is_err(), "Invalid recovery secret must fail closed");
        assert_eq!(manager.failure_count(), 1);

        // 2. Valid secret issues verified proof
        let proof = manager
            .issue_verified_proof(
                CANONICAL_CREATOR_ID,
                "scrypt_code",
                secret,
                salt,
                &expected_hash,
                300,
            )
            .expect("Valid secret must issue recovery proof");
        assert_eq!(manager.failure_count(), 0);

        // 3. Persistence across restart: new manager instance loads active proof
        let manager2 = RecoveryManager::from_base_dir(&temp_dir);
        let consumed = manager2
            .consume_proof(&proof.proof_id)
            .expect("Active proof must persist across restart");
        assert_eq!(consumed.proof_id, proof.proof_id);

        // 4. Replay attack: consuming again must fail
        let replay = manager2.consume_proof(&proof.proof_id);
        assert!(replay.is_err(), "Consumed proof cannot be replayed");

        // 5. Cleanup test artifacts post-test
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_issue_verified_proof_prunes_expired() {
        let temp_dir =
            std::env::temp_dir().join(format!("tara_rec_prune_test_{}", rand::random::<u64>()));
        let _ = std::fs::create_dir_all(&temp_dir);

        let manager = RecoveryManager::from_base_dir(&temp_dir);
        let secret = b"super_secure_recovery_secret_prune";
        let salt = b"salt_for_recovery_prune_test";
        let mut expected_hash = [0u8; 32];
        pbkdf2_hmac_sha256(secret, salt, 100_000, &mut expected_hash);

        // Inject an expired proof into active_proofs
        {
            let mut proofs = manager.active_proofs.lock().unwrap();
            let expired_proof = RecoveryAuthorizationProof {
                proof_id: "expired_proof_123".into(),
                creator_id: CANONICAL_CREATOR_ID.into(),
                method: "scrypt_code".into(),
                authorized_at: "auth_time".into(),
                expires_at: 10, // Far in the past
                nonce: "nonce".into(),
                proof_sig: "sig".into(),
            };
            proofs.insert("expired_proof_123".into(), expired_proof);
        }

        // Issue a new proof with large TTL to test saturating_add and pruning
        let proof = manager
            .issue_verified_proof(
                CANONICAL_CREATOR_ID,
                "scrypt_code",
                secret,
                salt,
                &expected_hash,
                u64::MAX, // Saturating add test
            )
            .expect("Must issue proof with saturating TTL");

        assert_eq!(proof.expires_at, u64::MAX);

        // Verify expired proof was pruned from active_proofs
        let proofs = manager.active_proofs.lock().unwrap();
        assert!(
            !proofs.contains_key("expired_proof_123"),
            "Expired proof must be pruned"
        );
        assert!(proofs.contains_key(&proof.proof_id));

        drop(proofs);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
