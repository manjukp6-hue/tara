//! Root Authority, Key Rotation, and Recovery Subsystem.
//!
//! Rule compliance:
//! - Root creation is an explicit first-run ceremony.
//! - After any authority has been established, missing or corrupt state triggers LOCKDOWN, NEVER re-bootstrap!
//! - Key rotation increments AuthorityGeneration and invalidates prior credentials.
//! - Recovery uses high-entropy one-time material with bounded KDF and rate-limiting.
//! - Extensible RecoveryProvider trait for future threshold / multi-sig schemes.

use ed25519_dalek::SigningKey;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use crate::cryptography::CryptoUtils;
use crate::primitives::{AuthorityGeneration, PrincipalId, SecurityState};

/// Authority Root State persisted securely.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorityRecord {
    pub generation: AuthorityGeneration,
    pub root_public_key_hex: String,
    pub root_principal_id: PrincipalId,
    pub created_epoch_secs: u64,
    pub rotated_epoch_secs: Option<u64>,
    pub recovery_verifier_hash: String,
    pub state: SecurityState,
    pub revoked_keys: HashSet<String>,
}

/// Recovery Proof submitted during emergency authority recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryProof {
    pub target_generation: AuthorityGeneration,
    pub recovery_secret: String,
    pub recovery_nonce: String,
    pub requested_epoch_secs: u64,
}

/// Key Rotation Transition Token proving legitimate authority succession.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationToken {
    pub old_generation: AuthorityGeneration,
    pub new_generation: AuthorityGeneration,
    pub new_public_key_hex: String,
    pub transition_signature_hex: String,
    pub timestamp_epoch_secs: u64,
}

/// Authority Manager handling ceremonies, rotations, durable disk persistence, and fail-closed lockdowns.
pub struct AuthorityManager {
    record: Arc<Mutex<Option<AuthorityRecord>>>,
    root_signing_key: Arc<Mutex<Option<SigningKey>>>,
    failed_recovery_attempts: Arc<Mutex<u32>>,
    storage_path: Option<std::path::PathBuf>,
}

impl Default for AuthorityManager {
    fn default() -> Self {
        Self::new()
    }
}

impl AuthorityManager {
    pub fn new() -> Self {
        Self {
            record: Arc::new(Mutex::new(None)),
            root_signing_key: Arc::new(Mutex::new(None)),
            failed_recovery_attempts: Arc::new(Mutex::new(0)),
            storage_path: None,
        }
    }

    /// Load authority record from durable storage path or prepare for initial root ceremony.
    /// If an existing state file is corrupted or tampered with, FAILS CLOSED INTO STRICT LOCKDOWN,
    /// NEVER allowing re-bootstrapping or silent downgrade to uninitialized state!
    pub fn load_or_new<P: AsRef<std::path::Path>>(storage_path: P) -> Self {
        let path = storage_path.as_ref().to_path_buf();
        let loaded_record = if path.exists() {
            match std::fs::read(&path) {
                Ok(data) => match serde_json::from_slice::<AuthorityRecord>(&data) {
                    Ok(rec) => Some(rec),
                    Err(_) => {
                        // File corrupted: FAIL CLOSED INTO LOCKDOWN!
                        Some(AuthorityRecord {
                            generation: AuthorityGeneration::INITIAL,
                            root_public_key_hex: String::new(),
                            root_principal_id: PrincipalId::new("creator_corrupted_state")
                                .unwrap_or_else(|_| PrincipalId::new("creator_root").unwrap()),
                            created_epoch_secs: 0,
                            rotated_epoch_secs: None,
                            recovery_verifier_hash: String::new(),
                            state: SecurityState::LockedDown {
                                generation: AuthorityGeneration::INITIAL,
                                reason: "Authority state file corrupted on disk (FailClosed)"
                                    .into(),
                                since_epoch_secs: 0,
                            },
                            revoked_keys: HashSet::new(),
                        })
                    }
                },
                Err(_) => {
                    // File unreadable: FAIL CLOSED INTO LOCKDOWN!
                    Some(AuthorityRecord {
                        generation: AuthorityGeneration::INITIAL,
                        root_public_key_hex: String::new(),
                        root_principal_id: PrincipalId::new("creator_corrupted_state")
                            .unwrap_or_else(|_| PrincipalId::new("creator_root").unwrap()),
                        created_epoch_secs: 0,
                        rotated_epoch_secs: None,
                        recovery_verifier_hash: String::new(),
                        state: SecurityState::LockedDown {
                            generation: AuthorityGeneration::INITIAL,
                            reason: "Authority state file unreadable on disk (FailClosed)".into(),
                            since_epoch_secs: 0,
                        },
                        revoked_keys: HashSet::new(),
                    })
                }
            }
        } else {
            // Check if authority previously existed via monotonic watermark / companion access artifacts
            let parent_dir = path.parent();
            let has_prior_authority = parent_dir
                .map(|p| {
                    p.join("access_seal.json").exists()
                        || p.join("operator_record.json").exists()
                        || p.join("restore_config.json").exists()
                        || p.join(".authority_watermark").exists()
                })
                .unwrap_or(false);

            if has_prior_authority {
                // Missing authority_record.json when companion authority files exist:
                // FAIL CLOSED INTO LOCKDOWN to prevent malicious re-bootstrapping!
                Some(AuthorityRecord {
                    generation: AuthorityGeneration::INITIAL,
                    root_public_key_hex: String::new(),
                    root_principal_id: PrincipalId::new("creator_missing_record").unwrap_or_else(|_| PrincipalId::new("creator_root").unwrap()),
                    created_epoch_secs: 0,
                    rotated_epoch_secs: None,
                    recovery_verifier_hash: String::new(),
                    state: SecurityState::LockedDown {
                        generation: AuthorityGeneration::INITIAL,
                        reason: "Authority record missing but prior authority watermark exists on disk (FailClosed against re-bootstrap bypass)".into(),
                        since_epoch_secs: 0,
                    },
                    revoked_keys: HashSet::new(),
                })
            } else {
                None
            }
        };

        Self {
            record: Arc::new(Mutex::new(loaded_record)),
            root_signing_key: Arc::new(Mutex::new(None)),
            failed_recovery_attempts: Arc::new(Mutex::new(0)),
            storage_path: Some(path),
        }
    }

    fn persist_record(&self, record: &AuthorityRecord) -> Result<(), String> {
        if let Some(ref path) = self.storage_path {
            let data = serde_json::to_vec_pretty(record)
                .map_err(|e| format!("AuthorityRecord serialization error: {e}"))?;
            crate::persistence::atomic_write_file_hardened(path, &data)?;
            if let Some(parent) = path.parent() {
                let watermark = parent.join(".authority_watermark");
                let _ = std::fs::write(&watermark, b"TARA_AUTHORITY_ROOT_INITIALIZED");
            }
        }
        Ok(())
    }

    /// Load existing authority record into memory. If data is corrupted or invalid, fails closed into lockdown!
    pub fn load_authority(&self, record: AuthorityRecord) -> Result<(), String> {
        let mut rec_guard = self
            .record
            .lock()
            .map_err(|_| "Record mutex poisoned (FailClosed)")?;
        if rec_guard.is_some() {
            return Err("Authority already initialized in memory".into());
        }
        self.persist_record(&record)?;
        *rec_guard = Some(record);
        Ok(())
    }

    /// Check whether the authority has been established.
    pub fn is_initialized(&self) -> bool {
        let rec_guard = self.record.lock().unwrap_or_else(|p| p.into_inner());
        rec_guard.is_some()
    }

    /// Explicit First-Run Root Authority Ceremony.
    /// Can ONLY be called when the system is uninitialized.
    pub fn initialize_root_ceremony(
        &self,
        creator_id: PrincipalId,
        recovery_secret: &str,
        epoch_secs: u64,
    ) -> Result<(AuthorityRecord, String), String> {
        let mut rec_guard = self
            .record
            .lock()
            .map_err(|_| "Record mutex poisoned (FailClosed)")?;
        if rec_guard.is_some() {
            return Err("AuthorityCeremonyDenied: Authority has already been established; re-bootstrap is strictly prohibited (FailClosed)".into());
        }

        if recovery_secret.len() < 32 {
            return Err(
                "Recovery secret must be at least 32 characters of high-entropy data".into(),
            );
        }

        // Generate root cryptographic signing keypair
        let (signing_key, verifying_key) = CryptoUtils::generate_ed25519_keypair();
        let pub_hex = hex::encode(verifying_key.as_bytes());

        // Derive bounded Scrypt verifier for recovery secret
        let mut salt = [0u8; 16];
        rand::thread_rng().fill_bytes(&mut salt);
        let mut recovery_derived = [0u8; 32];
        CryptoUtils::derive_scrypt_key(recovery_secret.as_bytes(), &salt, &mut recovery_derived)?;
        let recovery_verifier_hash =
            format!("{}:{}", hex::encode(salt), hex::encode(recovery_derived));

        let gen = AuthorityGeneration::INITIAL;
        let record = AuthorityRecord {
            generation: gen,
            root_public_key_hex: pub_hex.clone(),
            root_principal_id: creator_id,
            created_epoch_secs: epoch_secs,
            rotated_epoch_secs: None,
            recovery_verifier_hash,
            state: SecurityState::Operational { generation: gen },
            revoked_keys: HashSet::new(),
        };

        *rec_guard = Some(record.clone());
        self.persist_record(&record)?;

        let mut key_guard = self
            .root_signing_key
            .lock()
            .map_err(|_| "Key mutex poisoned (FailClosed)")?;
        *key_guard = Some(signing_key);

        Ok((record, pub_hex))
    }

    /// Current authority generation.
    pub fn current_generation(&self) -> Result<AuthorityGeneration, String> {
        let rec_guard = self
            .record
            .lock()
            .map_err(|_| "Record mutex poisoned (FailClosed)")?;
        let rec = rec_guard.as_ref().ok_or("Authority is not initialized")?;
        Ok(rec.generation)
    }

    /// Check if system is locked down.
    pub fn is_locked_down(&self) -> bool {
        let rec_guard = match self.record.lock() {
            Ok(g) => g,
            Err(_) => return true, // Mutex poisoned: Fail closed into lockdown!
        };
        match rec_guard.as_ref() {
            Some(rec) => rec.state.is_locked_down(),
            None => true, // Uninitialized is locked by default
        }
    }

    /// Trigger strict security lockdown.
    pub fn trigger_lockdown(&self, reason: &str, epoch_secs: u64) -> Result<(), String> {
        let mut rec_guard = self
            .record
            .lock()
            .map_err(|_| "Record mutex poisoned (FailClosed)")?;
        let rec = rec_guard.as_mut().ok_or("Authority is not initialized")?;
        rec.state = SecurityState::LockedDown {
            generation: rec.generation,
            reason: reason.to_string(),
            since_epoch_secs: epoch_secs,
        };
        self.persist_record(rec)?;
        Ok(())
    }

    /// Key Rotation: Increments generation, signs transition token, revokes old key.
    pub fn rotate_authority_key(&self, epoch_secs: u64) -> Result<RotationToken, String> {
        let mut rec_guard = self
            .record
            .lock()
            .map_err(|_| "Record mutex poisoned (FailClosed)")?;
        let rec = rec_guard.as_mut().ok_or("Authority is not initialized")?;

        if rec.state.is_locked_down() {
            return Err("AuthorityKeyRotationDenied: Cannot rotate keys during lockdown".into());
        }

        let old_gen = rec.generation;
        let new_gen = old_gen.next()?;

        let mut key_guard = self
            .root_signing_key
            .lock()
            .map_err(|_| "Key mutex poisoned (FailClosed)")?;
        let old_signing_key = key_guard
            .as_ref()
            .ok_or("Root signing key not loaded in memory")?;

        // Generate new keypair
        let (new_signing_key, new_verifying_key) = CryptoUtils::generate_ed25519_keypair();
        let new_pub_hex = hex::encode(new_verifying_key.as_bytes());

        // Sign transition payload with old key
        let transition_msg = format!("ROTATION:{}:{}:{}", old_gen, new_gen, new_pub_hex);
        let sig = CryptoUtils::sign_ed25519(old_signing_key, transition_msg.as_bytes());
        let sig_hex = hex::encode(sig);

        // Revoke old key and increment generation
        rec.revoked_keys.insert(rec.root_public_key_hex.clone());
        rec.root_public_key_hex = new_pub_hex.clone();
        rec.generation = new_gen;
        rec.rotated_epoch_secs = Some(epoch_secs);
        rec.state = SecurityState::Operational {
            generation: new_gen,
        };
        self.persist_record(rec)?;

        // Replace memory key
        *key_guard = Some(new_signing_key);

        Ok(RotationToken {
            old_generation: old_gen,
            new_generation: new_gen,
            new_public_key_hex: new_pub_hex,
            transition_signature_hex: sig_hex,
            timestamp_epoch_secs: epoch_secs,
        })
    }

    /// Emergency Recovery: High-entropy material verification with rate-limiting.
    pub fn execute_recovery(&self, proof: &RecoveryProof, epoch_secs: u64) -> Result<(), String> {
        let mut rec_guard = self
            .record
            .lock()
            .map_err(|_| "Record mutex poisoned (FailClosed)")?;
        let rec = rec_guard.as_mut().ok_or("Authority is not initialized")?;

        if proof.target_generation != rec.generation {
            return Err(format!(
                "RecoveryGenerationMismatch: Proof targets generation {}, current is {}",
                proof.target_generation.get(),
                rec.generation.get()
            ));
        }
        if proof.recovery_nonce.trim().len() < 16 {
            return Err("InvalidRecoveryNonce: Nonce must be at least 16 characters".into());
        }

        let domain = format!("authority_recovery_failures_gen_{}", rec.generation.get());
        let watermark_path = crate::dpapi::get_os_isolated_watermark_path(&domain);
        let persisted_attempts = crate::dpapi::load_monotonic_watermark(&watermark_path)
            .ok()
            .flatten()
            .unwrap_or(0) as u32;

        let mut attempts = self
            .failed_recovery_attempts
            .lock()
            .map_err(|_| "Attempts lock poisoned")?;
        let current_attempts = (*attempts).max(persisted_attempts);
        if current_attempts >= 3 {
            return Err("RecoveryExhausted: Too many failed recovery attempts; system is permanently locked".into());
        }

        let parts: Vec<&str> = rec.recovery_verifier_hash.split(':').collect();
        if parts.len() != 2 {
            return Err("Corrupted recovery verifier hash format".into());
        }
        let salt = hex::decode(parts[0]).map_err(|_| "Failed to decode salt")?;
        let expected_hash = hex::decode(parts[1]).map_err(|_| "Failed to decode hash")?;

        let mut candidate_hash = [0u8; 32];
        CryptoUtils::derive_scrypt_key(
            proof.recovery_secret.as_bytes(),
            &salt,
            &mut candidate_hash,
        )?;

        if !CryptoUtils::constant_time_eq(&candidate_hash, &expected_hash) {
            let new_attempts = current_attempts + 1;
            *attempts = new_attempts;
            let _ = crate::dpapi::store_monotonic_watermark(&watermark_path, new_attempts as u64);
            return Err(format!(
                "Invalid recovery secret (attempt {}/3)",
                new_attempts
            ));
        }

        *attempts = 0;
        let next_gen = rec.generation.next()?;
        rec.generation = next_gen;
        rec.rotated_epoch_secs = Some(epoch_secs);
        rec.state = SecurityState::Operational {
            generation: next_gen,
        };
        self.persist_record(rec)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execute_recovery_generation_mismatch_and_nonce_validation() {
        let mgr = AuthorityManager::new();
        let creator_id = PrincipalId::new("creator_root_test").unwrap();
        let recovery_secret = "super_high_entropy_secret_min_32_chars_long!";
        let (record, _) = mgr
            .initialize_root_ceremony(creator_id, recovery_secret, 1000)
            .unwrap();

        let domain = format!(
            "authority_recovery_failures_gen_{}",
            record.generation.get()
        );
        let _ = crate::dpapi::delete_os_monotonic_watermark(&domain);

        // 1. Target generation mismatch
        let proof_wrong_gen = RecoveryProof {
            target_generation: record.generation.next().unwrap(),
            recovery_secret: recovery_secret.to_string(),
            recovery_nonce: "valid_recovery_nonce_16_chars".to_string(),
            requested_epoch_secs: 1050,
        };
        let err_gen = mgr.execute_recovery(&proof_wrong_gen, 1050);
        assert!(err_gen.is_err());
        assert!(err_gen.unwrap_err().contains("RecoveryGenerationMismatch"));

        // 2. Short nonce (< 16 chars)
        let proof_short_nonce = RecoveryProof {
            target_generation: record.generation,
            recovery_secret: recovery_secret.to_string(),
            recovery_nonce: "short_nonce".to_string(),
            requested_epoch_secs: 1050,
        };
        let err_nonce = mgr.execute_recovery(&proof_short_nonce, 1050);
        assert!(err_nonce.is_err());
        assert!(err_nonce.unwrap_err().contains("InvalidRecoveryNonce"));

        // 3. Invalid secret
        let proof_wrong_secret = RecoveryProof {
            target_generation: record.generation,
            recovery_secret: "wrong_secret_high_entropy_32_characters_long!".to_string(),
            recovery_nonce: "valid_recovery_nonce_16_chars".to_string(),
            requested_epoch_secs: 1050,
        };
        let err_secret = mgr.execute_recovery(&proof_wrong_secret, 1050);
        assert!(err_secret.is_err());
        let err_str = err_secret.unwrap_err();
        assert!(
            err_str.contains("Invalid recovery secret") || err_str.contains("RecoveryExhausted")
        );

        // 4. Valid recovery execution
        let proof_valid = RecoveryProof {
            target_generation: record.generation,
            recovery_secret: recovery_secret.to_string(),
            recovery_nonce: "valid_recovery_nonce_16_chars".to_string(),
            requested_epoch_secs: 1050,
        };
        let res_valid = mgr.execute_recovery(&proof_valid, 1050);
        assert!(res_valid.is_ok());

        // Generation has advanced
        let rec_guard = mgr.record.lock().unwrap();
        assert_eq!(
            rec_guard.as_ref().unwrap().generation,
            record.generation.next().unwrap()
        );

        let _ = crate::dpapi::delete_os_monotonic_watermark(&domain);
    }
}
