//! Multi-path Recovery & Key Rotation System for TARA Creator Identity.
//!
//! Supported Recovery Paths:
//! 1. Cryptographic Recovery Code (PBKDF2-HMAC-SHA256 verifier with lockout)
//! 2. Verified Google Creator Account (cryptographically verified ID token / claims)
//! 3. Trusted Authorized Device Signature (registered Ed25519 device key)

use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

pub const CANONICAL_CREATOR_ID: &str = "ROOT_OPERATOR";
pub const DEFAULT_DISPLAY_NAME: &str = "OPERATOR_ROOT";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryAuthorizationProof {
    pub proof_id: String,
    pub creator_id: String,
    pub method: String, // "RECOVERY_CODE", "GOOGLE_AUTH", "TRUSTED_DEVICE"
    pub authorized_at: String,
    pub expires_at: u64,
    pub nonce: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryConfig {
    pub verifier_version: u32,
    pub salt_hex: String,
    pub verifier_hash_hex: String,
    pub iterations: u32,
    pub lockout_failures: u32,
    pub lockout_until: Option<u64>,
}

pub struct RecoveryManager {
    pub config_path: PathBuf,
    pub active_proofs: HashMap<String, RecoveryAuthorizationProof>,
    pub failure_count: u32,
    pub lockout_until: Option<u64>,
}

impl RecoveryManager {
    pub fn new<P: AsRef<Path>>(config_path: Option<P>) -> Self {
        let path = match config_path {
            Some(p) => p.as_ref().to_path_buf(),
            None => PathBuf::from("storage/recovery/recovery_config.json"),
        };

        Self {
            config_path: path,
            active_proofs: HashMap::new(),
            failure_count: 0,
            lockout_until: None,
        }
    }

    pub fn is_locked_out(&self) -> bool {
        if let Some(until) = self.lockout_until {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            return now < until;
        }
        false
    }

    pub fn generate_recovery_code(
        &mut self,
        plaintext_code: &str,
    ) -> Result<RecoveryConfig, String> {
        let mut salt = [0u8; 16];
        rand::Rng::fill(&mut rand::thread_rng(), &mut salt);
        let salt_hex = hex::encode(salt);

        let mut hash_out = [0u8; 32];
        let iterations = 100_000u32;
        pbkdf2_hmac_sha256(plaintext_code.as_bytes(), &salt, iterations, &mut hash_out);
        let verifier_hash_hex = hex::encode(hash_out);

        let config = RecoveryConfig {
            verifier_version: 2,
            salt_hex,
            verifier_hash_hex,
            iterations,
            lockout_failures: 0,
            lockout_until: None,
        };

        if let Some(parent) = self.config_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let data = serde_json::to_string_pretty(&config).map_err(|e| e.to_string())?;
        fs::write(&self.config_path, data).map_err(|e| e.to_string())?;

        Ok(config)
    }

    pub fn verify_recovery_code(
        &mut self,
        candidate_code: &str,
    ) -> Result<RecoveryAuthorizationProof, String> {
        if self.is_locked_out() {
            return Err("Recovery is currently locked out due to excessive failed attempts".into());
        }

        if !self.config_path.exists() {
            return Err("Recovery configuration not initialized".into());
        }

        let data = fs::read_to_string(&self.config_path).map_err(|e| e.to_string())?;
        let config: RecoveryConfig = serde_json::from_str(&data).map_err(|e| e.to_string())?;

        let salt = hex::decode(&config.salt_hex).map_err(|e| e.to_string())?;
        let mut hash_out = [0u8; 32];
        pbkdf2_hmac_sha256(
            candidate_code.as_bytes(),
            &salt,
            config.iterations,
            &mut hash_out,
        );

        if hex::encode(hash_out) != config.verifier_hash_hex {
            self.failure_count += 1;
            if self.failure_count >= 5 {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                self.lockout_until = Some(now + 900); // 15 min lockout
            }
            return Err("Invalid recovery code".into());
        }

        self.failure_count = 0;
        self.lockout_until = None;

        let proof = RecoveryAuthorizationProof {
            proof_id: format!("proof_{}", rand::random::<u64>()),
            creator_id: CANONICAL_CREATOR_ID.to_string(),
            method: "RECOVERY_CODE".to_string(),
            authorized_at: crate::now_iso(),
            expires_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                + 600,
            nonce: format!("{:x}", rand::random::<u128>()),
        };

        self.active_proofs
            .insert(proof.proof_id.clone(), proof.clone());
        Ok(proof)
    }

    pub fn verify_google_recovery(
        &mut self,
        verified_email: &str,
        expected_email: &str,
    ) -> Result<RecoveryAuthorizationProof, String> {
        if self.is_locked_out() {
            return Err("Recovery is locked out".into());
        }

        if verified_email
            .trim()
            .eq_ignore_ascii_case(expected_email.trim())
        {
            let proof = RecoveryAuthorizationProof {
                proof_id: format!("proof_g_{}", rand::random::<u64>()),
                creator_id: CANONICAL_CREATOR_ID.to_string(),
                method: "GOOGLE_AUTH".to_string(),
                authorized_at: crate::now_iso(),
                expires_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
                    + 600,
                nonce: format!("{:x}", rand::random::<u128>()),
            };
            self.active_proofs
                .insert(proof.proof_id.clone(), proof.clone());
            Ok(proof)
        } else {
            Err("Google account email does not match registered creator email".into())
        }
    }

    pub fn verify_trusted_device_recovery(
        &mut self,
        device_id: &str,
        signature_valid: bool,
    ) -> Result<RecoveryAuthorizationProof, String> {
        if !signature_valid {
            return Err("Invalid trusted device cryptographic signature".into());
        }

        let proof = RecoveryAuthorizationProof {
            proof_id: format!("proof_dev_{}", rand::random::<u64>()),
            creator_id: CANONICAL_CREATOR_ID.to_string(),
            method: format!("TRUSTED_DEVICE:{}", device_id),
            authorized_at: crate::now_iso(),
            expires_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                + 600,
            nonce: format!("{:x}", rand::random::<u128>()),
        };
        self.active_proofs
            .insert(proof.proof_id.clone(), proof.clone());
        Ok(proof)
    }

    pub fn recover_and_rotate_key(
        &mut self,
        proof_id: &str,
        new_public_key_hex: &str,
        profile: &mut crate::access::profile::CreatorIdentity,
    ) -> Result<(), String> {
        let proof = self
            .active_proofs
            .remove(proof_id)
            .ok_or_else(|| "Invalid or expired recovery authorization proof".to_string())?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if now > proof.expires_at {
            return Err("Recovery authorization proof has expired".into());
        }

        profile.rotate_root_key(
            new_public_key_hex,
            true,
            &format!("RECOVERY:{}", proof.method),
        )
    }

    pub fn verify_storage_integrity(&self) -> bool {
        if !self.config_path.exists() {
            return true; // Not configured yet is valid
        }
        if let Ok(data) = fs::read_to_string(&self.config_path) {
            serde_json::from_str::<RecoveryConfig>(&data).is_ok()
        } else {
            false
        }
    }
}

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
