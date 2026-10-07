//! Model Integrity: Creator-authorized HMAC-SHA256 seal for versions_manifest.json
//! and the active model weights SHA-256.
//!
//! Design:
//!   - A HMAC-SHA256 seal is computed over: `manifest_sha256 + model_sha256`
//!   - The HMAC key is derived via Scrypt from `TARA_MANIFEST_SEAL_KEY` env var + stored salt.
//!   - The seal is stored in `storage/vault/access/manifest_seal.json`.
//!   - Verification runs on every startup: if seal absent/invalid → REJECT and refuse to load model.
//!   - Only a holder of the TARA_MANIFEST_SEAL_KEY can produce a valid seal → Creator-authorized.
//!
//! Attack scenarios:
//!   A) valid model + valid manifest → ACCEPT (seal verifies)
//!   B) model file modified only → model SHA changes → seal HMAC fails → REJECT
//!   C) manifest file modified only → manifest SHA changes → seal HMAC fails → REJECT
//!   D) model + manifest both modified → seal HMAC still fails (attacker lacks HMAC key) → REJECT
//!   E) unauthorized signer (wrong TARA_MANIFEST_SEAL_KEY) → HMAC tag mismatch → REJECT
//!   The only way to update the seal is to know TARA_MANIFEST_SEAL_KEY and call seal_manifest().

use constant_time_eq::constant_time_eq;
use hmac::{Hmac, Mac};
use rand::RngCore;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

type HmacSha256 = Hmac<Sha256>;

/// Error type for integrity operations.
#[derive(Debug)]
pub enum IntegrityError {
    /// Seal file is missing — manifest has never been authorized.
    SealMissing,
    /// HMAC verification failed — model or manifest has been tampered with.
    HmacMismatch,
    /// A required file could not be read.
    IoError(String),
    /// The TARA_MANIFEST_SEAL_KEY env var is not set.
    KeyNotSet,
    /// JSON parsing or format error.
    FormatError(String),
}

impl std::fmt::Display for IntegrityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SealMissing => write!(
                f,
                "manifest_seal.json is missing — run seal_manifest() with Creator authority"
            ),
            Self::HmacMismatch => write!(
                f,
                "HMAC verification failed — model or manifest has been tampered"
            ),
            Self::IoError(e) => write!(f, "IO error: {}", e),
            Self::KeyNotSet => write!(
                f,
                "TARA_MANIFEST_SEAL_KEY is not set — Creator must set this env var"
            ),
            Self::FormatError(e) => write!(f, "Format error: {}", e),
        }
    }
}

pub struct ModelIntegrityEngine {
    seal_path: PathBuf,
    manifest_path: PathBuf,
    model_safetensors_path: PathBuf,
}

impl ModelIntegrityEngine {
    pub fn new<P: AsRef<Path>>(repo_root: P) -> Self {
        let root = repo_root.as_ref();
        Self {
            seal_path: root.join("storage/vault/access/manifest_seal.json"),
            manifest_path: root.join("storage/models/versions_manifest.json"),
            model_safetensors_path: root.join("storage/models/tara/model.safetensors"),
        }
    }

    /// Derive HMAC key from TARA_MANIFEST_SEAL_KEY + stored salt using Scrypt (N=2^14, r=8, p=1).
    /// Lower Scrypt cost (n=14) vs keystore (n=17) is intentional — seal is verified on every startup.
    fn derive_hmac_key(&self, salt: &[u8]) -> Result<[u8; 32], IntegrityError> {
        let secret =
            std::env::var("TARA_MANIFEST_SEAL_KEY").map_err(|_| IntegrityError::KeyNotSet)?;
        if secret.trim().is_empty() {
            return Err(IntegrityError::KeyNotSet);
        }

        const SCRYPT_LOG_N: u8 = 14;
        const SCRYPT_R: u32 = 8;
        const SCRYPT_P: u32 = 1;
        const SCRYPT_KEY_LEN: usize = 32;

        let params = scrypt::Params::new(SCRYPT_LOG_N, SCRYPT_R, SCRYPT_P, SCRYPT_KEY_LEN)
            .map_err(|e| IntegrityError::FormatError(format!("Scrypt params: {:?}", e)))?;
        let mut key = [0u8; 32];
        scrypt::scrypt(secret.as_bytes(), salt, &params, &mut key)
            .map_err(|e| IntegrityError::FormatError(format!("Scrypt: {:?}", e)))?;
        Ok(key)
    }

    /// Compute SHA-256 of a file on disk.
    fn file_sha256<P: AsRef<Path>>(path: P) -> Result<String, IntegrityError> {
        let bytes = fs::read(path.as_ref())
            .map_err(|e| IntegrityError::IoError(format!("{}: {}", path.as_ref().display(), e)))?;
        let hash = Sha256::digest(&bytes);
        Ok(format!("{:x}", hash))
    }

    /// Compute the canonical HMAC message: `manifest_sha256 || ":" || model_sha256`
    fn build_message(manifest_sha: &str, model_sha: &str) -> Vec<u8> {
        format!("TARA_MANIFEST_SEAL:{}:{}", manifest_sha, model_sha).into_bytes()
    }

    /// Compute HMAC-SHA256 tag over the message using the derived key.
    fn compute_hmac(key: &[u8; 32], message: &[u8]) -> String {
        let mut mac = HmacSha256::new_from_slice(key).expect("HMAC-SHA256 accepts any key length");
        mac.update(message);
        hex::encode(mac.finalize().into_bytes())
    }

    /// Creator-authorized: compute fresh SHAs, compute HMAC, write seal.
    /// Only callable by someone who holds TARA_MANIFEST_SEAL_KEY.
    pub fn seal_manifest(&self) -> Result<SealResult, IntegrityError> {
        // Require key is set before doing anything
        let _ = std::env::var("TARA_MANIFEST_SEAL_KEY").map_err(|_| IntegrityError::KeyNotSet)?;

        // Generate a fresh random salt for this seal
        let mut salt = [0u8; 16];
        rand::thread_rng().fill_bytes(&mut salt);

        let manifest_sha = Self::file_sha256(&self.manifest_path)?;
        let model_sha = Self::file_sha256(&self.model_safetensors_path)?;
        let message = Self::build_message(&manifest_sha, &model_sha);

        let key = self.derive_hmac_key(&salt)?;
        let hmac_tag = Self::compute_hmac(&key, &message);

        let seal = json!({
            "version": 1,
            "algorithm": "HMAC-SHA256+Scrypt(n=14,r=8,p=1)",
            "sealed_at": tara_engine::now_iso(),
            "salt": hex::encode(salt),
            "manifest_sha256": manifest_sha,
            "model_sha256": model_sha,
            "hmac_tag": hmac_tag,
            "note": "Creator-authorized integrity seal. Requires TARA_MANIFEST_SEAL_KEY to produce or verify."
        });

        // Atomic write via temp file
        let seal_dir = self.seal_path.parent().expect("seal path has parent");
        let _ = fs::create_dir_all(seal_dir);
        let temp = seal_dir.join(format!("manifest_seal.tmp.{}", rand::random::<u32>()));
        fs::write(&temp, serde_json::to_string_pretty(&seal).unwrap())
            .map_err(|e| IntegrityError::IoError(format!("write temp seal: {}", e)))?;
        fs::rename(&temp, &self.seal_path)
            .map_err(|e| IntegrityError::IoError(format!("rename seal: {}", e)))?;

        Ok(SealResult {
            manifest_sha256: manifest_sha,
            model_sha256: model_sha,
            hmac_tag,
            seal_path: self.seal_path.to_string_lossy().to_string(),
        })
    }

    /// Verify the integrity seal. Returns Ok(()) only if:
    ///   1. Seal file exists
    ///   2. TARA_MANIFEST_SEAL_KEY is set
    ///   3. Current manifest SHA matches seal
    ///   4. Current model SHA matches seal
    ///   5. HMAC tag matches (computed fresh with same key+salt)
    ///
    /// Fails CLOSED on any error.
    pub fn verify(&self) -> Result<VerifyResult, IntegrityError> {
        // 1. Load seal file
        let raw = fs::read_to_string(&self.seal_path).map_err(|_| IntegrityError::SealMissing)?;
        let seal: Value = serde_json::from_str(&raw)
            .map_err(|e| IntegrityError::FormatError(format!("seal JSON: {}", e)))?;

        let salt_hex = seal["salt"]
            .as_str()
            .ok_or_else(|| IntegrityError::FormatError("missing salt".into()))?;
        let stored_hmac = seal["hmac_tag"]
            .as_str()
            .ok_or_else(|| IntegrityError::FormatError("missing hmac_tag".into()))?;
        let stored_manifest_sha = seal["manifest_sha256"]
            .as_str()
            .ok_or_else(|| IntegrityError::FormatError("missing manifest_sha256".into()))?;
        let stored_model_sha = seal["model_sha256"]
            .as_str()
            .ok_or_else(|| IntegrityError::FormatError("missing model_sha256".into()))?;

        let salt = hex::decode(salt_hex)
            .map_err(|e| IntegrityError::FormatError(format!("salt hex: {}", e)))?;

        // 2. Compute CURRENT SHAs from disk
        let current_manifest_sha = Self::file_sha256(&self.manifest_path)?;
        let current_model_sha = Self::file_sha256(&self.model_safetensors_path)?;

        // 3. Derive HMAC key (requires TARA_MANIFEST_SEAL_KEY)
        let key = self.derive_hmac_key(&salt)?;

        // 4. Compute expected HMAC over stored SHAs (what was sealed)
        let message = Self::build_message(stored_manifest_sha, stored_model_sha);
        let expected_hmac = Self::compute_hmac(&key, &message);

        // 5. Constant-time HMAC comparison (prevent timing attacks)
        let hmac_ok = constant_time_eq(stored_hmac.as_bytes(), expected_hmac.as_bytes());
        let manifest_sha_ok = current_manifest_sha == stored_manifest_sha;
        let model_sha_ok = current_model_sha == stored_model_sha;

        if !hmac_ok || !manifest_sha_ok || !model_sha_ok {
            return Err(IntegrityError::HmacMismatch);
        }

        Ok(VerifyResult {
            manifest_sha256: current_manifest_sha,
            model_sha256: current_model_sha,
            hmac_verified: true,
            manifest_match: true,
            model_match: true,
        })
    }
}

pub struct SealResult {
    pub manifest_sha256: String,
    pub model_sha256: String,
    pub hmac_tag: String,
    pub seal_path: String,
}

pub struct VerifyResult {
    pub manifest_sha256: String,
    pub model_sha256: String,
    pub hmac_verified: bool,
    pub manifest_match: bool,
    pub model_match: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static TEST_MUTEX: Mutex<()> = Mutex::new(());

    struct TestDir(std::path::PathBuf);
    impl TestDir {
        fn new() -> Self {
            let p = std::env::temp_dir().join(format!("tara_test_{}", rand::random::<u64>()));
            let _ = std::fs::create_dir_all(&p);
            Self(p)
        }
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }
    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn setup_test_env(dir: &std::path::Path) -> ModelIntegrityEngine {
        // Create minimal versions_manifest.json
        let models_dir = dir.join("storage/models");
        std::fs::create_dir_all(&models_dir).unwrap();
        std::fs::create_dir_all(dir.join("storage/models/tara")).unwrap();
        std::fs::create_dir_all(dir.join("storage/vault/access")).unwrap();

        std::fs::write(
            models_dir.join("versions_manifest.json"),
            r#"{"versions":{},"active_version":"test"}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("storage/models/tara/model.safetensors"),
            b"FAKE_MODEL_BYTES_FOR_TEST_DO_NOT_PROMOTE",
        )
        .unwrap();

        ModelIntegrityEngine::new(dir)
    }

    /// A) valid model + valid manifest → ACCEPT
    #[test]
    fn test_a_valid_seal_accepts() {
        let _lock = TEST_MUTEX.lock().unwrap();
        let dir = TestDir::new();
        std::env::set_var("TARA_MANIFEST_SEAL_KEY", "test_creator_secret_a");
        let engine = setup_test_env(dir.path());
        let seal = engine.seal_manifest().unwrap();
        assert!(!seal.hmac_tag.is_empty());
        let result = engine.verify().unwrap();
        assert!(result.hmac_verified);
        assert!(result.manifest_match);
        assert!(result.model_match);
    }

    /// B) model file modified only → REJECT
    #[test]
    fn test_b_model_tampered_reject() {
        let _lock = TEST_MUTEX.lock().unwrap();
        let dir = TestDir::new();
        std::env::set_var("TARA_MANIFEST_SEAL_KEY", "test_creator_secret_b");
        let engine = setup_test_env(dir.path());
        engine.seal_manifest().unwrap();
        // Tamper model
        let model_path = dir.path().join("storage/models/tara/model.safetensors");
        std::fs::write(&model_path, b"TAMPERED_MODEL_BYTES").unwrap();
        let result = engine.verify();
        assert!(matches!(result, Err(IntegrityError::HmacMismatch)));
    }

    /// C) manifest file modified only → REJECT
    #[test]
    fn test_c_manifest_tampered_reject() {
        let _lock = TEST_MUTEX.lock().unwrap();
        let dir = TestDir::new();
        std::env::set_var("TARA_MANIFEST_SEAL_KEY", "test_creator_secret_c");
        let engine = setup_test_env(dir.path());
        engine.seal_manifest().unwrap();
        // Tamper manifest
        let manifest_path = dir.path().join("storage/models/versions_manifest.json");
        std::fs::write(
            &manifest_path,
            r#"{"versions":{},"active_version":"TAMPERED"}"#,
        )
        .unwrap();
        let result = engine.verify();
        assert!(matches!(result, Err(IntegrityError::HmacMismatch)));
    }

    /// D) model + manifest both modified → REJECT (attacker lacks HMAC key)
    #[test]
    fn test_d_both_tampered_reject() {
        let _lock = TEST_MUTEX.lock().unwrap();
        let dir = TestDir::new();
        std::env::set_var("TARA_MANIFEST_SEAL_KEY", "test_creator_secret_d");
        let engine = setup_test_env(dir.path());
        engine.seal_manifest().unwrap();
        // Tamper both
        std::fs::write(
            dir.path().join("storage/models/tara/model.safetensors"),
            b"TAMPERED",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("storage/models/versions_manifest.json"),
            r#"{"tampered":true}"#,
        )
        .unwrap();
        let result = engine.verify();
        assert!(matches!(result, Err(IntegrityError::HmacMismatch)));
    }

    /// E) wrong TARA_MANIFEST_SEAL_KEY (unauthorized signer) → REJECT
    #[test]
    fn test_e_wrong_key_reject() {
        let _lock = TEST_MUTEX.lock().unwrap();
        let dir = TestDir::new();
        std::env::set_var("TARA_MANIFEST_SEAL_KEY", "correct_creator_key");
        let engine = setup_test_env(dir.path());
        engine.seal_manifest().unwrap();
        // Now verify with wrong key
        std::env::set_var("TARA_MANIFEST_SEAL_KEY", "attacker_wrong_key");
        let result = engine.verify();
        assert!(matches!(result, Err(IntegrityError::HmacMismatch)));
    }

    /// Seal missing → REJECT
    #[test]
    fn test_seal_missing_reject() {
        let _lock = TEST_MUTEX.lock().unwrap();
        let dir = TestDir::new();
        std::env::set_var("TARA_MANIFEST_SEAL_KEY", "test_key_f");
        let engine = setup_test_env(dir.path());
        // Do NOT call seal_manifest — seal file absent
        let result = engine.verify();
        assert!(matches!(result, Err(IntegrityError::SealMissing)));
    }
}
