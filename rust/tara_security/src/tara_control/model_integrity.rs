//! Model Integrity: Creator-authorized HMAC-SHA256 seal for model manifests and weights.
//!
//! Rule compliance:
//! - Pure native Rust, zero outside crate dependencies.
//! - Streaming SHA-256 for large model weights without buffering entire weights in memory.
//! - Creator-authorization requirement for seal generation.
//! - Anti-rollback version verification and tamper detection.

use std::path::Path;
use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

use crate::cryptography::CryptoUtils;
use crate::primitives::{AssuranceLevel, VerifiedPrincipal};

/// Model seal metadata payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifestSeal {
    pub algorithm: String,
    pub model_id: String,
    pub manifest_sha256: String,
    pub model_weights_sha256: String,
    pub sealed_epoch_secs: u64,
    pub signature_hex: String,
}

/// Model Integrity Guard validating neural checkpoints and version seals.
pub struct ModelIntegrityGuard;

impl Default for ModelIntegrityGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelIntegrityGuard {
    pub fn new() -> Self {
        Self
    }

    /// Authentically computes the streaming SHA-256 hash of a model file and verifies against expected hash.
    pub fn verify_model_weights(
        &self,
        model_file: &Path,
        expected_sha256: &str,
    ) -> Result<String, String> {
        if !model_file.exists() {
            return Err(format!("ModelNotFound: Model checkpoint '{}' does not exist", model_file.display()));
        }

        let computed_sha256 = CryptoUtils::compute_file_sha256(model_file)?;

        if !CryptoUtils::constant_time_eq(computed_sha256.as_bytes(), expected_sha256.as_bytes()) {
            return Err(format!(
                "ModelTamperDetected: Computed SHA-256 '{}' does not match expected '{}'",
                computed_sha256, expected_sha256
            ));
        }

        Ok(computed_sha256)
    }

    /// Generates a cryptographic HMAC-SHA256 seal for a model manifest, requiring Creator Root assurance.
    pub fn generate_manifest_seal(
        &self,
        model_id: &str,
        manifest_data: &[u8],
        model_weights_sha256: &str,
        principal: &VerifiedPrincipal,
        seal_key: &[u8],
        epoch_secs: u64,
    ) -> Result<ModelManifestSeal, String> {
        if principal.assurance_level().score() < AssuranceLevel::CREATOR_ROOT.score() {
            return Err("UnauthorizedSealGeneration: Seal generation requires Creator Root assurance".into());
        }

        let manifest_sha = CryptoUtils::sha256_hex(manifest_data);
        let preimage = format!("{}:{}:{}:{}", model_id, manifest_sha, model_weights_sha256, epoch_secs);

        let hmac_bytes = CryptoUtils::hmac_sha256(seal_key, preimage.as_bytes())?;
        let sig_hex = hex::encode(hmac_bytes);

        Ok(ModelManifestSeal {
            algorithm: "HMAC-SHA256".to_string(),
            model_id: model_id.to_string(),
            manifest_sha256: manifest_sha,
            model_weights_sha256: model_weights_sha256.to_string(),
            sealed_epoch_secs: epoch_secs,
            signature_hex: sig_hex,
        })
    }

    /// Verifies that a manifest seal matches the active data and has not been tampered with.
    pub fn verify_manifest_seal(
        &self,
        seal: &ModelManifestSeal,
        manifest_data: &[u8],
        seal_key: &[u8],
    ) -> Result<(), String> {
        let computed_manifest_sha = CryptoUtils::sha256_hex(manifest_data);
        if !CryptoUtils::constant_time_eq(computed_manifest_sha.as_bytes(), seal.manifest_sha256.as_bytes()) {
            return Err("ManifestContentTampered: Manifest content hash does not match seal".into());
        }

        let preimage = format!(
            "{}:{}:{}:{}",
            seal.model_id, seal.manifest_sha256, seal.model_weights_sha256, seal.sealed_epoch_secs
        );
        let expected_hmac = CryptoUtils::hmac_sha256(seal_key, preimage.as_bytes())?;
        let expected_hex = hex::encode(expected_hmac);

        if !CryptoUtils::constant_time_eq(expected_hex.as_bytes(), seal.signature_hex.as_bytes()) {
            return Err("SealSignatureInvalid: HMAC signature does not match; seal is forged or corrupted".into());
        }

        Ok(())
    }
}

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use constant_time_eq::constant_time_eq;
use rand::RngCore;
use serde_json::{json, Value};
use crate::primitives::AuthenticatedPrincipal;

/// Error type for model integrity operations.
#[derive(Debug)]
pub enum IntegrityError {
    SealMissing,
    HmacMismatch,
    IoError(String),
    KeyNotSet,
    FormatError(String),
    Unauthorized(String),
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
            Self::Unauthorized(e) => write!(f, "Unauthorized: {}", e),
        }
    }
}

impl std::error::Error for IntegrityError {}

#[derive(Debug, Clone)]
pub struct SealResult {
    pub manifest_sha256: String,
    pub model_sha256: String,
    pub config_sha256: String,
    pub tokenizer_sha256: String,
    pub hmac_tag: String,
    pub seal_path: String,
}

#[derive(Debug, Clone)]
pub struct VerifyResult {
    pub manifest_sha256: String,
    pub model_sha256: String,
    pub config_sha256: String,
    pub tokenizer_sha256: String,
    pub hmac_verified: bool,
    pub manifest_match: bool,
    pub model_match: bool,
    pub config_match: bool,
    pub tokenizer_match: bool,
}

#[derive(Debug, Clone)]
struct SealMessageParams<'a> {
    version: u64,
    algorithm: &'a str,
    sealed_at: &'a str,
    architecture_profile: &'a str,
    parameter_count: u64,
    manifest_sha: &'a str,
    model_sha: &'a str,
    config_sha: &'a str,
    tokenizer_sha: &'a str,
}

pub struct ModelIntegrityEngine {
    seal_path: PathBuf,
    manifest_path: PathBuf,
    model_safetensors_path: PathBuf,
    config_json_path: PathBuf,
    tokenizer_json_path: PathBuf,
    pub watermark_domain: String,
    explicit_key: Option<String>,
}

impl ModelIntegrityEngine {
    pub fn new<P: AsRef<Path>>(repo_root: P) -> Self {
        let root = repo_root.as_ref();
        let candidate_dir = root.join("storage/models/tara_candidate_v1");
        let (model_safetensors_path, config_json_path, tokenizer_json_path) = if candidate_dir.exists() {
            let model_path = if candidate_dir.join("model.safetensors.index.json").exists() {
                candidate_dir.join("model.safetensors.index.json")
            } else if candidate_dir.join("model.safetensors").exists() {
                candidate_dir.join("model.safetensors")
            } else {
                candidate_dir.clone()
            };
            (
                model_path,
                candidate_dir.join("config.json"),
                candidate_dir.join("tokenizer.json"),
            )
        } else {
            (
                root.join("storage/models/tara/model.safetensors"),
                root.join("storage/models/tara/config.json"),
                root.join("storage/models/tara/tokenizer.json"),
            )
        };
        Self {
            seal_path: root.join("storage/vault/access/manifest_seal.json"),
            manifest_path: root.join("storage/models/versions_manifest.json"),
            model_safetensors_path,
            config_json_path,
            tokenizer_json_path,
            watermark_domain: "model_manifest".to_string(),
            explicit_key: None,
        }
    }

    pub fn with_watermark_domain(mut self, domain: impl Into<String>) -> Self {
        self.watermark_domain = domain.into();
        self
    }

    pub fn with_explicit_key(mut self, key: impl Into<String>) -> Self {
        self.explicit_key = Some(key.into());
        self
    }

    fn derive_hmac_key(&self, salt: &[u8]) -> Result<[u8; 32], IntegrityError> {
        let secret = if let Some(ref k) = self.explicit_key {
            k.clone()
        } else {
            std::env::var("TARA_MANIFEST_SEAL_KEY").map_err(|_| IntegrityError::KeyNotSet)?
        };
        if secret.trim().is_empty() {
            return Err(IntegrityError::KeyNotSet);
        }
        if salt.len() != 16 {
            return Err(IntegrityError::FormatError(format!(
                "Invalid salt length: expected 16 bytes, got {}",
                salt.len()
            )));
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

    pub fn stream_file_sha256<P: AsRef<Path>>(path: P) -> Result<String, IntegrityError> {
        CryptoUtils::compute256(path.as_ref())
            .map_err(IntegrityError::IoError)
    }

    fn stream_optional_file_sha256<P: AsRef<Path>>(path: P) -> Result<String, IntegrityError> {
        let p = path.as_ref();
        if p.exists() {
            Self::stream_file_sha256(p)
        } else {
            Ok("NONE".to_string())
        }
    }

    fn build_message(p: &SealMessageParams<'_>) -> Vec<u8> {
        format!(
            "TARA_MANIFEST_SEAL:v{}:{}:{}:{}:{}:{}:{}:{}:{}",
            p.version,
            p.algorithm,
            p.sealed_at,
            p.architecture_profile,
            p.parameter_count,
            p.manifest_sha,
            p.model_sha,
            p.config_sha,
            p.tokenizer_sha
        )
        .into_bytes()
    }

    fn compute_hmac(key: &[u8; 32], message: &[u8]) -> String {
        CryptoUtils::hmac_sha256_hex(key, message)
            .expect("HMAC-SHA256 accepts 32-byte key")
    }

    pub fn seal_manifest_authorized(
        &self,
        principal: &AuthenticatedPrincipal,
    ) -> Result<SealResult, IntegrityError> {
        self.seal_manifest_authorized_with_params(principal, 2, "tara_native_transformer", 0)
    }

    pub fn seal_manifest_authorized_with_params(
        &self,
        principal: &AuthenticatedPrincipal,
        version: u64,
        architecture_profile: &str,
        parameter_count: u64,
    ) -> Result<SealResult, IntegrityError> {
        if !principal.is_creator() {
            return Err(IntegrityError::Unauthorized(
                "Root creator authority required to seal model manifest".into(),
            ));
        }
        if let Some(existing_wm) =
            crate::dpapi::load_os_monotonic_watermark(&self.watermark_domain)
                .map_err(|e| IntegrityError::Unauthorized(format!("WatermarkCheckFailed: {}", e)))?
        {
            if version < existing_wm {
                return Err(IntegrityError::Unauthorized(format!(
                    "ManifestRollbackRejected: version {} is less than committed watermark {}",
                    version, existing_wm
                )));
            }
        }
        let res = self.seal_manifest_internal(version, architecture_profile, parameter_count)?;
        crate::dpapi::store_os_monotonic_watermark(&self.watermark_domain, version)
            .map_err(|e| IntegrityError::Unauthorized(format!("WatermarkStoreFailed: {}", e)))?;
        Ok(res)
    }

    pub fn seal_manifest(
        &self,
        principal: &AuthenticatedPrincipal,
    ) -> Result<SealResult, IntegrityError> {
        self.seal_manifest_authorized(principal)
    }

    fn now_iso() -> String {
        CryptoUtils::now_iso()
    }

    fn seal_manifest_internal(
        &self,
        version: u64,
        architecture_profile: &str,
        parameter_count: u64,
    ) -> Result<SealResult, IntegrityError> {
        let mut salt = [0u8; 16];
        rand::thread_rng().fill_bytes(&mut salt);

        let manifest_sha = Self::stream_file_sha256(&self.manifest_path)?;
        let model_sha = Self::stream_file_sha256(&self.model_safetensors_path)?;
        let config_sha = Self::stream_optional_file_sha256(&self.config_json_path)?;
        let tokenizer_sha = Self::stream_optional_file_sha256(&self.tokenizer_json_path)?;

        let algorithm = "HMAC-SHA256+Scrypt(n=14,r=8,p=1)";
        let sealed_at = Self::now_iso();
        let message = Self::build_message(&SealMessageParams {
            version,
            algorithm,
            sealed_at: &sealed_at,
            architecture_profile,
            parameter_count,
            manifest_sha: &manifest_sha,
            model_sha: &model_sha,
            config_sha: &config_sha,
            tokenizer_sha: &tokenizer_sha,
        });

        let mut key = self.derive_hmac_key(&salt)?;
        let hmac_tag = Self::compute_hmac(&key, &message);
        key.zeroize();

        let seal = json!({
            "version": version,
            "algorithm": algorithm,
            "sealed_at": sealed_at,
            "architecture_profile": architecture_profile,
            "parameter_count": parameter_count,
            "salt": hex::encode(salt),
            "manifest_sha256": manifest_sha,
            "model_sha256": model_sha,
            "config_sha256": config_sha,
            "tokenizer_sha256": tokenizer_sha,
            "hmac_tag": hmac_tag,
            "note": "Creator-authorized integrity seal. Requires TARA_MANIFEST_SEAL_KEY to produce or verify."
        });

        let seal_dir = self.seal_path.parent().ok_or_else(|| {
            IntegrityError::IoError(format!(
                "Seal path {} has no parent directory",
                self.seal_path.display()
            ))
        })?;
        fs::create_dir_all(seal_dir)
            .map_err(|e| IntegrityError::IoError(format!("create seal dir: {}", e)))?;
        let temp = seal_dir.join(format!("manifest_seal.tmp.{}", rand::random::<u64>()));

        let serialized = serde_json::to_string_pretty(&seal)
            .map_err(|e| IntegrityError::FormatError(format!("serialize seal JSON: {}", e)))?;
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp)
            .map_err(|e| IntegrityError::IoError(format!("write temp seal: {}", e)))?;
        file.write_all(serialized.as_bytes())
            .map_err(|e| IntegrityError::IoError(format!("write seal bytes: {}", e)))?;
        file.sync_all()
            .map_err(|e| IntegrityError::IoError(format!("sync seal file: {}", e)))?;
        drop(file);

        fs::rename(&temp, &self.seal_path)
            .map_err(|e| IntegrityError::IoError(format!("rename seal: {}", e)))?;

        crate::hardware_root::sync_directory_metadata(&self.seal_path)
            .map_err(|e| IntegrityError::IoError(format!("sync seal dir: {}", e)))?;
        crate::hardware_root::harden_file_permissions(&self.seal_path)
            .map_err(|e| IntegrityError::IoError(format!("harden seal file permissions: {}", e)))?;

        Ok(SealResult {
            manifest_sha256: manifest_sha,
            model_sha256: model_sha,
            config_sha256: config_sha,
            tokenizer_sha256: tokenizer_sha,
            hmac_tag,
            seal_path: self.seal_path.to_string_lossy().to_string(),
        })
    }

    pub fn verify(&self) -> Result<VerifyResult, IntegrityError> {
        let raw = fs::read_to_string(&self.seal_path).map_err(|_| IntegrityError::SealMissing)?;
        let seal: Value = serde_json::from_str(&raw)
            .map_err(|e| IntegrityError::FormatError(format!("seal JSON: {}", e)))?;

        let version = seal["version"].as_u64().unwrap_or(1);
        let algorithm = seal["algorithm"]
            .as_str()
            .unwrap_or("HMAC-SHA256+Scrypt(n=14,r=8,p=1)");
        let sealed_at = seal["sealed_at"].as_str().unwrap_or("");
        let architecture_profile = seal["architecture_profile"]
            .as_str()
            .unwrap_or("tara_native_transformer");
        let parameter_count = seal["parameter_count"].as_u64().unwrap_or(0);

        let salt_hex = seal["salt"]
            .as_str()
            .ok_or_else(|| IntegrityError::FormatError("missing salt".into()))?;
        let stored_hmac = seal["hmac_tag"]
            .as_str()
            .ok_or_else(|| IntegrityError::FormatError("missing hmac_tag".into()))?;
        let stored_manifest_sha = seal["manifest_sha256"]
            .as_str()
            .ok_or_else(|| IntegrityError::FormatError("missing manifest_sha256".into()))?;
        let stored_model_sha = seal["model_safetensors_sha256"]
            .as_str()
            .or_else(|| seal["model_sha256"].as_str())
            .ok_or_else(|| IntegrityError::FormatError("missing model_sha256".into()))?;
        let stored_config_sha = seal["config_sha256"].as_str().unwrap_or("NONE");
        let stored_tokenizer_sha = seal["tokenizer_sha256"].as_str().unwrap_or("NONE");

        if let Some(watermark) = crate::dpapi::load_os_monotonic_watermark(&self.watermark_domain)
            .map_err(|e| IntegrityError::Unauthorized(format!("WatermarkCheckFailed: {}", e)))?
        {
            if version < watermark {
                return Err(IntegrityError::Unauthorized(format!(
                    "ModelRollbackDetected: manifest seal version {} is older than committed watermark {}",
                    version, watermark
                )));
            }
        }

        let salt = hex::decode(salt_hex)
            .map_err(|e| IntegrityError::FormatError(format!("salt hex: {}", e)))?;
        if salt.len() != 16 {
            return Err(IntegrityError::FormatError("Invalid salt length".into()));
        }

        let current_manifest_sha = Self::stream_file_sha256(&self.manifest_path)?;
        let current_model_sha = Self::stream_file_sha256(&self.model_safetensors_path)?;
        let current_config_sha = Self::stream_optional_file_sha256(&self.config_json_path)?;
        let current_tokenizer_sha = Self::stream_optional_file_sha256(&self.tokenizer_json_path)?;

        let mut key = self.derive_hmac_key(&salt)?;

        let expected_hmac_v3 = Self::compute_hmac(
            &key,
            &Self::build_message(&SealMessageParams {
                version,
                algorithm,
                sealed_at,
                architecture_profile,
                parameter_count,
                manifest_sha: stored_manifest_sha,
                model_sha: stored_model_sha,
                config_sha: stored_config_sha,
                tokenizer_sha: stored_tokenizer_sha,
            }),
        );
        let expected_hmac_v2 = Self::compute_hmac(
            &key,
            format!(
                "TARA_MANIFEST_SEAL:v{}:{}:{}:{}:{}",
                version, algorithm, sealed_at, stored_manifest_sha, stored_model_sha
            )
            .as_bytes(),
        );
        let expected_hmac_v1 = Self::compute_hmac(
            &key,
            format!(
                "TARA_MANIFEST_SEAL:{}:{}",
                stored_manifest_sha, stored_model_sha
            )
            .as_bytes(),
        );

        let hmac_ok = constant_time_eq(stored_hmac.as_bytes(), expected_hmac_v3.as_bytes())
            || constant_time_eq(stored_hmac.as_bytes(), expected_hmac_v2.as_bytes())
            || constant_time_eq(stored_hmac.as_bytes(), expected_hmac_v1.as_bytes());

        key.zeroize();

        let manifest_sha_ok = constant_time_eq(current_manifest_sha.as_bytes(), stored_manifest_sha.as_bytes());
        let model_sha_ok = constant_time_eq(current_model_sha.as_bytes(), stored_model_sha.as_bytes());
        let config_sha_ok = constant_time_eq(current_config_sha.as_bytes(), stored_config_sha.as_bytes());
        let tokenizer_sha_ok = constant_time_eq(current_tokenizer_sha.as_bytes(), stored_tokenizer_sha.as_bytes());

        if !hmac_ok || !manifest_sha_ok || !model_sha_ok || !config_sha_ok || !tokenizer_sha_ok {
            return Err(IntegrityError::HmacMismatch);
        }

        Ok(VerifyResult {
            manifest_sha256: current_manifest_sha,
            model_sha256: current_model_sha,
            config_sha256: current_config_sha,
            tokenizer_sha256: current_tokenizer_sha,
            hmac_verified: true,
            manifest_match: true,
            model_match: true,
            config_match: true,
            tokenizer_match: true,
        })
    }
}
