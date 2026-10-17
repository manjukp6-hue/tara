//! Extensible Identity Verifier module.
//!
//! Rule compliance:
//! - Strictly NO caller-supplied boolean flags (e.g., `is_creator: bool` or `verified: bool`).
//! - VerifiedPrincipal is manufactured strictly upon cryptographic proof validation.
//! - Tokens must have valid cryptographic signatures verified against trusted issuer keys.
//! - Tokens CANNOT manufacture Creator authority from claims.
//! - Devices must be enrolled in the device registry before their signatures are trusted.

use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use rsa::signature::Verifier;
use rsa::{pkcs1v15::VerifyingKey, BigUint, RsaPublicKey};
use sha2::Sha256;

use crate::cryptography::{
    AlgorithmId, CryptoUtils, Ed25519Verifier, Rs256Verifier, SignatureVerifier,
};
use crate::primitives::{
    AssuranceLevel, AuthorityGeneration, DeviceId, PrincipalId, PrincipalRole, VerifiedPrincipal,
};

/// Record of an enrolled device within the trusted device registry.
#[derive(Debug, Clone)]
pub struct DeviceEnrollmentRecord {
    pub device_id: DeviceId,
    pub enrolled_public_key: Vec<u8>,
    pub bound_principal_id: PrincipalId,
    pub is_revoked: bool,
}

/// Registry of trusted devices with their verified hardware public keys.
#[derive(Debug, Clone)]
pub struct DeviceEnrollmentRegistry {
    enrolled_devices: Arc<RwLock<HashMap<String, DeviceEnrollmentRecord>>>,
}

impl Default for DeviceEnrollmentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceEnrollmentRegistry {
    pub fn new() -> Self {
        Self {
            enrolled_devices: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Authoritatively synchronize from persistent disk-backed IdentityManager.
    pub fn from_identity_manager(id_mgr: &crate::devices::IdentityManager) -> Result<Self, String> {
        let registry = Self::new();
        for dev in id_mgr.list_devices() {
            if let (Ok(device_id), Ok(pub_bytes)) = (
                DeviceId::new(&dev.device_id),
                hex::decode(&dev.device_public_key),
            ) {
                let bound_principal = PrincipalId::new(&dev.creator_id)
                    .unwrap_or_else(|_| PrincipalId::new("creator_root").unwrap());
                let is_revoked = dev.status.to_uppercase() == "REVOKED";
                let record = DeviceEnrollmentRecord {
                    device_id,
                    enrolled_public_key: pub_bytes,
                    bound_principal_id: bound_principal,
                    is_revoked,
                };
                let _ = registry.enroll_device(record);
            }
        }
        Ok(registry)
    }

    pub fn enroll_device(&self, record: DeviceEnrollmentRecord) -> Result<(), String> {
        let mut map = self
            .enrolled_devices
            .write()
            .map_err(|_| "Device registry lock poisoned (FailClosed)")?;
        map.insert(record.device_id.as_str().to_string(), record);
        Ok(())
    }

    pub fn revoke_device(&self, device_id: &DeviceId) -> Result<(), String> {
        let mut map = self
            .enrolled_devices
            .write()
            .map_err(|_| "Device registry lock poisoned (FailClosed)")?;
        if let Some(record) = map.get_mut(device_id.as_str()) {
            record.is_revoked = true;
        }
        Ok(())
    }

    pub fn get_device(
        &self,
        device_id: &DeviceId,
    ) -> Result<Option<DeviceEnrollmentRecord>, String> {
        let map = self
            .enrolled_devices
            .read()
            .map_err(|_| "Device registry lock poisoned (FailClosed)")?;
        Ok(map.get(device_id.as_str()).cloned())
    }
}

/// Policy for a trusted OIDC/OAuth identity token issuer.
#[derive(Debug, Clone)]
pub struct TrustedIssuerPolicy {
    pub issuer_url: String,
    pub allowed_algorithms: HashSet<AlgorithmId>,
    pub public_keys: HashMap<String, (AlgorithmId, Vec<u8>)>, // kid -> (alg, public_key_der_or_bytes)
}

/// Registry of configured trusted external identity token issuers.
#[derive(Debug, Clone)]
pub struct TrustedIssuerRegistry {
    issuers: Arc<RwLock<HashMap<String, TrustedIssuerPolicy>>>,
}

impl Default for TrustedIssuerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl TrustedIssuerRegistry {
    pub fn new() -> Self {
        Self {
            issuers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn register_issuer(&self, policy: TrustedIssuerPolicy) -> Result<(), String> {
        let mut map = self
            .issuers
            .write()
            .map_err(|_| "Issuer registry lock poisoned (FailClosed)")?;
        map.insert(policy.issuer_url.clone(), policy);
        Ok(())
    }

    pub fn get_issuer(&self, issuer_url: &str) -> Result<Option<TrustedIssuerPolicy>, String> {
        let map = self
            .issuers
            .read()
            .map_err(|_| "Issuer registry lock poisoned (FailClosed)")?;
        Ok(map.get(issuer_url).cloned())
    }
}

/// An external identity proof submitted by a caller.
#[derive(Debug, Clone)]
pub enum IdentityProof {
    /// OIDC / OAuth JWT ID Token with claims and signature.
    IdToken {
        token: String,
        expected_audience: String,
    },
    /// Google OpenID Connect ID token with optional single-use server nonce.
    GoogleIdToken {
        token: String,
        expected_nonce: Option<String>,
    },
    /// Cryptographic challenge-response signed by an enrolled device key.
    DeviceSignature {
        device_id: DeviceId,
        challenge_nonce: String,
        signature: Vec<u8>,
    },
    /// Multi-factor root passkey assertion.
    PasskeyAssertion {
        credential_id: Vec<u8>,
        authenticator_data: Vec<u8>,
        client_data_json: Vec<u8>,
        signature: Vec<u8>,
    },
    /// Custom extensible proof payload.
    Custom { proof_type: String, payload: Value },
}

/// Trait for pluggable identity verifiers.
pub trait IdentityVerifier: Send + Sync {
    fn proof_type(&self) -> &'static str;

    fn verify_proof(
        &self,
        proof: &IdentityProof,
        current_generation: AuthorityGeneration,
        current_epoch_secs: u64,
    ) -> Result<VerifiedPrincipal, String>;

    fn create_challenge_nonce(&self, _ttl_secs: u64) -> Result<String, String> {
        Err("Challenge creation not supported for this verifier".into())
    }
}

/// Registry of identity verifiers.
pub struct IdentityVerifierRegistry {
    verifiers: Arc<RwLock<HashMap<String, Box<dyn IdentityVerifier>>>>,
}

impl Default for IdentityVerifierRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl IdentityVerifierRegistry {
    pub fn new() -> Self {
        let registry = Self {
            verifiers: Arc::new(RwLock::new(HashMap::new())),
        };
        // Register authentic CentralizedPasskeyVerifier by default
        let _ = registry.register_verifier(Box::new(
            crate::hardware_root::CentralizedPasskeyVerifier::new(),
        ));
        registry
    }

    pub fn register_verifier(&self, verifier: Box<dyn IdentityVerifier>) -> Result<(), String> {
        let mut map = self
            .verifiers
            .write()
            .map_err(|_| "Verifier registry lock poisoned (FailClosed)")?;
        map.insert(verifier.proof_type().to_string(), verifier);
        Ok(())
    }

    /// Issue a single-use challenge nonce for the specified verifier proof type.
    pub fn create_challenge_nonce(
        &self,
        proof_type: &str,
        ttl_secs: u64,
    ) -> Result<String, String> {
        let map = self
            .verifiers
            .read()
            .map_err(|_| "Verifier registry lock poisoned (FailClosed)")?;
        let verifier = map
            .get(proof_type)
            .ok_or_else(|| format!("No verifier registered for proof type: '{}'", proof_type))?;
        verifier.create_challenge_nonce(ttl_secs)
    }

    /// Verifies external proof and constructs a VerifiedPrincipal.
    pub fn verify(
        &self,
        proof: &IdentityProof,
        current_generation: AuthorityGeneration,
        current_epoch_secs: u64,
    ) -> Result<VerifiedPrincipal, String> {
        let proof_type_key = match proof {
            IdentityProof::IdToken { .. } => "id_token",
            IdentityProof::GoogleIdToken { .. } => "google_id_token",
            IdentityProof::DeviceSignature { .. } => "device_signature",
            IdentityProof::PasskeyAssertion { .. } => "passkey_assertion",
            IdentityProof::Custom { proof_type, .. } => proof_type.as_str(),
        };

        let map = self
            .verifiers
            .read()
            .map_err(|_| "Verifier registry lock poisoned (FailClosed)")?;
        let verifier = map.get(proof_type_key).ok_or_else(|| {
            format!(
                "No verifier registered for proof type: '{}'",
                proof_type_key
            )
        })?;

        verifier.verify_proof(proof, current_generation, current_epoch_secs)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Built-in Identity Verifiers
// ─────────────────────────────────────────────────────────────────────────────

/// Device signature verifier anchored to the device enrollment registry.
pub struct DeviceSignatureVerifier {
    enrollment_registry: Arc<DeviceEnrollmentRegistry>,
    challenge_nonces: Arc<RwLock<HashMap<String, u64>>>,
}

impl DeviceSignatureVerifier {
    pub fn new(enrollment_registry: Arc<DeviceEnrollmentRegistry>) -> Self {
        Self {
            enrollment_registry,
            challenge_nonces: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Issue a single-use cryptographically random challenge nonce for device verification.
    pub fn create_challenge_nonce(&self, ttl_secs: u64) -> Result<String, String> {
        let mut nonce_bytes = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut nonce_bytes);
        let nonce = hex::encode(nonce_bytes);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let exp = now + ttl_secs.clamp(60, 600);
        let mut map = self
            .challenge_nonces
            .write()
            .map_err(|_| "Nonce cache lock poisoned")?;
        map.retain(|_, expiry| *expiry > now);
        if map.len() >= 5000 {
            // Anti-saturation: evict only the earliest-expiring entry (never clear entire cache)
            if let Some(oldest_key) = map
                .iter()
                .min_by_key(|(_, expiry)| *expiry)
                .map(|(k, _)| k.clone())
            {
                map.remove(&oldest_key);
            }
        }
        map.insert(nonce.clone(), exp);
        Ok(nonce)
    }

    /// Check and consume a challenge nonce (single-use anti-replay defense).
    pub fn consume_challenge_nonce(
        &self,
        nonce: &str,
        current_epoch_secs: u64,
    ) -> Result<(), String> {
        let mut map = self
            .challenge_nonces
            .write()
            .map_err(|_| "Nonce cache lock poisoned")?;
        map.retain(|_, exp| *exp > current_epoch_secs);
        match map.remove(nonce) {
            Some(exp) if exp >= current_epoch_secs => Ok(()),
            Some(_) => Err("DeviceChallengeExpired: The challenge nonce has expired".into()),
            None => Err("DeviceChallengeReplayedOrInvalid: Challenge nonce is invalid, expired, or already consumed".into()),
        }
    }
}

impl IdentityVerifier for DeviceSignatureVerifier {
    fn proof_type(&self) -> &'static str {
        "device_signature"
    }

    fn create_challenge_nonce(&self, ttl_secs: u64) -> Result<String, String> {
        self.create_challenge_nonce(ttl_secs)
    }

    fn verify_proof(
        &self,
        proof: &IdentityProof,
        current_generation: AuthorityGeneration,
        current_epoch_secs: u64,
    ) -> Result<VerifiedPrincipal, String> {
        let (device_id, nonce, signature) = match proof {
            IdentityProof::DeviceSignature {
                device_id,
                challenge_nonce,
                signature,
            } => (device_id, challenge_nonce, signature),
            _ => return Err("Invalid proof variant for DeviceSignatureVerifier".into()),
        };

        // 1. Single-use challenge nonce replay protection
        self.consume_challenge_nonce(nonce, current_epoch_secs)?;

        // 2. Look up device in the enrollment registry (anchored trust, ID-07)
        let record = self
            .enrollment_registry
            .get_device(device_id)?
            .ok_or_else(|| {
                format!(
                    "Device '{}' is not registered in enrollment registry",
                    device_id.as_str()
                )
            })?;

        // 2. Reject revoked devices
        if record.is_revoked {
            return Err(format!(
                "DeviceRevoked: Device '{}' has been revoked",
                device_id.as_str()
            ));
        }

        // 3. Verify signature using the registered, enrolled public key
        if record.enrolled_public_key.len() != 32 {
            return Err(
                "Invalid enrolled public key length for Ed25519 device verification".into(),
            );
        }

        let verifier = Ed25519Verifier;
        verifier.verify(&record.enrolled_public_key, nonce.as_bytes(), signature)?;

        let principal_id = record.bound_principal_id;
        let role = PrincipalRole::operator();
        let assurance = AssuranceLevel::HARDWARE_ATTESTED;

        let mut claims = HashMap::new();
        claims.insert(
            "device_id".into(),
            Value::String(device_id.as_str().to_string()),
        );
        claims.insert(
            "verified_via".into(),
            Value::String("Enrolled_Ed25519_Device_Challenge".into()),
        );

        VerifiedPrincipal::construct_verified(
            principal_id,
            role,
            assurance,
            Some(device_id.clone()),
            current_generation,
            current_epoch_secs,
            claims,
        )
    }
}

/// Cryptographically verified Token Identity Verifier.
pub struct TokenIdentityVerifier {
    issuer_registry: Arc<TrustedIssuerRegistry>,
}

impl TokenIdentityVerifier {
    pub fn new(issuer_registry: Arc<TrustedIssuerRegistry>) -> Self {
        Self { issuer_registry }
    }
}

impl IdentityVerifier for TokenIdentityVerifier {
    fn proof_type(&self) -> &'static str {
        "id_token"
    }

    fn verify_proof(
        &self,
        proof: &IdentityProof,
        current_generation: AuthorityGeneration,
        current_epoch_secs: u64,
    ) -> Result<VerifiedPrincipal, String> {
        let (token, expected_audience) = match proof {
            IdentityProof::IdToken {
                token,
                expected_audience,
            } => (token, expected_audience),
            _ => return Err("Invalid proof variant for TokenIdentityVerifier".into()),
        };

        if expected_audience.trim().is_empty() {
            return Err("EmptyExpectedAudience: Identity proof verification requires non-empty expected audience (FailClosed)".into());
        }

        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return Err("Malformed JWT token structure: expected 3 dot-separated parts".into());
        }

        // 1. Decode Header
        let header_bytes =
            base64::Engine::decode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, parts[0])
                .or_else(|_| {
                    base64::Engine::decode(
                        &base64::engine::general_purpose::STANDARD_NO_PAD,
                        parts[0],
                    )
                })
                .map_err(|e| format!("Failed to base64 decode JWT header: {e}"))?;

        let header_value: Value = serde_json::from_slice(&header_bytes)
            .map_err(|e| format!("Failed to parse JWT header JSON: {e}"))?;

        let alg_str = header_value
            .get("alg")
            .and_then(|v| v.as_str())
            .ok_or("JWT header missing 'alg' parameter")?;

        let kid = header_value
            .get("kid")
            .and_then(|v| v.as_str())
            .unwrap_or("default");

        // 2. Decode Payload
        let payload_bytes =
            base64::Engine::decode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, parts[1])
                .or_else(|_| {
                    base64::Engine::decode(
                        &base64::engine::general_purpose::STANDARD_NO_PAD,
                        parts[1],
                    )
                })
                .map_err(|e| format!("Failed to base64 decode JWT payload: {e}"))?;

        let claims_value: Value = serde_json::from_slice(&payload_bytes)
            .map_err(|e| format!("Failed to parse JWT claims JSON: {e}"))?;

        // 3. Decode Signature
        let signature_bytes =
            base64::Engine::decode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, parts[2])
                .or_else(|_| {
                    base64::Engine::decode(
                        &base64::engine::general_purpose::STANDARD_NO_PAD,
                        parts[2],
                    )
                })
                .map_err(|e| format!("Failed to base64 decode JWT signature: {e}"))?;

        // 4. Validate Issuer against configured TrustedIssuerRegistry (ID-01, ID-19)
        let iss = claims_value
            .get("iss")
            .and_then(|v| v.as_str())
            .ok_or("JWT token missing 'iss' claim")?;

        let issuer_policy = self.issuer_registry.get_issuer(iss)?.ok_or_else(|| {
            format!(
                "UntrustedIssuer: Token issuer '{}' is not registered as trusted",
                iss
            )
        })?;

        // 5. Verify algorithm is allowlisted by issuer policy
        let token_alg = match alg_str {
            "EdDSA" => AlgorithmId::Ed25519,
            "RS256" => AlgorithmId::Rs256,
            other => {
                return Err(format!(
                    "Unsupported or unallowlisted JWT algorithm: {}",
                    other
                ))
            }
        };

        if !issuer_policy.allowed_algorithms.contains(&token_alg) {
            return Err(format!(
                "AlgorithmDisallowed: Algorithm '{}' not permitted for issuer '{}'",
                alg_str, iss
            ));
        }

        // 6. Look up trusted public key for (issuer, kid)
        let (key_alg, public_key_bytes) = issuer_policy.public_keys.get(kid).ok_or_else(|| {
            format!(
                "UnknownKeyId: Key ID '{}' not found for issuer '{}'",
                kid, iss
            )
        })?;

        if *key_alg != token_alg {
            return Err(
                "AlgorithmMismatch: Key algorithm does not match token header algorithm".into(),
            );
        }

        // 7. Cryptographically verify signature over `${parts[0]}.${parts[1]}` (ID-01)
        let signed_message = format!("{}.{}", parts[0], parts[1]);
        match token_alg {
            AlgorithmId::Ed25519 => {
                let verifier = Ed25519Verifier;
                verifier
                    .verify(
                        public_key_bytes,
                        signed_message.as_bytes(),
                        &signature_bytes,
                    )
                    .map_err(|e| format!("JwtSignatureVerificationFailed: {e}"))?;
            }
            AlgorithmId::Rs256 => {
                let verifier = Rs256Verifier;
                verifier
                    .verify(
                        public_key_bytes,
                        signed_message.as_bytes(),
                        &signature_bytes,
                    )
                    .map_err(|e| format!("JwtSignatureVerificationFailed: {e}"))?;
            }
            _ => return Err("Unsupported signature verification algorithm".into()),
        }

        // 8. Validate Expiration (exp) with bounded clock skew tolerance
        let exp = claims_value
            .get("exp")
            .and_then(|v| v.as_u64())
            .ok_or("JWT token missing 'exp' claim")?;

        if exp <= current_epoch_secs {
            return Err(format!(
                "JwtTokenExpired: Token expired at {} (current: {})",
                exp, current_epoch_secs
            ));
        }

        // 9. Validate Not-Before (nbf) if present
        if let Some(nbf) = claims_value.get("nbf").and_then(|v| v.as_u64()) {
            if current_epoch_secs < nbf {
                return Err(format!(
                    "JwtTokenNotYetValid: Token not valid before {} (current: {})",
                    nbf, current_epoch_secs
                ));
            }
        }

        // 10. Validate Audience (aud) supporting string or array (ID-19)
        let aud_match = if let Some(aud_str) = claims_value.get("aud").and_then(|v| v.as_str()) {
            CryptoUtils::constant_time_eq(aud_str.as_bytes(), expected_audience.as_bytes())
        } else if let Some(aud_array) = claims_value.get("aud").and_then(|v| v.as_array()) {
            aud_array.iter().any(|item| {
                item.as_str().is_some_and(|s| {
                    CryptoUtils::constant_time_eq(s.as_bytes(), expected_audience.as_bytes())
                })
            })
        } else {
            return Err("JWT token missing or invalid 'aud' claim".into());
        };

        if !aud_match {
            return Err("JwtAudienceMismatch: Token 'aud' does not match expected audience".into());
        }

        // 11. Extract Subject (sub) - stable identity anchor (ID-19)
        let sub = claims_value
            .get("sub")
            .and_then(|v| v.as_str())
            .ok_or("JWT token missing 'sub' claim")?;

        // 12. Enforce Role: External tokens CANNOT manufacture Creator privileges (ID-02)
        // Tokens can only grant standard User or Operator roles, NEVER CreatorRoot.
        let role = PrincipalRole::user();
        let assurance = AssuranceLevel::STANDARD_TOKEN;
        // ID-19: Stable principal identity anchored to (iss, sub) with bounded namespace encoding
        let safe_namespace = if iss
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            && !iss.is_empty()
            && iss.len() <= 64
        {
            iss.to_string()
        } else {
            let digest = hex::encode(&CryptoUtils::sha256(iss.as_bytes())[..8]);
            format!("iss_{}", digest)
        };
        let principal_id = PrincipalId::with_namespace(&safe_namespace, sub)?;

        let mut claims_map = HashMap::new();
        if let Some(obj) = claims_value.as_object() {
            for (k, v) in obj {
                claims_map.insert(k.clone(), v.clone());
            }
        }

        VerifiedPrincipal::construct_verified(
            principal_id,
            role,
            assurance,
            None,
            current_generation,
            current_epoch_secs,
            claims_map,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Authentic Google OAuth OpenID Connect Service & Verifier
// ─────────────────────────────────────────────────────────────────────────────

pub const GOOGLE_ISSUERS: &[&str] = &["accounts.google.com", "https://accounts.google.com"];
pub const GOOGLE_JWKS_URL: &str = "https://www.googleapis.com/oauth2/v3/certs";
pub const MAX_NONCE_STORE_SIZE: usize = 1000;
pub const MAX_JWKS_RESPONSE_BYTES: usize = 1048576; // 1 MiB

fn decode_b64url(s: &str) -> Result<Vec<u8>, String> {
    let clean = s.trim().trim_end_matches('=');
    URL_SAFE_NO_PAD
        .decode(clean.as_bytes())
        .map_err(|e| format!("Base64URL decode failed: {}", e))
}

/// Cryptographically validates Google OpenID Connect ID tokens (JWT with RS256 signatures).
pub struct GoogleAuthService {
    authorized_email: Option<String>,
    expected_client_id: Option<String>,
    key_cache: Mutex<HashMap<String, (RsaPublicKey, f64)>>,
    last_key_fetch: Mutex<f64>,
    pending_nonces: Mutex<HashMap<String, (f64, bool)>>,
}

impl GoogleAuthService {
    pub fn new(authorized_email: Option<&str>, expected_client_id: Option<&str>) -> Self {
        Self {
            authorized_email: authorized_email.map(|e| e.trim().to_lowercase()),
            expected_client_id: expected_client_id.map(|c| c.trim().to_string()),
            key_cache: Mutex::new(HashMap::new()),
            last_key_fetch: Mutex::new(0.0),
            pending_nonces: Mutex::new(HashMap::new()),
        }
    }

    pub fn now_seconds() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    }

    pub fn register_jwk(&self, kid: &str, n_b64: &str, e_b64: &str) -> Result<(), String> {
        let n_bytes = decode_b64url(n_b64)?;
        let e_bytes = decode_b64url(e_b64)?;
        let n = BigUint::from_bytes_be(&n_bytes);
        let e = BigUint::from_bytes_be(&e_bytes);
        let key = RsaPublicKey::new(n, e)
            .map_err(|err| format!("Invalid RSA key components: {:?}", err))?;
        self.key_cache
            .lock()
            .map_err(|_| "Key cache poisoned")?
            .insert(kid.to_string(), (key, Self::now_seconds()));
        Ok(())
    }

    pub fn register_rsa_public_key(&self, kid: &str, key: RsaPublicKey) -> Result<(), String> {
        self.key_cache
            .lock()
            .map_err(|_| "Key cache poisoned")?
            .insert(kid.to_string(), (key, Self::now_seconds()));
        Ok(())
    }

    /// Creates an authoritative server-issued nonce with bounded TTL and capacity.
    /// Strictly rejects new nonces if capacity is saturated with unexpired entries (fixes finding #2).
    pub fn create_auth_nonce(&self, ttl_seconds: f64) -> Result<String, String> {
        let safe_ttl = if ttl_seconds <= 0.0 || !ttl_seconds.is_finite() || ttl_seconds > 300.0 {
            300.0
        } else {
            ttl_seconds
        };

        let now = Self::now_seconds();
        let mut nonces = self
            .pending_nonces
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        // Purge expired nonces only
        nonces.retain(|_, (exp, _)| now <= *exp);

        if nonces.len() >= MAX_NONCE_STORE_SIZE {
            return Err("NonceCacheCapacityExhausted: Pending nonce store is full of unexpired nonces; rejected to preserve replay integrity".into());
        }

        let mut bytes = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut bytes);
        let nonce = URL_SAFE_NO_PAD.encode(bytes);

        nonces.insert(nonce.clone(), (now + safe_ttl, false));
        Ok(nonce)
    }

    pub fn fetch_google_jwks(&self) -> bool {
        let now = Self::now_seconds();
        {
            let last = *self
                .last_key_fetch
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let cache_empty = self
                .key_cache
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .is_empty();
            if !cache_empty && (now - last) < 60.0 {
                return false;
            }
        }

        let output = std::process::Command::new("curl.exe")
            .args(["-s", "-m", "5", GOOGLE_JWKS_URL])
            .output();

        let json_str = match output {
            Ok(out) if out.status.success() => {
                if out.stdout.len() > MAX_JWKS_RESPONSE_BYTES {
                    return false;
                }
                String::from_utf8_lossy(&out.stdout).to_string()
            }
            _ => return false,
        };

        if let Ok(val) = serde_json::from_str::<Value>(&json_str) {
            if let Some(keys_arr) = val.get("keys").and_then(|k| k.as_array()) {
                let mut cache = self.key_cache.lock().unwrap_or_else(|p| p.into_inner());
                cache.retain(|_, (_, ts)| (now - *ts) < 86400.0);

                for k in keys_arr {
                    if k.get("kty").and_then(|v| v.as_str()) == Some("RSA")
                        && k.get("alg").and_then(|v| v.as_str()) == Some("RS256")
                    {
                        if let (Some(kid), Some(n_b64), Some(e_b64)) = (
                            k.get("kid").and_then(|v| v.as_str()),
                            k.get("n").and_then(|v| v.as_str()),
                            k.get("e").and_then(|v| v.as_str()),
                        ) {
                            if let (Ok(n_bytes), Ok(e_bytes)) =
                                (decode_b64url(n_b64), decode_b64url(e_b64))
                            {
                                let n = BigUint::from_bytes_be(&n_bytes);
                                let e = BigUint::from_bytes_be(&e_bytes);
                                if let Ok(rsa_key) = RsaPublicKey::new(n, e) {
                                    cache.insert(kid.to_string(), (rsa_key, now));
                                }
                            }
                        }
                    }
                }
                *self
                    .last_key_fetch
                    .lock()
                    .unwrap_or_else(|p| p.into_inner()) = now;
                return true;
            }
        }
        false
    }

    /// Cryptographically validates a Google OpenID Connect ID token (RS256 JWT).
    pub fn verify_id_token(
        &self,
        id_token: &str,
        expected_nonce: Option<&str>,
        require_nonce: bool,
    ) -> Result<Value, String> {
        let parts: Vec<&str> = id_token.split('.').collect();
        if parts.len() != 3 {
            return Err("Malformed JWT: expected exactly 3 dot-separated segments".to_string());
        }

        let header_bytes = decode_b64url(parts[0])?;
        let payload_bytes = decode_b64url(parts[1])?;
        let signature_bytes = decode_b64url(parts[2])?;

        let header: Value = serde_json::from_slice(&header_bytes)
            .map_err(|e| format!("Invalid JWT header JSON: {}", e))?;
        let payload: Value = serde_json::from_slice(&payload_bytes)
            .map_err(|e| format!("Invalid JWT payload JSON: {}", e))?;

        // 1. Verify header alg == "RS256"
        let alg = header.get("alg").and_then(|v| v.as_str()).unwrap_or("");
        if alg != "RS256" {
            return Err(format!("Unsupported algorithm '{}', expected RS256", alg));
        }

        let kid = header
            .get("kid")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing 'kid' in JWT header".to_string())?;

        // 2. Obtain RSA Public Key for kid
        let rsa_key = {
            let cache = self.key_cache.lock().unwrap_or_else(|p| p.into_inner());
            cache.get(kid).map(|(k, _)| k.clone())
        };

        let rsa_key = match rsa_key {
            Some(k) => k,
            None => {
                self.fetch_google_jwks();
                let cache = self.key_cache.lock().unwrap_or_else(|p| p.into_inner());
                cache
                    .get(kid)
                    .map(|(k, _)| k.clone())
                    .ok_or_else(|| format!("Unknown kid '{}' in JWKS and local cache", kid))?
            }
        };

        // 3. Cryptographically verify RS256 signature
        let signing_input = format!("{}.{}", parts[0], parts[1]);
        let verifying_key = VerifyingKey::<Sha256>::new(rsa_key);
        let rsa_sig = rsa::pkcs1v15::Signature::try_from(signature_bytes.as_slice())
            .map_err(|e| format!("Invalid RSA signature structure: {:?}", e))?;

        verifying_key
            .verify(signing_input.as_bytes(), &rsa_sig)
            .map_err(|_| {
                "Cryptographic RS256 signature verification failed against Google JWKS".to_string()
            })?;

        // 4. Verify Issuer
        let iss = payload.get("iss").and_then(|v| v.as_str()).unwrap_or("");
        if !GOOGLE_ISSUERS.contains(&iss) {
            return Err(format!("Invalid issuer '{}', expected Google issuer", iss));
        }

        // 5. Verify Audience: fail closed against Confused Deputy token substitution
        let aud = payload.get("aud").and_then(|v| v.as_str()).unwrap_or("");
        let env_aud = std::env::var("TARA_GOOGLE_CLIENT_ID").ok();
        let expected_aud = self.expected_client_id.as_deref().or(env_aud.as_deref());
        match expected_aud {
            Some(exp_aud) => {
                if aud != exp_aud {
                    return Err(format!(
                        "Audience mismatch: expected '{}', got '{}'",
                        exp_aud, aud
                    ));
                }
            }
            None => {
                return Err("MissingExpectedClientId: Audience validation requires a configured expected_client_id or TARA_GOOGLE_CLIENT_ID to prevent Confused Deputy attacks (FailClosed)".to_string());
            }
        }

        let now = Self::now_seconds();
        let clock_skew = 10.0;

        // 6. Verify Expiry
        let exp = payload
            .get("exp")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| "Missing 'exp' timestamp".to_string())?;
        if now > (exp + clock_skew) {
            return Err(format!("Token expired: exp={}, now={}", exp, now));
        }

        // 7. Verify Issued-At
        let iat = payload
            .get("iat")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| "Missing 'iat' timestamp".to_string())?;
        if iat > (now + clock_skew) {
            return Err(format!(
                "Token issued in the future: iat={}, now={}",
                iat, now
            ));
        }

        // 8. Strict boolean email_verified validation
        let email_verified = match payload.get("email_verified") {
            Some(Value::Bool(b)) => *b,
            _ => false,
        };
        if !email_verified {
            return Err(
                "Google account email is not verified (email_verified != true)".to_string(),
            );
        }

        // 9. Verify non-empty subject
        let sub = payload.get("sub").and_then(|v| v.as_str()).unwrap_or("");
        if sub.trim().is_empty() {
            return Err("Missing or empty 'sub' subject claim in ID token".to_string());
        }

        // 10. Server-side Nonce Anti-Replay Validation
        let token_nonce = payload.get("nonce").and_then(|v| v.as_str());
        if require_nonce && token_nonce.is_none() {
            return Err("Missing required nonce in ID token".to_string());
        }

        if let Some(exp_n) = expected_nonce {
            let tn = token_nonce
                .ok_or_else(|| "Expected nonce was not provided in token".to_string())?;
            if tn != exp_n {
                return Err(format!(
                    "Nonce mismatch: expected '{}', got '{}'",
                    exp_n, tn
                ));
            }

            let mut nonces = self
                .pending_nonces
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let entry = nonces
                .get_mut(exp_n)
                .ok_or_else(|| "Nonce is not registered in server nonce store".to_string())?;

            if entry.1 {
                return Err("Nonce has already been consumed (replay attack detected)".to_string());
            }
            if now > entry.0 {
                return Err("Server nonce has expired".to_string());
            }
            // Mark consumed strictly
            entry.1 = true;
        }

        // 11. Verify Email matches authorized email if set
        if let Some(ref auth_email) = self.authorized_email {
            let email = payload
                .get("email")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_lowercase();
            if &email != auth_email {
                return Err(format!(
                    "Email mismatch: token is for '{}', authorized creator is '{}'",
                    email, auth_email
                ));
            }
        }

        Ok(payload)
    }
}

/// Pluggable Google OAuth ID Token Identity Verifier.
pub struct GoogleOAuthVerifier {
    auth_service: Arc<GoogleAuthService>,
}

impl GoogleOAuthVerifier {
    pub fn new(auth_service: Arc<GoogleAuthService>) -> Self {
        Self { auth_service }
    }
}

impl IdentityVerifier for GoogleOAuthVerifier {
    fn proof_type(&self) -> &'static str {
        "google_id_token"
    }

    fn verify_proof(
        &self,
        proof: &IdentityProof,
        current_generation: AuthorityGeneration,
        current_epoch_secs: u64,
    ) -> Result<VerifiedPrincipal, String> {
        let (token, expected_nonce) = match proof {
            IdentityProof::GoogleIdToken {
                token,
                expected_nonce,
            } => (token, expected_nonce),
            _ => return Err("Invalid proof variant for GoogleOAuthVerifier".into()),
        };

        let nonce_str = expected_nonce
            .as_deref()
            .ok_or("Replay protection violation: expected_nonce is mandatory for Google ID token verification")?;
        let claims = self
            .auth_service
            .verify_id_token(token, Some(nonce_str), true)?;

        let sub = claims
            .get("sub")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'sub' in Google ID token")?;
        let email = claims.get("email").and_then(|v| v.as_str()).unwrap_or(sub);

        let role = PrincipalRole::user();
        let assurance = AssuranceLevel::STANDARD_TOKEN;
        let principal_id = PrincipalId::with_namespace("google_oauth", email)?;

        let mut claims_map = HashMap::new();
        if let Some(obj) = claims.as_object() {
            for (k, v) in obj {
                claims_map.insert(k.clone(), v.clone());
            }
        }
        claims_map.insert("provider".to_string(), Value::String("google".to_string()));

        VerifiedPrincipal::construct_verified(
            principal_id,
            role,
            assurance,
            None,
            current_generation,
            current_epoch_secs,
            claims_map,
        )
    }
}
