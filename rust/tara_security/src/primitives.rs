//! Strongly-typed security primitives: Identifiers, Actions, Resources, Roles, and States.
//!
//! Dynamic Extensibility and Zero Hardcoded Limits:
//! - Extensible by design: No rigid closed enums or fixed capability ceilings.
//! - Dynamic namespaces, custom actions, extensible claims, and configurable assurance levels.
//! - No public caller can fabricate a VerifiedPrincipal by passing boolean flags.

use constant_time_eq::constant_time_eq;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

/// Opaque, bounded Principal identifier with dynamic namespace support.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PrincipalId {
    namespace: String,
    name: String,
}

impl PrincipalId {
    pub fn new(name: impl Into<String>) -> Result<Self, String> {
        Self::with_namespace("default", name)
    }

    pub fn with_namespace(namespace: impl Into<String>, name: impl Into<String>) -> Result<Self, String> {
        let ns = namespace.into().trim().to_lowercase();
        let nm = name.into().trim().to_string();

        if ns.is_empty() || ns.len() > 64 {
            return Err("PrincipalId namespace must be 1-64 characters".into());
        }
        if nm.is_empty() || nm.len() > 128 {
            return Err("PrincipalId name must be 1-128 characters".into());
        }
        if !ns.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
            return Err("PrincipalId namespace contains invalid characters".into());
        }
        if !nm.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '@' || c == '.') {
            return Err("PrincipalId name contains invalid characters".into());
        }

        Ok(Self { namespace: ns, name: nm })
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn as_str(&self) -> String {
        format!("{}:{}", self.namespace, self.name)
    }
}

impl fmt::Display for PrincipalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.namespace, self.name)
    }
}

/// Opaque, bounded Device identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceId(String);

impl DeviceId {
    pub fn new(id: impl Into<String>) -> Result<Self, String> {
        let s = id.into();
        let trimmed = s.trim();
        if trimmed.is_empty() || trimmed.len() > 128 {
            return Err("DeviceId must be between 1 and 128 characters".into());
        }
        if !trimmed.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == ':') {
            return Err("DeviceId contains invalid characters".into());
        }
        Ok(Self(trimmed.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Opaque, strongly typed Session ID.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(String);

impl SessionId {
    pub fn new(id: impl Into<String>) -> Result<Self, String> {
        let s = id.into();
        let normalized = s.trim().to_lowercase();
        if normalized.len() != 64 || !normalized.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err("SessionId must be a 64-character hex string".into());
        }
        Ok(Self(normalized))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Opaque, strongly typed Challenge ID.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChallengeId(String);

impl ChallengeId {
    pub fn new(id: impl Into<String>) -> Result<Self, String> {
        let s = id.into();
        let normalized = s.trim().to_lowercase();
        if normalized.len() != 64 || !normalized.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err("ChallengeId must be a 64-character hex string".into());
        }
        Ok(Self(normalized))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ChallengeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Dynamic, extensible Security Action.
/// Supports both canonical well-known actions and dynamic custom extensions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SecurityAction {
    namespace: String,
    verb: String,
}

impl SecurityAction {
    pub fn new(namespace: impl Into<String>, verb: impl Into<String>) -> Result<Self, String> {
        let ns = namespace.into().trim().to_lowercase();
        let vb = verb.into().trim().to_lowercase();
        if ns.is_empty() || ns.len() > 64 || !ns.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
            return Err("Invalid action namespace".into());
        }
        if vb.is_empty() || vb.len() > 64 || !vb.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
            return Err("Invalid action verb".into());
        }
        Ok(Self { namespace: ns, verb: vb })
    }

    // Canonical built-in actions
    pub fn execute_task() -> Self {
        Self { namespace: "core".into(), verb: "execute_task".into() }
    }

    pub fn rotate_authority_key() -> Self {
        Self { namespace: "authority".into(), verb: "rotate_key".into() }
    }

    pub fn configure_policy() -> Self {
        Self { namespace: "policy".into(), verb: "configure".into() }
    }

    pub fn read_protected_resource() -> Self {
        Self { namespace: "storage".into(), verb: "read".into() }
    }

    pub fn write_protected_resource() -> Self {
        Self { namespace: "storage".into(), verb: "write".into() }
    }

    pub fn lift_lockdown() -> Self {
        Self { namespace: "authority".into(), verb: "lift_lockdown".into() }
    }

    pub fn enroll_device() -> Self {
        Self { namespace: "device".into(), verb: "enroll".into() }
    }

    pub fn revoke_device() -> Self {
        Self { namespace: "device".into(), verb: "revoke".into() }
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn verb(&self) -> &str {
        &self.verb
    }

    pub fn canonical_name(&self) -> String {
        format!("{}:{}", self.namespace, self.verb)
    }
}

impl fmt::Display for SecurityAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.namespace, self.verb)
    }
}

/// Dynamic, extensible Security Resource.
/// Can represent vaults, files, model checkpoints, endpoints, or custom plugins.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SecurityResource {
    SystemVault,
    PolicyRegistry,
    AuditStore,
    DeviceRegistry,
    AuthorityRoot,
    FilesystemPath(PathBuf),
    NamedEntity { namespace: String, id: String },
}

impl SecurityResource {
    pub fn custom(namespace: impl Into<String>, id: impl Into<String>) -> Result<Self, String> {
        let ns = namespace.into().trim().to_lowercase();
        let i = id.into().trim().to_string();
        if ns.is_empty() || ns.len() > 64 {
            return Err("Resource namespace must be 1-64 characters".into());
        }
        if i.is_empty() || i.len() > 128 {
            return Err("Resource id must be 1-128 characters".into());
        }
        // Strict canonical grammar: Colons (':') are reserved delimiters in URNs and prohibited in components
        if ns.contains(':') || i.contains(':') {
            return Err("InvalidResourceIdentifier: Colons (':') are reserved delimiters and forbidden inside namespace or id".into());
        }
        if !ns.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
            return Err("Resource namespace contains invalid characters (allowed: [a-z0-9_-])".into());
        }
        if !i.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' || c == '/') {
            return Err("Resource id contains invalid characters (allowed: [a-zA-Z0-9_.-/])".into());
        }
        Ok(Self::NamedEntity { namespace: ns, id: i })
    }

    pub fn path(p: impl AsRef<Path>) -> Result<Self, String> {
        let path = p.as_ref();
        for component in path.components() {
            if let std::path::Component::ParentDir = component {
                return Err(format!("PathTraversalDetected: Resource path '{}' contains parent directory traversal ('..')", path.display()));
            }
        }
        let s = path.to_string_lossy();
        if s.is_empty() || s.len() > 512 {
            return Err("Filesystem path length must be 1-512 characters".into());
        }
        Ok(Self::FilesystemPath(path.to_path_buf()))
    }

    pub fn filesystem(p: impl AsRef<Path>) -> Result<Self, String> {
        Self::path(p)
    }

    pub fn resource_urn(&self) -> String {
        match self {
            Self::SystemVault => "tara:res:system_vault".into(),
            Self::PolicyRegistry => "tara:res:policy_registry".into(),
            Self::AuditStore => "tara:res:audit_store".into(),
            Self::DeviceRegistry => "tara:res:device_registry".into(),
            Self::AuthorityRoot => "tara:res:authority_root".into(),
            Self::FilesystemPath(p) => format!("tara:res:fs:{}", p.to_string_lossy().replace('\\', "/")),
            Self::NamedEntity { namespace, id } => format!("tara:res:{}:{}", namespace, id),
        }
    }
}

/// Extensible dynamic Assurance Level rating.
/// Supports numerical gradation to allow future levels (e.g. quantum-resistant, biometric threshold).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AssuranceLevel(u32);

impl AssuranceLevel {
    pub const ANONYMOUS: Self = Self(0);
    pub const STANDARD_TOKEN: Self = Self(100);
    pub const HARDWARE_ATTESTED: Self = Self(200);
    pub const BIOMETRIC_VERIFIED: Self = Self(300);
    pub const MULTI_FACTOR_AUTHENTICATED: Self = Self(400);
    pub const CREATOR_ROOT: Self = Self(500);

    pub fn custom(level: u32) -> Self {
        Self(level)
    }

    pub fn score(&self) -> u32 {
        self.0
    }
}

/// Extensible Principal Role classification.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PrincipalRole(String);

impl PrincipalRole {
    pub fn new(role: impl Into<String>) -> Result<Self, String> {
        let r = role.into().trim().to_uppercase();
        if r.is_empty() || r.len() > 64 {
            return Err("Role name must be between 1 and 64 characters".into());
        }
        if !r.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err("Role name must contain only uppercase alphanumeric characters and underscores".into());
        }
        Ok(Self(r))
    }

    pub fn creator() -> Self {
        Self("CREATOR".into())
    }

    pub fn operator() -> Self {
        Self("OPERATOR".into())
    }

    pub fn worker() -> Self {
        Self("WORKER".into())
    }

    pub fn user() -> Self {
        Self("USER".into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Monotonically incrementing Authority Generation counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthorityGeneration(u64);

impl AuthorityGeneration {
    pub const INITIAL: Self = Self(1);

    pub fn initial() -> Self {
        Self::INITIAL
    }

    pub fn new(gen: u64) -> Result<Self, String> {
        if gen == 0 {
            return Err("AuthorityGeneration must be non-zero".into());
        }
        Ok(Self(gen))
    }

    pub fn next(&self) -> Result<Self, String> {
        self.0
            .checked_add(1)
            .map(Self)
            .ok_or_else(|| "AuthorityGeneration counter overflow".to_string())
    }

    pub fn get(&self) -> u64 {
        self.0
    }
}

impl fmt::Display for AuthorityGeneration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Monotonically incrementing Audit Sequence number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuditSequence(u64);

impl AuditSequence {
    pub const INITIAL: Self = Self(1);

    pub fn new(seq: u64) -> Result<Self, String> {
        if seq == 0 {
            return Err("AuditSequence must be non-zero".into());
        }
        Ok(Self(seq))
    }

    pub fn next(&self) -> Result<Self, String> {
        self.0
            .checked_add(1)
            .map(Self)
            .ok_or_else(|| "AuditSequence counter overflow".to_string())
    }

    pub fn get(&self) -> u64 {
        self.0
    }
}

/// Opaque, authenticated identity token.
///
/// Constructed strictly inside verifier modules upon successful cryptographic validation.
/// Callers cannot manufacture this struct by providing boolean assertions or deserializing raw data.
#[derive(Debug, Clone, Serialize)]
pub struct VerifiedPrincipal {
    id: PrincipalId,
    role: PrincipalRole,
    assurance_level: AssuranceLevel,
    bound_device: Option<DeviceId>,
    authority_generation: AuthorityGeneration,
    verified_at_epoch_secs: u64,
    claims: HashMap<String, serde_json::Value>,
}

impl VerifiedPrincipal {
    /// Authentic constructor for authorized verifier boundaries.
    pub fn construct_verified(
        id: PrincipalId,
        role: PrincipalRole,
        assurance_level: AssuranceLevel,
        bound_device: Option<DeviceId>,
        authority_generation: AuthorityGeneration,
        verified_at_epoch_secs: u64,
        claims: HashMap<String, serde_json::Value>,
    ) -> Result<Self, String> {
        // Enforce strict bounds on claims to prevent resource exhaustion attacks
        const MAX_CLAIMS_COUNT: usize = 64;
        const MAX_CLAIM_KEY_LEN: usize = 64;
        const MAX_CLAIM_VALUE_BYTES: usize = 4096;

        if claims.len() > MAX_CLAIMS_COUNT {
            return Err(format!("ClaimsCountExceeded: {} claims exceeds maximum limit of {}", claims.len(), MAX_CLAIMS_COUNT));
        }

        for (k, v) in &claims {
            if k.is_empty() || k.len() > MAX_CLAIM_KEY_LEN {
                return Err(format!("ClaimKeyInvalid: Key length {} exceeds maximum limit of {}", k.len(), MAX_CLAIM_KEY_LEN));
            }
            let val_str = v.to_string();
            if val_str.len() > MAX_CLAIM_VALUE_BYTES {
                return Err(format!("ClaimValueTooLarge: Claim '{}' value size {} exceeds limit of {}", k, val_str.len(), MAX_CLAIM_VALUE_BYTES));
            }
        }

        // Creator Root assurance gate: Creator privileges CANNOT be minted without verified cryptographic proof witness
        if role == PrincipalRole::creator() || assurance_level >= AssuranceLevel::CREATOR_ROOT {
            let s_id = claims.get("session_id").and_then(|s| s.as_str()).unwrap_or("");
            let has_valid_session_proof = claims.get("session_proof").and_then(|v| v.as_str()).map(|hex_proof| {
                if let Ok(bytes) = hex::decode(hex_proof.trim()) {
                    if bytes.len() == 32 {
                        let expected = compute_session_proof(&id.as_str(), s_id);
                        constant_time_eq(&bytes, &expected)
                    } else {
                        false
                    }
                } else {
                    false
                }
            }).unwrap_or(false);

            if !has_valid_session_proof {
                return Err("UnauthorizedPrincipalConstruction: Creator privileges require an authentic cryptographic session proof witness".into());
            }
        }

        Ok(Self {
            id,
            role,
            assurance_level,
            bound_device,
            authority_generation,
            verified_at_epoch_secs,
            claims,
        })
    }


    pub fn id(&self) -> &PrincipalId {
        &self.id
    }

    pub fn role(&self) -> &PrincipalRole {
        &self.role
    }

    pub fn assurance_level(&self) -> AssuranceLevel {
        self.assurance_level
    }

    pub fn bound_device(&self) -> Option<&DeviceId> {
        self.bound_device.as_ref()
    }

    pub fn authority_generation(&self) -> AuthorityGeneration {
        self.authority_generation
    }

    pub fn verified_at_epoch_secs(&self) -> u64 {
        self.verified_at_epoch_secs
    }

    pub fn get_claim(&self, key: &str) -> Option<&serde_json::Value> {
        self.claims.get(key)
    }

    pub fn is_creator_root(&self) -> bool {
        self.role == PrincipalRole::creator() && self.assurance_level >= AssuranceLevel::CREATOR_ROOT
    }
}

/// Typed security state machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityState {
    Uninitialized,
    Operational {
        generation: AuthorityGeneration,
    },
    LockedDown {
        generation: AuthorityGeneration,
        reason: String,
        since_epoch_secs: u64,
    },
    Recovering {
        target_generation: AuthorityGeneration,
        nonce: String,
    },
}

impl SecurityState {
    pub fn is_operational(&self) -> bool {
        matches!(self, Self::Operational { .. })
    }

    pub fn is_locked_down(&self) -> bool {
        matches!(self, Self::LockedDown { .. })
    }
}

/// Security context containing execution metadata for authorization decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContext {
    pub client_ip: String,
    pub request_epoch_secs: u64,
    pub challenge_nonce: Option<String>,
    pub attributes: HashMap<String, String>,
}

impl SecurityContext {
    pub fn new(client_ip: impl Into<String>, epoch_secs: u64) -> Self {
        Self {
            client_ip: client_ip.into(),
            request_epoch_secs: epoch_secs,
            challenge_nonce: None,
            attributes: HashMap::new(),
        }
    }

    pub fn with_challenge(mut self, nonce: impl Into<String>) -> Self {
        self.challenge_nonce = Some(nonce.into());
        self
    }

    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }
}

pub const CANONICAL_CREATOR_ID: &str = "ROOT_OPERATOR";
pub const CANONICAL_ISSUER: &str = "taracore.ai";
pub const CANONICAL_AUDIENCE: &str = "tara_server";

static AUTHORITY_KEY: std::sync::LazyLock<[u8; 32]> = std::sync::LazyLock::new(|| {
    crate::dpapi::derive_authority_hmac_key("canonical_authority_principal_seal")
        .expect("FAIL CLOSED: Mandatory durable root authority seal key must be derived from protected DPAPI storage")
});

/// Compute a cryptographic HMAC-SHA256 session proof witness for an authentic creator session.
pub(crate) fn compute_session_proof(actor_id: &str, session_id: &str) -> [u8; 32] {
    let mut signer =
        crate::cryptography::HmacSha256Signer::new(&*AUTHORITY_KEY).expect("HMAC accepts 32-byte key");
    signer.update(b"CREATOR_SESSION_PROOF_V1:");
    signer.update(actor_id.as_bytes());
    signer.update(b":");
    signer.update(session_id.as_bytes());
    signer.finalize()
}

pub(crate) fn compute_session_proof_hex(actor_id: &str, session_id: &str) -> String {
    hex::encode(compute_session_proof(actor_id, session_id))
}

/// Role classification for authenticated principals.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AuthenticatedRole {
    Creator,
    SystemAgent,
    User,
    Untrusted,
}

/// Strongly-typed authenticated principal carrying verified actor identity and cryptographic authority seal.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuthenticatedPrincipal {
    actor_id: String,
    role: AuthenticatedRole,
    session_id: String,
    issuer: String,
    audience: String,
    capabilities: Vec<String>,
    created_epoch_secs: u64,
    expires_epoch_secs: u64,
    authority_sig: [u8; 32],
}

impl AuthenticatedPrincipal {
    fn current_time_secs() -> u64 {
        crate::cryptography::CryptoUtils::now_secs()
    }

    fn compute_authority_sig(
        actor_id: &str,
        role: AuthenticatedRole,
        session_id: &str,
        issuer: &str,
        audience: &str,
        created_epoch_secs: u64,
        expires_epoch_secs: u64,
    ) -> [u8; 32] {
        let role_str = match role {
            AuthenticatedRole::Creator => "CREATOR",
            AuthenticatedRole::SystemAgent => "SYSTEM_AGENT",
            AuthenticatedRole::User => "USER",
            AuthenticatedRole::Untrusted => "UNTRUSTED",
        };
        let mut signer =
            crate::cryptography::HmacSha256Signer::new(&*AUTHORITY_KEY).expect("HMAC accepts 32-byte key");
        signer.update(actor_id.as_bytes());
        signer.update(b":");
        signer.update(role_str.as_bytes());
        signer.update(b":");
        signer.update(session_id.as_bytes());
        signer.update(b":");
        signer.update(issuer.as_bytes());
        signer.update(b":");
        signer.update(audience.as_bytes());
        signer.update(b":");
        signer.update(&created_epoch_secs.to_be_bytes());
        signer.update(b":");
        signer.update(&expires_epoch_secs.to_be_bytes());
        signer.finalize()
    }

    pub fn is_expired(&self) -> bool {
        Self::current_time_secs() >= self.expires_epoch_secs
    }

    pub fn verify_authority(&self) -> bool {
        if self.is_expired() {
            return false;
        }
        if self.issuer != CANONICAL_ISSUER || self.audience != CANONICAL_AUDIENCE {
            return false;
        }
        let expected = Self::compute_authority_sig(
            &self.actor_id,
            self.role,
            &self.session_id,
            &self.issuer,
            &self.audience,
            self.created_epoch_secs,
            self.expires_epoch_secs,
        );
        constant_time_eq(&self.authority_sig, &expected)
    }

    /// Mint an authenticated creator principal requiring authentic cryptographic proof witness.
    /// Strictly rejects unauthenticated actor strings (fixes finding #1: Root authority bypass).
    pub fn mint_creator(
        actor_id: impl Into<String>,
        session_id: impl Into<String>,
        proof_witness: &str,
    ) -> Result<Self, String> {
        let a_id = actor_id.into();
        let s_id = session_id.into();
        if a_id != CANONICAL_CREATOR_ID {
            return Err(format!(
                "Actor '{}' is not authorized as canonical creator '{}'",
                a_id, CANONICAL_CREATOR_ID
            ));
        }
        if s_id.trim().is_empty() {
            return Err("Session ID cannot be empty for creator principal".to_string());
        }

        // Cryptographic proof verification: Zero mocks, zero string bypasses
        let expected_proof = compute_session_proof(&a_id, &s_id);
        let witness_trimmed = proof_witness.trim();
        let is_valid = if let Ok(witness_bytes) = hex::decode(witness_trimmed) {
            if witness_bytes.len() == 32 {
                constant_time_eq(&witness_bytes, &expected_proof)
            } else {
                false
            }
        } else {
            false
        };

        if !is_valid {
            return Err("UnauthorizedPrincipalMinting: Creator privileges require authentic cryptographic proof witness".into());
        }

        let created = Self::current_time_secs();
        let expires = created.saturating_add(3600);
        let authority_sig = Self::compute_authority_sig(
            &a_id,
            AuthenticatedRole::Creator,
            &s_id,
            CANONICAL_ISSUER,
            CANONICAL_AUDIENCE,
            created,
            expires,
        );
        Ok(Self {
            actor_id: a_id,
            role: AuthenticatedRole::Creator,
            session_id: s_id,
            issuer: CANONICAL_ISSUER.to_string(),
            audience: CANONICAL_AUDIENCE.to_string(),
            capabilities: vec![
                "all".to_string(),
                "creator".to_string(),
                "admin".to_string(),
                "execute".to_string(),
            ],
            created_epoch_secs: created,
            expires_epoch_secs: expires,
            authority_sig,
        })
    }

    pub fn mint_creator_from_verified_session(
        actor_id: impl Into<String>,
        session_id: impl Into<String>,
        session_proof_hex: &str,
    ) -> Result<Self, String> {
        Self::mint_creator(actor_id, session_id, session_proof_hex)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn mint_creator_for_test(
        session_id: impl Into<String>,
    ) -> Result<Self, String> {
        let s_id = session_id.into();
        let proof = compute_session_proof_hex(CANONICAL_CREATOR_ID, &s_id);
        Self::mint_creator(CANONICAL_CREATOR_ID, s_id, &proof)
    }

    pub fn system_agent(actor_id: impl Into<String>) -> Self {
        let a_id = actor_id.into();
        let s_id = "system_session".to_string();
        let created = Self::current_time_secs();
        let expires = created.saturating_add(86400 * 365);
        let authority_sig = Self::compute_authority_sig(
            &a_id,
            AuthenticatedRole::SystemAgent,
            &s_id,
            CANONICAL_ISSUER,
            CANONICAL_AUDIENCE,
            created,
            expires,
        );
        Self {
            actor_id: a_id,
            role: AuthenticatedRole::SystemAgent,
            session_id: s_id,
            issuer: CANONICAL_ISSUER.to_string(),
            audience: CANONICAL_AUDIENCE.to_string(),
            capabilities: vec!["system".to_string(), "execute".to_string()],
            created_epoch_secs: created,
            expires_epoch_secs: expires,
            authority_sig,
        }
    }

    pub fn user(actor_id: impl Into<String>, session_id: impl Into<String>) -> Self {
        let a_id = actor_id.into();
        let s_id = session_id.into();
        let created = Self::current_time_secs();
        let expires = created.saturating_add(86400);
        let authority_sig = Self::compute_authority_sig(
            &a_id,
            AuthenticatedRole::User,
            &s_id,
            CANONICAL_ISSUER,
            CANONICAL_AUDIENCE,
            created,
            expires,
        );
        Self {
            actor_id: a_id,
            role: AuthenticatedRole::User,
            session_id: s_id,
            issuer: CANONICAL_ISSUER.to_string(),
            audience: CANONICAL_AUDIENCE.to_string(),
            capabilities: vec!["user".to_string()],
            created_epoch_secs: created,
            expires_epoch_secs: expires,
            authority_sig,
        }
    }

    pub fn untrusted(actor_id: impl Into<String>) -> Self {
        let a_id = actor_id.into();
        let s_id = String::new();
        let created = Self::current_time_secs();
        let expires = created;
        let authority_sig = Self::compute_authority_sig(
            &a_id,
            AuthenticatedRole::Untrusted,
            &s_id,
            CANONICAL_ISSUER,
            CANONICAL_AUDIENCE,
            created,
            expires,
        );
        Self {
            actor_id: a_id,
            role: AuthenticatedRole::Untrusted,
            session_id: s_id,
            issuer: CANONICAL_ISSUER.to_string(),
            audience: CANONICAL_AUDIENCE.to_string(),
            capabilities: Vec::new(),
            created_epoch_secs: created,
            expires_epoch_secs: expires,
            authority_sig,
        }
    }

    pub fn actor_id(&self) -> &str {
        &self.actor_id
    }

    pub fn name(&self) -> &str {
        &self.actor_id
    }

    pub fn role(&self) -> AuthenticatedRole {
        self.role
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn issuer(&self) -> &str {
        &self.issuer
    }

    pub fn audience(&self) -> &str {
        &self.audience
    }

    pub fn capabilities(&self) -> &[String] {
        &self.capabilities
    }

    pub fn has_capability(&self, capability: &str) -> bool {
        self.verify_authority()
            && (self.capabilities.iter().any(|c| c == capability || c == "all")
                || self.is_creator())
    }

    pub fn created_epoch_secs(&self) -> u64 {
        self.created_epoch_secs
    }

    pub fn expires_epoch_secs(&self) -> u64 {
        self.expires_epoch_secs
    }

    pub fn is_creator(&self) -> bool {
        self.role == AuthenticatedRole::Creator
            && self.actor_id == CANONICAL_CREATOR_ID
            && !self.session_id.is_empty()
            && self.issuer == CANONICAL_ISSUER
            && self.audience == CANONICAL_AUDIENCE
            && !self.is_expired()
            && self.verify_authority()
    }

    pub fn is_system_agent(&self) -> bool {
        self.role == AuthenticatedRole::SystemAgent
            && self.issuer == CANONICAL_ISSUER
            && self.audience == CANONICAL_AUDIENCE
            && !self.is_expired()
            && self.verify_authority()
    }

    pub fn is_authenticated(&self) -> bool {
        self.verify_authority() && self.role != AuthenticatedRole::Untrusted
    }
}

impl TryFrom<&AuthenticatedPrincipal> for VerifiedPrincipal {
    type Error = String;

    fn try_from(auth: &AuthenticatedPrincipal) -> Result<Self, Self::Error> {
        if !auth.verify_authority() {
            return Err("UnauthorizedPrincipalConversion: Authority signature verification failed or principal is expired".into());
        }
        let pid = PrincipalId::new(&auth.actor_id)?;
        let role = match auth.role {
            AuthenticatedRole::Creator => PrincipalRole::creator(),
            AuthenticatedRole::SystemAgent => PrincipalRole::operator(),
            AuthenticatedRole::User => PrincipalRole::user(),
            AuthenticatedRole::Untrusted => PrincipalRole::new("UNTRUSTED")?,
        };
        let assurance = match auth.role {
            AuthenticatedRole::Creator => AssuranceLevel::CREATOR_ROOT,
            AuthenticatedRole::SystemAgent => AssuranceLevel::HARDWARE_ATTESTED,
            AuthenticatedRole::User => AssuranceLevel::STANDARD_TOKEN,
            AuthenticatedRole::Untrusted => AssuranceLevel::ANONYMOUS,
        };
        let mut claims = HashMap::new();
        if auth.role == AuthenticatedRole::Creator {
            let session_proof = compute_session_proof_hex(&auth.actor_id, &auth.session_id);
            claims.insert(
                "session_proof".into(),
                serde_json::Value::String(session_proof),
            );
            claims.insert(
                "verified_via".into(),
                serde_json::Value::String("authenticated_creator_session".into()),
            );
        }
        claims.insert(
            "session_id".into(),
            serde_json::Value::String(auth.session_id.clone()),
        );
        claims.insert(
            "issuer".into(),
            serde_json::Value::String(auth.issuer.clone()),
        );
        claims.insert(
            "audience".into(),
            serde_json::Value::String(auth.audience.clone()),
        );

        VerifiedPrincipal::construct_verified(
            pid,
            role,
            assurance,
            None,
            AuthorityGeneration::INITIAL,
            auth.created_epoch_secs,
            claims,
        )
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verified_principal_creator_requires_witness() {
        let pid = PrincipalId::new("creator_alice").unwrap();
        let gen = AuthorityGeneration::INITIAL;
        let empty_claims = HashMap::new();

        // 1. Creator role with empty claims must FAIL CLOSED
        let res = VerifiedPrincipal::construct_verified(
            pid.clone(),
            PrincipalRole::creator(),
            AssuranceLevel::CREATOR_ROOT,
            None,
            gen,
            1000,
            empty_claims,
        );
        assert!(res.is_err(), "Creator principal without verified_via must fail closed");

        // 2. Creator role with unapproved claim value must FAIL CLOSED
        let mut forged_claims = HashMap::new();
        forged_claims.insert("verified_via".into(), serde_json::json!("arbitrary_unauthenticated_string"));
        let res_forged = VerifiedPrincipal::construct_verified(
            pid.clone(),
            PrincipalRole::creator(),
            AssuranceLevel::CREATOR_ROOT,
            None,
            gen,
            1000,
            forged_claims,
        );
        assert!(res_forged.is_err(), "Forged claim string must fail closed");

        // 3. Creator role with authentic session_proof claim must SUCCEED
        let mut valid_claims = HashMap::new();
        let proof = compute_session_proof_hex(&pid.as_str(), "sess_123");
        valid_claims.insert("session_id".into(), serde_json::json!("sess_123"));
        valid_claims.insert("session_proof".into(), serde_json::json!(proof));
        valid_claims.insert("verified_via".into(), serde_json::json!("authenticated_creator_session"));
        let res_valid = VerifiedPrincipal::construct_verified(
            pid.clone(),
            PrincipalRole::creator(),
            AssuranceLevel::CREATOR_ROOT,
            None,
            gen,
            1000,
            valid_claims,
        );
        assert!(res_valid.is_ok(), "Authentic witness claim must succeed");
        let principal = res_valid.unwrap();
        assert_eq!(principal.role(), &PrincipalRole::creator());
        assert_eq!(principal.assurance_level(), AssuranceLevel::CREATOR_ROOT);

        // 4. Non-privileged user does not require Creator root witness
        let user_pid = PrincipalId::new("standard_user").unwrap();
        let user_res = VerifiedPrincipal::construct_verified(
            user_pid,
            PrincipalRole::user(),
            AssuranceLevel::STANDARD_TOKEN,
            None,
            gen,
            1000,
            HashMap::new(),
        );
        assert!(user_res.is_ok(), "Standard user does not require root witness");
    }

    #[test]
    fn test_mint_creator_cryptographic_verification() {
        // 1. Arbitrary string bypass must FAIL CLOSED
        let res_bypass = AuthenticatedPrincipal::mint_creator(
            CANONICAL_CREATOR_ID,
            "session_123",
            "authenticated_creator_session",
        );
        assert!(res_bypass.is_err(), "Arbitrary string bypass must be rejected");

        // 2. Authentic proof witness must SUCCEED
        let valid_proof = compute_session_proof_hex(CANONICAL_CREATOR_ID, "session_123");
        let res_valid = AuthenticatedPrincipal::mint_creator(
            CANONICAL_CREATOR_ID,
            "session_123",
            &valid_proof,
        );
        assert!(res_valid.is_ok(), "Authentic cryptographic session proof must succeed");
        let principal = res_valid.unwrap();
        assert!(principal.verify_authority());
        assert!(principal.is_creator());

        // 3. Test harness helper mints authentic principal
        let test_principal = AuthenticatedPrincipal::mint_creator_for_test("session_test").unwrap();
        assert!(test_principal.verify_authority());
        assert!(test_principal.is_creator());
    }
}
