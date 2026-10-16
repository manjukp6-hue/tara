//! TARA Enterprise Zero-Trust Defense-in-Depth Security Subsystem.
//!
//! A high-security, dynamically extensible, production architecture built across the 6 Core Pillars:
//! 1. Hardware-Rooted Security & Attestation (`hardware_root`):
//!    - W3C WebAuthn Level 3 & TPM 2.0 PCR Quote measurement attestation.
//!    - Root authority first-run ceremony, key rotation, and emergency recovery.
//! 2. Zero-Trust Continuous Authorization (`zero_trust`):
//!    - Deny-by-default dynamic policy engine.
//!    - Continuous risk-adaptive UEBA evaluation with dynamic step-up authentication.
//!    - Macaroon-style capability tokens with offline HMAC caveat attenuation.
//!    - Server-side sessions and single-use challenge nonce replay protection.
//! 3. OS Process Isolation (`process_isolation`):
//!    - Dedicated OS process execution boundaries (`InferenceEngine`, `NetworkGateway`, `SecurityAuthority`, `StorageBroker`).
//!    - Mutually authenticated cryptographic IPC message envelopes (`IpcSecurityEnvelope`).
//! 4. Transparency Log & Anti-Rollback (`transparency_log`, `anti_rollback`):
//!    - RFC 6962 / RFC 9162 Merkle tree ledger with Signed Tree Heads (STH).
//!    - Inclusion proofs, consistency proofs, and fork/rollback detection.
//!    - Monotonic version anchors with explicit assurance levels.
//! 5. Multi-Party Threshold Quorum (`threshold_quorum`):
//!    - M-of-N multi-signature threshold quorum governance for critical administrative actions.
//! 6. Fail-Closed Defense (`fail_closed_defense`):
//!    - Active anti-tamper canary files and decoy honeypot token traps.
//!    - Security containment state with audited Creator-Root recovery.
//!
//! Pure native Rust, zero compiler warnings, zero external AI connection.

pub mod anti_rollback;
pub mod broker;
pub mod cbor;
pub mod creator;
pub mod cryptography;
pub mod devices;
pub mod dpapi;
pub mod fail_closed_defense;
pub mod hardware_root;
pub mod lockdown;
pub mod path_security;
pub mod persistence;
pub mod primitives;
pub mod process_isolation;
pub mod qr;
pub mod recovery;
pub mod tara_control;
pub mod telemetry_boundary;
pub mod threshold_quorum;
pub mod tokens;
pub mod transparency_log;
pub mod zero_trust;

/// ============================================================================
/// SECTION 1 — MAIN DYNAMIC CENTRALIZED SECURITY SYSTEM
/// ============================================================================
/// Protects the overall application, identities, access, APIs, and critical operations.
pub mod main_security {
    pub use crate::anti_rollback::*;
    pub use crate::broker::*;
    pub use crate::cryptography::*;
    pub use crate::fail_closed_defense::*;
    pub use crate::hardware_root::*;
    pub use crate::path_security::*;
    pub use crate::persistence::*;
    pub use crate::primitives::*;
    pub use crate::process_isolation::*;
    pub use crate::threshold_quorum::*;
    pub use crate::transparency_log::*;
    pub use crate::zero_trust::*;
}

/// ============================================================================
/// SECTION 2 — TARA-CONTROLLED SECURITY SYSTEM
/// ============================================================================
/// Protects TARA's own internal execution environment, autonomous agents,
/// source code, models, artifacts, and goal-execution lifecycle.
pub mod tara_controlled_security {
    pub use crate::tara_control::*;
}

// Ergonomic re-exports of core domain types
pub use anti_rollback::{
    AnchorAssuranceType, DurableMonotonicAnchor, LocalHostMonotonicAnchor, MonotonicAnchor,
    RollbackGuard,
};
pub use broker::{
    EnforcementGateway, EnforcementRequest, ModelAction, OperationBroker, OperationRequest,
    ProtectedCategory,
};
pub use path_security::{
    canonical_containment, check_lexical_traversal, is_symlink_escaping, normalize_path,
    validate_sandbox_path, validate_tool_command_path, PathSecurityError,
};
pub use cryptography::{
    AeadProvider, Aes256GcmProvider, AlgorithmId, CryptoUtils, Ed25519Verifier, Es256Verifier,
    HmacSha256Signer, Rs256Verifier, SignatureVerifier,
};
pub use fail_closed_defense::{CanarySentinel, TripwireDefenseEngine, TripwireStatus};
pub use hardware_root::{
    pcr_quote::{PcrQuoteVerificationResult, TpmPcrQuoteStatement, TpmPcrQuoteVerifier},
    AttestationFormatVerifier, AttestationRegistry, AttestationStatement, AttestationTrustStore,
    AttestationVerificationResult, AuthorityManager, AuthorityRecord, BiometricCapability,
    BiometricChallenge, BiometricChallengeService, BiometricFactorProvider, CoseAlgorithm,
    PlatformAuthenticatorType, PlatformBiometricAssertion, PlatformBiometricReport, RecoveryProof,
    RotationToken,
    UvmEntry, KEY_PROTECTION_HARDWARE, KEY_PROTECTION_SECURE_ELEMENT, KEY_PROTECTION_SOFTWARE,
    KEY_PROTECTION_TEE, MATCHER_PROTECTION_ON_CHIP, MATCHER_PROTECTION_SOFTWARE,
    MATCHER_PROTECTION_TEE, USER_VERIFY_ALL, USER_VERIFY_EYEPRINT, USER_VERIFY_FACEPRINT,
    USER_VERIFY_FINGERPRINT, USER_VERIFY_HANDPRINT, USER_VERIFY_LOCATION, USER_VERIFY_NONE,
    USER_VERIFY_PASSCODE, USER_VERIFY_PATTERN, USER_VERIFY_PRESENCE, USER_VERIFY_VOICEPRINT,
    extract_signed_uvm_entries_from_authenticator_data, CentralizedPasskeyVerifier,
    EnrolledBiometricCredential, ParsedAttestationObject, RelyingPartyConfig, TpmsAttest,
};
pub use qr::QrCodeMatrix;
pub use cbor::{CborDecoder, CborEncoder, CborValue};
pub use creator::{
    canonical_json_bytes, canonicalize_value, compute_canonical_hash,
    compute_recovery_verifier_hash, AuthorityLifecycleManager, AuthorityState, BootstrapWizard,
    CreatorAuthService, CreatorIdentity, CreatorIdentityRecord, CreatorSessionRecord,
    CreatorSetupEngine, CreatorSetupRequest, CreatorSetupResult, QrPrompt, RateLimiter,
    RevokedKey, CANONICAL_CREATOR_ID, DEFAULT_DISPLAY_NAME,
};
pub use devices::{
    DeviceCrypto, DeviceKeyPair, DeviceRecord, DeviceRegistryData, EnrollmentService,
    IdentityManager,
};
pub use dpapi::{
    cleanup_test_watermark_artifacts, delete_os_monotonic_watermark, derive_authority_hmac_key,
    get_os_isolated_watermark_path, get_system_witness_path, is_test_watermark_domain,
    load_monotonic_watermark, load_os_monotonic_watermark, native_dpapi,
    store_monotonic_watermark, store_os_monotonic_watermark, store_system_witness_watermark,
    SecureKeyStorage,
};
pub use lockdown::{LockdownCoordinator, LockdownRecord, LockdownState, ProgressiveLockdownCoordinator};
pub use recovery::{
    pbkdf2_hmac_sha256, RecoveryAuthorizationProof, RecoveryLockoutState, RecoveryManager,
};
pub use persistence::{atomic_write_file_hardened, HardenedFileSystemStorage, SecureStorage};
pub use primitives::{
    AssuranceLevel, AuditSequence, AuthenticatedPrincipal, AuthenticatedRole, AuthorityGeneration,
    ChallengeId, DeviceId, PrincipalId, PrincipalRole, SecurityAction, SecurityContext,
    SecurityResource, SecurityState, SessionId, VerifiedPrincipal, CANONICAL_AUDIENCE,
    CANONICAL_ISSUER,
};
pub use process_isolation::{
    AntiReplayCache, IpcEndpoint, IpcSecurityEnvelope, ProcessRole, SecureIpcBus,
};
pub use telemetry_boundary::{AuthenticatedAuthority, HeuristicTelemetry, SecurityDecision};
pub use threshold_quorum::{
    QuorumAuthorizationToken, QuorumPolicy, QuorumProposal, QuorumSignature, ThresholdQuorumEngine,
};
pub use tokens::{validate_protected_route, verify_api_key, verify_worker_token};
pub use transparency_log::{
    AuditRecord, AuditSink, AuthenticatedAuditLogger, FileAuditSink, MemoryAuditSink,
    MerkleConsistencyProof, MerkleInclusionProof, MerkleLeaf, MerkleTree, SignedTreeHead,
};
pub use tara_control::{
    canonical_manifest_payload, compute_file_sha256, AgentExecutionOutcome,
    AgentExecutionSandbox, AgentSandboxPolicy, AttackVector, AutonomousGoalGating,
    ContainmentManager, CopyleftConflictReport, DependencyVerificationReport,
    DependencyVerifier, GatedGoal, GoalState, HomoglyphNormalizer, JailbreakDefenseEngine,
    JailbreakDetector, JailbreakVerdict, LicenseFamily, LicenseProvenanceEngine,
    IntegrityError, ModelIntegrityEngine, ModelIntegrityGuard, ModelManifestSeal,
    ProvenanceLedgerGuard, ProvenanceRecord, QuarantineEngine, QuarantineRecord,
    QuarantineStage, QuarantineVerdict, ReleaseManifest, ReleaseManifestManager,
    RuntimeTamperSentinel, SealResult, SecretFinding, SecretLeakScanner, SourceProtector,
    SourceTamperDetector, SpdxLicenseInfo, TARAControlledSecurity, TokenScanner,
    VerifiedPackage, VerifyResult,
};
pub use zero_trust::{
    CapabilityCaveat, CapabilityToken, ChallengeRecord, ContinuousRiskEngine, GoogleAuthService,
    GoogleOAuthVerifier, IdentityProof, IdentityVerifier, IdentityVerifierRegistry, PolicyEngine,
    PolicyRule, RiskVerdict, SecurityDenial, SessionManager, SessionRecord, GOOGLE_ISSUERS,
    GOOGLE_JWKS_URL,
};

