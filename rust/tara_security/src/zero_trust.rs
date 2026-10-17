//! Zero-Trust Continuous Authorization & Capability Subsystem (Pillar 2).
//!
//! High-Security Guarantee:
//! - Deny-by-default dynamic policy engine.
//! - Continuous risk-adaptive UEBA evaluation with dynamic step-up authentication.
//! - Macaroon-style capability tokens with HMAC caveat attenuation.
//! - Dynamic identity proof verification with zero boolean bypass flags.
//! - Server-side session validation and single-use challenge nonce replay protection.

pub mod capabilities;
pub mod identity;
pub mod policy;
pub mod risk;
pub mod session;

pub use capabilities::{CapabilityCaveat, CapabilityToken};
pub use identity::{
    DeviceEnrollmentRecord, DeviceEnrollmentRegistry, DeviceSignatureVerifier, GoogleAuthService,
    GoogleOAuthVerifier, IdentityProof, IdentityVerifier, IdentityVerifierRegistry, GOOGLE_ISSUERS,
    GOOGLE_JWKS_URL,
};
pub use policy::{PolicyEngine, PolicyRule, SecurityDenial};
pub use risk::{ContinuousRiskEngine, RiskVerdict};
pub use session::{ChallengeRecord, SessionManager, SessionRecord};
