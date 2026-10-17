//! authority.rs / governance.rs
//!
//! Cryptographic Creator Authority & Identity Management for TARA Core.
//! Enforces immutable role hierarchies where CREATOR holds absolute root authority.
//!
//! # Architecture & Ownership Boundary
//! - **Scope**: In-crate entity role primitives and standalone Ed25519 creator signature validation.
//! - **Authoritative Runtime Zero-Trust**: Real-time cross-crate session authentication, hardware
//!   attestation, token lifecycle, and cryptographic zero-trust enforcement are authoritatively owned
//!   by `tara_security::zero_trust` and `tara_security::hardware_root`.
//! - **Usage**: Used by `tara_core::runtime` actors and supervisor hierarchies for local identity checks.

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Role {
    AI = 1,
    DEVICE = 2,
    USER = 3,
    STAFF = 4,
    ADMIN = 5,
    CREATOR = 100,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identity {
    pub id: String,
    pub name: String,
    pub role: Role,
    pub public_key: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CreatorAuthority {
    pub creator_id: String,
    creator_verifying_key: Option<VerifyingKey>,
}

impl CreatorAuthority {
    pub fn new(creator_id: impl Into<String>, public_key_bytes: Option<&[u8; 32]>) -> Self {
        let verifying_key = public_key_bytes.and_then(|bytes| VerifyingKey::from_bytes(bytes).ok());
        Self {
            creator_id: creator_id.into(),
            creator_verifying_key: verifying_key,
        }
    }

    /// Explicitly validates and builds CreatorAuthority from 32-byte Ed25519 public key
    pub fn try_new(
        creator_id: impl Into<String>,
        public_key_bytes: &[u8; 32],
    ) -> Result<Self, String> {
        let vk = VerifyingKey::from_bytes(public_key_bytes)
            .map_err(|e| format!("Invalid Ed25519 public key: {e}"))?;
        Ok(Self {
            creator_id: creator_id.into(),
            creator_verifying_key: Some(vk),
        })
    }

    /// Verifies cryptographic signature for high-privilege Creator actions
    pub fn verify_creator_signature(&self, message: &[u8], signature_bytes: &[u8; 64]) -> bool {
        if let Some(vk) = &self.creator_verifying_key {
            let sig = Signature::from_bytes(signature_bytes);
            vk.verify(message, &sig).is_ok()
        } else {
            false
        }
    }

    /// Strict guardrail: AI or external callers can never escalate to CREATOR
    pub fn can_escalate_role(&self, actor_role: Role, target_role: Role) -> bool {
        if actor_role != Role::CREATOR {
            if actor_role <= Role::USER {
                return false;
            }
            // Non-creator can never promote anyone to CREATOR or change protected rules
            if target_role == Role::CREATOR {
                return false;
            }
            // Cannot promote to a role higher than or equal to own role (strict hierarchy)
            return actor_role > target_role;
        }
        true
    }

    /// Evaluates whether an actor has authority to modify a target's role from current to new
    pub fn can_modify_target_role(
        &self,
        actor_role: Role,
        target_current_role: Role,
        target_new_role: Role,
    ) -> bool {
        if actor_role == Role::CREATOR {
            return true;
        }
        if actor_role <= Role::USER {
            return false;
        }
        if target_new_role == Role::CREATOR {
            return false;
        }
        actor_role > target_current_role && actor_role > target_new_role
    }

    /// Verifies that an AI cannot modify Creator identity
    pub fn can_modify_creator(&self, actor_role: Role) -> bool {
        actor_role == Role::CREATOR
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use rand::rngs::OsRng;

    #[test]
    fn test_role_hierarchy_ordering() {
        assert!(Role::AI < Role::DEVICE);
        assert!(Role::DEVICE < Role::USER);
        assert!(Role::USER < Role::STAFF);
        assert!(Role::STAFF < Role::ADMIN);
        assert!(Role::ADMIN < Role::CREATOR);
    }

    #[test]
    fn test_escalation_guardrails() {
        let auth = CreatorAuthority::new("creator_root", None);
        // AI cannot escalate to CREATOR or ADMIN
        assert!(!auth.can_escalate_role(Role::AI, Role::CREATOR));
        assert!(!auth.can_escalate_role(Role::AI, Role::ADMIN));
        // USER cannot escalate to CREATOR or self/peer
        assert!(!auth.can_escalate_role(Role::USER, Role::CREATOR));
        assert!(!auth.can_escalate_role(Role::USER, Role::USER));
        // ADMIN cannot escalate peer ADMIN to ADMIN
        assert!(!auth.can_escalate_role(Role::ADMIN, Role::ADMIN));
        // ADMIN can escalate STAFF to USER or DEVICE
        assert!(auth.can_escalate_role(Role::ADMIN, Role::STAFF));
        // CREATOR can escalate any role
        assert!(auth.can_escalate_role(Role::CREATOR, Role::CREATOR));
        assert!(auth.can_escalate_role(Role::CREATOR, Role::ADMIN));

        // Test can_modify_target_role
        assert!(auth.can_modify_target_role(Role::CREATOR, Role::ADMIN, Role::STAFF));
        assert!(auth.can_modify_target_role(Role::ADMIN, Role::STAFF, Role::USER));
        assert!(!auth.can_modify_target_role(Role::ADMIN, Role::ADMIN, Role::USER));
        assert!(!auth.can_modify_target_role(Role::ADMIN, Role::USER, Role::ADMIN));
        assert!(!auth.can_modify_target_role(Role::USER, Role::DEVICE, Role::AI));
    }

    #[test]
    fn test_creator_ed25519_signature_verification() {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        let pub_bytes = verifying_key.to_bytes();

        let auth = CreatorAuthority::new("creator_1", Some(&pub_bytes));

        let message = b"PROMOTE_MODEL_TO_PRODUCTION";
        let signature = signing_key.sign(message);
        let sig_bytes = signature.to_bytes();

        // Valid signature passes
        assert!(auth.verify_creator_signature(message, &sig_bytes));

        // Tampered message fails
        let tampered_message = b"PROMOTE_MODEL_TO_PRODUCTION_UNAUTHORIZED";
        assert!(!auth.verify_creator_signature(tampered_message, &sig_bytes));

        // Authority without public key fails closed
        let unkeyed_auth = CreatorAuthority::new("creator_unkeyed", None);
        assert!(!unkeyed_auth.verify_creator_signature(message, &sig_bytes));
    }

    #[test]
    fn test_creator_authority_try_new_valid_and_invalid() {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        let valid_bytes = verifying_key.to_bytes();

        // Valid public key bytes succeed
        let auth = CreatorAuthority::try_new("creator_valid", &valid_bytes);
        assert!(auth.is_ok());

        // Find an invalid Ed25519 point where decompression fails
        let mut invalid_bytes = [0u8; 32];
        invalid_bytes[0] = 5;
        let mut found = false;
        for b in 0..=255u8 {
            invalid_bytes[0] = b;
            if CreatorAuthority::try_new("creator_invalid", &invalid_bytes).is_err() {
                found = true;
                break;
            }
        }
        assert!(
            found,
            "Should find at least one invalid Ed25519 curve point"
        );
    }
}
