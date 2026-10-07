//! authority.rs
//!
//! Cryptographic Creator Authority & Identity Management for TARA Core.
//! Enforces immutable role hierarchies where CREATOR holds absolute root authority.

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
            // Non-creator can never promote anyone to CREATOR or change protected rules
            if target_role == Role::CREATOR {
                return false;
            }
            // Cannot promote to a role higher than own role
            return actor_role >= target_role;
        }
        true
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
        // USER cannot escalate to CREATOR
        assert!(!auth.can_escalate_role(Role::USER, Role::CREATOR));
        // CREATOR can escalate any role
        assert!(auth.can_escalate_role(Role::CREATOR, Role::CREATOR));
        assert!(auth.can_escalate_role(Role::CREATOR, Role::ADMIN));
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
}
