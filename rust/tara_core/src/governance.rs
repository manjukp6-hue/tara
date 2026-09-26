//! authority.rs
//! 
//! Cryptographic Creator Authority & Identity Management for TARA Core.
//! Enforces immutable role hierarchies where CREATOR holds absolute root authority.

use serde::{Deserialize, Serialize};
use ed25519_dalek::{Signature, VerifyingKey, Verifier};
use sha2::{Sha256, Digest};

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
