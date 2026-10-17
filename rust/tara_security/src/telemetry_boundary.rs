//! Type-Level Boundaries Between Heuristic Telemetry and Authoritative Security.
//!
//! Enforces:
//! - Strict nominal type separation between heuristic cognitive telemetry
//!   and authenticated security decisions.
//! - ZERO implicit conversions (`From`/`Into`) between heuristic data and security credentials.
//! - Only `AuthenticatedAuthority` (bearing a verified `VerifiedPrincipal`) can
//!   confer execution authorization.

use crate::primitives::VerifiedPrincipal;

/// Wrapper marking data strictly as non-authoritative heuristic telemetry.
/// This type cannot be converted or coerced into any security authorization type.
#[derive(Debug, Clone)]
pub struct HeuristicTelemetry<T> {
    pub value: T,
    pub description: &'static str,
}

impl<T> HeuristicTelemetry<T> {
    pub fn new(value: T, description: &'static str) -> Self {
        Self { value, description }
    }

    pub fn inner(&self) -> &T {
        &self.value
    }
}

/// Proof of verified, authenticated authority bound to a `VerifiedPrincipal`.
#[derive(Debug, Clone)]
pub struct AuthenticatedAuthority {
    principal: VerifiedPrincipal,
}

impl AuthenticatedAuthority {
    pub fn new(principal: VerifiedPrincipal) -> Result<Self, String> {
        if !principal.is_creator_root() {
            return Err(
                "Cannot construct AuthenticatedAuthority: Creator root authority required".into(),
            );
        }
        Ok(Self { principal })
    }

    /// Constructs an `AuthenticatedAuthority` from a verified non-anonymous principal.
    /// Rejects anonymous principals (`AssuranceLevel::ANONYMOUS`).
    pub fn from_verified(principal: VerifiedPrincipal) -> Result<Self, String> {
        if principal.assurance_level().score()
            < crate::primitives::AssuranceLevel::STANDARD_TOKEN.score()
        {
            return Err(
                "Cannot construct AuthenticatedAuthority: Minimum STANDARD_TOKEN assurance level required (anonymous principals rejected)".into(),
            );
        }
        Ok(Self { principal })
    }

    pub fn principal(&self) -> &VerifiedPrincipal {
        &self.principal
    }

    pub fn is_creator(&self) -> bool {
        self.principal.is_creator_root()
    }
}

/// Strongly-typed, authenticated security decision issued exclusively
/// by an authorized security gate or ExecutionGuard.
#[derive(Debug, Clone)]
pub struct SecurityDecision {
    allowed: bool,
    reason: String,
    authority: AuthenticatedAuthority,
}

impl SecurityDecision {
    pub fn allow(authority: AuthenticatedAuthority, reason: impl Into<String>) -> Self {
        Self {
            allowed: true,
            reason: reason.into(),
            authority,
        }
    }

    pub fn block(authority: AuthenticatedAuthority, reason: impl Into<String>) -> Self {
        Self {
            allowed: false,
            reason: reason.into(),
            authority,
        }
    }

    pub fn is_allowed(&self) -> bool {
        self.allowed
    }

    pub fn reason(&self) -> &str {
        &self.reason
    }

    pub fn authority(&self) -> &AuthenticatedAuthority {
        &self.authority
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::{
        AssuranceLevel, AuthorityGeneration, PrincipalId, PrincipalRole, VerifiedPrincipal,
    };
    use std::collections::HashMap;

    #[test]
    fn test_authenticated_authority_assurance_level_gating() {
        let anon_pid = PrincipalId::new("anon_user").unwrap();
        let anon_principal = VerifiedPrincipal::construct_verified(
            anon_pid,
            PrincipalRole::user(),
            AssuranceLevel::ANONYMOUS,
            None,
            AuthorityGeneration::INITIAL,
            crate::cryptography::CryptoUtils::now_secs(),
            HashMap::new(),
        )
        .unwrap();

        // 1. Anonymous principal must be rejected by from_verified
        let anon_res = AuthenticatedAuthority::from_verified(anon_principal);
        assert!(
            anon_res.is_err(),
            "Anonymous principal must not construct AuthenticatedAuthority"
        );
        assert!(anon_res
            .unwrap_err()
            .contains("Minimum STANDARD_TOKEN assurance level required"));

        // 2. Standard token principal is allowed by from_verified
        let token_pid = PrincipalId::new("operator_user").unwrap();
        let token_principal = VerifiedPrincipal::construct_verified(
            token_pid,
            PrincipalRole::operator(),
            AssuranceLevel::STANDARD_TOKEN,
            None,
            AuthorityGeneration::INITIAL,
            crate::cryptography::CryptoUtils::now_secs(),
            HashMap::new(),
        )
        .unwrap();

        let token_res = AuthenticatedAuthority::from_verified(token_principal.clone());
        assert!(
            token_res.is_ok(),
            "STANDARD_TOKEN principal must construct AuthenticatedAuthority"
        );
        let auth = token_res.unwrap();
        assert!(!auth.is_creator());

        // 3. AuthenticatedAuthority::new requires creator root
        let new_res = AuthenticatedAuthority::new(token_principal);
        assert!(
            new_res.is_err(),
            "Non-creator principal cannot use AuthenticatedAuthority::new"
        );
    }
}
