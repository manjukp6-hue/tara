//! Worker-Token, API-Key Validation, and Protected Route Enforcement.
//!
//! Enforces:
//! - Strict constant-time comparisons (`constant_time_eq`) to eliminate timing side-channels.
//! - Cryptographically sound validation for worker node registrations and protected control plane routes.
//! - Non-empty string validation and bounded token checks.

use constant_time_eq::constant_time_eq;

/// Constant-time worker token validation.
/// Returns true only if candidate and expected tokens match exactly.
pub fn verify_worker_token(candidate: &str, expected: &str) -> bool {
    let c = candidate.trim();
    let e = expected.trim();
    if c.is_empty() || e.is_empty() {
        return false;
    }
    if c.len() != e.len() {
        return false;
    }
    constant_time_eq(c.as_bytes(), e.as_bytes())
}

/// Constant-time API key validation.
/// Returns true only if candidate and expected API keys match exactly.
pub fn verify_api_key(candidate: &str, expected: &str) -> bool {
    let c = candidate.trim();
    let e = expected.trim();
    if c.is_empty() || e.is_empty() {
        return false;
    }
    if c.len() != e.len() {
        return false;
    }
    constant_time_eq(c.as_bytes(), e.as_bytes())
}

use crate::primitives::{AssuranceLevel, PrincipalRole, VerifiedPrincipal};

/// Validates protected route access using either an authenticated session principal
/// or a verified administrative API key.
/// Eliminates caller-supplied boolean flags to prevent authorization bypasses.
pub fn validate_protected_route(
    session_principal: Option<&VerifiedPrincipal>,
    api_key_candidate: Option<&str>,
    expected_api_key: Option<&str>,
) -> Result<PrincipalRole, String> {
    if let Some(principal) = session_principal {
        if principal.assurance_level().score() >= AssuranceLevel::STANDARD_TOKEN.score() {
            return Ok(principal.role().clone());
        }
    }

    if let (Some(cand), Some(exp)) = (api_key_candidate, expected_api_key) {
        if verify_api_key(cand, exp) {
            return Ok(PrincipalRole::operator());
        }
    }

    Err(
        "Protected route authorization rejected: Valid creator session or admin API key required"
            .into(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::{AuthorityGeneration, PrincipalId};
    use std::collections::HashMap;

    #[test]
    fn test_verify_worker_token_and_api_key() {
        assert!(verify_worker_token(
            "test_worker_cmp_aaa",
            "test_worker_cmp_aaa"
        ));
        assert!(!verify_worker_token(
            "test_worker_cmp_aaa",
            "test_worker_cmp_bbb"
        ));
        assert!(!verify_worker_token("", ""));
        assert!(!verify_worker_token("   ", "   "));

        assert!(verify_api_key(
            "test_api_cmp_valid_01",
            "test_api_cmp_valid_01"
        ));
        assert!(!verify_api_key(
            "test_api_cmp_valid_01",
            "test_api_cmp_wrong_01"
        ));
        assert!(!verify_api_key("", "key"));
    }

    #[test]
    fn test_validate_protected_route_fail_closed() {
        // 1. None / empty parameters must fail closed
        let rejected = validate_protected_route(None, None, None);
        assert!(rejected.is_err(), "Empty credentials must fail closed");

        // 2. Invalid API key must fail closed
        let bad_key = validate_protected_route(None, Some("bad_key"), Some("expected_key"));
        assert!(bad_key.is_err(), "Mismatched API key must fail closed");

        // 3. Valid API key grants operator role
        let valid_key = validate_protected_route(None, Some("correct_key"), Some("correct_key"));
        assert_eq!(valid_key.unwrap(), PrincipalRole::operator());

        // 4. Anonymous principal (< STANDARD_TOKEN) must fail closed
        let anon_pid = PrincipalId::new("anonymous_user").unwrap();
        let anon_principal = VerifiedPrincipal::construct_verified(
            anon_pid,
            PrincipalRole::new("anonymous").unwrap(),
            AssuranceLevel::ANONYMOUS,
            None,
            AuthorityGeneration::INITIAL,
            1000,
            HashMap::new(),
        )
        .unwrap();
        let anon_res = validate_protected_route(Some(&anon_principal), None, None);
        assert!(
            anon_res.is_err(),
            "Anonymous principal must be rejected on protected route"
        );

        // 5. Valid session principal (STANDARD_TOKEN) succeeds
        let user_pid = PrincipalId::new("user_bob").unwrap();
        let user_principal = VerifiedPrincipal::construct_verified(
            user_pid,
            PrincipalRole::user(),
            AssuranceLevel::STANDARD_TOKEN,
            None,
            AuthorityGeneration::INITIAL,
            1000,
            HashMap::new(),
        )
        .unwrap();
        let user_res = validate_protected_route(Some(&user_principal), None, None);
        assert_eq!(user_res.unwrap(), PrincipalRole::user());
    }
}
