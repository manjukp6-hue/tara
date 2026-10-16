//! Dynamic, Extensible Deny-by-Default Authorization Policy Engine.
//!
//! Rule compliance:
//! - Default decision is strictly DENY.
//! - Dynamic rule registry allows extending permissions without code recompilation.
//! - Checks lockdown state and minimum assurance levels.

use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

use crate::primitives::{
    AssuranceLevel, PrincipalRole, SecurityAction, SecurityContext, SecurityResource, SecurityState,
    VerifiedPrincipal,
};

/// Denial reason explaining why an operation was rejected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityDenial {
    SystemLockedDown { reason: String },
    AuthorityGenerationMismatch { expected: u64, actual: u64 },
    InsufficientAssurance { required: u32, actual: u32 },
    RoleUnauthorized { required: String, actual: String },
    ExplicitDeny { rule_id: String, reason: String },
    NoMatchingAllowRule,
}

impl std::fmt::Display for SecurityDenial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SystemLockedDown { reason } => write!(f, "SystemLockedDown: {reason}"),
            Self::AuthorityGenerationMismatch { expected, actual } => {
                write!(f, "AuthorityGenerationMismatch: expected gen {expected}, got {actual}")
            }
            Self::InsufficientAssurance { required, actual } => {
                write!(f, "InsufficientAssurance: required {required}, actual {actual}")
            }
            Self::RoleUnauthorized { required, actual } => {
                write!(f, "RoleUnauthorized: required {required}, actual {actual}")
            }
            Self::ExplicitDeny { rule_id, reason } => write!(f, "ExplicitDeny[{rule_id}]: {reason}"),
            Self::NoMatchingAllowRule => write!(f, "DefaultDeny: No matching policy rule granted authorization"),
        }
    }
}

/// A dynamic, configurable policy rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    pub rule_id: String,
    pub description: String,
    pub target_role: Option<String>,
    pub action_pattern: String,   // e.g. "core:*", "storage:read"
    pub resource_pattern: String, // e.g. "tara:res:system_vault", "tara:res:fs:*"
    pub min_assurance_level: u32,
    pub allow: bool,
}

/// Dynamic Policy Engine holding active rules and evaluating authorization decisions.
pub struct PolicyEngine {
    rules: Arc<RwLock<Vec<PolicyRule>>>,
}

impl Default for PolicyEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl PolicyEngine {
    pub fn new() -> Self {
        let default_rules = vec![
            // Policy Rule: Creator Root can perform any action on any resource
            PolicyRule {
                rule_id: "rule_creator_full_access".into(),
                description: "Creator root authority has full access to all resources".into(),
                target_role: Some(PrincipalRole::creator().as_str().to_string()),
                action_pattern: "*:*".into(),
                resource_pattern: "*".into(),
                min_assurance_level: AssuranceLevel::CREATOR_ROOT.score(),
                allow: true,
            },
            // Policy Rule: Hardware-attested Operators can execute tasks
            PolicyRule {
                rule_id: "rule_operator_execute_tasks".into(),
                description: "Attested operators can execute tasks".into(),
                target_role: Some(PrincipalRole::operator().as_str().to_string()),
                action_pattern: "core:execute_task".into(),
                resource_pattern: "*".into(),
                min_assurance_level: AssuranceLevel::HARDWARE_ATTESTED.score(),
                allow: true,
            },
        ];

        Self {
            rules: Arc::new(RwLock::new(default_rules)),
        }
    }

    /// Register a system baseline rule during engine initialization (governed by internal bootstrap).
    pub fn add_system_rule(&self, rule: PolicyRule) -> Result<(), String> {
        if rule.rule_id.trim().is_empty() {
            return Err("InvalidPolicyRule: rule_id cannot be empty".into());
        }
        if rule.action_pattern.trim().is_empty() || rule.resource_pattern.trim().is_empty() {
            return Err("InvalidPolicyRule: Action and resource patterns cannot be empty".into());
        }
        let mut rules = self.rules.write().map_err(|_| "Policy lock poisoned (FailClosed)")?;
        rules.push(rule);
        Ok(())
    }

    /// Add a dynamic policy rule at runtime, protected by governance authorization (ID-20).
    pub fn add_rule(
        &self,
        authorizer: &VerifiedPrincipal,
        rule: PolicyRule,
    ) -> Result<(), String> {
        // Privileged governance gate (ID-20): Only Creator or Hardware-Attested Operator can add rules
        if authorizer.role() != &PrincipalRole::creator()
            && authorizer.assurance_level().score() < AssuranceLevel::HARDWARE_ATTESTED.score()
        {
            return Err("UnauthorizedPolicyUpdate: Adding policy rules requires Creator or Hardware-Attested Operator".into());
        }

        if rule.rule_id.trim().is_empty() {
            return Err("InvalidPolicyRule: rule_id cannot be empty".into());
        }
        if rule.action_pattern.trim().is_empty() || rule.resource_pattern.trim().is_empty() {
            return Err("InvalidPolicyRule: Action and resource patterns cannot be empty".into());
        }

        let mut rules = self.rules.write().map_err(|_| "Policy lock poisoned (FailClosed)")?;
        rules.push(rule);
        Ok(())
    }

    /// Authorize a requested operation: strictly DENY by default with mandatory DENY PRECEDENCE (ID-06).
    pub fn authorize(
        &self,
        security_state: &SecurityState,
        principal: &VerifiedPrincipal,
        action: &SecurityAction,
        resource: &SecurityResource,
        _context: &SecurityContext,
    ) -> Result<(), SecurityDenial> {
        // 1. Strict state & generation check at boundary
        match security_state {
            SecurityState::LockedDown { reason, .. } => {
                return Err(SecurityDenial::SystemLockedDown {
                    reason: reason.clone(),
                });
            }
            SecurityState::Operational { generation } => {
                if principal.authority_generation() != *generation {
                    return Err(SecurityDenial::AuthorityGenerationMismatch {
                        expected: generation.get(),
                        actual: principal.authority_generation().get(),
                    });
                }
            }
            _ => {
                return Err(SecurityDenial::SystemLockedDown {
                    reason: "Security subsystem is not in Operational state (FailClosed)".into(),
                });
            }
        }

        let rules = self
            .rules
            .read()
            .map_err(|_| SecurityDenial::SystemLockedDown {
                reason: "Policy lock poisoned (FailClosed)".into(),
            })?;

        let action_name = action.canonical_name();
        let resource_urn = resource.resource_urn();
        let role_name = principal.role().as_str();

        // 3. PASS 1: Strict Explicit Deny Precedence (ID-06)
        // If ANY matching rule explicitly denies the operation, it overrides ALL allow rules!
        for rule in rules.iter() {
            if rule.allow {
                continue;
            }

            // Check role match
            if let Some(ref target) = rule.target_role {
                if !target.eq_ignore_ascii_case(role_name) && target != "*" {
                    continue;
                }
            }

            // Check action & resource match
            if matches_pattern(&rule.action_pattern, &action_name)
                && matches_pattern(&rule.resource_pattern, &resource_urn)
            {
                return Err(SecurityDenial::ExplicitDeny {
                    rule_id: rule.rule_id.clone(),
                    reason: rule.description.clone(),
                });
            }
        }

        // 4. PASS 2: Evaluate Allow Rules
        let mut matched_allow = false;
        let mut insufficient_assurance_error = None;

        for rule in rules.iter() {
            if !rule.allow {
                continue;
            }

            // Check role match
            if let Some(ref target) = rule.target_role {
                if !target.eq_ignore_ascii_case(role_name) && target != "*" {
                    continue;
                }
            }

            // Check action & resource match
            if !matches_pattern(&rule.action_pattern, &action_name)
                || !matches_pattern(&rule.resource_pattern, &resource_urn)
            {
                continue;
            }

            // Check assurance level
            if principal.assurance_level().score() < rule.min_assurance_level {
                insufficient_assurance_error = Some(SecurityDenial::InsufficientAssurance {
                    required: rule.min_assurance_level,
                    actual: principal.assurance_level().score(),
                });
                continue;
            }

            matched_allow = true;
            break;
        }

        if matched_allow {
            Ok(())
        } else if let Some(err) = insufficient_assurance_error {
            Err(err)
        } else {
            Err(SecurityDenial::NoMatchingAllowRule)
        }
    }
}

fn matches_pattern(pattern: &str, target: &str) -> bool {
    if pattern == "*" || pattern == "*:*" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        target.starts_with(prefix)
    } else {
        pattern == target
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::{AuthorityGeneration, PrincipalId};
    use std::collections::HashMap;

    #[test]
    fn test_policy_fail_closed_on_non_operational_states() {
        let engine = PolicyEngine::new();
        let gen = AuthorityGeneration::new(1).unwrap();
        let pid = PrincipalId::new("operator_test").unwrap();
        let principal = VerifiedPrincipal::construct_verified(
            pid,
            PrincipalRole::operator(),
            AssuranceLevel::HARDWARE_ATTESTED,
            None,
            gen,
            1_700_000_000,
            HashMap::new(),
        )
        .unwrap();

        let action = SecurityAction::new("core", "execute_task").unwrap();
        let resource = SecurityResource::custom("task", "task_1").unwrap();
        let context = SecurityContext::new("127.0.0.1", 1_700_000_000);

        // 1. Uninitialized security state MUST fail closed
        let uninit_res = engine.authorize(
            &SecurityState::Uninitialized,
            &principal,
            &action,
            &resource,
            &context,
        );
        assert!(matches!(uninit_res, Err(SecurityDenial::SystemLockedDown { .. })));

        // 2. Recovering security state MUST fail closed
        let recovering_state = SecurityState::Recovering {
            target_generation: gen,
            nonce: "recovery_nonce_123".into(),
        };
        let rec_res = engine.authorize(
            &recovering_state,
            &principal,
            &action,
            &resource,
            &context,
        );
        assert!(matches!(rec_res, Err(SecurityDenial::SystemLockedDown { .. })));

        // 3. LockedDown security state MUST fail closed
        let locked_state = SecurityState::LockedDown {
            generation: gen,
            reason: "Emergency tripwire triggered".into(),
            since_epoch_secs: 1_700_000_000,
        };
        let lock_res = engine.authorize(
            &locked_state,
            &principal,
            &action,
            &resource,
            &context,
        );
        assert!(matches!(lock_res, Err(SecurityDenial::SystemLockedDown { .. })));

        // 4. Operational state with authority generation mismatch MUST fail closed
        let next_gen = gen.next().unwrap();
        let stale_state = SecurityState::Operational { generation: next_gen };
        let gen_mismatch_res = engine.authorize(
            &stale_state,
            &principal,
            &action,
            &resource,
            &context,
        );
        assert!(matches!(
            gen_mismatch_res,
            Err(SecurityDenial::AuthorityGenerationMismatch { .. })
        ));

        // 5. Operational state with matching generation MUST succeed on allowed rule
        let valid_state = SecurityState::Operational { generation: gen };
        let ok_res = engine.authorize(
            &valid_state,
            &principal,
            &action,
            &resource,
            &context,
        );
        assert!(ok_res.is_ok());
    }
}

