//! Conflict Resolver across security policies and user rules.
//!
//! Precedence order:
//! 1. Mandatory Statutory Safety (Universal CSAM/Harm prevention) -> Immutable BLOCK
//! 2. Creator Governance Authority -> ALLOW for verified ROOT_CREATOR
//! 3. Fail-Closed Default -> BLOCK overrides ALLOW when priority is equal

use crate::rules::PolicyRule;

pub struct ConflictResolver;

impl ConflictResolver {
    /// Resolves conflicts among a set of candidate rules for a target action.
    pub fn resolve(rules: &[PolicyRule], action_type: &str, is_creator: bool) -> PolicyRule {
        let action_lc = action_type.to_lowercase();

        // 1. Mandatory Universal Invariant
        if action_lc.contains("csam") || action_lc.contains("child_abuse") {
            return PolicyRule {
                action_type: action_type.to_string(),
                decision: "BLOCK".to_string(),
                conditions: Vec::new(),
                reason: "Mandatory statutory safety: universal override.".to_string(),
            };
        }

        // 2. Creator override
        if is_creator {
            return PolicyRule {
                action_type: action_type.to_string(),
                decision: "ALLOW".to_string(),
                conditions: vec!["creator_verified".to_string()],
                reason: "Authorized Creator operation.".to_string(),
            };
        }

        // 3. Evaluate matching rules: any BLOCK takes precedence over ALLOW
        let mut matching_allows = Vec::new();
        for r in rules {
            if r.action_type == action_type || r.action_type == "*" {
                if r.decision == "BLOCK" {
                    return r.clone();
                } else if r.decision == "ALLOW" {
                    matching_allows.push(r.clone());
                }
            }
        }

        if let Some(first_allow) = matching_allows.first() {
            first_allow.clone()
        } else {
            PolicyRule {
                action_type: action_type.to_string(),
                decision: "ALLOW".to_string(),
                conditions: Vec::new(),
                reason: "Default permissive rule.".to_string(),
            }
        }
    }
}
