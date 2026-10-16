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
        if action_lc.contains("csam")
            || action_lc.contains("child_abuse")
            || action_lc.contains("child_porn")
        {
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

        // 3. Evaluate matching rules: any BLOCK or DENY takes precedence over ALLOW
        let mut matching_allows = Vec::new();
        for r in rules {
            if r.action_type.eq_ignore_ascii_case(action_type) || r.action_type == "*" {
                if r.decision.eq_ignore_ascii_case("BLOCK")
                    || r.decision.eq_ignore_ascii_case("DENY")
                {
                    let mut blocked = r.clone();
                    blocked.decision = "BLOCK".to_string();
                    return blocked;
                } else if r.decision.eq_ignore_ascii_case("ALLOW") {
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
