//! rule_engine.rs
//!
//! Rule-First Execution Engine for TARA Core.
//! Enforces that every consequential action must be evaluated and approved
//! by the RuleEngine before PermissionEngine check, sandboxing, or execution.
//!
//! # Architecture & Ownership Boundary
//! - **Scope**: Lightweight, deterministic rule evaluation engine for embedded core components.
//! - **Authoritative Multi-Tier Execution Guard**: Production-wide rule enforcement with dynamic
//!   reload, security state lockdown, and audit event dispatching is authoritatively owned
//!   by `tara_server::rules::ExecutionGuard`.

use crate::governance::Role;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuleDecision {
    Allow,
    Deny,
    RequireCreatorApproval,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    pub id: String,
    pub name: String,
    pub description: String,
    pub is_protected: bool,
    pub required_role: Role,
}

#[derive(Debug, Clone)]
pub struct RuleEngine {
    rules: HashMap<String, Rule>,
}

impl Default for RuleEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            rules: HashMap::new(),
        };
        engine.init_default_rules();
        engine
    }

    fn init_default_rules(&mut self) {
        self.register_rule(Rule {
            id: "RULE_CREATOR_SUPREME".into(),
            name: "Creator Supreme Authority".into(),
            description: "No AI or user can override Creator Authority or Root decisions".into(),
            is_protected: true,
            required_role: Role::CREATOR,
        });

        self.register_rule(Rule {
            id: "RULE_NO_SELF_PROMOTION".into(),
            name: "No Self Privilege Escalation".into(),
            description: "AI cannot grant itself permissions or alter security constraints".into(),
            is_protected: true,
            required_role: Role::CREATOR,
        });

        self.register_rule(Rule {
            id: "RULE_SANDBOX_EXECUTION".into(),
            name: "Strict Sandbox Isolation".into(),
            description: "Untrusted code must run inside ExecutionSandbox with resource throttling"
                .into(),
            is_protected: true,
            required_role: Role::ADMIN,
        });
    }

    pub fn register_rule(&mut self, mut rule: Rule) {
        rule.id = rule.id.trim().to_uppercase();
        self.rules.insert(rule.id.clone(), rule);
    }

    /// Evaluates an intent or action against rules (Rule-First Pipeline)
    pub fn evaluate_action(&self, action_name: &str, actor_role: Role) -> RuleDecision {
        let normalized = action_name.trim().to_uppercase();
        if normalized.is_empty() {
            return RuleDecision::Deny;
        }

        // Unconditionally denied destructive actions
        if normalized.starts_with("CORE_OVERWRITE")
            || normalized.starts_with("REVOKE_CREATOR")
            || normalized.contains("BYPASS_SECURITY")
        {
            return RuleDecision::Deny;
        }

        // Active enforcement of registered rules matching the action name / rule ID
        if let Some(rule) = self.rules.get(&normalized) {
            if actor_role < rule.required_role {
                return if rule.required_role == Role::CREATOR {
                    RuleDecision::RequireCreatorApproval
                } else {
                    RuleDecision::Deny
                };
            }
        }

        // High-privilege security, core modifications, rule alterations, or permission grants
        if normalized.starts_with("EDIT_SECURITY")
            || normalized.starts_with("UPDATE_CORE")
            || normalized.starts_with("MODIFY_RULE")
            || normalized.starts_with("GRANT_PERMISSION")
        {
            return if actor_role == Role::CREATOR {
                RuleDecision::Allow
            } else {
                RuleDecision::RequireCreatorApproval
            };
        }

        // Standard operational actions
        RuleDecision::Allow
    }

    /// Guardrail: Protected rules cannot be modified through learning or by AI
    pub fn can_modify_rule(&self, rule_id: &str, actor_role: Role) -> bool {
        let normalized = rule_id.trim().to_uppercase();
        if let Some(rule) = self.rules.get(&normalized) {
            if rule.is_protected {
                return actor_role == Role::CREATOR;
            }
            return actor_role >= rule.required_role;
        }
        actor_role >= Role::ADMIN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_rules_registered() {
        let engine = RuleEngine::default();
        assert!(engine.rules.contains_key("RULE_CREATOR_SUPREME"));
        assert!(engine.rules.contains_key("RULE_NO_SELF_PROMOTION"));
        assert!(engine.rules.contains_key("RULE_SANDBOX_EXECUTION"));
    }

    #[test]
    fn test_destructive_actions_denied() {
        let engine = RuleEngine::default();
        // Core overwrite, creator revocation, or security bypass is unconditionally denied even for creator
        assert_eq!(
            engine.evaluate_action("CORE_OVERWRITE_ATTEMPT", Role::CREATOR),
            RuleDecision::Deny
        );
        assert_eq!(
            engine.evaluate_action("REVOKE_CREATOR_KEY", Role::AI),
            RuleDecision::Deny
        );
        assert_eq!(
            engine.evaluate_action("TRY_BYPASS_SECURITY_NOW", Role::CREATOR),
            RuleDecision::Deny
        );
    }

    #[test]
    fn test_security_edits_require_creator() {
        let engine = RuleEngine::default();
        // Creator can edit security directly
        assert_eq!(
            engine.evaluate_action("EDIT_SECURITY_POLICY", Role::CREATOR),
            RuleDecision::Allow
        );
        // AI or Admin requires creator approval
        assert_eq!(
            engine.evaluate_action("EDIT_SECURITY_POLICY", Role::AI),
            RuleDecision::RequireCreatorApproval
        );
        assert_eq!(
            engine.evaluate_action("UPDATE_CORE_PARAMS", Role::ADMIN),
            RuleDecision::RequireCreatorApproval
        );
        assert_eq!(
            engine.evaluate_action("MODIFY_RULE_SET", Role::ADMIN),
            RuleDecision::RequireCreatorApproval
        );
        assert_eq!(
            engine.evaluate_action("GRANT_PERMISSION_ALL", Role::AI),
            RuleDecision::RequireCreatorApproval
        );
        assert_eq!(
            engine.evaluate_action("MODIFY_RULE_SET", Role::CREATOR),
            RuleDecision::Allow
        );
    }

    #[test]
    fn test_protected_rules_cannot_be_modified_by_ai() {
        let engine = RuleEngine::default();
        assert!(!engine.can_modify_rule("RULE_CREATOR_SUPREME", Role::AI));
        assert!(!engine.can_modify_rule("RULE_CREATOR_SUPREME", Role::ADMIN));
        assert!(engine.can_modify_rule("RULE_CREATOR_SUPREME", Role::CREATOR));
    }

    #[test]
    fn test_unprotected_rule_modification_respects_required_role() {
        let mut engine = RuleEngine::default();
        engine.register_rule(Rule {
            id: "CUSTOM_USER_RULE".into(),
            name: "Custom User Rule".into(),
            description: "Unprotected user rule".into(),
            is_protected: false,
            required_role: Role::ADMIN,
        });

        // USER role cannot modify an ADMIN required rule
        assert!(!engine.can_modify_rule("CUSTOM_USER_RULE", Role::USER));
        // ADMIN can modify it
        assert!(engine.can_modify_rule("CUSTOM_USER_RULE", Role::ADMIN));
        // Non-existent rule defaults to ADMIN requirement
        assert!(!engine.can_modify_rule("NON_EXISTENT_RULE", Role::USER));
        assert!(engine.can_modify_rule("NON_EXISTENT_RULE", Role::ADMIN));
    }

    #[test]
    fn test_rule_normalization_and_empty_action_denial() {
        let mut engine = RuleEngine::default();
        // Empty action is denied
        assert_eq!(engine.evaluate_action("", Role::CREATOR), RuleDecision::Deny);
        assert_eq!(engine.evaluate_action("   ", Role::CREATOR), RuleDecision::Deny);

        // Lowercase and whitespace normalization
        assert_eq!(
            engine.evaluate_action("  core_overwrite_test  ", Role::CREATOR),
            RuleDecision::Deny
        );

        // Registered rule enforcement
        engine.register_rule(Rule {
            id: "custom_admin_action".into(),
            name: "Custom Admin Action".into(),
            description: "Requires admin role".into(),
            is_protected: false,
            required_role: Role::ADMIN,
        });

        // Denied for USER
        assert_eq!(
            engine.evaluate_action("CUSTOM_ADMIN_ACTION", Role::USER),
            RuleDecision::Deny
        );
        // Allowed for ADMIN
        assert_eq!(
            engine.evaluate_action("custom_admin_action", Role::ADMIN),
            RuleDecision::Allow
        );
    }
}
