//! rule_engine.rs
//!
//! Rule-First Execution Engine for TARA Core.
//! Enforces that every consequential action must be evaluated and approved
//! by the RuleEngine before PermissionEngine check, sandboxing, or execution.

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

    pub fn register_rule(&mut self, rule: Rule) {
        self.rules.insert(rule.id.clone(), rule);
    }

    /// Evaluates an intent or action against rules (Rule-First Pipeline)
    pub fn evaluate_action(&self, action_name: &str, actor_role: Role) -> RuleDecision {
        // High-risk or protected actions require Creator
        if action_name.starts_with("CORE_OVERWRITE") || action_name.starts_with("REVOKE_CREATOR") {
            return RuleDecision::Deny;
        }

        if action_name.starts_with("EDIT_SECURITY") || action_name.starts_with("UPDATE_CORE") {
            if actor_role == Role::CREATOR {
                return RuleDecision::Allow;
            } else {
                return RuleDecision::RequireCreatorApproval;
            }
        }

        // Standard operational actions
        RuleDecision::Allow
    }

    /// Guardrail: Protected rules cannot be modified through learning or by AI
    pub fn can_modify_rule(&self, rule_id: &str, actor_role: Role) -> bool {
        if let Some(rule) = self.rules.get(rule_id) {
            if rule.is_protected {
                return actor_role == Role::CREATOR;
            }
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
        // Core overwrite or creator revocation is unconditionally denied even for creator
        assert_eq!(
            engine.evaluate_action("CORE_OVERWRITE_ATTEMPT", Role::CREATOR),
            RuleDecision::Deny
        );
        assert_eq!(
            engine.evaluate_action("REVOKE_CREATOR_KEY", Role::AI),
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
    }

    #[test]
    fn test_protected_rules_cannot_be_modified_by_ai() {
        let engine = RuleEngine::default();
        assert!(!engine.can_modify_rule("RULE_CREATOR_SUPREME", Role::AI));
        assert!(!engine.can_modify_rule("RULE_CREATOR_SUPREME", Role::ADMIN));
        assert!(engine.can_modify_rule("RULE_CREATOR_SUPREME", Role::CREATOR));
    }
}
