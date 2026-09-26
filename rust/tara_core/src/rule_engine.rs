//! rule_engine.rs
//! 
//! Rule-First Execution Engine for TARA Core.
//! Enforces that every consequential action must be evaluated and approved
//! by the RuleEngine before PermissionEngine check, sandboxing, or execution.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::governance::Role;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuleDecision {
    ALLOW,
    DENY,
    REQUIRE_CREATOR_APPROVAL,
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
            description: "Untrusted code must run inside ExecutionSandbox with resource throttling".into(),
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
            return RuleDecision::DENY;
        }

        if action_name.starts_with("EDIT_SECURITY") || action_name.starts_with("UPDATE_CORE") {
            if actor_role == Role::CREATOR {
                return RuleDecision::ALLOW;
            } else {
                return RuleDecision::REQUIRE_CREATOR_APPROVAL;
            }
        }

        // Standard operational actions
        RuleDecision::ALLOW
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
