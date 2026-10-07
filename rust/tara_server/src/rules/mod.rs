//! Rules subsystem: policy engine and execution guard.

use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;

/// Guard decision.
#[derive(Debug, Clone)]
pub struct GuardResult {
    pub decision: String, // "ALLOW" or "BLOCK"
    pub reason: String,
    pub creator_verified: bool,
}

/// A single policy rule.
#[derive(Debug, Clone)]
pub struct PolicyRule {
    pub action_type: String,
    pub decision: String,
    pub conditions: Vec<String>,
    pub reason: String,
}

/// Compiled policy loaded from JSON.
#[derive(Debug, Clone)]
pub struct CompiledPolicy {
    pub rules: Vec<PolicyRule>,
}

impl CompiledPolicy {
    pub fn from_json_file(path: &str) -> Result<Self, String> {
        let raw = fs::read_to_string(path).map_err(|e| e.to_string())?;
        let v: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
        let mut rules = Vec::new();
        if let Some(arr) = v.get("rules").and_then(|a| a.as_array()) {
            for rule in arr {
                let action_type = rule
                    .get("action_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("*")
                    .to_string();
                let decision = rule
                    .get("decision")
                    .and_then(|v| v.as_str())
                    .unwrap_or("ALLOW")
                    .to_string();
                let conditions = rule
                    .get("conditions")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|c| c.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                let reason = rule
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                rules.push(PolicyRule {
                    action_type,
                    decision,
                    conditions,
                    reason,
                });
            }
        }
        Ok(Self { rules })
    }

    pub fn to_json(&self) -> Value {
        json!({
            "rules": self.rules.iter().map(|r| json!({
                "action_type": r.action_type,
                "decision": r.decision,
                "conditions": r.conditions,
                "reason": r.reason
            })).collect::<Vec<_>>()
        })
    }
}

pub mod conflict_resolver;
pub mod language_detector;
pub mod policy_compiler;
pub mod policy_schema;
pub mod rule_parser;
pub mod rulebook_manager;
pub mod semantic_normalizer;
pub mod validator;

pub use conflict_resolver::ConflictResolver;
pub use language_detector::{LanguageDetectionResult, LanguageDetector};
pub use policy_compiler::PolicyCompiler;
pub use policy_schema::{
    PolicyPriority, RuleAction, RuleCategory, RuleItem, StructuredCompiledPolicy,
};
pub use rule_parser::RuleParser;
pub use rulebook_manager::RulebookManager;
pub use semantic_normalizer::SemanticNormalizer;
pub use validator::PolicyValidator;

/// Rule-based execution guard (policy enforcement).
pub struct ExecutionGuard {
    pub policy: Option<CompiledPolicy>,
    pub identity_manager: Option<Value>,
}

impl ExecutionGuard {
    pub fn new(policy: Option<CompiledPolicy>) -> Self {
        Self {
            policy,
            identity_manager: None,
        }
    }

    pub fn set_policy(&mut self, policy: CompiledPolicy) {
        self.policy = Some(policy);
    }

    pub fn set_identity_manager(&mut self, manager: Value) {
        self.identity_manager = Some(manager);
    }

    pub fn enforce_creator_confirmation(
        &self,
        action: &str,
        is_creator: bool,
    ) -> Result<(), String> {
        if !is_creator {
            Err(format!(
                "Access Denied: Action '{}' requires authenticated Creator confirmation",
                action
            ))
        } else {
            Ok(())
        }
    }

    /// Evaluate whether `action_type` should be ALLOWED or BLOCKED.
    /// - Creator: Unrestricted governance authority ("creatore no rule" - user restrictions do not apply).
    /// - Regular User: Strictly restricted to conversational/multimodal chat only. Zero edit, update, delete, or system configuration rights.
    pub fn evaluate_action(
        &self,
        action_type: &str,
        context: &HashMap<String, Value>,
    ) -> GuardResult {
        let is_creator = context
            .get("creator_verified")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let action_lower = action_type.to_lowercase();

        // 1. Mandatory Statutory Safety (Universal: CSAM and child sexual abuse materials are always blocked)
        if action_lower.contains("csam")
            || action_lower.contains("child_abuse")
            || action_lower.contains("child_porn")
        {
            return GuardResult {
                decision: "BLOCK".to_string(),
                reason: "Mandatory statutory safety: CSAM and child sexual abuse materials are strictly prohibited.".to_string(),
                creator_verified: is_creator,
            };
        }

        // 2. Creator Authority: Creator possesses unconstrained governance authority
        if is_creator {
            return GuardResult {
                decision: "ALLOW".to_string(),
                reason: "Authorized Creator operation.".to_string(),
                creator_verified: true,
            };
        }

        // 3. Regular User Restrictions:
        // Users are strictly confined to CONVERSATIONAL and multimodal chat/media interactions.
        // Any attempt by a regular user to edit, update, modify machine/config/rules, delete files,
        // dump secrets, or execute mutation tools is strictly BLOCKED.
        let guarded_user_actions = [
            "modify",
            "update",
            "edit",
            "delete",
            "remove",
            "alter",
            "modify_machine_config",
            "system_reconfigure",
            "alter_system_settings",
            "autonomous_evolution",
            "model_expansion",
            "export_key",
            "dump_credentials",
            "delete_file",
            "rule_update",
            "add_rule",
            "delete_rule",
            "execute_tool",
            "execute_skill",
            "guarded_action",
            "source_code_evolution",
            "add_source_function",
            "remove_source_function",
            "source_evolution",
            "code_evolution",
            "add_source",
            "code_edit",
        ];

        if guarded_user_actions
            .iter()
            .any(|&g| action_lower.contains(g))
        {
            return GuardResult {
                decision: "BLOCK".to_string(),
                reason: "User permission restricted: Only conversational/multimodal chat interactions are permitted for regular users. System edit, update, delete, and configuration actions are strictly restricted to the authorized Creator.".to_string(),
                creator_verified: false,
            };
        }

        // 4. Evaluate compiled policy rules if defined
        if let Some(ref policy) = self.policy {
            for rule in &policy.rules {
                if rule.action_type == action_type || rule.action_type == "*" {
                    let conditions_met = rule.conditions.is_empty()
                        || rule.conditions.iter().all(|cond| {
                            let cond_lc = cond.to_lowercase();
                            if cond_lc.contains("creator_verified") {
                                is_creator
                            } else {
                                true
                            }
                        });
                    if conditions_met && rule.decision == "BLOCK" {
                        return GuardResult {
                            decision: rule.decision.clone(),
                            reason: rule.reason.clone(),
                            creator_verified: is_creator,
                        };
                    }
                }
            }
        }

        // Default: Conversational chat allowed
        GuardResult {
            decision: "ALLOW".to_string(),
            reason: "Conversational access permitted".to_string(),
            creator_verified: false,
        }
    }

    pub fn get_policy_json(&self) -> Value {
        self.policy
            .as_ref()
            .map(|p| p.to_json())
            .unwrap_or(json!({ "rules": [] }))
    }
}
