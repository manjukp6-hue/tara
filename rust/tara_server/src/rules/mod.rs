//! Rules subsystem: policy engine and execution guard.

use std::collections::HashMap;
use std::fs;
use serde_json::{json, Value};

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
                let action_type = rule.get("action_type").and_then(|v| v.as_str()).unwrap_or("*").to_string();
                let decision = rule.get("decision").and_then(|v| v.as_str()).unwrap_or("ALLOW").to_string();
                let conditions = rule.get("conditions").and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|c| c.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                let reason = rule.get("reason").and_then(|v| v.as_str()).unwrap_or("").to_string();
                rules.push(PolicyRule { action_type, decision, conditions, reason });
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

/// Rule-based execution guard (policy enforcement).
pub struct ExecutionGuard {
    pub policy: Option<CompiledPolicy>,
}

impl ExecutionGuard {
    pub fn new(policy: Option<CompiledPolicy>) -> Self {
        Self { policy }
    }

    /// Evaluate whether `action_type` should be ALLOWED or BLOCKED.
    pub fn evaluate_action(&self, action_type: &str, context: &HashMap<String, Value>) -> GuardResult {
        if let Some(ref policy) = self.policy {
            for rule in &policy.rules {
                if rule.action_type == action_type || rule.action_type == "*" {
                    let conditions_met = rule.conditions.is_empty() || rule.conditions.iter().all(|cond| {
                        let cond_lc = cond.to_lowercase();
                        if cond_lc.contains("creator_verified") {
                            context.get("creator_verified")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false)
                        } else {
                            true
                        }
                    });
                    if conditions_met {
                        return GuardResult {
                            decision: rule.decision.clone(),
                            reason: rule.reason.clone(),
                            creator_verified: context.get("creator_verified")
                                .and_then(|v| v.as_bool()).unwrap_or(false),
                        };
                    }
                }
            }
        }
        // Default: ALLOW
        GuardResult {
            decision: "ALLOW".to_string(),
            reason: "Default policy: ALLOW".to_string(),
            creator_verified: context.get("creator_verified")
                .and_then(|v| v.as_bool()).unwrap_or(false),
        }
    }

    pub fn get_policy_json(&self) -> Value {
        self.policy.as_ref().map(|p| p.to_json()).unwrap_or(json!({ "rules": [] }))
    }
}
