//! authorization/mod.rs
//!
//! Automatic authorization gates for TARA Dynamic Runtime.
//! Evaluates entity identity, task capability requirements, sandbox isolation bounds,
//! security state (Lockdown/Quarantine), and active security policies before any execution.

use crate::runtime::lifecycle::EntityState;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionType {
    CreateSandbox,
    StartSandbox,
    ExecuteWorkload,
    AssignAgent,
    AssignWorker,
    AccessTool(String),
    RequestNetwork(String),
    TransferData,
    ModifyPolicy,
    AdminOperation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthorizationResult {
    Allowed,
    Denied {
        reason: String,
        security_violation: bool,
    },
}

impl AuthorizationResult {
    pub fn is_allowed(&self) -> bool {
        matches!(self, AuthorizationResult::Allowed)
    }
}

pub struct AuthorizationGate;

impl AuthorizationGate {
    /// Authorizes an operation across caller identity, target sandbox, requested capabilities, and system state.
    pub fn evaluate(
        actor_id: &str,
        actor_state: EntityState,
        action: &ActionType,
        target_sandbox_id: Option<&str>,
        sandbox_state: Option<EntityState>,
        caller_capabilities: &HashSet<String>,
        is_system_lockdown: bool,
    ) -> AuthorizationResult {
        // 1. In lockdown, deny all non-admin execution
        if is_system_lockdown && !matches!(action, ActionType::AdminOperation) {
            return AuthorizationResult::Denied {
                reason: "System is in LOCKDOWN mode. Non-administrative execution denied."
                    .to_string(),
                security_violation: true,
            };
        }

        // 2. Quarantined or Revoked entities cannot execute actions
        if matches!(
            actor_state,
            EntityState::Quarantined | EntityState::Revoked | EntityState::Failed
        ) {
            return AuthorizationResult::Denied {
                reason: format!("Actor '{}' is in invalid state '{}'", actor_id, actor_state),
                security_violation: true,
            };
        }

        // 3. Target sandbox state check
        if let Some(s_state) = sandbox_state {
            if matches!(
                s_state,
                EntityState::Quarantined | EntityState::Revoked | EntityState::Failed
            ) {
                return AuthorizationResult::Denied {
                    reason: format!(
                        "Target sandbox '{:?}' is in non-executable state '{}'",
                        target_sandbox_id, s_state
                    ),
                    security_violation: true,
                };
            }
        }

        // 4. Action-specific capability evaluation
        match action {
            ActionType::AccessTool(tool_name) => {
                let required_cap = format!("tool:{}", tool_name);
                if !caller_capabilities.contains(&required_cap)
                    && !caller_capabilities.contains("*")
                {
                    return AuthorizationResult::Denied {
                        reason: format!(
                            "Actor '{}' lacks required capability '{}'",
                            actor_id, required_cap
                        ),
                        security_violation: false,
                    };
                }
            }
            ActionType::RequestNetwork(domain) => {
                let net_cap = "network:outbound".to_string();
                if !caller_capabilities.contains(&net_cap) && !caller_capabilities.contains("*") {
                    return AuthorizationResult::Denied {
                        reason: format!(
                            "Actor '{}' denied network access to domain '{}'",
                            actor_id, domain
                        ),
                        security_violation: false,
                    };
                }
            }
            ActionType::ModifyPolicy | ActionType::AdminOperation
                if !caller_capabilities.contains("admin:root")
                    && !caller_capabilities.contains("*") =>
            {
                return AuthorizationResult::Denied {
                    reason: format!("Actor '{}' lacks administrative privileges", actor_id),
                    security_violation: true,
                };
            }
            _ => {}
        }

        AuthorizationResult::Allowed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_authorization_lockdown_denial() {
        let mut caps = HashSet::new();
        caps.insert("tool:file_inspector".to_string());

        let res = AuthorizationGate::evaluate(
            "actor_1",
            EntityState::Ready,
            &ActionType::AccessTool("file_inspector".to_string()),
            None,
            None,
            &caps,
            true, // is_system_lockdown
        );

        assert!(!res.is_allowed());
        if let AuthorizationResult::Denied { reason, security_violation } = res {
            assert!(reason.contains("LOCKDOWN"));
            assert!(security_violation);
        } else {
            panic!("Expected denial in lockdown");
        }
    }

    #[test]
    fn test_authorization_quarantined_actor_denial() {
        let caps = HashSet::new();
        let res = AuthorizationGate::evaluate(
            "actor_quarantine",
            EntityState::Quarantined,
            &ActionType::ExecuteWorkload,
            None,
            None,
            &caps,
            false,
        );
        assert!(!res.is_allowed());
    }

    #[test]
    fn test_authorization_capability_check() {
        let mut caps = HashSet::new();
        caps.insert("tool:allowed_tool".to_string());

        let ok_res = AuthorizationGate::evaluate(
            "actor_cap",
            EntityState::Ready,
            &ActionType::AccessTool("allowed_tool".to_string()),
            None,
            None,
            &caps,
            false,
        );
        assert!(ok_res.is_allowed());

        let deny_res = AuthorizationGate::evaluate(
            "actor_cap",
            EntityState::Ready,
            &ActionType::AccessTool("denied_tool".to_string()),
            None,
            None,
            &caps,
            false,
        );
        assert!(!deny_res.is_allowed());
    }
}

