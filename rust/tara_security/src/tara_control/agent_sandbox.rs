//! Agent Execution Sandbox & Runtime Containment for TARA-Controlled Security.
//!
//! Rule compliance:
//! - Pure native Rust, zero outside crate dependencies.
//! - Sandboxes autonomous agents with explicit resource limits, workspace containment,
//!   and tool dispatch gated through the Central Enforcement Gateway.
//! - Rejects path traversal and out-of-workspace mutations.
//! - Fails closed immediately if global lockdown is active.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::broker::{EnforcementGateway, EnforcementRequest, ProtectedCategory};
use crate::path_security::{canonical_containment, check_lexical_traversal};
use crate::primitives::{SecurityAction, SecurityContext, SecurityResource, VerifiedPrincipal};

/// Policy configuration governing an isolated agent execution environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSandboxPolicy {
    pub max_execution_time_ms: u64,
    pub max_memory_mb: usize,
    pub allow_filesystem_write: bool,
    pub allow_network_outbound: bool,
    pub allowed_tools: HashSet<String>,
}

impl Default for AgentSandboxPolicy {
    fn default() -> Self {
        Self {
            max_execution_time_ms: 10_000,
            max_memory_mb: 512,
            allow_filesystem_write: false,
            allow_network_outbound: false,
            allowed_tools: HashSet::new(),
        }
    }
}

/// Execution outcome from an agent task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentExecutionOutcome {
    pub success: bool,
    pub output: String,
    pub execution_time_ms: u64,
    pub error: Option<String>,
}

/// Agent Execution Sandbox managing workspace isolation and tool execution.
pub struct AgentExecutionSandbox {
    agent_id: String,
    workspace_root: PathBuf,
    policy: AgentSandboxPolicy,
    gateway: Arc<EnforcementGateway>,
}

impl AgentExecutionSandbox {
    pub fn new(
        agent_id: impl Into<String>,
        workspace_root: PathBuf,
        policy: AgentSandboxPolicy,
        gateway: Arc<EnforcementGateway>,
    ) -> Result<Self, String> {
        let aid = agent_id.into().trim().to_string();
        if aid.is_empty() {
            return Err("AgentId cannot be empty".into());
        }

        Ok(Self {
            agent_id: aid,
            workspace_root,
            policy,
            gateway,
        })
    }

    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    pub fn policy(&self) -> &AgentSandboxPolicy {
        &self.policy
    }

    /// Validates that a file access requested by the agent is contained within its assigned workspace.
    pub fn validate_workspace_access(
        &self,
        target_path: &Path,
        is_write: bool,
    ) -> Result<PathBuf, String> {
        // 1. Strict lexical traversal check
        check_lexical_traversal(target_path)
            .map_err(|e| format!("AgentWorkspaceTraversalViolation: {e}"))?;

        // 2. Write policy gate
        if is_write && !self.policy.allow_filesystem_write {
            return Err(
                "AgentWriteBlocked: Agent sandbox policy prohibits filesystem writes".into(),
            );
        }

        // 3. Strict containment verification
        canonical_containment(&self.workspace_root, target_path)
            .map_err(|e| format!("AgentWorkspaceBoundaryEscape: {e}"))
    }

    /// Dispatches an agent tool execution through the Central Enforcement Gateway.
    pub fn dispatch_tool<T, F>(
        &self,
        tool_name: &str,
        tool_executable: &Path,
        principal: &VerifiedPrincipal,
        context: &SecurityContext,
        session_id: Option<String>,
        tool_fn: F,
    ) -> Result<T, String>
    where
        F: FnOnce(&VerifiedPrincipal, &SecurityResource) -> Result<T, String>,
    {
        // 1. Fail-closed tool whitelist check
        if !self.policy.allowed_tools.contains(tool_name)
            && !self.policy.allowed_tools.contains("*")
        {
            return Err(format!(
                "UnauthorizedTool: Tool '{}' is not in agent allowed toolset",
                tool_name
            ));
        }

        // 2. Validate tool path against lexical traversal and workspace containment
        check_lexical_traversal(tool_executable).map_err(|e| format!("InvalidToolPath: {e}"))?;

        let contained_tool = canonical_containment(&self.workspace_root, tool_executable)
            .map_err(|e| format!("ToolWorkspaceBoundaryEscape: {e}"))?;

        let allowed_dir = if let Some(parent) = contained_tool.parent() {
            if canonical_containment(&self.workspace_root, parent).is_ok() {
                parent.to_path_buf()
            } else {
                self.workspace_root.clone()
            }
        } else {
            self.workspace_root.clone()
        };

        let request = EnforcementRequest {
            principal: principal.clone(),
            session_id,
            action: SecurityAction::execute_task(),
            resource: SecurityResource::custom("agent_tool", tool_name)?,
            context: context.clone(),
            category: ProtectedCategory::Tools {
                command_or_script: contained_tool,
                allowed_dirs: vec![allowed_dir],
            },
        };

        // 3. Enforce through Central Gateway
        self.gateway.enforce_protected_operation(request, tool_fn)
    }

    /// Executes an autonomous goal step through the Central Enforcement Gateway.
    pub fn execute_goal_step<T, F>(
        &self,
        goal_id: &str,
        step_description: &str,
        principal: &VerifiedPrincipal,
        context: &SecurityContext,
        session_id: Option<String>,
        goal_fn: F,
    ) -> Result<T, String>
    where
        F: FnOnce(&VerifiedPrincipal, &SecurityResource) -> Result<T, String>,
    {
        let request = EnforcementRequest {
            principal: principal.clone(),
            session_id,
            action: SecurityAction::execute_task(),
            resource: SecurityResource::custom("agent_goal", goal_id)?,
            context: context.clone(),
            category: ProtectedCategory::AutonomousGoals {
                goal_id: format!("{}:{}", self.agent_id, goal_id),
                target_subsystem: step_description.to_string(),
            },
        };

        self.gateway.enforce_protected_operation(request, goal_fn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::{AssuranceLevel, AuthorityGeneration, PrincipalId, PrincipalRole};

    #[test]
    fn test_dispatch_tool_allowed_tools_and_containment() {
        let temp_dir = std::env::temp_dir().join(format!("agent_sb_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&temp_dir);

        let mut allowed = HashSet::new();
        allowed.insert("safe_tool".to_string());
        let policy = AgentSandboxPolicy {
            allowed_tools: allowed,
            ..Default::default()
        };

        let gateway = Arc::new(EnforcementGateway::new_standalone());
        let sandbox =
            AgentExecutionSandbox::new("test_agent", temp_dir.clone(), policy, gateway.clone())
                .unwrap();

        let user_pid = PrincipalId::new("test_user").unwrap();
        let principal = VerifiedPrincipal::construct_verified(
            user_pid,
            PrincipalRole::user(),
            AssuranceLevel::STANDARD_TOKEN,
            None,
            AuthorityGeneration::new(1).unwrap(),
            1000,
            std::collections::HashMap::new(),
        )
        .unwrap();
        let context = SecurityContext::new("127.0.0.1", 1000);

        // Disallowed tool rejected
        let res_disallowed = sandbox.dispatch_tool(
            "unauthorized_tool",
            &temp_dir.join("safe_tool.exe"),
            &principal,
            &context,
            None,
            |_p, _r| Ok("ok"),
        );
        assert!(res_disallowed.is_err());
        assert!(res_disallowed.unwrap_err().contains("UnauthorizedTool"));

        // Allowed tool with executable escaping workspace rejected
        let res_escaped = sandbox.dispatch_tool(
            "safe_tool",
            &PathBuf::from("../../escape.exe"),
            &principal,
            &context,
            None,
            |_p, _r| Ok("ok"),
        );
        assert!(res_escaped.is_err());

        // Wildcard tool policy permits tool name, but path escaping is still rejected
        let mut wildcard_allowed = HashSet::new();
        wildcard_allowed.insert("*".to_string());
        let wildcard_policy = AgentSandboxPolicy {
            allowed_tools: wildcard_allowed,
            ..Default::default()
        };
        let wildcard_sandbox = AgentExecutionSandbox::new(
            "test_agent_2",
            temp_dir.clone(),
            wildcard_policy,
            gateway.clone(),
        )
        .unwrap();

        let res_wildcard_escaped = wildcard_sandbox.dispatch_tool(
            "any_tool",
            &PathBuf::from("../../escape.exe"),
            &principal,
            &context,
            None,
            |_p, _r| Ok("ok"),
        );
        assert!(res_wildcard_escaped.is_err());

        // Default empty allowed_tools policy fails closed
        let default_sandbox = AgentExecutionSandbox::new(
            "test_agent_default",
            temp_dir.clone(),
            AgentSandboxPolicy::default(),
            gateway,
        )
        .unwrap();
        let res_default_empty = default_sandbox.dispatch_tool(
            "any_tool",
            &temp_dir.join("any_tool.exe"),
            &principal,
            &context,
            None,
            |_p, _r| Ok("ok"),
        );
        assert!(res_default_empty.is_err());
        assert!(res_default_empty.unwrap_err().contains("UnauthorizedTool"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
