//! tools/mod.rs
//!
//! Task-scoped Tool Capability Controller.
//! Dynamically registers, assigns, and limits tools exposed to specific sandboxes.
//! Enforces least-privilege capability boundaries per task execution.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub tool_name: String,
    pub description: String,
    pub required_capability: String,
    pub is_high_risk: bool,
    pub allowed_extensions: Vec<String>,
}

#[derive(Debug, Default)]
pub struct ToolController {
    catalog: HashMap<String, ToolDefinition>,
    // sandbox_id -> Set of allowed tool names for active task
    active_assignments: HashMap<String, HashSet<String>>,
}

impl ToolController {
    pub fn new() -> Self {
        let mut ctrl = Self::default();
        ctrl.register_builtins();
        ctrl
    }

    fn register_builtins(&mut self) {
        self.register_tool(ToolDefinition {
            tool_name: "file_inspector".to_string(),
            description: "Read metadata and inspect head bytes inside sandbox".to_string(),
            required_capability: "tool:file_inspector".to_string(),
            is_high_risk: false,
            allowed_extensions: vec![".txt".into(), ".json".into(), ".csv".into(), ".rs".into()],
        });

        self.register_tool(ToolDefinition {
            tool_name: "hash_verifier".to_string(),
            description: "Compute and verify SHA256 of sandboxed artifact".to_string(),
            required_capability: "tool:hash_verifier".to_string(),
            is_high_risk: false,
            allowed_extensions: vec![],
        });

        self.register_tool(ToolDefinition {
            tool_name: "code_evaluator".to_string(),
            description: "Run verification script inside isolated container".to_string(),
            required_capability: "tool:code_evaluator".to_string(),
            is_high_risk: true,
            allowed_extensions: vec![".rs".into()],
        });
    }

    pub fn register_tool(&mut self, def: ToolDefinition) {
        self.catalog.insert(def.tool_name.clone(), def);
    }

    /// Dynamically assigns task-scoped tools to a sandbox.
    pub fn assign_tools_to_sandbox(
        &mut self,
        sandbox_id: &str,
        tool_names: &[String],
    ) -> Result<(), String> {
        let mut allowed = HashSet::new();
        for t in tool_names {
            if !self.catalog.contains_key(t) {
                return Err(format!("Unrecognized tool '{}'", t));
            }
            allowed.insert(t.clone());
        }
        self.active_assignments
            .insert(sandbox_id.to_string(), allowed);
        Ok(())
    }

    /// Verifies if a sandbox is authorized to execute a requested tool.
    pub fn is_tool_allowed(&self, sandbox_id: &str, tool_name: &str) -> bool {
        self.active_assignments
            .get(sandbox_id)
            .map(|set| set.contains(tool_name))
            .unwrap_or(false)
    }

    /// Revokes tool assignments upon task completion.
    pub fn revoke_sandbox_tools(&mut self, sandbox_id: &str) {
        self.active_assignments.remove(sandbox_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_controller_catalog_and_assignment() {
        let mut ctrl = ToolController::new();

        // Built-ins exist
        assert!(ctrl.catalog.contains_key("file_inspector"));
        assert!(ctrl.catalog.contains_key("code_evaluator"));

        // Assign valid tools
        let tools = vec!["file_inspector".to_string(), "hash_verifier".to_string()];
        assert!(ctrl.assign_tools_to_sandbox("sbx_task_01", &tools).is_ok());

        assert!(ctrl.is_tool_allowed("sbx_task_01", "file_inspector"));
        assert!(ctrl.is_tool_allowed("sbx_task_01", "hash_verifier"));
        assert!(!ctrl.is_tool_allowed("sbx_task_01", "code_evaluator"));

        // Revoke
        ctrl.revoke_sandbox_tools("sbx_task_01");
        assert!(!ctrl.is_tool_allowed("sbx_task_01", "file_inspector"));
    }

    #[test]
    fn test_unrecognized_tool_assignment_error() {
        let mut ctrl = ToolController::new();
        let tools = vec!["non_existent_tool".to_string()];
        let res = ctrl.assign_tools_to_sandbox("sbx_err", &tools);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("Unrecognized tool"));
    }
}

