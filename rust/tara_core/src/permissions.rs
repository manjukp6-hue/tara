//! permissions.rs
//!
//! Centralized Capability & Permission Engine for TARA Core.
//! Enforces that permissions are capability-driven and non-escalatable by AI.

use crate::governance::Role;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability {
    ExecuteCode,
    AccessFilesystem,
    NetworkAccess,
    DeviceControl,
    ManageSkills,
    ReadMemory,
    WriteMemory,
    ModelInference,
    AuditLogRead,
    CoreUpdate,
}

#[derive(Debug, Clone)]
pub struct PermissionEngine {
    role_capabilities: std::collections::HashMap<Role, HashSet<Capability>>,
}

impl Default for PermissionEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl PermissionEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            role_capabilities: std::collections::HashMap::new(),
        };
        engine.init_defaults();
        engine
    }

    fn init_defaults(&mut self) {
        // AI capabilities: restricted operational set
        let mut ai_caps = HashSet::new();
        ai_caps.insert(Capability::ModelInference);
        ai_caps.insert(Capability::ReadMemory);
        ai_caps.insert(Capability::WriteMemory);
        ai_caps.insert(Capability::ExecuteCode); // Must still be sandboxed
        self.role_capabilities.insert(Role::AI, ai_caps);

        // Creator: All capabilities
        let mut creator_caps = HashSet::new();
        creator_caps.insert(Capability::ExecuteCode);
        creator_caps.insert(Capability::AccessFilesystem);
        creator_caps.insert(Capability::NetworkAccess);
        creator_caps.insert(Capability::DeviceControl);
        creator_caps.insert(Capability::ManageSkills);
        creator_caps.insert(Capability::ReadMemory);
        creator_caps.insert(Capability::WriteMemory);
        creator_caps.insert(Capability::ModelInference);
        creator_caps.insert(Capability::AuditLogRead);
        creator_caps.insert(Capability::CoreUpdate);
        self.role_capabilities.insert(Role::CREATOR, creator_caps);
    }

    pub fn has_capability(&self, role: Role, capability: &Capability) -> bool {
        if role == Role::CREATOR {
            return true;
        }
        self.role_capabilities
            .get(&role)
            .map(|caps| caps.contains(capability))
            .unwrap_or(false)
    }

    /// AI cannot grant capabilities to itself or modify security permissions
    pub fn can_grant_permission(
        &self,
        granter_role: Role,
        _target_role: Role,
        _cap: &Capability,
    ) -> bool {
        granter_role == Role::CREATOR
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creator_has_all_capabilities() {
        let engine = PermissionEngine::default();
        let all_caps = [
            Capability::ExecuteCode,
            Capability::AccessFilesystem,
            Capability::NetworkAccess,
            Capability::DeviceControl,
            Capability::ManageSkills,
            Capability::ReadMemory,
            Capability::WriteMemory,
            Capability::ModelInference,
            Capability::AuditLogRead,
            Capability::CoreUpdate,
        ];
        for cap in &all_caps {
            assert!(engine.has_capability(Role::CREATOR, cap));
        }
    }

    #[test]
    fn test_ai_capabilities_restricted() {
        let engine = PermissionEngine::default();
        // Allowed for AI
        assert!(engine.has_capability(Role::AI, &Capability::ModelInference));
        assert!(engine.has_capability(Role::AI, &Capability::ReadMemory));
        assert!(engine.has_capability(Role::AI, &Capability::WriteMemory));
        assert!(engine.has_capability(Role::AI, &Capability::ExecuteCode));

        // Forbidden for AI (critical security boundary)
        assert!(!engine.has_capability(Role::AI, &Capability::CoreUpdate));
        assert!(!engine.has_capability(Role::AI, &Capability::AccessFilesystem));
        assert!(!engine.has_capability(Role::AI, &Capability::NetworkAccess));
        assert!(!engine.has_capability(Role::AI, &Capability::DeviceControl));
        assert!(!engine.has_capability(Role::AI, &Capability::ManageSkills));
    }

    #[test]
    fn test_ai_cannot_grant_permissions() {
        let engine = PermissionEngine::default();
        // AI attempting to grant permission to itself or others is rejected
        assert!(!engine.can_grant_permission(Role::AI, Role::AI, &Capability::CoreUpdate));
        assert!(!engine.can_grant_permission(Role::ADMIN, Role::AI, &Capability::CoreUpdate));
        // Only CREATOR can grant permission
        assert!(engine.can_grant_permission(Role::CREATOR, Role::AI, &Capability::ExecuteCode));
    }
}
