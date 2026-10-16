//! permissions.rs
//!
//! Centralized Capability & Permission Engine for TARA Core.
//! Enforces that permissions are capability-driven and non-escalatable by AI.
//!
//! # Architecture & Ownership Boundary
//! - **Scope**: In-crate capability definitions and localized permission checks for core tasks.
//! - **Authoritative Runtime Zero-Trust**: Cross-subsystem session tokens and permission sets
//!   are authoritatively granted and verified by `tara_security::zero_trust`.
//! - **Authoritative Execution Guard**: Full-system policy enforcement across API routes, tools,
//!   and server operations is authoritatively managed by `tara_server::rules::ExecutionGuard`.

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

        // Device capabilities
        let mut device_caps = HashSet::new();
        device_caps.insert(Capability::DeviceControl);
        device_caps.insert(Capability::ReadMemory);
        device_caps.insert(Capability::WriteMemory);
        self.role_capabilities.insert(Role::DEVICE, device_caps);

        // User capabilities
        let mut user_caps = HashSet::new();
        user_caps.insert(Capability::ModelInference);
        user_caps.insert(Capability::ReadMemory);
        user_caps.insert(Capability::WriteMemory);
        self.role_capabilities.insert(Role::USER, user_caps);

        // Staff capabilities
        let mut staff_caps = HashSet::new();
        staff_caps.insert(Capability::ModelInference);
        staff_caps.insert(Capability::ReadMemory);
        staff_caps.insert(Capability::WriteMemory);
        staff_caps.insert(Capability::ExecuteCode);
        staff_caps.insert(Capability::ManageSkills);
        staff_caps.insert(Capability::AuditLogRead);
        self.role_capabilities.insert(Role::STAFF, staff_caps);

        // Admin capabilities
        let mut admin_caps = HashSet::new();
        admin_caps.insert(Capability::ModelInference);
        admin_caps.insert(Capability::ReadMemory);
        admin_caps.insert(Capability::WriteMemory);
        admin_caps.insert(Capability::ExecuteCode);
        admin_caps.insert(Capability::ManageSkills);
        admin_caps.insert(Capability::AuditLogRead);
        admin_caps.insert(Capability::AccessFilesystem);
        admin_caps.insert(Capability::NetworkAccess);
        admin_caps.insert(Capability::DeviceControl);
        self.role_capabilities.insert(Role::ADMIN, admin_caps);

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

    /// Dynamically grants a capability to a role, gated strictly by Role::CREATOR authority.
    pub fn grant_permission(
        &mut self,
        granter_role: Role,
        target_role: Role,
        capability: Capability,
    ) -> Result<bool, String> {
        if !self.can_grant_permission(granter_role, target_role, &capability) {
            return Err("Permission modification rejected: requires CREATOR authority".to_string());
        }
        Ok(self.role_capabilities.entry(target_role).or_default().insert(capability))
    }

    /// Dynamically revokes a capability from a role, gated strictly by Role::CREATOR authority.
    /// CREATOR capabilities are permanent and unconditionally immutable against revocation.
    pub fn revoke_permission(
        &mut self,
        revoker_role: Role,
        target_role: Role,
        capability: &Capability,
    ) -> Result<bool, String> {
        if revoker_role != Role::CREATOR || target_role == Role::CREATOR {
            return Err("Permission modification rejected: illegal permission revocation attempt".to_string());
        }
        if let Some(caps) = self.role_capabilities.get_mut(&target_role) {
            Ok(caps.remove(capability))
        } else {
            Ok(false)
        }
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

    #[test]
    fn test_role_defaults_and_grant_revoke() {
        let mut engine = PermissionEngine::default();

        // Verify default capabilities
        assert!(engine.has_capability(Role::DEVICE, &Capability::DeviceControl));
        assert!(!engine.has_capability(Role::DEVICE, &Capability::NetworkAccess));

        assert!(engine.has_capability(Role::USER, &Capability::ModelInference));
        assert!(!engine.has_capability(Role::USER, &Capability::AccessFilesystem));

        assert!(engine.has_capability(Role::STAFF, &Capability::ManageSkills));
        assert!(!engine.has_capability(Role::STAFF, &Capability::AccessFilesystem));

        assert!(engine.has_capability(Role::ADMIN, &Capability::AccessFilesystem));
        assert!(!engine.has_capability(Role::ADMIN, &Capability::CoreUpdate));

        // Non-creator cannot grant or revoke
        assert!(engine
            .grant_permission(Role::ADMIN, Role::USER, Capability::NetworkAccess)
            .is_err());
        assert!(engine
            .revoke_permission(Role::ADMIN, Role::DEVICE, &Capability::DeviceControl)
            .is_err());

        // Creator can grant and revoke
        assert!(engine
            .grant_permission(Role::CREATOR, Role::USER, Capability::NetworkAccess)
            .unwrap());
        assert!(engine.has_capability(Role::USER, &Capability::NetworkAccess));

        assert!(engine
            .revoke_permission(Role::CREATOR, Role::USER, &Capability::NetworkAccess)
            .unwrap());
        assert!(!engine.has_capability(Role::USER, &Capability::NetworkAccess));

        // Attempting to revoke CREATOR permissions is unconditionally rejected
        assert!(engine
            .revoke_permission(Role::CREATOR, Role::CREATOR, &Capability::CoreUpdate)
            .is_err());
        assert!(engine.has_capability(Role::CREATOR, &Capability::CoreUpdate));
    }
}
