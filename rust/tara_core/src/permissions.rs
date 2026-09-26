//! permissions.rs
//! 
//! Centralized Capability & Permission Engine for TARA Core.
//! Enforces that permissions are capability-driven and non-escalatable by AI.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use crate::governance::Role;

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
    pub fn can_grant_permission(&self, granter_role: Role, _target_role: Role, _cap: &Capability) -> bool {
        granter_role == Role::CREATOR
    }
}
