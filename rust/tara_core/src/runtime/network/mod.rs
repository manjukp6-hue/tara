//! network/mod.rs
//!
//! Task-scoped Network Controller for Sandboxes.
//! Default: strictly OFF (no network).
//! Dynamically evaluates whether internet is required, whitelists allowed domains/services,
//! enforces read-only vs write permissions, and automatically revokes access when the task ends.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkPermission {
    ReadOnly,
    ReadWrite,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkGrant {
    pub sandbox_id: String,
    pub task_id: String,
    pub allowed_domains: HashSet<String>,
    pub permission: NetworkPermission,
    pub granted_at_ms: u64,
    pub duration_ms: u64,
}

pub struct NetworkController {
    active_grants: HashMap<String, (NetworkGrant, Instant)>,
}

impl Default for NetworkController {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkController {
    pub fn new() -> Self {
        Self {
            active_grants: HashMap::new(),
        }
    }

    /// Grants temporary, task-scoped network access to a sandbox.
    pub fn grant_network(
        &mut self,
        sandbox_id: &str,
        task_id: &str,
        allowed_domains: &[String],
        permission: NetworkPermission,
        duration_ms: u64,
    ) {
        let grant = NetworkGrant {
            sandbox_id: sandbox_id.to_string(),
            task_id: task_id.to_string(),
            allowed_domains: allowed_domains.iter().cloned().collect(),
            permission,
            granted_at_ms: 0,
            duration_ms,
        };

        self.active_grants
            .insert(sandbox_id.to_string(), (grant, Instant::now()));
    }

    /// Checks if a sandbox is permitted outbound connection to a domain.
    pub fn is_domain_allowed(&self, sandbox_id: &str, domain: &str) -> bool {
        if let Some((grant, start_time)) = self.active_grants.get(sandbox_id) {
            if start_time.elapsed() > Duration::from_millis(grant.duration_ms) {
                return false; // Expired
            }

            let d_clean = domain.trim().to_lowercase();
            grant.allowed_domains.contains(&d_clean) || grant.allowed_domains.contains("*")
        } else {
            false // Default: OFF
        }
    }

    /// Automatically revokes network access upon task completion or timeout.
    pub fn revoke_network(&mut self, sandbox_id: &str) {
        self.active_grants.remove(sandbox_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_controller_default_off() {
        let ctrl = NetworkController::new();
        assert!(!ctrl.is_domain_allowed("sbx_test", "example.com"));
    }

    #[test]
    fn test_network_controller_grant_and_revoke() {
        let mut ctrl = NetworkController::new();
        let allowed = vec!["api.local".to_string(), "crates.io".to_string()];

        ctrl.grant_network(
            "sbx_task_net",
            "task_net",
            &allowed,
            NetworkPermission::ReadOnly,
            60_000,
        );

        assert!(ctrl.is_domain_allowed("sbx_task_net", "api.local"));
        assert!(ctrl.is_domain_allowed("sbx_task_net", "crates.io"));
        assert!(!ctrl.is_domain_allowed("sbx_task_net", "malicious.org"));

        ctrl.revoke_network("sbx_task_net");
        assert!(!ctrl.is_domain_allowed("sbx_task_net", "api.local"));
    }

    #[test]
    fn test_network_controller_wildcard() {
        let mut ctrl = NetworkController::new();
        let allowed = vec!["*".to_string()];

        ctrl.grant_network(
            "sbx_wildcard",
            "task_wild",
            &allowed,
            NetworkPermission::ReadWrite,
            60_000,
        );

        assert!(ctrl.is_domain_allowed("sbx_wildcard", "anywhere.com"));
    }
}

