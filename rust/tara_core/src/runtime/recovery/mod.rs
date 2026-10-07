//! recovery/mod.rs
//!
//! Failure isolation, recovery orchestration, and quarantine management.
//! Enforces cross-entity failure isolation: a failure in Sandbox A does not compromise Sandbox B.
//! Fresh sandbox reconstruction for unverified state, preserved verified work, and rollback mechanisms.

use crate::runtime::lifecycle::EntityState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureCategory {
    ProcessCrash,
    Timeout,
    ResourceExhaustion,
    SecurityViolation,
    IntegrityMismatch,
    PolicyDenial,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncidentReport {
    pub incident_id: String,
    pub entity_id: String,
    pub failure_category: FailureCategory,
    pub description: String,
    pub recommended_action: String,
    pub quarantined: bool,
    pub timestamp_ms: u64,
}

pub struct RecoveryOrchestrator {
    incidents: HashMap<String, IncidentReport>,
}

impl Default for RecoveryOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

impl RecoveryOrchestrator {
    pub fn new() -> Self {
        Self {
            incidents: HashMap::new(),
        }
    }

    /// Diagnoses a failure and determines safe recovery vs quarantine.
    pub fn handle_failure(
        &mut self,
        entity_id: &str,
        failure: FailureCategory,
        details: &str,
    ) -> (EntityState, String) {
        let incident_id = format!("inc_{:x}", rand::random::<u64>());

        let (target_state, action) = match failure {
            FailureCategory::SecurityViolation | FailureCategory::IntegrityMismatch => {
                // Critical safety failure: immediately quarantine
                (
                    EntityState::Quarantined,
                    "Immediate quarantine enforced. Do not restore state; isolate entity and investigate.".to_string(),
                )
            }
            FailureCategory::ResourceExhaustion => {
                // Transient resource pressure: drain or pause
                (
                    EntityState::Degraded,
                    "Resource exhaustion detected. Drain active workload, release memory, and throttle.".to_string(),
                )
            }
            FailureCategory::Timeout | FailureCategory::ProcessCrash => {
                // Recoverable: destroy sandbox, preserve verified outputs, spin up clean instance
                (
                    EntityState::Recovering,
                    "Crash/timeout encountered. Isolate failed container, create fresh replacement, reassign task.".to_string(),
                )
            }
            FailureCategory::PolicyDenial => (
                EntityState::Paused,
                "Action denied by policy. Retain audit record; no automatic retry.".to_string(),
            ),
        };

        let report = IncidentReport {
            incident_id: incident_id.clone(),
            entity_id: entity_id.to_string(),
            failure_category: failure,
            description: details.to_string(),
            recommended_action: action.clone(),
            quarantined: target_state == EntityState::Quarantined,
            timestamp_ms: 0,
        };

        self.incidents.insert(incident_id, report);
        (target_state, action)
    }

    pub fn get_incident(&self, incident_id: &str) -> Option<&IncidentReport> {
        self.incidents.get(incident_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recovery_quarantine_on_security_violation() {
        let mut rec = RecoveryOrchestrator::new();
        let (state, action) = rec.handle_failure(
            "entity_malicious",
            FailureCategory::SecurityViolation,
            "Attempted host root escape",
        );

        assert_eq!(state, EntityState::Quarantined);
        assert!(action.contains("Immediate quarantine"));
    }

    #[test]
    fn test_recovery_degraded_on_resource_exhaustion() {
        let mut rec = RecoveryOrchestrator::new();
        let (state, _) = rec.handle_failure(
            "entity_heavy",
            FailureCategory::ResourceExhaustion,
            "Out of memory limit",
        );

        assert_eq!(state, EntityState::Degraded);
    }

    #[test]
    fn test_recovery_recoverable_on_process_crash() {
        let mut rec = RecoveryOrchestrator::new();
        let (state, _) = rec.handle_failure(
            "entity_crash",
            FailureCategory::ProcessCrash,
            "SIGSEGV trapped",
        );

        assert_eq!(state, EntityState::Recovering);
    }
}

