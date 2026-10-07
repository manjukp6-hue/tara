//! health/mod.rs
//!
//! Health monitoring and telemetry for dynamic runtime entities.
//! Tracks heartbeat, operational status, degraded states, and system health summaries.

use crate::runtime::lifecycle::EntityState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityHealthStatus {
    pub entity_id: String,
    pub entity_type: String,
    pub state: EntityState,
    pub is_healthy: bool,
    pub error_count: u32,
    pub latency_ms: u64,
}

pub struct HealthMonitor {
    last_heartbeat: HashMap<String, Instant>,
    error_counts: HashMap<String, u32>,
    heartbeat_timeout: Duration,
    max_consecutive_errors: u32,
}

/// Default heartbeat timeout before an entity is marked degraded/unresponsive (60 seconds).
pub const DEFAULT_HEARTBEAT_TIMEOUT_SECS: u64 = 60;

/// Default error count before health status is marked unhealthy.
pub const DEFAULT_MAX_CONSECUTIVE_ERRORS: u32 = 5;

impl Default for HealthMonitor {
    /// Constructs monitor using dynamic runtime configuration (`TARA_HEARTBEAT_TIMEOUT_SECS`, `TARA_HEALTH_MAX_ERRORS`),
    /// falling back to authoritative baselines if unspecified.
    fn default() -> Self {
        let timeout = std::env::var("TARA_HEARTBEAT_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_HEARTBEAT_TIMEOUT_SECS);
        let max_errors = std::env::var("TARA_HEALTH_MAX_ERRORS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MAX_CONSECUTIVE_ERRORS);
        Self::with_limits(timeout, max_errors)
    }
}

impl HealthMonitor {
    pub fn new(heartbeat_timeout_secs: u64) -> Self {
        Self::with_limits(heartbeat_timeout_secs, DEFAULT_MAX_CONSECUTIVE_ERRORS)
    }

    pub fn with_limits(heartbeat_timeout_secs: u64, max_consecutive_errors: u32) -> Self {
        Self {
            last_heartbeat: HashMap::new(),
            error_counts: HashMap::new(),
            heartbeat_timeout: Duration::from_secs(heartbeat_timeout_secs),
            max_consecutive_errors,
        }
    }

    pub fn record_heartbeat(&mut self, entity_id: &str) {
        self.last_heartbeat
            .insert(entity_id.to_string(), Instant::now());
    }

    pub fn record_error(&mut self, entity_id: &str) {
        *self.error_counts.entry(entity_id.to_string()).or_insert(0) += 1;
    }

    pub fn evaluate_health(
        &self,
        entity_id: &str,
        current_state: EntityState,
    ) -> EntityHealthStatus {
        let is_timed_out = self
            .last_heartbeat
            .get(entity_id)
            .map(|&t| t.elapsed() > self.heartbeat_timeout)
            .unwrap_or(false);

        let errs = self.error_counts.get(entity_id).copied().unwrap_or(0);
        let is_healthy = !is_timed_out
            && errs < self.max_consecutive_errors
            && !matches!(
                current_state,
                EntityState::Degraded
                    | EntityState::Failed
                    | EntityState::Quarantined
                    | EntityState::Revoked
            );

        EntityHealthStatus {
            entity_id: entity_id.to_string(),
            entity_type: "Entity".to_string(),
            state: current_state,
            is_healthy,
            error_count: errs,
            latency_ms: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_monitor_healthy_flow() {
        let mut monitor = HealthMonitor::new(60);
        monitor.record_heartbeat("entity_1");

        let status = monitor.evaluate_health("entity_1", EntityState::Ready);
        assert!(status.is_healthy);
        assert_eq!(status.error_count, 0);
    }

    #[test]
    fn test_health_monitor_error_threshold_degradation() {
        let mut monitor = HealthMonitor::with_limits(60, 3);
        monitor.record_heartbeat("entity_2");

        monitor.record_error("entity_2");
        monitor.record_error("entity_2");
        assert!(monitor.evaluate_health("entity_2", EntityState::Active).is_healthy);

        // Third error reaches threshold (3) -> unhealthy
        monitor.record_error("entity_2");
        let status = monitor.evaluate_health("entity_2", EntityState::Active);
        assert!(!status.is_healthy);
        assert_eq!(status.error_count, 3);
    }

    #[test]
    fn test_health_monitor_inherent_degraded_states() {
        let monitor = HealthMonitor::new(60);
        for bad_state in [
            EntityState::Degraded,
            EntityState::Failed,
            EntityState::Quarantined,
            EntityState::Revoked,
        ] {
            let status = monitor.evaluate_health("entity_bad", bad_state);
            assert!(!status.is_healthy);
        }
    }
}

