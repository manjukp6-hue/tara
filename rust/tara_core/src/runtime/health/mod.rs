//! health/mod.rs
//!
//! Health monitoring and telemetry for dynamic runtime entities.
//! Tracks heartbeat, operational status, degraded states, and system health summaries.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use crate::runtime::lifecycle::EntityState;

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
}

impl HealthMonitor {
    pub fn new(heartbeat_timeout_secs: u64) -> Self {
        Self {
            last_heartbeat: HashMap::new(),
            error_counts: HashMap::new(),
            heartbeat_timeout: Duration::from_secs(heartbeat_timeout_secs),
        }
    }

    pub fn record_heartbeat(&mut self, entity_id: &str) {
        self.last_heartbeat.insert(entity_id.to_string(), Instant::now());
    }

    pub fn record_error(&mut self, entity_id: &str) {
        *self.error_counts.entry(entity_id.to_string()).or_insert(0) += 1;
    }

    pub fn evaluate_health(&self, entity_id: &str, current_state: EntityState) -> EntityHealthStatus {
        let is_timed_out = self.last_heartbeat.get(entity_id)
            .map(|&t| t.elapsed() > self.heartbeat_timeout)
            .unwrap_or(false);

        let errs = self.error_counts.get(entity_id).copied().unwrap_or(0);
        let is_healthy = !is_timed_out && errs < 5 && !matches!(current_state, EntityState::Degraded | EntityState::Failed | EntityState::Quarantined | EntityState::Revoked);

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
