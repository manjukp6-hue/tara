//! resource/mod.rs
//!
//! Dynamic resource management, allocation limits, and abuse protection.
//! Adapts to host hardware (low RAM / GPU constraints) with on-demand allocation,
//! idle cleanup, and process storm / memory exhaustion detection.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuota {
    pub max_cpu_percentage: u32,
    pub max_memory_bytes: usize,
    pub max_processes: u32,
    pub max_disk_bytes: usize,
    pub max_execution_time_ms: u64,
}

pub const DEFAULT_RESOURCE_MAX_CPU_PERCENTAGE: u32 = 50;
pub const DEFAULT_RESOURCE_MAX_MEMORY_BYTES: usize = 256 * 1024 * 1024; // 256 MB
pub const DEFAULT_RESOURCE_MAX_PROCESSES: u32 = 4;
pub const DEFAULT_RESOURCE_MAX_DISK_BYTES: usize = 512 * 1024 * 1024; // 512 MB
pub const DEFAULT_RESOURCE_MAX_EXEC_MS: u64 = 15_000;

impl Default for ResourceQuota {
    fn default() -> Self {
        let max_cpu_percentage = std::env::var("TARA_RESOURCE_MAX_CPU_PERCENT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_RESOURCE_MAX_CPU_PERCENTAGE);
        let max_memory_bytes = std::env::var("TARA_RESOURCE_MAX_MEMORY_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_RESOURCE_MAX_MEMORY_BYTES);
        let max_processes = std::env::var("TARA_RESOURCE_MAX_PROCESSES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_RESOURCE_MAX_PROCESSES);
        let max_disk_bytes = std::env::var("TARA_RESOURCE_MAX_DISK_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_RESOURCE_MAX_DISK_BYTES);
        let max_execution_time_ms = std::env::var("TARA_RESOURCE_MAX_EXEC_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_RESOURCE_MAX_EXEC_MS);

        Self {
            max_cpu_percentage,
            max_memory_bytes,
            max_processes,
            max_disk_bytes,
            max_execution_time_ms,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub current_processes: u32,
    pub current_memory_bytes: usize,
    pub current_disk_bytes: usize,
    pub peak_processes: u32,
    pub peak_memory_bytes: usize,
    pub start_time_epoch_ms: u64,
}

pub struct ResourceGovernor {
    quotas: HashMap<String, ResourceQuota>,
    usages: HashMap<String, ResourceUsage>,
    last_activity: HashMap<String, Instant>,
    idle_timeout: Duration,
}

/// Default idle timeout before reclaiming unused resource allocations (5 minutes).
pub const DEFAULT_IDLE_TIMEOUT_SECS: u64 = 300;

impl Default for ResourceGovernor {
    /// Constructs governor using dynamic runtime configuration (`TARA_IDLE_TIMEOUT_SECS`),
    /// falling back to the 300-second baseline if unspecified.
    fn default() -> Self {
        let timeout = std::env::var("TARA_IDLE_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_IDLE_TIMEOUT_SECS);
        Self::new(timeout)
    }
}

impl ResourceGovernor {
    pub fn new(idle_timeout_secs: u64) -> Self {
        Self {
            quotas: HashMap::new(),
            usages: HashMap::new(),
            last_activity: HashMap::new(),
            idle_timeout: Duration::from_secs(idle_timeout_secs),
        }
    }

    pub fn assign_quota(&mut self, entity_id: &str, quota: ResourceQuota) {
        self.quotas.insert(entity_id.to_string(), quota);
        self.usages.insert(
            entity_id.to_string(),
            ResourceUsage {
                current_processes: 0,
                current_memory_bytes: 0,
                current_disk_bytes: 0,
                peak_processes: 0,
                peak_memory_bytes: 0,
                start_time_epoch_ms: 0,
            },
        );
        self.last_activity
            .insert(entity_id.to_string(), Instant::now());
    }

    /// Verifies if spawning a process would violate quota or trigger a process storm.
    pub fn check_process_spawn(&mut self, entity_id: &str) -> Result<(), String> {
        let quota = self.quotas.get(entity_id).cloned().unwrap_or_default();
        let usage = self
            .usages
            .get_mut(entity_id)
            .ok_or_else(|| "Entity not tracked".to_string())?;

        if usage.current_processes >= quota.max_processes {
            return Err(format!(
                "Process Storm Protection: active processes ({}) reached quota limit ({})",
                usage.current_processes, quota.max_processes
            ));
        }

        usage.current_processes += 1;
        if usage.current_processes > usage.peak_processes {
            usage.peak_processes = usage.current_processes;
        }
        self.last_activity
            .insert(entity_id.to_string(), Instant::now());
        Ok(())
    }

    pub fn release_process(&mut self, entity_id: &str) {
        if let Some(usage) = self.usages.get_mut(entity_id) {
            if usage.current_processes > 0 {
                usage.current_processes -= 1;
            }
        }
    }

    /// Identifies idle entities that have exceeded the idle timeout threshold.
    pub fn find_idle_entities(&self) -> Vec<String> {
        let now = Instant::now();
        self.last_activity
            .iter()
            .filter_map(|(id, &last)| {
                if now.duration_since(last) > self.idle_timeout {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_quota_default_and_env_resolution() {
        let quota = ResourceQuota::default();
        assert_eq!(quota.max_cpu_percentage, DEFAULT_RESOURCE_MAX_CPU_PERCENTAGE);
        assert_eq!(quota.max_processes, DEFAULT_RESOURCE_MAX_PROCESSES);
        assert_eq!(quota.max_memory_bytes, DEFAULT_RESOURCE_MAX_MEMORY_BYTES);
        assert_eq!(quota.max_disk_bytes, DEFAULT_RESOURCE_MAX_DISK_BYTES);
        assert_eq!(quota.max_execution_time_ms, DEFAULT_RESOURCE_MAX_EXEC_MS);
    }

    #[test]
    fn test_resource_governor_process_storm_protection() {
        let mut gov = ResourceGovernor::new(10);
        let quota = ResourceQuota {
            max_cpu_percentage: 20,
            max_memory_bytes: 1024,
            max_processes: 2,
            max_disk_bytes: 2048,
            max_execution_time_ms: 1000,
        };
        gov.assign_quota("entity_storm", quota);

        assert!(gov.check_process_spawn("entity_storm").is_ok());
        assert!(gov.check_process_spawn("entity_storm").is_ok());

        // Exceeding process quota must be rejected
        let storm_res = gov.check_process_spawn("entity_storm");
        assert!(storm_res.is_err());
        assert!(storm_res.unwrap_err().contains("Process Storm Protection"));

        // Releasing allows new spawn
        gov.release_process("entity_storm");
        assert!(gov.check_process_spawn("entity_storm").is_ok());
    }

    #[test]
    fn test_untracked_entity_spawn_rejected() {
        let mut gov = ResourceGovernor::new(10);
        let err = gov.check_process_spawn("untracked").unwrap_err();
        assert_eq!(err, "Entity not tracked");
    }
}

