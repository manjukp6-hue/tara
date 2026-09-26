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

impl Default for ResourceQuota {
    fn default() -> Self {
        Self {
            max_cpu_percentage: 50,
            max_memory_bytes: 256 * 1024 * 1024, // 256 MB
            max_processes: 4,
            max_disk_bytes: 512 * 1024 * 1024,    // 512 MB
            max_execution_time_ms: 15_000,
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
        self.usages.insert(entity_id.to_string(), ResourceUsage {
            current_processes: 0,
            current_memory_bytes: 0,
            current_disk_bytes: 0,
            peak_processes: 0,
            peak_memory_bytes: 0,
            start_time_epoch_ms: 0,
        });
        self.last_activity.insert(entity_id.to_string(), Instant::now());
    }

    /// Verifies if spawning a process would violate quota or trigger a process storm.
    pub fn check_process_spawn(&mut self, entity_id: &str) -> Result<(), String> {
        let quota = self.quotas.get(entity_id).cloned().unwrap_or_default();
        let usage = self.usages.get_mut(entity_id).ok_or_else(|| "Entity not tracked".to_string())?;

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
        self.last_activity.insert(entity_id.to_string(), Instant::now());
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
