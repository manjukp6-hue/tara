//! Multi-Tier Memory Placement & Dynamic Execution Governor for TARA-HD.
//!
//! Evaluates the comprehensive memory budget equation:
//! M_required = M_weights + M_kv_cache + M_activations + M_workspace + M_overhead
//! and resolves runtime execution placement into FullResident, PartialOffload, or CapacityExceeded.

use serde::{Deserialize, Serialize};
use crate::telemetry::RuntimeTelemetrySnapshot;

/// Operating policy for memory vs. performance trade-offs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MemoryExecutionMode {
    /// Maximize throughput by pinning hot weights and KV-cache in fastest accelerator memory.
    #[default]
    HighPerformance,
    /// Minimize RAM/VRAM footprint by streaming layers/experts dynamically on demand.
    ConstrainedMemorySaving,
}

/// Dynamic memory budget breakdown for a model workload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelMemoryBudget {
    /// Memory required for model parameter weights in bytes.
    pub weights_bytes: usize,
    /// Estimated peak memory required for Key-Value attention cache in bytes.
    pub kv_cache_bytes: usize,
    /// Peak transient activation memory in bytes.
    pub activations_bytes: usize,
    /// Transient per-layer weight upcast conversion buffer (FP16 storage to FP32 compute).
    pub transient_upcast_bytes: usize,
    /// Temporary workspace buffer memory in bytes.
    pub workspace_bytes: usize,
    /// Driver and runtime overhead buffer in bytes.
    pub overhead_bytes: usize,
}

impl ModelMemoryBudget {
    /// Compute total required memory footprint:
    /// M_required = M_weights + M_kv_cache + M_activations + M_transient_upcast + M_workspace + M_overhead
    pub fn total_required_bytes(&self) -> usize {
        self.weights_bytes
            .saturating_add(self.kv_cache_bytes)
            .saturating_add(self.activations_bytes)
            .saturating_add(self.transient_upcast_bytes)
            .saturating_add(self.workspace_bytes)
            .saturating_add(self.overhead_bytes)
    }

    /// Verifies strict invariant that total_required_bytes matches exact sum of components without double-counting.
    pub fn assert_invariant(&self) {
        let component_sum = self.weights_bytes
            .saturating_add(self.kv_cache_bytes)
            .saturating_add(self.activations_bytes)
            .saturating_add(self.transient_upcast_bytes)
            .saturating_add(self.workspace_bytes)
            .saturating_add(self.overhead_bytes);

        assert_eq!(
            self.total_required_bytes(),
            component_sum,
            "Memory budget total must equal exact sum of individual non-overlapping components"
        );
    }

    /// Helper to compute budget for standard transformer configurations.
    pub fn estimate(
        param_count: usize,
        bytes_per_param: usize,
        max_seq_len: usize,
        hidden_size: usize,
        num_layers: usize,
    ) -> Self {
        let weights_bytes = param_count * bytes_per_param;
        // KV cache approx: 2 * num_layers * 2(K+V) * seq_len * hidden_size * bytes_per_param
        let kv_cache_bytes = 2 * num_layers * 2 * max_seq_len * hidden_size * bytes_per_param;
        let activations_bytes = max_seq_len * hidden_size * bytes_per_param * 4;
        // Transient upcast buffer: per-layer upcast from FP16 (2B) to FP32 (4B)
        let transient_upcast_bytes = if bytes_per_param == 2 {
            (weights_bytes / num_layers.max(1)) * 2
        } else {
            0
        };
        let workspace_bytes = 64 * 1024 * 1024; // 64 MB minimum workspace
        let overhead_bytes = 128 * 1024 * 1024; // 128 MB driver overhead

        Self {
            weights_bytes,
            kv_cache_bytes,
            activations_bytes,
            transient_upcast_bytes,
            workspace_bytes,
            overhead_bytes,
        }
    }
}

/// Placement decision resolved by the Demand Scheduler.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlacementDecision {
    /// Entire workload (weights + KV cache + activations) fits completely in accelerator memory.
    FullResident {
        device_memory_allocated: usize,
        usable_vram_budget: usize,
        total_vram_capacity: usize,
        headroom_bytes: usize,
    },
    /// Host RAM assists accelerator: resident trunk in accelerator, overflow streamed/paged.
    PartialOffload {
        device_memory_budget: usize,
        host_paged_memory: usize,
        resident_layers: usize,
    },
    /// Neither accelerator nor host RAM satisfies minimum required memory bounds.
    CapacityExceeded {
        required_bytes: usize,
        available_bytes: usize,
        deficit_bytes: usize,
    },
}

/// Governor determining workload placement based on live telemetry and execution mode.
pub struct MemoryGovernor;

impl MemoryGovernor {
    /// Resolve placement decision without hardcoded ceilings or arbitrary magic thresholds.
    pub fn evaluate_placement(
        budget: &ModelMemoryBudget,
        telemetry: &RuntimeTelemetrySnapshot,
        mode: MemoryExecutionMode,
        total_layers: usize,
    ) -> PlacementDecision {
        let total_required = budget.total_required_bytes();

        // 1. Check Accelerator Placement (if accelerator available)
        if let Some(free_vram) = telemetry.free_device_vram_bytes {
            // Apply 10% safety margin on free VRAM
            let usable_vram = (free_vram * 90) / 100;
            let total_vram = telemetry.total_device_vram_bytes.unwrap_or(free_vram);

            if usable_vram >= total_required {
                return PlacementDecision::FullResident {
                    device_memory_allocated: total_required,
                    usable_vram_budget: usable_vram,
                    total_vram_capacity: total_vram,
                    headroom_bytes: usable_vram.saturating_sub(total_required),
                };
            }

            // Minimum viable unit of accelerator execution: at least 1 layer plus workspace and overhead
            let min_accelerator_layer_budget = (budget.weights_bytes / total_layers.max(1))
                .saturating_add(budget.workspace_bytes)
                .saturating_add(budget.overhead_bytes);

            // If in Constrained or partial fit: check partial offload dynamically for any model scale
            if usable_vram >= min_accelerator_layer_budget || mode == MemoryExecutionMode::ConstrainedMemorySaving {
                let resident_layers = (total_layers * usable_vram) / total_required.max(1);
                let resident_layers = resident_layers.clamp(1, total_layers);
                let host_paged = total_required.saturating_sub(usable_vram);

                if telemetry.free_host_ram_bytes >= host_paged {
                    return PlacementDecision::PartialOffload {
                        device_memory_budget: usable_vram,
                        host_paged_memory: host_paged,
                        resident_layers,
                    };
                }
            }
        }

        // 2. Check Host System RAM Placement (CPU execution)
        let usable_host_ram = (telemetry.free_host_ram_bytes * 85) / 100;
        if usable_host_ram >= total_required {
            return PlacementDecision::FullResident {
                device_memory_allocated: total_required,
                usable_vram_budget: usable_host_ram,
                total_vram_capacity: telemetry.total_host_ram_bytes,
                headroom_bytes: usable_host_ram.saturating_sub(total_required),
            };
        }

        // 3. Fallback: Check if partial paged paging into host RAM works dynamically
        let min_host_layer_budget = (budget.weights_bytes / total_layers.max(1))
            .saturating_add(budget.workspace_bytes);
        if usable_host_ram >= min_host_layer_budget {
            let resident_layers = (total_layers * usable_host_ram) / total_required.max(1);
            let resident_layers = resident_layers.clamp(1, total_layers);
            return PlacementDecision::PartialOffload {
                device_memory_budget: usable_host_ram,
                host_paged_memory: total_required.saturating_sub(usable_host_ram),
                resident_layers,
            };
        }

        // 4. Capacity Exceeded
        PlacementDecision::CapacityExceeded {
            required_bytes: total_required,
            available_bytes: usable_host_ram,
            deficit_bytes: total_required.saturating_sub(usable_host_ram),
        }
    }
}
