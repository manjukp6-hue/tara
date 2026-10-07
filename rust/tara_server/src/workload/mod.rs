//! Workload distribution: detect, route, distribute, aggregate across compute nodes.

pub mod workload_distributor {
    use serde_json::{json, Value};

    #[derive(Debug, Clone)]
    pub enum ExecutionMode {
        CpuSingle,
        CpuDistributed,
        GpuSingle,
        CpuGpuHybrid,
    }

    impl ExecutionMode {
        pub fn as_str(&self) -> &'static str {
            match self {
                ExecutionMode::CpuSingle => "CpuSingle",
                ExecutionMode::CpuDistributed => "CpuDistributed",
                ExecutionMode::GpuSingle => "GpuSingle",
                ExecutionMode::CpuGpuHybrid => "CpuGpuHybrid",
            }
        }
    }

    pub fn detect_workload(input_data: &Value) -> Value {
        let size = input_data.to_string().len();
        let workload_type = if size > 100_000 {
            "Training"
        } else {
            "Inference"
        };
        json!({
            "task_name": "inference",
            "workload_type": workload_type,
            "input_size": size,
            "estimated_ram_mb": (size / 1000).max(64),
        })
    }

    /// Detect the workload characteristics and select the appropriate execution mode.
    ///
    /// Decision matrix:
    ///  - Training workloads (large payloads >100 KB): route to distributed CPU workers
    ///    when multiple nodes exist, otherwise fall back to single-CPU execution.
    ///  - Inference workloads: single CPU unless a CUDA device is available
    ///    (detected via the TARA_CUDA_DEVICE environment variable).
    pub fn route(input_data: &Value) -> (String, Vec<String>) {
        let desc = detect_workload(input_data);
        let workload_type = desc["workload_type"].as_str().unwrap_or("Inference");

        let cuda_available = std::env::var("TARA_CUDA_DEVICE")
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false);

        let mode = match workload_type {
            "Training" => {
                if cuda_available {
                    ExecutionMode::GpuSingle
                } else {
                    ExecutionMode::CpuDistributed
                }
            }
            _ => {
                // Inference
                if cuda_available {
                    ExecutionMode::GpuSingle
                } else {
                    ExecutionMode::CpuSingle
                }
            }
        };

        (mode.as_str().to_string(), vec![])
    }
}

use crate::auto_connect::EndpointDefinition;
use serde_json::{json, Value};

/// Compute the distributed load assignment for a set of nodes.
pub fn distribute_load(nodes: &[EndpointDefinition]) -> Vec<(String, f32)> {
    if nodes.is_empty() {
        return vec![];
    }
    let total_capacity: f32 = nodes.iter().map(|n| n.cpu_capacity).sum();
    nodes
        .iter()
        .map(|n| {
            let share = n.cpu_capacity / total_capacity.max(0.001);
            (n.node_id.clone(), share)
        })
        .collect()
}

/// Aggregate results from multiple distributed nodes.
pub fn aggregate_results(results: Vec<Value>) -> Value {
    json!({
        "aggregate_count": results.len(),
        "results": results,
        "strategy": "FIRST_SUCCESS"
    })
}
