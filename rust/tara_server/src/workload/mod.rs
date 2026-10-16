//! Workload distribution: detect, route, distribute, aggregate across compute nodes.

pub mod workload_distributor {
    use serde_json::{json, Value};
    use std::io::Write;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum ExecutionMode {
        CpuSingle,
        CpuDistributed,
        GpuSingle,
        CpuGpuHybrid,
        TpuAccelerated,
    }

    impl ExecutionMode {
        pub fn as_str(&self) -> &'static str {
            match self {
                ExecutionMode::CpuSingle => "CpuSingle",
                ExecutionMode::CpuDistributed => "CpuDistributed",
                ExecutionMode::GpuSingle => "GpuSingle",
                ExecutionMode::CpuGpuHybrid => "CpuGpuHybrid",
                ExecutionMode::TpuAccelerated => "TpuAccelerated",
            }
        }
    }

    /// Zero-allocation byte counting writer to measure serialized JSON size.
    struct ByteCountWriter {
        count: usize,
    }

    impl Write for ByteCountWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.count = self.count.saturating_add(buf.len());
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    pub fn detect_workload(input_data: &Value) -> Value {
        let mut writer = ByteCountWriter { count: 0 };
        let size = if serde_json::to_writer(&mut writer, input_data).is_ok() {
            writer.count
        } else {
            0
        };

        let explicit_task = input_data
            .get("task_name")
            .and_then(|v| v.as_str())
            .or_else(|| input_data.get("task_type").and_then(|v| v.as_str()))
            .unwrap_or("");

        let is_training_task = explicit_task.eq_ignore_ascii_case("training")
            || explicit_task.eq_ignore_ascii_case("self_train")
            || explicit_task.eq_ignore_ascii_case("selftrain")
            || explicit_task.eq_ignore_ascii_case("train");

        let workload_type = if is_training_task || size > 100_000 {
            "Training"
        } else {
            "Inference"
        };

        let task_name = if !explicit_task.is_empty() {
            explicit_task
        } else if workload_type == "Training" {
            "training"
        } else {
            "inference"
        };

        json!({
            "task_name": task_name,
            "workload_type": workload_type,
            "input_size": size,
            "estimated_ram_mb": (size / 1000).max(64),
        })
    }

    /// Detect the workload characteristics and select the appropriate execution mode.
    ///
    /// Decision matrix via TARA-HD dynamic compute channel:
    ///  - GPU available: GpuSingle (or CpuGpuHybrid if requested).
    ///  - TPU available: TpuAccelerated.
    ///  - CPU fallback: CpuDistributed for large training payloads, CpuSingle for inference.
    pub fn route(input_data: &Value) -> (String, Vec<String>) {
        let desc = detect_workload(input_data);
        let workload_type = desc["workload_type"].as_str().unwrap_or("Inference");

        let channel = tara_hd::ComputeChannel::auto();
        let (has_gpu, has_tpu) = channel
            .map(|ch| (ch.is_gpu(), ch.is_tpu()))
            .unwrap_or((false, false));

        let hybrid_requested = input_data
            .get("hybrid")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let mode = if has_gpu && hybrid_requested {
            ExecutionMode::CpuGpuHybrid
        } else if has_gpu {
            ExecutionMode::GpuSingle
        } else if has_tpu {
            ExecutionMode::TpuAccelerated
        } else {
            match workload_type {
                "Training" => ExecutionMode::CpuDistributed,
                _ => ExecutionMode::CpuSingle,
            }
        };

        (mode.as_str().to_string(), vec![])
    }
}

use crate::control_plane::auto_connect::EndpointDefinition;
use serde_json::{json, Value};

/// Compute the distributed load assignment for a set of nodes.
///
/// Health-aware and NaN-safe:
/// - Excludes unhealthy nodes (`health <= 0.0`) when at least one healthy node exists.
/// - Sanitizes non-finite or non-positive `cpu_capacity` values to 1.0.
/// - Guarantees shares sum to 1.0.
pub fn distribute_load(nodes: &[EndpointDefinition]) -> Vec<(String, f32)> {
    if nodes.is_empty() {
        return vec![];
    }

    let has_healthy = nodes.iter().any(|n| n.health.is_finite() && n.health > 0.0);
    let candidate_nodes: Vec<&EndpointDefinition> = if has_healthy {
        nodes
            .iter()
            .filter(|n| !n.node_id.trim().is_empty() && n.health.is_finite() && n.health > 0.0)
            .collect()
    } else {
        nodes
            .iter()
            .filter(|n| !n.node_id.trim().is_empty())
            .collect()
    };

    if candidate_nodes.is_empty() {
        return vec![];
    }

    let sanitized_capacities: Vec<f32> = candidate_nodes
        .iter()
        .map(|n| {
            if n.cpu_capacity.is_finite() && n.cpu_capacity > 0.0 {
                n.cpu_capacity
            } else {
                1.0
            }
        })
        .collect();

    let total_capacity: f32 = sanitized_capacities.iter().sum();
    let safe_total = total_capacity.max(0.001);

    candidate_nodes
        .iter()
        .zip(sanitized_capacities.iter())
        .map(|(n, &cap)| {
            let share = cap / safe_total;
            (n.node_id.clone(), share)
        })
        .collect()
}

/// Aggregate results from multiple distributed nodes using true FIRST_SUCCESS strategy.
pub fn aggregate_results(results: Vec<Value>) -> Value {
    let selected_result = results
        .iter()
        .find(|res| {
            if let Some(status) = res.get("status").and_then(|s| s.as_str()) {
                if status.eq_ignore_ascii_case("ERROR") || status.eq_ignore_ascii_case("FAILED") {
                    return false;
                }
            }
            if res.get("error").is_some() {
                return false;
            }
            true
        })
        .cloned()
        .unwrap_or(Value::Null);

    let count = results.len();
    json!({
        "aggregate_count": count,
        "results": results,
        "selected_result": selected_result,
        "strategy": "FIRST_SUCCESS"
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn make_test_node(id: &str, health: f32, capacity: f32) -> EndpointDefinition {
        EndpointDefinition {
            node_id: id.to_string(),
            url: format!("http://{}:8080", id),
            capabilities: vec!["inference".to_string()],
            health,
            cpu_capacity: capacity,
            ram_gb: 16.0,
            gpu_available: false,
            gpu_model: None,
            gpu_vram_gb: None,
            current_load: 0.1,
            queue_depth: 0,
            latency_ms: 12.0,
            model_sha256: None,
            registered_at: Instant::now(),
        }
    }

    #[test]
    fn test_distribute_load_health_and_nan_safety() {
        let nodes = vec![
            make_test_node("node-1", 1.0, 2.0),
            make_test_node("node-unhealthy", 0.0, 4.0),
            make_test_node("node-nan", 0.8, f32::NAN),
            make_test_node("node-dead", -0.5, 3.0),
            make_test_node("node-2", 0.5, 0.0), // zero capacity -> sanitized to 1.0
        ];

        let shares = distribute_load(&nodes);

        // Verify unhealthy and dead nodes are excluded
        assert!(shares.iter().all(|(id, _)| id != "node-unhealthy" && id != "node-dead"));
        assert_eq!(shares.len(), 3);

        // Check shares sum to 1.0
        let total_share: f32 = shares.iter().map(|(_, s)| s).sum();
        assert!(
            (total_share - 1.0).abs() < 1e-5,
            "Shares sum must equal 1.0, got {}",
            total_share
        );

        // Check individual shares: node-1 (2.0), node-nan (1.0), node-2 (1.0) -> total 4.0
        let s1 = shares.iter().find(|(id, _)| id == "node-1").unwrap().1;
        let s_nan = shares.iter().find(|(id, _)| id == "node-nan").unwrap().1;
        let s2 = shares.iter().find(|(id, _)| id == "node-2").unwrap().1;

        assert!((s1 - 0.5).abs() < 1e-5);
        assert!((s_nan - 0.25).abs() < 1e-5);
        assert!((s2 - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_detect_workload_and_routing() {
        use workload_distributor::*;

        // Small inference payload
        let small = json!({"prompt": "hello"});
        let desc = detect_workload(&small);
        assert_eq!(desc["workload_type"], "Inference");

        // Explicit training task
        let train_task = json!({"task_name": "training", "epochs": 3});
        let desc_train = detect_workload(&train_task);
        assert_eq!(desc_train["workload_type"], "Training");

        // Hybrid routing when CUDA available and hybrid requested
        std::env::set_var("TARA_CUDA_DEVICE", "0");
        let hybrid_input = json!({"hybrid": true, "task_type": "inference"});
        let (mode, _) = route(&hybrid_input);
        assert_eq!(mode, "CpuGpuHybrid");
        std::env::remove_var("TARA_CUDA_DEVICE");
    }

    #[test]
    fn test_aggregate_results_first_success() {
        let results = vec![
            json!({"status": "ERROR", "message": "timeout"}),
            json!({"error": "connection failed"}),
            json!({"status": "SUCCESS", "output": "computed 42"}),
            json!({"status": "SUCCESS", "output": "computed 99"}),
        ];

        let agg = aggregate_results(results);
        assert_eq!(agg["aggregate_count"], 4);
        assert_eq!(agg["strategy"], "FIRST_SUCCESS");
        assert_eq!(agg["selected_result"]["status"], "SUCCESS");
        assert_eq!(agg["selected_result"]["output"], "computed 42");
    }
}
