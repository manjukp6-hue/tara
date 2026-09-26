//! Auto-connect sync engine: multi-node cluster management and workload routing.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

/// An endpoint registered in the cluster.
#[derive(Debug, Clone)]
pub struct EndpointDefinition {
    pub node_id: String,
    pub url: String,
    pub capabilities: Vec<String>,
    pub health: f32,
    pub cpu_capacity: f32,
    pub ram_gb: f32,
    pub gpu_available: bool,
    pub gpu_model: Option<String>,
    pub gpu_vram_gb: Option<f32>,
    pub current_load: f32,
    pub queue_depth: usize,
    pub latency_ms: f32,
    pub model_sha256: Option<String>,
    pub registered_at: Instant,
}

impl EndpointDefinition {
    fn to_json(&self) -> Value {
        json!({
            "node_id": self.node_id,
            "url": self.url,
            "capabilities": self.capabilities,
            "health": self.health,
            "cpu_capacity": self.cpu_capacity,
            "ram_gb": self.ram_gb,
            "gpu_available": self.gpu_available,
            "gpu_model": self.gpu_model,
            "gpu_vram_gb": self.gpu_vram_gb,
            "current_load": self.current_load,
            "queue_depth": self.queue_depth,
            "latency_ms": self.latency_ms,
            "model_sha256": self.model_sha256,
        })
    }
}

/// Manages cluster endpoint registration, selection, and workload routing.
pub struct AutoConnectSyncEngine {
    endpoints: Mutex<HashMap<String, EndpointDefinition>>,
    offline_journal: Mutex<Vec<Value>>,
}

impl AutoConnectSyncEngine {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            endpoints: Mutex::new(HashMap::new()),
            offline_journal: Mutex::new(Vec::new()),
        })
    }

    /// Register an endpoint from JSON payload.
    pub fn register_endpoint_from_json(&self, payload: &Value) -> Value {
        let node_id = payload.get("node_id").and_then(|v| v.as_str())
            .unwrap_or("node_unknown").to_string();
        let url = payload.get("url").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let capabilities: Vec<String> = payload.get("capabilities")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        let ep = EndpointDefinition {
            node_id: node_id.clone(),
            url,
            capabilities,
            health: payload.get("health").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
            cpu_capacity: payload.get("cpu_capacity").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
            ram_gb: payload.get("ram_gb").and_then(|v| v.as_f64()).unwrap_or(8.0) as f32,
            gpu_available: payload.get("gpu_available").and_then(|v| v.as_bool()).unwrap_or(false),
            gpu_model: payload.get("gpu_model").and_then(|v| v.as_str()).map(String::from),
            gpu_vram_gb: payload.get("gpu_vram_gb").and_then(|v| v.as_f64()).map(|v| v as f32),
            current_load: 0.0,
            queue_depth: 0,
            latency_ms: 0.0,
            model_sha256: payload.get("model_sha256").and_then(|v| v.as_str()).map(String::from),
            registered_at: Instant::now(),
        };

        self.endpoints.lock().unwrap().insert(node_id.clone(), ep);
        json!({ "status": "SUCCESS", "registered": node_id })
    }

    /// List all registered endpoints.
    pub fn list_endpoints(&self) -> Vec<Value> {
        self.endpoints.lock().unwrap().values().map(|ep| ep.to_json()).collect()
    }

    /// Get the best endpoint for given capabilities.
    pub fn get_active_endpoint(&self, required_caps: Option<&[String]>) -> Option<Value> {
        let endpoints = self.endpoints.lock().unwrap();
        let best = endpoints.values()
            .filter(|ep| {
                required_caps.map(|caps| {
                    caps.iter().all(|c| ep.capabilities.contains(c))
                }).unwrap_or(true)
            })
            .filter(|ep| ep.health > 0.5)
            .min_by(|a, b| {
                let score_a = a.current_load / a.cpu_capacity.max(0.01);
                let score_b = b.current_load / b.cpu_capacity.max(0.01);
                score_a.partial_cmp(&score_b).unwrap()
            });
        best.map(|ep| ep.to_json())
    }

    /// Probe all endpoints (stub — real implementation would HTTP GET /health).
    pub fn probe_all_endpoints(&self) -> Vec<Value> {
        self.endpoints.lock().unwrap().values()
            .map(|ep| json!({ "node_id": ep.node_id, "reachable": true, "health": ep.health }))
            .collect()
    }

    /// Accept a sync package from another node.
    pub fn receive_sync_package(&self, payload: &Value) -> (bool, String) {
        // Validate basic structure
        let node_id = payload.get("node_id").and_then(|v| v.as_str()).unwrap_or("");
        if node_id.is_empty() {
            return (false, "Missing node_id in sync package".to_string());
        }
        self.offline_journal.lock().unwrap().push(payload.clone());
        (true, format!("Sync package from '{}' accepted", node_id))
    }

    /// Get current sync status.
    pub fn get_status(&self) -> Value {
        let endpoints = self.endpoints.lock().unwrap();
        let journal_len = self.offline_journal.lock().unwrap().len();
        json!({
            "registered_endpoints": endpoints.len(),
            "offline_journal_entries": journal_len,
            "local_node": "local",
        })
    }

    /// Flush the offline journal (return and clear).
    pub fn flush_offline_journal(&self) -> Value {
        let mut journal = self.offline_journal.lock().unwrap();
        let entries = journal.drain(..).collect::<Vec<_>>();
        json!({ "flushed": entries.len(), "entries": entries })
    }

    /// Route a workload request to the best endpoint.
    pub fn route_workload(&self, payload: &Value) -> Value {
        let endpoint = self.get_active_endpoint(None);
        json!({
            "status": "SUCCESS",
            "selected_endpoint": endpoint,
            "routing_strategy": "LEAST_LOADED"
        })
    }
}
