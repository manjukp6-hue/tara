//! Auto-connect sync engine: multi-node cluster management and workload routing.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
        let node_id = payload
            .get("node_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let url = payload
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if node_id.trim().is_empty() {
            return json!({"status":"ERROR","error":"node_id is required"});
        }
        let parsed_url = match url::Url::parse(&url) {
            Ok(parsed) if parsed.scheme() == "http" && parsed.host_str().is_some() => parsed,
            _ => return json!({"status":"ERROR","error":"endpoint url must be a valid http URL"}),
        };
        let capabilities: Vec<String> = payload
            .get("capabilities")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let cpu_capacity = payload
            .get("cpu_capacity")
            .and_then(Value::as_f64)
            .unwrap_or(1.0) as f32;
        let health = payload.get("health").and_then(Value::as_f64).unwrap_or(1.0) as f32;
        if !cpu_capacity.is_finite()
            || cpu_capacity <= 0.0
            || !health.is_finite()
            || !(0.0..=1.0).contains(&health)
        {
            return json!({"status":"ERROR","error":"cpu_capacity must be positive and health must be between 0 and 1"});
        }
        let ep = EndpointDefinition {
            node_id: node_id.clone(),
            url,
            capabilities,
            health: 0.0,
            cpu_capacity,
            ram_gb: payload
                .get("ram_gb")
                .and_then(|v| v.as_f64())
                .unwrap_or(8.0) as f32,
            gpu_available: payload
                .get("gpu_available")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            gpu_model: payload
                .get("gpu_model")
                .and_then(|v| v.as_str())
                .map(String::from),
            gpu_vram_gb: payload
                .get("gpu_vram_gb")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),
            current_load: 0.0,
            queue_depth: 0,
            latency_ms: 0.0,
            model_sha256: payload
                .get("model_sha256")
                .and_then(|v| v.as_str())
                .map(String::from),
            registered_at: Instant::now(),
        };

        self.endpoints.lock().unwrap().insert(node_id.clone(), ep);
        let probe = self.probe_endpoint(&node_id, &parsed_url);
        json!({ "status": if probe["reachable"] == true { "SUCCESS" } else { "ERROR" }, "registered": node_id, "probe": probe })
    }

    /// List all registered endpoints.
    pub fn list_endpoints(&self) -> Vec<Value> {
        self.endpoints
            .lock()
            .unwrap()
            .values()
            .map(|ep| ep.to_json())
            .collect()
    }

    /// Get the best endpoint for given capabilities.
    pub fn get_active_endpoint(&self, required_caps: Option<&[String]>) -> Option<Value> {
        let endpoints = self.endpoints.lock().unwrap();
        let best = endpoints
            .values()
            .filter(|ep| {
                required_caps
                    .map(|caps| caps.iter().all(|c| ep.capabilities.contains(c)))
                    .unwrap_or(true)
            })
            .filter(|ep| ep.health > 0.5)
            .min_by(|a, b| {
                let score_a = a.current_load / a.cpu_capacity.max(0.01);
                let score_b = b.current_load / b.cpu_capacity.max(0.01);
                score_a.partial_cmp(&score_b).unwrap()
            });
        best.map(|ep| ep.to_json())
    }

    /// Probe endpoint health over HTTP and refresh its health/latency measurements.
    pub fn probe_all_endpoints(&self) -> Vec<Value> {
        let endpoints: Vec<(String, String)> = self
            .endpoints
            .lock()
            .unwrap()
            .values()
            .map(|ep| (ep.node_id.clone(), ep.url.clone()))
            .collect();
        endpoints
            .into_iter()
            .map(|(node_id, endpoint)| match url::Url::parse(&endpoint) {
                Ok(url) => self.probe_endpoint(&node_id, &url),
                Err(error) => {
                    json!({"node_id":node_id,"reachable":false,"error":error.to_string()})
                }
            })
            .collect()
    }

    fn probe_endpoint(&self, node_id: &str, endpoint: &url::Url) -> Value {
        let started = Instant::now();
        let result = (|| -> Result<u16, String> {
            let host = endpoint.host_str().ok_or("endpoint has no host")?;
            let port = endpoint
                .port_or_known_default()
                .ok_or("endpoint has no port")?;
            let address = (host, port)
                .to_socket_addrs()
                .map_err(|e| e.to_string())?
                .next()
                .ok_or("endpoint host did not resolve")?;
            let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(2))
                .map_err(|e| e.to_string())?;
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .map_err(|e| e.to_string())?;
            stream
                .set_write_timeout(Some(Duration::from_secs(2)))
                .map_err(|e| e.to_string())?;
            let path = format!("{}/health", endpoint.path().trim_end_matches('/'));
            write!(stream, "GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nAccept: application/json\r\n\r\n", path, host).map_err(|e| e.to_string())?;
            let mut response = Vec::new();
            stream
                .take(16_384)
                .read_to_end(&mut response)
                .map_err(|e| e.to_string())?;
            let response = String::from_utf8_lossy(&response);
            let status = response
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|value| value.parse::<u16>().ok())
                .ok_or("invalid HTTP status line")?;
            if !(200..300).contains(&status) {
                return Err(format!("health endpoint returned HTTP {status}"));
            }
            Ok(status)
        })();
        let latency_ms = started.elapsed().as_secs_f32() * 1000.0;
        let reachable = result.is_ok();
        if let Some(endpoint) = self.endpoints.lock().unwrap().get_mut(node_id) {
            endpoint.health = if reachable { 1.0 } else { 0.0 };
            endpoint.latency_ms = latency_ms;
        }
        match result {
            Ok(status) => {
                json!({"node_id":node_id,"reachable":true,"http_status":status,"latency_ms":latency_ms})
            }
            Err(error) => {
                json!({"node_id":node_id,"reachable":false,"health":0.0,"latency_ms":latency_ms,"error":error})
            }
        }
    }

    /// Accept a sync package from another node.
    pub fn receive_sync_package(&self, payload: &Value) -> (bool, String) {
        // Validate basic structure
        let node_id = payload
            .get("node_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
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
        let required_caps = payload
            .get("required_capabilities")
            .and_then(Value::as_array)
            .map(|caps| {
                caps.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            });
        let endpoint = self.get_active_endpoint(required_caps.as_deref());
        if let Some(selected) = endpoint {
            json!({"status":"SUCCESS","selected_endpoint":selected,"routing_strategy":"LEAST_LOADED"})
        } else {
            json!({"status":"ERROR","error":"No reachable endpoint satisfies the requested capabilities"})
        }
    }
}
