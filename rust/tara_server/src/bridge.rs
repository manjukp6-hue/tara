//! Native Rust inference dispatch and runtime telemetry.
//!
//! This module intentionally has no remote model or Python worker transport.

use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

use crate::contract::{CanonicalInferenceRequest, CanonicalInferenceResponse};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerMode {
    NativeRustCpu,
}

pub struct UnifiedBridge {
    stats: Mutex<(u64, u64)>,
    last_failover_latency_ms: Mutex<f64>,
}

impl UnifiedBridge {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            stats: Mutex::new((0, 0)),
            last_failover_latency_ms: Mutex::new(0.0),
        })
    }

    pub fn should_route_to_python(&self, _preferred_engine: Option<&str>) -> bool {
        false
    }

    pub fn probe_worker(&self) -> bool {
        false
    }

    pub fn dispatch_to_python(&self, _path: &str, _payload: &Value) -> Result<Value, String> {
        Err("Remote and Python model workers are disabled; TARA runs native Rust inference".into())
    }

    pub fn infer_with_contract(
        &self,
        req: &CanonicalInferenceRequest,
    ) -> Result<CanonicalInferenceResponse, String> {
        req.validate()?;
        let model_dir =
            std::env::var("TARA_MODEL_DIR").unwrap_or_else(|_| "storage/models/tara".to_string());
        let model = tara_engine::TaraForCausalLM::load(&model_dir)
            .map_err(|error| format!("Failed to load TARA model: {error}"))?;
        let tokenizer =
            tara_engine::TaraTokenizer::from_file(&format!("{model_dir}/tokenizer.json"))
                .map_err(|error| format!("Failed to load tokenizer: {error}"))?;
        let options = tara_engine::GenerateOptions {
            max_new_tokens: req.max_tokens,
            temperature: req.temperature,
            top_k: req.top_k,
            top_p: req.top_p,
            repetition_penalty: req.repetition_penalty,
            stop_tokens: None,
        };
        let result = tara_engine::generate_response(&model, &tokenizer, &req.prompt, &options)
            .map_err(|error| error.to_string())?;
        Ok(CanonicalInferenceResponse {
            request_id: req.request_id.clone(),
            status: "SUCCESS".to_string(),
            text: result.text,
            model_checksum: tara_engine::compute_sha256(&format!(
                "{model_dir}/model.safetensors"
            ))
            .map_err(|error| error.to_string())?,
            runtime_engine: "rust_tara_engine".to_string(),
            token_count: result.token_count,
            token_ids: Vec::new(),
            model_identity: crate::contract::CANONICAL_MODEL_IDENTITY.to_string(),
            first_latency_ms: result.first_latency_ms,
            total_latency_ms: result.total_latency_ms,
            tokens_per_second: result.tps,
        })
    }

    pub fn record_rust_dispatch(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.0 = stats.0.saturating_add(1);
        }
    }

    pub fn record_failover(&self, latency_ms: f64) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.1 = stats.1.saturating_add(1);
        }
        if let Ok(mut last) = self.last_failover_latency_ms.lock() {
            *last = latency_ms.max(0.0);
        }
    }

    pub fn get_telemetry(&self) -> Value {
        let stats = self.stats.lock().map(|value| *value).unwrap_or_default();
        let last_latency = self
            .last_failover_latency_ms
            .lock()
            .map(|value| *value)
            .unwrap_or_default();
        json!({
            "status": "ACTIVE",
            "bridge_architecture": "Native Rust model inference",
            "worker_mode": "NativeRustCpu",
            "remote_workers_enabled": false,
            "metrics": {
                "dispatched_python_gpu": 0,
                "dispatched_rust_cpu": stats.0,
                "automatic_failovers": stats.1,
                "last_failover_latency_ms": last_latency
            }
        })
    }
}
