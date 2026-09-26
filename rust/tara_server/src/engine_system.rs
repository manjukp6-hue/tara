//! Dynamic engine system: manifest registration, quarantine, health.

use std::collections::HashMap;
use std::sync::Mutex;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub enum EngineStatus {
    Active,
    Quarantined,
    Offline,
}

#[derive(Debug, Clone)]
pub struct EngineManifest {
    pub engine_id: String,
    pub name: String,
    pub version: String,
    pub capabilities: Vec<String>,
    pub status: EngineStatus,
    pub quarantine_reason: Option<String>,
}

pub struct DynamicEngineSystem {
    engines: Mutex<HashMap<String, EngineManifest>>,
}

impl DynamicEngineSystem {
    pub fn new() -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            engines: Mutex::new(HashMap::new()),
        })
    }

    pub fn list_engines(&self) -> Vec<Value> {
        self.engines.lock().unwrap().values().map(|e| json!({
            "engine_id": e.engine_id,
            "name": e.name,
            "version": e.version,
            "capabilities": e.capabilities,
            "status": format!("{:?}", e.status),
            "quarantine_reason": e.quarantine_reason
        })).collect()
    }

    pub fn register_manifest_from_json(&self, payload: &Value) -> (bool, Option<String>) {
        let engine_id = payload.get("engine_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        if engine_id.is_empty() {
            return (false, Some("engine_id is required".to_string()));
        }
        let caps: Vec<String> = payload.get("capabilities").and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        self.engines.lock().unwrap().insert(engine_id.clone(), EngineManifest {
            engine_id: engine_id.clone(),
            name: payload.get("name").and_then(|v| v.as_str()).unwrap_or(&engine_id).to_string(),
            version: payload.get("version").and_then(|v| v.as_str()).unwrap_or("1.0.0").to_string(),
            capabilities: caps,
            status: EngineStatus::Active,
            quarantine_reason: None,
        });
        (true, None)
    }

    pub fn execute(&self, task_type: &str, payload: Value, engine_id: Option<&str>) -> Value {
        let engines = self.engines.lock().unwrap();
        let engine = if let Some(eid) = engine_id {
            engines.get(eid)
        } else {
            engines.values()
                .find(|e| e.status == EngineStatus::Active && e.capabilities.contains(&task_type.to_string()))
        };

        if let Some(e) = engine {
            json!({
                "engine_id": e.engine_id,
                "task_type": task_type,
                "status": "SUCCESS",
                "result": { "executed_by": e.engine_id, "task": task_type, "payload_echo": payload }
            })
        } else {
            json!({ "status": "ERROR", "error": format!("No active engine for task '{}'", task_type) })
        }
    }

    pub fn quarantine_engine(&self, engine_id: &str, reason: &str) -> bool {
        if let Some(e) = self.engines.lock().unwrap().get_mut(engine_id) {
            e.status = EngineStatus::Quarantined;
            e.quarantine_reason = Some(reason.to_string());
            true
        } else {
            false
        }
    }

    pub fn get_health_report(&self) -> Value {
        let engines = self.engines.lock().unwrap();
        let total = engines.len();
        let active = engines.values().filter(|e| e.status == EngineStatus::Active).count();
        let quarantined = engines.values().filter(|e| e.status == EngineStatus::Quarantined).count();
        json!({ "total": total, "active": active, "quarantined": quarantined })
    }
}
