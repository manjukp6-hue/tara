//! Dynamic engine system: manifest registration, quarantine, health.

use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

type EngineExecutor = Arc<dyn Fn(Value) -> Result<Value, String> + Send + Sync>;

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
    executors: Mutex<HashMap<String, EngineExecutor>>,
}

impl DynamicEngineSystem {
    pub fn new() -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            engines: Mutex::new(HashMap::new()),
            executors: Mutex::new(HashMap::new()),
        })
    }

    pub fn list_engines(&self) -> Vec<Value> {
        self.engines
            .lock()
            .unwrap()
            .values()
            .map(|e| {
                json!({
                    "engine_id": e.engine_id,
                    "name": e.name,
                    "version": e.version,
                    "capabilities": e.capabilities,
                    "status": format!("{:?}", e.status),
                    "quarantine_reason": e.quarantine_reason
                })
            })
            .collect()
    }

    pub fn register_manifest_from_json(&self, payload: &Value) -> (bool, Option<String>) {
        let engine_id = payload
            .get("engine_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if engine_id.is_empty() {
            return (false, Some("engine_id is required".to_string()));
        }
        let caps: Vec<String> = payload
            .get("capabilities")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        self.engines.lock().unwrap().insert(
            engine_id.clone(),
            EngineManifest {
                engine_id: engine_id.clone(),
                name: payload
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&engine_id)
                    .to_string(),
                version: payload
                    .get("version")
                    .and_then(|v| v.as_str())
                    .unwrap_or("1.0.0")
                    .to_string(),
                capabilities: caps,
                // A manifest is metadata; it is not an executable backend.
                status: EngineStatus::Offline,
                quarantine_reason: None,
            },
        );
        (true, None)
    }

    pub fn execute(&self, task_type: &str, payload: Value, engine_id: Option<&str>) -> Value {
        let selected = {
            let engines = self.engines.lock().unwrap();
            if let Some(eid) = engine_id {
                engines
                    .get(eid)
                    .filter(|e| {
                        e.status == EngineStatus::Active
                            && e.capabilities.iter().any(|cap| cap == task_type)
                    })
                    .map(|e| e.engine_id.clone())
            } else {
                engines
                    .values()
                    .find(|e| {
                        e.status == EngineStatus::Active
                            && e.capabilities.iter().any(|cap| cap == task_type)
                    })
                    .map(|e| e.engine_id.clone())
            }
        };
        let Some(selected) = selected else {
            return json!({"status":"ERROR","error":format!("No active executable engine for task '{}'", task_type)});
        };
        let executor = self.executors.lock().unwrap().get(&selected).cloned();
        let Some(executor) = executor else {
            return json!({"status":"ERROR","error":format!("Engine '{}' has no executable backend", selected)});
        };
        match executor(payload) {
            Ok(result) => {
                json!({"status":"SUCCESS","engine_id":selected,"task_type":task_type,"result":result})
            }
            Err(error) => {
                json!({"status":"ERROR","engine_id":selected,"task_type":task_type,"error":error})
            }
        }
    }

    /// Attach a real in-process implementation to a previously registered capability manifest.
    pub fn register_executor<F>(&self, engine_id: &str, executor: F) -> Result<(), String>
    where
        F: Fn(Value) -> Result<Value, String> + Send + Sync + 'static,
    {
        let mut engines = self.engines.lock().map_err(|e| e.to_string())?;
        let engine = engines
            .get_mut(engine_id)
            .ok_or_else(|| format!("Engine '{}' is not registered", engine_id))?;
        if engine.capabilities.is_empty() {
            return Err("An executable engine must declare at least one capability".to_string());
        }
        self.executors
            .lock()
            .map_err(|e| e.to_string())?
            .insert(engine_id.to_string(), Arc::new(executor));
        engine.status = EngineStatus::Active;
        engine.quarantine_reason = None;
        Ok(())
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
        let active = engines
            .values()
            .filter(|e| e.status == EngineStatus::Active)
            .count();
        let quarantined = engines
            .values()
            .filter(|e| e.status == EngineStatus::Quarantined)
            .count();
        json!({ "total": total, "active": active, "quarantined": quarantined })
    }
}
