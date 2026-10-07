//! Dynamic Runtime Registry for TARA.
//!
//! Language-neutral registration, state tracking, and parity matrix generation.
//! Enforces:
//! - TARA = ONE MODEL / ONE SYSTEM IDENTITY.
//! - No fixed language order.
//! - Reusable dynamic registry with 10 lifecycle states.
//! - Canonical runtime addition & authenticated retirement flows.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub const CANONICAL_MODEL_IDENTITY: &str = "TARA";
// MODEL_SHA is intentionally NOT hardcoded. The 118,080-parameter smoke-test model has been removed.
// When a real production model is promoted via ModelRegistry::register_active_training(),
// the promoted SHA is stored in versions_manifest.json and read at runtime.
// UNREGISTERED sentinel prevents any stale SHA from being treated as valid.
pub const CANONICAL_MODEL_SHA256_UNREGISTERED: &str = "NO_MODEL_REGISTERED";
pub const CANONICAL_CONTRACT_VERSION: &str = "1.0.0";
pub const CANONICAL_SECURITY_VERSION: &str = "1.0.0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeState {
    DISCOVERED,
    VERIFYING,
    VERIFIED,
    FAILED,
    INCOMPATIBLE,
    UNAVAILABLE,
    DEGRADED,
    QUARANTINED,
    REVOKED,
    RETIRED,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeRecord {
    pub runtime_id: String,
    pub language: String,
    pub implementation_version: String,
    pub model_identity: String,
    pub model_version: String,
    pub model_sha: String,
    pub contract_version: String,
    pub security_version: String,
    pub supported_capabilities: Vec<String>,
    pub status: RuntimeState,
    pub required_for_promotion: bool,
    pub last_verification: Value,
    pub compatibility_state: Value,
}

impl RuntimeRecord {
    pub fn new(runtime_id: &str, language: &str, required: bool) -> Self {
        Self {
            runtime_id: runtime_id.to_string(),
            language: language.to_string(),
            implementation_version: "1.0.0".to_string(),
            model_identity: CANONICAL_MODEL_IDENTITY.to_string(),
            model_version: "1.0.0".to_string(),
            model_sha: CANONICAL_MODEL_SHA256_UNREGISTERED.to_string(),
            contract_version: CANONICAL_CONTRACT_VERSION.to_string(),
            security_version: CANONICAL_SECURITY_VERSION.to_string(),
            supported_capabilities: vec![
                "inference".to_string(),
                "skills".to_string(),
                "tools".to_string(),
                "voice".to_string(),
                "auth".to_string(),
            ],
            status: RuntimeState::VERIFIED,
            required_for_promotion: required,
            last_verification: json!({ "state": "VERIFIED" }),
            compatibility_state: json!({ "contract_compatible": true }),
        }
    }
}

pub struct DynamicRuntimeRegistry {
    runtimes: Arc<Mutex<HashMap<String, RuntimeRecord>>>,
}

impl Default for DynamicRuntimeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicRuntimeRegistry {
    pub fn new() -> Self {
        let mut map = HashMap::new();
        // The application runtime is native Rust; model weights remain SafeTensors data.
        map.insert("rust".to_string(), RuntimeRecord::new("rust", "Rust", true));

        Self {
            runtimes: Arc::new(Mutex::new(map)),
        }
    }

    pub fn get_runtime(&self, runtime_id: &str) -> Option<RuntimeRecord> {
        let lock = self.runtimes.lock().unwrap();
        lock.get(runtime_id).cloned()
    }

    pub fn list_all_runtimes(&self) -> Vec<RuntimeRecord> {
        let lock = self.runtimes.lock().unwrap();
        lock.values().cloned().collect()
    }

    pub fn list_required_runtimes(&self) -> Vec<RuntimeRecord> {
        let lock = self.runtimes.lock().unwrap();
        lock.values()
            .filter(|r| r.required_for_promotion && r.status != RuntimeState::RETIRED)
            .cloned()
            .collect()
    }

    pub fn register_runtime(&self, record: RuntimeRecord) -> Result<RuntimeRecord, String> {
        if record.model_identity != CANONICAL_MODEL_IDENTITY {
            return Err(format!(
                "Invalid model identity: expected {}, got {}",
                CANONICAL_MODEL_IDENTITY, record.model_identity
            ));
        }

        let mut lock = self.runtimes.lock().unwrap();
        let id = record.runtime_id.to_lowercase();
        lock.insert(id.clone(), record.clone());
        Ok(record)
    }

    pub fn update_runtime_state(
        &self,
        runtime_id: &str,
        new_state: RuntimeState,
    ) -> Result<RuntimeRecord, String> {
        let mut lock = self.runtimes.lock().unwrap();
        let id = runtime_id.to_lowercase();
        if let Some(rec) = lock.get_mut(&id) {
            rec.status = new_state;
            Ok(rec.clone())
        } else {
            Err(format!("Runtime '{}' not found", runtime_id))
        }
    }

    pub fn retire_runtime(
        &self,
        runtime_id: &str,
        creator_authenticated: bool,
        reason: &str,
    ) -> Result<Value, String> {
        if !creator_authenticated {
            return Err(
                "Runtime retirement requires authenticated creator authorization.".to_string(),
            );
        }

        let mut lock = self.runtimes.lock().unwrap();
        let id = runtime_id.to_lowercase();
        if let Some(rec) = lock.get_mut(&id) {
            rec.status = RuntimeState::RETIRED;
            rec.required_for_promotion = false;
            rec.last_verification = json!({ "action": "RETIRED", "reason": reason });
            Ok(json!({
                "status": "SUCCESS",
                "runtime_id": id,
                "message": format!("Runtime '{}' successfully retired", id)
            }))
        } else {
            Err(format!("Runtime '{}' not found", runtime_id))
        }
    }

    pub fn verify_runtime_ready(&self, runtime_id: &str) -> (bool, String, Value) {
        let lock = self.runtimes.lock().unwrap();
        let id = runtime_id.to_lowercase();
        if let Some(rec) = lock.get(&id) {
            if rec.model_identity != CANONICAL_MODEL_IDENTITY {
                return (false, "Model identity mismatch".to_string(), json!({}));
            }
            // Fail-closed: if no real model has been promoted, sentinel = NO_MODEL_REGISTERED
            if rec.model_sha == CANONICAL_MODEL_SHA256_UNREGISTERED || rec.model_sha.is_empty() {
                return (
                    false,
                    "No production model registered — promote a real model first".to_string(),
                    json!({"model_sha": "NO_MODEL_REGISTERED"}),
                );
            }
            if rec.contract_version != CANONICAL_CONTRACT_VERSION {
                return (false, "Contract mismatch".to_string(), json!({}));
            }
            if rec.status == RuntimeState::FAILED || rec.status == RuntimeState::QUARANTINED {
                return (false, "Runtime not in healthy state".to_string(), json!({}));
            }
            (
                true,
                "TARA_RUNTIME_READY".to_string(),
                json!({ "model_sha": rec.model_sha, "status": "VERIFIED" }),
            )
        } else {
            (
                false,
                format!("Runtime '{}' not registered", runtime_id),
                json!({}),
            )
        }
    }

    pub fn generate_parity_matrix(&self) -> Value {
        let lock = self.runtimes.lock().unwrap();
        let dimensions = vec!["Model", "Contract", "Security", "Skills", "Tools", "Update"];
        let mut matrix = json!({});

        let mut all_passed = true;
        for (r_id, rec) in lock.iter() {
            let mut row = json!({});
            let is_pass = rec.status == RuntimeState::VERIFIED;
            if rec.required_for_promotion && !is_pass {
                all_passed = false;
            }

            for dim in &dimensions {
                if rec.status == RuntimeState::RETIRED {
                    row[dim] = json!("RETIRED");
                } else if is_pass {
                    row[dim] = json!("PASS");
                } else {
                    row[dim] = json!("FAIL");
                }
            }
            matrix[r_id] = row;
        }

        json!({
            "dimensions": dimensions,
            "matrix": matrix,
            "all_parity_passed": all_passed
        })
    }
}
