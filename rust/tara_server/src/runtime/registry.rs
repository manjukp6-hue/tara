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
        let id = runtime_id.trim().to_lowercase();
        assert!(!id.is_empty(), "Runtime ID cannot be empty");
        Self {
            runtime_id: id,
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
        let lock = self.runtimes.lock().unwrap_or_else(|p| p.into_inner());
        let id = runtime_id.trim().to_lowercase();
        lock.get(&id).cloned()
    }

    pub fn list_all_runtimes(&self) -> Vec<RuntimeRecord> {
        let lock = self.runtimes.lock().unwrap_or_else(|p| p.into_inner());
        lock.values().cloned().collect()
    }

    pub fn list_required_runtimes(&self) -> Vec<RuntimeRecord> {
        let lock = self.runtimes.lock().unwrap_or_else(|p| p.into_inner());
        lock.values()
            .filter(|r| r.required_for_promotion && r.status != RuntimeState::RETIRED)
            .cloned()
            .collect()
    }

    pub fn register_runtime(&self, mut record: RuntimeRecord) -> Result<RuntimeRecord, String> {
        let id = record.runtime_id.trim().to_lowercase();
        if id.is_empty() {
            return Err("Runtime ID cannot be empty".to_string());
        }
        record.runtime_id = id.clone();

        if record.model_identity != CANONICAL_MODEL_IDENTITY {
            return Err(format!(
                "Invalid model identity: expected {}, got {}",
                CANONICAL_MODEL_IDENTITY, record.model_identity
            ));
        }

        let mut lock = self.runtimes.lock().unwrap_or_else(|p| p.into_inner());
        lock.insert(id, record.clone());
        Ok(record)
    }

    pub fn update_runtime_state(
        &self,
        runtime_id: &str,
        new_state: RuntimeState,
    ) -> Result<RuntimeRecord, String> {
        let mut lock = self.runtimes.lock().unwrap_or_else(|p| p.into_inner());
        let id = runtime_id.trim().to_lowercase();
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
        principal: &tara_security::primitives::AuthenticatedPrincipal,
        reason: &str,
    ) -> Result<Value, String> {
        if !principal.is_creator() {
            return Err(
                "Runtime retirement requires authenticated creator authorization.".to_string(),
            );
        }

        let mut lock = self.runtimes.lock().unwrap_or_else(|p| p.into_inner());
        let id = runtime_id.trim().to_lowercase();

        let is_active_required = match lock.get(&id) {
            Some(rec) => rec.required_for_promotion && rec.status != RuntimeState::RETIRED,
            None => return Err(format!("Runtime '{}' not found", runtime_id)),
        };

        if is_active_required {
            let active_required_count = lock
                .values()
                .filter(|r| r.required_for_promotion && r.status != RuntimeState::RETIRED)
                .count();
            if active_required_count <= 1 {
                return Err(format!(
                    "Cannot retire runtime '{}': it is the last remaining non-retired required runtime for promotion.",
                    id
                ));
            }
        }

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
        let lock = self.runtimes.lock().unwrap_or_else(|p| p.into_inner());
        let id = runtime_id.trim().to_lowercase();
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
            if rec.security_version != CANONICAL_SECURITY_VERSION {
                return (false, "Security version mismatch".to_string(), json!({}));
            }
            if rec.status != RuntimeState::VERIFIED {
                return (
                    false,
                    format!("Runtime not in healthy verified state: {:?}", rec.status),
                    json!({ "status": format!("{:?}", rec.status) }),
                );
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
        let lock = self.runtimes.lock().unwrap_or_else(|p| p.into_inner());
        let dimensions = vec!["Model", "Contract", "Security", "Skills", "Tools", "Update"];
        let mut matrix = json!({});

        let mut all_passed = true;
        for (r_id, rec) in lock.iter() {
            let mut row = json!({});
            let is_pass = rec.status == RuntimeState::VERIFIED
                && rec.model_identity == CANONICAL_MODEL_IDENTITY
                && rec.contract_version == CANONICAL_CONTRACT_VERSION
                && rec.security_version == CANONICAL_SECURITY_VERSION;
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
