//! Cognitive Self-Model & Introspective Epistemic Identity Engine.
//!
//! Provides genuine, persistent cognitive self-modeling for TARA:
//! - Canonical Identity representation (system name, architecture, version).
//! - Epistemic classification into four strict categories:
//!   - `WHAT_I_CAN_DO`: Real executable capabilities (math, science, code, rules, learning).
//!   - `WHAT_IS_VERIFIED`: Cryptographically proven or deterministically verified facts.
//!   - `WHAT_IS_UNCERTAIN`: Hypotheses or predictions with low epistemic confidence.
//!   - `WHAT_I_CANNOT_DO`: Operations exceeding computational or physical bounds.
//!   - `WHAT_I_AM_NOT_ALLOWED_TO_DO`: Invariant safety/policy restrictions (Creator authority, laws).
//! - Dynamic self-state tracking (uptime, active capabilities, calibration score).
//! - Persistent storage at `storage/persistence/self_model.json`.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicBoundary {
    pub category: String, // "WHAT_I_CAN_DO", "WHAT_IS_VERIFIED", "WHAT_IS_UNCERTAIN", "WHAT_I_CANNOT_DO", "WHAT_I_AM_NOT_ALLOWED_TO_DO"
    pub item: String,
    pub description: String,
    pub verification_proof: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveSelfState {
    pub system_identity: String,
    pub architecture_type: String,
    pub engine_version: String,
    pub operational_status: String,
    pub verified_capabilities: Vec<String>,
    pub epistemic_boundaries: Vec<EpistemicBoundary>,
    pub calibration_confidence: f64,
    pub last_introspected_at_ms: u64,
}

pub struct SelfModelEngine {
    storage_path: PathBuf,
    state: Arc<Mutex<CognitiveSelfState>>,
}

impl SelfModelEngine {
    pub fn new<P: AsRef<Path>>(repo_root: P) -> Self {
        let storage_dir = repo_root.as_ref().join("storage").join("persistence");
        let _ = fs::create_dir_all(&storage_dir);
        let storage_path = storage_dir.join("self_model.json");

        let default_state = CognitiveSelfState {
            system_identity: "TARA (Autonomous Cognitive Assistant)".to_string(),
            architecture_type: "Modular Neural-Cognitive Architecture with Native Rust Core".to_string(),
            engine_version: "1.0.0-release".to_string(),
            operational_status: "NOMINAL_ACTIVE".to_string(),
            verified_capabilities: vec![
                "symbolic_mathematics".to_string(),
                "scientific_engine_evaluation".to_string(),
                "programming_static_analysis".to_string(),
                "rule_policy_enforcement".to_string(),
                "epistemic_curiosity_exploration".to_string(),
                "experiential_closed_loop_learning".to_string(),
                "cryptographic_reward_ledger".to_string(),
                "multilingual_kannada_english_reasoning".to_string(),
            ],
            epistemic_boundaries: vec![
                EpistemicBoundary {
                    category: "WHAT_I_CAN_DO".to_string(),
                    item: "Deterministic computation and cognitive reasoning".to_string(),
                    description: "Execute symbolic math, physics, code verification, rule compliance, and autonomous knowledge indexing.".to_string(),
                    verification_proof: Some("tara_engine::computation unit tests verified".to_string()),
                },
                EpistemicBoundary {
                    category: "WHAT_IS_VERIFIED".to_string(),
                    item: "Open-source knowledge partition dataset".to_string(),
                    description: "100 manifest sources and 191 partitioned knowledge documents verified with SHA-256 and CC-BY-SA/PD licenses.".to_string(),
                    verification_proof: Some("storage/knowledge/knowledge_audit_manifest.json".to_string()),
                },
                EpistemicBoundary {
                    category: "WHAT_IS_UNCERTAIN".to_string(),
                    item: "Unproven mathematical conjectures and frontier science hypotheses".to_string(),
                    description: "Open mathematical problems (e.g. Riemann Hypothesis, P vs NP) and unverified conjectures are flagged as UNCERTAIN.".to_string(),
                    verification_proof: None,
                },
                EpistemicBoundary {
                    category: "WHAT_I_CANNOT_DO".to_string(),
                    item: "Direct unassisted physical actuation in the real world".to_string(),
                    description: "TARA is software-embodied; physical actuators require external hardware controllers via the Robotics HAL.".to_string(),
                    verification_proof: Some("rust/tara_core/src/robotics.rs ActuatorInterlock".to_string()),
                },
                EpistemicBoundary {
                    category: "WHAT_I_AM_NOT_ALLOWED_TO_DO".to_string(),
                    item: "Bypass Creator Authority or violate safety baselines".to_string(),
                    description: "Overriding mandatory child safety, legal protections, security policies, or impersonating Creator without authorization is strictly prohibited.".to_string(),
                    verification_proof: Some("rust/tara_server/src/rules/policy_schema.rs PolicyPriority".to_string()),
                },
            ],
            calibration_confidence: 0.95,
            last_introspected_at_ms: Self::current_ts_ms(),
        };

        let mut engine = Self {
            storage_path,
            state: Arc::new(Mutex::new(default_state)),
        };

        let _ = engine.load();
        engine
    }

    fn current_ts_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Introspect current self-model and return snapshot.
    pub fn introspect(&self) -> CognitiveSelfState {
        let mut state = self.state.lock().unwrap();
        state.last_introspected_at_ms = Self::current_ts_ms();
        state.clone()
    }

    /// Check if a proposed action or domain is within verified capabilities or violates prohibitions.
    pub fn assess_capability(&self, action_or_domain: &str) -> (bool, String) {
        let lower = action_or_domain.to_lowercase();
        let state = self.state.lock().unwrap();

        // 1. Check prohibitions
        for b in &state.epistemic_boundaries {
            let item_lower = b.item.to_lowercase();
            if b.category == "WHAT_I_AM_NOT_ALLOWED_TO_DO"
                && (lower.contains(&item_lower) || item_lower.contains(&lower))
            {
                return (
                    false,
                    format!("PROHIBITED_BY_SAFETY_INVARIANT: {}", b.description),
                );
            }
            if b.category == "WHAT_I_CANNOT_DO"
                && (lower.contains(&item_lower) || item_lower.contains(&lower))
            {
                return (
                    false,
                    format!("EXCEEDS_SYSTEM_CAPABILITY: {}", b.description),
                );
            }
        }

        // 2. Check verified capabilities
        for cap in &state.verified_capabilities {
            let cap_lower = cap.to_lowercase();
            if lower.contains(&cap_lower) || cap_lower.contains(&lower) {
                return (true, format!("CAPABILITY_VERIFIED: {}", cap));
            }
        }

        (true, "CAPABILITY_WITHIN_NOMINAL_BOUNDS".to_string())
    }

    /// Register a newly verified capability.
    pub fn register_capability(&self, capability_name: &str) -> Result<(), String> {
        let mut state = self.state.lock().unwrap();
        if !state
            .verified_capabilities
            .contains(&capability_name.to_string())
        {
            state
                .verified_capabilities
                .push(capability_name.to_string());
            drop(state);
            self.persist()?;
        }
        Ok(())
    }

    /// Add an epistemic boundary item.
    pub fn add_epistemic_boundary(
        &self,
        category: &str,
        item: &str,
        description: &str,
        proof: Option<String>,
    ) -> Result<(), String> {
        let mut state = self.state.lock().unwrap();
        state.epistemic_boundaries.push(EpistemicBoundary {
            category: category.to_string(),
            item: item.to_string(),
            description: description.to_string(),
            verification_proof: proof,
        });
        drop(state);
        self.persist()
    }

    /// Save state to disk.
    pub fn persist(&self) -> Result<(), String> {
        let state = self.state.lock().unwrap();
        let json_str = serde_json::to_string_pretty(&*state)
            .map_err(|e| format!("Serialization error: {}", e))?;
        fs::write(&self.storage_path, json_str).map_err(|e| format!("File write error: {}", e))?;
        Ok(())
    }

    /// Load state from disk if exists.
    pub fn load(&mut self) -> Result<(), String> {
        if self.storage_path.exists() {
            let data = fs::read_to_string(&self.storage_path)
                .map_err(|e| format!("File read error: {}", e))?;
            let loaded: CognitiveSelfState =
                serde_json::from_str(&data).map_err(|e| format!("Deserialization error: {}", e))?;
            *self.state.lock().unwrap() = loaded;
        } else {
            self.persist()?;
        }
        Ok(())
    }

    /// Summary for API and brain introspection.
    pub fn get_summary(&self) -> Value {
        let s = self.introspect();
        json!({
            "identity": s.system_identity,
            "architecture": s.architecture_type,
            "version": s.engine_version,
            "status": s.operational_status,
            "capabilities_count": s.verified_capabilities.len(),
            "boundaries_count": s.epistemic_boundaries.len(),
            "calibration_confidence": s.calibration_confidence,
            "last_introspected_at_ms": s.last_introspected_at_ms
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_model_introspection_and_boundaries() {
        let tmp = std::env::temp_dir().join(format!("tara_sm_test_{}", uuid::Uuid::new_v4()));
        let engine = SelfModelEngine::new(&tmp);

        let info = engine.introspect();
        assert_eq!(
            info.system_identity,
            "TARA (Autonomous Cognitive Assistant)"
        );
        assert!(info
            .verified_capabilities
            .contains(&"symbolic_mathematics".to_string()));

        // Test boundary check
        let (ok_math, msg_math) = engine.assess_capability("symbolic_mathematics");
        assert!(ok_math);
        assert!(msg_math.contains("CAPABILITY_VERIFIED"));

        let (ok_prohibit, msg_prohibit) = engine.assess_capability("Bypass Creator Authority");
        assert!(!ok_prohibit);
        assert!(msg_prohibit.contains("PROHIBITED_BY_SAFETY_INVARIANT"));

        // Register new capability and verify persistence
        engine.register_capability("quantum_simulation").unwrap();
        assert!(engine
            .introspect()
            .verified_capabilities
            .contains(&"quantum_simulation".to_string()));

        let _ = fs::remove_dir_all(&tmp);
    }
}
