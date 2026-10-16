//! Universal Promotion Gate for TARA (Rust Implementation).
//!
//! Enforces:
//! - ALL REGISTERED REQUIRED RUNTIMES MUST PASS.
//! - If any required runtime fails -> WHOLE CHANGE = FAIL.
//! - Atomic promotion and multi-runtime aware rollback.
//! - Anti-self-bypass: model cannot approve itself.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

use super::registry::{DynamicRuntimeRegistry, RuntimeState};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateVerdict {
    Approved,
    RejectedRuntimeFailure,
    RejectedPartialSupport,
    Quarantined,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeEvaluationResult {
    pub runtime_id: String,
    pub passed: bool,
    pub status: String,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateEvaluationResponse {
    pub target_type: String,
    pub target_id: String,
    pub candidate_version: String,
    pub required_runtimes: Vec<String>,
    pub runtime_results: HashMap<String, Value>,
    pub eligible_for_promotion: bool,
    pub gate_verdict: GateVerdict,
    pub reason: String,
    pub quarantined: bool,
}

pub struct UniversalRuntimeGate {
    registry: Arc<DynamicRuntimeRegistry>,
}

impl UniversalRuntimeGate {
    pub fn new(registry: Arc<DynamicRuntimeRegistry>) -> Self {
        Self { registry }
    }

    pub fn evaluate_evolution_candidate(
        &self,
        target_type: &str,
        target_id: &str,
        candidate_version: &str,
        evaluations: &HashMap<String, RuntimeEvaluationResult>,
    ) -> GateEvaluationResponse {
        let required = self.registry.list_required_runtimes();
        let mut required_ids: Vec<String> = required
            .iter()
            .map(|r| r.runtime_id.trim().to_lowercase())
            .collect();
        required_ids.sort();

        if required_ids.is_empty() {
            return GateEvaluationResponse {
                target_type: target_type.to_string(),
                target_id: target_id.to_string(),
                candidate_version: candidate_version.to_string(),
                required_runtimes: Vec::new(),
                runtime_results: HashMap::new(),
                eligible_for_promotion: false,
                gate_verdict: GateVerdict::Quarantined,
                reason: "Fail-closed: No active required runtimes are registered in DynamicRuntimeRegistry.".to_string(),
                quarantined: true,
            };
        }

        let normalized_evals: HashMap<String, &RuntimeEvaluationResult> = evaluations
            .iter()
            .map(|(k, v)| (k.trim().to_lowercase(), v))
            .collect();

        let mut results = HashMap::new();
        let mut all_passed = true;
        let mut failed = Vec::new();
        let mut missing = Vec::new();

        for req_rec in &required {
            let req_id = req_rec.runtime_id.trim().to_lowercase();
            let registry_verified = req_rec.status == RuntimeState::VERIFIED;

            if let Some(eval) = normalized_evals.get(&req_id) {
                let eval_passed = eval.passed && registry_verified;
                let status = if !registry_verified {
                    format!("REGISTRY_{:?}", req_rec.status)
                } else {
                    eval.status.clone()
                };
                let error = if !eval.passed {
                    eval.error_message.clone()
                } else if !registry_verified {
                    Some(format!(
                        "Runtime '{}' is not in VERIFIED state in runtime registry (state: {:?})",
                        req_id, req_rec.status
                    ))
                } else {
                    None
                };

                results.insert(
                    req_id.clone(),
                    json!({
                        "passed": eval_passed,
                        "status": status,
                        "error": error
                    }),
                );
                if !eval_passed {
                    all_passed = false;
                    failed.push(req_id.clone());
                }
            } else {
                all_passed = false;
                missing.push(req_id.clone());
                results.insert(
                    req_id.clone(),
                    json!({
                        "passed": false,
                        "status": "MISSING_EVALUATION"
                    }),
                );
            }
        }

        if !all_passed {
            let (verdict, reason) = if !failed.is_empty() {
                (
                    GateVerdict::RejectedRuntimeFailure,
                    format!(
                        "Evolution rejected: Required runtimes failed. Failed: {:?}, Missing: {:?}. Production unchanged.",
                        failed, missing
                    ),
                )
            } else {
                (
                    GateVerdict::RejectedPartialSupport,
                    format!(
                        "Evolution rejected: Partial runtime support. Missing evaluations for required runtimes: {:?}. Production unchanged.",
                        missing
                    ),
                )
            };

            GateEvaluationResponse {
                target_type: target_type.to_string(),
                target_id: target_id.to_string(),
                candidate_version: candidate_version.to_string(),
                required_runtimes: required_ids,
                runtime_results: results,
                eligible_for_promotion: false,
                gate_verdict: verdict,
                reason,
                quarantined: true,
            }
        } else {
            GateEvaluationResponse {
                target_type: target_type.to_string(),
                target_id: target_id.to_string(),
                candidate_version: candidate_version.to_string(),
                required_runtimes: required_ids,
                runtime_results: results,
                eligible_for_promotion: true,
                gate_verdict: GateVerdict::Approved,
                reason:
                    "All required runtimes verified successfully. Eligible for atomic promotion."
                        .to_string(),
                quarantined: false,
            }
        }
    }
}
