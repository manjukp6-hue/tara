//! TARA Runtime Management & Universal Promotion Gate.
//!
//! Language-neutral dynamic runtime registration, canonical contract validation,
//! and all-runtime promotion gating.

pub mod gate;
pub mod registry;

pub use gate::{
    GateEvaluationResponse, GateVerdict, RuntimeEvaluationResult, UniversalRuntimeGate,
};
pub use registry::{DynamicRuntimeRegistry, RuntimeRecord, RuntimeState};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tara_security::primitives::AuthenticatedPrincipal;

    #[test]
    fn test_case_insensitive_get_runtime() {
        let registry = DynamicRuntimeRegistry::new();
        assert!(registry.get_runtime("RUST").is_some());
        assert!(registry.get_runtime("rust").is_some());
        assert!(registry.get_runtime("  Rust  ").is_some());
        assert_eq!(
            registry.get_runtime("RUST").unwrap().runtime_id,
            "rust"
        );
    }

    #[test]
    fn test_verify_runtime_ready_unverified_states() {
        let registry = DynamicRuntimeRegistry::new();
        let mut rust_rec = registry.get_runtime("rust").unwrap();
        rust_rec.model_sha = "valid_model_sha_test_sentinel_passed".to_string();
        registry.register_runtime(rust_rec).unwrap();

        // Initially VERIFIED -> returns true
        let (ready, _, _) = registry.verify_runtime_ready("rust");
        assert!(ready);

        // REVOKED state must return false
        registry
            .update_runtime_state("rust", RuntimeState::REVOKED)
            .unwrap();
        let (ready_revoked, _, _) = registry.verify_runtime_ready("rust");
        assert!(!ready_revoked);

        // DEGRADED state must return false
        registry
            .update_runtime_state("rust", RuntimeState::DEGRADED)
            .unwrap();
        let (ready_degraded, _, _) = registry.verify_runtime_ready("rust");
        assert!(!ready_degraded);
    }

    #[test]
    fn test_retire_runtime_last_required_guard() {
        let registry = DynamicRuntimeRegistry::new();
        let creator = AuthenticatedPrincipal::mint_creator_for_test("session_test").unwrap();

        // "rust" is the sole required runtime
        let res_single = registry.retire_runtime("rust", &creator, "attempting to retire sole runtime");
        assert!(res_single.is_err());
        assert!(res_single
            .unwrap_err()
            .contains("last remaining non-retired required runtime"));

        // Register a second required runtime
        let second = RuntimeRecord::new("runtime_secondary", "Native", true);
        registry.register_runtime(second).unwrap();

        // Now retiring "rust" succeeds because another required runtime exists
        let res_ok = registry.retire_runtime("rust", &creator, "retiring replaced runtime");
        assert!(res_ok.is_ok());

        // Now "runtime_secondary" is the sole remaining required runtime, so retiring it is rejected
        let res_second_fail =
            registry.retire_runtime("runtime_secondary", &creator, "retiring last remaining runtime");
        assert!(res_second_fail.is_err());
        assert!(res_second_fail
            .unwrap_err()
            .contains("last remaining non-retired required runtime"));
    }

    #[test]
    fn test_gate_fail_closed_empty_required_set() {
        let registry = Arc::new(DynamicRuntimeRegistry::new());

        // Update default runtime to not be required for promotion, yielding an empty required set
        let non_required = RuntimeRecord::new("rust", "Rust", false);
        registry.register_runtime(non_required).unwrap();
        assert!(registry.list_required_runtimes().is_empty());

        let gate = UniversalRuntimeGate::new(registry);
        let evals = HashMap::new();
        let resp = gate.evaluate_evolution_candidate("model", "model_v2", "2.0.0", &evals);

        assert!(!resp.eligible_for_promotion);
        assert_eq!(resp.gate_verdict, GateVerdict::Quarantined);
        assert!(resp.quarantined);
    }

    #[test]
    fn test_gate_case_insensitive_evaluations_and_verdicts() {
        let registry = Arc::new(DynamicRuntimeRegistry::new());
        let gate = UniversalRuntimeGate::new(Arc::clone(&registry));

        // Missing evaluation -> RejectedPartialSupport
        let empty_evals = HashMap::new();
        let resp_missing = gate.evaluate_evolution_candidate("model", "target_1", "2.0.0", &empty_evals);
        assert!(!resp_missing.eligible_for_promotion);
        assert_eq!(resp_missing.gate_verdict, GateVerdict::RejectedPartialSupport);
        assert!(resp_missing.quarantined);

        // Case-insensitive evaluation passed ("RUST") -> Approved
        let mut passing_evals = HashMap::new();
        passing_evals.insert(
            "RUST".to_string(),
            RuntimeEvaluationResult {
                runtime_id: "RUST".to_string(),
                passed: true,
                status: "PASSED".to_string(),
                error_message: None,
            },
        );
        let resp_ok = gate.evaluate_evolution_candidate("model", "target_1", "2.0.0", &passing_evals);
        assert!(resp_ok.eligible_for_promotion);
        assert_eq!(resp_ok.gate_verdict, GateVerdict::Approved);
        assert!(!resp_ok.quarantined);

        // Explicit failure -> RejectedRuntimeFailure
        let mut failing_evals = HashMap::new();
        failing_evals.insert(
            "rust".to_string(),
            RuntimeEvaluationResult {
                runtime_id: "rust".to_string(),
                passed: false,
                status: "FAILED".to_string(),
                error_message: Some("Execution failure".to_string()),
            },
        );
        let resp_fail = gate.evaluate_evolution_candidate("model", "target_1", "2.0.0", &failing_evals);
        assert!(!resp_fail.eligible_for_promotion);
        assert_eq!(resp_fail.gate_verdict, GateVerdict::RejectedRuntimeFailure);
        assert!(resp_fail.quarantined);

        // Evaluation passed but registry state is DEGRADED -> RejectedRuntimeFailure
        registry
            .update_runtime_state("rust", RuntimeState::DEGRADED)
            .unwrap();
        let resp_unverified =
            gate.evaluate_evolution_candidate("model", "target_1", "2.0.0", &passing_evals);
        assert!(!resp_unverified.eligible_for_promotion);
        assert_eq!(resp_unverified.gate_verdict, GateVerdict::RejectedRuntimeFailure);
        assert!(resp_unverified.quarantined);
    }
}
