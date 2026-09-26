//! Evaluator: task verification, recovery, uncertainty detection, self-evaluation.

use serde_json::{json, Value};

/// Result of task verification.
#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub satisfied: bool,
    pub verification_notes: String,
}

/// Verifies whether a task action succeeded.
pub struct TaskVerifier;

impl TaskVerifier {
    pub fn verify(&self, action_intent: &str, target_name: &str, result: &Value) -> VerificationReport {
        let ok = !result.get("error").is_some()
            && result.get("status").and_then(|v| v.as_str()) != Some("ERROR");
        VerificationReport {
            satisfied: ok,
            verification_notes: if ok {
                format!("Action '{}' on '{}' completed successfully", action_intent, target_name)
            } else {
                format!("Action '{}' on '{}' failed: {:?}", action_intent, target_name,
                    result.get("error").or(result.get("status")))
            },
        }
    }
}

/// Attempts to recover from tool/skill failures.
pub struct RecoveryEngine {
    pub repo_root: String,
}

impl RecoveryEngine {
    pub fn new(repo_root: &str) -> Self {
        Self { repo_root: repo_root.to_string() }
    }

    pub fn attempt_recovery(&self, tool_name: &str, _params: &Value, error: &str) -> (bool, Value, String) {
        // Recovery strategies: retry, fallback, degrade
        let message = format!(
            "Recovery attempted for tool '{}': error was '{}'. Applying graceful degradation.",
            tool_name, error
        );
        (false, json!({ "recovery_attempted": true, "tool": tool_name, "error": error }), message)
    }
}

/// Detects uncertainty in TARA's knowledge for a query.
pub struct UncertaintyDetector;

impl UncertaintyDetector {
    pub fn evaluate_uncertainty(
        &self,
        text: &str,
        knowledge_results: &[Value],
        memory_results: &[Value],
    ) -> Option<String> {
        if knowledge_results.is_empty() && memory_results.is_empty() {
            let factual_indicators = ["what is", "who is", "when did", "where is", "how does", "define"];
            let text_lc = text.to_lowercase();
            if factual_indicators.iter().any(|&ind| text_lc.contains(ind)) {
                return Some(format!(
                    "I don't have specific information about that in my knowledge base. \
                    My response is based on my training and may not be fully accurate."
                ));
            }
        }
        None
    }
}

/// Self-evaluates and optionally refines the final response.
pub struct SelfEvaluator;

impl SelfEvaluator {
    pub fn evaluate_and_refine(
        &self,
        user_input: &str,
        tool_or_skill: &str,
        result: &Value,
        candidate_response: &str,
        verification: &VerificationReport,
    ) -> String {
        if !verification.satisfied {
            return format!(
                "{}\n\n[Note: The requested action encountered issues: {}]",
                candidate_response, verification.verification_notes
            );
        }
        if candidate_response.trim().is_empty() {
            return "I processed your request successfully.".to_string();
        }
        candidate_response.to_string()
    }
}
