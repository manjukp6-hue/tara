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
    pub fn verify(
        &self,
        action_intent: &str,
        target_name: &str,
        result: &Value,
    ) -> VerificationReport {
        let status = result.get("status").and_then(Value::as_str);
        let ok = !result.get("error").is_some()
            && status
                .map(|s| matches!(s, "SUCCESS" | "COMPLETED" | "PROCESSED" | "LOOKUP_COMPLETE"))
                .unwrap_or(true);
        VerificationReport {
            satisfied: ok,
            verification_notes: if ok {
                format!(
                    "Action '{}' on '{}' completed successfully",
                    action_intent, target_name
                )
            } else {
                format!(
                    "Action '{}' on '{}' failed: {:?}",
                    action_intent,
                    target_name,
                    result.get("error").or(result.get("status"))
                )
            },
        }
    }
}

/// Attempts to recover from tool/skill failures via parameter healing and path resolution.
pub struct RecoveryEngine {
    pub repo_root: String,
}

impl RecoveryEngine {
    pub fn new(repo_root: &str) -> Self {
        Self {
            repo_root: repo_root.to_string(),
        }
    }

    pub fn attempt_recovery(
        &self,
        tool_name: &str,
        params: &Value,
        error: &str,
    ) -> (bool, Value, String) {
        use std::path::Path;

        let mut healed_params = params.clone();
        let mut recovered = false;
        let mut strategies = Vec::new();

        if let Value::Object(ref mut map) = healed_params {
            // Strategy 1: Path resolution & key aliasing for file-based tools
            if (tool_name == "file_inspector" || tool_name == "hash_verifier")
                && !map.contains_key("path")
            {
                if let Some(f) = map.remove("file").or_else(|| map.remove("filepath")) {
                    map.insert("path".to_string(), f);
                    recovered = true;
                    strategies.push("normalized_path_key");
                }
            }

            // Strategy 2: File path cleanup and repository root resolution
            if let Some(Value::String(p)) = map.get("path") {
                let trimmed = p
                    .trim()
                    .trim_matches('`')
                    .trim_matches('"')
                    .trim_matches('\'');
                let candidate_path = Path::new(trimmed);
                if !candidate_path.exists() {
                    // Try joining with repo_root
                    let anchored = Path::new(&self.repo_root).join(trimmed);
                    if anchored.exists() {
                        map.insert(
                            "path".to_string(),
                            Value::String(anchored.to_string_lossy().to_string()),
                        );
                        recovered = true;
                        strategies.push("resolved_under_repo_root");
                    } else {
                        // Check if separator normalization works
                        let normalized_str = trimmed.replace('\\', "/");
                        let norm_anchored = Path::new(&self.repo_root).join(&normalized_str);
                        if norm_anchored.exists() {
                            map.insert(
                                "path".to_string(),
                                Value::String(norm_anchored.to_string_lossy().to_string()),
                            );
                            recovered = true;
                            strategies.push("normalized_separators_under_repo_root");
                        }
                    }
                } else if trimmed != p {
                    map.insert("path".to_string(), Value::String(trimmed.to_string()));
                    recovered = true;
                    strategies.push("trimmed_path_quotes");
                }
            }

            // Strategy 3: Markdown fence / string cleanup for query or code params
            for key in &["query", "code", "input", "text"] {
                if let Some(Value::String(s)) = map.get_mut(*key) {
                    if s.contains("```") {
                        let cleaned = s
                            .lines()
                            .filter(|l| !l.trim().starts_with("```"))
                            .collect::<Vec<_>>()
                            .join("\n");
                        *s = cleaned;
                        recovered = true;
                        strategies.push("stripped_markdown_code_fences");
                    }
                }
            }
        }

        if recovered {
            let desc = format!(
                "Recovered tool parameters via strategies: {}",
                strategies.join(", ")
            );
            (true, healed_params, desc)
        } else {
            let message = format!(
                "Recovery could not heal tool '{}' failure; original error was '{}'.",
                tool_name, error
            );
            (
                false,
                json!({ "recovery_attempted": true, "tool": tool_name, "error": error }),
                message,
            )
        }
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
            let factual_indicators = [
                "what is", "who is", "when did", "where is", "how does", "define",
            ];
            let text_lc = text.to_lowercase();
            if factual_indicators.iter().any(|&ind| text_lc.contains(ind)) {
                return Some(
                    "I don't have specific information about that in my knowledge base. \
                    My response is based on my training and may not be fully accurate."
                        .to_string(),
                );
            }
        }
        None
    }
}

/// Self-evaluates and optionally refines the final response.
///
/// Refinement logic:
///   1. Verification failure → append structured error context from the tool result.
///   2. Tool/skill produced a non-empty structured result the response doesn't mention →
///      append a formatted summary so the user sees the actual output.
///   3. Empty candidate response → emit the structured result directly.
///   4. User asked a question but the response doesn't include the key value fields from
///      the result → append the relevant fields.
pub struct SelfEvaluator;

impl SelfEvaluator {
    pub fn evaluate_and_refine(
        &self,
        _user_input: &str,
        tool_or_skill: &str,
        result: &Value,
        candidate_response: &str,
        verification: &VerificationReport,
    ) -> String {
        let trimmed = candidate_response.trim();

        // Step 1: Verification failure — append structured error detail.
        if !verification.satisfied {
            let error_detail = result
                .get("error")
                .and_then(Value::as_str)
                .or_else(|| result.get("message").and_then(Value::as_str))
                .unwrap_or(&verification.verification_notes);
            if trimmed.is_empty() {
                return format!(
                    "[Action '{}' encountered an issue: {}]",
                    tool_or_skill, error_detail
                );
            }
            return format!(
                "{}\n\n[Note: '{}' reported an issue — {}]",
                trimmed, tool_or_skill, error_detail
            );
        }

        // Step 2: Empty candidate response — surface the structured result directly.
        if trimmed.is_empty() {
            return Self::format_result_as_response(tool_or_skill, result);
        }

        // Step 3: The response exists but the tool produced structured data the
        // response text doesn't mention. Detect by checking if important result
        // fields appear verbatim in the response.
        if !tool_or_skill.is_empty() && tool_or_skill != "none" {
            let supplement = Self::extract_unmentioned_fields(trimmed, result);
            if let Some(extra) = supplement {
                return format!("{}\n\n{}", trimmed, extra);
            }
        }

        // Step 4: No refinement needed — return the genuine neural output.
        trimmed.to_string()
    }

    /// Format a structured JSON result into a human-readable response.
    fn format_result_as_response(tool_or_skill: &str, result: &Value) -> String {
        if let Some(obj) = result.as_object() {
            let mut parts = Vec::new();
            // Skip internal control fields.
            let skip = ["status", "action", "timestamp", "actor_id"];
            for (key, val) in obj {
                if skip.contains(&key.as_str()) {
                    continue;
                }
                let display = match val {
                    Value::String(s) => s.clone(),
                    Value::Number(n) => n.to_string(),
                    Value::Bool(b) => b.to_string(),
                    _ => serde_json::to_string(val).unwrap_or_default(),
                };
                if !display.is_empty() {
                    parts.push(format!("{}: {}", key, display));
                }
            }
            if !parts.is_empty() {
                return format!(
                    "[INFERENCE_NOTICE: No output tokens generated by neural model]\n\n{} result:\n{}",
                    tool_or_skill,
                    parts.join("\n")
                );
            }
        }
        "[INFERENCE_NOTICE: No output tokens generated by neural model]".to_string()
    }

    /// Check whether key result fields are already mentioned in the response.
    /// Returns a formatted supplement string for any fields that are absent.
    fn extract_unmentioned_fields(response: &str, result: &Value) -> Option<String> {
        let obj = result.as_object()?;
        let response_lower = response.to_lowercase();
        let skip = [
            "status",
            "action",
            "timestamp",
            "actor_id",
            "knowledge_updated",
        ];
        let mut missing = Vec::new();

        for (key, val) in obj {
            if skip.contains(&key.as_str()) {
                continue;
            }
            let display = match val {
                Value::String(s) if !s.is_empty() => s.clone(),
                Value::Number(n) => n.to_string(),
                Value::Bool(b) => b.to_string(),
                _ => continue,
            };
            // If neither the key nor the value appears in the response, mark as missing.
            if !response_lower.contains(&key.to_lowercase())
                && !response_lower.contains(&display.to_lowercase())
            {
                missing.push(format!("{}: {}", key, display));
            }
        }

        if missing.is_empty() {
            None
        } else {
            Some(format!("**Result details:**\n{}", missing.join("\n")))
        }
    }
}
