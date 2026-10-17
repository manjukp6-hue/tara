//! ProgrammingEngine: Production Native Programming Specialist Engine for TARA.
//!
//! Provides static code analysis, delimiter balancing, security issue audits,
//! Big-O algorithmic complexity classification, and real project compilation & test verification.

pub mod analyzer;
pub mod complexity;
pub mod language;
pub mod verifier;

pub use analyzer::{SecurityIssue, StaticAnalyzer, SyntaxHealth};
pub use complexity::{ComplexityAnalyzer, ComplexityProfile};
pub use language::{LanguageCapability, LanguageRegistry};
pub use verifier::{ProjectVerifier, VerificationReport};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgrammingEvaluationResult {
    pub success: bool,
    pub domain: String,
    pub operation: String,
    pub language: String,
    pub analysis: Value,
    pub explanation: String,
    pub execution_time_us: u64,
}

pub struct ProgrammingEngine {
    pub repo_root: String,
}

impl ProgrammingEngine {
    pub fn new(repo_root: &str) -> Self {
        Self {
            repo_root: repo_root.to_string(),
        }
    }

    /// Evaluates structured programming engine requests.
    pub fn evaluate(
        &self,
        operation: &str,
        params: &Value,
    ) -> Result<ProgrammingEvaluationResult, String> {
        let start = std::time::Instant::now();
        let lang = params
            .get("language")
            .and_then(Value::as_str)
            .unwrap_or("rust");

        const MAX_CODE_BYTES: usize = 512 * 1024;

        let (op_success, analysis, explanation) = match operation {
            "analyze_code" => {
                let code = params
                    .get("code")
                    .and_then(Value::as_str)
                    .ok_or("missing 'code'")?;
                if code.len() > MAX_CODE_BYTES {
                    return Err("Input code exceeds maximum allowed size of 512 KiB".to_string());
                }
                let health = StaticAnalyzer::analyze(code, lang);
                let clean = health.is_balanced && health.suspicious_patterns.is_empty();
                let json_val = json!({
                    "is_balanced": health.is_balanced,
                    "unclosed_delimiters": health.unclosed_delimiters,
                    "total_lines": health.total_lines,
                    "code_lines": health.code_lines,
                    "comment_lines": health.comment_lines,
                    "blank_lines": health.blank_lines,
                    "function_count": health.function_count,
                    "cyclomatic_complexity_est": health.cyclomatic_complexity_est,
                    "suspicious_patterns": health.suspicious_patterns,
                });
                let summary = if clean {
                    format!("Code analysis passed cleanly: {} lines of code, {} functions, cyclomatic complexity ~{}", health.code_lines, health.function_count, health.cyclomatic_complexity_est)
                } else {
                    format!("Code analysis flagged {} issues: {} delimiter errors, {} security patterns", health.unclosed_delimiters.len() + health.suspicious_patterns.len(), health.unclosed_delimiters.len(), health.suspicious_patterns.len())
                };
                (clean, json_val, summary)
            }

            "check_delimiters" => {
                let code = params
                    .get("code")
                    .and_then(Value::as_str)
                    .ok_or("missing 'code'")?;
                if code.len() > MAX_CODE_BYTES {
                    return Err("Input code exceeds maximum allowed size of 512 KiB".to_string());
                }
                let (balanced, errors) = StaticAnalyzer::check_delimiter_balance(code);
                let msg = if balanced {
                    "All parentheses, braces, and brackets are properly balanced".into()
                } else {
                    format!("Found {} delimiter balancing errors", errors.len())
                };
                (
                    balanced,
                    json!({ "balanced": balanced, "errors": errors }),
                    msg,
                )
            }

            "complexity_lookup" => {
                let algo = params
                    .get("algorithm")
                    .and_then(Value::as_str)
                    .ok_or("missing 'algorithm'")?;
                if let Some(prof) = ComplexityAnalyzer::lookup(algo) {
                    (
                        true,
                        json!({
                            "algorithm": prof.algorithm_or_operation,
                            "category": prof.category,
                            "time_best": prof.time_best,
                            "time_average": prof.time_average,
                            "time_worst": prof.time_worst,
                            "space_worst": prof.space_worst,
                            "explanation": prof.explanation,
                        }),
                        format!(
                            "{}: Average Time {}, Worst Time {}, Space {}",
                            prof.algorithm_or_operation,
                            prof.time_average,
                            prof.time_worst,
                            prof.space_worst
                        ),
                    )
                } else {
                    return Err(format!(
                        "algorithm '{}' not found in complexity knowledge base",
                        algo
                    ));
                }
            }

            "cargo_check" => {
                let target_dir = params
                    .get("workspace_dir")
                    .and_then(Value::as_str)
                    .unwrap_or(&self.repo_root);
                let target_path = Path::new(target_dir);
                tara_security::path_security::canonical_containment(Path::new(&self.repo_root), target_path)
                    .map_err(|e| format!("PathSecurityViolation: workspace_dir must be strictly contained within repository sandbox: {e}"))?;
                let rep = ProjectVerifier::cargo_check(target_dir)?;
                let msg = if rep.success {
                    "cargo check compiled successfully with no compilation errors".into()
                } else {
                    "cargo check reported compilation errors".into()
                };
                (
                    rep.success,
                    json!({
                        "success": rep.success,
                        "command": rep.command,
                        "working_dir": rep.working_dir,
                        "exit_code": rep.exit_code,
                        "stdout": rep.stdout,
                        "stderr": rep.stderr,
                        "error_summary": rep.error_summary,
                    }),
                    msg,
                )
            }

            "cargo_test" => {
                let target_dir = params
                    .get("workspace_dir")
                    .and_then(Value::as_str)
                    .unwrap_or(&self.repo_root);
                let target_path = Path::new(target_dir);
                tara_security::path_security::canonical_containment(Path::new(&self.repo_root), target_path)
                    .map_err(|e| format!("PathSecurityViolation: workspace_dir must be strictly contained within repository sandbox: {e}"))?;
                let filter = params.get("filter").and_then(Value::as_str);
                let rep = ProjectVerifier::cargo_test(target_dir, filter)?;
                let msg = if rep.success {
                    "cargo test executed and passed successfully".into()
                } else {
                    "cargo test reported test failures".into()
                };
                (
                    rep.success,
                    json!({
                        "success": rep.success,
                        "command": rep.command,
                        "working_dir": rep.working_dir,
                        "exit_code": rep.exit_code,
                        "stdout": rep.stdout,
                        "stderr": rep.stderr,
                        "error_summary": rep.error_summary,
                    }),
                    msg,
                )
            }

            "language_capability" => {
                let lang_cap = LanguageRegistry::get(lang)
                    .ok_or_else(|| format!("language '{}' not recognized", lang))?;
                (
                    true,
                    json!({
                        "name": lang_cap.name,
                        "extensions": lang_cap.extensions,
                        "paradigm": lang_cap.paradigm,
                        "typing": lang_cap.typing,
                        "native_compiler_available": lang_cap.native_compiler_available,
                        "static_analysis_supported": lang_cap.static_analysis_supported,
                    }),
                    format!("Language profile for {}", lang_cap.name),
                )
            }

            _ => {
                return Err(format!(
                    "unsupported ProgrammingEngine operation '{}'",
                    operation
                ))
            }
        };

        let elapsed = start.elapsed().as_micros() as u64;
        Ok(ProgrammingEvaluationResult {
            success: op_success,
            domain: "programming".into(),
            operation: operation.into(),
            language: lang.into(),
            analysis,
            explanation,
            execution_time_us: elapsed,
        })
    }

    /// Natural language query interpreter for standard programming questions.
    pub fn solve_query(&self, query: &str) -> Option<ProgrammingEvaluationResult> {
        let q = query.trim().to_lowercase();

        // Pattern: complexity of <algorithm> or Big-O of <algorithm>
        if q.contains("complexity") || q.contains("big-o") || q.contains("big o") {
            for algo in &[
                "binary_search",
                "linear_search",
                "quicksort",
                "mergesort",
                "heapsort",
                "hash_table_lookup",
                "bfs",
                "dfs",
                "dijkstra",
                "matrix_multiplication_naive",
            ] {
                let name = algo.replace('_', " ");
                if q.contains(algo)
                    || q.contains(&name)
                    || (algo == &"matrix_multiplication_naive"
                        && q.contains("matrix multiplication"))
                {
                    return self
                        .evaluate("complexity_lookup", &json!({ "algorithm": algo }))
                        .ok();
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_programming_engine_op_success_and_payload_limit() {
        let engine = ProgrammingEngine::new(".");

        // Clean code -> success is true
        let clean = "fn main() { let x = 42; println!(\"{}\", x); }";
        let res_clean = engine
            .evaluate(
                "analyze_code",
                &json!({ "code": clean, "language": "rust" }),
            )
            .unwrap();
        assert!(res_clean.success);

        // Unbalanced code -> success is false
        let unbal = "fn main() { let x = (1 + 2; }";
        let res_unbal = engine
            .evaluate(
                "analyze_code",
                &json!({ "code": unbal, "language": "rust" }),
            )
            .unwrap();
        assert!(!res_unbal.success);

        // Oversized payload (> 512 KiB) rejected
        let oversized = "a".repeat(512 * 1024 + 1);
        let res_err = engine.evaluate(
            "analyze_code",
            &json!({ "code": oversized, "language": "rust" }),
        );
        assert!(res_err.is_err());
    }

    #[test]
    fn test_solve_query_all_ten_algorithms() {
        let engine = ProgrammingEngine::new(".");
        let algos = [
            ("binary search", "binary_search"),
            ("linear search", "linear_search"),
            ("quicksort", "quicksort"),
            ("mergesort", "mergesort"),
            ("heapsort", "heapsort"),
            ("hash table lookup", "hash_table_lookup"),
            ("bfs", "bfs"),
            ("dfs", "dfs"),
            ("dijkstra", "dijkstra"),
            ("matrix multiplication", "matrix_multiplication_naive"),
        ];

        for (phrase, _key) in algos {
            let query = format!("what is the time complexity of {}?", phrase);
            let res = engine
                .solve_query(&query)
                .unwrap_or_else(|| panic!("failed for {}", phrase));
            assert!(res.success);
            assert!(res.analysis["time_worst"].as_str().is_some());
        }
    }
}
