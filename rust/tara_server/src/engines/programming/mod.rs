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

        let (analysis, explanation) = match operation {
            "analyze_code" => {
                let code = params
                    .get("code")
                    .and_then(Value::as_str)
                    .ok_or("missing 'code'")?;
                let health = StaticAnalyzer::analyze(code, lang);
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
                let summary = if health.is_balanced && health.suspicious_patterns.is_empty() {
                    format!("Code analysis passed cleanly: {} lines of code, {} functions, cyclomatic complexity ~{}", health.code_lines, health.function_count, health.cyclomatic_complexity_est)
                } else {
                    format!("Code analysis flagged {} issues: {} delimiter errors, {} security patterns", health.unclosed_delimiters.len() + health.suspicious_patterns.len(), health.unclosed_delimiters.len(), health.suspicious_patterns.len())
                };
                (json_val, summary)
            }

            "check_delimiters" => {
                let code = params
                    .get("code")
                    .and_then(Value::as_str)
                    .ok_or("missing 'code'")?;
                let (balanced, errors) = StaticAnalyzer::check_delimiter_balance(code);
                (
                    json!({ "balanced": balanced, "errors": errors }),
                    if balanced {
                        "All parentheses, braces, and brackets are properly balanced".into()
                    } else {
                        format!("Found {} delimiter balancing errors", errors.len())
                    },
                )
            }

            "complexity_lookup" => {
                let algo = params
                    .get("algorithm")
                    .and_then(Value::as_str)
                    .ok_or("missing 'algorithm'")?;
                if let Some(prof) = ComplexityAnalyzer::lookup(algo) {
                    (
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
                let rep = ProjectVerifier::cargo_check(target_dir)?;
                (
                    json!({
                        "success": rep.success,
                        "command": rep.command,
                        "working_dir": rep.working_dir,
                        "exit_code": rep.exit_code,
                        "stdout": rep.stdout,
                        "stderr": rep.stderr,
                        "error_summary": rep.error_summary,
                    }),
                    if rep.success {
                        "cargo check compiled successfully with no compilation errors".into()
                    } else {
                        "cargo check reported compilation errors".into()
                    },
                )
            }

            "cargo_test" => {
                let target_dir = params
                    .get("workspace_dir")
                    .and_then(Value::as_str)
                    .unwrap_or(&self.repo_root);
                let filter = params.get("filter").and_then(Value::as_str);
                let rep = ProjectVerifier::cargo_test(target_dir, filter)?;
                (
                    json!({
                        "success": rep.success,
                        "command": rep.command,
                        "working_dir": rep.working_dir,
                        "exit_code": rep.exit_code,
                        "stdout": rep.stdout,
                        "stderr": rep.stderr,
                        "error_summary": rep.error_summary,
                    }),
                    if rep.success {
                        "cargo test executed and passed successfully".into()
                    } else {
                        "cargo test reported test failures".into()
                    },
                )
            }

            "language_capability" => {
                let lang_cap = LanguageRegistry::get(lang)
                    .ok_or_else(|| format!("language '{}' not recognized", lang))?;
                (
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
            success: true,
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
                "quicksort",
                "mergesort",
                "heapsort",
                "bfs",
                "dfs",
                "dijkstra",
            ] {
                let name = algo.replace('_', " ");
                if q.contains(algo) || q.contains(&name) {
                    return self
                        .evaluate("complexity_lookup", &json!({ "algorithm": algo }))
                        .ok();
                }
            }
        }

        None
    }
}
