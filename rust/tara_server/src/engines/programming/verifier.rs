//! Real compiler and test verification engine for ProgrammingEngine.
//!
//! Executes real `cargo check` and `cargo test` in authorized workspace directories.
//! Never fabricates PASS: captures genuine exit codes, stdout, and stderr.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    pub success: bool,
    pub command: String,
    pub working_dir: String,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub error_summary: Option<String>,
}

pub struct ProjectVerifier;

pub const MAX_OUTPUT_CHARS: usize = 65_536;
pub const MAX_LOG_CHARS: usize = MAX_OUTPUT_CHARS;

fn truncate_output(s: String) -> String {
    if s.len() <= MAX_OUTPUT_CHARS {
        s
    } else {
        s.chars().take(MAX_OUTPUT_CHARS).collect::<String>() + "\n...[truncated]..."
    }
}

impl ProjectVerifier {
    /// Run real `cargo check` in target workspace directory.
    pub fn cargo_check(workspace_dir: &str) -> Result<VerificationReport, String> {
        let path = Path::new(workspace_dir);
        if !path.exists() || !path.join("Cargo.toml").exists() {
            return Err(format!(
                "path '{}' is not a valid Cargo project root",
                workspace_dir
            ));
        }

        let output = Command::new("cargo")
            .arg("check")
            .arg("--color")
            .arg("never")
            .current_dir(workspace_dir)
            .output()
            .map_err(|e| format!("failed to spawn cargo check: {}", e))?;

        let stdout = truncate_output(String::from_utf8_lossy(&output.stdout).into_owned());
        let stderr = truncate_output(String::from_utf8_lossy(&output.stderr).into_owned());
        let success = output.status.success();

        let error_summary = if !success {
            let mut errors = Vec::new();
            for line in stderr.lines() {
                if line.contains("error[") || line.contains("error:") {
                    errors.push(line.trim().to_string());
                }
            }
            if errors.is_empty() {
                Some("Compilation failed".to_string())
            } else {
                Some(errors.join("; "))
            }
        } else {
            None
        };

        Ok(VerificationReport {
            success,
            command: "cargo check".to_string(),
            working_dir: workspace_dir.to_string(),
            exit_code: output.status.code(),
            stdout,
            stderr,
            error_summary,
        })
    }

    /// Run real `cargo test` in target workspace directory.
    pub fn cargo_test(
        workspace_dir: &str,
        test_filter: Option<&str>,
    ) -> Result<VerificationReport, String> {
        let path = Path::new(workspace_dir);
        if !path.exists() || !path.join("Cargo.toml").exists() {
            return Err(format!(
                "path '{}' is not a valid Cargo project root",
                workspace_dir
            ));
        }

        let mut cmd = Command::new("cargo");
        cmd.arg("test");
        cmd.arg("--color");
        cmd.arg("never");
        if let Some(filter) = test_filter {
            if !filter.is_empty()
                && (!filter
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':' || c == '-')
                    || filter.starts_with('-'))
            {
                return Err(format!(
                    "Invalid test filter '{}': must be a valid Rust test identifier without command flags",
                    filter
                ));
            }
            cmd.arg(filter);
        }
        cmd.current_dir(workspace_dir);

        let output = cmd
            .output()
            .map_err(|e| format!("failed to spawn cargo test: {}", e))?;

        let stdout = truncate_output(String::from_utf8_lossy(&output.stdout).into_owned());
        let stderr = truncate_output(String::from_utf8_lossy(&output.stderr).into_owned());
        let success = output.status.success();

        let error_summary = if !success {
            Some("One or more tests failed during cargo test execution".to_string())
        } else {
            None
        };

        Ok(VerificationReport {
            success,
            command: format!("cargo test {}", test_filter.unwrap_or("")),
            working_dir: workspace_dir.to_string(),
            exit_code: output.status.code(),
            stdout,
            stderr,
            error_summary,
        })
    }
}
