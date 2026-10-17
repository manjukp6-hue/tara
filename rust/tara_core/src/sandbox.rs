//! sandbox.rs
//!
//! Isolated Execution Sandbox & Guardrails for TARA Core.
//! Real execution connected to OS-level Windows Job Object containment under runtime::sandbox.
//! Zero simulation, zero fake outputs.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::time::Instant;

use crate::destruction::SecureDestructionEngine;
use crate::runtime::sandbox::{IsolatedSandbox, SandboxConfig};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxPolicy {
    pub max_execution_time_ms: u64,
    pub max_memory_mb: usize,
    pub allow_filesystem_write: bool,
    pub allow_network_outbound: bool,
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        // Safe-baseline values for untrusted code execution.
        // Callers must supply explicit values for production workloads;
        // these defaults are the most restrictive operational baseline.
        const DEFAULT_EXEC_TIMEOUT_MS: u64 = 5_000;
        const DEFAULT_MEMORY_MB: usize = 256;
        Self {
            max_execution_time_ms: DEFAULT_EXEC_TIMEOUT_MS,
            max_memory_mb: DEFAULT_MEMORY_MB,
            allow_filesystem_write: false,
            allow_network_outbound: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub success: bool,
    pub output: String,
    pub execution_time_ms: u64,
    pub memory_used_mb: usize,
    pub error: Option<String>,
}

pub struct ExecutionSandbox {
    policy: SandboxPolicy,
}

impl ExecutionSandbox {
    pub fn new(policy: SandboxPolicy) -> Self {
        Self { policy }
    }

    /// Case-insensitive security check for forbidden commands and patterns.
    pub fn contains_forbidden_pattern(code: &str) -> bool {
        let lower = code.to_lowercase();
        let forbidden = [
            "rm -rf",
            "rm -fr",
            "rm -r -f",
            "del /s",
            "rd /s",
            "rmdir /s",
            "format ",
            "mkfs",
            "process.exit",
            "eval(",
            ":(){ :|:& };:",
            "shutdown",
            "reboot",
        ];
        forbidden.iter().any(|&p| lower.contains(p))
    }

    /// Safely executes code with policy bounds and real Windows Job Object isolation.
    pub fn execute_safe(&self, code: &str) -> ExecutionResult {
        if SecureDestructionEngine::is_destroyed() {
            return ExecutionResult {
                success: false,
                output: String::new(),
                execution_time_ms: 0,
                memory_used_mb: 0,
                error: Some("System is destroyed: sandbox execution rejected".into()),
            };
        }

        // Intercept forbidden code patterns
        if Self::contains_forbidden_pattern(code) {
            return ExecutionResult {
                success: false,
                output: String::new(),
                execution_time_ms: 0,
                memory_used_mb: 0,
                error: Some("Blocked: forbidden security pattern detected".into()),
            };
        }

        let start_time = Instant::now();
        let rand_id = rand::random::<u128>();
        let root_dir = std::env::temp_dir().join(format!("tara_sbx_exec_{:032x}", rand_id));

        let config = SandboxConfig {
            sandbox_id: format!("sbx_exec_{:032x}", rand_id),
            display_name: "ExecutionSandbox-Runner".to_string(),
            sandbox_type: "Code".to_string(),
            root_dir: root_dir.clone(),
            max_memory_bytes: self.policy.max_memory_mb * 1024 * 1024,
            max_processes: 2,
            max_disk_bytes: if self.policy.allow_filesystem_write {
                64 * 1024 * 1024
            } else {
                64 * 1024
            },
            execution_timeout_ms: self.policy.max_execution_time_ms,
            allow_network: self.policy.allow_network_outbound,
            allowed_domains: Vec::new(),
            allowed_tools: HashSet::new(),
            environment_variables: std::collections::HashMap::new(),
        };

        let mut sandbox = match IsolatedSandbox::new(config) {
            Ok(sb) => sb,
            Err(e) => {
                return ExecutionResult {
                    success: false,
                    output: String::new(),
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    memory_used_mb: 0,
                    error: Some(format!("Failed to initialize isolated sandbox: {}", e)),
                };
            }
        };

        #[cfg(windows)]
        let (executable, args) = {
            let script_name = "workload.cmd";
            let script_path = Path::new(script_name);
            let script_content = format!("@echo off\r\n{}", code);
            if let Err(e) = sandbox.write_file(script_path, script_content.as_bytes()) {
                let _ = fs::remove_dir_all(&root_dir);
                return ExecutionResult {
                    success: false,
                    output: String::new(),
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    memory_used_mb: 0,
                    error: Some(format!("Failed to write workload to sandbox: {}", e)),
                };
            }
            ("cmd.exe", vec!["/C", script_name])
        };

        #[cfg(not(windows))]
        let (executable, args) = {
            let script_name = "workload.sh";
            let script_path = Path::new(script_name);
            let script_content = format!("#!/bin/sh\n{}", code);
            if let Err(e) = sandbox.write_file(script_path, script_content.as_bytes()) {
                let _ = fs::remove_dir_all(&root_dir);
                return ExecutionResult {
                    success: false,
                    output: String::new(),
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    memory_used_mb: 0,
                    error: Some(format!("Failed to write workload to sandbox: {}", e)),
                };
            }
            ("/bin/sh", vec![script_name])
        };

        let res = sandbox.execute(executable, &args, None);
        let elapsed = start_time.elapsed().as_millis() as u64;
        let _ = sandbox.cleanup();
        let _ = fs::remove_dir_all(&root_dir);

        match res {
            Ok(exec_res) => {
                let success = exec_res.exit_code == 0
                    && !exec_res.timed_out
                    && exec_res.violated_policy.is_none();
                let output = if !exec_res.stdout.is_empty() {
                    exec_res.stdout.trim().to_string()
                } else {
                    exec_res.stderr.trim().to_string()
                };
                let memory_used_mb = (exec_res.peak_memory_bytes / (1024 * 1024)).max(1);
                let error = if !success {
                    if exec_res.timed_out {
                        Some("Execution timed out".to_string())
                    } else if let Some(pol) = exec_res.violated_policy {
                        Some(format!("Policy violation: {}", pol))
                    } else if !exec_res.stderr.is_empty() {
                        Some(exec_res.stderr.trim().to_string())
                    } else {
                        Some(format!(
                            "Process returned non-zero exit code: {}",
                            exec_res.exit_code
                        ))
                    }
                } else {
                    None
                };

                ExecutionResult {
                    success,
                    output,
                    execution_time_ms: elapsed,
                    memory_used_mb,
                    error,
                }
            }
            Err(e) => ExecutionResult {
                success: false,
                output: String::new(),
                execution_time_ms: elapsed,
                memory_used_mb: 0,
                error: Some(e),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contains_forbidden_pattern() {
        assert!(ExecutionSandbox::contains_forbidden_pattern("rm -rf /"));
        assert!(ExecutionSandbox::contains_forbidden_pattern("RM -RF /"));
        assert!(ExecutionSandbox::contains_forbidden_pattern("RM -FR /"));
        assert!(ExecutionSandbox::contains_forbidden_pattern("rm -r -f /"));
        assert!(ExecutionSandbox::contains_forbidden_pattern(
            "cmd /c DEL /S *.*"
        ));
        assert!(ExecutionSandbox::contains_forbidden_pattern(
            "rd /s /q c:\\"
        ));
        assert!(ExecutionSandbox::contains_forbidden_pattern(
            "rmdir /s /q c:\\"
        ));
        assert!(ExecutionSandbox::contains_forbidden_pattern("format c:"));
        assert!(ExecutionSandbox::contains_forbidden_pattern(
            "mkfs.ext4 /dev/sda1"
        ));
        assert!(ExecutionSandbox::contains_forbidden_pattern(
            ":(){ :|:& };:"
        ));
        assert!(ExecutionSandbox::contains_forbidden_pattern(
            "SHUTDOWN /s /t 0"
        ));
        assert!(ExecutionSandbox::contains_forbidden_pattern(
            "sudo reboot now"
        ));
        assert!(!ExecutionSandbox::contains_forbidden_pattern(
            "echo hello world"
        ));
    }

    #[test]
    fn test_execute_safe_echo() {
        let sandbox = ExecutionSandbox::new(SandboxPolicy::default());
        let res = sandbox.execute_safe("echo test_success");
        assert!(res.success);
        assert!(res.output.contains("test_success"));
        assert!(res.error.is_none());
    }
}
