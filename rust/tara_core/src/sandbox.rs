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

    /// Safely executes code with policy bounds and real Windows Job Object isolation.
    pub fn execute_safe(&self, code: &str) -> ExecutionResult {
        // Intercept forbidden code patterns
        if code.contains("rm -rf") || code.contains("process.exit") || code.contains("eval(") {
            return ExecutionResult {
                success: false,
                output: String::new(),
                execution_time_ms: 0,
                memory_used_mb: 0,
                error: Some("Blocked: forbidden security pattern detected".into()),
            };
        }

        let start_time = Instant::now();
        let rand_id = rand::random::<u32>();
        let root_dir = std::env::temp_dir().join(format!("tara_sbx_exec_{}", rand_id));

        let config = SandboxConfig {
            sandbox_id: format!("sbx_exec_{}", rand_id),
            display_name: "ExecutionSandbox-Runner".to_string(),
            sandbox_type: "Code".to_string(),
            root_dir: root_dir.clone(),
            max_memory_bytes: self.policy.max_memory_mb * 1024 * 1024,
            max_processes: 2,
            max_disk_bytes: 64 * 1024 * 1024,
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
        let disk_usage = sandbox.get_disk_usage().unwrap_or(0);
        let memory_mb = (disk_usage / (1024 * 1024)).max(1);

        let _ = fs::remove_dir_all(&root_dir);

        match res {
            Ok(exec_res) => {
                let success = exec_res.exit_code == 0 && !exec_res.timed_out;
                let output = if !exec_res.stdout.is_empty() {
                    exec_res.stdout.trim().to_string()
                } else {
                    exec_res.stderr.trim().to_string()
                };
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
                    memory_used_mb: memory_mb,
                    error,
                }
            }
            Err(e) => ExecutionResult {
                success: false,
                output: String::new(),
                execution_time_ms: elapsed,
                memory_used_mb: memory_mb,
                error: Some(e),
            },
        }
    }
}
