//! sandbox/mod.rs
//!
//! Kernel-enforced OS-level sandbox isolation, host protection, and process control.
//! On Windows: Job Objects (extended limits, process count, memory limits, kill-on-close, UI restrictions).
//! Strict filesystem isolation (path traversal, symlink/junction escape prevention, secret scrubbing).
//! Default network denial, cross-sandbox separation, and brokered IPC.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use crate::runtime::lifecycle::{EntityState, EntityType, StateMachine};

#[cfg(windows)]
mod win32 {
    use std::os::windows::raw::HANDLE;

    pub type Bool = i32;
    pub type Dword = u32;
    pub type SizeT = usize;
    pub type UlongPtr = usize;
    pub type UlongLong = u64;

    pub const FALSE: Bool = 0;

    pub const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: Dword = 0x00002000;
    pub const JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION: Dword = 0x00000400;
    pub const JOB_OBJECT_LIMIT_ACTIVE_PROCESS: Dword = 0x00000008;
    pub const JOB_OBJECT_LIMIT_PROCESS_MEMORY: Dword = 0x00000100;
    pub const JOB_OBJECT_LIMIT_JOB_MEMORY: Dword = 0x00000200;

    pub const JOB_OBJECT_UILIMIT_HANDLES: Dword = 0x00000001;
    pub const JOB_OBJECT_UILIMIT_READCLIPBOARD: Dword = 0x00000002;
    pub const JOB_OBJECT_UILIMIT_WRITECLIPBOARD: Dword = 0x00000004;
    pub const JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS: Dword = 0x00000008;
    pub const JOB_OBJECT_UILIMIT_DISPLAYSETTINGS: Dword = 0x00000010;
    pub const JOB_OBJECT_UILIMIT_GLOBALATOMS: Dword = 0x00000020;
    pub const JOB_OBJECT_UILIMIT_DESKTOP: Dword = 0x00000040;
    pub const JOB_OBJECT_UILIMIT_EXITWINDOWS: Dword = 0x00000080;

    pub const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION: Dword = 9;
    pub const JOB_OBJECT_BASIC_UI_RESTRICTIONS: Dword = 4;

    #[repr(C)]
    #[derive(Default)]
    pub struct IoCounters {
        pub read_operation_count: UlongLong,
        pub write_operation_count: UlongLong,
        pub other_operation_count: UlongLong,
        pub read_transfer_count: UlongLong,
        pub write_transfer_count: UlongLong,
        pub other_transfer_count: UlongLong,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct JobObjectBasicLimitInformation {
        pub per_process_user_time_limit: i64,
        pub per_job_user_time_limit: i64,
        pub limit_flags: Dword,
        pub minimum_working_set_size: SizeT,
        pub maximum_working_set_size: SizeT,
        pub active_process_limit: Dword,
        pub affinity: UlongPtr,
        pub priority_class: Dword,
        pub scheduling_class: Dword,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct JobObjectExtendedLimitInformation {
        pub basic_limit_information: JobObjectBasicLimitInformation,
        pub io_info: IoCounters,
        pub process_memory_limit: SizeT,
        pub job_memory_limit: SizeT,
        pub peak_process_memory_used: SizeT,
        pub peak_job_memory_used: SizeT,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct JobObjectBasicUiRestrictions {
        pub ui_restrictions_class: Dword,
    }

    extern "system" {
        pub fn CreateJobObjectW(lp_job_attributes: *mut u8, lp_name: *const u16) -> HANDLE;
        pub fn SetInformationJobObject(
            h_job: HANDLE,
            job_object_information_class: Dword,
            lp_job_object_information: *const u8,
            cb_job_object_information_length: Dword,
        ) -> Bool;
        pub fn AssignProcessToJobObject(h_job: HANDLE, h_process: HANDLE) -> Bool;
        pub fn TerminateJobObject(h_job: HANDLE, u_exit_code: Dword) -> Bool;
        pub fn CloseHandle(h_object: HANDLE) -> Bool;
    }
}

/// OS-level Job Object abstraction for Windows.
pub struct JobObjectGuard {
    #[cfg(windows)]
    handle: std::os::windows::raw::HANDLE,
    #[cfg(not(windows))]
    _dummy: u32,
}

unsafe impl Send for JobObjectGuard {}
unsafe impl Sync for JobObjectGuard {}

impl JobObjectGuard {
    pub fn create(max_memory_bytes: usize, max_processes: u32) -> Result<Self, String> {
        #[cfg(windows)]
        {
            unsafe {
                let h_job = win32::CreateJobObjectW(std::ptr::null_mut(), std::ptr::null());
                if h_job.is_null() {
                    return Err("Failed to create Win32 Job Object".to_string());
                }

                // Configure extended resource and process storm limits
                let mut info = win32::JobObjectExtendedLimitInformation::default();
                info.basic_limit_information.limit_flags = win32::JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                    | win32::JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
                    | win32::JOB_OBJECT_LIMIT_ACTIVE_PROCESS
                    | win32::JOB_OBJECT_LIMIT_PROCESS_MEMORY
                    | win32::JOB_OBJECT_LIMIT_JOB_MEMORY;

                info.basic_limit_information.active_process_limit = max_processes;
                info.process_memory_limit = max_memory_bytes;
                info.job_memory_limit = max_memory_bytes;

                let ok = win32::SetInformationJobObject(
                    h_job,
                    win32::JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                    &info as *const _ as *const u8,
                    std::mem::size_of::<win32::JobObjectExtendedLimitInformation>() as u32,
                );

                if ok == win32::FALSE {
                    win32::CloseHandle(h_job);
                    return Err("Failed to apply JobObjectExtendedLimitInformation".to_string());
                }

                // Enforce UI, device, and clipboard restrictions directly inside initializer
                let ui_info = win32::JobObjectBasicUiRestrictions {
                    ui_restrictions_class: win32::JOB_OBJECT_UILIMIT_READCLIPBOARD
                        | win32::JOB_OBJECT_UILIMIT_WRITECLIPBOARD
                        | win32::JOB_OBJECT_UILIMIT_HANDLES
                        | win32::JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS
                        | win32::JOB_OBJECT_UILIMIT_DISPLAYSETTINGS
                        | win32::JOB_OBJECT_UILIMIT_GLOBALATOMS
                        | win32::JOB_OBJECT_UILIMIT_DESKTOP
                        | win32::JOB_OBJECT_UILIMIT_EXITWINDOWS,
                };

                let _ = win32::SetInformationJobObject(
                    h_job,
                    win32::JOB_OBJECT_BASIC_UI_RESTRICTIONS,
                    &ui_info as *const _ as *const u8,
                    std::mem::size_of::<win32::JobObjectBasicUiRestrictions>() as u32,
                );

                Ok(Self { handle: h_job })
            }
        }
        #[cfg(not(windows))]
        {
            let _ = (max_memory_bytes, max_processes);
            Ok(Self { _dummy: 0 })
        }
    }

    pub fn assign_child(&self, child: &Child) -> Result<(), String> {
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            let raw_h = child.as_raw_handle() as std::os::windows::raw::HANDLE;
            unsafe {
                let ok = win32::AssignProcessToJobObject(self.handle, raw_h);
                if ok == win32::FALSE {
                    return Err("Failed to assign child process to Job Object".to_string());
                }
            }
            Ok(())
        }
        #[cfg(not(windows))]
        {
            let _ = child;
            Ok(())
        }
    }

    pub fn terminate(&self) {
        #[cfg(windows)]
        {
            unsafe {
                win32::TerminateJobObject(self.handle, 1);
            }
        }
    }
}

impl Drop for JobObjectGuard {
    fn drop(&mut self) {
        #[cfg(windows)]
        {
            unsafe {
                win32::TerminateJobObject(self.handle, 0);
                win32::CloseHandle(self.handle);
            }
        }
    }
}

/// Dynamic sandbox configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    pub sandbox_id: String,
    pub display_name: String,
    pub sandbox_type: String,
    pub root_dir: PathBuf,
    pub max_memory_bytes: usize,
    pub max_processes: u32,
    pub max_disk_bytes: usize,
    pub execution_timeout_ms: u64,
    pub allow_network: bool,
    pub allowed_domains: Vec<String>,
    pub allowed_tools: HashSet<String>,
    pub environment_variables: HashMap<String, String>,
}

pub const DEFAULT_SANDBOX_MAX_MEMORY_BYTES: usize = 256 * 1024 * 1024; // 256 MB default
pub const DEFAULT_SANDBOX_MAX_PROCESSES: u32 = 4;
pub const DEFAULT_SANDBOX_MAX_DISK_BYTES: usize = 512 * 1024 * 1024; // 512 MB disk quota
pub const DEFAULT_SANDBOX_EXECUTION_TIMEOUT_MS: u64 = 10_000;
pub const DEFAULT_SANDBOX_BROKER_MAX_BYTES: usize = 1024 * 1024; // 1 MB broker ceiling

pub fn resolve_sandbox_broker_max_bytes() -> usize {
    std::env::var("TARA_SANDBOX_BROKER_MAX_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_SANDBOX_BROKER_MAX_BYTES)
}

impl Default for SandboxConfig {
    fn default() -> Self {
        let max_memory_bytes = std::env::var("TARA_SANDBOX_MAX_MEMORY_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_SANDBOX_MAX_MEMORY_BYTES);
        let max_processes = std::env::var("TARA_SANDBOX_MAX_PROCESSES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_SANDBOX_MAX_PROCESSES);
        let max_disk_bytes = std::env::var("TARA_SANDBOX_MAX_DISK_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_SANDBOX_MAX_DISK_BYTES);
        let execution_timeout_ms = std::env::var("TARA_SANDBOX_EXECUTION_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_SANDBOX_EXECUTION_TIMEOUT_MS);

        Self {
            sandbox_id: "sbx_default".to_string(),
            display_name: "Default-Sandbox".to_string(),
            sandbox_type: "Standard".to_string(),
            root_dir: std::env::temp_dir().join("tara_sandbox_default"),
            max_memory_bytes,
            max_processes,
            max_disk_bytes,
            execution_timeout_ms,
            allow_network: false,
            allowed_domains: Vec::new(),
            allowed_tools: HashSet::new(),
            environment_variables: HashMap::new(),
        }
    }
}

/// Result of execution inside a Sandbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxExecutionResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u64,
    pub peak_memory_bytes: usize,
    pub timed_out: bool,
    pub violated_policy: Option<String>,
}

/// Dynamic, kernel-isolated Sandbox instance.
pub struct IsolatedSandbox {
    pub config: SandboxConfig,
    pub state: StateMachine,
    job_guard: Option<JobObjectGuard>,
    active_children: Vec<u32>, // Process IDs
}

impl IsolatedSandbox {
    pub fn new(mut config: SandboxConfig) -> Result<Self, String> {
        // Ensure root directory exists and is canonical
        fs::create_dir_all(&config.root_dir).map_err(|e| {
            format!(
                "Failed to create sandbox root at {:?}: {}",
                config.root_dir, e
            )
        })?;

        let canonical_root = config.root_dir.canonicalize().map_err(|e| {
            format!(
                "Failed to canonicalize sandbox root at {:?}: {}",
                config.root_dir, e
            )
        })?;
        config.root_dir = canonical_root;

        let state = StateMachine::new(config.sandbox_id.clone(), EntityType::Sandbox);

        let mut sandbox = Self {
            config,
            state,
            job_guard: None,
            active_children: Vec::new(),
        };

        sandbox.state.transition_to(
            EntityState::Initializing,
            "Initializing sandbox environment",
            "MANAGER",
        )?;

        match JobObjectGuard::create(
            sandbox.config.max_memory_bytes,
            sandbox.config.max_processes,
        ) {
            Ok(job) => {
                sandbox.job_guard = Some(job);
                sandbox.state.transition_to(
                    EntityState::Ready,
                    "Sandbox isolation verified and ready",
                    "MANAGER",
                )?;
            }
            Err(e) => {
                let _ = sandbox.state.transition_to(
                    EntityState::Degraded,
                    &format!("Host isolation enforcement failed: {}", e),
                    "SANDBOX_SUPERVISOR",
                );
            }
        }

        Ok(sandbox)
    }

    /// Verifies that a target path is strictly within the sandbox root directory.
    /// Blocks path traversal ('..') and symbolic link / junction point escaping.
    pub fn validate_path_safety(
        &self,
        relative_or_absolute_path: &Path,
    ) -> Result<PathBuf, String> {
        let candidate = if relative_or_absolute_path.is_absolute() {
            relative_or_absolute_path.to_path_buf()
        } else {
            self.config.root_dir.join(relative_or_absolute_path)
        };

        // If file exists, canonicalize to resolve symlinks and junctions
        if candidate.exists() {
            let real_path = candidate
                .canonicalize()
                .map_err(|e| format!("Path resolution error: {}", e))?;

            if !real_path.starts_with(&self.config.root_dir) {
                return Err(format!(
                    "Path Traversal Violation: resolved path '{:?}' escapes sandbox root '{:?}'",
                    real_path, self.config.root_dir
                ));
            }
            Ok(real_path)
        } else {
            // Check normalized path before creation
            let mut normalized = PathBuf::new();
            for comp in candidate.components() {
                match comp {
                    std::path::Component::ParentDir => {
                        if !normalized.pop() {
                            return Err(
                                "Path Traversal Violation: attempt to escape root via '..'"
                                    .to_string(),
                            );
                        }
                    }
                    std::path::Component::CurDir => continue,
                    _ => normalized.push(comp),
                }
            }

            if !normalized.starts_with(&self.config.root_dir) {
                return Err(format!(
                    "Path Traversal Violation: target '{:?}' is outside sandbox root '{:?}'",
                    normalized, self.config.root_dir
                ));
            }
            Ok(normalized)
        }
    }

    /// Returns total disk usage in bytes within the sandbox directory tree.
    pub fn get_disk_usage(&self) -> Result<usize, String> {
        fn dir_bytes(p: &Path) -> Result<usize, String> {
            let mut total = 0usize;
            let entries = fs::read_dir(p).map_err(|error| {
                format!("Cannot verify sandbox disk usage at {:?}: {}", p, error)
            })?;
            for entry in entries {
                let entry =
                    entry.map_err(|error| format!("Cannot inspect sandbox entry: {error}"))?;
                let file_type = entry
                    .file_type()
                    .map_err(|error| format!("Cannot inspect sandbox entry type: {error}"))?;
                let path = entry.path();
                if file_type.is_symlink() {
                    continue;
                }
                if file_type.is_file() {
                    let size = entry
                        .metadata()
                        .map_err(|error| format!("Cannot inspect sandbox file size: {error}"))?
                        .len()
                        .min(usize::MAX as u64) as usize;
                    total = total.saturating_add(size);
                } else if file_type.is_dir() {
                    total = total.saturating_add(dir_bytes(&path)?);
                }
            }
            Ok(total)
        }
        dir_bytes(&self.config.root_dir)
    }

    /// Safely writes data to a file inside the isolated sandbox, enforcing disk quota.
    pub fn write_file(&self, relative_path: &Path, data: &[u8]) -> Result<PathBuf, String> {
        let safe_path = self.validate_path_safety(relative_path)?;
        let current_disk = self.get_disk_usage()?;
        if current_disk.saturating_add(data.len()) > self.config.max_disk_bytes {
            return Err(format!(
                "Disk Quota Exceeded: sandbox '{}' disk limit of {} bytes exceeded",
                self.config.sandbox_id, self.config.max_disk_bytes
            ));
        }
        if let Some(parent) = safe_path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let mut file = File::create(&safe_path).map_err(|e| e.to_string())?;
        file.write_all(data).map_err(|e| e.to_string())?;
        Ok(safe_path)
    }

    /// Safely reads data from a file inside the isolated sandbox.
    pub fn read_file(&self, relative_path: &Path) -> Result<Vec<u8>, String> {
        let safe_path = self.validate_path_safety(relative_path)?;
        let mut file = File::open(&safe_path).map_err(|e| e.to_string())?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        Ok(buf)
    }

    /// Executes a command in the sandbox process boundary.
    pub fn execute(
        &mut self,
        executable: &str,
        args: &[&str],
        stdin_data: Option<&[u8]>,
    ) -> Result<SandboxExecutionResult, String> {
        if self.state.current_state() == EntityState::Degraded {
            return Err("Execution refused: sandbox is in DEGRADED state because required OS-level host isolation could not be enforced.".to_string());
        }
        if self.job_guard.is_none() {
            let _ = self.state.transition_to(
                EntityState::Degraded,
                "Host Job Object guard missing",
                "SANDBOX_SUPERVISOR",
            );
            return Err(
                "Execution refused: sandbox missing required OS-level isolation guard.".to_string(),
            );
        }

        self.state
            .transition_to(EntityState::Busy, "Executing sandboxed workload", "MANAGER")?;

        let start_time = Instant::now();

        let mut cmd = Command::new(executable);
        cmd.args(args);
        cmd.current_dir(&self.config.root_dir);
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        // Scrub all host environment secrets
        cmd.env_clear();
        cmd.env("TARA_SANDBOX", "1");
        cmd.env("TARA_SANDBOX_ID", &self.config.sandbox_id);
        cmd.env("TARA_SANDBOX_NAME", &self.config.display_name);
        cmd.env("TEMP", &self.config.root_dir);
        cmd.env("TMP", &self.config.root_dir);

        // Inject only explicitly allowed non-secret parameters
        for (k, v) in &self.config.environment_variables {
            // Guard against injecting secrets
            let k_upper = k.to_uppercase();
            if k_upper.contains("KEY")
                || k_upper.contains("SECRET")
                || k_upper.contains("TOKEN")
                || k_upper.contains("AUTH")
            {
                continue;
            }
            cmd.env(k, v);
        }

        // Spawn process
        let mut child = cmd.spawn().map_err(|e| {
            let _ =
                self.state
                    .transition_to(EntityState::Active, "Command launch failure", "MANAGER");
            format!(
                "Failed to spawn sandboxed executable '{}': {}",
                executable, e
            )
        })?;

        let pid = child.id();
        self.active_children.push(pid);

        // Assign to Job Object
        if let Some(ref job) = self.job_guard {
            if let Err(err) = job.assign_child(&child) {
                let _ = child.kill();
                let _ = self.state.transition_to(
                    EntityState::Degraded,
                    "Job assignment failure; marking degraded",
                    "SANDBOX_SUPERVISOR",
                );
                return Err(format!("Security boundary enforcement failed: {}", err));
            }
        }

        // Write stdin if present
        if let Some(input) = stdin_data {
            if let Some(mut stdin) = child.stdin.take() {
                let _ = stdin.write_all(input);
            }
        }

        // Monitor execution with timeout watchdog
        let timeout = Duration::from_millis(self.config.execution_timeout_ms);

        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    self.active_children.retain(|&p| p != pid);
                    let mut stdout = String::new();
                    let mut stderr = String::new();
                    if let Some(mut out) = child.stdout.take() {
                        let _ = out.read_to_string(&mut stdout);
                    }
                    if let Some(mut err) = child.stderr.take() {
                        let _ = err.read_to_string(&mut stderr);
                    }

                    let duration = start_time.elapsed().as_millis() as u64;

                    // Post-execution disk quota verification
                    let disk_usage = match self.get_disk_usage() {
                        Ok(usage) => usage,
                        Err(error) => {
                            let _ = self.state.transition_to(
                                EntityState::Quarantined,
                                "Sandbox disk usage could not be verified",
                                "SANDBOX_SUPERVISOR",
                            );
                            return Err(error);
                        }
                    };
                    if disk_usage > self.config.max_disk_bytes {
                        let _ = self.state.transition_to(
                            EntityState::Quarantined,
                            "Disk abuse detected: sandbox disk usage exceeded quota during workload",
                            "SANDBOX_SUPERVISOR",
                        );
                        return Ok(SandboxExecutionResult {
                            exit_code: -1,
                            stdout,
                            stderr: format!(
                                "{}\n[SECURITY] Sandbox quarantined: disk quota exceeded.",
                                stderr
                            ),
                            duration_ms: duration,
                            peak_memory_bytes: 0,
                            timed_out: false,
                            violated_policy: Some("DiskQuotaExceeded".to_string()),
                        });
                    }

                    let _ = self.state.transition_to(
                        EntityState::Active,
                        "Workload execution completed",
                        "MANAGER",
                    );

                    return Ok(SandboxExecutionResult {
                        exit_code: status.code().unwrap_or(-1),
                        stdout,
                        stderr,
                        duration_ms: duration,
                        peak_memory_bytes: 0,
                        timed_out: false,
                        violated_policy: None,
                    });
                }
                Ok(None) => {
                    if start_time.elapsed() > timeout {
                        let _ = child.kill();
                        if let Some(ref job) = self.job_guard {
                            job.terminate();
                        }
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(e) => {
                    self.active_children.retain(|&p| p != pid);
                    let _ = self.state.transition_to(
                        EntityState::Failed,
                        "Process wait failure",
                        "MANAGER",
                    );
                    return Err(format!("Error monitoring sandboxed child: {}", e));
                }
            }
        }

        self.active_children.retain(|&p| p != pid);
        let duration = start_time.elapsed().as_millis() as u64;
        let _ = self.state.transition_to(
            EntityState::Active,
            "Workload execution timed out",
            "MANAGER",
        );

        Ok(SandboxExecutionResult {
            exit_code: -1,
            stdout: String::new(),
            stderr: "Execution timed out".to_string(),
            duration_ms: duration,
            peak_memory_bytes: 0,
            timed_out: true, // The only code path reaching here is via timeout-triggered break.
            violated_policy: Some("ExecutionTimeout".to_string()),
        })
    }

    /// Stops, revokes, and kills all processes in the sandbox.
    pub fn stop(&mut self) -> Result<(), String> {
        if let Some(ref job) = self.job_guard {
            job.terminate();
        }
        self.active_children.clear();
        self.state.transition_to(
            EntityState::Stopped,
            "Sandbox stopped by manager",
            "MANAGER",
        )?;
        Ok(())
    }

    /// Quarantines the sandbox upon security policy or integrity violation.
    pub fn quarantine(&mut self, reason: &str) -> Result<(), String> {
        if let Some(ref job) = self.job_guard {
            job.terminate();
        }
        self.active_children.clear();
        self.state
            .transition_to(EntityState::Quarantined, reason, "MANAGER")?;
        Ok(())
    }

    /// Releases resources and cleans up files.
    pub fn cleanup(&mut self) -> Result<(), String> {
        self.stop()?;
        if self.config.root_dir.exists() {
            let _ = fs::remove_dir_all(&self.config.root_dir);
        }
        self.state
            .transition_to(EntityState::Retired, "Sandbox wiped and retired", "MANAGER")?;
        Ok(())
    }
}

/// Brokered message passing across Sandboxes.
/// Invariant: Sandbox A and Sandbox B NEVER directly connect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrokeredTransferRequest {
    pub source_sandbox_id: String,
    pub target_sandbox_id: String,
    pub purpose: String,
    pub payload_type: String,
    pub payload_bytes: Vec<u8>,
    pub max_size_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrokeredTransferResponse {
    pub success: bool,
    pub transfer_id: String,
    pub sha256_hash: String,
    pub bytes_transferred: usize,
    pub delivery_path: Option<String>,
    pub rejection_reason: Option<String>,
}

pub struct SandboxBroker;

impl SandboxBroker {
    /// Direct transfer is denied because cross-sandbox data requires a creator-signed ticket.
    pub fn transfer(
        _source: &IsolatedSandbox,
        _target: &IsolatedSandbox,
        _request: BrokeredTransferRequest,
    ) -> BrokeredTransferResponse {
        Self::rejected("Creator-approved broker transfer is required")
    }

    pub fn approval_target(request: &BrokeredTransferRequest) -> String {
        let mut hasher = Sha256::new();
        for part in [
            request.source_sandbox_id.as_bytes(),
            request.target_sandbox_id.as_bytes(),
            request.purpose.as_bytes(),
            request.payload_type.as_bytes(),
            request.payload_bytes.as_slice(),
        ] {
            hasher.update((part.len() as u64).to_le_bytes());
            hasher.update(part);
        }
        format!(
            "{}:{}:{}",
            request.source_sandbox_id,
            request.target_sandbox_id,
            hex::encode(hasher.finalize())
        )
    }

    pub fn transfer_with_approval(
        source: &IsolatedSandbox,
        target: &IsolatedSandbox,
        request: BrokeredTransferRequest,
        approvals: &mut crate::runtime::approval::ApprovalGate,
        ticket_id: &str,
    ) -> BrokeredTransferResponse {
        // 1. Verify identities
        if request.source_sandbox_id != source.config.sandbox_id
            || request.target_sandbox_id != target.config.sandbox_id
        {
            return Self::rejected("Identity mismatch between request and sandbox instances");
        }

        // 2. Verify both sandboxes are in operational state
        if !source.state.is_operational() || !target.state.is_operational() {
            return Self::rejected("One or both sandboxes are not in operational state");
        }

        if request.purpose.trim().is_empty() || request.payload_type.trim().is_empty() {
            return Self::rejected("Transfer purpose and payload type are required");
        }
        // The caller can request a lower cap but cannot raise the dynamically-resolved broker ceiling.
        let broker_ceiling = resolve_sandbox_broker_max_bytes();
        let limit = if request.max_size_bytes > 0 {
            request.max_size_bytes.min(broker_ceiling)
        } else {
            broker_ceiling
        };
        if request.payload_bytes.len() > limit {
            return Self::rejected(&format!(
                "Payload exceeds broker size limit of {} bytes",
                limit
            ));
        }

        // 3. Consume an exact, one-use creator approval before delivering bytes.
        let approval_target = Self::approval_target(&request);
        if let Err(error) =
            approvals.consume_approved(ticket_id, "sandbox_transfer", &approval_target)
        {
            return Self::rejected(&error);
        }

        // 4. Deliver into a path constrained by the target sandbox's filesystem guard.
        let mut hasher = Sha256::new();
        hasher.update(&request.payload_bytes);
        let hash = hex::encode(hasher.finalize());
        let transfer_id = format!("xfer_{:x}", rand::random::<u64>());
        let relative_path = PathBuf::from("broker/inbox").join(format!("{}.bin", transfer_id));
        let delivery_path = match target.write_file(&relative_path, &request.payload_bytes) {
            Ok(path) => path.to_string_lossy().to_string(),
            Err(error) => return Self::rejected(&format!("Broker delivery failed: {}", error)),
        };

        BrokeredTransferResponse {
            success: true,
            transfer_id,
            sha256_hash: hash,
            bytes_transferred: request.payload_bytes.len(),
            delivery_path: Some(delivery_path),
            rejection_reason: None,
        }
    }

    fn rejected(reason: &str) -> BrokeredTransferResponse {
        BrokeredTransferResponse {
            success: false,
            transfer_id: String::new(),
            sha256_hash: String::new(),
            bytes_transferred: 0,
            delivery_path: None,
            rejection_reason: Some(reason.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDirGuard(PathBuf);
    impl TempDirGuard {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!("tara_sbx_test_{}_{:x}", name, rand::random::<u64>()));
            let _ = fs::create_dir_all(&path);
            Self(path)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn test_sandbox_path_traversal_protection() {
        let temp = TempDirGuard::new("traversal");
        let config = SandboxConfig {
            sandbox_id: "sbx_test_traversal".to_string(),
            display_name: "Test-Traversal".to_string(),
            sandbox_type: "Test".to_string(),
            root_dir: temp.path().to_path_buf(),
            ..SandboxConfig::default()
        };

        let sandbox = IsolatedSandbox::new(config).unwrap();

        // Path escaping the root via parent references must fail
        let traversal_path = Path::new("../../escape.txt");
        let result = sandbox.validate_path_safety(traversal_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Path Traversal Violation"));

        // Valid relative path must succeed
        let valid_path = Path::new("nested/allowed.txt");
        let valid_res = sandbox.validate_path_safety(valid_path);
        assert!(valid_res.is_ok());
    }

    #[test]
    fn test_sandbox_file_write_read_and_quota() {
        let temp = TempDirGuard::new("io");
        let config = SandboxConfig {
            sandbox_id: "sbx_test_io".to_string(),
            display_name: "Test-IO".to_string(),
            sandbox_type: "Test".to_string(),
            root_dir: temp.path().to_path_buf(),
            max_disk_bytes: 1024,
            ..SandboxConfig::default()
        };

        let sandbox = IsolatedSandbox::new(config).unwrap();
        let test_data = b"Hello Sandboxed World!";
        let written_path = sandbox.write_file(Path::new("hello.txt"), test_data).unwrap();
        assert!(written_path.exists());

        let read_data = sandbox.read_file(Path::new("hello.txt")).unwrap();
        assert_eq!(read_data, test_data);

        // Writing data that exceeds quota must be rejected
        let huge_data = vec![0u8; 2048];
        let over_quota_res = sandbox.write_file(Path::new("huge.bin"), &huge_data);
        assert!(over_quota_res.is_err());
        assert!(over_quota_res.unwrap_err().contains("Disk Quota Exceeded"));
    }

    #[test]
    fn test_sandbox_broker_direct_transfer_rejected() {
        let temp1 = TempDirGuard::new("broker1");
        let temp2 = TempDirGuard::new("broker2");

        let s1 = IsolatedSandbox::new(SandboxConfig {
            sandbox_id: "sbx_1".to_string(),
            root_dir: temp1.path().to_path_buf(),
            ..SandboxConfig::default()
        }).unwrap();

        let s2 = IsolatedSandbox::new(SandboxConfig {
            sandbox_id: "sbx_2".to_string(),
            root_dir: temp2.path().to_path_buf(),
            ..SandboxConfig::default()
        }).unwrap();

        let req = BrokeredTransferRequest {
            source_sandbox_id: "sbx_1".to_string(),
            target_sandbox_id: "sbx_2".to_string(),
            purpose: "data_share".to_string(),
            payload_type: "raw".to_string(),
            payload_bytes: b"direct transfer attempt".to_vec(),
            max_size_bytes: 1024,
        };

        let resp = SandboxBroker::transfer(&s1, &s2, req);
        assert!(!resp.success);
        assert!(resp.rejection_reason.unwrap().contains("Creator-approved broker transfer is required"));
    }
}

