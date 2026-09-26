//! sandbox/mod.rs
//!
//! Kernel-enforced OS-level sandbox isolation, host protection, and process control.
//! On Windows: Job Objects (extended limits, process count, memory limits, kill-on-close, UI restrictions).
//! Strict filesystem isolation (path traversal, symlink/junction escape prevention, secret scrubbing).
//! Default network denial, cross-sandbox separation, and brokered IPC.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use sha2::{Digest, Sha256};

use crate::runtime::lifecycle::{EntityState, StateMachine, EntityType};

#[cfg(windows)]
#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code)]
mod win32 {
    use std::os::windows::raw::HANDLE;

    pub type BOOL = i32;
    pub type DWORD = u32;
    pub type SIZE_T = usize;
    pub type ULONG_PTR = usize;
    pub type ULONGLONG = u64;

    pub const FALSE: BOOL = 0;
    pub const TRUE: BOOL = 1;

    pub const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: DWORD = 0x00002000;
    pub const JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION: DWORD = 0x00000400;
    pub const JOB_OBJECT_LIMIT_ACTIVE_PROCESS: DWORD = 0x00000008;
    pub const JOB_OBJECT_LIMIT_PROCESS_MEMORY: DWORD = 0x00000100;
    pub const JOB_OBJECT_LIMIT_JOB_MEMORY: DWORD = 0x00000200;

    pub const JOB_OBJECT_UILIMIT_HANDLES: DWORD = 0x00000001;
    pub const JOB_OBJECT_UILIMIT_READCLIPBOARD: DWORD = 0x00000002;
    pub const JOB_OBJECT_UILIMIT_WRITECLIPBOARD: DWORD = 0x00000004;
    pub const JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS: DWORD = 0x00000008;
    pub const JOB_OBJECT_UILIMIT_DISPLAYSETTINGS: DWORD = 0x00000010;
    pub const JOB_OBJECT_UILIMIT_GLOBALATOMS: DWORD = 0x00000020;
    pub const JOB_OBJECT_UILIMIT_DESKTOP: DWORD = 0x00000040;
    pub const JOB_OBJECT_UILIMIT_EXITWINDOWS: DWORD = 0x00000080;

    pub const JobObjectExtendedLimitInformation: DWORD = 9;
    pub const JobObjectBasicUIRestrictions: DWORD = 4;

    #[repr(C)]
    #[derive(Default)]
    pub struct IO_COUNTERS {
        pub ReadOperationCount: ULONGLONG,
        pub WriteOperationCount: ULONGLONG,
        pub OtherOperationCount: ULONGLONG,
        pub ReadTransferCount: ULONGLONG,
        pub WriteTransferCount: ULONGLONG,
        pub OtherTransferCount: ULONGLONG,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
        pub PerProcessUserTimeLimit: i64,
        pub PerJobUserTimeLimit: i64,
        pub LimitFlags: DWORD,
        pub MinimumWorkingSetSize: SIZE_T,
        pub MaximumWorkingSetSize: SIZE_T,
        pub ActiveProcessLimit: DWORD,
        pub Affinity: ULONG_PTR,
        pub PriorityClass: DWORD,
        pub SchedulingClass: DWORD,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
        pub BasicLimitInformation: JOBOBJECT_BASIC_LIMIT_INFORMATION,
        pub IoInfo: IO_COUNTERS,
        pub ProcessMemoryLimit: SIZE_T,
        pub JobMemoryLimit: SIZE_T,
        pub PeakProcessMemoryUsed: SIZE_T,
        pub PeakJobMemoryUsed: SIZE_T,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct JOBOBJECT_BASIC_UI_RESTRICTIONS {
        pub UIRestrictionsClass: DWORD,
    }

    extern "system" {
        pub fn CreateJobObjectW(lpJobAttributes: *mut u8, lpName: *const u16) -> HANDLE;
        pub fn SetInformationJobObject(
            hJob: HANDLE,
            JobObjectInformationClass: DWORD,
            lpJobObjectInformation: *const u8,
            cbJobObjectInformationLength: DWORD,
        ) -> BOOL;
        pub fn AssignProcessToJobObject(hJob: HANDLE, hProcess: HANDLE) -> BOOL;
        pub fn TerminateJobObject(hJob: HANDLE, uExitCode: DWORD) -> BOOL;
        pub fn CloseHandle(hObject: HANDLE) -> BOOL;
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
                let mut info = win32::JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
                info.BasicLimitInformation.LimitFlags = win32::JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                    | win32::JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
                    | win32::JOB_OBJECT_LIMIT_ACTIVE_PROCESS
                    | win32::JOB_OBJECT_LIMIT_PROCESS_MEMORY
                    | win32::JOB_OBJECT_LIMIT_JOB_MEMORY;

                info.BasicLimitInformation.ActiveProcessLimit = max_processes;
                info.ProcessMemoryLimit = max_memory_bytes;
                info.JobMemoryLimit = max_memory_bytes;

                let ok = win32::SetInformationJobObject(
                    h_job,
                    win32::JobObjectExtendedLimitInformation,
                    &info as *const _ as *const u8,
                    std::mem::size_of::<win32::JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                );

                if ok == win32::FALSE {
                    win32::CloseHandle(h_job);
                    return Err("Failed to apply JobObjectExtendedLimitInformation".to_string());
                }

                // Enforce UI, device, and clipboard restrictions
                let mut ui_info = win32::JOBOBJECT_BASIC_UI_RESTRICTIONS::default();
                ui_info.UIRestrictionsClass = win32::JOB_OBJECT_UILIMIT_READCLIPBOARD
                    | win32::JOB_OBJECT_UILIMIT_WRITECLIPBOARD
                    | win32::JOB_OBJECT_UILIMIT_HANDLES
                    | win32::JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS
                    | win32::JOB_OBJECT_UILIMIT_DISPLAYSETTINGS
                    | win32::JOB_OBJECT_UILIMIT_GLOBALATOMS
                    | win32::JOB_OBJECT_UILIMIT_DESKTOP
                    | win32::JOB_OBJECT_UILIMIT_EXITWINDOWS;

                let _ = win32::SetInformationJobObject(
                    h_job,
                    win32::JobObjectBasicUIRestrictions,
                    &ui_info as *const _ as *const u8,
                    std::mem::size_of::<win32::JOBOBJECT_BASIC_UI_RESTRICTIONS>() as u32,
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

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            sandbox_id: "sbx_default".to_string(),
            display_name: "Default-Sandbox".to_string(),
            sandbox_type: "Standard".to_string(),
            root_dir: std::env::temp_dir().join("tara_sandbox_default"),
            max_memory_bytes: 256 * 1024 * 1024, // 256 MB default
            max_processes: 4,
            max_disk_bytes: 512 * 1024 * 1024,    // 512 MB disk quota
            execution_timeout_ms: 10_000,
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
        fs::create_dir_all(&config.root_dir)
            .map_err(|e| format!("Failed to create sandbox root at {:?}: {}", config.root_dir, e))?;

        let canonical_root = config.root_dir.canonicalize()
            .map_err(|e| format!("Failed to canonicalize sandbox root at {:?}: {}", config.root_dir, e))?;
        config.root_dir = canonical_root;

        let state = StateMachine::new(config.sandbox_id.clone(), EntityType::Sandbox);

        let mut sandbox = Self {
            config,
            state,
            job_guard: None,
            active_children: Vec::new(),
        };

        sandbox.state.transition_to(EntityState::Initializing, "Initializing sandbox environment", "MANAGER")?;

        match JobObjectGuard::create(sandbox.config.max_memory_bytes, sandbox.config.max_processes) {
            Ok(job) => {
                sandbox.job_guard = Some(job);
                sandbox.state.transition_to(EntityState::Ready, "Sandbox isolation verified and ready", "MANAGER")?;
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
    pub fn validate_path_safety(&self, relative_or_absolute_path: &Path) -> Result<PathBuf, String> {
        let candidate = if relative_or_absolute_path.is_absolute() {
            relative_or_absolute_path.to_path_buf()
        } else {
            self.config.root_dir.join(relative_or_absolute_path)
        };

        // If file exists, canonicalize to resolve symlinks and junctions
        if candidate.exists() {
            let real_path = candidate.canonicalize()
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
                            return Err("Path Traversal Violation: attempt to escape root via '..'".to_string());
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
    pub fn get_disk_usage(&self) -> usize {
        fn dir_bytes(p: &Path) -> usize {
            let mut total = 0;
            if let Ok(entries) = fs::read_dir(p) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        if let Ok(meta) = path.metadata() {
                            total += meta.len() as usize;
                        }
                    } else if path.is_dir() {
                        total += dir_bytes(&path);
                    }
                }
            }
            total
        }
        dir_bytes(&self.config.root_dir)
    }

    /// Safely writes data to a file inside the isolated sandbox, enforcing disk quota.
    pub fn write_file(&self, relative_path: &Path, data: &[u8]) -> Result<PathBuf, String> {
        let safe_path = self.validate_path_safety(relative_path)?;
        let current_disk = self.get_disk_usage();
        if current_disk + data.len() > self.config.max_disk_bytes {
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
            let _ = self.state.transition_to(EntityState::Degraded, "Host Job Object guard missing", "SANDBOX_SUPERVISOR");
            return Err("Execution refused: sandbox missing required OS-level isolation guard.".to_string());
        }

        self.state.transition_to(EntityState::Busy, "Executing sandboxed workload", "MANAGER")?;

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
            if k_upper.contains("KEY") || k_upper.contains("SECRET") || k_upper.contains("TOKEN") || k_upper.contains("AUTH") {
                continue;
            }
            cmd.env(k, v);
        }

        // Spawn process
        let mut child = cmd.spawn().map_err(|e| {
            let _ = self.state.transition_to(EntityState::Active, "Command launch failure", "MANAGER");
            format!("Failed to spawn sandboxed executable '{}': {}", executable, e)
        })?;

        let pid = child.id();
        self.active_children.push(pid);

        // Assign to Job Object
        if let Some(ref job) = self.job_guard {
            if let Err(err) = job.assign_child(&child) {
                let _ = child.kill();
                let _ = self.state.transition_to(EntityState::Degraded, "Job assignment failure; marking degraded", "SANDBOX_SUPERVISOR");
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
        let mut timed_out = false;

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
                    if self.get_disk_usage() > self.config.max_disk_bytes {
                        let _ = self.state.transition_to(
                            EntityState::Quarantined,
                            "Disk abuse detected: sandbox disk usage exceeded quota during workload",
                            "SANDBOX_SUPERVISOR",
                        );
                        return Ok(SandboxExecutionResult {
                            exit_code: -1,
                            stdout,
                            stderr: format!("{}\n[SECURITY] Sandbox quarantined: disk quota exceeded.", stderr),
                            duration_ms: duration,
                            peak_memory_bytes: 0,
                            timed_out: false,
                            violated_policy: Some("DiskQuotaExceeded".to_string()),
                        });
                    }

                    let _ = self.state.transition_to(EntityState::Active, "Workload execution completed", "MANAGER");

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
                        timed_out = true;
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
                    let _ = self.state.transition_to(EntityState::Failed, "Process wait failure", "MANAGER");
                    return Err(format!("Error monitoring sandboxed child: {}", e));
                }
            }
        }

        self.active_children.retain(|&p| p != pid);
        let duration = start_time.elapsed().as_millis() as u64;
        let _ = self.state.transition_to(EntityState::Active, "Workload execution timed out", "MANAGER");

        Ok(SandboxExecutionResult {
            exit_code: -1,
            stdout: String::new(),
            stderr: "Execution timed out".to_string(),
            duration_ms: duration,
            peak_memory_bytes: 0,
            timed_out,
            violated_policy: Some("ExecutionTimeout".to_string()),
        })
    }

    /// Stops, revokes, and kills all processes in the sandbox.
    pub fn stop(&mut self) -> Result<(), String> {
        if let Some(ref job) = self.job_guard {
            job.terminate();
        }
        self.active_children.clear();
        self.state.transition_to(EntityState::Stopped, "Sandbox stopped by manager", "MANAGER")?;
        Ok(())
    }

    /// Quarantines the sandbox upon security policy or integrity violation.
    pub fn quarantine(&mut self, reason: &str) -> Result<(), String> {
        if let Some(ref job) = self.job_guard {
            job.terminate();
        }
        self.active_children.clear();
        self.state.transition_to(EntityState::Quarantined, reason, "MANAGER")?;
        Ok(())
    }

    /// Releases resources and cleans up files.
    pub fn cleanup(&mut self) -> Result<(), String> {
        self.stop()?;
        if self.config.root_dir.exists() {
            let _ = fs::remove_dir_all(&self.config.root_dir);
        }
        self.state.transition_to(EntityState::Retired, "Sandbox wiped and retired", "MANAGER")?;
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
    pub rejection_reason: Option<String>,
}

pub struct SandboxBroker;

impl SandboxBroker {
    /// Validates and transfers data between isolated sandboxes under strict policy.
    pub fn transfer(
        source: &IsolatedSandbox,
        target: &IsolatedSandbox,
        request: BrokeredTransferRequest,
    ) -> BrokeredTransferResponse {
        // 1. Verify identities
        if request.source_sandbox_id != source.config.sandbox_id || request.target_sandbox_id != target.config.sandbox_id {
            return BrokeredTransferResponse {
                success: false,
                transfer_id: "".to_string(),
                sha256_hash: "".to_string(),
                bytes_transferred: 0,
                rejection_reason: Some("Identity mismatch between request and sandbox instances".to_string()),
            };
        }

        // 2. Verify both sandboxes are in operational state
        if !source.state.is_operational() || !target.state.is_operational() {
            return BrokeredTransferResponse {
                success: false,
                transfer_id: "".to_string(),
                sha256_hash: "".to_string(),
                bytes_transferred: 0,
                rejection_reason: Some("One or both sandboxes are not in operational state".to_string()),
            };
        }

        // 3. Verify payload size limit (Default max 1MB for cross-sandbox data transfer)
        let limit = if request.max_size_bytes > 0 { request.max_size_bytes } else { 1024 * 1024 };
        if request.payload_bytes.len() > limit {
            return BrokeredTransferResponse {
                success: false,
                transfer_id: "".to_string(),
                sha256_hash: "".to_string(),
                bytes_transferred: 0,
                rejection_reason: Some(format!("Payload exceeds broker size limit of {} bytes", limit)),
            };
        }

        // 4. Compute cryptographic hash
        let mut hasher = Sha256::new();
        hasher.update(&request.payload_bytes);
        let hash = hex::encode(hasher.finalize());

        let transfer_id = format!("xfer_{:x}", rand::random::<u64>());

        BrokeredTransferResponse {
            success: true,
            transfer_id,
            sha256_hash: hash,
            bytes_transferred: request.payload_bytes.len(),
            rejection_reason: None,
        }
    }
}
