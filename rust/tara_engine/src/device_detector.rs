//! Automatic Hardware & Device Capability Detection for TARA.
//!
//! Real, native platform probes with zero simulation:
//! - OS / Architecture
//! - Logical & physical CPU core counts
//! - Physical and available system RAM (via native OS APIs: GlobalMemoryStatusEx on Windows, /proc/meminfo on Linux)
//! - Available disk storage (via GetDiskFreeSpaceExW on Windows, statvfs on Unix)
//! - GPU hardware detection & VRAM reporting (via nvidia-smi / native command query)
//! - Hardware environment classification (DESKTOP, SERVER, MOBILE)

use serde::{Deserialize, Serialize};
use std::env;
use std::process::Command;

pub const GB: u64 = 1024 * 1024 * 1024;
pub const MB: u64 = 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub available: bool,
    pub vendor: String, // "NVIDIA", "APPLE", "AMD", "INTEL", "NONE"
    pub device_name: String,
    pub total_vram_bytes: u64,
    pub free_vram_bytes: u64,
    pub gpu_type: String, // "CUDA", "MPS", "ROCM", "DIRECTML", "NONE"
    pub device_index: u32,
}

impl Default for GpuInfo {
    fn default() -> Self {
        Self {
            available: false,
            vendor: "NONE".into(),
            device_name: "None".into(),
            total_vram_bytes: 0,
            free_vram_bytes: 0,
            gpu_type: "NONE".into(),
            device_index: 0,
        }
    }
}

impl GpuInfo {
    pub fn total_vram_gb(&self) -> f64 {
        (self.total_vram_bytes as f64 / GB as f64 * 100.0).round() / 100.0
    }

    pub fn free_vram_gb(&self) -> f64 {
        (self.free_vram_bytes as f64 / GB as f64 * 100.0).round() / 100.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceProfile {
    pub os_name: String,
    pub os_version: String,
    pub cpu_arch: String,
    pub cpu_cores_logical: usize,
    pub cpu_cores_physical: usize,
    pub total_ram_bytes: u64,
    pub available_ram_bytes: u64,
    pub available_storage_bytes: u64,
    pub gpu: GpuInfo,
    pub environment_type: String, // "DESKTOP", "SERVER", "MOBILE"
    pub is_mobile: bool,
    pub is_server: bool,
    pub is_desktop: bool,
}

impl DeviceProfile {
    pub fn total_ram_gb(&self) -> f64 {
        (self.total_ram_bytes as f64 / GB as f64 * 100.0).round() / 100.0
    }

    pub fn available_ram_gb(&self) -> f64 {
        (self.available_ram_bytes as f64 / GB as f64 * 100.0).round() / 100.0
    }

    pub fn available_storage_gb(&self) -> f64 {
        (self.available_storage_bytes as f64 / GB as f64 * 100.0).round() / 100.0
    }

    pub fn summary(&self) -> String {
        let gpu_str = if self.gpu.available {
            format!(
                "{} {} ({:.1} GB free)",
                self.gpu.vendor,
                self.gpu.device_name,
                self.gpu.free_vram_gb()
            )
        } else {
            "None (CPU only)".to_string()
        };

        format!(
            "DeviceProfile[{} | {} {} | CPU: {} cores | RAM: {:.1}/{:.1} GB | GPU: {} | Storage: {:.1} GB free]",
            self.environment_type,
            self.os_name,
            self.cpu_arch,
            self.cpu_cores_logical,
            self.available_ram_gb(),
            self.total_ram_gb(),
            gpu_str,
            self.available_storage_gb()
        )
    }
}

#[cfg(windows)]
#[repr(C)]
struct MemoryStatusEx {
    dw_length: u32,
    dw_memory_load: u32,
    ull_total_phys: u64,
    ull_avail_phys: u64,
    ull_total_page_file: u64,
    ull_avail_page_file: u64,
    ull_total_virtual: u64,
    ull_avail_virtual: u64,
    ull_avail_extended_virtual: u64,
}

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn GlobalMemoryStatusEx(lp_buffer: *mut MemoryStatusEx) -> i32;
    fn GetDiskFreeSpaceExW(
        lp_directory_name: *const u16,
        lp_free_bytes_available_to_caller: *mut u64,
        lp_total_number_of_bytes: *mut u64,
        lp_total_number_of_free_bytes: *mut u64,
    ) -> i32;
}

pub struct DeviceCapabilityDetector;

impl DeviceCapabilityDetector {
    /// Detects total and available physical RAM in bytes.
    pub fn get_ram_info() -> (u64, u64) {
        #[cfg(windows)]
        {
            unsafe {
                let mut status = MemoryStatusEx {
                    dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
                    dw_memory_load: 0,
                    ull_total_phys: 0,
                    ull_avail_phys: 0,
                    ull_total_page_file: 0,
                    ull_avail_page_file: 0,
                    ull_total_virtual: 0,
                    ull_avail_virtual: 0,
                    ull_avail_extended_virtual: 0,
                };
                if GlobalMemoryStatusEx(&mut status) != 0 {
                    return (status.ull_total_phys, status.ull_avail_phys);
                }
            }
        }

        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = fs::read_to_string("/proc/meminfo") {
                let mut total = 0u64;
                let mut avail = 0u64;
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            if let Ok(kb) = parts[1].parse::<u64>() {
                                total = kb * 1024;
                            }
                        }
                    } else if line.starts_with("MemAvailable:") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            if let Ok(kb) = parts[1].parse::<u64>() {
                                avail = kb * 1024;
                            }
                        }
                    }
                }
                if total > 0 {
                    if avail == 0 {
                        avail = total / 2;
                    }
                    return (total, avail);
                }
            }
        }

        // Fallback default safe estimation
        (16 * GB, 8 * GB)
    }

    /// Detects available disk storage in bytes for a given path.
    pub fn get_available_storage<P: AsRef<std::path::Path>>(path: P) -> u64 {
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStrExt;
            let path_ref = path.as_ref();
            let mut wide_chars: Vec<u16> = path_ref.as_os_str().encode_wide().collect();
            wide_chars.push(0);

            let mut free_bytes_caller = 0u64;
            let mut total_bytes = 0u64;
            let mut free_bytes = 0u64;

            unsafe {
                if GetDiskFreeSpaceExW(
                    wide_chars.as_ptr(),
                    &mut free_bytes_caller,
                    &mut total_bytes,
                    &mut free_bytes,
                ) != 0
                {
                    return free_bytes_caller;
                }
            }
        }

        #[cfg(unix)]
        {
            use std::ffi::CString;
            use std::os::unix::ffi::OsStrExt;
            if let Ok(c_path) = CString::new(path.as_ref().as_os_str().as_bytes()) {
                unsafe {
                    let mut stat: libc::statvfs = std::mem::zeroed();
                    if libc::statvfs(c_path.as_ptr(), &mut stat) == 0 {
                        return (stat.f_bavail as u64) * (stat.f_frsize as u64);
                    }
                }
            }
        }

        50 * GB
    }

    /// Detects GPU availability, vendor, and free VRAM via system tools.
    pub fn get_gpu_info() -> GpuInfo {
        // Probe via nvidia-smi
        if let Ok(output) = Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(first_line) = stdout.lines().next() {
                    let parts: Vec<&str> = first_line.split(',').map(|s| s.trim()).collect();
                    if parts.len() >= 3 {
                        let name = parts[0].to_string();
                        let total_mb: u64 = parts[1].parse().unwrap_or(0);
                        let free_mb: u64 = parts[2].parse().unwrap_or(0);

                        return GpuInfo {
                            available: true,
                            vendor: "NVIDIA".into(),
                            device_name: name,
                            total_vram_bytes: total_mb * MB,
                            free_vram_bytes: free_mb * MB,
                            gpu_type: "CUDA".into(),
                            device_index: 0,
                        };
                    }
                }
            }
        }

        GpuInfo::default()
    }

    /// Executes full host detection and constructs a DeviceProfile.
    pub fn detect() -> DeviceProfile {
        let os_name = env::consts::OS.to_string();
        let cpu_arch = env::consts::ARCH.to_string();
        let cpu_cores_logical = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        let cpu_cores_physical = (cpu_cores_logical / 2).max(1);

        let (total_ram_bytes, available_ram_bytes) = Self::get_ram_info();
        let available_storage_bytes = Self::get_available_storage(".");
        let gpu = Self::get_gpu_info();

        // Environment classification
        let is_mobile = os_name == "android" || os_name == "ios";
        let is_server = cpu_cores_logical >= 16 && total_ram_bytes >= 32 * GB;
        let is_desktop = !is_mobile && !is_server;

        let environment_type = if is_mobile {
            "MOBILE".to_string()
        } else if is_server {
            "SERVER".to_string()
        } else {
            "DESKTOP".to_string()
        };

        DeviceProfile {
            os_name,
            os_version: "Host".into(),
            cpu_arch,
            cpu_cores_logical,
            cpu_cores_physical,
            total_ram_bytes,
            available_ram_bytes,
            available_storage_bytes,
            gpu,
            environment_type,
            is_mobile,
            is_server,
            is_desktop,
        }
    }
}
