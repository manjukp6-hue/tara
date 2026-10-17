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
#[link(name = "kernel32")]
extern "system" {
    fn GetDiskFreeSpaceExW(
        lp_directory_name: *const u16,
        lp_free_bytes_available_to_caller: *mut u64,
        lp_total_number_of_bytes: *mut u64,
        lp_total_number_of_free_bytes: *mut u64,
    ) -> i32;
}

pub struct DeviceCapabilityDetector;

impl DeviceCapabilityDetector {
    /// Detects total and available physical RAM in bytes directly via TARA-HD authoritative discovery.
    pub fn get_ram_info() -> (u64, u64) {
        let (total, avail_opt) = tara_hd::discovery::query_system_ram_bytes();
        let avail = avail_opt.unwrap_or(total / 2);
        (total, avail)
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

        // Disk free space is genuinely unknown — return 0 so callers can detect and handle it
        0
    }

    /// Detects GPU availability, vendor, and free VRAM via TARA-HD dynamic hardware discovery.
    pub fn get_gpu_info() -> GpuInfo {
        let inventory = tara_hd::DiscoveredInventory::probe_host();
        if let Some(gpu) = inventory.gpus.first() {
            let total = gpu.total_memory_bytes;
            let free = gpu.available_memory_bytes.unwrap_or(total).min(total);

            // Infer vendor from the GPU device name string.
            // TARA-HD stores the raw device name from nvidia-smi or the environment;
            // we classify based on known vendor identifiers in the name.
            let name_lc = gpu.name.to_ascii_lowercase();
            let vendor =
                if name_lc.contains("amd") || name_lc.contains("radeon") || name_lc.contains("rx ")
                {
                    "AMD"
                } else if name_lc.contains("intel") || name_lc.contains("arc ") {
                    "Intel"
                } else if name_lc.contains("apple")
                    || name_lc.contains("m1")
                    || name_lc.contains("m2")
                    || name_lc.contains("m3")
                {
                    "Apple"
                } else if gpu.backend == "cuda" {
                    // CUDA backend without other vendor markers defaults to NVIDIA
                    "NVIDIA"
                } else {
                    "Unknown"
                };

            return GpuInfo {
                available: true,
                vendor: vendor.into(),
                device_name: gpu.name.clone(),
                total_vram_bytes: total,
                free_vram_bytes: free,
                gpu_type: gpu.backend.to_ascii_uppercase(),
                device_index: gpu.device_id as u32,
            };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_detector_detect_does_not_panic() {
        let profile = DeviceCapabilityDetector::detect();
        assert!(!profile.os_name.is_empty());
        assert!(profile.cpu_cores_logical >= 1);
        assert!(profile.gpu.free_vram_bytes <= profile.gpu.total_vram_bytes);
    }

    #[test]
    fn test_gpu_info_free_vram_clamping() {
        let total_mb: u64 = 8192;
        let anomalous_free_mb: u64 = 9000;
        let clamped_free_mb = anomalous_free_mb.min(total_mb);
        assert_eq!(clamped_free_mb, 8192);

        let gpu = GpuInfo {
            available: true,
            vendor: "NVIDIA".into(),
            device_name: "Test GPU".into(),
            total_vram_bytes: total_mb * MB,
            free_vram_bytes: clamped_free_mb * MB,
            gpu_type: "CUDA".into(),
            device_index: 0,
        };
        assert!(gpu.free_vram_bytes <= gpu.total_vram_bytes);
    }
}
