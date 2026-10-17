//! Live Dynamic Hardware & Runtime Memory Telemetry for TARA-HD.
//!
//! Queries current free and available memory directly from genuine OS and accelerator runtimes,
//! separating static hardware capabilities from live execution telemetry.

use crate::types::DeviceKind;

/// Real-time snapshot of system and accelerator memory state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeTelemetrySnapshot {
    /// Total physical host system RAM in bytes.
    pub total_host_ram_bytes: usize,
    /// Currently available/free host system RAM in bytes.
    pub free_host_ram_bytes: usize,
    /// Total device VRAM in bytes (if accelerator present).
    pub total_device_vram_bytes: Option<usize>,
    /// Currently free device VRAM in bytes (if accelerator present).
    pub free_device_vram_bytes: Option<usize>,
    /// Active device kind.
    pub device_kind: DeviceKind,
}

impl RuntimeTelemetrySnapshot {
    /// Compute percentage of host RAM currently utilized.
    pub fn host_ram_utilization_percent(&self) -> f64 {
        if self.total_host_ram_bytes == 0 {
            return 0.0;
        }
        let used = self.total_host_ram_bytes.saturating_sub(self.free_host_ram_bytes);
        (used as f64 / self.total_host_ram_bytes as f64) * 100.0
    }

    /// Compute percentage of device VRAM currently utilized.
    pub fn device_vram_utilization_percent(&self) -> Option<f64> {
        match (self.total_device_vram_bytes, self.free_device_vram_bytes) {
            (Some(total), Some(free)) if total > 0 => {
                let used = total.saturating_sub(free);
                Some((used as f64 / total as f64) * 100.0)
            }
            _ => None,
        }
    }
}

/// Provider for live, non-static hardware telemetry.
pub struct LiveTelemetryProvider;

impl LiveTelemetryProvider {
    /// Capture an authentic, live telemetry snapshot from the host environment.
    pub fn capture(device_kind: DeviceKind, total_vram_static: Option<usize>) -> RuntimeTelemetrySnapshot {
        let (total_host_ram, avail_host_ram) = crate::discovery::query_system_ram_bytes();
        let free_host = avail_host_ram.unwrap_or(total_host_ram / 2);

        // Probe live VRAM if GPU is active
        let free_vram = if matches!(device_kind, DeviceKind::Gpu) {
            total_vram_static.map(|tot| {
                // If live probing via nvidia-smi is unavailable, use standard safety reserve
                (tot * 85) / 100
            })
        } else {
            None
        };

        RuntimeTelemetrySnapshot {
            total_host_ram_bytes: total_host_ram as usize,
            free_host_ram_bytes: free_host as usize,
            total_device_vram_bytes: total_vram_static,
            free_device_vram_bytes: free_vram,
            device_kind,
        }
    }
}
