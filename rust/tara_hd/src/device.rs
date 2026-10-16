//! Unified device trait and concrete device implementations for TARA-HD.

use std::any::Any;
use std::fmt::Debug;

use crate::streaming::{CpuStreamHelper, StreamConfig};
use crate::types::{DeviceCapability, DeviceKind};

/// Unified interface representing an abstract compute device.
pub trait ComputeDevice: Send + Sync + Debug {
    /// Return static and dynamic capabilities of this device.
    fn info(&self) -> &DeviceCapability;

    /// Returns whether the device is currently healthy and operational.
    fn is_available(&self) -> bool;

    /// Wait for all pending asynchronous compute operations on this device to complete.
    fn synchronize(&self) -> Result<(), String>;

    /// Check if the device natively supports a specific named compute operation.
    fn supports_operation(&self, op_name: &str) -> bool;

    /// Downcast support for concrete device inspection.
    fn as_any(&self) -> &dyn Any;

    /// Optional reference to the CPU on-demand streaming helper (returns `Some` on `CpuDevice`).
    fn stream_helper(&self) -> Option<&CpuStreamHelper> {
        None
    }

    /// Returns human-readable name.
    fn name(&self) -> &str {
        &self.info().name
    }

    /// Returns high-level device category.
    fn kind(&self) -> &DeviceKind {
        &self.info().kind
    }
}

/// Accelerated NVIDIA CUDA GPU device implementation.
#[derive(Debug)]
pub struct CudaDevice {
    info: DeviceCapability,
}

impl CudaDevice {
    pub fn new(info: DeviceCapability) -> Self {
        Self { info }
    }
}

impl ComputeDevice for CudaDevice {
    fn info(&self) -> &DeviceCapability {
        &self.info
    }

    fn is_available(&self) -> bool {
        true
    }

    fn synchronize(&self) -> Result<(), String> {
        Ok(())
    }

    fn supports_operation(&self, op_name: &str) -> bool {
        matches!(
            op_name.trim().to_ascii_lowercase().as_str(),
            "matmul" | "attention" | "rope" | "rmsnorm" | "cross_entropy" | "adamw" | "gemm"
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Google Cloud TPU accelerator device implementation.
#[derive(Debug)]
pub struct TpuDevice {
    info: DeviceCapability,
}

impl TpuDevice {
    pub fn new(info: DeviceCapability) -> Self {
        Self { info }
    }
}

impl ComputeDevice for TpuDevice {
    fn info(&self) -> &DeviceCapability {
        &self.info
    }

    fn is_available(&self) -> bool {
        true
    }

    fn synchronize(&self) -> Result<(), String> {
        Ok(())
    }

    fn supports_operation(&self, op_name: &str) -> bool {
        matches!(
            op_name.trim().to_ascii_lowercase().as_str(),
            "matmul"
                | "attention"
                | "rope"
                | "rmsnorm"
                | "cross_entropy"
                | "adamw"
                | "gemm"
                | "xla_hlo"
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Host CPU compute device equipped with on-demand chunked streaming.
#[derive(Debug)]
pub struct CpuDevice {
    info: DeviceCapability,
    stream_helper: CpuStreamHelper,
}

impl CpuDevice {
    pub fn new(info: DeviceCapability) -> Self {
        // Dynamically size memory budget to 50% of available RAM (or 25% of total RAM), clamped to [512 MB, 16384 MB]
        let usable_bytes = info
            .available_memory_bytes
            .map(|avail| avail / 2)
            .unwrap_or(info.total_memory_bytes / 4);
        let budget_mb = ((usable_bytes / (1024 * 1024)) as usize).clamp(512, 16384);

        let config = StreamConfig {
            chunk_size_tokens: 256,
            max_resident_layers: 2,
            memory_budget_mb: budget_mb,
        };

        Self {
            info,
            stream_helper: CpuStreamHelper::with_config(config),
        }
    }

    /// Create a CPU device with an explicit streaming configuration.
    pub fn with_stream_config(info: DeviceCapability, config: StreamConfig) -> Self {
        Self {
            info,
            stream_helper: CpuStreamHelper::with_config(config),
        }
    }

    /// Reference to the on-demand CPU streaming helper.
    pub fn stream_helper(&self) -> &CpuStreamHelper {
        &self.stream_helper
    }
}

impl ComputeDevice for CpuDevice {
    fn info(&self) -> &DeviceCapability {
        &self.info
    }

    fn is_available(&self) -> bool {
        true
    }

    fn synchronize(&self) -> Result<(), String> {
        Ok(())
    }

    fn supports_operation(&self, _op_name: &str) -> bool {
        true
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn stream_helper(&self) -> Option<&CpuStreamHelper> {
        Some(&self.stream_helper)
    }
}

/// Pluggable adapter for future accelerators (Metal, ROCm, custom NPUs).
#[derive(Debug)]
pub struct CustomDevice {
    info: DeviceCapability,
    supported_ops: Vec<String>,
}

impl CustomDevice {
    pub fn new(info: DeviceCapability, supported_ops: Vec<String>) -> Self {
        Self {
            info,
            supported_ops,
        }
    }

    pub fn supported_operations(&self) -> &[String] {
        &self.supported_ops
    }
}

impl ComputeDevice for CustomDevice {
    fn info(&self) -> &DeviceCapability {
        &self.info
    }

    fn is_available(&self) -> bool {
        true
    }

    fn synchronize(&self) -> Result<(), String> {
        Ok(())
    }

    fn supports_operation(&self, op_name: &str) -> bool {
        let trimmed = op_name.trim();
        self.supported_ops
            .iter()
            .any(|op| op.eq_ignore_ascii_case(trimmed))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
