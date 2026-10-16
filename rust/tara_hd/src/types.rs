//! Core types, capabilities, error handling, and device enumerations for TARA-HD.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Unified error type for TARA-HD operations.
#[derive(Debug, Clone, PartialEq, Eq, Error, Serialize, Deserialize)]
pub enum TaraError {
    #[error("Requested device {kind:?} (ID: {device_id:?}) is unavailable on this host")]
    DeviceUnavailable {
        kind: DeviceKind,
        device_id: Option<usize>,
    },
    #[error("Device synchronization failed on {device}: {reason}")]
    SyncError { device: String, reason: String },
    #[error("Invalid selection policy string: '{0}'")]
    InvalidPolicy(String),
}

impl From<TaraError> for String {
    fn from(err: TaraError) -> Self {
        err.to_string()
    }
}

/// High-level device categories supported by TARA-HD.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceKind {
    /// Accelerated GPU hardware (NVIDIA CUDA, AMD ROCm, Apple Metal).
    Gpu,
    /// Tensor Processing Unit (Google Cloud TPU / XLA).
    Tpu,
    /// Host Central Processing Unit.
    Cpu,
    /// Extensible custom accelerator (e.g. specialized NPU or FPGA).
    Custom(String),
}

impl std::fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gpu => write!(f, "GPU"),
            Self::Tpu => write!(f, "TPU"),
            Self::Cpu => write!(f, "CPU"),
            Self::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// Precision formats supported across compute backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrecisionMode {
    /// Automatically select optimal accelerator precision (BF16 > FP16 > FP32).
    Auto,
    /// Single precision 32-bit IEEE 754 float.
    Fp32,
    /// Half precision 16-bit float.
    Fp16,
    /// Brain floating point 16-bit.
    Bf16,
}

impl std::fmt::Display for PrecisionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Fp32 => write!(f, "fp32"),
            Self::Fp16 => write!(f, "fp16"),
            Self::Bf16 => write!(f, "bf16"),
        }
    }
}

impl PrecisionMode {
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Self::Auto,
            "fp32" | "float32" | "f32" => Self::Fp32,
            "fp16" | "float16" | "f16" => Self::Fp16,
            "bf16" | "bfloat16" => Self::Bf16,
            other => {
                log::warn!("Unknown precision mode '{}', defaulting to Auto.", other);
                Self::Auto
            }
        }
    }
}

/// Dynamic capabilities and specifications of a detected compute device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceCapability {
    /// Device category (GPU, TPU, CPU, Custom).
    pub kind: DeviceKind,
    /// Logical device ordinal index on the host.
    pub device_id: usize,
    /// Human-readable brand and model name.
    pub name: String,
    /// Underlying backend driver (e.g., "cuda", "tpu_xla", "cpu_rayon").
    pub backend: String,
    /// Total device physical memory in bytes.
    pub total_memory_bytes: u64,
    /// Currently measurable free/available device memory in bytes.
    pub available_memory_bytes: Option<u64>,
    /// Precision formats natively executable on this device.
    pub supported_precisions: Vec<PrecisionMode>,
    /// Whether device supports accelerated matrix multiplication kernels.
    pub supports_matrix_multiply: bool,
    /// Whether device supports fused multi-head attention.
    pub supports_fused_attention: bool,
    /// Whether device supports in-place gradient and optimizer updates.
    pub supports_in_place_optimizer: bool,
}

/// User device selection policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionPolicy {
    /// Automatically discover and select highest priority accelerator (GPU > TPU > Custom > CPU).
    Auto,
    /// Explicit device selection without fallback.
    Explicit(DeviceKind, Option<usize>),
    /// Explicit device selection with fallback to CPU allowed on failure.
    ExplicitWithFallback(DeviceKind, Option<usize>),
}

impl SelectionPolicy {
    /// Parse a selection policy string.
    ///
    /// Supported formats:
    /// - `"auto"` -> `SelectionPolicy::Auto`
    /// - `"gpu"`, `"cuda"`, `"tpu"` -> `ExplicitWithFallback(kind, None)`
    /// - `"gpu:0"`, `"tpu:1"` -> `ExplicitWithFallback(kind, Some(id))`
    /// - `"strict:gpu:0"` or `"gpu:0!strict"` -> `Explicit(kind, Some(id))`
    /// - `"cpu"` or `"cpu:0"` -> `Explicit(DeviceKind::Cpu, id)`
    pub fn parse(s: &str) -> Self {
        let trimmed = s.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
            return Self::Auto;
        }

        // Check for strict prefix or suffix
        let (core_str, is_strict) = if let Some(rest) = trimmed.strip_prefix("strict:") {
            (rest.trim(), true)
        } else if let Some(rest) = trimmed.strip_suffix("!strict") {
            (rest.trim(), true)
        } else {
            (trimmed, false)
        };

        let parse_kind = |k: &str| -> DeviceKind {
            match k.to_ascii_lowercase().as_str() {
                "gpu" | "cuda" => DeviceKind::Gpu,
                "tpu" => DeviceKind::Tpu,
                "cpu" => DeviceKind::Cpu,
                custom => DeviceKind::Custom(custom.to_string()),
            }
        };

        let (kind, target_id) = if let Some((kind_str, id_str)) = core_str.split_once(':') {
            let id = match id_str.trim().parse::<usize>() {
                Ok(val) => Some(val),
                Err(_) => {
                    log::warn!(
                        "Invalid device ordinal '{}' in policy '{}'; ignoring ordinal.",
                        id_str,
                        s
                    );
                    None
                }
            };
            (parse_kind(kind_str.trim()), id)
        } else {
            (parse_kind(core_str), None)
        };

        if kind == DeviceKind::Cpu || is_strict {
            Self::Explicit(kind, target_id)
        } else {
            Self::ExplicitWithFallback(kind, target_id)
        }
    }
}

impl Default for SelectionPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

impl std::str::FromStr for SelectionPolicy {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Err("Device selection string cannot be empty".to_string());
        }
        let core = trimmed
            .strip_prefix("strict:")
            .unwrap_or(trimmed)
            .strip_suffix("!strict")
            .unwrap_or(trimmed);
        let base_kind = core.split(':').next().unwrap_or(core).trim().to_lowercase();
        match base_kind.as_str() {
            "auto" | "cpu" | "gpu" | "cuda" | "tpu" => Ok(SelectionPolicy::parse(trimmed)),
            "custom" => {
                let name = core.split_once(':').map(|(_, r)| r.trim()).unwrap_or("");
                if name.is_empty() {
                    Err("Custom device name cannot be empty".to_string())
                } else {
                    Ok(SelectionPolicy::parse(trimmed))
                }
            }
            other => Err(format!(
                "Unknown device '{other}'. Valid options: auto, cpu, gpu, cuda, tpu, custom:<name>"
            )),
        }
    }
}
