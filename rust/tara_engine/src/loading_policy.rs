//! Deterministic Hardware-Capability Policy Engine & Loading Strategy Selector.
//!
//! Selects optimal execution mode for the permanent TARA model:
//! - Strategy: FullLoad, LazyLoad, PartialLoad, QuantizedLoad
//! - Target Device: Gpu, Cpu
//! - Precision: NoneFp32, Fp16, Bf16, Int8, Int4
//! - Strict resource safety limits with deterministic fallback chains.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::device_detector::{DeviceProfile, MB};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelLoadingStrategy {
    FullLoad,
    LazyLoad,
    PartialLoad,
    QuantizedLoad,
}

impl ModelLoadingStrategy {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::FullLoad => "FULL_LOAD",
            Self::LazyLoad => "LAZY_LOAD",
            Self::PartialLoad => "PARTIAL_LOAD",
            Self::QuantizedLoad => "QUANTIZED_LOAD",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceTarget {
    Gpu,
    Cpu,
}

impl DeviceTarget {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Gpu => "GPU",
            Self::Cpu => "CPU",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationPrecision {
    NoneFp32,
    Fp16,
    Bf16,
    Int8,
    Int4,
}

impl QuantizationPrecision {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NoneFp32 => "NONE_FP32",
            Self::Fp16 => "FP16",
            Self::Bf16 => "BF16",
            Self::Int8 => "INT8",
            Self::Int4 => "INT4",
        }
    }

    pub fn bytes_per_param(&self) -> f64 {
        match self {
            Self::NoneFp32 => 4.0,
            Self::Fp16 | Self::Bf16 => 2.0,
            Self::Int8 => 1.0,
            Self::Int4 => 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadingPlan {
    pub strategy: ModelLoadingStrategy,
    pub device_target: DeviceTarget,
    pub precision: QuantizationPrecision,
    pub estimated_memory_bytes: u64,
    pub vram_budget_bytes: u64,
    pub ram_budget_bytes: u64,
    pub max_shard_cache_shards: usize,
    pub enable_mmap: bool,
    pub reason: String,
    pub fallback_chain: Vec<LoadingPlan>,
}

impl LoadingPlan {
    pub fn estimated_memory_mb(&self) -> f64 {
        (self.estimated_memory_bytes as f64 / MB as f64 * 100.0).round() / 100.0
    }

    pub fn summary(&self) -> String {
        format!(
            "LoadingPlan[Strategy={} | Device={} | Precision={} | EstMemory={:.1} MB | CacheShards={} | Reason='{}']",
            self.strategy.as_str(),
            self.device_target.as_str(),
            self.precision.as_str(),
            self.estimated_memory_mb(),
            self.max_shard_cache_shards,
            self.reason
        )
    }
}

#[derive(Debug, Error)]
#[error("Insufficient memory: available resources cannot safely support even the most constrained fallback model profile")]
pub struct InsufficientMemoryError;

pub struct LoadingPolicyEngine;

pub const DEFAULT_MIN_VRAM_BYTES: u64 = 512 * MB;
pub const DEFAULT_VRAM_BUDGET_RATIO: f64 = 0.8;
pub const DEFAULT_RAM_BUDGET_RATIO: f64 = 0.7;
pub const DEFAULT_FP32_OVERHEAD_MULTIPLIER: f64 = 1.25;
pub const DEFAULT_FP16_OVERHEAD_MULTIPLIER: f64 = 1.25;
pub const DEFAULT_INT8_OVERHEAD_MULTIPLIER: f64 = 1.20;
pub const DEFAULT_INT4_OVERHEAD_MULTIPLIER: f64 = 1.15;

pub fn resolve_min_vram_bytes() -> u64 {
    std::env::var("TARA_MIN_VRAM_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MIN_VRAM_BYTES)
}

pub fn resolve_vram_budget_ratio() -> f64 {
    std::env::var("TARA_VRAM_BUDGET_RATIO")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_VRAM_BUDGET_RATIO)
}

pub fn resolve_ram_budget_ratio() -> f64 {
    std::env::var("TARA_RAM_BUDGET_RATIO")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_RAM_BUDGET_RATIO)
}

impl LoadingPolicyEngine {
    pub fn estimate_model_memory(
        param_count: usize,
        precision: QuantizationPrecision,
        overhead_multiplier: f64,
    ) -> u64 {
        let bpp = precision.bytes_per_param();
        let raw = (param_count as f64 * bpp) as u64;
        ((raw as f64 * overhead_multiplier) as u64).max(1)
    }

    pub fn create_loading_plan(
        profile: &DeviceProfile,
        param_count: usize,
        weights_file_bytes: u64,
        requested_strategy: Option<ModelLoadingStrategy>,
    ) -> Result<LoadingPlan, InsufficientMemoryError> {
        let avail_ram = profile.available_ram_bytes;
        let vram_free = profile.gpu.free_vram_bytes;
        let min_vram = resolve_min_vram_bytes();
        let gpu_avail = profile.gpu.available && vram_free > min_vram;

        let vram_budget = (vram_free as f64 * resolve_vram_budget_ratio()) as u64;
        let ram_budget = (avail_ram as f64 * resolve_ram_budget_ratio()) as u64;

        let fp32_mem = Self::estimate_model_memory(
            param_count,
            QuantizationPrecision::NoneFp32,
            DEFAULT_FP32_OVERHEAD_MULTIPLIER,
        );
        let fp16_mem = Self::estimate_model_memory(
            param_count,
            QuantizationPrecision::Fp16,
            DEFAULT_FP16_OVERHEAD_MULTIPLIER,
        );
        let int8_mem = Self::estimate_model_memory(
            param_count,
            QuantizationPrecision::Int8,
            DEFAULT_INT8_OVERHEAD_MULTIPLIER,
        );
        let int4_mem = Self::estimate_model_memory(
            param_count,
            QuantizationPrecision::Int4,
            DEFAULT_INT4_OVERHEAD_MULTIPLIER,
        );

        // Check if explicit strategy requested
        if let Some(strat) = requested_strategy {
            match strat {
                ModelLoadingStrategy::FullLoad => {
                    if gpu_avail && fp32_mem <= vram_budget {
                        return Ok(LoadingPlan {
                            strategy: ModelLoadingStrategy::FullLoad,
                            device_target: DeviceTarget::Gpu,
                            precision: QuantizationPrecision::NoneFp32,
                            estimated_memory_bytes: fp32_mem,
                            vram_budget_bytes: vram_budget,
                            ram_budget_bytes: ram_budget,
                            max_shard_cache_shards: 16,
                            enable_mmap: true,
                            reason: "Explicit full load on GPU".into(),
                            fallback_chain: Vec::new(),
                        });
                    }
                    if fp32_mem <= ram_budget {
                        return Ok(LoadingPlan {
                            strategy: ModelLoadingStrategy::FullLoad,
                            device_target: DeviceTarget::Cpu,
                            precision: QuantizationPrecision::NoneFp32,
                            estimated_memory_bytes: fp32_mem,
                            vram_budget_bytes: 0,
                            ram_budget_bytes: ram_budget,
                            max_shard_cache_shards: 16,
                            enable_mmap: true,
                            reason: "Explicit full load on CPU RAM".into(),
                            fallback_chain: Vec::new(),
                        });
                    }
                }
                ModelLoadingStrategy::LazyLoad => {
                    return Ok(LoadingPlan {
                        strategy: ModelLoadingStrategy::LazyLoad,
                        device_target: DeviceTarget::Cpu,
                        precision: QuantizationPrecision::NoneFp32,
                        estimated_memory_bytes: weights_file_bytes.clamp(32 * MB, 200 * MB),
                        vram_budget_bytes: 0,
                        ram_budget_bytes: ram_budget,
                        max_shard_cache_shards: 4,
                        enable_mmap: true,
                        reason: "Explicit lazy load requested".into(),
                        fallback_chain: Vec::new(),
                    });
                }
                _ => {}
            }
        }

        // Automatic plan calculation:
        // Tier 1: Full Load on GPU FP32/FP16 if VRAM sufficient
        if gpu_avail && fp32_mem <= vram_budget {
            return Ok(LoadingPlan {
                strategy: ModelLoadingStrategy::FullLoad,
                device_target: DeviceTarget::Gpu,
                precision: QuantizationPrecision::NoneFp32,
                estimated_memory_bytes: fp32_mem,
                vram_budget_bytes: vram_budget,
                ram_budget_bytes: ram_budget,
                max_shard_cache_shards: 16,
                enable_mmap: true,
                reason: "Full weights fit comfortably in VRAM (FP32)".into(),
                fallback_chain: Vec::new(),
            });
        }

        if gpu_avail && fp16_mem <= vram_budget {
            return Ok(LoadingPlan {
                strategy: ModelLoadingStrategy::FullLoad,
                device_target: DeviceTarget::Gpu,
                precision: QuantizationPrecision::Fp16,
                estimated_memory_bytes: fp16_mem,
                vram_budget_bytes: vram_budget,
                ram_budget_bytes: ram_budget,
                max_shard_cache_shards: 16,
                enable_mmap: true,
                reason: "Full weights fit in VRAM with FP16 precision".into(),
                fallback_chain: Vec::new(),
            });
        }

        // Tier 2: Full Load on CPU RAM
        if fp32_mem <= ram_budget {
            return Ok(LoadingPlan {
                strategy: ModelLoadingStrategy::FullLoad,
                device_target: DeviceTarget::Cpu,
                precision: QuantizationPrecision::NoneFp32,
                estimated_memory_bytes: fp32_mem,
                vram_budget_bytes: 0,
                ram_budget_bytes: ram_budget,
                max_shard_cache_shards: 8,
                enable_mmap: true,
                reason: "Full weights fit in host RAM (FP32)".into(),
                fallback_chain: Vec::new(),
            });
        }

        // Tier 3: Quantized Load (INT8)
        if int8_mem <= ram_budget {
            return Ok(LoadingPlan {
                strategy: ModelLoadingStrategy::QuantizedLoad,
                device_target: DeviceTarget::Cpu,
                precision: QuantizationPrecision::Int8,
                estimated_memory_bytes: int8_mem,
                vram_budget_bytes: 0,
                ram_budget_bytes: ram_budget,
                max_shard_cache_shards: 4,
                enable_mmap: true,
                reason: "INT8 quantization selected to fit comfortably in RAM".into(),
                fallback_chain: Vec::new(),
            });
        }

        // Tier 4: Lazy Paged Shards
        let lazy_mem = (100 * MB * 2).min(ram_budget);
        if lazy_mem > 32 * MB {
            return Ok(LoadingPlan {
                strategy: ModelLoadingStrategy::LazyLoad,
                device_target: DeviceTarget::Cpu,
                precision: QuantizationPrecision::NoneFp32,
                estimated_memory_bytes: lazy_mem,
                vram_budget_bytes: 0,
                ram_budget_bytes: ram_budget,
                max_shard_cache_shards: 2,
                enable_mmap: true,
                reason: "Constrained RAM: Paging 100MB SafeTensors shards via LRU cache".into(),
                fallback_chain: Vec::new(),
            });
        }

        // Tier 5: Extreme fallback - INT4
        if int4_mem <= ram_budget {
            return Ok(LoadingPlan {
                strategy: ModelLoadingStrategy::QuantizedLoad,
                device_target: DeviceTarget::Cpu,
                precision: QuantizationPrecision::Int4,
                estimated_memory_bytes: int4_mem,
                vram_budget_bytes: 0,
                ram_budget_bytes: ram_budget,
                max_shard_cache_shards: 1,
                enable_mmap: true,
                reason: "Ultra-constrained RAM: INT4 quantization".into(),
                fallback_chain: Vec::new(),
            });
        }

        Err(InsufficientMemoryError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_detector::GpuInfo;

    #[test]
    fn test_estimate_model_memory_precisions() {
        let params = 1_000_000; // 1M params
        let fp32 = LoadingPolicyEngine::estimate_model_memory(
            params,
            QuantizationPrecision::NoneFp32,
            1.0,
        );
        assert_eq!(fp32, 4_000_000);

        let fp16 = LoadingPolicyEngine::estimate_model_memory(
            params,
            QuantizationPrecision::Fp16,
            1.0,
        );
        assert_eq!(fp16, 2_000_000);

        let int8 = LoadingPolicyEngine::estimate_model_memory(
            params,
            QuantizationPrecision::Int8,
            1.0,
        );
        assert_eq!(int8, 1_000_000);

        let int4 = LoadingPolicyEngine::estimate_model_memory(
            params,
            QuantizationPrecision::Int4,
            1.0,
        );
        assert_eq!(int4, 500_000);
    }

    #[test]
    fn test_create_loading_plan_cpu_full_load() {
        let profile = DeviceProfile {
            os_name: "windows".into(),
            os_version: "10".into(),
            cpu_arch: "x86_64".into(),
            cpu_cores_logical: 8,
            cpu_cores_physical: 4,
            total_ram_bytes: 16 * 1024 * MB,
            available_ram_bytes: 8 * 1024 * MB,
            available_storage_bytes: 50 * 1024 * MB,
            gpu: GpuInfo::default(),
            environment_type: "DESKTOP".into(),
            is_mobile: false,
            is_server: false,
            is_desktop: true,
        };

        let plan = LoadingPolicyEngine::create_loading_plan(
            &profile,
            10_000_000, // 10M params
            40 * MB,
            None,
        )
        .unwrap();

        assert_eq!(plan.device_target, DeviceTarget::Cpu);
        assert_eq!(plan.strategy, ModelLoadingStrategy::FullLoad);
        assert_eq!(plan.precision, QuantizationPrecision::NoneFp32);
    }

    #[test]
    fn test_create_loading_plan_insufficient_memory() {
        let profile = DeviceProfile {
            os_name: "windows".into(),
            os_version: "10".into(),
            cpu_arch: "x86_64".into(),
            cpu_cores_logical: 1,
            cpu_cores_physical: 1,
            total_ram_bytes: 10 * MB,
            available_ram_bytes: 5 * MB,
            available_storage_bytes: MB,
            gpu: GpuInfo::default(),
            environment_type: "DESKTOP".into(),
            is_mobile: false,
            is_server: false,
            is_desktop: true,
        };

        // Huge model cannot fit in 5MB RAM
        let res = LoadingPolicyEngine::create_loading_plan(
            &profile,
            1_000_000_000, // 1B params
            4_000 * MB,
            None,
        );
        assert!(res.is_err());
    }
}

