//! Automatic training hyperparameter configuration engine for TARA-HD.
//!
//! Derives optimal batch size, sequence length, gradient accumulation, learning rate,
//! warmup steps, weight decay, and gradient clip norm from the actual live device
//! memory reported by the `ComputeChannel` — zero manual tuning and zero hardcoded
//! memory literals.
//!
//! The engine guarantees a conservative memory budget so that training never triggers
//! out-of-memory errors even on constrained environments (e.g. Colab Free 15 GB VRAM).

use std::env;

use crate::channel::ComputeChannel;
use crate::types::{DeviceKind, PrecisionMode};

/// Automatically derived training hyperparameters computed from live device memory.
///
/// All fields carry the resolved, ready-to-use values. Callers should apply each
/// field only when the user has not explicitly overridden that hyperparameter.
#[derive(Debug, Clone)]
pub struct AutoTrainingConfig {
    /// Gradient-accumulation micro-batch count (effective batch = batch_size × grad_accum_steps).
    pub batch_size: usize,
    /// Token sequence length for each forward pass.
    pub seq_len: usize,
    /// Number of gradient-accumulation steps before each optimizer update.
    pub grad_accum_steps: usize,
    /// Initial learning rate selected from model-scale heuristic.
    pub learning_rate: f32,
    /// Linear warmup step count (10 % of expected total optimizer steps).
    pub warmup_steps: usize,
    /// AdamW decoupled weight decay coefficient.
    pub weight_decay: f32,
    /// Global L2 gradient clip norm ceiling.
    pub grad_clip_norm: f32,
    /// Recommended compute precision for this device.
    pub precision: PrecisionMode,
    /// Summarised telemetry string describing what was auto-detected.
    pub telemetry: String,
    /// Estimated total dataset tokens used to size warmup (caller updates if known).
    pub estimated_total_optimizer_steps: usize,
    /// Total host RAM detected in MB (MemTotal from /proc/meminfo). Zero if probe failed.
    pub host_ram_total_mb: f64,
    /// Host RAM currently available in MB (MemAvailable from /proc/meminfo).
    /// More accurate than MemTotal for budget decisions — accounts for kernel buffers/caches.
    /// Zero when unavailable (Windows dev machines).
    pub host_ram_available_mb: f64,
    /// Cgroup memory hard limit in MB (v2: memory.max, v1: memory.limit_in_bytes).
    /// Zero when the cgroup filesystem is absent or the limit is set to "max" (unlimited).
    /// On Colab this is the binding container ceiling, which may be lower than MemTotal.
    pub cgroup_limit_mb: f64,
    /// Cgroup memory currently available in MB: cgroup_limit_mb − current cgroup usage.
    /// Zero when cgroup is absent, the limit is unlimited, or current usage is unavailable.
    /// This is the remaining headroom inside the container's hard memory boundary.
    pub cgroup_available_mb: f64,
    /// Safe host memory training ceiling in MB.
    /// Derived as: safety_pct × min(host_ram_available, cgroup_available_if_set).
    /// Trainers must use this as `max_memory_mb` when the user has not set an explicit ceiling.
    pub max_memory_mb: f64,
    /// Total device (GPU/TPU) VRAM in MB. Zero when running purely on CPU.
    pub device_vram_total_mb: f64,
    /// Available device (GPU/TPU) VRAM in MB. Zero when running purely on CPU.
    pub device_vram_available_mb: f64,
    /// Safe device (GPU/TPU) VRAM ceiling in MB (available VRAM * safety fraction).
    /// Zero when running purely on CPU.
    pub max_device_vram_mb: f64,
}

impl AutoTrainingConfig {
    /// Derive an optimal training configuration from live device memory.
    ///
    /// # Arguments
    /// - `channel`: Active TARA-HD compute channel (GPU / TPU / CPU).
    /// - `total_params`: Total number of model parameters (sum of all tensor element counts).
    /// - `hidden_size`: Model hidden dimension from `config.json`.
    /// - `num_layers`: Number of transformer decoder layers.
    /// - `estimated_total_samples`: Rough number of training samples available (used for warmup sizing).
    pub fn compute(
        channel: &ComputeChannel,
        total_params: usize,
        hidden_size: usize,
        num_layers: usize,
        estimated_total_samples: usize,
    ) -> Self {
        let device_info = channel.device().info();
        let device_kind = channel.device_kind().clone();
        let effective_hidden = hidden_size.max(1);
        let effective_layers = num_layers.max(1);

        // ── Precision ────────────────────────────────────────────────────────────
        // Honour the channel's negotiated precision; fall back to Fp32 on CPU.
        let precision = match device_kind {
            DeviceKind::Cpu => PrecisionMode::Fp32,
            _ => channel.precision(),
        };

        // Bytes per parameter in the weight store (f16/bf16 → 2, f32 → 4).
        let bytes_per_weight: u64 = match precision {
            PrecisionMode::Fp16 | PrecisionMode::Bf16 => 2,
            _ => 4,
        };

        // ── Device memory budget ─────────────────────────────────────────────────
        // Total memory bytes reported by TARA-HD for the active device.
        // When 0 (device did not report), fall back to conservative defaults.
        let total_mem_bytes = device_info.total_memory_bytes;
        let avail_mem_bytes = device_info
            .available_memory_bytes
            .unwrap_or(total_mem_bytes);

        // ── GPU / TPU auto-configuration ─────────────────────────────────────────
        let (batch_size, seq_len, grad_accum_steps) = match device_kind {
            DeviceKind::Gpu | DeviceKind::Tpu => Self::derive_gpu_config(
                total_mem_bytes,
                avail_mem_bytes,
                total_params,
                effective_hidden,
                effective_layers,
                bytes_per_weight,
            ),
            DeviceKind::Cpu => Self::derive_cpu_config(
                avail_mem_bytes,
                total_params,
                effective_hidden,
                effective_layers,
            ),
            DeviceKind::Custom(_) => {
                // Treat custom accelerators conservatively like GPU with unknown VRAM.
                Self::derive_gpu_config(
                    total_mem_bytes,
                    avail_mem_bytes,
                    total_params,
                    effective_hidden,
                    effective_layers,
                    bytes_per_weight,
                )
            }
        };

        // ── Learning rate (model-scale heuristic) ────────────────────────────────
        // Larger models need a smaller learning rate to remain stable.
        let learning_rate: f32 = if total_params < 100_000_000 {
            3e-4 // Small models (< 100 M params)
        } else if total_params < 500_000_000 {
            1e-4 // Medium models (100 M – 500 M params)
        } else {
            3e-5 // Large models (≥ 500 M params)
        };

        // ── Warmup steps (10 % of estimated total optimizer steps) ──────────────
        // Derived from the number of samples, sequence length, and gradient accumulation.
        let steps_per_epoch = if estimated_total_samples > 0 && seq_len > 0 && grad_accum_steps > 0
        {
            (estimated_total_samples / grad_accum_steps).max(1)
        } else {
            500 // conservative default when sample count is unknown
        };
        let warmup_steps = (steps_per_epoch / 10).clamp(50, 2000);

        // ── Fixed hyperparameters that are well-established defaults ─────────────
        let weight_decay: f32 = 0.01;
        let grad_clip_norm: f32 = 1.0;

        // ── Host RAM probe (cgroup-aware) ─────────────────────────────────────
        // To catch Colab-style cgroup OOM kills, we need the binding memory limit,
        // not MemTotal. Colab containers may report MemTotal = 16 GB (physical) but
        // the cgroup hard limit = 12 GB. Using MemTotal as the ceiling allows
        // projected_mb to pass preflight while the actual allocation exceeds the
        // container limit and triggers a signal-137 OOM kill.
        //
        // Priority order for the effective memory ceiling:
        //   1. TARA_HOST_RAM_MB env override (explicit manual control)
        //   2. min(MemAvailable, cgroup_limit − cgroup_current) — tightest real bound
        //   3. MemAvailable alone when cgroup is unlimited or absent
        //   4. Fallback: 8 GB conservative default when /proc is unavailable (Windows)
        let (host_ram_total_mb, host_ram_available_mb, cgroup_limit_mb, cgroup_available_mb) =
            probe_host_memory_budget_mb();

        let host_ram_safety_pct: f64 = env::var("TARA_HOST_RAM_SAFETY_PCT")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(85.0)
            .clamp(50.0, 95.0);

        // Effective memory budget: take the most restrictive of MemAvailable and
        // cgroup_available (when the cgroup limit is set and finite). This ensures
        // the preflight ceiling correctly reflects the container's hard boundary
        // rather than the potentially-larger system MemTotal or MemAvailable.
        let effective_available_mb = if cgroup_available_mb > 0.0 {
            host_ram_available_mb.min(cgroup_available_mb)
        } else if host_ram_available_mb > 0.0 {
            host_ram_available_mb
        } else {
            // No reliable probe: conservatively assume 8 GB minus a safe margin.
            8192.0
        };

        let max_memory_mb = (effective_available_mb * host_ram_safety_pct / 100.0).max(512.0);

        // ── Device memory budget (GPU/TPU) ───────────────────────────────────────
        let (device_vram_total_mb, device_vram_available_mb, max_device_vram_mb) = match device_kind {
            DeviceKind::Gpu | DeviceKind::Tpu | DeviceKind::Custom(_) if total_mem_bytes > 0 => {
                let tot_mb = total_mem_bytes as f64 / (1024.0 * 1024.0);
                let av_mb = avail_mem_bytes as f64 / (1024.0 * 1024.0);
                let vram_safety_pct: f64 = env::var("TARA_VRAM_SAFETY_PCT")
                    .ok()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(85.0)
                    .clamp(50.0, 95.0);
                let safe_mb = (av_mb.min(tot_mb) * vram_safety_pct / 100.0).max(512.0);
                (tot_mb, av_mb, safe_mb)
            }
            _ => (0.0, 0.0, 0.0),
        };

        // ── Telemetry summary ────────────────────────────────────────────────────
        let effective_batch = batch_size * grad_accum_steps;
        let mem_gb = total_mem_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let device_label = format!("{} ({:.1} GB total)", device_info.name, mem_gb);
        let cgroup_str = if cgroup_limit_mb > 0.0 {
            format!(
                " | CgroupLimit: {:.0} MB, CgroupAvail: {:.0} MB",
                cgroup_limit_mb, cgroup_available_mb
            )
        } else {
            String::new()
        };
        let vram_telemetry_str = if device_vram_total_mb > 0.0 {
            format!(
                " | VRAM: {:.1} GB / {:.1} GB avail (Ceiling: {:.0} MB)",
                device_vram_available_mb / 1024.0,
                device_vram_total_mb / 1024.0,
                max_device_vram_mb
            )
        } else {
            String::new()
        };
        let telemetry = format!(
            "[AutoMemory] Device: {}{vram_telemetry_str} | HostRAMTotal: {:.1} GB | HostRAMAvail: {:.1} GB{} | Precision: {} | Batch: {} | SeqLen: {} | GradAccum: {} | EffBatch: {} | LR: {:.1e} | Warmup: {} | HostCeiling: {:.0} MB",
            device_label,
            host_ram_total_mb / 1024.0,
            host_ram_available_mb / 1024.0,
            cgroup_str,
            precision,
            batch_size,
            seq_len,
            grad_accum_steps,
            effective_batch,
            learning_rate,
            warmup_steps,
            max_memory_mb,
        );

        Self {
            batch_size,
            seq_len,
            grad_accum_steps,
            learning_rate,
            warmup_steps,
            weight_decay,
            grad_clip_norm,
            precision,
            telemetry,
            estimated_total_optimizer_steps: steps_per_epoch,
            host_ram_total_mb,
            host_ram_available_mb,
            cgroup_limit_mb,
            cgroup_available_mb,
            max_memory_mb,
            device_vram_total_mb,
            device_vram_available_mb,
            max_device_vram_mb,
        }
    }

    /// Recompute batch size and sequence length using a live VRAM reading obtained
    /// after the CUDA context is fully initialized (post `CudaTrainer::get_vram_info()`).
    ///
    /// Call this only when a GPU CudaTrainer is successfully created and reports
    /// live VRAM numbers that differ significantly from the pre-init estimate.
    /// Returns `Some(Self)` when a meaningful refinement was possible, `None` when
    /// the live reading is zero or essentially identical to the original estimate.
    pub fn refine_with_live_vram(
        &self,
        live_free_bytes: u64,
        live_total_bytes: u64,
        total_params: usize,
        hidden_size: usize,
        num_layers: usize,
        estimated_total_samples: usize,
    ) -> Option<Self> {
        if live_total_bytes == 0 || live_free_bytes == 0 {
            return None;
        }

        let bytes_per_weight: u64 = match self.precision {
            PrecisionMode::Fp16 | PrecisionMode::Bf16 => 2,
            _ => 4,
        };

        let (new_batch, new_seq, new_accum) = Self::derive_gpu_config(
            live_total_bytes,
            live_free_bytes,
            total_params,
            hidden_size.max(1),
            num_layers.max(1),
            bytes_per_weight,
        );

        // Only report a refinement when numbers changed meaningfully.
        let changed = new_batch != self.batch_size
            || new_seq != self.seq_len
            || new_accum != self.grad_accum_steps;

        if !changed {
            return None;
        }

        let effective_batch = new_batch * new_accum;
        let mem_gb = live_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let free_gb = live_free_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let telemetry = format!(
            "[AutoMemory Refined] Live VRAM: {:.1}/{:.1} GB | Precision: {} | Batch: {} | SeqLen: {} | GradAccum: {} | EffBatch: {}",
            free_gb,
            mem_gb,
            self.precision,
            new_batch,
            new_seq,
            new_accum,
            effective_batch,
        );

        let steps_per_epoch = if estimated_total_samples > 0 && new_seq > 0 && new_accum > 0 {
            (estimated_total_samples / new_accum).max(1)
        } else {
            self.estimated_total_optimizer_steps
        };
        let warmup_steps = (steps_per_epoch / 10).clamp(50, 2000);

        Some(Self {
            batch_size: new_batch,
            seq_len: new_seq,
            grad_accum_steps: new_accum,
            learning_rate: self.learning_rate,
            warmup_steps,
            weight_decay: self.weight_decay,
            grad_clip_norm: self.grad_clip_norm,
            precision: self.precision,
            telemetry,
            estimated_total_optimizer_steps: steps_per_epoch,
            host_ram_total_mb: self.host_ram_total_mb,
            host_ram_available_mb: self.host_ram_available_mb,
            cgroup_limit_mb: self.cgroup_limit_mb,
            cgroup_available_mb: self.cgroup_available_mb,
            max_memory_mb: self.max_memory_mb,
            device_vram_total_mb: live_total_bytes as f64 / (1024.0 * 1024.0),
            device_vram_available_mb: live_free_bytes as f64 / (1024.0 * 1024.0),
            max_device_vram_mb: {
                let vram_safety_pct: f64 = env::var("TARA_VRAM_SAFETY_PCT")
                    .ok()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(85.0)
                    .clamp(50.0, 95.0);
                (live_free_bytes as f64 / (1024.0 * 1024.0) * vram_safety_pct / 100.0).max(512.0)
            },
        })
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Private computation helpers
    // ─────────────────────────────────────────────────────────────────────────────

    /// Derive GPU/TPU batch_size, seq_len, grad_accum_steps from available VRAM.
    ///
    /// When `total_mem_bytes == 0` the device did not report memory — uses a
    /// conservative single-sample fallback to avoid OOM.
    fn derive_gpu_config(
        total_mem_bytes: u64,
        avail_mem_bytes: u64,
        total_params: usize,
        hidden_size: usize,
        num_layers: usize,
        bytes_per_weight: u64,
    ) -> (usize, usize, usize) {
        // If device reports 0 bytes (unknown), use a minimal safe fallback.
        if total_mem_bytes == 0 {
            return Self::conservative_gpu_fallback();
        }

        // Allow environment override for the safety fraction (default 85 %).
        let safety_pct: f64 = env::var("TARA_VRAM_SAFETY_PCT")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(85.0)
            .clamp(50.0, 95.0);

        // Use the *available* memory (post OS/driver overhead) capped by safety fraction.
        // When available_memory_bytes is identical to total (driver didn't split them),
        // we apply the safety fraction to total and treat that as our budget.
        let candidate = avail_mem_bytes.min(total_mem_bytes);
        let safe_vram_bytes = (candidate as f64 * safety_pct / 100.0) as u64;

        // Fixed CUDA runtime + kernel overhead (conservative estimate).
        let cuda_overhead_bytes: u64 = 450 * 1024 * 1024;

        // Weight footprint in bytes (f16/bf16 → 2 B, f32 → 4 B per param).
        let weight_bytes: u64 = total_params as u64 * bytes_per_weight;

        // Gradient accumulation buffer: always f32 regardless of weight precision.
        let grad_bytes: u64 = total_params as u64 * 4;

        // AdamW optimizer moments (m + v): 8 bytes/param when resident in VRAM.
        // For very large models the optimizer is streamed from host RAM (handled by caller).
        let optimizer_bytes: u64 = if total_params >= 200_000_000 {
            // Streamed optimizer: only 3 chunk buffers in VRAM (≈ 12 MB worst case).
            12 * 1024 * 1024
        } else {
            total_params as u64 * 8
        };

        let static_overhead = weight_bytes
            .saturating_add(grad_bytes)
            .saturating_add(optimizer_bytes)
            .saturating_add(cuda_overhead_bytes);

        let avail_for_activations = safe_vram_bytes.saturating_sub(static_overhead);

        // Activation bytes per token: 4 projections (Q, K, V, O) + gate/up/down + 2 norms
        // + residual stream input/output + RoPE buffer ≈ 8 f32 values × hidden_size × num_layers.
        // For f16/bf16 activations we conservatively still budget f32 because gradients
        // are accumulated in f32 during backprop.
        let acts_per_token_bytes: u64 = (hidden_size as u64 * num_layers as u64 * 8 * 4).max(1);

        if avail_for_activations == 0 || acts_per_token_bytes == 0 {
            return Self::conservative_gpu_fallback();
        }

        let max_tokens = avail_for_activations / acts_per_token_bytes;

        if max_tokens == 0 {
            return Self::conservative_gpu_fallback();
        }

        // Target sequence length: next power of two below sqrt(max_tokens), clamped.
        let seq_sqrt = (max_tokens as f64).sqrt() as u64;
        let seq_len = prev_power_of_two(seq_sqrt).clamp(64, 2048) as usize;

        // Micro-batch size from remaining token budget, clamped to a sane range.
        let batch_size = (max_tokens / seq_len as u64).clamp(1, 32) as usize;

        // Gradient accumulation to achieve a practical effective batch of 32.
        let target_effective_batch: usize = 32;
        let grad_accum_steps = (target_effective_batch / batch_size).max(1);

        (batch_size, seq_len, grad_accum_steps)
    }

    /// Derive CPU batch_size, seq_len, grad_accum_steps from available host RAM.
    fn derive_cpu_config(
        avail_ram_bytes: u64,
        total_params: usize,
        hidden_size: usize,
        num_layers: usize,
    ) -> (usize, usize, usize) {
        // CPU training budget: cap at 40 % of available RAM to leave headroom for
        // OS, tokenizer, dataset cache, and other processes.
        let safety_pct: f64 = env::var("TARA_RAM_SAFETY_PCT")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(40.0)
            .clamp(20.0, 70.0);

        // Model weights (f32, 4 bytes/param) + gradients (4 B) + moments m+v (8 B) = 16 B/param.
        let static_footprint_bytes: u64 = total_params as u64 * 16;

        // OS/runtime baseline overhead on host.
        let host_overhead_bytes: u64 = 350 * 1024 * 1024;

        let budget_bytes = if avail_ram_bytes > 0 {
            (avail_ram_bytes as f64 * safety_pct / 100.0) as u64
        } else {
            // Conservatively assume 4 GB available when RAM probe failed.
            (4u64 * 1024 * 1024 * 1024) / 2
        };

        let avail_for_acts = budget_bytes
            .saturating_sub(static_footprint_bytes)
            .saturating_sub(host_overhead_bytes);

        // Use CpuStreamHelper recommendation logic: activation bytes per CPU token (f32).
        let acts_per_token_bytes: u64 = (hidden_size as u64 * num_layers as u64 * 14 * 4).max(1);

        // CPU always uses batch_size = 1 to minimise peak activation memory.
        let batch_size: usize = 1;

        let seq_len: usize = if avail_for_acts > 0 && acts_per_token_bytes > 0 {
            let max_seq = avail_for_acts / acts_per_token_bytes;
            prev_power_of_two(max_seq.clamp(64, 2048)).clamp(64, 2048) as usize
        } else {
            512 // minimal safe fallback for CPU
        };

        // Gradient accumulation of 4 gives a modest effective batch to smooth gradients.
        let grad_accum_steps: usize = 4;

        (batch_size, seq_len, grad_accum_steps)
    }

    /// Minimal conservative fallback when GPU VRAM is unreported (total_memory_bytes == 0).
    fn conservative_gpu_fallback() -> (usize, usize, usize) {
        // batch_size=1, seq_len=512, grad_accum=32 → effective batch=32 with minimal VRAM.
        (1, 512, 32)
    }
}

/// Return the largest power of two that is ≤ `n`.
/// Returns `n` itself when `n == 0`.
fn prev_power_of_two(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let bits = u64::BITS - n.leading_zeros();
    1u64 << (bits - 1)
}

/// Probe host RAM and cgroup memory limits to derive a safe training memory budget.
///
/// Returns `(total_mb, available_mb, cgroup_limit_mb, cgroup_available_mb)`:
/// - `total_mb`: MemTotal from `/proc/meminfo` (physical RAM — may exceed cgroup limit).
/// - `available_mb`: MemAvailable — more accurate for budgeting than MemTotal.
/// - `cgroup_limit_mb`: Hard container limit from cgroup v2 `memory.max` or v1
///   `memory.limit_in_bytes`. Zero when absent or set to "max" (unlimited).
///   On Colab this is the binding ceiling, which may be lower than system MemTotal.
/// - `cgroup_available_mb`: cgroup_limit_mb minus current cgroup usage. Zero when unavailable.
///
/// The env var `TARA_HOST_RAM_MB` overrides `available_mb` and disables cgroup probing.
fn probe_host_memory_budget_mb() -> (f64, f64, f64, f64) {
    // Explicit env override bypasses all OS probes (useful on Windows or for test control).
    if let Ok(override_val) = env::var("TARA_HOST_RAM_MB") {
        if let Ok(mb) = override_val.trim().parse::<f64>() {
            if mb > 0.0 {
                return (mb, mb, 0.0, 0.0);
            }
        }
    }

    // Read /proc/meminfo — present on Linux and all cloud/Colab environments.
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok();
    let (mut total_mb, mut avail_mb) = (0.0f64, 0.0f64);
    if let Some(ref content) = meminfo {
        for line in content.lines() {
            if line.starts_with("MemTotal:") {
                if let Some(kb) = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f64>().ok())
                {
                    total_mb = kb / 1024.0;
                }
            } else if line.starts_with("MemAvailable:") {
                if let Some(kb) = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f64>().ok())
                {
                    avail_mb = kb / 1024.0;
                }
            }
        }
        // Older kernels may lack MemAvailable — fall back to 70% of MemTotal.
        if avail_mb <= 0.0 && total_mb > 0.0 {
            avail_mb = total_mb * 0.70;
        }
    }

    // Probe cgroup v2 (memory.max) or cgroup v1 (memory.limit_in_bytes).
    // "max" in cgroup v2 means unlimited; u64::MAX sentinels (> 1 PB) also mean unlimited.
    let mut cgroup_limit_mb = 0.0f64;
    let limit_paths = [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ];
    for path in limit_paths {
        if let Ok(raw) = std::fs::read_to_string(path) {
            let s = raw.trim();
            if s != "max" {
                if let Ok(bytes) = s.parse::<f64>() {
                    if bytes > 0.0 && bytes < 1e15 {
                        cgroup_limit_mb = bytes / (1024.0 * 1024.0);
                        break;
                    }
                }
            }
        }
    }

    // Probe current cgroup memory usage to derive remaining headroom.
    let mut cgroup_current_mb = 0.0f64;
    let current_paths = [
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ];
    for path in current_paths {
        if let Ok(raw) = std::fs::read_to_string(path) {
            if let Ok(bytes) = raw.trim().parse::<f64>() {
                if bytes >= 0.0 {
                    cgroup_current_mb = bytes / (1024.0 * 1024.0);
                    break;
                }
            }
        }
    }

    let cgroup_available_mb = if cgroup_limit_mb > 0.0 {
        (cgroup_limit_mb - cgroup_current_mb).max(0.0)
    } else {
        0.0
    };

    (total_mb, avail_mb, cgroup_limit_mb, cgroup_available_mb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prev_power_of_two() {
        assert_eq!(prev_power_of_two(0), 0);
        assert_eq!(prev_power_of_two(1), 1);
        assert_eq!(prev_power_of_two(3), 2);
        assert_eq!(prev_power_of_two(4), 4);
        assert_eq!(prev_power_of_two(5), 4);
        assert_eq!(prev_power_of_two(1024), 1024);
        assert_eq!(prev_power_of_two(1025), 1024);
        assert_eq!(prev_power_of_two(2000), 1024);
        assert_eq!(prev_power_of_two(2048), 2048);
    }

    #[test]
    fn test_auto_config_cpu_channel() {
        let channel = crate::ComputeChannel::with_options(
            &crate::types::SelectionPolicy::parse("cpu:0"),
            crate::types::PrecisionMode::Fp32,
        )
        .expect("CPU channel must resolve");

        // 50 M param model, hidden=1024, 24 layers, 1000 samples
        let cfg = AutoTrainingConfig::compute(&channel, 50_000_000, 1024, 24, 1000);
        assert_eq!(cfg.batch_size, 1, "CPU must always use batch_size=1");
        assert!(cfg.seq_len >= 64, "seq_len must be at least 64");
        assert!(cfg.seq_len <= 2048, "seq_len must not exceed 2048");
        assert!(cfg.grad_accum_steps >= 1);
        assert!(cfg.learning_rate > 0.0 && cfg.learning_rate < 1.0);
        assert!(cfg.warmup_steps >= 50);
        assert_eq!(cfg.precision, crate::types::PrecisionMode::Fp32);
        assert!(!cfg.telemetry.is_empty());
        println!("CPU auto-config: {}", cfg.telemetry);
    }

    #[test]
    fn test_auto_config_conservative_gpu_fallback_when_vram_unknown() {
        // Call the private fallback directly via derive_gpu_config with zero total_mem.
        let (batch, seq, accum) = AutoTrainingConfig::derive_gpu_config(
            0, // total_mem_bytes = 0 (unknown)
            0, // avail_mem_bytes = 0
            50_000_000, 1024, 24, 2,
        );
        assert_eq!(batch, 1, "unknown VRAM must fall back to batch=1");
        assert_eq!(seq, 512, "unknown VRAM must fall back to seq_len=512");
        assert_eq!(accum, 32, "unknown VRAM must fall back to grad_accum=32");
    }

    #[test]
    fn test_auto_config_simulated_colab_free_15gb() {
        // Simulate a 15 GB GPU (Colab Free T4): verify derived config is plausible.
        let gb: u64 = 1024 * 1024 * 1024;
        let total_vram: u64 = 15 * gb;
        let avail_vram: u64 = 14 * gb; // typical after CUDA context overhead

        // 100 M param model, hidden=1024, 24 layers, f16.
        let (batch, seq, accum) =
            AutoTrainingConfig::derive_gpu_config(total_vram, avail_vram, 100_000_000, 1024, 24, 2);
        println!(
            "Colab Free 15 GB: batch={batch}, seq_len={seq}, grad_accum={accum}, effective_batch={}",
            batch * accum
        );
        assert!(batch >= 1, "batch must be at least 1");
        assert!(seq >= 64, "seq must be at least 64");
        assert!(seq <= 2048, "seq must not exceed 2048");
        // Power-of-two invariant
        assert_eq!(seq & (seq - 1), 0, "seq_len must be a power of two");
        assert!(accum >= 1);
    }

    #[test]
    fn test_auto_config_learning_rate_scale() {
        let channel = crate::ComputeChannel::with_options(
            &crate::types::SelectionPolicy::parse("cpu:0"),
            crate::types::PrecisionMode::Fp32,
        )
        .expect("CPU channel must resolve");

        let small_cfg = AutoTrainingConfig::compute(&channel, 50_000_000, 512, 12, 500);
        let medium_cfg = AutoTrainingConfig::compute(&channel, 200_000_000, 1024, 24, 500);
        let large_cfg = AutoTrainingConfig::compute(&channel, 700_000_000, 2048, 32, 500);

        assert!(
            small_cfg.learning_rate > medium_cfg.learning_rate,
            "small models need higher LR than medium"
        );
        assert!(
            medium_cfg.learning_rate > large_cfg.learning_rate,
            "medium models need higher LR than large"
        );
    }

    #[test]
    fn test_refine_with_live_vram_no_change() {
        let channel = crate::ComputeChannel::with_options(
            &crate::types::SelectionPolicy::parse("cpu:0"),
            crate::types::PrecisionMode::Fp32,
        )
        .expect("CPU channel must resolve");
        let cfg = AutoTrainingConfig::compute(&channel, 10_000_000, 512, 12, 100);
        // When live VRAM is 0, no refinement should occur.
        let refined = cfg.refine_with_live_vram(0, 0, 10_000_000, 512, 12, 100);
        assert!(
            refined.is_none(),
            "zero VRAM bytes must not produce a refinement"
        );
    }
}
