//! Native self-trainer for TARA model evolution.
//!
//! Executes full-network backpropagation (Embeddings, RMSNorm, Attention/GQA,
//! RoPE, SwiGLU MLP, and LM Head) with AdamW optimization, gradient clipping,
//! gradient accumulation, streaming dataset buffering, and isolated candidate staging.

use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

use crate::config::TaraConfig;
use crate::dataset::compiler::{CompilerError, DynamicDatasetCompiler};
use crate::model::causal_lm::TaraForCausalLM;
use crate::safetensors::{
    compute_sha256, load_model_weights_with_purpose, write_safetensors_sharded, SafeTensorsError,
};
use crate::tokenizer::TaraTokenizer;

/// Errors from the self-trainer.
#[derive(Debug, Error)]
pub enum TrainerError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("SafeTensors error: {0}")]
    SafeTensors(#[from] SafeTensorsError),
    #[error("model load error: {0}")]
    Model(String),
    #[error("invalid training data: {0}")]
    Dataset(String),
    #[error("dataset compilation error: {0}")]
    Compiler(#[from] CompilerError),
}

/// Hyperparameters for AdamW optimization step.
#[derive(Debug, Clone, Copy)]
pub struct AdamWHyperparams {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
}

/// Dynamic AdamW optimizer state across all model parameter tensors.
pub struct DynamicAdamW {
    m: HashMap<String, Vec<f32>>,
    v: HashMap<String, Vec<f32>>,
    step: u64,
    last_grad_norm: f32,
}

impl Default for DynamicAdamW {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicAdamW {
    pub fn new() -> Self {
        Self {
            m: HashMap::new(),
            v: HashMap::new(),
            step: 0,
            last_grad_norm: 0.0,
        }
    }

    /// Construct a DynamicAdamW optimizer with eagerly pre-allocated moment buffers.
    ///
    /// Allocating moment vectors upfront prevents unexpected host memory spikes
    /// and OOM-killer invocations during the first optimizer step.
    pub fn with_preallocated_moments(weights: &HashMap<String, Vec<f32>>) -> Self {
        let mut m = HashMap::with_capacity(weights.len());
        let mut v = HashMap::with_capacity(weights.len());
        for (name, param) in weights {
            m.insert(name.clone(), vec![0.0f32; param.len()]);
            v.insert(name.clone(), vec![0.0f32; param.len()]);
        }
        Self {
            m,
            v,
            step: 0,
            last_grad_norm: 0.0,
        }
    }

    pub fn last_grad_norm(&self) -> f32 {
        self.last_grad_norm
    }

    /// Run single AdamW step with global L2 gradient clipping across all parameters.
    ///
    /// If the computed gradient norm is NaN or Inf (gradient explosion or degenerate input),
    /// the update step is skipped entirely to protect weights — no NaN is propagated.
    /// The step counter still advances so bias-correction remains monotone.
    pub fn step(
        &mut self,
        weights: &mut HashMap<String, Vec<f32>>,
        grads: &HashMap<String, Vec<f32>>,
        hp: &AdamWHyperparams,
    ) -> bool {
        let lr = hp.lr;
        let beta1 = hp.beta1;
        let beta2 = hp.beta2;
        let eps = hp.eps;
        let weight_decay = hp.weight_decay;
        let max_grad_norm = hp.max_grad_norm;

        // 1. Calculate global L2 gradient norm
        let mut total_norm_sq = 0.0f32;
        for g in grads.values() {
            for &val in g {
                total_norm_sq += val * val;
            }
        }
        let total_norm = total_norm_sq.sqrt();

        // Guard: skip the parameter update if gradients are NaN or Inf.
        // This prevents a single degenerate batch from poisoning the entire model.
        // Neither moments nor step count are updated on non-finite gradients.
        if !total_norm.is_finite() {
            return false;
        }

        self.last_grad_norm = total_norm;
        self.step += 1;
        let step_i32 = self.step.min(i32::MAX as u64) as i32;
        let bc1 = (1.0 - beta1.powi(step_i32)).max(1e-12);
        let bc2 = (1.0 - beta2.powi(step_i32)).max(1e-12);

        let clip_scale = if total_norm > max_grad_norm && total_norm > 0.0 {
            max_grad_norm / total_norm
        } else {
            1.0f32
        };

        // 2. Update all parameter weights with AdamW
        for (name, param) in weights.iter_mut() {
            if let Some(grad) = grads.get(name) {
                let n = param.len().min(grad.len());
                let m = self
                    .m
                    .entry(name.clone())
                    .or_insert_with(|| vec![0.0f32; param.len()]);
                if m.len() < n {
                    m.resize(param.len(), 0.0);
                }
                let v = self
                    .v
                    .entry(name.clone())
                    .or_insert_with(|| vec![0.0f32; param.len()]);
                if v.len() < n {
                    v.resize(param.len(), 0.0);
                }

                for i in 0..n {
                    let g = grad[i] * clip_scale;
                    m[i] = beta1 * m[i] + (1.0 - beta1) * g;
                    v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
                    let m_hat = m[i] / bc1;
                    let v_hat = v[i] / bc2;
                    // Decoupled weight decay applied to current parameter before Adam step
                    param[i] -= lr * weight_decay * param[i];
                    param[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                }
            }
        }
        true
    }

    pub fn get_step(&self) -> u64 {
        self.step
    }

    pub fn set_step(&mut self, s: u64) {
        self.step = s;
    }

    pub fn export_state(&self) -> HashMap<String, (Vec<f32>, Vec<f32>)> {
        let mut out = HashMap::new();
        for (k, m_vec) in &self.m {
            let v_vec = self
                .v
                .get(k)
                .cloned()
                .unwrap_or_else(|| vec![0.0f32; m_vec.len()]);
            out.insert(k.clone(), (m_vec.clone(), v_vec));
        }
        out
    }

    pub fn load_state(&mut self, state: HashMap<String, (Vec<f32>, Vec<f32>)>) {
        for (k, (m_vec, v_vec)) in state {
            self.m.insert(k.clone(), m_vec);
            self.v.insert(k, v_vec);
        }
    }
}

/// Streaming JSONL dataset reader that reads sample pairs line-by-line without buffering entire files.
pub struct StreamingDatasetReader {
    reader: BufReader<File>,
    path: String,
}

impl StreamingDatasetReader {
    pub fn open(path: &str) -> Result<Self, std::io::Error> {
        let file = File::open(path)?;
        Ok(Self {
            reader: BufReader::new(file),
            path: path.to_string(),
        })
    }

    /// Read next valid (input, output) pair.
    pub fn next_sample(&mut self) -> Result<Option<(String, String)>, TrainerError> {
        let mut line = String::new();
        while self.reader.read_line(&mut line)? > 0 {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                line.clear();
                continue;
            }
            let val: Value = serde_json::from_str(trimmed)
                .map_err(|e| TrainerError::Dataset(format!("{}: {e}", self.path)))?;
            line.clear();
            let has_text = val
                .get("text")
                .or_else(|| val.get("content"))
                .and_then(Value::as_str);

            let input = val
                .get("formatted_input")
                .and_then(Value::as_str)
                .map(|s| s.to_string())
                .or_else(|| {
                    val.get("input")
                        .or_else(|| val.get("prompt"))
                        .or_else(|| val.get("instruction"))
                        .or_else(|| val.get("trigger_pattern"))
                        .or_else(|| val.get("metadata").and_then(|m| m.get("input")))
                        .and_then(Value::as_str)
                        .map(|s| s.trim().to_string())
                })
                .or_else(|| {
                    if has_text.is_some() {
                        let source = val
                            .get("source")
                            .and_then(Value::as_str)
                            .unwrap_or("Knowledge");
                        let book = val.get("book").and_then(Value::as_str).unwrap_or("");
                        let chapter = val
                            .get("chapter")
                            .and_then(|c| c.as_u64())
                            .map(|c| format!(" Chapter {c}"))
                            .unwrap_or_default();
                        let header = if !book.is_empty() {
                            format!("{source}: {book}{chapter}")
                        } else {
                            source.to_string()
                        };
                        Some(format!(
                            "<|im_start|>user\n{header}\n<|im_end|>\n<|im_start|>assistant\n"
                        ))
                    } else {
                        None
                    }
                })
                .unwrap_or_default();

            let output = val
                .get("formatted_target")
                .and_then(Value::as_str)
                .map(|s| s.to_string())
                .or_else(|| {
                    val.get("output")
                        .or_else(|| val.get("completion"))
                        .or_else(|| val.get("response"))
                        .or_else(|| val.get("recommendation"))
                        .or_else(|| val.get("metadata").and_then(|m| m.get("output")))
                        .or_else(|| val.get("text"))
                        .or_else(|| val.get("content"))
                        .and_then(Value::as_str)
                        .map(|s| s.trim().to_string())
                })
                .unwrap_or_default();

            if !input.trim().is_empty() && !output.trim().is_empty() {
                return Ok(Some((input, output)));
            }
        }
        Ok(None)
    }
}

/// Authoritative compute device selection policy directly from TARA-HD (Single Source of Truth).
pub type TrainingDevice = tara_hd::SelectionPolicy;

/// Configuration and state bundle for atomic checkpoint persistence.
pub struct CheckpointBundleConfig<'a> {
    pub cp_dir: &'a str,
    pub model_dir: &'a str,
    pub weights: &'a HashMap<String, Vec<f32>>,
    pub config: &'a TaraConfig,
    pub opt_moments: HashMap<String, (Vec<f32>, Vec<f32>)>,
    pub optimizer_steps: usize,
    pub samples_seen: usize,
    pub epoch_idx: usize,
    pub current_eval_loss: Option<f64>,
    pub best_loss: f64,
    pub is_best: bool,
    pub val_tokens_empty: bool,
}

/// TARA native self-trainer.
pub struct NativeSelfTrainer {
    pub model_dir: String,
    pub repo_root: String,
    pub max_memory_mb: f64,
    pub dataset_dir: Option<String>,
    pub curriculum_path: Option<String>,
    pub learning_rate: Option<f32>,
    pub batch_size: Option<usize>,
    pub max_steps: Option<usize>,
    pub warmup_steps: Option<usize>,
    pub weight_decay: Option<f32>,
    pub grad_clip_norm: Option<f32>,
    pub device: TrainingDevice,
    pub precision: crate::cuda::TrainingPrecision,
    pub checkpoint_dir: Option<String>,
    pub checkpoint_interval: Option<usize>,
    pub resume: bool,
    pub resume_from: Option<String>,
    pub stage_name: Option<String>,
    pub anchor_weights: Option<HashMap<String, Vec<f32>>>,
    pub ewc_importance: Option<HashMap<String, Vec<f32>>>,
    pub ewc_lambda: f32,
}

impl NativeSelfTrainer {
    /// Create a self-trainer for the given model directory.
    pub fn new(model_dir: &str, repo_root: &str) -> Self {
        Self {
            model_dir: model_dir.to_string(),
            repo_root: repo_root.to_string(),
            max_memory_mb: 65536.0,
            dataset_dir: None,
            curriculum_path: None,
            learning_rate: None,
            batch_size: None,
            max_steps: None,
            warmup_steps: None,
            weight_decay: None,
            grad_clip_norm: None,
            device: TrainingDevice::Auto,
            precision: crate::cuda::TrainingPrecision::Auto,
            checkpoint_dir: None,
            checkpoint_interval: None,
            resume: false,
            resume_from: None,
            stage_name: None,
            anchor_weights: None,
            ewc_importance: None,
            ewc_lambda: 100.0,
        }
    }

    /// Set an explicit stage label for live telemetry and training progress logs.
    pub fn with_stage_name(mut self, name: &str) -> Self {
        self.stage_name = Some(name.to_string());
        self
    }

    /// Configure Elastic Weight Consolidation (EWC) anchor weights and Fisher importance matrix.
    pub fn with_ewc_anchors(
        mut self,
        anchor_weights: HashMap<String, Vec<f32>>,
        ewc_importance: HashMap<String, Vec<f32>>,
        lambda: f32,
    ) -> Self {
        self.anchor_weights = Some(anchor_weights);
        self.ewc_importance = Some(ewc_importance);
        self.ewc_lambda = lambda;
        self
    }

    /// Set an explicit source directory or checkpoint from which to resume.
    pub fn with_resume_from<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.resume_from = Some(path.as_ref().to_string_lossy().to_string());
        self.resume = true;
        self
    }

    /// Set an explicit checkpoint directory for periodic saving and resuming.
    pub fn with_checkpoint_dir(mut self, dir: &str) -> Self {
        self.checkpoint_dir = Some(dir.to_string());
        self
    }

    /// Set periodic checkpoint interval in steps.
    pub fn with_checkpoint_interval(mut self, interval: usize) -> Self {
        self.checkpoint_interval = Some(interval);
        self
    }

    /// Enable resuming from the latest checkpoint if found.
    pub fn with_resume(mut self, resume: bool) -> Self {
        self.resume = resume;
        self
    }

    /// Set an explicit execution device backend (CPU, GPU, or Auto).
    pub fn with_device(mut self, dev: TrainingDevice) -> Self {
        self.device = dev;
        self
    }

    /// Set an explicit training precision (FP32, FP16, or Auto).
    pub fn with_precision(mut self, prec: crate::cuda::TrainingPrecision) -> Self {
        self.precision = prec;
        self
    }

    /// Set an explicit dataset directory.
    pub fn with_dataset_dir<P: AsRef<Path>>(mut self, dir: P) -> Self {
        self.dataset_dir = Some(dir.as_ref().to_string_lossy().to_string());
        self
    }

    /// Set an explicit academic curriculum path.
    pub fn with_curriculum_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.curriculum_path = Some(path.as_ref().to_string_lossy().to_string());
        self
    }

    /// Set an explicit learning rate.
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = Some(lr);
        self
    }

    /// Set an explicit gradient accumulation / batch size.
    pub fn with_batch_size(mut self, bs: usize) -> Self {
        self.batch_size = Some(bs);
        self
    }

    /// Set an explicit maximum training steps ceiling.
    pub fn with_max_steps(mut self, ms: usize) -> Self {
        self.max_steps = Some(ms);
        self
    }

    /// Set an explicit linear learning-rate warmup step count.
    pub fn with_warmup_steps(mut self, ws: usize) -> Self {
        self.warmup_steps = Some(ws);
        self
    }

    /// Set an explicit AdamW weight decay coefficient.
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = Some(wd);
        self
    }

    /// Set an explicit gradient clipping norm ceiling.
    pub fn with_grad_clip_norm(mut self, gcn: f32) -> Self {
        self.grad_clip_norm = Some(gcn);
        self
    }
}

/// Detailed memory footprint breakdown across physical hardware memory domains.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingMemoryFootprint {
    /// Projected Host RAM required in MB (including safety margin).
    pub host_ram_mb: f64,
    /// Projected Device (GPU/TPU) VRAM required in MB (including safety margin).
    /// Zero when pure CPU training.
    pub device_vram_mb: f64,
    /// Highest memory consumption across active pools in MB.
    pub governing_peak_mb: f64,
    /// Diagnostic breakdown string detailing per-component memory allocations.
    pub breakdown: String,
}

impl NativeSelfTrainer {
    /// Set an explicit maximum memory ceiling in megabytes.
    pub fn with_max_memory_mb(mut self, mb: f64) -> Self {
        self.max_memory_mb = mb;
        self
    }

    /// Calculate projected training memory footprint in MB given total parameter count, target device, and precision.
    /// Explicitly accounts for Host RAM and GPU VRAM placements: model weights, gradients, AdamW moments,
    /// activations, chunk buffers, workspaces, and safety margins.
    pub fn estimate_memory_footprint_mb_detailed(
        total_params: usize,
        is_gpu: bool,
        is_f16: bool,
        max_tensor_len: usize,
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
    ) -> TrainingMemoryFootprint {
        // Use the actual model hidden_size from config to compute activation buffer sizes.
        // Callers must pass config.hidden_size; using a stale assumption here would produce
        // incorrect estimates as model architectures change.
        let hs = hidden_size.max(1);

        // 1. Host RAM Breakdown:
        // Canonical FP32 model weights on host: 4 bytes / param (single copy via from_weights_move)
        let host_model_mb = (total_params * 4) as f64 / (1024.0 * 1024.0);
        // Host optimizer states:
        let streamed_optimizer_threshold: usize = 200_000_000;
        let is_streamed_gpu = is_gpu && total_params >= streamed_optimizer_threshold;
        let disk_moments_forced = std::env::var("TARA_STREAM_MOMENTS_DISK")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false);
        let moments_size_mb = (total_params * 4) as f64 / (1024.0 * 1024.0);
        let snap = crate::cuda::gpu_trainer::capture_memory_snapshot(None);
        let host_avail_mb = match (snap.cgroup_limit_mb, snap.cgroup_current_mb) {
            (Some(lim), Some(cur)) if lim > cur => snap.host_avail_mb.min(lim - cur),
            _ => snap.host_avail_mb,
        };
        let will_use_disk_moments =
            is_streamed_gpu && (disk_moments_forced || host_avail_mb < (moments_size_mb + 4000.0));

        let (host_opt_mb, opt_tag) = if is_gpu {
            if will_use_disk_moments {
                // Chunked disk-backed moments: only active transfer chunk in host memory (12 MB)
                (12.0f64, format!("Opt(disk-streamed chunk)={:.1}MB", 12.0))
            } else if is_streamed_gpu {
                // Compact FP16 streamed moments in host RAM: 2B (m) + 2B (v) = 4B per param
                (
                    moments_size_mb,
                    format!("Opt(m+v fp16)={moments_size_mb:.1}MB"),
                )
            } else {
                // Non-streamed: moments live in GPU VRAM
                (0.0f64, "Opt(GPU VRAM resident)=0.0MB".to_string())
            }
        } else {
            (
                (total_params * 12) as f64 / (1024.0 * 1024.0),
                format!(
                    "Grads+Moments(fp32)={:.1}MB",
                    (total_params * 12) as f64 / (1024.0 * 1024.0)
                ),
            )
        };
        // Activations & cache buffers on host: micro-batch activations across transformer layers.
        // Layer count factor of 14 covers: q,k,v,o projections + gate,up,down MLP + 2x norms +
        // residual stream input/output + RoPE buffer + LM head partial row.
        let host_act_mb = (batch_size * seq_len * hs * 4 * 14) as f64 / (1024.0 * 1024.0);
        let host_overhead_mb = 500.0; // Tokenizer, runtime, OS, Colab kernel, dataset buffers
        let host_subtotal = host_model_mb + host_opt_mb + host_act_mb + host_overhead_mb;
        let host_total_with_margin = host_subtotal * 1.15; // 15% safety headroom

        // 2. GPU VRAM Breakdown:
        let (gpu_w_mb, gpu_g_mb, gpu_chunk_mb, gpu_act_mb, gpu_total_with_margin) = if is_gpu {
            let w_mb = if is_f16 {
                (total_params * 2) as f64 / (1024.0 * 1024.0)
            } else {
                (total_params * 4) as f64 / (1024.0 * 1024.0)
            };
            let g_mb = (total_params * 4) as f64 / (1024.0 * 1024.0);
            let chunk_mb = if is_streamed_gpu {
                // 3 chunk buffers (dw, dm, dv) of max_tensor_len in FP32 = 12 bytes / max element
                (max_tensor_len * 12) as f64 / (1024.0 * 1024.0)
            } else {
                // Resident moments m (4B) + v (4B) = 8 bytes / param in VRAM for small models
                (total_params * 8) as f64 / (1024.0 * 1024.0)
            };
            // GPU activation buffer: hidden_size * 8 bytes per token (FP16 acts + gradient buffer)
            let act_mb = (batch_size * seq_len * hs * 8) as f64 / (1024.0 * 1024.0) + 150.0;
            let cuda_overhead_mb = 450.0;
            let subtotal = w_mb + g_mb + chunk_mb + act_mb + cuda_overhead_mb;
            (w_mb, g_mb, chunk_mb, act_mb, subtotal * 1.10) // 10% safety headroom
        } else {
            (0.0, 0.0, 0.0, 0.0, 0.0)
        };

        // Governing peak: both host RAM AND GPU VRAM must be within limits.
        let governing_peak = if is_gpu {
            host_total_with_margin.max(gpu_total_with_margin)
        } else {
            host_total_with_margin
        };

        let breakdown = if is_gpu {
            format!(
                "Host RAM: Model={host_model_mb:.1}MB, {opt_tag}, Act={host_act_mb:.1}MB, Overhead={host_overhead_mb:.1}MB, Total(+15% Margin)={host_total_with_margin:.1}MB | GPU VRAM: Weights={gpu_w_mb:.1}MB, Grads={gpu_g_mb:.1}MB, Chunks/States={gpu_chunk_mb:.1}MB, Act={gpu_act_mb:.1}MB, Total(+10% Margin)={gpu_total_with_margin:.1}MB",
            )
        } else {
            format!(
                "Host RAM: Model={host_model_mb:.1}MB, {opt_tag}, Act={host_act_mb:.1}MB, Overhead={host_overhead_mb:.1}MB, Total(+15% Margin)={host_total_with_margin:.1}MB",
            )
        };

        TrainingMemoryFootprint {
            host_ram_mb: host_total_with_margin,
            device_vram_mb: gpu_total_with_margin,
            governing_peak_mb: governing_peak,
            breakdown,
        }
    }

    /// Calculate projected training memory footprint in MB given total parameter count.
    /// Uses a representative hidden_size of 2048 as a conservative estimate when config is unavailable.
    /// For accurate estimates, use `estimate_memory_footprint_mb_detailed` with the actual config.hidden_size.
    pub fn estimate_memory_footprint_mb(total_params: usize) -> f64 {
        Self::estimate_memory_footprint_mb_detailed(total_params, false, false, 0, 1, 512, 2048)
            .governing_peak_mb
    }

    /// Return the current trainer status (last cycle time, epoch count, etc.).
    pub fn get_status(&self) -> Value {
        let status_path = format!(
            "{}/storage/training/self_trainer_status.json",
            self.repo_root
        );
        if let Ok(raw) = fs::read_to_string(&status_path) {
            serde_json::from_str(&raw).unwrap_or_else(|_| json!({"status": "NO_CYCLE_RUN"}))
        } else {
            json!({
                "status": "IDLE",
                "model_dir": self.model_dir,
                "last_cycle": null,
                "cycles_completed": 0
            })
        }
    }

    fn copy_dir_all(src: &Path, dst: &Path) -> std::io::Result<()> {
        fs::create_dir_all(dst)?;
        for entry in fs::read_dir(src)? {
            let entry = entry?;
            let ft = entry.file_type()?;
            let dest_path = dst.join(entry.file_name());
            if ft.is_dir() {
                Self::copy_dir_all(&entry.path(), &dest_path)?;
            } else {
                let _ = fs::copy(entry.path(), dest_path)?;
            }
        }
        Ok(())
    }

    /// Formats seconds into human-readable HH:MM:SS or MM:SS format.
    pub fn format_duration_hhmmss(total_seconds: f64) -> String {
        let total_secs = total_seconds.max(0.0) as u64;
        let hours = total_secs / 3600;
        let minutes = (total_secs % 3600) / 60;
        let seconds = total_secs % 60;
        if hours > 0 {
            format!("{hours}h {minutes:02}m {seconds:02}s")
        } else {
            format!("{minutes}m {seconds:02}s")
        }
    }

    /// Automatically synchronize checkpoint milestones to Google Drive if present.
    pub fn sync_to_gdrive_if_present(source_dir: &Path) {
        let mut targets = Vec::new();
        if let Ok(env_dir) = std::env::var("GDRIVE_CHECKPOINT_DIR") {
            if !env_dir.trim().is_empty() {
                targets.push(PathBuf::from(env_dir));
            }
        }
        targets.push(PathBuf::from("/content/drive/MyDrive/TARA_CHECKPOINTS"));

        let folder_name = source_dir
            .file_name()
            .unwrap_or_else(|| std::ffi::OsStr::new("checkpoint"));

        for target in targets {
            if let Some(parent) = target.parent() {
                if parent.exists() {
                    let dest_dir = target.join(folder_name);
                    if let Err(e) = Self::copy_dir_all(source_dir, &dest_dir) {
                        eprintln!(
                            "[GDrive Sync] Warning: failed to copy '{}' to '{}': {e}",
                            source_dir.display(),
                            dest_dir.display()
                        );
                    } else {
                        println!(
                            "[GDrive Sync] Milestone checkpoint synchronized to Google Drive: {}",
                            dest_dir.display()
                        );
                    }
                    break;
                }
            }
        }
    }

    /// Persist an atomic sharded model checkpoint bundle, optimizer state, and metadata to disk and Google Drive.
    pub fn save_checkpoint_state_bundle(
        bundle: CheckpointBundleConfig<'_>,
    ) -> Result<(), TrainerError> {
        let cp_path = Path::new(bundle.cp_dir);
        let staging_cp = cp_path.join(".staging");
        std::fs::create_dir_all(&staging_cp)?;

        let shapes: HashMap<String, Vec<usize>> = bundle
            .weights
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    crate::model_expansion::tensor_shape(k, v.len(), bundle.config),
                )
            })
            .collect();

        write_safetensors_sharded(
            bundle.weights,
            &shapes,
            &staging_cp.to_string_lossy(),
            crate::safetensors::resolve_max_shard_size_bytes(),
        )?;

        let mut opt_tensors: HashMap<String, Vec<f32>> = HashMap::new();
        let mut opt_shapes: HashMap<String, Vec<usize>> = HashMap::new();
        for (k, (m_vec, v_vec)) in bundle.opt_moments {
            let m_name = format!("{k}.adam_m");
            let v_name = format!("{k}.adam_v");
            opt_shapes.insert(m_name.clone(), vec![m_vec.len()]);
            opt_shapes.insert(v_name.clone(), vec![v_vec.len()]);
            opt_tensors.insert(m_name, m_vec);
            opt_tensors.insert(v_name, v_vec);
        }
        crate::safetensors::write_safetensors_with_shapes(
            &opt_tensors,
            &opt_shapes,
            &staging_cp.join("optimizer.safetensors").to_string_lossy(),
        )?;

        let _ = std::fs::copy(
            Path::new(bundle.model_dir).join("config.json"),
            staging_cp.join("config.json"),
        );
        let _ = std::fs::copy(
            Path::new(bundle.model_dir).join("tokenizer.json"),
            staging_cp.join("tokenizer.json"),
        );

        let state_json = serde_json::json!({
            "optimizer_step": bundle.optimizer_steps,
            "samples_seen": bundle.samples_seen,
            "epoch": bundle.epoch_idx,
            "validation_loss": bundle.current_eval_loss,
            "best_validation_loss": if bundle.val_tokens_empty { None } else { Some(bundle.best_loss) },
            "is_best": bundle.is_best,
            "timestamp": crate::now_iso(),
        });
        std::fs::write(
            staging_cp.join("checkpoint_state.json"),
            serde_json::to_string_pretty(&state_json).unwrap_or_default(),
        )?;

        std::fs::create_dir_all(cp_path)?;
        for entry in std::fs::read_dir(&staging_cp)? {
            let entry = entry?;
            let dest = cp_path.join(entry.file_name());
            std::fs::copy(entry.path(), dest)?;
        }

        if bundle.is_best {
            let best_dir = cp_path.join("best");
            std::fs::create_dir_all(&best_dir)?;
            for entry in std::fs::read_dir(&staging_cp)? {
                let entry = entry?;
                let dest = best_dir.join(entry.file_name());
                std::fs::copy(entry.path(), dest)?;
            }
            println!(
                "[Checkpoint] ★ Dynamic milestone: New BEST checkpoint saved to '{}/best' (Validation Loss: {:.4})",
                bundle.cp_dir, bundle.best_loss
            );
            Self::sync_to_gdrive_if_present(&best_dir);
        }

        let _ = std::fs::remove_dir_all(&staging_cp);
        Self::sync_to_gdrive_if_present(cp_path);

        let val_loss_str = bundle
            .current_eval_loss
            .map(|l| format!("{:.4}", l))
            .unwrap_or_else(|| "N/A".to_string());
        let best_loss_str = if bundle.val_tokens_empty {
            "N/A".to_string()
        } else {
            format!("{:.4}", bundle.best_loss)
        };
        println!(
            "[Checkpoint] Saved persistent checkpoint to '{}' at step {} (Val Loss: {}, Best: {})",
            bundle.cp_dir, bundle.optimizer_steps, val_loss_str, best_loss_str
        );
        Ok(())
    }

    /// Run full-network self-learning cycle with dynamic shapes and crash-safe candidate staging.
    pub fn run_full_self_learning_cycle(
        &self,
        max_epochs: usize,
        force_now: bool,
    ) -> Result<Value, TrainerError> {
        self.run_full_self_learning_cycle_isolated(max_epochs, force_now, None)
    }

    /// Run full-network self-learning cycle writing to an isolated candidate output directory.
    pub fn run_full_self_learning_cycle_isolated(
        &self,
        max_epochs: usize,
        force_now: bool,
        candidate_dir_override: Option<&str>,
    ) -> Result<Value, TrainerError> {
        if max_epochs == 0 {
            return Err(TrainerError::Dataset(
                "max_epochs must be at least 1 — zero epochs produce no training progress".into(),
            ));
        }

        // 1. Cooldown enforcement
        if !force_now {
            let previous = self.get_status();
            if let Some(last_cycle) = previous
                .get("completed_at_epoch_seconds")
                .and_then(Value::as_u64)
            {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                if now.saturating_sub(last_cycle) < 3600 {
                    return Ok(json!({
                        "status": "COOLDOWN",
                        "last_cycle_epoch_seconds": last_cycle,
                        "retry_after_seconds": 3600 - now.saturating_sub(last_cycle)
                    }));
                }
            }
        }

        // 2. Discover and compile dataset (curriculum + pretraining pool)
        let mut samples = Vec::new();
        let target_samples_limit = self
            .max_steps
            .map(|ms| ms * self.batch_size.unwrap_or(4) * 2);
        let target_limit_str = target_samples_limit
            .map(|l| format!("{l} samples"))
            .unwrap_or_else(|| "unlimited".to_string());
        println!("[Dataset] Loading training data (target quota: {target_limit_str})...");

        // 2a. Load Academic Curriculum if available (storage/datasets/curriculum/foundational_curriculum.jsonl)
        let has_explicit_curriculum = self.curriculum_path.is_some();
        let curr_path = self.curriculum_path.clone().unwrap_or_else(|| {
            let found = format!(
                "{}/storage/datasets/curriculum/foundational_curriculum.jsonl",
                self.repo_root
            );
            if Path::new(&found).exists() {
                found
            } else {
                format!(
                    "{}/storage/datasets/curriculum/master_curriculum.jsonl",
                    self.repo_root
                )
            }
        });
        if Path::new(&curr_path).exists() && Path::new(&curr_path).is_file() {
            if let Ok(mut reader) = StreamingDatasetReader::open(&curr_path) {
                while let Ok(Some(pair)) = reader.next_sample() {
                    samples.push(pair);
                    if let Some(limit) = target_samples_limit {
                        if samples.len() >= limit {
                            break;
                        }
                    }
                }
            }
        }

        // 2b. Discover pretraining dataset if more samples needed.
        // If an explicit curriculum was provided and dataset_dir was NOT set,
        // do not fall back to pretraining pool; train strictly on the explicit curriculum.
        let skip_pretraining = (has_explicit_curriculum && self.dataset_dir.is_none())
            || self.dataset_dir.as_deref() == Some("none")
            || self.dataset_dir.as_deref() == Some("");

        let need_more = !skip_pretraining
            && target_samples_limit
                .map(|lim| samples.len() < lim)
                .unwrap_or(true);
        if need_more {
            let dataset_candidate = self.dataset_dir.clone().unwrap_or_else(|| {
                let main_v1_dir = format!("{}/storage/datasets/main_dataset_v1", self.repo_root);
                if Path::new(&main_v1_dir).exists() {
                    main_v1_dir
                } else {
                    let unified_dir = format!("{}/storage/datasets/tara_dataset", self.repo_root);
                    if Path::new(&unified_dir).exists() {
                        unified_dir
                    } else {
                        format!("{}/storage/datasets", self.repo_root)
                    }
                }
            });

            if Path::new(&dataset_candidate).is_file() {
                if let Ok(mut reader) = StreamingDatasetReader::open(&dataset_candidate) {
                    while let Ok(Some(pair)) = reader.next_sample() {
                        samples.push(pair);
                        if let Some(limit) = target_samples_limit {
                            if samples.len() >= limit {
                                break;
                            }
                        }
                    }
                }
            } else if Path::new(&dataset_candidate).is_dir() {
                // Read from shard files directly in dataset_candidate directory
                let mut shard_files = Vec::new();
                if let Ok(entries) = fs::read_dir(&dataset_candidate) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                            shard_files.push(path);
                        }
                    }
                }
                shard_files.sort();

                if !shard_files.is_empty() {
                    for shard_path in shard_files {
                        if let Ok(mut reader) =
                            StreamingDatasetReader::open(&shard_path.to_string_lossy())
                        {
                            while let Ok(Some(pair)) = reader.next_sample() {
                                samples.push(pair);
                                if samples.len() % 1000 == 0 {
                                    println!("[Dataset] Streamed {} samples...", samples.len());
                                }
                                if let Some(limit) = target_samples_limit {
                                    if samples.len() >= limit {
                                        break;
                                    }
                                }
                            }
                        }
                        if let Some(limit) = target_samples_limit {
                            if samples.len() >= limit {
                                break;
                            }
                        }
                    }
                } else {
                    let unified_path = format!("{}/unified_training.jsonl", dataset_candidate);
                    let exists_and_nonempty = Path::new(&unified_path).exists()
                        && fs::metadata(&unified_path)
                            .map(|m| m.len() > 0)
                            .unwrap_or(false);
                    if !exists_and_nonempty {
                        let _ = DynamicDatasetCompiler::new(&dataset_candidate, &unified_path)
                            .compile();
                    }
                    if Path::new(&unified_path).exists() {
                        if let Ok(mut reader) = StreamingDatasetReader::open(&unified_path) {
                            while let Ok(Some(pair)) = reader.next_sample() {
                                samples.push(pair);
                                if let Some(limit) = target_samples_limit {
                                    if samples.len() >= limit {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if samples.is_empty() {
            return Ok(json!({
                "status": "NO_SAMPLES",
                "message": "No training samples found in datasets or storage/datasets"
            }));
        }
        println!("[Dataset] Collected {} total samples.", samples.len());

        // 3. Load model weights, config, and tokenizer
        let (weights, shapes) = load_model_weights_with_purpose(&self.model_dir, "INPUT")?;
        let config_path = format!("{}/config.json", self.model_dir);
        let config = TaraConfig::from_json_file(&config_path)
            .map_err(|e| TrainerError::Model(e.to_string()))?;

        let tokenizer_path = format!("{}/tokenizer.json", self.model_dir);
        let tokenizer = TaraTokenizer::from_file(&tokenizer_path)
            .map_err(|e| TrainerError::Model(e.to_string()))?;
        if tokenizer.vocab_size != config.vocab_size {
            return Err(TrainerError::Model(format!(
                "tokenizer vocabulary has {} ids but model config declares {}",
                tokenizer.vocab_size, config.vocab_size
            )));
        }

        // 4. Hardware discovery and memory footprint preflight check via TARA-HD
        let total_params: usize = weights.values().map(|v| v.len()).sum();
        let max_tensor_len = weights.values().map(|v| v.len()).max().unwrap_or(0);
        let mut hd_channel = tara_hd::ComputeChannel::with_options(
            &self.device,
            match self.precision {
                crate::cuda::TrainingPrecision::Fp16 => tara_hd::PrecisionMode::Fp16,
                crate::cuda::TrainingPrecision::Fp32 => tara_hd::PrecisionMode::Fp32,
                crate::cuda::TrainingPrecision::Auto => tara_hd::PrecisionMode::Auto,
            },
        )
        .map_err(|e| {
            TrainerError::Model(format!("TARA-HD hardware channel selection failed: {e}"))
        })?;

        // ── Automatic memory-aware hyperparameter configuration ──────────────────
        // Derive optimal batch_size, seq_len, grad_accum_steps, learning_rate, and
        // warmup_steps from actual device memory reported by the TARA-HD channel.
        // User-supplied values take priority; auto-derived values fill only None slots.
        let mut auto_cfg = tara_hd::AutoTrainingConfig::compute(
            &hd_channel,
            total_params,
            config.hidden_size,
            config.num_hidden_layers,
            samples.len(),
        );
        println!("{}", auto_cfg.telemetry);

        // Resolve effective precision from auto-config when it wasn't explicitly set.
        let is_gpu = hd_channel.is_gpu();
        let is_f16 = match self.precision {
            crate::cuda::TrainingPrecision::Fp16 => true,
            crate::cuda::TrainingPrecision::Fp32 => false,
            crate::cuda::TrainingPrecision::Auto => {
                matches!(
                    auto_cfg.precision,
                    tara_hd::PrecisionMode::Fp16 | tara_hd::PrecisionMode::Bf16
                )
            }
        };

        // Use auto-derived batch_size and seq_len for memory preflight estimation so
        // the preflight correctly reflects the training configuration that will actually run.
        let effective_batch_for_preflight = self.batch_size.unwrap_or(auto_cfg.batch_size);
        let effective_seq_for_preflight = auto_cfg.seq_len;

        let footprint = Self::estimate_memory_footprint_mb_detailed(
            total_params,
            is_gpu,
            is_f16,
            max_tensor_len,
            effective_batch_for_preflight,
            effective_seq_for_preflight,
            config.hidden_size,
        );
        // Effective memory safety ceilings:
        // Host RAM admission control is based strictly on effective cgroup memory limit and available host RAM.
        // Device VRAM admission control is based strictly on physical accelerator VRAM capacity and available budget.
        // User overrides via --max-memory-mb apply to host memory, unless TARA_FORCE_UNSAFE_MEMORY=1 is explicitly declared.
        let default_max_memory_mb: f64 = 65536.0;
        let force_unsafe_memory = std::env::var("TARA_FORCE_UNSAFE_MEMORY")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false);
        let effective_host_max_memory_mb = if force_unsafe_memory {
            self.max_memory_mb
        } else if (self.max_memory_mb - default_max_memory_mb).abs() < 1.0 {
            auto_cfg.max_memory_mb
        } else {
            self.max_memory_mb.min(auto_cfg.max_memory_mb)
        };

        let effective_device_max_vram_mb = if force_unsafe_memory {
            f64::MAX
        } else if auto_cfg.max_device_vram_mb > 0.0 {
            auto_cfg.max_device_vram_mb
        } else if is_gpu {
            (auto_cfg.device_vram_total_mb * 0.90).max(4096.0)
        } else {
            0.0
        };

        let host_summary_str = if auto_cfg.cgroup_limit_mb > 0.0 {
            format!(
                "Host RAM: {:.1} MB avail / {:.1} MB total | Cgroup: {:.1} MB avail / {:.1} MB limit",
                auto_cfg.host_ram_available_mb,
                auto_cfg.host_ram_total_mb,
                auto_cfg.cgroup_available_mb,
                auto_cfg.cgroup_limit_mb
            )
        } else {
            format!(
                "Host RAM: {:.1} MB avail / {:.1} MB total",
                auto_cfg.host_ram_available_mb, auto_cfg.host_ram_total_mb
            )
        };

        println!(
            "[Memory Preflight] Host RAM: projected {:.1} MB (Safety ceiling: {:.1} MB | {host_summary_str})",
            footprint.host_ram_mb, effective_host_max_memory_mb,
        );
        if is_gpu {
            println!(
                "[Memory Preflight] Device VRAM: projected {:.1} MB (Safety ceiling: {:.1} MB | Total: {:.1} MB, Avail: {:.1} MB)",
                footprint.device_vram_mb,
                effective_device_max_vram_mb,
                auto_cfg.device_vram_total_mb,
                auto_cfg.device_vram_available_mb,
            );
        }
        println!("[Memory Preflight] Detailed breakdown: {}", footprint.breakdown);

        // Preflight Admission Gate 1: Host System RAM / cgroup container memory limit
        if footprint.host_ram_mb > effective_host_max_memory_mb {
            let host_ram_str = if auto_cfg.host_ram_total_mb > 0.0 {
                format!(
                    "{:.0} MB (avail: {:.0} MB)",
                    auto_cfg.host_ram_total_mb, auto_cfg.host_ram_available_mb
                )
            } else {
                "unknown".to_string()
            };
            let cgroup_detail = if auto_cfg.cgroup_limit_mb > 0.0 {
                format!(
                    ", cgroup limit: {:.0} MB, cgroup avail: {:.0} MB",
                    auto_cfg.cgroup_limit_mb, auto_cfg.cgroup_available_mb
                )
            } else {
                String::new()
            };
            return Err(TrainerError::Model(format!(
                "Projected Host RAM requirement {:.1} MB exceeds host safety ceiling \
                {effective_host_max_memory_mb:.1} MB (detected host RAM: {host_ram_str}{cgroup_detail}). \
                Breakdown: {}",
                footprint.host_ram_mb,
                footprint.breakdown,
            )));
        }

        // Preflight Admission Gate 2: Device Accelerator VRAM (when GPU active)
        if is_gpu && footprint.device_vram_mb > effective_device_max_vram_mb {
            return Err(TrainerError::Model(format!(
                "Projected Device VRAM requirement {:.1} MB exceeds device safety ceiling \
                {effective_device_max_vram_mb:.1} MB (detected VRAM total: {:.0} MB, avail: {:.0} MB). \
                Breakdown: {}",
                footprint.device_vram_mb,
                auto_cfg.device_vram_total_mb,
                auto_cfg.device_vram_available_mb,
                footprint.breakdown,
            )));
        }

        // 5. Tokenize training samples
        println!(
            "[Tokenizer] Tokenizing {} samples (vocabulary size {})...",
            samples.len(),
            config.vocab_size
        );
        let training_tokens: Vec<(Vec<u32>, usize)> = samples
            .iter()
            .filter_map(|(input, target)| {
                let input_ids = tokenizer.encode(input);
                let target_ids = tokenizer.encode(target);
                if input_ids.is_empty() || target_ids.is_empty() {
                    return None;
                }
                if input_ids
                    .iter()
                    .chain(target_ids.iter())
                    .any(|id| *id as usize >= config.vocab_size)
                {
                    return None;
                }
                let max_seq_default = tara_hd::CpuStreamHelper::new()
                    .recommend_chunk_size(config.max_position_embeddings, config.hidden_size);
                let max_seq_override = std::env::var("TARA_MAX_SEQ_LEN")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok());
                let max_len = max_seq_override
                    .unwrap_or(max_seq_default)
                    .clamp(2, config.max_position_embeddings);
                let input_start = input_ids.len().saturating_sub(max_len - 1);
                let mut tokens = input_ids[input_start..].to_vec();
                let target_start = tokens.len();
                tokens.extend(
                    target_ids
                        .into_iter()
                        .take(max_len.saturating_sub(tokens.len())),
                );
                (tokens.len() > target_start).then_some((tokens, target_start))
            })
            .collect();

        if training_tokens.is_empty() {
            return Err(TrainerError::Dataset(
                "samples contain no encodable prompt/target token pairs".into(),
            ));
        }
        drop(samples);
        println!(
            "[Tokenizer] Successfully tokenized {} sequence pairs.",
            training_tokens.len()
        );

        // 6. Split into Train (90%) and Held-out Validation (10%) to prevent evaluation contamination
        let total_seq_count = training_tokens.len();
        let (train_tokens, val_tokens) = if total_seq_count > 1 {
            let val_count = (total_seq_count / 10).clamp(1, total_seq_count - 1);
            let train = training_tokens[..total_seq_count - val_count].to_vec();
            let val = training_tokens[total_seq_count - val_count..].to_vec();
            (train, val)
        } else {
            (training_tokens.clone(), Vec::new())
        };

        if val_tokens.is_empty() {
            println!(
                "[Dataset Split] Partitioned {} train sequence(s). Held-out validation is unavailable (single-sequence boundary — zero self-contamination).",
                train_tokens.len()
            );
        } else {
            println!(
                "[Dataset Split] Partitioned {} train sequences and {} held-out validation sequences.",
                train_tokens.len(),
                val_tokens.len()
            );
        }

        // 7. Full-Network Backpropagation with AdamW and Gradient Accumulation
        // Initialize hardware acceleration device backend via TARA-HD channel
        let gpu_trainer_opt: Option<crate::cuda::CudaTrainer> = if hd_channel.is_gpu() {
            let dev_id = hd_channel.device().info().device_id as i32;
            match crate::cuda::CudaTrainer::new_with_precision(dev_id, self.precision) {
                Ok(mut t) => {
                    let dev = t.device().clone();
                    let (free, total) = t.get_vram_info().unwrap_or((0, 0));
                    println!("[Device] Selected: GPU Backend (NVIDIA {})", dev.name);
                    println!(
                        "         Compute Capability: {}.{}",
                        dev.compute_capability.0, dev.compute_capability.1
                    );
                    println!("         Precision: {:?}", t.precision());
                    println!(
                        "         VRAM: {:.2} MB free / {:.2} MB total",
                        free as f64 / (1024.0 * 1024.0),
                        total as f64 / (1024.0 * 1024.0)
                    );
                    // Refine auto-config using the live VRAM reading from the initialized
                    // CUDA context, which is more accurate than the pre-init device probe.
                    if let Some(refined) = auto_cfg.refine_with_live_vram(
                        free as u64,
                        total as u64,
                        total_params,
                        config.hidden_size,
                        config.num_hidden_layers,
                        train_tokens.len() + val_tokens.len(),
                    ) {
                        println!("{}", refined.telemetry);
                        auto_cfg = refined;
                    }
                    crate::cuda::gpu_trainer::log_flushed_memory_checkpoint(
                        "PRE_GPU_WEIGHT_REGISTRATION",
                        t.get_vram_info().ok(),
                    );
                    let reg_res = t.register_weights(&weights);
                    crate::cuda::gpu_trainer::log_flushed_memory_checkpoint(
                        "POST_GPU_WEIGHT_REGISTRATION",
                        t.get_vram_info().ok(),
                    );
                    match reg_res {
                        Ok(()) => Some(t),
                        Err(e) => {
                            if matches!(self.device, tara_hd::SelectionPolicy::Explicit(..)) {
                                return Err(TrainerError::Model(format!(
                                    "GPU buffer allocation failed under explicit device policy: {e}"
                                )));
                            }
                            if total_params >= 50_000_000 {
                                return Err(TrainerError::Model(format!(
                                    "GPU buffer allocation failed for large model ({total_params} params): {e}. \
                                     Silent CPU fallback is disabled under --device auto to prevent host memory exhaustion (OOM). \
                                     Explicitly specify --device cpu if CPU execution is genuinely intended."
                                )));
                            }
                            println!(
                                "[Device] GPU buffer allocation failed ({e}); falling back to CPU."
                            );
                            None
                        }
                    }
                }
                Err(e) => {
                    if matches!(self.device, tara_hd::SelectionPolicy::Explicit(..)) {
                        return Err(TrainerError::Model(format!(
                            "CUDA initialization failed for explicit device: {e}."
                        )));
                    }
                    if total_params >= 50_000_000 {
                        return Err(TrainerError::Model(format!(
                            "CUDA initialization/compilation failed for large model ({total_params} params): {e}. \
                             Silent CPU fallback is disabled under --device auto to prevent host memory exhaustion (OOM). \
                             Explicitly specify --device cpu if CPU execution is genuinely intended."
                        )));
                    }
                    println!("[Device] CUDA GPU not available ({e}); safely falling back to CPU.");
                    None
                }
            }
        } else if hd_channel.is_tpu() {
            println!(
                "[Device] Hardware backend 'TPU' ({}) active in TARA-HD channel; using CPU fallback.",
                hd_channel.device_name()
            );
            None
        } else {
            println!(
                "[Device] Selected: CPU Backend via TARA-HD ({})",
                hd_channel.device_name()
            );
            None
        };

        // Eagerly pre-allocate moments if running on CPU before moving weights ownership
        let mut optimizer = if gpu_trainer_opt.is_none() {
            DynamicAdamW::with_preallocated_moments(&weights)
        } else {
            DynamicAdamW::new()
        };

        // Construct canonical in-memory model by moving ownership of weights (zero duplicate memory!)
        let mut model = TaraForCausalLM::from_weights_move(weights, &config, &self.model_dir)
            .map_err(|e| TrainerError::Model(e.to_string()))?;

        // Connect CudaComputeDevice into TARA-HD ComputeChannel as Single Source of Truth
        let cuda_compute_dev_opt: Option<Arc<crate::cuda::CudaComputeDevice>> =
            if let Some(t) = gpu_trainer_opt {
                let slice_map = model.parameter_slice_map();
                let cuda_dev = Arc::new(crate::cuda::CudaComputeDevice::new(
                    hd_channel.device().info().clone(),
                    Arc::new(Mutex::new(t)),
                    slice_map,
                ));
                hd_channel.set_device(cuda_dev.clone());
                Some(cuda_dev)
            } else {
                None
            };

        // Compute baseline loss strictly across held-out validation sequences
        let initial_loss: Option<f64> = if val_tokens.is_empty() {
            println!(
                "[Preflight] Skipping baseline validation loss (held-out validation unavailable)."
            );
            None
        } else {
            // Optional debugging smoke test (allows 1-2 sample verification when requested)
            if std::env::var("TARA_VALIDATION_SMOKE_TEST")
                .map(|v| v == "1" || v == "true")
                .unwrap_or(false)
                || std::env::var("TARA_SMOKE_VAL")
                    .map(|v| v == "1" || v == "true")
                    .unwrap_or(false)
            {
                let smoke_count = 2.min(val_tokens.len());
                let smoke_val = &val_tokens[..smoke_count];
                println!("[Preflight Smoke Test] Running {smoke_count}-sample baseline validation smoke test...");
                let _ = compute_model_loss(&model, &hd_channel, smoke_val, "Preflight Smoke")?;
                println!("[Preflight Smoke Test] Smoke test passed!");
            }

            // Full validation across all held-out validation samples (or bounded by TARA_MAX_VAL_SAMPLES if explicitly set)
            let max_val_override = std::env::var("TARA_MAX_VAL_SAMPLES")
                .ok()
                .and_then(|v| v.parse::<usize>().ok());
            let active_val_tokens = if let Some(lim) = max_val_override {
                let lim = lim.clamp(1, val_tokens.len());
                println!("[Preflight] Limiting validation samples to {lim} per TARA_MAX_VAL_SAMPLES override");
                &val_tokens[..lim]
            } else {
                &val_tokens[..]
            };

            let loss = compute_model_loss(&model, &hd_channel, active_val_tokens, "Preflight")?;
            println!("[Preflight] Held-out validation baseline loss: {:.4}", loss);
            Some(loss)
        };

        let device_label = if hd_channel.is_gpu() {
            format!("GPU ({})", hd_channel.device_name())
        } else if hd_channel.is_tpu() {
            format!("TPU ({})", hd_channel.device_name())
        } else {
            format!("CPU ({})", hd_channel.device_name())
        };
        let cpu_stream_helper = hd_channel
            .device()
            .stream_helper()
            .cloned()
            .unwrap_or_default();
        // is_gpu was resolved earlier alongside auto_cfg computation.
        let precision_label = format!("{:?}", hd_channel.precision());
        // Hyperparameter resolution: user-supplied values are authoritative.
        // When the user has not explicitly set a value (None), the auto-derived value
        // from AutoTrainingConfig — computed from actual device memory via TARA-HD —
        // is used instead of a static fallback, enabling full zero-config training.
        let learning_rate = self.learning_rate.unwrap_or(auto_cfg.learning_rate);
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let weight_decay = self.weight_decay.unwrap_or(auto_cfg.weight_decay);
        let max_grad_norm = self.grad_clip_norm.unwrap_or(auto_cfg.grad_clip_norm);
        let warmup_steps = self.warmup_steps.unwrap_or(auto_cfg.warmup_steps);
        // accumulation_steps doubles as the gradient accumulation micro-batch count.
        let accumulation_steps = self.batch_size.unwrap_or(auto_cfg.batch_size);

        let mut samples_seen = 0usize;
        let mut optimizer_steps = 0usize;
        let mut epochs_completed = 0;
        let mut best_loss = initial_loss.unwrap_or(f64::INFINITY);

        // Check for resumable checkpoint state
        if self.resume {
            let explicit_resume = if let Some(ref r_from) = self.resume_from {
                let p = Path::new(r_from);
                if !p.exists() {
                    return Err(TrainerError::Model(format!(
                        "Explicit resume_from path '{}' does not exist",
                        r_from
                    )));
                }
                if !p.join("checkpoint_state.json").exists() {
                    return Err(TrainerError::Model(format!(
                        "Explicit resume_from path '{}' is missing checkpoint_state.json (not continuation-ready)",
                        r_from
                    )));
                }
                if !p.join("optimizer.safetensors").exists() {
                    return Err(TrainerError::Model(format!(
                        "Explicit resume_from path '{}' is missing optimizer.safetensors (not continuation-ready)",
                        r_from
                    )));
                }
                Some(r_from.clone())
            } else {
                None
            };

            let resume_path_str = explicit_resume
                .or_else(|| {
                    let m_path = Path::new(&self.model_dir);
                    if m_path.join("checkpoint_state.json").exists() {
                        Some(self.model_dir.clone())
                    } else {
                        None
                    }
                })
                .or_else(|| {
                    if let Some(ref cp_dir) = self.checkpoint_dir {
                        let cp_path = Path::new(cp_dir);
                        if cp_path.join("checkpoint_state.json").exists() {
                            Some(cp_dir.clone())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                });
            if let Some(ref resume_dir) = resume_path_str {
                let cp_path = Path::new(resume_dir);
                let state_file = cp_path.join("checkpoint_state.json");
                let has_model = cp_path.join("model.safetensors").exists()
                    || cp_path.join("model.safetensors.index.json").exists();
                if state_file.exists() && has_model {
                    println!(
                        "[Checkpoint] Resuming from existing checkpoint at '{}'...",
                        resume_dir
                    );
                    let state_str = std::fs::read_to_string(&state_file)?;
                    let state_json: serde_json::Value =
                        serde_json::from_str(&state_str).map_err(|e| {
                            TrainerError::Model(format!(
                                "Corrupt checkpoint_state.json in '{}': {}",
                                resume_dir, e
                            ))
                        })?;
                    let step_val = state_json
                        .get("optimizer_step")
                        .or_else(|| state_json.get("step"))
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| {
                            TrainerError::Model(format!(
                                "checkpoint_state.json in '{}' missing valid 'optimizer_step'",
                                resume_dir
                            ))
                        })?;
                    optimizer_steps = step_val as usize;
                    if let Some(s) = state_json.get("samples_seen").and_then(|v| v.as_u64()) {
                        samples_seen = s as usize;
                    } else {
                        samples_seen = optimizer_steps * accumulation_steps;
                    }
                    if let Some(e) = state_json.get("epoch").and_then(|v| v.as_u64()) {
                        epochs_completed = e as usize;
                    }
                    println!(
                        "[Checkpoint] Resumed at optimizer step {}, samples {}, epoch {}",
                        optimizer_steps, samples_seen, epochs_completed
                    );

                    let (resumed_weights, _) =
                        crate::safetensors::load_model_weights_with_purpose(resume_dir, "RESUME")?;
                    if let Some(ref dev) = cuda_compute_dev_opt {
                        let mut t = dev
                            .trainer()
                            .lock()
                            .map_err(|e| TrainerError::Model(e.to_string()))?;
                        let _ = t.register_weights(&resumed_weights);
                    }

                    // Restore and strictly validate optimizer state moments
                    let opt_file = cp_path.join("optimizer.safetensors");
                    if opt_file.exists() {
                        let (opt_tensors, _) = crate::safetensors::load_safetensors_with_shapes(
                            &opt_file.to_string_lossy(),
                        )
                        .map_err(|e| {
                            TrainerError::Model(format!(
                                "Corrupt optimizer.safetensors in '{}': {}",
                                resume_dir, e
                            ))
                        })?;
                        if opt_tensors.is_empty() {
                            return Err(TrainerError::Model(format!(
                                "optimizer.safetensors in '{}' contains zero tensors",
                                resume_dir
                            )));
                        }
                        let mut loaded_moments: HashMap<String, (Vec<f32>, Vec<f32>)> =
                            HashMap::new();
                        for (k, v) in opt_tensors {
                            if let Some(base_name) = k.strip_suffix(".adam_m") {
                                let entry = loaded_moments
                                    .entry(base_name.to_string())
                                    .or_insert_with(|| (Vec::new(), Vec::new()));
                                entry.0 = v;
                            } else if let Some(base_name) = k.strip_suffix(".adam_v") {
                                let entry = loaded_moments
                                    .entry(base_name.to_string())
                                    .or_insert_with(|| (Vec::new(), Vec::new()));
                                entry.1 = v;
                            }
                        }
                        for (param_name, w_vec) in &resumed_weights {
                            if let Some((m_vec, v_vec)) = loaded_moments.get(param_name) {
                                if m_vec.len() != w_vec.len() || v_vec.len() != w_vec.len() {
                                    return Err(TrainerError::Model(format!(
                                        "Optimizer moment shape mismatch for '{}': weight len={}, m len={}, v len={}",
                                        param_name,
                                        w_vec.len(),
                                        m_vec.len(),
                                        v_vec.len()
                                    )));
                                }
                            }
                        }
                        if let Some(ref dev) = cuda_compute_dev_opt {
                            let mut t = dev
                                .trainer()
                                .lock()
                                .map_err(|e| TrainerError::Model(e.to_string()))?;
                            t.load_optimizer_state(&loaded_moments).map_err(|e| {
                                TrainerError::Model(format!(
                                    "Failed to load GPU optimizer state from '{}': {e}",
                                    resume_dir
                                ))
                            })?;
                            t.set_step_count(optimizer_steps as u64);
                        }
                        optimizer.load_state(loaded_moments);
                        optimizer.set_step(optimizer_steps as u64);
                        model = TaraForCausalLM::from_weights_move(
                            resumed_weights,
                            &config,
                            &self.model_dir,
                        )
                        .map_err(|e| TrainerError::Model(e.to_string()))?;
                        if let Some(ref dev) = cuda_compute_dev_opt {
                            dev.update_slice_map(model.parameter_slice_map());
                        }
                        println!("[Checkpoint] Successfully restored optimizer moments and step count ({optimizer_steps}).");
                    } else if self.resume_from.is_some() {
                        return Err(TrainerError::Model(format!(
                            "Explicit resume_from checkpoint '{}' missing optimizer.safetensors",
                            resume_dir
                        )));
                    }
                }
            }
        }

        let mut initial_sample_skip = if self.resume && !train_tokens.is_empty() {
            samples_seen % train_tokens.len()
        } else {
            0
        };

        let train_start_instant = std::time::Instant::now();
        let mut running_avg_loss: Option<f32> = None;
        let mut ema_step_duration: Option<f64> = None;
        let stage_env = std::env::var("STAGE_NAME").ok();
        let stage_label = self
            .stage_name
            .as_deref()
            .or(stage_env.as_deref())
            .unwrap_or("Stage")
            .to_string();

        let start_epoch = epochs_completed;
        for epoch_idx in start_epoch..max_epochs {
            let mut accum_grads: HashMap<String, Vec<f32>> = HashMap::new();
            let mut accum_count = 0usize;
            let mut batch_loss_sum = 0.0f32;
            let mut batch_tokens_count = 0usize;
            let mut batch_bytes_count = 0usize;
            let mut step_start_instant = std::time::Instant::now();

            let skip_count = initial_sample_skip;
            initial_sample_skip = 0;

            println!(
                "[{stage_label}] Starting Epoch {}/{} ({} samples in pool, accumulation: {} micro-steps/optimizer step, device: {})...",
                epoch_idx + 1,
                max_epochs,
                train_tokens.len(),
                accumulation_steps,
                device_label
            );
            let _ = std::io::stdout().flush();

            for (sample_offset, (tokens, target_start)) in
                train_tokens.iter().skip(skip_count).enumerate()
            {
                if let Some(max_s) = self.max_steps {
                    if optimizer_steps >= max_s {
                        break;
                    }
                }
                let sample_idx_in_epoch = skip_count + sample_offset;
                let sample_in_epoch = sample_idx_in_epoch + 1;
                let sample_start = std::time::Instant::now();
                let stream_chunk_size =
                    cpu_stream_helper.recommend_chunk_size(tokens.len(), config.hidden_size);
                let active_seq_limit = std::env::var("TARA_MAX_SEQ_LEN")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or_else(|| {
                        if hd_channel.is_gpu() {
                            let vram_free_mb = cuda_compute_dev_opt
                                .as_ref()
                                .and_then(|d| d.trainer().lock().ok()?.get_vram_info().ok())
                                .map(|(free, _)| free as f64 / (1024.0 * 1024.0))
                                .unwrap_or(8192.0);
                            if vram_free_mb < 2048.0 {
                                512
                            } else if vram_free_mb < 4096.0 {
                                1024
                            } else {
                                2048.min(config.max_position_embeddings)
                            }
                        } else {
                            stream_chunk_size
                        }
                    });
                let (active_tokens, raw_target_start) = if tokens.len() > active_seq_limit {
                    println!(
                        "[{stage_label} | Bounded Stream] Bound sequence ({} tokens) to {} tokens to preserve compute latency & RAM",
                        tokens.len(), active_seq_limit
                    );
                    (
                        &tokens[..active_seq_limit],
                        (*target_start).min(active_seq_limit),
                    )
                } else {
                    (&tokens[..], *target_start)
                };
                let seq_len = active_tokens.len();
                let sample_pct =
                    (sample_in_epoch as f64 / train_tokens.len() as f64 * 100.0).min(100.0);
                println!(
                    "[{stage_label} | Epoch {}/{} | Sample {}/{} ({:.1}%)] Starting forward/backward (seq_len: {}, micro-step {}/{})...",
                    epoch_idx + 1,
                    max_epochs,
                    sample_in_epoch,
                    train_tokens.len(),
                    sample_pct,
                    seq_len,
                    accum_count + 1,
                    accumulation_steps
                );
                let _ = std::io::stdout().flush();

                let (layer_inputs, layer_activations, final_normed, logits) =
                    model.forward_with_cache_channel_saved(active_tokens, &hd_channel);

                // Compute d_logits for supervised causal cross-entropy (Rayon-parallelized across token positions)
                let vs = config.vocab_size;
                let eff_target_start = raw_target_start.max(1);
                let mut d_logits = vec![0.0f32; seq_len * vs];

                let (unnorm_sample_loss, valid_targets) = d_logits
                    .par_chunks_exact_mut(vs)
                    .enumerate()
                    .map(|(predictor, grad_slice)| {
                        let pos = predictor + 1;
                        if pos >= eff_target_start && pos < seq_len {
                            let target = tokens[pos] as usize;
                            if target < vs {
                                let logit_start = predictor * vs;
                                let logits_slice = &logits[logit_start..logit_start + vs];
                                let max_logit = logits_slice
                                    .iter()
                                    .copied()
                                    .fold(f32::NEG_INFINITY, f32::max);
                                let mut sum_exp = 0.0f32;
                                for (v, &l) in logits_slice.iter().enumerate() {
                                    let exp_val = (l - max_logit).exp();
                                    grad_slice[v] = exp_val;
                                    sum_exp += exp_val;
                                }

                                if sum_exp > 0.0 && sum_exp.is_finite() {
                                    let inv_sum = 1.0 / sum_exp;
                                    for val in grad_slice.iter_mut().take(vs) {
                                        *val *= inv_sum;
                                    }
                                    let loss = sum_exp.ln() - (logits_slice[target] - max_logit);
                                    grad_slice[target] -= 1.0;
                                    return (loss, 1usize);
                                }
                            }
                        }
                        (0.0f32, 0usize)
                    })
                    .reduce(|| (0.0f32, 0usize), |a, b| (a.0 + b.0, a.1 + b.1));

                // Drop large forward logits buffer immediately — reclaims ~570 MB host RAM
                drop(logits);

                if valid_targets > 0 {
                    let sample_loss = unnorm_sample_loss / valid_targets as f32;
                    batch_loss_sum += sample_loss;
                    batch_tokens_count += valid_targets;
                    let target_bytes: usize = active_tokens[eff_target_start..]
                        .iter()
                        .map(|id| tokenizer.id_to_token.get(id).map(|s| s.len()).unwrap_or(1))
                        .sum();
                    batch_bytes_count += target_bytes.max(1);

                    // Normalise by number of target tokens
                    let norm_factor = 1.0 / valid_targets as f32;
                    for g in d_logits.iter_mut() {
                        *g *= norm_factor;
                    }

                    // Stream gradients directly through TARA-HD channel
                    if hd_channel.is_gpu() {
                        // Hardware-accelerated LM Head backward projection through TARA-HD channel:
                        // Accumulates gradients directly on GPU, avoiding dense 819.2 MB host buffer download
                        let (_lm_norm_sq, gpu_dnorm) = hd_channel
                            .backward_lm_head_accumulate(
                                &d_logits,
                                &final_normed,
                                seq_len,
                                config.hidden_size,
                                config.vocab_size,
                            )
                            .map_err(|e| {
                                TrainerError::Model(format!("Hardware backward failed: {e}"))
                            })?;

                        // Drop logits and norm activation buffers immediately
                        drop(d_logits);
                        drop(final_normed);

                        // Backward through RMSNorm with hardware-computed d_final_normed
                        let (mut dx, d_norm) = model.norm.backward(
                            &gpu_dnorm,
                            layer_inputs.last().unwrap(),
                            seq_len,
                            config.hidden_size,
                        );
                        drop(gpu_dnorm);

                        // Accumulate Norm gradient into hardware buffer and drop host buffer immediately
                        hd_channel
                            .accumulate_gradient("model.norm.weight", &d_norm)
                            .map_err(|e| {
                                TrainerError::Model(format!("Gradient accumulation failed: {e}"))
                            })?;
                        drop(d_norm);

                        // Backward through decoder layers in reverse order:
                        // Uses pre-computed activations to eliminate forward recomputation during backward!
                        for l in (0..model.layers.len()).rev() {
                            let l_grads = model.layers[l].backward_channel(
                                &dx,
                                &layer_inputs[l],
                                &layer_activations[l],
                                seq_len,
                                &hd_channel,
                            );

                            let prefix = format!("model.layers.{}", l);
                            hd_channel
                                .accumulate_gradient(
                                    &format!("{}.input_layernorm.weight", prefix),
                                    &l_grads.d_input_layernorm,
                                )
                                .map_err(|e| {
                                    TrainerError::Model(format!(
                                        "Gradient accumulation failed: {e}"
                                    ))
                                })?;
                            hd_channel
                                .accumulate_gradient(
                                    &format!("{}.self_attn.q_proj.weight", prefix),
                                    &l_grads.d_q_proj,
                                )
                                .map_err(|e| {
                                    TrainerError::Model(format!(
                                        "Gradient accumulation failed: {e}"
                                    ))
                                })?;
                            hd_channel
                                .accumulate_gradient(
                                    &format!("{}.self_attn.k_proj.weight", prefix),
                                    &l_grads.d_k_proj,
                                )
                                .map_err(|e| {
                                    TrainerError::Model(format!(
                                        "Gradient accumulation failed: {e}"
                                    ))
                                })?;
                            hd_channel
                                .accumulate_gradient(
                                    &format!("{}.self_attn.v_proj.weight", prefix),
                                    &l_grads.d_v_proj,
                                )
                                .map_err(|e| {
                                    TrainerError::Model(format!(
                                        "Gradient accumulation failed: {e}"
                                    ))
                                })?;
                            hd_channel
                                .accumulate_gradient(
                                    &format!("{}.self_attn.o_proj.weight", prefix),
                                    &l_grads.d_o_proj,
                                )
                                .map_err(|e| {
                                    TrainerError::Model(format!(
                                        "Gradient accumulation failed: {e}"
                                    ))
                                })?;
                            hd_channel
                                .accumulate_gradient(
                                    &format!("{}.post_attention_layernorm.weight", prefix),
                                    &l_grads.d_post_attention_layernorm,
                                )
                                .map_err(|e| {
                                    TrainerError::Model(format!(
                                        "Gradient accumulation failed: {e}"
                                    ))
                                })?;
                            hd_channel
                                .accumulate_gradient(
                                    &format!("{}.mlp.gate_proj.weight", prefix),
                                    &l_grads.d_gate_proj,
                                )
                                .map_err(|e| {
                                    TrainerError::Model(format!(
                                        "Gradient accumulation failed: {e}"
                                    ))
                                })?;
                            hd_channel
                                .accumulate_gradient(
                                    &format!("{}.mlp.up_proj.weight", prefix),
                                    &l_grads.d_up_proj,
                                )
                                .map_err(|e| {
                                    TrainerError::Model(format!(
                                        "Gradient accumulation failed: {e}"
                                    ))
                                })?;
                            hd_channel
                                .accumulate_gradient(
                                    &format!("{}.mlp.down_proj.weight", prefix),
                                    &l_grads.d_down_proj,
                                )
                                .map_err(|e| {
                                    TrainerError::Model(format!(
                                        "Gradient accumulation failed: {e}"
                                    ))
                                })?;
                            dx = l_grads.dx;
                        }
                        drop(layer_inputs);
                        drop(layer_activations);

                        // Accumulate token embeddings gradient sparsely — ZERO 819.2 MB host buffer allocation!
                        hd_channel
                            .accumulate_sparse_embedding_gradient(active_tokens, &dx, config.hidden_size)
                            .map_err(|e| {
                                TrainerError::Model(format!("Sparse embedding gradient accumulation failed: {e}"))
                            })?;
                        drop(dx);

                        // Single driver synchronization per micro-step
                        hd_channel.synchronize().map_err(|e| {
                            TrainerError::Model(format!("Device synchronize failed: {e}"))
                        })?;
                    } else {
                        // Full-network CPU analytical backward pass: into_map() moves buffers with zero cloning
                        let model_grads =
                            model.backward(active_tokens, &layer_inputs, &final_normed, &d_logits);
                        drop(d_logits);
                        drop(final_normed);
                        drop(layer_inputs);
                        drop(layer_activations);
                        for (k, v) in model_grads.into_map() {
                            let entry = accum_grads
                                .entry(k)
                                .or_insert_with(|| vec![0.0f32; v.len()]);
                            for i in 0..v.len() {
                                entry[i] += v[i];
                            }
                        }
                    }

                    accum_count += 1;
                    samples_seen += 1;

                    // Update running average loss
                    running_avg_loss = Some(match running_avg_loss {
                        None => sample_loss,
                        Some(prev) => 0.95 * prev + 0.05 * sample_loss,
                    });
                    let running_loss_display = running_avg_loss.unwrap_or(sample_loss);

                    let sample_time = sample_start.elapsed().as_secs_f32();
                    let sample_throughput = (valid_targets as f32) / sample_time.max(1e-4);

                    let snap = crate::cuda::gpu_trainer::capture_memory_snapshot(
                        cuda_compute_dev_opt
                            .as_ref()
                            .and_then(|d| d.trainer().lock().ok()?.get_vram_info().ok()),
                    );
                    let vram_str = match (snap.gpu_used_vram_mb, snap.gpu_total_vram_mb) {
                        (Some(u), Some(t)) => format!(" | GPU VRAM: {:.1}/{:.1} MB", u, t),
                        _ => String::new(),
                    };

                    println!(
                        "[{stage_label} | Epoch {}/{} | Sample {}/{} ({:.1}%)] Micro-step {}/{} | Loss: {:.4} (Running: {:.4}) | Tok/s: {:.1} | Time: {:.2}s | Host RSS: {:.1} MB{}",
                        epoch_idx + 1,
                        max_epochs,
                        sample_in_epoch,
                        train_tokens.len(),
                        sample_pct,
                        accum_count,
                        accumulation_steps,
                        sample_loss,
                        running_loss_display,
                        sample_throughput,
                        sample_time,
                        snap.process_rss_mb,
                        vram_str
                    );
                    let _ = std::io::stdout().flush();
                } else {
                    drop(d_logits);
                    drop(final_normed);
                    drop(layer_inputs);
                    drop(layer_activations);
                }

                let dynamic_token_budget = accumulation_steps * stream_chunk_size.max(256);
                let should_step = accum_count >= accumulation_steps
                    || (batch_tokens_count >= dynamic_token_budget && accum_count >= 1);

                if should_step {
                    let mut step_taken = false;
                    let next_step = optimizer_steps + 1;
                    let effective_lr = if warmup_steps > 0 && next_step <= warmup_steps {
                        learning_rate * (next_step as f32 / warmup_steps as f32)
                    } else {
                        learning_rate
                    };

                    let is_tied = config.is_tied();
                    let mut weights = model.into_weights_map();

                    // Calculate Continual Learning (EWC) penalty & gradient
                    let mut ewc_loss = 0.0f32;
                    if let (Some(ref anchors), Some(ref importance)) =
                        (&self.anchor_weights, &self.ewc_importance)
                    {
                        for (k, w) in &weights {
                            if let (Some(a), Some(f)) = (anchors.get(k), importance.get(k)) {
                                let n = w.len().min(a.len()).min(f.len());
                                for i in 0..n {
                                    let diff = w[i] - a[i];
                                    ewc_loss += 0.5 * self.ewc_lambda * f[i] * diff * diff;
                                }
                            }
                        }
                    }

                    if let Some(ref dev) = cuda_compute_dev_opt {
                        let mut gpu_trainer = dev
                            .trainer()
                            .lock()
                            .map_err(|e| TrainerError::Model(e.to_string()))?;
                        // Inject EWC gradient into accumulated GPU buffers
                        if let (Some(ref anchors), Some(ref importance)) =
                            (&self.anchor_weights, &self.ewc_importance)
                        {
                            for (k, w) in &weights {
                                if let (Some(a), Some(f)) = (anchors.get(k), importance.get(k)) {
                                    let n = w.len().min(a.len()).min(f.len());
                                    let mut ewc_g = vec![0.0f32; n];
                                    for i in 0..n {
                                        let diff = w[i] - a[i];
                                        ewc_g[i] =
                                            (self.ewc_lambda * f[i] * diff) * accum_count as f32;
                                    }
                                    let _ = gpu_trainer.accumulate_gradient(k, &ewc_g);
                                }
                            }
                        }

                        // GPU-accelerated gradient scaling and AdamW step
                        gpu_trainer
                            .scale_accumulated_gradients(accum_count)
                            .map_err(|e| {
                                TrainerError::Model(format!("GPU gradient scaling failed: {e}"))
                            })?;
                        let stepped = gpu_trainer
                            .step_adamw_with_weights(
                                &mut weights,
                                effective_lr,
                                beta1,
                                beta2,
                                eps,
                                weight_decay,
                                max_grad_norm,
                            )
                            .map_err(|e| {
                                TrainerError::Model(format!("GPU AdamW step failed: {e}"))
                            })?;
                        if stepped {
                            step_taken = true;
                            if optimizer_steps == 0 {
                                println!(
                                    "[Train] Verified first training optimizer step executed on CUDA GPU (k_adamw_step_f16, step_count={})",
                                    gpu_trainer.get_step_count()
                                );
                                crate::cuda::gpu_trainer::log_flushed_memory_checkpoint(
                                    "FIRST_OPTIMIZER_UPDATE",
                                    gpu_trainer.get_vram_info().ok(),
                                );
                            }
                        }

                        if !gpu_trainer.is_streaming() {
                            gpu_trainer
                                .download_weights_into(&mut weights)
                                .map_err(|e| {
                                    TrainerError::Model(format!("GPU weights download failed: {e}"))
                                })?;
                        }
                    } else {
                        // CPU AdamW step with EWC gradient penalty
                        let factor = 1.0 / accum_count as f32;
                        for g in accum_grads.values_mut() {
                            for val in g.iter_mut() {
                                *val *= factor;
                            }
                        }
                        if is_tied {
                            if let Some(lm_g) = accum_grads.get("lm_head.weight").cloned() {
                                if let Some(emb_g) = accum_grads.get_mut("model.embed_tokens.weight") {
                                    for (e, l) in emb_g.iter_mut().zip(lm_g.iter()) {
                                        *e += *l;
                                    }
                                }
                            }
                        }
                        if let (Some(ref anchors), Some(ref importance)) =
                            (&self.anchor_weights, &self.ewc_importance)
                        {
                            for (k, g) in accum_grads.iter_mut() {
                                if let (Some(w), Some(a), Some(f)) =
                                    (weights.get(k), anchors.get(k), importance.get(k))
                                {
                                    let n = g.len().min(w.len()).min(a.len()).min(f.len());
                                    for i in 0..n {
                                        let diff = w[i] - a[i];
                                        g[i] += self.ewc_lambda * f[i] * diff;
                                    }
                                }
                            }
                        }
                        crate::cuda::gpu_trainer::log_flushed_memory_checkpoint(
                            "Before CPU Optimizer Step",
                            None,
                        );
                        let stepped = optimizer.step(
                            &mut weights,
                            &accum_grads,
                            &AdamWHyperparams {
                                lr: effective_lr,
                                beta1,
                                beta2,
                                eps,
                                weight_decay,
                                max_grad_norm,
                            },
                        );
                        if is_tied {
                            if let Some(emb_w) = weights.get("model.embed_tokens.weight").cloned() {
                                weights.insert("lm_head.weight".to_string(), emb_w);
                            }
                        }
                        crate::cuda::gpu_trainer::log_flushed_memory_checkpoint(
                            "After CPU Optimizer Step",
                            None,
                        );
                        if stepped {
                            step_taken = true;
                        }
                        accum_grads.clear();
                    }

                    if step_taken {
                        optimizer_steps += 1;
                    }

                    let batch_lm_loss = if accum_count > 0 {
                        batch_loss_sum / accum_count as f32
                    } else {
                        0.0f32
                    };
                    let total_step_loss = batch_lm_loss + ewc_loss;
                    let ppl = (batch_lm_loss as f64).min(50.0).exp();
                    let bpb = if batch_bytes_count > 0 {
                        ((batch_lm_loss as f64) / std::f64::consts::LN_2)
                            * (batch_tokens_count as f64)
                            / (batch_bytes_count as f64)
                    } else {
                        (batch_lm_loss as f64) / std::f64::consts::LN_2
                    };
                    let grad_norm = if let Some(ref dev) = cuda_compute_dev_opt {
                        dev.trainer()
                            .lock()
                            .map(|t| t.last_grad_norm())
                            .unwrap_or(0.0)
                    } else {
                        optimizer.last_grad_norm()
                    };
                    let step_duration = step_start_instant.elapsed();
                    let batch_time_sec = step_duration.as_secs_f32().max(1e-6);
                    let throughput = batch_tokens_count as f32 / batch_time_sec;
                    let step_duration_secs = step_duration.as_secs_f64();

                    ema_step_duration = Some(match ema_step_duration {
                        None => step_duration_secs,
                        Some(prev) => 0.8 * prev + 0.2 * step_duration_secs,
                    });

                    let step_pct_str = if let Some(max_s) = self.max_steps {
                        format!(
                            " ({:.1}%)",
                            (optimizer_steps as f64 / max_s as f64 * 100.0).min(100.0)
                        )
                    } else {
                        String::new()
                    };

                    let eta_str =
                        if let (Some(max_s), Some(ema_sec)) = (self.max_steps, ema_step_duration) {
                            let rem_steps = max_s.saturating_sub(optimizer_steps);
                            let eta_sec = rem_steps as f64 * ema_sec;
                            format!(" | ETA: {}", Self::format_duration_hhmmss(eta_sec))
                        } else {
                            String::new()
                        };
                    let elapsed_str =
                        Self::format_duration_hhmmss(train_start_instant.elapsed().as_secs_f64());

                    let snap = crate::cuda::gpu_trainer::capture_memory_snapshot(
                        cuda_compute_dev_opt
                            .as_ref()
                            .and_then(|d| d.trainer().lock().ok()?.get_vram_info().ok()),
                    );
                    let vram_str = match (snap.gpu_used_vram_mb, snap.gpu_total_vram_mb) {
                        (Some(u), Some(t)) => format!(" | GPU VRAM: {:.1}/{:.1} MB", u, t),
                        _ => String::new(),
                    };

                    let max_str = self
                        .max_steps
                        .map(|m| m.to_string())
                        .unwrap_or_else(|| "∞".to_string());
                    let running_loss_display = running_avg_loss.unwrap_or(batch_lm_loss);

                    println!(
                            "[{stage_label} | Step {:>4}/{}{}] Total Loss: {:.4} | LM Loss: {:.4} (Running: {:.4}) | PPL: {:.2} | BPB: {:.4} | Aux/EWC: {:.4} | Grad Norm: {:.4} | LR: {:.6} | Throughput: {:.1} tok/s | Step Time: {:.2}s | Elapsed: {}{} | Host RSS: {:.1} MB{}",
                            optimizer_steps,
                            max_str,
                            step_pct_str,
                            total_step_loss,
                            batch_lm_loss,
                            running_loss_display,
                            ppl,
                            bpb,
                            ewc_loss,
                            grad_norm,
                            effective_lr,
                            throughput,
                            batch_time_sec,
                            elapsed_str,
                            eta_str,
                            snap.process_rss_mb,
                            vram_str
                        );
                    let _ = std::io::stdout().flush();

                    // Reset step accumulator counters
                    accum_count = 0;
                    batch_loss_sum = 0.0;
                    batch_tokens_count = 0;
                    batch_bytes_count = 0;
                    step_start_instant = std::time::Instant::now();

                    // Reconstruct model in-memory with updated weights without cloning
                    model = TaraForCausalLM::from_weights_move(weights, &config, &self.model_dir)
                        .map_err(|e| TrainerError::Model(e.to_string()))?;
                    if let Some(ref dev) = cuda_compute_dev_opt {
                        dev.update_slice_map(model.parameter_slice_map());
                    }

                    // Dynamic milestone and periodic persistent checkpoint saving
                    if let Some(ref cp_dir) = self.checkpoint_dir {
                        let cp_interval = self.checkpoint_interval.unwrap_or(usize::MAX);
                        let is_interval_step =
                            optimizer_steps > 0 && optimizer_steps.is_multiple_of(cp_interval);
                        let should_eval = is_interval_step
                            || optimizer_steps.is_multiple_of(25)
                            || (self.checkpoint_interval.is_none()
                                && optimizer_steps.is_multiple_of(10));
                        let current_eval_loss = if val_tokens.is_empty() || !should_eval {
                            None
                        } else {
                            compute_model_loss(&model, &hd_channel, &val_tokens, "Step Eval").ok()
                        };
                        let is_best = match current_eval_loss {
                            Some(loss) if loss < best_loss => {
                                best_loss = loss;
                                true
                            }
                            _ => false,
                        };

                        if is_interval_step || is_best {
                            let opt_moments = if let Some(ref dev) = cuda_compute_dev_opt {
                                dev.trainer()
                                    .lock()
                                    .map(|t| t.export_optimizer_state().unwrap_or_default())
                                    .unwrap_or_default()
                            } else {
                                optimizer.export_state()
                            };
                            let weights = model.into_weights_map();
                            crate::cuda::gpu_trainer::log_flushed_memory_checkpoint(
                                "PRE_CHECKPOINT_SERIALIZATION",
                                cuda_compute_dev_opt
                                    .as_ref()
                                    .and_then(|d| d.trainer().lock().ok()?.get_vram_info().ok()),
                            );
                            Self::save_checkpoint_state_bundle(CheckpointBundleConfig {
                                cp_dir,
                                model_dir: &self.model_dir,
                                weights: &weights,
                                config: &config,
                                opt_moments,
                                optimizer_steps,
                                samples_seen,
                                epoch_idx,
                                current_eval_loss,
                                best_loss,
                                is_best,
                                val_tokens_empty: val_tokens.is_empty(),
                            })?;
                            crate::cuda::gpu_trainer::log_flushed_memory_checkpoint(
                                "POST_CHECKPOINT_SERIALIZATION",
                                cuda_compute_dev_opt
                                    .as_ref()
                                    .and_then(|d| d.trainer().lock().ok()?.get_vram_info().ok()),
                            );
                            model = TaraForCausalLM::from_weights_move(
                                weights,
                                &config,
                                &self.model_dir,
                            )
                            .map_err(|e| TrainerError::Model(e.to_string()))?;
                            if let Some(ref dev) = cuda_compute_dev_opt {
                                dev.update_slice_map(model.parameter_slice_map());
                            }
                            let eval_str = current_eval_loss
                                .map(|l| format!("{:.4}", l))
                                .unwrap_or_else(|| "N/A".to_string());
                            let best_str = if val_tokens.is_empty() {
                                "N/A".to_string()
                            } else {
                                format!("{:.4}", best_loss)
                            };
                            println!(
                                    "[{stage_label} | Checkpoint Saved] Step {} (Epoch {}) | Eval Loss: {} | Best Loss: {} | Path: {} | Sync: GDrive",
                                    optimizer_steps,
                                    epoch_idx + 1,
                                    eval_str,
                                    best_str,
                                    cp_dir
                                );
                            let _ = std::io::stdout().flush();
                        }
                    }
                }
            }

            // Flush remaining accumulated gradients at epoch boundary
            if accum_count > 0 {
                let mut step_taken = false;
                let next_step = optimizer_steps + 1;
                let effective_lr = if warmup_steps > 0 && next_step <= warmup_steps {
                    learning_rate * (next_step as f32 / warmup_steps as f32)
                } else {
                    learning_rate
                };

                let is_tied = config.is_tied();
                let mut weights = model.into_weights_map();

                let mut ewc_loss = 0.0f32;
                if let (Some(ref anchors), Some(ref importance)) =
                    (&self.anchor_weights, &self.ewc_importance)
                {
                    for (k, w) in &weights {
                        if let (Some(a), Some(f)) = (anchors.get(k), importance.get(k)) {
                            let n = w.len().min(a.len()).min(f.len());
                            for i in 0..n {
                                let diff = w[i] - a[i];
                                ewc_loss += 0.5 * self.ewc_lambda * f[i] * diff * diff;
                            }
                        }
                    }
                }

                if let Some(ref dev) = cuda_compute_dev_opt {
                    let mut gpu_trainer = dev
                        .trainer()
                        .lock()
                        .map_err(|e| TrainerError::Model(e.to_string()))?;
                    if let (Some(ref anchors), Some(ref importance)) =
                        (&self.anchor_weights, &self.ewc_importance)
                    {
                        for (k, w) in &weights {
                            if let (Some(a), Some(f)) = (anchors.get(k), importance.get(k)) {
                                let n = w.len().min(a.len()).min(f.len());
                                let mut ewc_g = vec![0.0f32; n];
                                for i in 0..n {
                                    let diff = w[i] - a[i];
                                    ewc_g[i] = (self.ewc_lambda * f[i] * diff) * accum_count as f32;
                                }
                                let _ = gpu_trainer.accumulate_gradient(k, &ewc_g);
                            }
                        }
                    }
                    gpu_trainer
                        .scale_accumulated_gradients(accum_count)
                        .map_err(|e| {
                            TrainerError::Model(format!("GPU gradient scaling failed: {e}"))
                        })?;
                    let stepped = gpu_trainer
                        .step_adamw_with_weights(
                            &mut weights,
                            effective_lr,
                            beta1,
                            beta2,
                            eps,
                            weight_decay,
                            max_grad_norm,
                        )
                        .map_err(|e| TrainerError::Model(format!("GPU AdamW step failed: {e}")))?;
                    if stepped {
                        step_taken = true;
                    }

                    if !gpu_trainer.is_streaming() {
                        gpu_trainer
                            .download_weights_into(&mut weights)
                            .map_err(|e| {
                                TrainerError::Model(format!("GPU weights download failed: {e}"))
                            })?;
                    }
                } else {
                    let factor = 1.0 / accum_count as f32;
                    for g in accum_grads.values_mut() {
                        for val in g.iter_mut() {
                            *val *= factor;
                        }
                    }
                    if is_tied {
                        if let Some(lm_g) = accum_grads.get("lm_head.weight").cloned() {
                            if let Some(emb_g) = accum_grads.get_mut("model.embed_tokens.weight") {
                                for (e, l) in emb_g.iter_mut().zip(lm_g.iter()) {
                                    *e += *l;
                                }
                            }
                        }
                    }
                    if let (Some(ref anchors), Some(ref importance)) =
                        (&self.anchor_weights, &self.ewc_importance)
                    {
                        for (k, g) in accum_grads.iter_mut() {
                            if let (Some(w), Some(a), Some(f)) =
                                (weights.get(k), anchors.get(k), importance.get(k))
                            {
                                let n = g.len().min(w.len()).min(a.len()).min(f.len());
                                for i in 0..n {
                                    let diff = w[i] - a[i];
                                    g[i] += self.ewc_lambda * f[i] * diff;
                                }
                            }
                        }
                    }
                    let stepped = optimizer.step(
                        &mut weights,
                        &accum_grads,
                        &AdamWHyperparams {
                            lr: effective_lr,
                            beta1,
                            beta2,
                            eps,
                            weight_decay,
                            max_grad_norm,
                        },
                    );
                    if is_tied {
                        if let Some(emb_w) = weights.get("model.embed_tokens.weight").cloned() {
                            weights.insert("lm_head.weight".to_string(), emb_w);
                        }
                    }
                    if stepped {
                        step_taken = true;
                    }
                    accum_grads.clear();
                }
                if step_taken {
                    optimizer_steps += 1;
                }

                let batch_lm_loss = batch_loss_sum / accum_count as f32;
                let total_step_loss = batch_lm_loss + ewc_loss;
                let ppl = (batch_lm_loss as f64).min(50.0).exp();
                let bpb = if batch_bytes_count > 0 {
                    ((batch_lm_loss as f64) / std::f64::consts::LN_2) * (batch_tokens_count as f64)
                        / (batch_bytes_count as f64)
                } else {
                    (batch_lm_loss as f64) / std::f64::consts::LN_2
                };
                let grad_norm = if let Some(ref dev) = cuda_compute_dev_opt {
                    dev.trainer()
                        .lock()
                        .map(|t| t.last_grad_norm())
                        .unwrap_or(0.0)
                } else {
                    optimizer.last_grad_norm()
                };
                let step_duration = step_start_instant.elapsed();
                let batch_time_sec = step_duration.as_secs_f32().max(1e-6);
                let throughput = batch_tokens_count as f32 / batch_time_sec;

                let step_duration_secs = step_duration.as_secs_f64();
                ema_step_duration = Some(match ema_step_duration {
                    None => step_duration_secs,
                    Some(prev) => 0.8 * prev + 0.2 * step_duration_secs,
                });

                let step_pct_str = if let Some(max_s) = self.max_steps {
                    format!(
                        " ({:.1}%)",
                        (optimizer_steps as f64 / max_s as f64 * 100.0).min(100.0)
                    )
                } else {
                    String::new()
                };

                let eta_str =
                    if let (Some(max_s), Some(ema_sec)) = (self.max_steps, ema_step_duration) {
                        let rem_steps = max_s.saturating_sub(optimizer_steps);
                        let eta_sec = rem_steps as f64 * ema_sec;
                        format!(" | ETA: {}", Self::format_duration_hhmmss(eta_sec))
                    } else {
                        String::new()
                    };
                let elapsed_str =
                    Self::format_duration_hhmmss(train_start_instant.elapsed().as_secs_f64());

                let snap = crate::cuda::gpu_trainer::capture_memory_snapshot(
                    cuda_compute_dev_opt
                        .as_ref()
                        .and_then(|d| d.trainer().lock().ok()?.get_vram_info().ok()),
                );
                let vram_str = match (snap.gpu_used_vram_mb, snap.gpu_total_vram_mb) {
                    (Some(u), Some(t)) => format!(" | GPU VRAM: {:.1}/{:.1} MB", u, t),
                    _ => String::new(),
                };

                let max_str = self
                    .max_steps
                    .map(|m| m.to_string())
                    .unwrap_or_else(|| "∞".to_string());
                let running_loss_display = running_avg_loss.unwrap_or(batch_lm_loss);

                println!(
                    "[{stage_label} | Step {:>4}/{}{} (Epoch Flush)] Total Loss: {:.4} | LM Loss: {:.4} (Running: {:.4}) | PPL: {:.2} | BPB: {:.4} | Aux/EWC: {:.4} | Grad Norm: {:.4} | LR: {:.6} | Throughput: {:.1} tok/s | Step Time: {:.2}s | Elapsed: {}{} | Host RSS: {:.1} MB{}",
                    optimizer_steps,
                    max_str,
                    step_pct_str,
                    total_step_loss,
                    batch_lm_loss,
                    running_loss_display,
                    ppl,
                    bpb,
                    ewc_loss,
                    grad_norm,
                    effective_lr,
                    throughput,
                    batch_time_sec,
                    elapsed_str,
                    eta_str,
                    snap.process_rss_mb,
                    vram_str
                );
                let _ = std::io::stdout().flush();

                model = TaraForCausalLM::from_weights_move(weights, &config, &self.model_dir)
                    .map_err(|e| TrainerError::Model(e.to_string()))?;
                if let Some(ref dev) = cuda_compute_dev_opt {
                    dev.update_slice_map(model.parameter_slice_map());
                }

                // Checkpoint at epoch boundary flush
                if let Some(ref cp_dir) = self.checkpoint_dir {
                    let current_eval_loss = if val_tokens.is_empty() {
                        None
                    } else {
                        compute_model_loss(&model, &hd_channel, &val_tokens, "Epoch Eval").ok()
                    };
                    let is_best = match current_eval_loss {
                        Some(loss) if loss < best_loss => {
                            best_loss = loss;
                            true
                        }
                        _ => false,
                    };

                    let opt_moments = if let Some(ref dev) = cuda_compute_dev_opt {
                        dev.trainer()
                            .lock()
                            .map(|t| t.export_optimizer_state().unwrap_or_default())
                            .unwrap_or_default()
                    } else {
                        optimizer.export_state()
                    };
                    let weights = model.into_weights_map();
                    Self::save_checkpoint_state_bundle(CheckpointBundleConfig {
                        cp_dir,
                        model_dir: &self.model_dir,
                        weights: &weights,
                        config: &config,
                        opt_moments,
                        optimizer_steps,
                        samples_seen,
                        epoch_idx,
                        current_eval_loss,
                        best_loss,
                        is_best,
                        val_tokens_empty: val_tokens.is_empty(),
                    })?;
                    model = TaraForCausalLM::from_weights_move(weights, &config, &self.model_dir)
                        .map_err(|e| TrainerError::Model(e.to_string()))?;
                    if let Some(ref dev) = cuda_compute_dev_opt {
                        dev.update_slice_map(model.parameter_slice_map());
                    }
                    let eval_str = current_eval_loss
                        .map(|l| format!("{:.4}", l))
                        .unwrap_or_else(|| "N/A".to_string());
                    let best_str = if val_tokens.is_empty() {
                        "N/A".to_string()
                    } else {
                        format!("{:.4}", best_loss)
                    };
                    println!(
                        "[{stage_label} | Checkpoint Saved] Step {} (Epoch {}) | Eval Loss: {} | Best Loss: {} | Path: {} | Sync: GDrive",
                        optimizer_steps,
                        epoch_idx + 1,
                        eval_str,
                        best_str,
                        cp_dir
                    );
                    let _ = std::io::stdout().flush();
                }
            }

            epochs_completed = epoch_idx + 1;
            if let Some(max_s) = self.max_steps {
                if optimizer_steps >= max_s {
                    break;
                }
            }
        }

        let final_loss: Option<f64> = if val_tokens.is_empty() {
            println!("[Validation Gate] Held-out validation unavailable (single-sequence boundary). Skipping regression check.");
            None
        } else {
            let fl = compute_model_loss(&model, &hd_channel, &val_tokens, "Final Validation")?;
            // Validation Gate: reject non-finite loss or severe regression
            if !fl.is_finite() {
                return Err(TrainerError::Model(format!(
                    "Candidate rejected: final validation loss is non-finite ({fl}). Model update aborted."
                )));
            }
            if let Some(init_l) = initial_loss {
                if init_l > 0.0 && fl > init_l * 1.5 {
                    return Err(TrainerError::Model(format!(
                        "Candidate rejected: validation loss regressed from {init_l:.4} to {fl:.4}. Model update aborted."
                    )));
                }
                println!(
                    "[Validation Gate] PASSED: initial baseline = {:.4}, final validation loss = {:.4}",
                    init_l, fl
                );
            }
            Some(fl)
        };

        // 8. Determine destination and staging directory (Candidate Isolation)
        let now_stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let candidate_id = format!("TARA_CANDIDATE_{now_stamp}");

        let (target_candidate_dir, is_source_preserved) = if let Some(dir) = candidate_dir_override
        {
            (dir.to_string(), true)
        } else {
            let normalized = self.model_dir.replace('\\', "/");
            if normalized.ends_with("storage/models/tara")
                || normalized.ends_with("storage/models/tara/")
            {
                (
                    format!(
                        "{}/storage/models/candidates/{}",
                        self.repo_root, candidate_id
                    ),
                    true,
                )
            } else {
                // If model_dir was a temporary isolated fixture (unit tests)
                (self.model_dir.clone(), false)
            }
        };

        let staging_dir = format!("{}.staging_{}", target_candidate_dir, now_stamp);
        fs::create_dir_all(&staging_dir)?;

        let weights = model.into_weights_map();

        // Write sharded or single SafeTensors into staging directory (100 MB canonical boundary)
        let shard_files =
            write_safetensors_sharded(&weights, &shapes, &staging_dir, 100 * 1024 * 1024)?;

        // Copy config.json and tokenizer.json into staging directory
        let cfg_dest = format!("{}/config.json", staging_dir);
        fs::copy(&config_path, &cfg_dest)?;
        let tok_dest = format!("{}/tokenizer.json", staging_dir);
        fs::copy(&tokenizer_path, &tok_dest)?;

        // Export optimizer state moments so every candidate checkpoint is continuation-ready
        let opt_moments = if let Some(ref dev) = cuda_compute_dev_opt {
            dev.trainer()
                .lock()
                .map(|t| t.export_optimizer_state().unwrap_or_default())
                .unwrap_or_default()
        } else {
            optimizer.export_state()
        };
        let mut opt_tensors: HashMap<String, Vec<f32>> = HashMap::new();
        let mut opt_shapes: HashMap<String, Vec<usize>> = HashMap::new();
        for (k, (m_vec, v_vec)) in opt_moments {
            let m_name = format!("{k}.adam_m");
            let v_name = format!("{k}.adam_v");
            opt_shapes.insert(m_name.clone(), vec![m_vec.len()]);
            opt_shapes.insert(v_name.clone(), vec![v_vec.len()]);
            opt_tensors.insert(m_name, m_vec);
            opt_tensors.insert(v_name, v_vec);
        }
        crate::safetensors::write_safetensors_with_shapes(
            &opt_tensors,
            &opt_shapes,
            &format!("{}/optimizer.safetensors", staging_dir),
        )?;

        let state_json = json!({
            "step": optimizer_steps,
            "optimizer_step": optimizer_steps,
            "samples_seen": samples_seen,
            "epoch": epochs_completed,
            "candidate_id": candidate_id,
            "loss_before": initial_loss,
            "loss_after": final_loss,
            "timestamp": crate::now_iso(),
            "resumable": true,
        });
        fs::write(
            format!("{}/checkpoint_state.json", staging_dir),
            serde_json::to_string_pretty(&state_json)?,
        )?;

        // Compute candidate weights sha256
        let first_shard = shard_files
            .first()
            .map(String::as_str)
            .unwrap_or("model.safetensors");
        let candidate_sha256 = compute_sha256(&format!("{}/{}", staging_dir, first_shard))
            .unwrap_or_else(|_| "unknown_sha256".to_string());

        let previous_sha256 =
            TaraForCausalLM::get_model_sha256(&self.model_dir).unwrap_or_default();

        let updated_tensors: Vec<String> = weights.keys().cloned().collect();

        let result = json!({
            "status": "COMPLETED",
            "candidate_id": candidate_id,
            "completed_at_epoch_seconds": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
            "cycles_completed": self.get_status().get("cycles_completed").and_then(Value::as_u64).unwrap_or(0) + 1,
            "samples_trained": samples_seen,
            "epochs": epochs_completed,
            "steps": optimizer_steps,
            "optimizer_steps": optimizer_steps,
            "loss_before": initial_loss,
            "loss_after": final_loss,
            "candidate_sha256": candidate_sha256,
            "previous_model_sha256": previous_sha256,
            "source_model_preserved": is_source_preserved,
            "source_model_dir": self.model_dir,
            "candidate_dir": target_candidate_dir,
            "device_selected": device_label,
            "device_type": if is_gpu { "GPU" } else { "CPU" },
            "precision_used": precision_label,
            "shards_exported": shard_files,
            "updated_tensors": updated_tensors
        });

        // Write training summary into staging dir
        fs::write(
            format!("{}/training_summary.json", staging_dir),
            serde_json::to_string_pretty(&result)?,
        )?;

        // Two-phase atomic rename from staging to target candidate directory
        if target_candidate_dir == self.model_dir {
            // In unit tests where model_dir is the target fixture, copy files across
            for shard in &shard_files {
                fs::copy(
                    format!("{}/{}", staging_dir, shard),
                    format!("{}/{}", target_candidate_dir, shard),
                )?;
            }
            if Path::new(&format!("{}/model.safetensors.index.json", staging_dir)).exists() {
                fs::copy(
                    format!("{}/model.safetensors.index.json", staging_dir),
                    format!("{}/model.safetensors.index.json", target_candidate_dir),
                )?;
            }
            fs::copy(
                format!("{}/config.json", staging_dir),
                format!("{}/config.json", target_candidate_dir),
            )?;
            fs::copy(
                format!("{}/tokenizer.json", staging_dir),
                format!("{}/tokenizer.json", target_candidate_dir),
            )?;
            if Path::new(&format!("{}/optimizer.safetensors", staging_dir)).exists() {
                fs::copy(
                    format!("{}/optimizer.safetensors", staging_dir),
                    format!("{}/optimizer.safetensors", target_candidate_dir),
                )?;
            }
            if Path::new(&format!("{}/checkpoint_state.json", staging_dir)).exists() {
                fs::copy(
                    format!("{}/checkpoint_state.json", staging_dir),
                    format!("{}/checkpoint_state.json", target_candidate_dir),
                )?;
            }
            let _ = fs::remove_dir_all(&staging_dir);
        } else {
            promote_directory_atomically(
                Path::new(&staging_dir),
                Path::new(&target_candidate_dir),
            )?;
        }

        // Persist status
        let status_dir = format!("{}/storage/training", self.repo_root);
        fs::create_dir_all(&status_dir)?;
        let status_path = format!("{}/self_trainer_status.json", status_dir);
        fs::write(&status_path, serde_json::to_string_pretty(&result)?)?;

        Ok(result)
    }
}

/// Recovers from an interrupted directory promotion if `<target_dir>` is missing or incomplete
/// while a valid `<target_dir>.bak_<timestamp>` exists in the same parent directory, and purges
/// stale `.staging_*` or superseded `.bak_*` directories once `<target_dir>` is verified intact.
pub fn recover_interrupted_promotion(target_dir: &Path) -> Result<bool, TrainerError> {
    recover_interrupted_promotion_inner(target_dir, None)
}

fn is_promoted_directory_intact(dir: &Path) -> bool {
    if !dir.exists() || !dir.is_dir() {
        return false;
    }
    if dir.join("world_model_state.json").exists() {
        return true;
    }
    let has_config = dir.join("config.json").exists();
    let has_tokenizer = dir.join("tokenizer.json").exists();
    let has_weights =
        dir.join("model.safetensors").exists() || dir.join("model.safetensors.index.json").exists();
    has_config && has_tokenizer && has_weights
}

fn recover_interrupted_promotion_inner(
    target_dir: &Path,
    active_staging: Option<&Path>,
) -> Result<bool, TrainerError> {
    let parent = match target_dir.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    if !parent.exists() {
        return Ok(false);
    }
    let base_name = match target_dir.file_name().and_then(|n| n.to_str()) {
        Some(n) if !n.is_empty() => n,
        _ => return Ok(false),
    };

    let bak_prefix = format!("{}.bak_", base_name);
    let staging_prefix = format!("{}.staging_", base_name);
    let active_staging_name = active_staging
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str());

    let mut backups: Vec<PathBuf> = Vec::new();
    let mut stagings: Vec<PathBuf> = Vec::new();

    if let Ok(entries) = fs::read_dir(parent) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with(&bak_prefix) {
                        backups.push(p);
                    } else if name.starts_with(&staging_prefix) && Some(name) != active_staging_name
                    {
                        stagings.push(p);
                    }
                }
            }
        }
    }

    backups.sort();
    let target_intact = is_promoted_directory_intact(target_dir);
    let mut recovered = false;

    if !target_intact {
        if let Some(valid_bak) = backups
            .iter()
            .rev()
            .find(|b| is_promoted_directory_intact(b))
            .cloned()
        {
            if target_dir.exists() {
                let _ = fs::remove_dir_all(target_dir);
            }
            fs::rename(&valid_bak, target_dir)?;
            recovered = true;
            backups.retain(|b| b != &valid_bak);
        }
    }

    // If target is now intact, clean up stale backups and orphan staging dirs (excluding active_staging)
    if is_promoted_directory_intact(target_dir) {
        for b in backups {
            let _ = fs::remove_dir_all(b);
        }
        for s in stagings {
            let _ = fs::remove_dir_all(s);
        }
    }

    Ok(recovered)
}

/// Atomically promotes `staging_dir` to `target_dir` on the same parent filesystem with
/// file-level `sync_all`, `.bak_<timestamp>` backup, and automatic rollback if rename fails.
pub fn promote_directory_atomically(
    staging_dir: &Path,
    target_dir: &Path,
) -> Result<(), TrainerError> {
    if !staging_dir.exists() || !staging_dir.is_dir() {
        return Err(TrainerError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Staging directory does not exist: {}",
                staging_dir.display()
            ),
        )));
    }

    // Recover any prior interrupted promotion first while preserving our active staging_dir
    let _ = recover_interrupted_promotion_inner(target_dir, Some(staging_dir))?;

    if let Some(parent) = target_dir.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    // Flush all regular files in staging_dir to disk before rename
    if let Ok(entries) = fs::read_dir(staging_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() {
                if let Ok(f) = File::open(&p) {
                    let _ = f.sync_all();
                }
            }
        }
    }

    let now_stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let backup_path = PathBuf::from(format!("{}.bak_{}", target_dir.display(), now_stamp));

    if target_dir.exists() {
        fs::rename(target_dir, &backup_path)?;
    }

    let mut rename_res = fs::rename(staging_dir, target_dir);
    if rename_res.is_err() {
        for attempt in 1..=10 {
            std::thread::sleep(std::time::Duration::from_millis(50 * attempt));
            rename_res = fs::rename(staging_dir, target_dir);
            if rename_res.is_ok() {
                break;
            }
        }
    }

    match rename_res {
        Ok(()) => {
            if backup_path.exists() {
                let _ = fs::remove_dir_all(&backup_path);
            }
            Ok(())
        }
        Err(e) => {
            if backup_path.exists() {
                let _ = fs::rename(&backup_path, target_dir);
            }
            Err(TrainerError::Io(e))
        }
    }
}

fn compute_model_loss(
    model: &TaraForCausalLM,
    channel: &tara_hd::ComputeChannel,
    val_tokens: &[(Vec<u32>, usize)],
    stage_desc: &str,
) -> Result<f64, TrainerError> {
    use std::io::Write;

    if val_tokens.is_empty() {
        return Ok(f64::INFINITY);
    }

    let vs = model.config.vocab_size;
    let hs = model.config.hidden_size;
    let max_pos = model.config.max_position_embeddings;
    let total_samples = val_tokens.len();

    let mut total_loss = 0.0f64;
    let mut total_target_tokens = 0usize;
    let start_time = std::time::Instant::now();

    println!(
        "[{stage_desc}] Evaluating validation loss across {} samples (context ceiling: {} tokens, micro-batch: 1)...",
        total_samples, max_pos
    );
    let _ = std::io::stdout().flush();

    for (idx, (tokens, target_start)) in val_tokens.iter().enumerate() {
        let sample_start = std::time::Instant::now();
        let orig_len = tokens.len();
        let val_seq_ceiling = std::env::var("TARA_MAX_VAL_SEQ_LEN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(256);
        let seq_len = orig_len.min(max_pos).min(val_seq_ceiling);
        if seq_len == 0 {
            continue;
        }

        let active_tokens = &tokens[..seq_len];
        let eff_target_start = (*target_start).min(seq_len);

        // Forward hidden activations and LM head routed through TARA-HD channel
        let final_normed = model.forward_hidden_channel(active_tokens, channel);
        let logits = channel
            .forward_linear_named("lm_head.weight", &model.lm_head, &final_normed, seq_len, hs, vs)
            .unwrap_or_else(|_| {
                use rayon::prelude::*;
                let mut lgs = vec![0.0f32; seq_len * vs];
                lgs.par_chunks_exact_mut(vs)
                    .enumerate()
                    .for_each(|(t, l_row)| {
                        let h_row = &final_normed[t * hs..(t + 1) * hs];
                        for (v, val) in l_row.iter_mut().enumerate() {
                            let lm_row = &model.lm_head[v * hs..(v + 1) * hs];
                            *val = h_row.iter().zip(lm_row.iter()).map(|(&a, &b)| a * b).sum();
                        }
                    });
                lgs
            });
        drop(final_normed);

        // Cross-entropy loss over target tokens (Rayon-parallelized across CPU cores)
        let target_positions: Vec<(usize, usize)> = active_tokens
            .iter()
            .enumerate()
            .take(seq_len)
            .skip(eff_target_start.max(1))
            .filter_map(|(pos, &token_val)| {
                let target = token_val as usize;
                if target < vs {
                    Some((pos.saturating_sub(1), target))
                } else {
                    None
                }
            })
            .collect();

        use rayon::prelude::*;
        let (sample_loss, sample_targets) = target_positions
            .par_iter()
            .map(|&(predictor, target)| {
                let logit_start = predictor * vs;
                let logits_slice = &logits[logit_start..logit_start + vs];
                let max_logit = logits_slice
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let sum_exp: f64 = logits_slice
                    .iter()
                    .map(|&l| ((l - max_logit) as f64).exp())
                    .sum();
                if sum_exp > 0.0 && sum_exp.is_finite() {
                    let token_loss = sum_exp.ln() - ((logits_slice[target] - max_logit) as f64);
                    (token_loss, 1usize)
                } else {
                    (0.0, 0usize)
                }
            })
            .reduce(|| (0.0f64, 0usize), |a, b| (a.0 + b.0, a.1 + b.1));

        drop(logits);

        let sample_mean_loss = if sample_targets > 0 {
            sample_loss / sample_targets as f64
        } else {
            0.0
        };
        total_loss += sample_loss;
        total_target_tokens += sample_targets;

        let elapsed = sample_start.elapsed().as_secs_f64();
        let snap = crate::cuda::capture_memory_snapshot(None);

        if idx == 0 || (idx + 1) == total_samples || (idx + 1) % 50 == 0 {
            println!(
                "[{stage_desc}] Validation sample {}/{}: tokens={}, time={:.3}s, loss={:.4}, Host RSS={:.1} MB, forward_completed=true",
                idx + 1,
                total_samples,
                seq_len,
                elapsed,
                sample_mean_loss,
                snap.process_rss_mb
            );
            let _ = std::io::stdout().flush();
        }
    }

    let overall_elapsed = start_time.elapsed().as_secs_f64();
    if total_target_tokens == 0 {
        return Ok(f64::INFINITY);
    }

    let mean_loss = total_loss / total_target_tokens as f64;
    let avg_ms = (overall_elapsed * 1000.0) / total_samples as f64;
    let tok_per_sec = (total_target_tokens as f64) / overall_elapsed.max(1e-4);

    println!(
        "[{stage_desc} Complete] Evaluated {} samples ({} tokens) in {:.2}s (avg {:.1} ms/sample, {:.1} tok/sec) | Loss: {:.4}",
        total_samples, total_target_tokens, overall_elapsed, avg_ms, tok_per_sec, mean_loss
    );
    let _ = std::io::stdout().flush();

    if !mean_loss.is_finite() || mean_loss <= 0.0 {
        return Err(TrainerError::Model(format!(
            "Validation loss is non-finite or non-positive: {mean_loss}"
        )));
    }

    Ok(mean_loss)
}

#[cfg(test)]
mod tests {
    use super::{NativeSelfTrainer, StreamingDatasetReader};
    use crate::config::TaraConfig;
    use crate::safetensors::{load_model_weights, write_safetensors_with_shapes};
    use std::collections::HashMap;

    #[test]
    fn supervised_cycle_updates_real_model_parameters_and_reduces_loss() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("tara_train_{}_{}", std::process::id(), stamp));
        let model_dir = root.join("model");
        let dataset_dir = root.join("storage/datasets");
        std::fs::create_dir_all(&model_dir).expect("create model fixture");
        std::fs::create_dir_all(&dataset_dir).expect("create dataset fixture");
        let config = TaraConfig {
            vocab_size: 3,
            hidden_size: 2,
            intermediate_size: 2,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            max_position_embeddings: 8,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            initializer_range: 0.02,
            version: "test".into(),
            model_type: "tara".into(),
            tie_word_embeddings: None,
        };
        std::fs::write(
            model_dir.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        std::fs::write(
            model_dir.join("tokenizer.json"),
            r#"{"vocab":{"a":0,"b":1,"<|unk|>":2}}"#,
        )
        .unwrap();

        let mut weights = HashMap::<String, Vec<f32>>::new();
        weights.insert(
            "model.embed_tokens.weight".into(),
            vec![1.0, 0.5, 0.2, 1.0, 0.1, 0.1],
        );
        weights.insert("model.norm.weight".into(), vec![1.0, 1.0]);
        weights.insert("lm_head.weight".into(), vec![0.1; 6]);
        for name in [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            weights.insert(format!("model.layers.0.{name}"), vec![0.1; 4]);
        }
        weights.insert("model.layers.0.input_layernorm.weight".into(), vec![1.0; 2]);
        weights.insert(
            "model.layers.0.post_attention_layernorm.weight".into(),
            vec![1.0; 2],
        );
        let shapes = weights
            .iter()
            .map(|(name, tensor)| {
                let shape = match name.as_str() {
                    "model.embed_tokens.weight" | "lm_head.weight" => vec![3, 2],
                    "model.norm.weight"
                    | "model.layers.0.input_layernorm.weight"
                    | "model.layers.0.post_attention_layernorm.weight" => vec![2],
                    _ => vec![2, 2],
                };
                (
                    name.clone(),
                    if shape.iter().product::<usize>() == tensor.len() {
                        shape
                    } else {
                        vec![tensor.len()]
                    },
                )
            })
            .collect();
        write_safetensors_with_shapes(
            &weights,
            &shapes,
            &model_dir.join("model.safetensors").to_string_lossy(),
        )
        .unwrap();
        std::fs::write(
            dataset_dir.join("examples.jsonl"),
            "{\"input\":\"a\",\"output\":\"b\"}\n{\"input\":\"a\",\"output\":\"b\"}\n",
        )
        .unwrap();

        let trainer = NativeSelfTrainer::new(&model_dir.to_string_lossy(), &root.to_string_lossy())
            .with_dataset_dir(&dataset_dir);
        let result = trainer
            .run_full_self_learning_cycle(5, true)
            .expect("train model");
        assert_eq!(result["status"], "COMPLETED");
        assert!(result["loss_after"].as_f64().unwrap() < result["loss_before"].as_f64().unwrap());
        let updated = load_model_weights(&model_dir.to_string_lossy()).unwrap();
        assert_ne!(updated["lm_head.weight"], weights["lm_head.weight"]);
        assert_ne!(
            updated["model.embed_tokens.weight"],
            weights["model.embed_tokens.weight"]
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn test_single_sample_held_out_validation_cleanly_unavailable() {
        let temp_dir =
            std::env::temp_dir().join(format!("tara_train_single_{}", std::process::id()));
        let root = temp_dir.clone();
        let model_dir = root.join("model");
        let dataset_dir = root.join("dataset");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::create_dir_all(&dataset_dir).unwrap();

        let config = TaraConfig {
            vocab_size: 3,
            hidden_size: 2,
            intermediate_size: 2,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            max_position_embeddings: 8,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            initializer_range: 0.02,
            version: "test".into(),
            model_type: "tara".into(),
            tie_word_embeddings: None,
        };
        std::fs::write(
            model_dir.join("config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();

        let mut vocab = HashMap::new();
        vocab.insert("<unk>".to_string(), 0);
        vocab.insert("a".to_string(), 1);
        vocab.insert("b".to_string(), 2);
        let tokenizer_data = serde_json::json!({
            "vocab": vocab
        });
        std::fs::write(
            model_dir.join("tokenizer.json"),
            serde_json::to_string(&tokenizer_data).unwrap(),
        )
        .unwrap();

        let mut weights = HashMap::new();
        weights.insert("model.embed_tokens.weight".to_string(), vec![0.1; 6]);
        weights.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            vec![1.0; 2],
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            vec![0.1; 4],
        );
        weights.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            vec![0.1; 4],
        );
        weights.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            vec![0.1; 4],
        );
        weights.insert(
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            vec![0.1; 4],
        );
        weights.insert(
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            vec![1.0; 2],
        );
        weights.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            vec![0.1; 4],
        );
        weights.insert(
            "model.layers.0.mlp.up_proj.weight".to_string(),
            vec![0.1; 4],
        );
        weights.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            vec![0.1; 4],
        );
        weights.insert("model.norm.weight".to_string(), vec![1.0; 2]);
        weights.insert("lm_head.weight".to_string(), vec![0.1; 6]);

        let shapes: HashMap<String, Vec<usize>> = weights
            .keys()
            .map(|name| {
                let shape = match name.as_str() {
                    "model.embed_tokens.weight" | "lm_head.weight" => vec![3, 2],
                    "model.norm.weight"
                    | "model.layers.0.input_layernorm.weight"
                    | "model.layers.0.post_attention_layernorm.weight" => vec![2],
                    _ => vec![2, 2],
                };
                (name.clone(), shape)
            })
            .collect();
        write_safetensors_with_shapes(
            &weights,
            &shapes,
            &model_dir.join("model.safetensors").to_string_lossy(),
        )
        .unwrap();
        std::fs::write(
            dataset_dir.join("examples.jsonl"),
            r#"{"input":"a","output":"b"}"#,
        )
        .unwrap();

        let trainer = NativeSelfTrainer::new(&model_dir.to_string_lossy(), &root.to_string_lossy())
            .with_dataset_dir(&dataset_dir);
        let result = trainer
            .run_full_self_learning_cycle(2, true)
            .expect("train model");
        assert_eq!(result["status"], "COMPLETED");
        assert!(result["loss_before"].is_null());
        assert!(result["loss_after"].is_null());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn test_streaming_dataset_reader_format_compatibility() {
        let temp_dir =
            std::env::temp_dir().join(format!("tara_stream_test_{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir).unwrap();
        let file_path = temp_dir.join("multi_format.jsonl");

        let content = [
            // 1. Standard ChatML format emitted by tara_tokenizer_stage
            r#"{"id":"rec_1","formatted_input":"<|im_start|>user\nWhat is Pi?<|im_end|>\n<|im_start|>assistant\n","formatted_target":"3.14159<|im_end|>","metadata":{"input":"What is Pi?","output":"3.14159"}}"#,
            // 2. Standard raw input / output format
            r#"{"id":"rec_2","input":"Explain gravity","output":"Gravity is an attraction between masses."}"#,
            // 3. Instruction / Response format
            r#"{"id":"rec_3","instruction":"Write rust hello world","response":"fn main() { println!(\"hello\"); }"}"#,
            // 4. Prompt / Completion format
            r#"{"id":"rec_4","prompt":"Capital of Karnataka?","completion":"Bengaluru."}"#,
            // 5. Nested metadata only format
            r#"{"id":"rec_5","metadata":{"input":"Define force","output":"Mass times acceleration."}}"#,
            // 6. Experiential learning lessons format (trigger_pattern / recommendation)
            r#"{"lesson_id":"lesson_6","task_category":"CONVERSATIONAL","trigger_pattern":"calculate 20 + 30","strategy_used":"natural_query","outcome_score":1.0,"recommendation":"Execute validated path for 'CONVERSATIONAL'"}"#,
        ].join("\n");

        std::fs::write(&file_path, content).unwrap();

        let mut reader = StreamingDatasetReader::open(&file_path.to_string_lossy()).unwrap();
        let mut samples = Vec::new();
        while let Ok(Some(sample)) = reader.next_sample() {
            samples.push(sample);
        }

        assert_eq!(
            samples.len(),
            6,
            "All 6 dataset formats must be successfully parsed"
        );
        assert_eq!(
            samples[0].0,
            "<|im_start|>user\nWhat is Pi?<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(samples[0].1, "3.14159<|im_end|>");
        assert_eq!(samples[1].0, "Explain gravity");
        assert_eq!(samples[1].1, "Gravity is an attraction between masses.");
        assert_eq!(samples[2].0, "Write rust hello world");
        assert_eq!(samples[2].1, "fn main() { println!(\"hello\"); }");
        assert_eq!(samples[3].0, "Capital of Karnataka?");
        assert_eq!(samples[3].1, "Bengaluru.");
        assert_eq!(samples[4].0, "Define force");
        assert_eq!(samples[4].1, "Mass times acceleration.");
        assert_eq!(samples[5].0, "calculate 20 + 30");
        assert_eq!(samples[5].1, "Execute validated path for 'CONVERSATIONAL'");

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_estimate_memory_footprint_streamed_gpu_accounting() {
        // ~1B model on GPU: verify host RAM correctly accounts for optimizer moments
        let total_params = 995_809_280;
        let footprint = NativeSelfTrainer::estimate_memory_footprint_mb_detailed(
            total_params,
            true,        // is_gpu
            true,        // is_f16
            2048 * 2048, // max_tensor_len
            1,           // batch_size
            512,         // seq_len
            2048,        // hidden_size — representative for ~1B parameter transformer
        );

        assert!(footprint.governing_peak_mb > 0.0);
        assert!(footprint.host_ram_mb > 0.0);
        assert!(footprint.device_vram_mb > 0.0);
        assert!(footprint.breakdown.contains("Opt("), "breakdown was: {}", footprint.breakdown);
        assert!(footprint.breakdown.contains("Host RAM:"));
        assert!(footprint.breakdown.contains("GPU VRAM:"));

        // Small GPU model (~10M params) has resident moments in GPU VRAM
        let small_params = 10_000_000;
        let small_footprint = NativeSelfTrainer::estimate_memory_footprint_mb_detailed(
            small_params,
            true,
            true,
            512 * 512,
            1,
            512,
            512, // hidden_size — representative for small model
        );
        assert!(
            small_footprint.breakdown.contains("Opt(GPU VRAM resident)="),
            "breakdown was: {}", small_footprint.breakdown
        );

        // CPU mode accounts for gradients + moments on host
        let cpu_footprint = NativeSelfTrainer::estimate_memory_footprint_mb_detailed(
            small_params,
            false,
            false,
            0,
            1,
            512,
            512, // hidden_size — representative for small model
        );
        assert!(
            cpu_footprint.breakdown.contains("Grads+Moments(fp32)="),
            "CPU breakdown was: {}", cpu_footprint.breakdown
        );
    }

    #[test]
    fn test_memory_preflight_cgroup_awareness() {
        // Test that preflight ceiling logic correctly identifies memory limits
        // and respects the tighter of host RAM and cgroup limit.
        let channel = tara_hd::ComputeChannel::with_options(
            &tara_hd::SelectionPolicy::parse("cpu:0"),
            tara_hd::PrecisionMode::Fp32,
        )
        .expect("CPU channel must resolve");

        // Set an explicit TARA_HOST_RAM_MB to simulate a constrained environment (e.g., 4 GB container)
        std::env::set_var("TARA_HOST_RAM_MB", "4096");
        let auto_cfg = tara_hd::AutoTrainingConfig::compute(&channel, 100_000_000, 1024, 24, 500);
        std::env::remove_var("TARA_HOST_RAM_MB");

        assert_eq!(auto_cfg.host_ram_total_mb, 4096.0);
        assert_eq!(auto_cfg.host_ram_available_mb, 4096.0);
        // Default safety pct is 85%: 4096 * 0.85 = 3481.6 MB
        assert!((auto_cfg.max_memory_mb - 3481.6).abs() < 1.0);

        // Preflight estimate for 950M model: host RAM ~4.9 GB > 3481.6 MB safety ceiling
        let footprint = NativeSelfTrainer::estimate_memory_footprint_mb_detailed(
            950_000_000,
            true,
            true,
            2048 * 2048,
            1,
            128,
            2048,
        );
        assert!(
            footprint.host_ram_mb > auto_cfg.max_memory_mb,
            "950M model host RAM requirement ({} MB) must exceed 4GB container ceiling ({:.1} MB)",
            footprint.host_ram_mb,
            auto_cfg.max_memory_mb
        );
    }

    #[test]
    fn test_dual_pool_independent_admission() {
        // Verify that ~1B model on 15 GB GPU with 12 GB Host RAM safely admits:
        // Host RAM: ~4.9 GB <= 6.3 GB ceiling (Passes)
        // GPU VRAM: ~9.5 GB <= 12.6 GB ceiling (Passes)
        // Simulate Colab environment where host RAM budget triggers disk moments
        std::env::set_var("TARA_STREAM_MOMENTS_DISK", "1");
        let total_params = 995_809_280;
        let footprint = NativeSelfTrainer::estimate_memory_footprint_mb_detailed(
            total_params,
            true,        // is_gpu
            true,        // is_f16
            2048 * 2048, // max_tensor_len
            32,          // batch_size
            64,          // seq_len
            2048,        // hidden_size
        );
        std::env::remove_var("TARA_STREAM_MOMENTS_DISK");

        // Host RAM usage must be around 4.9 - 5.0 GB (with 15% margin)
        assert!(footprint.host_ram_mb < 6000.0, "Host RAM was: {:.1} MB", footprint.host_ram_mb);
        // GPU VRAM usage must be around 9.5 GB (with 10% margin)
        assert!(footprint.device_vram_mb < 11000.0, "GPU VRAM was: {:.1} MB", footprint.device_vram_mb);

        // Simulated Colab T4 (15 GB VRAM) + 12 GB Host RAM
        let host_ceiling_mb = 7467.0 * 0.85; // 6347 MB
        let vram_ceiling_mb = 14912.0 * 0.85; // 12675 MB

        assert!(footprint.host_ram_mb <= host_ceiling_mb, "Host RAM must fit inside host ceiling");
        assert!(footprint.device_vram_mb <= vram_ceiling_mb, "GPU VRAM must fit inside VRAM ceiling");
    }
}
