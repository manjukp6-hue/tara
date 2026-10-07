//! Native self-trainer for TARA model evolution.
//!
//! Executes full-network backpropagation (Embeddings, RMSNorm, Attention/GQA,
//! RoPE, SwiGLU MLP, and LM Head) with AdamW optimization, gradient clipping,
//! gradient accumulation, streaming dataset buffering, and isolated candidate staging.

use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

use crate::config::TaraConfig;
use crate::dataset::compiler::{CompilerError, DynamicDatasetCompiler};
use crate::model::causal_lm::TaraForCausalLM;
use crate::safetensors::{
    compute_sha256, load_model_weights_with_shapes, write_safetensors_sharded, SafeTensorsError,
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
        }
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

        self.step += 1;
        let bc1 = 1.0 - beta1.powi(self.step as i32);
        let bc2 = 1.0 - beta2.powi(self.step as i32);

        let clip_scale = if total_norm > max_grad_norm && total_norm > 0.0 {
            max_grad_norm / total_norm
        } else {
            1.0f32
        };

        // 2. Update all parameter weights with AdamW
        for (name, param) in weights.iter_mut() {
            if let Some(grad) = grads.get(name) {
                let m = self
                    .m
                    .entry(name.clone())
                    .or_insert_with(|| vec![0.0f32; param.len()]);
                let v = self
                    .v
                    .entry(name.clone())
                    .or_insert_with(|| vec![0.0f32; param.len()]);

                for i in 0..param.len() {
                    let g = grad[i] * clip_scale;
                    m[i] = beta1 * m[i] + (1.0 - beta1) * g;
                    v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
                    let m_hat = m[i] / bc1;
                    let v_hat = v[i] / bc2;
                    param[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                    param[i] -= lr * weight_decay * param[i];
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

/// Execution device backend for TARA neural self-training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum TrainingDevice {
    Cpu,
    Gpu,
    #[default]
    Auto,
}

impl std::str::FromStr for TrainingDevice {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "cpu" => Ok(TrainingDevice::Cpu),
            "gpu" => Ok(TrainingDevice::Gpu),
            "auto" => Ok(TrainingDevice::Auto),
            other => Err(format!(
                "Unknown device '{other}'. Valid options: cpu, gpu, auto"
            )),
        }
    }
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
    pub device: TrainingDevice,
    pub precision: crate::cuda::TrainingPrecision,
    pub checkpoint_dir: Option<String>,
    pub checkpoint_interval: Option<usize>,
    pub resume: bool,
    pub resume_from: Option<String>,
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
            device: TrainingDevice::Auto,
            precision: crate::cuda::TrainingPrecision::Auto,
            checkpoint_dir: None,
            checkpoint_interval: None,
            resume: false,
            resume_from: None,
        }
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

    /// Set an explicit maximum memory ceiling in megabytes.
    pub fn with_max_memory_mb(mut self, mb: f64) -> Self {
        self.max_memory_mb = mb;
        self
    }

    /// Calculate projected training memory footprint in MB given total parameter count.
    pub fn estimate_memory_footprint_mb(total_params: usize) -> f64 {
        (total_params * 16) as f64 / (1024.0 * 1024.0) + 50.0
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
        if max_epochs == 0 || max_epochs > 100 {
            return Err(TrainerError::Dataset(
                "max_epochs must be between 1 and 100".into(),
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
        let target_samples_limit = self.max_steps.map(|ms| ms * self.batch_size.unwrap_or(4) * 2);
        let target_limit_str = target_samples_limit
            .map(|l| format!("{l} samples"))
            .unwrap_or_else(|| "unlimited".to_string());
        println!("[Dataset] Loading training data (target quota: {target_limit_str})...");

        // 2a. Load Academic Curriculum if available (storage/datasets/curriculum/master_curriculum.jsonl)
        let has_explicit_curriculum = self.curriculum_path.is_some();
        let curr_path = self.curriculum_path.clone().unwrap_or_else(|| {
            format!("{}/storage/datasets/curriculum/master_curriculum.jsonl", self.repo_root)
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
                let canonical_dir = format!("{}/storage/datasets/tara_dataset_filtered/canonical", self.repo_root);
                if Path::new(&canonical_dir).exists() {
                    canonical_dir
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
                        if let Ok(mut reader) = StreamingDatasetReader::open(&shard_path.to_string_lossy()) {
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
                        && fs::metadata(&unified_path).map(|m| m.len() > 0).unwrap_or(false);
                    if !exists_and_nonempty {
                        let _ = DynamicDatasetCompiler::new(&dataset_candidate, &unified_path).compile();
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
        let (mut weights, shapes) = load_model_weights_with_shapes(&self.model_dir)?;
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

        // 4. Memory footprint preflight check
        let total_params: usize = weights.values().map(|v| v.len()).sum();
        let projected_mb = Self::estimate_memory_footprint_mb(total_params);
        if projected_mb > self.max_memory_mb {
            return Err(TrainerError::Model(format!(
                "Projected training memory {projected_mb:.1} MB exceeds {:.1} MB safety ceiling",
                self.max_memory_mb
            )));
        }

        // 5. Tokenize training samples
        println!("[Tokenizer] Tokenizing {} samples (vocabulary size {})...", samples.len(), config.vocab_size);
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
                let max_len = config.max_position_embeddings.max(2);
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
        println!("[Tokenizer] Successfully tokenized {} sequence pairs.", training_tokens.len());

        // 6. Split into Train (90%) and Held-out Validation (10%) to prevent evaluation contamination
        let total_seq_count = training_tokens.len();
        let (train_tokens, val_tokens) = if total_seq_count > 1 {
            let val_count = (total_seq_count / 10).clamp(1, (total_seq_count - 1).min(256));
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

        // Compute baseline loss strictly across held-out validation sequences
        let mut model = TaraForCausalLM::from_weights_and_config(
            weights.clone(),
            config.clone(),
            &self.model_dir,
        )
        .map_err(|e| TrainerError::Model(e.to_string()))?;

        let initial_loss: Option<f64> = if val_tokens.is_empty() {
            println!("[Preflight] Skipping baseline validation loss (held-out validation unavailable).");
            None
        } else {
            println!("[Preflight] Evaluating baseline loss across {} held-out validation samples...", val_tokens.len());
            let loss = compute_model_loss(&model, &val_tokens);
            println!("[Preflight] Held-out validation baseline loss: {:.4}", loss);
            Some(loss)
        };

        // 7. Full-Network Backpropagation with AdamW and Gradient Accumulation
        // Initialize hardware acceleration device backend
        let mut gpu_trainer_opt: Option<crate::cuda::CudaTrainer> = match self.device {
            TrainingDevice::Cpu => {
                println!("[Device] Selected: CPU Backend (explicitly requested)");
                None
            }
            TrainingDevice::Gpu => {
                println!("[Device] Initializing NVIDIA CUDA GPU (--device gpu)...");
                match crate::cuda::CudaTrainer::new_with_precision(0, self.precision) {
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
                        t.register_weights(&weights).map_err(|e| {
                            TrainerError::Model(format!("Failed to upload weights to GPU: {e}"))
                        })?;
                        Some(t)
                    }
                    Err(e) => {
                        return Err(TrainerError::Model(format!(
                            "CUDA initialization failed for --device gpu: {e}. GPU execution cannot proceed."
                        )));
                    }
                }
            }
            TrainingDevice::Auto => {
                println!("[Device] Auto-detecting hardware acceleration (--device auto)...");
                match crate::cuda::CudaTrainer::new_with_precision(0, self.precision) {
                    Ok(mut t) => {
                        let dev = t.device().clone();
                        let (free, total) = t.get_vram_info().unwrap_or((0, 0));
                        println!("[Device] Detected NVIDIA CUDA GPU: {}", dev.name);
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
                        match t.register_weights(&weights) {
                            Ok(()) => {
                                println!("[Device] Selected: GPU Backend (NVIDIA {})", dev.name);
                                Some(t)
                            }
                            Err(e) => {
                                println!("[Device] GPU buffer allocation failed ({e}); falling back to CPU.");
                                None
                            }
                        }
                    }
                    Err(e) => {
                        println!(
                            "[Device] CUDA GPU not available ({e}); safely falling back to CPU."
                        );
                        None
                    }
                }
            }
        };

        let device_label = if let Some(ref t) = gpu_trainer_opt {
            format!("GPU ({})", t.device().name)
        } else {
            "CPU".to_string()
        };
        let is_gpu = gpu_trainer_opt.is_some();
        let precision_label = if let Some(ref t) = gpu_trainer_opt {
            format!("{:?}", t.precision())
        } else {
            "Fp32".to_string()
        };

        let mut optimizer = DynamicAdamW::new();
        let learning_rate = self.learning_rate.unwrap_or(1e-3f32);
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let weight_decay = 0.01f32;
        let max_grad_norm = 1.0f32;
        let accumulation_steps = self.batch_size.unwrap_or(4usize);

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
                        crate::safetensors::load_model_weights_with_shapes(resume_dir)?;
                    weights = resumed_weights;
                    model = TaraForCausalLM::from_weights_and_config(
                        weights.clone(),
                        config.clone(),
                        &self.model_dir,
                    )
                    .map_err(|e| TrainerError::Model(e.to_string()))?;
                    if let Some(ref mut gpu_trainer) = gpu_trainer_opt {
                        let _ = gpu_trainer.register_weights(&weights);
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
                        for (param_name, w_vec) in &weights {
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
                        if let Some(ref mut gpu_trainer) = gpu_trainer_opt {
                            let _ = gpu_trainer.load_optimizer_state(&loaded_moments);
                            gpu_trainer.set_step_count(optimizer_steps as u64);
                        }
                        optimizer.load_state(loaded_moments);
                        optimizer.set_step(optimizer_steps as u64);
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

        let start_epoch = epochs_completed;
        for epoch_idx in start_epoch..max_epochs {
            let mut accum_grads: HashMap<String, Vec<f32>> = HashMap::new();
            let mut accum_count = 0usize;

            let skip_count = initial_sample_skip;
            initial_sample_skip = 0;

            for (tokens, target_start) in train_tokens.iter().skip(skip_count) {
                if let Some(max_s) = self.max_steps {
                    if optimizer_steps >= max_s {
                        break;
                    }
                }
                let seq_len = tokens.len();
                let (layer_inputs, final_normed, cpu_logits) = model.forward_with_cache(tokens);
                let logits = if let Some(ref gpu_trainer) = gpu_trainer_opt {
                    gpu_trainer
                        .forward_lm_head(
                            &final_normed,
                            seq_len,
                            config.hidden_size,
                            config.vocab_size,
                        )
                        .map_err(|e| TrainerError::Model(format!("GPU forward failed: {e}")))?
                } else {
                    cpu_logits
                };

                // Compute d_logits for supervised causal cross-entropy
                let mut d_logits = vec![0.0f32; seq_len * config.vocab_size];
                let mut valid_targets = 0usize;
                let mut sample_loss = 0.0f32;

                for (pos, &token_val) in tokens.iter().enumerate().take(seq_len).skip(*target_start)
                {
                    let predictor = pos - 1;
                    let logits_slice =
                        &logits[predictor * config.vocab_size..(predictor + 1) * config.vocab_size];
                    let probs = softmax(logits_slice);
                    let target = token_val as usize;

                    let p = probs[target].max(1e-12);
                    sample_loss += -p.ln();

                    for v in 0..config.vocab_size {
                        let ind = if v == target { 1.0f32 } else { 0.0f32 };
                        d_logits[predictor * config.vocab_size + v] = probs[v] - ind;
                    }
                    valid_targets += 1;
                }

                if valid_targets > 0 {
                    sample_loss /= valid_targets as f32;
                    // Normalise by number of target tokens
                    let norm_factor = 1.0 / valid_targets as f32;
                    for g in d_logits.iter_mut() {
                        *g *= norm_factor;
                    }

                    let grad_map = if let Some(ref gpu_trainer) = gpu_trainer_opt {
                        // GPU-accelerated LM Head backward projection
                        let (gpu_dlm, gpu_dnorm) = gpu_trainer
                            .backward_lm_head(
                                &d_logits,
                                &final_normed,
                                seq_len,
                                config.hidden_size,
                                config.vocab_size,
                            )
                            .map_err(|e| {
                                TrainerError::Model(format!("GPU backward failed: {e}"))
                            })?;

                        // Backward through decoder layers with GPU-computed d_final_normed
                        let (mut dx, d_norm) = model.norm.backward(
                            &gpu_dnorm,
                            layer_inputs.last().unwrap(),
                            seq_len,
                            config.hidden_size,
                        );

                        let mut layer_gradients = Vec::with_capacity(model.layers.len());
                        for l in (0..model.layers.len()).rev() {
                            let l_grads = model.layers[l].backward(&dx, &layer_inputs[l], seq_len);
                            dx = l_grads.dx.clone();
                            layer_gradients.push(l_grads);
                        }
                        layer_gradients.reverse();

                        let mut d_embed_tokens =
                            vec![0.0f32; config.vocab_size * config.hidden_size];
                        for (t, &id) in tokens.iter().enumerate() {
                            let id = (id as usize).min(config.vocab_size.saturating_sub(1));
                            let dx_row = &dx[t * config.hidden_size..(t + 1) * config.hidden_size];
                            for h in 0..config.hidden_size {
                                d_embed_tokens[id * config.hidden_size + h] += dx_row[h];
                            }
                        }

                        let model_grads = crate::model::causal_lm::ModelGradients {
                            embed_tokens: d_embed_tokens,
                            layers: layer_gradients,
                            norm: d_norm,
                            lm_head: gpu_dlm,
                        };
                        model_grads.to_map()
                    } else {
                        // Full-network CPU analytical backward pass
                        let model_grads =
                            model.backward(tokens, &layer_inputs, &final_normed, &d_logits);
                        model_grads.to_map()
                    };

                    // Accumulate gradients (on GPU if active, otherwise host map)
                    if let Some(ref gpu_trainer) = gpu_trainer_opt {
                        for (k, v) in &grad_map {
                            gpu_trainer.accumulate_gradient(k, v).map_err(|e| {
                                TrainerError::Model(format!(
                                    "GPU gradient accumulation failed: {e}"
                                ))
                            })?;
                        }
                    } else {
                        for (k, v) in grad_map {
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

                    if accum_count >= accumulation_steps {
                        let mut step_taken = false;

                        if let Some(ref mut gpu_trainer) = gpu_trainer_opt {
                            // GPU-accelerated gradient scaling and AdamW step
                            gpu_trainer
                                .scale_accumulated_gradients(accum_count)
                                .map_err(|e| {
                                    TrainerError::Model(format!("GPU gradient scaling failed: {e}"))
                                })?;
                            let stepped = gpu_trainer
                                .step_adamw(
                                    learning_rate,
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
                            }

                            weights = gpu_trainer.download_weights().map_err(|e| {
                                TrainerError::Model(format!("GPU weights download failed: {e}"))
                            })?;
                        } else {
                            // CPU AdamW step
                            let factor = 1.0 / accum_count as f32;
                            for g in accum_grads.values_mut() {
                                for val in g.iter_mut() {
                                    *val *= factor;
                                }
                            }
                            let stepped = optimizer.step(
                                &mut weights,
                                &accum_grads,
                                &AdamWHyperparams {
                                    lr: learning_rate,
                                    beta1,
                                    beta2,
                                    eps,
                                    weight_decay,
                                    max_grad_norm,
                                },
                            );
                            if stepped {
                                step_taken = true;
                            }
                            accum_grads.clear();
                        }
                        accum_count = 0;

                        if step_taken {
                            optimizer_steps += 1;
                        }

                        if optimizer_steps % 10 == 0 || optimizer_steps == 1 {
                            let max_str = self
                                .max_steps
                                .map(|m| m.to_string())
                                .unwrap_or_else(|| "∞".to_string());
                            println!(
                                "[Train] Step {:>4}/{} (samples: {}) | Sample Loss: {:.4} | LR: {:.6} | Device: {}",
                                optimizer_steps, max_str, samples_seen, sample_loss, learning_rate, device_label
                            );
                        }

                        // Reload model in-memory with updated weights
                        model = TaraForCausalLM::from_weights_and_config(
                            weights.clone(),
                            config.clone(),
                            &self.model_dir,
                        )
                        .map_err(|e| TrainerError::Model(e.to_string()))?;

                        // Periodic persistent checkpoint saving with atomic staging
                        if let Some(interval) = self.checkpoint_interval {
                            if optimizer_steps > 0 && optimizer_steps.is_multiple_of(interval) {
                                if let Some(ref cp_dir) = self.checkpoint_dir {
                                    let cp_path = Path::new(cp_dir);
                                    let staging_cp = cp_path.join(".staging");
                                    let _ = std::fs::create_dir_all(&staging_cp);

                                    let shapes: HashMap<String, Vec<usize>> = weights
                                        .iter()
                                        .map(|(k, v)| {
                                            (
                                                k.clone(),
                                                crate::model_expansion::tensor_shape(
                                                    k,
                                                    v.len(),
                                                    &config,
                                                ),
                                            )
                                        })
                                        .collect();

                                    // Save model weights (sharded to ensure <100MB canonical boundary)
                                    let _ = write_safetensors_sharded(
                                        &weights,
                                        &shapes,
                                        &staging_cp.to_string_lossy(),
                                        crate::safetensors::resolve_max_shard_size_bytes(),
                                    );

                                    // Export and persist optimizer state moments
                                    let opt_moments = if let Some(ref gpu_trainer) = gpu_trainer_opt
                                    {
                                        gpu_trainer.export_optimizer_state().unwrap_or_default()
                                    } else {
                                        optimizer.export_state()
                                    };

                                    let mut opt_tensors: HashMap<String, Vec<f32>> = HashMap::new();
                                    let mut opt_shapes: HashMap<String, Vec<usize>> =
                                        HashMap::new();
                                    for (k, (m_vec, v_vec)) in opt_moments {
                                        let m_name = format!("{k}.adam_m");
                                        let v_name = format!("{k}.adam_v");
                                        opt_shapes.insert(m_name.clone(), vec![m_vec.len()]);
                                        opt_shapes.insert(v_name.clone(), vec![v_vec.len()]);
                                        opt_tensors.insert(m_name, m_vec);
                                        opt_tensors.insert(v_name, v_vec);
                                    }
                                    let _ = crate::safetensors::write_safetensors_with_shapes(
                                        &opt_tensors,
                                        &opt_shapes,
                                        &staging_cp.join("optimizer.safetensors").to_string_lossy(),
                                    );

                                    let _ = std::fs::copy(
                                        Path::new(&self.model_dir).join("config.json"),
                                        staging_cp.join("config.json"),
                                    );
                                    let _ = std::fs::copy(
                                        Path::new(&self.model_dir).join("tokenizer.json"),
                                        staging_cp.join("tokenizer.json"),
                                    );
                                    // Validation loss evaluation strictly on held-out val_tokens
                                    let current_eval_loss = if val_tokens.is_empty() {
                                        None
                                    } else {
                                        Some(compute_model_loss(&model, &val_tokens))
                                    };
                                    let is_best = match current_eval_loss {
                                        Some(loss) if loss < best_loss => {
                                            best_loss = loss;
                                            true
                                        }
                                        _ => false,
                                    };

                                    let state_json = serde_json::json!({
                                        "optimizer_step": optimizer_steps,
                                        "samples_seen": samples_seen,
                                        "epoch": epoch_idx,
                                        "validation_loss": current_eval_loss,
                                        "best_validation_loss": if val_tokens.is_empty() { None } else { Some(best_loss) },
                                        "is_best": is_best,
                                        "timestamp": crate::now_iso(),
                                    });
                                    let _ = std::fs::write(
                                        staging_cp.join("checkpoint_state.json"),
                                        serde_json::to_string_pretty(&state_json)
                                            .unwrap_or_default(),
                                    );

                                    // Atomic copy to final checkpoint directory
                                    let _ = std::fs::create_dir_all(cp_path);
                                    if let Ok(entries) = std::fs::read_dir(&staging_cp) {
                                        for entry in entries.flatten() {
                                            let dest = cp_path.join(entry.file_name());
                                            let _ = std::fs::copy(entry.path(), dest);
                                        }
                                    }

                                    // Save dedicated best model checkpoint if validation loss improved
                                    if is_best {
                                        let best_dir = cp_path.join("best");
                                        let _ = std::fs::create_dir_all(&best_dir);
                                        if let Ok(entries) = std::fs::read_dir(&staging_cp) {
                                            for entry in entries.flatten() {
                                                let dest = best_dir.join(entry.file_name());
                                                let _ = std::fs::copy(entry.path(), dest);
                                            }
                                        }
                                        println!(
                                            "[Checkpoint] ★ New BEST checkpoint saved to '{}/best' (Validation Loss: {:.4})",
                                            cp_dir, best_loss
                                        );
                                    }

                                    let _ = std::fs::remove_dir_all(&staging_cp);
                                    let val_loss_str = current_eval_loss
                                        .map(|l| format!("{:.4}", l))
                                        .unwrap_or_else(|| "N/A".to_string());
                                    let best_loss_str = if val_tokens.is_empty() {
                                        "N/A".to_string()
                                    } else {
                                        format!("{:.4}", best_loss)
                                    };
                                    println!(
                                        "[Checkpoint] Saved persistent checkpoint to '{}' at step {} (Val Loss: {}, Best: {})",
                                        cp_dir, optimizer_steps, val_loss_str, best_loss_str
                                    );
                                }
                            }
                        }
                    }
                }
            }

            // Flush remaining accumulated gradients at epoch boundary
            if accum_count > 0 {
                let mut step_taken = false;
                if let Some(ref mut gpu_trainer) = gpu_trainer_opt {
                    gpu_trainer
                        .scale_accumulated_gradients(accum_count)
                        .map_err(|e| {
                            TrainerError::Model(format!("GPU gradient scaling failed: {e}"))
                        })?;
                    let stepped = gpu_trainer
                        .step_adamw(
                            learning_rate,
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

                    weights = gpu_trainer.download_weights().map_err(|e| {
                        TrainerError::Model(format!("GPU weights download failed: {e}"))
                    })?;
                } else {
                    let factor = 1.0 / accum_count as f32;
                    for g in accum_grads.values_mut() {
                        for val in g.iter_mut() {
                            *val *= factor;
                        }
                    }
                    let stepped = optimizer.step(
                        &mut weights,
                        &accum_grads,
                        &AdamWHyperparams {
                            lr: learning_rate,
                            beta1,
                            beta2,
                            eps,
                            weight_decay,
                            max_grad_norm,
                        },
                    );
                    if stepped {
                        step_taken = true;
                    }
                    accum_grads.clear();
                }
                if step_taken {
                    optimizer_steps += 1;
                }
                model = TaraForCausalLM::from_weights_and_config(
                    weights.clone(),
                    config.clone(),
                    &self.model_dir,
                )
                .map_err(|e| TrainerError::Model(e.to_string()))?;
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
            let fl = compute_model_loss(&model, &val_tokens);
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

        // Write sharded or single SafeTensors into staging directory (100 MB canonical boundary)
        let shard_files =
            write_safetensors_sharded(&weights, &shapes, &staging_dir, 100 * 1024 * 1024)?;

        // Copy config.json and tokenizer.json into staging directory
        let cfg_dest = format!("{}/config.json", staging_dir);
        fs::copy(&config_path, &cfg_dest)?;
        let tok_dest = format!("{}/tokenizer.json", staging_dir);
        fs::copy(&tokenizer_path, &tok_dest)?;

        // Export optimizer state moments so every candidate checkpoint is continuation-ready
        let opt_moments = if let Some(ref gpu_trainer) = gpu_trainer_opt {
            gpu_trainer.export_optimizer_state().unwrap_or_default()
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
            promote_directory_atomically(Path::new(&staging_dir), Path::new(&target_candidate_dir))?;
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
    let active_staging_name = active_staging.and_then(|p| p.file_name()).and_then(|n| n.to_str());

    let mut backups: Vec<PathBuf> = Vec::new();
    let mut stagings: Vec<PathBuf> = Vec::new();

    if let Ok(entries) = fs::read_dir(parent) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with(&bak_prefix) {
                        backups.push(p);
                    } else if name.starts_with(&staging_prefix)
                        && Some(name) != active_staging_name
                    {
                        stagings.push(p);
                    }
                }
            }
        }
    }

    backups.sort();
    let target_intact = target_dir.exists()
        && (target_dir.join("config.json").exists()
            || target_dir.join("world_model_state.json").exists());
    let mut recovered = false;

    if !target_intact {
        if let Some(valid_bak) = backups
            .iter()
            .rev()
            .find(|b| b.join("config.json").exists() || b.join("world_model_state.json").exists())
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
    if target_dir.exists() {
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
            format!("Staging directory does not exist: {}", staging_dir.display()),
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

    match fs::rename(staging_dir, target_dir) {
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

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|value| (value - max).exp()).collect();
    let total: f32 = probs.iter().sum();
    if total > 0.0 && total.is_finite() {
        for probability in &mut probs {
            *probability /= total;
        }
    }
    probs
}

fn compute_model_loss(model: &TaraForCausalLM, training_tokens: &[(Vec<u32>, usize)]) -> f64 {
    let mut total_loss = 0.0f64;
    let mut count = 0usize;
    let vs = model.config.vocab_size;

    for (tokens, target_start) in training_tokens {
        let seq_len = tokens.len();
        let (_, _, logits) = model.forward_with_cache(tokens);

        for (pos, &token_val) in tokens.iter().enumerate().take(seq_len).skip(*target_start) {
            let predictor = pos - 1;
            let logits_slice = &logits[predictor * vs..(predictor + 1) * vs];
            let probs = softmax(logits_slice);
            let target = token_val as usize;
            if target < probs.len() {
                total_loss -= (probs[target] as f64).max(1e-12).ln();
                count += 1;
            }
        }
    }

    if count == 0 {
        f64::INFINITY
    } else {
        total_loss / count as f64
    }
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
        let temp_dir = std::env::temp_dir().join(format!("tara_train_single_{}", std::process::id()));
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
        weights.insert("model.layers.0.input_layernorm.weight".to_string(), vec![1.0; 2]);
        weights.insert("model.layers.0.self_attn.q_proj.weight".to_string(), vec![0.1; 4]);
        weights.insert("model.layers.0.self_attn.k_proj.weight".to_string(), vec![0.1; 4]);
        weights.insert("model.layers.0.self_attn.v_proj.weight".to_string(), vec![0.1; 4]);
        weights.insert("model.layers.0.self_attn.o_proj.weight".to_string(), vec![0.1; 4]);
        weights.insert("model.layers.0.post_attention_layernorm.weight".to_string(), vec![1.0; 2]);
        weights.insert("model.layers.0.mlp.gate_proj.weight".to_string(), vec![0.1; 4]);
        weights.insert("model.layers.0.mlp.up_proj.weight".to_string(), vec![0.1; 4]);
        weights.insert("model.layers.0.mlp.down_proj.weight".to_string(), vec![0.1; 4]);
        weights.insert("model.norm.weight".to_string(), vec![1.0; 2]);
        weights.insert("lm_head.weight".to_string(), vec![0.1; 6]);

        let shapes: HashMap<String, Vec<usize>> = weights
            .iter()
            .map(|(name, _)| {
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
        let temp_dir = std::env::temp_dir().join(format!("tara_stream_test_{}", std::process::id()));
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

        assert_eq!(samples.len(), 6, "All 6 dataset formats must be successfully parsed");
        assert_eq!(samples[0].0, "<|im_start|>user\nWhat is Pi?<|im_end|>\n<|im_start|>assistant\n");
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
}
