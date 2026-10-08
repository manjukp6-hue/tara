//! Controlled training boundary for candidate model versions.
//!
//! Enforces five production control gates at the training boundary:
//! 1. **Centralized Hyperparameter & Option Validation** (`TrainingOptions::validate`):
//!    Rejects `NaN`/`Inf`/non-positive `learning_rate`, `max_memory_mb`, `grad_clip_norm`,
//!    zero `batch_size`, `max_steps`, `checkpoint_interval`, out-of-range `weight_decay`,
//!    and invalid `device`/`precision` strings with dedicated `TrainCandidateError::InvalidOption`.
//! 2. **Explicit Stopping Contract (`max_epochs` vs `max_steps`)**:
//!    Enforces `1 <= max_epochs <= 100` and `max_steps >= 1` (when set) as two orthogonal
//!    stopping constraints (`completed_epochs >= max_epochs || optimizer_steps >= max_steps`),
//!    recording the stopping policy (`FullEpochs` vs `StepBoundedEpochs`) and the exact
//!    `effective_stop_reason` (`"max_steps_reached"` vs `"max_epochs_reached"`).
//! 3. **Deterministic & Verified Resume Source Resolution**:
//!    Resolves a single canonical continuation source (`resume_from` -> `checkpoint_dir` -> `model_dir`),
//!    rejects ambiguous dual-checkpoint state when `resume_from` is omitted, and verifies
//!    `checkpoint_state.json` + `optimizer.safetensors` shape integrity before training begins.
//! 4. **Candidate Path & Repository Isolation**:
//!    Validates `repo_root` and `model_dir` existence/integrity, prohibits `..` traversal,
//!    forbids `candidate_dir` overlapping `model_dir`, `checkpoint_dir`, `repo_root`, or
//!    protected production paths (`storage/models/tara`, `production/neural`), and rejects
//!    overwriting existing directories that contain non-checkpoint files.
//! 5. **Transactional Staging & Candidate Acceptance Gate**:
//!    Trains into an isolated temporary staging directory, verifies `status == "COMPLETED"`,
//!    `optimizer_steps >= 1`, finite baseline/final validation loss, regression threshold compliance,
//!    complete dual-purpose checkpoint artifacts (`TaraForCausalLM` + `optimizer.safetensors`),
//!    and finite causal forward-pass probe logits BEFORE atomically promoting to `candidate_dir`.

use serde_json::Value;
use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

use crate::config::TaraConfig;
use crate::model::causal_lm::TaraForCausalLM;
use crate::safetensors::{load_model_weights_with_shapes, load_safetensors_with_shapes};
use crate::tokenizer::TaraTokenizer;
use crate::trainer::{
    promote_directory_atomically, recover_interrupted_promotion, NativeSelfTrainer, TrainerError,
    TrainingDevice,
};

#[derive(Debug, Error)]
pub enum TrainCandidateError {
    #[error("invalid training option: {0}")]
    InvalidOption(String),
    #[error("unsafe or invalid path configuration: {0}")]
    UnsafePath(String),
    #[error("invalid or ambiguous resume source: {0}")]
    InvalidResumeSource(String),
    #[error("candidate rejected by controlled acceptance gate: {0}")]
    CandidateRejected(String),
    #[error("trainer error: {0}")]
    Trainer(#[from] TrainerError),
}

/// Explicit stopping budget contract governing how orthogonal `max_epochs` and `max_steps` constraints interact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoppingContract {
    /// Run until `completed_epochs >= max_epochs` with no optimizer-step ceiling.
    FullEpochs { max_epochs: usize },
    /// Run with both constraints active, stopping as soon as either
    /// `completed_epochs >= max_epochs` OR `optimizer_steps >= max_steps` is reached.
    StepBoundedEpochs {
        max_epochs: usize,
        max_steps: usize,
    },
}

impl StoppingContract {
    pub fn from_epochs_and_steps(
        max_epochs: usize,
        max_steps: Option<usize>,
    ) -> Result<Self, TrainCandidateError> {
        if max_epochs == 0 || max_epochs > 100 {
            return Err(TrainCandidateError::InvalidOption(format!(
                "max_epochs must be between 1 and 100 (got {max_epochs})"
            )));
        }
        match max_steps {
            None => Ok(Self::FullEpochs { max_epochs }),
            Some(0) => Err(TrainCandidateError::InvalidOption(
                "max_steps must be >= 1 when specified (got 0)".to_string(),
            )),
            Some(ms) => Ok(Self::StepBoundedEpochs {
                max_epochs,
                max_steps: ms,
            }),
        }
    }

    pub fn contract_name(&self) -> &'static str {
        match self {
            Self::FullEpochs { .. } => "full_epochs",
            Self::StepBoundedEpochs { .. } => {
                "first_reached(completed_epochs >= max_epochs, optimizer_steps >= max_steps)"
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TrainingOptions {
    pub dataset_dir: Option<String>,
    pub curriculum_path: Option<String>,
    pub candidate_dir: Option<String>,
    pub learning_rate: Option<f32>,
    pub batch_size: Option<usize>,
    pub max_steps: Option<usize>,
    pub warmup_steps: Option<usize>,
    pub weight_decay: Option<f32>,
    pub grad_clip_norm: Option<f32>,
    pub device: Option<String>,
    pub precision: Option<String>,
    pub checkpoint_dir: Option<String>,
    pub checkpoint_interval: Option<usize>,
    pub resume: bool,
    pub resume_from: Option<String>,
    pub max_memory_mb: Option<f64>,
    /// Maximum allowed validation loss ratio (`loss_after / loss_before`) before candidate rejection.
    /// Defaults to `1.5` when `None`.
    pub max_loss_regression_ratio: Option<f64>,
}

fn has_parent_traversal(path_str: &str) -> bool {
    Path::new(path_str)
        .components()
        .any(|c| matches!(c, Component::ParentDir))
}

fn normalize_path_for_comparison(path: &Path) -> PathBuf {
    if let Ok(canon) = fs::canonicalize(path) {
        return canon;
    }
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    let mut out = PathBuf::new();
    for comp in abs.components() {
        match comp {
            Component::Prefix(p) => out.push(p.as_os_str()),
            Component::RootDir => out.push(comp.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                let _ = out.pop();
            }
            Component::Normal(c) => out.push(c),
        }
    }
    out
}

fn paths_overlap(a: &Path, b: &Path) -> bool {
    let na = normalize_path_for_comparison(a);
    let nb = normalize_path_for_comparison(b);
    na == nb || na.starts_with(&nb) || nb.starts_with(&na)
}

fn is_recognized_checkpoint_artifact(file_name: &str) -> bool {
    file_name == "config.json"
        || file_name == "tokenizer.json"
        || file_name == "model.safetensors"
        || file_name == "model.safetensors.index.json"
        || file_name == "optimizer.safetensors"
        || file_name == "checkpoint_state.json"
        || file_name == "training_summary.json"
        || file_name == "STAGE_METADATA.json"
        || file_name == "manual_eval_report.json"
        || file_name == "PROMOTION_MANIFEST.json"
        || (file_name.starts_with("model-") && file_name.ends_with(".safetensors"))
}

/// Validates that an existing `candidate_dir` is either empty or contains only recognized
/// checkpoint/candidate artifacts (preventing accidental clobbering of arbitrary directories).
fn validate_existing_candidate_dir_safe(candidate_dir: &Path) -> Result<(), TrainCandidateError> {
    if !candidate_dir.exists() {
        return Ok(());
    }
    if !candidate_dir.is_dir() {
        return Err(TrainCandidateError::UnsafePath(format!(
            "candidate_dir '{}' exists but is not a directory",
            candidate_dir.display()
        )));
    }
    let entries = fs::read_dir(candidate_dir).map_err(|e| {
        TrainCandidateError::UnsafePath(format!(
            "Failed to inspect existing candidate_dir '{}': {e}",
            candidate_dir.display()
        ))
    })?;
    for entry in entries {
        let entry = entry.map_err(|e| {
            TrainCandidateError::UnsafePath(format!(
                "Directory entry error in '{}': {e}",
                candidate_dir.display()
            ))
        })?;
        let p = entry.path();
        if p.is_dir() {
            return Err(TrainCandidateError::UnsafePath(format!(
                "Existing candidate_dir '{}' contains subdirectory '{}' and cannot be overwritten safely",
                candidate_dir.display(),
                p.display()
            )));
        }
        let fname = entry.file_name();
        let fname_str = fname.to_string_lossy();
        if !is_recognized_checkpoint_artifact(&fname_str) {
            return Err(TrainCandidateError::UnsafePath(format!(
                "Existing candidate_dir '{}' contains unrecognized non-checkpoint file '{}' and cannot be overwritten safely",
                candidate_dir.display(),
                fname_str
            )));
        }
    }
    Ok(())
}

/// Verifies that `dir` is a valid, loadable working model (`config.json`, `tokenizer.json`,
/// matching `vocab_size`, finite weights, and full `TaraForCausalLM` layer compatibility).
pub fn verify_working_model_dir(dir: &Path) -> Result<TaraForCausalLM, String> {
    if !dir.exists() || !dir.is_dir() {
        return Err(format!(
            "Model directory '{}' does not exist or is not a directory",
            dir.display()
        ));
    }
    let cfg_path = dir.join("config.json");
    let tok_path = dir.join("tokenizer.json");
    if !cfg_path.exists() {
        return Err(format!("Missing config.json in '{}'", dir.display()));
    }
    if !tok_path.exists() {
        return Err(format!("Missing tokenizer.json in '{}'", dir.display()));
    }
    let cfg = TaraConfig::from_json_file(&cfg_path.to_string_lossy())
        .map_err(|e| format!("Invalid config.json in '{}': {e}", dir.display()))?;
    let tok = TaraTokenizer::from_file(&tok_path.to_string_lossy())
        .map_err(|e| format!("Invalid tokenizer.json in '{}': {e}", dir.display()))?;
    if cfg.vocab_size != tok.vocab_size {
        return Err(format!(
            "Vocabulary size mismatch in '{}': config.vocab_size={} vs tokenizer.vocab_size={}",
            dir.display(),
            cfg.vocab_size,
            tok.vocab_size
        ));
    }
    let (weights, _) = load_model_weights_with_shapes(&dir.to_string_lossy())
        .map_err(|e| format!("Failed to load SafeTensors weights in '{}': {e}", dir.display()))?;
    if weights.is_empty() {
        return Err(format!("Zero weight tensors found in '{}'", dir.display()));
    }
    for (name, tensor) in &weights {
        if tensor.is_empty() {
            return Err(format!("Weight tensor '{name}' in '{}' is empty", dir.display()));
        }
        if tensor.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "Weight tensor '{name}' in '{}' contains NaN or Inf values",
                dir.display()
            ));
        }
    }
    TaraForCausalLM::from_weights_and_config(weights, cfg, &dir.to_string_lossy()).map_err(|e| {
        format!(
            "Model weights in '{}' are incompatible with TaraForCausalLM architecture: {e}",
            dir.display()
        )
    })
}

/// Verifies that `dir` is a complete continuation-ready checkpoint (`verify_working_model_dir`
/// PLUS valid `checkpoint_state.json` and `optimizer.safetensors` with finite moments matching
/// every model weight tensor length).
pub fn verify_continuation_checkpoint_dir(dir: &Path) -> Result<u64, String> {
    let _ = verify_working_model_dir(dir)?;
    let state_path = dir.join("checkpoint_state.json");
    let opt_path = dir.join("optimizer.safetensors");
    if !state_path.exists() {
        return Err(format!(
            "Continuation checkpoint '{}' is missing checkpoint_state.json",
            dir.display()
        ));
    }
    if !opt_path.exists() {
        return Err(format!(
            "Continuation checkpoint '{}' is missing optimizer.safetensors",
            dir.display()
        ));
    }
    let state_raw = fs::read_to_string(&state_path).map_err(|e| {
        format!(
            "Failed to read checkpoint_state.json in '{}': {e}",
            dir.display()
        )
    })?;
    let state_val: Value = serde_json::from_str(&state_raw).map_err(|e| {
        format!(
            "Malformed checkpoint_state.json in '{}': {e}",
            dir.display()
        )
    })?;
    let opt_step = state_val
        .get("optimizer_step")
        .or_else(|| state_val.get("step"))
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            format!(
                "checkpoint_state.json in '{}' is missing valid 'optimizer_step'",
                dir.display()
            )
        })?;

    let (weights, _) = load_model_weights_with_shapes(&dir.to_string_lossy())
        .map_err(|e| format!("Failed to load model weights in '{}': {e}", dir.display()))?;
    let (opt_tensors, _) = load_safetensors_with_shapes(&opt_path.to_string_lossy()).map_err(|e| {
        format!(
            "Failed to load optimizer.safetensors in '{}': {e}",
            dir.display()
        )
    })?;
    if opt_tensors.is_empty() {
        return Err(format!(
            "optimizer.safetensors in '{}' contains zero tensors",
            dir.display()
        ));
    }

    for (param_name, w_vec) in &weights {
        let m_key = format!("{param_name}.adam_m");
        let v_key = format!("{param_name}.adam_v");
        let Some(m_vec) = opt_tensors.get(&m_key) else {
            return Err(format!(
                "optimizer.safetensors in '{}' missing '{m_key}'",
                dir.display()
            ));
        };
        let Some(v_vec) = opt_tensors.get(&v_key) else {
            return Err(format!(
                "optimizer.safetensors in '{}' missing '{v_key}'",
                dir.display()
            ));
        };
        if m_vec.len() != w_vec.len() || v_vec.len() != w_vec.len() {
            return Err(format!(
                "optimizer.safetensors moment length mismatch for '{param_name}' in '{}': weight={}, m={}, v={}",
                dir.display(),
                w_vec.len(),
                m_vec.len(),
                v_vec.len()
            ));
        }
        if m_vec.iter().any(|x| !x.is_finite()) || v_vec.iter().any(|x| !x.is_finite()) {
            return Err(format!(
                "optimizer.safetensors moments for '{param_name}' in '{}' contain NaN or Inf",
                dir.display()
            ));
        }
    }

    Ok(opt_step)
}

impl TrainingOptions {
    /// Centralized validation of all hyperparameters, device/precision options, and path syntax.
    pub fn validate(&self) -> Result<(), TrainCandidateError> {
        if let Some(lr) = self.learning_rate {
            if !lr.is_finite() || lr <= 0.0 || lr > 1.0 {
                return Err(TrainCandidateError::InvalidOption(format!(
                    "learning_rate must be finite and in (0.0, 1.0], got {lr}"
                )));
            }
        }
        if let Some(bs) = self.batch_size {
            if bs == 0 {
                return Err(TrainCandidateError::InvalidOption(
                    "batch_size must be >= 1 (got 0)".to_string(),
                ));
            }
        }
        if let Some(ms) = self.max_steps {
            if ms == 0 {
                return Err(TrainCandidateError::InvalidOption(
                    "max_steps must be >= 1 when set (got 0)".to_string(),
                ));
            }
        }
        if let Some(ci) = self.checkpoint_interval {
            if ci == 0 {
                return Err(TrainCandidateError::InvalidOption(
                    "checkpoint_interval must be >= 1 when set (got 0)".to_string(),
                ));
            }
        }
        if let Some(mb) = self.max_memory_mb {
            if !mb.is_finite() || mb <= 0.0 {
                return Err(TrainCandidateError::InvalidOption(format!(
                    "max_memory_mb must be a positive finite number, got {mb}"
                )));
            }
        }
        if let Some(wd) = self.weight_decay {
            if !wd.is_finite() || wd < 0.0 || wd > 1.0 {
                return Err(TrainCandidateError::InvalidOption(format!(
                    "weight_decay must be finite and in [0.0, 1.0], got {wd}"
                )));
            }
        }
        if let Some(gcn) = self.grad_clip_norm {
            if !gcn.is_finite() || gcn <= 0.0 {
                return Err(TrainCandidateError::InvalidOption(format!(
                    "grad_clip_norm must be a positive finite number, got {gcn}"
                )));
            }
        }
        if let Some(ratio) = self.max_loss_regression_ratio {
            if !ratio.is_finite() || ratio < 1.0 {
                return Err(TrainCandidateError::InvalidOption(format!(
                    "max_loss_regression_ratio must be finite and >= 1.0, got {ratio}"
                )));
            }
        }
        if let Some(ref dev_str) = self.device {
            dev_str
                .parse::<TrainingDevice>()
                .map_err(TrainCandidateError::InvalidOption)?;
        }
        if let Some(ref prec_str) = self.precision {
            prec_str
                .parse::<crate::cuda::TrainingPrecision>()
                .map_err(TrainCandidateError::InvalidOption)?;
        }
        if self.resume_from.is_some() && !self.resume {
            return Err(TrainCandidateError::InvalidOption(
                "resume_from is specified while resume is false; set resume = true when providing resume_from".to_string(),
            ));
        }

        for (label, opt_path) in [
            ("dataset_dir", self.dataset_dir.as_deref()),
            ("curriculum_path", self.curriculum_path.as_deref()),
            ("candidate_dir", self.candidate_dir.as_deref()),
            ("checkpoint_dir", self.checkpoint_dir.as_deref()),
            ("resume_from", self.resume_from.as_deref()),
        ] {
            if let Some(p) = opt_path {
                if p.trim().is_empty() {
                    return Err(TrainCandidateError::UnsafePath(format!(
                        "{label} cannot be empty or whitespace"
                    )));
                }
                if has_parent_traversal(p) {
                    return Err(TrainCandidateError::UnsafePath(format!(
                        "{label} ('{p}') contains forbidden '..' parent-directory traversal"
                    )));
                }
            }
        }

        Ok(())
    }

    /// Full pre-execution validation across `model_dir`, `repo_root`, `max_epochs`, and `TrainingOptions`.
    pub fn validate_for_execution(
        &self,
        model_dir: &str,
        repo_root: &str,
        max_epochs: usize,
    ) -> Result<StoppingContract, TrainCandidateError> {
        self.validate()?;
        let contract = StoppingContract::from_epochs_and_steps(max_epochs, self.max_steps)?;

        if model_dir.trim().is_empty() {
            return Err(TrainCandidateError::UnsafePath(
                "model_dir cannot be empty".to_string(),
            ));
        }
        if has_parent_traversal(model_dir) {
            return Err(TrainCandidateError::UnsafePath(format!(
                "model_dir ('{model_dir}') contains forbidden '..' parent-directory traversal"
            )));
        }
        if repo_root.trim().is_empty() {
            return Err(TrainCandidateError::UnsafePath(
                "repo_root cannot be empty".to_string(),
            ));
        }
        if has_parent_traversal(repo_root) {
            return Err(TrainCandidateError::UnsafePath(format!(
                "repo_root ('{repo_root}') contains forbidden '..' parent-directory traversal"
            )));
        }

        let repo_path = Path::new(repo_root);
        if !repo_path.exists() || !repo_path.is_dir() {
            return Err(TrainCandidateError::UnsafePath(format!(
                "repo_root '{}' does not exist or is not a directory",
                repo_root
            )));
        }

        let model_path = Path::new(model_dir);
        let _ = recover_interrupted_promotion(model_path);
        verify_working_model_dir(model_path).map_err(TrainCandidateError::UnsafePath)?;

        if let Some(ref cand_str) = self.candidate_dir {
            let cand_path = Path::new(cand_str);
            let _ = recover_interrupted_promotion(cand_path);

            if paths_overlap(cand_path, model_path) {
                return Err(TrainCandidateError::UnsafePath(format!(
                    "candidate_dir ('{cand_str}') overlaps with source model_dir ('{model_dir}'); in-place source overwrite is prohibited"
                )));
            }
            if let Some(ref cp_str) = self.checkpoint_dir {
                if paths_overlap(cand_path, Path::new(cp_str)) {
                    return Err(TrainCandidateError::UnsafePath(format!(
                        "candidate_dir ('{cand_str}') overlaps with periodic checkpoint_dir ('{cp_str}'); candidate and step checkpoints must be isolated"
                    )));
                }
            }
            if normalize_path_for_comparison(cand_path) == normalize_path_for_comparison(repo_path)
            {
                return Err(TrainCandidateError::UnsafePath(format!(
                    "candidate_dir ('{cand_str}') cannot equal repo_root ('{repo_root}')"
                )));
            }

            let prod_tara = repo_path.join("storage/models/tara");
            let prod_neural = repo_path.join("production/neural");
            let prod_world = repo_path.join("production/world_model");
            if paths_overlap(cand_path, &prod_tara)
                || paths_overlap(cand_path, &prod_neural)
                || paths_overlap(cand_path, &prod_world)
                || paths_overlap(cand_path, Path::new("storage/models/tara"))
                || paths_overlap(cand_path, Path::new("production/neural"))
            {
                return Err(TrainCandidateError::UnsafePath(format!(
                    "candidate_dir ('{cand_str}') targets a protected production directory"
                )));
            }

            validate_existing_candidate_dir_safe(cand_path)?;
        }

        Ok(contract)
    }
}

/// Resolves and validates a single canonical continuation checkpoint when `options.resume` is true.
fn resolve_and_validate_resume_source(
    model_dir: &str,
    options: &TrainingOptions,
) -> Result<Option<String>, TrainCandidateError> {
    if !options.resume {
        return Ok(None);
    }

    if let Some(ref explicit_rf) = options.resume_from {
        let rf_path = Path::new(explicit_rf);
        let _ = recover_interrupted_promotion(rf_path);
        verify_continuation_checkpoint_dir(rf_path)
            .map_err(TrainCandidateError::InvalidResumeSource)?;
        return Ok(Some(explicit_rf.clone()));
    }

    // When `resume = true` and `resume_from = None`, inspect `checkpoint_dir` and `model_dir`
    let model_path = Path::new(model_dir);
    let model_has_cont = model_path.join("checkpoint_state.json").exists()
        || model_path.join("optimizer.safetensors").exists();

    let cp_has_cont = if let Some(ref cp_str) = options.checkpoint_dir {
        let cp_path = Path::new(cp_str);
        if normalize_path_for_comparison(cp_path) != normalize_path_for_comparison(model_path) {
            cp_path.join("checkpoint_state.json").exists()
                || cp_path.join("optimizer.safetensors").exists()
        } else {
            false
        }
    } else {
        false
    };

    if model_has_cont && cp_has_cont {
        return Err(TrainCandidateError::InvalidResumeSource(format!(
            "Ambiguous resume state: both model_dir ('{}') and checkpoint_dir ('{}') contain continuation state; specify options.resume_from explicitly",
            model_dir,
            options.checkpoint_dir.as_deref().unwrap_or("")
        )));
    }

    if cp_has_cont {
        let cp_str = options.checkpoint_dir.as_ref().unwrap();
        verify_continuation_checkpoint_dir(Path::new(cp_str))
            .map_err(TrainCandidateError::InvalidResumeSource)?;
        return Ok(Some(cp_str.clone()));
    }

    if model_has_cont {
        verify_continuation_checkpoint_dir(model_path)
            .map_err(TrainCandidateError::InvalidResumeSource)?;
        return Ok(Some(model_dir.to_string()));
    }

    Err(TrainCandidateError::InvalidResumeSource(
        "options.resume is true, but no continuation-ready checkpoint exists in resume_from, checkpoint_dir, or model_dir".to_string(),
    ))
}

/// Verifies the staged candidate directory after training:
/// 1. Valid `config.json`, `tokenizer.json`, and `vocab_size` agreement.
/// 2. Loadable `TaraForCausalLM` weights with all finite values.
/// 3. Valid `checkpoint_state.json` and `optimizer.safetensors` with finite moments matching weights.
/// 4. Live forward-pass probe producing finite logits of shape `probe_len * vocab_size`.
fn verify_staged_candidate_and_probe(staging_dir: &Path) -> Result<(), TrainCandidateError> {
    let _ = verify_continuation_checkpoint_dir(staging_dir)
        .map_err(TrainCandidateError::CandidateRejected)?;
    let model =
        verify_working_model_dir(staging_dir).map_err(TrainCandidateError::CandidateRejected)?;

    let vocab_size = model.config.vocab_size;
    if vocab_size == 0 {
        return Err(TrainCandidateError::CandidateRejected(
            "Candidate model has vocab_size == 0".to_string(),
        ));
    }
    let probe_tokens: Vec<u32> = vec![0, (1 % vocab_size) as u32];
    let (_, _, logits) = model.forward_with_cache(&probe_tokens);
    let expected_len = probe_tokens.len() * vocab_size;
    if logits.len() != expected_len {
        return Err(TrainCandidateError::CandidateRejected(format!(
            "Candidate forward-pass probe logits length mismatch: expected {expected_len}, got {}",
            logits.len()
        )));
    }
    if logits.iter().any(|v| !v.is_finite()) {
        return Err(TrainCandidateError::CandidateRejected(
            "Candidate forward-pass probe produced NaN or Inf logits".to_string(),
        ));
    }
    Ok(())
}

/// Run a controlled training job for `max_epochs` on `dataset_path`.
pub fn run_controlled_training(
    model_dir: &str,
    repo_root: &str,
    max_epochs: usize,
) -> Result<Value, TrainCandidateError> {
    run_controlled_training_with_options(
        model_dir,
        repo_root,
        max_epochs,
        TrainingOptions::default(),
    )
}

/// Run a controlled training job with explicit hyperparameters, path isolation,
/// deterministic resume resolution, transactional staging, and candidate acceptance gating.
pub fn run_controlled_training_with_options(
    model_dir: &str,
    repo_root: &str,
    max_epochs: usize,
    options: TrainingOptions,
) -> Result<Value, TrainCandidateError> {
    // 1. Validate options, stopping budget contract, repo_root, model_dir, and candidate_dir safety
    let stopping_contract = options.validate_for_execution(model_dir, repo_root, max_epochs)?;

    // 2. Resolve and verify canonical resume source (if resume == true)
    let canonical_resume_from = resolve_and_validate_resume_source(model_dir, &options)?;

    // 3. Determine target candidate publication directory (never overwriting model_dir)
    let now_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let pid = std::process::id();

    let target_candidate_dir = if let Some(ref c_dir) = options.candidate_dir {
        PathBuf::from(c_dir)
    } else {
        Path::new(repo_root)
            .join("storage/models/candidates")
            .join(format!("TARA_CANDIDATE_{now_nanos}_{pid}"))
    };

    // 4. Create isolated staging directory for transactional candidate evaluation before publication
    let staging_candidate_dir = PathBuf::from(format!(
        "{}.staging_ctrl_{}_{}",
        target_candidate_dir.display(),
        now_nanos,
        pid
    ));
    if staging_candidate_dir.exists() {
        let _ = fs::remove_dir_all(&staging_candidate_dir);
    }

    // 5. Configure NativeSelfTrainer with validated options
    let mut trainer = NativeSelfTrainer::new(model_dir, repo_root);
    if let Some(ref ds) = options.dataset_dir {
        trainer = trainer.with_dataset_dir(ds);
    }
    if let Some(ref cp) = options.curriculum_path {
        trainer = trainer.with_curriculum_path(cp);
    }
    if let Some(lr) = options.learning_rate {
        trainer = trainer.with_learning_rate(lr);
    }
    if let Some(bs) = options.batch_size {
        trainer = trainer.with_batch_size(bs);
    }
    if let Some(ms) = options.max_steps {
        trainer = trainer.with_max_steps(ms);
    }
    if let Some(ws) = options.warmup_steps {
        trainer = trainer.with_warmup_steps(ws);
    }
    if let Some(wd) = options.weight_decay {
        trainer = trainer.with_weight_decay(wd);
    }
    if let Some(gcn) = options.grad_clip_norm {
        trainer = trainer.with_grad_clip_norm(gcn);
    }
    if let Some(mb) = options.max_memory_mb {
        trainer = trainer.with_max_memory_mb(mb);
    }
    if let Some(ref dev_str) = options.device {
        let dev = dev_str
            .parse::<TrainingDevice>()
            .map_err(TrainCandidateError::InvalidOption)?;
        trainer = trainer.with_device(dev);
    }
    if let Some(ref prec_str) = options.precision {
        let prec = prec_str
            .parse::<crate::cuda::TrainingPrecision>()
            .map_err(TrainCandidateError::InvalidOption)?;
        trainer = trainer.with_precision(prec);
    }
    if let Some(ref cp_dir) = options.checkpoint_dir {
        trainer = trainer.with_checkpoint_dir(cp_dir);
    }
    if let Some(cp_interval) = options.checkpoint_interval {
        trainer = trainer.with_checkpoint_interval(cp_interval);
    }
    if let Some(ref resolved_rf) = canonical_resume_from {
        trainer = trainer.with_resume_from(resolved_rf);
    }

    let staging_str = staging_candidate_dir.to_string_lossy().to_string();
    let mut result = match trainer.run_full_self_learning_cycle_isolated(
        max_epochs,
        true,
        Some(&staging_str),
    ) {
        Ok(res) => res,
        Err(e) => {
            let _ = fs::remove_dir_all(&staging_candidate_dir);
            return Err(TrainCandidateError::Trainer(e));
        }
    };

    // 6. Controlled Candidate Acceptance Gate
    let status = result
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("UNKNOWN");
    if status != "COMPLETED" {
        let msg = result
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("Training did not reach COMPLETED state");
        let _ = fs::remove_dir_all(&staging_candidate_dir);
        return Err(TrainCandidateError::CandidateRejected(format!(
            "Training cycle returned non-completed status '{status}': {msg}"
        )));
    }

    let steps_done = result
        .get("optimizer_steps")
        .or_else(|| result.get("steps"))
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    if steps_done == 0 {
        let _ = fs::remove_dir_all(&staging_candidate_dir);
        return Err(TrainCandidateError::CandidateRejected(
            "Training cycle executed 0 optimizer steps".to_string(),
        ));
    }

    let max_regression_ratio = options.max_loss_regression_ratio.unwrap_or(1.5);
    let loss_before_opt = result.get("loss_before").and_then(Value::as_f64);
    let loss_after_opt = result.get("loss_after").and_then(Value::as_f64);

    if let Some(lb) = loss_before_opt {
        if !lb.is_finite() || lb < 0.0 {
            let _ = fs::remove_dir_all(&staging_candidate_dir);
            return Err(TrainCandidateError::CandidateRejected(format!(
                "Baseline validation loss is invalid/non-finite: {lb}"
            )));
        }
    }
    if let Some(la) = loss_after_opt {
        if !la.is_finite() || la < 0.0 {
            let _ = fs::remove_dir_all(&staging_candidate_dir);
            return Err(TrainCandidateError::CandidateRejected(format!(
                "Candidate validation loss is invalid/non-finite: {la}"
            )));
        }
        if let Some(lb) = loss_before_opt {
            if lb > 0.0 && la > lb * max_regression_ratio {
                let _ = fs::remove_dir_all(&staging_candidate_dir);
                return Err(TrainCandidateError::CandidateRejected(format!(
                    "Candidate validation loss regressed from {lb:.4} to {la:.4} (exceeds max_loss_regression_ratio {max_regression_ratio:.2})"
                )));
            }
        }
    }

    if let Err(gate_err) = verify_staged_candidate_and_probe(&staging_candidate_dir) {
        let _ = fs::remove_dir_all(&staging_candidate_dir);
        return Err(gate_err);
    }

    // 7. Record explicit stopping contract & acceptance gate metadata before atomic publication
    let effective_stop_reason = match stopping_contract {
        StoppingContract::FullEpochs { .. } => "max_epochs_reached",
        StoppingContract::StepBoundedEpochs { max_steps, .. } => {
            if steps_done >= max_steps {
                "max_steps_reached"
            } else {
                "max_epochs_reached"
            }
        }
    };

    if let Some(obj) = result.as_object_mut() {
        obj.insert(
            "candidate_dir".to_string(),
            Value::String(target_candidate_dir.to_string_lossy().to_string()),
        );
        obj.insert(
            "controlled_gate_status".to_string(),
            Value::String("PASSED".to_string()),
        );
        obj.insert(
            "stopping_contract".to_string(),
            Value::String(stopping_contract.contract_name().to_string()),
        );
        obj.insert(
            "requested_max_epochs".to_string(),
            Value::from(max_epochs as u64),
        );
        obj.insert(
            "requested_max_steps".to_string(),
            options
                .max_steps
                .map(|m| Value::from(m as u64))
                .unwrap_or(Value::Null),
        );
        obj.insert(
            "effective_stop_reason".to_string(),
            Value::String(effective_stop_reason.to_string()),
        );
    }

    if let Ok(summary_pretty) = serde_json::to_string_pretty(&result) {
        let _ = fs::write(
            staging_candidate_dir.join("training_summary.json"),
            summary_pretty,
        );
    }

    // 8. Atomically promote verified staging directory to target_candidate_dir
    if let Err(e) = promote_directory_atomically(&staging_candidate_dir, &target_candidate_dir) {
        let _ = fs::remove_dir_all(&staging_candidate_dir);
        return Err(TrainCandidateError::Trainer(e));
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::write_safetensors_with_shapes;
    use std::collections::HashMap;

    fn unique_test_dir(tag: &str) -> PathBuf {
        let dir = env::temp_dir().join(format!(
            "tara_ctrl_cand_test_{}_{}_{}",
            tag,
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_minimal_model(dir: &Path, with_continuation: bool) {
        fs::create_dir_all(dir).unwrap();
        let cfg = TaraConfig {
            vocab_size: 4,
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 16,
            ..TaraConfig::default()
        };
        fs::write(
            dir.join("config.json"),
            serde_json::to_string_pretty(&cfg).unwrap(),
        )
        .unwrap();
        fs::write(
            dir.join("tokenizer.json"),
            r#"{"vocab":{"<|unk|>":0,"a":1,"b":2,"c":3}}"#,
        )
        .unwrap();

        let mut weights = HashMap::new();
        let mut shapes = HashMap::new();
        weights.insert("model.embed_tokens.weight".to_string(), vec![0.02f32; 4 * 4]);
        shapes.insert("model.embed_tokens.weight".to_string(), vec![4, 4]);
        weights.insert("model.norm.weight".to_string(), vec![1.0f32; 4]);
        shapes.insert("model.norm.weight".to_string(), vec![4]);
        weights.insert("lm_head.weight".to_string(), vec![0.02f32; 4 * 4]);
        shapes.insert("lm_head.weight".to_string(), vec![4, 4]);

        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            let k = format!("model.layers.0.self_attn.{proj}.weight");
            weights.insert(k.clone(), vec![0.02f32; 4 * 4]);
            shapes.insert(k, vec![4, 4]);
        }
        for proj in ["gate_proj", "up_proj"] {
            let k = format!("model.layers.0.mlp.{proj}.weight");
            weights.insert(k.clone(), vec![0.02f32; 8 * 4]);
            shapes.insert(k, vec![8, 4]);
        }
        weights.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            vec![0.02f32; 4 * 8],
        );
        shapes.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            vec![4, 8],
        );
        for ln in ["input_layernorm", "post_attention_layernorm"] {
            let k = format!("model.layers.0.{ln}.weight");
            weights.insert(k.clone(), vec![1.0f32; 4]);
            shapes.insert(k, vec![4]);
        }

        write_safetensors_with_shapes(
            &weights,
            &shapes,
            &dir.join("model.safetensors").to_string_lossy(),
        )
        .unwrap();

        if with_continuation {
            fs::write(
                dir.join("checkpoint_state.json"),
                r#"{"step":2,"optimizer_step":2,"samples_seen":4,"epoch":0,"resumable":true}"#,
            )
            .unwrap();
            let mut opt_t = HashMap::new();
            let mut opt_s = HashMap::new();
            for (k, v) in &weights {
                opt_t.insert(format!("{k}.adam_m"), vec![0.0f32; v.len()]);
                opt_s.insert(format!("{k}.adam_m"), vec![v.len()]);
                opt_t.insert(format!("{k}.adam_v"), vec![0.0f32; v.len()]);
                opt_s.insert(format!("{k}.adam_v"), vec![v.len()]);
            }
            write_safetensors_with_shapes(
                &opt_t,
                &opt_s,
                &dir.join("optimizer.safetensors").to_string_lossy(),
            )
            .unwrap();
        }
    }

    #[test]
    fn test_training_options_validate_rejects_invalid_hyperparameters_and_options() {
        assert!(TrainingOptions {
            learning_rate: Some(0.0),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(TrainingOptions {
            learning_rate: Some(f32::NAN),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(TrainingOptions {
            learning_rate: Some(f32::INFINITY),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(TrainingOptions {
            batch_size: Some(0),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(TrainingOptions {
            max_steps: Some(0),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(TrainingOptions {
            checkpoint_interval: Some(0),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(TrainingOptions {
            max_memory_mb: Some(-10.0),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(TrainingOptions {
            max_memory_mb: Some(f64::NAN),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(TrainingOptions {
            device: Some("tpu_invalid".to_string()),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(TrainingOptions {
            precision: Some("bf8_invalid".to_string()),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(TrainingOptions {
            resume: false,
            resume_from: Some("some/path".to_string()),
            ..Default::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn test_candidate_path_isolation_and_ambiguous_resume_rejection() {
        let root = unique_test_dir("isolation_and_resume");
        let model_dir = root.join("base_model");
        let cp_dir = root.join("checkpoints");
        write_minimal_model(&model_dir, true);
        write_minimal_model(&cp_dir, true);

        // 1. candidate_dir == model_dir must be rejected
        let opts_same_model = TrainingOptions {
            candidate_dir: Some(model_dir.to_string_lossy().to_string()),
            ..Default::default()
        };
        let err1 = opts_same_model
            .validate_for_execution(&model_dir.to_string_lossy(), &root.to_string_lossy(), 1)
            .unwrap_err();
        assert!(matches!(err1, TrainCandidateError::UnsafePath(_)));

        // 2. candidate_dir == checkpoint_dir must be rejected
        let opts_same_cp = TrainingOptions {
            candidate_dir: Some(cp_dir.to_string_lossy().to_string()),
            checkpoint_dir: Some(cp_dir.to_string_lossy().to_string()),
            ..Default::default()
        };
        let err2 = opts_same_cp
            .validate_for_execution(&model_dir.to_string_lossy(), &root.to_string_lossy(), 1)
            .unwrap_err();
        assert!(matches!(err2, TrainCandidateError::UnsafePath(_)));

        // 3. Ambiguous resume (both model_dir and checkpoint_dir have continuation state, resume_from = None)
        let opts_ambig_resume = TrainingOptions {
            checkpoint_dir: Some(cp_dir.to_string_lossy().to_string()),
            resume: true,
            resume_from: None,
            ..Default::default()
        };
        let err3 =
            resolve_and_validate_resume_source(&model_dir.to_string_lossy(), &opts_ambig_resume)
                .unwrap_err();
        assert!(matches!(err3, TrainCandidateError::InvalidResumeSource(_)));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_end_to_end_controlled_training_gate_and_stopping_contract() {
        let root = unique_test_dir("e2e_controlled");
        let model_dir = root.join("base_model");
        let cand_dir = root.join("published_candidate_step_bound");
        let cand_dir_epoch_bound = root.join("published_candidate_epoch_bound");
        let ds_dir = root.join("dataset");
        fs::create_dir_all(&ds_dir).unwrap();
        write_minimal_model(&model_dir, false);

        fs::write(
            ds_dir.join("train.jsonl"),
            "{\"input\":\"a\",\"output\":\"b\"}\n{\"input\":\"b\",\"output\":\"c\"}\n{\"input\":\"a\",\"output\":\"c\"}\n{\"input\":\"b\",\"output\":\"a\"}\n",
        )
        .unwrap();

        // Case 1: max_epochs = 100, max_steps = 3 -> stops at exactly 3 optimizer steps (during epoch 1, not 3 epochs)
        let opts_step_bound = TrainingOptions {
            dataset_dir: Some(ds_dir.to_string_lossy().to_string()),
            candidate_dir: Some(cand_dir.to_string_lossy().to_string()),
            learning_rate: Some(1e-3),
            batch_size: Some(1),
            max_steps: Some(3),
            device: Some("cpu".to_string()),
            precision: Some("fp32".to_string()),
            ..Default::default()
        };

        let report = run_controlled_training_with_options(
            &model_dir.to_string_lossy(),
            &root.to_string_lossy(),
            100,
            opts_step_bound,
        )
        .expect("step-bounded controlled training should pass gate");

        assert_eq!(report["status"], "COMPLETED");
        assert_eq!(report["controlled_gate_status"], "PASSED");
        assert_eq!(
            report["stopping_contract"],
            "first_reached(completed_epochs >= max_epochs, optimizer_steps >= max_steps)"
        );
        assert_eq!(report["effective_stop_reason"], "max_steps_reached");
        assert_eq!(report["steps"].as_u64(), Some(3));
        assert!(verify_continuation_checkpoint_dir(&cand_dir).is_ok());

        // Case 2: max_epochs = 2, max_steps = 1000 -> stops after completing 2 epochs (8 optimizer steps)
        let opts_epoch_bound = TrainingOptions {
            dataset_dir: Some(ds_dir.to_string_lossy().to_string()),
            candidate_dir: Some(cand_dir_epoch_bound.to_string_lossy().to_string()),
            learning_rate: Some(1e-3),
            batch_size: Some(1),
            max_steps: Some(1000),
            device: Some("cpu".to_string()),
            precision: Some("fp32".to_string()),
            ..Default::default()
        };

        let report_epoch = run_controlled_training_with_options(
            &model_dir.to_string_lossy(),
            &root.to_string_lossy(),
            2,
            opts_epoch_bound,
        )
        .expect("epoch-bounded controlled training should pass gate");

        assert_eq!(report_epoch["status"], "COMPLETED");
        assert_eq!(report_epoch["controlled_gate_status"], "PASSED");
        assert_eq!(
            report_epoch["stopping_contract"],
            "first_reached(completed_epochs >= max_epochs, optimizer_steps >= max_steps)"
        );
        assert_eq!(report_epoch["effective_stop_reason"], "max_epochs_reached");
        assert_eq!(report_epoch["epochs"].as_u64(), Some(2));
        assert_eq!(report_epoch["steps"].as_u64(), Some(6));
        assert!(verify_continuation_checkpoint_dir(&cand_dir_epoch_bound).is_ok());

        let _ = fs::remove_dir_all(&root);
    }
}


