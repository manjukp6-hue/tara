//! TARA Manual Training Workspace CLI & Runner (100% Native Rust).
//!
//! Enforces:
//! - Complete separation between Manual Training and Autonomous Training.
//! - Uses the SHARED TRAINING INFRASTRUCTURE (`NativeSelfTrainer`, `DatasetEngine`,
//!   `CurriculumEngine`, `SafeTensors`, `promote_directory_atomically`).
//! - Staging isolation & path containment:
//!   * Manual Neural Checkpoints -> `manual_training/checkpoints/neural/`
//!   * Manual World Model State  -> `manual_training/checkpoints/world_model/`
//!   * Production Repository     -> `storage/models/tara/` (Never overwritten during training;
//!     only updated via atomic two-phase promotion after passing evaluation gates)
//! - Fail-closed error handling: malformed config JSON, invalid CLI flags, missing source models,
//!   training errors, and failed promotion gates all return `Err(...)` (non-zero exit code).
//! - Zero Python, zero mocks, zero synthetic snapshots, zero hardcoded hashes.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use tara_engine::config::TaraConfig;
use tara_engine::curriculum_engine::{CurriculumConfig, CurriculumEngine};
use tara_engine::dataset_engine::{DatasetEngine, DatasetEngineConfig, DatasetSource};
use tara_engine::tokenizer::TaraTokenizer;
use tara_engine::train_candidate::{run_controlled_training_with_options, TrainingOptions};
use tara_engine::{
    compute_sha256, load_model_weights_with_shapes, load_safetensors_with_shapes,
    promote_directory_atomically, recover_interrupted_promotion, SkillsEvaluator,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManualTrainingConfig {
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default = "default_mode")]
    pub training_mode: String, // "neural", "world_model", or "both"
    #[serde(default = "default_model_dir")]
    pub model_source_dir: String,
    #[serde(default = "default_dataset_dir")]
    pub dataset_dir: String,
    #[serde(default = "default_curriculum_path")]
    pub curriculum_path: String,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f32,
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    #[serde(default = "default_warmup_steps")]
    pub warmup_steps: usize,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f32,
    #[serde(default = "default_grad_clip_norm")]
    pub grad_clip_norm: f32,
    #[serde(default = "default_checkpoint_interval")]
    pub checkpoint_interval_steps: usize,
    #[serde(default = "default_neural_chk_dir")]
    pub neural_checkpoint_dir: String,
    #[serde(default = "default_world_model_chk_dir")]
    pub world_model_checkpoint_dir: String,
    #[serde(default = "default_logs_dir")]
    pub logs_dir: String,
    #[serde(default = "default_runs_dir")]
    pub runs_dir: String,
    #[serde(default = "default_evaluation_dir")]
    pub evaluation_dir: String,
}

fn default_version() -> String {
    "1.0.0".to_string()
}
fn default_mode() -> String {
    "neural".to_string()
}
fn default_model_dir() -> String {
    "storage/models/tara".to_string()
}
fn default_dataset_dir() -> String {
    "storage/datasets/tara_dataset_filtered/canonical".to_string()
}
fn default_curriculum_path() -> String {
    "storage/datasets/curriculum/master_curriculum.jsonl".to_string()
}
fn default_batch_size() -> usize {
    4
}
fn default_learning_rate() -> f32 {
    0.0003
}
fn default_max_steps() -> usize {
    100
}
fn default_warmup_steps() -> usize {
    10
}
fn default_weight_decay() -> f32 {
    0.01
}
fn default_grad_clip_norm() -> f32 {
    1.0
}
fn default_checkpoint_interval() -> usize {
    50
}
fn default_neural_chk_dir() -> String {
    "manual_training/checkpoints/neural".to_string()
}
fn default_world_model_chk_dir() -> String {
    "manual_training/checkpoints/world_model".to_string()
}
fn default_logs_dir() -> String {
    "manual_training/logs".to_string()
}
fn default_runs_dir() -> String {
    "manual_training/runs".to_string()
}
fn default_evaluation_dir() -> String {
    "manual_training/evaluation".to_string()
}

impl Default for ManualTrainingConfig {
    fn default() -> Self {
        Self {
            version: default_version(),
            training_mode: default_mode(),
            model_source_dir: default_model_dir(),
            dataset_dir: default_dataset_dir(),
            curriculum_path: default_curriculum_path(),
            batch_size: default_batch_size(),
            learning_rate: default_learning_rate(),
            max_steps: default_max_steps(),
            warmup_steps: default_warmup_steps(),
            weight_decay: default_weight_decay(),
            grad_clip_norm: default_grad_clip_norm(),
            checkpoint_interval_steps: default_checkpoint_interval(),
            neural_checkpoint_dir: default_neural_chk_dir(),
            world_model_checkpoint_dir: default_world_model_chk_dir(),
            logs_dir: default_logs_dir(),
            runs_dir: default_runs_dir(),
            evaluation_dir: default_evaluation_dir(),
        }
    }
}

pub fn normalize_for_comparison(path: &Path) -> PathBuf {
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

    let mut normalized = PathBuf::new();
    for comp in abs.components() {
        match comp {
            Component::Prefix(p) => normalized.push(p.as_os_str()),
            Component::RootDir => normalized.push(comp.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                let _ = normalized.pop();
            }
            Component::Normal(c) => normalized.push(c),
        }
    }
    normalized
}

pub fn paths_overlap(a: &Path, b: &Path) -> bool {
    let na = normalize_for_comparison(a);
    let nb = normalize_for_comparison(b);
    na == nb || na.starts_with(&nb) || nb.starts_with(&na)
}

pub const EVALUATION_SUITE_ID: &str = "tara_core_competencies_v1";
pub const PROMOTION_PASS_RATE_THRESHOLD: f64 = 0.60;
pub const CANONICAL_PROMOTION_PROMPTS: &[&str] = &[
    "Explain the Pythagorean theorem in geometry.",
    "Write a simple function to reverse a vector in Rust.",
    "What is a causal graph in probabilistic reasoning?",
    "Explain how AdamW optimizer handles weight decay.",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStatus {
    TrainingCompleted,
    EvaluationPassed,
    Promoted,
    PromotionRejected,
    ValidationError,
    ExecutionFailed,
}

impl RunStatus {
    pub fn exit_code(self) -> i32 {
        match self {
            RunStatus::TrainingCompleted | RunStatus::EvaluationPassed | RunStatus::Promoted => 0,
            RunStatus::PromotionRejected => 10,
            RunStatus::ValidationError => 20,
            RunStatus::ExecutionFailed => 30,
        }
    }
}

pub fn has_parent_dir_traversal(path: &Path) -> bool {
    path.components().any(|c| matches!(c, Component::ParentDir))
}

pub fn validate_manual_config(config: &ManualTrainingConfig) -> Result<(), String> {
    if config.version.trim() != "1.0.0" {
        return Err(format!(
            "Unsupported manual config version '{}': expected '1.0.0'",
            config.version
        ));
    }
    match config.training_mode.as_str() {
        "neural" | "world_model" | "both" => {}
        other => {
            return Err(format!(
                "Invalid training_mode '{}': must be 'neural', 'world_model', or 'both'",
                other
            ));
        }
    }
    if config.batch_size == 0 {
        return Err("batch_size must be >= 1".to_string());
    }
    if config.max_steps == 0 {
        return Err("max_steps must be >= 1".to_string());
    }
    if config.checkpoint_interval_steps == 0 {
        return Err("checkpoint_interval_steps must be >= 1".to_string());
    }
    if config.checkpoint_interval_steps > config.max_steps {
        return Err(format!(
            "checkpoint_interval_steps ({}) must be <= max_steps ({})",
            config.checkpoint_interval_steps, config.max_steps
        ));
    }
    if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
        return Err(format!(
            "learning_rate must be a positive finite float, got {}",
            config.learning_rate
        ));
    }
    if !config.weight_decay.is_finite() || config.weight_decay < 0.0 {
        return Err(format!(
            "weight_decay must be a non-negative finite float, got {}",
            config.weight_decay
        ));
    }
    if !config.grad_clip_norm.is_finite() || config.grad_clip_norm <= 0.0 {
        return Err(format!(
            "grad_clip_norm must be a positive finite float, got {}",
            config.grad_clip_norm
        ));
    }

    let managed_dirs = [
        ("neural_checkpoint_dir", Path::new(&config.neural_checkpoint_dir)),
        ("world_model_checkpoint_dir", Path::new(&config.world_model_checkpoint_dir)),
        ("logs_dir", Path::new(&config.logs_dir)),
        ("runs_dir", Path::new(&config.runs_dir)),
        ("evaluation_dir", Path::new(&config.evaluation_dir)),
    ];

    for (label, dir) in &managed_dirs {
        if dir.as_os_str().is_empty() {
            return Err(format!("{label} cannot be empty"));
        }
        if has_parent_dir_traversal(dir) {
            return Err(format!(
                "{label} ('{}') contains forbidden '..' parent directory traversal",
                dir.display()
            ));
        }
    }

    let model_src = Path::new(&config.model_source_dir);
    let protected_dirs = [
        Path::new("storage/models/tara"),
        Path::new("production/neural"),
        Path::new("production/world_model"),
        Path::new("storage/persistence"),
    ];

    for (label, dir) in &managed_dirs {
        for prot in &protected_dirs {
            if paths_overlap(dir, prot) {
                return Err(format!(
                    "{label} ('{}') overlaps with protected directory '{}'",
                    dir.display(),
                    prot.display()
                ));
            }
        }
        if paths_overlap(dir, model_src) {
            return Err(format!(
                "{label} ('{}') overlaps with model_source_dir ('{}'). In-place overwrite of source model is prohibited.",
                dir.display(),
                config.model_source_dir
            ));
        }
    }

    for i in 0..managed_dirs.len() {
        for j in (i + 1)..managed_dirs.len() {
            let (label_a, dir_a) = managed_dirs[i];
            let (label_b, dir_b) = managed_dirs[j];
            if paths_overlap(dir_a, dir_b) {
                return Err(format!(
                    "{label_a} ('{}') overlaps with {label_b} ('{}')",
                    dir_a.display(),
                    dir_b.display()
                ));
            }
        }
    }

    Ok(())
}

pub fn load_manual_config(
    config_path: &Path,
    explicit_path: bool,
) -> Result<ManualTrainingConfig, String> {
    if !config_path.exists() {
        if explicit_path {
            return Err(format!(
                "Specified --config file does not exist: {}",
                config_path.display()
            ));
        }
        println!(
            "  [Notice] Default config file {:?} not found, using default configuration.",
            config_path
        );
        return Ok(ManualTrainingConfig::default());
    }

    let data = fs::read_to_string(config_path)
        .map_err(|e| format!("Failed to read config file '{}': {}", config_path.display(), e))?;
    serde_json::from_str::<ManualTrainingConfig>(&data).map_err(|e| {
        format!(
            "Malformed JSON in config file '{}': {}",
            config_path.display(),
            e
        )
    })
}

fn require_arg_value(args: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    if *i + 1 >= args.len() {
        return Err(format!("Missing value for CLI option '{}'", flag));
    }
    let val = &args[*i + 1];
    if val.starts_with("--") {
        return Err(format!(
            "Option '{}' requires a value, but found flag '{}'",
            flag, val
        ));
    }
    *i += 1;
    Ok(val.clone())
}

#[derive(Debug, Default)]
pub struct ParsedManualCli {
    pub config_path: PathBuf,
    pub explicit_config: bool,
    pub mode_override: Option<String>,
    pub steps_override: Option<usize>,
    pub batch_size_override: Option<usize>,
    pub lr_override: Option<f32>,
    pub dataset_override: Option<String>,
    pub resume: bool,
    pub resume_from: Option<String>,
    pub fresh: bool,
    pub run_evaluation: bool,
    pub run_promote: bool,
    pub show_help: bool,
}

pub fn parse_manual_cli(
    args: &[String],
    default_config_path: PathBuf,
) -> Result<ParsedManualCli, String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        return Ok(ParsedManualCli {
            show_help: true,
            ..Default::default()
        });
    }

    let mut out = ParsedManualCli {
        config_path: default_config_path,
        ..Default::default()
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                let val = require_arg_value(args, &mut i, "--config")?;
                out.config_path = PathBuf::from(val);
                out.explicit_config = true;
            }
            "--mode" => {
                let val = require_arg_value(args, &mut i, "--mode")?;
                if !matches!(val.as_str(), "neural" | "world_model" | "both") {
                    return Err(format!(
                        "Invalid --mode '{}': must be 'neural', 'world_model', or 'both'",
                        val
                    ));
                }
                out.mode_override = Some(val);
            }
            "--steps" => {
                let val = require_arg_value(args, &mut i, "--steps")?;
                let s = val
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid integer for --steps: '{}'", val))?;
                if s == 0 {
                    return Err("--steps must be >= 1".to_string());
                }
                out.steps_override = Some(s);
            }
            "--batch-size" => {
                let val = require_arg_value(args, &mut i, "--batch-size")?;
                let b = val
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid integer for --batch-size: '{}'", val))?;
                if b == 0 {
                    return Err("--batch-size must be >= 1".to_string());
                }
                out.batch_size_override = Some(b);
            }
            "--lr" => {
                let val = require_arg_value(args, &mut i, "--lr")?;
                let lr = val
                    .parse::<f32>()
                    .map_err(|_| format!("Invalid float for --lr: '{}'", val))?;
                if !lr.is_finite() || lr <= 0.0 {
                    return Err("--lr must be a positive finite float".to_string());
                }
                out.lr_override = Some(lr);
            }
            "--dataset" => {
                let val = require_arg_value(args, &mut i, "--dataset")?;
                out.dataset_override = Some(val);
            }
            "--resume" => {
                out.resume = true;
            }
            "--resume-from" => {
                let val = require_arg_value(args, &mut i, "--resume-from")?;
                out.resume_from = Some(val);
                out.resume = true;
            }
            "--fresh" => {
                out.fresh = true;
            }
            "--evaluate" => {
                out.run_evaluation = true;
            }
            "--promote" => {
                out.run_promote = true;
            }
            unknown => {
                return Err(format!("Unknown CLI option: '{}'", unknown));
            }
        }
        i += 1;
    }

    if out.run_evaluation && out.run_promote {
        return Err(
            "Conflicting CLI flags: '--evaluate' and '--promote' are mutually exclusive action modes"
                .to_string(),
        );
    }
    if out.fresh && (out.resume || out.resume_from.is_some()) {
        return Err(
            "Conflicting CLI flags: '--fresh' cannot be combined with '--resume' or '--resume-from'"
                .to_string(),
        );
    }
    let has_training_flags = out.mode_override.is_some()
        || out.steps_override.is_some()
        || out.batch_size_override.is_some()
        || out.lr_override.is_some()
        || out.dataset_override.is_some()
        || out.resume
        || out.resume_from.is_some()
        || out.fresh;
    if (out.run_evaluation || out.run_promote) && has_training_flags {
        return Err(
            "Conflicting CLI flags: training flags (--mode, --steps, --batch-size, --lr, --dataset, --resume, --resume-from, --fresh) cannot be combined with '--evaluate' or '--promote'"
                .to_string(),
        );
    }

    Ok(out)
}

/// Computes dynamic SHA-256 over a candidate model's SafeTensors file(s).
pub fn compute_candidate_weights_sha256(model_dir: &Path) -> Result<String, String> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return compute_sha256(&single.to_string_lossy())
            .map_err(|e| format!("Failed to hash '{}': {e}", single.display()));
    }
    let index = model_dir.join("model.safetensors.index.json");
    if index.exists() {
        return compute_sha256(&index.to_string_lossy())
            .map_err(|e| format!("Failed to hash '{}': {e}", index.display()));
    }
    Err(format!(
        "Neither model.safetensors nor model.safetensors.index.json found in '{}'",
        model_dir.display()
    ))
}

/// Verifies that a model directory contains valid `config.json`, `tokenizer.json`, and SafeTensors weights
/// whose shapes match `TaraConfig`.
pub fn verify_model_bundle(model_dir: &Path) -> Result<usize, String> {
    if !model_dir.exists() || !model_dir.is_dir() {
        return Err(format!(
            "Model directory does not exist: {}",
            model_dir.display()
        ));
    }
    let cfg_path = model_dir.join("config.json");
    if !cfg_path.exists() {
        return Err(format!("config.json missing in '{}'", model_dir.display()));
    }
    let cfg = TaraConfig::from_json_file(&cfg_path.to_string_lossy())
        .map_err(|e| format!("Invalid config.json in '{}': {}", model_dir.display(), e))?;

    let tok_path = model_dir.join("tokenizer.json");
    if !tok_path.exists() {
        return Err(format!(
            "tokenizer.json missing in '{}'",
            model_dir.display()
        ));
    }
    TaraTokenizer::from_file(&tok_path.to_string_lossy())
        .map_err(|e| format!("Invalid tokenizer.json in '{}': {}", model_dir.display(), e))?;

    let (weights, _) = load_model_weights_with_shapes(&model_dir.to_string_lossy())
        .map_err(|e| format!("Failed to load SafeTensors weights in '{}': {}", model_dir.display(), e))?;
    if weights.is_empty() {
        return Err(format!(
            "SafeTensors weights dictionary is empty in '{}'",
            model_dir.display()
        ));
    }
    for req in ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"] {
        if !weights.contains_key(req) {
            return Err(format!(
                "SafeTensors in '{}' missing required layer '{}'",
                model_dir.display(),
                req
            ));
        }
    }

    let expected_embed_elems = cfg.vocab_size * cfg.hidden_size;
    if let Some(embed) = weights.get("model.embed_tokens.weight") {
        if embed.len() != expected_embed_elems {
            return Err(format!(
                "Shape mismatch for 'model.embed_tokens.weight': expected {} elements (vocab_size={} * hidden_size={}), got {}",
                expected_embed_elems,
                cfg.vocab_size,
                cfg.hidden_size,
                embed.len()
            ));
        }
    }
    if let Some(norm) = weights.get("model.norm.weight") {
        if norm.len() != cfg.hidden_size {
            return Err(format!(
                "Shape mismatch for 'model.norm.weight': expected {} elements (hidden_size={}), got {}",
                cfg.hidden_size,
                cfg.hidden_size,
                norm.len()
            ));
        }
    }
    if let Some(lm_head) = weights.get("lm_head.weight") {
        if lm_head.len() != expected_embed_elems {
            return Err(format!(
                "Shape mismatch for 'lm_head.weight': expected {} elements, got {}",
                expected_embed_elems,
                lm_head.len()
            ));
        }
    }

    let param_count: usize = weights.values().map(|v| v.len()).sum();
    tara_engine::TaraForCausalLM::from_weights_and_config(
        weights,
        cfg,
        &model_dir.to_string_lossy(),
    )
    .map_err(|e| {
        format!(
            "Model weights in '{}' incompatible with TaraForCausalLM architecture: {e}",
            model_dir.display()
        )
    })?;

    Ok(param_count)
}

/// Verifies continuation state (`checkpoint_state.json` + `optimizer.safetensors`) in a candidate directory.
pub fn verify_candidate_continuation(cand_dir: &Path) -> Result<u64, String> {
    let state_path = cand_dir.join("checkpoint_state.json");
    if !state_path.exists() {
        return Err(format!(
            "checkpoint_state.json missing in '{}'",
            cand_dir.display()
        ));
    }
    let state_str = fs::read_to_string(&state_path)
        .map_err(|e| format!("Failed to read checkpoint_state.json: {e}"))?;
    let val: Value = serde_json::from_str(&state_str)
        .map_err(|e| format!("Malformed checkpoint_state.json: {e}"))?;
    let step = val
        .get("optimizer_step")
        .or_else(|| val.get("step"))
        .and_then(Value::as_u64)
        .ok_or_else(|| "checkpoint_state.json missing valid 'optimizer_step'".to_string())?;

    let opt_path = cand_dir.join("optimizer.safetensors");
    if !opt_path.exists() {
        return Err(format!(
            "optimizer.safetensors missing in '{}'",
            cand_dir.display()
        ));
    }
    let (tensors, _) = load_safetensors_with_shapes(&opt_path.to_string_lossy())
        .map_err(|e| format!("Corrupt optimizer.safetensors: {e}"))?;
    if tensors.is_empty() {
        return Err("optimizer.safetensors contains zero moment tensors".to_string());
    }
    Ok(step)
}

/// Reusable end-to-end candidate verification: verifies model bundle (`config.json`, `tokenizer.json`,
/// SafeTensors weights & shape compatibility), continuation state (`checkpoint_state.json`,
/// `optimizer.safetensors`), and returns `(parameter_count, optimizer_step, weights_sha256)`.
pub fn verify_candidate(cand_dir: &Path) -> Result<(usize, u64, String), String> {
    let param_count = verify_model_bundle(cand_dir)?;
    let opt_step = verify_candidate_continuation(cand_dir)?;
    let sha256 = compute_candidate_weights_sha256(cand_dir)?;
    Ok((param_count, opt_step, sha256))
}

/// Runs the canonical evaluation suite (`EVALUATION_SUITE_ID`) against `candidate_dir`, strictly
/// validates the evaluator output schema, binds `candidate_sha256`, and persists `manual_eval_report.json`.
pub fn run_and_validate_evaluation(
    candidate_dir: &Path,
    candidate_sha256: &str,
    eval_output_dir: &Path,
) -> Result<(Value, u64, u64, f64), String> {
    let evaluator = SkillsEvaluator::new(&candidate_dir.to_string_lossy());
    let prompts: Vec<String> = CANONICAL_PROMOTION_PROMPTS
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    let mut eval_report = evaluator.evaluate(EVALUATION_SUITE_ID, &prompts);

    let pass_count = eval_report
        .get("passed")
        .and_then(Value::as_u64)
        .ok_or_else(|| "SkillsEvaluator report missing numeric 'passed' field".to_string())?;
    let total = eval_report
        .get("total_prompts")
        .and_then(Value::as_u64)
        .ok_or_else(|| "SkillsEvaluator report missing numeric 'total_prompts' field".to_string())?;
    if total == 0 || total != CANONICAL_PROMOTION_PROMPTS.len() as u64 {
        return Err(format!(
            "SkillsEvaluator report invalid 'total_prompts': expected {}, got {}",
            CANONICAL_PROMOTION_PROMPTS.len(),
            total
        ));
    }
    let pass_rate = eval_report
        .get("pass_rate")
        .and_then(Value::as_f64)
        .ok_or_else(|| "SkillsEvaluator report missing numeric 'pass_rate' field".to_string())?;
    if !pass_rate.is_finite() || !(0.0..=1.0).contains(&pass_rate) {
        return Err(format!(
            "SkillsEvaluator report invalid 'pass_rate': {}",
            pass_rate
        ));
    }

    if let Some(obj) = eval_report.as_object_mut() {
        obj.insert(
            "suite_id".to_string(),
            Value::String(EVALUATION_SUITE_ID.to_string()),
        );
        obj.insert(
            "candidate_sha256".to_string(),
            Value::String(candidate_sha256.to_string()),
        );
        obj.insert(
            "evaluated_at_epoch".to_string(),
            Value::Number(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
                    .into(),
            ),
        );
    }

    fs::create_dir_all(eval_output_dir)
        .map_err(|e| format!("Failed to create evaluation_dir '{}': {e}", eval_output_dir.display()))?;
    let eval_file = eval_output_dir.join("manual_eval_report.json");
    let json_bytes = serde_json::to_string_pretty(&eval_report).map_err(|e| e.to_string())?;
    fs::write(&eval_file, json_bytes)
        .map_err(|e| format!("Failed to write '{}': {e}", eval_file.display()))?;

    Ok((eval_report, pass_count, total, pass_rate))
}

/// Executes authentic World Model state transition: loads prior world model state (if present in
/// `world_out_dir/world_model_state.json` or `storage/persistence/world_model_state.json`),
/// extracts relational facts via `DatasetEngine` + `CurriculumEngine`, merges/evolves prior facts
/// with newly extracted facts (resolving conflicts by higher confidence), verifies graph consistency,
/// and atomically publishes the updated `world_model_state.json`.
pub fn execute_world_model_update(
    dataset_dir: &Path,
    curriculum_path: &Path,
    world_out_dir: &Path,
) -> Result<Value, String> {
    let _ = recover_interrupted_promotion(world_out_dir).map_err(|e| e.to_string())?;

    // 1. Load prior world model state if available for stateful transition
    let prior_candidate_state = world_out_dir.join("world_model_state.json");
    let prior_prod_state = Path::new("storage/persistence/world_model_state.json");
    let prior_path = if prior_candidate_state.exists() {
        Some(prior_candidate_state)
    } else if prior_prod_state.exists() {
        Some(prior_prod_state.to_path_buf())
    } else {
        None
    };

    let mut previous_state_sha256: Option<String> = None;
    let mut merged_facts: std::collections::BTreeMap<(String, String), Value> =
        std::collections::BTreeMap::new();
    let mut prior_facts_count = 0usize;

    if let Some(ref p_path) = prior_path {
        let raw = fs::read_to_string(p_path)
            .map_err(|e| format!("Failed to read prior world model state '{}': {e}", p_path.display()))?;
        let prior_val: Value = serde_json::from_str(&raw)
            .map_err(|e| format!("Malformed prior world model state '{}': {e}", p_path.display()))?;
        previous_state_sha256 = prior_val
            .get("state_graph_sha256")
            .and_then(Value::as_str)
            .map(|s| s.to_string());
        if let Some(arr) = prior_val.get("verified_facts").and_then(Value::as_array) {
            for fact in arr {
                let subj = fact.get("subject").and_then(Value::as_str).unwrap_or("").trim();
                let pred = fact.get("predicate").and_then(Value::as_str).unwrap_or("").trim();
                let obj = fact.get("object").and_then(Value::as_str).unwrap_or("").trim();
                if !subj.is_empty() && !pred.is_empty() && !obj.is_empty() {
                    merged_facts.insert((subj.to_string(), pred.to_string()), fact.clone());
                    prior_facts_count += 1;
                }
            }
        }
    }

    let mut engine_cfg = DatasetEngineConfig::default();
    if dataset_dir.exists() && dataset_dir.is_dir() {
        engine_cfg.manual_dir = dataset_dir.to_path_buf();
    }
    let dataset_engine =
        DatasetEngine::new(engine_cfg).map_err(|e| format!("DatasetEngine init error: {e}"))?;

    let mut ingested_files = 0usize;
    if curriculum_path.exists() && curriculum_path.is_file() {
        dataset_engine
            .process_dataset_file(curriculum_path, DatasetSource::Manual)
            .map_err(|e| format!("Failed to ingest curriculum file '{}': {e}", curriculum_path.display()))?;
        ingested_files += 1;
    }
    if dataset_dir.exists() {
        if dataset_dir.is_file() {
            dataset_engine
                .process_dataset_file(dataset_dir, DatasetSource::Manual)
                .map_err(|e| format!("Failed to ingest dataset file '{}': {e}", dataset_dir.display()))?;
            ingested_files += 1;
        } else if dataset_dir.is_dir() {
            let entries = fs::read_dir(dataset_dir)
                .map_err(|e| format!("Failed to read dataset directory '{}': {e}", dataset_dir.display()))?;
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_file() && p.extension().and_then(|ext| ext.to_str()) == Some("jsonl") {
                    dataset_engine
                        .process_dataset_file(&p, DatasetSource::Manual)
                        .map_err(|e| format!("Failed to ingest shard '{}': {e}", p.display()))?;
                    ingested_files += 1;
                }
            }
        }
    }

    if ingested_files == 0 {
        return Err(format!(
            "World Model training failed: neither curriculum_path ('{}') nor dataset_dir ('{}') contained any valid JSONL files",
            curriculum_path.display(),
            dataset_dir.display()
        ));
    }

    let curriculum_engine = CurriculumEngine::new(CurriculumConfig::default());
    let report = curriculum_engine.receive_and_update(&dataset_engine);
    let (_, world_curriculum) = curriculum_engine.get_curriculum_snapshots();

    if world_curriculum.ordered_records.is_empty() {
        return Err(
            "World Model training rejected: zero relational/world-model records extracted from dataset"
                .to_string(),
        );
    }

    let mut added_facts_count = 0usize;
    let mut updated_facts_count = 0usize;
    let mut conflicts_resolved_count = 0usize;

    for rec in &world_curriculum.ordered_records {
        if rec.subject.trim().is_empty()
            || rec.predicate.trim().is_empty()
            || rec.object.trim().is_empty()
        {
            return Err(format!(
                "World Model consistency check FAILED on record '{}': empty subject/predicate/object",
                rec.id
            ));
        }
        if !rec.confidence.is_finite() || !(0.0..=1.0).contains(&rec.confidence) {
            return Err(format!(
                "World Model consistency check FAILED on record '{}': invalid confidence {}",
                rec.id, rec.confidence
            ));
        }

        let key = (rec.subject.trim().to_string(), rec.predicate.trim().to_string());
        let rec_val = serde_json::to_value(rec).map_err(|e| e.to_string())?;

        if let Some(existing) = merged_facts.get(&key) {
            let old_obj = existing.get("object").and_then(Value::as_str).unwrap_or("");
            let old_conf = existing
                .get("confidence")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            if old_obj != rec.object.trim() {
                conflicts_resolved_count += 1;
            }
            if (rec.confidence as f64) >= old_conf {
                merged_facts.insert(key, rec_val);
                updated_facts_count += 1;
            }
        } else {
            merged_facts.insert(key, rec_val);
            added_facts_count += 1;
        }
    }

    let mut distinct_entities = HashSet::new();
    let mut distinct_predicates = HashSet::new();
    let mut hasher = Sha256::new();
    let mut final_facts = Vec::with_capacity(merged_facts.len());
    let mut conf_sum = 0.0f64;

    for fact in merged_facts.values() {
        let id = fact.get("id").and_then(Value::as_str).unwrap_or("unknown");
        let subj = fact.get("subject").and_then(Value::as_str).unwrap_or("");
        let pred = fact.get("predicate").and_then(Value::as_str).unwrap_or("");
        let obj = fact.get("object").and_then(Value::as_str).unwrap_or("");
        let conf = fact.get("confidence").and_then(Value::as_f64).unwrap_or(0.0);

        distinct_entities.insert(subj.to_string());
        distinct_entities.insert(obj.to_string());
        distinct_predicates.insert(pred.to_string());
        conf_sum += conf;

        hasher.update(id.as_bytes());
        hasher.update(b"|");
        hasher.update(subj.as_bytes());
        hasher.update(b"|");
        hasher.update(pred.as_bytes());
        hasher.update(b"|");
        hasher.update(obj.as_bytes());
        hasher.update(&conf.to_bits().to_le_bytes());

        final_facts.push(fact.clone());
    }

    let state_digest = hex::encode(hasher.finalize());
    let now_stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let avg_conf = if final_facts.is_empty() {
        0.0
    } else {
        conf_sum / final_facts.len() as f64
    };

    let snapshot_data = serde_json::json!({
        "schema_version": "2.0.0",
        "snapshot_timestamp_epoch": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
        "status": "candidate_verified",
        "previous_state_sha256": previous_state_sha256,
        "prior_facts_count": prior_facts_count,
        "added_facts_count": added_facts_count,
        "updated_facts_count": updated_facts_count,
        "conflicts_resolved_count": conflicts_resolved_count,
        "entity_count": distinct_entities.len(),
        "predicate_count": distinct_predicates.len(),
        "fact_count": final_facts.len(),
        "curriculum_stages_count": report.world_stages_count,
        "average_confidence": avg_conf,
        "state_graph_sha256": state_digest,
        "consistency_check": "PASS",
        "stages": world_curriculum.stages,
        "verified_facts": final_facts,
    });

    if let Some(parent) = world_out_dir.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }

    let staging_dir = PathBuf::from(format!(
        "{}.staging_{}_{}",
        world_out_dir.display(),
        now_stamp,
        std::process::id()
    ));
    fs::create_dir_all(&staging_dir).map_err(|e| e.to_string())?;
    fs::write(
        staging_dir.join("world_model_state.json"),
        serde_json::to_string_pretty(&snapshot_data).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    promote_directory_atomically(&staging_dir, world_out_dir).map_err(|e| e.to_string())?;
    Ok(snapshot_data)
}

/// Atomically stages and promotes a verified manual candidate bundle (`model.safetensors`,
/// `config.json`, `tokenizer.json`, `checkpoint_state.json`, `optimizer.safetensors`,
/// `manual_eval_report.json`, and `PROMOTION_MANIFEST.json`) into `prod_dir` using two-phase backup and rollback.
pub fn promote_candidate_bundle_atomically(
    candidate_dir: &Path,
    prod_dir: &Path,
    dynamic_sha: &str,
    pass_rate: f64,
    eval_report_opt: Option<&Value>,
) -> Result<(), String> {
    let _ = recover_interrupted_promotion(prod_dir).map_err(|e| e.to_string())?;
    if let Some(parent) = prod_dir.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }

    let now_stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let staging_dir = PathBuf::from(format!(
        "{}.staging_{}_{}",
        prod_dir.display(),
        now_stamp,
        std::process::id()
    ));
    fs::create_dir_all(&staging_dir).map_err(|e| e.to_string())?;

    // Copy all model shards, config, tokenizer, and continuation artifacts
    for entry in fs::read_dir(candidate_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let p = entry.path();
        if p.is_file() {
            if let Some(name) = p.file_name() {
                fs::copy(&p, staging_dir.join(name)).map_err(|e| e.to_string())?;
            }
        }
    }

    if let Some(eval_rep) = eval_report_opt {
        fs::write(
            staging_dir.join("manual_eval_report.json"),
            serde_json::to_string_pretty(eval_rep).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    }

    let manifest = serde_json::json!({
        "promoted_from": candidate_dir.to_string_lossy(),
        "promoted_to": prod_dir.to_string_lossy(),
        "model_weights_sha256": dynamic_sha,
        "evaluation_suite_id": EVALUATION_SUITE_ID,
        "evaluation_pass_rate": pass_rate,
        "promoted_at_epoch": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
    });
    fs::write(
        staging_dir.join("PROMOTION_MANIFEST.json"),
        serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    // Verify staged bundle before swapping into production
    verify_candidate(&staging_dir)?;

    promote_directory_atomically(&staging_dir, prod_dir).map_err(|e| e.to_string())?;

    // Post-promotion verification of target directory
    verify_candidate(prod_dir)?;
    Ok(())
}

fn print_usage() {
    println!("================================================================================");
    println!("  TARA MANUAL TRAINING WORKSPACE RUNNER (100% NATIVE RUST)");
    println!("================================================================================");
    println!("Usage: cargo run --bin manual_trainer -- [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --config <path>         Path to JSON configuration (default: manual_training/config/default_config.json)");
    println!("  --mode <mode>           Training mode: 'neural', 'world_model', or 'both' (default from config)");
    println!("  --steps <n>             Override maximum training steps (must be >= 1)");
    println!("  --batch-size <n>        Override batch size (must be >= 1)");
    println!("  --lr <rate>             Override learning rate (must be > 0.0)");
    println!("  --dataset <path>        Override canonical dataset directory");
    println!("  --resume                Resume training from existing checkpoint state if present");
    println!("  --resume-from <path>    Explicitly resume training from a specific checkpoint directory");
    println!("  --fresh                 Force fresh training run (mutually exclusive with --resume/--resume-from)");
    println!("  --evaluate              Run canonical benchmark evaluation suite against manual candidate");
    println!("  --promote               Promote verified manual candidate to production (requires gate pass)");
    println!("  -h, --help              Print this help information");
    println!("================================================================================");
}

pub fn append_log(log_path: &Path, message: &str) -> Result<(), std::io::Error> {
    if let Some(parent) = log_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    writeln!(file, "[{timestamp}] {message}")
}

pub fn generate_collision_free_run_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let pid = std::process::id();
    format!("run_{nanos}_{pid}")
}

fn execute_manual_trainer() -> Result<RunStatus, (RunStatus, String)> {
    let args: Vec<String> = env::args().collect();

    let candidates = [
        "rust/tara_training_system/manual_training/config.json",
        "tara_training_system/manual_training/config.json",
        "manual_training/config/default_config.json",
        "../manual_training/config/default_config.json",
    ];
    let mut default_config_path =
        PathBuf::from("rust/tara_training_system/manual_training/config.json");
    for cand in &candidates {
        if Path::new(cand).exists() {
            default_config_path = PathBuf::from(cand);
            break;
        }
    }

    let cli = parse_manual_cli(&args, default_config_path)
        .map_err(|e| (RunStatus::ValidationError, e))?;
    if cli.show_help {
        print_usage();
        return Ok(RunStatus::TrainingCompleted);
    }

    // 1. Load and strictly validate configuration (zero silent fallback on malformed JSON)
    let mut config = load_manual_config(&cli.config_path, cli.explicit_config)
        .map_err(|e| (RunStatus::ValidationError, e))?;

    if let Some(ref m) = cli.mode_override {
        config.training_mode = m.clone();
    }
    if let Some(s) = cli.steps_override {
        config.max_steps = s;
        if config.checkpoint_interval_steps > config.max_steps {
            config.checkpoint_interval_steps = config.max_steps;
        }
    }
    if let Some(b) = cli.batch_size_override {
        config.batch_size = b;
    }
    if let Some(lr) = cli.lr_override {
        config.learning_rate = lr;
    }
    if let Some(ref ds) = cli.dataset_override {
        config.dataset_dir = ds.clone();
    }

    validate_manual_config(&config).map_err(|e| (RunStatus::ValidationError, e))?;

    println!("================================================================================");
    println!("  TARA MANUAL TRAINING WORKSPACE INITIALIZATION");
    println!("================================================================================");
    println!("Config Version    : {}", config.version);
    println!("Mode              : {}", config.training_mode);
    println!("Model Source      : {}", config.model_source_dir);
    println!("Dataset Path      : {}", config.dataset_dir);
    println!("Batch Size        : {}", config.batch_size);
    println!("Learning Rate     : {}", config.learning_rate);
    println!("Max Steps         : {}", config.max_steps);
    println!("Warmup Steps      : {}", config.warmup_steps);
    println!("Weight Decay      : {}", config.weight_decay);
    println!("Grad Clip Norm    : {}", config.grad_clip_norm);
    println!("Neural Target Dir : {}", config.neural_checkpoint_dir);
    println!("World Target Dir  : {}", config.world_model_checkpoint_dir);
    println!("--------------------------------------------------------------------------------");

    // 2. Promotion flow (Fail-closed gate + Unified evaluation suite + Atomic two-phase promotion)
    if cli.run_promote {
        println!("=== [PROMOTION GATE: MANUAL CANDIDATE -> PRODUCTION] ===");
        let candidate_neural = PathBuf::from(&config.neural_checkpoint_dir);
        let _ = recover_interrupted_promotion(&candidate_neural)
            .map_err(|e| (RunStatus::ExecutionFailed, e.to_string()))?;

        let (param_count, opt_step, dynamic_sha) = verify_candidate(&candidate_neural).map_err(|e| {
            (
                RunStatus::PromotionRejected,
                format!("PROMOTION ABORTED: Candidate verification failed: {e}"),
            )
        })?;
        println!("  Candidate Parameters             : {}", param_count);
        println!("  Candidate Neural Weights SHA-256 : {}", dynamic_sha);
        println!("  Candidate Optimizer Step         : {}", opt_step);

        let eval_output_dir = PathBuf::from(&config.evaluation_dir);
        let (eval_report, pass_count, total, pass_rate) =
            run_and_validate_evaluation(&candidate_neural, &dynamic_sha, &eval_output_dir)
                .map_err(|e| (RunStatus::ExecutionFailed, e))?;
        println!(
            "  Evaluation Suite ({}) : {}/{} passed ({:.2}%)",
            EVALUATION_SUITE_ID,
            pass_count,
            total,
            pass_rate * 100.0
        );

        if pass_rate < PROMOTION_PASS_RATE_THRESHOLD {
            return Err((
                RunStatus::PromotionRejected,
                format!(
                    "PROMOTION REJECTED: Candidate pass rate {:.2}% is below required {:.2}% threshold",
                    pass_rate * 100.0,
                    PROMOTION_PASS_RATE_THRESHOLD * 100.0
                ),
            ));
        }

        println!("  Evaluation PASSED: Candidate meets promotion threshold.");
        let prod_dirs = [
            PathBuf::from("storage/models/tara"),
            PathBuf::from("production/neural"),
        ];
        for prod_dir in &prod_dirs {
            promote_candidate_bundle_atomically(
                &candidate_neural,
                prod_dir,
                &dynamic_sha,
                pass_rate,
                Some(&eval_report),
            )
            .map_err(|e| (RunStatus::ExecutionFailed, e))?;
            println!("  Atomically promoted candidate bundle to {:?}", prod_dir);
        }

        // Promote world model atomically if candidate exists
        let cand_world_dir = PathBuf::from(&config.world_model_checkpoint_dir);
        let cand_world_file = cand_world_dir.join("world_model_state.json");
        if cand_world_file.exists() {
            let wm_target = PathBuf::from("production/world_model");
            let now_stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let wm_staging = PathBuf::from(format!(
                "{}.staging_{}_{}",
                wm_target.display(),
                now_stamp,
                std::process::id()
            ));
            fs::create_dir_all(&wm_staging)
                .map_err(|e| (RunStatus::ExecutionFailed, e.to_string()))?;
            fs::copy(&cand_world_file, wm_staging.join("world_model_state.json"))
                .map_err(|e| (RunStatus::ExecutionFailed, e.to_string()))?;
            promote_directory_atomically(&wm_staging, &wm_target)
                .map_err(|e| (RunStatus::ExecutionFailed, e.to_string()))?;
            println!("  Atomically promoted candidate world model to {:?}", wm_target);
        }

        println!("SUCCESS: Promoted manual candidate to production.");
        let neural_log = PathBuf::from(&config.logs_dir).join("neural_loss.log");
        append_log(
            &neural_log,
            &format!(
                "PROMOTION_EXECUTED: status=Promoted, Candidate SHA-256={dynamic_sha}, pass_rate={:.2}% -> storage/models/tara & production/neural",
                pass_rate * 100.0
            ),
        )
        .map_err(|e| (RunStatus::ExecutionFailed, format!("Failed to write log: {e}")))?;
        return Ok(RunStatus::Promoted);
    }

    // 3. Evaluation-only flow (Unified suite + strict schema check + candidate verification)
    if cli.run_evaluation {
        println!("=== [EVALUATION: MANUAL CANDIDATE BENCHMARK SUITE] ===");
        let candidate_neural = PathBuf::from(&config.neural_checkpoint_dir);
        let _ = recover_interrupted_promotion(&candidate_neural)
            .map_err(|e| (RunStatus::ExecutionFailed, e.to_string()))?;
        let (_, _, dynamic_sha) = verify_candidate(&candidate_neural)
            .map_err(|e| (RunStatus::ExecutionFailed, format!("EVALUATION ABORTED: {e}")))?;

        let eval_output_dir = PathBuf::from(&config.evaluation_dir);
        let (_, pass_count, total, pass_rate) =
            run_and_validate_evaluation(&candidate_neural, &dynamic_sha, &eval_output_dir)
                .map_err(|e| (RunStatus::ExecutionFailed, e))?;

        println!("Evaluation Result ({EVALUATION_SUITE_ID}):");
        println!("  Candidate SHA256 : {}", dynamic_sha);
        println!("  Total Benchmarks : {}", total);
        println!("  Passed Tests     : {}", pass_count);
        println!("  Pass Rate        : {:.2}%", pass_rate * 100.0);
        println!(
            "Wrote evaluation report to {:?}",
            eval_output_dir.join("manual_eval_report.json")
        );

        if pass_rate < PROMOTION_PASS_RATE_THRESHOLD {
            return Err((
                RunStatus::ExecutionFailed,
                format!(
                    "EVALUATION FAILED: Pass rate {:.2}% is below {:.2}% threshold",
                    pass_rate * 100.0,
                    PROMOTION_PASS_RATE_THRESHOLD * 100.0
                ),
            ));
        }
        return Ok(RunStatus::EvaluationPassed);
    }

    let run_id = generate_collision_free_run_id();
    let run_dir = PathBuf::from(&config.runs_dir).join(&run_id);
    fs::create_dir_all(&run_dir)
        .map_err(|e| (RunStatus::ExecutionFailed, format!("Failed to create run directory: {e}")))?;

    // 4. Execution: Neural Model Training (Strict model_source_dir verification & separated intermediate checkpoints)
    if config.training_mode == "neural" || config.training_mode == "both" {
        println!("\n=== [EXECUTING NEURAL MODEL TRAINING (run_id={run_id})] ===");
        let src_dir = PathBuf::from(&config.model_source_dir);
        let _ = recover_interrupted_promotion(&src_dir)
            .map_err(|e| (RunStatus::ExecutionFailed, e.to_string()))?;
        verify_model_bundle(&src_dir).map_err(|e| {
            (
                RunStatus::ValidationError,
                format!(
                    "Configured model_source_dir '{}' is missing or invalid (silent fallback prohibited): {}",
                    config.model_source_dir, e
                ),
            )
        })?;

        let neural_out = PathBuf::from(&config.neural_checkpoint_dir);
        let _ = recover_interrupted_promotion(&neural_out)
            .map_err(|e| (RunStatus::ExecutionFailed, e.to_string()))?;
        let intermediate_cp_dir = run_dir.join("checkpoints");
        fs::create_dir_all(&intermediate_cp_dir)
            .map_err(|e| (RunStatus::ExecutionFailed, e.to_string()))?;

        let (should_resume, resolved_resume_from) = if cli.fresh {
            (false, None)
        } else if let Some(ref rf) = cli.resume_from {
            (true, Some(rf.clone()))
        } else if cli.resume {
            (true, Some(config.neural_checkpoint_dir.clone()))
        } else {
            (false, None)
        };

        let options = TrainingOptions {
            dataset_dir: Some(config.dataset_dir.clone()),
            curriculum_path: if Path::new(&config.curriculum_path).exists() {
                Some(config.curriculum_path.clone())
            } else {
                None
            },
            candidate_dir: Some(config.neural_checkpoint_dir.clone()),
            learning_rate: Some(config.learning_rate),
            batch_size: Some(config.batch_size),
            max_steps: Some(config.max_steps),
            warmup_steps: Some(config.warmup_steps),
            weight_decay: Some(config.weight_decay),
            grad_clip_norm: Some(config.grad_clip_norm),
            device: Some("auto".to_string()),
            precision: Some("auto".to_string()),
            checkpoint_dir: Some(intermediate_cp_dir.to_string_lossy().to_string()),
            checkpoint_interval: Some(config.checkpoint_interval_steps),
            resume: should_resume,
            resume_from: resolved_resume_from,
            max_memory_mb: Some(65536.0),
            ..Default::default()
        };

        println!("Calling shared training infrastructure (NativeSelfTrainer)...");
        let report = run_controlled_training_with_options(
            &config.model_source_dir,
            ".",
            1,
            options,
        )
        .map_err(|e| (RunStatus::ExecutionFailed, e.to_string()))?;

        // Post-training candidate verification before claiming completion
        let (param_count, opt_step, cand_sha) = verify_candidate(&neural_out).map_err(|e| {
            (
                RunStatus::ExecutionFailed,
                format!("Post-training candidate verification failed in '{}': {e}", neural_out.display()),
            )
        })?;

        println!(
            "  Neural Training & Candidate Verification Completed (params={}, step={}, sha256={})!",
            param_count, opt_step, cand_sha
        );

        let run_manifest = serde_json::json!({
            "run_id": run_id,
            "status": RunStatus::TrainingCompleted,
            "training_mode": config.training_mode,
            "candidate_dir": config.neural_checkpoint_dir,
            "intermediate_checkpoints_dir": intermediate_cp_dir.to_string_lossy(),
            "candidate_sha256": cand_sha,
            "optimizer_step": opt_step,
            "parameter_count": param_count,
            "config": config,
            "telemetry": report,
        });
        let manifest_path = run_dir.join("manifest.json");
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&run_manifest)
                .map_err(|e| (RunStatus::ExecutionFailed, e.to_string()))?,
        )
        .map_err(|e| (RunStatus::ExecutionFailed, e.to_string()))?;
        println!("  Saved structured run manifest to {:?}", manifest_path);

        let neural_log = PathBuf::from(&config.logs_dir).join("neural_loss.log");
        let loss = report
            .get("loss_after")
            .or_else(|| report.get("final_loss"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        append_log(
            &neural_log,
            &format!(
                "RUN_COMPLETED: id={run_id}, steps={opt_step}, sha256={cand_sha}, final_loss={loss:.4}"
            ),
        )
        .map_err(|e| (RunStatus::ExecutionFailed, format!("Failed to write neural log: {e}")))?;
    }

    // 5. Execution: Genuine World Model State Update & Consistency Verification
    if config.training_mode == "world_model" || config.training_mode == "both" {
        println!("\n=== [EXECUTING WORLD MODEL UPDATE & SNAPSHOT (run_id={run_id})] ===");
        let world_out = PathBuf::from(&config.world_model_checkpoint_dir);
        let snapshot = execute_world_model_update(
            Path::new(&config.dataset_dir),
            Path::new(&config.curriculum_path),
            &world_out,
        )
        .map_err(|e| (RunStatus::ExecutionFailed, e))?;

        let entities = snapshot
            .get("entity_count")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let facts = snapshot
            .get("fact_count")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let sha = snapshot
            .get("state_graph_sha256")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        println!(
            "  World Model Updated: {} entities, {} verified facts (SHA-256: {})",
            entities, facts, sha
        );

        if config.training_mode == "world_model" {
            let wm_manifest = serde_json::json!({
                "run_id": run_id,
                "status": RunStatus::TrainingCompleted,
                "training_mode": config.training_mode,
                "world_model_checkpoint_dir": config.world_model_checkpoint_dir,
                "state_graph_sha256": sha,
                "entity_count": entities,
                "fact_count": facts,
                "config": config,
            });
            let manifest_path = run_dir.join("manifest.json");
            fs::write(
                &manifest_path,
                serde_json::to_string_pretty(&wm_manifest)
                    .map_err(|e| (RunStatus::ExecutionFailed, e.to_string()))?,
            )
            .map_err(|e| (RunStatus::ExecutionFailed, e.to_string()))?;
        }

        let world_log = PathBuf::from(&config.logs_dir).join("world_model_consistency.log");
        append_log(
            &world_log,
            &format!(
                "WORLD_MODEL_VERIFIED: id={run_id}, destination={:?}, entities={}, facts={}, sha256={}",
                world_out, entities, facts, sha
            ),
        )
        .map_err(|e| (RunStatus::ExecutionFailed, format!("Failed to write world model log: {e}")))?;
    }

    println!("================================================================================");
    println!("  MANUAL TRAINING WORKSPACE RUN FINISHED (status=TrainingCompleted).");
    println!("  Checkpoints staged in isolated manual directory. Production unaffected.");
    println!("================================================================================");

    Ok(RunStatus::TrainingCompleted)
}

fn main() {
    match execute_manual_trainer() {
        Ok(status) => {
            std::process::exit(status.exit_code());
        }
        Err((status, msg)) => {
            eprintln!("ERROR [{:?}]: {}", status, msg);
            std::process::exit(status.exit_code());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tara_engine::write_safetensors_with_shapes;

    fn unique_test_dir(tag: &str) -> PathBuf {
        let dir = env::temp_dir().join(format!(
            "tara_manual_test_{}_{}_{}",
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

    fn write_working_candidate(dir: &Path) {
        fs::create_dir_all(dir).unwrap();
        let cfg = TaraConfig {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 4,
            max_position_embeddings: 32,
            ..TaraConfig::default()
        };
        fs::write(
            dir.join("config.json"),
            serde_json::to_string_pretty(&cfg).unwrap(),
        )
        .unwrap();
        fs::write(
            dir.join("tokenizer.json"),
            r#"{"vocab":{"<|pad|>":0,"<|im_start|>":1,"<|im_end|>":2,"<|unk|>":3,"a":4,"b":5}}"#,
        )
        .unwrap();

        let mut weights = HashMap::new();
        let mut shapes = HashMap::new();
        weights.insert("model.embed_tokens.weight".to_string(), vec![0.01f32; 16 * 8]);
        shapes.insert("model.embed_tokens.weight".to_string(), vec![16, 8]);
        weights.insert("model.norm.weight".to_string(), vec![1.0f32; 8]);
        shapes.insert("model.norm.weight".to_string(), vec![8]);
        weights.insert("lm_head.weight".to_string(), vec![0.01f32; 16 * 8]);
        shapes.insert("lm_head.weight".to_string(), vec![16, 8]);

        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            let k = format!("model.layers.0.self_attn.{proj}.weight");
            weights.insert(k.clone(), vec![0.01f32; 8 * 8]);
            shapes.insert(k, vec![8, 8]);
        }
        for proj in ["gate_proj", "up_proj"] {
            let k = format!("model.layers.0.mlp.{proj}.weight");
            weights.insert(k.clone(), vec![0.01f32; 16 * 8]);
            shapes.insert(k, vec![16, 8]);
        }
        weights.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            vec![0.01f32; 8 * 16],
        );
        shapes.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            vec![8, 16],
        );
        for ln in ["input_layernorm", "post_attention_layernorm"] {
            let k = format!("model.layers.0.{ln}.weight");
            weights.insert(k.clone(), vec![1.0f32; 8]);
            shapes.insert(k, vec![8]);
        }

        write_safetensors_with_shapes(
            &weights,
            &shapes,
            &dir.join("model.safetensors").to_string_lossy(),
        )
        .unwrap();

        fs::write(
            dir.join("checkpoint_state.json"),
            r#"{"step":5,"optimizer_step":5,"samples_seen":20,"epoch":1,"resumable":true}"#,
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

    #[test]
    fn test_cli_and_malformed_config_reject_invalid_inputs() {
        let dir = unique_test_dir("cli_and_config");
        let bad_cfg = dir.join("corrupt.json");
        fs::write(&bad_cfg, "{ invalid json syntax ").unwrap();

        // Malformed config JSON must fail hard, never fall back to default
        assert!(load_manual_config(&bad_cfg, true).is_err());

        // Missing explicit config must fail hard
        assert!(load_manual_config(&dir.join("nonexistent.json"), true).is_err());

        // Invalid CLI arguments must fail hard
        let args_bad_steps = vec!["manual_trainer".to_string(), "--steps".to_string(), "abc".to_string()];
        assert!(parse_manual_cli(&args_bad_steps, bad_cfg.clone()).is_err());

        let args_unknown = vec!["manual_trainer".to_string(), "--unknown-flag".to_string()];
        assert!(parse_manual_cli(&args_unknown, bad_cfg.clone()).is_err());

        let args_missing_val = vec!["manual_trainer".to_string(), "--lr".to_string()];
        assert!(parse_manual_cli(&args_missing_val, bad_cfg.clone()).is_err());

        // Mutually exclusive --evaluate and --promote must fail
        let args_conflict_eval_prom = vec![
            "manual_trainer".to_string(),
            "--evaluate".to_string(),
            "--promote".to_string(),
        ];
        assert!(parse_manual_cli(&args_conflict_eval_prom, bad_cfg.clone()).is_err());

        // Mutually exclusive --fresh and --resume must fail
        let args_conflict_fresh_res = vec![
            "manual_trainer".to_string(),
            "--fresh".to_string(),
            "--resume".to_string(),
        ];
        assert!(parse_manual_cli(&args_conflict_fresh_res, bad_cfg).is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_path_safety_rejects_production_traversal_and_overlaps() {
        let mut cfg = ManualTrainingConfig::default();
        assert!(validate_manual_config(&cfg).is_ok());

        // Pointing neural_checkpoint_dir to production must fail
        cfg.neural_checkpoint_dir = "storage/models/tara/sub".to_string();
        assert!(validate_manual_config(&cfg).is_err());
        cfg = ManualTrainingConfig::default();

        // Pointing neural_checkpoint_dir to model_source_dir must fail
        cfg.model_source_dir = "custom_models/base".to_string();
        cfg.neural_checkpoint_dir = "custom_models/base".to_string();
        assert!(validate_manual_config(&cfg).is_err());
        cfg = ManualTrainingConfig::default();

        // Parent traversal '..' in logs_dir or evaluation_dir must fail
        cfg.logs_dir = "manual_training/../storage/persistence".to_string();
        assert!(validate_manual_config(&cfg).is_err());
        cfg = ManualTrainingConfig::default();

        // checkpoint_interval_steps > max_steps must fail
        cfg.max_steps = 10;
        cfg.checkpoint_interval_steps = 20;
        assert!(validate_manual_config(&cfg).is_err());
        cfg = ManualTrainingConfig::default();

        // Unsupported version must fail
        cfg.version = "9.9.9".to_string();
        assert!(validate_manual_config(&cfg).is_err());
    }

    #[test]
    fn test_world_model_stateful_transition_and_atomic_promotion() {
        let dir = unique_test_dir("world_model_and_promote");
        let ds_file1 = dir.join("world_facts_1.jsonl");
        fs::write(
            &ds_file1,
            concat!(
                r#"{"id":"w1","subject":"TaraEngine","predicate":"executes_on","object":"NativeRust","context":"Core architecture","confidence":0.90}"#,
                "\n"
            ),
        )
        .unwrap();

        let world_out = dir.join("world_chk");
        let snap1 =
            execute_world_model_update(&dir, &ds_file1, &world_out).expect("first world model update");
        let sha1 = snap1
            .get("state_graph_sha256")
            .and_then(Value::as_str)
            .unwrap()
            .to_string();
        assert_eq!(snap1.get("fact_count").and_then(Value::as_u64), Some(1));

        // Second update with updated confidence and new fact must transition from prior state
        fs::remove_file(&ds_file1).unwrap();
        let ds_file2 = dir.join("world_facts_2.jsonl");
        fs::write(
            &ds_file2,
            concat!(
                r#"{"id":"w1_v2","subject":"TaraEngine","predicate":"executes_on","object":"NativeRust","context":"Core architecture","confidence":0.98}"#,
                "\n",
                r#"{"id":"w2","subject":"AdamW","predicate":"updates","object":"ModelWeights","context":"Optimization","confidence":0.92}"#,
                "\n"
            ),
        )
        .unwrap();

        let snap2 =
            execute_world_model_update(&dir, &ds_file2, &world_out).expect("second world model update");
        assert_eq!(
            snap2.get("previous_state_sha256").and_then(Value::as_str),
            Some(sha1.as_str())
        );
        assert_eq!(snap2.get("prior_facts_count").and_then(Value::as_u64), Some(1));
        assert_eq!(snap2.get("added_facts_count").and_then(Value::as_u64), Some(1));
        assert_eq!(snap2.get("updated_facts_count").and_then(Value::as_u64), Some(1));
        assert_eq!(snap2.get("fact_count").and_then(Value::as_u64), Some(2));

        // Test atomic neural candidate bundle promotion with evaluation report
        let cand_dir = dir.join("cand_neural");
        let prod_dir = dir.join("prod_neural");
        let eval_dir = dir.join("eval_out");
        write_working_candidate(&cand_dir);

        let (_, _, sha) = verify_candidate(&cand_dir).expect("candidate must verify");
        let (eval_rep, _, total, _) =
            run_and_validate_evaluation(&cand_dir, &sha, &eval_dir).expect("eval must run");
        assert_eq!(total, CANONICAL_PROMOTION_PROMPTS.len() as u64);

        promote_candidate_bundle_atomically(&cand_dir, &prod_dir, &sha, 0.85, Some(&eval_rep))
            .unwrap();

        assert!(verify_candidate(&prod_dir).is_ok());
        assert!(prod_dir.join("PROMOTION_MANIFEST.json").exists());
        assert!(prod_dir.join("manual_eval_report.json").exists());

        let _ = fs::remove_dir_all(&dir);
    }
}
