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
use tara_engine::safetensors::{
    compute_sha256, load_model_weights_with_shapes, load_safetensors_with_shapes,
};
use tara_engine::skills_evaluator::SkillsEvaluator;
use tara_engine::tokenizer::TaraTokenizer;
use tara_engine::train_candidate::{run_controlled_training_with_options, TrainingOptions};
use tara_engine::trainer::{promote_directory_atomically, recover_interrupted_promotion};

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

pub fn validate_manual_config(config: &ManualTrainingConfig) -> Result<(), String> {
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

    let neural_chk = Path::new(&config.neural_checkpoint_dir);
    let world_chk = Path::new(&config.world_model_checkpoint_dir);
    let model_src = Path::new(&config.model_source_dir);

    let protected_dirs = [
        Path::new("storage/models/tara"),
        Path::new("production/neural"),
        Path::new("production/world_model"),
        Path::new("storage/persistence"),
    ];

    for prot in &protected_dirs {
        if paths_overlap(neural_chk, prot) {
            return Err(format!(
                "neural_checkpoint_dir ('{}') overlaps with protected directory '{}'",
                config.neural_checkpoint_dir,
                prot.display()
            ));
        }
        if paths_overlap(world_chk, prot) {
            return Err(format!(
                "world_model_checkpoint_dir ('{}') overlaps with protected directory '{}'",
                config.world_model_checkpoint_dir,
                prot.display()
            ));
        }
    }

    if paths_overlap(neural_chk, model_src) {
        return Err(format!(
            "neural_checkpoint_dir ('{}') overlaps with model_source_dir ('{}'). In-place overwrite of source model is prohibited.",
            config.neural_checkpoint_dir, config.model_source_dir
        ));
    }
    if paths_overlap(neural_chk, world_chk) {
        return Err(format!(
            "neural_checkpoint_dir ('{}') overlaps with world_model_checkpoint_dir ('{}')",
            config.neural_checkpoint_dir, config.world_model_checkpoint_dir
        ));
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
    pub run_evaluation: bool,
    pub run_promote: bool,
    pub show_help: bool,
}

pub fn parse_manual_cli(args: &[String], default_config_path: PathBuf) -> Result<ParsedManualCli, String> {
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

    Ok(out)
}

/// Verifies that a model directory contains valid `config.json`, `tokenizer.json`, and SafeTensors weights.
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
    TaraConfig::from_json_file(&cfg_path.to_string_lossy())
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
    Ok(weights.values().map(|v| v.len()).sum())
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

/// Executes authentic World Model state transition extraction, topological curriculum ordering,
/// consistency verification, and atomic snapshot staging.
pub fn execute_world_model_update(
    dataset_dir: &Path,
    curriculum_path: &Path,
    world_out_dir: &Path,
) -> Result<Value, String> {
    let _ = recover_interrupted_promotion(world_out_dir).map_err(|e| e.to_string())?;

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

    let mut distinct_entities = HashSet::new();
    let mut distinct_predicates = HashSet::new();
    let mut hasher = Sha256::new();

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
        distinct_entities.insert(rec.subject.clone());
        distinct_entities.insert(rec.object.clone());
        distinct_predicates.insert(rec.predicate.clone());

        hasher.update(rec.id.as_bytes());
        hasher.update(b"|");
        hasher.update(rec.subject.as_bytes());
        hasher.update(b"|");
        hasher.update(rec.predicate.as_bytes());
        hasher.update(b"|");
        hasher.update(rec.object.as_bytes());
        hasher.update(&rec.confidence.to_bits().to_le_bytes());
    }

    let state_digest = hex::encode(hasher.finalize());
    let now_stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    let snapshot_data = serde_json::json!({
        "schema_version": "2.0.0",
        "snapshot_timestamp_epoch": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
        "status": "candidate_verified",
        "entity_count": distinct_entities.len(),
        "predicate_count": distinct_predicates.len(),
        "fact_count": world_curriculum.total_facts,
        "curriculum_stages_count": report.world_stages_count,
        "average_confidence": world_curriculum.average_confidence,
        "state_graph_sha256": state_digest,
        "consistency_check": "PASS",
        "stages": world_curriculum.stages,
        "verified_facts": world_curriculum.ordered_records,
    });

    if let Some(parent) = world_out_dir.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }

    let staging_dir = PathBuf::from(format!("{}.staging_{}", world_out_dir.display(), now_stamp));
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
/// `config.json`, `tokenizer.json`, `checkpoint_state.json`, `optimizer.safetensors`, and
/// `PROMOTION_MANIFEST.json`) into `prod_dir` using two-phase backup and rollback.
pub fn promote_candidate_bundle_atomically(
    candidate_dir: &Path,
    prod_dir: &Path,
    dynamic_sha: &str,
    pass_rate: f64,
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
    let staging_dir = PathBuf::from(format!("{}.staging_{}", prod_dir.display(), now_stamp));
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

    let manifest = serde_json::json!({
        "promoted_from": candidate_dir.to_string_lossy(),
        "promoted_to": prod_dir.to_string_lossy(),
        "model_weights_sha256": dynamic_sha,
        "evaluation_pass_rate": pass_rate,
        "promoted_at_epoch": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
    });
    fs::write(
        staging_dir.join("PROMOTION_MANIFEST.json"),
        serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    promote_directory_atomically(&staging_dir, prod_dir).map_err(|e| e.to_string())?;

    // Post-promotion verification of target directory
    verify_model_bundle(prod_dir)?;
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
    println!("  --evaluate              Run regression skills evaluation against manual candidate checkpoints");
    println!("  --promote               Promote verified manual candidate to production (requires gate pass)");
    println!("  -h, --help              Print this help information");
    println!("================================================================================");
}

fn append_log(log_path: &Path, message: &str) {
    if let Some(parent) = log_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(log_path) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let _ = writeln!(file, "[{timestamp}] {message}");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    let cli = parse_manual_cli(&args, default_config_path)?;
    if cli.show_help {
        print_usage();
        return Ok(());
    }

    // 1. Load and strictly validate configuration (zero silent fallback on malformed JSON)
    let mut config = load_manual_config(&cli.config_path, cli.explicit_config)?;

    if let Some(m) = cli.mode_override {
        config.training_mode = m;
    }
    if let Some(s) = cli.steps_override {
        config.max_steps = s;
    }
    if let Some(b) = cli.batch_size_override {
        config.batch_size = b;
    }
    if let Some(lr) = cli.lr_override {
        config.learning_rate = lr;
    }
    if let Some(ds) = cli.dataset_override {
        config.dataset_dir = ds;
    }

    validate_manual_config(&config)?;

    println!("================================================================================");
    println!("  TARA MANUAL TRAINING WORKSPACE INITIALIZATION");
    println!("================================================================================");
    println!("Mode              : {}", config.training_mode);
    println!("Model Source      : {}", config.model_source_dir);
    println!("Dataset Path      : {}", config.dataset_dir);
    println!("Batch Size        : {}", config.batch_size);
    println!("Learning Rate     : {}", config.learning_rate);
    println!("Max Steps         : {}", config.max_steps);
    println!("Neural Target Dir : {}", config.neural_checkpoint_dir);
    println!("World Target Dir  : {}", config.world_model_checkpoint_dir);
    println!("--------------------------------------------------------------------------------");

    // 2. Promotion flow (Fail-closed gate + Atomic two-phase directory promotion)
    if cli.run_promote {
        println!("=== [PROMOTION GATE: MANUAL CANDIDATE -> PRODUCTION] ===");
        let candidate_neural = PathBuf::from(&config.neural_checkpoint_dir);
        let _ = recover_interrupted_promotion(&candidate_neural)?;

        verify_model_bundle(&candidate_neural)
            .map_err(|e| format!("PROMOTION ABORTED: Invalid manual candidate model bundle: {e}"))?;
        let opt_step = verify_candidate_continuation(&candidate_neural).map_err(|e| {
            format!("PROMOTION ABORTED: Manual candidate is not continuation-ready: {e}")
        })?;

        let candidate_weights = candidate_neural.join("model.safetensors");
        let dynamic_sha = compute_sha256(&candidate_weights.to_string_lossy())
            .map_err(|e| format!("Dynamic SHA computation failed: {e}"))?;
        println!("  Candidate Neural Weights SHA-256 : {}", dynamic_sha);
        println!("  Candidate Optimizer Step         : {}", opt_step);

        let evaluator = SkillsEvaluator::new(&candidate_neural.to_string_lossy());
        let prompts = vec![
            "Explain the Pythagorean theorem in geometry.".to_string(),
            "Write a simple function to reverse a vector in Rust.".to_string(),
            "What is a causal graph in probabilistic reasoning?".to_string(),
        ];
        let eval_report = evaluator.evaluate("core_reasoning", &prompts);
        let pass_rate = eval_report
            .get("pass_rate")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        println!("  Skills Evaluation Pass Rate      : {:.2}%", pass_rate * 100.0);

        if pass_rate < 0.60 {
            return Err(format!(
                "PROMOTION REJECTED: Candidate pass rate {:.2}% is below required 60.00% threshold",
                pass_rate * 100.0
            )
            .into());
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
            )?;
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
            let wm_staging = PathBuf::from(format!("{}.staging_{}", wm_target.display(), now_stamp));
            fs::create_dir_all(&wm_staging)?;
            fs::copy(&cand_world_file, wm_staging.join("world_model_state.json"))?;
            promote_directory_atomically(&wm_staging, &wm_target)?;
            println!("  Atomically promoted candidate world model to {:?}", wm_target);
        }

        println!("SUCCESS: Promoted manual candidate to production.");
        let neural_log = PathBuf::from(&config.logs_dir).join("neural_loss.log");
        append_log(
            &neural_log,
            &format!("PROMOTION_EXECUTED: Candidate SHA-256={dynamic_sha} -> storage/models/tara & production/neural"),
        );
        return Ok(());
    }

    // 3. Evaluation-only flow (Fail-closed on missing or invalid candidate)
    if cli.run_evaluation {
        println!("=== [EVALUATION: MANUAL CANDIDATE BENCHMARK SUITE] ===");
        let candidate_neural = PathBuf::from(&config.neural_checkpoint_dir);
        let _ = recover_interrupted_promotion(&candidate_neural)?;
        verify_model_bundle(&candidate_neural)
            .map_err(|e| format!("EVALUATION ABORTED: {e}"))?;

        let evaluator = SkillsEvaluator::new(&candidate_neural.to_string_lossy());
        let prompts = vec![
            "Explain the Pythagorean theorem in geometry.".to_string(),
            "Write a simple function to reverse a vector in Rust.".to_string(),
            "What is a causal graph in probabilistic reasoning?".to_string(),
            "Explain how AdamW optimizer handles weight decay.".to_string(),
        ];
        let eval_report = evaluator.evaluate("core_competencies", &prompts);
        let pass_count = eval_report
            .get("passed")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let total = eval_report
            .get("total_prompts")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let pass_rate = eval_report
            .get("pass_rate")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        println!("Evaluation Result:");
        println!("  Total Benchmarks : {}", total);
        println!("  Passed Tests     : {}", pass_count);
        println!("  Pass Rate        : {:.2}%", pass_rate * 100.0);

        let eval_output_dir = PathBuf::from(&config.evaluation_dir);
        fs::create_dir_all(&eval_output_dir)?;
        let eval_file = eval_output_dir.join("manual_eval_report.json");
        fs::write(&eval_file, serde_json::to_string_pretty(&eval_report)?)?;
        println!("Wrote evaluation report to {:?}", eval_file);
        return Ok(());
    }

    // 4. Execution: Neural Model Training (Strict model_source_dir verification & separated intermediate checkpoints)
    if config.training_mode == "neural" || config.training_mode == "both" {
        println!("\n=== [EXECUTING NEURAL MODEL TRAINING] ===");
        let src_dir = PathBuf::from(&config.model_source_dir);
        let _ = recover_interrupted_promotion(&src_dir)?;
        verify_model_bundle(&src_dir).map_err(|e| {
            format!(
                "Configured model_source_dir '{}' is missing or invalid (silent fallback prohibited): {}",
                config.model_source_dir, e
            )
        })?;

        let neural_out = PathBuf::from(&config.neural_checkpoint_dir);
        let _ = recover_interrupted_promotion(&neural_out)?;
        let intermediate_cp_dir = neural_out.join("checkpoints");
        fs::create_dir_all(&intermediate_cp_dir)?;

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
            device: Some("auto".to_string()),
            precision: Some("auto".to_string()),
            checkpoint_dir: Some(intermediate_cp_dir.to_string_lossy().to_string()),
            checkpoint_interval: Some(config.checkpoint_interval_steps),
            resume: false,
            resume_from: None,
            max_memory_mb: Some(65536.0),
        };

        println!("Calling shared training infrastructure (NativeSelfTrainer)...");
        let report = run_controlled_training_with_options(
            &config.model_source_dir,
            ".",
            1,
            options,
        )?;

        println!("  Neural Training Completed Successfully!");
        let run_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let runs_dir = PathBuf::from(&config.runs_dir);
        fs::create_dir_all(&runs_dir)?;
        let run_file = runs_dir.join(format!("run_neural_{run_id}.json"));
        fs::write(&run_file, serde_json::to_string_pretty(&report)?)?;
        println!("  Saved run telemetry to {:?}", run_file);

        let neural_log = PathBuf::from(&config.logs_dir).join("neural_loss.log");
        let loss = report
            .get("loss_after")
            .or_else(|| report.get("final_loss"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        append_log(
            &neural_log,
            &format!(
                "RUN_COMPLETED: id={run_id}, steps={}, final_loss={loss:.4}",
                config.max_steps
            ),
        );
    }

    // 5. Execution: Genuine World Model State Update & Consistency Verification
    if config.training_mode == "world_model" || config.training_mode == "both" {
        println!("\n=== [EXECUTING WORLD MODEL UPDATE & SNAPSHOT] ===");
        let world_out = PathBuf::from(&config.world_model_checkpoint_dir);
        let snapshot = execute_world_model_update(
            Path::new(&config.dataset_dir),
            Path::new(&config.curriculum_path),
            &world_out,
        )?;

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

        let world_log = PathBuf::from(&config.logs_dir).join("world_model_consistency.log");
        append_log(
            &world_log,
            &format!(
                "WORLD_MODEL_VERIFIED: destination={:?}, entities={}, facts={}, sha256={}",
                world_out, entities, facts, sha
            ),
        );
    }

    println!("================================================================================");
    println!("  MANUAL TRAINING WORKSPACE RUN FINISHED.");
    println!("  Checkpoints staged in isolated manual directory. Production unaffected.");
    println!("================================================================================");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tara_engine::safetensors::write_safetensors_with_shapes;

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
        assert!(parse_manual_cli(&args_missing_val, bad_cfg).is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_path_safety_rejects_production_and_source_overwrites() {
        let mut cfg = ManualTrainingConfig::default();
        assert!(validate_manual_config(&cfg).is_ok());

        // Pointing neural_checkpoint_dir to production must fail
        cfg.neural_checkpoint_dir = "storage/models/tara/sub".to_string();
        assert!(validate_manual_config(&cfg).is_err());

        // Pointing neural_checkpoint_dir to model_source_dir must fail
        cfg.model_source_dir = "custom_models/base".to_string();
        cfg.neural_checkpoint_dir = "custom_models/base".to_string();
        assert!(validate_manual_config(&cfg).is_err());
    }

    #[test]
    fn test_world_model_real_extraction_and_atomic_promotion() {
        let dir = unique_test_dir("world_model_and_promote");
        let ds_file = dir.join("world_facts.jsonl");
        fs::write(
            &ds_file,
            concat!(
                r#"{"id":"w1","subject":"TaraEngine","predicate":"executes_on","object":"NativeRust","context":"Core architecture","confidence":0.95}"#,
                "\n",
                r#"{"id":"w2","subject":"AdamW","predicate":"updates","object":"ModelWeights","context":"Optimization","confidence":0.92}"#,
                "\n"
            ),
        )
        .unwrap();

        let world_out = dir.join("world_chk");
        let snapshot =
            execute_world_model_update(&dir, &ds_file, &world_out).expect("world model update must succeed");

        assert!(world_out.join("world_model_state.json").exists());
        assert!(snapshot.get("entity_count").and_then(Value::as_u64).unwrap_or(0) >= 4);
        assert!(snapshot.get("fact_count").and_then(Value::as_u64).unwrap_or(0) >= 2);
        assert_eq!(
            snapshot.get("consistency_check").and_then(Value::as_str),
            Some("PASS")
        );

        // Test atomic neural candidate bundle promotion
        let cand_dir = dir.join("cand_neural");
        let prod_dir = dir.join("prod_neural");
        write_working_candidate(&cand_dir);

        let sha = compute_sha256(&cand_dir.join("model.safetensors").to_string_lossy()).unwrap();
        promote_candidate_bundle_atomically(&cand_dir, &prod_dir, &sha, 0.85).unwrap();

        assert!(verify_model_bundle(&prod_dir).is_ok());
        assert!(verify_candidate_continuation(&prod_dir).is_ok());
        assert!(prod_dir.join("PROMOTION_MANIFEST.json").exists());

        let _ = fs::remove_dir_all(&dir);
    }
}
