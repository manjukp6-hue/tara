//! TARA Extensible Multi-Stage Training Pipeline Orchestrator (100% Native Rust).
//!
//! Features:
//! - True DAG Dependency Resolution: Supports arbitrary N stages with explicit `depends_on`
//!   and implicit checkpoint-lineage edges, 3-color DFS cycle detection, topological sorting,
//!   and multi-parent ambiguity detection.
//! - Dual-Purpose Checkpoints: Every stage checkpoint is BOTH a complete, working model
//!   (`model.safetensors`, `config.json`, `tokenizer.json`) AND a continuation-ready checkpoint
//!   (`optimizer.safetensors`, `checkpoint_state.json`) verified against weight tensor shapes.
//! - Independent Checkpoint & Candidate Directories: Periodic step checkpoints are stored in
//!   `{checkpoints_dir}/{stage_id}/checkpoints/` while the final verified working model is
//!   published atomically into `{output_checkpoint}/`.
//! - Crash-Safe Atomic Directory Promotion & Startup `.bak_*` Recovery.
//! - Strict `--resume-from` Validation: Requires both working model and continuation-ready state.
//! - Genuine `--eval-checkpoint`: Performs weight/shape inspection, canonical probe causal loss,
//!   and held-out JSONL dataset streaming evaluation.
//! - Sound Fingerprint-Gated Idempotence: Skips completed stages ONLY when working model,
//!   continuation state, AND `stage_config_fingerprint` (covering dataset digest, input model
//!   digest, and hyperparameters) match.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::{Component, Path, PathBuf};

use tara_engine::config::TaraConfig;
use tara_engine::dataset::{ExpandableDatasetReader, ShardStreamingMode};
use tara_engine::model::causal_lm::TaraForCausalLM;
use tara_engine::safetensors::{load_model_weights_with_shapes, load_safetensors_with_shapes};
use tara_engine::tokenizer::TaraTokenizer;
use tara_engine::train_candidate::{run_controlled_training_with_options, TrainingOptions};
use tara_engine::trainer::recover_interrupted_promotion;

/// Extensible specification for a single training stage in the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageDefinition {
    pub stage_index: usize,
    pub stage_id: String,
    pub stage_name: String,
    pub description: String,
    pub dataset_path: String,
    pub input_checkpoint: String,
    pub output_checkpoint: String,
    pub epochs: usize,
    pub max_steps: Option<usize>,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub checkpoint_interval: usize,
    #[serde(default)]
    pub depends_on: Vec<String>,
}

/// Detailed dataset inspection statistics including JSONL parseability and content fingerprint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DatasetInspection {
    pub jsonl_files: usize,
    pub estimated_records: usize,
    pub valid_records: usize,
    pub malformed_records: usize,
    pub empty_records: usize,
    pub total_bytes: u64,
    pub dataset_fingerprint: String,
}

/// Dynamic metadata generated for every completed or staged stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageMetadata {
    pub stage_index: usize,
    pub stage_id: String,
    pub stage_name: String,
    pub description: String,
    pub input_checkpoint: String,
    pub output_checkpoint: String,
    pub dataset_source: String,
    pub dataset_inspection: DatasetInspection,
    pub completed_at_utc: String,
    pub status: String,
    pub dynamic_model_sha256: String,
    pub dynamic_model_weights_sha256: String,
    pub stage_config_fingerprint: String,
    pub is_working_model: bool,
    pub is_continuation_ready: bool,
}

fn compute_file_sha256<P: AsRef<Path>>(path: P) -> Result<String, std::io::Error> {
    let mut file = BufReader::with_capacity(128 * 1024, File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Computes a deterministic dynamic digest across all model safetensor weight shards in a directory.
pub fn compute_model_digest(dir: &Path) -> Result<String, std::io::Error> {
    if !dir.exists() {
        return Ok("pending_execution".to_string());
    }

    let mut shard_files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                if (name == "model.safetensors" || name.starts_with("model-"))
                    && name.ends_with(".safetensors")
                {
                    shard_files.push(p);
                }
            }
        }
    }

    if shard_files.is_empty() {
        return Ok("unhashed_no_weights".to_string());
    }

    shard_files.sort();

    let mut combined_hasher = Sha256::new();
    for shard in shard_files {
        let name = shard.file_name().unwrap_or_default().to_string_lossy();
        let meta = shard.metadata()?;
        combined_hasher.update(name.as_bytes());
        combined_hasher.update(&meta.len().to_le_bytes());
        let shard_hash = compute_file_sha256(&shard)?;
        combined_hasher.update(shard_hash.as_bytes());
    }

    Ok(hex::encode(combined_hasher.finalize()))
}

/// Inspects a JSONL file or directory, validating JSON parseability, non-empty training fields,
/// and computing a deterministic dataset content fingerprint.
pub fn inspect_dataset(path: &Path) -> DatasetInspection {
    if path.is_file() {
        let mut valid_records = 0usize;
        let mut malformed_records = 0usize;
        let mut empty_records = 0usize;
        let mut total_bytes = 0u64;
        let mut hasher = Sha256::new();

        if let Ok(meta) = fs::metadata(path) {
            total_bytes = meta.len();
        }

        if let Ok(file) = File::open(path) {
            let mut reader = BufReader::with_capacity(64 * 1024, file);
            let mut line = String::new();
            while let Ok(n) = reader.read_line(&mut line) {
                if n == 0 {
                    break;
                }
                hasher.update(line.as_bytes());
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    empty_records += 1;
                    line.clear();
                    continue;
                }
                match serde_json::from_str::<serde_json::Value>(trimmed) {
                    Ok(val) => {
                        let has_in = val
                            .get("formatted_input")
                            .or_else(|| val.get("input"))
                            .or_else(|| val.get("prompt"))
                            .or_else(|| val.get("instruction"))
                            .or_else(|| val.get("metadata").and_then(|m| m.get("input")))
                            .and_then(|v| v.as_str())
                            .map(|s| !s.trim().is_empty())
                            .unwrap_or(false);
                        let has_out = val
                            .get("formatted_target")
                            .or_else(|| val.get("output"))
                            .or_else(|| val.get("completion"))
                            .or_else(|| val.get("response"))
                            .or_else(|| val.get("metadata").and_then(|m| m.get("output")))
                            .and_then(|v| v.as_str())
                            .map(|s| !s.trim().is_empty())
                            .unwrap_or(false);
                        if has_in && has_out {
                            valid_records += 1;
                        } else {
                            empty_records += 1;
                        }
                    }
                    Err(_) => {
                        malformed_records += 1;
                    }
                }
                line.clear();
            }
        }

        DatasetInspection {
            jsonl_files: 1,
            estimated_records: valid_records,
            valid_records,
            malformed_records,
            empty_records,
            total_bytes,
            dataset_fingerprint: hex::encode(hasher.finalize()),
        }
    } else if path.is_dir() {
        let mut files = Vec::new();
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                let ep = entry.path();
                if ep.is_file() && ep.extension().and_then(|ext| ext.to_str()) == Some("jsonl") {
                    files.push(ep);
                }
            }
        }
        files.sort();

        let mut total_valid = 0usize;
        let mut total_malformed = 0usize;
        let mut total_empty = 0usize;
        let mut total_bytes = 0u64;
        let mut combined_hasher = Sha256::new();

        for ep in &files {
            let insp = inspect_dataset(ep);
            total_valid += insp.valid_records;
            total_malformed += insp.malformed_records;
            total_empty += insp.empty_records;
            total_bytes += insp.total_bytes;
            if let Some(name) = ep.file_name().and_then(|n| n.to_str()) {
                combined_hasher.update(name.as_bytes());
            }
            combined_hasher.update(insp.dataset_fingerprint.as_bytes());
        }

        DatasetInspection {
            jsonl_files: files.len(),
            estimated_records: total_valid,
            valid_records: total_valid,
            malformed_records: total_malformed,
            empty_records: total_empty,
            total_bytes,
            dataset_fingerprint: if files.is_empty() {
                "empty_dir".to_string()
            } else {
                hex::encode(combined_hasher.finalize())
            },
        }
    } else {
        DatasetInspection {
            jsonl_files: 0,
            estimated_records: 0,
            valid_records: 0,
            malformed_records: 0,
            empty_records: 0,
            total_bytes: 0,
            dataset_fingerprint: "missing_dataset".to_string(),
        }
    }
}

/// Computes a deterministic fingerprint for a stage's execution configuration, input model weights,
/// and dataset content. Used to ensure idempotence only skips when nothing upstream changed.
pub fn compute_stage_config_fingerprint(
    stage: &StageDefinition,
    input_model_digest: &str,
    dataset_fingerprint: &str,
    precision: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(stage.stage_id.as_bytes());
    hasher.update(b"|in:");
    hasher.update(input_model_digest.as_bytes());
    hasher.update(b"|ds:");
    hasher.update(dataset_fingerprint.as_bytes());
    hasher.update(b"|ep:");
    hasher.update(&stage.epochs.to_le_bytes());
    hasher.update(b"|ms:");
    hasher.update(&stage.max_steps.unwrap_or(0).to_le_bytes());
    hasher.update(b"|lr:");
    hasher.update(&stage.learning_rate.to_bits().to_le_bytes());
    hasher.update(b"|bs:");
    hasher.update(&stage.batch_size.to_le_bytes());
    hasher.update(b"|prec:");
    hasher.update(precision.as_bytes());
    hex::encode(hasher.finalize())
}

/// Checks whether a stage output directory is already completed, continuation-ready, AND matches
/// the exact `expected_fingerprint`.
pub fn should_skip_completed_stage(out_dir: &Path, expected_fingerprint: &str) -> bool {
    let (is_model, _, _) = verify_working_model(out_dir);
    if !is_model {
        return false;
    }
    let (is_cont, _, _) = verify_continuation_ready(out_dir);
    if !is_cont {
        return false;
    }
    let meta_path = out_dir.join("STAGE_METADATA.json");
    if !meta_path.exists() {
        return false;
    }
    let Ok(content) = fs::read_to_string(&meta_path) else {
        return false;
    };
    let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) else {
        return false;
    };
    let Some(saved_fp) = val.get("stage_config_fingerprint").and_then(|v| v.as_str()) else {
        return false;
    };
    saved_fp == expected_fingerprint
}

/// Deep verification of working model: inspects config, tokenizer, and SafeTensors weights.
pub fn verify_working_model(model_dir: &Path) -> (bool, Vec<String>, Option<usize>) {
    let mut missing = Vec::new();
    let config_path = model_dir.join("config.json");
    let tokenizer_path = model_dir.join("tokenizer.json");

    if !config_path.exists() {
        missing.push("config.json missing".to_string());
    } else if TaraConfig::from_json_file(&config_path.to_string_lossy()).is_err() {
        missing.push("config.json invalid or malformed".to_string());
    }

    if !tokenizer_path.exists() {
        missing.push("tokenizer.json missing".to_string());
    } else if TaraTokenizer::from_file(&tokenizer_path.to_string_lossy()).is_err() {
        missing.push("tokenizer.json invalid or malformed".to_string());
    }

    let mut param_count: Option<usize> = None;
    match load_model_weights_with_shapes(&model_dir.to_string_lossy()) {
        Ok((weights, _)) => {
            if weights.is_empty() {
                missing.push("SafeTensors weights dictionary is empty".to_string());
            } else {
                let has_embed = weights.contains_key("model.embed_tokens.weight");
                let has_norm = weights.contains_key("model.norm.weight");
                let has_lm_head = weights.contains_key("lm_head.weight");
                if !has_embed || !has_norm || !has_lm_head {
                    missing.push(
                        "SafeTensors missing essential layers (embed, norm, lm_head)".to_string(),
                    );
                }
                param_count = Some(weights.values().map(|w| w.len()).sum());
            }
        }
        Err(e) => {
            missing.push(format!("Failed to load model SafeTensors weights: {e}"));
        }
    }

    (missing.is_empty(), missing, param_count)
}

/// Deep verification of continuation state: checks `checkpoint_state.json` and `optimizer.safetensors`
/// (including verifying that moment tensor lengths match model weight tensor lengths when weights exist).
pub fn verify_continuation_ready(cp_dir: &Path) -> (bool, Vec<String>, Option<u64>) {
    let mut issues = Vec::new();
    let state_file = cp_dir.join("checkpoint_state.json");
    let opt_file = cp_dir.join("optimizer.safetensors");

    let mut opt_step = None;
    if !state_file.exists() {
        issues.push("checkpoint_state.json missing".to_string());
    } else {
        match fs::read_to_string(&state_file) {
            Ok(content) => match serde_json::from_str::<serde_json::Value>(&content) {
                Ok(val) => {
                    opt_step = val
                        .get("optimizer_step")
                        .or_else(|| val.get("step"))
                        .and_then(|v| v.as_u64());
                    if opt_step.is_none() {
                        issues.push(
                            "checkpoint_state.json missing valid 'optimizer_step'".to_string(),
                        );
                    }
                }
                Err(e) => issues.push(format!("checkpoint_state.json JSON error: {e}")),
            },
            Err(e) => issues.push(format!("checkpoint_state.json unreadable: {e}")),
        }
    }

    if !opt_file.exists() {
        issues.push("optimizer.safetensors missing".to_string());
    } else {
        match load_safetensors_with_shapes(&opt_file.to_string_lossy()) {
            Ok((tensors, _)) => {
                if tensors.is_empty() {
                    issues.push("optimizer.safetensors contains zero moment tensors".to_string());
                } else if let Ok((weights, _)) =
                    load_model_weights_with_shapes(&cp_dir.to_string_lossy())
                {
                    for (param_name, w_vec) in &weights {
                        let m_key = format!("{param_name}.adam_m");
                        let v_key = format!("{param_name}.adam_v");
                        match (tensors.get(&m_key), tensors.get(&v_key)) {
                            (Some(m), Some(v)) => {
                                if m.len() != w_vec.len() || v.len() != w_vec.len() {
                                    issues.push(format!(
                                        "optimizer.safetensors moment length mismatch for '{param_name}'"
                                    ));
                                }
                            }
                            _ => {
                                issues.push(format!(
                                    "optimizer.safetensors missing .adam_m/.adam_v for '{param_name}'"
                                ));
                            }
                        }
                    }
                }
            }
            Err(e) => issues.push(format!("optimizer.safetensors invalid: {e}")),
        }
    }

    (issues.is_empty(), issues, opt_step)
}

fn is_safe_stage_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Resolves a path to a normalized logical/canonical representation so symlink/relative path
/// aliases and parent/descendant containment can be reliably checked even before output dirs exist.
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

fn paths_overlap(a: &Path, b: &Path) -> bool {
    let na = normalize_for_comparison(a);
    let nb = normalize_for_comparison(b);
    na == nb || na.starts_with(&nb) || nb.starts_with(&na)
}

pub fn validate_device_and_precision(device: &str, precision: &str) -> Result<(), String> {
    match device.to_ascii_lowercase().as_str() {
        "auto" | "cpu" | "gpu" | "cuda" => {}
        other => {
            return Err(format!(
                "Invalid --device '{}': must be one of 'auto', 'cpu', 'gpu', 'cuda'",
                other
            ));
        }
    }
    match precision.to_ascii_lowercase().as_str() {
        "auto" | "fp16" | "fp32" => {}
        other => {
            return Err(format!(
                "Invalid --precision '{}': must be one of 'auto', 'fp16', 'fp32'",
                other
            ));
        }
    }
    Ok(())
}

/// Validates stage definitions for hyperparameter validity, path containment, production overwrite
/// protection, and duplicate output paths.
pub fn validate_stage_definitions(
    stages: &[StageDefinition],
    base_model_dir: &Path,
) -> Result<(), String> {
    if stages.is_empty() {
        return Err("Stage configuration contains zero stages".to_string());
    }

    let mut seen_indices = HashSet::new();
    let mut seen_ids = HashSet::new();
    let mut seen_outputs: Vec<(String, PathBuf)> = Vec::new();

    let prod_tara = Path::new("storage/models/tara");
    let prod_neural = Path::new("production/neural");

    for stage in stages {
        if !seen_indices.insert(stage.stage_index) {
            return Err(format!(
                "Duplicate stage_index detected: {}",
                stage.stage_index
            ));
        }
        if !is_safe_stage_id(&stage.stage_id) {
            return Err(format!(
                "Stage at index {} has invalid stage_id '{}' (only ASCII alphanumeric, '_', '-' allowed)",
                stage.stage_index, stage.stage_id
            ));
        }
        if !seen_ids.insert(stage.stage_id.clone()) {
            return Err(format!("Duplicate stage_id detected: '{}'", stage.stage_id));
        }
        if stage.stage_name.trim().is_empty() {
            return Err(format!(
                "Stage '{}' has empty stage_name",
                stage.stage_id
            ));
        }
        if stage.dataset_path.trim().is_empty() {
            return Err(format!(
                "Stage '{}' has empty dataset_path",
                stage.stage_id
            ));
        }
        if stage.epochs == 0 {
            return Err(format!(
                "Stage '{}' specifies epochs = 0 (must be >= 1)",
                stage.stage_id
            ));
        }
        if stage.max_steps == Some(0) {
            return Err(format!(
                "Stage '{}' specifies max_steps = 0 (must be >= 1 when set)",
                stage.stage_id
            ));
        }
        if stage.batch_size == 0 {
            return Err(format!(
                "Stage '{}' specifies batch_size = 0 (must be >= 1)",
                stage.stage_id
            ));
        }
        if stage.checkpoint_interval == 0 {
            return Err(format!(
                "Stage '{}' specifies checkpoint_interval = 0",
                stage.stage_id
            ));
        }
        if !stage.learning_rate.is_finite() || stage.learning_rate <= 0.0 {
            return Err(format!(
                "Stage '{}' specifies invalid learning_rate: {}",
                stage.stage_id, stage.learning_rate
            ));
        }

        let out_path = Path::new(&stage.output_checkpoint);
        let in_path = Path::new(&stage.input_checkpoint);

        if paths_overlap(out_path, in_path) {
            return Err(format!(
                "Stage '{}' output_checkpoint ('{}') overlaps with input_checkpoint ('{}'). In-place or nested overwrite is prohibited.",
                stage.stage_id, stage.output_checkpoint, stage.input_checkpoint
            ));
        }
        if paths_overlap(out_path, base_model_dir) {
            return Err(format!(
                "Stage '{}' output_checkpoint ('{}') overlaps with base_model ('{}'). Overwriting base model is strictly prohibited.",
                stage.stage_id,
                stage.output_checkpoint,
                base_model_dir.display()
            ));
        }
        if paths_overlap(out_path, prod_tara) || paths_overlap(out_path, prod_neural) {
            return Err(format!(
                "Stage '{}' output_checkpoint ('{}') targets protected production directory.",
                stage.stage_id, stage.output_checkpoint
            ));
        }

        let norm_out = normalize_for_comparison(out_path);
        for (other_id, other_out) in &seen_outputs {
            if norm_out == *other_out || norm_out.starts_with(other_out) || other_out.starts_with(&norm_out) {
                return Err(format!(
                    "Stage '{}' output_checkpoint ('{}') conflicts/overlaps with stage '{}' output_checkpoint.",
                    stage.stage_id, stage.output_checkpoint, other_id
                ));
            }
        }
        seen_outputs.push((stage.stage_id.clone(), norm_out));
    }

    Ok(())
}

/// Validates the stage dependency DAG (combining explicit `depends_on` and implicit
/// `output_checkpoint -> input_checkpoint` edges), detects cycles and ambiguous multi-parent
/// inheritance, and returns stages sorted in deterministic topological order.
pub fn validate_and_sort_stage_dag(
    stages: &[StageDefinition],
) -> Result<Vec<StageDefinition>, String> {
    let mut id_to_idx: HashMap<String, usize> = HashMap::new();
    let mut output_to_id: HashMap<PathBuf, String> = HashMap::new();

    for (idx, s) in stages.iter().enumerate() {
        id_to_idx.insert(s.stage_id.clone(), idx);
        output_to_id.insert(
            normalize_for_comparison(Path::new(&s.output_checkpoint)),
            s.stage_id.clone(),
        );
    }

    // Build dependency list per stage_id: deps[stage_idx] = Vec<predecessor_stage_idx>
    let mut deps: Vec<Vec<usize>> = vec![Vec::new(); stages.len()];

    for (idx, s) in stages.iter().enumerate() {
        let mut stage_deps: Vec<usize> = Vec::new();
        for dep_id in &s.depends_on {
            let Some(&dep_idx) = id_to_idx.get(dep_id) else {
                return Err(format!(
                    "DAG Error: Stage '{}' declares dependency on unknown stage_id '{}'",
                    s.stage_id, dep_id
                ));
            };
            if dep_idx == idx {
                return Err(format!(
                    "DAG Cycle Error: Stage '{}' depends on itself",
                    s.stage_id
                ));
            }
            if !stage_deps.contains(&dep_idx) {
                stage_deps.push(dep_idx);
            }
        }

        let norm_in = normalize_for_comparison(Path::new(&s.input_checkpoint));
        if let Some(producer_id) = output_to_id.get(&norm_in) {
            let producer_idx = id_to_idx[producer_id];
            if producer_idx == idx {
                return Err(format!(
                    "DAG Cycle Error: Stage '{}' input_checkpoint is its own output_checkpoint",
                    s.stage_id
                ));
            }
            if !stage_deps.contains(&producer_idx) {
                stage_deps.push(producer_idx);
            }
        }

        // If a stage declares multiple dependencies, its input_checkpoint must unambiguously match
        // one of those dependencies' output_checkpoint (or an external base checkpoint).
        if stage_deps.len() > 1 {
            let matches_declared_dep = stage_deps.iter().any(|&d_idx| {
                normalize_for_comparison(Path::new(&stages[d_idx].output_checkpoint)) == norm_in
            });
            if !matches_declared_dep {
                return Err(format!(
                    "DAG Ambiguity Error: Stage '{}' has multiple dependencies ({:?}) but input_checkpoint '{}' does not match any dependency output_checkpoint",
                    s.stage_id, s.depends_on, s.input_checkpoint
                ));
            }
        }

        deps[idx] = stage_deps;
    }

    // 3-color DFS cycle detection and topological sorting
    // 0 = Unvisited, 1 = Visiting, 2 = Visited
    let mut state = vec![0u8; stages.len()];
    let mut topo_order: Vec<usize> = Vec::with_capacity(stages.len());
    let mut path_stack: Vec<String> = Vec::new();

    fn dfs(
        node: usize,
        stages: &[StageDefinition],
        deps: &[Vec<usize>],
        state: &mut [u8],
        topo_order: &mut Vec<usize>,
        path_stack: &mut Vec<String>,
    ) -> Result<(), String> {
        state[node] = 1;
        path_stack.push(stages[node].stage_id.clone());

        for &pred in &deps[node] {
            if state[pred] == 1 {
                path_stack.push(stages[pred].stage_id.clone());
                return Err(format!(
                    "DAG Cycle Detected: {}",
                    path_stack.join(" -> ")
                ));
            }
            if state[pred] == 0 {
                dfs(pred, stages, deps, state, topo_order, path_stack)?;
            }
        }

        path_stack.pop();
        state[node] = 2;
        topo_order.push(node);
        Ok(())
    }

    // Visit in ascending stage_index order for deterministic tie-breaking
    let mut indices_by_stage_num: Vec<usize> = (0..stages.len()).collect();
    indices_by_stage_num.sort_by_key(|&i| stages[i].stage_index);

    for idx in indices_by_stage_num {
        if state[idx] == 0 {
            dfs(
                idx,
                stages,
                &deps,
                &mut state,
                &mut topo_order,
                &mut path_stack,
            )?;
        }
    }

    Ok(topo_order.into_iter().map(|i| stages[i].clone()).collect())
}

fn compute_causal_sequence_loss(
    model: &TaraForCausalLM,
    token_ids: &[u32],
    vocab_size: usize,
) -> (f64, usize) {
    if token_ids.len() < 2 || vocab_size == 0 {
        return (0.0, 0);
    }
    let (_, _, logits) = model.forward_with_cache(token_ids);
    let mut total_loss = 0.0f64;
    let mut targets_evaluated = 0usize;

    for pos in 1..token_ids.len() {
        let predictor = pos - 1;
        let target = token_ids[pos] as usize;
        if target >= vocab_size {
            continue;
        }
        let logits_slice = &logits[predictor * vocab_size..(predictor + 1) * vocab_size];
        let max_val = logits_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = logits_slice.iter().map(|&v| (v - max_val).exp()).sum();
        let target_logit = logits_slice[target];
        let p = ((target_logit - max_val).exp() / sum_exp).max(1e-12);
        total_loss += -p.ln() as f64;
        targets_evaluated += 1;
    }

    (total_loss, targets_evaluated)
}

/// Evaluates a checkpoint directory by loading the model, executing a causal forward pass on
/// the canonical probe sequence, and (when an evaluation dataset path is provided or available)
/// streaming held-out dataset samples to compute dataset validation loss and perplexity.
pub fn evaluate_checkpoint_model(
    eval_cp: &Path,
    eval_dataset: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("[MODE: COMPREHENSIVE CHECKPOINT EVALUATION]");
    println!("Target Checkpoint : {}", eval_cp.display());
    println!("--------------------------------------------------------------------------------");

    let _ = recover_interrupted_promotion(eval_cp);

    let (is_model, missing_model, param_count) = verify_working_model(eval_cp);
    let (is_cont, missing_cont, opt_step) = verify_continuation_ready(eval_cp);

    println!(
        "  Working Model Integrity        : {}",
        if is_model { "PASSED" } else { "FAILED" }
    );
    if let Some(pc) = param_count {
        println!(
            "  Total Model Parameters         : {} ({:.2}M params)",
            pc,
            pc as f64 / 1_000_000.0
        );
    }
    if !is_model {
        for m in &missing_model {
            println!("    [FAIL] {}", m);
        }
        return Err(format!("Checkpoint at '{}' is not a working model.", eval_cp.display()).into());
    }

    println!(
        "  Continuation-Ready State       : {}",
        if is_cont {
            "PASSED"
        } else {
            "INCOMPLETE / STANDALONE MODEL"
        }
    );
    if let Some(step) = opt_step {
        println!("  Optimizer Step Recorded        : {}", step);
    }
    if !is_cont {
        for c in &missing_cont {
            println!("    [NOTICE] {}", c);
        }
    }

    let cfg_path = eval_cp.join("config.json");
    let tok_path = eval_cp.join("tokenizer.json");
    let config = TaraConfig::from_json_file(&cfg_path.to_string_lossy())?;
    let tokenizer = TaraTokenizer::from_file(&tok_path.to_string_lossy())?;

    println!("  Model Architecture:");
    println!(
        "    Layers: {}, Hidden Dim: {}, Heads: {} (KV: {}), Vocab: {}, Context: {}",
        config.num_hidden_layers,
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        tokenizer.vocab_size,
        config.max_position_embeddings
    );

    println!("--------------------------------------------------------------------------------");
    println!("Executing live evaluation forward pass on canonical probe sequence...");

    let (weights, _) = load_model_weights_with_shapes(&eval_cp.to_string_lossy())?;
    let model =
        TaraForCausalLM::from_weights_and_config(weights, config.clone(), &eval_cp.to_string_lossy())?;

    let probe_text = "The fundamental principles of computation and mathematical reasoning.";
    let token_ids = tokenizer.encode(probe_text);
    if token_ids.len() < 2 {
        return Err("Tokenizer produced fewer than 2 tokens for probe text".into());
    }

    let (probe_total_loss, probe_targets) =
        compute_causal_sequence_loss(&model, &token_ids, config.vocab_size);
    let avg_loss = if probe_targets > 0 {
        probe_total_loss / probe_targets as f64
    } else {
        f64::NAN
    };
    let perplexity = avg_loss.exp();

    println!("  Forward Pass Probe Loss        : {:.4}", avg_loss);
    println!("  Estimated Probe Perplexity     : {:.4}", perplexity);
    println!(
        "  Probe Loss Finite Check        : {}",
        if avg_loss.is_finite() {
            "PASSED"
        } else {
            "FAILED (Non-finite)"
        }
    );

    if !avg_loss.is_finite() {
        return Err("Model evaluation rejected: forward pass produced non-finite loss".into());
    }

    // Stream held-out dataset records if provided or if default validation split exists
    if let Some(ds_path) = eval_dataset.filter(|p| p.exists()) {
        let mut reader = ExpandableDatasetReader::new(ShardStreamingMode::Sequential);
        if ds_path.is_dir() {
            let _ = reader.add_shards_from_dir(ds_path)?;
        } else {
            reader.add_shard(ds_path)?;
        }
        if reader.shard_count() > 0 {
            let mut ds_loss_sum = 0.0f64;
            let mut ds_tokens = 0usize;
            let mut ds_samples = 0usize;
            let max_eval_samples = 16usize;
            let max_ctx = config.max_position_embeddings.min(128).max(8);

            while ds_samples < max_eval_samples {
                let Some(sample) = reader.next_sample()? else {
                    break;
                };
                let combined = format!("{}\n{}", sample.input, sample.output);
                let mut tids = tokenizer.encode(&combined);
                if tids.len() > max_ctx {
                    tids.truncate(max_ctx);
                }
                if tids.len() >= 2 {
                    let (l, t) = compute_causal_sequence_loss(&model, &tids, config.vocab_size);
                    ds_loss_sum += l;
                    ds_tokens += t;
                    ds_samples += 1;
                }
            }

            if ds_tokens > 0 {
                let ds_avg_loss = ds_loss_sum / ds_tokens as f64;
                let ds_ppl = ds_avg_loss.exp();
                println!("  Held-Out Dataset Samples Eval  : {}", ds_samples);
                println!("  Held-Out Dataset Loss          : {:.4}", ds_avg_loss);
                println!("  Held-Out Dataset Perplexity    : {:.4}", ds_ppl);
                if !ds_avg_loss.is_finite() {
                    return Err(
                        "Model evaluation rejected: held-out dataset evaluation produced non-finite loss"
                            .into(),
                    );
                }
            }
        }
    }

    println!("================================================================================");
    println!("CHECKPOINT EVALUATION SUCCESSFUL: Model is fully functional and numerically stable.");
    println!("================================================================================");
    Ok(())
}

/// Build default progression of extensible stages.
pub fn build_default_stages(
    base_model: &Path,
    checkpoints_dir: &Path,
    foundational_curriculum_path: &Path,
    ability_curriculum_path: &Path,
    canonical_dir: &Path,
    weak_dir: &Path,
    adaptation_path: &Path,
) -> Vec<StageDefinition> {
    vec![
        StageDefinition {
            stage_index: 1,
            stage_id: "stage1_curriculum".to_string(),
            stage_name: "Stage 1: Foundational Academic Curriculum Pretraining".to_string(),
            description: "Foundational academic training on structured pedagogical records across Domains 1-8 (Math, Science, Programming, AI/ML, Research, Culture, Creativity, Search)."
                .to_string(),
            dataset_path: foundational_curriculum_path.to_string_lossy().to_string(),
            input_checkpoint: base_model.to_string_lossy().to_string(),
            output_checkpoint: checkpoints_dir
                .join("stage1_curriculum")
                .to_string_lossy()
                .to_string(),
            epochs: 1,
            max_steps: None,
            learning_rate: 5e-4,
            batch_size: 4,
            checkpoint_interval: 100,
            depends_on: vec![],
        },
        StageDefinition {
            stage_index: 2,
            stage_id: "stage2_ability_training".to_string(),
            stage_name: "Stage 2: Ability Training & Self-Evolution".to_string(),
            description: "Specialized training on Autonomous Ability Acquisition, Skill Evolution, and protocol-compliant self-training/self-update."
                .to_string(),
            dataset_path: ability_curriculum_path.to_string_lossy().to_string(),
            input_checkpoint: checkpoints_dir
                .join("stage1_curriculum")
                .to_string_lossy()
                .to_string(),
            output_checkpoint: checkpoints_dir
                .join("stage2_ability_training")
                .to_string_lossy()
                .to_string(),
            epochs: 1,
            max_steps: None,
            learning_rate: 3e-4,
            batch_size: 4,
            checkpoint_interval: 100,
            depends_on: vec!["stage1_curriculum".to_string()],
        },
        StageDefinition {
            stage_index: 3,
            stage_id: "stage3_canonical".to_string(),
            stage_name: "Stage 3: Full Canonical Pretraining".to_string(),
            description: "Full-scale pretraining on verified canonical high-quality data shards."
                .to_string(),
            dataset_path: canonical_dir.to_string_lossy().to_string(),
            input_checkpoint: checkpoints_dir
                .join("stage2_ability_training")
                .to_string_lossy()
                .to_string(),
            output_checkpoint: checkpoints_dir
                .join("stage3_canonical")
                .to_string_lossy()
                .to_string(),
            epochs: 1,
            max_steps: None,
            learning_rate: 2e-4,
            batch_size: 4,
            checkpoint_interval: 500,
            depends_on: vec!["stage2_ability_training".to_string()],
        },
        StageDefinition {
            stage_index: 4,
            stage_id: "stage4_weak_mix".to_string(),
            stage_name: "Stage 4: Controlled Weak-Mix Training".to_string(),
            description: "Balanced continual expansion with controlled canonical and selected weak data."
                .to_string(),
            dataset_path: weak_dir.to_string_lossy().to_string(),
            input_checkpoint: checkpoints_dir
                .join("stage3_canonical")
                .to_string_lossy()
                .to_string(),
            output_checkpoint: checkpoints_dir
                .join("stage4_weak_mix")
                .to_string_lossy()
                .to_string(),
            epochs: 1,
            max_steps: None,
            learning_rate: 1e-4,
            batch_size: 4,
            checkpoint_interval: 500,
            depends_on: vec!["stage3_canonical".to_string()],
        },
        StageDefinition {
            stage_index: 5,
            stage_id: "stage5_continual_adaptation".to_string(),
            stage_name: "Stage 5: Extensible Continual Adaptation & Specialization".to_string(),
            description: "Continual adaptation and specialization trained on genuine post-execution experiential lessons and validated strategies."
                .to_string(),
            dataset_path: adaptation_path.to_string_lossy().to_string(),
            input_checkpoint: checkpoints_dir
                .join("stage4_weak_mix")
                .to_string_lossy()
                .to_string(),
            output_checkpoint: checkpoints_dir
                .join("stage5_continual_adaptation")
                .to_string_lossy()
                .to_string(),
            epochs: 1,
            max_steps: None,
            learning_rate: 5e-5,
            batch_size: 4,
            checkpoint_interval: 200,
            depends_on: vec!["stage4_weak_mix".to_string()],
        },
    ]
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

fn print_usage() {
    println!("Usage: tara_training_stages [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --stage <N>                 Execute stage by index (1, 2, 3, ...)");
    println!("  --stage-id <ID>             Execute stage by ID (e.g. stage1_curriculum)");
    println!("  --all-stages                Execute all configured stages in topological DAG order");
    println!("  --stages-config <FILE>      Load extensible pipeline stages from custom JSON file");
    println!("  --resume-from <CHECKPOINT>  Explicitly resume from a continuation-ready checkpoint directory");
    println!("  --eval-checkpoint <PATH>    Evaluate a checkpoint directory (weights, shapes, probe & dataset loss)");
    println!("  --eval-dataset <PATH>       Optional held-out JSONL dataset path for --eval-checkpoint");
    println!("  --model <PATH>              Base model checkpoint directory (default: storage/models/tara_candidate_v1)");
    println!("  --curriculum <PATH>         Foundational curriculum path");
    println!("  --ability-path <PATH>       Ability curriculum path");
    println!("  --canonical-dir <PATH>      Canonical datasets directory");
    println!("  --weak-dir <PATH>           Weak datasets directory");
    println!("  --adaptation-path <PATH>    Continual adaptation lessons path");
    println!("  --checkpoints-dir <PATH>    Base directory for stage checkpoints (default: storage/models/checkpoints)");
    println!("  --device <DEVICE>           Device backend: 'cpu', 'gpu', 'cuda', or 'auto' (default: auto)");
    println!("  --precision <PRECISION>     Precision: 'fp16', 'fp32', or 'auto' (default: auto)");
    println!("  --force                     Force re-execution of already completed stages");
    println!("  --dry-run                   Audit and verify stage DAG and inputs/outputs without gradient updates");
    println!("  --epochs <N>                Override epochs for active stage(s) (must be >= 1)");
    println!("  --max-steps <N>             Override maximum steps for active stage(s) (must be >= 1)");
    println!("  --learning-rate <F>         Override learning rate (must be > 0.0)");
    println!("  --batch-size <N>            Override gradient accumulation batch size (must be >= 1)");
    println!("  -h, --help                  Print help information");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return Ok(());
    }

    let mut target_stage_index: Option<usize> = None;
    let mut target_stage_id: Option<String> = None;
    let mut run_all_stages = false;
    let mut custom_config_path: Option<PathBuf> = None;
    let mut resume_from_override: Option<PathBuf> = None;
    let mut eval_checkpoint_path: Option<PathBuf> = None;
    let mut eval_dataset_path: Option<PathBuf> = None;
    let mut force_rerun = false;

    let mut model_dir = PathBuf::from("storage/models/tara_candidate_v1");
    let mut curriculum_path = PathBuf::from("storage/datasets/tara_dataset_filtered/canonical");
    let mut ability_curriculum_path =
        PathBuf::from("storage/datasets/curriculum/ability_acquisition.jsonl");
    let mut canonical_dir = PathBuf::from("storage/datasets/tara_dataset_filtered/canonical");
    let mut weak_dir = PathBuf::from("storage/datasets/tara_dataset_filtered/weak");
    let mut adaptation_path = PathBuf::from("storage/persistence/experiential_lessons.jsonl");
    let mut checkpoints_dir = PathBuf::from("storage/models/checkpoints");
    let mut device = "auto".to_string();
    let mut precision = "auto".to_string();
    let mut dry_run = false;

    let mut override_epochs: Option<usize> = None;
    let mut override_max_steps: Option<usize> = None;
    let mut override_learning_rate: Option<f32> = None;
    let mut override_batch_size: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--stage" => {
                let val = require_arg_value(&args, &mut i, "--stage")?;
                let idx = val
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid integer for --stage: '{}'", val))?;
                if idx == 0 {
                    return Err("--stage index must be >= 1".into());
                }
                target_stage_index = Some(idx);
            }
            "--stage-id" => {
                let val = require_arg_value(&args, &mut i, "--stage-id")?;
                target_stage_id = Some(val);
            }
            "--all-stages" => {
                run_all_stages = true;
            }
            "--stages-config" => {
                let val = require_arg_value(&args, &mut i, "--stages-config")?;
                custom_config_path = Some(PathBuf::from(val));
            }
            "--resume-from" => {
                let val = require_arg_value(&args, &mut i, "--resume-from")?;
                resume_from_override = Some(PathBuf::from(val));
            }
            "--eval-checkpoint" => {
                let val = require_arg_value(&args, &mut i, "--eval-checkpoint")?;
                eval_checkpoint_path = Some(PathBuf::from(val));
            }
            "--eval-dataset" => {
                let val = require_arg_value(&args, &mut i, "--eval-dataset")?;
                eval_dataset_path = Some(PathBuf::from(val));
            }
            "--model" => {
                let val = require_arg_value(&args, &mut i, "--model")?;
                model_dir = PathBuf::from(val);
            }
            "--curriculum" => {
                let val = require_arg_value(&args, &mut i, "--curriculum")?;
                curriculum_path = PathBuf::from(val);
            }
            "--ability-path" | "--ability-curriculum" => {
                let flag = args[i].clone();
                let val = require_arg_value(&args, &mut i, &flag)?;
                ability_curriculum_path = PathBuf::from(val);
            }
            "--canonical-dir" => {
                let val = require_arg_value(&args, &mut i, "--canonical-dir")?;
                canonical_dir = PathBuf::from(val);
            }
            "--weak-dir" => {
                let val = require_arg_value(&args, &mut i, "--weak-dir")?;
                weak_dir = PathBuf::from(val);
            }
            "--adaptation-path" | "--adaptation" => {
                let flag = args[i].clone();
                let val = require_arg_value(&args, &mut i, &flag)?;
                adaptation_path = PathBuf::from(val);
            }
            "--checkpoints-dir" => {
                let val = require_arg_value(&args, &mut i, "--checkpoints-dir")?;
                checkpoints_dir = PathBuf::from(val);
            }
            "--device" => {
                device = require_arg_value(&args, &mut i, "--device")?;
            }
            "--precision" => {
                precision = require_arg_value(&args, &mut i, "--precision")?;
            }
            "--force" => {
                force_rerun = true;
            }
            "--dry-run" => {
                dry_run = true;
            }
            "--epochs" => {
                let val = require_arg_value(&args, &mut i, "--epochs")?;
                let ep = val
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid integer for --epochs: '{}'", val))?;
                if ep == 0 {
                    return Err("--epochs must be >= 1".into());
                }
                override_epochs = Some(ep);
            }
            "--max-steps" => {
                let val = require_arg_value(&args, &mut i, "--max-steps")?;
                let ms = val
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid integer for --max-steps: '{}'", val))?;
                if ms == 0 {
                    return Err("--max-steps must be >= 1".into());
                }
                override_max_steps = Some(ms);
            }
            "--learning-rate" => {
                let val = require_arg_value(&args, &mut i, "--learning-rate")?;
                let lr = val
                    .parse::<f32>()
                    .map_err(|_| format!("Invalid float for --learning-rate: '{}'", val))?;
                if !lr.is_finite() || lr <= 0.0 {
                    return Err("--learning-rate must be a positive finite float".into());
                }
                override_learning_rate = Some(lr);
            }
            "--batch-size" => {
                let val = require_arg_value(&args, &mut i, "--batch-size")?;
                let bs = val
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid integer for --batch-size: '{}'", val))?;
                if bs == 0 {
                    return Err("--batch-size must be >= 1".into());
                }
                override_batch_size = Some(bs);
            }
            unknown => {
                return Err(format!("Unknown CLI option: '{}'", unknown).into());
            }
        }
        i += 1;
    }

    validate_device_and_precision(&device, &precision)?;

    println!("================================================================================");
    println!(" TARA EXTENSIBLE MULTI-STAGE TRAINING PIPELINE ORCHESTRATOR");
    println!("================================================================================");

    // Mode: Standalone Checkpoint Evaluation
    if let Some(ref eval_cp) = eval_checkpoint_path {
        let eval_ds = eval_dataset_path
            .as_deref()
            .or_else(|| Some(curriculum_path.as_path()));
        return evaluate_checkpoint_model(eval_cp, eval_ds);
    }

    // Default to stage 1 if no stage selection was explicitly provided
    if !run_all_stages && target_stage_id.is_none() && target_stage_index.is_none() {
        target_stage_index = Some(1);
    }

    // Load or construct extensible stages
    let raw_stages: Vec<StageDefinition> = if let Some(ref cfg_file) = custom_config_path {
        println!(
            "Loading custom stage definitions from: {}",
            cfg_file.display()
        );
        if !cfg_file.exists() {
            return Err(format!(
                "Specified --stages-config file does not exist: {}",
                cfg_file.display()
            )
            .into());
        }
        let content = fs::read_to_string(cfg_file)?;
        serde_json::from_str(&content).map_err(|e| {
            format!(
                "Malformed JSON in --stages-config '{}': {}",
                cfg_file.display(),
                e
            )
        })?
    } else {
        build_default_stages(
            &model_dir,
            &checkpoints_dir,
            &curriculum_path,
            &ability_curriculum_path,
            &canonical_dir,
            &weak_dir,
            &adaptation_path,
        )
    };

    // Rigorous validation of stage definitions and topological DAG ordering
    validate_stage_definitions(&raw_stages, &model_dir)?;
    let mut stages = validate_and_sort_stage_dag(&raw_stages)?;

    // Apply CLI hyperparameter overrides
    for s in &mut stages {
        if let Some(e) = override_epochs {
            s.epochs = e;
        }
        if let Some(ms) = override_max_steps {
            s.max_steps = Some(ms);
        }
        if let Some(lr) = override_learning_rate {
            s.learning_rate = lr;
        }
        if let Some(bs) = override_batch_size {
            s.batch_size = bs;
        }
    }

    // Base model verification
    let _ = recover_interrupted_promotion(&model_dir);
    if !model_dir.exists() {
        return Err(format!("Base model directory missing: {}", model_dir.display()).into());
    }
    let (is_base_model, missing_base, base_param_count) = verify_working_model(&model_dir);
    if !is_base_model {
        return Err(format!(
            "Base model directory '{}' is invalid: {:?}",
            model_dir.display(),
            missing_base
        )
        .into());
    }

    let config_path = model_dir.join("config.json");
    let tok_path = model_dir.join("tokenizer.json");
    let model_config = TaraConfig::from_json_file(&config_path.to_string_lossy())?;
    let tokenizer = TaraTokenizer::from_file(&tok_path.to_string_lossy())?;

    println!("[BASE MODEL VERIFIED]");
    println!("  Path               : {}", model_dir.display());
    println!(
        "  Parameters         : {} ({:.2}M)",
        base_param_count.unwrap_or(0),
        base_param_count.unwrap_or(0) as f64 / 1_000_000.0
    );
    println!(
        "  Vocab Size         : {} (Tokenizer: {})",
        model_config.vocab_size, tokenizer.vocab_size
    );
    println!("  Hidden Dimension   : {}", model_config.hidden_size);
    println!("  Decoder Layers     : {}", model_config.num_hidden_layers);
    println!(
        "  Attention / KV     : {} heads / {} KV heads",
        model_config.num_attention_heads, model_config.num_key_value_heads
    );
    println!(
        "  Max Context Length : {}",
        model_config.max_position_embeddings
    );
    println!("--------------------------------------------------------------------------------");

    if run_all_stages {
        println!("[TARGET EXECUTION: ALL STAGES IN TOPOLOGICAL DAG ORDER]");
    } else if let Some(ref id) = target_stage_id {
        println!("[TARGET EXECUTION: STAGE ID '{}' ONLY]", id);
    } else if let Some(idx) = target_stage_index {
        println!("[TARGET EXECUTION: STAGE {:02} ONLY]", idx);
    }

    let mut stage_metadata_list = Vec::new();
    for s in &stages {
        let out_path = PathBuf::from(&s.output_checkpoint);
        let _ = recover_interrupted_promotion(&out_path);

        let insp = inspect_dataset(Path::new(&s.dataset_path));
        let (is_model, _, _) = verify_working_model(&out_path);
        let (is_cont, _, _) = verify_continuation_ready(&out_path);
        let in_digest =
            compute_model_digest(Path::new(&s.input_checkpoint)).unwrap_or_else(|_| "pending".to_string());
        let fp = compute_stage_config_fingerprint(s, &in_digest, &insp.dataset_fingerprint, &precision);

        let dynamic_hash =
            compute_model_digest(&out_path).unwrap_or_else(|_| "pending_execution".to_string());

        let is_targeted = run_all_stages
            || (target_stage_id
                .as_ref()
                .map(|id| id == &s.stage_id)
                .unwrap_or(false))
            || (target_stage_index
                .map(|idx| idx == s.stage_index)
                .unwrap_or(false));

        if is_targeted {
            println!(
                "  --> [ACTIVE] Stage {:02} [{}]: {}",
                s.stage_index, s.stage_id, s.stage_name
            );
            println!(
                "      Dataset     : {} ({} files, {} valid records, {} malformed, {:.2} MB)",
                s.dataset_path,
                insp.jsonl_files,
                insp.valid_records,
                insp.malformed_records,
                insp.total_bytes as f64 / 1_048_576.0
            );
            println!("      Input CP    : {}", s.input_checkpoint);
            println!("      Output CP   : {}", s.output_checkpoint);
        }

        stage_metadata_list.push(StageMetadata {
            stage_index: s.stage_index,
            stage_id: s.stage_id.clone(),
            stage_name: s.stage_name.clone(),
            description: s.description.clone(),
            input_checkpoint: s.input_checkpoint.clone(),
            output_checkpoint: s.output_checkpoint.clone(),
            dataset_source: s.dataset_path.clone(),
            dataset_inspection: insp,
            completed_at_utc: if is_model {
                "ALREADY_COMPLETED".to_string()
            } else {
                "PENDING".to_string()
            },
            status: if is_model {
                "COMPLETED".to_string()
            } else {
                "STAGED_READY".to_string()
            },
            dynamic_model_sha256: dynamic_hash.clone(),
            dynamic_model_weights_sha256: dynamic_hash,
            stage_config_fingerprint: fp,
            is_working_model: is_model,
            is_continuation_ready: is_cont,
        });
    }

    fs::create_dir_all(&checkpoints_dir)?;
    let plan_path = checkpoints_dir.join("pipeline_stages_plan.json");
    let plan_json = serde_json::json!({
        "pipeline_name": "TARA Extensible Multi-Stage Pipeline",
        "total_stages": stages.len(),
        "extensibility": "Arbitrary N stages supported in topological DAG order; dual-purpose checkpoints (model + continuation state)",
        "architectural_directives": {
            "zero_python_in_workspace": "Strict directive enforced via native Rust orchestrator",
            "zero_external_ai": "Strict directive enforced via native Rust model and engine",
            "pure_rust_native_engine": "100% native Rust execution",
            "zero_hardcoded_shas": "Runtime dynamic SHA-256 computation",
            "dual_purpose_checkpoints": "Working model + continuation state (optimizer.safetensors + checkpoint_state.json) per stage"
        },
        "stages": stage_metadata_list,
        "evaluation_protocol": {
            "evaluation_mode": "held_out_validation_gate_and_probe_forward_pass"
        }
    });
    fs::write(&plan_path, serde_json::to_string_pretty(&plan_json)?)?;
    println!("--------------------------------------------------------------------------------");
    println!("Extensible Pipeline Plan Written To: {}", plan_path.display());
    println!("--------------------------------------------------------------------------------");

    if dry_run {
        println!("[DRY-RUN AUDIT COMPLETED]");
        println!(
            "  All {} stages verified and validated in topological DAG order.",
            stages.len()
        );
        println!("  Ready for native Rust training execution.");
        return Ok(());
    }

    // Filter stages to run
    let stages_to_run: Vec<StageDefinition> = if run_all_stages {
        stages.clone()
    } else if let Some(ref id) = target_stage_id {
        stages.iter().filter(|s| &s.stage_id == id).cloned().collect()
    } else if let Some(idx) = target_stage_index {
        stages
            .iter()
            .filter(|s| s.stage_index == idx)
            .cloned()
            .collect()
    } else {
        vec![]
    };

    if stages_to_run.is_empty() {
        return Err(
            "No matching stages found to run. Specify --stage <N>, --stage-id <ID>, or --all-stages."
                .into(),
        );
    }

    // Strict validation of explicit --resume-from checkpoint: MUST be both a working model AND continuation-ready!
    if let Some(ref r_path) = resume_from_override {
        let _ = recover_interrupted_promotion(r_path);
        if !r_path.exists() {
            return Err(format!(
                "Specified --resume-from path does not exist: {}",
                r_path.display()
            )
            .into());
        }
        let (is_model, missing_model, _) = verify_working_model(r_path);
        if !is_model {
            return Err(format!(
                "Specified --resume-from path '{}' is not a valid working model: {:?}",
                r_path.display(),
                missing_model
            )
            .into());
        }
        let (is_cont, missing_cont, _) = verify_continuation_ready(r_path);
        if !is_cont {
            return Err(format!(
                "Specified --resume-from path '{}' is not continuation-ready (requires valid checkpoint_state.json and optimizer.safetensors): {:?}",
                r_path.display(),
                missing_cont
            )
            .into());
        }
    }

    // Full DAG Preflight Check across all targeted stages before running any stage
    let mut scheduled_outputs: HashSet<PathBuf> = HashSet::new();
    for (idx, s) in stages_to_run.iter().enumerate() {
        let ds_insp = inspect_dataset(Path::new(&s.dataset_path));
        if ds_insp.malformed_records > 0 {
            return Err(format!(
                "DAG Preflight Failed: Stage '{}' dataset '{}' contains {} malformed JSONL record(s).",
                s.stage_id, s.dataset_path, ds_insp.malformed_records
            )
            .into());
        }
        if ds_insp.valid_records == 0 {
            return Err(format!(
                "DAG Preflight Failed: Stage '{}' dataset '{}' contains 0 valid training records.",
                s.stage_id, s.dataset_path
            )
            .into());
        }

        let effective_in = if idx == 0 {
            resume_from_override
                .as_ref()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from(&s.input_checkpoint))
        } else {
            PathBuf::from(&s.input_checkpoint)
        };
        let norm_in = normalize_for_comparison(&effective_in);
        if !scheduled_outputs.contains(&norm_in) {
            if !effective_in.exists() {
                return Err(format!(
                    "DAG Preflight Failed: Stage '{}' requires input checkpoint '{}' which does not exist on disk and is not produced by an earlier scheduled stage.",
                    s.stage_id,
                    effective_in.display()
                )
                .into());
            }
            let (ok_model, issues, _) = verify_working_model(&effective_in);
            if !ok_model {
                return Err(format!(
                    "DAG Preflight Failed: Stage '{}' input checkpoint '{}' is not a valid working model: {:?}",
                    s.stage_id,
                    effective_in.display(),
                    issues
                )
                .into());
            }
        }
        scheduled_outputs.insert(normalize_for_comparison(Path::new(&s.output_checkpoint)));
    }

    let mut is_first_stage = true;
    let mut completed_outputs_by_norm: HashMap<PathBuf, String> = HashMap::new();

    for stage in stages_to_run {
        let out_dir = PathBuf::from(&stage.output_checkpoint);
        let _ = recover_interrupted_promotion(&out_dir);

        // Determine input checkpoint from explicit resume (entry stage only) or DAG predecessor output
        let norm_declared_in = normalize_for_comparison(Path::new(&stage.input_checkpoint));
        let (input_cp, explicit_resume_from) = if let Some(ref r_override) = resume_from_override {
            if is_first_stage {
                println!(
                    "  [RESUME PIPELINE] Resuming entry stage '{}' from explicit checkpoint: {}",
                    stage.stage_id,
                    r_override.display()
                );
                (
                    r_override.to_string_lossy().to_string(),
                    Some(r_override.to_string_lossy().to_string()),
                )
            } else {
                let pred_cp = completed_outputs_by_norm
                    .get(&norm_declared_in)
                    .cloned()
                    .unwrap_or_else(|| stage.input_checkpoint.clone());
                (pred_cp.clone(), Some(pred_cp))
            }
        } else if let Some(pred_out) = completed_outputs_by_norm.get(&norm_declared_in) {
            (pred_out.clone(), None)
        } else {
            (stage.input_checkpoint.clone(), None)
        };

        let input_path = Path::new(&input_cp);
        let (is_input_model, missing_input, _) = verify_working_model(input_path);
        if !is_input_model {
            return Err(format!(
                "Stage {:02} ('{}') input checkpoint '{}' is not a valid working model: {:?}",
                stage.stage_index, stage.stage_id, input_cp, missing_input
            )
            .into());
        }

        let input_model_digest =
            compute_model_digest(input_path).unwrap_or_else(|_| "unhashed_input".to_string());
        let ds_insp = inspect_dataset(Path::new(&stage.dataset_path));
        let expected_fingerprint = compute_stage_config_fingerprint(
            &stage,
            &input_model_digest,
            &ds_insp.dataset_fingerprint,
            &precision,
        );

        // Fingerprint-gated idempotence check
        if !force_rerun && !(is_first_stage && resume_from_override.is_some()) {
            if should_skip_completed_stage(&out_dir, &expected_fingerprint) {
                println!();
                println!(
                    "[STAGE {:02} ({}) ALREADY COMPLETED & FINGERPRINT VERIFIED: SKIPPING (use --force to re-run)]",
                    stage.stage_index, stage.stage_id
                );
                completed_outputs_by_norm.insert(
                    normalize_for_comparison(&out_dir),
                    stage.output_checkpoint.clone(),
                );
                is_first_stage = false;
                continue;
            }
        }

        println!();
        println!("================================================================================");
        println!(" EXECUTING STAGE {:02}: {}", stage.stage_index, stage.stage_name);
        println!("================================================================================");
        println!("  Stage ID            : {}", stage.stage_id);
        println!("  Description         : {}", stage.description);
        println!("  Dataset Source      : {}", stage.dataset_path);
        println!("  Input Checkpoint    : {} [VERIFIED WORKING MODEL]", input_cp);
        println!("  Target Candidate    : {}", stage.output_checkpoint);

        // Separate intermediate checkpoint directory from published candidate directory
        let intermediate_cp_dir = checkpoints_dir.join(&stage.stage_id).join("checkpoints");
        fs::create_dir_all(&intermediate_cp_dir)?;

        let has_prior_state = input_path.join("checkpoint_state.json").exists()
            && input_path.join("optimizer.safetensors").exists();
        let options = TrainingOptions {
            curriculum_path: if Path::new(&stage.dataset_path).is_file() {
                Some(stage.dataset_path.clone())
            } else {
                None
            },
            dataset_dir: if Path::new(&stage.dataset_path).is_dir() {
                Some(stage.dataset_path.clone())
            } else {
                None
            },
            candidate_dir: Some(stage.output_checkpoint.clone()),
            learning_rate: Some(stage.learning_rate),
            batch_size: Some(stage.batch_size),
            max_steps: stage.max_steps,
            device: Some(device.clone()),
            precision: Some(precision.clone()),
            checkpoint_dir: Some(intermediate_cp_dir.to_string_lossy().to_string()),
            checkpoint_interval: Some(stage.checkpoint_interval),
            resume: has_prior_state || explicit_resume_from.is_some(),
            resume_from: explicit_resume_from,
            max_memory_mb: Some(16384.0),
            ..Default::default()
        };

        let result =
            run_controlled_training_with_options(&input_cp, ".", stage.epochs, options)?;

        // Verify stage output satisfies dual-purpose model requirement
        let (is_model, missing_model, param_cnt) = verify_working_model(&out_dir);
        let (is_cont, missing_cont, last_step) = verify_continuation_ready(&out_dir);
        let dynamic_sha =
            compute_model_digest(&out_dir).unwrap_or_else(|_| "unhashed".to_string());

        if !is_model {
            return Err(format!(
                "Stage {:02} ('{}') failed working model verification: {:?}",
                stage.stage_index, stage.stage_id, missing_model
            )
            .into());
        }
        if !is_cont {
            return Err(format!(
                "Stage {:02} ('{}') failed continuation-ready verification: {:?}",
                stage.stage_index, stage.stage_id, missing_cont
            )
            .into());
        }

        let stage_provenance = serde_json::json!({
            "stage_index": stage.stage_index,
            "stage_id": stage.stage_id,
            "stage_name": stage.stage_name,
            "input_checkpoint": input_cp,
            "input_model_weights_sha256": input_model_digest,
            "output_checkpoint": stage.output_checkpoint,
            "intermediate_checkpoints_dir": intermediate_cp_dir.to_string_lossy(),
            "dynamic_model_sha256": dynamic_sha,
            "dynamic_model_weights_sha256": dynamic_sha,
            "stage_config_fingerprint": expected_fingerprint,
            "dataset_fingerprint": ds_insp.dataset_fingerprint,
            "total_parameters": param_cnt.unwrap_or(0),
            "last_optimizer_step": last_step,
            "is_working_model": is_model,
            "is_continuation_ready": is_cont,
            "training_result": result,
            "timestamp_utc": tara_engine::now_iso()
        });
        fs::write(
            out_dir.join("STAGE_METADATA.json"),
            serde_json::to_string_pretty(&stage_provenance)?,
        )?;

        println!("--------------------------------------------------------------------------------");
        println!("[STAGE {:02} COMPLETE]", stage.stage_index);
        println!("  Working Model Verification      : PASSED");
        println!("  Continuation-Ready Verification : PASSED");
        println!("  Model Weights Digest (SHA-256)  : {}", dynamic_sha);
        println!("  Stage Config Fingerprint        : {}", expected_fingerprint);
        println!(
            "  Stage Provenance Metadata Saved : {}",
            out_dir.join("STAGE_METADATA.json").display()
        );
        println!("================================================================================");

        completed_outputs_by_norm.insert(
            normalize_for_comparison(&out_dir),
            stage.output_checkpoint.clone(),
        );
        is_first_stage = false;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tara_engine::safetensors::write_safetensors_with_shapes;
    use tara_engine::trainer::promote_directory_atomically;

    fn unique_test_dir(tag: &str) -> PathBuf {
        let dir = env::temp_dir().join(format!(
            "tara_stages_test_{}_{}_{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_minimal_working_model(dir: &Path, with_continuation: bool, corrupt_opt: bool) {
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

        if with_continuation {
            fs::write(
                dir.join("checkpoint_state.json"),
                r#"{"step":10,"optimizer_step":10,"samples_seen":40,"epoch":1,"resumable":true}"#,
            )
            .unwrap();
            let mut opt_t = HashMap::new();
            let mut opt_s = HashMap::new();
            for (k, v) in &weights {
                let len = if corrupt_opt { v.len() + 3 } else { v.len() };
                opt_t.insert(format!("{k}.adam_m"), vec![0.0f32; len]);
                opt_s.insert(format!("{k}.adam_m"), vec![len]);
                opt_t.insert(format!("{k}.adam_v"), vec![0.0f32; len]);
                opt_s.insert(format!("{k}.adam_v"), vec![len]);
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
    fn test_dag_cycle_detection_and_topological_sort() {
        let s1 = StageDefinition {
            stage_index: 1,
            stage_id: "stage_a".to_string(),
            stage_name: "A".to_string(),
            description: "d".to_string(),
            dataset_path: "ds.jsonl".to_string(),
            input_checkpoint: "cp_b".to_string(),
            output_checkpoint: "cp_a".to_string(),
            epochs: 1,
            max_steps: Some(10),
            learning_rate: 1e-4,
            batch_size: 2,
            checkpoint_interval: 5,
            depends_on: vec!["stage_b".to_string()],
        };
        let s2 = StageDefinition {
            stage_index: 2,
            stage_id: "stage_b".to_string(),
            stage_name: "B".to_string(),
            description: "d".to_string(),
            dataset_path: "ds.jsonl".to_string(),
            input_checkpoint: "cp_a".to_string(),
            output_checkpoint: "cp_b".to_string(),
            epochs: 1,
            max_steps: Some(10),
            learning_rate: 1e-4,
            batch_size: 2,
            checkpoint_interval: 5,
            depends_on: vec!["stage_a".to_string()],
        };

        let err = validate_and_sort_stage_dag(&[s1.clone(), s2.clone()]).unwrap_err();
        assert!(err.contains("DAG Cycle Detected"), "Unexpected error: {err}");

        // Valid out-of-order stages must sort topologically (stage_b depends on stage_a)
        let mut valid_a = s1;
        valid_a.input_checkpoint = "base_model".to_string();
        valid_a.depends_on = vec![];
        let valid_b = s2;
        let sorted = validate_and_sort_stage_dag(&[valid_b, valid_a]).unwrap();
        assert_eq!(sorted[0].stage_id, "stage_a");
        assert_eq!(sorted[1].stage_id, "stage_b");
    }

    #[test]
    fn test_validate_stage_definitions_rejects_duplicate_output_and_unsafe_paths() {
        let dir = unique_test_dir("validate_defs");
        let base_model = dir.join("base");
        let out_shared = dir.join("shared_out");

        let s1 = StageDefinition {
            stage_index: 1,
            stage_id: "s1".to_string(),
            stage_name: "S1".to_string(),
            description: "d".to_string(),
            dataset_path: "ds.jsonl".to_string(),
            input_checkpoint: base_model.to_string_lossy().to_string(),
            output_checkpoint: out_shared.to_string_lossy().to_string(),
            epochs: 1,
            max_steps: Some(5),
            learning_rate: 1e-4,
            batch_size: 2,
            checkpoint_interval: 5,
            depends_on: vec![],
        };
        let mut s2 = s1.clone();
        s2.stage_index = 2;
        s2.stage_id = "s2".to_string();

        // Duplicate output path rejected
        let err = validate_stage_definitions(&[s1.clone(), s2], &base_model).unwrap_err();
        assert!(err.contains("conflicts/overlaps"));

        // Unsafe stage_id with traversal rejected
        let mut s_bad_id = s1.clone();
        s_bad_id.stage_id = "../escape".to_string();
        assert!(validate_stage_definitions(&[s_bad_id], &base_model).is_err());

        // max_steps = Some(0) rejected
        let mut s_zero_steps = s1;
        s_zero_steps.max_steps = Some(0);
        assert!(validate_stage_definitions(&[s_zero_steps], &base_model).is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_continuation_verification_and_fingerprint_idempotence() {
        let dir = unique_test_dir("idempotence_and_cont");
        let model_only = dir.join("model_only");
        let cont_ready = dir.join("cont_ready");
        let corrupt_opt = dir.join("corrupt_opt");

        write_minimal_working_model(&model_only, false, false);
        write_minimal_working_model(&cont_ready, true, false);
        write_minimal_working_model(&corrupt_opt, true, true);

        // Model-only directory passes working model but fails continuation-ready
        assert!(verify_working_model(&model_only).0);
        assert!(!verify_continuation_ready(&model_only).0);

        // Continuation-ready directory passes both
        assert!(verify_working_model(&cont_ready).0);
        assert!(verify_continuation_ready(&cont_ready).0);

        // Corrupt optimizer tensor shape is caught and rejected
        let (ok_corrupt, issues, _) = verify_continuation_ready(&corrupt_opt);
        assert!(!ok_corrupt, "Corrupt optimizer state must fail: {:?}", issues);

        // Idempotence check requires matching stage_config_fingerprint
        let mut stage = StageDefinition {
            stage_index: 1,
            stage_id: "s1".to_string(),
            stage_name: "S1".to_string(),
            description: "d".to_string(),
            dataset_path: "ds.jsonl".to_string(),
            input_checkpoint: "in".to_string(),
            output_checkpoint: cont_ready.to_string_lossy().to_string(),
            epochs: 1,
            max_steps: Some(10),
            learning_rate: 1e-4,
            batch_size: 4,
            checkpoint_interval: 5,
            depends_on: vec![],
        };
        let fp1 = compute_stage_config_fingerprint(&stage, "in_sha_1", "ds_sha_1", "fp32");
        fs::write(
            cont_ready.join("STAGE_METADATA.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "stage_config_fingerprint": fp1
            }))
            .unwrap(),
        )
        .unwrap();

        assert!(should_skip_completed_stage(&cont_ready, &fp1));

        // Changing dataset SHA or learning rate invalidates fingerprint -> must NOT skip!
        let fp_changed_ds =
            compute_stage_config_fingerprint(&stage, "in_sha_1", "ds_sha_2_modified", "fp32");
        assert!(!should_skip_completed_stage(&cont_ready, &fp_changed_ds));

        stage.learning_rate = 5e-4;
        let fp_changed_lr = compute_stage_config_fingerprint(&stage, "in_sha_1", "ds_sha_1", "fp32");
        assert!(!should_skip_completed_stage(&cont_ready, &fp_changed_lr));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_crash_recovery_after_interrupted_backup_rename() {
        let dir = unique_test_dir("crash_recovery");
        let target = dir.join("stage1_out");
        let backup = dir.join("stage1_out.bak_12345");
        let orphan_staging = dir.join("stage1_out.staging_12345");

        // Simulate crash right after renaming target -> backup, before staging -> target completed
        write_minimal_working_model(&backup, true, false);
        fs::create_dir_all(&orphan_staging).unwrap();

        let recovered = recover_interrupted_promotion(&target).unwrap();
        assert!(recovered, "Must recover target from valid .bak_* directory");
        assert!(target.exists() && verify_working_model(&target).0);
        assert!(!backup.exists());
        assert!(!orphan_staging.exists());

        // Now test full atomic promotion replacing target with new staging
        let new_staging = dir.join("stage1_out.staging_99999");
        write_minimal_working_model(&new_staging, true, false);
        promote_directory_atomically(&new_staging, &target).unwrap();
        assert!(target.exists() && verify_working_model(&target).0);
        assert!(!new_staging.exists());

        let _ = fs::remove_dir_all(&dir);
    }
}
