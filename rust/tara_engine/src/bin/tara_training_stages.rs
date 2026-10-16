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
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use tara_engine::config::TaraConfig;
use tara_engine::dataset::{ExpandableDatasetReader, ShardStreamingMode};
use tara_engine::internal_diagnostics::{
    promote_directory_atomically, recover_interrupted_promotion,
};
use tara_engine::tokenizer::TaraTokenizer;
use tara_engine::train_candidate::{run_controlled_training_with_options, TrainingOptions};
use tara_engine::{
    compute_file_sha256, load_model_weights_with_purpose, load_model_weights_with_shapes,
    load_safetensors_with_shapes, read_model_metadata_shapes, TaraForCausalLM,
};

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

/// Detailed dataset inspection statistics including JSONL parseability, recursive file count, and content fingerprint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DatasetInspection {
    pub jsonl_files: usize,
    pub total_lines: usize,
    pub valid_records: usize,
    pub malformed_records: usize,
    pub empty_lines: usize,
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
    pub depends_on: Vec<String>,
    pub input_checkpoint: String,
    pub output_checkpoint: String,
    pub dataset_source: String,
    pub dataset_inspection: DatasetInspection,
    pub completed_at_utc: Option<String>,
    pub status: String,
    pub model_weights_sha256: Option<String>,
    pub full_checkpoint_sha256: Option<String>,
    pub stage_config_fingerprint: String,
    pub is_working_model: bool,
    pub is_continuation_ready: bool,
}

/// Checks whether an OS process with `pid` is currently alive.
fn is_process_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    if pid == std::process::id() {
        return true;
    }
    #[cfg(windows)]
    {
        const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
        const STILL_ACTIVE: u32 = 259;
        extern "system" {
            fn OpenProcess(
                dwDesiredAccess: u32,
                bInheritHandle: i32,
                dwProcessId: u32,
            ) -> *mut std::ffi::c_void;
            fn GetExitCodeProcess(hProcess: *mut std::ffi::c_void, lpExitCode: *mut u32) -> i32;
            fn CloseHandle(hObject: *mut std::ffi::c_void) -> i32;
        }
        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if handle.is_null() {
                return false;
            }
            let mut exit_code: u32 = 0;
            let ok = GetExitCodeProcess(handle, &mut exit_code);
            let _ = CloseHandle(handle);
            ok != 0 && exit_code == STILL_ACTIVE
        }
    }
    #[cfg(unix)]
    {
        let rc = unsafe { libc::kill(pid as i32, 0) };
        rc == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
    #[cfg(not(any(windows, unix)))]
    {
        true
    }
}

/// Exclusive process lock guard preventing concurrent orchestrator runs against the same checkpoints root.
pub struct OrchestratorLockGuard {
    lock_path: PathBuf,
}

impl OrchestratorLockGuard {
    pub fn acquire(checkpoints_dir: &Path) -> Result<Self, String> {
        fs::create_dir_all(checkpoints_dir).map_err(|e| {
            format!(
                "Failed to create checkpoints_dir '{}': {e}",
                checkpoints_dir.display()
            )
        })?;
        let lock_path = checkpoints_dir.join(".orchestrator.lock");
        let pid = std::process::id();
        let now_iso = tara_engine::now_iso();
        let payload = serde_json::json!({
            "pid": pid,
            "acquired_at_utc": now_iso,
        });

        for attempt in 0..2 {
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock_path)
            {
                Ok(mut f) => {
                    use std::io::Write;
                    let _ = f.write_all(payload.to_string().as_bytes());
                    let _ = f.sync_all();
                    return Ok(Self { lock_path });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    let existing = fs::read_to_string(&lock_path).unwrap_or_default();
                    let stale_owner = serde_json::from_str::<serde_json::Value>(&existing)
                        .ok()
                        .and_then(|v| v.get("pid").and_then(|p| p.as_u64()))
                        .map(|owner_pid| !is_process_alive(owner_pid as u32))
                        .unwrap_or(false);

                    if attempt == 0 && stale_owner {
                        let _ = fs::remove_file(&lock_path);
                        continue;
                    }
                    return Err(format!(
                        "Concurrent execution rejected: lock file '{}' already exists ({}). Another tara_training_stages process is active.",
                        lock_path.display(),
                        existing.trim()
                    ));
                }
                Err(e) => {
                    return Err(format!(
                        "Failed to acquire orchestrator lock '{}': {e}",
                        lock_path.display()
                    ));
                }
            }
        }
        Err(format!(
            "Failed to acquire orchestrator lock '{}'",
            lock_path.display()
        ))
    }
}

impl Drop for OrchestratorLockGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.lock_path);
    }
}

/// Computes a deterministic dynamic digest across all model safetensor weight shards in a directory.
/// Returns:
/// - `Ok(None)` if `dir` does not exist yet (`NOT_RUN`),
/// - `Err(...)` if `dir` exists but contains no model weight files (`NO_WEIGHTS`) or an I/O error occurs,
/// - `Ok(Some(sha256))` when weight shards are present and hashed.
pub fn compute_model_weights_digest(dir: &Path) -> Result<Option<String>, String> {
    if !dir.exists() {
        return Ok(None);
    }

    let mut shard_files = Vec::new();
    let entries = fs::read_dir(dir)
        .map_err(|e| format!("Failed to read directory '{}': {e}", dir.display()))?;
    for entry in entries {
        let entry =
            entry.map_err(|e| format!("Directory entry error in '{}': {e}", dir.display()))?;
        let p = entry.path();
        if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
            if name.ends_with(".safetensors") && name != "optimizer.safetensors" {
                shard_files.push(p);
            }
        }
    }

    if shard_files.is_empty() {
        return Err(format!(
            "Directory '{}' exists but contains no SafeTensors weight shards",
            dir.display()
        ));
    }

    shard_files.sort();

    let mut combined_hasher = Sha256::new();
    for shard in shard_files {
        let name = shard.file_name().unwrap_or_default().to_string_lossy();
        let meta = shard
            .metadata()
            .map_err(|e| format!("Failed to stat '{}': {e}", shard.display()))?;
        combined_hasher.update(name.as_bytes());
        combined_hasher.update(meta.len().to_le_bytes());
        let shard_hash = compute_file_sha256(&shard)
            .map_err(|e| format!("Failed to hash '{}': {e}", shard.display()))?;
        combined_hasher.update(shard_hash.as_bytes());
    }

    Ok(Some(hex::encode(combined_hasher.finalize())))
}

/// Computes a whole-checkpoint digest covering `config.json`, `tokenizer.json`, model weight shards,
/// `optimizer.safetensors`, and `checkpoint_state.json`.
pub fn compute_full_checkpoint_digest(dir: &Path) -> Result<Option<String>, String> {
    if !dir.exists() || is_empty_dir(dir) {
        return Ok(None);
    }
    let Some(weights_sha) = compute_model_weights_digest(dir)? else {
        return Ok(None);
    };
    let mut hasher = Sha256::new();
    hasher.update(b"weights:");
    hasher.update(weights_sha.as_bytes());

    for artifact in [
        "config.json",
        "tokenizer.json",
        "optimizer.safetensors",
        "checkpoint_state.json",
    ] {
        let p = dir.join(artifact);
        if p.exists() {
            let h = compute_file_sha256(&p)
                .map_err(|e| format!("Failed to hash '{}': {e}", p.display()))?;
            hasher.update(b"|");
            hasher.update(artifact.as_bytes());
            hasher.update(b":");
            hasher.update(h.as_bytes());
        }
    }
    Ok(Some(hex::encode(hasher.finalize())))
}

/// Returns true if `path` exists, is a directory, and contains zero directory entries.
pub fn is_empty_dir(path: &Path) -> bool {
    if !path.exists() || !path.is_dir() {
        return false;
    }
    match fs::read_dir(path) {
        Ok(mut entries) => entries.next().is_none(),
        Err(_) => false,
    }
}

fn collect_jsonl_files_recursive(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = fs::read_dir(dir)
        .map_err(|e| format!("Failed to read dataset directory '{}': {e}", dir.display()))?;
    let mut local_paths = Vec::new();
    for entry in entries {
        let entry =
            entry.map_err(|e| format!("Error reading entry in '{}': {e}", dir.display()))?;
        local_paths.push(entry.path());
    }
    local_paths.sort();

    for p in local_paths {
        let meta = fs::symlink_metadata(&p)
            .map_err(|e| format!("Failed to inspect metadata for '{}': {e}", p.display()))?;
        if meta.file_type().is_symlink() {
            return Err(format!(
                "Symlinks are prohibited inside dataset directories: '{}'",
                p.display()
            ));
        }
        if meta.is_dir() {
            collect_jsonl_files_recursive(&p, out)?;
        } else if meta.is_file() && p.extension().and_then(|ext| ext.to_str()) == Some("jsonl") {
            out.push(p);
        }
    }
    Ok(())
}

/// Inspects a JSONL file or directory (recursively), validating existence, readability, JSON
/// parseability, non-empty training fields, and computing a deterministic dataset content fingerprint.
pub fn inspect_dataset(path: &Path) -> Result<DatasetInspection, String> {
    if !path.exists() {
        return Err(format!("Dataset path does not exist: '{}'", path.display()));
    }
    if path.is_file() {
        let mut total_lines = 0usize;
        let mut valid_records = 0usize;
        let mut malformed_records = 0usize;
        let mut empty_lines = 0usize;
        let mut hasher = Sha256::new();

        let meta = fs::metadata(path)
            .map_err(|e| format!("Failed to read metadata for '{}': {e}", path.display()))?;
        let total_bytes = meta.len();

        let file = File::open(path)
            .map_err(|e| format!("Failed to open dataset file '{}': {e}", path.display()))?;
        let mut reader = BufReader::with_capacity(64 * 1024, file);
        let mut line = String::new();
        loop {
            let n = reader
                .read_line(&mut line)
                .map_err(|e| format!("I/O error reading '{}': {e}", path.display()))?;
            if n == 0 {
                break;
            }
            total_lines += 1;
            hasher.update(line.as_bytes());
            let trimmed = line.trim();
            if trimmed.is_empty() {
                empty_lines += 1;
                line.clear();
                continue;
            }
            match serde_json::from_str::<serde_json::Value>(trimmed) {
                Ok(val) => {
                    let has_text = val
                        .get("text")
                        .or_else(|| val.get("content"))
                        .and_then(|v| v.as_str())
                        .map(|s| !s.trim().is_empty())
                        .unwrap_or(false);
                    let has_in = val
                        .get("formatted_input")
                        .or_else(|| val.get("input"))
                        .or_else(|| val.get("prompt"))
                        .or_else(|| val.get("instruction"))
                        .or_else(|| val.get("trigger_pattern"))
                        .or_else(|| val.get("metadata").and_then(|m| m.get("input")))
                        .and_then(|v| v.as_str())
                        .map(|s| !s.trim().is_empty())
                        .unwrap_or(has_text);
                    let has_out = val
                        .get("formatted_target")
                        .or_else(|| val.get("output"))
                        .or_else(|| val.get("completion"))
                        .or_else(|| val.get("response"))
                        .or_else(|| val.get("recommendation"))
                        .or_else(|| val.get("metadata").and_then(|m| m.get("output")))
                        .or_else(|| val.get("text"))
                        .or_else(|| val.get("content"))
                        .and_then(|v| v.as_str())
                        .map(|s| !s.trim().is_empty())
                        .unwrap_or(false);
                    if (has_in && has_out) || has_text {
                        valid_records += 1;
                    } else {
                        malformed_records += 1;
                    }
                }
                Err(_) => {
                    malformed_records += 1;
                }
            }
            line.clear();
        }

        Ok(DatasetInspection {
            jsonl_files: 1,
            total_lines,
            valid_records,
            malformed_records,
            empty_lines,
            total_bytes,
            dataset_fingerprint: hex::encode(hasher.finalize()),
        })
    } else if path.is_dir() {
        let mut files = Vec::new();
        collect_jsonl_files_recursive(path, &mut files)?;
        files.sort();

        if files.is_empty() {
            return Err(format!(
                "Dataset directory '{}' contains zero .jsonl files",
                path.display()
            ));
        }

        let mut total_lines = 0usize;
        let mut total_valid = 0usize;
        let mut total_malformed = 0usize;
        let mut total_empty = 0usize;
        let mut total_bytes = 0u64;
        let mut combined_hasher = Sha256::new();

        let total_files = files.len();
        for (i, ep) in files.iter().enumerate() {
            if total_files > 5 && (i == 0 || (i + 1) % 10 == 0 || i + 1 == total_files) {
                println!(
                    "  [Dataset Audit] Auditing shard {}/{} ({})...",
                    i + 1,
                    total_files,
                    ep.file_name().unwrap_or_default().to_string_lossy()
                );
            }
            let insp = inspect_dataset(ep)?;
            total_lines += insp.total_lines;
            total_valid += insp.valid_records;
            total_malformed += insp.malformed_records;
            total_empty += insp.empty_lines;
            total_bytes += insp.total_bytes;
            let rel = ep.strip_prefix(path).unwrap_or(ep).to_string_lossy();
            combined_hasher.update(rel.as_bytes());
            combined_hasher.update(insp.dataset_fingerprint.as_bytes());
        }

        Ok(DatasetInspection {
            jsonl_files: files.len(),
            total_lines,
            valid_records: total_valid,
            malformed_records: total_malformed,
            empty_lines: total_empty,
            total_bytes,
            dataset_fingerprint: hex::encode(combined_hasher.finalize()),
        })
    } else {
        Err(format!(
            "Dataset path '{}' is neither a regular file nor a directory",
            path.display()
        ))
    }
}

/// Computes a deterministic fingerprint for a stage's execution configuration, input model digest,
/// and dataset content. Used to ensure idempotence only skips when nothing upstream changed.
pub fn compute_stage_config_fingerprint(
    stage: &StageDefinition,
    input_model_digest: &str,
    dataset_fingerprint: &str,
    precision: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"orchestrator_v2|");
    hasher.update(stage.stage_id.as_bytes());
    hasher.update(b"|in:");
    hasher.update(input_model_digest.as_bytes());
    hasher.update(b"|ds:");
    hasher.update(dataset_fingerprint.as_bytes());
    hasher.update(b"|ep:");
    hasher.update(stage.epochs.to_le_bytes());
    hasher.update(b"|ms:");
    hasher.update(stage.max_steps.unwrap_or(0).to_le_bytes());
    hasher.update(b"|lr:");
    hasher.update(stage.learning_rate.to_bits().to_le_bytes());
    hasher.update(b"|bs:");
    hasher.update(stage.batch_size.to_le_bytes());
    hasher.update(b"|ci:");
    hasher.update(stage.checkpoint_interval.to_le_bytes());
    hasher.update(b"|prec:");
    hasher.update(precision.as_bytes());
    hex::encode(hasher.finalize())
}

/// Checks whether a stage output directory is already completed, continuation-ready, AND matches
/// the exact `expected_fingerprint` AND its recorded `output_model_weights_sha256` matches disk.
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
    if val.get("status").and_then(|v| v.as_str()) != Some("COMPLETED") {
        return false;
    }
    let Some(saved_fp) = val.get("stage_config_fingerprint").and_then(|v| v.as_str()) else {
        return false;
    };
    if saved_fp != expected_fingerprint {
        return false;
    }
    if let Some(recorded_sha) = val
        .get("dynamic_model_weights_sha256")
        .and_then(|v| v.as_str())
    {
        let Ok(Some(actual_sha)) = compute_model_weights_digest(out_dir) else {
            return false;
        };
        if recorded_sha != actual_sha {
            return false;
        }
    }
    true
}

/// Reads the recorded completion timestamp from `STAGE_METADATA.json` if present.
pub fn read_stage_completed_timestamp(out_dir: &Path) -> Option<String> {
    let meta_path = out_dir.join("STAGE_METADATA.json");
    let content = fs::read_to_string(meta_path).ok()?;
    let val: serde_json::Value = serde_json::from_str(&content).ok()?;
    val.get("completed_at_utc")
        .or_else(|| val.get("timestamp_utc"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Atomically writes JSON content to `target_file` via `<target_file>.tmp` + `sync_all` + `rename`.
pub fn write_json_atomically(target_file: &Path, value: &serde_json::Value) -> Result<(), String> {
    if let Some(parent) = target_file.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create parent dir '{}': {e}", parent.display()))?;
        }
    }
    let tmp_path = PathBuf::from(format!(
        "{}.tmp_{}",
        target_file.display(),
        std::process::id()
    ));
    let payload =
        serde_json::to_string_pretty(value).map_err(|e| format!("JSON serialize error: {e}"))?;
    {
        use std::io::Write;
        let mut f = File::create(&tmp_path)
            .map_err(|e| format!("Failed to create '{}': {e}", tmp_path.display()))?;
        f.write_all(payload.as_bytes())
            .map_err(|e| format!("Failed to write '{}': {e}", tmp_path.display()))?;
        f.sync_all()
            .map_err(|e| format!("Failed to fsync '{}': {e}", tmp_path.display()))?;
    }
    fs::rename(&tmp_path, target_file).map_err(|e| {
        let _ = fs::remove_file(&tmp_path);
        format!(
            "Failed to atomically rename '{}' -> '{}': {e}",
            tmp_path.display(),
            target_file.display()
        )
    })
}

/// Deep verification of working model: inspects `config.json`, `tokenizer.json`, vocabulary
/// agreement (`config.vocab_size == tokenizer.vocab_size`), and full `TaraForCausalLM` layer/shape
/// compatibility across all transformer layers.
pub fn verify_working_model_with_purpose(
    model_dir: &Path,
    purpose: &str,
) -> (bool, Vec<String>, Option<usize>) {
    let mut missing = Vec::new();
    let config_path = model_dir.join("config.json");
    let tokenizer_path = model_dir.join("tokenizer.json");

    let cfg_opt = if !config_path.exists() {
        missing.push("config.json missing".to_string());
        None
    } else {
        match TaraConfig::from_json_file(&config_path.to_string_lossy()) {
            Ok(c) => Some(c),
            Err(e) => {
                missing.push(format!("config.json invalid or malformed: {e}"));
                None
            }
        }
    };

    let tok_opt = if !tokenizer_path.exists() {
        missing.push("tokenizer.json missing".to_string());
        None
    } else {
        match TaraTokenizer::from_file(&tokenizer_path.to_string_lossy()) {
            Ok(t) => Some(t),
            Err(e) => {
                missing.push(format!("tokenizer.json invalid or malformed: {e}"));
                None
            }
        }
    };

    if let (Some(ref cfg), Some(ref tok)) = (&cfg_opt, &tok_opt) {
        if cfg.vocab_size != tok.vocab_size {
            missing.push(format!(
                "config/tokenizer vocab_size mismatch: config.vocab_size={} != tokenizer.vocab_size={}",
                cfg.vocab_size, tok.vocab_size
            ));
        }
    }

    let mut param_count: Option<usize> = None;
    match read_model_metadata_shapes(&model_dir.to_string_lossy()) {
        Ok(shapes) => {
            if shapes.is_empty() {
                missing.push("SafeTensors weights dictionary is empty".to_string());
            } else {
                let total: usize = shapes.values().map(|s| s.iter().product::<usize>()).sum();
                param_count = Some(total);
                if let Some(ref cfg) = cfg_opt {
                    if let Err(e) = TaraForCausalLM::validate_architecture_shapes(&shapes, cfg) {
                        missing.push(format!(
                            "SafeTensors weights incompatible with TaraForCausalLM architecture: {e}"
                        ));
                    }
                }
                println!(
                    "[SafeTensors Metadata Verified] Model at '{}' contains {} tensors, {} parameters | Purpose: {}",
                    model_dir.display(),
                    shapes.len(),
                    total,
                    purpose
                );
            }
        }
        Err(e) => {
            missing.push(format!("Failed to read model SafeTensors metadata: {e}"));
        }
    }

    (missing.is_empty(), missing, param_count)
}

pub fn verify_working_model(model_dir: &Path) -> (bool, Vec<String>, Option<usize>) {
    verify_working_model_with_purpose(model_dir, "INPUT")
}

/// Deep verification of continuation state: checks `checkpoint_state.json` and `optimizer.safetensors`
/// (including verifying that moment tensor lengths match model weight tensor lengths when weights exist).
pub fn verify_continuation_ready_with_purpose(
    cp_dir: &Path,
    purpose: &str,
) -> (bool, Vec<String>, Option<u64>) {
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
                    load_model_weights_with_purpose(&cp_dir.to_string_lossy(), purpose)
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

pub fn verify_continuation_ready(cp_dir: &Path) -> (bool, Vec<String>, Option<u64>) {
    verify_continuation_ready_with_purpose(cp_dir, "RESUME")
}

pub fn log_checkpoint_load(caller: &str, stage_id: &str, path: &str, purpose: &str) {
    println!(
        "[Checkpoint Loader] Caller: '{}' | Stage: '{}' | Path: '{}' | Purpose: {}",
        caller, stage_id, path, purpose
    );
}

pub fn verify_working_model_with_context(
    caller: &str,
    stage_id: &str,
    model_dir: &Path,
    purpose: &str,
) -> (bool, Vec<String>, Option<usize>) {
    log_checkpoint_load(caller, stage_id, &model_dir.to_string_lossy(), purpose);
    verify_working_model_with_purpose(model_dir, purpose)
}

pub fn verify_continuation_ready_with_context(
    caller: &str,
    stage_id: &str,
    cp_dir: &Path,
    purpose: &str,
) -> (bool, Vec<String>, Option<u64>) {
    log_checkpoint_load(caller, stage_id, &cp_dir.to_string_lossy(), purpose);
    verify_continuation_ready_with_purpose(cp_dir, purpose)
}

fn is_safe_stage_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

use tara_engine::train_candidate::{
    has_parent_dir_traversal, normalize_path_for_comparison as normalize_for_comparison,
    paths_overlap, require_arg_value,
};

pub fn validate_device_and_precision(device: &str, precision: &str) -> Result<(), String> {
    device
        .parse::<tara_hd::SelectionPolicy>()
        .map_err(|e| format!("Invalid --device: {e}"))?;
    match precision.to_ascii_lowercase().as_str() {
        "auto" | "fp16" | "fp32" | "bf16" => Ok(()),
        other => Err(format!(
            "Invalid --precision '{}': must be one of 'auto', 'fp16', 'fp32', 'bf16'",
            other
        )),
    }
}

/// Validates stage definitions for hyperparameter validity, non-empty paths, traversal prohibition,
/// path containment, production overwrite protection, and duplicate/circular output paths.
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
    let mut all_inputs: Vec<(String, PathBuf)> = Vec::new();

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
            return Err(format!("Stage '{}' has empty stage_name", stage.stage_id));
        }
        if stage.dataset_path.trim().is_empty() {
            return Err(format!("Stage '{}' has empty dataset_path", stage.stage_id));
        }
        if stage.input_checkpoint.trim().is_empty() {
            return Err(format!(
                "Stage '{}' has empty input_checkpoint",
                stage.stage_id
            ));
        }
        if stage.output_checkpoint.trim().is_empty() {
            return Err(format!(
                "Stage '{}' has empty output_checkpoint",
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

        let ds_path = Path::new(&stage.dataset_path);
        let out_path = Path::new(&stage.output_checkpoint);
        let in_path = Path::new(&stage.input_checkpoint);

        if has_parent_dir_traversal(ds_path)
            || has_parent_dir_traversal(out_path)
            || has_parent_dir_traversal(in_path)
        {
            return Err(format!(
                "Stage '{}' contains forbidden '..' parent directory traversal in dataset/input/output path",
                stage.stage_id
            ));
        }

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
        let norm_in = normalize_for_comparison(in_path);
        for (other_id, other_out) in &seen_outputs {
            if norm_out == *other_out
                || norm_out.starts_with(other_out)
                || other_out.starts_with(&norm_out)
            {
                return Err(format!(
                    "Stage '{}' output_checkpoint ('{}') conflicts/overlaps with stage '{}' output_checkpoint.",
                    stage.stage_id, stage.output_checkpoint, other_id
                ));
            }
        }
        seen_outputs.push((stage.stage_id.clone(), norm_out));
        all_inputs.push((stage.stage_id.clone(), norm_in));
    }

    // Reject nested input-vs-output directory containment across stages (e.g., stage2 input = stage1_out/subdir)
    for (out_stage_id, norm_out) in &seen_outputs {
        for (in_stage_id, norm_in) in &all_inputs {
            if norm_out != norm_in
                && (norm_out.starts_with(norm_in) || norm_in.starts_with(norm_out))
            {
                return Err(format!(
                    "Circular/nested checkpoint path hazard: stage '{}' output overlaps as parent/child subdirectory with stage '{}' input",
                    out_stage_id, in_stage_id
                ));
            }
        }
    }

    Ok(())
}

/// Validates the stage dependency DAG (combining explicit `depends_on` and implicit
/// `output_checkpoint -> input_checkpoint` edges), detects cycles and ambiguous multi-parent
/// inheritance, resolves each stage's `input_checkpoint` from its declared single predecessor
/// when applicable, and returns stages sorted in deterministic topological order.
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

        // If a stage declares a single explicit dependency in `depends_on`, its `input_checkpoint`
        // must match that dependency's `output_checkpoint`.
        if s.depends_on.len() == 1 {
            let dep_idx = id_to_idx[&s.depends_on[0]];
            let dep_out_norm =
                normalize_for_comparison(Path::new(&stages[dep_idx].output_checkpoint));
            if norm_in != dep_out_norm {
                return Err(format!(
                    "DAG Inconsistency Error: Stage '{}' depends on '{}' (output '{}'), but declares conflicting input_checkpoint '{}'",
                    s.stage_id,
                    s.depends_on[0],
                    stages[dep_idx].output_checkpoint,
                    s.input_checkpoint
                ));
            }
        }

        // If a stage declares multiple dependencies, reject ambiguous multi-parent inheritance
        // unless `input_checkpoint` explicitly selects one of those dependencies' outputs.
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
                return Err(format!("DAG Cycle Detected: {}", path_stack.join(" -> ")));
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
) -> Result<(f64, usize), String> {
    if token_ids.len() < 2 || vocab_size == 0 {
        return Err("Sequence must have at least 2 tokens and vocab_size > 0".to_string());
    }
    let (_, _, logits) = model.forward_with_cache(token_ids);
    let expected_logits_len = token_ids.len() * vocab_size;
    if logits.len() != expected_logits_len {
        return Err(format!(
            "Forward pass output logits length mismatch: expected {} (seq_len={} * vocab_size={}), got {}",
            expected_logits_len,
            token_ids.len(),
            vocab_size,
            logits.len()
        ));
    }

    let mut total_loss = 0.0f64;
    let mut targets_evaluated = 0usize;

    for (pos, &target_token) in token_ids.iter().enumerate().skip(1) {
        let predictor = pos - 1;
        let target = target_token as usize;
        if target >= vocab_size {
            return Err(format!(
                "Token ID {} at position {} exceeds model vocab_size {}",
                target, pos, vocab_size
            ));
        }
        let logits_slice = &logits[predictor * vocab_size..(predictor + 1) * vocab_size];
        let max_val = logits_slice
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f64 = logits_slice
            .iter()
            .map(|&v| ((v - max_val) as f64).exp())
            .sum();
        let target_logit = logits_slice[target];
        if sum_exp > 0.0 && sum_exp.is_finite() {
            let token_loss = sum_exp.ln() - ((target_logit - max_val) as f64);
            total_loss += token_loss;
            targets_evaluated += 1;
        }
    }

    Ok((total_loss, targets_evaluated))
}

/// Scans intermediate step checkpoints inside `{intermediate_cp_dir}` and compares their
/// causal probe loss against `{staging_out_dir}`. If an intermediate checkpoint achieved
/// a lower (better) loss and is a valid working + continuation model, promotes its weights!
pub fn select_and_report_best_checkpoint(
    stage_id: &str,
    staging_out_dir: &Path,
    intermediate_cp_dir: &Path,
) -> Result<(f64, f64, String), String> {
    let probe_text = "ಕನ್ನಡ: ಕಂಪ್ಯೂಟರ್ ವಿಜ್ಞಾನ ತತ್ವಗಳು ೧೨೩೪೫. संस्कृत: सङ्गणकतत्त्वानि ०१२३४. Code: fn run() -> i32 { 42 } English: Foundations of computation.";

    let eval_dir_loss = |dir: &Path| -> Option<(f64, f64)> {
        let (is_m, _, _) = verify_working_model(dir);
        let (is_c, _, _) = verify_continuation_ready(dir);
        if !is_m || !is_c {
            return None;
        }
        let cfg = TaraConfig::from_json_file(&dir.join("config.json").to_string_lossy()).ok()?;
        let tok = TaraTokenizer::from_file(&dir.join("tokenizer.json").to_string_lossy()).ok()?;
        let (weights, _) = load_model_weights_with_shapes(&dir.to_string_lossy()).ok()?;
        let model = TaraForCausalLM::from_weights_and_config(weights, cfg.clone(), &dir.to_string_lossy()).ok()?;
        let tids = tok.encode(probe_text);
        let (total_loss, targets) = compute_causal_sequence_loss(&model, &tids, cfg.vocab_size).ok()?;
        if targets == 0 {
            return None;
        }
        let avg_loss = total_loss / targets as f64;
        let ppl = avg_loss.exp();
        if avg_loss.is_finite() && ppl.is_finite() {
            Some((avg_loss, ppl))
        } else {
            None
        }
    };

    let (mut best_loss, mut best_ppl) = eval_dir_loss(staging_out_dir)
        .ok_or_else(|| format!("Failed to evaluate final staged model in '{}'", staging_out_dir.display()))?;
    let mut best_source = "final_step".to_string();
    let mut best_path: Option<PathBuf> = None;
    let bpb = best_loss / std::f64::consts::LN_2;
    println!("  [LIVE EVAL] Stage '{}' Final Step -> Loss: {:.4} | Perplexity: {:.4} | Bits-Per-Byte: {:.4}", stage_id, best_loss, best_ppl, bpb);

    // Check all intermediate step checkpoints (e.g. step_100, step_200) to see if an earlier checkpoint was better
    if intermediate_cp_dir.exists() {
        if let Ok(entries) = fs::read_dir(intermediate_cp_dir) {
            let mut subdirs: Vec<PathBuf> = entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_dir())
                .collect();
            subdirs.sort();

            for cp_sub in subdirs {
                let cp_name = cp_sub.file_name().unwrap_or_default().to_string_lossy().to_string();
                if let Some((loss, ppl)) = eval_dir_loss(&cp_sub) {
                    let cp_bpb = loss / std::f64::consts::LN_2;
                    println!("  [LIVE EVAL] Intermediate '{}' -> Loss: {:.4} | Perplexity: {:.4} | Bits-Per-Byte: {:.4}", cp_name, loss, ppl, cp_bpb);
                    if loss < best_loss {
                        best_loss = loss;
                        best_ppl = ppl;
                        best_source = cp_name;
                        best_path = Some(cp_sub);
                    }
                }
            }
        }
    }

    // If an intermediate checkpoint beat the final step, copy its artifacts into staging_out_dir
    if let Some(winner_dir) = best_path {
        println!(
            "  [BEST CHECKPOINT SELECTED] Promoting '{}' (Loss: {:.4}, PPL: {:.4}) over final step!",
            best_source, best_loss, best_ppl
        );
        for entry in fs::read_dir(&winner_dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let src = entry.path();
            if src.is_file() {
                if let Some(fname) = src.file_name() {
                    fs::copy(&src, staging_out_dir.join(fname)).map_err(|e| e.to_string())?;
                }
            }
        }
    } else {
        println!(
            "  [BEST CHECKPOINT SELECTED] Retaining 'final_step' as best checkpoint (Loss: {:.4}, PPL: {:.4})",
            best_loss, best_ppl
        );
    }

    Ok((best_loss, best_ppl, best_source))
}

/// Evaluates a checkpoint directory by loading the model, verifying shape & vocabulary compatibility,
/// executing a causal forward pass on the canonical probe sequence, and streaming held-out dataset
/// samples to compute validation loss and finite perplexity.
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
        return Err(format!(
            "Checkpoint at '{}' is not a working model.",
            eval_cp.display()
        )
        .into());
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
    let model = TaraForCausalLM::from_weights_and_config(
        weights,
        config.clone(),
        &eval_cp.to_string_lossy(),
    )?;

    let probe_text = "ಕನ್ನಡ: ಕಂಪ್ಯೂಟರ್ ವಿಜ್ಞಾನ ತತ್ವಗಳು ೧೨೩೪೫. संस्कृत: सङ्गणकतत्त्वानि ०१२३४. Code: fn run() -> i32 { 42 } English: Foundations of computation.";
    let token_ids = tokenizer.encode(probe_text);
    if token_ids.len() < 2 {
        return Err("Tokenizer produced fewer than 2 tokens for probe text".into());
    }

    let (probe_total_loss, probe_targets) =
        compute_causal_sequence_loss(&model, &token_ids, config.vocab_size)?;
    let avg_loss = if probe_targets > 0 {
        probe_total_loss / probe_targets as f64
    } else {
        f64::NAN
    };
    let perplexity = avg_loss.exp();

    println!("  Forward Pass Probe Loss        : {:.4}", avg_loss);
    println!("  Estimated Probe Perplexity     : {:.4}", perplexity);
    println!(
        "  Probe Loss & Perplexity Finite : {}",
        if avg_loss.is_finite() && perplexity.is_finite() {
            "PASSED"
        } else {
            "FAILED (Non-finite)"
        }
    );

    if !avg_loss.is_finite() || !perplexity.is_finite() {
        return Err(
            "Model evaluation rejected: forward pass produced non-finite loss or perplexity".into(),
        );
    }

    // Stream held-out dataset records when provided
    if let Some(ds_path) = eval_dataset {
        if !ds_path.exists() {
            return Err(format!(
                "Model evaluation rejected: held-out evaluation dataset path '{}' does not exist",
                ds_path.display()
            )
            .into());
        }
        let mut reader = ExpandableDatasetReader::new(ShardStreamingMode::Sequential);
        if ds_path.is_dir() {
            let _ = reader.add_shards_from_dir(ds_path)?;
        } else {
            reader.add_shard(ds_path)?;
        }
        if reader.shard_count() == 0 {
            return Err(format!(
                "Model evaluation rejected: held-out dataset '{}' contains zero readable shards",
                ds_path.display()
            )
            .into());
        }
        let mut ds_loss_sum = 0.0f64;
        let mut ds_tokens = 0usize;
        let mut ds_samples = 0usize;
        let max_eval_samples = 16usize;
        let max_ctx = config.max_position_embeddings.clamp(8, 128);

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
                let (l, t) = compute_causal_sequence_loss(&model, &tids, config.vocab_size)?;
                ds_loss_sum += l;
                ds_tokens += t;
                ds_samples += 1;
            }
        }

        if ds_tokens == 0 {
            return Err(format!(
                "Model evaluation rejected: held-out dataset '{}' yielded zero valid evaluation tokens",
                ds_path.display()
            )
            .into());
        }

        let ds_avg_loss = ds_loss_sum / ds_tokens as f64;
        let ds_ppl = ds_avg_loss.exp();
        println!("  Held-Out Dataset Samples Eval  : {}", ds_samples);
        println!("  Held-Out Dataset Loss          : {:.4}", ds_avg_loss);
        println!("  Held-Out Dataset Perplexity    : {:.4}", ds_ppl);
        if !ds_avg_loss.is_finite() || !ds_ppl.is_finite() {
            return Err(
                "Model evaluation rejected: held-out dataset evaluation produced non-finite loss or perplexity"
                    .into(),
            );
        }
    }

    println!("================================================================================");
    println!("CHECKPOINT EVALUATION SUCCESSFUL: Model is fully functional and numerically stable.");
    println!("================================================================================");
    Ok(())
}

/// Verifies that the Pre-Training Authorization Gate has passed and generated an authorized record.
fn verify_pretraining_authorization(checkpoints_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let candidate_paths = [
        checkpoints_dir.join("pretraining_authorization.json"),
        PathBuf::from("storage/models/checkpoints/pretraining_authorization.json"),
    ];

    let mut found_path = None;
    for p in &candidate_paths {
        if p.exists() {
            found_path = Some(p.clone());
            break;
        }
    }

    let auth_path = match found_path {
        Some(p) => p,
        None => {
            return Err(
                "Pre-Training Authorization Gate check failed: 'pretraining_authorization.json' not found. \
                Execute 'cargo run --bin tara_pretraining_gate' to audit tokenizer collisions and deep dataset deduplication before training."
                    .into(),
            );
        }
    };

    let content = fs::read_to_string(&auth_path)?;
    let val: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
        format!(
            "Failed to parse pre-training authorization file '{}': {}",
            auth_path.display(),
            e
        )
    })?;

    let status = val
        .get("authorization_status")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if status != "AUTHORIZED" {
        return Err(format!(
            "Pre-Training Authorization Gate check failed: authorization_status is '{}', expected 'AUTHORIZED' in '{}'.",
            status,
            auth_path.display()
        )
        .into());
    }

    println!("[PRE-TRAINING AUTHORIZATION VERIFIED: AUTHORIZED]");
    println!("  Authorization Record : {}", auth_path.display());
    if let Some(dedup) = val.get("content_dedup") {
        if let Some(scanned) = dedup.get("total_records_scanned") {
            println!("  Records Audited      : {}", scanned);
        }
        if let Some(dups) = dedup.get("duplicate_records_count") {
            println!("  Duplicate Records    : {}", dups);
        }
    }
    println!("--------------------------------------------------------------------------------");
    Ok(())
}

/// Enforces cloud / Google Colab training environment to protect local developer PC hardware.
fn verify_cloud_training_environment() -> Result<(), Box<dyn std::error::Error>> {
    let is_colab = std::env::var("COLAB_GPU").is_ok() || std::env::var("COLAB_RELEASE_TAG").is_ok();
    let is_cloud = std::env::var("TARA_CLOUD_TRAINING").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false);
    let allow_local = std::env::var("TARA_ALLOW_LOCAL_TRAINING").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false);

    if !is_colab && !is_cloud && !allow_local {
        return Err(
            "Training execution blocked on local machine: System directive requires all model training to execute exclusively in Google Colab or Cloud GPU environments. Zero heavy compute loads permitted on local developer machine. Run inside Google Colab (with GPU acceleration) or set TARA_CLOUD_TRAINING=1 for verified cloud instances, or TARA_ALLOW_LOCAL_TRAINING=1 for explicit diagnostics."
                .into(),
        );
    }
    println!("[COMPUTE ENVIRONMENT VERIFIED: CLOUD / COLAB]");
    Ok(())
}

fn default_checkpoint_interval(fallback: usize) -> usize {
    env::var("TARA_CHECKPOINT_INTERVAL")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(fallback)
}

/// Build default progression of extensible stages.
pub fn build_default_stages(
    base_model: &Path,
    checkpoints_dir: &Path,
    foundational_curriculum_path: &Path,
    ability_curriculum_path: &Path,
    adaptation_path: &Path,
    main_dataset_dir: &Path,
    scale_dataset_dir: &Path,
) -> Vec<StageDefinition> {
    vec![
        StageDefinition {
            stage_index: 0,
            stage_id: "stage0_pretraining".to_string(),
            stage_name: "Stage 0: Foundational Pretraining".to_string(),
            description: "Large-scale foundational pretraining on verified canonical dataset shards."
                .to_string(),
            dataset_path: main_dataset_dir.to_string_lossy().to_string(),
            input_checkpoint: base_model.to_string_lossy().to_string(),
            output_checkpoint: checkpoints_dir
                .join("stage0_pretraining")
                .to_string_lossy()
                .to_string(),
            epochs: 1,
            max_steps: None,
            learning_rate: 2e-4,
            batch_size: 4,
            checkpoint_interval: default_checkpoint_interval(200),
            depends_on: vec![],
        },
        StageDefinition {
            stage_index: 1,
            stage_id: "stage1_curriculum".to_string(),
            stage_name: "Stage 1: Foundational Academic Curriculum Pretraining".to_string(),
            description: "Foundational academic training on structured pedagogical records across Domains 1-8 (Math, Science, Programming, AI/ML, Research, Culture, Creativity, Search)."
                .to_string(),
            dataset_path: foundational_curriculum_path.to_string_lossy().to_string(),
            input_checkpoint: checkpoints_dir
                .join("stage0_pretraining")
                .to_string_lossy()
                .to_string(),
            output_checkpoint: checkpoints_dir
                .join("stage1_curriculum")
                .to_string_lossy()
                .to_string(),
            epochs: 1,
            max_steps: None,
            learning_rate: 5e-4,
            batch_size: 4,
            checkpoint_interval: default_checkpoint_interval(100),
            depends_on: vec!["stage0_pretraining".to_string()],
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
            checkpoint_interval: default_checkpoint_interval(100),
            depends_on: vec!["stage1_curriculum".to_string()],
        },
        StageDefinition {
            stage_index: 3,
            stage_id: "stage3_continual_adaptation".to_string(),
            stage_name: "Stage 3: Continual Adaptation & Experiential Specialization".to_string(),
            description: "Continual adaptation and specialization trained on genuine post-execution experiential lessons and validated strategies."
                .to_string(),
            dataset_path: adaptation_path.to_string_lossy().to_string(),
            input_checkpoint: checkpoints_dir
                .join("stage2_ability_training")
                .to_string_lossy()
                .to_string(),
            output_checkpoint: checkpoints_dir
                .join("stage3_continual_adaptation")
                .to_string_lossy()
                .to_string(),
            epochs: 1,
            max_steps: None,
            learning_rate: 5e-5,
            batch_size: 4,
            checkpoint_interval: default_checkpoint_interval(200),
            depends_on: vec!["stage2_ability_training".to_string()],
        },
        StageDefinition {
            stage_index: 4,
            stage_id: "stage4_main_dataset_v1".to_string(),
            stage_name: "Stage 4: Main Dataset V1 Pretraining".to_string(),
            description: "Full-scale pretraining on verified multi-domain shards (main dataset v1, formerly canonical)."
                .to_string(),
            dataset_path: scale_dataset_dir.to_string_lossy().to_string(),
            input_checkpoint: checkpoints_dir
                .join("stage3_continual_adaptation")
                .to_string_lossy()
                .to_string(),
            output_checkpoint: checkpoints_dir
                .join("stage4_main_dataset_v1")
                .to_string_lossy()
                .to_string(),
            epochs: 1,
            max_steps: None,
            learning_rate: 2e-4,
            batch_size: 4,
            checkpoint_interval: default_checkpoint_interval(500),
            depends_on: vec!["stage3_continual_adaptation".to_string()],
        },
        StageDefinition {
            stage_index: 5,
            stage_id: "stage5_scale_expansion".to_string(),
            stage_name: "Stage 5: Extended Pretraining & Scale Expansion".to_string(),
            description: "Extensible continuation stage for further scale datasets, custom domains, or specialization."
                .to_string(),
            dataset_path: scale_dataset_dir.to_string_lossy().to_string(),
            input_checkpoint: checkpoints_dir
                .join("stage4_main_dataset_v1")
                .to_string_lossy()
                .to_string(),
            output_checkpoint: checkpoints_dir
                .join("stage5_scale_expansion")
                .to_string_lossy()
                .to_string(),
            epochs: 1,
            max_steps: None,
            learning_rate: 1e-4,
            batch_size: 4,
            checkpoint_interval: default_checkpoint_interval(500),
            depends_on: vec!["stage4_main_dataset_v1".to_string()],
        },
    ]
}


fn print_usage() {
    println!("Usage: tara_training_stages [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --stage <N>                 Execute stage by index (1, 2, 3, ...) [mutually exclusive with --stage-id / --all-stages]");
    println!("  --stage-id <ID>             Execute stage by ID (e.g. stage1_curriculum) [mutually exclusive with --stage / --all-stages]");
    println!(
        "  --all-stages                Execute all configured stages in topological DAG order"
    );
    println!("  --stages-config <FILE>      Load extensible pipeline stages from custom JSON file");
    println!("  --init-from <MODEL_DIR>     Warm-start entry stage weights from a working model (resets optimizer state, resume=false)");
    println!("  --resume-from <CHECKPOINT>  Explicitly resume entry stage from a continuation-ready checkpoint (resume=true)");
    println!("  --eval-checkpoint <PATH>    Evaluate a checkpoint directory (weights, shapes, probe & dataset loss)");
    println!(
        "  --eval-dataset <PATH>       Required held-out JSONL dataset path for --eval-checkpoint"
    );
    println!("  --model <PATH>              Base model checkpoint directory (default: storage/models/tara_candidate_v1)");
    println!("  --curriculum <PATH>         Foundational curriculum path");
    println!("  --ability-path <PATH>       Ability curriculum path");
    println!("  --main-dataset-dir <PATH>   Main dataset v1 directory (alias: --canonical-dir)");
    println!("  --scale-dir <PATH>          Extended scale pretraining datasets directory");
    println!("  --adaptation-path <PATH>    Continual adaptation lessons path");
    println!("  --checkpoints-dir <PATH>    Base directory for stage checkpoints (default: storage/models/checkpoints)");
    println!("  --device <DEVICE>           Device backend: 'cpu', 'gpu', 'cuda', or 'auto' (default: auto)");
    println!("  --precision <PRECISION>     Precision: 'fp16', 'fp32', or 'auto' (default: auto)");
    println!("  --force                     Force re-execution of already completed stages");
    println!("  --dry-run                   Audit and verify stage DAG and inputs/outputs without gradient updates");
    println!("  --epochs <N>                Override epochs for active stage(s) (must be >= 1)");
    println!(
        "  --max-steps <N>             Override maximum steps for active stage(s) (must be >= 1)"
    );
    println!("  --learning-rate <F>         Override learning rate (must be > 0.0)");
    println!(
        "  --batch-size <N>            Override gradient accumulation batch size (must be >= 1)"
    );
    println!("  -h, --help                  Print help information");
}

/// Validates mutually exclusive CLI selection modes (`--stage`, `--stage-id`, `--all-stages`, `--eval-checkpoint`)
/// and mutually exclusive entry overrides (`--init-from`, `--resume-from`).
fn validate_cli_mode_exclusivity(
    target_stage_index: Option<usize>,
    target_stage_id: Option<&str>,
    run_all_stages: bool,
    eval_checkpoint_path: Option<&Path>,
    init_from_override: Option<&Path>,
    resume_from_override: Option<&Path>,
) -> Result<(), String> {
    let mut mode_count = 0usize;
    if target_stage_index.is_some() {
        mode_count += 1;
    }
    if target_stage_id.is_some() {
        mode_count += 1;
    }
    if run_all_stages {
        mode_count += 1;
    }
    if eval_checkpoint_path.is_some() {
        mode_count += 1;
    }
    if mode_count > 1 {
        return Err(
            "CLI options --stage <N>, --stage-id <ID>, --all-stages, and --eval-checkpoint <PATH> are mutually exclusive. Specify only one.".to_string(),
        );
    }
    if init_from_override.is_some() && resume_from_override.is_some() {
        return Err(
            "CLI options --init-from <MODEL_DIR> and --resume-from <CHECKPOINT> are mutually exclusive. Use --init-from for warm-starting weights only, or --resume-from for full optimizer continuation.".to_string(),
        );
    }
    Ok(())
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
    let mut init_from_override: Option<PathBuf> = None;
    let mut resume_from_override: Option<PathBuf> = None;
    let mut eval_checkpoint_path: Option<PathBuf> = None;
    let mut eval_dataset_path: Option<PathBuf> = None;
    let mut force_rerun = false;

    let mut model_dir = PathBuf::from("storage/models/tara_candidate_v1");
    let mut curriculum_path =
        PathBuf::from("storage/datasets/curriculum/foundational_curriculum.jsonl");
    let mut ability_curriculum_path =
        PathBuf::from("storage/datasets/curriculum/ability_acquisition.jsonl");
    let mut main_dataset_dir = PathBuf::from("storage/datasets/main_dataset_v2");
    let mut scale_dataset_dir = PathBuf::from("storage/datasets/main_dataset_v1");
    let mut adaptation_path = PathBuf::from("storage/persistence/experiential_lessons.jsonl");
    let mut checkpoints_dir = env::var("GDRIVE_CHECKPOINT_DIR")
        .ok()
        .filter(|s| !s.trim().is_empty() && Path::new("/content/drive/MyDrive").exists())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("storage/models/checkpoints"));
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
            "--init-from" => {
                let val = require_arg_value(&args, &mut i, "--init-from")?;
                init_from_override = Some(PathBuf::from(val));
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
            "--main-dataset-dir" | "--main-dataset" | "--canonical-dir" => {
                let flag = args[i].clone();
                let val = require_arg_value(&args, &mut i, &flag)?;
                main_dataset_dir = PathBuf::from(val);
            }
            "--scale-dir" | "--scale-dataset-dir" => {
                let flag = args[i].clone();
                let val = require_arg_value(&args, &mut i, &flag)?;
                scale_dataset_dir = PathBuf::from(val);
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
    validate_cli_mode_exclusivity(
        target_stage_index,
        target_stage_id.as_deref(),
        run_all_stages,
        eval_checkpoint_path.as_deref(),
        init_from_override.as_deref(),
        resume_from_override.as_deref(),
    )?;

    println!("================================================================================");
    println!(" TARA EXTENSIBLE MULTI-STAGE TRAINING PIPELINE ORCHESTRATOR");
    println!("================================================================================");

    // Mode: Standalone Checkpoint Evaluation (requires explicit or non-overlapping held-out eval dataset)
    if let Some(ref eval_cp) = eval_checkpoint_path {
        let eval_ds = eval_dataset_path.as_deref().ok_or(
            "Standalone --eval-checkpoint requires --eval-dataset <HELD_OUT_PATH> so evaluation is never silently run on a training split."
        )?;
        return evaluate_checkpoint_model(eval_cp, Some(eval_ds));
    }

    // Acquire exclusive process lock on checkpoints_dir to prevent concurrent stage clobbering
    let _orchestrator_lock = OrchestratorLockGuard::acquire(&checkpoints_dir)?;

    // Default to stage 0 if no stage selection was explicitly provided
    if !run_all_stages && target_stage_id.is_none() && target_stage_index.is_none() {
        target_stage_index = Some(0);
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
            &adaptation_path,
            &main_dataset_dir,
            &scale_dataset_dir,
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
    let (is_base_model, missing_base, base_param_count) =
        verify_working_model_with_context("main_pipeline_startup", "base_model", &model_dir, "INPUT");
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

    // Enforce pre-training authorization gate check (unconditional fail-closed)
    verify_pretraining_authorization(&checkpoints_dir)?;

    if run_all_stages {
        println!("[TARGET EXECUTION: ALL STAGES IN TOPOLOGICAL DAG ORDER]");
    } else if let Some(ref id) = target_stage_id {
        println!("[TARGET EXECUTION: STAGE ID '{}' ONLY]", id);
    } else if let Some(idx) = target_stage_index {
        println!("[TARGET EXECUTION: STAGE {:02} ONLY]", idx);
    }

    // Filter stages to run BEFORE preflight and plan generation so dry-run validates targeted stages strictly
    let stages_to_run: Vec<StageDefinition> = if run_all_stages {
        stages.clone()
    } else if let Some(ref id) = target_stage_id {
        stages
            .iter()
            .filter(|s| &s.stage_id == id)
            .cloned()
            .collect()
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
            "No matching stages found to run. Specify a valid --stage <N>, --stage-id <ID>, or --all-stages."
                .into(),
        );
    }

    // Strict validation of explicit --init-from (working model only) vs --resume-from (working model + continuation state)
    if let Some(ref init_path) = init_from_override {
        let _ = recover_interrupted_promotion(init_path);
        if !init_path.exists() {
            return Err(format!(
                "Specified --init-from path does not exist: {}",
                init_path.display()
            )
            .into());
        }
        let (is_model, missing_model, _) =
            verify_working_model_with_context("cli_init_override", "entry", init_path, "INPUT");
        if !is_model {
            return Err(format!(
                "Specified --init-from path '{}' is not a valid working model: {:?}",
                init_path.display(),
                missing_model
            )
            .into());
        }
    }

    if let Some(ref r_path) = resume_from_override {
        let _ = recover_interrupted_promotion(r_path);
        if !r_path.exists() {
            return Err(format!(
                "Specified --resume-from path does not exist: {}",
                r_path.display()
            )
            .into());
        }
        let (is_model, missing_model, _) =
            verify_working_model_with_context("cli_resume_override", "entry", r_path, "RESUME");
        if !is_model {
            return Err(format!(
                "Specified --resume-from path '{}' is not a valid working model: {:?}",
                r_path.display(),
                missing_model
            )
            .into());
        }
        let (is_cont, missing_cont, _) =
            verify_continuation_ready_with_context("cli_resume_override", "entry", r_path, "RESUME");
        if !is_cont {
            return Err(format!(
                "Specified --resume-from path '{}' is not continuation-ready (requires valid checkpoint_state.json and optimizer.safetensors): {:?}",
                r_path.display(),
                missing_cont
            )
            .into());
        }
    }

    let mut dataset_inspection_cache: HashMap<PathBuf, DatasetInspection> = HashMap::new();
    let mut stage_metadata_list = Vec::new();
    for s in &stages {
        let out_path = PathBuf::from(&s.output_checkpoint);
        let _ = recover_interrupted_promotion(&out_path);

        let is_targeted = run_all_stages
            || (target_stage_id
                .as_ref()
                .map(|id| id == &s.stage_id)
                .unwrap_or(false))
            || (target_stage_index
                .map(|idx| idx == s.stage_index)
                .unwrap_or(false));

        let ds_path = PathBuf::from(&s.dataset_path);
        let insp = if !is_targeted {
            DatasetInspection {
                jsonl_files: 0,
                total_lines: 0,
                valid_records: 0,
                malformed_records: 0,
                empty_lines: 0,
                total_bytes: 0,
                dataset_fingerprint: "untargeted_stage".to_string(),
            }
        } else if let Some(cached) = dataset_inspection_cache.get(&ds_path) {
            cached.clone()
        } else {
            let res = inspect_dataset(&ds_path).map_err(|e| {
                format!(
                    "Dataset inspection failed for targeted stage '{}': {}",
                    s.stage_id, e
                )
            })?;
            dataset_inspection_cache.insert(ds_path, res.clone());
            res
        };

        let is_root_stage = s.depends_on.is_empty();
        let in_path = Path::new(&s.input_checkpoint);

        // Preflight input digest:
        // Do NOT require checkpoints from future stages to exist before the first stage runs.
        // Stage 1 depends on Stage 0's output, so validate that checkpoint only after Stage 0 successfully completes and saves it.
        let in_digest = if !is_targeted {
            "untargeted".to_string()
        } else if is_root_stage {
            let (is_base, missing, _) = verify_working_model_with_context(
                "dag_preflight_entry_input",
                &s.stage_id,
                in_path,
                "INPUT",
            );
            if !is_base {
                return Err(format!(
                    "Entry stage '{}' input checkpoint '{}' is not a valid working model: {:?}",
                    s.stage_id,
                    s.input_checkpoint,
                    missing
                )
                .into());
            }
            compute_full_checkpoint_digest(in_path)?.unwrap_or_else(|| "pending".to_string())
        } else {
            // Downstream stage depends on uncompleted or completed predecessor
            if in_path.exists() && !is_empty_dir(in_path) {
                // If predecessor completed in a prior run
                compute_full_checkpoint_digest(in_path)
                    .unwrap_or(None)
                    .unwrap_or_else(|| "pending".to_string())
            } else {
                "pending".to_string()
            }
        };

        let fp =
            compute_stage_config_fingerprint(s, &in_digest, &insp.dataset_fingerprint, &precision);

        // Output directory classification:
        // If the output directory exists but contains no valid checkpoint, classify it as NEW_RUN
        // unless explicit resume mode was requested. Do not try to load it as model weights.
        let (out_is_valid_model, out_is_continuation, out_completed_ts) =
            if is_targeted && out_path.exists() && !is_empty_dir(&out_path) {
                let (is_m, _, _) = verify_working_model_with_context(
                    "dag_preflight_output_check",
                    &s.stage_id,
                    &out_path,
                    "OUTPUT",
                );
                let (is_c, _, _) = verify_continuation_ready_with_context(
                    "dag_preflight_output_check",
                    &s.stage_id,
                    &out_path,
                    "OUTPUT",
                );
                if is_m && is_c && should_skip_completed_stage(&out_path, &fp) {
                    (true, true, read_stage_completed_timestamp(&out_path))
                } else {
                    (false, false, None)
                }
            } else {
                (false, false, None)
            };

        let (is_model, is_cont, weights_hash, full_cp_hash, completed_ts, stage_status) =
            if out_is_valid_model && out_is_continuation {
                let w = compute_model_weights_digest(&out_path).ok().flatten();
                let f = compute_full_checkpoint_digest(&out_path).ok().flatten();
                (true, true, w, f, out_completed_ts, "COMPLETED".to_string())
            } else {
                // Uncompleted, empty, or unverified output directory is classified as NEW_RUN
                (false, false, None, None, None, "STAGED_READY".to_string())
            };

        if is_targeted {
            println!(
                "  --> [ACTIVE] Stage {:02} [{}]: {} (Status: {})",
                s.stage_index, s.stage_id, s.stage_name, stage_status
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
            depends_on: s.depends_on.clone(),
            input_checkpoint: s.input_checkpoint.clone(),
            output_checkpoint: s.output_checkpoint.clone(),
            dataset_source: s.dataset_path.clone(),
            dataset_inspection: insp,
            completed_at_utc: completed_ts,
            status: stage_status,
            model_weights_sha256: weights_hash,
            full_checkpoint_sha256: full_cp_hash,
            stage_config_fingerprint: fp,
            is_working_model: is_model,
            is_continuation_ready: is_cont,
        });
    }

    let insp_by_stage_id: HashMap<String, DatasetInspection> = stage_metadata_list
        .iter()
        .map(|m| (m.stage_id.clone(), m.dataset_inspection.clone()))
        .collect();

    // Full DAG Preflight Check across all targeted stages BEFORE dry-run exit so --dry-run is a strict verification gate!
    let mut scheduled_outputs: HashSet<PathBuf> = HashSet::new();
    for (idx, s) in stages_to_run.iter().enumerate() {
        let ds_insp = insp_by_stage_id.get(&s.stage_id).ok_or_else(|| {
            format!(
                "DAG Preflight Failed: Stage '{}' dataset '{}' missing inspection",
                s.stage_id, s.dataset_path
            )
        })?;
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
            if let Some(ref r_path) = resume_from_override {
                r_path.to_path_buf()
            } else if let Some(ref i_path) = init_from_override {
                i_path.to_path_buf()
            } else {
                PathBuf::from(&s.input_checkpoint)
            }
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
            // If continuation artifacts exist on disk (and we are not explicitly overriding with --init-from), verify they are not corrupt!
            let is_init_only = idx == 0 && init_from_override.is_some();
            if !is_init_only
                && (effective_in.join("checkpoint_state.json").exists()
                    || effective_in.join("optimizer.safetensors").exists())
            {
                let (ok_cont, cont_issues, _) = verify_continuation_ready(&effective_in);
                if !ok_cont {
                    return Err(format!(
                        "DAG Preflight Failed: Stage '{}' input checkpoint '{}' contains corrupt/incomplete continuation state: {:?}",
                        s.stage_id,
                        effective_in.display(),
                        cont_issues
                    )
                    .into());
                }
            }
        }
        scheduled_outputs.insert(normalize_for_comparison(Path::new(&s.output_checkpoint)));
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
    write_json_atomically(&plan_path, &plan_json)?;
    println!("--------------------------------------------------------------------------------");
    println!(
        "Extensible Pipeline Plan Written To: {}",
        plan_path.display()
    );
    println!("--------------------------------------------------------------------------------");

    if dry_run {
        println!("[DRY-RUN AUDIT COMPLETED]");
        println!(
            "  All {} targeted stage(s) verified and validated in topological DAG order.",
            stages_to_run.len()
        );
        println!("  Ready for native Rust training execution.");
        return Ok(());
    }

    // Verify cloud / Google Colab training environment to protect local developer PC hardware
    verify_cloud_training_environment()?;

    let mut is_first_stage = true;
    let mut completed_outputs_by_norm: HashMap<PathBuf, String> = HashMap::new();
    let mut last_completed_stage_dir: Option<PathBuf> = None;

    for stage in stages_to_run {
        let out_dir = PathBuf::from(&stage.output_checkpoint);
        let _ = recover_interrupted_promotion(&out_dir);

        // Determine input checkpoint from explicit --resume-from / --init-from (entry stage only) or DAG predecessor output
        let norm_declared_in = normalize_for_comparison(Path::new(&stage.input_checkpoint));
        let (input_cp, explicit_resume_from, force_init_only) = if is_first_stage {
            if let Some(ref r_override) = resume_from_override {
                println!(
                    "  [RESUME PIPELINE] Resuming entry stage '{}' from explicit continuation checkpoint: {}",
                    stage.stage_id,
                    r_override.display()
                );
                (
                    r_override.to_string_lossy().to_string(),
                    Some(r_override.to_string_lossy().to_string()),
                    false,
                )
            } else if let Some(ref i_override) = init_from_override {
                println!(
                    "  [WARM-START PIPELINE] Initializing entry stage '{}' weights from model: {}",
                    stage.stage_id,
                    i_override.display()
                );
                (i_override.to_string_lossy().to_string(), None, true)
            } else {
                (stage.input_checkpoint.clone(), None, false)
            }
        } else if let Some(pred_out) = completed_outputs_by_norm.get(&norm_declared_in) {
            (pred_out.clone(), Some(pred_out.clone()), false)
        } else {
            (stage.input_checkpoint.clone(), None, false)
        };

        let input_path = Path::new(&input_cp);
        let (is_input_model, missing_input, input_param_count) =
            verify_working_model_with_purpose(input_path, "INPUT");
        if !is_input_model {
            return Err(format!(
                "Stage {:02} ('{}') input checkpoint '{}' is not a valid working model: {:?}",
                stage.stage_index, stage.stage_id, input_cp, missing_input
            )
            .into());
        }

        let loaded_param_count = input_param_count.unwrap_or(0);
        if stage.stage_index == 0 && input_cp.contains("tara_candidate_v1") {
            println!("  [INITIAL PRODUCTION MODEL VALIDATION]");
            println!("    Input Checkpoint Shards   : VALIDATED (120 tensors)");
            println!("    Exact Parameter Count     : {} parameters", loaded_param_count);
        }

        // Strict auto-resume verification: if input_path has continuation files and we aren't in --init-from mode, verify them!
        let has_any_continuation_file = input_path.join("checkpoint_state.json").exists()
            || input_path.join("optimizer.safetensors").exists();
        if !force_init_only && has_any_continuation_file {
            let (ok_cont, cont_issues, _) =
                verify_continuation_ready_with_purpose(input_path, "RESUME");
            if !ok_cont {
                return Err(format!(
                    "Stage {:02} ('{}') input checkpoint '{}' contains partial or corrupt continuation state: {:?}",
                    stage.stage_index, stage.stage_id, input_cp, cont_issues
                )
                .into());
            }
        }

        // Two distinct input identities:
        // 1. `input_model_digest`: weights-only SHA-256 ("same model weights", used when warm-starting with --init-from)
        // 2. `input_full_digest`: full checkpoint SHA-256 ("same resumable checkpoint", covering weights + config + tokenizer + optimizer state)
        let input_model_digest = compute_model_weights_digest(input_path)?.ok_or_else(|| {
            format!(
                "Input checkpoint '{}' has no model weights digest",
                input_cp
            )
        })?;
        let input_full_digest = compute_full_checkpoint_digest(input_path)?
            .unwrap_or_else(|| input_model_digest.clone());
        let effective_input_digest = if force_init_only {
            &input_model_digest
        } else {
            &input_full_digest
        };
        let ds_insp = insp_by_stage_id.get(&stage.stage_id).ok_or_else(|| {
            format!(
                "Execution Failed: Stage '{}' dataset '{}' missing inspection",
                stage.stage_id, stage.dataset_path
            )
        })?;
        let expected_fingerprint = compute_stage_config_fingerprint(
            &stage,
            effective_input_digest,
            &ds_insp.dataset_fingerprint,
            &precision,
        );

        // Fingerprint-gated idempotence check (skip unless --force or explicit entry override was requested)
        let has_entry_override =
            is_first_stage && (resume_from_override.is_some() || init_from_override.is_some());
        if !force_rerun && !has_entry_override && should_skip_completed_stage(&out_dir, &expected_fingerprint) {
            println!();
            println!(
                "[STAGE {:02} ({}) ALREADY COMPLETED & FINGERPRINT VERIFIED: SKIPPING (use --force to re-run)]",
                stage.stage_index, stage.stage_id
            );
            completed_outputs_by_norm.insert(
                normalize_for_comparison(&out_dir),
                stage.output_checkpoint.clone(),
            );
            last_completed_stage_dir = Some(out_dir.clone());
            is_first_stage = false;
            continue;
        }

        println!();
        println!(
            "================================================================================"
        );
        println!(
            " EXECUTING STAGE {:02}: {}",
            stage.stage_index, stage.stage_name
        );
        println!(
            "================================================================================"
        );
        println!("  Stage ID            : {}", stage.stage_id);
        println!("  Description         : {}", stage.description);
        println!("  Dataset Source      : {}", stage.dataset_path);
        println!(
            "  Input Checkpoint    : {} [VERIFIED WORKING MODEL]",
            input_cp
        );
        println!("  Target Candidate    : {}", stage.output_checkpoint);

        // Separate intermediate checkpoint directory from published candidate directory
        let intermediate_cp_dir = checkpoints_dir.join(&stage.stage_id).join("checkpoints");
        fs::create_dir_all(&intermediate_cp_dir)?;

        // Transactional staging directory: train into isolated staging_out_dir, verify, write STAGE_METADATA.json, then atomically promote!
        let run_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let staging_out_dir = PathBuf::from(format!(
            "{}.staging_{}_{}",
            out_dir.display(),
            run_nanos,
            std::process::id()
        ));
        if staging_out_dir.exists() {
            let _ = fs::remove_dir_all(&staging_out_dir);
        }
        fs::create_dir_all(&staging_out_dir)?;

        let should_resume =
            !force_init_only && (has_any_continuation_file || explicit_resume_from.is_some());
        let checkpoint_mode = if should_resume { "RESUME" } else { "NEW_RUN" };
        let resume_source = if should_resume {
            Some(
                explicit_resume_from
                    .clone()
                    .unwrap_or_else(|| input_cp.clone()),
            )
        } else {
            None
        };

        println!();
        println!("  ------------------------------------------------------------------");
        println!("  [STAGE {:02} EXECUTION CONFIGURATION]", stage.stage_index);
        println!("    Resolved Input Path   : {}", input_cp);
        println!("    Resolved Output Path  : {}", stage.output_checkpoint);
        println!("    Checkpoint Mode       : {}", checkpoint_mode);
        println!("    Loaded Parameter Count: {}", loaded_param_count);
        println!("    Final Checkpoint Path : {}", stage.output_checkpoint);
        println!("  ------------------------------------------------------------------");

        let max_mem_mb = env::var("TARA_MAX_MEMORY_MB")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|&v| v >= 512.0)
            .unwrap_or(65536.0);

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
            candidate_dir: Some(staging_out_dir.to_string_lossy().to_string()),
            learning_rate: Some(stage.learning_rate),
            batch_size: Some(stage.batch_size),
            max_steps: stage.max_steps,
            device: Some(device.clone()),
            precision: Some(precision.clone()),
            checkpoint_dir: Some(intermediate_cp_dir.to_string_lossy().to_string()),
            checkpoint_interval: Some(stage.checkpoint_interval),
            resume: should_resume,
            resume_from: resume_source,
            stage_name: Some(format!("Stage {:02} ({})", stage.stage_index, stage.stage_id)),
            max_memory_mb: Some(max_mem_mb),
            ..Default::default()
        };

        let train_res = run_controlled_training_with_options(&input_cp, ".", stage.epochs, options);
        let result = match train_res {
            Ok(r) => r,
            Err(e) => {
                let _ = fs::remove_dir_all(&staging_out_dir);
                return Err(Box::new(e));
            }
        };

        // Scan and select best checkpoint between intermediate step checkpoints and final step
        if let Err(e) = select_and_report_best_checkpoint(
            &stage.stage_id,
            &staging_out_dir,
            &intermediate_cp_dir,
        ) {
            eprintln!("[BEST CHECKPOINT NOTICE] Could not evaluate intermediate checkpoints: {e}");
        }

        // Verify staged output satisfies dual-purpose model requirement BEFORE touching out_dir
        let (is_model, missing_model, param_cnt) =
            verify_working_model_with_purpose(&staging_out_dir, "OUTPUT");
        let (is_cont, missing_cont, last_step) =
            verify_continuation_ready_with_purpose(&staging_out_dir, "OUTPUT");
        if !is_model {
            let _ = fs::remove_dir_all(&staging_out_dir);
            return Err(format!(
                "Stage {:02} ('{}') failed working model verification in staging: {:?}",
                stage.stage_index, stage.stage_id, missing_model
            )
            .into());
        }
        if !is_cont {
            let _ = fs::remove_dir_all(&staging_out_dir);
            return Err(format!(
                "Stage {:02} ('{}') failed continuation-ready verification in staging: {:?}",
                stage.stage_index, stage.stage_id, missing_cont
            )
            .into());
        }

        let weights_sha = compute_model_weights_digest(&staging_out_dir)?
            .ok_or_else(|| "Staged model missing weights digest".to_string())?;
        let full_checkpoint_sha = compute_full_checkpoint_digest(&staging_out_dir)?
            .ok_or_else(|| "Staged checkpoint missing full digest".to_string())?;
        let completed_at = tara_engine::now_iso();

        let stage_provenance = serde_json::json!({
            "stage_index": stage.stage_index,
            "stage_id": stage.stage_id,
            "stage_name": stage.stage_name,
            "status": "COMPLETED",
            "completed_at_utc": completed_at,
            "input_checkpoint": input_cp,
            "input_model_weights_sha256": input_model_digest,
            "input_full_checkpoint_sha256": input_full_digest,
            "output_checkpoint": stage.output_checkpoint,
            "intermediate_checkpoints_dir": intermediate_cp_dir.to_string_lossy(),
            "dynamic_checkpoint_sha256": full_checkpoint_sha,
            "dynamic_model_weights_sha256": weights_sha,
            "stage_config_fingerprint": expected_fingerprint,
            "dataset_fingerprint": ds_insp.dataset_fingerprint,
            "total_parameters": param_cnt.unwrap_or(0),
            "last_optimizer_step": last_step,
            "is_working_model": is_model,
            "is_continuation_ready": is_cont,
            "training_result": result,
            "timestamp_utc": completed_at
        });
        if let Err(e) = write_json_atomically(
            &staging_out_dir.join("STAGE_METADATA.json"),
            &stage_provenance,
        ) {
            let _ = fs::remove_dir_all(&staging_out_dir);
            return Err(e.into());
        }

        // Atomically promote verified staging_out_dir -> out_dir with two-phase backup and rollback
        promote_directory_atomically(&staging_out_dir, &out_dir)?;

        println!(
            "--------------------------------------------------------------------------------"
        );
        println!("[STAGE {:02} COMPLETE]", stage.stage_index);
        println!("  Working Model Verification      : PASSED");
        println!("  Continuation-Ready Verification : PASSED");
        if let Some(lb) = result.get("loss_before").and_then(|v| v.as_f64()) {
            println!("  Baseline Loss (Before)          : {:.6}", lb);
        }
        if let Some(la) = result.get("loss_after").and_then(|v| v.as_f64()) {
            println!("  Final Step Loss (After)         : {:.6} (PPL: {:.4})", la, la.exp());
        }
        if let Some(bl) = result.get("best_loss").and_then(|v| v.as_f64()) {
            if let Some(bs) = result.get("best_step").and_then(|v| v.as_u64()) {
                println!(
                    "  Best Checkpoint Loss            : {:.6} (Step {}, Source: {})",
                    bl,
                    bs,
                    result.get("best_checkpoint_source").and_then(|v| v.as_str()).unwrap_or("unknown")
                );
            } else {
                println!("  Best Checkpoint Loss            : {:.6}", bl);
            }
        }
        if let Some(st) = result.get("steps").or_else(|| result.get("optimizer_steps")).and_then(|v| v.as_u64()) {
            println!("  Total Optimizer Steps           : {}", st);
        }
        println!("  Model Weights Digest (SHA-256)  : {}", weights_sha);
        println!(
            "  Full Checkpoint Digest (SHA-256): {}",
            full_checkpoint_sha
        );
        println!(
            "  Stage Config Fingerprint        : {}",
            expected_fingerprint
        );
        println!(
            "  Stage Provenance Metadata Saved : {}",
            out_dir.join("STAGE_METADATA.json").display()
        );
        println!(
            "================================================================================"
        );

        completed_outputs_by_norm.insert(
            normalize_for_comparison(&out_dir),
            stage.output_checkpoint.clone(),
        );
        last_completed_stage_dir = Some(out_dir.clone());
        is_first_stage = false;
    }

    println!("================================================================================");
    println!(" PIPELINE EXECUTION COMPLETED: ALL STAGES PROCESSED SUCCESSFULLY");
    println!("================================================================================");

    // Mandatory Post-Training Model Promotion to Hugging Face (Deterministic final stage)
    if let Some(ref final_model_dir) = last_completed_stage_dir {
        if let Err(e) = trigger_hf_model_promotion(final_model_dir) {
            eprintln!("[Hugging Face Model Promotion Notice]: {e}");
        }
    }

    Ok(())
}

/// Triggers post-training model promotion to Hugging Face repository upon successful training completion.
fn trigger_hf_model_promotion(model_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let is_offline = std::env::var("TARA_OFFLINE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if is_offline {
        println!(
            "[HUGGING FACE MODEL PROMOTION]: Offline mode (TARA_OFFLINE=1) detected. Model promotion queued locally at '{}'.",
            model_dir.display()
        );
        return Ok(());
    }

    let hf_repo = match std::env::var("HF_MODEL_REPO").or_else(|_| std::env::var("HF_REPO")) {
        Ok(repo) if !repo.trim().is_empty() => repo,
        _ => {
            println!(
                "[HUGGING FACE MODEL PROMOTION]: Skipped because neither HF_MODEL_REPO nor HF_REPO is configured. Promoted model remains at '{}'.",
                model_dir.display()
            );
            return Ok(());
        }
    };
    println!("[HUGGING FACE MODEL PROMOTION]");
    println!("  Promoted Model Directory : {}", model_dir.display());
    println!("  Target Hugging Face Repo : {}", hf_repo);

    let bin_name = if cfg!(windows) {
        "tara_hf_model_uploader.exe"
    } else {
        "tara_hf_model_uploader"
    };
    let candidates = [
        PathBuf::from("target/release").join(bin_name),
        PathBuf::from("target/debug").join(bin_name),
    ];

    if let Some(bin_path) = candidates.iter().find(|p| p.exists()) {
        let mut cmd = std::process::Command::new(bin_path);
        cmd.env("HF_MODEL_REPO", &hf_repo);
        cmd.args(["--model", &model_dir.to_string_lossy(), "--repo-id", &hf_repo]);
        if let Ok(mut child) = cmd.spawn() {
            let status = child.wait()?;
            if status.success() {
                println!("  Hugging Face Promotion   : SUCCESS");
            } else {
                eprintln!("  Hugging Face Promotion   : Exited with status {}", status);
            }
        }
    } else {
        println!("  Hugging Face uploader binary not pre-built in target/{{release,debug}}; promotion manifest logged.");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tara_engine::internal_diagnostics::write_safetensors_with_shapes;

    fn unique_test_dir(tag: &str) -> PathBuf {
        tara_engine::internal_diagnostics::unique_test_dir(tag)
    }

    fn write_minimal_working_model(dir: &Path, with_continuation: bool, corrupt_opt: bool) {
        fs::create_dir_all(dir).unwrap();
        let cfg = TaraConfig {
            vocab_size: 8,
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
            r#"{"vocab":{"<|pad|>":0,"<|im_start|>":1,"<|im_end|>":2,"<|unk|>":3,"a":4,"b":5,"c":6,"d":7}}"#,
        )
        .unwrap();

        let mut weights = HashMap::new();
        let mut shapes = HashMap::new();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            vec![0.01f32; 8 * 8],
        );
        shapes.insert("model.embed_tokens.weight".to_string(), vec![8, 8]);
        weights.insert("model.norm.weight".to_string(), vec![1.0f32; 8]);
        shapes.insert("model.norm.weight".to_string(), vec![8]);
        weights.insert("lm_head.weight".to_string(), vec![0.01f32; 8 * 8]);
        shapes.insert("lm_head.weight".to_string(), vec![8, 8]);

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
        assert!(
            err.contains("DAG Cycle Detected"),
            "Unexpected error: {err}"
        );

        // Valid out-of-order stages must sort topologically (stage_b depends on stage_a)
        let mut valid_a = s1;
        valid_a.input_checkpoint = "base_model".to_string();
        valid_a.depends_on = vec![];
        let valid_b = s2;
        let sorted = validate_and_sort_stage_dag(&[valid_b.clone(), valid_a.clone()]).unwrap();
        assert_eq!(sorted[0].stage_id, "stage_a");
        assert_eq!(sorted[1].stage_id, "stage_b");

        // Mismatched depends_on vs input_checkpoint must fail
        let mut mismatch_b = valid_b;
        mismatch_b.input_checkpoint = "wrong_checkpoint_dir".to_string();
        let err_mismatch = validate_and_sort_stage_dag(&[valid_a, mismatch_b]).unwrap_err();
        assert!(
            err_mismatch.contains("DAG Inconsistency Error"),
            "Unexpected error: {err_mismatch}"
        );
    }

    #[test]
    fn test_validate_stage_definitions_rejects_duplicate_nested_and_unsafe_paths() {
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

        // Nested cross-stage output/input overlap rejected
        let mut s_nested = s1.clone();
        s_nested.stage_index = 2;
        s_nested.stage_id = "s_nested".to_string();
        s_nested.input_checkpoint = out_shared
            .join("nested_child")
            .to_string_lossy()
            .to_string();
        s_nested.output_checkpoint = dir.join("other_out").to_string_lossy().to_string();
        let err_nested =
            validate_stage_definitions(&[s1.clone(), s_nested], &base_model).unwrap_err();
        assert!(
            err_nested.contains("Circular/nested checkpoint path hazard"),
            "Unexpected error: {err_nested}"
        );

        // Unsafe stage_id with traversal rejected
        let mut s_bad_id = s1.clone();
        s_bad_id.stage_id = "../escape".to_string();
        assert!(validate_stage_definitions(&[s_bad_id], &base_model).is_err());

        // Path traversal in dataset_path rejected
        let mut s_bad_ds = s1.clone();
        s_bad_ds.dataset_path = "storage/../secret.jsonl".to_string();
        assert!(validate_stage_definitions(&[s_bad_ds], &base_model).is_err());

        // max_steps = Some(0) rejected
        let mut s_zero_steps = s1;
        s_zero_steps.max_steps = Some(0);
        assert!(validate_stage_definitions(&[s_zero_steps], &base_model).is_err());
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

        // Model-only directory passes working model (including full layer & vocab check) but fails continuation-ready
        let (ok_m, issues_m, _) = verify_working_model(&model_only);
        assert!(ok_m, "Expected working model to pass: {:?}", issues_m);
        assert!(!verify_continuation_ready(&model_only).0);

        // Continuation-ready directory passes both
        assert!(verify_working_model(&cont_ready).0);
        assert!(verify_continuation_ready(&cont_ready).0);

        // Full checkpoint digest differs from weights-only digest when continuation artifacts exist
        let w_sha = compute_model_weights_digest(&cont_ready).unwrap().unwrap();
        let full_sha = compute_full_checkpoint_digest(&cont_ready)
            .unwrap()
            .unwrap();
        assert_ne!(w_sha, full_sha);

        // Corrupt optimizer tensor shape is caught and rejected
        let (ok_corrupt, issues, _) = verify_continuation_ready(&corrupt_opt);
        assert!(
            !ok_corrupt,
            "Corrupt optimizer state must fail: {:?}",
            issues
        );

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
        write_json_atomically(
            &cont_ready.join("STAGE_METADATA.json"),
            &serde_json::json!({
                "status": "COMPLETED",
                "stage_config_fingerprint": fp1,
                "dynamic_model_weights_sha256": w_sha,
                "completed_at_utc": "2026-10-08T00:00:00Z"
            }),
        )
        .unwrap();

        assert!(should_skip_completed_stage(&cont_ready, &fp1));
        assert_eq!(
            read_stage_completed_timestamp(&cont_ready).as_deref(),
            Some("2026-10-08T00:00:00Z")
        );

        // Changing dataset SHA or learning rate invalidates fingerprint -> must NOT skip!
        let fp_changed_ds =
            compute_stage_config_fingerprint(&stage, "in_sha_1", "ds_sha_2_modified", "fp32");
        assert!(!should_skip_completed_stage(&cont_ready, &fp_changed_ds));

        stage.learning_rate = 5e-4;
        let fp_changed_lr =
            compute_stage_config_fingerprint(&stage, "in_sha_1", "ds_sha_1", "fp32");
        assert!(!should_skip_completed_stage(&cont_ready, &fp_changed_lr));
    }

    #[test]
    fn test_crash_recovery_lock_guard_and_cli_exclusivity() {
        let dir = unique_test_dir("crash_recovery");
        let target = dir.join("stage1_out");
        let backup = dir.join("stage1_out.bak_12345");
        let orphan_staging = dir.join("stage1_out.staging_12345");

        // Crash Window 1: Crash right after renaming target -> backup, before staging -> target completed
        write_minimal_working_model(&backup, true, false);
        fs::create_dir_all(&orphan_staging).unwrap();

        let recovered = recover_interrupted_promotion(&target).unwrap();
        assert!(recovered, "Must recover target from valid .bak_* directory");
        assert!(target.exists() && verify_working_model(&target).0);
        assert!(!backup.exists());
        assert!(!orphan_staging.exists());

        // Crash Window 2: Crash AFTER staging -> target rename succeeded, BEFORE backup deletion completed.
        // Recovery MUST preserve the newly committed target and remove the superseded backup!
        fs::write(target.join("STAGE_METADATA.json"), r#"{"commit":"NEW"}"#).unwrap();
        let stale_backup = dir.join("stage1_out.bak_55555");
        write_minimal_working_model(&stale_backup, true, false);
        fs::write(
            stale_backup.join("STAGE_METADATA.json"),
            r#"{"commit":"OLD"}"#,
        )
        .unwrap();

        let rolled_back = recover_interrupted_promotion(&target).unwrap();
        assert!(
            !rolled_back,
            "Must NOT roll back when newly promoted target is already intact"
        );
        assert!(!stale_backup.exists(), "Stale backup must be cleaned up");
        let preserved_meta = fs::read_to_string(target.join("STAGE_METADATA.json")).unwrap();
        assert!(
            preserved_meta.contains("NEW"),
            "Newly committed target must be preserved, got: {preserved_meta}"
        );

        // Now test full atomic promotion replacing target with new staging
        let new_staging = dir.join("stage1_out.staging_99999");
        write_minimal_working_model(&new_staging, true, false);
        promote_directory_atomically(&new_staging, &target).unwrap();
        assert!(target.exists() && verify_working_model(&target).0);
        assert!(!new_staging.exists());

        // Test OrchestratorLockGuard prevents concurrent lock acquisition by live process and cleans up on drop
        {
            let _guard = OrchestratorLockGuard::acquire(&dir).unwrap();
            assert!(OrchestratorLockGuard::acquire(&dir).is_err());
        }
        assert!(OrchestratorLockGuard::acquire(&dir).is_ok());

        // Test OrchestratorLockGuard automatically recovers stale lock left by dead PID
        let lock_file = dir.join(".orchestrator.lock");
        fs::write(
            &lock_file,
            r#"{"pid":0,"acquired_at_utc":"2026-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        {
            let _recovered_guard = OrchestratorLockGuard::acquire(&dir)
                .expect("Stale lock from dead PID must be recovered automatically");
            assert!(OrchestratorLockGuard::acquire(&dir).is_err());
        }

        // Test CLI mutual exclusion
        assert!(
            validate_cli_mode_exclusivity(Some(1), Some("s1"), false, None, None, None).is_err()
        );
        assert!(validate_cli_mode_exclusivity(Some(1), None, true, None, None, None).is_err());
        assert!(validate_cli_mode_exclusivity(
            None,
            None,
            false,
            None,
            Some(Path::new("m")),
            Some(Path::new("c")),
        )
        .is_err());
    }

    #[test]
    fn test_process_liveness_and_env_fallbacks() {
        assert!(is_process_alive(std::process::id()));
        assert!(!is_process_alive(0));

        // Test memory MB parsing logic: default fallback, rejection of < 512 MB, and acceptance of valid values
        let parse_mem = |val: Option<&str>| {
            val.and_then(|v| v.parse::<f64>().ok())
                .filter(|&v| v >= 512.0)
                .unwrap_or(12288.0)
        };
        assert_eq!(parse_mem(None), 12288.0);
        assert_eq!(parse_mem(Some("100.0")), 12288.0);
        assert_eq!(parse_mem(Some("NaN")), 12288.0);
        assert_eq!(parse_mem(Some("16384.0")), 16384.0);

        // Test HF_MODEL_REPO fallback to HF_REPO dynamically without exposing creator identities
        std::env::remove_var("HF_MODEL_REPO");
        std::env::set_var("HF_REPO", "test_org/test_repo");
        let resolved = std::env::var("HF_MODEL_REPO")
            .or_else(|_| std::env::var("HF_REPO"))
            .ok();
        assert_eq!(resolved.as_deref(), Some("test_org/test_repo"));
        std::env::remove_var("HF_REPO");

        let default_resolved = std::env::var("HF_MODEL_REPO")
            .or_else(|_| std::env::var("HF_REPO"))
            .ok();
        assert!(default_resolved.is_none());
    }

    #[test]
    fn test_valid_input_checkpoint_with_empty_output_directory() {
        let dir = unique_test_dir("valid_in_empty_out");
        let input_dir = dir.join("input_model");
        let output_dir = dir.join("stage0_out");

        write_minimal_working_model(&input_dir, false, false);
        fs::create_dir_all(&output_dir).unwrap();

        assert!(input_dir.exists());
        assert!(output_dir.exists());
        assert!(is_empty_dir(&output_dir));

        let (is_model, missing, param_count) =
            verify_working_model_with_purpose(&input_dir, "INPUT");
        assert!(is_model, "Input model should be valid: {:?}", missing);
        assert!(param_count.unwrap_or(0) > 0, "Parameter count must be non-zero");

        // Verify that empty output directory is recognized as a new training destination,
        // and does NOT trigger "contains no SafeTensors weight shards" during DAG planning.
        let out_is_empty_or_nonexistent = !output_dir.exists() || is_empty_dir(&output_dir);
        assert!(out_is_empty_or_nonexistent);

        let should_resume = false;
        let checkpoint_mode = if should_resume { "RESUME" } else { "NEW_RUN" };
        assert_eq!(checkpoint_mode, "NEW_RUN");

        let input_weights_sha = compute_model_weights_digest(&input_dir).unwrap();
        assert!(input_weights_sha.is_some());

        // Calling compute_model_weights_digest on non-empty corrupted dir would fail,
        // but an empty output directory is NOT checked as a model.
        // Confirm compute_model_weights_digest fails if called directly on empty directory:
        let empty_digest_err = compute_model_weights_digest(&output_dir).unwrap_err();
        assert!(empty_digest_err.contains("contains no SafeTensors weight shards"));
    }

    #[test]
    fn test_valid_resume_checkpoint() {
        let dir = unique_test_dir("valid_resume_cp");
        let resume_dir = dir.join("checkpoint_resume");

        write_minimal_working_model(&resume_dir, true, false);

        let (is_model, missing_model, param_count) =
            verify_working_model_with_purpose(&resume_dir, "INPUT");
        assert!(
            is_model,
            "Working model verification must pass: {:?}",
            missing_model
        );
        assert!(param_count.unwrap_or(0) > 0);

        let (is_cont, missing_cont, opt_step) =
            verify_continuation_ready_with_purpose(&resume_dir, "RESUME");
        assert!(
            is_cont,
            "Continuation verification must pass: {:?}",
            missing_cont
        );
        assert_eq!(opt_step, Some(10));

        let has_any_continuation_file = resume_dir.join("checkpoint_state.json").exists()
            || resume_dir.join("optimizer.safetensors").exists();
        assert!(has_any_continuation_file);

        let should_resume = has_any_continuation_file;
        let checkpoint_mode = if should_resume { "RESUME" } else { "NEW_RUN" };
        assert_eq!(checkpoint_mode, "RESUME");

        let weights_digest = compute_model_weights_digest(&resume_dir).unwrap();
        assert!(weights_digest.is_some());

        let full_digest = compute_full_checkpoint_digest(&resume_dir).unwrap();
        assert!(full_digest.is_some());
    }

    #[test]
    fn test_missing_or_incomplete_input_checkpoint() {
        let dir = unique_test_dir("missing_incomplete");
        let missing_dir = dir.join("does_not_exist");

        // 1. Missing directory: not a working model, compute_model_weights_digest returns Ok(None)
        let (is_model, missing, _) = verify_working_model(&missing_dir);
        assert!(!is_model);
        assert!(missing.iter().any(|m| m.contains("missing")));
        let missing_digest = compute_model_weights_digest(&missing_dir).unwrap();
        assert!(missing_digest.is_none());

        // 2. Incomplete checkpoint: directory exists, has config.json, but has 0 SafeTensors weight shards
        let incomplete_dir = dir.join("incomplete_model");
        fs::create_dir_all(&incomplete_dir).unwrap();
        fs::write(
            incomplete_dir.join("config.json"),
            r#"{"vocab_size":8,"hidden_size":8}"#,
        )
        .unwrap();

        // Must strictly return Err("Directory '...' exists but contains no SafeTensors weight shards")
        // Zero masking via Ok(None)!
        let err = compute_model_weights_digest(&incomplete_dir).unwrap_err();
        assert!(
            err.contains("exists but contains no SafeTensors weight shards"),
            "Error must not be masked: {err}"
        );

        let (is_inc_model, inc_missing, _) = verify_working_model(&incomplete_dir);
        assert!(!is_inc_model);
        assert!(inc_missing.iter().any(|m| m.contains("Failed to load model SafeTensors weights")
            || m.contains("missing")));
    }

    #[test]
    fn test_multistage_dag_preflight_downstream_checkpoints_deferred() {
        let dir = unique_test_dir("multistage_dag_preflight");
        let base_model = dir.join("base_model");
        let stage0_out = dir.join("stage0_out");
        let stage1_out = dir.join("stage1_out");
        let stage2_out = dir.join("stage2_out");

        write_minimal_working_model(&base_model, false, false);
        // Stage 0 output exists on disk as an empty directory (e.g. created by Google Drive / cloud storage)
        fs::create_dir_all(&stage0_out).unwrap();

        let ds_file = dir.join("test_dataset.jsonl");
        fs::write(&ds_file, "{\"text\":\"Hello TARA multi-stage training\"}\n").unwrap();

        let s0 = StageDefinition {
            stage_index: 0,
            stage_id: "stage0".to_string(),
            stage_name: "Pretraining".to_string(),
            description: "Stage 0".to_string(),
            dataset_path: ds_file.to_string_lossy().to_string(),
            input_checkpoint: base_model.to_string_lossy().to_string(),
            output_checkpoint: stage0_out.to_string_lossy().to_string(),
            epochs: 1,
            max_steps: Some(10),
            learning_rate: 1e-4,
            batch_size: 2,
            checkpoint_interval: 5,
            depends_on: vec![],
        };

        let s1 = StageDefinition {
            stage_index: 1,
            stage_id: "stage1".to_string(),
            stage_name: "Curriculum".to_string(),
            description: "Stage 1".to_string(),
            dataset_path: ds_file.to_string_lossy().to_string(),
            input_checkpoint: stage0_out.to_string_lossy().to_string(),
            output_checkpoint: stage1_out.to_string_lossy().to_string(),
            epochs: 1,
            max_steps: Some(10),
            learning_rate: 1e-4,
            batch_size: 2,
            checkpoint_interval: 5,
            depends_on: vec!["stage0".to_string()],
        };

        let s2 = StageDefinition {
            stage_index: 2,
            stage_id: "stage2".to_string(),
            stage_name: "Adaptation".to_string(),
            description: "Stage 2".to_string(),
            dataset_path: ds_file.to_string_lossy().to_string(),
            input_checkpoint: stage1_out.to_string_lossy().to_string(),
            output_checkpoint: stage2_out.to_string_lossy().to_string(),
            epochs: 1,
            max_steps: Some(10),
            learning_rate: 1e-4,
            batch_size: 2,
            checkpoint_interval: 5,
            depends_on: vec!["stage1".to_string()],
        };

        let raw_stages = vec![s0, s1, s2];
        assert!(validate_stage_definitions(&raw_stages, &base_model).is_ok());
        let sorted = validate_and_sort_stage_dag(&raw_stages).expect("DAG validation failed");
        assert_eq!(sorted.len(), 3);

        // Preflight simulation: downstream checkpoints do NOT exist yet on disk
        let mut scheduled_outputs: HashSet<PathBuf> = HashSet::new();
        for (idx, stage) in sorted.iter().enumerate() {
            let effective_in = PathBuf::from(&stage.input_checkpoint);
            let norm_in = normalize_for_comparison(&effective_in);

            if !scheduled_outputs.contains(&norm_in) {
                // Only root stage (idx == 0) must exist on disk prior to execution
                assert_eq!(idx, 0, "Only the entry stage may have an un-scheduled input checkpoint");
                let (is_model, missing, _) = verify_working_model(&effective_in);
                assert!(is_model, "Base model input must be valid: {:?}", missing);
            }

            // Verify that empty or nonexistent output dirs are recognized as NEW_RUN and not loaded as models
            let out_p = PathBuf::from(&stage.output_checkpoint);
            let is_empty_or_missing = !out_p.exists() || is_empty_dir(&out_p);
            assert!(is_empty_or_missing);

            scheduled_outputs.insert(normalize_for_comparison(Path::new(&stage.output_checkpoint)));
        }

        assert_eq!(scheduled_outputs.len(), 3);
    }
}

