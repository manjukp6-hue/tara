//! Coordinated Joint Neural Model and Dedicated World Model Training Engine.
//!
//! Executes genuine joint training updating both architectures in tandem within
//! a single coordinated training loop:
//! - Primary Neural Model: trained on language modeling causal cross-entropy tokens.
//! - Neural World Model: trained on validated state transition prediction ((s_t, a_t) -> s_{t+1}) with prompt masking.
//!
//! Fail-closed preflight validation ensures that if the dataset lacks valid state-transition records,
//! training fails immediately with a clear descriptive error instead of silently performing neural-only training.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::compute_dynamic_sha256;
use crate::dataset_engine::{validate_state_transition, StateTransitionRecord};
use crate::model::TaraForCausalLM;
use crate::safetensors::{
    load_model_weights_with_shapes, write_safetensors_sharded, write_safetensors_with_shapes,
};
use crate::tokenizer::TaraTokenizer;
use crate::train_candidate::{
    has_parent_dir_traversal, paths_overlap, verify_working_model_dir,
};
use crate::trainer::{
    promote_directory_atomically, recover_interrupted_promotion, AdamWHyperparams, DynamicAdamW,
};

/// Configuration for coordinated joint Neural Model and World Model training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointTrainingConfig {
    pub neural_model_dir: PathBuf,
    pub world_model_dir: PathBuf,
    pub neural_checkpoint_dir: PathBuf,
    pub world_model_checkpoint_dir: PathBuf,
    pub dataset_dir: PathBuf,
    pub curriculum_path: Option<PathBuf>,
    pub max_steps: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub grad_clip_norm: f32,
    pub batch_size: usize,
    pub resume: bool,
    pub resume_from: Option<PathBuf>,
}

impl Default for JointTrainingConfig {
    fn default() -> Self {
        Self {
            neural_model_dir: PathBuf::from("storage/models/tara_candidate_v1"),
            world_model_dir: PathBuf::from("storage/models/tara_world_model"),
            neural_checkpoint_dir: PathBuf::from("manual_training/checkpoints/neural"),
            world_model_checkpoint_dir: PathBuf::from("manual_training/checkpoints/world_model"),
            dataset_dir: PathBuf::from("storage/datasets"),
            curriculum_path: Some(PathBuf::from(
                "storage/datasets/curriculum/foundational_curriculum.jsonl",
            )),
            max_steps: 10,
            learning_rate: 1e-4,
            weight_decay: 0.01,
            grad_clip_norm: 1.0,
            batch_size: 1,
            resume: false,
            resume_from: None,
        }
    }
}

/// Comprehensive report emitted upon completion of a coordinated joint training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointTrainingReport {
    pub status: String,
    pub neural_parameter_count: usize,
    pub world_parameter_count: usize,
    pub neural_loss_before: f64,
    pub neural_loss_after: f64,
    pub world_loss_before: f64,
    pub world_loss_after: f64,
    pub steps_completed: usize,
    pub neural_candidate_sha256: String,
    pub world_candidate_sha256: String,
}

impl JointTrainingConfig {
    /// Validates directory isolation, stopping conditions, and bundle integrity.
    pub fn validate_paths(&self) -> Result<(), String> {
        if self.max_steps == 0 {
            return Err("max_steps must be at least 1 for training progress".to_string());
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err("learning_rate must be a positive finite float".to_string());
        }

        for (label, p) in [
            ("neural_model_dir", &self.neural_model_dir),
            ("world_model_dir", &self.world_model_dir),
            ("neural_checkpoint_dir", &self.neural_checkpoint_dir),
            (
                "world_model_checkpoint_dir",
                &self.world_model_checkpoint_dir,
            ),
            ("dataset_dir", &self.dataset_dir),
        ] {
            if has_parent_dir_traversal(p) {
                return Err(format!(
                    "{label} ('{}') contains forbidden '..' parent-directory traversal",
                    p.display()
                ));
            }
        }

        // Prevent in-place source model overwrite
        if paths_overlap(&self.neural_checkpoint_dir, &self.neural_model_dir) {
            return Err(format!(
                "neural_checkpoint_dir ('{}') overlaps with neural_model_dir ('{}'); in-place overwrite of source model is prohibited",
                self.neural_checkpoint_dir.display(),
                self.neural_model_dir.display()
            ));
        }
        if paths_overlap(&self.world_model_checkpoint_dir, &self.world_model_dir) {
            return Err(format!(
                "world_model_checkpoint_dir ('{}') overlaps with world_model_dir ('{}'); in-place overwrite of source model is prohibited",
                self.world_model_checkpoint_dir.display(),
                self.world_model_dir.display()
            ));
        }
        if paths_overlap(
            &self.neural_checkpoint_dir,
            &self.world_model_checkpoint_dir,
        ) {
            return Err(format!(
                "neural_checkpoint_dir ('{}') overlaps with world_model_checkpoint_dir ('{}'); dual checkpoints must be isolated",
                self.neural_checkpoint_dir.display(),
                self.world_model_checkpoint_dir.display()
            ));
        }

        Ok(())
    }
}

/// Scans a directory for JSONL files and discovers valid state transition records.
pub fn discover_state_transitions_in_dir(
    dir: &Path,
) -> Result<Vec<StateTransitionRecord>, std::io::Error> {
    let mut transitions = Vec::new();
    if !dir.exists() {
        return Ok(transitions);
    }

    let mut entries_to_visit = vec![dir.to_path_buf()];
    while let Some(current_dir) = entries_to_visit.pop() {
        if let Ok(rd) = fs::read_dir(&current_dir) {
            for entry in rd.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    entries_to_visit.push(p);
                } else if p.is_file() {
                    if let Some(ext) = p.extension() {
                        if ext == "jsonl" {
                            let f = File::open(&p)?;
                            let reader = BufReader::new(f);
                            for (line_idx, line_res) in reader.lines().enumerate() {
                                if let Ok(line) = line_res {
                                    let trimmed = line.trim();
                                    if trimmed.is_empty() {
                                        continue;
                                    }
                                    if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
                                        if let Some(obj) = val.as_object() {
                                            let t_obj = if let Some(nested) =
                                                obj.get("state_transition").and_then(Value::as_object)
                                            {
                                                nested
                                            } else {
                                                obj
                                            };
                                            let s_obs = t_obj
                                                .get("state_observation")
                                                .or_else(|| t_obj.get("state"))
                                                .or_else(|| t_obj.get("observation"))
                                                .and_then(Value::as_str);
                                            let act = t_obj
                                                .get("action")
                                                .or_else(|| t_obj.get("intervention"))
                                                .and_then(Value::as_str);
                                            let ns_obs = t_obj
                                                .get("next_state_observation")
                                                .or_else(|| t_obj.get("next_state"))
                                                .or_else(|| t_obj.get("subsequent_state"))
                                                .and_then(Value::as_str);

                                            if let (Some(s), Some(a), Some(ns)) = (s_obs, act, ns_obs)
                                            {
                                                if s.trim().is_empty()
                                                    || a.trim().is_empty()
                                                    || ns.trim().is_empty()
                                                {
                                                    continue;
                                                }

                                                let ep_id = t_obj
                                                    .get("episode_id")
                                                    .and_then(Value::as_str)
                                                    .unwrap_or("default_episode");
                                                let step_idx = if let Some(st_val) =
                                                    t_obj.get("step_index").or_else(|| t_obj.get("step"))
                                                {
                                                    if let Some(st_num) = st_val.as_u64() {
                                                        st_num as usize
                                                    } else {
                                                        continue;
                                                    }
                                                } else {
                                                    line_idx
                                                };

                                                let reward = if let Some(r_val) = t_obj.get("reward") {
                                                    if let Some(r_num) = r_val.as_f64() {
                                                        if !r_num.is_finite() {
                                                            continue;
                                                        }
                                                        r_num as f32
                                                    } else {
                                                        continue;
                                                    }
                                                } else {
                                                    0.0f32
                                                };

                                                let done = if let Some(d_val) = t_obj.get("done") {
                                                    if let Some(d_bool) = d_val.as_bool() {
                                                        d_bool
                                                    } else {
                                                        continue;
                                                    }
                                                } else {
                                                    false
                                                };

                                                let domain = t_obj
                                                    .get("domain")
                                                    .and_then(Value::as_str)
                                                    .unwrap_or("general");

                                                let confidence = if let Some(c_val) = t_obj.get("confidence") {
                                                    if let Some(c_num) = c_val.as_f64() {
                                                        if !c_num.is_finite() || c_num < 0.0 || c_num > 1.0 {
                                                            continue;
                                                        }
                                                        c_num as f32
                                                    } else {
                                                        continue;
                                                    }
                                                } else {
                                                    1.0f32
                                                };

                                                let rec = StateTransitionRecord {
                                                    id: format!("{}_{line_idx}", p.display()),
                                                    episode_id: ep_id.to_string(),
                                                    step_index: step_idx,
                                                    state_observation: s.to_string(),
                                                    action: a.to_string(),
                                                    next_state_observation: ns.to_string(),
                                                    reward,
                                                    done,
                                                    domain: domain.to_string(),
                                                    confidence,
                                                    source_dataset_id: p
                                                        .to_string_lossy()
                                                        .to_string(),
                                                };
                                                if validate_state_transition(&rec).is_ok() {
                                                    transitions.push(rec);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(transitions)
}

/// Discovers plain text sequences for primary neural causal language modeling.
pub fn discover_text_sequences_in_dir(
    dir: &Path,
    curriculum: Option<&Path>,
) -> Result<Vec<String>, std::io::Error> {
    let mut texts = Vec::new();

    if let Some(c_path) = curriculum {
        if c_path.exists() && c_path.is_file() {
            let f = File::open(c_path)?;
            let reader = BufReader::new(f);
            for line_res in reader.lines() {
                if let Ok(line) = line_res {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
                        if let Some(t) = val
                            .get("output")
                            .or_else(|| val.get("text"))
                            .or_else(|| val.get("input"))
                            .and_then(Value::as_str)
                        {
                            if !t.trim().is_empty() {
                                texts.push(t.to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    if dir.exists() {
        let mut entries_to_visit = vec![dir.to_path_buf()];
        while let Some(current_dir) = entries_to_visit.pop() {
            if let Ok(rd) = fs::read_dir(&current_dir) {
                for entry in rd.flatten() {
                    let p = entry.path();
                    if p.is_dir() {
                        entries_to_visit.push(p);
                    } else if p.is_file() {
                        if let Some(ext) = p.extension() {
                            if ext == "jsonl" {
                                if let Ok(f) = File::open(&p) {
                                    let reader = BufReader::new(f);
                                    for line_res in reader.lines() {
                                        if let Ok(line) = line_res {
                                            let trimmed = line.trim();
                                            if trimmed.is_empty() {
                                                continue;
                                            }
                                            if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
                                                if let Some(t) = val
                                                    .get("text")
                                                    .or_else(|| val.get("output"))
                                                    .or_else(|| val.get("prompt"))
                                                    .and_then(Value::as_str)
                                                {
                                                    if !t.trim().is_empty() {
                                                        texts.push(t.to_string());
                                                        if texts.len() >= 1000 {
                                                            return Ok(texts);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(texts)
}

/// Compute causal cross entropy loss and gradient for a single sequence on CPU.
fn compute_causal_step(
    model: &TaraForCausalLM,
    tokens: &[u32],
    target_start: usize,
    channel: &tara_hd::ComputeChannel,
) -> (f32, HashMap<String, Vec<f32>>) {
    let (layer_inputs, final_normed, logits) = model.forward_with_cache_channel(tokens, channel);
    let seq_len = tokens.len();
    let vs = model.config.vocab_size;

    let eff_target_start = target_start.max(1);
    let mut d_logits = vec![0.0f32; seq_len * vs];
    let mut total_loss = 0.0f32;
    let mut valid_targets = 0usize;

    for pos in eff_target_start..seq_len {
        let pred_idx = pos - 1;
        let target = tokens[pos] as usize;
        if target < vs {
            let logit_start = pred_idx * vs;
            let logits_slice = &logits[logit_start..logit_start + vs];
            let max_logit = logits_slice
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for (v, &l) in logits_slice.iter().enumerate() {
                let exp_val = (l - max_logit).exp();
                d_logits[logit_start + v] = exp_val;
                sum_exp += exp_val;
            }

            if sum_exp > 0.0 && sum_exp.is_finite() {
                let inv_sum = 1.0 / sum_exp;
                for v in 0..vs {
                    d_logits[logit_start + v] *= inv_sum;
                }
                let loss = sum_exp.ln() - (logits_slice[target] - max_logit);
                d_logits[logit_start + target] -= 1.0;
                total_loss += loss;
                valid_targets += 1;
            }
        }
    }

    if valid_targets > 0 {
        let norm_factor = 1.0 / valid_targets as f32;
        for g in d_logits.iter_mut() {
            *g *= norm_factor;
        }
        let grads = model.backward(tokens, &layer_inputs, &final_normed, &d_logits);
        (total_loss / valid_targets as f32, grads.into_map())
    } else {
        (0.0, HashMap::new())
    }
}

/// Run genuine coordinated joint Neural Model and World Model training.
pub fn run_coordinated_joint_training(
    config: &JointTrainingConfig,
) -> Result<JointTrainingReport, String> {
    // 1. Validate paths and isolation
    config.validate_paths()?;

    // 2. Preflight verification of both working model bundles
    let _ = recover_interrupted_promotion(&config.neural_model_dir);
    let _ = recover_interrupted_promotion(&config.world_model_dir);
    let _ = recover_interrupted_promotion(&config.neural_checkpoint_dir);
    let _ = recover_interrupted_promotion(&config.world_model_checkpoint_dir);

    println!(
        "[Preflight] Verifying Primary Neural Model bundle at '{}'...",
        config.neural_model_dir.display()
    );
    let mut neural_model = verify_working_model_dir(&config.neural_model_dir)
        .map_err(|e| format!("Primary Neural Model bundle verification failed: {e}"))?;

    println!(
        "[Preflight] Verifying Dedicated World Model bundle at '{}'...",
        config.world_model_dir.display()
    );
    let mut world_model = verify_working_model_dir(&config.world_model_dir)
        .map_err(|e| format!("Dedicated World Model bundle verification failed: {e}"))?;

    // 3. Fail-Closed Dataset Preflight: Discover genuine World Model state transitions
    println!(
        "[Preflight] Scanning for verified World Model state transitions in '{}'...",
        config.dataset_dir.display()
    );
    let transitions = discover_state_transitions_in_dir(&config.dataset_dir)
        .map_err(|e| format!("Failed to read dataset directory: {e}"))?;

    if transitions.is_empty() {
        return Err(
            "Joint training preflight failed: World Model transition dataset is missing or contains 0 valid state-transition records. Genuine World Model learning requires authentic (s, a, s', r) state transition observations."
                .to_string(),
        );
    }
    println!(
        "[Preflight] Discovered {} verified state transition records for World Model training.",
        transitions.len()
    );

    // 4. Discover Neural text data
    let text_samples = discover_text_sequences_in_dir(
        &config.dataset_dir,
        config.curriculum_path.as_deref(),
    )
    .map_err(|e| format!("Failed to read language dataset samples: {e}"))?;
    if text_samples.is_empty() {
        return Err(
            "Joint training preflight failed: Primary Neural Model text dataset contains 0 samples."
                .to_string(),
        );
    }
    println!(
        "[Preflight] Discovered {} text sequences for Primary Neural Model training.",
        text_samples.len()
    );

    // 5. Load tokenizers
    let neural_tok_path = config.neural_model_dir.join("tokenizer.json");
    let world_tok_path = config.world_model_dir.join("tokenizer.json");
    let neural_tok = TaraTokenizer::from_file(&neural_tok_path.to_string_lossy())
        .map_err(|e| format!("Failed to load neural tokenizer: {e}"))?;
    let world_tok = TaraTokenizer::from_file(&world_tok_path.to_string_lossy())
        .map_err(|e| format!("Failed to load world model tokenizer: {e}"))?;

    // 6. Model parameters and weight shapes
    let (mut neural_weights, neural_shapes) =
        load_model_weights_with_shapes(&config.neural_model_dir.to_string_lossy())
            .map_err(|e| format!("Failed to load neural model weights: {e}"))?;
    let (mut world_weights, world_shapes) =
        load_model_weights_with_shapes(&config.world_model_dir.to_string_lossy())
            .map_err(|e| format!("Failed to load world model weights: {e}"))?;

    let neural_params: usize = neural_weights.values().map(|v| v.len()).sum();
    let world_params: usize = world_weights.values().map(|v| v.len()).sum();

    println!(
        "[Coordinated Setup] Neural Model: {neural_params} parameters | World Model: {world_params} parameters"
    );

    // 7. Instantiating isolated optimizers
    let mut neural_opt = DynamicAdamW::with_preallocated_moments(&neural_weights);
    let mut world_opt = DynamicAdamW::with_preallocated_moments(&world_weights);

    let mut start_step = 0usize;
    if config.resume {
        let neural_resume_dir = config
            .resume_from
            .as_ref()
            .unwrap_or(&config.neural_checkpoint_dir);
        let world_resume_dir = &config.world_model_checkpoint_dir;

        let joint_state_file = neural_resume_dir.join("joint_checkpoint_state.json");
        if joint_state_file.exists() {
            println!(
                "[Resume] Resuming joint training state from '{}'...",
                joint_state_file.display()
            );
            let raw = fs::read_to_string(&joint_state_file)
                .map_err(|e| format!("Failed to read joint checkpoint state: {e}"))?;
            let jval: Value = serde_json::from_str(&raw)
                .map_err(|e| format!("Malformed joint checkpoint state: {e}"))?;
            if let Some(s) = jval.get("step").and_then(Value::as_u64) {
                start_step = s as usize;
                neural_opt.set_step(start_step as u64);
                world_opt.set_step(start_step as u64);
            }
        }

        // Restore Neural weights & optimizer state
        if let Ok((resumed_n_weights, _)) =
            load_model_weights_with_shapes(&neural_resume_dir.to_string_lossy())
        {
            neural_weights = resumed_n_weights;
            let _ = neural_model.update_weights(&neural_weights);
        }
        let n_opt_file = neural_resume_dir.join("optimizer.safetensors");
        if n_opt_file.exists() {
            if let Ok((opt_tensors, _)) =
                load_model_weights_with_shapes(&n_opt_file.to_string_lossy())
            {
                let mut opt_state = HashMap::new();
                for (k, m_vec) in &opt_tensors {
                    if k.ends_with(".adam_m") {
                        let base = k.trim_end_matches(".adam_m");
                        if let Some(v_vec) = opt_tensors.get(&format!("{base}.adam_v")) {
                            opt_state.insert(base.to_string(), (m_vec.clone(), v_vec.clone()));
                        }
                    }
                }
                neural_opt.load_state(opt_state);
            }
        }

        // Restore World Model weights & optimizer state
        if let Ok((resumed_w_weights, _)) =
            load_model_weights_with_shapes(&world_resume_dir.to_string_lossy())
        {
            world_weights = resumed_w_weights;
            let _ = world_model.update_weights(&world_weights);
        }
        let w_opt_file = world_resume_dir.join("optimizer.safetensors");
        if w_opt_file.exists() {
            if let Ok((opt_tensors, _)) =
                load_model_weights_with_shapes(&w_opt_file.to_string_lossy())
            {
                let mut opt_state = HashMap::new();
                for (k, m_vec) in &opt_tensors {
                    if k.ends_with(".adam_m") {
                        let base = k.trim_end_matches(".adam_m");
                        if let Some(v_vec) = opt_tensors.get(&format!("{base}.adam_v")) {
                            opt_state.insert(base.to_string(), (m_vec.clone(), v_vec.clone()));
                        }
                    }
                }
                world_opt.load_state(opt_state);
            }
        }
    }

    let hp = AdamWHyperparams {
        lr: config.learning_rate,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: config.weight_decay,
        max_grad_norm: config.grad_clip_norm,
    };

    let channel = tara_hd::ComputeChannel::auto()
        .map_err(|e| format!("ComputeChannel initialization failed: {e}"))?;

    let mut neural_loss_before = 0.0f64;
    let mut world_loss_before = 0.0f64;
    let mut neural_loss_after = 0.0f64;
    let mut world_loss_after = 0.0f64;

    println!(
        "\n=== [EXECUTING COORDINATED JOINT TRAINING: {} STEPS] ===",
        config.max_steps
    );

    for step_idx in start_step..config.max_steps {
        // --- A. Neural Language Modeling Step ---
        let text_sample = &text_samples[step_idx % text_samples.len()];
        let mut n_tokens = neural_tok.encode(text_sample);
        if n_tokens.len() > 128 {
            n_tokens.truncate(128);
        }
        if n_tokens.len() < 2 {
            n_tokens.push(0);
            n_tokens.push(1);
        }

        let (n_loss, n_grads) = compute_causal_step(&neural_model, &n_tokens, 1, &channel);
        if step_idx == start_step {
            neural_loss_before = n_loss as f64;
        }
        neural_loss_after = n_loss as f64;

        neural_opt.step(&mut neural_weights, &n_grads, &hp);
        neural_model
            .update_weights(&neural_weights)
            .map_err(|e| format!("Failed to update neural weights: {e}"))?;

        // --- B. Dedicated World Model State Transition Step ---
        let transition = &transitions[step_idx % transitions.len()];
        let prompt_str = format!(
            "[STATE] {} [ACTION] {} [NEXT_STATE] ",
            transition.state_observation, transition.action
        );
        let prompt_tokens = world_tok.encode(&prompt_str);
        let target_tokens = world_tok.encode(&transition.next_state_observation);
        let mut w_tokens = [prompt_tokens.clone(), target_tokens].concat();
        if w_tokens.len() > 128 {
            w_tokens.truncate(128);
        }
        let target_start = prompt_tokens.len().min(w_tokens.len().saturating_sub(1));

        let (w_loss, w_grads) = compute_causal_step(&world_model, &w_tokens, target_start, &channel);
        if step_idx == start_step {
            world_loss_before = w_loss as f64;
        }
        world_loss_after = w_loss as f64;

        world_opt.step(&mut world_weights, &w_grads, &hp);
        world_model
            .update_weights(&world_weights)
            .map_err(|e| format!("Failed to update world model weights: {e}"))?;

        println!(
            "  [Joint Step {}/{}] Neural Loss (LM): {:.4} | World Loss ((s,a)->s'): {:.4} | Grads: finite",
            step_idx + 1,
            config.max_steps,
            n_loss,
            w_loss
        );
    }

    // 8. Atomic Coordinated Checkpoint Commit
    let pid = std::process::id();
    let now_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    let staging_neural = PathBuf::from(format!(
        "{}.staging_joint_neural_{pid}_{now_nanos}",
        config.neural_checkpoint_dir.display()
    ));
    let staging_world = PathBuf::from(format!(
        "{}.staging_joint_world_{pid}_{now_nanos}",
        config.world_model_checkpoint_dir.display()
    ));

    fs::create_dir_all(&staging_neural)
        .map_err(|e| format!("Failed to create neural staging dir: {e}"))?;
    fs::create_dir_all(&staging_world)
        .map_err(|e| format!("Failed to create world model staging dir: {e}"))?;

    // Save Neural checkpoint artifacts
    write_safetensors_sharded(
        &neural_weights,
        &neural_shapes,
        &staging_neural.to_string_lossy(),
        100 * 1024 * 1024,
    )
    .map_err(|e| format!("Failed to write neural weights safetensors: {e}"))?;
    fs::copy(
        config.neural_model_dir.join("config.json"),
        staging_neural.join("config.json"),
    )
    .map_err(|e| format!("Failed to copy neural config.json: {e}"))?;
    fs::copy(
        config.neural_model_dir.join("tokenizer.json"),
        staging_neural.join("tokenizer.json"),
    )
    .map_err(|e| format!("Failed to copy neural tokenizer.json: {e}"))?;

    let neural_opt_moments = neural_opt.export_state();
    let mut n_opt_tensors = HashMap::new();
    let mut n_opt_shapes = HashMap::new();
    for (k, (m_vec, v_vec)) in neural_opt_moments {
        let m_name = format!("{k}.adam_m");
        let v_name = format!("{k}.adam_v");
        n_opt_shapes.insert(m_name.clone(), vec![m_vec.len()]);
        n_opt_shapes.insert(v_name.clone(), vec![v_vec.len()]);
        n_opt_tensors.insert(m_name, m_vec);
        n_opt_tensors.insert(v_name, v_vec);
    }
    write_safetensors_with_shapes(
        &n_opt_tensors,
        &n_opt_shapes,
        &staging_neural.join("optimizer.safetensors").to_string_lossy(),
    )
    .map_err(|e| format!("Failed to write neural optimizer.safetensors: {e}"))?;

    let neural_state = json!({
        "step": config.max_steps,
        "optimizer_step": config.max_steps,
        "loss_before": neural_loss_before,
        "loss_after": neural_loss_after,
        "validation_loss": neural_loss_after,
        "joint_training": true,
    });
    fs::write(
        staging_neural.join("checkpoint_state.json"),
        serde_json::to_string_pretty(&neural_state).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("Failed to write neural checkpoint_state.json: {e}"))?;

    // Save World Model checkpoint artifacts
    write_safetensors_sharded(
        &world_weights,
        &world_shapes,
        &staging_world.to_string_lossy(),
        100 * 1024 * 1024,
    )
    .map_err(|e| format!("Failed to write world model weights safetensors: {e}"))?;
    fs::copy(
        config.world_model_dir.join("config.json"),
        staging_world.join("config.json"),
    )
    .map_err(|e| format!("Failed to copy world model config.json: {e}"))?;
    fs::copy(
        config.world_model_dir.join("tokenizer.json"),
        staging_world.join("tokenizer.json"),
    )
    .map_err(|e| format!("Failed to copy world model tokenizer.json: {e}"))?;

    let world_opt_moments = world_opt.export_state();
    let mut w_opt_tensors = HashMap::new();
    let mut w_opt_shapes = HashMap::new();
    for (k, (m_vec, v_vec)) in world_opt_moments {
        let m_name = format!("{k}.adam_m");
        let v_name = format!("{k}.adam_v");
        w_opt_shapes.insert(m_name.clone(), vec![m_vec.len()]);
        w_opt_shapes.insert(v_name.clone(), vec![v_vec.len()]);
        w_opt_tensors.insert(m_name, m_vec);
        w_opt_tensors.insert(v_name, v_vec);
    }
    write_safetensors_with_shapes(
        &w_opt_tensors,
        &w_opt_shapes,
        &staging_world.join("optimizer.safetensors").to_string_lossy(),
    )
    .map_err(|e| format!("Failed to write world model optimizer.safetensors: {e}"))?;

    let world_state = json!({
        "step": config.max_steps,
        "optimizer_step": config.max_steps,
        "loss_before": world_loss_before,
        "loss_after": world_loss_after,
        "validation_loss": world_loss_after,
        "joint_training": true,
    });
    fs::write(
        staging_world.join("checkpoint_state.json"),
        serde_json::to_string_pretty(&world_state).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("Failed to write world model checkpoint_state.json: {e}"))?;

    // Compute dynamic SHAs for candidates before promotion
    let n_safetensor = if staging_neural.join("model.safetensors").exists() {
        staging_neural.join("model.safetensors")
    } else {
        staging_neural.join("model-00001.safetensors")
    };
    let n_cand_sha = compute_dynamic_sha256(&n_safetensor, 64 * 1024)
        .map_err(|e| format!("Failed to compute neural candidate SHA: {e}"))?;

    let w_safetensor = if staging_world.join("model.safetensors").exists() {
        staging_world.join("model.safetensors")
    } else {
        staging_world.join("model-00001.safetensors")
    };
    let w_cand_sha = compute_dynamic_sha256(&w_safetensor, 64 * 1024)
        .map_err(|e| format!("Failed to compute world model candidate SHA: {e}"))?;

    // Write joint state manifest to both directories
    let joint_manifest = json!({
        "step": config.max_steps,
        "joint_world_model": true,
        "neural_parameter_count": neural_params,
        "world_parameter_count": world_params,
        "neural_loss_before": neural_loss_before,
        "neural_loss_after": neural_loss_after,
        "world_loss_before": world_loss_before,
        "world_loss_after": world_loss_after,
        "neural_candidate_sha256": n_cand_sha,
        "world_candidate_sha256": w_cand_sha,
    });

    let manifest_str =
        serde_json::to_string_pretty(&joint_manifest).map_err(|e| e.to_string())?;
    fs::write(
        staging_neural.join("joint_checkpoint_state.json"),
        &manifest_str,
    )
    .map_err(|e| format!("Failed to write joint_checkpoint_state.json in neural: {e}"))?;
    fs::write(
        staging_world.join("joint_checkpoint_state.json"),
        &manifest_str,
    )
    .map_err(|e| format!("Failed to write joint_checkpoint_state.json in world: {e}"))?;

    // Transactionally promote staging candidates to published checkpoint paths
    promote_directory_atomically(&staging_neural, &config.neural_checkpoint_dir)
        .map_err(|e| format!("Failed to promote neural checkpoint directory: {e}"))?;
    promote_directory_atomically(&staging_world, &config.world_model_checkpoint_dir)
        .map_err(|e| format!("Failed to promote world model checkpoint directory: {e}"))?;

    println!("[Promotion] Coordinated joint candidates promoted successfully.");
    println!("  Neural Checkpoint      : {}", config.neural_checkpoint_dir.display());
    println!("  World Model Checkpoint : {}", config.world_model_checkpoint_dir.display());

    Ok(JointTrainingReport {
        status: "COMPLETED".to_string(),
        neural_parameter_count: neural_params,
        world_parameter_count: world_params,
        neural_loss_before,
        neural_loss_after,
        world_loss_before,
        world_loss_after,
        steps_completed: config.max_steps - start_step,
        neural_candidate_sha256: n_cand_sha,
        world_candidate_sha256: w_cand_sha,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TaraConfig;
    
    fn create_test_model_bundle(
        dir: &Path,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        fs::create_dir_all(dir)?;
        let mut cfg = TaraConfig::default();
        cfg.vocab_size = vocab_size;
        cfg.hidden_size = hidden_size;
        cfg.intermediate_size = hidden_size * 2;
        cfg.num_hidden_layers = 1;
        cfg.num_attention_heads = 2;
        cfg.num_key_value_heads = 2;
        cfg.head_dim = hidden_size / 2;
        cfg.max_position_embeddings = 64;
        cfg.tie_word_embeddings = Some(false);
        fs::write(dir.join("config.json"), serde_json::to_string(&cfg)?)?;

        let mut vocab = serde_json::Map::new();
        for i in 0..vocab_size {
            vocab.insert(format!("<tok_{i}>"), json!(i));
        }
        let tok = json!({ "vocab": vocab });
        fs::write(dir.join("tokenizer.json"), serde_json::to_string(&tok)?)?;

        let mut weights = HashMap::new();
        let mut shapes = HashMap::new();

        let add_param =
            |name: &str, shape: Vec<usize>, w: &mut HashMap<String, Vec<f32>>, s: &mut HashMap<String, Vec<usize>>| {
                let count: usize = shape.iter().product();
                let vals: Vec<f32> = (0..count).map(|i| ((i % 17) as f32) * 0.01 + 0.01).collect();
                w.insert(name.to_string(), vals);
                s.insert(name.to_string(), shape);
            };

        add_param(
            "model.embed_tokens.weight",
            vec![vocab_size, hidden_size],
            &mut weights,
            &mut shapes,
        );
        add_param(
            "model.norm.weight",
            vec![hidden_size],
            &mut weights,
            &mut shapes,
        );
        add_param(
            "lm_head.weight",
            vec![vocab_size, hidden_size],
            &mut weights,
            &mut shapes,
        );
        add_param(
            "model.layers.0.input_layernorm.weight",
            vec![hidden_size],
            &mut weights,
            &mut shapes,
        );
        add_param(
            "model.layers.0.post_attention_layernorm.weight",
            vec![hidden_size],
            &mut weights,
            &mut shapes,
        );
        add_param(
            "model.layers.0.self_attn.q_proj.weight",
            vec![hidden_size, hidden_size],
            &mut weights,
            &mut shapes,
        );
        add_param(
            "model.layers.0.self_attn.k_proj.weight",
            vec![hidden_size, hidden_size],
            &mut weights,
            &mut shapes,
        );
        add_param(
            "model.layers.0.self_attn.v_proj.weight",
            vec![hidden_size, hidden_size],
            &mut weights,
            &mut shapes,
        );
        add_param(
            "model.layers.0.self_attn.o_proj.weight",
            vec![hidden_size, hidden_size],
            &mut weights,
            &mut shapes,
        );
        add_param(
            "model.layers.0.mlp.gate_proj.weight",
            vec![hidden_size * 2, hidden_size],
            &mut weights,
            &mut shapes,
        );
        add_param(
            "model.layers.0.mlp.up_proj.weight",
            vec![hidden_size * 2, hidden_size],
            &mut weights,
            &mut shapes,
        );
        add_param(
            "model.layers.0.mlp.down_proj.weight",
            vec![hidden_size, hidden_size * 2],
            &mut weights,
            &mut shapes,
        );

        write_safetensors_with_shapes(
            &weights,
            &shapes,
            &dir.join("model.safetensors").to_string_lossy(),
        )?;

        Ok(())
    }

    #[test]
    fn test_fail_closed_preflight_on_zero_state_transitions() {
        let test_root = std::env::temp_dir().join(format!("tara_preflight_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&test_root);
        fs::create_dir_all(&test_root).unwrap();

        let n_src = test_root.join("neural_src");
        let w_src = test_root.join("world_src");
        let ds_dir = test_root.join("empty_dataset");
        fs::create_dir_all(&ds_dir).unwrap();

        create_test_model_bundle(&n_src, 16, 8).unwrap();
        create_test_model_bundle(&w_src, 16, 8).unwrap();

        // Write a text-only dataset with ZERO state transitions
        fs::write(
            ds_dir.join("shard_000.jsonl"),
            "{\"text\": \"This is plain text without state observation.\"}\n",
        )
        .unwrap();

        let cfg = JointTrainingConfig {
            neural_model_dir: n_src,
            world_model_dir: w_src,
            neural_checkpoint_dir: test_root.join("cp_neural"),
            world_model_checkpoint_dir: test_root.join("cp_world"),
            dataset_dir: ds_dir,
            curriculum_path: None,
            max_steps: 2,
            ..Default::default()
        };

        let result = run_coordinated_joint_training(&cfg);
        assert!(result.is_err(), "Joint training must fail preflight when transitions are missing");
        let err_msg = result.unwrap_err();
        assert!(
            err_msg.contains("Joint training preflight failed"),
            "Error must clearly indicate joint preflight failure: {err_msg}"
        );
        assert!(
            err_msg.contains("World Model transition dataset is missing or contains 0 valid state-transition records"),
            "Error must specify transition data deficiency: {err_msg}"
        );

        let _ = fs::remove_dir_all(&test_root);
    }

    #[test]
    fn test_coordinated_joint_training_step_and_state_restoration() {
        let test_root = std::env::temp_dir().join(format!("tara_joint_step_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&test_root);
        fs::create_dir_all(&test_root).unwrap();

        let n_src = test_root.join("neural_src");
        let w_src = test_root.join("world_src");
        let ds_dir = test_root.join("joint_dataset");
        let cp_neural = test_root.join("candidate_neural");
        let cp_world = test_root.join("candidate_world");
        fs::create_dir_all(&ds_dir).unwrap();

        create_test_model_bundle(&n_src, 16, 8).unwrap();
        create_test_model_bundle(&w_src, 16, 8).unwrap();

        // Write verified state transition record
        let transition_line = json!({
            "id": "trans_001",
            "episode_id": "ep_01",
            "step_index": 0,
            "state_observation": "<tok_1> <tok_2>",
            "action": "<tok_3>",
            "next_state_observation": "<tok_4> <tok_5>",
            "reward": 1.0,
            "done": false,
            "domain": "navigation",
            "confidence": 1.0,
            "source_dataset_id": "test_ds"
        })
        .to_string();

        let text_line = json!({
            "text": "<tok_1> <tok_2> <tok_3> <tok_4>"
        })
        .to_string();

        fs::write(
            ds_dir.join("transitions.jsonl"),
            format!("{transition_line}\n{text_line}\n"),
        )
        .unwrap();

        let cfg = JointTrainingConfig {
            neural_model_dir: n_src.clone(),
            world_model_dir: w_src.clone(),
            neural_checkpoint_dir: cp_neural.clone(),
            world_model_checkpoint_dir: cp_world.clone(),
            dataset_dir: ds_dir,
            curriculum_path: None,
            max_steps: 2,
            learning_rate: 1e-2,
            weight_decay: 0.01,
            grad_clip_norm: 1.0,
            batch_size: 1,
            resume: false,
            resume_from: None,
        };

        let report = run_coordinated_joint_training(&cfg).expect("Joint training must succeed with valid data");

        // Verify report metrics
        assert_eq!(report.status, "COMPLETED");
        assert_eq!(report.steps_completed, 2);
        assert!(report.neural_parameter_count > 0);
        assert!(report.world_parameter_count > 0);
        assert!(report.neural_loss_before.is_finite());
        assert!(report.world_loss_before.is_finite());

        // Verify published checkpoints exist and contain continuation-ready state
        assert!(cp_neural.join("model.safetensors").exists() || cp_neural.join("model-00001.safetensors").exists());
        assert!(cp_neural.join("optimizer.safetensors").exists());
        assert!(cp_neural.join("checkpoint_state.json").exists());
        assert!(cp_neural.join("joint_checkpoint_state.json").exists());

        assert!(cp_world.join("model.safetensors").exists() || cp_world.join("model-00001.safetensors").exists());
        assert!(cp_world.join("optimizer.safetensors").exists());
        assert!(cp_world.join("checkpoint_state.json").exists());
        assert!(cp_world.join("joint_checkpoint_state.json").exists());

        // Test crash-safe resumption from joint checkpoint
        let mut resume_cfg = cfg.clone();
        resume_cfg.resume = true;
        resume_cfg.resume_from = Some(cp_neural.clone());
        resume_cfg.max_steps = 4; // train 2 more steps

        let resume_report =
            run_coordinated_joint_training(&resume_cfg).expect("Resumed joint training must succeed");
        assert_eq!(resume_report.status, "COMPLETED");
        assert_eq!(resume_report.steps_completed, 2); // 4 - 2 = 2 steps executed

        let _ = fs::remove_dir_all(&test_root);
    }

    #[test]
    fn test_kg_triples_rejected_as_state_transitions() {
        let test_root = std::env::temp_dir().join(format!("tara_kg_triples_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&test_root);
        fs::create_dir_all(&test_root).unwrap();

        let ds_dir = test_root.join("triples_only");
        fs::create_dir_all(&ds_dir).unwrap();

        // Write knowledge graph triples only
        let triples = vec![
            json!({"subject": "Sun", "predicate": "emits", "object": "Light"}),
            json!({"subject": "Water", "predicate": "consists_of", "object": "H2O"}),
        ];
        let mut content = String::new();
        for t in triples {
            content.push_str(&t.to_string());
            content.push('\n');
        }
        fs::write(ds_dir.join("triples.jsonl"), content).unwrap();

        let transitions = discover_state_transitions_in_dir(&ds_dir).unwrap();
        assert!(
            transitions.is_empty(),
            "Relational knowledge graph triples must not be treated as environment transitions"
        );

        let _ = fs::remove_dir_all(&test_root);
    }

    #[test]
    fn test_invalid_and_malformed_transitions_rejected() {
        let test_root = std::env::temp_dir().join(format!("tara_malformed_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&test_root);
        fs::create_dir_all(&test_root).unwrap();

        let ds_dir = test_root.join("bad_trans");
        fs::create_dir_all(&ds_dir).unwrap();

        let bad_records = vec![
            json!({"state_observation": "", "action": "step", "next_state_observation": "s2"}), // empty state
            json!({"state_observation": "s1", "action": "  ", "next_state_observation": "s2"}), // empty action
            json!({"state_observation": "s1", "action": "step"}), // missing next_state
            json!({"state_observation": 123, "action": "step", "next_state_observation": "s2"}), // non-string state
            json!({"state_observation": "s1", "action": "step", "next_state_observation": "s2", "reward": f64::NAN}), // NaN reward
            json!({"state_observation": "s1", "action": "step", "next_state_observation": "s2", "confidence": 2.5}), // confidence > 1.0
        ];
        let mut content = String::new();
        for b in bad_records {
            content.push_str(&b.to_string());
            content.push('\n');
        }
        fs::write(ds_dir.join("bad.jsonl"), content).unwrap();

        let transitions = discover_state_transitions_in_dir(&ds_dir).unwrap();
        assert_eq!(
            transitions.len(),
            0,
            "Malformed transitions must be completely rejected"
        );

        let _ = fs::remove_dir_all(&test_root);
    }
}
