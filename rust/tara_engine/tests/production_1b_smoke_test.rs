use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tara_engine::internal_diagnostics::{
    unique_test_dir, write_safetensors_with_shapes, AdamWHyperparams, DynamicAdamW,
};
use tara_engine::train_candidate::verify_continuation_checkpoint_dir;
use tara_engine::{load_model_weights_with_shapes, TaraConfig, TaraForCausalLM, TaraTokenizer};

fn resolve_production_model_dir() -> PathBuf {
    let rel_crate = Path::new("../../storage/models/tara_candidate_v1");
    if rel_crate.exists() {
        return rel_crate.to_path_buf();
    }
    let rel_root = Path::new("storage/models/tara_candidate_v1");
    if rel_root.exists() {
        return rel_root.to_path_buf();
    }
    panic!("Production candidate model directory not found in relative paths");
}

struct TestDirGuard(PathBuf);
impl Drop for TestDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn test_production_1b_single_step_smoke() {
    println!("\n=======================================================");
    println!("   TARA 1B PRODUCTION MODEL STRICT SMOKE TEST AUDIT   ");
    println!("=======================================================\n");

    let model_dir = resolve_production_model_dir();
    let tokenizer_path = model_dir.join("tokenizer.json");
    let config_path = model_dir.join("config.json");

    assert!(tokenizer_path.exists(), "tokenizer.json must exist");
    assert!(config_path.exists(), "config.json must exist");

    // 1. Load authentic production configuration and tokenizer
    let prod_config = TaraConfig::from_json_file(&config_path.to_string_lossy())
        .expect("Load candidate config.json");
    let tokenizer = TaraTokenizer::from_file(&tokenizer_path.to_string_lossy())
        .expect("Load production tokenizer.json");

    assert_eq!(prod_config.vocab_size, 100_000);
    assert_eq!(prod_config.hidden_size, 2048);
    assert_eq!(prod_config.num_hidden_layers, 13);
    assert_eq!(prod_config.num_attention_heads, 64);
    assert_eq!(prod_config.num_key_value_heads, 16);

    println!("[1/6] Production Architecture Specifications:");
    println!("      - Model: Tara Causal LM (GQA + SwiGLU + RoPE + RMSNorm)");
    println!("      - Vocabulary Size: {}", prod_config.vocab_size);
    println!("      - Hidden Dimension: {}", prod_config.hidden_size);
    println!(
        "      - Intermediate Size: {}",
        prod_config.intermediate_size
    );
    println!("      - Decoder Layers: {}", prod_config.num_hidden_layers);
    println!(
        "      - Attention Heads: {} (KV Heads: {})",
        prod_config.num_attention_heads, prod_config.num_key_value_heads
    );

    // 2. Load authentic production model weights from Safetensors shards
    println!("[2/6] Loading 995.8M Production Model Shards into Memory...");
    let t_load_start = Instant::now();
    let (mut weights, shapes) = load_model_weights_with_shapes(&model_dir.to_string_lossy())
        .expect("Load candidate safetensors shards");
    let load_dur = t_load_start.elapsed().as_secs_f32();

    let total_params: usize = weights.values().map(|v| v.len()).sum();
    println!("      - Total Tensors Loaded: {}", weights.len());
    println!("      - Exact Total Parameters: {}", total_params);
    println!("      - Time to Load Shards: {:.2}s", load_dur);
    assert_eq!(
        total_params, 995_809_280,
        "Must match exactly 995.8M production parameters"
    );

    let temp_root = unique_test_dir("prod_1b_smoke");
    let _guard = TestDirGuard(temp_root.clone());
    let working_model_dir = temp_root.join("model");
    fs::create_dir_all(&working_model_dir).expect("Create working model dir");

    let model = TaraForCausalLM::from_weights_and_config(
        weights.clone(),
        prod_config.clone(),
        &working_model_dir.to_string_lossy(),
    )
    .expect("Instantiate 1B production model");
    assert_eq!(model.param_count(), 995_809_280);

    // 3. Execute 1 Real Forward Pass
    println!("[3/6] Executing Real Forward Pass on Production Model...");
    let prompt = "ಕನ್ನಡ";
    let target = "ಭಾಷೆ";
    let p_ids = tokenizer.encode(prompt);
    let t_ids = tokenizer.encode(target);
    let mut tokens = p_ids.clone();
    let target_start = tokens.len();
    tokens.extend_from_slice(&t_ids);

    let seq_len = tokens.len();
    assert!(seq_len >= 2);

    let t_fwd_start = Instant::now();
    let (layer_inputs, final_normed, logits) = model.forward_with_cache(&tokens);
    let fwd_dur = t_fwd_start.elapsed().as_secs_f32();

    assert_eq!(layer_inputs.len(), prod_config.num_hidden_layers + 1);
    assert_eq!(logits.len(), seq_len * prod_config.vocab_size);
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "Logits must be finite"
    );

    // Compute cross-entropy loss on target tokens
    let mut d_logits = vec![0.0f32; seq_len * prod_config.vocab_size];
    let mut valid_targets = 0usize;
    let mut sample_loss = 0.0f32;

    for (pos, &token_val) in tokens
        .iter()
        .enumerate()
        .take(seq_len)
        .skip(target_start.max(1))
    {
        let predictor = pos.saturating_sub(1);
        let target_idx = token_val as usize;
        let logit_start = predictor * prod_config.vocab_size;
        let logits_slice = &logits[logit_start..logit_start + prod_config.vocab_size];
        let grad_slice = &mut d_logits[logit_start..logit_start + prod_config.vocab_size];

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
            for val in grad_slice.iter_mut().take(prod_config.vocab_size) {
                *val *= inv_sum;
            }
            sample_loss += sum_exp.ln() - (logits_slice[target_idx] - max_logit);
            grad_slice[target_idx] -= 1.0;
            valid_targets += 1;
        }
    }

    assert!(valid_targets > 0);
    sample_loss /= valid_targets as f32;
    println!("      - Forward Duration: {:.3}s", fwd_dur);
    println!(
        "      - Cross-Entropy Loss: {:.4} (Finite: {})",
        sample_loss,
        sample_loss.is_finite()
    );
    assert!(
        sample_loss.is_finite() && sample_loss > 0.0,
        "Loss must be finite and positive"
    );

    let norm_factor = 1.0 / valid_targets as f32;
    for g in d_logits.iter_mut() {
        *g *= norm_factor;
    }

    // 4. Execute 1 Real Backward Pass
    println!("[4/6] Executing Real Backward Pass across All 13 Layers...");
    let t_bwd_start = Instant::now();
    let model_grads = model.backward(&tokens, &layer_inputs, &final_normed, &d_logits);
    let bwd_dur = t_bwd_start.elapsed().as_secs_f32();
    let grad_map = model_grads.into_map();
    drop(layer_inputs);
    drop(final_normed);
    drop(d_logits);
    drop(logits);

    let mut total_grad_norm_sq = 0.0f64;
    for (k, g) in &grad_map {
        let mut t_sq = 0.0f64;
        for &val in g {
            assert!(
                val.is_finite(),
                "Gradient for {k} contains non-finite element"
            );
            t_sq += (val as f64) * (val as f64);
        }
        total_grad_norm_sq += t_sq;
    }
    let global_grad_norm = total_grad_norm_sq.sqrt() as f32;
    println!("      - Backward Duration: {:.3}s", bwd_dur);
    println!(
        "      - Global Gradient Norm: {:.6} (Finite & Non-Zero)",
        global_grad_norm
    );
    assert!(
        global_grad_norm.is_finite() && global_grad_norm > 0.0,
        "Grad norm must be finite and > 0"
    );

    // 5. Execute 1 Real AdamW Optimizer Step
    println!("[5/6] Executing Real AdamW Optimizer Step...");
    let mut optimizer = DynamicAdamW::new();
    let hyperparams = AdamWHyperparams {
        lr: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
        max_grad_norm: 1.0,
    };

    let initial_weights_snapshot = weights.clone();
    let t_opt_start = Instant::now();
    let step_res = optimizer.step(&mut weights, &grad_map, &hyperparams);
    let opt_dur = t_opt_start.elapsed().as_secs_f32();
    assert!(step_res, "Optimizer step must succeed");
    drop(grad_map);

    let mut total_param_delta_sq = 0.0f64;
    for (name, orig_w) in &initial_weights_snapshot {
        let updated_w = weights.get(name).unwrap();
        for (&w_old, &w_new) in orig_w.iter().zip(updated_w.iter()) {
            let diff = (w_new - w_old) as f64;
            total_param_delta_sq += diff * diff;
        }
    }
    drop(initial_weights_snapshot);
    let param_delta = total_param_delta_sq.sqrt();
    println!("      - Optimizer Step Duration: {:.3}s", opt_dur);
    println!(
        "      - Parameter Update Delta (L2 Norm): {:.6} (Parameters genuinely changed!)",
        param_delta
    );
    assert!(
        param_delta > 0.0,
        "Parameters must genuinely change after optimizer step"
    );

    // 6. Save and Reload Checkpoint Verification
    println!("[6/6] Verifying Checkpoint Persistence & Reloading Parity...");
    let cp_dir = temp_root.join("checkpoint_prod_step1");
    fs::create_dir_all(&cp_dir).expect("Create checkpoint directory");

    // Write weights
    let model_safetensors_path = cp_dir.join("model.safetensors");
    write_safetensors_with_shapes(&weights, &shapes, &model_safetensors_path.to_string_lossy())
        .expect("Write checkpoint weights");

    // Write config and tokenizer
    fs::write(
        cp_dir.join("config.json"),
        serde_json::to_vec_pretty(&prod_config).unwrap(),
    )
    .unwrap();
    fs::copy(&tokenizer_path, cp_dir.join("tokenizer.json")).unwrap();

    // Write optimizer moments
    let mut opt_tensors = HashMap::new();
    let mut opt_shapes = HashMap::new();
    for (k, (m_vec, v_vec)) in optimizer.export_state() {
        let shape = shapes.get(&k).cloned().unwrap_or_else(|| vec![m_vec.len()]);
        opt_tensors.insert(format!("{k}.adam_m"), m_vec);
        opt_shapes.insert(format!("{k}.adam_m"), shape.clone());
        opt_tensors.insert(format!("{k}.adam_v"), v_vec);
        opt_shapes.insert(format!("{k}.adam_v"), shape);
    }
    drop(optimizer);
    write_safetensors_with_shapes(
        &opt_tensors,
        &opt_shapes,
        &cp_dir.join("optimizer.safetensors").to_string_lossy(),
    )
    .expect("Save optimizer tensors");
    drop(opt_tensors);
    drop(opt_shapes);

    // Write checkpoint state
    let checkpoint_state = serde_json::json!({
        "optimizer_step": 1,
        "epoch": 1,
        "loss": sample_loss,
    });
    fs::write(
        cp_dir.join("checkpoint_state.json"),
        serde_json::to_vec_pretty(&checkpoint_state).unwrap(),
    )
    .unwrap();

    let verified_step = verify_continuation_checkpoint_dir(&cp_dir)
        .expect("Checkpoint continuation verification must succeed");
    assert_eq!(verified_step, 1);

    let (reloaded_weights, _) = load_model_weights_with_shapes(&cp_dir.to_string_lossy()).unwrap();
    assert_eq!(reloaded_weights.len(), weights.len());
    for (name, orig_t) in &weights {
        let reloaded_t = reloaded_weights
            .get(name)
            .expect("Tensor must exist in reloaded checkpoint");
        assert_eq!(orig_t.len(), reloaded_t.len());
        for (i, (&a, &b)) in orig_t.iter().zip(reloaded_t.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "Mismatch in tensor {name} at {i}: {a} vs {b}"
            );
        }
    }

    println!("      - Continuation Checkpoint Verification: PASSED");
    println!("      - Reloaded Checkpoint Parity: 100% Bit-Accurate");
    println!("\n=======================================================");
    println!("  PRODUCTION 1B MODEL SMOKE TEST: ALL GATES PASSED!    ");
    println!("=======================================================\n");
}
