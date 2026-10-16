use std::collections::HashSet;
use std::path::{Path, PathBuf};
use tara_engine::{
    generate_response, GenerateOptions, TaraForCausalLM, TaraTokenizer,
};

fn resolve_candidate_path() -> PathBuf {
    let rel_crate = Path::new("../../storage/models/tara_candidate_v1");
    if rel_crate.exists() {
        return rel_crate.to_path_buf();
    }
    let rel_root = Path::new("storage/models/tara_candidate_v1");
    if rel_root.exists() {
        return rel_root.to_path_buf();
    }
    panic!("Candidate model directory not found in relative paths");
}

fn compute_causal_cross_entropy(model: &TaraForCausalLM, tokens: &[u32]) -> Option<f64> {
    let vocab_size = model.config.vocab_size;
    if vocab_size < 2 || tokens.len() < 2 {
        return None;
    }

    let (_, _, logits) = model.forward_with_cache(tokens);
    if logits.len() != tokens.len() * vocab_size {
        return None;
    }

    let mut total_loss = 0.0f64;
    let mut count = 0usize;
    for (pos, &target_tok) in tokens.iter().enumerate().skip(1) {
        let pred_idx = pos - 1;
        let target = target_tok as usize;
        if target >= vocab_size {
            continue;
        }
        let slice = &logits[pred_idx * vocab_size..(pred_idx + 1) * vocab_size];
        let max_v = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f64 = slice.iter().map(|&v| ((v - max_v) as f64).exp()).sum();
        if sum_exp > 0.0 && sum_exp.is_finite() {
            total_loss += sum_exp.ln() - ((slice[target] - max_v) as f64);
            count += 1;
        }
    }

    if count > 0 {
        let avg = total_loss / count as f64;
        if avg.is_finite() {
            return Some(avg);
        }
    }
    None
}

fn analyze_repetition(tokens: &[u32]) -> (usize, usize, f64) {
    if tokens.is_empty() {
        return (0, 0, 0.0);
    }
    let total = tokens.len();
    let mut unique_set = HashSet::new();
    let mut max_consecutive = 1usize;
    let mut current_consecutive = 1usize;

    for (i, &tok) in tokens.iter().enumerate() {
        unique_set.insert(tok);
        if i > 0 {
            if tok == tokens[i - 1] {
                current_consecutive += 1;
                if current_consecutive > max_consecutive {
                    max_consecutive = current_consecutive;
                }
            } else {
                current_consecutive = 1;
            }
        }
    }

    let unique_ratio = unique_set.len() as f64 / total as f64;
    (unique_set.len(), max_consecutive, unique_ratio)
}

#[test]
fn test_live_checkpoint_load_generation_and_evaluation() {
    let dir = resolve_candidate_path();
    let dir_str = dir.to_string_lossy().to_string();

    println!("\n=======================================================");
    println!("      TARA CHECKPOINT LOAD & LIVE EVALUATION           ");
    println!("=======================================================");
    println!("Target checkpoint directory: {}", dir_str);

    // 1. Checkpoint load test
    println!("\n[1/3] CHECKPOINT LOAD TEST");
    let tok_path = dir.join("tokenizer.json");
    assert!(tok_path.exists(), "tokenizer.json must exist in candidate directory");

    let tokenizer = TaraTokenizer::from_file(&tok_path.to_string_lossy())
        .expect("Tokenizer must parse and load successfully");
    println!("  -> Tokenizer loaded successfully.");
    println!("     Vocab Size: {}", tokenizer.vocab_size);
    println!("     Max Token Length: {}", tokenizer.max_token_len);

    let model = TaraForCausalLM::load(&dir_str)
        .expect("Model weights and architecture must load successfully");
    println!("  -> Model loaded successfully.");
    println!("     Layers: {}", model.config.num_hidden_layers);
    println!("     Hidden Size: {}", model.config.hidden_size);
    println!("     Intermediate Size: {}", model.config.intermediate_size);
    println!("     Attention Heads: {}", model.config.num_attention_heads);
    println!("     KV Heads: {}", model.config.num_key_value_heads);
    println!("     Head Dim: {}", model.config.effective_head_dim());

    let mut total_params = 0usize;
    let mut nan_weights = 0usize;
    let mut inf_weights = 0usize;

    for val in &model.embed_tokens {
        total_params += 1;
        if val.is_nan() { nan_weights += 1; }
        if val.is_infinite() { inf_weights += 1; }
    }
    for val in &model.norm.weight {
        total_params += 1;
        if val.is_nan() { nan_weights += 1; }
        if val.is_infinite() { inf_weights += 1; }
    }
    for val in &model.lm_head {
        total_params += 1;
        if val.is_nan() { nan_weights += 1; }
        if val.is_infinite() { inf_weights += 1; }
    }
    for layer in &model.layers {
        for val in &layer.self_attn.q_proj {
            total_params += 1;
            if val.is_nan() { nan_weights += 1; }
            if val.is_infinite() { inf_weights += 1; }
        }
        for val in &layer.self_attn.k_proj {
            total_params += 1;
            if val.is_nan() { nan_weights += 1; }
            if val.is_infinite() { inf_weights += 1; }
        }
        for val in &layer.self_attn.v_proj {
            total_params += 1;
            if val.is_nan() { nan_weights += 1; }
            if val.is_infinite() { inf_weights += 1; }
        }
        for val in &layer.self_attn.o_proj {
            total_params += 1;
            if val.is_nan() { nan_weights += 1; }
            if val.is_infinite() { inf_weights += 1; }
        }
        for val in &layer.input_layernorm.weight {
            total_params += 1;
            if val.is_nan() { nan_weights += 1; }
            if val.is_infinite() { inf_weights += 1; }
        }
        for val in &layer.post_attention_layernorm.weight {
            total_params += 1;
            if val.is_nan() { nan_weights += 1; }
            if val.is_infinite() { inf_weights += 1; }
        }
        for val in &layer.mlp.gate_proj {
            total_params += 1;
            if val.is_nan() { nan_weights += 1; }
            if val.is_infinite() { inf_weights += 1; }
        }
        for val in &layer.mlp.up_proj {
            total_params += 1;
            if val.is_nan() { nan_weights += 1; }
            if val.is_infinite() { inf_weights += 1; }
        }
        for val in &layer.mlp.down_proj {
            total_params += 1;
            if val.is_nan() { nan_weights += 1; }
            if val.is_infinite() { inf_weights += 1; }
        }
    }

    println!("     Total Inspected Parameters: {}", total_params);
    println!("     NaN Weights: {}", nan_weights);
    println!("     Inf Weights: {}", inf_weights);
    assert_eq!(nan_weights, 0, "No weights may be NaN");
    assert_eq!(inf_weights, 0, "No weights may be Inf");

    // 2. Generation Demo across Kannada, English, and Code
    println!("\n[2/3] GENERATION DEMO");
    let test_prompts = [
        ("Kannada", "ನಮಸ್ಕಾರ, ನೀನು ಯಾರು?"),
        ("English", "The fundamental principles of computation are"),
        ("Code", "fn fibonacci(n: u32) -> u64 {"),
    ];

    let gen_opts = GenerateOptions {
        max_new_tokens: 16,
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        repetition_penalty: 1.15,
        stop_tokens: None,
    };

    let mut all_generated_tokens: Vec<(String, Vec<u32>, String)> = Vec::new();

    for (label, prompt) in &test_prompts {
        println!("\n  --- Generating for {} prompt ---", label);
        println!("  Prompt: \"{}\"", prompt);

        let input_ids = tokenizer.encode(prompt);
        println!("  Prompt token count: {}", input_ids.len());

        let res = generate_response(&model, &tokenizer, prompt, &gen_opts)
            .expect("Generation should complete without runtime error");

        let gen_ids = tokenizer.encode(&res.text);
        println!("  Generated token count: {}", res.token_count);
        println!("  Total latency: {:.2}ms ({:.2} tokens/sec)", res.total_latency_ms, res.tps);
        println!("  Generated output:\n--------------------------------------------------");
        println!("{}", res.text);
        println!("--------------------------------------------------");

        all_generated_tokens.push((label.to_string(), gen_ids, res.text.clone()));
    }

    // 3. Quick Evaluation: Loss finiteness, output emptiness, repetition analysis
    println!("\n[3/3] QUICK EVALUATION");

    // Loss computation on probe texts
    let probe_texts = [
        ("Kannada Probe", "ನಮಸ್ಕಾರ ತಾರಾ ಹೇಗಿದ್ದೀರಾ"),
        ("English Probe", "Computation and reasoning foundation."),
        ("Code Probe", "fn add(a: i32) -> i32 { a + 1 }"),
    ];

    println!("  [Loss Check]");
    for (name, text) in &probe_texts {
        let enc = tokenizer.encode(text);
        let loss_opt = compute_causal_cross_entropy(&model, &enc);
        match loss_opt {
            Some(l) => {
                println!("    {}: Loss = {:.4} (is_finite: {})", name, l, l.is_finite());
                assert!(l.is_finite(), "Loss must be finite");
                assert!(l > 0.0, "Loss must be strictly positive");
            }
            None => {
                println!("    {}: Loss could not be evaluated (sequence too short or vocab mismatch)", name);
            }
        }
    }

    // Emptiness and repetition check
    println!("\n  [Output Emptiness & Token Repetition Check]");
    for (label, gen_ids, text) in &all_generated_tokens {
        let is_empty = text.trim().is_empty();
        let (unique_count, max_consecutive, unique_ratio) = analyze_repetition(gen_ids);

        println!("    {} prompt evaluation:", label);
        println!("      - Output empty: {}", is_empty);
        println!("      - Unique tokens: {} / {}", unique_count, gen_ids.len());
        println!("      - Unique token ratio: {:.2}%", unique_ratio * 100.0);
        println!("      - Max consecutive identical token repetitions: {}", max_consecutive);

        let is_stuck_in_loop = max_consecutive > 10;
        println!("      - Stuck in single-token repetition loop: {}", is_stuck_in_loop);

        assert!(!is_empty, "Generated text should not be empty");
    }

    println!("\n=======================================================");
    println!("           EVALUATION COMPLETED SUCCESSFULLY           ");
    println!("=======================================================\n");
}
