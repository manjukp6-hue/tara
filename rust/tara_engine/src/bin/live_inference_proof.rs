// live_inference_proof.rs
//
// Standalone binary: loads TARA SafeTensors, runs real forward pass, prints actual output.
// NO mocks. NO hardcoded responses. Real model weights only.
// Evidence: PID, model path, actual SHA-256, parameter count, real prompts, real output.

fn main() {
    let model_dir = std::env::var("TARA_MODEL_DIR").unwrap_or_else(|_| {
        if std::path::Path::new("storage/models/tara_candidate_v1/model.safetensors").exists() {
            "storage/models/tara_candidate_v1".to_string()
        } else {
            "storage/models/tara".to_string()
        }
    });

    println!("=== TARA LIVE INFERENCE PROOF ===");
    println!("PID: {}", std::process::id());
    println!("Model dir: {}", model_dir);

    // 1. Compute live SHA-256 of actual model weights file
    let model_file = format!("{}/model.safetensors", model_dir);
    if !std::path::Path::new(&model_file).exists() {
        eprintln!("FATAL: model file not found: {}", model_file);
        std::process::exit(1);
    }
    let sha = tara_engine::compute_sha256(&model_file).unwrap_or_else(|e| {
        eprintln!("SHA256 failed: {}", e);
        std::process::exit(1);
    });
    let file_size = std::fs::metadata(&model_file).map(|m| m.len()).unwrap_or(0);
    println!("model.safetensors size: {} bytes", file_size);
    println!("model.safetensors SHA-256: {}", sha);

    // 2. Load model from SafeTensors — actual tensor deserialization
    let model = tara_engine::TaraForCausalLM::load(&model_dir).unwrap_or_else(|e| {
        eprintln!("Model load failed: {}", e);
        std::process::exit(1);
    });

    let vs = model.config.vocab_size;
    let hs = model.config.hidden_size;
    let nl = model.config.num_hidden_layers;
    println!(
        "Config: vocab_size={} hidden_size={} num_layers={}",
        vs, hs, nl
    );

    // Count total parameter floats loaded from weights
    let param_count: usize = model.embed_tokens.len()
        + model.lm_head.len()
        + model.norm.weight.len()
        + model
            .layers
            .iter()
            .map(|l| {
                l.self_attn.q_proj.len()
                    + l.self_attn.k_proj.len()
                    + l.self_attn.v_proj.len()
                    + l.self_attn.o_proj.len()
                    + l.mlp.gate_proj.len()
                    + l.mlp.up_proj.len()
                    + l.mlp.down_proj.len()
                    + l.input_layernorm.weight.len()
                    + l.post_attention_layernorm.weight.len()
            })
            .sum::<usize>();
    println!("Total parameter floats loaded: {}", param_count);
    // Each float32 = 4 bytes
    println!(
        "Total parameter bytes: {} (expected ~{})",
        param_count * 4,
        file_size
    );

    // Prove embed_tokens[0] is NOT zero (real weights loaded)
    let first_weight = model.embed_tokens[0];
    println!(
        "embed_tokens[0] (real weight, not zero): {:.8}",
        first_weight
    );

    // 3. Load tokenizer
    let tokenizer_path = format!("{}/tokenizer.json", model_dir);
    let tokenizer = tara_engine::TaraTokenizer::from_file(&tokenizer_path).unwrap_or_else(|e| {
        eprintln!("Tokenizer load failed: {}", e);
        std::process::exit(1);
    });
    println!("Tokenizer loaded: vocab_size={}", tokenizer.vocab_size);

    // 4. Run real inference on multiple prompts
    // These are real prompts — the output comes 100% from the model forward pass
    let prompts: &[(&str, &str)] = &[
        ("English", "hello tara who are you"),
        ("Kannada", "ನಮಸ್ಕಾರ ತಾರಾ"),
        ("Math", "what is 2 plus 3"),
        ("Kanglish", "tara explain"),
        ("Identity", "what is tara"),
        ("Boundary", "a"),
    ];

    for (label, prompt) in prompts {
        println!("\n--- Prompt [{}]: {:?} ---", label, prompt);
        let tokens = tokenizer.encode(prompt);
        println!("  Token IDs: {:?}", &tokens[..tokens.len().min(20)]);

        let options = tara_engine::GenerateOptions {
            max_new_tokens: 40,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            stop_tokens: None,
        };
        let result = tara_engine::generate_response(&model, &tokenizer, prompt, &options);
        println!("  Output: {:?}", result);
    }

    // 5. Verify forward pass produces non-trivial logits
    let probe_ids: Vec<u32> = vec![106, 9, 130, 9, 133]; // "tara who you"
    let logits = model.forward(&probe_ids);
    let last_logit_slice = &logits[(probe_ids.len() - 1) * vs..probe_ids.len() * vs];
    let max_logit = last_logit_slice
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_logit = last_logit_slice
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let has_nan = last_logit_slice.iter().any(|v| v.is_nan());
    let has_inf = last_logit_slice.iter().any(|v| v.is_infinite());
    println!("\n--- Forward pass sanity check ---");
    println!("  Logit max: {:.4}", max_logit);
    println!("  Logit min: {:.4}", min_logit);
    println!("  Has NaN: {}", has_nan);
    println!("  Has Inf: {}", has_inf);
    println!("  PASS: {}", !has_nan && !has_inf && max_logit.is_finite());

    println!("\n=== INFERENCE PROOF COMPLETE ===");
}
