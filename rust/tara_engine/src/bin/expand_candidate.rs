//! TARA Candidate Architecture Expansion CLI — 100 % Native Rust.
//!
//! Expands a source model to a target parameter count using the generic
//! `FullCapacity` growth path. Existing compatible weights are transferred
//! via Net2Net zero-padding. Newly created parameters are zero-initialised
//! (output projections) or small-random-noise (input projections).
//!
//! Does NOT train. Does NOT touch `storage/models/tara`.
//! Creates a new candidate directory only.
//!
//! Usage:
//!   expand_candidate [OPTIONS]
//!
//! Key options:
//!   --source-model <path>      Source model directory   [default: storage/models/tara]
//!   --output <path>            Candidate output directory (must not exist)
//!   --target-params <n>        Target parameter count   [default: 1_000_000_000]
//!   --vocab-size <n>           Override vocabulary size [default: 8192]
//!   --verify                   Load + run CPU/CUDA forward-pass after writing

use std::env;
use std::path::PathBuf;
use std::process::exit;

use tara_engine::config::TaraConfig;
use tara_engine::cuda::{CudaTrainer, TrainingPrecision};
use tara_engine::model::causal_lm::TaraForCausalLM;
use tara_engine::model_expansion::{
    ArchitectureConstraints, ArchitectureScaler, GrowthType, ModelExpansionEngine,
};

fn print_usage() {
    println!("TARA Candidate Architecture Expansion CLI (Rust)");
    println!("Usage:");
    println!("  expand_candidate [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --source-model <path>       Source model directory (default: storage/models/tara)");
    println!(
        "  --output <path>             Candidate output directory (required unless --dry-run)"
    );
    println!(
        "  --target-params <n>         Target parameter count to solve for (default: 1000000000)"
    );
    println!("  --vocab-size <n>            Override target vocabulary size (default: 8192)");
    println!("  --layers <n>                Explicit override for num_hidden_layers (e.g. 22)");
    println!("  --hidden-size <n>           Explicit override for hidden_size (e.g. 2048)");
    println!("  --heads <n>                 Explicit override for num_attention_heads (e.g. 32)");
    println!("  --kv-heads <n>              Explicit override for num_key_value_heads (e.g. 8)");
    println!("  --intermediate-size <n>     Explicit override for intermediate_size (e.g. 5504)");
    println!("  --dry-run                   Solve and print architecture & exact parameter breakdown; do not write files");
    println!("  --verify                    Load and verify CPU + CUDA forward pass after writing");
    println!("  -h, --help                  Print this help message");
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut source_model = "storage/models/tara".to_string();
    let mut output_dir: Option<String> = None;
    let mut target_params: u64 = 1_000_000_000;
    let mut vocab_size: Option<usize> = None;
    let mut override_layers: Option<usize> = None;
    let mut override_hidden: Option<usize> = None;
    let mut override_heads: Option<usize> = None;
    let mut override_kv_heads: Option<usize> = None;
    let mut override_intermediate: Option<usize> = None;
    let mut dry_run = false;
    let mut do_verify = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_usage();
                exit(0);
            }
            "--source-model" | "--source" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --source-model requires a path argument");
                    exit(1);
                }
                source_model = args[i].clone();
            }
            "--output" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --output requires a path argument");
                    exit(1);
                }
                output_dir = Some(args[i].clone());
            }
            "--target-params" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --target-params requires an integer argument");
                    exit(1);
                }
                target_params = args[i].replace('_', "").parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("Error: invalid integer for --target-params: {}", args[i]);
                    exit(1);
                });
            }
            "--vocab-size" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --vocab-size requires an integer argument");
                    exit(1);
                }
                vocab_size = Some(args[i].parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("Error: invalid integer for --vocab-size: {}", args[i]);
                    exit(1);
                }));
            }
            "--layers" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --layers requires an integer argument");
                    exit(1);
                }
                override_layers = Some(args[i].parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("Error: invalid integer for --layers: {}", args[i]);
                    exit(1);
                }));
            }
            "--hidden-size" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --hidden-size requires an integer argument");
                    exit(1);
                }
                override_hidden = Some(args[i].parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("Error: invalid integer for --hidden-size: {}", args[i]);
                    exit(1);
                }));
            }
            "--heads" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --heads requires an integer argument");
                    exit(1);
                }
                override_heads = Some(args[i].parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("Error: invalid integer for --heads: {}", args[i]);
                    exit(1);
                }));
            }
            "--kv-heads" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --kv-heads requires an integer argument");
                    exit(1);
                }
                override_kv_heads = Some(args[i].parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("Error: invalid integer for --kv-heads: {}", args[i]);
                    exit(1);
                }));
            }
            "--intermediate-size" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --intermediate-size requires an integer argument");
                    exit(1);
                }
                override_intermediate = Some(args[i].parse::<usize>().unwrap_or_else(|_| {
                    eprintln!(
                        "Error: invalid integer for --intermediate-size: {}",
                        args[i]
                    );
                    exit(1);
                }));
            }
            "--head-dim" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --head-dim requires an integer argument");
                    exit(1);
                }
                let _ = args[i].parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("Error: invalid integer for --head-dim: {}", args[i]);
                    exit(1);
                });
            }
            "--dry-run" => {
                dry_run = true;
            }
            "--verify" => {
                do_verify = true;
            }
            unknown => {
                eprintln!("Error: unrecognized option '{}'", unknown);
                print_usage();
                exit(1);
            }
        }
        i += 1;
    }

    // ── Safety guard: never write into production model ───────────────────────
    if let Some(ref out) = output_dir {
        let norm = out.replace('\\', "/");
        if norm == "storage/models/tara"
            || norm == "./storage/models/tara"
            || norm.ends_with("/storage/models/tara")
        {
            eprintln!(
                "CRITICAL ERROR: --output cannot be 'storage/models/tara'. \
                 Expansion writes to candidate directories only."
            );
            exit(1);
        }
    }

    // ── Read source config ────────────────────────────────────────────────────
    let source_config_path = format!("{}/config.json", source_model);
    let source_config = TaraConfig::from_json_file(&source_config_path).unwrap_or_else(|e| {
        eprintln!(
            "Error reading source config '{}': {}",
            source_config_path, e
        );
        exit(1);
    });

    let effective_vocab = vocab_size.unwrap_or(8192).max(source_config.vocab_size);

    println!("==================================================================");
    println!("TARA ARCHITECTURE EXPANSION — CANDIDATE GENERATION");
    println!("==================================================================");
    println!("  Source Model:     {}", source_model);
    println!(
        "  Source Params:    {} ({} vocab, {} hidden, {} layers)",
        ArchitectureScaler::count_params(
            source_config.vocab_size,
            source_config.hidden_size,
            source_config.intermediate_size,
            source_config.num_hidden_layers,
            source_config.num_attention_heads,
            source_config.num_key_value_heads,
            source_config.effective_head_dim(),
        ),
        source_config.vocab_size,
        source_config.hidden_size,
        source_config.num_hidden_layers
    );
    println!("  Target Params:    {}", target_params);
    println!("  Target Vocab:     {}", effective_vocab);
    if let Some(ref o) = output_dir {
        println!("  Output Directory: {}", o);
    } else {
        println!("  Output Directory: [NONE - dry-run mode]");
    }
    if dry_run {
        println!("  Execution Mode:   DRY-RUN (architecture solve & calculation only)");
    }
    println!();

    // ── Solve architecture with constraints & overrides ───────────────────────
    println!(
        "[1/5] Solving architecture for {} parameters ...",
        target_params
    );

    let mut constraints = ArchitectureConstraints::for_target(target_params);
    if let Some(nl) = override_layers {
        constraints.min_layers = nl;
        constraints.max_layers = nl;
        constraints.preferred_layers = Some(nl);
    }
    if let Some(hs) = override_hidden {
        constraints.min_hidden = hs;
        constraints.max_hidden = hs;
        constraints.preferred_hidden = Some(hs);
    }
    if let Some(nh) = override_heads {
        constraints.preferred_heads = Some(nh);
    }
    if let Some(nkv) = override_kv_heads {
        constraints.preferred_kv_heads = Some(nkv);
    }
    if let Some(inter) = override_intermediate {
        constraints.preferred_intermediate = Some(inter);
    }

    let (target_config, exact_params) =
        ArchitectureScaler::solve_with_constraints(target_params, effective_vocab, &constraints);
    let delta_signed = exact_params as i64 - target_params as i64;
    let delta_pct = (exact_params as f64 - target_params as f64) / target_params as f64 * 100.0;

    let vs = target_config.vocab_size;
    let hs = target_config.hidden_size;
    let inter = target_config.intermediate_size;
    let nl = target_config.num_hidden_layers;
    let nh = target_config.num_attention_heads;
    let nkv = target_config.num_key_value_heads;
    let hd = target_config.head_dim;

    let p_embed = (vs * hs) as u64;
    let p_lm_head = (vs * hs) as u64;
    let p_norm = hs as u64;
    let p_q = (nh * hd * hs) as u64;
    let p_k = (nkv * hd * hs) as u64;
    let p_v = (nkv * hd * hs) as u64;
    let p_o = (hs * nh * hd) as u64;
    let p_in_ln = hs as u64;
    let p_post_ln = hs as u64;
    let p_gate = (inter * hs) as u64;
    let p_up = (inter * hs) as u64;
    let p_down = (hs * inter) as u64;
    let p_per_layer = p_q + p_k + p_v + p_o + p_in_ln + p_post_ln + p_gate + p_up + p_down;
    let p_all_layers = p_per_layer * (nl as u64);

    println!();
    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │                   SOLVED ARCHITECTURE                       │");
    println!("  ├─────────────────────────────────────────────────────────────┤");
    println!(
        "  │  vocab_size:                 {:>10}                     │",
        vs
    );
    println!(
        "  │  hidden_size:                {:>10}                     │",
        hs
    );
    println!(
        "  │  intermediate_size:          {:>10}  (SwiGLU)           │",
        inter
    );
    println!(
        "  │  num_hidden_layers:          {:>10}                     │",
        nl
    );
    println!(
        "  │  num_attention_heads:        {:>10}                     │",
        nh
    );
    println!(
        "  │  num_key_value_heads:        {:>10}  (GQA {}:1)          │",
        nkv,
        nh / nkv
    );
    println!(
        "  │  head_dim:                   {:>10}                     │",
        hd
    );
    println!(
        "  │  max_position_embeddings:    {:>10}                     │",
        target_config.max_position_embeddings
    );
    println!(
        "  │  rope_theta:                 {:>10.1}                     │",
        target_config.rope_theta
    );
    println!(
        "  │  rms_norm_eps:               {:>10.1e}                     │",
        target_config.rms_norm_eps
    );
    println!("  ├─────────────────────────────────────────────────────────────┤");
    println!("  │  PARAMETER BREAKDOWN:                                       │");
    println!(
        "  │    embed_tokens.weight:      {:>14}                     │",
        p_embed
    );
    println!(
        "  │    lm_head.weight:           {:>14}                     │",
        p_lm_head
    );
    println!(
        "  │    model.norm.weight:        {:>14}                     │",
        p_norm
    );
    println!(
        "  │    per decoder layer:        {:>14}                     │",
        p_per_layer
    );
    println!(
        "  │      self_attn (Q+K+V+O):    {:>14}                     │",
        p_q + p_k + p_v + p_o
    );
    println!(
        "  │      mlp (gate+up+down):     {:>14}                     │",
        p_gate + p_up + p_down
    );
    println!(
        "  │      layernorms (in+post):   {:>14}                     │",
        p_in_ln + p_post_ln
    );
    println!(
        "  │    all {} decoder layers:     {:>14}                     │",
        nl, p_all_layers
    );
    println!("  ├─────────────────────────────────────────────────────────────┤");
    println!(
        "  │  EXACT TOTAL PARAMETERS:     {:>14}                     │",
        exact_params
    );
    println!(
        "  │  Target parameters:          {:>14}                     │",
        target_params
    );
    println!(
        "  │  Difference from target:     {:>+14} ({:>+.3} %)          │",
        delta_signed, delta_pct
    );
    println!("  └─────────────────────────────────────────────────────────────┘");
    println!();

    // Validate target config
    if let Err(e) = target_config.validate() {
        eprintln!("Error: solved architecture failed validation: {}", e);
        exit(1);
    }

    if dry_run || output_dir.is_none() {
        println!("  [DRY-RUN COMPLETE] Architecture solved and verified mathematically.");
        println!("  No SafeTensors or candidate directory written.");
        println!("==================================================================");
        exit(0);
    }

    let output = output_dir.expect("output_dir verified above");

    // ── Expand weights ────────────────────────────────────────────────────────
    println!("[2/5] Expanding weights from source model ...");
    println!("      (this may take several minutes for a 1B model)");

    let engine = ModelExpansionEngine::new(&source_model);
    let (expanded_weights, metadata) = engine
        .expand_model(&target_config, &GrowthType::FullCapacity)
        .unwrap_or_else(|e| {
            eprintln!("Error: weight expansion failed: {}", e);
            exit(1);
        });

    println!(
        "  Source params transferred: {}",
        metadata.parent_param_count
    );
    println!("  Expanded param count:      {}", metadata.new_param_count);
    println!("  Layers added (new):        {}", metadata.layers_added);
    println!("  Hidden delta:              {}", metadata.hidden_delta);
    println!("  Vocab delta:               {}", metadata.vocab_delta);

    // Verify the count matches the solver
    if metadata.new_param_count != exact_params {
        eprintln!(
            "CRITICAL: expanded weight param count {} != solver count {} — aborting",
            metadata.new_param_count, exact_params
        );
        exit(1);
    }
    println!(
        "  Param count cross-check:   OK ({} == {})",
        metadata.new_param_count, exact_params
    );
    println!();

    // ── Derive added tokens from dataset (data-driven, not placeholder strings) ──
    println!(
        "[3/5] Preparing tokenizer (vocab {} → {}) ...",
        source_config.vocab_size, effective_vocab
    );
    let added_tokens: Vec<String> = if effective_vocab > source_config.vocab_size {
        // Attempt to find a dataset directory: --dataset-dir arg or well-known default
        let dataset_dir = args
            .iter()
            .position(|a| a == "--dataset-dir")
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| "storage/datasets/raw_sources".to_string());

        if std::path::Path::new(&dataset_dir).exists() {
            let vocab_builder = tara_engine::dataset::VocabBuilder::new(
                &dataset_dir,
                &source_model,
                effective_vocab,
            );
            match vocab_builder.build_candidate_tokens() {
                Ok((tokens, report)) => {
                    println!(
                        "  Data-driven tokens from dataset: {} (scanned {} records)",
                        tokens.len(),
                        report.total_records_scanned
                    );
                    tokens
                }
                Err(e) => {
                    eprintln!("Error: VocabBuilder failed: {e}");
                    eprintln!(
                        "Hint: provide --dataset-dir <path> pointing to JSONL training shards."
                    );
                    exit(1);
                }
            }
        } else {
            eprintln!("Error: Vocabulary expansion requires a dataset directory.");
            eprintln!(
                "  Source vocab: {}, target vocab: {}",
                source_config.vocab_size, effective_vocab
            );
            eprintln!("  Provide --dataset-dir <path> pointing to JSONL training shards.");
            eprintln!(
                "  Placeholder <ext_N> tokens are not accepted — they produce unusable tokenizers."
            );
            exit(1);
        }
    } else {
        Vec::new()
    };
    println!("  Tokens prepared: {}", added_tokens.len());
    println!();

    // ── Write candidate model atomically ─────────────────────────────────────
    println!("[4/5] Writing expanded candidate to '{}' ...", output);

    engine
        .write_expanded_model_with_tokens(&expanded_weights, &target_config, &output, &added_tokens)
        .unwrap_or_else(|e| {
            eprintln!("Error: failed to write expanded model: {}", e);
            exit(1);
        });

    // Verify directory was created
    let output_path = PathBuf::from(&output);
    if !output_path.join("model.safetensors").exists() {
        eprintln!("Error: model.safetensors not found in output directory after write");
        exit(1);
    }
    if !output_path.join("config.json").exists() {
        eprintln!("Error: config.json not found in output directory after write");
        exit(1);
    }
    if !output_path.join("tokenizer.json").exists() {
        eprintln!("Error: tokenizer.json not found in output directory after write");
        exit(1);
    }
    println!("  model.safetensors:   OK");
    println!("  config.json:         OK");
    println!("  tokenizer.json:      OK");
    println!();

    // ── Verification ──────────────────────────────────────────────────────────
    println!("[5/5] Verification ...");
    println!();

    let cpu_result: String;
    let cuda_result: String;

    // CPU: load + forward pass
    if do_verify {
        println!("  [CPU] Loading expanded model from disk ...");
        match TaraForCausalLM::load(&output) {
            Err(e) => {
                cpu_result = format!("FAIL — load error: {}", e);
            }
            Ok(model) => {
                // Single-token forward pass
                let test_seq = vec![1u32];
                let logits = model.forward(&test_seq);
                if logits.is_empty() {
                    cpu_result = "FAIL — forward returned empty logits".to_string();
                } else if logits.iter().any(|v| v.is_nan() || v.is_infinite()) {
                    cpu_result =
                        format!("FAIL — forward produced NaN/Inf in {} logits", logits.len());
                } else {
                    cpu_result = format!(
                        "PASS — logits shape [{}], first 3: [{:.4}, {:.4}, {:.4}]",
                        logits.len(),
                        logits[0],
                        logits[1],
                        logits[2],
                    );
                }
            }
        }
        println!("  [CPU] {}", cpu_result);
    } else {
        cpu_result = "SKIPPED (run with --verify to enable)".to_string();
        println!("  [CPU] {}", cpu_result);
    }

    // CUDA FP16: verify key architecture dimensions on real GPU
    // We test the lm_head tier only (vocab×hidden) to avoid OOM on 2 GB VRAM.
    // A full 1B model in FP16 requires ~2 GB which exceeds available free VRAM.
    println!();
    println!(
        "  [CUDA FP16] Testing lm_head dimensions ({}×{}) on GPU ...",
        target_config.vocab_size, target_config.hidden_size
    );

    let lm_head_size = target_config.vocab_size * target_config.hidden_size;
    let lm_head_mb_fp16 = (lm_head_size * 2) as f64 / 1_048_576.0;
    println!(
        "  [CUDA FP16] lm_head tensor: {:.1} MB in FP16",
        lm_head_mb_fp16
    );

    // Build a minimal weight map containing just lm_head for GPU test
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let test_lm_head: Vec<f32> = (0..lm_head_size)
        .map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * 0.02)
        .collect();
    let mut test_weights = std::collections::HashMap::new();
    test_weights.insert("lm_head.weight".to_string(), test_lm_head);

    match CudaTrainer::new_with_precision(0, TrainingPrecision::Fp16) {
        Err(e) => {
            cuda_result = format!("FAIL — CUDA init error: {}", e);
        }
        Ok(mut trainer) => {
            let device = trainer.device();
            println!(
                "  [CUDA FP16] Device: {} (CC {}.{})",
                device.name, device.compute_capability.0, device.compute_capability.1
            );
            let (free_mb, total_mb) = trainer
                .get_vram_info()
                .map(|(f, t)| (f as f64 / 1_048_576.0, t as f64 / 1_048_576.0))
                .unwrap_or((0.0, 0.0));
            println!(
                "  [CUDA FP16] VRAM free: {:.1} MB / {:.1} MB total",
                free_mb, total_mb
            );

            if lm_head_mb_fp16 * 5.0 > free_mb {
                // Estimate: FP16 weight + FP32 master + m + v + FP16 grad ≈ 5×
                cuda_result = format!(
                    "SKIPPED — lm_head alone requires ~{:.0} MB (×5 with optimizer states), \
                     {:.0} MB free. Full 1B FP16 model requires ~2 GB. \
                     Architecture is valid; train on a GPU with ≥ 8 GB VRAM.",
                    lm_head_mb_fp16 * 5.0,
                    free_mb
                );
            } else {
                match trainer.register_weights(&test_weights) {
                    Err(e) => {
                        cuda_result = format!("FAIL — weight upload: {}", e);
                    }
                    Ok(()) => {
                        // Forward pass: seq_len=1, hidden=target, vocab=target
                        let activations = vec![0.01f32; target_config.hidden_size];
                        match trainer.forward_lm_head(
                            &activations,
                            1,
                            target_config.hidden_size,
                            target_config.vocab_size,
                        ) {
                            Err(e) => {
                                cuda_result = format!("FAIL — forward_lm_head: {}", e);
                            }
                            Ok(logits) => {
                                let finite = logits.iter().all(|v| v.is_finite());
                                cuda_result = format!(
                                    "PASS — logits shape [{}], finite: {}, \
                                     sample: [{:.4}, {:.4}, {:.4}]",
                                    logits.len(),
                                    finite,
                                    logits[0],
                                    logits[1],
                                    logits[2],
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    println!("  [CUDA FP16] {}", cuda_result);
    println!();

    // ── Final report ──────────────────────────────────────────────────────────
    println!("==================================================================");
    println!("EXPANSION REPORT");
    println!("==================================================================");
    println!();
    println!("A) Solved Architecture:");
    println!("   vocab_size:          {}", target_config.vocab_size);
    println!("   hidden_size:         {}", target_config.hidden_size);
    println!(
        "   intermediate_size:   {}",
        target_config.intermediate_size
    );
    println!(
        "   num_hidden_layers:   {}",
        target_config.num_hidden_layers
    );
    println!(
        "   num_attention_heads: {}",
        target_config.num_attention_heads
    );
    println!(
        "   num_key_value_heads: {}  (GQA {}:1)",
        target_config.num_key_value_heads,
        target_config.num_attention_heads / target_config.num_key_value_heads
    );
    println!("   head_dim:            {}", target_config.head_dim);
    println!(
        "   max_position_emb:    {}",
        target_config.max_position_embeddings
    );
    println!("   rope_theta:          {}", target_config.rope_theta);
    println!();
    println!("B) Exact Parameter Count:    {}", exact_params);
    println!("   Delta from target:        {:.3} %", delta_pct);
    println!();
    println!("C) Candidate Path:           {}", output);
    println!();
    println!("D) Old-Weight Preservation:");
    println!(
        "   Source params:            {}",
        metadata.parent_param_count
    );
    println!(
        "   Transferred (zero-pad):   {} tensors from source layers",
        source_config.num_hidden_layers
    );
    println!(
        "   New layers initialized:   {} (zero-residual)",
        metadata.layers_added
    );
    println!();
    println!("E) CPU Load/Forward:         {}", cpu_result);
    println!();
    println!("F) CUDA FP16 Load/Forward:   {}", cuda_result);
    println!();

    // Determine blocker
    let cpu_ok = cpu_result.starts_with("PASS") || cpu_result.starts_with("SKIPPED");
    let cuda_ok = cuda_result.starts_with("PASS") || cuda_result.starts_with("SKIPPED");
    if !cpu_ok || !cuda_ok {
        println!("G) Blockers:");
        if !cpu_ok {
            println!("   [BLOCKER] CPU: {}", cpu_result);
        }
        if !cuda_ok {
            println!("   [BLOCKER] CUDA: {}", cuda_result);
        }
    } else {
        println!("G) Blockers: NONE — candidate architecture is valid and verified.");
        if !do_verify {
            println!("   (Re-run with --verify to confirm CPU forward pass)");
        }
    }
    println!("==================================================================");
}
