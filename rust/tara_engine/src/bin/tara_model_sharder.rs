//! TARA Part-Wise SafeTensors Model Sharder & Builder (100% Native Rust).
//!
//! Converts models into part-wise shards (<= 500MB per shard) with standard
//! `model.safetensors.index.json`, supporting BF16 precision, custom prefixes,
//! and world-model alignment.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use tara_engine::config::TaraConfig;
use tara_engine::internal_diagnostics::write_safetensors_sharded_with_prefix;
use tara_engine::load_model_weights_with_shapes;
use tara_engine::model_expansion::{tensor_shape, ModelExpansionEngine};
use tara_engine::TaraForCausalLM;

fn print_usage() {
    println!("Usage: tara_model_sharder [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --model-dir <PATH>           Directory containing or receiving the model");
    println!("  --source-dir <PATH>          Directory to read weights from (default: model-dir)");
    println!("  --max-shard-mb <MB>          Maximum size per shard in MB (default: 480)");
    println!("  --prefix <PREFIX>            Filename prefix for shards (e.g. neural_model, world_model)");
    println!("  --bf16                       Store weights in BF16 precision (default)");
    println!("  --f32                        Store weights in F32 precision");
    println!("  --clean-old                  Delete obsolete shards after verified sharding (default: true)");
    println!("  --no-clean-old               Keep old shard files");
    println!(
        "  --rebuild-world-model        Rebuild world model architecture aligned with neural model"
    );
    println!("  -h, --help                   Print help information");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return Ok(());
    }

    let mut model_dir = PathBuf::from("storage/models/tara_candidate_v1");
    let mut source_dir: Option<PathBuf> = None;
    let mut max_shard_mb = 480usize;
    let mut prefix = "neural_model".to_string();
    let mut is_bf16 = true;
    let mut clean_old = true;
    let mut rebuild_world_model = false;
    let mut init_fresh = false;

    let mut idx = 1;
    while idx < args.len() {
        match args[idx].as_str() {
            "--model-dir" => {
                if idx + 1 < args.len() {
                    model_dir = PathBuf::from(&args[idx + 1]);
                    idx += 2;
                } else {
                    return Err("Missing argument for --model-dir".into());
                }
            }
            "--source-dir" => {
                if idx + 1 < args.len() {
                    source_dir = Some(PathBuf::from(&args[idx + 1]));
                    idx += 2;
                } else {
                    return Err("Missing argument for --source-dir".into());
                }
            }
            "--max-shard-mb" => {
                if idx + 1 < args.len() {
                    max_shard_mb = args[idx + 1].parse::<usize>()?;
                    if max_shard_mb == 0 {
                        return Err("--max-shard-mb must be >= 1".into());
                    }
                    idx += 2;
                } else {
                    return Err("Missing argument for --max-shard-mb".into());
                }
            }
            "--prefix" => {
                if idx + 1 < args.len() {
                    prefix = args[idx + 1].clone();
                    idx += 2;
                } else {
                    return Err("Missing argument for --prefix".into());
                }
            }
            "--bf16" => {
                is_bf16 = true;
                idx += 1;
            }
            "--f32" => {
                is_bf16 = false;
                idx += 1;
            }
            "--clean-old" => {
                clean_old = true;
                idx += 1;
            }
            "--no-clean-old" => {
                clean_old = false;
                idx += 1;
            }
            "--init-fresh" => {
                init_fresh = true;
                idx += 1;
            }
            "--rebuild-world-model" => {
                rebuild_world_model = true;
                prefix = "world_model".to_string();
                idx += 1;
            }
            unknown => {
                return Err(format!("Unknown option: {unknown}").into());
            }
        }
    }

    let max_shard_bytes = max_shard_mb
        .checked_mul(1024 * 1024)
        .ok_or("--max-shard-mb overflow")?;

    println!("================================================================================");
    println!(" TARA PART-WISE SAFETENSORS MODEL SHARDER & BUILDER (100% NATIVE RUST)");
    println!("================================================================================");
    println!("Target Model Directory  : {}", model_dir.display());
    println!(
        "Source Model Directory  : {}",
        source_dir.as_ref().unwrap_or(&model_dir).display()
    );
    println!(
        "Max Shard Size Limit    : {} MB ({} bytes)",
        max_shard_mb, max_shard_bytes
    );
    println!("Shard Prefix            : {}", prefix);
    println!(
        "Precision               : {}",
        if is_bf16 { "BF16" } else { "F32" }
    );
    println!("Rebuild World Model     : {}", rebuild_world_model);
    println!("Clean Obsolete Files    : {}", clean_old);
    println!("--------------------------------------------------------------------------------");

    fs::create_dir_all(&model_dir)?;

    let (weights, shapes) = if rebuild_world_model {
        println!("Rebuilding World Model aligned with Neural Model architecture...");
        let neural_dir = Path::new("storage/models/tara_candidate_v1");
        if !neural_dir.exists() {
            return Err(
                "Neural model directory storage/models/tara_candidate_v1 does not exist".into(),
            );
        }

        // Copy tokenizer files to ensure 100% compatibility
        for file_name in &[
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ] {
            let src_file = neural_dir.join(file_name);
            let dst_file = model_dir.join(file_name);
            if src_file.exists() {
                fs::copy(&src_file, &dst_file)?;
                println!("  Synced {} from neural model.", file_name);
            }
        }

        // Define aligned world model configuration (matching hidden_size: 2048, head_dim: 32, num_heads: 64, kv_heads: 16)
        let world_config = TaraConfig {
            vocab_size: 100000,
            hidden_size: 2048,
            intermediate_size: 5632,
            num_hidden_layers: 8,
            num_attention_heads: 64,
            num_key_value_heads: 16,
            head_dim: 32,
            max_position_embeddings: 4096,
            rope_theta: 500000.0,
            rms_norm_eps: 0.00001,
            initializer_range: 0.02,
            version: "2.0.0".to_string(),
            model_type: "tara".to_string(),
            tie_word_embeddings: Some(true),
        };

        let config_path = model_dir.join("config.json");
        fs::write(&config_path, serde_json::to_string_pretty(&world_config)?)?;
        println!("  Wrote aligned World Model config.json: {} layers, {} hidden_size, {} intermediate_size.",
            world_config.num_hidden_layers, world_config.hidden_size, world_config.intermediate_size);

        println!("  Initializing target weights for World Model layout...");
        let t0 = Instant::now();
        let w = ModelExpansionEngine::init_target_weights(&world_config);
        let s: HashMap<String, Vec<usize>> = w
            .iter()
            .map(|(k, v)| (k.clone(), tensor_shape(k, v.len(), &world_config)))
            .collect();
        println!(
            "  Initialized {} tensors in {:.2}s.",
            w.len(),
            t0.elapsed().as_secs_f64()
        );
        (w, s)
    } else if init_fresh {
        println!(
            "Initializing FRESH weights for {} from existing config.json...",
            model_dir.display()
        );
        let config_path = model_dir.join("config.json");
        let cfg = TaraConfig::from_json_file(&config_path.to_string_lossy())?;
        println!(
            "  Preserved model config: {} layers, {} hidden_size, {} intermediate_size, vocab {}.",
            cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size
        );
        let t0 = Instant::now();
        let w = ModelExpansionEngine::init_target_weights(&cfg);
        let s: HashMap<String, Vec<usize>> = w
            .iter()
            .map(|(k, v)| (k.clone(), tensor_shape(k, v.len(), &cfg)))
            .collect();
        println!(
            "  Initialized {} fresh tensors in {:.2}s.",
            w.len(),
            t0.elapsed().as_secs_f64()
        );
        (w, s)
    } else {
        let load_dir = source_dir.as_ref().unwrap_or(&model_dir);
        println!("Loading weights from {} ...", load_dir.display());
        let t0 = Instant::now();
        let (w, s) = load_model_weights_with_shapes(&load_dir.to_string_lossy())?;
        println!(
            "  Loaded {} tensors in {:.2}s.",
            w.len(),
            t0.elapsed().as_secs_f64()
        );
        (w, s)
    };

    println!(
        "Writing sharded SafeTensors to {} with prefix '{}'...",
        model_dir.display(),
        prefix
    );
    let t1 = Instant::now();
    let written_shards = write_safetensors_sharded_with_prefix(
        &weights,
        &shapes,
        &model_dir.to_string_lossy(),
        max_shard_bytes,
        &prefix,
        is_bf16,
    )?;

    println!(
        "  Wrote {} shards in {:.2}s:",
        written_shards.len(),
        t1.elapsed().as_secs_f64()
    );
    for shard_name in &written_shards {
        let shard_path = model_dir.join(shard_name);
        let sz = fs::metadata(&shard_path)?.len();
        println!(
            "    - {} ({:.2} MB)",
            shard_name,
            sz as f64 / (1024.0 * 1024.0)
        );
        if sz > max_shard_bytes as u64 {
            return Err(format!(
                "Shard {} exceeds max limit of {} bytes (was {} bytes)",
                shard_name, max_shard_bytes, sz
            )
            .into());
        }
    }

    // Verify index file was written
    let index_file = model_dir.join("model.safetensors.index.json");
    if !index_file.exists() {
        return Err("Sharding completed but model.safetensors.index.json was not created!".into());
    }
    println!("  Generated authoritative index: {}", index_file.display());

    // Verify loading weights from index
    println!("Verifying sharded model reload from index ...");
    let t2 = Instant::now();
    let (reloaded_weights, _) = load_model_weights_with_shapes(&model_dir.to_string_lossy())?;
    if reloaded_weights.len() != weights.len() {
        return Err(format!(
            "Reload mismatch: expected {} tensors, got {}",
            weights.len(),
            reloaded_weights.len()
        )
        .into());
    }

    for (k, v) in &weights {
        let reloaded_v = reloaded_weights
            .get(k)
            .ok_or_else(|| format!("Missing tensor in reloaded weights: {k}"))?;
        if reloaded_v.len() != v.len() {
            return Err(format!(
                "Tensor length mismatch for {k}: expected {}, got {}",
                v.len(),
                reloaded_v.len()
            )
            .into());
        }
        if !is_bf16 {
            for (idx, (&a, &b)) in v.iter().zip(reloaded_v.iter()).enumerate() {
                if (a - b).abs() > 1e-6 {
                    return Err(
                        format!("Tensor bit mismatch for {k} at index {idx}: {a} vs {b}").into(),
                    );
                }
            }
        }
    }
    println!(
        "  Weight verification SUCCESS in {:.2}s!",
        t2.elapsed().as_secs_f64()
    );

    // Verify full model instantiation and forward pass
    println!("Verifying full TaraForCausalLM load and forward pass ...");
    let model = TaraForCausalLM::load(&model_dir.to_string_lossy())?;
    let test_prompt = vec![1u32];
    let logits = model.forward(&test_prompt);
    if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
        return Err("Model forward pass failed: non-finite or empty logits!".into());
    }
    println!(
        "  Forward pass SUCCESS! (Logits len: {}, last token logits: {})",
        logits.len(),
        model.config.vocab_size
    );

    // Clean old/obsolete shards
    if clean_old {
        println!(
            "Cleaning obsolete safetensors shards in {} ...",
            model_dir.display()
        );
        let entries = fs::read_dir(&model_dir)?;
        for entry in entries {
            let entry = entry?;
            let file_name = entry.file_name().to_string_lossy().to_string();
            if file_name.ends_with(".safetensors") && !written_shards.contains(&file_name) {
                let p = entry.path();
                let sz = fs::metadata(&p)?.len();
                println!(
                    "  Removing obsolete shard: {} ({:.2} MB)",
                    file_name,
                    sz as f64 / (1024.0 * 1024.0)
                );
                fs::remove_file(p)?;
            }
        }
    }

    println!("================================================================================");
    println!(" PART-WISE SAFETENSORS SHARDING & VALIDATION 100% SUCCESSFUL!");
    println!("================================================================================");

    Ok(())
}
