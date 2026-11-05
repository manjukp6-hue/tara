//! Dynamic World Model Initializer & Part-Wise SafeTensors Sharder (100% Native Rust).
//!
//! Generates production-grade Transformer Latent Dynamics World Model
//! dynamically aligned with any base model configuration or custom parameters:
//! - State/Latent Dimension
//! - Action Dimension
//! - Hidden Dimension
//! - Attention Tensor Geometry Invariant (head_dim * num_heads == hidden_size)
//! - Part-Wise SafeTensors Sharding with model.safetensors.index.json
//! - Pure Latent Dynamics Engine (zero text tokenizer pollution)

use std::env;
use std::fs;
use std::path::PathBuf;
use tara_engine::resolve_max_shard_size_bytes;
use tara_engine::world_model::{WorldModel, WorldModelConfig};

fn print_usage() {
    println!("Usage: init_world_model [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --output-dir <PATH>        Destination directory (default: storage/models/tara_world_model)");
    println!("  --base-config <PATH>       Base neural model config.json to inherit architecture from");
    println!("  --max-shard-mb <MB>        Maximum size per shard in MB (default: dynamic / 480)");
    println!("  --latent-dim <DIM>         Override latent state dimension");
    println!("  --action-dim <DIM>         Override action dimension");
    println!("  --hidden-size <SIZE>       Override hidden size");
    println!("  --layers <NUM>             Override number of dynamics layers");
    println!("  --heads <NUM>              Override attention heads");
    println!("  --kv-heads <NUM>           Override KV heads");
    println!("  --head-dim <DIM>           Override head dimension");
    println!("  -h, --help                 Print help information");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return Ok(());
    }

    let mut output_dir = PathBuf::from("storage/models/tara_world_model");
    let mut base_config_path = PathBuf::from("storage/models/tara_candidate_v1/config.json");
    let mut custom_max_shard_mb: Option<usize> = None;
    let mut custom_latent_dim: Option<usize> = None;
    let mut custom_action_dim: Option<usize> = None;
    let mut custom_hidden_size: Option<usize> = None;
    let mut custom_layers: Option<usize> = None;
    let mut custom_heads: Option<usize> = None;
    let mut custom_kv_heads: Option<usize> = None;
    let mut custom_head_dim: Option<usize> = None;

    let mut idx = 1;
    while idx < args.len() {
        match args[idx].as_str() {
            "--output-dir" if idx + 1 < args.len() => {
                output_dir = PathBuf::from(&args[idx + 1]);
                idx += 2;
            }
            "--base-config" if idx + 1 < args.len() => {
                base_config_path = PathBuf::from(&args[idx + 1]);
                idx += 2;
            }
            "--max-shard-mb" if idx + 1 < args.len() => {
                custom_max_shard_mb = args[idx + 1].parse().ok();
                idx += 2;
            }
            "--latent-dim" if idx + 1 < args.len() => {
                custom_latent_dim = args[idx + 1].parse().ok();
                idx += 2;
            }
            "--action-dim" if idx + 1 < args.len() => {
                custom_action_dim = args[idx + 1].parse().ok();
                idx += 2;
            }
            "--hidden-size" if idx + 1 < args.len() => {
                custom_hidden_size = args[idx + 1].parse().ok();
                idx += 2;
            }
            "--layers" if idx + 1 < args.len() => {
                custom_layers = args[idx + 1].parse().ok();
                idx += 2;
            }
            "--heads" if idx + 1 < args.len() => {
                custom_heads = args[idx + 1].parse().ok();
                idx += 2;
            }
            "--kv-heads" if idx + 1 < args.len() => {
                custom_kv_heads = args[idx + 1].parse().ok();
                idx += 2;
            }
            "--head-dim" if idx + 1 < args.len() => {
                custom_head_dim = args[idx + 1].parse().ok();
                idx += 2;
            }
            _ => {
                idx += 1;
            }
        }
    }

    fs::create_dir_all(&output_dir)?;

    // Dynamically derive configuration from base model config if present,
    // otherwise construct dynamically from parameters.
    let mut config = if base_config_path.exists() {
        println!(
            "[World Model Init] Dynamically deriving architecture from '{}'...",
            base_config_path.display()
        );
        WorldModelConfig::from_base_config_file(&base_config_path)?
    } else {
        println!("[World Model Init] Constructing dynamic World Model configuration...");
        let h = custom_hidden_size.unwrap_or(2048);
        let l = custom_latent_dim.unwrap_or(h);
        let a = custom_action_dim.unwrap_or((h / 4).max(64));
        WorldModelConfig::new(l, a, h)
    };

    // Apply any explicit parameter overrides dynamically
    if let Some(l) = custom_latent_dim {
        config.latent_dim = l;
    }
    if let Some(a) = custom_action_dim {
        config.action_dim = a;
    }
    if let Some(h) = custom_hidden_size {
        config.hidden_size = h;
    }
    if let Some(layers) = custom_layers {
        config.num_hidden_layers = layers;
    }
    if let Some(heads) = custom_heads {
        config.num_attention_heads = heads;
    }
    if let Some(kv) = custom_kv_heads {
        config.num_key_value_heads = kv;
    }
    if let Some(hd) = custom_head_dim {
        config.head_dim = hd;
    }

    // Verify strict Attention Tensor Geometry Invariant (head_dim * num_heads == hidden_size)
    config.validate()?;

    println!("[World Model Init] Dynamic Transformer Latent Dynamics Architecture:");
    println!("  - State/Latent Dim: {}", config.latent_dim);
    println!("  - Action Dim:       {}", config.action_dim);
    println!("  - Hidden Size:      {}", config.hidden_size);
    println!("  - Intermediate Size:{}", config.intermediate_size);
    println!("  - Hidden Layers:    {}", config.num_hidden_layers);
    println!("  - Attention Heads:  {}", config.num_attention_heads);
    println!("  - KV Heads (GQA):   {}", config.num_key_value_heads);
    println!("  - Head Dim:         {}", config.head_dim);
    println!("  - Geometric Check:  {} * {} == {} (VALID)", config.head_dim, config.num_attention_heads, config.hidden_size);

    println!("[World Model Init] Initializing variance-scaled Gaussian/Xavier weights...");
    let wm = WorldModel::new(config.clone());

    // Resolve max shard size dynamically
    let max_shard_bytes = custom_max_shard_mb
        .map(|mb| mb * 1024 * 1024)
        .unwrap_or_else(|| {
            std::env::var("TARA_MAX_SHARD_MB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .map(|mb| mb * 1024 * 1024)
                .unwrap_or_else(resolve_max_shard_size_bytes)
        });

    println!(
        "[World Model Init] Saving part-wise sharded SafeTensors to '{}' (shard limit: {:.2} MB)...",
        output_dir.display(),
        max_shard_bytes as f64 / (1024.0 * 1024.0)
    );
    let shards = wm.save_to_dir_sharded(&output_dir, max_shard_bytes)?;

    // Remove any text tokenizer files from world model folder (Pure Latent Dynamics isolation)
    for tok_file in &["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"] {
        let p = output_dir.join(tok_file);
        if p.exists() {
            let _ = fs::remove_file(&p);
            println!("  - Removed text tokenizer artifact: {}", tok_file);
        }
    }

    // Clean up obsolete single monolithic safetensors file if present
    let old_monolithic = output_dir.join("world_model.safetensors");
    if old_monolithic.exists() {
        let _ = fs::remove_file(&old_monolithic);
        println!("  - Removed obsolete monolithic safetensors file.");
    }

    let index_path = output_dir.join("model.safetensors.index.json");
    let config_path = output_dir.join("world_model_config.json");

    if index_path.exists() && config_path.exists() && !shards.is_empty() {
        let mut total_shard_bytes: u64 = 0;
        for shard_name in &shards {
            let shard_path = output_dir.join(shard_name);
            if let Ok(meta) = fs::metadata(&shard_path) {
                total_shard_bytes += meta.len();
                println!("  - Shard: {} ({:.2} MB)", shard_name, meta.len() as f64 / (1024.0 * 1024.0));
            }
        }
        println!(
            "[World Model Init] SUCCESS: Generated {} part-wise shards ({:.2} MB total) and index at '{}'",
            shards.len(),
            total_shard_bytes as f64 / (1024.0 * 1024.0),
            index_path.display()
        );
    } else {
        return Err("Failed to generate authoritative part-wise World Model files".into());
    }

    Ok(())
}
