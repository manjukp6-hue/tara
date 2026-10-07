//! TARA Candidate Training CLI — 100% Native Rust Standalone Trainer.
//!
//! Directly invokes the production TARA training engine (`NativeSelfTrainer` / `run_controlled_training_with_options`).
//! Pure native Rust, zero Python, zero PyTorch, zero external AI connection.

use std::env;
use std::path::PathBuf;
use std::process::exit;

use tara_engine::train_candidate::{run_controlled_training_with_options, TrainingOptions};

fn print_usage() {
    println!("TARA Native Candidate Model Training Utility (Rust)");
    println!("Usage:");
    println!("  train_candidate [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --curriculum <path>     Academic curriculum jsonl path (default: datasets/master_curriculum.jsonl, use 'none' to disable)");
    println!("  --dataset <path>        Dataset directory containing canonical jsonl shards (default: storage/datasets/tara_dataset_filtered/canonical)");
    println!("  --model <path>          Base model / checkpoint directory (default: storage/models/tara_candidate_v1)");
    println!("  --output <path>         Output candidate directory for newly trained SafeTensors checkpoint");
    println!("  --device <dev>          Training device backend: 'cpu', 'gpu', or 'auto' (default: auto)");
    println!(
        "  --precision <prec>      Training precision: 'fp32', 'fp16', or 'auto' (default: auto)"
    );
    println!(
        "                            auto = fp16 when GPU available + compute >= 5.0, else fp32"
    );
    println!("  --epochs <n>            Number of training epochs (default: 1)");
    println!("  --learning-rate <f>     Learning rate for AdamW optimizer (default: 0.001)");
    println!("  --batch-size <n>        Gradient accumulation batch size (default: 4)");
    println!("  --max-steps <n>         Maximum training steps ceiling (optional, default: entire epoch)");
    println!(
        "  --checkpoint-dir <path> Directory for persistent periodic checkpoints and resumption"
    );
    println!(
        "  --checkpoint-interval <n> Periodic checkpoint saving interval in steps (default: 500)"
    );
    println!("  --resume                Resume training from last saved checkpoint if present");
    println!("  --resume-from <path>    Explicit checkpoint or model directory to resume from");
    println!("  --max-memory-mb <f>     Maximum memory footprint ceiling in MB (default: 65536.0)");
    println!("  -h, --help              Print this help message");
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut curriculum_path: Option<String> = Some("datasets/master_curriculum.jsonl".to_string());
    let mut dataset_dir = "storage/datasets/tara_dataset_filtered/canonical".to_string();
    let mut model_dir = "storage/models/tara_candidate_v1".to_string();
    let mut output_dir: Option<String> = None;
    let mut device = "auto".to_string();
    let mut precision = "auto".to_string();
    let mut epochs = 1usize;
    let mut learning_rate = 1e-3f32;
    let mut batch_size = 4usize;
    let mut max_steps: Option<usize> = None;
    let mut checkpoint_dir: Option<String> = None;
    let mut checkpoint_interval: Option<usize> = None;
    let mut resume = false;
    let mut resume_from: Option<String> = None;
    let mut max_memory_mb: Option<f64> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_usage();
                exit(0);
            }
            "--curriculum" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --curriculum requires a path argument (or 'none')");
                    exit(1);
                }
                curriculum_path = if args[i].is_empty() || args[i].to_lowercase() == "none" {
                    None
                } else {
                    Some(args[i].clone())
                };
            }
            "--device" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --device requires an argument ('cpu', 'gpu', or 'auto')");
                    exit(1);
                }
                let d = args[i].to_lowercase();
                if d != "cpu" && d != "gpu" && d != "auto" {
                    eprintln!(
                        "Error: invalid --device '{}'. Valid options: 'cpu', 'gpu', 'auto'",
                        args[i]
                    );
                    exit(1);
                }
                device = d;
            }
            "--precision" => {
                i += 1;
                if i >= args.len() {
                    eprintln!(
                        "Error: --precision requires an argument ('fp32', 'fp16', or 'auto')"
                    );
                    exit(1);
                }
                let p = args[i].to_lowercase();
                if p != "fp32" && p != "fp16" && p != "auto" {
                    eprintln!(
                        "Error: invalid --precision '{}'. Valid options: 'fp32', 'fp16', 'auto'",
                        args[i]
                    );
                    exit(1);
                }
                precision = p;
            }
            "--dataset" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --dataset requires a path argument");
                    exit(1);
                }
                dataset_dir = args[i].clone();
            }
            "--model" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --model requires a path argument");
                    exit(1);
                }
                model_dir = args[i].clone();
            }
            "--output" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --output requires a path argument");
                    exit(1);
                }
                output_dir = Some(args[i].clone());
            }
            "--epochs" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --epochs requires an integer argument");
                    exit(1);
                }
                epochs = args[i].parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("Error: invalid integer for --epochs: {}", args[i]);
                    exit(1);
                });
            }
            "--learning-rate" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --learning-rate requires a float argument");
                    exit(1);
                }
                learning_rate = args[i].parse::<f32>().unwrap_or_else(|_| {
                    eprintln!("Error: invalid float for --learning-rate: {}", args[i]);
                    exit(1);
                });
            }
            "--batch-size" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --batch-size requires an integer argument");
                    exit(1);
                }
                batch_size = args[i].parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("Error: invalid integer for --batch-size: {}", args[i]);
                    exit(1);
                });
            }
            "--max-steps" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --max-steps requires an integer argument");
                    exit(1);
                }
                max_steps = Some(args[i].parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("Error: invalid integer for --max-steps: {}", args[i]);
                    exit(1);
                }));
            }
            "--checkpoint-dir" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --checkpoint-dir requires a path argument");
                    exit(1);
                }
                checkpoint_dir = Some(args[i].clone());
            }
            "--checkpoint-interval" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --checkpoint-interval requires an integer argument");
                    exit(1);
                }
                checkpoint_interval = Some(args[i].parse::<usize>().unwrap_or_else(|_| {
                    eprintln!(
                        "Error: invalid integer for --checkpoint-interval: {}",
                        args[i]
                    );
                    exit(1);
                }));
            }
            "--resume" => {
                resume = true;
            }
            "--resume-from" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --resume-from requires a path argument");
                    exit(1);
                }
                resume_from = Some(args[i].clone());
                resume = true;
            }
            "--max-memory-mb" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --max-memory-mb requires a float argument in MB");
                    exit(1);
                }
                max_memory_mb = Some(args[i].parse::<f64>().unwrap_or_else(|_| {
                    eprintln!("Error: invalid float for --max-memory-mb: {}", args[i]);
                    exit(1);
                }));
            }
            unknown => {
                eprintln!("Error: unrecognized option '{}'", unknown);
                print_usage();
                exit(1);
            }
        }
        i += 1;
    }

    // Safety constraint: Never write directly into production storage/models/tara
    if let Some(ref out) = output_dir {
        let norm_out = out.replace('\\', "/");
        if norm_out == "storage/models/tara"
            || norm_out == "./storage/models/tara"
            || norm_out.ends_with("/storage/models/tara")
        {
            eprintln!("CRITICAL ERROR: Destination cannot be production 'storage/models/tara'. Training writes to candidate directories only.");
            exit(1);
        }
    }

    let repo_root = env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .to_string_lossy()
        .to_string();

    println!("==================================================================");
    println!("TARA NATIVE RUST CANDIDATE MODEL TRAINER");
    println!("==================================================================");
    println!("  Repo Root:      {}", repo_root);
    if let Some(ref cp) = curriculum_path {
        println!("  Curriculum:     {}", cp);
    } else {
        println!("  Curriculum:     [none]");
    }
    println!("  Dataset Dir:    {}", dataset_dir);
    println!("  Source Model:   {}", model_dir);
    if let Some(ref out) = output_dir {
        println!("  Target Output:  {}", out);
    } else {
        println!("  Target Output:  [isolated candidate staging]");
    }
    println!("  Epochs:         {}", epochs);
    println!("  Learning Rate:  {}", learning_rate);
    println!("  Batch Size:     {}", batch_size);
    println!("  Device Option:  {}", device);
    println!("  Precision:      {}", precision);
    if let Some(ref cp) = checkpoint_dir {
        println!("  Checkpoint Dir: {}", cp);
    }
    if let Some(ci) = checkpoint_interval {
        println!("  Checkpoint Int: {} steps", ci);
    }
    println!("  Resume:         {}", resume);
    if let Some(ms) = max_steps {
        println!("  Max Steps:      {}", ms);
    } else {
        println!("  Max Steps:      [unlimited]");
    }
    println!("==================================================================\n");

    let options = TrainingOptions {
        dataset_dir: Some(dataset_dir),
        curriculum_path,
        candidate_dir: output_dir,
        learning_rate: Some(learning_rate),
        batch_size: Some(batch_size),
        max_steps,
        device: Some(device),
        precision: Some(precision),
        checkpoint_dir,
        checkpoint_interval,
        resume,
        resume_from,
        max_memory_mb,
        ..Default::default()
    };

    println!("[1/3] Starting full-network backpropagation training cycle...");
    let result = match run_controlled_training_with_options(&model_dir, &repo_root, epochs, options)
    {
        Ok(v) => v,
        Err(e) => {
            eprintln!("\n[FAILURE] Training error: {}", e);
            exit(2);
        }
    };

    println!("[2/3] Training cycle completed successfully.");
    println!("\n==================================================================");
    println!("TRAINING EXECUTION REPORT");
    println!("==================================================================");
    let status = result
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("UNKNOWN");
    let loss_before = result
        .get("loss_before")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let loss_after = result
        .get("loss_after")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let steps_done = result.get("steps").and_then(|v| v.as_u64()).unwrap_or(0);
    let epochs_done = result.get("epochs").and_then(|v| v.as_u64()).unwrap_or(0);
    let out_path = result
        .get("candidate_dir")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let candidate_sha = result
        .get("candidate_sha256")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let source_preserved = result
        .get("source_model_preserved")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let dev_selected = result
        .get("device_selected")
        .and_then(|v| v.as_str())
        .unwrap_or("CPU");
    let dev_type = result
        .get("device_type")
        .and_then(|v| v.as_str())
        .unwrap_or("CPU");
    let precision_used = result
        .get("precision_used")
        .and_then(|v| v.as_str())
        .unwrap_or("fp32");

    println!("  Status:                   {}", status);
    println!("  Device Selected:          {}", dev_selected);
    println!("  Device Type:              {}", dev_type);
    println!("  Precision Used:           {}", precision_used);
    println!("  Loss Before:              {:.6}", loss_before);
    println!("  Loss After:               {:.6}", loss_after);
    println!("  Total Steps:              {}", steps_done);
    println!("  Epochs Completed:         {}", epochs_done);
    println!("  Candidate Output Path:    {}", out_path);
    println!("  Candidate SafeTensors:    {}", candidate_sha);
    println!("  Source Model Preserved:   {}", source_preserved);
    println!("==================================================================\n");

    if status != "COMPLETED" {
        eprintln!("Training did not complete normally: status = {}", status);
        exit(3);
    }

    println!("[3/3] Candidate checkpoint saved and ready for verification.");
    exit(0);
}
