//! TARA Candidate Training CLI — 100% Native Rust Standalone Trainer.
//!
//! Directly invokes the production TARA training engine (`NativeSelfTrainer` / `run_controlled_training_with_options`).
//! Pure native Rust, zero Python, zero PyTorch, zero external AI connection.

use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::exit;

use tara_engine::train_candidate::{
    has_parent_dir_traversal, paths_overlap, run_controlled_training_with_options, TrainingOptions,
};

fn print_usage() {
    println!("TARA Native Candidate Model Training Utility (Rust)");
    println!("Usage:");
    println!("  train_candidate [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --curriculum <path>     Academic curriculum jsonl path (default: storage/datasets/curriculum/foundational_curriculum.jsonl, use 'none' to disable)");
    println!("  --dataset <path>        Dataset directory containing jsonl shards (default: storage/datasets/main_dataset_v1)");
    println!("  --model <path>          Base model / checkpoint directory (default: storage/models/tara_candidate_v1)");
    println!("  --output <path>         Output candidate directory for newly trained SafeTensors checkpoint");
    println!("  --device <dev>          Training device backend: 'cpu', 'gpu', 'cuda', or 'auto' (default: auto)");
    println!(
        "  --precision <prec>      Training precision: 'fp32', 'fp16', or 'auto' (default: auto)"
    );
    println!(
        "                            auto = fp16 when GPU available + compute >= 5.0, else fp32"
    );
    println!("  --epochs <n>            Number of training epochs (default: 1, must be >= 1)");
    println!("  --learning-rate <f>     Learning rate for AdamW optimizer (default: 0.001, must be > 0.0)");
    println!(
        "  --batch-size <n>        Gradient accumulation batch size (default: 4, must be >= 1)"
    );
    println!("  --max-steps <n>         Maximum training steps ceiling (optional, default: entire epoch)");
    println!(
        "  --checkpoint-dir <path> Directory for persistent periodic checkpoints and resumption"
    );
    println!(
        "  --checkpoint-interval <n> Periodic checkpoint saving interval in steps (default: 500)"
    );
    println!("  --resume                Resume training from last saved checkpoint if present");
    println!("  --resume-from <path>    Explicit checkpoint or model directory to resume from");
    println!("  --max-memory-mb <f>     Maximum memory footprint ceiling in MB (default: 12288.0 on Colab)");
    println!("  -h, --help              Print this help message");
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut curriculum_path: Option<String> =
        Some("storage/datasets/curriculum/foundational_curriculum.jsonl".to_string());
    let mut curriculum_explicitly_set = false;
    let mut dataset_explicitly_set = false;
    let mut dataset_dir = "storage/datasets/main_dataset_v1".to_string();
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
    let mut max_memory_mb: Option<f64> = env::var("TARA_MAX_MEMORY_MB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|&v| v >= 512.0);

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
                curriculum_explicitly_set = true;
                curriculum_path = if args[i].is_empty() || args[i].eq_ignore_ascii_case("none") {
                    None
                } else {
                    Some(args[i].clone())
                };
            }
            "--device" => {
                i += 1;
                if i >= args.len() {
                    eprintln!(
                        "Error: --device requires an argument ('cpu', 'gpu', 'cuda', or 'auto')"
                    );
                    exit(1);
                }
                let d = args[i].to_lowercase();
                if d != "cpu" && d != "gpu" && d != "cuda" && d != "auto" {
                    eprintln!(
                        "Error: invalid --device '{}'. Valid options: 'cpu', 'gpu', 'cuda', 'auto'",
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
                dataset_explicitly_set = true;
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
                if epochs == 0 {
                    eprintln!("Error: --epochs must be >= 1");
                    exit(1);
                }
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
                if !learning_rate.is_finite() || learning_rate <= 0.0 {
                    eprintln!("Error: --learning-rate must be a finite positive float (> 0.0)");
                    exit(1);
                }
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
                if batch_size == 0 {
                    eprintln!("Error: --batch-size must be >= 1");
                    exit(1);
                }
            }
            "--max-steps" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --max-steps requires an integer argument");
                    exit(1);
                }
                let ms = args[i].parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("Error: invalid integer for --max-steps: {}", args[i]);
                    exit(1);
                });
                if ms == 0 {
                    eprintln!("Error: --max-steps must be >= 1");
                    exit(1);
                }
                max_steps = Some(ms);
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
                let ci = args[i].parse::<usize>().unwrap_or_else(|_| {
                    eprintln!(
                        "Error: invalid integer for --checkpoint-interval: {}",
                        args[i]
                    );
                    exit(1);
                });
                if ci == 0 {
                    eprintln!("Error: --checkpoint-interval must be >= 1");
                    exit(1);
                }
                checkpoint_interval = Some(ci);
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
                let mm = args[i].parse::<f64>().unwrap_or_else(|_| {
                    eprintln!("Error: invalid float for --max-memory-mb: {}", args[i]);
                    exit(1);
                });
                if !mm.is_finite() || mm < 512.0 {
                    eprintln!("Error: --max-memory-mb must be >= 512.0");
                    exit(1);
                }
                max_memory_mb = Some(mm);
            }
            unknown => {
                eprintln!("Error: unrecognized option '{}'", unknown);
                print_usage();
                exit(1);
            }
        }
        i += 1;
    }

    // If user explicitly passed --dataset without --curriculum, disable default curriculum override
    if dataset_explicitly_set && !curriculum_explicitly_set {
        curriculum_path = None;
    }

    // Safety guard: never write into production model or overlapping paths
    if let Some(ref out) = output_dir {
        let out_path = Path::new(out);
        if has_parent_dir_traversal(out_path) {
            eprintln!(
                "CRITICAL ERROR: --output cannot contain parent directory traversal ('..'): {}",
                out
            );
            exit(1);
        }

        let forbidden_targets = [
            Path::new("storage/models/tara"),
            Path::new("production/neural"),
            Path::new("production/world_model"),
            Path::new(&model_dir),
        ];

        for forbidden in &forbidden_targets {
            if paths_overlap(out_path, forbidden) {
                eprintln!(
                    "CRITICAL ERROR: --output cannot overlap with '{}'. Training writes to candidate directories only.",
                    forbidden.display()
                );
                exit(1);
            }
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
    let _ = std::io::stdout().flush();

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
    let _ = std::io::stdout().flush();

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
    let best_loss = result
        .get("best_loss")
        .and_then(|v| v.as_f64())
        .unwrap_or(loss_after.min(loss_before));
    let best_step = result.get("best_step").and_then(|v| v.as_u64());
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
    println!("  Loss After (Final Step):  {:.6}", loss_after);
    if let Some(bs) = best_step {
        println!(
            "  Best Checkpoint Loss:     {:.6} (at step {})",
            best_loss, bs
        );
    } else {
        println!("  Best Loss Observed:       {:.6}", best_loss);
    }
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
