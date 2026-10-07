//! TARA Canonical Dataset Splitter & Leakage Control Engine Binary.
//!
//! Provides deterministic partitioning of canonical datasets into train, val,
//! and test splits, followed by full cross-split leakage verification (exact match
//! and 13-gram shingle contamination checks).

use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tara_engine::dataset::compute_file_sha256;
use tara_engine::dataset::splitter::{DatasetSplitter, LeakageChecker, SplitRatio};

fn print_usage() {
    println!("Usage: tara_dataset_splitter [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --input-dir, --input <PATH>    Path to canonical dataset file or directory containing .jsonl shards");
    println!("  --output-dir, --output <PATH>  Output directory for split files (default: <input-dir>/splits)");
    println!("  --train <RATIO>                Training split ratio (default: 0.80)");
    println!("  --val <RATIO>                  Validation split ratio (default: 0.10)");
    println!("  --test <RATIO>                 Testing split ratio (default: 0.10)");
    println!("  --skip-leakage                 Skip cross-split leakage verification");
    println!("  -h, --help                     Print help information");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return Ok(());
    }

    let mut input_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut train_ratio = 0.80f64;
    let mut val_ratio = 0.10f64;
    let mut test_ratio = 0.10f64;
    let mut skip_leakage = false;

    let mut idx = 1;
    while idx < args.len() {
        match args[idx].as_str() {
            "--input-dir" | "--input" => {
                if idx + 1 < args.len() {
                    input_path = Some(PathBuf::from(&args[idx + 1]));
                    idx += 2;
                } else {
                    return Err("Missing argument for --input-dir".into());
                }
            }
            "--output-dir" | "--output" => {
                if idx + 1 < args.len() {
                    output_path = Some(PathBuf::from(&args[idx + 1]));
                    idx += 2;
                } else {
                    return Err("Missing argument for --output-dir".into());
                }
            }
            "--train" => {
                if idx + 1 < args.len() {
                    train_ratio = args[idx + 1].parse::<f64>()?;
                    idx += 2;
                } else {
                    return Err("Missing argument for --train".into());
                }
            }
            "--val" => {
                if idx + 1 < args.len() {
                    val_ratio = args[idx + 1].parse::<f64>()?;
                    idx += 2;
                } else {
                    return Err("Missing argument for --val".into());
                }
            }
            "--test" => {
                if idx + 1 < args.len() {
                    test_ratio = args[idx + 1].parse::<f64>()?;
                    idx += 2;
                } else {
                    return Err("Missing argument for --test".into());
                }
            }
            "--skip-leakage" => {
                skip_leakage = true;
                idx += 1;
            }
            unknown => {
                return Err(format!("Unknown option: {unknown}").into());
            }
        }
    }

    let input_dir = match input_path {
        Some(p) => p,
        None => {
            print_usage();
            return Err("Error: --input-dir is required".into());
        }
    };

    if !input_dir.exists() {
        return Err(format!("Input path does not exist: {}", input_dir.display()).into());
    }

    let output_dir = match output_path {
        Some(p) => p,
        None => {
            if input_dir.is_dir() {
                input_dir.join("splits")
            } else {
                input_dir
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join("splits")
            }
        }
    };

    println!("================================================================================");
    println!(" TARA CANONICAL DATASET SPLITTER & LEAKAGE AUDITOR");
    println!("================================================================================");
    println!("Input Source : {}", input_dir.display());
    println!("Output Target: {}", output_dir.display());
    println!(
        "Split Targets: Train = {:.2} | Val = {:.2} | Test = {:.2}",
        train_ratio, val_ratio, test_ratio
    );
    println!("--------------------------------------------------------------------------------");

    let split_ratio = SplitRatio::new(train_ratio, val_ratio, test_ratio)?;

    let split_start = Instant::now();
    let report = DatasetSplitter::split(&input_dir, &output_dir, split_ratio)?;
    let split_elapsed = split_start.elapsed();

    let train_pct = (report.train_samples as f64 / report.total_samples as f64) * 100.0;
    let val_pct = (report.val_samples as f64 / report.total_samples as f64) * 100.0;
    let test_pct = (report.test_samples as f64 / report.total_samples as f64) * 100.0;

    let sum_parts = report.train_samples + report.val_samples + report.test_samples;
    let conservation_pass = sum_parts == report.total_samples;

    println!("[SPLIT EXECUTION COMPLETE]");
    println!("  Total Canonical Records : {}", report.total_samples);
    println!(
        "  Train Samples           : {} ({:.2}%) -> {}",
        report.train_samples, train_pct, report.train_path
    );
    println!(
        "  Val Samples             : {} ({:.2}%) -> {}",
        report.val_samples, val_pct, report.val_path
    );
    println!(
        "  Test Samples            : {} ({:.2}%) -> {}",
        report.test_samples, test_pct, report.test_path
    );
    println!(
        "  Conservation Check      : {} = {} + {} + {} -> {}",
        report.total_samples,
        report.train_samples,
        report.val_samples,
        report.test_samples,
        if conservation_pass {
            "PASS (100% Exact Match)"
        } else {
            "FAIL (Record Count Mismatch)"
        }
    );
    println!("  Split Wall Time         : {:.2?}", split_elapsed);
    println!("--------------------------------------------------------------------------------");

    if !conservation_pass {
        return Err("Conservation check failed: sum of splits != total records".into());
    }

    println!("[COMPUTING DYNAMIC CRYPTOGRAPHIC SHAS]");
    let train_sha = compute_file_sha256(Path::new(&report.train_path)).map_err(std::io::Error::other)?;
    let val_sha = compute_file_sha256(Path::new(&report.val_path)).map_err(std::io::Error::other)?;
    let test_sha = compute_file_sha256(Path::new(&report.test_path)).map_err(std::io::Error::other)?;
    println!("  TRAIN SHA-256 : {}", train_sha);
    println!("  VAL   SHA-256 : {}", val_sha);
    println!("  TEST  SHA-256 : {}", test_sha);
    println!("--------------------------------------------------------------------------------");

    if !skip_leakage {
        println!("[EXECUTING CROSS-SPLIT LEAKAGE AUDIT]");

        // 1. Train <-> Val
        let tv_start = Instant::now();
        let tv = LeakageChecker::check_leakage(&report.train_path, &report.val_path)?;
        let tv_elapsed = tv_start.elapsed();
        println!(
            "  Train <-> Val  : exact_match = {} | ngram_shingle = {} | total_leaked = {} (time: {:.2?}) -> {}",
            tv.exact_match_leaks,
            tv.ngram_shingle_leaks,
            tv.total_leaked_samples,
            tv_elapsed,
            if tv.is_clean { "CLEAN (0.00% Leakage)" } else { "CONTAMINATED" }
        );

        // 2. Train <-> Test
        let tt_start = Instant::now();
        let tt = LeakageChecker::check_leakage(&report.train_path, &report.test_path)?;
        let tt_elapsed = tt_start.elapsed();
        println!(
            "  Train <-> Test : exact_match = {} | ngram_shingle = {} | total_leaked = {} (time: {:.2?}) -> {}",
            tt.exact_match_leaks,
            tt.ngram_shingle_leaks,
            tt.total_leaked_samples,
            tt_elapsed,
            if tt.is_clean { "CLEAN (0.00% Leakage)" } else { "CONTAMINATED" }
        );

        // 3. Val <-> Test
        let vt_start = Instant::now();
        let vt = LeakageChecker::check_leakage(&report.val_path, &report.test_path)?;
        let vt_elapsed = vt_start.elapsed();
        println!(
            "  Val   <-> Test : exact_match = {} | ngram_shingle = {} | total_leaked = {} (time: {:.2?}) -> {}",
            vt.exact_match_leaks,
            vt.ngram_shingle_leaks,
            vt.total_leaked_samples,
            vt_elapsed,
            if vt.is_clean { "CLEAN (0.00% Leakage)" } else { "CONTAMINATED" }
        );

        let all_clean = tv.is_clean && tt.is_clean && vt.is_clean;
        println!(
            "--------------------------------------------------------------------------------"
        );
        if all_clean {
            println!("  Overall Cross-Split Leakage: ZERO CONTAMINATION DETECTED (PASSED)");
        } else {
            return Err("Cross-split contamination detected during leakage audit!".into());
        }
    }

    println!("================================================================================");
    println!(" GATE 13: FULL CANONICAL DATASET SPLIT AUDIT -> VERIFIED SUCCESS");
    println!("================================================================================");

    Ok(())
}
