//! Standalone Dataset Downloader CLI
//!
//! Provides isolated execution for dataset ingestion verification,
//! cryptographic integrity auditing, and provenance ledger maintenance.

use std::env;
use std::path::{Path, PathBuf};
use std::process;
use tara_standalone_downloader::{DownloaderService, LicenseEvaluator};

fn print_usage(prog_name: &str) {
    println!("Standalone Dataset Downloader — Operational Tool");
    println!("Usage:");
    println!("  {} --download <url> --file-name <name> --license <license> [--proof <proof_url>] [--dir <output_dir>]", prog_name);
    println!("  {} --verify --file <path> --url <source_url> --license <license> --proof <proof_url> [--dir <output_dir>]", prog_name);
    println!("  {} --ledger [--dir <output_dir>]", prog_name);
    println!("  {} --check-license <license_id>", prog_name);
    println!("  {} --shard-all --input-dir <dir> --output-dir <dir> [--max-mb <mb>]", prog_name);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage(&args[0]);
        process::exit(1);
    }

    let mut mode = String::new();
    let mut file_path = String::new();
    let mut file_name = String::new();
    let mut source_url = String::new();
    let mut license_id = String::new();
    let mut proof_url = String::new();
    let mut output_dir: Option<PathBuf> = None;
    let mut input_dir: Option<PathBuf> = None;
    let mut max_mb: u64 = 500;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--download" => {
                mode = "download".to_string();
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    source_url = args[i + 1].clone();
                    i += 1;
                }
            }
            "--file-name" => {
                if i + 1 < args.len() {
                    file_name = args[i + 1].clone();
                    i += 1;
                }
            }
            "--shard-all" => {
                mode = "shard-all".to_string();
            }
            "--input-dir" => {
                if i + 1 < args.len() {
                    input_dir = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--output-dir" => {
                if i + 1 < args.len() {
                    output_dir = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--max-mb" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<u64>() {
                        max_mb = v;
                    }
                    i += 1;
                }
            }
            "--verify" => {
                mode = "verify".to_string();
            }
            "--ledger" => {
                mode = "ledger".to_string();
            }
            "--check-license" => {
                mode = "check-license".to_string();
                if i + 1 < args.len() {
                    license_id = args[i + 1].clone();
                    i += 1;
                }
            }
            "--file" | "--output" => {
                if i + 1 < args.len() {
                    file_path = args[i + 1].clone();
                    if file_name.is_empty() {
                        file_name = args[i + 1].clone();
                    }
                    i += 1;
                }
            }
            "--url" => {
                if i + 1 < args.len() {
                    source_url = args[i + 1].clone();
                    i += 1;
                }
            }
            "--license" => {
                if i + 1 < args.len() {
                    license_id = args[i + 1].clone();
                    i += 1;
                }
            }
            "--proof" => {
                if i + 1 < args.len() {
                    proof_url = args[i + 1].clone();
                    i += 1;
                }
            }
            "--dir" => {
                if i + 1 < args.len() {
                    output_dir = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--help" | "-h" => {
                print_usage(&args[0]);
                process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    if file_name.is_empty() && !file_path.is_empty() {
        file_name = file_path.clone();
    }
    if file_path.is_empty() && !file_name.is_empty() {
        file_path = file_name.clone();
    }

    match mode.as_str() {
        "shard-all" => {
            let in_dir = match input_dir {
                Some(d) => d,
                None => {
                    eprintln!("Error: --input-dir is required for shard-all");
                    process::exit(1);
                }
            };
            let out_dir = match output_dir {
                Some(d) => d,
                None => {
                    eprintln!("Error: --output-dir is required for shard-all");
                    process::exit(1);
                }
            };

            if let Err(e) = std::fs::create_dir_all(&out_dir) {
                eprintln!("Failed to create output directory: {}", e);
                process::exit(1);
            }

            fn collect_files(dir: &Path, parquets: &mut Vec<PathBuf>) {
                if let Ok(entries) = std::fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.is_dir() {
                            collect_files(&path, parquets);
                        } else if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                            if ext == "parquet" {
                                parquets.push(path);
                            }
                        }
                    }
                }
            }

            let mut parquet_files = Vec::new();
            collect_files(&in_dir, &mut parquet_files);
            parquet_files.sort();

            println!(
                "Found {} Parquet files to convert to JSONL (max {} MB/shard)...",
                parquet_files.len(),
                max_mb
            );

            let max_bytes = max_mb * 1024 * 1024;
            // Dynamically discover highest existing shard index in out_dir to avoid hardcoding or gaps
            let mut shard_idx = 0;
            if let Ok(entries) = std::fs::read_dir(&out_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(file_name) = path.file_name().and_then(|s| s.to_str()) {
                        if file_name.starts_with("shard_") && file_name.ends_with(".jsonl") {
                            let stem = &file_name[6..file_name.len() - 6];
                            if let Ok(idx) = stem.parse::<usize>() {
                                if idx >= shard_idx {
                                    shard_idx = idx + 1;
                                }
                            }
                        }
                    }
                }
            }
            let mut current_bytes = 0;
            let mut total_records = 0;
            let mut current_file: Option<std::fs::File> = None;

            use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
            use std::io::Write;

            for (p_idx, p_path) in parquet_files.iter().enumerate() {
                if (p_idx + 1) % 20 == 0 || p_idx == 0 || p_idx + 1 == parquet_files.len() {
                    println!("Converting parquet [{}/{}] {:?}", p_idx + 1, parquet_files.len(), p_path.file_name().unwrap_or_default());
                }

                let file = match std::fs::File::open(p_path) {
                    Ok(f) => f,
                    Err(e) => {
                        eprintln!("Warning: skipping unreadable parquet {:?}: {}", p_path, e);
                        continue;
                    }
                };

                let builder = match ParquetRecordBatchReaderBuilder::try_new(file) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("Warning: failed reading parquet {:?}: {}", p_path, e);
                        continue;
                    }
                };

                let mut reader = match builder.build() {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("Warning: failed building parquet reader {:?}: {}", p_path, e);
                        continue;
                    }
                };

                for maybe_batch in &mut reader {
                    let batch = match maybe_batch {
                        Ok(b) => b,
                        Err(e) => {
                            eprintln!("Warning: batch error in {:?}: {}", p_path, e);
                            continue;
                        }
                    };

                    let mut json_buf = Vec::new();
                    {
                        let mut writer = arrow_json::LineDelimitedWriter::new(&mut json_buf);
                        if let Err(e) = writer.write(&batch) {
                            eprintln!("Warning: failed writing batch to json: {}", e);
                            continue;
                        }
                        let _ = writer.finish();
                    }

                    let json_str = match std::str::from_utf8(&json_buf) {
                        Ok(s) => s,
                        Err(_) => continue,
                    };

                    for line in json_str.lines() {
                        if line.trim().is_empty() {
                            continue;
                        }
                        if current_file.is_none() || current_bytes >= max_bytes {
                            let shard_name = format!("shard_{:05}.jsonl", shard_idx);
                            let shard_path = out_dir.join(&shard_name);
                            println!("Writing to new shard: {:?}", shard_path);
                            let new_f = match std::fs::File::create(&shard_path) {
                                Ok(f) => f,
                                Err(e) => {
                                    eprintln!("Failed to create shard {:?}: {}", shard_path, e);
                                    process::exit(1);
                                }
                            };
                            current_file = Some(new_f);
                            current_bytes = 0;
                            shard_idx += 1;
                        }

                        if let Some(ref mut out_f) = current_file {
                            let bytes = line.as_bytes();
                            out_f.write_all(bytes)?;
                            out_f.write_all(b"\n")?;
                            current_bytes += bytes.len() as u64 + 1;
                            total_records += 1;
                        }
                    }
                }
            }

            println!("All Parquet files converted and sharded successfully!");
            println!("Total additional records written: {}", total_records);
            println!("Final shard index reached: {}", shard_idx);
        }
        "download" => {
            if source_url.is_empty() || file_name.is_empty() || license_id.is_empty() {
                eprintln!("Error: --download <URL>, --file-name <NAME>, and --license <LICENSE> are required for download.");
                process::exit(1);
            }
            let mut service = match DownloaderService::new(output_dir) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Failed to initialize downloader service: {}", e);
                    process::exit(1);
                }
            };
            println!("Downloading from {} to {} ...", source_url, file_name);
            match service.download_and_register_asset(
                &source_url,
                &file_name,
                &license_id,
                &proof_url,
            ) {
                Ok(rec) => {
                    println!("Asset downloaded, verified, and registered successfully!");
                    println!("File: {}", file_name);
                    println!("SHA-256: {}", rec.sha256);
                    println!("Bytes: {}", rec.file_size_bytes);
                    println!("License: {}", rec.license);
                    println!("Status: {}", rec.status);
                }
                Err(e) => {
                    eprintln!("Download failed: {}", e);
                    process::exit(1);
                }
            }
        }
        "check-license" => {
            if license_id.is_empty() {
                eprintln!("Error: Missing license identifier to evaluate.");
                process::exit(1);
            }
            let compliant = LicenseEvaluator::is_permissive_compliant(&license_id);
            println!(
                "License '{}': Permissive Commercial Compliance = {}",
                license_id, compliant
            );
            if !compliant {
                process::exit(2);
            }
        }
        "ledger" => {
            let service = match DownloaderService::new(output_dir) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Failed to initialize downloader service: {}", e);
                    process::exit(1);
                }
            };
            println!("Target directory: {:?}", service.target_dir());
            println!(
                "Registered verified assets: {}",
                service.ledger().records.len()
            );
            for (key, record) in &service.ledger().records {
                println!(
                    " - [{}] {} | sha256: {} | {} bytes | license: {}",
                    record.status, key, record.sha256, record.file_size_bytes, record.license
                );
            }
        }
        "verify" => {
            if file_path.is_empty() || source_url.is_empty() || license_id.is_empty() {
                eprintln!("Error: --file, --url, and --license are required for verification.");
                process::exit(1);
            }
            let mut service = match DownloaderService::new(output_dir) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Failed to initialize downloader service: {}", e);
                    process::exit(1);
                }
            };
            match service.verify_and_register_asset(
                &file_path,
                &source_url,
                &proof_url,
                &license_id,
            ) {
                Ok(rec) => {
                    println!("Asset verified and registered successfully!");
                    println!("SHA-256: {}", rec.sha256);
                    println!("Bytes: {}", rec.file_size_bytes);
                    println!("Status: {}", rec.status);
                }
                Err(e) => {
                    eprintln!("Verification rejected: {}", e);
                    process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("Error: Unknown or missing operational mode.");
            print_usage(&args[0]);
            process::exit(1);
        }
    }

    Ok(())
}
