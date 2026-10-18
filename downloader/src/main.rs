//! Standalone Dataset Downloader CLI
//!
//! Provides isolated execution for dataset ingestion verification,
//! cryptographic integrity auditing, and provenance ledger maintenance.

use std::env;
use std::path::PathBuf;
use std::process;
use tara_standalone_downloader::{DownloaderService, LicenseEvaluator};

fn print_usage(prog_name: &str) {
    println!("Standalone Dataset Downloader — Operational Tool");
    println!("Usage:");
    println!("  {} --verify --file <path> --url <source_url> --license <license> --proof <proof_url> [--dir <output_dir>]", prog_name);
    println!("  {} --ledger [--dir <output_dir>]", prog_name);
    println!("  {} --check-license <license_id>", prog_name);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage(&args[0]);
        process::exit(1);
    }

    let mut mode = String::new();
    let mut file_path = String::new();
    let mut source_url = String::new();
    let mut license_id = String::new();
    let mut proof_url = String::new();
    let mut output_dir: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
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
            "--file" => {
                if i + 1 < args.len() {
                    file_path = args[i + 1].clone();
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

    match mode.as_str() {
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
}
