//! # TARA GitHub Sync & Cross-Platform CLI Runner (100% Native Rust)
//!
//! Enforces:
//! - POSIX relative path verification across all architecture maps.
//! - Pre-push repository validation (0 warnings, 0 secrets, 0 hardcoded SHAs).
//! - Generation of .github/workflows/tara_ci.yml for Linux/Windows CI matrix.
//! - Generation of ARCHITECTURE/GITHUB_EXPORT_MANIFEST.json.
//! - Post-clone import integrity verification on remote GitHub runners.

use std::env;
use tara_engine::github_adapter::{GitHubAdapter, GitHubAdapterConfig};

fn print_usage() {
    println!("================================================================================");
    println!("  TARA GITHUB ADAPTER & CROSS-PLATFORM SYNC ENGINE");
    println!("================================================================================");
    println!("Usage: cargo run -p tara_engine --bin tara_github_sync -- [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --verify        Run pre-push verification audit (default)");
    println!("  --export        Run audit and export ARCHITECTURE/GITHUB_EXPORT_MANIFEST.json");
    println!("  --import-check  Verify dynamic SHA-256 integrity after clone from GitHub");
    println!("  -h, --help      Display this help information");
    println!("================================================================================");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return Ok(());
    }

    let mode_export = args.iter().any(|a| a == "--export");
    let mode_import = args.iter().any(|a| a == "--import-check");

    println!("================================================================================");
    println!("  TARA GITHUB ADAPTER — CROSS-PLATFORM (WINDOWS / GITHUB / LINUX) AUDIT");
    println!("================================================================================");

    let config = GitHubAdapterConfig::default();
    println!("Workspace Root : {}", config.workspace_root.display());
    println!("Architecture   : {}", config.arch_dir.display());
    println!("--------------------------------------------------------------------------------");

    let adapter = GitHubAdapter::new(config);

    if mode_import {
        println!("=== RUNNING POST-CLONE / IMPORT INTEGRITY VERIFICATION ===");
        match adapter.verify_github_import() {
            Ok(total_checked) => {
                println!("PASS: Successfully verified {total_checked} files on disk against registry.");
                println!("SUCCESS: GitHub import verified 100% intact.");
                return Ok(());
            }
            Err(errors) => {
                eprintln!("ERROR: GitHub import verification failed with {} issues:", errors.len());
                for err in &errors {
                    eprintln!("  - {err}");
                }
                std::process::exit(1);
            }
        }
    }

    println!("=== 1. AUDITING CROSS-PLATFORM PATHS & GIT REPOSITORY READINESS ===");
    let audit = adapter.pre_push_audit()?;

    println!("  POSIX Paths Verified       : {}", audit.total_posix_paths_verified);
    println!("  Architecture Engine Status  : {}", audit.architecture_status);
    println!("  Total Files in Architecture : {}", audit.total_indexed_files);
    println!("  Missing Disk Files Count   : {}", audit.missing_disk_files.len());
    println!("  Unindexed Rust Files Count  : {}", audit.unindexed_rust_files.len());
    println!("  Compiler Warnings Count     : {}", audit.compiler_warnings_count);
    println!("  Hardcoded Secrets Found     : {}", audit.hardcoded_secrets_found.len());
    println!("  Hardcoded SHA-256 Literals  : {}", audit.hardcoded_sha_found.len());
    println!("  .github/ Unblocked in git   : {}", audit.gitignore_github_unblocked);
    println!("  Audit Duration              : {} ms", audit.duration_ms);

    if !audit.non_posix_paths_found.is_empty() {
        eprintln!("ERROR: Non-POSIX backslash paths found in architecture:");
        for p in &audit.non_posix_paths_found {
            eprintln!("  - {p}");
        }
    }

    if !audit.missing_disk_files.is_empty() {
        eprintln!("ERROR: Missing files referenced in SOURCE_INDEX.json:");
        for p in &audit.missing_disk_files {
            eprintln!("  - {p}");
        }
    }

    if !audit.unindexed_rust_files.is_empty() {
        eprintln!("ERROR: Unindexed Rust source files found on disk:");
        for p in &audit.unindexed_rust_files {
            eprintln!("  - {p}");
        }
    }

    if audit.compiler_warnings_count > 0 {
        eprintln!("ERROR: Compiler warnings detected! (Rule 7 requires 0 warnings)");
    }

    if !audit.hardcoded_secrets_found.is_empty() {
        eprintln!("ERROR: Hardcoded secrets found (Rule 12):");
        for s in &audit.hardcoded_secrets_found {
            eprintln!("  - {s}");
        }
    }

    if !audit.gitignore_github_unblocked {
        eprintln!("ERROR: .github/ is blocked in .gitignore! GitHub Actions will not sync.");
    }

    if !audit.is_ready_for_push {
        eprintln!("--------------------------------------------------------------------------------");
        eprintln!("FAILURE: Repository is NOT ready for GitHub push. Fix above errors first.");
        eprintln!("================================================================================");
        std::process::exit(1);
    }

    println!("\nPASS: All cross-platform checks passed with 0 errors and 0 warnings.");

    if mode_export || true {
        println!("\n=== 2. EXPORTING GITHUB BUNDLE & CI WORKFLOW ===");
        let manifest = adapter.export_github_bundle()?;
        println!("  Generated: ARCHITECTURE/GITHUB_EXPORT_MANIFEST.json");
        println!("  Generated: .github/workflows/tara_ci.yml");
        println!("  Platforms: {:?}", manifest.supported_platforms);
        println!("  Digest   : {}", manifest.registries_digest_sha256);
        println!("PASS: GitHub export bundle verified.");
    }

    println!("--------------------------------------------------------------------------------");
    println!("SUCCESS: Repository is 100% verified and ready for GitHub push (Windows + Linux).");
    println!("================================================================================");

    Ok(())
}
