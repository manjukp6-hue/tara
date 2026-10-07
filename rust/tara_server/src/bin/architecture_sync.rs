//! TARA Architecture Sync Engine Standalone CLI Runner (100% Native Rust).
//!
//! Enforces:
//! - Full filesystem reconciliation against project registries.
//! - Live continuous watcher mode for filesystem event synchronization.
//! - Read-only observation of project files: NEVER mutates project code.
//! - Updates:
//!   * ARCHITECTURE/project_tree.json
//!   * ARCHITECTURE/file_registry.json
//!   * ARCHITECTURE/folder_registry.json
//!   * ARCHITECTURE/architecture_state.json
//! - Zero Python, zero mocks, dynamic SHA-256 integrity.

use std::env;
use std::sync::Arc;
use tara_server::runtime::architecture_sync::{
    find_workspace_root, ArchitectureSyncConfig, ArchitectureSyncEngine,
};

fn print_usage() {
    println!("================================================================================");
    println!("  TARA ARCHITECTURE SYNC ENGINE (100% NATIVE RUST)");
    println!("================================================================================");
    println!("Usage: cargo run -p tara_server --bin architecture_sync -- [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --reconcile     Execute Layer 2 Full Reconciler immediately and exit (Default)");
    println!("  --watch         Start Layer 1 Live Watcher in continuous daemon mode");
    println!("  --status        Read and display current architecture state snapshot");
    println!("  --interval <ms> Polling/debounce interval in milliseconds (default: 500)");
    println!("  -h, --help      Display this help information");
    println!("================================================================================");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return Ok(());
    }

    let ws_root = find_workspace_root().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let mut config = ArchitectureSyncConfig {
        workspace_root: ws_root,
        ..Default::default()
    };

    let mut mode_watch = false;
    let mut mode_status = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--watch" => {
                mode_watch = true;
                i += 1;
            }
            "--reconcile" => {
                mode_watch = false;
                i += 1;
            }
            "--status" => {
                mode_status = true;
                i += 1;
            }
            "--interval" if i + 1 < args.len() => {
                if let Ok(ms) = args[i + 1].parse() {
                    config.poll_interval_ms = ms;
                }
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    let engine = Arc::new(ArchitectureSyncEngine::new(config));

    if mode_status {
        let state_path = engine.config.arch_dir.join("architecture_state.json");
        if state_path.exists() {
            let content = std::fs::read_to_string(&state_path)?;
            println!("Current Architecture State:\n{}", content);
        } else {
            println!("No architecture state found at {:?}", state_path);
        }
        return Ok(());
    }

    println!("================================================================================");
    println!("  TARA ARCHITECTURE SYNC ENGINE");
    println!("================================================================================");
    println!("Workspace Root : {}", engine.config.workspace_root.display());
    println!("Architecture Dir: {}", engine.config.arch_dir.display());
    println!("Ignored Dirs    : {:?}", engine.config.ignored_dirs);
    println!("--------------------------------------------------------------------------------");

    println!("=== LAYER 2: EXECUTING FULL RECONCILIATION ===");
    let report = engine.reconcile_full()?;

    println!("PASS: Full Reconciliation Completed in {} ms.", report.duration_ms);
    println!("  Total Files Indexed  : {}", report.total_files);
    println!("  Total Folders Indexed: {}", report.total_folders);
    println!("  Files Added          : {}", report.files_added.len());
    println!("  Files Updated        : {}", report.files_updated.len());
    println!("  Files Removed        : {}", report.files_removed.len());
    println!("  Folders Added        : {}", report.folders_added.len());
    println!("  Folders Removed      : {}", report.folders_removed.len());
    println!("--------------------------------------------------------------------------------");
    println!("  Artifacts Generated:");
    println!("  1. project_tree.json       -> Hierarchical project filesystem tree");
    println!("  2. file_registry.json      -> Flat index with dynamic SHA-256 and lines");
    println!("  3. folder_registry.json    -> Folder depth and child counts");
    println!("  4. architecture_state.json -> Operational status & registry integrity");
    println!("================================================================================");

    if mode_watch {
        println!("\n=== LAYER 1: STARTING CONTINUOUS LIVE WATCHER ===");
        println!("Press Ctrl+C to terminate live watcher.");
        let _shutdown_tx = engine.clone().start_live_watcher()?;

        // Keep main thread alive
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }

    Ok(())
}
