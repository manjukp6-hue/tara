// process_restart_proof.rs
//
// Tests OS-level restart persistence:
// Process A: writes a state value and terminates.
// Process B: reads the state value written by A and proves it survived.
//
// Usage:
//   cargo run --bin process_restart_proof -- write   → writes state, exits (Process A)
//   cargo run --bin process_restart_proof -- read    → reads state, proves persistence (Process B)

use std::fs;
use std::path::PathBuf;

fn state_file() -> PathBuf {
    let root = std::env::var("TARA_REPO_ROOT").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(root).join("storage/persistence/restart_proof_state.json")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("write");

    match mode {
        "write" => run_write(),
        "read" => run_read(),
        _ => {
            eprintln!("Usage: process_restart_proof [write|read]");
            std::process::exit(1);
        }
    }
}

fn run_write() {
    let pid = std::process::id();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let magic = format!("TARA_RESTART_PROOF_{}", timestamp);

    let state = serde_json::json!({
        "pid_a": pid,
        "magic_value": magic,
        "written_at_unix": timestamp,
        "written_at_iso": tara_engine::now_iso(),
        "test": "Process A state — must survive full process termination"
    });

    let path = state_file();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("Cannot create state directory");
    }
    fs::write(&path, serde_json::to_string_pretty(&state).unwrap())
        .expect("Cannot write state file");

    println!("=== PROCESS A: STATE WRITTEN ===");
    println!("PID A: {}", pid);
    println!("Magic value: {}", magic);
    println!("State written to: {}", path.display());
    println!("Process A will now exit. Run with 'read' mode as Process B.");
    // Process A terminates here
}

fn run_read() {
    let pid = std::process::id();
    let path = state_file();

    println!("=== PROCESS B: READING PERSISTED STATE ===");
    println!("PID B: {}", pid);
    println!("Reading state from: {}", path.display());

    let raw = fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("FAIL: Cannot read state file: {}", e);
        std::process::exit(1);
    });

    let state: serde_json::Value = serde_json::from_str(&raw).unwrap_or_else(|e| {
        eprintln!("FAIL: Cannot parse state JSON: {}", e);
        std::process::exit(1);
    });

    let pid_a = state["pid_a"].as_u64().unwrap_or(0);
    let magic = state["magic_value"].as_str().unwrap_or("MISSING");
    let written_at = state["written_at_iso"].as_str().unwrap_or("MISSING");

    println!("PID A (from state): {}", pid_a);
    println!("PID B (current): {}", pid);
    println!("Magic value read: {}", magic);
    println!("Written at: {}", written_at);

    // Verify: B's PID must be different from A's PID
    if pid_a == pid as u64 {
        eprintln!(
            "FAIL: PID A == PID B ({}) — not a real process restart!",
            pid
        );
        std::process::exit(1);
    }
    if magic.is_empty() || magic == "MISSING" {
        eprintln!("FAIL: Magic value missing from state");
        std::process::exit(1);
    }

    println!("\n=== RESTART PERSISTENCE PROOF ===");
    println!(
        "✓ PID A ({}) ≠ PID B ({}) — real OS-level process restart confirmed",
        pid_a, pid
    );
    println!("✓ State written by Process A survived termination");
    println!("✓ Process B successfully read magic value: {}", magic);
    println!("=== RESTART PROOF COMPLETE ===");
}
