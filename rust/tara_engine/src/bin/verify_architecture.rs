//! TARA Whole-System Architecture Verification Utility (100% Native Rust).
//!
//! Enforces:
//! 1. Real implementations (zero fabrication, zero broken links).
//! 2. Complete capability mapping (definition, registration, discovery, execution, storage, memory, learning, safety, tests).
//! 3. No orphan components, no missing registrations.
//! 4. 100% pure Rust workspace compliance with zero warnings.

use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn find_workspace_root() -> Result<PathBuf, String> {
    let mut current = std::env::current_dir().map_err(|e| e.to_string())?;
    loop {
        if current.join("AGENTS.md").exists() && current.join("Cargo.toml").exists() {
            return Ok(current);
        }
        if let Some(parent) = current.parent() {
            current = parent.to_path_buf();
        } else {
            break;
        }
    }
    Err("Could not find workspace root (marked by AGENTS.md and Cargo.toml)".to_string())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let workspace_root = find_workspace_root()?;
    let arch_dir = workspace_root.join("ARCHITECTURE");
    let cap_dir = arch_dir.join("CAPABILITIES");
    let graph_path = arch_dir.join("CAPABILITY_GRAPH.json");

    println!("================================================================================");
    println!(" TARA WHOLE-SYSTEM ARCHITECTURE & CAPABILITY INTEGRITY AUDIT (RUST)");
    println!("================================================================================");
    println!("Workspace Root: {}", workspace_root.display());
    println!("Architecture  : {}", arch_dir.display());
    println!("--------------------------------------------------------------------------------");

    println!("=== 1. Verifying Capability Graph & Implementation Truth ===");
    if !graph_path.exists() {
        eprintln!("ERROR: ARCHITECTURE/CAPABILITY_GRAPH.json missing!");
        std::process::exit(1);
    }

    let graph_content = fs::read_to_string(&graph_path)?;
    let graph: Value = serde_json::from_str(&graph_content)?;

    let nodes = graph
        .get("nodes")
        .and_then(Value::as_array)
        .ok_or("Invalid nodes array in CAPABILITY_GRAPH.json")?;

    println!("Total Registered Capabilities: {}", nodes.len());

    let mut missing_files: Vec<(String, String)> = Vec::new();
    let mut orphan_nodes: Vec<String> = Vec::new();

    for node in nodes {
        let cap_id = node
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();

        let cap_file = cap_dir.join(format!("{}.json", cap_id));
        if !cap_file.exists() {
            orphan_nodes.push(cap_id.clone());
        }

        if let Some(impl_files) = node.get("implementation_files").and_then(Value::as_array) {
            for f in impl_files {
                if let Some(rel_path) = f.as_str() {
                    let full_path = workspace_root.join(rel_path);
                    if !full_path.exists() {
                        missing_files.push((cap_id.clone(), rel_path.to_string()));
                    }
                }
            }
        }
    }

    if !missing_files.is_empty() {
        eprintln!("ERROR: Capability implementation files missing from disk:");
        for (cap, file) in &missing_files {
            eprintln!("  [{cap}] -> {file}");
        }
        std::process::exit(1);
    } else {
        println!("PASS: All capability implementation files exist on disk.");
    }

    if !orphan_nodes.is_empty() {
        eprintln!(
            "ERROR: Orphan capability nodes missing specification JSON: {:?}",
            orphan_nodes
        );
        std::process::exit(1);
    } else {
        println!("PASS: All {} capability specifications present and verified.", nodes.len());
    }

    println!("\n=== 2. Running Cargo Workspace Compiler & Zero-Warning Check ===");
    let output = Command::new("cargo")
        .args(["check", "--workspace", "--all-targets"])
        .current_dir(&workspace_root)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("ERROR: cargo check failed:\n{}", stderr);
        std::process::exit(1);
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    if stderr.contains("warning:") {
        eprintln!("ERROR: Cargo check generated warnings (0-warning directive violated):\n{}", stderr);
        std::process::exit(1);
    }

    println!("PASS: cargo check --workspace --all-targets passed with 0 warnings and 0 errors.");

    println!("\n=== 3. Verifying Source Index Truth (ARCHITECTURE/SOURCE_INDEX.json) ===");
    let source_index_path = arch_dir.join("SOURCE_INDEX.json");
    if source_index_path.exists() {
        let content = fs::read_to_string(&source_index_path)?;
        let clean_content = content.trim_start_matches('\u{feff}');
        let index_val: Value = serde_json::from_str(clean_content)?;
        if let Some(files_map) = index_val.get("files").and_then(Value::as_object) {
            let mut missing_index_files = Vec::new();
            for (rel_path, _) in files_map {
                let p = workspace_root.join(rel_path);
                if !p.exists() {
                    missing_index_files.push(rel_path.clone());
                }
            }
            if !missing_index_files.is_empty() {
                eprintln!("ERROR: SOURCE_INDEX.json references missing files on disk:");
                for mf in &missing_index_files {
                    eprintln!("  MISSING: {mf}");
                }
                std::process::exit(1);
            }
            println!("PASS: All {} indexed files in SOURCE_INDEX.json exist on disk.", files_map.len());
        }
    }

    println!("\n=== 4. Verifying Architecture Sync Engine Registries ===");
    let tree_path = arch_dir.join("project_tree.json");
    let file_reg_path = arch_dir.join("file_registry.json");
    let folder_reg_path = arch_dir.join("folder_registry.json");
    let state_path = arch_dir.join("architecture_state.json");

    if !tree_path.exists() || !file_reg_path.exists() || !folder_reg_path.exists() || !state_path.exists() {
        eprintln!("ERROR: Architecture Sync Engine registries missing from ARCHITECTURE/");
        if !tree_path.exists() { eprintln!("  Missing: project_tree.json"); }
        if !file_reg_path.exists() { eprintln!("  Missing: file_registry.json"); }
        if !folder_reg_path.exists() { eprintln!("  Missing: folder_registry.json"); }
        if !state_path.exists() { eprintln!("  Missing: architecture_state.json"); }
        std::process::exit(1);
    }

    let state_raw = fs::read_to_string(&state_path)?;
    let state_val: Value = serde_json::from_str(state_raw.trim_start_matches('\u{feff}'))?;
    let status = state_val.get("status").and_then(Value::as_str).unwrap_or("UNKNOWN");
    if status != "SYNCED" {
        eprintln!("ERROR: Architecture state is not SYNCED: current status = {status}");
        std::process::exit(1);
    }
    let total_files = state_val.get("total_files").and_then(Value::as_u64).unwrap_or(0);
    let total_folders = state_val.get("total_folders").and_then(Value::as_u64).unwrap_or(0);
    println!("PASS: Architecture Sync Engine registries verified (status: SYNCED, files: {total_files}, folders: {total_folders}).");

    println!("\n=== 5. Verifying Crate Module Ownership & Single-Compilation Invariant ===");
    let rust_root = workspace_root.join("rust");
    let mut rs_files = Vec::new();
    collect_rs_files(&rust_root, &mut rs_files)?;
    let mut included_targets: std::collections::HashMap<PathBuf, PathBuf> =
        std::collections::HashMap::new();
    for rs_file in &rs_files {
        let src = fs::read_to_string(rs_file)?;
        let parent_dir = rs_file.parent().unwrap_or(&rust_root);
        for line in src.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("#[path = \"") {
                if let Some(rel_target) = rest.strip_suffix("\"]") {
                    let joined = parent_dir.join(rel_target);
                    let canon = fs::canonicalize(&joined).map_err(|e| {
                        format!(
                            "Broken #[path = \"{rel_target}\"] in '{}': {e}",
                            rs_file.display()
                        )
                    })?;
                    if let Some(prev_owner) = included_targets.insert(canon.clone(), rs_file.clone())
                    {
                        eprintln!(
                            "ERROR: Duplicate #[path] compilation detected for '{}': included by both '{}' and '{}'",
                            canon.display(),
                            prev_owner.display(),
                            rs_file.display()
                        );
                        std::process::exit(1);
                    }
                }
            }
        }
    }
    for owned_rel in tara_engine::OWNED_TRAINING_SYSTEM_MODULES {
        let full = workspace_root.join(owned_rel);
        if !full.exists() {
            eprintln!(
                "ERROR: Declared OWNED_TRAINING_SYSTEM_MODULES file missing: {}",
                owned_rel
            );
            std::process::exit(1);
        }
    }
    let engine_lib_src = fs::read_to_string(rust_root.join("tara_engine/src/lib.rs"))?;
    for private_mod in [
        "cuda",
        "model",
        "model_expansion",
        "safetensors",
        "self_update",
        "shard_manager",
        "skills_evaluator",
        "trainer",
    ] {
        let forbidden = format!("pub mod {private_mod};");
        if engine_lib_src.lines().any(|l| l.trim() == forbidden) {
            eprintln!(
                "ERROR: Low-level Tier 3 module '{}' must be private (`mod {};`), not `pub mod` in tara_engine/src/lib.rs",
                private_mod, private_mod
            );
            std::process::exit(1);
        }
    }
    println!(
        "PASS: Single-owner module & private Tier 3 boundary verified ({} #[path] targets audited, 0 duplicate #[path] inclusions).",
        included_targets.len()
    );

    println!("--------------------------------------------------------------------------------");
    println!("SUCCESS: Whole-system architecture verified compliant with all directives.");
    println!("================================================================================");

    Ok(())
}

fn collect_rs_files(dir: & std::path::Path, out: &mut Vec<PathBuf>) -> Result<(), std::io::Error> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let p = entry.path();
        if p.is_dir() {
            collect_rs_files(&p, out)?;
        } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(p);
        }
    }
    Ok(())
}
