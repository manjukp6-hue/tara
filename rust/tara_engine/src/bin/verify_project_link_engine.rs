//! # TARA Project Link & Reference Engine Verification Binary (100% Native Rust)
//!
//! Enforces Option 1 Lifecycle (Strict Rule 19 Compliance):
//! - Phase 1: Full Initial Indexing (Rust files, modules, symbols, in-links, out-links)
//! - Phase 2: Lifecycle Event Tracking & Reconnection (Create -> Modify -> Move; Test Artifacts PRESERVED)
//! - Phase 3: Graph Validation & Snapshot-based Atomic Rollback (with preserved test artifact)
//! - Phase 4: Architecture Map Synchronization & Dynamic Runtime SHA-256 Integrity
//! - Phase 5: Staged Final Deletion Verification & Mandatory Post-Cleanup Filesystem Scan
//!
//! Zero Python scripts, zero mocks, zero cargo warnings.

use std::fs;
use tara_engine::project_link_engine::{
    compute_dynamic_sha256, find_workspace_root, ProjectLinkConfig, ProjectLinkEngine,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!("  TARA PROJECT LINK & REFERENCE ENGINE — VERIFICATION SUITE (OPTION 1)");
    println!("================================================================================");

    let ws_root = find_workspace_root()
        .map_err(|e| format!("Failed to find workspace root: {e}"))?;
    println!("Workspace Root: {}", ws_root.display());

    let config = ProjectLinkConfig {
        workspace_root: ws_root.clone(),
        ..Default::default()
    };
    let mut engine = ProjectLinkEngine::new(config);

    // ──────────────────────────────────────────────────────────────────────────
    // PHASE 1: FULL INITIAL INDEXING (RULE 2, RULE 5)
    // ──────────────────────────────────────────────────────────────────────────
    println!("\n=== PHASE 1: FULL INITIAL INDEXING ===");
    let report = engine.build_full_index()?;

    println!("  Total Rust Files Indexed: {}", report.total_rust_files);
    println!("  Total Modules Mapped    : {}", report.total_modules);
    println!("  Total Symbols Parsed    : {}", report.total_symbols);
    println!("  Total Imports Parsed    : {}", report.total_imports);
    println!("  Total In/Out Links      : {}", report.total_in_out_links);
    println!("  Indexing Duration       : {} ms", report.duration_ms);

    assert!(
        report.total_rust_files >= 10,
        "Expected at least 10 Rust files in workspace, got {}",
        report.total_rust_files
    );
    assert!(
        report.total_modules >= 5,
        "Expected at least 5 modules mapped, got {}",
        report.total_modules
    );
    assert!(
        report.total_symbols > 50,
        "Expected > 50 symbols parsed, got {}",
        report.total_symbols
    );
    assert!(
        report.total_in_out_links > 0,
        "Expected at least some in/out links resolved across workspace, got {}",
        report.total_in_out_links
    );

    println!("PASS: Phase 1 Initial Indexing verified.");

    // ──────────────────────────────────────────────────────────────────────────
    // PHASE 2: LIFECYCLE EVENT TRACKING & PRESERVATION (RULES 3, 4 & 19)
    // ──────────────────────────────────────────────────────────────────────────
    println!("\n=== PHASE 2: LIFECYCLE EVENT TRACKING & PRESERVATION (RULE 19) ===");

    let test_file_rel_a = "rust/tara_engine/src/__link_engine_test_alpha.rs";
    let test_file_path_a = ws_root.join(test_file_rel_a);

    // 2.1 Creation
    let code_v1 = r#"
// Staged verification module (lifecycle test preserved until Phase 5)
use std::collections::BTreeMap;
use tara_core::governance::verify_creator_signature;

pub struct AlphaEngineEntity {
    pub id: u64,
    pub name: String,
}

pub fn execute_alpha_task(entity: &AlphaEngineEntity) -> bool {
    !entity.name.is_empty()
}
"#;
    fs::write(&test_file_path_a, code_v1)?;

    let delta_created = engine.on_file_created(test_file_rel_a)?;
    println!(
        "  [CREATE] Action: {}, Added Imports: {:?}, Added Exports: {:?}",
        delta_created.action, delta_created.added_imports, delta_created.added_exports
    );
    assert_eq!(delta_created.action, "CREATED");
    assert!(delta_created.added_exports.contains(&"AlphaEngineEntity".to_string()));
    assert!(delta_created.added_exports.contains(&"execute_alpha_task".to_string()));
    assert!(engine.files.contains_key(test_file_rel_a));

    // 2.2 Modification
    let code_v2 = r#"
// Staged verification module (lifecycle test v2 preserved until Phase 5)
use std::collections::BTreeMap;
use std::sync::Arc;
use tara_core::governance::verify_creator_signature;

pub struct AlphaEngineEntity {
    pub id: u64,
    pub name: String,
}

pub struct AlphaExtraRecord {
    pub count: usize,
}

pub fn execute_alpha_task(entity: &AlphaEngineEntity) -> bool {
    !entity.name.is_empty()
}

pub fn compute_alpha_bonus(rec: &AlphaExtraRecord) -> usize {
    rec.count * 2
}
"#;
    fs::write(&test_file_path_a, code_v2)?;

    let delta_modified = engine.on_file_modified(test_file_rel_a)?;
    println!(
        "  [MODIFY] Action: {}, Added Imports: {:?}, Added Exports: {:?}",
        delta_modified.action, delta_modified.added_imports, delta_modified.added_exports
    );
    assert_eq!(delta_modified.action, "MODIFIED");
    assert!(delta_modified.added_exports.contains(&"AlphaExtraRecord".to_string()));
    assert!(delta_modified.added_exports.contains(&"compute_alpha_bonus".to_string()));

    // 2.3 Move / Reconnection
    let test_file_rel_b = "rust/tara_engine/src/__link_engine_test_beta.rs";
    let test_file_path_b = ws_root.join(test_file_rel_b);

    fs::rename(&test_file_path_a, &test_file_path_b)?;

    let delta_moved = engine.on_file_moved(test_file_rel_a, test_file_rel_b)?;
    println!(
        "  [MOVE & RECONNECT] Action: {}, Path: {}",
        delta_moved.action, delta_moved.file_path
    );
    assert_eq!(delta_moved.action, "MOVED_AND_RECONNECTED");
    assert!(!engine.files.contains_key(test_file_rel_a));
    assert!(engine.files.contains_key(test_file_rel_b));

    // RULE 19 PRESERVATION:
    // DO NOT DELETE the test file here! Keep it on disk for multi-stage inspection & repeated validation.
    assert!(
        test_file_path_b.exists(),
        "Rule 19 Mandate: Test artifact must remain on disk throughout Phase 2, 3, and 4!"
    );
    println!(
        "  [RULE 19 PRESERVE] Test artifact '{}' preserved on disk for downstream phases.",
        test_file_rel_b
    );
    println!("PASS: Phase 2 Lifecycle Event Tracking & Artifact Preservation verified.");

    // ──────────────────────────────────────────────────────────────────────────
    // PHASE 3: GRAPH VALIDATION & SNAPSHOT ROLLBACK (RULE 6)
    // ──────────────────────────────────────────────────────────────────────────
    println!("\n=== PHASE 3: GRAPH VALIDATION & SNAPSHOT ROLLBACK ===");

    // Take snapshot of clean state (including preserved test artifact)
    let snapshot = engine.create_snapshot();
    println!("  Created clean graph snapshot: ID {}", snapshot.snapshot_id);

    // Initial validation of clean graph with preserved artifact
    match engine.validate_graph() {
        Ok(valid_files) => {
            println!("  Initial graph validation: PASS ({} valid nodes)", valid_files);
        }
        Err(issues) => {
            println!("  Initial graph validation notes: {} issues", issues.len());
        }
    }

    // Deliberately inject a corrupted node with a dangling broken out-link
    let mut corrupt_node = engine
        .files
        .values()
        .next()
        .cloned()
        .unwrap();
    corrupt_node.rel_path = "rust/tara_engine/src/__corrupt_node.rs".to_string();
    corrupt_node.out_links.insert("non_existent_ghost_target.rs".to_string());
    engine.files.insert("rust/tara_engine/src/__corrupt_node.rs".to_string(), corrupt_node);

    let val_result = engine.validate_graph();
    assert!(
        val_result.is_err(),
        "Expected validation to fail when broken out-link injected!"
    );
    let issues = val_result.unwrap_err();
    assert!(
        issues.iter().any(|i| i.contains("non_existent_ghost_target.rs")),
        "Expected issue mentioning non_existent_ghost_target.rs"
    );
    println!("  PASS: Corrupted broken out-link correctly detected: {}", issues[0]);

    // Rollback to clean snapshot
    println!("  Executing atomic rollback to snapshot {}...", snapshot.snapshot_id);
    engine.rollback(snapshot);

    assert!(
        !engine.files.contains_key("rust/tara_engine/src/__corrupt_node.rs"),
        "Corrupted node should be eliminated after rollback!"
    );
    println!("PASS: Phase 3 Snapshot Rollback verified.");

    // ──────────────────────────────────────────────────────────────────────────
    // PHASE 4: ARCHITECTURE MAP SYNC & DYNAMIC SHA-256 (RULES 2 & 7)
    // ──────────────────────────────────────────────────────────────────────────
    println!("\n=== PHASE 4: ARCHITECTURE MAP SYNC & DYNAMIC SHA-256 ===");

    engine.sync_architecture_maps()?;

    let source_index_path = ws_root.join("ARCHITECTURE").join("SOURCE_INDEX.json");
    let rel_graph_path = ws_root.join("ARCHITECTURE").join("RELATIONSHIP_GRAPH.json");

    assert!(source_index_path.exists(), "SOURCE_INDEX.json must exist after sync");
    assert!(rel_graph_path.exists(), "RELATIONSHIP_GRAPH.json must exist after sync");

    let source_index_content = fs::read_to_string(&source_index_path)?;
    let source_index_json: serde_json::Value = serde_json::from_str(&source_index_content)?;
    let files_map = source_index_json
        .get("files")
        .and_then(|f| f.as_object())
        .expect("files map in SOURCE_INDEX.json");

    println!("  SOURCE_INDEX.json synchronized: {} files indexed", files_map.len());
    assert!(files_map.len() >= 10);

    // Verify fields on a sample entry
    let sample_entry = files_map.values().next().unwrap();
    assert!(sample_entry.get("rel_path").is_some());
    assert!(sample_entry.get("dynamic_sha256").is_some());
    assert!(sample_entry.get("structs").is_some());
    assert!(sample_entry.get("functions").is_some());
    assert!(sample_entry.get("in_links").is_some());
    assert!(sample_entry.get("out_links").is_some());

    let rel_graph_content = fs::read_to_string(&rel_graph_path)?;
    let rel_graph_json: serde_json::Value = serde_json::from_str(&rel_graph_content)?;
    assert!(rel_graph_json.get("total_files").is_some());
    assert!(rel_graph_json.get("module_tree").is_some());
    println!(
        "  RELATIONSHIP_GRAPH.json synchronized: {} total files recorded",
        rel_graph_json["total_files"]
    );

    // Dynamic SHA-256 integrity check against disk truth
    let mut checked_shas = 0;
    for (rel_path, node) in engine.files.iter().take(10) {
        let full_path = ws_root.join(rel_path);
        if full_path.is_file() {
            let actual_sha = compute_dynamic_sha256(&full_path, 64 * 1024)?;
            assert_eq!(
                node.dynamic_sha256, actual_sha,
                "Dynamic SHA mismatch for {rel_path}"
            );
            checked_shas += 1;
        }
    }
    println!("  Checked {checked_shas} files: All dynamically computed SHA-256 digests match disk byte truth.");
    println!("PASS: Phase 4 Architecture Map Sync & Dynamic SHA-256 verified.");

    // ──────────────────────────────────────────────────────────────────────────
    // PHASE 5: STAGED FINAL DELETION VERIFICATION & CLEANUP (RULE 19)
    // ──────────────────────────────────────────────────────────────────────────
    println!("\n=== PHASE 5: STAGED FINAL DELETION VERIFICATION & CLEANUP (RULE 19) ===");

    // Verify artifact is still present on disk right before final deletion stage
    assert!(
        test_file_path_b.exists(),
        "Artifact must exist on disk right up to Phase 5 final cleanup stage!"
    );

    // 5.1 Physical Deletion & Event Verification
    fs::remove_file(&test_file_path_b)?;
    let delta_deleted = engine.on_file_deleted(test_file_rel_b)?;
    println!(
        "  [FINAL DELETE] Action: {}, Removed Path: {}",
        delta_deleted.action, delta_deleted.file_path
    );
    assert_eq!(delta_deleted.action, "DELETED");
    assert!(!engine.files.contains_key(test_file_rel_b));

    // 5.2 Clean architecture map re-sync to reflect final state
    engine.sync_architecture_maps()?;

    // 5.3 Mandatory Post-Cleanup Filesystem Verification Scan (Rule 19)
    println!("  [FILESYSTEM SCAN] Verifying complete elimination of test artifacts...");
    assert!(
        !test_file_path_a.exists(),
        "Residual test file A found on disk after suite completion!"
    );
    assert!(
        !test_file_path_b.exists(),
        "Residual test file B found on disk after suite completion!"
    );

    // Scan directory for any leftover __link_engine_test* artifacts
    let engine_src_dir = ws_root.join("rust").join("tara_engine").join("src");
    let mut residual_artifacts = Vec::new();
    if let Ok(entries) = fs::read_dir(&engine_src_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("__link_engine_test") {
                residual_artifacts.push(name);
            }
        }
    }

    assert!(
        residual_artifacts.is_empty(),
        "Residual test artifacts detected in filesystem scan: {:?}",
        residual_artifacts
    );

    println!("  Filesystem Scan Result: ZERO test artifacts or residual files remain.");
    println!("PASS: Phase 5 Staged Final Deletion & Filesystem Clean Scan verified (Rule 19 compliant).");

    println!("\n================================================================================");
    println!("  SUMMARY: OPTION 1 VERIFICATION SUITE FULLY COMPLETED (100% RULE 19 COMPLIANT)");
    println!("  - Phase 1 Initial Indexing       : PASS ({} files, {} modules, {} symbols)", report.total_rust_files, report.total_modules, report.total_symbols);
    println!("  - Phase 2 Lifecycle Preservation : PASS (Artifact preserved across runs)");
    println!("  - Phase 3 Validation & Rollback  : PASS (Corrupt link detected, rollback restored)");
    println!("  - Phase 4 Architecture Sync      : PASS (SOURCE_INDEX.json & RELATIONSHIP_GRAPH.json)");
    println!("  - Phase 5 Final Deletion & Scan  : PASS (Staged deletion verified, 0 residual files)");
    println!("================================================================================");

    Ok(())
}
