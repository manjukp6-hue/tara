//! Subsystem Self-Upgrade Engine for TARA.
//!
//! Enables TARA to inspect, benchmark, and safely stage subsystem upgrades
//! with automatic rollback preservation.
//!
//! `inspect_subsystem` performs real, filesystem-measured health checks:
//!   - Verifies the subsystem source directory exists and is non-empty.
//!   - Measures total source bytes (code volume as a proxy for integrity).
//!   - Reads the upgrade history log to report previous upgrade events.
//!   - Runs `cargo check -p tara_server` on the live workspace to confirm the
//!     subsystem compiles without errors. Status is HEALTHY only when all checks pass.

use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub const ALLOWED_SUBSYSTEMS: &[&str] = &[
    "MODEL",
    "TOKENIZER",
    "SKILLS",
    "KNOWLEDGE",
    "TOOLS",
    "MEMORY",
    "LEARNING",
    "RETRIEVAL",
    "INFERENCE",
    "STORAGE",
    "PERFORMANCE",
];

pub struct SelfUpgradeEngine {
    pub repo_root: PathBuf,
    pub rollback_dir: PathBuf,
    pub history_file: PathBuf,
}

impl SelfUpgradeEngine {
    pub fn new<P: AsRef<Path>>(repo_root: P) -> Self {
        let root = repo_root.as_ref().to_path_buf();
        let rollback = root.join("storage").join("rollback");
        let history = root.join("storage").join("upgrade_history.json");
        let _ = fs::create_dir_all(&rollback);

        Self {
            repo_root: root,
            rollback_dir: rollback,
            history_file: history,
        }
    }

    /// Inspect a subsystem with real filesystem measurements and live `cargo check`.
    ///
    /// Returns HEALTHY only when:
    ///   - The subsystem source directory exists and contains at least one `.rs` file.
    ///   - `cargo check -p tara_server` exits 0 (the workspace compiles).
    ///
    /// Returns DEGRADED with a reason for any check that fails.
    pub fn inspect_subsystem(&self, subsystem: &str) -> Result<Value, String> {
        let sub = subsystem.to_uppercase();
        if !ALLOWED_SUBSYSTEMS.contains(&sub.as_str()) {
            return Err(format!(
                "Unknown subsystem '{}'. Allowed: {:?}",
                subsystem, ALLOWED_SUBSYSTEMS
            ));
        }

        let source_dir = self.subsystem_source_dir(&sub);
        let inspected_at = crate::now_iso();

        // Check 1: Directory existence.
        if !source_dir.exists() || !source_dir.is_dir() {
            return Ok(json!({
                "subsystem": sub,
                "path": source_dir.display().to_string(),
                "exists": false,
                "rs_file_count": 0,
                "total_source_bytes": 0,
                "compile_ok": false,
                "compile_error": "Source directory not found",
                "status": "DEGRADED",
                "reason": format!("Source directory '{}' does not exist", source_dir.display()),
                "inspected_at": inspected_at,
            }));
        }

        // Check 2: Count .rs files and measure total source bytes.
        let (rs_count, total_bytes) = Self::measure_source_dir(&source_dir);
        if rs_count == 0 {
            return Ok(json!({
                "subsystem": sub,
                "path": source_dir.display().to_string(),
                "exists": true,
                "rs_file_count": 0,
                "total_source_bytes": 0,
                "compile_ok": false,
                "compile_error": "No Rust source files found",
                "status": "DEGRADED",
                "reason": "Subsystem source directory exists but contains no .rs files",
                "inspected_at": inspected_at,
            }));
        }

        // Check 3: cargo check — real compilation gate.
        let (compile_ok, compile_error) = self.run_cargo_check();

        // Collect previous upgrade history entries for this subsystem.
        let history_entries = self.load_history_for(&sub);

        let status = if compile_ok { "HEALTHY" } else { "DEGRADED" };
        let reason = if compile_ok {
            format!(
                "{} .rs files ({} bytes), workspace compiles",
                rs_count, total_bytes
            )
        } else {
            format!(
                "cargo check failed: {}",
                compile_error.as_deref().unwrap_or("unknown error")
            )
        };

        Ok(json!({
            "subsystem": sub,
            "path": source_dir.display().to_string(),
            "exists": true,
            "rs_file_count": rs_count,
            "total_source_bytes": total_bytes,
            "compile_ok": compile_ok,
            "compile_error": compile_error,
            "previous_upgrades": history_entries.len(),
            "upgrade_history": history_entries,
            "status": status,
            "reason": reason,
            "inspected_at": inspected_at,
        }))
    }

    pub fn stage_upgrade(
        &self,
        subsystem: &str,
        upgrade_id: &str,
        affected_files: &[PathBuf],
        is_creator: bool,
    ) -> Result<Value, String> {
        if !is_creator {
            return Err(
                "Unauthorized: Subsystem upgrade requires ROOT_OPERATOR authorization".into(),
            );
        }

        // Validate subsystem name.
        let sub = subsystem.to_uppercase();
        if !ALLOWED_SUBSYSTEMS.contains(&sub.as_str()) {
            return Err(format!("Unknown subsystem '{}'", subsystem));
        }

        // Create rollback snapshot of all affected files before any change.
        let snap_dir = self.rollback_dir.join(upgrade_id);
        fs::create_dir_all(&snap_dir).map_err(|e| e.to_string())?;

        let mut snapshotted = Vec::new();
        for f in affected_files {
            if f.exists() && f.is_file() {
                if let Ok(rel) = f.strip_prefix(&self.repo_root) {
                    let dest = snap_dir.join(rel);
                    if let Some(parent) = dest.parent() {
                        let _ = fs::create_dir_all(parent);
                    }
                    fs::copy(f, &dest)
                        .map_err(|e| format!("Snapshot copy failed for '{}': {e}", f.display()))?;
                    snapshotted.push(rel.to_string_lossy().replace('\\', "/"));
                }
            }
        }

        let staged_at = crate::now_iso();
        let record = json!({
            "upgrade_id": upgrade_id,
            "subsystem": sub,
            "staged_at": staged_at,
            "rollback_path": snap_dir.display().to_string(),
            "snapshotted_files": snapshotted,
            "status": "STAGED",
        });

        // Append to upgrade history log.
        self.append_history(&record);

        Ok(record)
    }

    /// Rollback a staged upgrade by restoring files from the rollback snapshot.
    pub fn rollback_upgrade(&self, upgrade_id: &str, is_creator: bool) -> Result<Value, String> {
        if !is_creator {
            return Err("Unauthorized: Subsystem rollback requires ROOT_OPERATOR authorization".into());
        }

        let snap_dir = self.rollback_dir.join(upgrade_id);
        if !snap_dir.exists() || !snap_dir.is_dir() {
            return Err(format!("Rollback snapshot '{}' not found in {}", upgrade_id, self.rollback_dir.display()));
        }

        let mut restored = Vec::new();
        fn restore_recursive(src: &Path, base_src: &Path, target_root: &Path, restored: &mut Vec<String>) -> Result<(), String> {
            let entries = fs::read_dir(src).map_err(|e| e.to_string())?;
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    restore_recursive(&p, base_src, target_root, restored)?;
                } else if p.is_file() {
                    if let Ok(rel) = p.strip_prefix(base_src) {
                        let dest = target_root.join(rel);
                        if let Some(parent) = dest.parent() {
                            let _ = fs::create_dir_all(parent);
                        }
                        fs::copy(&p, &dest).map_err(|e| format!("Restore copy failed for '{}': {e}", dest.display()))?;
                        restored.push(rel.to_string_lossy().replace('\\', "/"));
                    }
                }
            }
            Ok(())
        }

        restore_recursive(&snap_dir, &snap_dir, &self.repo_root, &mut restored)?;

        let rolled_back_at = crate::now_iso();
        let record = json!({
            "upgrade_id": upgrade_id,
            "rolled_back_at": rolled_back_at,
            "restored_files": restored,
            "status": "ROLLED_BACK",
        });

        self.append_history(&record);
        Ok(record)
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    fn subsystem_source_dir(&self, sub: &str) -> PathBuf {
        match sub {
            "MODEL" => self
                .repo_root
                .join("rust")
                .join("tara_engine")
                .join("src")
                .join("model"),
            "TOKENIZER" => self.repo_root.join("rust").join("tara_engine").join("src"),
            "SKILLS" => self
                .repo_root
                .join("rust")
                .join("tara_server")
                .join("src")
                .join("skills"),
            "KNOWLEDGE" => self
                .repo_root
                .join("rust")
                .join("tara_server")
                .join("src")
                .join("knowledge"),
            "MEMORY" => self
                .repo_root
                .join("rust")
                .join("tara_server")
                .join("src")
                .join("memory"),
            "LEARNING" => self
                .repo_root
                .join("rust")
                .join("tara_server")
                .join("src")
                .join("learning"),
            "TOOLS" => self
                .repo_root
                .join("rust")
                .join("tara_server")
                .join("src")
                .join("tools_registry.rs")
                .parent()
                .unwrap_or(&self.repo_root)
                .to_path_buf(),
            "STORAGE" => self.repo_root.join("storage"),
            _ => self.repo_root.join("rust"),
        }
    }

    /// Count `.rs` files and total bytes under a directory (recursive).
    fn measure_source_dir(dir: &Path) -> (usize, u64) {
        fn walk(dir: &Path, count: &mut usize, bytes: &mut u64) {
            let Ok(entries) = fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, count, bytes);
                } else if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("rs")
                {
                    *count += 1;
                    *bytes += entry.metadata().map(|m| m.len()).unwrap_or(0);
                }
            }
        }
        let mut count = 0usize;
        let mut bytes = 0u64;
        walk(dir, &mut count, &mut bytes);
        (count, bytes)
    }

    /// Run `cargo check -p tara_server` from the workspace root.
    /// Returns (success: bool, stderr: Option<String>).
    fn run_cargo_check(&self) -> (bool, Option<String>) {
        match Command::new("cargo")
            .args(["check", "-p", "tara_server", "--message-format", "short"])
            .current_dir(&self.repo_root)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()
        {
            Ok(out) => {
                if out.status.success() {
                    (true, None)
                } else {
                    let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
                    (false, Some(stderr))
                }
            }
            Err(e) => (false, Some(format!("Failed to launch cargo: {e}"))),
        }
    }

    fn load_history_for(&self, subsystem: &str) -> Vec<Value> {
        if !self.history_file.exists() {
            return Vec::new();
        }
        let Ok(text) = fs::read_to_string(&self.history_file) else {
            return Vec::new();
        };
        let Ok(arr) = serde_json::from_str::<Vec<Value>>(&text) else {
            return Vec::new();
        };
        arr.into_iter()
            .filter(|v| v.get("subsystem").and_then(Value::as_str) == Some(subsystem))
            .collect()
    }

    fn append_history(&self, record: &Value) {
        let mut entries: Vec<Value> = if self.history_file.exists() {
            fs::read_to_string(&self.history_file)
                .ok()
                .and_then(|t| serde_json::from_str(&t).ok())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        entries.push(record.clone());
        if let Ok(serialized) = serde_json::to_string_pretty(&entries) {
            let _ = fs::write(&self.history_file, serialized);
        }
    }
}
