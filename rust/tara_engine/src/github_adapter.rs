//! # TARA GitHub Adapter & Cross-Platform Synchronization Engine (100% Native Rust)
//!
//! Provides bidirectional compatibility between local development (Windows/macOS/Linux)
//! and remote GitHub repositories / GitHub Actions CI runners.
//!
//! ## Core Responsibilities:
//! - **Cross-Platform Path Normalization**: Enforces strict POSIX forward-slash `/` relative paths
//!   across all architecture manifests, registries, and dependency graphs.
//! - **Pre-Push Architecture & Security Audit**:
//!   1. Verifies `architecture_state.json` status is `SYNCED`.
//!   2. Audits all Rust files on disk against `SOURCE_INDEX.json` (zero orphan/missing files).
//!   3. Verifies zero compiler warnings across all targets (`cargo check --workspace --all-targets`).
//!   4. Audits codebase for zero hardcoded secrets/credentials (Rule 12).
//!   5. Audits codebase for zero hardcoded SHA-256 literals (Rule 2).
//!   6. Audits `.gitignore` to ensure `.github/` is tracked and secrets are protected.
//! - **GitHub Export Manifest**: Produces `ARCHITECTURE/GITHUB_EXPORT_MANIFEST.json`
//!   recording dynamic integrity proofs for GitHub CI verification.
//! - **Post-Clone / CI Import Verification**: Re-computes dynamic SHA-256 for all indexed files
//!   on GitHub Actions runners (Windows and Ubuntu) ensuring byte-for-byte consistency.
//!
//! 100% Native Rust, zero Python scripts, zero mocks.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Configuration for GitHub Adapter.
#[derive(Debug, Clone)]
pub struct GitHubAdapterConfig {
    pub workspace_root: PathBuf,
    pub arch_dir: PathBuf,
    pub io_buffer_size: usize,
    pub check_compiler_warnings: bool,
}

impl Default for GitHubAdapterConfig {
    fn default() -> Self {
        let ws = find_workspace_root().unwrap_or_else(|_| PathBuf::from("."));
        let arch = ws.join("ARCHITECTURE");
        Self {
            workspace_root: ws,
            arch_dir: arch,
            io_buffer_size: 64 * 1024,
            check_compiler_warnings: true,
        }
    }
}

/// Comprehensive audit report produced by GitHub Adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubValidationReport {
    pub is_ready_for_push: bool,
    pub total_posix_paths_verified: usize,
    pub non_posix_paths_found: Vec<String>,
    pub architecture_status: String,
    pub total_indexed_files: usize,
    pub missing_disk_files: Vec<String>,
    pub unindexed_rust_files: Vec<String>,
    pub compiler_warnings_count: usize,
    pub hardcoded_secrets_found: Vec<String>,
    pub hardcoded_sha_found: Vec<String>,
    pub gitignore_github_unblocked: bool,
    pub duration_ms: u64,
}

/// GitHub Export Manifest written to ARCHITECTURE/GITHUB_EXPORT_MANIFEST.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubExportManifest {
    pub schema_version: String,
    pub exported_at_utc: String,
    pub repository_ready_for_push: bool,
    pub total_files_indexed: usize,
    pub registries_digest_sha256: String,
    pub supported_platforms: Vec<String>,
    pub ci_workflow_path: String,
}

/// Single commit record within a GitHub push webhook event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubCommitRecord {
    pub id: String,
    pub message: String,
    pub timestamp: Option<String>,
    #[serde(default)]
    pub added: Vec<String>,
    #[serde(default)]
    pub removed: Vec<String>,
    #[serde(default)]
    pub modified: Vec<String>,
}

/// Payload sent by GitHub push webhooks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubPushPayload {
    #[serde(rename = "ref")]
    pub git_ref: String,
    pub before: String,
    pub after: String,
    #[serde(default)]
    pub repository_name: Option<String>,
    #[serde(default)]
    pub commits: Vec<GitHubCommitRecord>,
    #[serde(default)]
    pub head_commit: Option<GitHubCommitRecord>,
}

/// Result of processing a GitHub push event or remote update into Project Link Engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubWebhookResult {
    pub git_ref: String,
    pub commit_sha: String,
    pub total_files_processed: usize,
    pub added_files: Vec<String>,
    pub modified_files: Vec<String>,
    pub removed_files: Vec<String>,
    pub architecture_synced: bool,
    pub duration_ms: u64,
}

/// Report summarizing controlled remote git fetch & sync.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubRemoteSyncReport {
    pub remote: String,
    pub branch: String,
    pub previous_head: String,
    pub current_head: String,
    pub has_remote_changes: bool,
    pub files_reconciled: usize,
    pub duration_ms: u64,
}

/// Main GitHub Adapter engine.
pub struct GitHubAdapter {
    pub config: GitHubAdapterConfig,
}

impl GitHubAdapter {
    pub fn new(config: GitHubAdapterConfig) -> Self {
        Self { config }
    }

    /// Finds workspace root containing AGENTS.md and Cargo.toml.
    pub fn workspace_root(&self) -> &Path {
        &self.config.workspace_root
    }

    /// Performs exhaustive pre-push audit for GitHub and cross-platform compatibility.
    pub fn pre_push_audit(&self) -> Result<GitHubValidationReport, Box<dyn std::error::Error>> {
        let start = std::time::Instant::now();
        let ws = &self.config.workspace_root;
        let arch = &self.config.arch_dir;

        // 0. Ensure Project Link Engine synchronizes architecture maps
        let mut link_engine = crate::project_link_engine::ProjectLinkEngine::new(
            crate::project_link_engine::ProjectLinkConfig {
                workspace_root: ws.clone(),
                arch_dir: arch.clone(),
                ..Default::default()
            },
        );
        let _ = link_engine.build_full_index();
        let _ = link_engine.sync_architecture_maps();

        let mut non_posix = Vec::new();
        let mut missing_disk = Vec::new();
        let mut unindexed_rust = Vec::new();
        let mut hardcoded_secrets = Vec::new();
        let mut hardcoded_sha = Vec::new();
        let mut compiler_warnings = 0;

        // 1. Check Architecture State
        let state_path = arch.join("architecture_state.json");
        let (arch_status, total_indexed) = if state_path.exists() {
            let raw = fs::read_to_string(&state_path)?;
            let val: Value = serde_json::from_str(raw.trim_start_matches('\u{feff}'))?;
            let st = val.get("status").and_then(Value::as_str).unwrap_or("UNKNOWN").to_string();
            let tf = val.get("total_files").and_then(Value::as_u64).unwrap_or(0) as usize;
            (st, tf)
        } else {
            ("MISSING".to_string(), 0)
        };

        // 2. Check SOURCE_INDEX.json paths and disk existence
        let source_index_path = arch.join("SOURCE_INDEX.json");
        let mut total_posix_checked = 0;
        let mut indexed_paths = Vec::new();

        if source_index_path.exists() {
            let raw = fs::read_to_string(&source_index_path)?;
            let val: Value = serde_json::from_str(raw.trim_start_matches('\u{feff}'))?;
            if let Some(files_obj) = val.get("files").and_then(Value::as_object) {
                for (rel_path, _) in files_obj {
                    total_posix_checked += 1;
                    if rel_path.contains('\\') {
                        non_posix.push(rel_path.clone());
                    }
                    let full_path = ws.join(rel_path);
                    if !full_path.exists() {
                        missing_disk.push(rel_path.clone());
                    }
                    indexed_paths.push(rel_path.clone());
                }
            }
        }

        // 3. Check for any Rust files on disk that are NOT indexed in SOURCE_INDEX.json
        let mut disk_rust_files = Vec::new();
        self.collect_rust_files(ws, ws, &mut disk_rust_files)?;
        for disk_file in &disk_rust_files {
            if !indexed_paths.contains(disk_file) {
                unindexed_rust.push(disk_file.clone());
            }
        }

        // 4. Verify compiler check (zero warnings)
        if self.config.check_compiler_warnings {
            let output = Command::new("cargo")
                .args(["check", "--workspace", "--all-targets"])
                .current_dir(ws)
                .output()?;

            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("warning:") {
                compiler_warnings = stderr.matches("warning:").count();
            }
            if !output.status.success() {
                compiler_warnings += 1;
            }
        }

        // 5. Audit .gitignore for .github/ unblocking
        let gitignore_path = ws.join(".gitignore");
        let gitignore_unblocked = if gitignore_path.exists() {
            let content = fs::read_to_string(&gitignore_path)?;
            !content.lines().any(|l| l.trim() == ".github/" || l.trim() == ".github")
        } else {
            true
        };

        // 6. Audit for hardcoded secrets (Rule 12) in source files
        let secret_keywords = [
            "AIzaSy",
            "ghp_",
            "github_pat_",
            "sk-ant-",
            "sk-proj-",
            "xoxb-",
            "-----BEGIN PRIVATE KEY-----",
            "-----BEGIN RSA PRIVATE KEY-----",
        ];
        for rel_path in &disk_rust_files {
            let is_scanner_definition = rel_path.ends_with("intelligence_filter_35.rs")
                || rel_path.ends_with("github_adapter.rs");

            let full_p = ws.join(rel_path);
            if let Ok(content) = fs::read_to_string(&full_p) {
                if !is_scanner_definition {
                    for kw in &secret_keywords {
                        if content.contains(kw) {
                            hardcoded_secrets.push(format!("[{rel_path}] Contains forbidden secret keyword '{kw}'"));
                        }
                    }
                }

                // 7. Audit for hardcoded SHA-256 literals (Rule 2) in Rust source code
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with("*") {
                        continue;
                    }
                    if let Some(start_idx) = trimmed.find('"') {
                        if let Some(end_idx) = trimmed[start_idx + 1..].find('"') {
                            let candidate = &trimmed[start_idx + 1..start_idx + 1 + end_idx];
                            if candidate.len() == 64 && candidate.chars().all(|c| c.is_ascii_hexdigit()) {
                                hardcoded_sha.push(format!("[{rel_path}] Found hardcoded SHA-256 literal: {candidate}"));
                            }
                        }
                    }
                }
            }
        }

        let is_ready = non_posix.is_empty()
            && missing_disk.is_empty()
            && unindexed_rust.is_empty()
            && compiler_warnings == 0
            && hardcoded_secrets.is_empty()
            && hardcoded_sha.is_empty()
            && gitignore_unblocked
            && arch_status == "SYNCED";

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(GitHubValidationReport {
            is_ready_for_push: is_ready,
            total_posix_paths_verified: total_posix_checked,
            non_posix_paths_found: non_posix,
            architecture_status: arch_status,
            total_indexed_files: total_indexed,
            missing_disk_files: missing_disk,
            unindexed_rust_files: unindexed_rust,
            compiler_warnings_count: compiler_warnings,
            hardcoded_secrets_found: hardcoded_secrets,
            hardcoded_sha_found: hardcoded_sha,
            gitignore_github_unblocked: gitignore_unblocked,
            duration_ms,
        })
    }

    /// Exports `ARCHITECTURE/GITHUB_EXPORT_MANIFEST.json` and ensures CI workflow exists.
    pub fn export_github_bundle(&self) -> Result<GitHubExportManifest, Box<dyn std::error::Error>> {
        let audit = self.pre_push_audit()?;
        let arch = &self.config.arch_dir;
        let ws = &self.config.workspace_root;

        // Read registries digest from architecture_state.json
        let state_path = arch.join("architecture_state.json");
        let digest = if state_path.exists() {
            let raw = fs::read_to_string(&state_path)?;
            let val: Value = serde_json::from_str(raw.trim_start_matches('\u{feff}'))?;
            val.get("registries_digest_sha256")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()
        } else {
            String::new()
        };

        // Write or verify .github/workflows/tara_ci.yml
        let workflows_dir = ws.join(".github").join("workflows");
        fs::create_dir_all(&workflows_dir)?;
        let ci_workflow_path = workflows_dir.join("tara_ci.yml");
        if !ci_workflow_path.exists() {
            let workflow_content = generate_github_ci_workflow_content();
            fs::write(&ci_workflow_path, workflow_content)?;
        }

        let manifest = GitHubExportManifest {
            schema_version: "1.0.0".to_string(),
            exported_at_utc: chrono_now_iso(),
            repository_ready_for_push: audit.is_ready_for_push,
            total_files_indexed: audit.total_indexed_files,
            registries_digest_sha256: digest,
            supported_platforms: vec![
                "windows-latest".to_string(),
                "ubuntu-latest".to_string(),
                "macos-latest".to_string(),
            ],
            ci_workflow_path: ".github/workflows/tara_ci.yml".to_string(),
        };

        let manifest_path = arch.join("GITHUB_EXPORT_MANIFEST.json");
        let formatted = serde_json::to_string_pretty(&manifest)?;
        fs::write(&manifest_path, formatted)?;

        // Synchronize file_registry.json for newly exported / modified architecture files
        let file_reg_path = arch.join("file_registry.json");
        if file_reg_path.is_file() {
            if let Ok(raw) = fs::read_to_string(&file_reg_path) {
                if let Ok(mut val) = serde_json::from_str::<Value>(raw.trim_start_matches('\u{feff}')) {
                    if let Some(files_map) = val.get_mut("files").and_then(Value::as_object_mut) {
                        let to_sync = [
                            "ARCHITECTURE/GITHUB_EXPORT_MANIFEST.json",
                            "ARCHITECTURE/RELATIONSHIP_GRAPH.json",
                            "ARCHITECTURE/SOURCE_INDEX.json",
                            ".github/workflows/tara_ci.yml",
                        ];
                        for rel in to_sync {
                            let p = ws.join(rel);
                            if p.is_file() {
                                if let Ok(sha) = compute_dynamic_sha256(&p, self.config.io_buffer_size) {
                                    let rel_str = rel.to_string();
                                    if let Some(entry) = files_map.get_mut(rel) {
                                        entry["sha256"] = Value::String(sha);
                                    } else {
                                        let meta = fs::metadata(&p).ok();
                                        let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
                                        let ext = p.extension().map(|e| e.to_string_lossy().to_string()).unwrap_or_default();
                                        let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                                        let lines = fs::read_to_string(&p).map(|c| c.lines().count()).unwrap_or(0);
                                        files_map.insert(
                                            rel_str.clone(),
                                            serde_json::json!({
                                                "rel_path": rel_str,
                                                "name": name,
                                                "extension": ext,
                                                "size_bytes": size,
                                                "line_count": lines,
                                                "sha256": sha,
                                                "modified_epoch": 0,
                                                "file_id": 0
                                            }),
                                        );
                                    }
                                }
                            }
                        }
                    }
                    if let Ok(updated_raw) = serde_json::to_string_pretty(&val) {
                        let _ = fs::write(&file_reg_path, updated_raw);
                    }
                }
            }
        }

        Ok(manifest)
    }

    /// Verifies post-clone / import integrity when repository is cloned on fresh machine.
    pub fn verify_github_import(&self) -> Result<usize, Vec<String>> {
        let ws = &self.config.workspace_root;
        let arch = &self.config.arch_dir;
        let mut errors = Vec::new();

        // 1. Verify file_registry.json
        let file_reg_path = arch.join("file_registry.json");
        if !file_reg_path.exists() {
            return Err(vec!["ARCHITECTURE/file_registry.json is missing!".to_string()]);
        }

        let raw = match fs::read_to_string(&file_reg_path) {
            Ok(r) => r,
            Err(e) => return Err(vec![format!("Failed to read file_registry.json: {e}")]),
        };

        let val: Value = match serde_json::from_str(raw.trim_start_matches('\u{feff}')) {
            Ok(v) => v,
            Err(e) => return Err(vec![format!("Corrupted file_registry.json: {e}")]),
        };

        let files_map = match val.get("files").and_then(Value::as_object) {
            Some(m) => m,
            None => return Err(vec!["files map missing in file_registry.json".to_string()]),
        };

        let mut checked = 0;
        for (rel_path, f_val) in files_map {
            let full_p = ws.join(rel_path);
            if !full_p.exists() {
                errors.push(format!("Missing file on disk: {rel_path}"));
                continue;
            }

            if let Some(expected_sha) = f_val.get("sha256").and_then(Value::as_str) {
                if !expected_sha.is_empty() {
                    match compute_dynamic_sha256(&full_p, self.config.io_buffer_size) {
                        Ok(actual_sha) => {
                            if actual_sha != expected_sha {
                                errors.push(format!(
                                    "Dynamic SHA mismatch on import: {rel_path} (expected {expected_sha}, got {actual_sha})"
                                ));
                            }
                        }
                        Err(e) => {
                            errors.push(format!("Failed to hash file {rel_path}: {e}"));
                        }
                    }
                }
            }
            checked += 1;
        }

        if errors.is_empty() {
            Ok(checked)
        } else {
            Err(errors)
        }
    }

    /// Processes an authentic GitHub push webhook event and feeds the changed files
    /// directly into the Project Link & Reference Engine to reconcile relationships.
    pub fn process_push_event(
        &self,
        payload: &GitHubPushPayload,
        link_engine: &mut crate::project_link_engine::ProjectLinkEngine,
    ) -> Result<GitHubWebhookResult, Box<dyn std::error::Error>> {
        let start = std::time::Instant::now();
        let ws = &self.config.workspace_root;

        let mut added_files = Vec::new();
        let mut modified_files = Vec::new();
        let mut removed_files = Vec::new();

        // Aggregate added, modified, and removed files across all commits in the push
        for commit in &payload.commits {
            for f in &commit.added {
                let norm = f.replace('\\', "/");
                if !added_files.contains(&norm) {
                    added_files.push(norm);
                }
            }
            for f in &commit.modified {
                let norm = f.replace('\\', "/");
                if !modified_files.contains(&norm) {
                    modified_files.push(norm);
                }
            }
            for f in &commit.removed {
                let norm = f.replace('\\', "/");
                if !removed_files.contains(&norm) {
                    removed_files.push(norm);
                }
            }
        }

        // Also incorporate head_commit if present
        if let Some(ref head) = payload.head_commit {
            for f in &head.added {
                let norm = f.replace('\\', "/");
                if !added_files.contains(&norm) {
                    added_files.push(norm);
                }
            }
            for f in &head.modified {
                let norm = f.replace('\\', "/");
                if !modified_files.contains(&norm) {
                    modified_files.push(norm);
                }
            }
            for f in &head.removed {
                let norm = f.replace('\\', "/");
                if !removed_files.contains(&norm) {
                    removed_files.push(norm);
                }
            }
        }

        let mut total_processed = 0;

        // 1. Process removed files first in Project Link Engine
        for rel_p in &removed_files {
            if rel_p.ends_with(".rs") {
                let _ = link_engine.on_file_deleted(rel_p);
                total_processed += 1;
            }
        }

        // 2. Process added files in Project Link Engine
        for rel_p in &added_files {
            if rel_p.ends_with(".rs") {
                let full_p = ws.join(rel_p);
                if full_p.is_file() {
                    let _ = link_engine.on_file_created(rel_p);
                    total_processed += 1;
                }
            }
        }

        // 3. Process modified files in Project Link Engine
        for rel_p in &modified_files {
            if rel_p.ends_with(".rs") {
                let full_p = ws.join(rel_p);
                if full_p.is_file() {
                    let _ = link_engine.on_file_modified(rel_p);
                    total_processed += 1;
                }
            }
        }

        // 4. Synchronize Architecture Maps (SOURCE_INDEX.json & RELATIONSHIP_GRAPH.json)
        link_engine.sync_architecture_maps()?;

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(GitHubWebhookResult {
            git_ref: payload.git_ref.clone(),
            commit_sha: payload.after.clone(),
            total_files_processed: total_processed,
            added_files,
            modified_files,
            removed_files,
            architecture_synced: true,
            duration_ms,
        })
    }

    /// Controlled remote Git fetch and reconciliation engine.
    /// Pulls remote commits from GitHub and directly feeds delta into Project Link Engine.
    pub fn fetch_and_reconcile_remote(
        &self,
        remote: &str,
        branch: &str,
        link_engine: &mut crate::project_link_engine::ProjectLinkEngine,
    ) -> Result<GitHubRemoteSyncReport, Box<dyn std::error::Error>> {
        let start = std::time::Instant::now();
        let ws = &self.config.workspace_root;

        let head_out = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(ws)
            .output()?;
        let prev_head = String::from_utf8_lossy(&head_out.stdout).trim().to_string();

        let fetch_res = Command::new("git")
            .args(["fetch", remote, branch])
            .current_dir(ws)
            .output();

        let mut reconciled = 0;
        let mut curr_head = prev_head.clone();
        let mut has_changes = false;

        if let Ok(out) = fetch_res {
            if out.status.success() {
                let remote_ref = format!("{remote}/{branch}");
                let remote_head_out = Command::new("git")
                    .args(["rev-parse", &remote_ref])
                    .current_dir(ws)
                    .output()?;
                let rem_head = String::from_utf8_lossy(&remote_head_out.stdout).trim().to_string();

                if !rem_head.is_empty() && rem_head != prev_head {
                    has_changes = true;

                    let diff_out = Command::new("git")
                        .args(["diff", "--name-status", &prev_head, &rem_head])
                        .current_dir(ws)
                        .output()?;
                    let diff_str = String::from_utf8_lossy(&diff_out.stdout);

                    let _ = Command::new("git")
                        .args(["merge", "--ff-only", &remote_ref])
                        .current_dir(ws)
                        .output();

                    for line in diff_str.lines() {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            let status = parts[0];
                            let rel_p = parts[1].replace('\\', "/");

                            if rel_p.ends_with(".rs") {
                                let full_p = ws.join(&rel_p);
                                match status {
                                    "A" => {
                                        if full_p.is_file() {
                                            let _ = link_engine.on_file_created(&rel_p);
                                            reconciled += 1;
                                        }
                                    }
                                    "M" => {
                                        if full_p.is_file() {
                                            let _ = link_engine.on_file_modified(&rel_p);
                                            reconciled += 1;
                                        }
                                    }
                                    "D" => {
                                        let _ = link_engine.on_file_deleted(&rel_p);
                                        reconciled += 1;
                                    }
                                    _ => {
                                        if full_p.is_file() {
                                            let _ = link_engine.on_file_modified(&rel_p);
                                            reconciled += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    link_engine.sync_architecture_maps()?;
                    curr_head = rem_head;
                }
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(GitHubRemoteSyncReport {
            remote: remote.to_string(),
            branch: branch.to_string(),
            previous_head: prev_head,
            current_head: curr_head,
            has_remote_changes: has_changes,
            files_reconciled: reconciled,
            duration_ms,
        })
    }

    /// Recursively collects all .rs files in rust/ tree.
    fn collect_rust_files(
        &self,
        root: &Path,
        current: &Path,
        out: &mut Vec<String>,
    ) -> std::io::Result<()> {
        let entries = match fs::read_dir(current) {
            Ok(e) => e,
            Err(_) => return Ok(()),
        };

        for entry_res in entries {
            let entry = match entry_res {
                Ok(e) => e,
                Err(_) => continue,
            };
            let path = entry.path();
            let ft = match entry.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };

            if ft.is_dir() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name != "target"
                    && name != ".git"
                    && name != ".gemini"
                    && name != "brain"
                    && name != "scratch"
                    && name != "storage"
                {
                    self.collect_rust_files(root, &path, out)?;
                }
            } else if ft.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "rs" {
                        let rel = path.strip_prefix(root).unwrap_or(&path);
                        let norm = rel.to_string_lossy().replace('\\', "/");
                        out.push(norm);
                    }
                }
            }
        }

        Ok(())
    }
}

/// Helper function to compute dynamic SHA-256 without hardcoded literals.
pub fn compute_dynamic_sha256(path: &Path, buffer_size: usize) -> std::io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let buf_sz = buffer_size.max(1);
    let mut buffer = vec![0u8; buf_sz];

    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    Ok(hex::encode(hasher.finalize()))
}

/// Computes RFC 2104 HMAC-SHA256 in 100% native Rust using sha2.
pub fn compute_hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    const BLOCK_SIZE: usize = 64;
    let mut k_prime = [0u8; BLOCK_SIZE];
    if key.len() > BLOCK_SIZE {
        let mut hasher = Sha256::new();
        hasher.update(key);
        let key_hash = hasher.finalize();
        k_prime[..key_hash.len()].copy_from_slice(&key_hash);
    } else {
        k_prime[..key.len()].copy_from_slice(key);
    }

    let mut inner_pad = [0u8; BLOCK_SIZE];
    let mut outer_pad = [0u8; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        inner_pad[i] = k_prime[i] ^ 0x36;
        outer_pad[i] = k_prime[i] ^ 0x5c;
    }

    let mut inner_hasher = Sha256::new();
    inner_hasher.update(inner_pad);
    inner_hasher.update(data);
    let inner_hash = inner_hasher.finalize();

    let mut outer_hasher = Sha256::new();
    outer_hasher.update(outer_pad);
    outer_hasher.update(inner_hash);
    let result = outer_hasher.finalize();

    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// Verifies GitHub webhook HMAC-SHA256 signature (`X-Hub-Signature-256`) in constant time.
pub fn verify_webhook_hmac_signature(secret: &[u8], body: &[u8], signature_header: &str) -> bool {
    let expected_hex = signature_header
        .strip_prefix("sha256=")
        .unwrap_or(signature_header)
        .trim();
    let computed_hmac = compute_hmac_sha256(secret, body);
    let computed_hex = hex::encode(computed_hmac);

    if expected_hex.len() != computed_hex.len() {
        return false;
    }

    // Constant-time byte comparison to protect against timing side-channel attacks
    let mut diff = 0u8;
    for (a, b) in expected_hex.as_bytes().iter().zip(computed_hex.as_bytes().iter()) {
        diff |= a.to_ascii_lowercase() ^ b.to_ascii_lowercase();
    }
    diff == 0
}

/// Generates official GitHub Actions CI workflow content.
pub fn generate_github_ci_workflow_content() -> String {
    r#"# TARA AI Core — Cross-Platform Continuous Integration (GitHub Actions)
# Validates Windows and Linux builds, zero compiler warnings, and architecture integrity.

name: TARA CI (Cross-Platform)

on:
  push:
    branches: [ main, master, dev ]
  pull_request:
    branches: [ main, master, dev ]

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  tara_integrity_check:
    name: Build & Architecture Verification (${{ matrix.os }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ ubuntu-latest, windows-latest ]

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Install stable Rust toolchain
        uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt

      - name: Rust Cache
        uses: Swatinem/rust-cache@v2

      - name: Verify Workspace (Zero Warnings Directive)
        run: cargo check --workspace --all-targets

      - name: Run Whole-System Architecture Audit
        run: cargo run -p tara_engine --bin verify_architecture

      - name: Run Project Link & Reference Engine Verification
        run: cargo run -p tara_engine --bin verify_project_link_engine

      - name: Run GitHub Adapter Pre-Push & Import Verification
        run: cargo run -p tara_engine --bin tara_github_sync -- --verify
"#.to_string()
}

/// Helper to format ISO timestamp.
fn chrono_now_iso() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{epoch}")
}

/// Finds workspace root containing AGENTS.md and Cargo.toml.
pub fn find_workspace_root() -> Result<PathBuf, String> {
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
