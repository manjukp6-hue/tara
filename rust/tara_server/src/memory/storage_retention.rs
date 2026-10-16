//! Storage Retention Manager — prevents unbounded disk growth.
//!
//! Three growth sources are addressed:
//!
//! 1. **Knowledge partitions** (`storage/knowledge/partitions/`)
//!    — file-per-entry; deduplicated by subject but grows without bound
//!    across topics. `enforce_knowledge_retention()` removes lowest-confidence
//!    entries when a domain exceeds `max_entries_per_domain`.
//!
//! 2. **Model versions manifest** (`versions_manifest.json`)
//!    — `rollback_history` and superseded `versions` entries accumulate.
//!    `prune_model_versions()` trims both to `max_rollback_history` and
//!    removes superseded version records whose `artifact_location` dir is gone.
//!
//! 3. **Candidate staging directories** (`storage/models/candidates/CYCLE_*/`)
//!    — left on disk after promotion or rejection.
//!    `cleanup_stale_candidates()` deletes all candidate dirs older than
//!    `candidate_max_age_secs` that are NOT the currently active model dir.
//!
//! All operations are non-destructive to the active model and active knowledge.
//! Errors are logged but do not propagate — retention is best-effort.

use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};

/// Default: keep at most this many entries per knowledge domain.
pub const DEFAULT_MAX_ENTRIES_PER_DOMAIN: usize = 10_000;
/// Default: keep at most this many rollback history entries in the manifest.
pub const DEFAULT_MAX_ROLLBACK_HISTORY: usize = 10;
/// Default: delete candidate staging dirs older than 7 days.
pub const DEFAULT_CANDIDATE_MAX_AGE_SECS: u64 = 7 * 24 * 3600;

pub struct StorageRetentionManager {
    repo_root: PathBuf,
    /// Maximum knowledge entries per domain directory.
    max_entries_per_domain: usize,
    /// Maximum rollback_history entries kept in the model manifest.
    max_rollback_history: usize,
    /// Maximum age of a candidate staging directory before deletion.
    candidate_max_age_secs: u64,
}

impl StorageRetentionManager {
    pub fn new(repo_root: &str) -> Self {
        Self {
            repo_root: PathBuf::from(repo_root),
            max_entries_per_domain: DEFAULT_MAX_ENTRIES_PER_DOMAIN,
            max_rollback_history: DEFAULT_MAX_ROLLBACK_HISTORY,
            candidate_max_age_secs: DEFAULT_CANDIDATE_MAX_AGE_SECS,
        }
    }

    /// Override limits (for testing or operator configuration). Enforces sensible minimum bounds.
    pub fn with_limits(
        mut self,
        max_entries_per_domain: usize,
        max_rollback_history: usize,
        candidate_max_age_secs: u64,
    ) -> Self {
        self.max_entries_per_domain = max_entries_per_domain.max(10);
        self.max_rollback_history = max_rollback_history.max(1);
        self.candidate_max_age_secs = candidate_max_age_secs.max(3600);
        self
    }

    /// Run all three retention passes. Returns a JSON summary.
    pub fn run_all(&self) -> Value {
        let knowledge_result = self.enforce_knowledge_retention();
        let manifest_result = self.prune_model_versions();
        let candidate_result = self.cleanup_stale_candidates();
        json!({
            "status": "DONE",
            "knowledge": knowledge_result,
            "manifest": manifest_result,
            "candidates": candidate_result
        })
    }

    // ── 1. Knowledge retention ──────────────────────────────────────────────

    /// For each domain subdirectory under `storage/knowledge/partitions/`,
    /// if the entry count exceeds `max_entries_per_domain`, delete the
    /// lowest-confidence entries until the count is within the limit.
    ///
    /// Confidence is read from the JSON field `"confidence"`.
    /// Files that cannot be read/parsed are treated as confidence=0.0
    /// (first candidates for deletion).
    pub fn enforce_knowledge_retention(&self) -> Value {
        let partitions = self
            .repo_root
            .join("storage")
            .join("knowledge")
            .join("partitions");
        if !partitions.exists() {
            return json!({"skipped": true, "reason": "no partitions directory"});
        }

        let mut total_checked = 0u64;
        let mut total_deleted = 0u64;
        let mut domains_pruned = 0u32;
        let mut errors: Vec<String> = Vec::new();

        let domain_dirs = match fs::read_dir(&partitions) {
            Ok(rd) => rd
                .flatten()
                .filter(|e| e.path().is_dir())
                .collect::<Vec<_>>(),
            Err(e) => {
                return json!({"status": "ERROR", "error": format!("read_dir failed: {e}")});
            }
        };

        for domain_entry in domain_dirs {
            let domain_path = domain_entry.path();
            let domain_name = domain_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?")
                .to_string();

            // Collect all .json files in this domain dir (recursively traversing subdirectories)
            let mut files: Vec<(PathBuf, f64)> = Vec::new();
            let mut dirs_to_visit = vec![domain_path.clone()];
            while let Some(current_dir) = dirs_to_visit.pop() {
                match fs::read_dir(&current_dir) {
                    Ok(rd) => {
                        for entry in rd.flatten() {
                            let p = entry.path();
                            if p.is_dir() {
                                dirs_to_visit.push(p);
                            } else if p.extension().and_then(|e| e.to_str()) == Some("json") {
                                // Skip index/metadata files
                                let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                                if stem == "index" || stem == "metadata" || stem == "_index" {
                                    continue;
                                }
                                let confidence = Self::read_confidence(&p);
                                files.push((p, confidence));
                                total_checked += 1;
                            }
                        }
                    }
                    Err(e) => {
                        errors.push(format!("domain {domain_name} dir {}: {e}", current_dir.display()));
                    }
                }
            }

            if files.len() <= self.max_entries_per_domain {
                continue;
            }

            // Sort ascending by confidence — lowest confidence first for deletion
            files.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let to_delete = files.len() - self.max_entries_per_domain;
            for (path, _conf) in files.iter().take(to_delete) {
                match fs::remove_file(path) {
                    Ok(_) => total_deleted += 1,
                    Err(e) => errors.push(format!("delete {}: {e}", path.display())),
                }
            }
            domains_pruned += 1;
        }

        json!({
            "total_files_checked": total_checked,
            "total_files_deleted": total_deleted,
            "domains_pruned": domains_pruned,
            "max_entries_per_domain": self.max_entries_per_domain,
            "errors": errors
        })
    }

    fn read_confidence(path: &Path) -> f64 {
        fs::read_to_string(path)
            .ok()
            .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
            .and_then(|v| v.get("confidence").and_then(Value::as_f64))
            .unwrap_or(0.0)
    }

    // ── 2. Model manifest pruning ───────────────────────────────────────────

    /// Prune the `versions_manifest.json`:
    /// - Trim `rollback_history` to the latest `max_rollback_history` entries.
    /// - Remove `versions` entries whose `status == "superseded"` AND whose
    ///   `artifact_location` directory no longer exists on disk.
    ///   The active version entry is NEVER removed.
    pub fn prune_model_versions(&self) -> Value {
        let manifest_path = self
            .repo_root
            .join("storage")
            .join("models")
            .join("versions_manifest.json");

        if !manifest_path.exists() {
            return json!({"skipped": true, "reason": "no versions_manifest.json"});
        }

        let raw = match fs::read_to_string(&manifest_path) {
            Ok(r) => r,
            Err(e) => return json!({"status": "ERROR", "error": format!("read: {e}")}),
        };
        let mut manifest: Value = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(e) => return json!({"status": "ERROR", "error": format!("parse: {e}")}),
        };

        let active_id = manifest
            .get("active_version")
            .and_then(Value::as_str)
            .map(str::to_string);

        let mut rollback_trimmed = 0u32;
        let mut versions_removed = 0u32;

        // Trim rollback_history
        if let Some(history) = manifest
            .get_mut("rollback_history")
            .and_then(Value::as_array_mut)
        {
            let keep = self.max_rollback_history;
            if history.len() > keep {
                let excess = history.len() - keep;
                rollback_trimmed = excess as u32;
                // Drain oldest entries (front of array = oldest)
                history.drain(..excess);
            }
        }

        // Remove obsolete superseded versions
        if let Some(versions) = manifest.get_mut("versions").and_then(Value::as_object_mut) {
            let active = active_id.as_deref().unwrap_or("");
            let keys_to_remove: Vec<String> = versions
                .iter()
                .filter_map(|(id, entry)| {
                    if id == active {
                        return None; // Never remove active
                    }
                    let status = entry.get("status").and_then(Value::as_str).unwrap_or("");
                    if status != "superseded" {
                        return None;
                    }
                    // Only remove if the artifact location is gone
                    let artifact = entry
                        .get("artifact_location")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    if artifact.is_empty() {
                        return Some(id.clone()); // No location → safe to remove record
                    }
                    let artifact_path = self.repo_root.join(artifact);
                    if !artifact_path.exists() {
                        Some(id.clone())
                    } else {
                        None // Artifact still on disk → keep record
                    }
                })
                .collect();

            versions_removed = keys_to_remove.len() as u32;
            for k in &keys_to_remove {
                versions.remove(k);
            }
        }

        // Write back atomically and durably only if something changed
        if rollback_trimmed > 0 || versions_removed > 0 {
            let tmp = manifest_path.with_extension("json.tmp");
            let write_res = (|| -> Result<(), std::io::Error> {
                let json_bytes = serde_json::to_vec_pretty(&manifest)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                let mut f = fs::File::create(&tmp)?;
                use std::io::Write;
                f.write_all(&json_bytes)?;
                f.sync_all()?;
                drop(f);
                fs::rename(&tmp, &manifest_path)?;
                Ok(())
            })();
            if let Err(e) = write_res {
                let _ = fs::remove_file(&tmp);
                return json!({"status": "ERROR", "error": format!("write: {e}")});
            }
        }

        json!({
            "rollback_history_trimmed": rollback_trimmed,
            "superseded_versions_removed": versions_removed,
            "max_rollback_history": self.max_rollback_history
        })
    }

    // ── 3. Candidate staging directory cleanup ──────────────────────────────

    /// Delete candidate staging directories under `storage/models/candidates/`
    /// (and any other `CYCLE_*` prefixed directories under `storage/models/`)
    /// that are:
    ///   - Older than `candidate_max_age_secs` (based on directory mtime), AND
    ///   - NOT the currently active model directory.
    ///
    /// The active model directory is read from `versions_manifest.json`.
    pub fn cleanup_stale_candidates(&self) -> Value {
        let candidates_root = self.repo_root.join("storage").join("models");
        if !candidates_root.exists() {
            return json!({"skipped": true, "reason": "no storage/models directory"});
        }

        // Determine the active model's artifact path (to protect it)
        let active_artifact = match self.active_artifact_location() {
            Some(loc) => loc,
            None => {
                let manifest_path = self
                    .repo_root
                    .join("storage")
                    .join("models")
                    .join("versions_manifest.json");
                if manifest_path.exists() {
                    return json!({
                        "status": "ABORTED",
                        "error": "Cannot verify active model artifact location from versions_manifest.json. Failing closed to protect active model."
                    });
                }
                return json!({"skipped": true, "reason": "no versions_manifest.json"});
            }
        };

        let cutoff = SystemTime::now()
            .checked_sub(Duration::from_secs(self.candidate_max_age_secs))
            .unwrap_or(SystemTime::UNIX_EPOCH);

        let mut deleted = 0u32;
        let mut skipped_active = 0u32;
        let mut errors: Vec<String> = Vec::new();

        let mut scan_dirs = vec![candidates_root.clone()];
        let nested_candidates = candidates_root.join("candidates");
        if nested_candidates.exists() && nested_candidates.is_dir() {
            scan_dirs.push(nested_candidates);
        }

        let mut entries = Vec::new();
        for s_dir in scan_dirs {
            if let Ok(rd) = fs::read_dir(&s_dir) {
                entries.extend(rd.flatten());
            }
        }

        for entry in entries {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

            // Only touch CYCLE_* prefixed directories (candidate staging dirs)
            if !name.starts_with("CYCLE_") && !name.starts_with("candidate") {
                continue;
            }

            // Never delete the active artifact
            let active_path = self.repo_root.join(&active_artifact);
            if path.starts_with(&active_path) || active_path.starts_with(&path) {
                skipped_active += 1;
                continue;
            }

            // Check age via mtime
            let is_old = fs::metadata(&path)
                .and_then(|m| m.modified())
                .map(|mtime| mtime < cutoff)
                .unwrap_or(false);

            if !is_old {
                continue;
            }

            match fs::remove_dir_all(&path) {
                Ok(_) => deleted += 1,
                Err(e) => errors.push(format!("remove {}: {e}", path.display())),
            }
        }

        json!({
            "candidate_dirs_deleted": deleted,
            "skipped_active_model": skipped_active,
            "candidate_max_age_secs": self.candidate_max_age_secs,
            "errors": errors
        })
    }

    fn active_artifact_location(&self) -> Option<String> {
        let root_str = self.repo_root.to_str().unwrap_or(".");
        crate::learning::model_registry::ModelRegistry::new(root_str)
            .get_active_version_location()
    }

    /// Returns the total disk usage of the knowledge and model storage directories in bytes.
    pub fn storage_usage_bytes(&self) -> Value {
        let knowledge_bytes =
            Self::dir_size_bytes(&self.repo_root.join("storage").join("knowledge"));
        let models_bytes = Self::dir_size_bytes(&self.repo_root.join("storage").join("models"));
        json!({
            "knowledge_bytes": knowledge_bytes,
            "models_bytes": models_bytes,
            "total_bytes": knowledge_bytes + models_bytes
        })
    }

    fn dir_size_bytes(dir: &Path) -> u64 {
        let mut total = 0u64;
        let mut stack = vec![dir.to_path_buf()];
        let mut visited = std::collections::HashSet::new();
        while let Some(current) = stack.pop() {
            if let Ok(canonical) = current.canonicalize() {
                if !visited.insert(canonical) {
                    continue;
                }
            }
            match fs::read_dir(&current) {
                Ok(rd) => {
                    for entry in rd.flatten() {
                        let p = entry.path();
                        if let Ok(m) = fs::symlink_metadata(&p) {
                            if m.file_type().is_symlink() {
                                continue;
                            }
                            if m.is_dir() {
                                stack.push(p);
                            } else {
                                total += m.len();
                            }
                        }
                    }
                }
                Err(_) => continue,
            }
        }
        total
    }
}
