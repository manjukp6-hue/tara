//! Fail-Closed Runtime Tamper Detection Sentinel.
//!
//! Rule compliance:
//! - Pure native Rust, zero outside crate dependencies.
//! - Validates filesystem integrity against dynamically registered baselines (dynamic hash registry).
//! - Fails closed: any modified or missing baseline file reports tamper.
//! - Can signal AuthorityManager for instant lockdown upon tamper detection.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::cryptography::CryptoUtils;
use crate::hardware_root::AuthorityManager;
use crate::path_security::check_lexical_traversal;

/// Runtime Tamper Detection Sentinel.
pub struct RuntimeTamperSentinel {
    repo_root: PathBuf,
    baseline_hashes: Arc<Mutex<HashMap<String, String>>>,
    authority: Option<Arc<AuthorityManager>>,
}

impl RuntimeTamperSentinel {
    pub fn new(repo_root: PathBuf) -> Self {
        Self {
            repo_root,
            baseline_hashes: Arc::new(Mutex::new(HashMap::new())),
            authority: None,
        }
    }

    pub fn with_authority(mut self, authority: Arc<AuthorityManager>) -> Self {
        self.authority = Some(authority);
        self
    }

    /// Registers baseline hashes for a set of critical files dynamically at runtime.
    pub fn register_baseline(&self, relative_paths: &[String]) -> Result<usize, String> {
        let mut map = self
            .baseline_hashes
            .lock()
            .map_err(|_| "Baseline lock poisoned")?;
        let mut registered = 0;

        for rel in relative_paths {
            check_lexical_traversal(Path::new(rel)).map_err(|e| format!("InvalidPath: {e}"))?;

            let full = self.repo_root.join(rel);
            if full.is_symlink() {
                return Err(format!(
                    "Symlinks are forbidden in baseline registration: {}",
                    rel
                ));
            }
            if full.exists() && full.is_file() {
                let hash = CryptoUtils::compute_file_sha256(&full)?;
                map.insert(rel.clone(), hash);
                registered += 1;
            }
        }

        Ok(registered)
    }

    /// Verifies current files against the baseline.
    /// Returns (is_intact: bool, tampered_files: Vec<String>).
    pub fn verify_integrity(&self, epoch_secs: u64) -> (bool, Vec<String>) {
        let map = match self.baseline_hashes.lock() {
            Ok(m) => m,
            Err(_) => return (false, vec!["LOCK_POISONED".to_string()]),
        };

        if map.is_empty() {
            return (false, vec!["NO_BASELINE_REGISTERED".to_string()]);
        }

        let mut tampered = Vec::new();

        for (rel, expected_hash) in map.iter() {
            let full = self.repo_root.join(rel);
            if full.is_symlink() {
                tampered.push(format!("SYMLINK_REPLACED:{}", rel));
                continue;
            }
            if !full.exists() {
                tampered.push(format!("MISSING:{}", rel));
                continue;
            }

            match CryptoUtils::compute_file_sha256(&full) {
                Ok(current_hash) => {
                    if !CryptoUtils::constant_time_eq(
                        current_hash.as_bytes(),
                        expected_hash.as_bytes(),
                    ) {
                        tampered.push(format!("ALTERED:{}", rel));
                    }
                }
                Err(e) => {
                    tampered.push(format!("UNREADABLE:{}:{}", rel, e));
                }
            }
        }

        let is_intact = tampered.is_empty();

        // If tamper detected and authority is attached, trigger instant global lockdown!
        if !is_intact {
            if let Some(ref auth) = self.authority {
                let reason = format!("RuntimeTamperDetected: {:?}", tampered);
                let _ = auth.trigger_lockdown(&reason, epoch_secs);
            }
        }

        (is_intact, tampered)
    }

    /// Persists dynamically computed baseline registry to a JSON file.
    pub fn save_baseline_registry(&self, registry_file: &Path) -> Result<(), String> {
        let map = self
            .baseline_hashes
            .lock()
            .map_err(|_| "Baseline lock poisoned")?;
        let serialized = serde_json::to_vec_pretty(&*map)
            .map_err(|e| format!("Failed to serialize baseline registry: {e}"))?;

        crate::persistence::atomic_write_file_hardened(registry_file, &serialized)
    }

    /// Loads baseline hashes from an operational registry file.
    pub fn load_baseline_registry(&self, registry_file: &Path) -> Result<usize, String> {
        if !registry_file.exists() {
            return Err("Registry file not found".into());
        }
        let data = std::fs::read(registry_file)
            .map_err(|e| format!("Failed to read registry file: {e}"))?;
        let loaded: HashMap<String, String> = serde_json::from_slice(&data)
            .map_err(|e| format!("Corrupted baseline registry JSON: {e}"))?;

        let count = loaded.len();
        let mut map = self
            .baseline_hashes
            .lock()
            .map_err(|_| "Baseline lock poisoned")?;
        *map = loaded;
        Ok(count)
    }
}

pub type SourceTamperDetector = RuntimeTamperSentinel;
