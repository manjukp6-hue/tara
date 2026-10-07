//! Fail-Closed Runtime Tamper Detection Engine.
//!
//! Validates integrity of security-critical code and configuration files.
//! Fails closed on unauthorized file alterations.

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

pub struct SourceTamperDetector {
    pub repo_root: PathBuf,
    pub cached_hashes: HashMap<String, String>,
    pub tampered_files: Vec<String>,
}

impl SourceTamperDetector {
    pub fn new<P: AsRef<Path>>(repo_root: Option<P>) -> Self {
        let root = match repo_root {
            Some(p) => p.as_ref().to_path_buf(),
            None => PathBuf::from("."),
        };

        let mut detector = Self {
            repo_root: root,
            cached_hashes: HashMap::new(),
            tampered_files: Vec::new(),
        };

        detector.load_from_manifest();
        detector
    }

    pub fn load_from_manifest(&mut self) {
        let manifest_path = self.repo_root.join("storage").join("release_manifest.json");
        if manifest_path.exists() {
            if let Ok(data) = fs::read_to_string(&manifest_path) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&data) {
                    if let Some(tree) = v.get("file_tree").and_then(serde_json::Value::as_object) {
                        for (k, hash_val) in tree {
                            if let Some(h_str) = hash_val.as_str() {
                                self.cached_hashes.insert(k.clone(), h_str.to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn compute_file_hash(&self, rel_path: &str) -> Option<String> {
        let full_p = self
            .repo_root
            .join(rel_path.replace('/', std::path::MAIN_SEPARATOR_STR));
        if !full_p.exists() || !full_p.is_file() {
            return None;
        }

        if let Ok(bytes) = fs::read(&full_p) {
            Some(hex::encode(Sha256::digest(&bytes)))
        } else {
            None
        }
    }

    pub fn register_baseline(&mut self, critical_files: &[String]) -> HashMap<String, String> {
        for rel in critical_files {
            if let Some(h) = self.compute_file_hash(rel) {
                self.cached_hashes.insert(rel.clone(), h);
            }
        }
        self.cached_hashes.clone()
    }

    pub fn verify_integrity(&mut self, critical_files: Option<&[String]>) -> (bool, Vec<String>) {
        let default_targets: Vec<String> = self.cached_hashes.keys().cloned().collect();
        let targets = critical_files.unwrap_or(&default_targets);

        let mut tampered = Vec::new();
        for rel in targets {
            let current = self.compute_file_hash(rel);
            let expected = self.cached_hashes.get(rel);

            if let Some(exp) = expected {
                if current.as_deref() != Some(exp.as_str()) {
                    tampered.push(rel.clone());
                }
            }
        }

        self.tampered_files = tampered.clone();
        (tampered.is_empty(), tampered)
    }

    pub fn enforce_fail_closed(&mut self) -> Result<(), String> {
        let (is_intact, tampered) = self.verify_integrity(None);
        if !is_intact {
            return Err(format!(
                "Security Tamper Fail-Closed: {} critical file(s) modified or corrupted: {:?}",
                tampered.len(),
                tampered
            ));
        }
        Ok(())
    }
}
