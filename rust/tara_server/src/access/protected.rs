//! Action Broker and Protected Policy Store.
//!
//! Enforces:
//! - Only authenticated ROOT_CREATOR with active session can modify sources or create policies.
//! - Path validation preventing traversal and mutations to immutable weights.
//! - Snapshot rollback creation before any modification.
//! - Cryptographic hash-chaining of policy records.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

pub const GENESIS_POLICY_HASH: &str = "GENESIS_POLICY_ROOT_HASH";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyEntry {
    pub policy_id: String,
    pub version: u64,
    pub policy_text: String,
    pub status: String,
    pub prev_hash: String,
    pub entry_hash: String,
    pub updated_at: String,
}

pub struct PolicyRecordStore {
    pub store_path: PathBuf,
    pub policies: HashMap<String, PolicyEntry>,
    pub last_hash: String,
}

impl PolicyRecordStore {
    pub fn new<P: AsRef<Path>>(store_path: Option<P>) -> Self {
        let p = match store_path {
            Some(path) => path.as_ref().to_path_buf(),
            None => PathBuf::from("storage/root_policies/root_policies.json"),
        };

        let mut store = Self {
            store_path: p,
            policies: HashMap::new(),
            last_hash: GENESIS_POLICY_HASH.to_string(),
        };

        let _ = store.load();
        store
    }

    pub fn compute_entry_hash(
        prev_hash: &str,
        policy_id: &str,
        version: u64,
        policy_text: &str,
        status: &str,
        updated_at: &str,
    ) -> String {
        let payload = format!(
            "{}|{}|{}|{}|{}|{}",
            prev_hash, policy_id, version, policy_text, status, updated_at
        );
        hex::encode(Sha256::digest(payload.as_bytes()))
    }

    pub fn load(&mut self) -> Result<(), String> {
        if self.store_path.exists() {
            let data = fs::read_to_string(&self.store_path).map_err(|e| e.to_string())?;
            let v: Value = serde_json::from_str(&data).map_err(|e| e.to_string())?;
            self.last_hash = v
                .get("last_hash")
                .and_then(Value::as_str)
                .unwrap_or(GENESIS_POLICY_HASH)
                .to_string();
            if let Some(map) = v.get("policies").and_then(Value::as_object) {
                for (k, val) in map {
                    if let Ok(entry) = serde_json::from_value::<PolicyEntry>(val.clone()) {
                        self.policies.insert(k.clone(), entry);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn save(&self) -> Result<(), String> {
        if let Some(parent) = self.store_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let data = json!({
            "store_version": "1.0",
            "last_hash": self.last_hash,
            "policies": self.policies,
        });
        fs::write(
            &self.store_path,
            serde_json::to_string_pretty(&data).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())
    }

    pub fn create_policy(
        &mut self,
        is_auth: bool,
        policy_id: &str,
        policy_text: &str,
    ) -> Result<PolicyEntry, String> {
        if !is_auth {
            return Err("Unauthorized: Root creator authority required".into());
        }
        if policy_text.trim().is_empty() {
            return Err("Policy text cannot be empty".into());
        }

        // Version is the insertion sequence across the store (1-based).
        // This makes every entry's version unique and monotonically increasing
        // so the chain can be reconstructed in sorted order by verify_store_integrity().
        let version = self.policies.len() as u64 + 1;
        let updated_at = crate::now_iso();
        let entry_hash = Self::compute_entry_hash(
            &self.last_hash,
            policy_id,
            version,
            policy_text,
            "ACTIVE",
            &updated_at,
        );
        let entry = PolicyEntry {
            policy_id: policy_id.to_string(),
            version,
            policy_text: policy_text.to_string(),
            status: "ACTIVE".to_string(),
            prev_hash: self.last_hash.clone(),
            entry_hash: entry_hash.clone(),
            updated_at,
        };

        self.last_hash = entry_hash;
        self.policies.insert(policy_id.to_string(), entry.clone());
        self.save()?;
        Ok(entry)
    }

    pub fn set_policy_status(
        &mut self,
        is_auth: bool,
        policy_id: &str,
        new_status: &str,
    ) -> Result<PolicyEntry, String> {
        if !is_auth {
            return Err("Unauthorized".into());
        }
        let mut entry = self
            .policies
            .get(policy_id)
            .cloned()
            .ok_or_else(|| "Policy not found".to_string())?;
        entry.version += 1;
        entry.status = new_status.to_string();
        entry.updated_at = crate::now_iso();
        entry.prev_hash = self.last_hash.clone();
        entry.entry_hash = Self::compute_entry_hash(
            &entry.prev_hash,
            &entry.policy_id,
            entry.version,
            &entry.policy_text,
            &entry.status,
            &entry.updated_at,
        );

        self.last_hash = entry.entry_hash.clone();
        self.policies.insert(policy_id.to_string(), entry.clone());
        self.save()?;
        Ok(entry)
    }

    pub fn delete_policy(&mut self, is_auth: bool, policy_id: &str) -> Result<bool, String> {
        if !is_auth {
            return Err("Unauthorized".into());
        }
        let res = self.policies.remove(policy_id).is_some();
        if res {
            self.save()?;
        }
        Ok(res)
    }

    pub fn verify_store_integrity(&self) -> bool {
        // Sort entries by version (ascending) to reconstruct the chain in insertion order.
        // An entry with version 1 must declare prev_hash == GENESIS. Every subsequent entry
        // must declare prev_hash equal to the entry_hash of the previous version in the chain.
        let mut entries: Vec<&PolicyEntry> = self.policies.values().collect();
        entries.sort_by_key(|e| e.version);

        let mut running_hash = GENESIS_POLICY_HASH.to_string();
        for entry in &entries {
            // Verify that the entry correctly links to the current chain head.
            if entry.prev_hash != running_hash {
                return false;
            }
            // Re-derive the entry hash from its content to detect any tampered field.
            let expected = Self::compute_entry_hash(
                &entry.prev_hash,
                &entry.policy_id,
                entry.version,
                &entry.policy_text,
                &entry.status,
                &entry.updated_at,
            );
            if entry.entry_hash != expected {
                return false;
            }
            // Advance the chain pointer to this entry's hash.
            running_hash = entry.entry_hash.clone();
        }
        true
    }
}

pub struct ActionBroker {
    pub repo_root: PathBuf,
    pub snapshot_dir: PathBuf,
}

impl ActionBroker {
    pub fn new<P: AsRef<Path>>(repo_root: P) -> Self {
        let root = repo_root.as_ref().to_path_buf();
        let snap = root.join("storage").join("snapshots");
        let _ = fs::create_dir_all(&snap);
        Self {
            repo_root: root,
            snapshot_dir: snap,
        }
    }

    pub fn validate_path(&self, rel_or_abs: &str) -> Result<PathBuf, String> {
        if rel_or_abs.trim().is_empty() {
            return Err("Empty path provided".into());
        }

        // Normalise the input against repo_root.
        let raw = if Path::new(rel_or_abs).is_absolute() {
            PathBuf::from(rel_or_abs)
        } else {
            self.repo_root.join(rel_or_abs)
        };

        // canonicalize() resolves symlinks and ".." components.
        // If the file doesn't exist yet we cannot canonicalize the full path,
        // so we canonicalize only the repo_root anchor (which must exist) and
        // then check that the normalised raw path starts with that anchor.
        let root_canon = self
            .repo_root
            .canonicalize()
            .unwrap_or_else(|_| self.repo_root.clone());

        // Attempt to canonicalize the target; if the path doesn't exist yet,
        // produce a "logical absolute" by resolving ".." manually.
        let canon = raw.canonicalize().unwrap_or_else(|_| {
            // Walk the components and resolve "..".
            let mut resolved = root_canon.clone();
            let suffix = if Path::new(rel_or_abs).is_absolute() {
                Path::new(rel_or_abs).to_path_buf()
            } else {
                PathBuf::from(rel_or_abs)
            };
            for component in suffix.components() {
                use std::path::Component;
                match component {
                    Component::ParentDir => {
                        resolved.pop();
                    }
                    Component::Normal(part) => resolved.push(part),
                    Component::CurDir => {}
                    Component::RootDir => resolved = PathBuf::from("/"),
                    Component::Prefix(p) => resolved = PathBuf::from(p.as_os_str()),
                }
            }
            resolved
        });

        // The resolved path MUST be inside repo_root.
        if !canon.starts_with(&root_canon) {
            return Err(format!(
                "Path traversal rejected: '{}' resolves outside repository root",
                rel_or_abs
            ));
        }

        let rel_str = rel_or_abs.replace('\\', "/");
        if rel_str.contains("/.git") || rel_str.starts_with(".git") {
            return Err("Access denied: .git is protected".into());
        }
        if rel_str.to_lowercase().contains("model.safetensors") {
            return Err("Access denied: Production model weights are immutable".into());
        }

        Ok(raw)
    }

    pub fn create_snapshot(
        &self,
        affected_files: &[PathBuf],
        change_id: &str,
    ) -> Result<PathBuf, String> {
        let snap_path = self.snapshot_dir.join(change_id);
        fs::create_dir_all(&snap_path).map_err(|e| e.to_string())?;

        for f in affected_files {
            if f.exists() && f.is_file() {
                if let Ok(rel) = f.strip_prefix(&self.repo_root) {
                    let dest = snap_path.join(rel);
                    if let Some(parent) = dest.parent() {
                        let _ = fs::create_dir_all(parent);
                    }
                    let _ = fs::copy(f, dest);
                }
            }
        }
        Ok(snap_path)
    }

    pub fn read_source(&self, is_auth: bool, rel_path: &str) -> Result<String, String> {
        if !is_auth {
            return Err("Unauthorized".into());
        }
        let full_path = self.validate_path(rel_path)?;
        fs::read_to_string(full_path).map_err(|e| e.to_string())
    }

    pub fn modify_source(
        &self,
        is_auth: bool,
        rel_path: &str,
        new_content: &str,
        change_id: &str,
    ) -> Result<(), String> {
        if !is_auth {
            return Err("Unauthorized".into());
        }
        let full_path = self.validate_path(rel_path)?;

        // Create snapshot before modifying
        self.create_snapshot(std::slice::from_ref(&full_path), change_id)?;

        if let Some(parent) = full_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(full_path, new_content).map_err(|e| e.to_string())
    }

    pub fn create_source(
        &self,
        is_auth: bool,
        rel_path: &str,
        content: &str,
        change_id: &str,
    ) -> Result<(), String> {
        if !is_auth {
            return Err("Unauthorized".into());
        }
        let full_path = self.validate_path(rel_path)?;
        if full_path.exists() {
            return Err("File already exists".into());
        }
        self.create_snapshot(std::slice::from_ref(&full_path), change_id)?;
        if let Some(parent) = full_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(full_path, content).map_err(|e| e.to_string())
    }

    pub fn delete_source(
        &self,
        is_auth: bool,
        rel_path: &str,
        change_id: &str,
    ) -> Result<(), String> {
        if !is_auth {
            return Err("Unauthorized".into());
        }
        let full_path = self.validate_path(rel_path)?;
        if full_path.exists() {
            self.create_snapshot(std::slice::from_ref(&full_path), change_id)?;
            fs::remove_file(full_path).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}
