//! memory_hub.rs
//!
//! TARA Dynamic Memory Hub — ports Python's `memory_interfaces.py`.
//! Provides a simple key-value store backed by JSON files in a configured
//! storage directory. Keys map to arbitrary JSON values.

use std::fs;
use std::path::{Path, PathBuf};

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum MemoryHubError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialisation error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Invalid key '{0}': must be non-empty and contain only alphanumeric, dash, or underscore chars")]
    InvalidKey(String),
    #[error("Storage directory could not be created: {0}")]
    StorageInit(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// DynamicMemoryHub
// ─────────────────────────────────────────────────────────────────────────────

/// A flat, file-backed key-value memory store for TARA.
///
/// Each key is persisted as `<storage_dir>/<key>.json`.
/// Keys must be non-empty and match `[A-Za-z0-9_\-]+`.
pub struct DynamicMemoryHub {
    /// Absolute path to the storage directory.
    pub storage_dir: String,
}

impl DynamicMemoryHub {
    /// Create a new hub that stores data in `storage_dir`.
    /// The directory is created if it does not exist.
    pub fn new(storage_dir: impl Into<String>) -> Result<Self, MemoryHubError> {
        let dir = storage_dir.into();
        fs::create_dir_all(&dir).map_err(|e| MemoryHubError::StorageInit(e.to_string()))?;
        Ok(Self { storage_dir: dir })
    }

    /// Validate that a key contains only safe filesystem characters.
    fn validate_key(&self, key: &str) -> Result<(), MemoryHubError> {
        if key.is_empty()
            || !key
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.')
        {
            return Err(MemoryHubError::InvalidKey(key.to_string()));
        }
        Ok(())
    }

    /// Derive the full file path for a key.
    fn key_path(&self, key: &str) -> PathBuf {
        Path::new(&self.storage_dir).join(format!("{}.json", key))
    }

    /// Persist a JSON value under the given key (overwrites existing).
    pub fn store(&self, key: &str, value: serde_json::Value) -> Result<(), MemoryHubError> {
        self.validate_key(key)?;
        let path = self.key_path(key);
        let serialised = serde_json::to_string_pretty(&value)?;
        fs::write(&path, serialised)?;
        Ok(())
    }

    /// Retrieve the JSON value stored under `key`.
    /// Returns `None` if the key does not exist.
    pub fn retrieve(&self, key: &str) -> Result<Option<serde_json::Value>, MemoryHubError> {
        self.validate_key(key)?;
        let path = self.key_path(key);
        if !path.exists() {
            return Ok(None);
        }
        let raw = fs::read_to_string(&path)?;
        let value: serde_json::Value = serde_json::from_str(&raw)?;
        Ok(Some(value))
    }

    /// Delete a key. Returns `true` if the key existed and was deleted.
    pub fn delete(&self, key: &str) -> Result<bool, MemoryHubError> {
        self.validate_key(key)?;
        let path = self.key_path(key);
        if path.exists() {
            fs::remove_file(&path)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// List all stored keys (sorted alphabetically).
    pub fn list_keys(&self) -> Result<Vec<String>, MemoryHubError> {
        let dir = Path::new(&self.storage_dir);
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut keys = Vec::new();
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let fname = entry.file_name();
            let fname_str = fname.to_string_lossy();
            if fname_str.ends_with(".json") {
                let key = fname_str.trim_end_matches(".json").to_string();
                keys.push(key);
            }
        }
        keys.sort();
        Ok(keys)
    }

    /// Return a JSON summary of all stored key-value pairs.
    pub fn dump_all(&self) -> Result<serde_json::Value, MemoryHubError> {
        let keys = self.list_keys()?;
        let mut map = serde_json::Map::new();
        for key in &keys {
            if let Some(val) = self.retrieve(key)? {
                map.insert(key.clone(), val);
            }
        }
        Ok(serde_json::Value::Object(map))
    }

    /// Return storage statistics.
    pub fn stats(&self) -> Result<serde_json::Value, MemoryHubError> {
        let keys = self.list_keys()?;
        let total_size: u64 = keys
            .iter()
            .map(|k| fs::metadata(self.key_path(k)).map(|m| m.len()).unwrap_or(0))
            .sum();
        Ok(serde_json::json!({
            "storage_dir": self.storage_dir,
            "key_count": keys.len(),
            "total_bytes": total_size,
            "keys": keys
        }))
    }
}
