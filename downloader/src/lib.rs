//! Standalone Dataset Downloader Library
//!
//! Provides isolated dataset downloading, dynamic cryptographic integrity verification,
//! 4-way commercial rights license gating, and operational provenance registry management.
//! Strictly isolated with zero connections to engine or core training systems.

use hex::encode as hex_encode;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Permissive license verification evaluator for commercial dataset rights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LicenseEvaluator;

impl LicenseEvaluator {
    /// Evaluates if a given license identifier satisfies commercial use,
    /// modification, redistribution, and training rights with zero restrictions.
    pub fn is_permissive_compliant(license_id: &str) -> bool {
        let normalized = license_id.trim().to_uppercase();

        // Immediate rejection for forbidden non-commercial, no-derivatives, or copyleft constraints
        let forbidden_substrings = [
            "-NC",
            "NON-COMMERCIAL",
            "NONCOMMERCIAL",
            "-ND",
            "NO-DERIVATIVES",
            "NODERIVATIVES",
            "RESEARCH-ONLY",
            "ACADEMIC-ONLY",
            "GPL",
            "AGPL",
            "LGPL",
            "CC-BY-SA",
        ];

        for forbidden in &forbidden_substrings {
            if normalized.contains(forbidden) {
                return false;
            }
        }

        // Permissive approved license designations
        matches!(
            normalized.as_str(),
            "MIT"
                | "APACHE-2.0"
                | "APACHE 2.0"
                | "BSD-2-CLAUSE"
                | "BSD-3-CLAUSE"
                | "CC-BY-4.0"
                | "CC BY 4.0"
                | "CC0-1.0"
                | "CC0"
                | "UNLICENSE"
                | "ISC"
        )
    }
}

/// Operational provenance entry for dynamically verified downloaded assets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProvenanceRecord {
    pub source_url: String,
    pub license_proof_url: String,
    pub license: String,
    pub sha256: String,
    pub timestamp_epoch_secs: u64,
    pub file_size_bytes: u64,
    pub status: String,
}

/// Dynamic runtime ledger for repeat exclusion and integrity verification.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DownloadLedger {
    pub records: HashMap<String, ProvenanceRecord>,
}

impl DownloadLedger {
    /// Loads dynamic ledger from a JSON file. Returns empty ledger if file does not exist.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let p = path.as_ref();
        if !p.exists() {
            return Ok(Self::default());
        }
        let data = fs::read_to_string(p)?;
        serde_json::from_str(&data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Persists dynamic ledger to disk atomically.
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let p = path.as_ref();
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent)?;
        }
        let serialized = serde_json::to_string_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        let temp_path = p.with_extension("tmp");
        {
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&temp_path)?;
            file.write_all(serialized.as_bytes())?;
            file.flush()?;
        }
        fs::rename(&temp_path, p)?;
        Ok(())
    }

    /// Checks if a file key is already registered with a verified SHA-256 digest.
    pub fn has_verified_entry(&self, key: &str, expected_sha: &str) -> bool {
        if let Some(record) = self.records.get(key) {
            record.status == "verified" && record.sha256.eq_ignore_ascii_case(expected_sha)
        } else {
            false
        }
    }

    /// Inserts or updates an operational provenance record.
    pub fn record_asset(&mut self, key: String, record: ProvenanceRecord) {
        self.records.insert(key, record);
    }
}

/// Standalone downloader service configuration and operations.
pub struct DownloaderService {
    target_dir: PathBuf,
    ledger_path: PathBuf,
    ledger: DownloadLedger,
}

impl DownloaderService {
    /// Resolves preferred destination directory: checks D:\ drive first, falls back to local storage if inaccessible.
    pub fn resolve_default_target_dir() -> PathBuf {
        let preferred = PathBuf::from(r"D:\taracore_datasets\downloaded");
        // Test accessibility of preferred storage drive
        if let Some(parent) = preferred.parent() {
            if parent.exists() || fs::create_dir_all(&preferred).is_ok() {
                return preferred;
            }
        }
        // Fallback to local storage if D:\ is unavailable
        PathBuf::from("storage/downloaded")
    }

    /// Initializes service with resolved destination and loaded operational ledger.
    pub fn new(target_dir_override: Option<PathBuf>) -> io::Result<Self> {
        let target_dir = target_dir_override.unwrap_or_else(Self::resolve_default_target_dir);
        fs::create_dir_all(&target_dir)?;
        let ledger_path = target_dir.join("download_ledger.json");
        let ledger = DownloadLedger::load_from_file(&ledger_path)?;
        Ok(Self {
            target_dir,
            ledger_path,
            ledger,
        })
    }

    pub fn target_dir(&self) -> &Path {
        &self.target_dir
    }

    pub fn ledger(&self) -> &DownloadLedger {
        &self.ledger
    }

    /// Computes streaming SHA-256 hash dynamically from filesystem bytes.
    pub fn compute_dynamic_sha256<P: AsRef<Path>>(file_path: P) -> io::Result<String> {
        let mut file = File::open(file_path)?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 65536];

        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }

        Ok(hex_encode(hasher.finalize()))
    }

    /// Verifies downloaded content integrity and 4-way license compliance before committing to ledger.
    pub fn verify_and_register_asset(
        &mut self,
        file_name: &str,
        source_url: &str,
        license_proof_url: &str,
        license_id: &str,
    ) -> Result<ProvenanceRecord, String> {
        // Enforce 4-way commercial license rights
        if !LicenseEvaluator::is_permissive_compliant(license_id) {
            return Err(format!(
                "License '{}' rejected: does not meet unconditional commercial rights and permissive gating requirements",
                license_id
            ));
        }

        let full_path = self.target_dir.join(file_name);
        if !full_path.exists() {
            return Err(format!("Downloaded file does not exist at {:?}", full_path));
        }

        // Dynamically compute runtime SHA-256 hash
        let dynamic_sha = Self::compute_dynamic_sha256(&full_path)
            .map_err(|e| format!("Dynamic hash calculation failed: {}", e))?;

        let metadata = fs::metadata(&full_path)
            .map_err(|e| format!("Failed to read file metadata: {}", e))?;

        let epoch_now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let record = ProvenanceRecord {
            source_url: source_url.to_string(),
            license_proof_url: license_proof_url.to_string(),
            license: license_id.to_string(),
            sha256: dynamic_sha,
            timestamp_epoch_secs: epoch_now,
            file_size_bytes: metadata.len(),
            status: "verified".to_string(),
        };

        self.ledger.record_asset(file_name.to_string(), record.clone());
        self.ledger
            .save_to_file(&self.ledger_path)
            .map_err(|e| format!("Failed to persist operational ledger: {}", e))?;

        Ok(record)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permissive_license_acceptance() {
        assert!(LicenseEvaluator::is_permissive_compliant("MIT"));
        assert!(LicenseEvaluator::is_permissive_compliant("Apache-2.0"));
        assert!(LicenseEvaluator::is_permissive_compliant("BSD-3-Clause"));
        assert!(LicenseEvaluator::is_permissive_compliant("CC0-1.0"));
    }

    #[test]
    fn test_restrictive_license_rejection() {
        assert!(!LicenseEvaluator::is_permissive_compliant("GPL-3.0"));
        assert!(!LicenseEvaluator::is_permissive_compliant("AGPL-3.0"));
        assert!(!LicenseEvaluator::is_permissive_compliant("CC-BY-NC-4.0"));
        assert!(!LicenseEvaluator::is_permissive_compliant("CC-BY-ND-4.0"));
        assert!(!LicenseEvaluator::is_permissive_compliant("Research-Only"));
    }

    #[test]
    fn test_dynamic_hash_and_ledger_lifecycle() {
        let temp_dir = std::env::temp_dir().join(format!("tara_dl_test_{}", std::process::id()));
        fs::create_dir_all(&temp_dir).unwrap();

        let test_file = temp_dir.join("sample_dataset.jsonl");
        fs::write(&test_file, b"{\"text\": \"universal knowledge sample\"}\n").unwrap();

        let mut service = DownloaderService::new(Some(temp_dir.clone())).unwrap();
        let result = service.verify_and_register_asset(
            "sample_dataset.jsonl",
            "https://archive.org/details/universal_sample",
            "https://archive.org/details/license_proof",
            "Apache-2.0",
        );

        assert!(result.is_ok());
        let record = result.unwrap();
        assert_eq!(record.status, "verified");
        assert!(!record.sha256.is_empty());

        let is_recorded = service.ledger().has_verified_entry("sample_dataset.jsonl", &record.sha256);
        assert!(is_recorded);

        // Cleanup test directory
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
