//! Monotonic Witness and Rollback Prevention Subsystem (Pillar 4).
//!
//! Threat Model & Architecture:
//! - Local files, DPAPI blobs, or registry entries alone DO NOT prevent rollback against an attacker
//!   who can restore the entire disk or virtual machine snapshot.
//! - MonotonicAnchor trait explicitly declares whether an anchor is Hardware-Backed,
//!   Remote-Witnessed, or Local-Host-Bounded.
//! - If hardware monotonic facilities are unavailable on the host, the anchor explicitly reports
//!   HardwareMonotonic = false rather than making false claims.

use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::cryptography::CryptoUtils;

/// Classification of anti-rollback assurance provided by an anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnchorAssuranceType {
    /// Anchored to hardware counter (e.g. TPM NV monotonic counter, RPMB).
    HardwareMonotonic,
    /// Anchored to an independent remote witness service via signed checkpoints.
    RemoteWitnessed,
    /// Bounded to the local host OS instance (vulnerable to full VM/disk rollback).
    LocalHostWitness,
    /// Anti-rollback facilities are completely unavailable on this environment.
    Unavailable,
}

/// Abstract Monotonic Anchor trait.
pub trait MonotonicAnchor: Send + Sync {
    fn assurance_type(&self) -> AnchorAssuranceType;
    fn read_watermark(&self) -> Result<u64, String>;
    fn advance_watermark(&self, candidate: u64) -> Result<(), String>;
}

/// Persistent record format for DurableMonotonicAnchor.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MonotonicDiskRecord {
    magic: String,
    anchor_name: String,
    watermark: u64,
    updated_epoch_secs: u64,
    record_sha256: String,
}

impl MonotonicDiskRecord {
    const MAGIC: &'static str = "TARA_MONOTONIC_ANCHOR_V1";

    fn compute_digest(magic: &str, name: &str, watermark: u64, timestamp: u64) -> String {
        let preimage = format!("{}:{}:{}:{}", magic, name, watermark, timestamp);
        CryptoUtils::sha256_hex(preimage.as_bytes())
    }

    fn new_record(anchor_name: &str, watermark: u64, timestamp: u64) -> Self {
        let digest = Self::compute_digest(Self::MAGIC, anchor_name, watermark, timestamp);
        Self {
            magic: Self::MAGIC.to_string(),
            anchor_name: anchor_name.to_string(),
            watermark,
            updated_epoch_secs: timestamp,
            record_sha256: digest,
        }
    }

    fn verify_integrity(&self, expected_name: &str) -> Result<(), String> {
        if self.magic != Self::MAGIC {
            return Err(format!(
                "CorruptedAnchorMagic: Expected {}, got {}",
                Self::MAGIC,
                self.magic
            ));
        }
        if self.anchor_name != expected_name {
            return Err(format!(
                "AnchorNameMismatch: Expected {}, got {}",
                expected_name, self.anchor_name
            ));
        }
        let expected_digest = Self::compute_digest(
            &self.magic,
            &self.anchor_name,
            self.watermark,
            self.updated_epoch_secs,
        );
        if !CryptoUtils::constant_time_eq(self.record_sha256.as_bytes(), expected_digest.as_bytes())
        {
            return Err(
                "AnchorRecordTampered: SHA-256 integrity checksum does not match payload".into(),
            );
        }
        Ok(())
    }
}

/// Durable, crash-safe, and atomic Monotonic Anchor backed by persistent storage.
/// Survives process crashes, restarts, and reboots without losing watermark monotonicity.
pub struct DurableMonotonicAnchor {
    backing_path: PathBuf,
    anchor_name: String,
    current_watermark: Arc<Mutex<u64>>,
}

impl DurableMonotonicAnchor {
    /// Opens existing anchor state from disk or initializes a new one.
    pub fn open_or_create<P: AsRef<Path>>(
        path: P,
        anchor_name: impl Into<String>,
        initial_val: u64,
    ) -> Result<Self, String> {
        let p = path.as_ref().to_path_buf();
        let name = anchor_name.into();

        if let Some(parent) = p.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).map_err(|e| {
                    format!(
                        "Failed to create anchor directory {}: {}",
                        parent.display(),
                        e
                    )
                })?;
            }
        }

        let loaded_watermark = if p.exists() {
            let data = fs::read(&p)
                .map_err(|e| format!("Failed to read anchor file {}: {}", p.display(), e))?;
            let record: MonotonicDiskRecord = serde_json::from_slice(&data).map_err(|e| {
                format!("Corrupted monotonic anchor JSON at {}: {}", p.display(), e)
            })?;
            record.verify_integrity(&name)?;
            record.watermark
        } else {
            // Create initial persistent record atomically
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let record = MonotonicDiskRecord::new_record(&name, initial_val, now);
            Self::atomic_write_record(&p, &record)?;
            initial_val
        };

        Ok(Self {
            backing_path: p,
            anchor_name: name,
            current_watermark: Arc::new(Mutex::new(loaded_watermark)),
        })
    }

    fn atomic_write_record(target_path: &Path, record: &MonotonicDiskRecord) -> Result<(), String> {
        let serialized = serde_json::to_vec_pretty(record)
            .map_err(|e| format!("Serialization error for monotonic record: {e}"))?;

        let temp_path = target_path.with_extension(format!(
            "tmp.{}.{}",
            std::process::id(),
            rand::random::<u64>()
        ));

        // 1. Write and sync temp file
        {
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&temp_path)
                .map_err(|e| {
                    format!(
                        "Failed to open temp anchor file {}: {}",
                        temp_path.display(),
                        e
                    )
                })?;

            file.write_all(&serialized)
                .map_err(|e| format!("Failed to write anchor record: {e}"))?;

            file.sync_all()
                .map_err(|e| format!("Failed to sync anchor record to disk: {e}"))?;
        }

        // 2. Harden permissions on temp file
        crate::hardware_root::harden_file_permissions(&temp_path)?;

        // 3. Atomically replace destination file
        fs::rename(&temp_path, target_path).map_err(|e| {
            let _ = fs::remove_file(&temp_path);
            format!("Atomic rename failed for monotonic anchor: {e}")
        })?;

        // 4. Sync parent directory metadata
        crate::hardware_root::sync_directory_metadata(target_path)?;

        Ok(())
    }

    pub fn backing_path(&self) -> &Path {
        &self.backing_path
    }
}

impl MonotonicAnchor for DurableMonotonicAnchor {
    fn assurance_type(&self) -> AnchorAssuranceType {
        AnchorAssuranceType::LocalHostWitness
    }

    fn read_watermark(&self) -> Result<u64, String> {
        let val = self
            .current_watermark
            .lock()
            .map_err(|_| "Watermark lock poisoned")?;
        Ok(*val)
    }

    fn advance_watermark(&self, candidate: u64) -> Result<(), String> {
        let mut val = self
            .current_watermark
            .lock()
            .map_err(|_| "Watermark lock poisoned")?;
        if candidate <= *val {
            return Err(format!(
                "RollbackDetected[{}]: Candidate watermark {} is not strictly greater than committed watermark {}",
                self.anchor_name, candidate, *val
            ));
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let record = MonotonicDiskRecord::new_record(&self.anchor_name, candidate, now);

        // Atomic persistent commit
        Self::atomic_write_record(&self.backing_path, &record)?;

        *val = candidate;
        Ok(())
    }
}

/// Local Host Witness Monotonic Anchor.
/// Supports both in-memory and durable modes.
pub struct LocalHostMonotonicAnchor {
    inner: Box<dyn MonotonicAnchor>,
}

impl LocalHostMonotonicAnchor {
    pub fn new(anchor_name: impl Into<String>, initial_val: u64) -> Self {
        struct InMemoryAnchor {
            name: String,
            val: Arc<Mutex<u64>>,
        }
        impl MonotonicAnchor for InMemoryAnchor {
            fn assurance_type(&self) -> AnchorAssuranceType {
                AnchorAssuranceType::LocalHostWitness
            }
            fn read_watermark(&self) -> Result<u64, String> {
                let v = self.val.lock().map_err(|_| "Watermark lock poisoned")?;
                Ok(*v)
            }
            fn advance_watermark(&self, candidate: u64) -> Result<(), String> {
                let mut v = self.val.lock().map_err(|_| "Watermark lock poisoned")?;
                if candidate <= *v {
                    return Err(format!(
                        "RollbackDetected[{}]: Candidate watermark {} is not strictly greater than committed watermark {}",
                        self.name, candidate, *v
                    ));
                }
                *v = candidate;
                Ok(())
            }
        }

        Self {
            inner: Box::new(InMemoryAnchor {
                name: anchor_name.into(),
                val: Arc::new(Mutex::new(initial_val)),
            }),
        }
    }

    /// Creates a persistent, crash-safe host anchor backed by disk.
    pub fn new_persistent<P: AsRef<Path>>(
        path: P,
        anchor_name: impl Into<String>,
        initial_val: u64,
    ) -> Result<Self, String> {
        let anchor = DurableMonotonicAnchor::open_or_create(path, anchor_name, initial_val)?;
        Ok(Self {
            inner: Box::new(anchor),
        })
    }
}

impl MonotonicAnchor for LocalHostMonotonicAnchor {
    fn assurance_type(&self) -> AnchorAssuranceType {
        self.inner.assurance_type()
    }

    fn read_watermark(&self) -> Result<u64, String> {
        self.inner.read_watermark()
    }

    fn advance_watermark(&self, candidate: u64) -> Result<(), String> {
        self.inner.advance_watermark(candidate)
    }
}

/// Rollback Guard enforcing policy requirements on required anchor assurance.
pub struct RollbackGuard {
    anchor: Box<dyn MonotonicAnchor>,
    min_required_assurance: AnchorAssuranceType,
}

impl RollbackGuard {
    pub fn new(anchor: Box<dyn MonotonicAnchor>, min_assurance: AnchorAssuranceType) -> Self {
        Self {
            anchor,
            min_required_assurance: min_assurance,
        }
    }

    /// Evaluates candidate version counter against monotonic watermark.
    pub fn verify_and_advance(&self, candidate_version: u64) -> Result<(), String> {
        if self.anchor.assurance_type() == AnchorAssuranceType::Unavailable {
            return Err("RollbackGuardViolation: Required anti-rollback anchor is unavailable on this platform".into());
        }

        if self.min_required_assurance == AnchorAssuranceType::HardwareMonotonic
            && self.anchor.assurance_type() != AnchorAssuranceType::HardwareMonotonic
        {
            return Err("RollbackGuardPolicyViolation: Security policy mandates HardwareMonotonic anchor, but platform provides only LocalHostWitness".into());
        }

        self.anchor.advance_watermark(candidate_version)
    }

    pub fn current_watermark(&self) -> Result<u64, String> {
        self.anchor.read_watermark()
    }

    pub fn anchor_assurance(&self) -> AnchorAssuranceType {
        self.anchor.assurance_type()
    }
}
