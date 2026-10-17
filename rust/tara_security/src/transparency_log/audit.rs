//! Authenticated, Hash-Chained Audit Logging Subsystem.
//!
//! Rule compliance:
//! - Records schema version, sequence number, actor, action, resource, result, generation, and epoch.
//! - Cryptographically chained HMAC-SHA256 signatures detect tampering and truncation.
//! - Operations FAIL CLOSED if mandatory audit persistence fails.
//! - Canonical, unambiguous JSON serialization for preimage computation (ID-13).
//! - Chain verification method (`verify_chain`) validates complete history integrity (ID-13).
//! - Zeroization of memory key material on drop (ID-14).
//! - Crash-recovery allows restoring state from durable sink (ID-09).

use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use zeroize::Zeroizing;

use crate::cryptography::CryptoUtils;
use crate::primitives::{
    AuditSequence, AuthorityGeneration, PrincipalId, SecurityAction, SecurityResource,
};

pub const GENESIS_PREVIOUS_HASH: &str =
    "GENESIS_BLOCK_0000000000000000000000000000000000000000000000000000000000000000";

/// An authenticated audit record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    pub schema_version: u32,
    pub sequence: AuditSequence,
    pub authority_generation: AuthorityGeneration,
    pub actor: PrincipalId,
    pub action: SecurityAction,
    pub resource: SecurityResource,
    pub result: String, // "SUCCESS" or "DENIED" or error details
    pub timestamp_epoch_secs: u64,
    pub previous_record_hash: String,
    pub record_hmac_signature: String,
}

/// Canonical preimage input for HMAC calculation and integrity verification (ID-13).
#[derive(Debug, Clone, Serialize)]
pub struct AuditPreimageInput<'a> {
    pub schema_version: u32,
    pub sequence: u64,
    pub authority_generation: u64,
    pub actor: &'a str,
    pub action: &'a str,
    pub resource: &'a str,
    pub result: &'a str,
    pub timestamp_epoch_secs: u64,
    pub previous_record_hash: &'a str,
}

impl<'a> AuditPreimageInput<'a> {
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(self).map_err(|e| format!("Failed to serialize canonical preimage: {e}"))
    }
}

impl AuditRecord {
    /// Compute canonical preimage bytes for HMAC calculation and verification.
    pub fn canonical_preimage(&self) -> Result<Vec<u8>, String> {
        let actor_str = self.actor.as_str();
        let action_str = self.action.canonical_name();
        let resource_str = self.resource.resource_urn();
        let input = AuditPreimageInput {
            schema_version: self.schema_version,
            sequence: self.sequence.get(),
            authority_generation: self.authority_generation.get(),
            actor: &actor_str,
            action: &action_str,
            resource: &resource_str,
            result: &self.result,
            timestamp_epoch_secs: self.timestamp_epoch_secs,
            previous_record_hash: &self.previous_record_hash,
        };
        input.to_canonical_bytes()
    }
}

/// Pluggable sink for audit persistence.
pub trait AuditSink: Send + Sync {
    fn append_record(&self, record: &AuditRecord) -> Result<(), String>;
    fn read_all_records(&self) -> Result<Vec<AuditRecord>, String>;
}

/// In-memory / transactional Audit Sink.
pub struct MemoryAuditSink {
    records: Arc<Mutex<Vec<AuditRecord>>>,
}

impl Default for MemoryAuditSink {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryAuditSink {
    pub fn new() -> Self {
        Self {
            records: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn records(&self) -> Vec<AuditRecord> {
        self.records
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }
}

impl AuditSink for MemoryAuditSink {
    fn append_record(&self, record: &AuditRecord) -> Result<(), String> {
        let mut list = self
            .records
            .lock()
            .map_err(|_| "Audit sink lock poisoned (FailClosed)")?;
        list.push(record.clone());
        Ok(())
    }

    fn read_all_records(&self) -> Result<Vec<AuditRecord>, String> {
        Ok(self.records())
    }
}

/// Durable append-only crash-resilient file audit sink.
pub struct FileAuditSink {
    path: std::path::PathBuf,
    lock: Mutex<()>,
}

impl FileAuditSink {
    pub fn new<P: AsRef<std::path::Path>>(path: P) -> Result<Self, String> {
        let p = path.as_ref().to_path_buf();
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create audit dir: {e}"))?;
        }
        Ok(Self {
            path: p,
            lock: Mutex::new(()),
        })
    }
}

impl AuditSink for FileAuditSink {
    fn append_record(&self, record: &AuditRecord) -> Result<(), String> {
        let _guard = self.lock.lock().map_err(|_| "Audit file lock poisoned")?;
        let line = serde_json::to_string(record)
            .map_err(|e| format!("AuditRecord serialization error: {e}"))?;
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| format!("Failed to open audit log for append: {e}"))?;
        writeln!(file, "{}", line).map_err(|e| format!("Failed to write audit record: {e}"))?;
        file.sync_data()
            .map_err(|e| format!("Failed to fsync audit record: {e}"))?;
        Ok(())
    }

    fn read_all_records(&self) -> Result<Vec<AuditRecord>, String> {
        let _guard = self.lock.lock().map_err(|_| "Audit file lock poisoned")?;
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        use std::io::BufRead;
        let file = std::fs::File::open(&self.path)
            .map_err(|e| format!("Failed to open audit log for reading: {e}"))?;
        let reader = std::io::BufReader::new(file);
        let mut records = Vec::new();
        for (line_no, line_res) in reader.lines().enumerate() {
            let line = line_res.map_err(|e| format!("IO error reading line {line_no}: {e}"))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let rec = serde_json::from_str::<AuditRecord>(trimmed)
                .map_err(|e| format!("Failed to parse audit record on line {line_no}: {e}"))?;
            records.push(rec);
        }
        Ok(records)
    }
}

/// Authenticated Audit Logger managing hash-chain continuity and cryptographic signatures.
pub struct AuthenticatedAuditLogger {
    current_sequence: Arc<Mutex<AuditSequence>>,
    last_record_hash: Arc<Mutex<String>>,
    audit_hmac_key: Zeroizing<[u8; 32]>,
    sink: Box<dyn AuditSink>,
}

impl AuthenticatedAuditLogger {
    pub fn new(hmac_key: [u8; 32], sink: Box<dyn AuditSink>) -> Self {
        Self {
            current_sequence: Arc::new(Mutex::new(AuditSequence::INITIAL)),
            last_record_hash: Arc::new(Mutex::new(GENESIS_PREVIOUS_HASH.to_string())),
            audit_hmac_key: Zeroizing::new(hmac_key),
            sink,
        }
    }

    /// Recover sequence and previous hash state from durable sink records (ID-09).
    pub fn recover_from_sink(&self) -> Result<(), String> {
        let records = self.sink.read_all_records()?;
        if !records.is_empty() {
            Self::verify_chain(&records, &self.audit_hmac_key[..]).map_err(|e| {
                format!(
                    "Audit log recovery failed: hash chain verification failed (FailClosed): {e}"
                )
            })?;
            let last_rec = records.last().unwrap();
            let next_seq = last_rec.sequence.next()?;
            let current_hash = CryptoUtils::sha256_hex(last_rec.record_hmac_signature.as_bytes());

            let mut seq_guard = self
                .current_sequence
                .lock()
                .map_err(|_| "Sequence lock poisoned")?;
            let mut hash_guard = self
                .last_record_hash
                .lock()
                .map_err(|_| "Hash lock poisoned")?;

            *seq_guard = next_seq;
            *hash_guard = current_hash;
        }
        Ok(())
    }

    /// Record an authenticated audit entry.
    /// Strictly FAILS CLOSED if sink append fails.
    pub fn log_event(
        &self,
        actor: PrincipalId,
        action: SecurityAction,
        resource: SecurityResource,
        result: &str,
        generation: AuthorityGeneration,
        epoch_secs: u64,
    ) -> Result<AuditRecord, String> {
        let mut seq_guard = self
            .current_sequence
            .lock()
            .map_err(|_| "Sequence lock poisoned (FailClosed)")?;
        let mut prev_hash_guard = self
            .last_record_hash
            .lock()
            .map_err(|_| "Hash lock poisoned (FailClosed)")?;

        let seq = *seq_guard;
        let prev_hash = prev_hash_guard.clone();

        let actor_str = actor.as_str();
        let action_str = action.canonical_name();
        let resource_str = resource.resource_urn();
        let preimage_input = AuditPreimageInput {
            schema_version: 1,
            sequence: seq.get(),
            authority_generation: generation.get(),
            actor: &actor_str,
            action: &action_str,
            resource: &resource_str,
            result,
            timestamp_epoch_secs: epoch_secs,
            previous_record_hash: &prev_hash,
        };
        let preimage_bytes = preimage_input.to_canonical_bytes()?;

        let record_hmac = CryptoUtils::hmac_sha256(&*self.audit_hmac_key, &preimage_bytes)?;
        let record_hmac_hex = hex::encode(record_hmac);
        let current_hash = CryptoUtils::sha256_hex(record_hmac_hex.as_bytes());

        let record = AuditRecord {
            schema_version: 1,
            sequence: seq,
            authority_generation: generation,
            actor,
            action,
            resource,
            result: result.to_string(),
            timestamp_epoch_secs: epoch_secs,
            previous_record_hash: prev_hash,
            record_hmac_signature: record_hmac_hex,
        };

        // 1. Append to sink FIRST (ID-09)
        self.sink.append_record(&record)?;

        // 2. Only advance memory state once sink commit succeeds!
        *seq_guard = seq.next()?;
        *prev_hash_guard = current_hash;

        Ok(record)
    }

    /// Verifies the cryptographic integrity and unbroken continuity of an entire audit log chain (ID-13).
    pub fn verify_chain(records: &[AuditRecord], hmac_key: &[u8]) -> Result<(), String> {
        if records.is_empty() {
            return Ok(());
        }

        let mut expected_prev_hash = GENESIS_PREVIOUS_HASH.to_string();

        for (idx, record) in records.iter().enumerate() {
            let expected_seq = AuditSequence::INITIAL.get() + (idx as u64);
            // 1. Verify sequence numbers are strictly contiguous
            if record.sequence.get() != expected_seq {
                return Err(format!(
                    "AuditChainIntegrityFailure: Sequence gap at record {}: expected {}, found {}",
                    idx,
                    expected_seq,
                    record.sequence.get()
                ));
            }

            // 2. Verify previous hash chaining
            if !CryptoUtils::constant_time_eq(
                record.previous_record_hash.as_bytes(),
                expected_prev_hash.as_bytes(),
            ) {
                return Err(format!(
                    "AuditChainIntegrityFailure: Hash chain broken at record {}: expected prev {}, found {}",
                    idx, expected_prev_hash, record.previous_record_hash
                ));
            }

            // 3. Recompute canonical preimage and verify HMAC signature
            let preimage = record.canonical_preimage()?;

            let expected_hmac = CryptoUtils::hmac_sha256(hmac_key, &preimage)?;
            let provided_hmac = hex::decode(&record.record_hmac_signature)
                .map_err(|e| format!("Malformed HMAC hex at record {}: {e}", idx))?;

            if !CryptoUtils::constant_time_eq(&expected_hmac, &provided_hmac) {
                return Err(format!(
                    "AuditChainIntegrityFailure: Invalid HMAC signature on record {} (Tampering Detected)",
                    idx
                ));
            }

            // 4. Advance expected values for next link in chain
            expected_prev_hash = CryptoUtils::sha256_hex(record.record_hmac_signature.as_bytes());
        }

        Ok(())
    }
}
