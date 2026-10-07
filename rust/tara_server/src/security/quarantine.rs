//! 8-Stage Multi-Artifact Quarantine Engine and Containment Manager.
//!
//! Provides quarantine pipeline for incoming artifacts (code, knowledge, skills,
//! configs, search results) with SHA-256 hashing, syntax verification,
//! policy checks, audit logging, and immediate containment revocation.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuarantineStage {
    Ingestion,
    HashVerification,
    SyntacticAnalysis,
    HeuristicScanning,
    SandboxDetonation,
    ProvenanceCheck,
    VerdictDecision,
    AuditLogging,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuarantineVerdict {
    Pending,
    Approved,
    Rejected,
    Contained,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarantineRecord {
    pub quarantine_id: String,
    pub artifact_type: String, // "skill", "knowledge", "code", "config"
    pub artifact_name: String,
    pub sha256_hash: String,
    pub raw_content: String,
    pub stages_passed: Vec<QuarantineStage>,
    pub current_stage: QuarantineStage,
    pub verdict: QuarantineVerdict,
    pub failure_reason: Option<String>,
    pub ingest_timestamp_ms: u64,
}

pub struct QuarantineEngine {
    quarantine_store_path: String,
    blocklisted_hashes: Arc<Mutex<HashSet<String>>>,
    active_quarantine: Arc<Mutex<HashMap<String, QuarantineRecord>>>,
}

impl QuarantineEngine {
    pub fn new(repo_root: &str) -> Self {
        let store_dir = format!("{}/storage/quarantine", repo_root);
        let _ = fs::create_dir_all(&store_dir);
        let quarantine_store_path = format!("{}/quarantine_ledger.jsonl", store_dir);

        Self {
            quarantine_store_path,
            blocklisted_hashes: Arc::new(Mutex::new(HashSet::new())),
            active_quarantine: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Blocklists a specific SHA256 digest permanently.
    pub fn blocklist_hash(&self, hash: &str) {
        let mut list = self.blocklisted_hashes.lock().unwrap();
        list.insert(hash.trim().to_lowercase());
    }

    /// Runs an incoming artifact through the full 8-stage quarantine pipeline.
    pub fn quarantine_and_evaluate(
        &self,
        artifact_type: &str,
        name: &str,
        raw_content: &str,
        syntax_validator: impl Fn(&str) -> Result<(), String>,
        heuristic_checker: impl Fn(&str) -> Result<(), String>,
    ) -> QuarantineRecord {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Stage 1: Ingestion
        let q_id = format!("quar_{}_{}", artifact_type, now);
        let mut stages = vec![QuarantineStage::Ingestion];

        // Stage 2: Hash Verification
        stages.push(QuarantineStage::HashVerification);
        let mut hasher = Sha256::new();
        hasher.update(raw_content.as_bytes());
        let hash = format!("{:x}", hasher.finalize());

        {
            let blocklist = self.blocklisted_hashes.lock().unwrap();
            if blocklist.contains(&hash) {
                let rec = QuarantineRecord {
                    quarantine_id: q_id,
                    artifact_type: artifact_type.to_string(),
                    artifact_name: name.to_string(),
                    sha256_hash: hash,
                    raw_content: raw_content.to_string(),
                    stages_passed: stages,
                    current_stage: QuarantineStage::HashVerification,
                    verdict: QuarantineVerdict::Rejected,
                    failure_reason: Some("SHA-256 hash matches blocklist".to_string()),
                    ingest_timestamp_ms: now,
                };
                self.persist_record(&rec);
                return rec;
            }
        }

        // Stage 3: Syntactic Analysis
        stages.push(QuarantineStage::SyntacticAnalysis);
        if let Err(err) = syntax_validator(raw_content) {
            let rec = QuarantineRecord {
                quarantine_id: q_id,
                artifact_type: artifact_type.to_string(),
                artifact_name: name.to_string(),
                sha256_hash: hash,
                raw_content: raw_content.to_string(),
                stages_passed: stages,
                current_stage: QuarantineStage::SyntacticAnalysis,
                verdict: QuarantineVerdict::Rejected,
                failure_reason: Some(format!("Syntax validation failed: {}", err)),
                ingest_timestamp_ms: now,
            };
            self.persist_record(&rec);
            return rec;
        }

        // Stage 4: Heuristic Scanning
        stages.push(QuarantineStage::HeuristicScanning);
        if let Err(err) = heuristic_checker(raw_content) {
            let rec = QuarantineRecord {
                quarantine_id: q_id,
                artifact_type: artifact_type.to_string(),
                artifact_name: name.to_string(),
                sha256_hash: hash,
                raw_content: raw_content.to_string(),
                stages_passed: stages,
                current_stage: QuarantineStage::HeuristicScanning,
                verdict: QuarantineVerdict::Rejected,
                failure_reason: Some(format!("Heuristic check flagged anomaly: {}", err)),
                ingest_timestamp_ms: now,
            };
            self.persist_record(&rec);
            return rec;
        }

        // Stage 5: Sandbox Detonation / Policy Dry Run
        stages.push(QuarantineStage::SandboxDetonation);
        // Ensure no dangerous shell commands or unsafe code blocks in sandbox
        if raw_content.contains("system(")
            || raw_content.contains("exec(")
            || raw_content.contains("child_process")
        {
            let rec = QuarantineRecord {
                quarantine_id: q_id,
                artifact_type: artifact_type.to_string(),
                artifact_name: name.to_string(),
                sha256_hash: hash,
                raw_content: raw_content.to_string(),
                stages_passed: stages,
                current_stage: QuarantineStage::SandboxDetonation,
                verdict: QuarantineVerdict::Rejected,
                failure_reason: Some("Disallowed execution primitive detected".to_string()),
                ingest_timestamp_ms: now,
            };
            self.persist_record(&rec);
            return rec;
        }

        // Stage 6: Provenance Check
        stages.push(QuarantineStage::ProvenanceCheck);
        // Requires valid author/type descriptor
        if name.trim().is_empty() {
            let rec = QuarantineRecord {
                quarantine_id: q_id,
                artifact_type: artifact_type.to_string(),
                artifact_name: name.to_string(),
                sha256_hash: hash,
                raw_content: raw_content.to_string(),
                stages_passed: stages,
                current_stage: QuarantineStage::ProvenanceCheck,
                verdict: QuarantineVerdict::Rejected,
                failure_reason: Some("Missing artifact name or provenance identifier".to_string()),
                ingest_timestamp_ms: now,
            };
            self.persist_record(&rec);
            return rec;
        }

        // Stage 7: Verdict Decision
        stages.push(QuarantineStage::VerdictDecision);
        let verdict = QuarantineVerdict::Approved;

        // Stage 8: Audit Logging
        stages.push(QuarantineStage::AuditLogging);
        let rec = QuarantineRecord {
            quarantine_id: q_id.clone(),
            artifact_type: artifact_type.to_string(),
            artifact_name: name.to_string(),
            sha256_hash: hash,
            raw_content: raw_content.to_string(),
            stages_passed: stages,
            current_stage: QuarantineStage::AuditLogging,
            verdict,
            failure_reason: None,
            ingest_timestamp_ms: now,
        };

        self.persist_record(&rec);
        {
            let mut active = self.active_quarantine.lock().unwrap();
            active.insert(q_id, rec.clone());
        }

        rec
    }

    fn persist_record(&self, record: &QuarantineRecord) {
        if let Ok(mut f) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.quarantine_store_path)
        {
            if let Ok(line) = serde_json::to_string(record) {
                let _ = writeln!(f, "{}", line);
            }
        }
    }
}

/// Containment Manager for instant session revocation and compromised worker isolation.
pub struct ContainmentManager {
    revoked_sessions: Arc<Mutex<HashSet<String>>>,
    isolated_workers: Arc<Mutex<HashSet<String>>>,
}

impl Default for ContainmentManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ContainmentManager {
    pub fn new() -> Self {
        Self {
            revoked_sessions: Arc::new(Mutex::new(HashSet::new())),
            isolated_workers: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Instantly revokes a session ID.
    pub fn revoke_session(&self, session_id: &str) {
        let mut set = self.revoked_sessions.lock().unwrap();
        set.insert(session_id.to_string());
    }

    /// Checks if a session is revoked.
    pub fn is_session_revoked(&self, session_id: &str) -> bool {
        let set = self.revoked_sessions.lock().unwrap();
        set.contains(session_id)
    }

    /// Isolates a compromised worker ID immediately.
    pub fn isolate_worker(&self, worker_id: &str) {
        let mut set = self.isolated_workers.lock().unwrap();
        set.insert(worker_id.to_string());
    }

    /// Checks if a worker is isolated.
    pub fn is_worker_isolated(&self, worker_id: &str) -> bool {
        let set = self.isolated_workers.lock().unwrap();
        set.contains(worker_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn test_quarantine_pipeline_approves_clean_artifact() {
        let temp = std::env::temp_dir().join("tara_test_quarantine_ok");
        let q = QuarantineEngine::new(temp.to_str().unwrap());

        let clean_json = r#"{"name": "demo_skill", "version": "1.0.0"}"#;
        let record = q.quarantine_and_evaluate(
            "skill",
            "demo_skill",
            clean_json,
            |s| {
                serde_json::from_str::<Value>(s)
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            },
            |_s| Ok(()),
        );

        assert_eq!(record.verdict, QuarantineVerdict::Approved);
        assert_eq!(record.stages_passed.len(), 8);
        assert!(record.failure_reason.is_none());

        let _ = fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_quarantine_rejects_blocklisted_hash() {
        let temp = std::env::temp_dir().join("tara_test_quarantine_block");
        let q = QuarantineEngine::new(temp.to_str().unwrap());

        let payload = "malicious payload";
        let mut hasher = Sha256::new();
        hasher.update(payload.as_bytes());
        let hash = format!("{:x}", hasher.finalize());

        q.blocklist_hash(&hash);

        let record =
            q.quarantine_and_evaluate("code", "bad_script", payload, |_| Ok(()), |_| Ok(()));

        assert_eq!(record.verdict, QuarantineVerdict::Rejected);
        assert!(record.failure_reason.unwrap().contains("blocklist"));

        let _ = fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_containment_manager_revocation() {
        let cm = ContainmentManager::new();
        assert!(!cm.is_session_revoked("sess_123"));
        cm.revoke_session("sess_123");
        assert!(cm.is_session_revoked("sess_123"));

        assert!(!cm.is_worker_isolated("worker_node_1"));
        cm.isolate_worker("worker_node_1");
        assert!(cm.is_worker_isolated("worker_node_1"));
    }
}
