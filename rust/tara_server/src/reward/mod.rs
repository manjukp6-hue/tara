//! Canonical Encrypted & Tamper-Evident Reward System for TARA AI.
//!
//! Enforces:
//! - Exact mathematical reward tracking across 10 canonical reward dimensions:
//!   TaskCompletion, LearningProgress, SuccessfulReasoning, SuccessfulResearch,
//!   ValidatedCapability, UserApproved, Consistency, SafetyAdherence,
//!   ContinuousImprovement, LongTermGoal.
//! - Cryptographic hash-chaining of all reward events with exact binary IEEE-754 precision.
//! - Authenticated AES-256-GCM encryption at rest with independent, secure key derivation.
//! - Single mutex transaction lock preventing lock-order deadlocks and state divergence.
//! - Atomic, durable temp-file + fsync + rename disk persistence before memory mutation.
//! - Creator-authorized operations with strongly-typed `AuthenticatedPrincipal`.
//! - Fail-closed initialization: corrupted or tampered ledgers prevent normal operation.

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;
use zeroize::Zeroize;
use sha2::{Digest, Sha256};

pub const GENESIS_REWARD_HASH: &str = "GENESIS_REWARD_ROOT_HASH";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RewardCategory {
    TaskCompletion,
    LearningProgress,
    SuccessfulReasoning,
    SuccessfulResearch,
    ValidatedCapability,
    UserApproved,
    Consistency,
    SafetyAdherence,
    ContinuousImprovement,
    LongTermGoal,
}

impl RewardCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TaskCompletion => "task_completion",
            Self::LearningProgress => "learning_progress",
            Self::SuccessfulReasoning => "successful_reasoning",
            Self::SuccessfulResearch => "successful_research",
            Self::ValidatedCapability => "validated_capability",
            Self::UserApproved => "user_approved",
            Self::Consistency => "consistency",
            Self::SafetyAdherence => "safety_adherence",
            Self::ContinuousImprovement => "continuous_improvement",
            Self::LongTermGoal => "long_term_goal",
        }
    }

    pub fn parse_category(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

impl std::str::FromStr for RewardCategory {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "task_completion" | "task" => Ok(Self::TaskCompletion),
            "learning_progress" | "learning" => Ok(Self::LearningProgress),
            "successful_reasoning" | "reasoning" => Ok(Self::SuccessfulReasoning),
            "successful_research" | "research" => Ok(Self::SuccessfulResearch),
            "validated_capability" | "capability" => Ok(Self::ValidatedCapability),
            "user_approved" | "approval" => Ok(Self::UserApproved),
            "consistency" => Ok(Self::Consistency),
            "safety_adherence" | "safety" => Ok(Self::SafetyAdherence),
            "continuous_improvement" | "improvement" => Ok(Self::ContinuousImprovement),
            "long_term_goal" | "goal" => Ok(Self::LongTermGoal),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RewardEvent {
    pub event_id: String,
    pub timestamp_ms: u64,
    pub actor_id: String,
    pub category: RewardCategory,
    pub delta: f64,
    pub reason: String,
    pub prev_hash: String,
    pub signature_hash: String,
}

impl RewardEvent {
    /// Compute exact canonical hash over all event fields.
    /// Uses exact binary IEEE-754 bytes for `delta` to prevent rounding collisions.
    pub fn compute_hash(
        prev_hash: &str,
        event_id: &str,
        timestamp_ms: u64,
        actor_id: &str,
        category: RewardCategory,
        delta: f64,
        reason: &str,
    ) -> String {
        let mut payload = Vec::new();
        payload.extend_from_slice(prev_hash.as_bytes());
        payload.push(b'|');
        payload.extend_from_slice(event_id.as_bytes());
        payload.push(b'|');
        payload.extend_from_slice(timestamp_ms.to_string().as_bytes());
        payload.push(b'|');
        payload.extend_from_slice(actor_id.as_bytes());
        payload.push(b'|');
        payload.extend_from_slice(category.as_str().as_bytes());
        payload.push(b'|');
        payload.extend_from_slice(&delta.to_bits().to_le_bytes());
        payload.push(b'|');
        payload.extend_from_slice(reason.as_bytes());
        tara_security::cryptography::CryptoUtils::compute256_bytes_hex(&payload)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardState {
    pub total_score: f64,
    pub category_scores: HashMap<String, f64>,
    pub events_count: usize,
    pub last_event_hash: String,
    pub last_updated_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardLedger {
    pub events: Vec<RewardEvent>,
    pub state: RewardState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LedgerStatus {
    Valid,
    Corrupt(String),
}

pub struct RewardSystem {
    storage_path: PathBuf,
    ledger: Mutex<RewardLedger>,
    key: [u8; 32],
    status: Mutex<LedgerStatus>,
}

impl Drop for RewardSystem {
    fn drop(&mut self) {
        self.key.zeroize();
    }
}

impl RewardSystem {
    pub fn watermark_domain(&self) -> String {
        format!(
            "reward_ledger_{}",
            &hex::encode(Sha256::digest(self.storage_path.to_string_lossy().as_bytes()))[..16]
        )
    }

    /// Try initializing the reward system rooted at `repo_root`.
    pub fn try_new<P: AsRef<Path>>(repo_root: P) -> Result<Self, String> {
        let storage_path = repo_root
            .as_ref()
            .join("storage")
            .join("vault")
            .join("reward");
        let _ = fs::create_dir_all(&storage_path);

        let salt_file = storage_path.join("salt.bin");
        let mut salt = [0u8; 16];
        if salt_file.exists()
            && fs::metadata(&salt_file)
                .map(|m| m.len() == 16)
                .unwrap_or(false)
        {
            let mut f = File::open(&salt_file).map_err(|e| format!("Failed to read salt: {e}"))?;
            f.read_exact(&mut salt)
                .map_err(|e| format!("Corrupt salt file: {e}"))?;
        } else {
            rand::thread_rng().fill_bytes(&mut salt);
            let mut f =
                File::create(&salt_file).map_err(|e| format!("Failed to create salt file: {e}"))?;
            f.write_all(&salt)
                .map_err(|e| format!("Failed to write salt: {e}"))?;
            f.sync_all()
                .map_err(|e| format!("Failed to sync salt file: {e}"))?;
            let _ = tara_security::hardware_root::harden_file_permissions(&salt_file);
        }

        let key = Self::resolve_or_generate_key(&storage_path, &salt)?;

        let initial_state = RewardState {
            total_score: 0.0,
            category_scores: HashMap::new(),
            events_count: 0,
            last_event_hash: GENESIS_REWARD_HASH.to_string(),
            last_updated_ms: 0,
        };

        let sys = Self {
            storage_path,
            ledger: Mutex::new(RewardLedger {
                events: Vec::new(),
                state: initial_state,
            }),
            key,
            status: Mutex::new(LedgerStatus::Valid),
        };

        let enc_file = sys.storage_path.join("reward_ledger.enc");
        let domain = sys.watermark_domain();
        let watermark = tara_security::dpapi::load_os_monotonic_watermark(&domain).unwrap_or(None);

        if !enc_file.exists() {
            if let Some(wm) = watermark {
                if wm > 0 {
                    *sys.status.lock().unwrap_or_else(|p| p.into_inner()) =
                        LedgerStatus::Corrupt(format!(
                            "AntiRollbackViolation: Encrypted reward ledger missing while non-zero watermark ({wm}) exists"
                        ));
                    return Err(format!(
                        "FailClosed: Missing reward ledger with active watermark ({wm})"
                    ));
                }
            }
        } else {
            match sys.load_and_verify_encrypted() {
                Ok(loaded_ledger) => {
                    *sys.ledger.lock().unwrap_or_else(|p| p.into_inner()) = loaded_ledger;
                }
                Err(e) => {
                    *sys.status.lock().unwrap_or_else(|p| p.into_inner()) =
                        LedgerStatus::Corrupt(e.clone());
                    return Err(format!(
                        "FailClosed: Reward ledger verification failed on startup: {e}"
                    ));
                }
            }
        }

        Ok(sys)
    }

    /// Primary constructor for RewardSystem. Fails closed if encrypted ledger exists and is corrupt.
    pub fn new<P: AsRef<Path>>(repo_root: P) -> Self {
        match Self::try_new(repo_root) {
            Ok(sys) => sys,
            Err(e) => {
                // If corrupted or tampered, construct locked-down fail-closed instance
                let storage_path = PathBuf::from("storage/vault/reward");
                Self {
                    storage_path,
                    ledger: Mutex::new(RewardLedger {
                        events: Vec::new(),
                        state: RewardState {
                            total_score: 0.0,
                            category_scores: HashMap::new(),
                            events_count: 0,
                            last_event_hash: GENESIS_REWARD_HASH.to_string(),
                            last_updated_ms: 0,
                        },
                    }),
                    key: [0u8; 32],
                    status: Mutex::new(LedgerStatus::Corrupt(e)),
                }
            }
        }
    }

    /// Resolve cryptographic key from `TARA_REWARD_KEY` environment variable,
    /// legacy `key.bin`, or derive from DPAPI authority key combined with salt.
    fn resolve_or_generate_key(storage_path: &Path, salt: &[u8; 16]) -> Result<[u8; 32], String> {
        if let Ok(env_secret) = std::env::var("TARA_REWARD_KEY") {
            if !env_secret.trim().is_empty() {
                const SCRYPT_LOG_N: u8 = 14;
                const SCRYPT_R: u32 = 8;
                const SCRYPT_P: u32 = 1;
                const SCRYPT_KEY_LEN: usize = 32;

                let mut key = [0u8; 32];
                let params = scrypt::Params::new(SCRYPT_LOG_N, SCRYPT_R, SCRYPT_P, SCRYPT_KEY_LEN)
                    .map_err(|e| format!("Invalid scrypt params: {e}"))?;
                scrypt::scrypt(env_secret.as_bytes(), salt, &params, &mut key)
                    .map_err(|e| format!("Scrypt key derivation failed: {e:?}"))?;
                return Ok(key);
            }
        }

        // Dedicated local secure key file (backward compatibility)
        let key_file = storage_path.join("key.bin");
        if key_file.exists() {
            let mut f = File::open(&key_file)
                .map_err(|e| format!("Failed to open reward key file: {e}"))?;
            let mut key = [0u8; 32];
            f.read_exact(&mut key)
                .map_err(|e| format!("Corrupt reward key file: {e}"))?;
            let _ = tara_security::hardware_root::harden_file_permissions(&key_file);
            return Ok(key);
        }

        let authority_key = tara_security::dpapi::derive_authority_hmac_key("reward_ledger_vault")
            .map_err(|e| format!("Failed to derive authority HMAC key: {e}"))?;

        const SCRYPT_LOG_N: u8 = 14;
        const SCRYPT_R: u32 = 8;
        const SCRYPT_P: u32 = 1;
        const SCRYPT_KEY_LEN: usize = 32;

        let mut key = [0u8; 32];
        let params = scrypt::Params::new(SCRYPT_LOG_N, SCRYPT_R, SCRYPT_P, SCRYPT_KEY_LEN)
            .map_err(|e| format!("Invalid scrypt params: {e}"))?;
        scrypt::scrypt(&authority_key, salt, &params, &mut key)
            .map_err(|e| format!("Scrypt key derivation failed: {e:?}"))?;

        Ok(key)
    }

    /// Record a validated reward event into the tamper-evident ledger.
    /// Transactional: writes and syncs to disk before mutating in-memory state.
    pub fn record_reward(
        &self,
        actor_id: &str,
        category: RewardCategory,
        delta: f64,
        reason: &str,
    ) -> Result<RewardEvent, String> {
        // Enforce ledger status
        {
            let status = self.status.lock().unwrap_or_else(|p| p.into_inner());
            if let LedgerStatus::Corrupt(ref err) = *status {
                return Err(format!(
                    "LedgerDisabled: Reward system is in a fail-closed corrupt state: {err}"
                ));
            }
        }

        if !delta.is_finite() || delta.abs() > 1000.0 {
            return Err(
                "InvalidRewardDelta: delta must be finite and within magnitude 1000.0".into(),
            );
        }
        if actor_id.trim().is_empty() {
            return Err("InvalidActorId: actor_id cannot be empty".into());
        }

        let mut ledger = self.ledger.lock().unwrap_or_else(|p| p.into_inner());

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let event_id = format!(
            "rew_{}_{}_{}",
            ts,
            ledger.events.len(),
            &Uuid::new_v4().to_string()[..8]
        );
        let prev_hash = ledger.state.last_event_hash.clone();
        let sig_hash =
            RewardEvent::compute_hash(&prev_hash, &event_id, ts, actor_id, category, delta, reason);

        let event = RewardEvent {
            event_id,
            timestamp_ms: ts,
            actor_id: actor_id.to_string(),
            category,
            delta,
            reason: reason.to_string(),
            prev_hash,
            signature_hash: sig_hash.clone(),
        };

        // Construct candidate ledger
        let mut candidate_events = ledger.events.clone();
        candidate_events.push(event.clone());

        let mut candidate_state = ledger.state.clone();
        candidate_state.total_score += delta;
        *candidate_state
            .category_scores
            .entry(category.as_str().to_string())
            .or_insert(0.0) += delta;
        candidate_state.events_count += 1;
        candidate_state.last_event_hash = sig_hash;
        candidate_state.last_updated_ms = ts;

        // Persist candidate ledger durably to disk BEFORE mutating memory
        self.save_encrypted_transactional(&candidate_events)?;

        // Commit in-memory state
        ledger.events = candidate_events;
        ledger.state = candidate_state;

        Ok(event)
    }

    /// Creator-authorized manual reward adjustment with strongly-typed `AuthenticatedPrincipal`.
    pub fn manual_adjustment_with_principal(
        &self,
        principal: &tara_security::primitives::AuthenticatedPrincipal,
        actor_id: &str,
        category: RewardCategory,
        delta: f64,
        justification: &str,
    ) -> Result<RewardEvent, String> {
        if !principal.is_creator() {
            return Err(
                "Unauthorized: manual reward adjustments require validated Creator Authority"
                    .into(),
            );
        }
        let reason = format!("[CREATOR_ADJUSTMENT]: {}", justification);
        self.record_reward(actor_id, category, delta, &reason)
    }

    /// Deprecated backward-compatible manual reward adjustment.
    /// Strictly requires an authentic verified principal in production.
    pub fn manual_adjustment(
        &self,
        is_creator: bool,
        actor_id: &str,
        category: RewardCategory,
        delta: f64,
        justification: &str,
    ) -> Result<RewardEvent, String> {
        #[cfg(test)]
        {
            if is_creator {
                let principal = tara_security::primitives::AuthenticatedPrincipal::mint_creator_for_test("reward_test_session")
                    .map_err(|e| format!("Failed to mint test creator: {}", e))?;
                return self.manual_adjustment_with_principal(&principal, actor_id, category, delta, justification);
            }
        }
        let _ = is_creator;
        let principal = tara_security::primitives::AuthenticatedPrincipal::untrusted(actor_id);
        self.manual_adjustment_with_principal(&principal, actor_id, category, delta, justification)
    }

    /// Verify the complete cryptographic hash chain of the reward ledger.
    pub fn verify_ledger_integrity(&self) -> (bool, Option<String>) {
        {
            let status = self.status.lock().unwrap_or_else(|p| p.into_inner());
            if let LedgerStatus::Corrupt(ref err) = *status {
                return (false, Some(err.clone()));
            }
        }
        let ledger = self.ledger.lock().unwrap_or_else(|p| p.into_inner());
        Self::verify_events_integrity(&ledger.events)
    }

    fn verify_events_integrity(events: &[RewardEvent]) -> (bool, Option<String>) {
        if events.is_empty() {
            return (true, None);
        }

        let mut running_hash = GENESIS_REWARD_HASH.to_string();
        for (i, ev) in events.iter().enumerate() {
            if ev.prev_hash != running_hash {
                return (
                    false,
                    Some(format!(
                        "Ledger broken at index {}: expected prev_hash '{}', found '{}'",
                        i, running_hash, ev.prev_hash
                    )),
                );
            }
            let expected_sig = RewardEvent::compute_hash(
                &ev.prev_hash,
                &ev.event_id,
                ev.timestamp_ms,
                &ev.actor_id,
                ev.category,
                ev.delta,
                &ev.reason,
            );
            if ev.signature_hash != expected_sig {
                return (
                    false,
                    Some(format!(
                        "Ledger signature mismatch at index {}: event '{}' has been tampered with",
                        i, ev.event_id
                    )),
                );
            }
            running_hash = ev.signature_hash.clone();
        }

        (true, None)
    }

    /// Save ledger events encrypted with AES-256-GCM using unique temp-file + fsync + rename.
    fn save_encrypted_transactional(&self, events: &[RewardEvent]) -> Result<(), String> {
        let payload = serde_json::to_vec(events).map_err(|e| e.to_string())?;

        let cipher = Aes256Gcm::new_from_slice(&self.key).map_err(|e| e.to_string())?;

        let mut nonce_bytes = [0u8; 12];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher
            .encrypt(nonce, payload.as_ref())
            .map_err(|e| format!("encryption failed: {}", e))?;

        let temp_file = self
            .storage_path
            .join(format!("reward_ledger.enc.{}.tmp", Uuid::new_v4()));

        let write_and_rename = (|| -> Result<(), String> {
            let mut out = File::create(&temp_file).map_err(|e| e.to_string())?;
            out.write_all(&nonce_bytes).map_err(|e| e.to_string())?;
            out.write_all(&ciphertext).map_err(|e| e.to_string())?;
            out.sync_all().map_err(|e| e.to_string())?;
            drop(out);

            let enc_file = self.storage_path.join("reward_ledger.enc");
            fs::rename(&temp_file, &enc_file).map_err(|e| e.to_string())?;
            let _ = tara_security::hardware_root::harden_file_permissions(&enc_file);
            Ok(())
        })();

        if let Err(e) = write_and_rename {
            let _ = fs::remove_file(&temp_file);
            return Err(e);
        }

        let domain = self.watermark_domain();
        tara_security::dpapi::store_os_monotonic_watermark(&domain, events.len() as u64)?;

        Ok(())
    }

    /// Load and verify ledger events from AES-256-GCM encrypted file.
    fn load_and_verify_encrypted(&self) -> Result<RewardLedger, String> {
        let enc_file = self.storage_path.join("reward_ledger.enc");
        let domain = self.watermark_domain();
        let watermark = tara_security::dpapi::load_os_monotonic_watermark(&domain)
            .map_err(|e| format!("Failed to load watermark: {e}"))?;

        if !enc_file.exists() {
            if let Some(wm) = watermark {
                if wm > 0 {
                    return Err(format!(
                        "AntiRollbackViolation: Encrypted reward ledger missing while non-zero watermark ({wm}) exists"
                    ));
                }
            }
            return Ok(RewardLedger {
                events: Vec::new(),
                state: RewardState {
                    total_score: 0.0,
                    category_scores: HashMap::new(),
                    events_count: 0,
                    last_event_hash: GENESIS_REWARD_HASH.to_string(),
                    last_updated_ms: 0,
                },
            });
        }

        let mut f = File::open(&enc_file).map_err(|e| e.to_string())?;
        let mut data = Vec::new();
        f.read_to_end(&mut data).map_err(|e| e.to_string())?;

        if data.len() < 12 {
            return Err("corrupt encrypted reward ledger: file too short".into());
        }

        let (nonce_bytes, ciphertext) = data.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);

        let cipher = Aes256Gcm::new_from_slice(&self.key).map_err(|e| e.to_string())?;

        let plaintext = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| format!("reward ledger decryption failed: {e}"))?;

        let loaded_events: Vec<RewardEvent> =
            serde_json::from_slice(&plaintext).map_err(|e| format!("corrupt ledger JSON: {e}"))?;

        // Mandatory integrity check before live state installation
        let (valid, err) = Self::verify_events_integrity(&loaded_events);
        if !valid {
            return Err(format!(
                "Tampered reward ledger: hash chain break: {:?}",
                err
            ));
        }

        let event_count = loaded_events.len();
        if let Some(wm) = watermark {
            if (event_count as u64) < wm {
                return Err(format!(
                    "AntiRollbackViolation: Loaded ledger event count ({}) is less than monotonic watermark ({})",
                    event_count, wm
                ));
            }
        }

        // Reconstruct verified state
        let mut total = 0.0;
        let mut cat_scores: HashMap<String, f64> = HashMap::new();
        let last_hash = loaded_events
            .last()
            .map(|e| e.signature_hash.clone())
            .unwrap_or_else(|| GENESIS_REWARD_HASH.to_string());
        let last_ts = loaded_events.last().map(|e| e.timestamp_ms).unwrap_or(0);

        for ev in &loaded_events {
            total += ev.delta;
            *cat_scores
                .entry(ev.category.as_str().to_string())
                .or_insert(0.0) += ev.delta;
        }

        Ok(RewardLedger {
            events: loaded_events,
            state: RewardState {
                total_score: total,
                category_scores: cat_scores,
                events_count: event_count,
                last_event_hash: last_hash,
                last_updated_ms: last_ts,
            },
        })
    }

    /// Retrieve summary of current reward standings.
    pub fn get_summary(&self) -> Value {
        let (valid, err) = self.verify_ledger_integrity();
        let ledger = self.ledger.lock().unwrap_or_else(|p| p.into_inner());

        json!({
            "total_score": ledger.state.total_score,
            "category_scores": ledger.state.category_scores,
            "events_count": ledger.events.len(),
            "last_event_hash": ledger.state.last_event_hash,
            "last_updated_ms": ledger.state.last_updated_ms,
            "ledger_verified": valid,
            "integrity_error": err,
        })
    }

    /// Retrieve clone of all reward events for auditing.
    pub fn get_events(&self) -> Vec<RewardEvent> {
        let ledger = self.ledger.lock().unwrap_or_else(|p| p.into_inner());
        ledger.events.clone()
    }

    /// Compute category strategy weight bias.
    pub fn compute_category_bias(&self, category: RewardCategory) -> f64 {
        let ledger = self.ledger.lock().unwrap_or_else(|p| p.into_inner());
        let score = ledger
            .state
            .category_scores
            .get(category.as_str())
            .copied()
            .unwrap_or(0.0);

        1.0 + (score / 20.0).tanh()
    }

    /// Credits reward points requiring verified external oracle or creator.
    /// Rejects non-authoritative heuristic metrics and self-awarded points.
    pub fn credit_points(
        &self,
        principal: &tara_security::primitives::AuthenticatedPrincipal,
        points: f64,
        reason: &str,
    ) -> Result<RewardEvent, String> {
        if !principal.is_creator() && !principal.is_system_agent() {
            return Err(
                "AuthorizationDenied: Rewards require verified external oracle or creator".into(),
            );
        }
        if reason.to_lowercase().contains("heuristic")
            || reason.to_lowercase().contains("self_awarded")
        {
            return Err("RejectedHeuristicReward: Self-awarded telemetry points cannot confer reward credits".into());
        }
        self.record_reward(
            principal.actor_id(),
            RewardCategory::TaskCompletion,
            points,
            reason,
        )
    }
}

pub mod gated;
pub use gated::{
    ActiveTaskView, FailureRule, GatedGoalSystem, GatedTask, GoalHierarchySpec,
    GoalRewardDisburser, MilestoneSpec, RetryRule, ServerRewardDisburser, StageSpec,
    TaskSpec, TaskStatus, VerificationResult,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir() -> PathBuf {
        let p = std::env::temp_dir().join(format!("tara_rew_test_{}", Uuid::new_v4()));
        let _ = fs::create_dir_all(&p);
        p
    }

    #[test]
    fn test_reward_event_hash_chaining() {
        let tmp = test_dir();
        let reward_sys = RewardSystem::new(&tmp);

        let ev1 = reward_sys
            .record_reward(
                "agent_primary",
                RewardCategory::TaskCompletion,
                10.0,
                "solved calculus problem",
            )
            .unwrap();
        assert_eq!(ev1.prev_hash, GENESIS_REWARD_HASH);

        let ev2 = reward_sys
            .record_reward(
                "agent_primary",
                RewardCategory::LearningProgress,
                5.0,
                "mastered quadratic equations",
            )
            .unwrap();
        assert_eq!(ev2.prev_hash, ev1.signature_hash);

        let ev3 = reward_sys
            .record_reward(
                "agent_secondary",
                RewardCategory::SafetyAdherence,
                2.5,
                "clean security audit",
            )
            .unwrap();
        assert_eq!(ev3.prev_hash, ev2.signature_hash);

        let (valid, err) = reward_sys.verify_ledger_integrity();
        assert!(valid, "ledger should be valid: {:?}", err);

        let summary = reward_sys.get_summary();
        assert_eq!(summary["total_score"], 17.5);
        assert_eq!(summary["events_count"], 3);
        assert_eq!(summary["ledger_verified"], true);

        let domain = reward_sys.watermark_domain();
        let _ = tara_security::dpapi::delete_os_monotonic_watermark(&domain);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_unauthorized_manual_adjustment_rejected() {
        let tmp = test_dir();
        let reward_sys = RewardSystem::new(&tmp);

        let res = reward_sys.manual_adjustment(
            false,
            "intruder",
            RewardCategory::TaskCompletion,
            100.0,
            "giving myself points",
        );
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("Unauthorized"));

        let res_auth = reward_sys.manual_adjustment(
            true,
            "creator",
            RewardCategory::ContinuousImprovement,
            20.0,
            "approved model self-upgrade",
        );
        assert!(res_auth.is_ok());

        let domain = reward_sys.watermark_domain();
        let _ = tara_security::dpapi::delete_os_monotonic_watermark(&domain);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_tamper_detection() {
        let tmp = test_dir();
        let reward_sys = RewardSystem::new(&tmp);

        reward_sys
            .record_reward("agent_primary", RewardCategory::TaskCompletion, 5.0, "step 1")
            .unwrap();
        reward_sys
            .record_reward("agent_primary", RewardCategory::TaskCompletion, 5.0, "step 2")
            .unwrap();

        // Intentionally tamper with event 0
        {
            let mut ledger = reward_sys.ledger.lock().unwrap_or_else(|p| p.into_inner());
            ledger.events[0].delta = 50.0; // Tampered delta!
        }

        let (valid, err) = reward_sys.verify_ledger_integrity();
        assert!(!valid, "tampered ledger must fail integrity check");
        assert!(err.unwrap().contains("tampered"));

        let domain = reward_sys.watermark_domain();
        let _ = tara_security::dpapi::delete_os_monotonic_watermark(&domain);
        let _ = fs::remove_dir_all(&tmp);
    }

    struct TestDirGuard(PathBuf, String);
    impl Drop for TestDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
            let _ = tara_security::dpapi::delete_os_monotonic_watermark(&self.1);
        }
    }

    #[test]
    fn test_persistence_reload_and_anti_rollback_detection() {
        let tmp = test_dir();
        let reward_sys = RewardSystem::try_new(&tmp).unwrap();
        let domain = reward_sys.watermark_domain();
        let _guard = TestDirGuard(tmp.clone(), domain.clone());

        reward_sys
            .record_reward("agent_1", RewardCategory::TaskCompletion, 10.0, "initial task")
            .unwrap();
        reward_sys
            .record_reward("agent_1", RewardCategory::LearningProgress, 5.0, "second task")
            .unwrap();

        let enc_path = tmp
            .join("storage")
            .join("vault")
            .join("reward")
            .join("reward_ledger.enc");
        let enc_backup_2_events = fs::read(&enc_path).unwrap();

        reward_sys
            .record_reward("agent_1", RewardCategory::ContinuousImprovement, 2.5, "third task")
            .unwrap();

        drop(reward_sys);

        let reloaded_sys = RewardSystem::try_new(&tmp).unwrap();
        assert_eq!(reloaded_sys.get_events().len(), 3);
        assert_eq!(reloaded_sys.get_summary()["total_score"], 17.5);
        drop(reloaded_sys);

        // Restore older 2-event backup over 3-event ledger to simulate rollback
        fs::write(&enc_path, &enc_backup_2_events).unwrap();

        let rollback_attempt = RewardSystem::try_new(&tmp);
        assert!(
            rollback_attempt.is_err(),
            "Older truncated ledger must be rejected by anti-rollback watermark"
        );
        let err_msg = rollback_attempt.err().unwrap();
        assert!(err_msg.contains("AntiRollbackViolation"));
    }
}
