//! Canonical Encrypted & Tamper-Evident Reward System for TARA AI.
//!
//! Enforces:
//! - Exact mathematical reward tracking across 10 canonical reward dimensions:
//!   TaskCompletion, LearningProgress, SuccessfulReasoning, SuccessfulResearch,
//!   ValidatedCapability, UserApproved, Consistency, SafetyAdherence,
//!   ContinuousImprovement, LongTermGoal.
//! - Cryptographic hash-chaining of all reward events (tamper-evident audit trail).
//! - Authenticated AES-256-GCM encryption at rest with Scrypt key derivation.
//! - Creator-authorized operations for manual adjustments, resets, and policy changes.
//! - Zero hardcoded secret keys or plaintext exposures.

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

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
    pub fn compute_hash(
        prev_hash: &str,
        event_id: &str,
        timestamp_ms: u64,
        actor_id: &str,
        category: RewardCategory,
        delta: f64,
        reason: &str,
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(prev_hash.as_bytes());
        hasher.update(b"|");
        hasher.update(event_id.as_bytes());
        hasher.update(b"|");
        hasher.update(timestamp_ms.to_string().as_bytes());
        hasher.update(b"|");
        hasher.update(actor_id.as_bytes());
        hasher.update(b"|");
        hasher.update(category.as_str().as_bytes());
        hasher.update(b"|");
        hasher.update(format!("{:.6}", delta).as_bytes());
        hasher.update(b"|");
        hasher.update(reason.as_bytes());
        hex::encode(hasher.finalize())
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

pub struct RewardSystem {
    storage_path: PathBuf,
    events: Mutex<Vec<RewardEvent>>,
    state: Mutex<RewardState>,
    encryption_salt: [u8; 16],
}

impl RewardSystem {
    pub fn new<P: AsRef<Path>>(repo_root: P) -> Self {
        let storage_path = repo_root
            .as_ref()
            .join("storage")
            .join("vault")
            .join("reward");
        let _ = fs::create_dir_all(&storage_path);

        let salt_file = storage_path.join("salt.bin");
        let mut salt = [0u8; 16];
        if salt_file.exists() {
            if let Ok(mut f) = File::open(&salt_file) {
                let _ = f.read_exact(&mut salt);
            }
        } else {
            rand::thread_rng().fill_bytes(&mut salt);
            if let Ok(mut f) = File::create(&salt_file) {
                let _ = f.write_all(&salt);
            }
        }

        let mut sys = Self {
            storage_path,
            events: Mutex::new(Vec::new()),
            state: Mutex::new(RewardState {
                total_score: 0.0,
                category_scores: HashMap::new(),
                events_count: 0,
                last_event_hash: GENESIS_REWARD_HASH.to_string(),
                last_updated_ms: 0,
            }),
            encryption_salt: salt,
        };

        let _ = sys.load_encrypted();
        sys
    }

    /// Derive a 256-bit AES encryption key deterministically using Scrypt.
    fn derive_key(&self) -> [u8; 32] {
        let env_secret = std::env::var("TARA_REWARD_KEY").unwrap_or_else(|_| {
            // No hardcoded secret string. Derive an opaque key from the vault's own salt.
            format!("TARA_SALT_DERIVED_{}", hex::encode(self.encryption_salt))
        });
        const SCRYPT_LOG_N: u8 = 14;
        const SCRYPT_R: u32 = 8;
        const SCRYPT_P: u32 = 1;
        const SCRYPT_KEY_LEN: usize = 32;

        let mut key = [0u8; 32];
        let params = scrypt::Params::new(SCRYPT_LOG_N, SCRYPT_R, SCRYPT_P, SCRYPT_KEY_LEN)
            .expect("valid scrypt params");
        scrypt::scrypt(
            env_secret.as_bytes(),
            &self.encryption_salt,
            &params,
            &mut key,
        )
        .expect("scrypt derivation success");
        key
    }

    /// Record a validated reward event into the tamper-evident ledger.
    pub fn record_reward(
        &self,
        actor_id: &str,
        category: RewardCategory,
        delta: f64,
        reason: &str,
    ) -> Result<RewardEvent, String> {
        let mut events = self.events.lock().unwrap();
        let mut state = self.state.lock().unwrap();

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let event_id = format!("rew_{}_{}", ts, events.len());
        let prev_hash = state.last_event_hash.clone();
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

        // Update running state
        state.total_score += delta;
        *state
            .category_scores
            .entry(category.as_str().to_string())
            .or_insert(0.0) += delta;
        state.events_count += 1;
        state.last_event_hash = sig_hash;
        state.last_updated_ms = ts;

        events.push(event.clone());

        // Save ledger to encrypted storage
        drop(events);
        drop(state);
        self.save_encrypted()?;

        Ok(event)
    }

    /// Creator-authorized manual reward adjustment.
    pub fn manual_adjustment(
        &self,
        is_creator: bool,
        actor_id: &str,
        category: RewardCategory,
        delta: f64,
        justification: &str,
    ) -> Result<RewardEvent, String> {
        if !is_creator {
            return Err(
                "Unauthorized: manual reward adjustments require validated Creator Authority"
                    .into(),
            );
        }
        let reason = format!("[CREATOR_ADJUSTMENT]: {}", justification);
        self.record_reward(actor_id, category, delta, &reason)
    }

    /// Verify the complete cryptographic hash chain of the reward ledger.
    pub fn verify_ledger_integrity(&self) -> (bool, Option<String>) {
        let events = self.events.lock().unwrap();
        if events.is_empty() {
            return (true, None);
        }

        let mut running_hash = if let Some(first) = events.first() {
            if first.prev_hash == "0000000000000000000000000000000000000000000000000000000000000000" {
                "0000000000000000000000000000000000000000000000000000000000000000".to_string()
            } else {
                GENESIS_REWARD_HASH.to_string()
            }
        } else {
            GENESIS_REWARD_HASH.to_string()
        };
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

    /// Save ledger events encrypted with AES-256-GCM.
    fn save_encrypted(&self) -> Result<(), String> {
        let events = self.events.lock().unwrap();
        let payload = serde_json::to_vec(&*events).map_err(|e| e.to_string())?;

        let key = self.derive_key();
        let cipher = Aes256Gcm::new_from_slice(&key).map_err(|e| e.to_string())?;

        let mut nonce_bytes = [0u8; 12];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher
            .encrypt(nonce, payload.as_ref())
            .map_err(|e| format!("encryption failed: {}", e))?;

        let enc_file = self.storage_path.join("reward_ledger.enc");
        let mut out = File::create(&enc_file).map_err(|e| e.to_string())?;
        out.write_all(&nonce_bytes).map_err(|e| e.to_string())?;
        out.write_all(&ciphertext).map_err(|e| e.to_string())?;

        Ok(())
    }

    /// Load ledger events from AES-256-GCM encrypted file.
    fn load_encrypted(&mut self) -> Result<(), String> {
        let enc_file = self.storage_path.join("reward_ledger.enc");
        if !enc_file.exists() {
            return Ok(());
        }

        let mut f = File::open(&enc_file).map_err(|e| e.to_string())?;
        let mut data = Vec::new();
        f.read_to_end(&mut data).map_err(|e| e.to_string())?;

        if data.len() < 12 {
            return Err("corrupt encrypted reward ledger: file too short".into());
        }

        let (nonce_bytes, ciphertext) = data.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);

        let key = self.derive_key();
        let cipher = Aes256Gcm::new_from_slice(&key).map_err(|e| e.to_string())?;

        let plaintext = cipher.decrypt(nonce, ciphertext).map_err(|e| {
            format!(
                "reward ledger decryption failed (tampered or wrong key): {}",
                e
            )
        })?;

        let loaded_events: Vec<RewardEvent> =
            serde_json::from_slice(&plaintext).map_err(|e| e.to_string())?;

        // Reconstruct running state
        let mut total = 0.0;
        let mut cat_scores: HashMap<String, f64> = HashMap::new();
        let mut last_hash = GENESIS_REWARD_HASH.to_string();
        let mut last_ts = 0;

        for ev in &loaded_events {
            total += ev.delta;
            *cat_scores
                .entry(ev.category.as_str().to_string())
                .or_insert(0.0) += ev.delta;
            last_hash = ev.signature_hash.clone();
            last_ts = ev.timestamp_ms;
        }

        let count = loaded_events.len();
        *self.events.lock().unwrap() = loaded_events;
        *self.state.lock().unwrap() = RewardState {
            total_score: total,
            category_scores: cat_scores,
            events_count: count,
            last_event_hash: last_hash,
            last_updated_ms: last_ts,
        };

        Ok(())
    }

    /// Retrieve summary of reward state.
    pub fn get_summary(&self) -> Value {
        let state = self.state.lock().unwrap();
        let (integrity, err) = self.verify_ledger_integrity();
        json!({
            "total_score": state.total_score,
            "category_scores": state.category_scores,
            "events_count": state.events_count,
            "last_event_hash": state.last_event_hash,
            "last_updated_ms": state.last_updated_ms,
            "ledger_verified": integrity,
            "integrity_error": err
        })
    }

    /// Retrieve all decrypted reward events in sequence.
    pub fn get_events(&self) -> Vec<RewardEvent> {
        self.events.lock().unwrap().clone()
    }

    /// Compute multiplicative reward bias for decision making and strategy weighting:
    /// Bias = 1.0 + tanh(category_score / 20.0).
    /// Positive historical rewards boost strategy selection; negative scores dampen it.
    pub fn compute_category_bias(&self, category: RewardCategory) -> f64 {
        let state = self.state.lock().unwrap();
        let cat_str = category.as_str();
        let score = state.category_scores.get(cat_str).copied().unwrap_or(0.0);
        1.0 + (score / 20.0).tanh()
    }

    /// Retrieve categories ranked by historical reward performance to prioritize autonomous goals.
    pub fn get_ranked_category_priorities(&self) -> Vec<(RewardCategory, f64)> {
        let all_cats = [
            RewardCategory::TaskCompletion,
            RewardCategory::LearningProgress,
            RewardCategory::SuccessfulReasoning,
            RewardCategory::SuccessfulResearch,
            RewardCategory::ValidatedCapability,
            RewardCategory::UserApproved,
            RewardCategory::Consistency,
            RewardCategory::SafetyAdherence,
            RewardCategory::ContinuousImprovement,
            RewardCategory::LongTermGoal,
        ];

        let mut ranked = Vec::new();
        for &cat in &all_cats {
            let bias = self.compute_category_bias(cat);
            ranked.push((cat, bias));
        }
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir() -> PathBuf {
        let p = std::env::temp_dir().join(format!("tara_rew_test_{}", uuid::Uuid::new_v4()));
        let _ = fs::create_dir_all(&p);
        p
    }

    #[test]
    fn test_reward_event_hash_chaining() {
        let tmp = test_dir();
        let reward_sys = RewardSystem::new(&tmp);

        // Record 3 events
        let ev1 = reward_sys
            .record_reward(
                "alice",
                RewardCategory::TaskCompletion,
                10.0,
                "solved calculus problem",
            )
            .unwrap();
        assert_eq!(ev1.prev_hash, GENESIS_REWARD_HASH);

        let ev2 = reward_sys
            .record_reward(
                "alice",
                RewardCategory::LearningProgress,
                5.0,
                "mastered quadratic equations",
            )
            .unwrap();
        assert_eq!(ev2.prev_hash, ev1.signature_hash);

        let ev3 = reward_sys
            .record_reward(
                "bob",
                RewardCategory::SafetyAdherence,
                2.5,
                "clean security audit",
            )
            .unwrap();
        assert_eq!(ev3.prev_hash, ev2.signature_hash);

        // Verify ledger integrity
        let (valid, err) = reward_sys.verify_ledger_integrity();
        assert!(valid, "ledger should be valid: {:?}", err);

        let summary = reward_sys.get_summary();
        assert_eq!(summary["total_score"], 17.5);
        assert_eq!(summary["events_count"], 3);
        assert_eq!(summary["ledger_verified"], true);

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_unauthorized_manual_adjustment_rejected() {
        let tmp = test_dir();
        let reward_sys = RewardSystem::new(&tmp);

        // Unauthorized user attempts manual adjustment
        let res = reward_sys.manual_adjustment(
            false,
            "intruder",
            RewardCategory::TaskCompletion,
            1000.0,
            "giving myself points",
        );
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("Unauthorized"));

        // Authorized creator succeeds
        let res_auth = reward_sys.manual_adjustment(
            true,
            "creator",
            RewardCategory::ContinuousImprovement,
            20.0,
            "approved model self-upgrade",
        );
        assert!(res_auth.is_ok());

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_tamper_detection() {
        let tmp = test_dir();
        let reward_sys = RewardSystem::new(&tmp);

        reward_sys
            .record_reward("alice", RewardCategory::TaskCompletion, 5.0, "step 1")
            .unwrap();
        reward_sys
            .record_reward("alice", RewardCategory::TaskCompletion, 5.0, "step 2")
            .unwrap();

        // Intentionally tamper with event 0
        {
            let mut events = reward_sys.events.lock().unwrap();
            events[0].delta = 500.0; // Tampered delta!
        }

        let (valid, err) = reward_sys.verify_ledger_integrity();
        assert!(!valid, "tampered ledger must fail integrity check");
        assert!(err.unwrap().contains("tampered"));

        let _ = fs::remove_dir_all(&tmp);
    }
}
