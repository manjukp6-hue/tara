//! Progressive and Full Lockdown Engine for TARA Security Subsystem.
//!
//! Controls system security states:
//! NORMAL -> LOCKDOWN -> RECOVERY -> DESTROYING -> DESTROYED
//!
//! Hardened invariants:
//! - Corrupt, unreadable, or missing signature files fail closed into LOCKDOWN.
//! - Unknown state strings fail closed into LOCKDOWN.
//! - State is HMAC-SHA256 authenticated to prevent offline tampering.
//! - Progressive authentication failures trigger exponential backoff and lockdown.
//! - Clearing lockdown requires verified creator identity and recovery proof.
//! - Atomic crash-safe disk persistence.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LockdownState {
    Normal,
    Lockdown,
    Recovery,
    Destroying,
    Destroyed,
}

impl LockdownState {
    pub fn as_str(&self) -> &'static str {
        match self {
            LockdownState::Normal => "NORMAL",
            LockdownState::Lockdown => "LOCKDOWN",
            LockdownState::Recovery => "RECOVERY",
            LockdownState::Destroying => "DESTROYING",
            LockdownState::Destroyed => "DESTROYED",
        }
    }
}

impl std::str::FromStr for LockdownState {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.trim().to_uppercase().as_str() {
            "NORMAL" => LockdownState::Normal,
            "LOCKDOWN" => LockdownState::Lockdown,
            "RECOVERY" => LockdownState::Recovery,
            "DESTROYING" => LockdownState::Destroying,
            "DESTROYED" => LockdownState::Destroyed,
            _ => LockdownState::Lockdown, // Fail closed on any unknown state!
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockdownRecord {
    pub current_state: String,
    pub crypto_failure_count: usize,
    pub biometric_failure_count: usize,
    pub temporary_locked_until: Option<u64>,
    pub lockdown_reason: Option<String>,
    pub write_block_active: bool,
    #[serde(default)]
    pub hmac_signature: String,
}

impl LockdownRecord {
    pub fn compute_hmac(&self, hmac_key: &[u8]) -> String {
        let mut signer = crate::cryptography::HmacSha256Signer::new(hmac_key).expect("HMAC accepts key");
        signer.update(self.current_state.as_bytes());
        signer.update(b":");
        signer.update(&self.crypto_failure_count.to_be_bytes());
        signer.update(b":");
        signer.update(&self.biometric_failure_count.to_be_bytes());
        signer.update(b":");
        if let Some(until) = self.temporary_locked_until {
            signer.update(&until.to_be_bytes());
        }
        signer.update(b":");
        if let Some(ref reason) = self.lockdown_reason {
            signer.update(reason.as_bytes());
        }
        signer.update(b":");
        signer.update(if self.write_block_active { b"1" } else { b"0" });
        signer.finalize_hex()
    }

    pub fn verify_hmac(&self, hmac_key: &[u8]) -> bool {
        if self.hmac_signature.is_empty() {
            return false;
        }
        let expected = self.compute_hmac(hmac_key);
        crate::cryptography::CryptoUtils::constant_time_eq(
            self.hmac_signature.as_bytes(),
            expected.as_bytes(),
        )
    }
}

/// Progressive Lockdown Coordinator managing dynamic threats and backoff.
pub struct ProgressiveLockdownCoordinator {
    state_file: PathBuf,
    hmac_key: [u8; 32],
    state: Mutex<LockdownRecord>,
    generation: std::sync::atomic::AtomicU64,
}

pub type LockdownCoordinator = ProgressiveLockdownCoordinator;

impl ProgressiveLockdownCoordinator {
    pub fn new<P: AsRef<Path>>(base_dir: P, hmac_key: [u8; 32]) -> Self {
        let dir = base_dir.as_ref().to_path_buf();
        let state_file = dir.join("storage").join("vault").join("access").join("lockdown_state.json");
        if let Some(parent) = state_file.parent() {
            let _ = fs::create_dir_all(parent);
        }

        let loaded = Self::load_or_default(&state_file, &hmac_key);
        Self {
            state_file,
            hmac_key,
            state: Mutex::new(loaded),
            generation: std::sync::atomic::AtomicU64::new(1),
        }
    }

    pub fn new_with_root<P: AsRef<Path>>(base_dir: P) -> Self {
        let key = crate::dpapi::derive_authority_hmac_key("lockdown_state")
            .expect("FAIL CLOSED: Mandatory durable lockdown state HMAC key must be derived from protected DPAPI storage");
        Self::new(base_dir, key)
    }

    fn now_secs() -> u64 {
        crate::cryptography::CryptoUtils::now_secs()
    }

    fn load_or_default(path: &Path, hmac_key: &[u8]) -> LockdownRecord {
        if path.exists() {
            if let Ok(data) = fs::read(path) {
                if let Ok(record) = serde_json::from_slice::<LockdownRecord>(&data) {
                    if record.verify_hmac(hmac_key) {
                        return record;
                    }
                }
            }
            // Tampered or corrupt state: fail closed into full lockdown!
            return LockdownRecord {
                current_state: LockdownState::Lockdown.as_str().to_string(),
                crypto_failure_count: 10,
                biometric_failure_count: 10,
                temporary_locked_until: None,
                lockdown_reason: Some("State file corrupted or HMAC signature invalid (FailClosed)".into()),
                write_block_active: true,
                hmac_signature: String::new(),
            };
        }

        // If path does NOT exist, check if parent directory has established security records.
        // If security records exist but lockdown state was deleted, fail closed (prevents lockdown reset via file deletion).
        if let Some(parent) = path.parent() {
            let watermark_path = crate::dpapi::get_os_isolated_watermark_path("creator_authority_initialized");
            let was_previously_initialized = crate::dpapi::load_monotonic_watermark(&watermark_path)
                .ok()
                .flatten()
                .map(|val| val > 0)
                .unwrap_or(false);

            let has_established_security = was_previously_initialized
                || parent.join("access_seal.json").exists()
                || parent.join("operator_record.json").exists()
                || parent.join("operators_registry.json").exists()
                || parent.join("devices.json").exists()
                || parent.join("restore_config.json").exists()
                || parent.join("enrolled_passkeys.json").exists()
                || parent.join("recovery_master_key.bin").exists()
                || parent.join("authority_state.json").exists()
                || parent.join("enrolled_devices.json").exists()
                || parent.join("creator_identity.json").exists();
            if has_established_security {
                return LockdownRecord {
                    current_state: LockdownState::Lockdown.as_str().to_string(),
                    crypto_failure_count: 10,
                    biometric_failure_count: 10,
                    temporary_locked_until: None,
                    lockdown_reason: Some("Lockdown state missing in established security vault: deletion attack detected (FailClosed)".into()),
                    write_block_active: true,
                    hmac_signature: String::new(),
                };
            }
        }

        let mut rec = LockdownRecord {
            current_state: LockdownState::Normal.as_str().to_string(),
            crypto_failure_count: 0,
            biometric_failure_count: 0,
            temporary_locked_until: None,
            lockdown_reason: None,
            write_block_active: false,
            hmac_signature: String::new(),
        };
        rec.hmac_signature = rec.compute_hmac(hmac_key);
        if let Ok(data) = serde_json::to_vec_pretty(&rec) {
            let _ = crate::persistence::atomic_write_file_hardened(path, &data);
        }
        rec
    }

    pub fn is_locked_down(&self) -> bool {
        let guard = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if guard.current_state != LockdownState::Normal.as_str() {
            return true;
        }
        if let Some(until) = guard.temporary_locked_until {
            if Self::now_secs() < until {
                return true;
            }
        }
        false
    }

    pub fn record_failure(&self, is_biometric: bool) -> Result<(), String> {
        let mut guard = self.state.lock().map_err(|_| "Lockdown state mutex poisoned")?;
        if is_biometric {
            guard.biometric_failure_count = guard.biometric_failure_count.saturating_add(1);
        } else {
            guard.crypto_failure_count = guard.crypto_failure_count.saturating_add(1);
        }

        let total_failures = guard.crypto_failure_count + guard.biometric_failure_count;
        let now = Self::now_secs();

        if total_failures >= 10 {
            if guard.current_state != LockdownState::Destroying.as_str()
                && guard.current_state != LockdownState::Destroyed.as_str()
            {
                guard.current_state = LockdownState::Lockdown.as_str().to_string();
            }
            guard.write_block_active = true;
            guard.lockdown_reason = Some(format!("Progressive escalation: {} failed attempts", total_failures));
        } else if total_failures >= 5 {
            guard.temporary_locked_until = Some(now + 900); // 15 min
            guard.lockdown_reason = Some("Temporary backoff: 5+ failed attempts".into());
        } else if total_failures >= 3 {
            guard.temporary_locked_until = Some(now + 300); // 5 min
            guard.lockdown_reason = Some("Temporary backoff: 3+ failed attempts".into());
        }

        guard.hmac_signature = guard.compute_hmac(&self.hmac_key);
        self.persist_state(&guard)?;
        self.generation.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    pub fn record_success(&self) -> Result<(), String> {
        let mut guard = self.state.lock().map_err(|_| "Lockdown state mutex poisoned")?;
        if guard.current_state == LockdownState::Normal.as_str() {
            guard.crypto_failure_count = 0;
            guard.biometric_failure_count = 0;
            guard.temporary_locked_until = None;
            guard.lockdown_reason = None;
            guard.hmac_signature = guard.compute_hmac(&self.hmac_key);
            self.persist_state(&guard)?;
            self.generation.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        Ok(())
    }

    pub fn trigger_lockdown(&self, reason: &str) -> Result<(), String> {
        let mut guard = self.state.lock().map_err(|_| "Lockdown state mutex poisoned")?;
        guard.current_state = LockdownState::Lockdown.as_str().to_string();
        guard.write_block_active = true;
        guard.lockdown_reason = Some(reason.to_string());
        guard.hmac_signature = guard.compute_hmac(&self.hmac_key);
        self.persist_state(&guard)?;
        self.generation.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    pub fn current_generation(&self) -> u64 {
        self.generation.load(std::sync::atomic::Ordering::SeqCst)
    }

    pub fn assert_unlocked(&self, gen: u64) -> Result<(), String> {
        if self.is_locked_down() {
            return Err("System is currently locked down".to_string());
        }
        let current = self.current_generation();
        if current != gen {
            return Err(format!(
                "Lockdown state modified concurrently during execution (generation mismatch: expected {}, current {})",
                gen, current
            ));
        }
        Ok(())
    }

    pub fn is_in_lockdown(&self) -> bool {
        let guard = self.state.lock().unwrap_or_else(|p| p.into_inner());
        guard.current_state != LockdownState::Normal.as_str()
    }

    pub fn is_temporarily_locked(&self) -> bool {
        let guard = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(until) = guard.temporary_locked_until {
            Self::now_secs() < until
        } else {
            false
        }
    }

    pub fn is_write_blocked(&self) -> bool {
        let guard = self.state.lock().unwrap_or_else(|p| p.into_inner());
        guard.write_block_active || guard.current_state != LockdownState::Normal.as_str()
    }

    pub fn record_crypto_failure(&self, _details: &str) -> serde_json::Value {
        let _ = self.record_failure(false);
        serde_json::json!({
            "is_locked_down": self.is_locked_down(),
            "write_blocked": self.is_write_blocked()
        })
    }

    pub fn record_biometric_failure(&self, _details: &str) -> serde_json::Value {
        let _ = self.record_failure(true);
        serde_json::json!({
            "is_locked_down": self.is_locked_down(),
            "write_blocked": self.is_write_blocked()
        })
    }

    /// Lifts lockdown using a cryptographically verified RecoveryAuthorizationProof via RecoveryManager.
    pub fn clear_lockdown_with_proof(
        &self,
        proof: &crate::recovery::RecoveryAuthorizationProof,
        recovery_mgr: &crate::recovery::RecoveryManager,
    ) -> Result<(), String> {
        let _ = recovery_mgr.consume_proof(&proof.proof_id)?;
        self.clear_lockdown_internal()
    }

    /// Lifts lockdown using a cryptographically verified Creator Principal with CREATOR_ROOT assurance.
    pub fn clear_lockdown_with_principal(
        &self,
        principal: &crate::primitives::VerifiedPrincipal,
    ) -> Result<(), String> {
        if principal.role() != &crate::primitives::PrincipalRole::creator()
            || principal.assurance_level() < crate::primitives::AssuranceLevel::CREATOR_ROOT
        {
            return Err("LockdownClearDenied: Lifting lockdown requires verified Creator authority with CREATOR_ROOT assurance".into());
        }
        self.clear_lockdown_internal()
    }

    pub fn clear_lockdown_internal(&self) -> Result<(), String> {
        let mut guard = self.state.lock().map_err(|_| "Lockdown state mutex poisoned")?;
        if guard.current_state == LockdownState::Destroying.as_str()
            || guard.current_state == LockdownState::Destroyed.as_str()
        {
            return Err("LockdownClearDenied: Cannot clear terminal DESTROYING/DESTROYED state".into());
        }
        guard.current_state = LockdownState::Normal.as_str().to_string();
        guard.crypto_failure_count = 0;
        guard.biometric_failure_count = 0;
        guard.temporary_locked_until = None;
        guard.lockdown_reason = None;
        guard.write_block_active = false;
        guard.hmac_signature = guard.compute_hmac(&self.hmac_key);
        self.persist_state(&guard)?;
        self.generation.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    pub fn clear_lockdown(&self) -> Result<(), String> {
        Err("LockdownClearDenied: Lifting lockdown requires verified proof (use clear_lockdown_with_proof or clear_lockdown_with_principal)".into())
    }

    fn persist_state(&self, record: &LockdownRecord) -> Result<(), String> {
        let data = serde_json::to_vec_pretty(record)
            .map_err(|e| format!("Serialization error: {e}"))?;
        crate::persistence::atomic_write_file_hardened(&self.state_file, &data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lockdown_coordinator_initializes_and_clears() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let root = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .unwrap_or(&manifest_dir);
        let coord = LockdownCoordinator::new_with_root(root);
        let gen_before = coord.current_generation();
        let res = coord.clear_lockdown_internal();
        assert!(res.is_ok());
        assert!(!coord.is_locked_down());
        assert_eq!(coord.current_generation(), gen_before + 1);
    }

    #[test]
    fn test_lockdown_coordinator_refuses_clear_when_destroying_or_destroyed() {
        let temp_dir = std::env::temp_dir().join(format!("tara_lockdown_test_{}", rand::random::<u64>()));
        let coord = LockdownCoordinator::new(&temp_dir, [7u8; 32]);

        // Manually transition to DESTROYING
        {
            let mut guard = coord.state.lock().unwrap();
            guard.current_state = LockdownState::Destroying.as_str().to_string();
            guard.hmac_signature = guard.compute_hmac(&coord.hmac_key);
            coord.persist_state(&guard).unwrap();
        }

        // Clearing must be refused
        let res = coord.clear_lockdown_internal();
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("Cannot clear terminal DESTROYING/DESTROYED state"));

        // Failures must not downgrade DESTROYING to LOCKDOWN
        for _ in 0..12 {
            let _ = coord.record_failure(false);
        }
        {
            let guard = coord.state.lock().unwrap();
            assert_eq!(guard.current_state, "DESTROYING");
        }

        // Manually transition to DESTROYED
        {
            let mut guard = coord.state.lock().unwrap();
            guard.current_state = LockdownState::Destroyed.as_str().to_string();
            guard.hmac_signature = guard.compute_hmac(&coord.hmac_key);
            coord.persist_state(&guard).unwrap();
        }

        // Clearing must be refused
        let res_destroyed = coord.clear_lockdown_internal();
        assert!(res_destroyed.is_err());
        assert!(res_destroyed.unwrap_err().contains("Cannot clear terminal DESTROYING/DESTROYED state"));

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}

