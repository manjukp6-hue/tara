//! Progressive and Full Lockdown Engine for TARA Security System.
//! Controls system security states:
//! NORMAL -> LOCKDOWN -> RECOVERY -> DESTROYING -> DESTROYED

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityState {
    Normal,
    Lockdown,
    Recovery,
    Destroying,
    Destroyed,
}

impl SecurityState {
    pub fn as_str(&self) -> &'static str {
        match self {
            SecurityState::Normal => "NORMAL",
            SecurityState::Lockdown => "LOCKDOWN",
            SecurityState::Recovery => "RECOVERY",
            SecurityState::Destroying => "DESTROYING",
            SecurityState::Destroyed => "DESTROYED",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockdownStateData {
    pub current_state: String,
    pub crypto_failure_count: usize,
    pub biometric_failure_count: usize,
    pub temporary_locked_until: Option<u64>,
    pub lockdown_reason: Option<String>,
    pub write_block_active: bool,
}

pub struct LockdownCoordinator {
    pub state_file: PathBuf,
    inner: Mutex<LockdownStateData>,
}

impl LockdownCoordinator {
    pub const TEMPORARY_LOCK_THRESHOLD: usize = 3;
    pub const FULL_LOCKDOWN_THRESHOLD: usize = 5;
    pub const TEMPORARY_LOCK_DURATION_SECONDS: u64 = 300;

    pub fn new(repo_root: &str) -> Self {
        let state_file = Path::new(repo_root).join("TARA/ACCESS/lockdown/lockdown_state.json");

        let mut data = LockdownStateData {
            current_state: "NORMAL".to_string(),
            crypto_failure_count: 0,
            biometric_failure_count: 0,
            temporary_locked_until: None,
            lockdown_reason: None,
            write_block_active: false,
        };

        if state_file.exists() {
            if let Ok(content) = fs::read_to_string(&state_file) {
                if let Ok(loaded) = serde_json::from_str::<LockdownStateData>(&content) {
                    data = loaded;
                }
            }
        }

        Self {
            state_file,
            inner: Mutex::new(data),
        }
    }

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    pub fn is_temporarily_locked(&self) -> bool {
        let mut d = self.inner.lock().unwrap();
        if let Some(until) = d.temporary_locked_until {
            let now = Self::now_secs();
            if now < until {
                return true;
            }
            d.temporary_locked_until = None;
            self.save_locked(&d);
        }
        false
    }

    pub fn is_in_lockdown(&self) -> bool {
        let d = self.inner.lock().unwrap();
        d.current_state == "LOCKDOWN"
            || d.current_state == "DESTROYING"
            || d.current_state == "DESTROYED"
    }

    pub fn is_write_blocked(&self) -> bool {
        let d = self.inner.lock().unwrap();
        d.write_block_active || d.current_state == "DESTROYING" || d.current_state == "DESTROYED"
    }

    pub fn record_crypto_failure(&self, details: &str) -> Value {
        let mut d = self.inner.lock().unwrap();
        if d.current_state == "DESTROYING" || d.current_state == "DESTROYED" {
            return json!({ "state": d.current_state, "action": "DENIED" });
        }

        d.crypto_failure_count += 1;
        let mut action = "DENY";

        if d.crypto_failure_count >= Self::FULL_LOCKDOWN_THRESHOLD {
            d.current_state = "LOCKDOWN".to_string();
            d.lockdown_reason = Some(format!(
                "FULL LOCKDOWN engaged: {} consecutive cryptographic attack failures: {}",
                d.crypto_failure_count, details
            ));
            action = "ENGAGE_FULL_LOCKDOWN";
        } else if d.crypto_failure_count >= Self::TEMPORARY_LOCK_THRESHOLD {
            d.temporary_locked_until =
                Some(Self::now_secs() + Self::TEMPORARY_LOCK_DURATION_SECONDS);
            action = "TEMPORARY_LOCKOUT";
        }

        self.save_locked(&d);

        json!({
            "state": d.current_state,
            "action": action,
            "failures": d.crypto_failure_count,
            "lockdown_reason": d.lockdown_reason
        })
    }

    pub fn record_biometric_failure(&self) -> Value {
        let mut d = self.inner.lock().unwrap();
        d.biometric_failure_count += 1;
        self.save_locked(&d);

        json!({
            "state": d.current_state,
            "action": "DEVICE_PIN_FALLBACK_REQUIRED",
            "biometric_failures": d.biometric_failure_count
        })
    }

    pub fn record_success(&self) {
        let mut d = self.inner.lock().unwrap();
        if d.current_state == "NORMAL" {
            d.crypto_failure_count = 0;
            d.biometric_failure_count = 0;
            d.temporary_locked_until = None;
            self.save_locked(&d);
        }
    }

    pub fn engage_manual_lockdown(&self, reason: &str) {
        let mut d = self.inner.lock().unwrap();
        if d.current_state != "DESTROYING" && d.current_state != "DESTROYED" {
            d.current_state = "LOCKDOWN".to_string();
            d.lockdown_reason = Some(reason.to_string());
            self.save_locked(&d);
        }
    }

    pub fn clear_lockdown_via_recovery(&self) -> bool {
        let mut d = self.inner.lock().unwrap();
        if d.current_state == "DESTROYING" || d.current_state == "DESTROYED" {
            return false;
        }
        d.current_state = "NORMAL".to_string();
        d.crypto_failure_count = 0;
        d.biometric_failure_count = 0;
        d.temporary_locked_until = None;
        d.lockdown_reason = None;
        self.save_locked(&d);
        true
    }

    pub fn can_execute_privileged_operation(&self, operation: &str) -> bool {
        let d = self.inner.lock().unwrap();
        if d.current_state == "DESTROYING" || d.current_state == "DESTROYED" {
            return false;
        }
        let in_lock = d.current_state == "LOCKDOWN";
        let is_temp = d
            .temporary_locked_until
            .map(|u| Self::now_secs() < u)
            .unwrap_or(false);

        if in_lock || is_temp {
            let allowed = [
                "status",
                "get_status",
                "initiate_recovery",
                "verify_recovery",
            ];
            return allowed.contains(&operation.to_lowercase().as_str());
        }
        true
    }

    pub fn set_destroying_state(&self) {
        let mut d = self.inner.lock().unwrap();
        d.current_state = "DESTROYING".to_string();
        d.write_block_active = true;
        d.lockdown_reason = Some("SYSTEM_DESTROYING".to_string());
        self.save_locked(&d);
    }

    pub fn set_destroyed_state(&self) {
        let mut d = self.inner.lock().unwrap();
        d.current_state = "DESTROYED".to_string();
        d.write_block_active = true;
        d.lockdown_reason = Some("SYSTEM_DESTROYED".to_string());
        self.save_locked(&d);
    }

    fn save_locked(&self, d: &LockdownStateData) {
        if let Some(parent) = self.state_file.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(serialized) = serde_json::to_string_pretty(d) {
            let _ = fs::write(&self.state_file, serialized);
        }
    }
}
