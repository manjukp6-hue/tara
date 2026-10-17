//! Server-Side Session and Challenge Replay-Protection Subsystem.
//!
//! Rule compliance:
//! - Sessions have fixed issuer, audience, expiry, device binding, and authority generation.
//! - Challenge nonces are server-generated, bounded in lifetime, bound to session/action, atomically consumed, and rejected on replay.
//! - Revocation immediately invalidates all associated sessions.
//! - Fail-closed on mutex poison or expired/replayed tokens.

use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::primitives::{
    AuthorityGeneration, ChallengeId, DeviceId, PrincipalId, SecurityAction, SessionId,
    VerifiedPrincipal,
};

/// Server-side authenticated session record.
#[derive(Debug, Clone, Serialize)]
pub struct SessionRecord {
    pub session_id: SessionId,
    pub principal: VerifiedPrincipal,
    pub bound_device: Option<DeviceId>,
    pub issuer: String,
    pub audience: String,
    pub authority_generation: AuthorityGeneration,
    pub created_epoch_secs: u64,
    pub expires_epoch_secs: u64,
}

/// A one-time cryptographic challenge nonce.
#[derive(Debug, Clone)]
pub struct ChallengeRecord {
    pub challenge_id: ChallengeId,
    pub session_id: Option<SessionId>,
    pub bound_action: Option<SecurityAction>,
    pub created_at: Instant,
    pub ttl: Duration,
}

pub const MAX_ACTIVE_CHALLENGES: usize = 10_000;

/// Internal session state managed under a single concurrency lock to eliminate races (ID-11).
struct InnerSessionState {
    sessions: HashMap<String, SessionRecord>,
    challenges: HashMap<String, ChallengeRecord>,
    revoked_principals: HashSet<String>,
    revoked_devices: HashSet<String>,
}

#[derive(Serialize, Deserialize)]
struct RevocationStore {
    revoked_principals: HashSet<String>,
    revoked_devices: HashSet<String>,
}

/// Central Session and Replay Protection Manager.
pub struct SessionManager {
    state: Arc<Mutex<InnerSessionState>>,
    default_session_ttl_secs: u64,
    default_challenge_ttl: Duration,
    storage_path: Option<PathBuf>,
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionManager {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(InnerSessionState {
                sessions: HashMap::new(),
                challenges: HashMap::new(),
                revoked_principals: HashSet::new(),
                revoked_devices: HashSet::new(),
            })),
            default_session_ttl_secs: 3600,                  // 1 hour
            default_challenge_ttl: Duration::from_secs(300), // 5 minutes
            storage_path: None,
        }
    }

    pub fn with_storage<P: AsRef<Path>>(storage_path: P) -> Self {
        let path = storage_path.as_ref().to_path_buf();
        let (rev_principals, rev_devices) = if path.exists() {
            if let Ok(bytes) = std::fs::read(&path) {
                if let Ok(store) = serde_json::from_slice::<RevocationStore>(&bytes) {
                    (store.revoked_principals, store.revoked_devices)
                } else {
                    (HashSet::new(), HashSet::new())
                }
            } else {
                (HashSet::new(), HashSet::new())
            }
        } else {
            (HashSet::new(), HashSet::new())
        };

        Self {
            state: Arc::new(Mutex::new(InnerSessionState {
                sessions: HashMap::new(),
                challenges: HashMap::new(),
                revoked_principals: rev_principals,
                revoked_devices: rev_devices,
            })),
            default_session_ttl_secs: 3600,
            default_challenge_ttl: Duration::from_secs(300),
            storage_path: Some(path),
        }
    }

    fn persist_revocations(&self, state: &InnerSessionState) {
        if let Some(ref path) = self.storage_path {
            let store = RevocationStore {
                revoked_principals: state.revoked_principals.clone(),
                revoked_devices: state.revoked_devices.clone(),
            };
            if let Ok(data) = serde_json::to_vec_pretty(&store) {
                let _ = crate::persistence::atomic_write_file_hardened(path, &data);
            }
        }
    }

    /// Issue a new server-side authenticated session with verified generation (ID-08, ID-11, ID-17).
    pub fn create_session(
        &self,
        principal: VerifiedPrincipal,
        issuer: &str,
        audience: &str,
        current_generation: AuthorityGeneration,
        current_epoch_secs: u64,
    ) -> Result<SessionRecord, String> {
        // 1. Generation alignment check (ID-08)
        if principal.authority_generation() != current_generation {
            return Err(format!(
                "AuthorityGenerationMismatch: Principal generation {} != current generation {}",
                principal.authority_generation().get(),
                current_generation.get()
            ));
        }

        // Strict audience check (fail closed against empty audience)
        if audience.trim().is_empty() {
            return Err(
                "EmptyAudience: Session creation requires non-empty audience (FailClosed)".into(),
            );
        }

        let principal_key = principal.id().as_str();
        let bound_device = principal.bound_device().cloned();

        // 2. Atomic check and insertion under unified lock (ID-11, ID-17)
        let mut state = self
            .state
            .lock()
            .map_err(|_| "Session state lock poisoned (FailClosed)")?;

        if state.revoked_principals.contains(&principal_key) {
            return Err("PrincipalRevoked: Cannot issue session to revoked principal".into());
        }

        if let Some(ref dev) = bound_device {
            if state.revoked_devices.contains(dev.as_str()) {
                return Err(
                    "DeviceRevoked: Cannot issue session with a revoked bound device".into(),
                );
            }
        }

        // Generate 32-byte cryptographically secure session ID
        let mut rand_bytes = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut rand_bytes);
        let session_id = SessionId::new(hex::encode(rand_bytes))?;

        let expires_epoch_secs = current_epoch_secs
            .checked_add(self.default_session_ttl_secs)
            .ok_or("Session expiry calculation overflow")?;

        let record = SessionRecord {
            session_id: session_id.clone(),
            principal,
            bound_device,
            issuer: issuer.to_string(),
            audience: audience.to_string(),
            authority_generation: current_generation,
            created_epoch_secs: current_epoch_secs,
            expires_epoch_secs,
        };

        state
            .sessions
            .insert(session_id.as_str().to_string(), record.clone());

        Ok(record)
    }

    /// Register or update an authenticated session for an authenticated principal (e.g. Creator login / OAuth).
    pub fn ensure_session_registered(
        &self,
        session_id_str: &str,
        principal: &VerifiedPrincipal,
        current_generation: AuthorityGeneration,
        current_epoch_secs: u64,
    ) -> Result<(), String> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "Session state lock poisoned (FailClosed)")?;
        if state.revoked_principals.contains(&principal.id().as_str()) {
            return Err("PrincipalRevoked: Principal has been revoked in zero-trust session store (FailClosed)".into());
        }
        if let Some(dev) = principal.bound_device() {
            if state.revoked_devices.contains(dev.as_str()) {
                return Err("DeviceRevoked: Bound device has been revoked in zero-trust session store (FailClosed)".into());
            }
        }
        if !state.sessions.contains_key(session_id_str) {
            let expires_epoch_secs =
                current_epoch_secs.saturating_add(self.default_session_ttl_secs);
            let s_id = SessionId::new(session_id_str)?;
            let record = SessionRecord {
                session_id: s_id,
                principal: principal.clone(),
                bound_device: principal.bound_device().cloned(),
                issuer: crate::primitives::CANONICAL_ISSUER.to_string(),
                audience: crate::primitives::CANONICAL_AUDIENCE.to_string(),
                authority_generation: current_generation,
                created_epoch_secs: current_epoch_secs,
                expires_epoch_secs,
            };
            state.sessions.insert(session_id_str.to_string(), record);
        }
        Ok(())
    }

    /// Validate a session token strictly fail-closed against current authority generation.
    pub fn validate_session(
        &self,
        session_id_str: &str,
        current_generation: AuthorityGeneration,
        current_epoch_secs: u64,
    ) -> Result<SessionRecord, String> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "Session state lock poisoned (FailClosed)")?;

        let record = state.sessions.get(session_id_str).cloned().ok_or_else(|| {
            "InvalidSession: Session ID not found or already terminated".to_string()
        })?;

        // 1. Check expiration
        if record.expires_epoch_secs <= current_epoch_secs {
            state.sessions.remove(session_id_str);
            return Err("SessionExpired: Session TTL has lapsed".into());
        }

        // 2. Check authority generation alignment
        if record.authority_generation != current_generation {
            state.sessions.remove(session_id_str);
            return Err(format!(
                "AuthorityGenerationMismatch: Session generation {} is stale (current {})",
                record.authority_generation.get(),
                current_generation.get()
            ));
        }

        // 3. Check principal revocation
        if state
            .revoked_principals
            .contains(&record.principal.id().as_str())
        {
            state.sessions.remove(session_id_str);
            return Err("PrincipalRevoked: Session terminated due to principal revocation".into());
        }

        // 4. Check device revocation
        if let Some(ref dev) = record.bound_device {
            if state.revoked_devices.contains(dev.as_str()) {
                state.sessions.remove(session_id_str);
                return Err(
                    "DeviceRevoked: Session terminated due to bound device revocation".into(),
                );
            }
        }

        // 5. Strict audience check (fail-closed against unauthorized audience tokens)
        if record.audience.trim().is_empty() {
            state.sessions.remove(session_id_str);
            return Err("EmptyAudience: Session record has empty audience (FailClosed)".into());
        }
        if record.audience != crate::primitives::CANONICAL_AUDIENCE && record.audience != "tara_api"
        {
            state.sessions.remove(session_id_str);
            return Err(format!(
                "AudienceMismatch: Session audience '{}' is not an authorized server audience (FailClosed)",
                record.audience
            ));
        }

        Ok(record)
    }

    /// Issue a one-time challenge nonce bound to an action and/or session with bounded capacity (ID-16).
    pub fn create_challenge(
        &self,
        session_id: Option<SessionId>,
        bound_action: Option<SecurityAction>,
    ) -> Result<ChallengeId, String> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "Session state lock poisoned (FailClosed)")?;

        // Evict expired challenges automatically (ID-16)
        state
            .challenges
            .retain(|_, c| c.created_at.elapsed() < c.ttl);

        if state.challenges.len() >= MAX_ACTIVE_CHALLENGES {
            return Err(format!(
                "ChallengeStorageCapacityExceeded: Active challenge count reached limit ({})",
                MAX_ACTIVE_CHALLENGES
            ));
        }

        let mut rand_bytes = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut rand_bytes);
        let challenge_id = ChallengeId::new(hex::encode(rand_bytes))?;

        let record = ChallengeRecord {
            challenge_id: challenge_id.clone(),
            session_id,
            bound_action,
            created_at: Instant::now(),
            ttl: self.default_challenge_ttl,
        };

        state
            .challenges
            .insert(challenge_id.as_str().to_string(), record);

        Ok(challenge_id)
    }

    /// Atomically consume a challenge nonce: strictly single-use to prevent replay attacks.
    pub fn consume_challenge(
        &self,
        challenge_id_str: &str,
        session_id: Option<&SessionId>,
        action: Option<&SecurityAction>,
    ) -> Result<(), String> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "Session state lock poisoned (FailClosed)")?;

        // Atomic removal
        let record = state.challenges.remove(challenge_id_str).ok_or_else(|| {
            "ChallengeReplayDetected: Challenge nonce does not exist or has already been consumed".to_string()
        })?;

        // Check freshness
        if record.created_at.elapsed() >= record.ttl {
            return Err("ChallengeExpired: Nonce lifetime exceeded TTL".into());
        }

        // Check session binding if bound
        if let Some(ref bound_s) = record.session_id {
            match session_id {
                Some(s) if s == bound_s => {}
                _ => {
                    return Err(
                        "ChallengeBindingViolation: Challenge is bound to a different session"
                            .into(),
                    )
                }
            }
        }

        // Check action binding if bound
        if let Some(ref bound_a) = record.bound_action {
            match action {
                Some(a) if a == bound_a => {}
                _ => {
                    return Err(
                        "ChallengeBindingViolation: Challenge is bound to a different action"
                            .into(),
                    )
                }
            }
        }

        Ok(())
    }

    /// Revoke principal and atomically terminate all associated sessions (ID-11).
    pub fn revoke_principal(&self, principal_id: &PrincipalId) -> Result<(), String> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "Session state lock poisoned (FailClosed)")?;
        state.revoked_principals.insert(principal_id.as_str());
        state
            .sessions
            .retain(|_, s| s.principal.id() != principal_id);
        self.persist_revocations(&state);
        Ok(())
    }

    /// Revoke device and atomically terminate all associated sessions (ID-17).
    pub fn revoke_device(&self, device_id: &DeviceId) -> Result<(), String> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "Session state lock poisoned (FailClosed)")?;
        state.revoked_devices.insert(device_id.as_str().to_string());
        state
            .sessions
            .retain(|_, s| s.bound_device.as_ref() != Some(device_id));
        self.persist_revocations(&state);
        Ok(())
    }

    /// Check if a principal is revoked (Fail-closed).
    pub fn is_principal_revoked(&self, principal_id: &PrincipalId) -> bool {
        let state = match self.state.lock() {
            Ok(s) => s,
            Err(_) => return true, // Fail-closed on poisoned lock
        };
        state.revoked_principals.contains(&principal_id.as_str())
    }

    /// Check if a device is revoked (Fail-closed).
    pub fn is_device_revoked(&self, device_id: &DeviceId) -> bool {
        let state = match self.state.lock() {
            Ok(s) => s,
            Err(_) => return true, // Fail-closed on poisoned lock
        };
        state.revoked_devices.contains(device_id.as_str())
    }

    /// Explicitly terminate and revoke a single session ID.
    pub fn revoke_session(&self, session_id: &str) -> Result<bool, String> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "Session state lock poisoned (FailClosed)")?;
        Ok(state.sessions.remove(session_id).is_some())
    }
}
