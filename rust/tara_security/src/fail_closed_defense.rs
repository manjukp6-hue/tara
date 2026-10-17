//! Active Anti-Tamper Canary & Tripwire Defense Subsystem with Controlled Recovery (Pillar 6).
//!
//! Threat Model & DoS Mitigation:
//! - An unrecoverable bricking lockdown creates an easy Denial-of-Service attack path for an adversary.
//! - Tripwire violations trigger immediate Security Containment (freezing sensitive keys and actions).
//! - Forensic evidence is preserved automatically.
//! - Containment can be lifted ONLY through a strictly audited, multi-party or Creator-Root recovery
//!   ceremony with explicit acknowledgment.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::cryptography::CryptoUtils;
use crate::primitives::{PrincipalId, VerifiedPrincipal};

/// Status report from Tripwire verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TripwireStatus {
    ArmedAndIntact,
    Contained {
        tripwire_id: String,
        details: String,
        contained_epoch_secs: u64,
        forensic_hash: String,
    },
}

/// A monitored canary resource (file, token, or memory sentinel).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanarySentinel {
    pub tripwire_id: String,
    pub canary_path: Option<PathBuf>,
    pub honeypot_token_hash: Option<String>,
    pub expected_canary_hash: Option<String>,
}

/// Persistent state format for crash-safe tripwire state.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TripwireStateFile {
    sentinels: Vec<CanarySentinel>,
    honeypot_tokens: Vec<String>,
    containment: Option<TripwireStatus>,
}

/// Active Anti-Tamper Tripwire Defense Engine.
pub struct TripwireDefenseEngine {
    sentinels: Arc<Mutex<Vec<CanarySentinel>>>,
    honeypot_tokens: Arc<Mutex<HashSet<String>>>,
    active_containment: Arc<Mutex<Option<TripwireStatus>>>,
    persistence_path: Option<PathBuf>,
}

impl Default for TripwireDefenseEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl TripwireDefenseEngine {
    pub fn new() -> Self {
        Self {
            sentinels: Arc::new(Mutex::new(Vec::new())),
            honeypot_tokens: Arc::new(Mutex::new(HashSet::new())),
            active_containment: Arc::new(Mutex::new(None)),
            persistence_path: None,
        }
    }

    /// Opens or initializes a durable tripwire engine backed by persistent storage.
    pub fn open_or_create_persistent<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let p = path.as_ref().to_path_buf();
        if let Some(parent) = p.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    format!(
                        "Failed to create tripwire directory {}: {}",
                        parent.display(),
                        e
                    )
                })?;
            }
        }

        let mut sentinels = Vec::new();
        let mut tokens = HashSet::new();
        let mut containment = None;

        if p.exists() {
            let data = std::fs::read(&p)
                .map_err(|e| format!("Failed to read tripwire state {}: {}", p.display(), e))?;
            let state: TripwireStateFile = serde_json::from_slice(&data)
                .map_err(|e| format!("Corrupted tripwire state at {}: {}", p.display(), e))?;
            sentinels = state.sentinels;
            tokens = state.honeypot_tokens.into_iter().collect();
            containment = state.containment;
        }

        let engine = Self {
            sentinels: Arc::new(Mutex::new(sentinels)),
            honeypot_tokens: Arc::new(Mutex::new(tokens)),
            active_containment: Arc::new(Mutex::new(containment)),
            persistence_path: Some(p),
        };

        if !engine.persistence_path.as_ref().unwrap().exists() {
            engine.flush_persistent_state()?;
        }

        Ok(engine)
    }

    fn flush_persistent_state(&self) -> Result<(), String> {
        let p = match &self.persistence_path {
            Some(path) => path,
            None => return Ok(()),
        };

        let sentinels = self
            .sentinels
            .lock()
            .map_err(|_| "Sentinels lock poisoned")?
            .clone();
        let tokens: Vec<String> = self
            .honeypot_tokens
            .lock()
            .map_err(|_| "Tokens lock poisoned")?
            .iter()
            .cloned()
            .collect();
        let containment = self
            .active_containment
            .lock()
            .map_err(|_| "Containment lock poisoned")?
            .clone();

        let state = TripwireStateFile {
            sentinels,
            honeypot_tokens: tokens,
            containment,
        };

        let serialized = serde_json::to_vec_pretty(&state)
            .map_err(|e| format!("Serialization error for tripwire state: {e}"))?;

        let temp_path = p.with_extension(format!(
            "tmp.{}.{}",
            std::process::id(),
            rand::random::<u64>()
        ));

        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&temp_path)
                .map_err(|e| format!("Failed to open temp tripwire file: {e}"))?;

            f.write_all(&serialized)
                .map_err(|e| format!("Failed to write tripwire file: {e}"))?;

            f.sync_all()
                .map_err(|e| format!("Failed to sync tripwire file: {e}"))?;
        }

        crate::hardware_root::harden_file_permissions(&temp_path)?;
        std::fs::rename(&temp_path, p).map_err(|e| {
            let _ = std::fs::remove_file(&temp_path);
            format!("Atomic rename failed for tripwire state: {e}")
        })?;
        crate::hardware_root::sync_directory_metadata(p)?;

        Ok(())
    }

    /// Arm a canary file tripwire. The canary file must remain unmodified.
    pub fn arm_file_canary(&self, tripwire_id: &str, file_path: &Path) -> Result<(), String> {
        if !file_path.exists() {
            return Err(format!(
                "Canary file '{}' does not exist to arm",
                file_path.display()
            ));
        }

        let content =
            std::fs::read(file_path).map_err(|e| format!("Failed to read canary file: {e}"))?;
        let hash = CryptoUtils::sha256_hex(&content);

        let sentinel = CanarySentinel {
            tripwire_id: tripwire_id.to_string(),
            canary_path: Some(file_path.to_path_buf()),
            honeypot_token_hash: None,
            expected_canary_hash: Some(hash),
        };

        let mut list = self
            .sentinels
            .lock()
            .map_err(|_| "Tripwire lock poisoned")?;
        list.push(sentinel);
        drop(list);
        self.flush_persistent_state()?;
        Ok(())
    }

    /// Plant a honeypot decoy token.
    pub fn plant_honeypot_token(&self, decoy_token: &str) -> Result<(), String> {
        let hash = CryptoUtils::sha256_hex(decoy_token.as_bytes());
        let mut tokens = self
            .honeypot_tokens
            .lock()
            .map_err(|_| "Honeypot lock poisoned")?;
        tokens.insert(hash);
        drop(tokens);
        self.flush_persistent_state()?;
        Ok(())
    }

    /// Inspect incoming token against honeypot traps.
    pub fn check_token_trap(&self, candidate_token: &str, epoch_secs: u64) -> bool {
        let hash = CryptoUtils::sha256_hex(candidate_token.as_bytes());
        let is_trap = {
            let tokens = self
                .honeypot_tokens
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            tokens.contains(&hash)
        };

        if is_trap {
            let mut containment = self
                .active_containment
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            *containment = Some(TripwireStatus::Contained {
                tripwire_id: "HONEYPOT_TOKEN_BREACH".into(),
                details: "Planted decoy honeypot credential was submitted in authorization header"
                    .into(),
                contained_epoch_secs: epoch_secs,
                forensic_hash: hash,
            });
            drop(containment);
            let _ = self.flush_persistent_state();
            true
        } else {
            false
        }
    }

    /// Scan all armed canaries for tampering.
    pub fn scan_integrity(&self, epoch_secs: u64) -> TripwireStatus {
        let containment = self
            .active_containment
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if let Some(ref st) = *containment {
            return st.clone();
        }
        drop(containment);

        let sentinels_snapshot: Vec<CanarySentinel> = {
            let sentinels = self.sentinels.lock().unwrap_or_else(|p| p.into_inner());
            sentinels.clone()
        };

        for s in sentinels_snapshot.iter() {
            if let (Some(ref path), Some(ref expected_hash)) =
                (&s.canary_path, &s.expected_canary_hash)
            {
                if !path.exists() {
                    let st = TripwireStatus::Contained {
                        tripwire_id: s.tripwire_id.clone(),
                        details: format!("Canary file deleted: {}", path.display()),
                        contained_epoch_secs: epoch_secs,
                        forensic_hash: "FILE_DELETED".into(),
                    };
                    let mut containment = self
                        .active_containment
                        .lock()
                        .unwrap_or_else(|p| p.into_inner());
                    *containment = Some(st.clone());
                    drop(containment);
                    let _ = self.flush_persistent_state();
                    return st;
                }

                match std::fs::read(path) {
                    Ok(bytes) => {
                        let current_hash = CryptoUtils::sha256_hex(&bytes);
                        if !CryptoUtils::constant_time_eq(
                            current_hash.as_bytes(),
                            expected_hash.as_bytes(),
                        ) {
                            let st = TripwireStatus::Contained {
                                tripwire_id: s.tripwire_id.clone(),
                                details: format!("Canary file tampered with: {}", path.display()),
                                contained_epoch_secs: epoch_secs,
                                forensic_hash: current_hash,
                            };
                            let mut containment = self
                                .active_containment
                                .lock()
                                .unwrap_or_else(|p| p.into_inner());
                            *containment = Some(st.clone());
                            drop(containment);
                            let _ = self.flush_persistent_state();
                            return st;
                        }
                    }
                    Err(e) => {
                        let st = TripwireStatus::Contained {
                            tripwire_id: s.tripwire_id.clone(),
                            details: format!("I/O failure reading canary: {}", e),
                            contained_epoch_secs: epoch_secs,
                            forensic_hash: "IO_ERROR".into(),
                        };
                        let mut containment = self
                            .active_containment
                            .lock()
                            .unwrap_or_else(|p| p.into_inner());
                        *containment = Some(st.clone());
                        drop(containment);
                        let _ = self.flush_persistent_state();
                        return st;
                    }
                }
            }
        }

        TripwireStatus::ArmedAndIntact
    }

    /// Returns whether the system is currently under tripwire containment.
    pub fn is_contained(&self) -> bool {
        let containment = self
            .active_containment
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        containment.is_some()
    }

    /// Audited Recovery from Containment:
    /// Requires verified Creator Root Principal authority and explicit forensic acknowledgment.
    /// Resets the containment state and re-arms the defense system cleanly.
    pub fn recover_from_containment(
        &self,
        principal: &VerifiedPrincipal,
        forensic_acknowledgment: &str,
    ) -> Result<PrincipalId, String> {
        if !principal.is_creator_root() {
            return Err(
                "RecoveryDenied: Lifting tripwire containment requires Creator Root authority"
                    .into(),
            );
        }

        if forensic_acknowledgment.trim().len() < 16 {
            return Err("RecoveryDenied: Forensic incident review acknowledgment must be at least 16 characters".into());
        }

        let mut containment = self
            .active_containment
            .lock()
            .map_err(|_| "Containment lock poisoned")?;
        *containment = None;
        drop(containment);
        self.flush_persistent_state()?;

        Ok(principal.id().clone())
    }
}
