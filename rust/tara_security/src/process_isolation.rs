//! Operating System Process Isolation Architecture & IPC Security Boundary (Pillar 3).
//!
//! Threat Model & Architecture:
//! - A Rust crate boundary is a software design abstraction; it is NOT an OS security boundary.
//! - For true defense-in-depth, TARA partitions components across dedicated OS processes running
//!   with distinct privileges and sandboxing profiles:
//!
//!   1. `InferenceEngineProcess`:
//!      - Privilege: Lowest privilege (Windows AppContainer / Linux unshare/seccomp).
//!      - Capabilities: Pure compute (CPU/GPU tensor math). Zero network, zero filesystem write.
//!   2. `NetworkGatewayProcess`:
//!      - Privilege: Network listener only.
//!      - Capabilities: Ingests TLS traffic. Zero direct access to hardware TPM or vault keys.
//!   3. `SecurityAuthorityProcess`:
//!      - Privilege: High privilege service.
//!      - Capabilities: Exclusive owner of TPM 2.0 / DPAPI keys, authority generation, and root ceremonies.
//!   4. `StorageBrokerProcess`:
//!      - Privilege: Controlled filesystem mediator.
//!      - Capabilities: Exclusive handle-relative file reader/writer with canonical traversal verification.
//!
//! - Inter-Process Communication (IPC):
//!   - Anonymous pipes / Unix domain sockets authenticated via mutually signed capability tokens.
//!   - A compromise in the AI model execution process CANNOT exfiltrate or mutate authority keys!

use serde::{Deserialize, Serialize};

use crate::cryptography::CryptoUtils;
use crate::zero_trust::CapabilityToken;

/// Dedicated OS Process execution boundaries for TARA subsystems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProcessRole {
    InferenceEngine,
    NetworkGateway,
    SecurityAuthority,
    StorageBroker,
}

impl ProcessRole {
    pub fn allowed_privilege_level(&self) -> &'static str {
        match self {
            Self::InferenceEngine => "SE_RESTRICTED_APP_CONTAINER",
            Self::NetworkGateway => "SE_NETWORK_SERVICE_ONLY",
            Self::SecurityAuthority => "SE_ISOLATED_SECURITY_SERVICE",
            Self::StorageBroker => "SE_STORAGE_MEDIAS_ONLY",
        }
    }
}

/// A mutually authenticated IPC message envelope exchanged across process boundaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcSecurityEnvelope {
    pub sender_role: ProcessRole,
    pub target_role: ProcessRole,
    pub message_nonce: String,
    pub timestamp_epoch_secs: u64,
    pub capability_token: CapabilityToken,
    pub payload_bytes: Vec<u8>,
    pub envelope_hmac_hex: String,
}

impl IpcSecurityEnvelope {
    /// Constructs a cryptographically authenticated IPC envelope.
    pub fn construct(
        sender_role: ProcessRole,
        target_role: ProcessRole,
        capability_token: CapabilityToken,
        payload_bytes: Vec<u8>,
        shared_ipc_key: &[u8],
        epoch_secs: u64,
    ) -> Result<Self, String> {
        let message_nonce = hex::encode(rand::random::<[u8; 16]>());

        let mut preimage = Vec::new();
        preimage.extend_from_slice(sender_role.allowed_privilege_level().as_bytes());
        preimage.extend_from_slice(target_role.allowed_privilege_level().as_bytes());
        preimage.extend_from_slice(message_nonce.as_bytes());
        preimage.extend_from_slice(&epoch_secs.to_le_bytes());
        let cap_bytes = serde_json::to_vec(&capability_token)
            .map_err(|e| format!("Serialization error: {e}"))?;
        preimage.extend_from_slice(&cap_bytes);
        preimage.extend_from_slice(&payload_bytes);

        let hmac_sig = CryptoUtils::hmac_sha256(shared_ipc_key, &preimage)?;

        Ok(Self {
            sender_role,
            target_role,
            message_nonce,
            timestamp_epoch_secs: epoch_secs,
            capability_token,
            payload_bytes,
            envelope_hmac_hex: hex::encode(hmac_sig),
        })
    }

    /// Verifies the authenticity and boundary integrity of an incoming IPC message.
    pub fn verify_boundary(
        &self,
        expected_target: ProcessRole,
        shared_ipc_key: &[u8],
        max_clock_skew_secs: u64,
        current_epoch_secs: u64,
    ) -> Result<(), String> {
        // 1. Verify intended target process
        if self.target_role != expected_target {
            return Err(format!(
                "IpcBoundaryViolation: Envelope targeted {:?}, received by {:?}",
                self.target_role, expected_target
            ));
        }

        // 2. Verify clock skew / freshness
        let diff = (current_epoch_secs as i64 - self.timestamp_epoch_secs as i64).unsigned_abs();
        if diff > max_clock_skew_secs {
            return Err(format!(
                "IpcReplayOrSkewViolation: Envelope timestamp skew ({}s) exceeds max allowed",
                diff
            ));
        }

        // 3. Verify cryptographic HMAC signature over payload and capability token
        let mut preimage = Vec::new();
        preimage.extend_from_slice(self.sender_role.allowed_privilege_level().as_bytes());
        preimage.extend_from_slice(self.target_role.allowed_privilege_level().as_bytes());
        preimage.extend_from_slice(self.message_nonce.as_bytes());
        preimage.extend_from_slice(&self.timestamp_epoch_secs.to_le_bytes());
        let cap_bytes = serde_json::to_vec(&self.capability_token)
            .map_err(|e| format!("Serialization error: {e}"))?;
        preimage.extend_from_slice(&cap_bytes);
        preimage.extend_from_slice(&self.payload_bytes);

        let expected_hmac = CryptoUtils::hmac_sha256(shared_ipc_key, &preimage)?;
        let provided_hmac = hex::decode(&self.envelope_hmac_hex).map_err(|e| e.to_string())?;

        if !CryptoUtils::constant_time_eq(&expected_hmac, &provided_hmac) {
            return Err("IpcAuthenticationFailed: Envelope HMAC signature invalid (tampered or spoofed message)".into());
        }

        Ok(())
    }
}

/// Anti-replay cache tracking observed nonces within the clock skew window.
pub struct AntiReplayCache {
    seen_nonces: std::sync::Mutex<std::collections::HashMap<String, u64>>,
}

impl Default for AntiReplayCache {
    fn default() -> Self {
        Self::new()
    }
}

impl AntiReplayCache {
    pub fn new() -> Self {
        Self {
            seen_nonces: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Validates that a message nonce has not been previously observed and records it.
    /// Purges stale nonces outside the clock skew window.
    pub fn check_and_record(
        &self,
        nonce: &str,
        msg_timestamp: u64,
        max_skew_secs: u64,
        current_epoch_secs: u64,
    ) -> Result<(), String> {
        let mut map = self
            .seen_nonces
            .lock()
            .map_err(|_| "Replay cache mutex poisoned")?;

        // Purge expired nonces
        let cutoff = current_epoch_secs.saturating_sub(max_skew_secs);
        map.retain(|_, ts| *ts >= cutoff);

        if map.contains_key(nonce) {
            return Err(format!(
                "IpcReplayAttackDetected: Nonce '{}' has already been processed",
                nonce
            ));
        }

        // Strict DoS defense: never evict unexpired nonces to make room
        if map.len() >= 50_000 {
            return Err("ReplayCacheCapacityExhausted: Bounded replay protection cache is full of unexpired nonces; request rejected to preserve replay integrity (FailClosed)".into());
        }

        map.insert(nonce.to_string(), msg_timestamp);
        Ok(())
    }
}

/// Secure IPC Message Bus coordinating authenticated communication across isolated process roles.
pub struct SecureIpcBus {
    mailboxes: std::sync::RwLock<
        std::collections::HashMap<ProcessRole, std::collections::VecDeque<IpcSecurityEnvelope>>,
    >,
    replay_cache: AntiReplayCache,
}

impl Default for SecureIpcBus {
    fn default() -> Self {
        Self::new()
    }
}

impl SecureIpcBus {
    pub fn new() -> Self {
        let mut mailboxes = std::collections::HashMap::new();
        mailboxes.insert(
            ProcessRole::InferenceEngine,
            std::collections::VecDeque::new(),
        );
        mailboxes.insert(
            ProcessRole::NetworkGateway,
            std::collections::VecDeque::new(),
        );
        mailboxes.insert(
            ProcessRole::SecurityAuthority,
            std::collections::VecDeque::new(),
        );
        mailboxes.insert(
            ProcessRole::StorageBroker,
            std::collections::VecDeque::new(),
        );

        Self {
            mailboxes: std::sync::RwLock::new(mailboxes),
            replay_cache: AntiReplayCache::new(),
        }
    }

    /// Dispatches an authenticated envelope into the recipient role's queue.
    pub fn dispatch(&self, envelope: IpcSecurityEnvelope) -> Result<(), String> {
        let mut guard = self
            .mailboxes
            .write()
            .map_err(|_| "Mailbox lock poisoned")?;
        let queue = guard.get_mut(&envelope.target_role).ok_or_else(|| {
            format!(
                "No mailbox registered for target role {:?}",
                envelope.target_role
            )
        })?;
        queue.push_back(envelope);
        Ok(())
    }

    /// Receives, verifies, and anti-replay validates the next pending envelope for a target role.
    pub fn receive(
        &self,
        target_role: ProcessRole,
        shared_ipc_key: &[u8],
        max_clock_skew_secs: u64,
        current_epoch_secs: u64,
    ) -> Result<Option<IpcSecurityEnvelope>, String> {
        let mut guard = self
            .mailboxes
            .write()
            .map_err(|_| "Mailbox lock poisoned")?;
        let queue = guard
            .get_mut(&target_role)
            .ok_or_else(|| format!("No mailbox registered for role {:?}", target_role))?;

        if let Some(envelope) = queue.pop_front() {
            // 1. Verify cryptographic boundary and intended recipient
            envelope.verify_boundary(
                target_role,
                shared_ipc_key,
                max_clock_skew_secs,
                current_epoch_secs,
            )?;

            // 2. Anti-replay verification
            self.replay_cache.check_and_record(
                &envelope.message_nonce,
                envelope.timestamp_epoch_secs,
                max_clock_skew_secs,
                current_epoch_secs,
            )?;

            Ok(Some(envelope))
        } else {
            Ok(None)
        }
    }

    /// Number of pending messages for a given process role.
    pub fn pending_count(&self, role: ProcessRole) -> usize {
        if let Ok(guard) = self.mailboxes.read() {
            guard.get(&role).map(|q| q.len()).unwrap_or(0)
        } else {
            0
        }
    }
}

/// Dedicated endpoint handle bound to a specific ProcessRole.
pub struct IpcEndpoint {
    role: ProcessRole,
    bus: std::sync::Arc<SecureIpcBus>,
    shared_key: Vec<u8>,
}

impl IpcEndpoint {
    pub fn new(role: ProcessRole, bus: std::sync::Arc<SecureIpcBus>, shared_key: Vec<u8>) -> Self {
        Self {
            role,
            bus,
            shared_key,
        }
    }

    pub fn role(&self) -> ProcessRole {
        self.role
    }

    /// Constructs, signs, and dispatches a secure IPC message to another role.
    pub fn send(
        &self,
        target_role: ProcessRole,
        capability_token: CapabilityToken,
        payload_bytes: Vec<u8>,
        epoch_secs: u64,
    ) -> Result<IpcSecurityEnvelope, String> {
        let envelope = IpcSecurityEnvelope::construct(
            self.role,
            target_role,
            capability_token,
            payload_bytes,
            &self.shared_key,
            epoch_secs,
        )?;

        self.bus.dispatch(envelope.clone())?;
        Ok(envelope)
    }

    /// Receives and validates the next incoming message for this endpoint.
    pub fn receive(
        &self,
        max_clock_skew_secs: u64,
        current_epoch_secs: u64,
    ) -> Result<Option<IpcSecurityEnvelope>, String> {
        self.bus.receive(
            self.role,
            &self.shared_key,
            max_clock_skew_secs,
            current_epoch_secs,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipc_envelope_tampered_capability_token_fails_verification() {
        let root_key = [7u8; 32];
        let token = CapabilityToken::mint_root(&root_key, "token_priv_001").unwrap();
        let shared_key = [9u8; 32];

        let mut envelope = IpcSecurityEnvelope::construct(
            ProcessRole::SecurityAuthority,
            ProcessRole::InferenceEngine,
            token,
            b"trusted_payload".to_vec(),
            &shared_key,
            1000,
        )
        .unwrap();

        assert!(envelope
            .verify_boundary(ProcessRole::InferenceEngine, &shared_key, 60, 1000)
            .is_ok());

        envelope.capability_token.token_id = "token_swapped_evil".to_string();

        let err = envelope.verify_boundary(ProcessRole::InferenceEngine, &shared_key, 60, 1000);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("Envelope HMAC signature invalid"));
    }
}
