//! Per-device cryptographic identity and enrollment service.
//!
//! Rule compliance:
//! - Every authorized device generates its own local key pair and receives a unique
//!   Device ID (e.g. TARA-DEV-<hex>). Creator root private key is never shared across devices.
//! - Enforces proof-of-possession for enrollment, memory zeroization, and strict public key validation.
//! - Persistent authorized device registry with hardened atomic writes.

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use zeroize::Zeroize;

use crate::persistence::atomic_write_file_hardened;

pub const CANONICAL_CREATOR_ID: &str = "ROOT_OPERATOR";

/// Device keypair with memory zeroization on drop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceKeyPair {
    pub public_key_hex: String,
    #[serde(skip_serializing)]
    private_key_hex: String,
}

impl DeviceKeyPair {
    pub fn new(public_key_hex: String, private_key_hex: String) -> Self {
        Self {
            public_key_hex,
            private_key_hex,
        }
    }

    pub fn private_key_hex(&self) -> &str {
        &self.private_key_hex
    }
}

impl Drop for DeviceKeyPair {
    fn drop(&mut self) {
        self.private_key_hex.zeroize();
    }
}

pub struct DeviceCrypto;

impl DeviceCrypto {
    pub fn generate_device_keypair() -> DeviceKeyPair {
        let mut csprng = OsRng;
        let signing_key = SigningKey::generate(&mut csprng);
        let verifying_key = signing_key.verifying_key();
        DeviceKeyPair::new(
            hex::encode(verifying_key.as_bytes()),
            hex::encode(signing_key.to_bytes()),
        )
    }

    pub fn sign_message(private_key_hex: &str, message: &[u8]) -> Result<String, String> {
        let mut priv_bytes = hex::decode(private_key_hex).map_err(|e| e.to_string())?;
        if priv_bytes.len() != 32 {
            return Err("Invalid private key length: expected 32 bytes".into());
        }
        let mut key_arr = [0u8; 32];
        key_arr.copy_from_slice(&priv_bytes);
        priv_bytes.zeroize();
        let signing_key = SigningKey::from_bytes(&key_arr);
        key_arr.zeroize();
        let signature = signing_key.sign(message);
        Ok(hex::encode(signature.to_bytes()))
    }

    pub fn verify_signature_detailed(
        public_key_hex: &str,
        message: &[u8],
        signature_hex: &str,
    ) -> Result<(), String> {
        let pub_bytes = hex::decode(public_key_hex)
            .map_err(|e| format!("Malformed public key hex: {}", e))?;
        if pub_bytes.len() != 32 {
            return Err(format!("Public key must be 32 bytes, got {}", pub_bytes.len()));
        }
        let mut pk_arr = [0u8; 32];
        pk_arr.copy_from_slice(&pub_bytes);

        let sig_bytes = hex::decode(signature_hex)
            .map_err(|e| format!("Malformed signature hex: {}", e))?;
        if sig_bytes.len() != 64 {
            return Err(format!("Signature must be 64 bytes, got {}", sig_bytes.len()));
        }
        let mut sig_arr = [0u8; 64];
        sig_arr.copy_from_slice(&sig_bytes);

        let vk = VerifyingKey::from_bytes(&pk_arr)
            .map_err(|e| format!("Invalid Ed25519 public key: {}", e))?;
        let sig = Signature::from_bytes(&sig_arr);
        vk.verify_strict(message, &sig)
            .map_err(|e| format!("Ed25519 signature verification failed: {}", e))?;
        Ok(())
    }

    pub fn verify_signature(public_key_hex: &str, message: &[u8], signature_hex: &str) -> bool {
        Self::verify_signature_detailed(public_key_hex, message, signature_hex).is_ok()
    }

    pub fn derive_device_id(public_key_hex: &str) -> Result<String, String> {
        let bytes =
            hex::decode(public_key_hex).map_err(|e| format!("Invalid hex public key: {}", e))?;
        if bytes.len() != 32 {
            return Err("Device public key must be exactly 32 bytes".to_string());
        }
        let hash = Sha256::digest(&bytes);
        Ok(format!("TARA-DEV-{}", hex::encode(&hash[..8])))
    }
}

/// Device record in the authorized device registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceRecord {
    pub device_id: String,
    pub creator_id: String,
    pub device_name: String,
    pub device_public_key: String,
    pub status: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceRegistryData {
    pub devices: HashMap<String, DeviceRecord>,
}

impl Default for DeviceRegistryData {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceRegistryData {
    pub fn new() -> Self {
        Self {
            devices: HashMap::new(),
        }
    }

    pub fn register_device(
        &mut self,
        public_key_hex: &str,
        device_name: &str,
        status: &str,
        timestamp_iso: &str,
    ) -> Result<DeviceRecord, String> {
        let device_id = DeviceCrypto::derive_device_id(public_key_hex)?;

        let rec = DeviceRecord {
            device_id: device_id.clone(),
            creator_id: CANONICAL_CREATOR_ID.to_string(),
            device_name: device_name.to_string(),
            device_public_key: public_key_hex.to_string(),
            status: status.to_string(),
            created_at: timestamp_iso.to_string(),
        };
        self.devices.insert(device_id, rec.clone());
        Ok(rec)
    }

    pub fn revoke_device(&mut self, device_id: &str) -> bool {
        if let Some(d) = self.devices.get_mut(device_id) {
            d.status = "REVOKED".to_string();
            true
        } else {
            false
        }
    }

    pub fn revoke_all(&mut self) {
        for d in self.devices.values_mut() {
            d.status = "REVOKED".to_string();
        }
    }

    pub fn list_devices(&self) -> Vec<DeviceRecord> {
        self.devices.values().cloned().collect()
    }
}

/// Durable Identity and Device Manager with atomic persistence.
pub struct IdentityManager {
    pub base_dir: String,
    devices: Mutex<DeviceRegistryData>,
    pub creator_id: String,
}

impl IdentityManager {
    pub fn new(base_dir: &str) -> Self {
        if !Path::new(base_dir).exists() {
            let _ = fs::create_dir_all(base_dir);
        }
        let dev_path = PathBuf::from(base_dir).join("devices.json");
        let registry = if dev_path.exists() {
            match fs::read_to_string(&dev_path) {
                Ok(data) => serde_json::from_str::<HashMap<String, DeviceRecord>>(&data)
                    .map(|devices| DeviceRegistryData { devices })
                    .unwrap_or_else(|_| DeviceRegistryData::new()),
                Err(_) => DeviceRegistryData::new(),
            }
        } else {
            DeviceRegistryData::new()
        };

        Self {
            base_dir: base_dir.to_string(),
            devices: Mutex::new(registry),
            creator_id: CANONICAL_CREATOR_ID.to_string(),
        }
    }

    fn persist_devices(&self, reg: &DeviceRegistryData) -> Result<(), String> {
        let path = PathBuf::from(&self.base_dir).join("devices.json");
        let data = serde_json::to_string_pretty(&reg.devices).map_err(|e| e.to_string())?;
        atomic_write_file_hardened(&path, data.as_bytes())
    }

    pub fn register_device(
        &self,
        public_key_hex: &str,
        device_name: &str,
        status: &str,
        timestamp_iso: &str,
    ) -> Result<DeviceRecord, String> {
        let mut reg = self.devices.lock().map_err(|_| "Devices lock poisoned")?;
        let rec = reg.register_device(public_key_hex, device_name, status, timestamp_iso)?;
        self.persist_devices(&reg)?;
        Ok(rec)
    }

    pub fn revoke_device(&self, device_id: &str) -> bool {
        let mut reg = match self.devices.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        let res = reg.revoke_device(device_id);
        if res {
            let _ = self.persist_devices(&reg);
        }
        res
    }

    pub fn revoke_all_devices(&self) {
        if let Ok(mut reg) = self.devices.lock() {
            reg.revoke_all();
            let _ = self.persist_devices(&reg);
        }
    }

    pub fn list_devices(&self) -> Vec<DeviceRecord> {
        self.devices
            .lock()
            .map(|g| g.list_devices())
            .unwrap_or_default()
    }

    pub fn get_device(&self, device_id: &str) -> Option<DeviceRecord> {
        self.devices
            .lock()
            .ok()?
            .devices
            .get(device_id)
            .cloned()
    }

    pub fn authorize_device(&self, device_id: &str) -> bool {
        let mut reg = match self.devices.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        if let Some(d) = reg.devices.get_mut(device_id) {
            d.status = "AUTHORIZED".to_string();
            let _ = self.persist_devices(&reg);
            true
        } else {
            false
        }
    }

    pub fn get_security_summary(&self) -> Value {
        let devices = self.devices.lock();
        let authorized_count = match devices {
            Ok(ref d) => d
                .devices
                .values()
                .filter(|dev| dev.status == "AUTHORIZED")
                .count(),
            Err(_) => 0,
        };
        json!({
            "creator_id": self.creator_id,
            "authorized_devices": authorized_count,
            "lockdown_state": "NORMAL",
            "base_dir": self.base_dir
        })
    }
}

/// Dynamic Challenge Enrollment Service with anti-replay tracking.
pub struct EnrollmentService {
    active_challenges: Mutex<HashMap<String, (String, Instant)>>,
}

impl Default for EnrollmentService {
    fn default() -> Self {
        Self::new()
    }
}

impl EnrollmentService {
    pub const MAX_CHALLENGES: usize = 1000;

    pub fn new() -> Self {
        Self {
            active_challenges: Mutex::new(HashMap::new()),
        }
    }

    pub fn create_enrollment_challenge(&self, device_name: &str) -> Value {
        let mut rand_bytes = [0u8; 32];
        rand::RngCore::fill_bytes(&mut OsRng, &mut rand_bytes);
        let token = hex::encode(rand_bytes);

        let mut ch = self.active_challenges.lock().unwrap_or_else(|p| p.into_inner());
        // Evict expired challenges
        ch.retain(|_, (_, ts)| ts.elapsed() < Duration::from_secs(300));
        if ch.len() >= Self::MAX_CHALLENGES {
            if let Some(oldest_token) = ch.iter().min_by_key(|(_, (_, ts))| *ts).map(|(k, _)| k.clone()) {
                ch.remove(&oldest_token);
            }
        }
        ch.insert(token.clone(), (device_name.to_string(), Instant::now()));

        json!({
            "status": "CHALLENGE_CREATED",
            "enrollment_token": token,
            "device_name": device_name,
            "expires_in_seconds": 300
        })
    }

    pub fn verify_enrollment(
        &self,
        enrollment_token: &str,
        device_pubkey_hex: &str,
        proof_signature_hex: &str,
    ) -> Result<String, String> {
        let mut ch = self.active_challenges.lock().map_err(|_| "Challenges lock poisoned")?;
        let (device_name, created_at) = ch
            .remove(enrollment_token)
            .ok_or_else(|| "Enrollment token expired or not found".to_string())?;

        if created_at.elapsed() > Duration::from_secs(300) {
            return Err("Enrollment token has expired".to_string());
        }

        let verified = DeviceCrypto::verify_signature(
            device_pubkey_hex,
            enrollment_token.as_bytes(),
            proof_signature_hex,
        );

        if !verified {
            return Err("Device cryptographic signature proof verification failed".to_string());
        }

        Ok(device_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_device_derives_canonical_device_id() {
        let (_signing_key, verifying_key) = crate::cryptography::CryptoUtils::generate_ed25519_keypair();
        let pub_hex = hex::encode(verifying_key.as_bytes());

        let mut reg = DeviceRegistryData::new();
        let rec = reg.register_device(&pub_hex, "TestDevice", "ACTIVE", "2026-10-14T12:00:00Z").unwrap();

        let expected_id = DeviceCrypto::derive_device_id(&pub_hex).unwrap();
        assert_eq!(rec.device_id, expected_id);
        assert!(rec.device_id.starts_with("TARA-DEV-"));
        assert_eq!(rec.device_id.len(), 25);
    }
}

