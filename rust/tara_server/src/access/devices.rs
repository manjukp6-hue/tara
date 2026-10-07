//! Per-device cryptographic identity and enrollment service.
//! Every authorized device generates its own local key pair and receives a unique
//! Device ID (e.g. TARA-DEVICE-001). Creator root private key is never shared across devices.

use ed25519_dalek::{Signature, SigningKey, VerifyingKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceKeyPair {
    pub public_key_hex: String,
    pub private_key_hex: String,
}

pub struct DeviceCrypto;

impl DeviceCrypto {
    pub fn generate_device_keypair() -> DeviceKeyPair {
        let mut csprng = OsRng;
        let signing_key = SigningKey::generate(&mut csprng);
        let verifying_key = signing_key.verifying_key();
        DeviceKeyPair {
            public_key_hex: hex::encode(verifying_key.as_bytes()),
            private_key_hex: hex::encode(signing_key.to_bytes()),
        }
    }

    pub fn sign_message(private_key_hex: &str, message: &[u8]) -> Result<String, String> {
        let priv_bytes = hex::decode(private_key_hex).map_err(|e| e.to_string())?;
        if priv_bytes.len() != 32 {
            return Err("Invalid private key length".into());
        }
        let mut key_arr = [0u8; 32];
        key_arr.copy_from_slice(&priv_bytes);
        let signing_key = SigningKey::from_bytes(&key_arr);
        use ed25519_dalek::Signer;
        let signature = signing_key.sign(message);
        Ok(hex::encode(signature.to_bytes()))
    }

    pub fn verify_signature(public_key_hex: &str, message: &[u8], signature_hex: &str) -> bool {
        let pub_bytes = match hex::decode(public_key_hex) {
            Ok(b) if b.len() == 32 => {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&b);
                arr
            }
            _ => return false,
        };

        let sig_bytes = match hex::decode(signature_hex) {
            Ok(b) if b.len() == 64 => {
                let mut arr = [0u8; 64];
                arr.copy_from_slice(&b);
                arr
            }
            _ => return false,
        };

        if let Ok(vk) = VerifyingKey::from_bytes(&pub_bytes) {
            let sig = Signature::from_bytes(&sig_bytes);
            return vk.verify_strict(message, &sig).is_ok();
        }
        false
    }
}

pub struct EnrollmentService {
    active_challenges: Mutex<HashMap<String, (String, Instant)>>,
}

impl Default for EnrollmentService {
    fn default() -> Self {
        Self::new()
    }
}

impl EnrollmentService {
    pub fn new() -> Self {
        Self {
            active_challenges: Mutex::new(HashMap::new()),
        }
    }

    pub fn create_enrollment_challenge(&self, device_name: &str) -> Value {
        let mut rng = OsRng;
        let mut rand_bytes = [0u8; 32];
        use rand::RngCore;
        rng.fill_bytes(&mut rand_bytes);
        let token = hex::encode(rand_bytes);

        let mut ch = self.active_challenges.lock().unwrap();
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
        let mut ch = self.active_challenges.lock().unwrap();
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
