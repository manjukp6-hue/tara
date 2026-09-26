//! Hardened platform-secure encrypted storage for cryptographic keys in Rust.
//! Enforces exact parity with Python TARA/ACCESS/crypto/secure_storage.py:
//! - Scrypt KDF: N=131072 (2^17), r=8, p=1, 32-byte key
//! - AES-256-GCM AEAD: 96-bit (12-byte) random nonces, Associated Data binding key_id
//! - Windows DPAPI via native crypt32.dll CryptProtectData / CryptUnprotectData
//! - Zero plaintext keys on disk; atomic crash-safe file writes
//! - Rejection of empty passphrases

use std::fs;
use std::path::{Path, PathBuf};
use aes_gcm::{Aes256Gcm, KeyInit, aead::{Aead, Payload}};
use aes_gcm::Nonce;
use rand::RngCore;
use serde_json::{json, Value};

#[cfg(windows)]
pub mod dpapi {
    use std::ptr;

    #[repr(C)]
    struct DataBlob {
        cb_data: u32,
        pb_data: *mut u8,
    }

    #[link(name = "crypt32")]
    extern "system" {
        fn CryptProtectData(
            pDataIn: *const DataBlob,
            szDataDescr: *const u16,
            pOptionalEntropy: *const DataBlob,
            pvReserved: *mut std::ffi::c_void,
            pPromptStruct: *mut std::ffi::c_void,
            dwFlags: u32,
            pDataOut: *mut DataBlob,
        ) -> i32;

        fn CryptUnprotectData(
            pDataIn: *const DataBlob,
            ppszDataDescr: *mut *mut u16,
            pOptionalEntropy: *const DataBlob,
            pvReserved: *mut std::ffi::c_void,
            pPromptStruct: *mut std::ffi::c_void,
            dwFlags: u32,
            pDataOut: *mut DataBlob,
        ) -> i32;
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn LocalFree(hMem: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
    }

    pub fn protect_bytes(plaintext: &[u8]) -> Result<Vec<u8>, String> {
        if plaintext.is_empty() {
            return Ok(Vec::new());
        }
        let in_blob = DataBlob {
            cb_data: plaintext.len() as u32,
            pb_data: plaintext.as_ptr() as *mut u8,
        };
        let mut out_blob = DataBlob {
            cb_data: 0,
            pb_data: ptr::null_mut(),
        };
        let res = unsafe {
            CryptProtectData(
                &in_blob,
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
                ptr::null_mut(),
                0x1, // CRYPTPROTECT_UI_FORBIDDEN
                &mut out_blob,
            )
        };
        if res == 0 {
            return Err("CryptProtectData failed".to_string());
        }
        let slice = unsafe {
            std::slice::from_raw_parts(out_blob.pb_data, out_blob.cb_data as usize).to_vec()
        };
        unsafe {
            LocalFree(out_blob.pb_data as *mut std::ffi::c_void);
        }
        Ok(slice)
    }

    pub fn unprotect_bytes(ciphertext: &[u8]) -> Result<Vec<u8>, String> {
        if ciphertext.is_empty() {
            return Ok(Vec::new());
        }
        let in_blob = DataBlob {
            cb_data: ciphertext.len() as u32,
            pb_data: ciphertext.as_ptr() as *mut u8,
        };
        let mut out_blob = DataBlob {
            cb_data: 0,
            pb_data: ptr::null_mut(),
        };
        let res = unsafe {
            CryptUnprotectData(
                &in_blob,
                ptr::null_mut(),
                ptr::null(),
                ptr::null_mut(),
                ptr::null_mut(),
                0x1, // CRYPTPROTECT_UI_FORBIDDEN
                &mut out_blob,
            )
        };
        if res == 0 {
            return Err("CryptUnprotectData failed".to_string());
        }
        let slice = unsafe {
            std::slice::from_raw_parts(out_blob.pb_data, out_blob.cb_data as usize).to_vec()
        };
        unsafe {
            LocalFree(out_blob.pb_data as *mut std::ffi::c_void);
        }
        Ok(slice)
    }
}

#[cfg(not(windows))]
pub mod dpapi {
    pub fn protect_bytes(plaintext: &[u8]) -> Result<Vec<u8>, String> {
        Ok(plaintext.to_vec())
    }
    pub fn unprotect_bytes(ciphertext: &[u8]) -> Result<Vec<u8>, String> {
        Ok(ciphertext.to_vec())
    }
}

pub struct SecureKeyStorage {
    pub storage_dir: PathBuf,
}

impl SecureKeyStorage {
    pub const KEY_SIZE_BYTES: usize = 32;
    pub const NONCE_SIZE_BYTES: usize = 12;

    pub fn new<P: AsRef<Path>>(storage_dir: P) -> Self {
        let dir = storage_dir.as_ref().to_path_buf();
        let _ = fs::create_dir_all(&dir);
        Self { storage_dir: dir }
    }

    /// Derives 256-bit key using Scrypt (N=131072, r=8, p=1).
    pub fn derive_scrypt_key(passphrase: &str, salt: &[u8]) -> Result<[u8; 32], String> {
        if passphrase.trim().is_empty() {
            return Err("Passphrase cannot be empty".to_string());
        }
        // log_n = 17 => 2^17 = 131072, r = 8, p = 1, len = 32
        let params = scrypt::Params::new(17, 8, 1, Self::KEY_SIZE_BYTES)
            .map_err(|e| format!("Invalid Scrypt params: {:?}", e))?;
        let mut derived = [0u8; 32];
        scrypt::scrypt(passphrase.as_bytes(), salt, &params, &mut derived)
            .map_err(|e| format!("Scrypt derivation failed: {:?}", e))?;
        Ok(derived)
    }

    /// Stores private key in Version 3 keystore (Scrypt + AES-256-GCM + DPAPI).
    pub fn store_private_key_modern(
        &self,
        key_id: &str,
        private_bytes: &[u8],
        passphrase: &str,
        use_dpapi: bool,
    ) -> Result<PathBuf, String> {
        if passphrase.trim().is_empty() {
            return Err("Passphrase cannot be empty for key storage".to_string());
        }
        let mut salt = [0u8; 16];
        rand::thread_rng().fill_bytes(&mut salt);

        let mut nonce_bytes = [0u8; Self::NONCE_SIZE_BYTES];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);

        let derived_key = Self::derive_scrypt_key(passphrase, &salt)?;

        // AES-256-GCM encryption with Associated Data (key_id)
        let cipher = Aes256Gcm::new_from_slice(&derived_key)
            .map_err(|e| format!("Failed to create AES-GCM cipher: {:?}", e))?;
        let nonce = Nonce::from_slice(&nonce_bytes);
        let payload = Payload {
            msg: private_bytes,
            aad: key_id.as_bytes(),
        };
        let ciphertext = cipher.encrypt(nonce, payload)
            .map_err(|e| format!("AES-GCM encryption failed: {:?}", e))?;

        let mut dpapi_protected = false;
        let mut dpapi_blob = None;
        if use_dpapi && cfg!(windows) {
            match dpapi::protect_bytes(&derived_key) {
                Ok(enc) => {
                    dpapi_blob = Some(hex::encode(enc));
                    dpapi_protected = true;
                }
                Err(_) => {
                    dpapi_protected = false;
                }
            }
        }

        let keystore_json = json!({
            "key_id": key_id,
            "aead": "AES-256-GCM",
            "kdf": "Scrypt",
            "kdf_params": {
                "n": 131072,
                "r": 8,
                "p": 1,
                "salt": hex::encode(salt),
                "length": Self::KEY_SIZE_BYTES
            },
            "nonce": hex::encode(nonce_bytes),
            "ciphertext": hex::encode(ciphertext),
            "dpapi_protected": dpapi_protected,
            "dpapi_blob": dpapi_blob,
            "version": 3
        });

        let target_path = self.storage_dir.join(format!("{}.keystore", key_id));
        let temp_path = self.storage_dir.join(format!("{}.keystore.tmp.{}", key_id, rand::random::<u32>()));

        fs::write(&temp_path, serde_json::to_string_pretty(&keystore_json).map_err(|e| e.to_string())?)
            .map_err(|e| format!("Failed to write temp keystore: {}", e))?;

        fs::rename(&temp_path, &target_path)
            .map_err(|e| format!("Failed to atomically rename keystore: {}", e))?;

        Ok(target_path)
    }

    /// Loads private key from Version 3 keystore.
    /// Never accepts empty passphrases.
    pub fn load_private_key(
        &self,
        key_id: &str,
        passphrase: Option<&str>,
    ) -> Result<Vec<u8>, String> {
        let keystore_path = self.storage_dir.join(format!("{}.keystore", key_id));
        if !keystore_path.exists() {
            return Err(format!("Keystore for '{}' does not exist", key_id));
        }

        let raw = fs::read_to_string(&keystore_path)
            .map_err(|e| format!("Failed to read keystore: {}", e))?;
        let payload: Value = serde_json::from_str(&raw)
            .map_err(|e| format!("Failed to parse keystore JSON: {}", e))?;

        let version = payload.get("version").and_then(|v| v.as_u64()).unwrap_or(1);
        if version != 3 {
            return Err(format!("Unsupported keystore version: {}", version));
        }

        let nonce_hex = payload.get("nonce").and_then(|v| v.as_str())
            .ok_or_else(|| "Missing nonce in keystore".to_string())?;
        let ciphertext_hex = payload.get("ciphertext").and_then(|v| v.as_str())
            .ok_or_else(|| "Missing ciphertext in keystore".to_string())?;

        let nonce_bytes = hex::decode(nonce_hex)
            .map_err(|e| format!("Invalid nonce hex: {}", e))?;
        let ciphertext_bytes = hex::decode(ciphertext_hex)
            .map_err(|e| format!("Invalid ciphertext hex: {}", e))?;

        let mut derived_key: Option<[u8; 32]> = None;

        // 1. Try Scrypt if passphrase provided
        if let Some(pp) = passphrase {
            if !pp.trim().is_empty() {
                let kdf_params = payload.get("kdf_params").ok_or_else(|| "Missing kdf_params".to_string())?;
                let salt_hex = kdf_params.get("salt").and_then(|v| v.as_str()).unwrap_or("");
                let salt = hex::decode(salt_hex).map_err(|e| format!("Invalid salt: {}", e))?;
                derived_key = Some(Self::derive_scrypt_key(pp, &salt)?);
            }
        }

        // 2. Try DPAPI OS unwrap if no valid passphrase provided but DPAPI protected
        if derived_key.is_none() {
            let dpapi_prot = payload.get("dpapi_protected").and_then(|v| v.as_bool()).unwrap_or(false);
            if dpapi_prot && cfg!(windows) {
                if let Some(blob_hex) = payload.get("dpapi_blob").and_then(|v| v.as_str()) {
                    if let Ok(blob_bytes) = hex::decode(blob_hex) {
                        if let Ok(unprotected) = dpapi::unprotect_bytes(&blob_bytes) {
                            if unprotected.len() == 32 {
                                let mut arr = [0u8; 32];
                                arr.copy_from_slice(&unprotected);
                                derived_key = Some(arr);
                            }
                        }
                    }
                }
            }
        }

        let key = derived_key.ok_or_else(|| "Failed to unlock keystore: invalid passphrase or unprotect failure".to_string())?;

        let cipher = Aes256Gcm::new_from_slice(&key)
            .map_err(|e| format!("AES-GCM init failed: {:?}", e))?;
        let nonce = Nonce::from_slice(&nonce_bytes);
        let aad = key_id.as_bytes();
        let payload = Payload {
            msg: &ciphertext_bytes,
            aad,
        };

        let decrypted = cipher.decrypt(nonce, payload)
            .map_err(|_| "Keystore decryption failed: authentication tag mismatch or wrong key".to_string())?;

        Ok(decrypted)
    }
}
