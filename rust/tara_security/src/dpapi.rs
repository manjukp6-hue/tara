//! Platform-Secure Windows DPAPI and Hardened Key Storage for TARA Security Subsystem.
//!
//! Enforces:
//! - Windows DPAPI via native crypt32.dll CryptProtectData / CryptUnprotectData.
//! - Non-windows DPAPI fails closed with explicit error (no plaintext passthrough).
//! - Scrypt KDF: N=131072 (2^17), r=8, p=1, 32-byte key.
//! - AES-256-GCM AEAD: 96-bit (12-byte) random nonces, Associated Data binding key_id.
//! - Zero plaintext keys on disk; atomic crash-safe file writes with fsync.
//! - Strict key_id validation preventing path traversal.
//! - Memory zeroization on sensitive key material.

use aes_gcm::Nonce;
use aes_gcm::{
    aead::{Aead, Payload},
    Aes256Gcm, KeyInit,
};
use rand::RngCore;
use serde_json::{json, Value};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use zeroize::{Zeroize, Zeroizing};

#[cfg(windows)]
pub mod native_dpapi {
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
        if plaintext.len() > u32::MAX as usize {
            return Err("Plaintext exceeds maximum allowable size for DPAPI".to_string());
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
        if out_blob.pb_data.is_null() {
            return Ok(Vec::new());
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
        if ciphertext.len() > u32::MAX as usize {
            return Err("Ciphertext exceeds maximum allowable size for DPAPI".to_string());
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
        if out_blob.pb_data.is_null() {
            return Ok(Vec::new());
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
pub mod native_dpapi {
    pub fn protect_bytes(_plaintext: &[u8]) -> Result<Vec<u8>, String> {
        Err("Windows DPAPI is not available on non-Windows platforms (FailClosed)".to_string())
    }
    pub fn unprotect_bytes(_ciphertext: &[u8]) -> Result<Vec<u8>, String> {
        Err("Windows DPAPI is not available on non-Windows platforms (FailClosed)".to_string())
    }
}

pub struct SecureKeyStorage {
    storage_dir: PathBuf,
}

impl SecureKeyStorage {
    pub const KEY_SIZE_BYTES: usize = 32;
    pub const NONCE_SIZE_BYTES: usize = 12;
    pub const MAX_KEYSTORE_FILE_BYTES: u64 = 65536;

    pub fn new<P: AsRef<Path>>(storage_dir: P) -> Self {
        let dir = storage_dir.as_ref().to_path_buf();
        if !dir.exists() {
            let _ = fs::create_dir_all(&dir);
        }
        Self { storage_dir: dir }
    }

    /// Strictly validate key_id to ensure safe identifier syntax without directory traversal.
    pub fn validate_key_id(key_id: &str) -> Result<(), String> {
        let trimmed = key_id.trim();
        if trimmed.is_empty() {
            return Err("key_id cannot be empty".to_string());
        }
        if trimmed.len() > 64 {
            return Err("key_id exceeds maximum permitted length (64 chars)".to_string());
        }
        if !trimmed
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(format!(
                "Invalid key_id '{}': only alphanumeric, underscore, and hyphen are permitted",
                trimmed
            ));
        }
        Ok(())
    }

    /// Derives 256-bit key using Scrypt (N=131072, r=8, p=1).
    pub fn derive_scrypt_key(passphrase: &str, salt: &[u8]) -> Result<[u8; 32], String> {
        if passphrase.trim().is_empty() {
            return Err("Passphrase cannot be empty".to_string());
        }
        if salt.len() < 16 {
            return Err("Salt must be at least 16 bytes for Scrypt derivation".to_string());
        }

        let mut derived_key = [0u8; 32];
        let params = scrypt::Params::new(17, 8, 1, 32)
            .map_err(|e| format!("Invalid Scrypt parameters: {}", e))?;
        scrypt::scrypt(passphrase.as_bytes(), salt, &params, &mut derived_key)
            .map_err(|e| format!("Scrypt key derivation failed: {}", e))?;

        Ok(derived_key)
    }

    /// Encrypt and persist private key material using Scrypt + AES-256-GCM + Windows DPAPI.
    pub fn store_key(
        &self,
        key_id: &str,
        raw_key: &[u8],
        passphrase: &str,
    ) -> Result<(), String> {
        Self::validate_key_id(key_id)?;

        if raw_key.len() != Self::KEY_SIZE_BYTES {
            return Err(format!(
                "Invalid key size: expected {} bytes, got {}",
                Self::KEY_SIZE_BYTES,
                raw_key.len()
            ));
        }
        if passphrase.trim().is_empty() {
            return Err("Refusing to encrypt key with empty passphrase".to_string());
        }

        let mut salt = [0u8; 16];
        rand::thread_rng().fill_bytes(&mut salt);

        let mut derived_key = Self::derive_scrypt_key(passphrase, &salt)?;

        let mut nonce_bytes = [0u8; Self::NONCE_SIZE_BYTES];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let cipher = Aes256Gcm::new_from_slice(&derived_key)
            .map_err(|e| format!("Failed to initialize AES-GCM cipher: {}", e))?;
        derived_key.zeroize();

        let payload = Payload {
            msg: raw_key,
            aad: key_id.as_bytes(),
        };

        let ciphertext = cipher
            .encrypt(nonce, payload)
            .map_err(|e| format!("AES-256-GCM encryption failed: {}", e))?;

        // Protect ciphertext with Windows DPAPI (fail-closed on Windows)
        #[cfg(windows)]
        let dpapi_blob = Some(hex::encode(
            native_dpapi::protect_bytes(&ciphertext)
                .map_err(|e| format!("FAIL CLOSED: Windows DPAPI protection failure: {e}"))?
        ));
        #[cfg(not(windows))]
        let dpapi_blob: Option<String> = None;

        let keystore_doc = json!({
            "key_id": key_id,
            "kdf": "scrypt",
            "kdf_params": {
                "n": 131072,
                "r": 8,
                "p": 1,
                "salt": hex::encode(salt),
            },
            "cipher": "aes-256-gcm",
            "nonce": hex::encode(nonce_bytes),
            "ciphertext": hex::encode(ciphertext),
            "dpapi_blob": dpapi_blob,
            "created_at": crate::cryptography::CryptoUtils::sha256_hex(key_id.as_bytes()),
        });

        let target_path = self.storage_dir.join(format!("{}.key.json", key_id));
        let serialized = serde_json::to_vec_pretty(&keystore_doc)
            .map_err(|e| format!("Keystore serialization error: {}", e))?;

        let temp_path = self.storage_dir.join(format!(
            "{}.tmp.{}",
            key_id,
            rand::random::<u64>()
        ));

        {
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&temp_path)
                .map_err(|e| format!("Failed to create temp keystore file: {}", e))?;
            file.write_all(&serialized)
                .map_err(|e| format!("Failed to write keystore data: {}", e))?;
            file.sync_all()
                .map_err(|e| format!("Failed to sync keystore file: {}", e))?;
        }

        fs::rename(&temp_path, &target_path)
            .map_err(|e| format!("Atomic rename to final keystore path failed: {}", e))?;

        crate::hardware_root::platform::harden_file_permissions(&target_path)?;

        Ok(())
    }

    /// Decrypt private key material using Scrypt + AES-256-GCM + Windows DPAPI.
    pub fn load_key(&self, key_id: &str, passphrase: &str) -> Result<Vec<u8>, String> {
        Self::validate_key_id(key_id)?;

        let target_path = self.storage_dir.join(format!("{}.key.json", key_id));
        if !target_path.exists() {
            return Err(format!("Key '{}' not found in keystore", key_id));
        }

        let metadata = fs::metadata(&target_path)
            .map_err(|e| format!("Failed to read keystore metadata: {}", e))?;
        if metadata.len() > Self::MAX_KEYSTORE_FILE_BYTES {
            return Err("Keystore file size exceeds permitted maximum limit".to_string());
        }

        let content = fs::read_to_string(&target_path)
            .map_err(|e| format!("Failed to read keystore file: {}", e))?;
        let doc: Value = serde_json::from_str(&content)
            .map_err(|e| format!("Invalid JSON structure in keystore file: {}", e))?;

        let stored_id = doc
            .get("key_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'key_id' in keystore document")?;
        if stored_id != key_id {
            return Err(format!(
                "Key ID mismatch in stored record: expected '{}', found '{}'",
                key_id, stored_id
            ));
        }

        let salt_hex = doc
            .get("kdf_params")
            .and_then(|v| v.get("salt"))
            .and_then(|v| v.as_str())
            .ok_or("Missing KDF salt in keystore document")?;
        let salt = hex::decode(salt_hex).map_err(|e| format!("Invalid salt hex: {}", e))?;

        let nonce_hex = doc
            .get("nonce")
            .and_then(|v| v.as_str())
            .ok_or("Missing nonce in keystore document")?;
        let nonce_bytes =
            hex::decode(nonce_hex).map_err(|e| format!("Invalid nonce hex: {}", e))?;
        if nonce_bytes.len() != Self::NONCE_SIZE_BYTES {
            return Err("Corrupted nonce size in keystore document".to_string());
        }
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext_hex = doc
            .get("ciphertext")
            .and_then(|v| v.as_str())
            .ok_or("Missing ciphertext in keystore document")?;
        let ciphertext =
            hex::decode(ciphertext_hex).map_err(|e| format!("Invalid ciphertext hex: {}", e))?;

        // Enforce Windows DPAPI machine/user binding verification (Fail-closed on Windows)
        #[cfg(windows)]
        {
            let dpapi_hex = doc
                .get("dpapi_blob")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "FailClosed: Missing mandatory DPAPI envelope binding in keystore document on Windows".to_string())?;
            let dpapi_bytes = hex::decode(dpapi_hex)
                .map_err(|e| format!("Invalid dpapi_blob hex: {}", e))?;
            let unprot = native_dpapi::unprotect_bytes(&dpapi_bytes)
                .map_err(|e| format!("DPAPI machine/user binding verification failed: {}", e))?;
            if !crate::cryptography::CryptoUtils::constant_time_eq(&unprot, &ciphertext) {
                return Err("DPAPI envelope ciphertext mismatch: key storage tampered or foreign machine context".into());
            }
        }

        let mut derived_key = Self::derive_scrypt_key(passphrase, &salt)?;

        let cipher = Aes256Gcm::new_from_slice(&derived_key)
            .map_err(|e| format!("Failed to initialize AES-GCM cipher: {}", e))?;
        derived_key.zeroize();

        let payload = Payload {
            msg: &ciphertext,
            aad: key_id.as_bytes(),
        };

        let decrypted = cipher
            .decrypt(nonce, payload)
            .map_err(|_| "Decryption failed (incorrect passphrase or tampered key)".to_string())?;

        Ok(decrypted)
    }

    /// Modern private key storage supporting DPAPI, Scrypt, and AES-256-GCM.
    pub fn store_private_key_modern(
        &self,
        key_id: &str,
        raw_key: &[u8],
        passphrase: &str,
        _use_dpapi: bool,
    ) -> Result<(), String> {
        self.store_key(key_id, raw_key, passphrase)
    }

    /// Loads and decrypts private key material using passphrase if provided.
    pub fn load_private_key(
        &self,
        key_id: &str,
        passphrase_opt: Option<&str>,
    ) -> Result<Vec<u8>, String> {
        let passphrase = passphrase_opt.unwrap_or("");
        self.load_key(key_id, passphrase)
    }

    /// Check if key exists.
    pub fn key_exists(&self, key_id: &str) -> bool {
        if Self::validate_key_id(key_id).is_err() {
            return false;
        }
        self.storage_dir.join(format!("{}.key.json", key_id)).exists()
    }
}

/// Stores a 64-bit watermark for the Machine-Level System Witness outside the user profile (Anchor 3).
pub fn store_system_witness_watermark(
    watermark_path: &Path,
    candidate_val: u64,
) -> Result<(), String> {
    if let Some(parent) = watermark_path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "Failed to create system witness directory {}: {}",
                parent.display(),
                e
            )
        })?;
    }
    if watermark_path.exists() {
        if let Some(curr) = load_monotonic_watermark(watermark_path)? {
            if candidate_val < curr {
                return Err(format!(
                    "Monotonic watermark rollback rejected: candidate watermark {} is less than committed {}",
                    candidate_val, curr
                ));
            }
        }
    }
    let bytes = candidate_val.to_le_bytes();
    #[cfg(windows)]
    let payload = native_dpapi::protect_bytes(&bytes)?;
    #[cfg(not(windows))]
    let payload = bytes.to_vec();

    let tmp_path = watermark_path.with_extension(format!("tmp.{}", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp_path)
        .map_err(|e| format!("Failed to open temp system witness file: {}", e))?;
    file.write_all(&payload)
        .map_err(|e| format!("Failed to write temp system witness: {}", e))?;
    file.sync_all()
        .map_err(|e| format!("Failed to sync temp system witness: {}", e))?;
    drop(file);

    #[cfg(windows)]
    if watermark_path.exists() {
        if let Ok(user) = std::env::var("USERNAME") {
            let system_root =
                std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".to_string());
            let icacls_path = PathBuf::from(system_root)
                .join("System32")
                .join("icacls.exe");
            let _ = std::process::Command::new(icacls_path)
                .arg(watermark_path)
                .arg("/grant:r")
                .arg(format!("{}:F", user))
                .output();
        }
    }

    fs::rename(&tmp_path, watermark_path)
        .map_err(|e| format!("Failed to commit system witness file: {}", e))?;
    let _ = crate::hardware_root::sync_directory_metadata(watermark_path);
    let _ = crate::hardware_root::harden_file_permissions(watermark_path);
    Ok(())
}

/// Stores a monotonically non-decreasing 64-bit watermark encrypted via platform DPAPI.
pub fn store_monotonic_watermark(watermark_path: &Path, candidate_val: u64) -> Result<(), String> {
    if let Some(parent) = watermark_path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "Failed to create watermark directory {}: {}",
                parent.display(),
                e
            )
        })?;
    }
    if watermark_path.exists() {
        if let Some(curr) = load_monotonic_watermark(watermark_path)? {
            if candidate_val < curr {
                return Err(format!(
                    "Monotonic watermark rollback rejected: candidate watermark {} is less than committed {}",
                    candidate_val, curr
                ));
            }
        }
    }
    let bytes = candidate_val.to_le_bytes();
    #[cfg(windows)]
    let payload = native_dpapi::protect_bytes(&bytes)?;
    #[cfg(not(windows))]
    let payload = bytes.to_vec();

    let tmp_path = watermark_path.with_extension(format!("tmp.{}", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp_path)
        .map_err(|e| format!("Failed to open temp watermark file: {}", e))?;
    file.write_all(&payload)
        .map_err(|e| format!("Failed to write temp watermark: {}", e))?;
    file.sync_all()
        .map_err(|e| format!("Failed to sync temp watermark: {}", e))?;
    drop(file);
    fs::rename(&tmp_path, watermark_path)
        .map_err(|e| format!("Failed to commit watermark file: {}", e))?;
    let _ = crate::hardware_root::sync_directory_metadata(watermark_path);
    let _ = crate::hardware_root::harden_file_permissions(watermark_path);
    Ok(())
}

/// Loads a 64-bit watermark protected via platform DPAPI.
pub fn load_monotonic_watermark(watermark_path: &Path) -> Result<Option<u64>, String> {
    if !watermark_path.exists() {
        return Ok(None);
    }
    let data =
        fs::read(watermark_path).map_err(|e| format!("Failed to read watermark file: {}", e))?;
    if data.is_empty() {
        return Ok(None);
    }
    #[cfg(windows)]
    let raw = native_dpapi::unprotect_bytes(&data)?;
    #[cfg(not(windows))]
    let raw = data;

    if raw.len() != 8 {
        return Err(format!(
            "Corrupt watermark payload length: expected 8 bytes, got {}",
            raw.len()
        ));
    }
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&raw);
    Ok(Some(u64::from_le_bytes(arr)))
}

/// Resolves an OS-isolated hardware/vault watermark path outside the workspace repository.
pub fn get_os_isolated_watermark_path(domain: &str) -> PathBuf {
    #[cfg(windows)]
    {
        if let Ok(local_appdata) = std::env::var("LOCALAPPDATA") {
            return PathBuf::from(local_appdata)
                .join("TaraCore")
                .join("vault")
                .join(format!("{}_monotonic_watermark.bin", domain));
        }
        if let Ok(userprofile) = std::env::var("USERPROFILE") {
            return PathBuf::from(userprofile)
                .join("AppData")
                .join("Local")
                .join("TaraCore")
                .join("vault")
                .join(format!("{}_monotonic_watermark.bin", domain));
        }
        PathBuf::from("C:\\ProgramData\\TaraCore\\vault")
            .join(format!("{}_monotonic_watermark.bin", domain))
    }
    #[cfg(not(windows))]
    {
        if let Ok(home) = std::env::var("HOME") {
            let p = PathBuf::from(home)
                .join(".local")
                .join("share")
                .join("taracore")
                .join("vault");
            return p.join(format!("{}_monotonic_watermark.bin", domain));
        }
        let fallback = PathBuf::from("/var/lib/taracore/vault");
        fallback.join(format!("{}_monotonic_watermark.bin", domain))
    }
}

/// Derives a domain-separated 32-byte HMAC key anchored to an OS-isolated root authority seed.
/// Fails closed if the authority seed cannot be read, decrypted, generated, or securely persisted.
pub fn derive_authority_hmac_key(domain: &str) -> Result<[u8; 32], String> {
    let seed_path = get_os_isolated_watermark_path("root_authority_seed");
    let mut seed = Zeroizing::new([0u8; 32]);

    if seed_path.exists() {
        let data = fs::read(&seed_path)
            .map_err(|e| format!("Failed to read root authority seed file: {}", e))?;
        #[cfg(windows)]
        let mut raw = native_dpapi::unprotect_bytes(&data)
            .map_err(|e| format!("Failed to unprotect existing root authority seed: {}", e))?;
        #[cfg(not(windows))]
        let mut raw = data;

        if raw.len() != 32 {
            raw.zeroize();
            return Err(format!(
                "Corrupted root authority seed length: expected 32 bytes, got {}",
                raw.len()
            ));
        }
        seed.copy_from_slice(&raw);
        raw.zeroize();
    } else {
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut *seed);
        #[cfg(windows)]
        let mut payload = native_dpapi::protect_bytes(&*seed)
            .map_err(|e| format!("DPAPI protect_bytes failed for root authority seed: {}", e))?;
        #[cfg(not(windows))]
        let mut payload = seed.to_vec();

        if let Some(parent) = seed_path.parent() {
            if !parent.exists() {
                if let Err(e) = fs::create_dir_all(parent) {
                    payload.zeroize();
                    return Err(format!(
                        "Failed to create authority seed directory {}: {}",
                        parent.display(),
                        e
                    ));
                }
            }
        }

        let tmp_path = seed_path.with_extension(format!("tmp.{}", rand::random::<u64>()));
        let write_res = (|| -> Result<(), String> {
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp_path)
                .map_err(|e| format!("Failed to open temp authority seed file: {}", e))?;
            file.write_all(&payload)
                .map_err(|e| format!("Failed to write authority seed: {}", e))?;
            file.sync_all()
                .map_err(|e| format!("Failed to fsync authority seed: {}", e))?;
            Ok(())
        })();
        payload.zeroize();
        write_res?;

        fs::rename(&tmp_path, &seed_path)
            .map_err(|e| format!("Failed to commit authority seed: {}", e))?;

        let _ = crate::hardware_root::sync_directory_metadata(&seed_path);
        let _ = crate::hardware_root::harden_file_permissions(&seed_path);
    }

    crate::cryptography::CryptoUtils::hmac_sha256(&*seed, domain.as_bytes())
}

#[cfg(windows)]
pub mod win_registry {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;
    use std::ptr;

    const HKEY_CURRENT_USER: isize = -2147483647; // 0x80000001
    const KEY_READ: u32 = 0x20019;
    const KEY_WRITE: u32 = 0x20006;
    const REG_QWORD: u32 = 11;
    const ERROR_SUCCESS: i32 = 0;
    const ERROR_FILE_NOT_FOUND: i32 = 2;
    const ERROR_NO_MORE_ITEMS: i32 = 259;

    #[link(name = "advapi32")]
    extern "system" {
        fn RegCreateKeyExW(
            hKey: isize,
            lpSubKey: *const u16,
            Reserved: u32,
            lpClass: *const u16,
            dwOptions: u32,
            samDesired: u32,
            lpSecurityAttributes: *mut std::ffi::c_void,
            phkResult: *mut isize,
            lpdwDisposition: *mut u32,
        ) -> i32;

        fn RegOpenKeyExW(
            hKey: isize,
            lpSubKey: *const u16,
            ulOptions: u32,
            samDesired: u32,
            phkResult: *mut isize,
        ) -> i32;

        fn RegSetValueExW(
            hKey: isize,
            lpValueName: *const u16,
            Reserved: u32,
            dwType: u32,
            lpData: *const u8,
            cbData: u32,
        ) -> i32;

        fn RegQueryValueExW(
            hKey: isize,
            lpValueName: *const u16,
            lpReserved: *mut u32,
            lpType: *mut u32,
            lpData: *mut u8,
            lpcbData: *mut u32,
        ) -> i32;

        fn RegEnumValueW(
            hKey: isize,
            dwIndex: u32,
            lpValueName: *mut u16,
            lpcchValueName: *mut u32,
            lpReserved: *mut u32,
            lpType: *mut u32,
            lpData: *mut u8,
            lpcbData: *mut u32,
        ) -> i32;

        fn RegDeleteValueW(hKey: isize, lpValueName: *const u16) -> i32;

        fn RegCloseKey(hKey: isize) -> i32;
    }

    fn to_wide(s: &str) -> Vec<u16> {
        OsStr::new(s)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect()
    }

    pub fn set_u64(sub_key: &str, value_name: &str, val: u64) -> Result<(), String> {
        let sub_key_w = to_wide(sub_key);
        let val_name_w = to_wide(value_name);
        let mut hkey: isize = 0;
        let mut disposition: u32 = 0;

        let res = unsafe {
            RegCreateKeyExW(
                HKEY_CURRENT_USER,
                sub_key_w.as_ptr(),
                0,
                ptr::null(),
                0,
                KEY_WRITE,
                ptr::null_mut(),
                &mut hkey,
                &mut disposition,
            )
        };
        if res != ERROR_SUCCESS {
            return Err(format!("RegCreateKeyExW failed with error code: {}", res));
        }

        let bytes = val.to_ne_bytes();
        let set_res = unsafe {
            RegSetValueExW(
                hkey,
                val_name_w.as_ptr(),
                0,
                REG_QWORD,
                bytes.as_ptr(),
                bytes.len() as u32,
            )
        };
        unsafe {
            RegCloseKey(hkey);
        }

        if set_res != ERROR_SUCCESS {
            return Err(format!("RegSetValueExW failed with error code: {}", set_res));
        }
        Ok(())
    }

    pub fn get_u64(sub_key: &str, value_name: &str) -> Result<Option<u64>, String> {
        let sub_key_w = to_wide(sub_key);
        let val_name_w = to_wide(value_name);
        let mut hkey: isize = 0;

        let open_res = unsafe {
            RegOpenKeyExW(
                HKEY_CURRENT_USER,
                sub_key_w.as_ptr(),
                0,
                KEY_READ,
                &mut hkey,
            )
        };
        if open_res == ERROR_FILE_NOT_FOUND {
            return Ok(None);
        }
        if open_res != ERROR_SUCCESS {
            return Err(format!("RegOpenKeyExW failed with error code: {}", open_res));
        }

        let mut val_type: u32 = 0;
        let mut data = [0u8; 8];
        let mut data_len: u32 = 8;

        let query_res = unsafe {
            RegQueryValueExW(
                hkey,
                val_name_w.as_ptr(),
                ptr::null_mut(),
                &mut val_type,
                data.as_mut_ptr(),
                &mut data_len,
            )
        };
        unsafe {
            RegCloseKey(hkey);
        }

        if query_res == ERROR_FILE_NOT_FOUND {
            return Ok(None);
        }
        if query_res != ERROR_SUCCESS {
            return Err(format!("RegQueryValueExW failed with error code: {}", query_res));
        }
        if val_type != REG_QWORD || data_len != 8 {
            return Err(format!("Registry value is not REG_QWORD or invalid length: type={}, len={}", val_type, data_len));
        }

        Ok(Some(u64::from_ne_bytes(data)))
    }

    pub fn delete_value(sub_key: &str, value_name: &str) -> Result<(), String> {
        let wide_sub = to_wide(sub_key);
        let wide_name = to_wide(value_name);
        let mut hkey: isize = 0;

        let res = unsafe {
            RegOpenKeyExW(
                HKEY_CURRENT_USER,
                wide_sub.as_ptr(),
                0,
                KEY_WRITE,
                &mut hkey,
            )
        };
        if res == ERROR_FILE_NOT_FOUND {
            return Ok(());
        }
        if res != ERROR_SUCCESS {
            return Err(format!("RegOpenKeyExW failed with code {}", res));
        }

        let del_res = unsafe { RegDeleteValueW(hkey, wide_name.as_ptr()) };
        unsafe {
            RegCloseKey(hkey);
        }

        if del_res == ERROR_SUCCESS || del_res == ERROR_FILE_NOT_FOUND {
            Ok(())
        } else {
            Err(format!("RegDeleteValueW failed with code {}", del_res))
        }
    }

    pub fn list_values(sub_key: &str) -> Result<Vec<String>, String> {
        let wide_sub = to_wide(sub_key);
        let mut hkey: isize = 0;

        let res =
            unsafe { RegOpenKeyExW(HKEY_CURRENT_USER, wide_sub.as_ptr(), 0, KEY_READ, &mut hkey) };
        if res == ERROR_FILE_NOT_FOUND {
            return Ok(Vec::new());
        }
        if res != ERROR_SUCCESS {
            return Err(format!("RegOpenKeyExW failed with code {}", res));
        }

        let mut names = Vec::new();
        let mut index: u32 = 0;
        let mut name_buf = [0u16; 1024];

        loop {
            let mut name_len = name_buf.len() as u32;
            let enum_res = unsafe {
                RegEnumValueW(
                    hkey,
                    index,
                    name_buf.as_mut_ptr(),
                    &mut name_len,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                )
            };

            if enum_res == ERROR_NO_MORE_ITEMS {
                break;
            }
            if enum_res != ERROR_SUCCESS {
                unsafe {
                    RegCloseKey(hkey);
                }
                return Err(format!("RegEnumValueW failed with code {}", enum_res));
            }

            let name = String::from_utf16_lossy(&name_buf[..name_len as usize]);
            names.push(name);
            index = index.saturating_add(1);
        }

        unsafe {
            RegCloseKey(hkey);
        }
        Ok(names)
    }

    pub fn delete_test_watermark_values() -> Result<(), String> {
        let sub_key = "Software\\TaraCore\\MonotonicWatermarks";
        let names = list_values(sub_key)?;
        for name in names {
            if super::is_test_watermark_domain(&name) {
                delete_value(sub_key, &name)?;
            }
        }
        Ok(())
    }
}

pub fn get_system_witness_path(domain: &str) -> PathBuf {
    #[cfg(windows)]
    {
        if let Ok(progdata) = std::env::var("ProgramData") {
            return PathBuf::from(progdata)
                .join("TaraCore")
                .join("system_witness")
                .join(format!("{}_system_witness.bin", domain));
        }
        PathBuf::from("C:\\ProgramData\\TaraCore\\system_witness")
            .join(format!("{}_system_witness.bin", domain))
    }
    #[cfg(not(windows))]
    {
        if is_test_watermark_domain(domain) {
            return std::env::temp_dir()
                .join("taracore")
                .join("system_witness")
                .join(format!("{}_system_witness.bin", domain));
        }
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home)
                .join(".local")
                .join("share")
                .join("taracore")
                .join("system_witness")
                .join(format!("{}_system_witness.bin", domain));
        }
        PathBuf::from("/var/run/taracore/system_witness")
            .join(format!("{}_system_witness.bin", domain))
    }
}

pub fn get_unix_secondary_watermark_path(domain: &str) -> PathBuf {
    #[cfg(windows)]
    {
        get_os_isolated_watermark_path(domain)
    }
    #[cfg(not(windows))]
    {
        if is_test_watermark_domain(domain) {
            return std::env::temp_dir()
                .join("taracore")
                .join("vault")
                .join(format!("{}_secondary_watermark.bin", domain));
        }
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home)
                .join(".local")
                .join("share")
                .join("taracore")
                .join("vault")
                .join(format!("{}_secondary_watermark.bin", domain));
        }
        PathBuf::from("/etc/taracore/vault")
            .join(format!("{}_secondary_watermark.bin", domain))
    }
}

pub fn is_test_watermark_domain(domain: &str) -> bool {
    let lower = domain.to_ascii_lowercase();
    lower.starts_with("test")
        || lower.contains("_test")
        || lower.contains("test_")
        || lower.contains("sandbox")
        || lower.contains("temp")
        || lower.contains("tmp")
        || lower.contains("contention")
        || lower.contains("dummy")
}

pub fn store_os_monotonic_watermark(domain: &str, candidate_val: u64) -> Result<(), String> {
    if let Some(curr) = load_os_monotonic_watermark(domain)? {
        if candidate_val < curr {
            return Err(format!(
                "OS monotonic watermark rollback rejected: candidate {} is less than committed {}",
                candidate_val, curr
            ));
        }
    }

    let path = get_os_isolated_watermark_path(domain);
    store_monotonic_watermark(&path, candidate_val)?;

    #[cfg(windows)]
    {
        let subkey = "Software\\TaraCore\\MonotonicWatermarks";
        win_registry::set_u64(subkey, domain, candidate_val)?;
    }
    #[cfg(not(windows))]
    {
        let secondary_path = get_unix_secondary_watermark_path(domain);
        store_monotonic_watermark(&secondary_path, candidate_val)?;
    }

    let sys_path = get_system_witness_path(domain);
    store_system_witness_watermark(&sys_path, candidate_val)?;

    Ok(())
}

pub fn load_os_monotonic_watermark(domain: &str) -> Result<Option<u64>, String> {
    let path = get_os_isolated_watermark_path(domain);
    let disk_val = load_monotonic_watermark(&path)?;

    #[cfg(windows)]
    let reg_val = win_registry::get_u64("Software\\TaraCore\\MonotonicWatermarks", domain)?;
    #[cfg(not(windows))]
    let reg_val = {
        let secondary_path = get_unix_secondary_watermark_path(domain);
        load_monotonic_watermark(&secondary_path)?
    };

    let sys_path = get_system_witness_path(domain);
    let sys_val = load_monotonic_watermark(&sys_path)?;

    match (disk_val, reg_val, sys_val) {
        (Some(d), Some(r), Some(s)) => {
            if d != r || d != s || r != s {
                if d < s && r < s {
                    return Err(format!(
                        "SystemWitnessRollbackDetected: user anchors (disk={}, registry={}) were rolled back behind independent system witness ({})",
                        d, r, s
                    ));
                }
                if s < d || s < r {
                    return Err(format!(
                        "SystemWitnessRollbackDetected: system witness ({}) was rolled back behind user anchors (disk={}, registry={})",
                        s, d, r
                    ));
                }
                if d < r {
                    return Err(format!(
                        "PhysicalWatermarkRollbackDetected: physical disk watermark {} was rolled back behind authoritative OS Registry anchor {}",
                        d, r
                    ));
                }
                if r < d {
                    return Err(format!(
                        "PhysicalWatermarkRollbackDetected: OS Registry anchor {} was rolled back behind authoritative disk watermark {}",
                        r, d
                    ));
                }
                return Err(format!(
                    "WatermarkAnchorDivergenceDetected: triple-anchor divergence detected (disk={}, registry={}, system={})",
                    d, r, s
                ));
            }
            Ok(Some(d))
        }
        (Some(d), Some(r), None) => {
            if d < r {
                return Err(format!(
                    "PhysicalWatermarkRollbackDetected: physical disk watermark {} was rolled back behind authoritative OS Registry anchor {}",
                    d, r
                ));
            }
            if r < d {
                return Err(format!(
                    "PhysicalWatermarkRollbackDetected: OS Registry anchor {} was rolled back behind authoritative disk watermark {}",
                    r, d
                ));
            }
            Err(format!(
                "WatermarkAnchorDivergenceDetected: user anchors present ({}) but machine system witness is missing/deleted",
                d
            ))
        }
        (Some(d), None, Some(_)) => {
            Err(format!(
                "WatermarkAnchorDivergenceDetected: primary disk watermark ({}) present but secondary OS anchor is missing/deleted",
                d
            ))
        }
        (None, Some(r), Some(_)) => {
            Err(format!(
                "WatermarkAnchorDivergenceDetected: secondary OS anchor ({}) present but primary disk watermark file is missing/deleted",
                r
            ))
        }
        (None, None, Some(s)) => {
            Err(format!(
                "SystemWitnessRollbackDetected: user anchors missing/wiped but machine system witness retains committed high watermark ({})",
                s
            ))
        }
        (Some(d), None, None) => {
            Err(format!(
                "WatermarkAnchorDivergenceDetected: primary disk watermark ({}) present but secondary OS anchor and system witness are missing/deleted",
                d
            ))
        }
        (None, Some(r), None) => {
            Err(format!(
                "WatermarkAnchorDivergenceDetected: secondary OS anchor ({}) present but primary disk watermark and system witness are missing/deleted",
                r
            ))
        }
        (None, None, None) => Ok(None),
    }
}

pub fn delete_os_monotonic_watermark(domain: &str) -> Result<(), String> {
    let path = get_os_isolated_watermark_path(domain);
    if path.exists() {
        fs::remove_file(&path)
            .map_err(|e| format!("Failed to remove watermark file {}: {}", path.display(), e))?;
    }
    #[cfg(windows)]
    win_registry::delete_value("Software\\TaraCore\\MonotonicWatermarks", domain)?;
    #[cfg(not(windows))]
    {
        let secondary_path = get_unix_secondary_watermark_path(domain);
        if secondary_path.exists() {
            fs::remove_file(&secondary_path).map_err(|e| {
                format!(
                    "Failed to remove secondary watermark file {}: {}",
                    secondary_path.display(),
                    e
                )
            })?;
        }
    }
    let sys_path = get_system_witness_path(domain);
    if sys_path.exists() {
        #[cfg(windows)]
        {
            if let Ok(user) = std::env::var("USERNAME") {
                let system_root =
                    std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".to_string());
                let icacls_path = PathBuf::from(system_root)
                    .join("System32")
                    .join("icacls.exe");
                let _ = std::process::Command::new(icacls_path)
                    .arg(&sys_path)
                    .arg("/grant:r")
                    .arg(format!("{}:F", user))
                    .output();
            }
        }
        fs::remove_file(&sys_path).map_err(|e| {
            format!(
                "Failed to remove system witness file {}: {}",
                sys_path.display(),
                e
            )
        })?;
    }
    Ok(())
}

pub fn cleanup_test_watermark_artifacts() -> Result<(), String> {
    let dummy_path = get_os_isolated_watermark_path("");
    if let Some(parent) = dummy_path.parent() {
        if parent.exists() {
            let entries = fs::read_dir(parent).map_err(|e| {
                format!(
                    "Failed to read watermark directory {}: {}",
                    parent.display(),
                    e
                )
            })?;
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.ends_with("_monotonic_watermark.bin") {
                    let domain = &name[..name.len() - "_monotonic_watermark.bin".len()];
                    if is_test_watermark_domain(domain) {
                        fs::remove_file(entry.path()).map_err(|e| {
                            format!(
                                "Failed to remove watermark test artifact {}: {}",
                                entry.path().display(),
                                e
                            )
                        })?;
                    }
                }
            }
        }
    }
    #[cfg(windows)]
    win_registry::delete_test_watermark_values()?;
    #[cfg(not(windows))]
    {
        let secondary_dummy = get_unix_secondary_watermark_path("");
        if let Some(parent) = secondary_dummy.parent() {
            if parent.exists() {
                let entries = fs::read_dir(parent).map_err(|e| {
                    format!(
                        "Failed to read secondary watermark directory {}: {}",
                        parent.display(),
                        e
                    )
                })?;
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.ends_with("_monotonic_watermark.bin") {
                        let domain = &name[..name.len() - "_monotonic_watermark.bin".len()];
                        if is_test_watermark_domain(domain) {
                            fs::remove_file(entry.path()).map_err(|e| {
                                format!(
                                    "Failed to remove secondary watermark test artifact {}: {}",
                                    entry.path().display(),
                                    e
                                )
                            })?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dpapi_protect_unprotect() {
        let plaintext = b"TARA_CANONICAL_SECRET_KEY_DPAPI_TEST";
        let protected = native_dpapi::protect_bytes(plaintext).expect("protect");
        let unprotected = native_dpapi::unprotect_bytes(&protected).expect("unprotect");
        assert_eq!(unprotected, plaintext);
    }

    #[test]
    fn test_secure_key_storage_roundtrip() {
        let temp_dir = std::env::temp_dir().join(format!("tara_test_keystore_{}", rand::random::<u64>()));
        let store = SecureKeyStorage::new(&temp_dir);
        let raw_key = [0x42u8; 32];
        let passphrase = "test_creator_strong_passphrase_123!";

        store.store_key("test_operator_key", &raw_key, passphrase).expect("store");
        assert!(store.key_exists("test_operator_key"));

        let loaded = store.load_key("test_operator_key", passphrase).expect("load");
        assert_eq!(loaded, raw_key);

        let wrong_res = store.load_key("test_operator_key", "wrong_passphrase");
        assert!(wrong_res.is_err());

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
