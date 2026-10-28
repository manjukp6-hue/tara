//! Pluggable, extensible cryptographic adapters and security utilities.
//!
//! Rule compliance:
//! - Strictly vetted platform & RustCrypto algorithms (no custom crypto).
//! - Trait-based architecture allows plugging in hardware HSMs, TPM keys, or future Post-Quantum suites.
//! - Constant-time comparison for all security-sensitive verification checks.

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use constant_time_eq::constant_time_eq;
use ed25519_dalek::{Signature as Ed25519Sig, Signer, SigningKey, Verifier, VerifyingKey};

pub use ed25519_dalek::{Signature as DalekSignature, SigningKey as Ed25519SigningKey, VerifyingKey as Ed25519VerifyingKey};
use hmac::{Hmac, Mac};
use rand::RngCore;
use rsa::pkcs1::DecodeRsaPublicKey;
use rsa::pkcs1v15::VerifyingKey as RsaVerifyingKey;
use rsa::sha2::Sha256 as RsaSha256;
use rsa::traits::PublicKeyParts;
use rsa::RsaPublicKey;
use sha2::{Digest, Sha256, Sha512};
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub type HmacSha256 = Hmac<Sha256>;

/// Cryptographic algorithm identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AlgorithmId {
    Ed25519,
    Es256,
    Rs256,
    Aes256Gcm,
    HmacSha256,
    Custom(String),
}

impl fmt::Display for AlgorithmId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ed25519 => write!(f, "Ed25519"),
            Self::Es256 => write!(f, "ES256"),
            Self::Rs256 => write!(f, "RS256"),
            Self::Aes256Gcm => write!(f, "AES-256-GCM"),
            Self::HmacSha256 => write!(f, "HMAC-SHA256"),
            Self::Custom(c) => write!(f, "Custom({})", c),
        }
    }
}

/// Abstract Signature Verifier trait for extensible algorithm support.
pub trait SignatureVerifier: Send + Sync {
    fn algorithm(&self) -> AlgorithmId;
    fn verify(&self, public_key: &[u8], message: &[u8], signature: &[u8]) -> Result<(), String>;
}

/// Standard Ed25519 Signature Verifier.
pub struct Ed25519Verifier;

impl SignatureVerifier for Ed25519Verifier {
    fn algorithm(&self) -> AlgorithmId {
        AlgorithmId::Ed25519
    }

    fn verify(&self, public_key: &[u8], message: &[u8], signature: &[u8]) -> Result<(), String> {
        if public_key.len() != 32 {
            return Err("Invalid Ed25519 public key length: expected 32 bytes".into());
        }
        if signature.len() != 64 {
            return Err("Invalid Ed25519 signature length: expected 64 bytes".into());
        }

        let key_bytes: [u8; 32] = public_key
            .try_into()
            .map_err(|_| "Failed to cast key bytes")?;
        let verifier = VerifyingKey::from_bytes(&key_bytes)
            .map_err(|e| format!("Invalid Ed25519 public key: {e}"))?;

        let sig_bytes: [u8; 64] = signature
            .try_into()
            .map_err(|_| "Failed to cast signature bytes")?;
        let sig = Ed25519Sig::from_bytes(&sig_bytes);

        verifier
            .verify(message, &sig)
            .map_err(|e| format!("Ed25519 signature verification failed: {e}"))
    }
}

/// Standard RSA-SHA256 (RS256) Signature Verifier.
pub struct Rs256Verifier;

impl SignatureVerifier for Rs256Verifier {
    fn algorithm(&self) -> AlgorithmId {
        AlgorithmId::Rs256
    }

    fn verify(
        &self,
        public_key_der: &[u8],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), String> {
        let pub_key = RsaPublicKey::from_pkcs1_der(public_key_der)
            .or_else(|_| rsa::pkcs8::DecodePublicKey::from_public_key_der(public_key_der))
            .map_err(|e| format!("Failed to parse RSA public key: {e}"))?;

        // Policy Gate: Enforce minimum 2048-bit RSA key size
        let key_bits = pub_key.n().bits();
        if key_bits < 2048 {
            return Err(format!("InsecureKeySize: RSA key size of {key_bits} bits is below mandatory 2048-bit threshold"));
        }

        let verifying_key: RsaVerifyingKey<RsaSha256> = RsaVerifyingKey::new(pub_key);
        let sig = rsa::pkcs1v15::Signature::try_from(signature)
            .map_err(|e| format!("Invalid RSA signature structure: {e}"))?;

        verifying_key
            .verify(message, &sig)
            .map_err(|e| format!("RS256 verification failed: {e}"))
    }
}

/// Standard ECDSA P-256 with SHA-256 (ES256) Signature Verifier.
pub struct Es256Verifier;

impl SignatureVerifier for Es256Verifier {
    fn algorithm(&self) -> AlgorithmId {
        AlgorithmId::Es256
    }

    fn verify(&self, public_key: &[u8], message: &[u8], signature: &[u8]) -> Result<(), String> {
        let (x, y) = extract_p256_public_coords(public_key)?;
        let sig64 = normalize_p256_signature(signature)?;

        #[cfg(windows)]
        {
            let mut hasher = Sha256::new();
            hasher.update(message);
            let hash: [u8; 32] = hasher.finalize().into();
            // Try Windows platform CNG first, then pure-Rust p256 verifier
            cng_p256::verify_p256_cng(&x, &y, &hash, &sig64)
                .or_else(|_| verify_p256_pure_rust(&x, &y, message, &sig64))
        }

        #[cfg(not(windows))]
        {
            verify_p256_pure_rust(&x, &y, message, &sig64)
        }
    }
}

/// Verify P-256 ECDSA/SHA-256 signature using the pure-Rust `p256` crate.
/// This runs identically across Windows, Linux, and macOS.
pub fn verify_p256_pure_rust(
    x: &[u8; 32],
    y: &[u8; 32],
    message: &[u8],
    sig64: &[u8; 64],
) -> Result<(), String> {
    let encoded_point = p256::EncodedPoint::from_affine_coordinates(x.into(), y.into(), false);
    let verifying_key = p256::ecdsa::VerifyingKey::from_encoded_point(&encoded_point)
        .map_err(|e| format!("Invalid P-256 public key point coordinates: {e}"))?;

    let signature = p256::ecdsa::Signature::from_bytes(sig64.into())
        .map_err(|e| format!("Invalid ECDSA P-256 signature format: {e}"))?;

    use p256::ecdsa::signature::Verifier;
    verifying_key
        .verify(message, &signature)
        .map_err(|e| format!("P-256 ECDSA signature verification failed: {e}"))
}

fn parse_der_length(input: &[u8], offset: &mut usize) -> Result<usize, String> {
    if *offset >= input.len() {
        return Err("Truncated DER input".into());
    }
    let first = input[*offset];
    *offset += 1;
    if first & 0x80 == 0 {
        Ok(first as usize)
    } else {
        let num_octets = (first & 0x7F) as usize;
        if num_octets == 0 || num_octets > 4 || *offset + num_octets > input.len() {
            return Err("Invalid DER multi-byte length".into());
        }
        let mut len = 0usize;
        for _ in 0..num_octets {
            len = (len << 8) | (input[*offset] as usize);
            *offset += 1;
        }
        Ok(len)
    }
}

fn parse_spki_p256(pk: &[u8]) -> Result<([u8; 32], [u8; 32]), String> {
    let mut offset = 0;
    if offset >= pk.len() || pk[offset] != 0x30 {
        return Err("SPKI must start with SEQUENCE (0x30)".into());
    }
    offset += 1;
    let total_len = parse_der_length(pk, &mut offset)?;
    if offset + total_len > pk.len() {
        return Err("SPKI total length exceeds buffer".into());
    }

    // 1. AlgorithmIdentifier SEQUENCE
    if offset >= pk.len() || pk[offset] != 0x30 {
        return Err("SPKI missing AlgorithmIdentifier SEQUENCE".into());
    }
    offset += 1;
    let alg_len = parse_der_length(pk, &mut offset)?;
    if offset + alg_len > pk.len() {
        return Err("AlgorithmIdentifier length exceeds buffer".into());
    }
    let alg_bytes = &pk[offset..offset + alg_len];
    offset += alg_len;

    // Check OID id-ecPublicKey: 1.2.840.10045.2.1 = 06 07 2A 86 48 CE 3D 02 01
    const EC_PUB_KEY_OID: [u8; 9] = [0x06, 0x07, 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x02, 0x01];
    if !alg_bytes
        .windows(EC_PUB_KEY_OID.len())
        .any(|w| w == EC_PUB_KEY_OID)
    {
        return Err("AlgorithmIdentifier is not id-ecPublicKey (1.2.840.10045.2.1)".into());
    }

    // Check curve parameter OID prime256v1 / secp256r1: 1.2.840.10045.3.1.7 = 06 08 2A 86 48 CE 3D 03 01 07
    const SECP256R1_OID: [u8; 10] = [0x06, 0x08, 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x03, 0x01, 0x07];
    if !alg_bytes
        .windows(SECP256R1_OID.len())
        .any(|w| w == SECP256R1_OID)
    {
        return Err(
            "AlgorithmIdentifier curve parameter is not secp256r1/prime256v1 (1.2.840.10045.3.1.7)"
                .into(),
        );
    }

    // 2. subjectPublicKey BIT STRING
    if offset >= pk.len() || pk[offset] != 0x03 {
        return Err("SPKI missing subjectPublicKey BIT STRING (0x03)".into());
    }
    offset += 1;
    let bit_len = parse_der_length(pk, &mut offset)?;
    if offset + bit_len > pk.len() {
        return Err("subjectPublicKey BIT STRING length exceeds buffer".into());
    }
    let bit_bytes = &pk[offset..offset + bit_len];

    // BIT STRING must have 0 unused bits followed by uncompressed point (0x04 || X || Y) = 66 bytes total
    if bit_bytes.len() != 66 || bit_bytes[0] != 0x00 || bit_bytes[1] != 0x04 {
        return Err("subjectPublicKey BIT STRING is not an uncompressed P-256 point (expected 66 bytes: 0x00 0x04 || X || Y)".into());
    }

    let mut x = [0u8; 32];
    let mut y = [0u8; 32];
    x.copy_from_slice(&bit_bytes[2..34]);
    y.copy_from_slice(&bit_bytes[34..66]);
    Ok((x, y))
}

pub fn extract_p256_public_coords(pk: &[u8]) -> Result<([u8; 32], [u8; 32]), String> {
    if pk.len() == 64 {
        let mut x = [0u8; 32];
        let mut y = [0u8; 32];
        x.copy_from_slice(&pk[0..32]);
        y.copy_from_slice(&pk[32..64]);
        return Ok((x, y));
    }
    if pk.len() == 65 && pk[0] == 0x04 {
        let mut x = [0u8; 32];
        let mut y = [0u8; 32];
        x.copy_from_slice(&pk[1..33]);
        y.copy_from_slice(&pk[33..65]);
        return Ok((x, y));
    }
    if pk.len() == 33 && (pk[0] == 0x02 || pk[0] == 0x03) {
        let point = p256::EncodedPoint::from_bytes(pk)
            .map_err(|e| format!("Invalid compressed P-256 public key: {e}"))?;
        let vk = p256::ecdsa::VerifyingKey::from_encoded_point(&point)
            .map_err(|e| format!("Invalid compressed P-256 curve point: {e}"))?;
        let decompressed = vk.to_encoded_point(false);
        let x_bytes = decompressed
            .x()
            .ok_or_else(|| "Missing X coordinate".to_string())?;
        let y_bytes = decompressed
            .y()
            .ok_or_else(|| "Missing Y coordinate".to_string())?;
        let mut x = [0u8; 32];
        let mut y = [0u8; 32];
        x.copy_from_slice(x_bytes);
        y.copy_from_slice(y_bytes);
        return Ok((x, y));
    }
    // Check if pk is a CBOR COSE_Key map containing P-256 EC coordinates (-2: X, -3: Y)
    const CBOR_X_HEADER: [u8; 3] = [0x21, 0x58, 0x20];
    const CBOR_Y_HEADER: [u8; 3] = [0x22, 0x58, 0x20];
    if let (Some(x_pos), Some(y_pos)) = (
        pk.windows(3).position(|w| w == CBOR_X_HEADER),
        pk.windows(3).position(|w| w == CBOR_Y_HEADER),
    ) {
        if x_pos + 3 + 32 <= pk.len() && y_pos + 3 + 32 <= pk.len() {
            let mut x = [0u8; 32];
            let mut y = [0u8; 32];
            x.copy_from_slice(&pk[x_pos + 3..x_pos + 3 + 32]);
            y.copy_from_slice(&pk[y_pos + 3..y_pos + 3 + 32]);
            return Ok((x, y));
        }
    }
    if pk.starts_with(&[0x30]) {
        return parse_spki_p256(pk);
    }
    Err("Invalid P-256 public key format: expected raw 64-byte (X||Y), 65-byte uncompressed (0x04||X||Y), 33-byte compressed, CBOR COSE_Key, or ASN.1 DER SubjectPublicKeyInfo".into())
}

fn normalize_p256_signature(sig: &[u8]) -> Result<[u8; 64], String> {
    if sig.len() == 64 {
        let mut out = [0u8; 64];
        out.copy_from_slice(sig);
        return Ok(out);
    }
    if sig.len() >= 8 && sig[0] == 0x30 {
        let mut offset = 2;
        if sig.len() > offset && sig[1] & 0x80 != 0 {
            let num_len_bytes = (sig[1] & 0x7f) as usize;
            offset = 2 + num_len_bytes;
        }
        if offset + 2 > sig.len() || sig[offset] != 0x02 {
            return Err("Invalid ASN.1 DER signature: missing r marker".into());
        }
        let r_len = sig[offset + 1] as usize;
        offset += 2;
        if offset + r_len > sig.len() {
            return Err("Invalid ASN.1 DER signature: truncated r".into());
        }
        let mut r_slice = &sig[offset..offset + r_len];
        offset += r_len;
        if r_slice.len() > 32 && r_slice[0] == 0x00 {
            r_slice = &r_slice[1..];
        }
        if r_slice.len() > 32 {
            return Err("ASN.1 DER signature r exceeds 32 bytes".into());
        }

        if offset + 2 > sig.len() || sig[offset] != 0x02 {
            return Err("Invalid ASN.1 DER signature: missing s marker".into());
        }
        let s_len = sig[offset + 1] as usize;
        offset += 2;
        if offset + s_len > sig.len() {
            return Err("Invalid ASN.1 DER signature: truncated s".into());
        }
        let mut s_slice = &sig[offset..offset + s_len];
        if s_slice.len() > 32 && s_slice[0] == 0x00 {
            s_slice = &s_slice[1..];
        }
        if s_slice.len() > 32 {
            return Err("ASN.1 DER signature s exceeds 32 bytes".into());
        }

        let mut out = [0u8; 64];
        let r_pad = 32 - r_slice.len();
        out[r_pad..32].copy_from_slice(r_slice);
        let s_pad = 32 - s_slice.len();
        out[32 + s_pad..64].copy_from_slice(s_slice);
        return Ok(out);
    }
    Err(format!(
        "Invalid ECDSA P-256 signature length: {}, expected 64 raw or valid ASN.1 DER",
        sig.len()
    ))
}

#[cfg(windows)]
mod cng_p256 {
    use std::ptr;

    const BCRYPT_ECDSA_PUBLIC_P256_MAGIC: u32 = 0x31534345; // "ECS1"
    const BCRYPT_ECDSA_P256_ALGORITHM: &[u16] = &[
        'E' as u16, 'C' as u16, 'D' as u16, 'S' as u16, 'A' as u16, '_' as u16, 'P' as u16,
        '2' as u16, '5' as u16, '6' as u16, 0,
    ];
    const BCRYPT_ECCPUBLIC_BLOB: &[u16] = &[
        'E' as u16, 'C' as u16, 'C' as u16, 'P' as u16, 'U' as u16, 'B' as u16, 'L' as u16,
        'I' as u16, 'C' as u16, 'B' as u16, 'L' as u16, 'O' as u16, 'B' as u16, 0,
    ];

    #[link(name = "bcrypt")]
    extern "system" {
        fn BCryptOpenAlgorithmProvider(
            phAlgorithm: *mut *mut std::ffi::c_void,
            pszAlgId: *const u16,
            pszImplementation: *const u16,
            dwFlags: u32,
        ) -> i32;

        fn BCryptCloseAlgorithmProvider(hAlgorithm: *mut std::ffi::c_void, dwFlags: u32) -> i32;

        fn BCryptImportKeyPair(
            hAlgorithm: *mut std::ffi::c_void,
            hImportKey: *mut std::ffi::c_void,
            pszBlobType: *const u16,
            phKey: *mut *mut std::ffi::c_void,
            pbInput: *const u8,
            cbInput: u32,
            dwFlags: u32,
        ) -> i32;

        fn BCryptDestroyKey(hKey: *mut std::ffi::c_void) -> i32;

        fn BCryptVerifySignature(
            hKey: *mut std::ffi::c_void,
            pPaddingInfo: *const std::ffi::c_void,
            pbHash: *const u8,
            cbHash: u32,
            pbSignature: *const u8,
            cbSignature: u32,
            dwFlags: u32,
        ) -> i32;
    }

    pub fn verify_p256_cng(
        x: &[u8; 32],
        y: &[u8; 32],
        hash32: &[u8; 32],
        sig64: &[u8; 64],
    ) -> Result<(), String> {
        unsafe {
            let mut h_alg: *mut std::ffi::c_void = ptr::null_mut();
            let status = BCryptOpenAlgorithmProvider(
                &mut h_alg,
                BCRYPT_ECDSA_P256_ALGORITHM.as_ptr(),
                ptr::null(),
                0,
            );
            if status != 0 {
                return Err(format!(
                    "BCryptOpenAlgorithmProvider failed (NTSTATUS 0x{:08X})",
                    status
                ));
            }

            let mut blob = Vec::with_capacity(8 + 32 + 32);
            blob.extend_from_slice(&BCRYPT_ECDSA_PUBLIC_P256_MAGIC.to_ne_bytes());
            blob.extend_from_slice(&32u32.to_ne_bytes());
            blob.extend_from_slice(x);
            blob.extend_from_slice(y);

            let mut h_key: *mut std::ffi::c_void = ptr::null_mut();
            let import_status = BCryptImportKeyPair(
                h_alg,
                ptr::null_mut(),
                BCRYPT_ECCPUBLIC_BLOB.as_ptr(),
                &mut h_key,
                blob.as_ptr(),
                blob.len() as u32,
                0,
            );

            if import_status != 0 {
                BCryptCloseAlgorithmProvider(h_alg, 0);
                return Err(format!(
                    "BCryptImportKeyPair failed (NTSTATUS 0x{:08X})",
                    import_status
                ));
            }

            let verify_status = BCryptVerifySignature(
                h_key,
                ptr::null(),
                hash32.as_ptr(),
                32,
                sig64.as_ptr(),
                64,
                0,
            );

            BCryptDestroyKey(h_key);
            BCryptCloseAlgorithmProvider(h_alg, 0);

            if verify_status == 0 {
                Ok(())
            } else {
                Err(format!(
                    "ECDSA P-256 signature verification failed (NTSTATUS 0x{:08X})",
                    verify_status
                ))
            }
        }
    }
}

/// Standard Authenticated Encryption (AEAD) provider.
pub trait AeadProvider: Send + Sync {
    fn encrypt(&self, key: &[u8], plaintext: &[u8], aad: &[u8]) -> Result<Vec<u8>, String>;
    fn decrypt(
        &self,
        key: &[u8],
        ciphertext_with_nonce: &[u8],
        aad: &[u8],
    ) -> Result<Vec<u8>, String>;
}

/// AES-256-GCM AEAD implementation with random 96-bit nonce prepended and authenticated AAD.
pub struct Aes256GcmProvider;

impl AeadProvider for Aes256GcmProvider {
    fn encrypt(&self, key: &[u8], plaintext: &[u8], aad: &[u8]) -> Result<Vec<u8>, String> {
        if key.len() != 32 {
            return Err("AES-256-GCM requires a 32-byte key".into());
        }
        let cipher = Aes256Gcm::new_from_slice(key).map_err(|e| e.to_string())?;

        let mut nonce_bytes = [0u8; 12];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher
            .encrypt(
                nonce,
                aes_gcm::aead::Payload {
                    msg: plaintext,
                    aad,
                },
            )
            .map_err(|e| format!("AEAD encryption failed: {e}"))?;

        let mut out = Vec::with_capacity(12 + ciphertext.len());
        out.extend_from_slice(&nonce_bytes);
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    fn decrypt(&self, key: &[u8], payload: &[u8], aad: &[u8]) -> Result<Vec<u8>, String> {
        if key.len() != 32 {
            return Err("AES-256-GCM requires a 32-byte key".into());
        }
        if payload.len() < 12 {
            return Err("Ciphertext payload too short: missing 12-byte nonce".into());
        }

        let (nonce_bytes, ciphertext) = payload.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);
        let cipher = Aes256Gcm::new_from_slice(key).map_err(|e| e.to_string())?;

        cipher
            .decrypt(
                nonce,
                aes_gcm::aead::Payload {
                    msg: ciphertext,
                    aad,
                },
            )
            .map_err(|e| {
                format!("AEAD authentication/decryption failed (tampered data or wrong key): {e}")
            })
    }
}

/// Core Cryptographic Utility functions.
pub struct CryptoUtils;

impl CryptoUtils {
    /// Canonical SHA-256 computation on raw in-memory bytes.
    pub fn compute256_bytes(data: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().into()
    }

    /// Canonical SHA-256 hex string computation on raw in-memory bytes.
    pub fn compute256_bytes_hex(data: &[u8]) -> String {
        hex::encode(Self::compute256_bytes(data))
    }

    /// Compute SHA-256 digest on in-memory bytes.
    /// Thin compatibility alias delegating to `compute256_bytes`.
    pub fn sha256(data: &[u8]) -> [u8; 32] {
        Self::compute256_bytes(data)
    }

    /// Compute SHA-256 hex string on in-memory bytes.
    /// Thin compatibility alias delegating to `compute256_bytes_hex`.
    pub fn sha256_hex(data: &[u8]) -> String {
        Self::compute256_bytes_hex(data)
    }

    /// Compute SHA-512 digest.
    pub fn sha512(data: &[u8]) -> [u8; 64] {
        let mut hasher = Sha512::new();
        hasher.update(data);
        hasher.finalize().into()
    }

    /// Compute HMAC-SHA256 signature bytes.
    pub fn hmac_sha256(key: &[u8], data: &[u8]) -> Result<[u8; 32], String> {
        let mut mac = <HmacSha256 as Mac>::new_from_slice(key).map_err(|e| e.to_string())?;
        mac.update(data);
        Ok(mac.finalize().into_bytes().into())
    }

    /// Compute HMAC-SHA256 signature hex string.
    pub fn hmac_sha256_hex(key: &[u8], data: &[u8]) -> Result<String, String> {
        Self::hmac_sha256(key, data).map(hex::encode)
    }

    /// Constant-time comparison for authenticating cryptographic hashes or signatures.
    pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
        constant_time_eq(a, b)
    }

    /// Verify HMAC-SHA256 signature in constant time.
    pub fn verify_hmac_sha256(key: &[u8], data: &[u8], expected: &[u8; 32]) -> bool {
        match Self::hmac_sha256(key, data) {
            Ok(computed) => constant_time_eq(&computed, expected),
            Err(_) => false,
        }
    }

    /// Verify HMAC-SHA256 signature against hex-encoded string in constant time.
    pub fn verify_hmac_sha256_hex(key: &[u8], data: &[u8], expected_hex: &str) -> bool {
        match Self::hmac_sha256_hex(key, data) {
            Ok(computed) => constant_time_eq(computed.as_bytes(), expected_hex.as_bytes()),
            Err(_) => false,
        }
    }

    /// Derive key using Scrypt with bounded secure parameters.
    pub fn derive_scrypt_key(
        passphrase: &[u8],
        salt: &[u8],
        output: &mut [u8; 32],
    ) -> Result<(), String> {
        if salt.len() < 16 {
            return Err(
                "InsecureSalt: Scrypt salt must be at least 16 bytes of independent entropy".into(),
            );
        }
        const LOG_N: u8 = 15; // N = 32768
        const R: u32 = 8;
        const P: u32 = 1;
        let params = scrypt::Params::new(LOG_N, R, P, 32)
            .map_err(|e| format!("Scrypt params error: {e}"))?;
        scrypt::scrypt(passphrase, salt, &params, output)
            .map_err(|e| format!("Scrypt derivation error: {e}"))
    }

    /// Generate an authentic Ed25519 signing keypair with stack-zeroizing memory hygiene.
    pub fn generate_ed25519_keypair() -> (SigningKey, VerifyingKey) {
        let mut seed = zeroize::Zeroizing::new([0u8; 32]);
        rand::thread_rng().fill_bytes(&mut *seed);
        let signing_key = SigningKey::from_bytes(&seed);
        let verifying_key = signing_key.verifying_key();
        (signing_key, verifying_key)
    }

    /// Sign data using Ed25519 signing key.
    pub fn sign_ed25519(signing_key: &SigningKey, message: &[u8]) -> [u8; 64] {
        signing_key.sign(message).to_bytes()
    }

    /// Canonical SHA-256 entry point and source of truth for file hashing.
    /// Authentically computes SHA-256 hash of a file on disk using a streaming 64KB buffer.
    pub fn compute256<P: AsRef<Path>>(path: P) -> Result<String, String> {
        let p = path.as_ref();
        let mut file = File::open(p).map_err(|e| {
            format!(
                "Failed to open file '{}' for SHA-256 computation: {e}",
                p.display()
            )
        })?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 65536];

        loop {
            let bytes_read = file.read(&mut buffer).map_err(|e| {
                format!("Failed reading file '{}' during hashing: {e}", p.display())
            })?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }

        let digest = hasher.finalize();
        Ok(hex::encode(digest))
    }

    /// Authentically compute SHA-256 hash of a file on disk.
    /// Thin compatibility alias delegating to canonical `compute256`.
    pub fn compute_file_sha256<P: AsRef<Path>>(path: P) -> Result<String, String> {
        Self::compute256(path)
    }

    /// Return current UTC epoch timestamp in seconds.
    pub fn now_secs() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Return current UTC epoch timestamp in milliseconds.
    pub fn now_millis() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Return current UTC epoch timestamp in fractional seconds (f64).
    pub fn now_secs_f64() -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }

    /// Formats Unix epoch seconds (`secs` since `1970-01-01T00:00:00Z`, signed) into an
    /// exact UTC ISO-8601 timestamp (`YYYY-MM-DDTHH:MM:SSZ`) using Howard Hinnant's proleptic
    /// Gregorian civil-from-days algorithm in full `i128` arithmetic (handling 4/100/400-year leap rules accurately
    /// without domain truncation or integer overflow for any `i128` / `u64` input).
    pub fn format_unix_seconds_iso_signed(secs: i128) -> String {
        let (days, rem_secs) = if secs >= 0 {
            (secs / 86_400, secs % 86_400)
        } else {
            let d = (secs - 86_399) / 86_400;
            let r = secs - d * 86_400;
            (d, r)
        };

        let s = (rem_secs % 60) as u32;
        let m = ((rem_secs / 60) % 60) as u32;
        let h = (rem_secs / 3600) as u32;

        let z = days + 719468;
        let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
        let doe = (z - era * 146097) as u64;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let y = (yoe as i128) + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let month = if mp < 10 { mp + 3 } else { mp - 9 };
        let year = if month <= 2 { y + 1 } else { y };

        if (0..=9999).contains(&year) {
            format!("{year:04}-{month:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
        } else {
            format!("{year}-{month:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
        }
    }

    /// Formats non-negative Unix epoch seconds (`secs` since `1970-01-01T00:00:00Z`) into an
    /// exact UTC ISO-8601 timestamp (`YYYY-MM-DDTHH:MM:SSZ`) safely across the entire `u64` domain.
    pub fn format_unix_seconds_iso(secs: u64) -> String {
        Self::format_unix_seconds_iso_signed(secs as i128)
    }

    /// Formats a [`std::time::SystemTime`] as an exact UTC ISO-8601 timestamp (`YYYY-MM-DDTHH:MM:SSZ`),
    /// correctly handling both pre-epoch and post-epoch timestamps without silent data destruction.
    pub fn format_system_time_iso(time: std::time::SystemTime) -> String {
        use std::time::UNIX_EPOCH;
        match time.duration_since(UNIX_EPOCH) {
            Ok(dur) => Self::format_unix_seconds_iso(dur.as_secs()),
            Err(e) => {
                let neg_secs = e.duration().as_secs();
                Self::format_unix_seconds_iso_signed(-(neg_secs as i128))
            }
        }
    }

    /// Return current UTC time as an exact ISO-8601 string (`YYYY-MM-DDTHH:MM:SSZ`).
    pub fn now_iso() -> String {
        Self::format_system_time_iso(std::time::SystemTime::now())
    }
}

/// Streaming / multi-part HMAC-SHA256 signer.
pub struct HmacSha256Signer {
    mac: HmacSha256,
}

impl HmacSha256Signer {
    /// Initialize a new streaming HMAC-SHA256 signer with the provided secret key.
    pub fn new(key: &[u8]) -> Result<Self, String> {
        let mac = <HmacSha256 as Mac>::new_from_slice(key).map_err(|e| e.to_string())?;
        Ok(Self { mac })
    }

    /// Update the streaming HMAC state with additional data bytes.
    pub fn update(&mut self, data: &[u8]) {
        self.mac.update(data);
    }

    /// Finalize the HMAC computation and return the 32-byte signature.
    pub fn finalize(self) -> [u8; 32] {
        self.mac.finalize().into_bytes().into()
    }

    /// Finalize the HMAC computation and return the hex-encoded signature string.
    pub fn finalize_hex(self) -> String {
        hex::encode(self.finalize())
    }
}
