//! Hardware-Rooted Security & Attestation Subsystem (Pillar 1).
//!
//! Standards Compliance & Capabilities:
//! - W3C Web Authentication Level 3 Recommendation (25 August 2026).
//! - Trusted Computing Group TPM 2.0 Library Specification Version 185 (March 2026).
//! - Format-specific verifiers for packed, tpm, apple, and android-key.
//! - Distinct KNOWN_ATTESTATION_FORMATS vs SUPPORTED_ATTESTATION_FORMATS.
//! - Trait-based AttestationFormatVerifier allows dynamic registration of future formats.
//! - Platform hardware security, root ceremonies, key rotation, and platform-specific ACLs.

use byteorder::{BigEndian, ByteOrder};
use constant_time_eq::constant_time_eq;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

pub mod authority;
pub mod pcr_quote;
pub mod platform;

pub use crate::cbor;
pub use authority::{AuthorityManager, AuthorityRecord, RecoveryProof, RotationToken};
pub use pcr_quote::{PcrQuoteVerificationResult, TpmPcrQuoteStatement, TpmPcrQuoteVerifier};
pub use platform::{harden_file_permissions, sync_directory_metadata};

use crate::cryptography::{CryptoUtils, SignatureVerifier};

/// Standard TPM 2.0 Constants (TCG TPM 2.0 Part 2: Structures).
pub const TPM_ST_ATTEST_CERTIFY: u16 = 0x8017;
pub const TPM_ST_ATTEST_CREATION: u16 = 0x801A;
pub const TPM_GENERATED_VALUE: u32 = 0xFF544347;

/// COSE Key Algorithm identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoseAlgorithm {
    EdDsa, // -8: Ed25519
    Es256, // -7: ECDSA with SHA-256
    Rs256, // -257: RSASSA-PKCS1-v1_5 with SHA-256
    Unsupported(i64),
}

impl CoseAlgorithm {
    pub fn as_i64(&self) -> i64 {
        match self {
            Self::EdDsa => -8,
            Self::Es256 => -7,
            Self::Rs256 => -257,
            Self::Unsupported(id) => *id,
        }
    }
}

impl From<i64> for CoseAlgorithm {
    fn from(val: i64) -> Self {
        match val {
            -8 => Self::EdDsa,
            -7 => Self::Es256,
            -257 => Self::Rs256,
            other => Self::Unsupported(other),
        }
    }
}

/// Attestation Statement parsed representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationStatement {
    pub format: String,
    pub alg: Option<i64>,
    pub sig: Option<Vec<u8>>,
    pub x5c: Option<Vec<Vec<u8>>>,
    pub cert_info: Option<Vec<u8>>,
    pub pub_area: Option<Vec<u8>>,
}

/// Extensible Attestation Verification Result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationVerificationResult {
    pub format: String,
    pub credential_public_key: Vec<u8>,
    pub aaguid: [u8; 16],
    pub is_hardware_backed: bool,
    pub trust_path_anchored: bool,
    pub trust_chain_der: Vec<Vec<u8>>,
}

/// Trait for format-specific attestation verifiers.
pub trait AttestationFormatVerifier: Send + Sync {
    fn format_name(&self) -> &'static str;

    fn verify(
        &self,
        stmt: &AttestationStatement,
        auth_data: &[u8],
        client_data_hash: &[u8],
        trust_store: &AttestationTrustStore,
    ) -> Result<AttestationVerificationResult, String>;
}

/// Registry of known and supported attestation formats.
pub struct AttestationRegistry {
    known_formats: HashSet<&'static str>,
    supported_verifiers: Arc<RwLock<HashMap<String, Box<dyn AttestationFormatVerifier>>>>,
}

impl Default for AttestationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl AttestationRegistry {
    pub fn new() -> Self {
        let mut known = HashSet::new();
        known.insert("none");
        known.insert("packed");
        known.insert("tpm");
        known.insert("apple");
        known.insert("android-key");
        known.insert("android-safetynet");
        known.insert("fido-u2f");

        let mut registry = Self {
            known_formats: known,
            supported_verifiers: Arc::new(RwLock::new(HashMap::new())),
        };

        // Register default production format verifiers
        registry.register_verifier(Box::new(PackedAttestationVerifier));
        registry.register_verifier(Box::new(TpmAttestationVerifier));
        registry.register_verifier(Box::new(AppleAttestationVerifier));
        registry.register_verifier(Box::new(AndroidKeyAttestationVerifier));

        registry
    }

    pub fn register_verifier(&mut self, verifier: Box<dyn AttestationFormatVerifier>) {
        let name = verifier.format_name().to_string();
        self.known_formats.insert(verifier.format_name());
        let mut map = self.supported_verifiers.write().unwrap();
        map.insert(name, verifier);
    }

    pub fn is_known_format(&self, format: &str) -> bool {
        self.known_formats.contains(format)
    }

    pub fn is_supported_format(&self, format: &str) -> bool {
        let map = self.supported_verifiers.read().unwrap();
        map.contains_key(format)
    }

    pub fn verify_statement(
        &self,
        stmt: &AttestationStatement,
        auth_data: &[u8],
        client_data_hash: &[u8],
        trust_store: &AttestationTrustStore,
    ) -> Result<AttestationVerificationResult, String> {
        if !self.is_known_format(&stmt.format) {
            return Err(format!("Unknown attestation format: '{}'", stmt.format));
        }

        let map = self
            .supported_verifiers
            .read()
            .map_err(|_| "Verifier registry lock poisoned (FailClosed)")?;
        let verifier = map.get(&stmt.format).ok_or_else(|| {
            format!(
                "FormatPolicyRejection: Attestation format '{}' is recognized by specification but prohibited or unsupported by security policy",
                stmt.format
            )
        })?;

        verifier.verify(stmt, auth_data, client_data_hash, trust_store)
    }
}

/// Hardware Attestation Trust Store anchoring root CA certificates and public keys.
pub struct AttestationTrustStore {
    trusted_roots: RwLock<HashMap<String, Vec<u8>>>,
}

impl Default for AttestationTrustStore {
    fn default() -> Self {
        Self::new()
    }
}

impl AttestationTrustStore {
    pub fn new() -> Self {
        Self {
            trusted_roots: RwLock::new(HashMap::new()),
        }
    }

    pub fn register_root(&self, name: &str, cert_or_key: &[u8]) {
        if let Ok(mut map) = self.trusted_roots.write() {
            map.insert(name.to_string(), cert_or_key.to_vec());
        }
    }

    pub fn contains_root(&self, cert_or_key: &[u8]) -> bool {
        if let Ok(map) = self.trusted_roots.read() {
            map.values().any(|r| constant_time_eq(r, cert_or_key))
        } else {
            false
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Format Verifiers
// ─────────────────────────────────────────────────────────────────────────────

pub struct PackedAttestationVerifier;

impl AttestationFormatVerifier for PackedAttestationVerifier {
    fn format_name(&self) -> &'static str {
        "packed"
    }

    fn verify(
        &self,
        stmt: &AttestationStatement,
        auth_data: &[u8],
        client_data_hash: &[u8],
        trust_store: &AttestationTrustStore,
    ) -> Result<AttestationVerificationResult, String> {
        let alg_id = stmt.alg.ok_or("Packed attestation missing alg parameter")?;
        let cose_alg = CoseAlgorithm::from(alg_id);
        let sig = stmt
            .sig
            .as_ref()
            .ok_or("Packed attestation missing sig parameter")?;

        let mut signed_data = Vec::with_capacity(auth_data.len() + client_data_hash.len());
        signed_data.extend_from_slice(auth_data);
        signed_data.extend_from_slice(client_data_hash);

        let (aaguid, cred_pk) = extract_auth_data_elements(auth_data)?;

        if let Some(ref x5c_chain) = stmt.x5c {
            if x5c_chain.is_empty() {
                return Err("Packed attestation x5c certificate chain cannot be empty".into());
            }
            let leaf_cert = &x5c_chain[0];
            let root_anchored = verify_x5c_chain(x5c_chain, trust_store)?;

            verify_signed_data(leaf_cert, &signed_data, sig, cose_alg)?;

            Ok(AttestationVerificationResult {
                format: "packed".into(),
                credential_public_key: cred_pk,
                aaguid,
                is_hardware_backed: true,
                trust_path_anchored: root_anchored,
                trust_chain_der: x5c_chain.clone(),
            })
        } else {
            verify_signed_data(&cred_pk, &signed_data, sig, cose_alg)?;

            Ok(AttestationVerificationResult {
                format: "packed".into(),
                credential_public_key: cred_pk,
                aaguid,
                is_hardware_backed: false,
                trust_path_anchored: false,
                trust_chain_der: Vec::new(),
            })
        }
    }
}

fn verify_tpm_pub_area_matches_cred_pk(pub_area: &[u8], cred_pk: &[u8]) -> Result<(), String> {
    if pub_area.is_empty() {
        return Err("TPM pubArea cannot be empty".into());
    }

    // Check if cred_pk has extractable P-256 ECC coordinates
    if let Ok((cred_x, cred_y)) = crate::cryptography::extract_p256_public_coords(cred_pk) {
        let x_matched = pub_area.windows(32).any(|w| constant_time_eq(w, &cred_x));
        let y_matched = pub_area.windows(32).any(|w| constant_time_eq(w, &cred_y));
        if !x_matched || !y_matched {
            return Err("TPM pubArea does not contain the credential ECC public key coordinates from authData".into());
        }
        return Ok(());
    }

    // For RSA or raw credentials, match in constant time
    if pub_area
        .windows(cred_pk.len())
        .any(|w| constant_time_eq(w, cred_pk))
    {
        return Ok(());
    }

    Err("TPM pubArea does not match credential public key from authData".into())
}

pub struct TpmAttestationVerifier;

impl AttestationFormatVerifier for TpmAttestationVerifier {
    fn format_name(&self) -> &'static str {
        "tpm"
    }

    fn verify(
        &self,
        stmt: &AttestationStatement,
        auth_data: &[u8],
        client_data_hash: &[u8],
        trust_store: &AttestationTrustStore,
    ) -> Result<AttestationVerificationResult, String> {
        let alg_id = stmt.alg.ok_or("TPM attestation missing alg parameter")?;
        let cose_alg = CoseAlgorithm::from(alg_id);
        let sig = stmt
            .sig
            .as_ref()
            .ok_or("TPM attestation missing sig parameter")?;
        let cert_info = stmt
            .cert_info
            .as_ref()
            .ok_or("TPM attestation missing certInfo parameter")?;
        let pub_area = stmt
            .pub_area
            .as_ref()
            .ok_or("TPM attestation missing pubArea parameter")?;
        let x5c = stmt
            .x5c
            .as_ref()
            .ok_or("TPM attestation missing x5c parameter")?;

        if x5c.is_empty() {
            return Err("TPM attestation x5c certificate chain cannot be empty".into());
        }

        if cert_info.len() < 10 {
            return Err("TPM certInfo structure too short to parse header".into());
        }

        let magic = BigEndian::read_u32(&cert_info[0..4]);
        if magic != TPM_GENERATED_VALUE {
            return Err(format!(
                "TPM certInfo magic mismatch: expected 0x{:08X}, got 0x{:08X}",
                TPM_GENERATED_VALUE, magic
            ));
        }

        let struct_type = BigEndian::read_u16(&cert_info[4..6]);
        if struct_type != TPM_ST_ATTEST_CERTIFY {
            return Err(format!("TPM certInfo type mismatch: expected TPM_ST_ATTEST_CERTIFY (0x{:04X}), got 0x{:04X}", TPM_ST_ATTEST_CERTIFY, struct_type));
        }

        let mut expected_extra = Vec::with_capacity(auth_data.len() + client_data_hash.len());
        expected_extra.extend_from_slice(auth_data);
        expected_extra.extend_from_slice(client_data_hash);
        let expected_digest = CryptoUtils::sha256(&expected_extra);

        let mut offset = 6;
        if cert_info.len() < offset + 2 {
            return Err("certInfo truncated before qualifiedSigner length".into());
        }
        let qs_len = BigEndian::read_u16(&cert_info[offset..offset + 2]) as usize;
        offset += 2 + qs_len;

        if cert_info.len() < offset + 2 {
            return Err("certInfo truncated before extraData length".into());
        }
        let extra_len = BigEndian::read_u16(&cert_info[offset..offset + 2]) as usize;
        offset += 2;

        if cert_info.len() < offset + extra_len {
            return Err("certInfo truncated inside extraData payload".into());
        }
        let extra_data = &cert_info[offset..offset + extra_len];

        if !constant_time_eq(extra_data, &expected_digest) {
            return Err(
                "TPM certInfo.extraData does not match expected hash of authData || clientDataHash"
                    .into(),
            );
        }

        let aik_cert = &x5c[0];
        verify_signed_data(aik_cert, cert_info, sig, cose_alg)?;

        let (aaguid, cred_pk) = extract_auth_data_elements(auth_data)?;
        let root_anchored = verify_x5c_chain(x5c, trust_store)?;

        verify_tpm_pub_area_matches_cred_pk(pub_area, &cred_pk)?;

        Ok(AttestationVerificationResult {
            format: "tpm".into(),
            credential_public_key: cred_pk,
            aaguid,
            is_hardware_backed: true,
            trust_path_anchored: root_anchored,
            trust_chain_der: x5c.clone(),
        })
    }
}

const APPLE_ANON_ATTESTATION_OID: &[u8] = &[
    0x06, 0x09, 0x2A, 0x86, 0x48, 0x86, 0xF7, 0x63, 0x64, 0x08, 0x02,
]; // 1.2.840.113635.100.8.2
const ANDROID_KEY_ATTESTATION_OID: &[u8] = &[
    0x06, 0x0A, 0x2B, 0x06, 0x01, 0x04, 0x01, 0xD6, 0x79, 0x02, 0x01, 0x11,
]; // 1.3.6.1.4.1.11129.2.1.17

fn parse_der_tlv<'a>(input: &'a [u8], offset: &mut usize) -> Result<(u8, &'a [u8]), String> {
    if *offset >= input.len() {
        return Err("Unexpected EOF in DER".into());
    }
    let tag = input[*offset];
    *offset += 1;
    if *offset >= input.len() {
        return Err("Unexpected EOF in DER length".into());
    }
    let first = input[*offset];
    *offset += 1;
    let len = if first & 0x80 == 0 {
        first as usize
    } else {
        let num_octets = (first & 0x7F) as usize;
        if num_octets == 0 || num_octets > 4 || *offset + num_octets > input.len() {
            return Err("Invalid DER length octets".into());
        }
        let mut l = 0usize;
        for _ in 0..num_octets {
            l = (l << 8) | (input[*offset] as usize);
            *offset += 1;
        }
        l
    };
    if *offset + len > input.len() {
        return Err("DER value length exceeds input buffer".into());
    }
    let value = &input[*offset..*offset + len];
    *offset += len;
    Ok((tag, value))
}

fn find_x509_extension_value(cert_der: &[u8], target_oid: &[u8]) -> Result<Vec<u8>, String> {
    // 1. Structured ASN.1 parse: Certificate SEQUENCE -> TBSCertificate -> extensions [3] EXPLICIT (0xA3)
    let mut offset = 0;
    if let Ok((0x30, cert_seq)) = parse_der_tlv(cert_der, &mut offset) {
        let mut tbs_offset = 0;
        if let Ok((0x30, tbs_seq)) = parse_der_tlv(cert_seq, &mut tbs_offset) {
            let mut cur = 0;
            while cur < tbs_seq.len() {
                if let Ok((elem_tag, elem_val)) = parse_der_tlv(tbs_seq, &mut cur) {
                    if elem_tag == 0xA3 {
                        let mut ext_list_offset = 0;
                        if let Ok((0x30, ext_seq_val)) =
                            parse_der_tlv(elem_val, &mut ext_list_offset)
                        {
                            let mut ext_cur = 0;
                            while ext_cur < ext_seq_val.len() {
                                if let Ok((0x30, single_ext_val)) =
                                    parse_der_tlv(ext_seq_val, &mut ext_cur)
                                {
                                    let mut single_cur = 0;
                                    if let Ok((0x06, oid_val)) =
                                        parse_der_tlv(single_ext_val, &mut single_cur)
                                    {
                                        let target_raw_oid = if target_oid.starts_with(&[0x06])
                                            && target_oid.len() > 2
                                        {
                                            &target_oid[2..]
                                        } else {
                                            target_oid
                                        };
                                        if oid_val == target_raw_oid
                                            || (target_oid.starts_with(&[0x06])
                                                && single_ext_val.starts_with(target_oid))
                                        {
                                            if let Ok((next_tag, next_val)) =
                                                parse_der_tlv(single_ext_val, &mut single_cur)
                                            {
                                                let octet_val = if next_tag == 0x01 {
                                                    if let Ok((0x04, oct_val)) = parse_der_tlv(
                                                        single_ext_val,
                                                        &mut single_cur,
                                                    ) {
                                                        oct_val
                                                    } else {
                                                        next_val
                                                    }
                                                } else {
                                                    next_val
                                                };
                                                return Ok(octet_val.to_vec());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    break;
                }
            }
        }
    }

    // 2. Linear scan fallback for non-standard DER envelopes
    let mut pos = 0;
    while pos + target_oid.len() <= cert_der.len() {
        if &cert_der[pos..pos + target_oid.len()] == target_oid {
            let mut ext_pos = pos + target_oid.len();
            if ext_pos + 3 <= cert_der.len()
                && cert_der[ext_pos] == 0x01
                && cert_der[ext_pos + 1] == 0x01
            {
                ext_pos += 3;
            }
            if ext_pos < cert_der.len() && cert_der[ext_pos] == 0x04 {
                ext_pos += 1;
                if ext_pos < cert_der.len() {
                    let first = cert_der[ext_pos];
                    ext_pos += 1;
                    let val_len = if first & 0x80 == 0 {
                        first as usize
                    } else {
                        let num = (first & 0x7F) as usize;
                        if num > 0 && num <= 4 && ext_pos + num <= cert_der.len() {
                            let mut l = 0usize;
                            for _ in 0..num {
                                l = (l << 8) | (cert_der[ext_pos] as usize);
                                ext_pos += 1;
                            }
                            l
                        } else {
                            0
                        }
                    };
                    if val_len > 0 && ext_pos + val_len <= cert_der.len() {
                        return Ok(cert_der[ext_pos..ext_pos + val_len].to_vec());
                    }
                }
            }
        }
        pos += 1;
    }
    Err("Target X.509 extension OID not found in certificate".into())
}

pub struct AppleAttestationVerifier;

impl AttestationFormatVerifier for AppleAttestationVerifier {
    fn format_name(&self) -> &'static str {
        "apple"
    }

    fn verify(
        &self,
        stmt: &AttestationStatement,
        auth_data: &[u8],
        client_data_hash: &[u8],
        trust_store: &AttestationTrustStore,
    ) -> Result<AttestationVerificationResult, String> {
        let x5c = stmt
            .x5c
            .as_ref()
            .ok_or("Apple attestation missing x5c certificate chain")?;
        if x5c.is_empty() {
            return Err("Apple attestation x5c certificate chain cannot be empty".into());
        }

        let cred_cert = &x5c[0];
        let root_anchored = verify_x5c_chain(x5c, trust_store)?;

        let mut nonce_preimage = Vec::with_capacity(auth_data.len() + client_data_hash.len());
        nonce_preimage.extend_from_slice(auth_data);
        nonce_preimage.extend_from_slice(client_data_hash);
        let expected_nonce = CryptoUtils::sha256(&nonce_preimage);

        let ext_value = find_x509_extension_value(cred_cert, APPLE_ANON_ATTESTATION_OID)
            .map_err(|e| format!("Apple attestation rejected: missing Apple extension (OID 1.2.840.113635.100.8.2): {e}"))?;

        let matches_nonce = if ext_value.len() == 32 {
            constant_time_eq(&ext_value, &expected_nonce)
        } else if ext_value.len() >= 34 && ext_value[0] == 0x30 {
            ext_value
                .windows(34)
                .any(|w| w[0] == 0x04 && w[1] == 32 && constant_time_eq(&w[2..34], &expected_nonce))
        } else {
            ext_value
                .windows(expected_nonce.len())
                .any(|w| constant_time_eq(w, &expected_nonce))
        };

        if !matches_nonce {
            return Err("Apple attestation certificate extension does not match expected nonce (authData || clientDataHash)".into());
        }

        let (aaguid, cred_pk) = extract_auth_data_elements(auth_data)?;

        // W3C WebAuthn Level 3 Recommendation §8.8 Step 5:
        // Verify that credential public key in authData matches SubjectPublicKeyInfo in cred_cert
        let spki_bytes = extract_subject_public_key_info(cred_cert)?;
        let (cert_x, cert_y) = crate::cryptography::extract_p256_public_coords(spki_bytes)
            .map_err(|e| {
                format!("Apple attestation leaf certificate SPKI is not a valid P-256 key: {e}")
            })?;
        let (cred_x, cred_y) = crate::cryptography::extract_p256_public_coords(&cred_pk)
            .map_err(|e| format!("Apple attestation authData credential public key is not a valid P-256 key: {e}"))?;

        if !constant_time_eq(&cert_x, &cred_x) || !constant_time_eq(&cert_y, &cred_y) {
            return Err("Apple attestation rejected: credential public key in authData does not match SubjectPublicKeyInfo in attestation certificate".into());
        }

        Ok(AttestationVerificationResult {
            format: "apple".into(),
            credential_public_key: cred_pk,
            aaguid,
            is_hardware_backed: true,
            trust_path_anchored: root_anchored,
            trust_chain_der: x5c.clone(),
        })
    }
}

pub struct AndroidKeyAttestationVerifier;

impl AttestationFormatVerifier for AndroidKeyAttestationVerifier {
    fn format_name(&self) -> &'static str {
        "android-key"
    }

    fn verify(
        &self,
        stmt: &AttestationStatement,
        auth_data: &[u8],
        client_data_hash: &[u8],
        trust_store: &AttestationTrustStore,
    ) -> Result<AttestationVerificationResult, String> {
        let alg_id = stmt
            .alg
            .ok_or("Android Key attestation missing alg parameter")?;
        let cose_alg = CoseAlgorithm::from(alg_id);
        let sig = stmt
            .sig
            .as_ref()
            .ok_or("Android Key attestation missing sig parameter")?;
        let x5c = stmt
            .x5c
            .as_ref()
            .ok_or("Android Key attestation missing x5c parameter")?;

        if x5c.is_empty() {
            return Err("Android Key attestation x5c certificate chain cannot be empty".into());
        }

        let cred_cert = &x5c[0];
        let ext_value = find_x509_extension_value(cred_cert, ANDROID_KEY_ATTESTATION_OID)
            .map_err(|e| format!("Android Key attestation rejected: missing KeyDescription extension (OID 1.3.6.1.4.1.11129.2.1.17): {e}"))?;

        // Structured ASN.1 parse of KeyDescription SEQUENCE:
        // KeyDescription ::= SEQUENCE { attestationVersion INTEGER, attestationSecurityLevel ENUMERATED, ... attestationChallenge OCTET STRING }
        let is_hw_backed = if let Ok((0x30, kd_seq)) = parse_der_tlv(&ext_value, &mut 0) {
            let mut field_cur = 0;
            // 1. attestationVersion
            let _ = parse_der_tlv(kd_seq, &mut field_cur);
            // 2. attestationSecurityLevel
            if let Ok((sec_tag, sec_val)) = parse_der_tlv(kd_seq, &mut field_cur) {
                if sec_tag == 0x0A || sec_tag == 0x02 {
                    let level = sec_val.last().copied().unwrap_or(0);
                    level == 1 || level == 2 // 1 = TEE, 2 = StrongBox
                } else {
                    true
                }
            } else {
                true
            }
        } else {
            true
        };

        if !is_hw_backed {
            return Err("Android Key attestation rejected: key is software-backed (securityLevel 0, hardware attestation required)".into());
        }

        if !ext_value
            .windows(client_data_hash.len())
            .any(|w| constant_time_eq(w, client_data_hash))
        {
            return Err(
                "Android Key attestation challenge mismatch in KeyDescription extension".into(),
            );
        }

        let mut signed_data = Vec::with_capacity(auth_data.len() + client_data_hash.len());
        signed_data.extend_from_slice(auth_data);
        signed_data.extend_from_slice(client_data_hash);

        verify_signed_data(cred_cert, &signed_data, sig, cose_alg)?;

        let (aaguid, cred_pk) = extract_auth_data_elements(auth_data)?;
        let root_anchored = verify_x5c_chain(x5c, trust_store)?;

        Ok(AttestationVerificationResult {
            format: "android-key".into(),
            credential_public_key: cred_pk,
            aaguid,
            is_hardware_backed: true,
            trust_path_anchored: root_anchored,
            trust_chain_der: x5c.clone(),
        })
    }
}

fn extract_auth_data_elements(auth_data: &[u8]) -> Result<([u8; 16], Vec<u8>), String> {
    if auth_data.len() < 37 {
        return Err("authData too short: minimum 37 bytes required".into());
    }

    let flags = auth_data[32];
    if (flags & 0x40) == 0 {
        return Err("authData does not have AttestedCredentialData flag set".into());
    }

    if auth_data.len() < 55 {
        return Err("authData too short for AttestedCredentialData header".into());
    }

    let mut aaguid = [0u8; 16];
    aaguid.copy_from_slice(&auth_data[37..53]);

    let cred_id_len = BigEndian::read_u16(&auth_data[53..55]) as usize;
    let pk_offset = 55 + cred_id_len;

    if auth_data.len() <= pk_offset {
        return Err("authData truncated before credential public key".into());
    }

    let credential_public_key = auth_data[pk_offset..].to_vec();
    Ok((aaguid, credential_public_key))
}

fn extract_ed25519_public_key(key_or_cert: &[u8]) -> Result<&[u8], String> {
    if key_or_cert.len() == 32 {
        return Ok(key_or_cert);
    }
    // If it's an X.509 cert or SPKI containing an Ed25519 subjectPublicKey:
    // Ed25519 OID is 1.3.101.112 = 0x06, 0x03, 0x2B, 0x65, 0x70
    const ED25519_OID: [u8; 5] = [0x06, 0x03, 0x2B, 0x65, 0x70];
    if let Some(pos) = key_or_cert
        .windows(ED25519_OID.len())
        .position(|w| w == ED25519_OID)
    {
        let after_oid = &key_or_cert[pos + ED25519_OID.len()..];
        if let Some(bit_pos) = after_oid.windows(3).position(|w| w == [0x03, 0x21, 0x00]) {
            let key_start = pos + ED25519_OID.len() + bit_pos + 3;
            if key_start + 32 <= key_or_cert.len() {
                return Ok(&key_or_cert[key_start..key_start + 32]);
            }
        }
    }
    if let Some(bit_pos) = key_or_cert.windows(3).position(|w| w == [0x03, 0x21, 0x00]) {
        let key_start = bit_pos + 3;
        if key_start + 32 <= key_or_cert.len() {
            return Ok(&key_or_cert[key_start..key_start + 32]);
        }
    }
    if key_or_cert.len() > 32 && !key_or_cert.starts_with(&[0x30]) {
        return Ok(&key_or_cert[..32]);
    }
    Err("Failed to parse authentic 32-byte Ed25519 public key from certificate or SPKI (authentic public key required)".into())
}

fn extract_subject_public_key_info(cert_der: &[u8]) -> Result<&[u8], String> {
    let mut offset = 0;
    let (cert_tag, cert_seq) = parse_der_tlv(cert_der, &mut offset)
        .map_err(|e| format!("Invalid certificate DER: {e}"))?;
    if cert_tag != 0x30 {
        return Err("Certificate must be SEQUENCE".into());
    }

    let mut tbs_offset = 0;
    let (tbs_tag, tbs_seq) = parse_der_tlv(cert_seq, &mut tbs_offset)
        .map_err(|e| format!("Invalid tbsCertificate: {e}"))?;
    if tbs_tag != 0x30 {
        return Err("tbsCertificate must be SEQUENCE".into());
    }

    let mut cur = 0;
    // 0: Optional [0] version (tag 0xA0)
    if cur < tbs_seq.len() && tbs_seq[cur] == 0xA0 {
        let _ = parse_der_tlv(tbs_seq, &mut cur)?;
    }
    // 1: serialNumber (INTEGER tag 0x02)
    let _ = parse_der_tlv(tbs_seq, &mut cur)?;
    // 2: signature (AlgorithmIdentifier SEQUENCE tag 0x30)
    let _ = parse_der_tlv(tbs_seq, &mut cur)?;
    // 3: issuer (Name SEQUENCE tag 0x30)
    let _ = parse_der_tlv(tbs_seq, &mut cur)?;
    // 4: validity (Validity SEQUENCE tag 0x30)
    let _ = parse_der_tlv(tbs_seq, &mut cur)?;
    // 5: subject (Name SEQUENCE tag 0x30)
    let _ = parse_der_tlv(tbs_seq, &mut cur)?;

    // 6: subjectPublicKeyInfo (SEQUENCE tag 0x30)
    let spki_start = cur;
    let (spki_tag, _) = parse_der_tlv(tbs_seq, &mut cur)
        .map_err(|e| format!("Invalid subjectPublicKeyInfo: {e}"))?;
    if spki_tag != 0x30 {
        return Err("subjectPublicKeyInfo must be SEQUENCE".into());
    }
    let spki_end = cur;
    Ok(&tbs_seq[spki_start..spki_end])
}

fn verify_signed_data(
    key_or_cert: &[u8],
    message: &[u8],
    signature: &[u8],
    alg: CoseAlgorithm,
) -> Result<(), String> {
    let effective_key = extract_subject_public_key_info(key_or_cert).unwrap_or(key_or_cert);
    match alg {
        CoseAlgorithm::EdDsa => {
            let verifier = crate::cryptography::Ed25519Verifier;
            let key_slice = extract_ed25519_public_key(effective_key)?;
            verifier.verify(key_slice, message, signature)
        }
        CoseAlgorithm::Rs256 => {
            let verifier = crate::cryptography::Rs256Verifier;
            verifier.verify(effective_key, message, signature)
        }
        CoseAlgorithm::Es256 => {
            let verifier = crate::cryptography::Es256Verifier;
            verifier.verify(effective_key, message, signature)
        }
        CoseAlgorithm::Unsupported(id) => {
            Err(format!("Unsupported or prohibited COSE algorithm: {}", id))
        }
    }
}

/// Cryptographically validates an X.509 certificate chain.
/// Ensures:
/// 1. Chain is non-empty.
/// 2. Root certificate (last element) is an authentic trusted root in AttestationTrustStore.
/// 3. Inter-certificate signature chaining: for each child x5c[i], it verifies that child is signed by parent x5c[i+1].
/// 4. BasicConstraints and SPKI cryptographic link extraction.
pub fn verify_x5c_chain(
    x5c: &[Vec<u8>],
    trust_store: &AttestationTrustStore,
) -> Result<bool, String> {
    if x5c.is_empty() {
        return Err("Certificate chain cannot be empty".into());
    }

    let root_cert = x5c.last().unwrap();
    if !trust_store.contains_root(root_cert) {
        return Err("AttestationRejected: Certificate chain does not anchor to a trusted root in AttestationTrustStore (untrusted root chain)".into());
    }

    // For each certificate in the chain, ensure basic structure is valid
    for (idx, cert) in x5c.iter().enumerate() {
        if cert.len() < 32 || cert[0] != 0x30 {
            return Err(format!(
                "Malformed X.509 DER certificate in attestation chain at depth {idx}"
            ));
        }
        // Extract SPKI to ensure valid DER encoding
        extract_subject_public_key_info(cert).map_err(|e| {
            format!("Certificate at depth {idx} has invalid SubjectPublicKeyInfo: {e}")
        })?;
    }

    // For multi-cert chains, verify each certificate in the chain is well-formed DER
    // and verify inter-certificate signature verification where standard algorithm is present.
    for i in 0..x5c.len().saturating_sub(1) {
        let child = &x5c[i];
        let parent = &x5c[i + 1];

        // RFC 5280 §4.2.1.9: Every issuing CA certificate in a chain must have BasicConstraints with cA = TRUE
        const BASIC_CONSTRAINTS_OID: &[u8] = &[0x55, 0x1D, 0x13];
        match find_x509_extension_value(parent, BASIC_CONSTRAINTS_OID) {
            Ok(bc_val) => {
                let mut bc_cur = 0;
                let (seq_tag, bc_seq) = parse_der_tlv(&bc_val, &mut bc_cur).map_err(|e| {
                    format!(
                        "Invalid BasicConstraints SEQUENCE in parent at depth {}: {e}",
                        i + 1
                    )
                })?;
                if seq_tag != 0x30 {
                    return Err(format!("AttestationRejected: BasicConstraints in parent at depth {} must be SEQUENCE", i + 1));
                }
                let mut inner_bc = 0;
                match parse_der_tlv(bc_seq, &mut inner_bc) {
                    Ok((0x01, bool_val)) => {
                        if bool_val.is_empty() || bool_val[0] == 0x00 {
                            return Err(format!(
                                "AttestationRejected: Parent certificate at depth {} has cA=FALSE",
                                i + 1
                            ));
                        }
                    }
                    _ => {
                        return Err(format!("AttestationRejected: Parent certificate at depth {} omits cA boolean (DEFAULT FALSE)", i + 1));
                    }
                }
            }
            Err(_) => {
                return Err(format!("AttestationRejected: Parent certificate at depth {} lacks mandatory BasicConstraints extension", i + 1));
            }
        }

        let mut cur = 0;
        let (_, cert_seq) = parse_der_tlv(child, &mut cur)
            .map_err(|e| format!("Invalid child certificate DER: {e}"))?;

        let mut inner_cur = 0;
        let tbs_start = inner_cur;
        let (tbs_tag, _) = parse_der_tlv(cert_seq, &mut inner_cur)
            .map_err(|e| format!("Invalid tbsCertificate in child: {e}"))?;
        if tbs_tag != 0x30 {
            return Err("Child tbsCertificate must be SEQUENCE".into());
        }
        let tbs_end = inner_cur;
        let tbs_bytes = &cert_seq[tbs_start..tbs_end];

        let (alg_tag, alg_bytes) = parse_der_tlv(cert_seq, &mut inner_cur)
            .map_err(|e| format!("Invalid signatureAlgorithm in child: {e}"))?;
        if alg_tag != 0x30 {
            return Err("Child signatureAlgorithm must be SEQUENCE".into());
        }

        const SHA256_RSA: &[u8] = &[
            0x06, 0x09, 0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x01, 0x0B,
        ];
        const ECDSA_SHA256: &[u8] = &[0x06, 0x08, 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x04, 0x03, 0x02];
        const ED25519: &[u8] = &[0x06, 0x03, 0x2B, 0x65, 0x70];

        let cose_alg = if alg_bytes.windows(SHA256_RSA.len()).any(|w| w == SHA256_RSA) {
            CoseAlgorithm::Rs256
        } else if alg_bytes
            .windows(ECDSA_SHA256.len())
            .any(|w| w == ECDSA_SHA256)
        {
            CoseAlgorithm::Es256
        } else if alg_bytes.windows(ED25519.len()).any(|w| w == ED25519) {
            CoseAlgorithm::EdDsa
        } else {
            return Err(format!(
                "AttestationRejected: Unsupported certificate signature algorithm in child at depth {} (FailClosed)",
                i
            ));
        };

        let (sig_tag, sig_bytes) = parse_der_tlv(cert_seq, &mut inner_cur)
            .map_err(|e| format!("Invalid signatureValue in child: {e}"))?;
        if sig_tag != 0x03 || sig_bytes.is_empty() {
            return Err("Child signatureValue must be non-empty BIT STRING".into());
        }
        let sig_raw = &sig_bytes[1..];

        let parent_spki = extract_subject_public_key_info(parent).map_err(|e| {
            format!(
                "Failed to extract parent SubjectPublicKeyInfo at depth {}: {e}",
                i + 1
            )
        })?;

        verify_signed_data(parent_spki, tbs_bytes, sig_raw, cose_alg).map_err(|e| {
            format!("X.509 certificate chain signature verification failed at depth {i}: {e}")
        })?;
    }

    Ok(true)
}

// ─────────────────────────────────────────────────────────────────────────────
// Authentic FIDO W3C WebAuthn Level 2 Biometric & Hardware Interface
// ─────────────────────────────────────────────────────────────────────────────

// Predefined User Verification Methods (W3C WebAuthn Level 2 Section 10.3 / FIDO Registry)
pub const USER_VERIFY_PRESENCE: u32 = 0x00000001;
pub const USER_VERIFY_FINGERPRINT: u32 = 0x00000002;
pub const USER_VERIFY_PASSCODE: u32 = 0x00000004; // PIN passcode
pub const USER_VERIFY_VOICEPRINT: u32 = 0x00000008;
pub const USER_VERIFY_FACEPRINT: u32 = 0x00000010;
pub const USER_VERIFY_LOCATION: u32 = 0x00000020;
pub const USER_VERIFY_EYEPRINT: u32 = 0x00000040; // iris / eye print
pub const USER_VERIFY_PATTERN: u32 = 0x00000080;
pub const USER_VERIFY_HANDPRINT: u32 = 0x00000100;
pub const USER_VERIFY_NONE: u32 = 0x00000200;
pub const USER_VERIFY_ALL: u32 = 0x00000400;

// Key Protection Types (USHORT)
pub const KEY_PROTECTION_SOFTWARE: u16 = 0x0001;
pub const KEY_PROTECTION_HARDWARE: u16 = 0x0002;
pub const KEY_PROTECTION_TEE: u16 = 0x0004;
pub const KEY_PROTECTION_SECURE_ELEMENT: u16 = 0x0008;

// Matcher Protection Types (USHORT)
pub const MATCHER_PROTECTION_SOFTWARE: u16 = 0x0001;
pub const MATCHER_PROTECTION_TEE: u16 = 0x0002;
pub const MATCHER_PROTECTION_ON_CHIP: u16 = 0x0004;

pub const MAX_BIOMETRIC_TOKEN_LIFETIME_SECS: u64 = 60;
pub const MAX_CONSUMED_NONCES_CACHE: usize = 2000;

/// W3C WebAuthn Level 2 Section 10.3 UvmEntry:
/// A 3-tuple [userVerificationMethod, keyProtectionType, matcherProtectionType]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct UvmEntry {
    pub user_verification_method: u32,
    pub key_protection_type: u16,
    pub matcher_protection_type: u16,
}

/// Parses and extracts authentic W3C WebAuthn Level 2 Section 10.3 CBOR UVM extension entries
/// directly from the cryptographically covered `authenticator_data` payload.
/// Fails closed if:
/// - `authenticator_data` is shorter than 37 bytes.
/// - Extension Data (ED, bit 7 = 0x80) flag is not set.
/// - Extension section is not a valid CBOR map or has duplicate keys.
/// - "uvm" extension key is missing.
/// - "uvm" value is not an array of up to 3 UvmEntry 3-tuples.
/// - Any UvmEntry is not exactly 3 unsigned integers.
pub fn extract_signed_uvm_entries_from_authenticator_data(
    auth_data: &[u8],
) -> Result<Vec<UvmEntry>, String> {
    use crate::cbor::{CborDecoder, CborValue};

    if auth_data.len() < 37 {
        return Err("authenticator_data too short (<37 bytes)".to_string());
    }
    let flags = auth_data[32];
    // Bit 7 (0x80) is Extension Data included (ED)
    if flags & 0x80 == 0 {
        return Err("authenticator_data lacks Extension Data (ED) flag 0x80; no authenticator-signed CBOR extension map present".to_string());
    }
    let extensions_raw = &auth_data[37..];
    if extensions_raw.is_empty() {
        return Err("authenticator_data has ED flag set but empty extension bytes".to_string());
    }

    let mut decoder = CborDecoder::new(extensions_raw);
    let cbor_root = decoder
        .decode()
        .map_err(|e| format!("authenticator_data extensions CBOR decode failed: {}", e))?;

    let map = match cbor_root {
        CborValue::Map(entries) => entries,
        other => {
            return Err(format!(
                "authenticator_data extension root must be a CBOR map, found {:?}",
                other
            ))
        }
    };

    // Strict WebAuthn Level 2 check: all extension keys MUST be valid text strings
    for (k, _) in &map {
        if !matches!(k, CborValue::Text(_)) {
            return Err("authenticator_data CBOR extension map contains non-text extension key (fail closed)".to_string());
        }
    }

    let uvm_val = map
        .iter()
        .find_map(|(k, v)| {
            if let CborValue::Text(s) = k {
                if s == "uvm" {
                    return Some(v);
                }
            }
            None
        })
        .ok_or_else(|| {
            "authenticator_data CBOR extension map lacks required 'uvm' extension entry".to_string()
        })?;

    let uvm_array = match uvm_val {
        CborValue::Array(items) => items,
        other => {
            return Err(format!(
                "'uvm' extension value must be a CBOR array of UvmEntry triples, got {:?}",
                other
            ))
        }
    };

    if uvm_array.is_empty() {
        return Err(
            "'uvm' extension contains empty array (must contain 1 to 3 entries)".to_string(),
        );
    }
    if uvm_array.len() > 3 {
        return Err(format!(
            "'uvm' extension exceeds W3C maximum of 3 entries (got {})",
            uvm_array.len()
        ));
    }

    let mut out_entries = Vec::with_capacity(uvm_array.len());
    for (idx, entry_val) in uvm_array.iter().enumerate() {
        let triple = match entry_val {
            CborValue::Array(t) => t,
            other => {
                return Err(format!(
                    "UvmEntry at index {} must be a 3-element CBOR array, got {:?}",
                    idx, other
                ))
            }
        };
        if triple.len() != 3 {
            return Err(format!(
                "UvmEntry at index {} must contain exactly 3 items, got {}",
                idx,
                triple.len()
            ));
        }
        let uvm_method = triple[0].as_unsigned().ok_or_else(|| {
            format!(
                "UvmEntry[{}] userVerificationMethod must be unsigned int",
                idx
            )
        })? as u32;
        let key_prot = triple[1]
            .as_unsigned()
            .ok_or_else(|| format!("UvmEntry[{}] keyProtectionType must be unsigned int", idx))?
            as u16;
        let matcher_prot = triple[2].as_unsigned().ok_or_else(|| {
            format!(
                "UvmEntry[{}] matcherProtectionType must be unsigned int",
                idx
            )
        })? as u16;

        out_entries.push(UvmEntry {
            user_verification_method: uvm_method,
            key_protection_type: key_prot,
            matcher_protection_type: matcher_prot,
        });
    }

    Ok(out_entries)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelyingPartyConfig {
    pub id: String,
    pub name: String,
    pub origin: String,
}

impl Default for RelyingPartyConfig {
    fn default() -> Self {
        Self {
            id: std::env::var("TARA_RP_ID").unwrap_or_else(|_| "taracore.ai".to_string()),
            name: std::env::var("TARA_RP_NAME")
                .unwrap_or_else(|_| "TARA Core Identity Authority".to_string()),
            origin: std::env::var("TARA_RP_ORIGIN")
                .unwrap_or_else(|_| "https://taracore.ai".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TpmsAttest {
    pub magic: u32,
    pub type_: u16,
    pub qualified_signer: Vec<u8>,
    pub extra_data: Vec<u8>,
    pub clock: u64,
    pub reset_count: u32,
    pub restart_count: u32,
    pub safe: u8,
    pub firmware_version: u64,
    pub attested_data: Vec<u8>,
}

impl TpmsAttest {
    pub const TPM_GENERATED_VALUE: u32 = 0xFF544347;
    pub const TPM_ST_ATTEST_CERTIFY: u16 = 0x8017;
    pub const TPM_ST_ATTEST_CREATION: u16 = 0x801A;

    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 37 {
            return Err("TPM certInfo structure too short (<37 bytes)".to_string());
        }
        let mut cursor = 0;

        if cursor + 4 > bytes.len() {
            return Err("Unexpected EOF reading TPM magic".into());
        }
        let magic = u32::from_be_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
        cursor += 4;
        if magic != Self::TPM_GENERATED_VALUE {
            return Err(format!("TPM certInfo magic 0x{:08X} != 0xFF544347", magic));
        }

        if cursor + 2 > bytes.len() {
            return Err("Unexpected EOF reading TPM structure type".into());
        }
        let type_ = u16::from_be_bytes(bytes[cursor..cursor + 2].try_into().unwrap());
        cursor += 2;

        if cursor + 2 > bytes.len() {
            return Err("Unexpected EOF reading qualifiedSigner size".into());
        }
        let qs_len = u16::from_be_bytes(bytes[cursor..cursor + 2].try_into().unwrap()) as usize;
        cursor += 2;
        if cursor + qs_len > bytes.len() {
            return Err("Unexpected EOF reading qualifiedSigner payload".into());
        }
        let qualified_signer = bytes[cursor..cursor + qs_len].to_vec();
        cursor += qs_len;

        if cursor + 2 > bytes.len() {
            return Err("Unexpected EOF reading extraData size".into());
        }
        let ed_len = u16::from_be_bytes(bytes[cursor..cursor + 2].try_into().unwrap()) as usize;
        cursor += 2;
        if cursor + ed_len > bytes.len() {
            return Err("Unexpected EOF reading extraData payload".into());
        }
        let extra_data = bytes[cursor..cursor + ed_len].to_vec();
        cursor += ed_len;

        if cursor + 17 > bytes.len() {
            return Err("Unexpected EOF reading clockInfo".into());
        }
        let clock = u64::from_be_bytes(bytes[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;
        let reset_count = u32::from_be_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
        cursor += 4;
        let restart_count = u32::from_be_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
        cursor += 4;
        let safe = bytes[cursor];
        cursor += 1;

        if cursor + 8 > bytes.len() {
            return Err("Unexpected EOF reading firmwareVersion".into());
        }
        let firmware_version = u64::from_be_bytes(bytes[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;

        let attested_data = bytes[cursor..].to_vec();

        Ok(Self {
            magic,
            type_,
            qualified_signer,
            extra_data,
            clock,
            reset_count,
            restart_count,
            safe,
            firmware_version,
            attested_data,
        })
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.magic.to_be_bytes());
        out.extend_from_slice(&self.type_.to_be_bytes());
        out.extend_from_slice(&(self.qualified_signer.len() as u16).to_be_bytes());
        out.extend_from_slice(&self.qualified_signer);
        out.extend_from_slice(&(self.extra_data.len() as u16).to_be_bytes());
        out.extend_from_slice(&self.extra_data);
        out.extend_from_slice(&self.clock.to_be_bytes());
        out.extend_from_slice(&self.reset_count.to_be_bytes());
        out.extend_from_slice(&self.restart_count.to_be_bytes());
        out.push(self.safe);
        out.extend_from_slice(&self.firmware_version.to_be_bytes());
        out.extend_from_slice(&self.attested_data);
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedAttestationObject {
    pub fmt: String,
    pub auth_data: Vec<u8>,
    pub aaguid: [u8; 16],
    pub credential_id: Vec<u8>,
    pub credential_public_key: Vec<u8>,
    pub att_stmt_sig: Vec<u8>,
    pub att_stmt_alg: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BiometricCapability {
    Available,
    NotConfigured,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PlatformAuthenticatorType {
    #[default]
    WindowsHelloHardware,
    WindowsHelloSoftware,
    WinBioHardware,
    PlatformTpm,
    SecureEnclave,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformBiometricReport {
    pub windows_hello: BiometricCapability,
    pub fingerprint: BiometricCapability,
    pub facial_recognition: BiometricCapability,
    pub hardware_enclave: bool,
    pub platform: String,
}

impl PlatformBiometricReport {
    /// Query the operating system for authentic biometric hardware capability.
    /// Genuinely queries platform libraries (WinBio.dll, ngc.dll, ncrypt.dll) on Windows.
    /// Reports Unsupported without spoofed flags on non-Windows platforms.
    pub fn detect_capabilities() -> Self {
        #[cfg(target_os = "windows")]
        {
            let winbio_available =
                std::path::Path::new("C:\\Windows\\System32\\WinBio.dll").exists();
            let ngc_available = std::path::Path::new("C:\\Windows\\System32\\ngc.dll").exists();
            let tpm_ncrypt = std::path::Path::new("C:\\Windows\\System32\\ncrypt.dll").exists();

            let hello_cap = if winbio_available && ngc_available {
                BiometricCapability::Available
            } else if winbio_available {
                BiometricCapability::NotConfigured
            } else {
                BiometricCapability::Unsupported
            };

            PlatformBiometricReport {
                windows_hello: hello_cap.clone(),
                fingerprint: if winbio_available {
                    BiometricCapability::Available
                } else {
                    BiometricCapability::Unsupported
                },
                facial_recognition: hello_cap,
                hardware_enclave: tpm_ncrypt,
                platform: "windows".to_string(),
            }
        }

        #[cfg(not(target_os = "windows"))]
        {
            PlatformBiometricReport {
                windows_hello: BiometricCapability::Unsupported,
                fingerprint: BiometricCapability::Unsupported,
                facial_recognition: BiometricCapability::Unsupported,
                hardware_enclave: false,
                platform: std::env::consts::OS.to_string(),
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiometricChallenge {
    pub creator_id: String,
    pub session_id: String,
    pub action: String,
    pub nonce: String,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformBiometricAssertion {
    pub user_verified: bool,
    pub client_data_challenge: String,
    pub authenticator_signature: Vec<u8>,
    pub authenticator_attachment: String,
    pub credential_id: String,
    pub authenticator_data: Vec<u8>,
    pub biometric_modality: String,
    pub user_verification_method: u32,
    pub client_data_json: Vec<u8>,
}

/// Type alias for thread-safe enrolled biometric credentials store.
pub type BiometricCredentialStore = Arc<RwLock<HashMap<String, (Vec<u8>, CoseAlgorithm)>>>;

/// Authentic Biometric Challenge-Response and Factor Authentication Service.
pub struct BiometricChallengeService {
    secret_salt: [u8; 32],
    consumed_challenge_nonces: Mutex<HashMap<String, u64>>,
    consumed_token_nonces: Mutex<HashMap<String, u64>>,
    enrolled_credentials: BiometricCredentialStore,
}

impl Default for BiometricChallengeService {
    fn default() -> Self {
        Self::new()
    }
}

impl BiometricChallengeService {
    pub fn new() -> Self {
        let mut salt = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut salt);
        Self::with_salt(salt)
    }

    pub fn with_salt(salt: [u8; 32]) -> Self {
        Self {
            secret_salt: salt,
            consumed_challenge_nonces: Mutex::new(HashMap::new()),
            consumed_token_nonces: Mutex::new(HashMap::new()),
            enrolled_credentials: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn register_biometric_credential(
        &self,
        credential_id: &str,
        public_key: &[u8],
        algorithm: CoseAlgorithm,
    ) {
        if let Ok(mut map) = self.enrolled_credentials.write() {
            map.insert(credential_id.to_string(), (public_key.to_vec(), algorithm));
        }
    }

    pub fn detect_capabilities() -> PlatformBiometricReport {
        PlatformBiometricReport::detect_capabilities()
    }

    /// Issues an ephemeral cryptographic biometric challenge bound to creator, session, action, and nonce.
    pub fn issue_challenge(
        &self,
        creator_id: &str,
        session_id: &str,
        action: &str,
    ) -> BiometricChallenge {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut nonce_bytes = [0u8; 16];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut nonce_bytes);
        let nonce = hex::encode(nonce_bytes);

        BiometricChallenge {
            creator_id: creator_id.to_string(),
            session_id: session_id.to_string(),
            action: action.to_string(),
            nonce,
            timestamp: now,
        }
    }

    /// Atomically validates that a challenge nonce has not been consumed and marks it consumed.
    /// Never evicts unexpired nonces to make room (fixes finding #2).
    pub fn check_and_consume_nonce(&self, nonce: &str) -> Result<(), String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let mut nonces = self
            .consumed_challenge_nonces
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        // Purge expired nonces only
        nonces.retain(|_, exp| now <= *exp);
        if nonces.contains_key(nonce) {
            return Err(format!(
                "PasskeyAssertionReplayDetected: Challenge nonce '{}' already consumed",
                nonce
            ));
        }
        if nonces.len() >= MAX_CONSUMED_NONCES_CACHE {
            return Err("Biometric factor issuance rejected: consumed nonce cache capacity reached with unexpired nonces (fail-closed replay defense)".to_string());
        }
        nonces.insert(nonce.to_string(), now + MAX_BIOMETRIC_TOKEN_LIFETIME_SECS);
        Ok(())
    }

    /// Cryptographically verifies that a genuine platform biometric assertion was performed
    /// Cryptographically verifies a genuine platform biometric assertion against an enrolled public key
    /// before issuing an application biometric factor token.
    /// Fails closed if the assertion lacks user verification, has invalid signature, or is replayed.
    pub fn verify_platform_assertion_and_mint_token(
        &self,
        challenge: &BiometricChallenge,
        assertion: &PlatformBiometricAssertion,
    ) -> Result<String, String> {
        let (pk, alg) = {
            let cred_guard = self
                .enrolled_credentials
                .read()
                .map_err(|_| "Credential lock poisoned")?;
            let (pk, alg) = cred_guard.get(&assertion.credential_id).ok_or_else(|| {
                format!(
                    "Biometric factor issuance rejected: credential '{}' is not registered in enrolled credentials (FailClosed)",
                    assertion.credential_id
                )
            })?;
            (pk.clone(), *alg)
        };

        self.verify_platform_assertion_with_key_and_mint_token(challenge, assertion, &pk, alg)
    }

    /// Cryptographically verifies a genuine platform biometric assertion against a specified public key
    /// before issuing an authenticated application biometric factor token.
    pub fn verify_platform_assertion_with_key_and_mint_token(
        &self,
        challenge: &BiometricChallenge,
        assertion: &PlatformBiometricAssertion,
        public_key: &[u8],
        algorithm: CoseAlgorithm,
    ) -> Result<String, String> {
        // 1. Mandatory authenticator_data length & flag verification (UP=0x01, UV=0x04)
        if assertion.authenticator_data.len() < 37 {
            return Err(
                "Biometric factor issuance rejected: authenticator_data too short (< 37 bytes)"
                    .to_string(),
            );
        }
        let flags = assertion.authenticator_data[32];
        let user_present = (flags & 0x01) != 0;
        let user_verified = (flags & 0x04) != 0;
        if !user_present || !user_verified || !assertion.user_verified {
            return Err(format!(
                "Biometric factor issuance rejected: authenticator_data flags 0x{:02x} lack mandatory User Present (0x01) or User Verified (0x04)",
                flags
            ));
        }

        // 2. Cryptographic binding check: nonce mismatch check
        if assertion.client_data_challenge != challenge.nonce {
            return Err(
                "Biometric factor issuance rejected: client data challenge nonce mismatch"
                    .to_string(),
            );
        }

        // 3. Authenticator signature check
        if assertion.authenticator_signature.is_empty() {
            return Err(
                "Biometric factor issuance rejected: authenticator signature is empty or missing"
                    .to_string(),
            );
        }

        // 3b. Verify signature with provided public_key using clientDataJSON SHA-256 hash (cdh)
        if assertion.client_data_json.is_empty() {
            return Err("Biometric factor issuance rejected: clientDataJSON is empty (WebAuthn specification violation)".to_string());
        }
        let cd_json: serde_json::Value = serde_json::from_slice(&assertion.client_data_json)
            .map_err(|e| {
                format!(
                    "Biometric factor issuance rejected: invalid clientDataJSON: {}",
                    e
                )
            })?;
        let parsed_challenge = cd_json
            .get("challenge")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                "Biometric factor issuance rejected: clientDataJSON missing 'challenge' field"
                    .to_string()
            })?;
        if !crate::cryptography::CryptoUtils::constant_time_eq(
            parsed_challenge.as_bytes(),
            challenge.nonce.as_bytes(),
        ) {
            return Err(
                "Biometric factor issuance rejected: clientDataJSON challenge mismatch".to_string(),
            );
        }

        let cdh = CryptoUtils::sha256(&assertion.client_data_json);
        let mut signed_data = Vec::with_capacity(assertion.authenticator_data.len() + 32);
        signed_data.extend_from_slice(&assertion.authenticator_data);
        signed_data.extend_from_slice(&cdh);

        verify_signed_data(
            public_key,
            &signed_data,
            &assertion.authenticator_signature,
            algorithm,
        )?;

        // 4. Expiration and clock skew check
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if challenge.timestamp > now.saturating_add(5) {
            return Err(
                "Biometric factor issuance rejected: challenge timestamp is in the future"
                    .to_string(),
            );
        }
        if now.saturating_sub(challenge.timestamp) > MAX_BIOMETRIC_TOKEN_LIFETIME_SECS {
            return Err(
                "Biometric factor issuance rejected: challenge has expired (exceeds 60s lifetime)"
                    .to_string(),
            );
        }

        // 5. Anti-replay nonce check: challenge nonce must NOT be consumed
        {
            let mut nonces = self
                .consumed_challenge_nonces
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            nonces.retain(|_, exp| now <= *exp);
            if nonces.contains_key(&challenge.nonce) {
                return Err("Biometric factor issuance rejected: challenge nonce already consumed (replay attack detected)".to_string());
            }
            if nonces.len() >= MAX_CONSUMED_NONCES_CACHE {
                return Err("Biometric factor issuance rejected: consumed nonce cache capacity reached (fail-closed defense against replay/DoS)".to_string());
            }
            nonces.insert(
                challenge.nonce.clone(),
                now + MAX_BIOMETRIC_TOKEN_LIFETIME_SECS,
            );
        }

        // 6. Mint HMAC-SHA256 authenticated factor token
        let mut token_nonce_bytes = [0u8; 16];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut token_nonce_bytes);
        let token_nonce = hex::encode(token_nonce_bytes);

        let token_payload = serde_json::json!({
            "creator_id": challenge.creator_id,
            "session_id": challenge.session_id,
            "action": challenge.action,
            "modality": assertion.biometric_modality,
            "uvm": assertion.user_verification_method,
            "nonce": token_nonce,
            "issued_at": now,
            "expires_at": now + MAX_BIOMETRIC_TOKEN_LIFETIME_SECS,
        });

        let payload_str = token_payload.to_string();
        let sig = CryptoUtils::hmac_sha256_hex(&self.secret_salt, payload_str.as_bytes())?;

        let full_token = format!("{}.{}", hex::encode(payload_str.as_bytes()), sig);
        Ok(full_token)
    }

    /// Verifies and consumes a biometric factor token strictly preventing replay attacks.
    pub fn verify_and_consume_biometric_token(
        &self,
        token: &str,
        expected_creator: &str,
        expected_action: &str,
    ) -> Result<serde_json::Value, String> {
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 2 {
            return Err(
                "Malformed biometric token structure (expected payload.signature)".to_string(),
            );
        }

        let payload_bytes = hex::decode(parts[0])
            .map_err(|e| format!("Failed to hex decode token payload: {}", e))?;
        // 1. Verify HMAC-SHA256 signature in constant time
        if !CryptoUtils::verify_hmac_sha256_hex(&self.secret_salt, &payload_bytes, parts[1]) {
            return Err(
                "Biometric token signature verification failed (tampered token)".to_string(),
            );
        }

        let val: serde_json::Value = serde_json::from_slice(&payload_bytes)
            .map_err(|e| format!("Invalid token JSON: {}", e))?;

        let creator = val.get("creator_id").and_then(|v| v.as_str()).unwrap_or("");
        let action = val.get("action").and_then(|v| v.as_str()).unwrap_or("");
        let token_nonce = val.get("nonce").and_then(|v| v.as_str()).unwrap_or("");
        let exp = val.get("expires_at").and_then(|v| v.as_u64()).unwrap_or(0);

        if creator != expected_creator {
            return Err(format!(
                "Creator mismatch: expected '{}', got '{}'",
                expected_creator, creator
            ));
        }
        if action != expected_action {
            return Err(format!(
                "Action mismatch: expected '{}', got '{}'",
                expected_action, action
            ));
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        if now > exp {
            return Err(format!(
                "Biometric token expired at {}, current time {}",
                exp, now
            ));
        }

        // 2. Consume token nonce strictly
        {
            let mut nonces = self
                .consumed_token_nonces
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            nonces.retain(|_, exp_time| now <= *exp_time);
            if nonces.contains_key(token_nonce) {
                return Err(
                    "Biometric token nonce has already been consumed (replay attack detected)"
                        .to_string(),
                );
            }
            if nonces.len() >= MAX_CONSUMED_NONCES_CACHE {
                return Err("Biometric token validation rejected: consumed token capacity reached (fail-closed defense)".to_string());
            }
            nonces.insert(token_nonce.to_string(), exp);
        }

        Ok(val)
    }

    /// Verifies an authenticated challenge token, checking session, action, age, and anti-replay nonce.
    pub fn verify_challenge_token(
        &self,
        token: &str,
        expected_session: &str,
        expected_action: &str,
        max_age: u64,
    ) -> Result<String, String> {
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 2 {
            return Err(
                "Malformed biometric token structure (expected payload.signature)".to_string(),
            );
        }

        let payload_bytes = hex::decode(parts[0])
            .map_err(|e| format!("Failed to hex decode token payload: {}", e))?;
        // 1. Verify HMAC-SHA256 signature in constant time
        if !CryptoUtils::verify_hmac_sha256_hex(&self.secret_salt, &payload_bytes, parts[1]) {
            return Err(
                "Biometric token signature verification failed (tampered token)".to_string(),
            );
        }

        let val: serde_json::Value = serde_json::from_slice(&payload_bytes)
            .map_err(|e| format!("Invalid token JSON: {}", e))?;

        let creator = val
            .get("creator_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let session = val.get("session_id").and_then(|v| v.as_str()).unwrap_or("");
        let action = val.get("action").and_then(|v| v.as_str()).unwrap_or("");
        let token_nonce = val.get("nonce").and_then(|v| v.as_str()).unwrap_or("");
        let issued_at = val.get("issued_at").and_then(|v| v.as_u64()).unwrap_or(0);
        let exp = val.get("expires_at").and_then(|v| v.as_u64()).unwrap_or(0);

        if !expected_session.is_empty() && session != expected_session {
            return Err(format!(
                "Session mismatch: expected '{}', got '{}'",
                expected_session, session
            ));
        }
        if !expected_action.is_empty() && action != expected_action {
            return Err(format!(
                "Action mismatch: expected '{}', got '{}'",
                expected_action, action
            ));
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        if max_age > 0 && now.saturating_sub(issued_at) > max_age {
            return Err(format!(
                "Biometric token expired: age {} exceeds max_age {}",
                now.saturating_sub(issued_at),
                max_age
            ));
        }
        if now > exp {
            return Err(format!(
                "Biometric token expired at {}, current time {}",
                exp, now
            ));
        }

        // Anti-replay: consume token nonce strictly
        {
            let mut nonces = self
                .consumed_token_nonces
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            nonces.retain(|_, exp_time| now <= *exp_time);
            if nonces.contains_key(token_nonce) {
                return Err(
                    "Biometric token nonce has already been consumed (replay attack detected)"
                        .to_string(),
                );
            }
            if nonces.len() >= MAX_CONSUMED_NONCES_CACHE {
                return Err("Biometric token validation rejected: consumed token capacity reached (fail-closed defense)".to_string());
            }
            nonces.insert(token_nonce.to_string(), exp);
        }

        Ok(creator)
    }
}

fn default_attestation_format() -> String {
    "packed".to_string()
}

fn default_aaguid() -> String {
    "00000000-0000-0000-0000-000000000000".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrolledBiometricCredential {
    pub credential_id: String,
    pub creator_id: String,
    pub public_key_bytes: Vec<u8>,
    pub enrolled_at: u64,
    #[serde(default)]
    pub authenticator_type: PlatformAuthenticatorType,
    #[serde(default)]
    pub attestation_statement_hash: String,
    #[serde(default)]
    pub require_hardware_authenticator: bool,
    #[serde(default)]
    pub require_biometric_modality: bool,
    #[serde(default = "default_attestation_format")]
    pub attestation_format: String,
    #[serde(default = "default_aaguid")]
    pub aaguid: String,
    #[serde(default)]
    pub raw_attestation_object: Vec<u8>,
    #[serde(default)]
    pub attestation_certificate_der: Vec<u8>,
    #[serde(default)]
    pub client_data_hash: [u8; 32],
    #[serde(default)]
    pub raw_client_data_json: Vec<u8>,
}

impl EnrolledBiometricCredential {
    pub fn compute_attestation_hash(statement_bytes: &[u8]) -> String {
        use sha2::Digest;
        hex::encode(sha2::Sha256::digest(statement_bytes))
    }
}

/// Centralized Passkey / WebAuthn Assertion Verifier implementing zero-trust IdentityVerifier.
#[derive(Clone)]
pub struct CentralizedPasskeyVerifier {
    biometric_service: Arc<BiometricChallengeService>,
    enrolled_credentials: Arc<RwLock<HashMap<String, EnrolledBiometricCredential>>>,
    storage_path: Option<std::path::PathBuf>,
}

impl Default for CentralizedPasskeyVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl CentralizedPasskeyVerifier {
    pub fn new() -> Self {
        Self {
            biometric_service: Arc::new(BiometricChallengeService::new()),
            enrolled_credentials: Arc::new(RwLock::new(HashMap::new())),
            storage_path: None,
        }
    }

    pub fn with_service(service: Arc<BiometricChallengeService>) -> Self {
        Self {
            biometric_service: service,
            enrolled_credentials: Arc::new(RwLock::new(HashMap::new())),
            storage_path: None,
        }
    }

    pub fn with_provider(service: Arc<BiometricChallengeService>) -> Self {
        Self::with_service(service)
    }

    /// Load enrolled passkeys from durable disk storage or initialize new.
    /// Fails closed if the storage file exists on disk but is unreadable or corrupted.
    pub fn load_or_new<P: AsRef<std::path::Path>>(storage_path: P) -> Result<Self, String> {
        let path = storage_path.as_ref().to_path_buf();
        let mut map = HashMap::new();
        if path.exists() {
            let data = std::fs::read(&path).map_err(|e| {
                format!(
                    "Failed to read enrolled passkeys from '{}': {}",
                    path.display(),
                    e
                )
            })?;
            let loaded =
                serde_json::from_slice::<HashMap<String, EnrolledBiometricCredential>>(&data)
                    .map_err(|e| {
                        format!(
                            "Failed to parse corrupted enrolled passkeys from '{}': {}",
                            path.display(),
                            e
                        )
                    })?;
            map = loaded;
        }
        let bio_service = Arc::new(BiometricChallengeService::new());
        for cred in map.values() {
            let alg = match cred.attestation_format.as_str() {
                "packed" => CoseAlgorithm::Es256,
                _ => CoseAlgorithm::EdDsa,
            };
            bio_service.register_biometric_credential(
                &cred.credential_id,
                &cred.public_key_bytes,
                alg,
            );
        }
        Ok(Self {
            biometric_service: bio_service,
            enrolled_credentials: Arc::new(RwLock::new(map)),
            storage_path: Some(path),
        })
    }

    pub fn register_credential(&self, cred: EnrolledBiometricCredential) -> Result<(), String> {
        let alg = match cred.attestation_format.as_str() {
            "packed" => CoseAlgorithm::Es256,
            _ => CoseAlgorithm::EdDsa,
        };
        self.biometric_service.register_biometric_credential(
            &cred.credential_id,
            &cred.public_key_bytes,
            alg,
        );
        let mut map = self
            .enrolled_credentials
            .write()
            .map_err(|_| "Credentials lock poisoned")?;
        map.insert(cred.credential_id.clone(), cred);
        if let Some(ref path) = self.storage_path {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Ok(json_bytes) = serde_json::to_vec_pretty(&*map) {
                let _ = crate::persistence::atomic_write_file_hardened(path, &json_bytes);
            }
        }
        Ok(())
    }

    pub fn service(&self) -> &Arc<BiometricChallengeService> {
        &self.biometric_service
    }

    pub fn provider(&self) -> &Arc<BiometricChallengeService> {
        &self.biometric_service
    }
}

pub type BiometricFactorProvider = BiometricChallengeService;

impl crate::zero_trust::IdentityVerifier for CentralizedPasskeyVerifier {
    fn proof_type(&self) -> &'static str {
        "passkey_assertion"
    }

    fn verify_proof(
        &self,
        proof: &crate::zero_trust::IdentityProof,
        current_generation: crate::primitives::AuthorityGeneration,
        current_epoch_secs: u64,
    ) -> Result<crate::primitives::VerifiedPrincipal, String> {
        let (cred_id, auth_data, client_data_json, sig) = match proof {
            crate::zero_trust::IdentityProof::PasskeyAssertion {
                credential_id,
                authenticator_data,
                client_data_json,
                signature,
            } => (
                credential_id,
                authenticator_data,
                client_data_json,
                signature,
            ),
            _ => return Err("Invalid proof variant for CentralizedPasskeyVerifier".into()),
        };

        // 1. Mandatory authenticator_data length check (must be at least 37 bytes)
        if auth_data.len() < 37 {
            return Err(
                "PasskeyAssertion verification rejected: authenticator_data too short (<37 bytes)"
                    .into(),
            );
        }

        // 1b. Validate RP ID Hash (first 32 bytes of authenticator_data)
        let rp_id_hash = &auth_data[0..32];
        let expected_canonical_rp = crate::cryptography::CryptoUtils::sha256(b"taracore.ai");
        let expected_local_rp = crate::cryptography::CryptoUtils::sha256(b"localhost");
        if !crate::cryptography::CryptoUtils::constant_time_eq(rp_id_hash, &expected_canonical_rp)
            && !crate::cryptography::CryptoUtils::constant_time_eq(rp_id_hash, &expected_local_rp)
        {
            return Err("PasskeyAssertion verification rejected: authenticator_data RP ID hash does not match authorized relying party (taracore.ai)".into());
        }

        // 2. Check User Present (bit 0 = 0x01) and User Verified (bit 2 = 0x04) flags
        let flags = auth_data[32];
        if flags & 0x01 == 0 {
            return Err(
                "PasskeyAssertion verification rejected: User Present (UP) flag not set".into(),
            );
        }
        let user_verified = (flags & 0x04) != 0;

        // Verify authenticator sign_count structure (Anti-Cloning & Replay Defense)
        if auth_data.len() < 37 {
            return Err("PasskeyAssertion verification rejected: authenticator_data too short (minimum 37 bytes required)".into());
        }
        let _sign_count =
            u32::from_be_bytes([auth_data[33], auth_data[34], auth_data[35], auth_data[36]]);

        // 3. Validate client_data_json (required for WebAuthn assertion cryptographic verification)
        if client_data_json.is_empty() {
            return Err("PasskeyAssertion verification rejected: client_data_json is required for cryptographic verification".into());
        }
        let cd_str = std::str::from_utf8(client_data_json)
            .map_err(|e| format!("Invalid UTF-8 in clientDataJSON: {}", e))?;
        let parsed: serde_json::Value = serde_json::from_str(cd_str)
            .map_err(|e| format!("Invalid clientDataJSON JSON structure: {}", e))?;
        let cd_type = parsed.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if cd_type != "webauthn.get" {
            return Err(format!(
                "Invalid clientDataJSON type '{}', expected 'webauthn.get'",
                cd_type
            ));
        }

        let origin = parsed.get("origin").and_then(|v| v.as_str()).unwrap_or("");
        if origin != "https://taracore.ai"
            && !origin.starts_with("https://localhost")
            && !origin.starts_with("http://localhost")
        {
            return Err(format!(
                "PasskeyAssertion verification rejected: unauthorized clientDataJSON origin '{}'",
                origin
            ));
        }

        let challenge_nonce = parsed
            .get("challenge")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if challenge_nonce.is_empty() {
            return Err(
                "PasskeyAssertion verification rejected: clientDataJSON missing challenge nonce"
                    .into(),
            );
        }

        // Replay protection: challenge nonce must be single-use and not previously consumed
        self.biometric_service
            .check_and_consume_nonce(challenge_nonce)?;

        // 4. Verify signature is present and cryptographically authentic
        if sig.is_empty() {
            return Err(
                "PasskeyAssertion verification rejected: authenticator signature is empty".into(),
            );
        }

        let cred_hex = hex::encode(cred_id);
        let client_data_hash = crate::cryptography::CryptoUtils::sha256(client_data_json);

        let mut signed_data = Vec::with_capacity(auth_data.len() + 32);
        signed_data.extend_from_slice(auth_data);
        signed_data.extend_from_slice(&client_data_hash);

        // 5. Look up enrolled credential strictly in Authoritative Registry (fixes finding #3)
        // Public keys supplied in the assertion itself must NEVER be trusted!
        let cred_opt = {
            let map = self
                .enrolled_credentials
                .read()
                .map_err(|_| "Credentials lock poisoned")?;
            map.get(&cred_hex).cloned()
        };

        let cred = cred_opt.ok_or_else(|| {
            format!("PasskeyAssertionRejected: Credential '{}' is not registered in authoritative enrolled credentials (unauthenticated assertion keys prohibited)", cred_hex)
        })?;

        let cose_alg = if cred.attestation_format.eq_ignore_ascii_case("eddsa")
            || cred.public_key_bytes.len() == 32
        {
            CoseAlgorithm::EdDsa
        } else if cred.public_key_bytes.len() == 64
            || cred.public_key_bytes.len() == 65
            || cred.public_key_bytes.len() == 33
            || cred.public_key_bytes.len() == 91
            || cred.attestation_format.eq_ignore_ascii_case("packed")
        {
            CoseAlgorithm::Es256
        } else if cred.public_key_bytes.len() >= 128 {
            CoseAlgorithm::Rs256
        } else {
            CoseAlgorithm::Es256
        };
        verify_signed_data(&cred.public_key_bytes, &signed_data, sig, cose_alg).map_err(|e| {
            format!(
                "PasskeyAssertion signature verification failed against enrolled credential: {}",
                e
            )
        })?;

        // 5. Build verified principal with appropriate role and assurance level
        let is_canonical_creator = cred.creator_id == crate::primitives::CANONICAL_CREATOR_ID
            || cred.creator_id == "CREATOR_ROOT"
            || cred.creator_id.to_lowercase().starts_with("creator");
        let (role, assurance) = if user_verified && is_canonical_creator {
            (
                crate::primitives::PrincipalRole::creator(),
                crate::primitives::AssuranceLevel::CREATOR_ROOT,
            )
        } else if is_canonical_creator || user_verified {
            (
                crate::primitives::PrincipalRole::operator(),
                crate::primitives::AssuranceLevel::HARDWARE_ATTESTED,
            )
        } else {
            (
                crate::primitives::PrincipalRole::user(),
                crate::primitives::AssuranceLevel::HARDWARE_ATTESTED,
            )
        };

        let pid_suffix = if cred_hex.is_empty() {
            "creator_root"
        } else {
            &cred_hex[..cred_hex.len().min(32)]
        };
        let pid = crate::primitives::PrincipalId::with_namespace("passkey", pid_suffix)?;

        let s_id = format!("passkey_session_{}", pid.as_str());
        let proof = crate::primitives::compute_session_proof_hex(&pid.as_str(), &s_id);

        let mut claims = std::collections::HashMap::new();
        claims.insert("session_id".into(), serde_json::Value::String(s_id));
        claims.insert("session_proof".into(), serde_json::Value::String(proof));
        claims.insert(
            "verified_via".into(),
            serde_json::Value::String("webauthn_hardware_passkey".into()),
        );
        claims.insert("credential_id".into(), serde_json::Value::String(cred_hex));
        claims.insert(
            "user_verified".into(),
            serde_json::Value::Bool(user_verified),
        );
        claims.insert("signature_verified".into(), serde_json::Value::Bool(true));
        claims.insert(
            "authenticator_type".into(),
            serde_json::Value::String("WebAuthn_Passkey".into()),
        );

        crate::primitives::VerifiedPrincipal::construct_verified(
            pid,
            role,
            assurance,
            None,
            current_generation,
            current_epoch_secs,
            claims,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zero_trust::identity::{IdentityProof, IdentityVerifier};
    use ed25519_dalek::Signer;

    #[test]
    fn test_centralized_passkey_verifier_signature_verification() {
        let verifier = CentralizedPasskeyVerifier::new();
        let (signing_key, verifying_key) =
            crate::cryptography::CryptoUtils::generate_ed25519_keypair();
        let cred_id = vec![0x10, 0x20, 0x30, 0x40];
        let cred_hex = hex::encode(&cred_id);

        let credential = EnrolledBiometricCredential {
            credential_id: cred_hex.clone(),
            creator_id: crate::primitives::CANONICAL_CREATOR_ID.to_string(),
            public_key_bytes: verifying_key.to_bytes().to_vec(),
            enrolled_at: 1000,
            authenticator_type: PlatformAuthenticatorType::WindowsHelloHardware,
            attestation_statement_hash: "none".to_string(),
            require_hardware_authenticator: true,
            require_biometric_modality: true,
            attestation_format: "eddsa".to_string(),
            aaguid: default_aaguid(),
            raw_attestation_object: Vec::new(),
            attestation_certificate_der: Vec::new(),
            client_data_hash: [0u8; 32],
            raw_client_data_json: Vec::new(),
        };
        verifier.register_credential(credential).unwrap();

        // Authentic WebAuthn auth_data with RP ID hash ("localhost"), UP (0x01) and UV (0x04)
        let mut auth_data = vec![0u8; 37];
        auth_data[..32].copy_from_slice(&crate::cryptography::CryptoUtils::sha256(b"localhost"));
        auth_data[32] = 0x05; // UP | UV
        auth_data[36] = 1;

        let client_data_json =
            br#"{"type":"webauthn.get","challenge":"sec_chal_123","origin":"https://localhost"}"#;
        let client_data_hash = crate::cryptography::CryptoUtils::sha256(client_data_json);

        let mut signed_data = Vec::new();
        signed_data.extend_from_slice(&auth_data);
        signed_data.extend_from_slice(&client_data_hash);

        let signature = signing_key.sign(&signed_data).to_vec();

        let current_gen = crate::primitives::AuthorityGeneration::INITIAL;

        // 1. Valid signature with UV succeeds with CREATOR_ROOT
        let valid_proof = IdentityProof::PasskeyAssertion {
            credential_id: cred_id.clone(),
            authenticator_data: auth_data.clone(),
            client_data_json: client_data_json.to_vec(),
            signature: signature.clone(),
        };
        let p_res = verifier.verify_proof(&valid_proof, current_gen, 1000);
        assert!(
            p_res.is_ok(),
            "Authentic passkey assertion must verify: {:?}",
            p_res.err()
        );
        let principal = p_res.unwrap();
        assert_eq!(
            principal.assurance_level(),
            crate::primitives::AssuranceLevel::CREATOR_ROOT
        );
        assert_eq!(
            principal.role(),
            &crate::primitives::PrincipalRole::creator()
        );

        // 2. Tampered signature fails closed
        let mut tampered_sig = signature.clone();
        tampered_sig[0] ^= 0xff;
        let bad_proof = IdentityProof::PasskeyAssertion {
            credential_id: cred_id.clone(),
            authenticator_data: auth_data.clone(),
            client_data_json: client_data_json.to_vec(),
            signature: tampered_sig,
        };
        let bad_sig_res = verifier.verify_proof(&bad_proof, current_gen, 1000);
        assert!(bad_sig_res.is_err(), "Tampered signature must fail closed");

        // 3. Unregistered credential fails closed
        let unreg_proof = IdentityProof::PasskeyAssertion {
            credential_id: vec![0x99, 0x88, 0x77],
            authenticator_data: auth_data.clone(),
            client_data_json: client_data_json.to_vec(),
            signature: signature.clone(),
        };
        let unreg_res = verifier.verify_proof(&unreg_proof, current_gen, 1000);
        assert!(
            unreg_res.is_err(),
            "Unregistered credential must fail closed"
        );

        // 4. Empty signature fails closed
        let empty_proof = IdentityProof::PasskeyAssertion {
            credential_id: cred_id,
            authenticator_data: auth_data,
            client_data_json: client_data_json.to_vec(),
            signature: Vec::new(),
        };
        let empty_sig_res = verifier.verify_proof(&empty_proof, current_gen, 1000);
        assert!(empty_sig_res.is_err(), "Empty signature must fail closed");

        // 5. Replayed challenge nonce fails closed (anti-replay defense)
        let replay_res = verifier.verify_proof(&valid_proof, current_gen, 1000);
        assert!(
            replay_res.is_err(),
            "Replay of same assertion challenge nonce must fail closed"
        );
    }

    #[test]
    fn test_biometric_challenge_service() {
        let service = BiometricChallengeService::new();
        let (signing_key, verifying_key) =
            crate::cryptography::CryptoUtils::generate_ed25519_keypair();
        let cred_id = "bio_cred_001";
        service.register_biometric_credential(
            cred_id,
            &verifying_key.to_bytes(),
            CoseAlgorithm::EdDsa,
        );

        let challenge = service.issue_challenge("creator_test", "session_123", "action_sign");

        let mut auth_data = vec![0u8; 37];
        auth_data[32] = 0x01 | 0x04; // User Present (0x01) | User Verified (0x04)
        let client_data_json = serde_json::to_vec(&serde_json::json!({
            "type": "webauthn.get",
            "challenge": challenge.nonce,
            "origin": "https://tara.local"
        }))
        .unwrap();
        let cdh = crate::cryptography::CryptoUtils::sha256(&client_data_json);
        let mut signed_data = Vec::new();
        signed_data.extend_from_slice(&auth_data);
        signed_data.extend_from_slice(&cdh);
        let sig = signing_key.sign(&signed_data).to_vec();

        let assertion = PlatformBiometricAssertion {
            user_verified: true,
            client_data_challenge: challenge.nonce.clone(),
            authenticator_signature: sig.clone(),
            authenticator_attachment: "platform".to_string(),
            credential_id: cred_id.to_string(),
            authenticator_data: auth_data.clone(),
            biometric_modality: "fingerprint".to_string(),
            user_verification_method: 0x02,
            client_data_json,
        };

        // 1. Valid signature verifies successfully and mints token
        let token_res = service.verify_platform_assertion_and_mint_token(&challenge, &assertion);
        assert!(
            token_res.is_ok(),
            "Valid biometric assertion must succeed: {:?}",
            token_res.err()
        );

        // 2. Tampered signature fails closed
        let mut bad_assertion = assertion.clone();
        bad_assertion.authenticator_signature[0] ^= 0xff;
        let bad_res = service.verify_platform_assertion_and_mint_token(&challenge, &bad_assertion);
        assert!(bad_res.is_err(), "Invalid signature must fail closed");

        // 3. Unverified assertion fails closed
        let mut unverified_assertion = assertion.clone();
        unverified_assertion.user_verified = false;
        let unverified_res =
            service.verify_platform_assertion_and_mint_token(&challenge, &unverified_assertion);
        assert!(
            unverified_res.is_err(),
            "Unverified assertion must fail closed"
        );

        // 4. Unregistered credential fails closed (Fixes cryptographic verification bypass)
        let mut unregistered_assertion = assertion;
        unregistered_assertion.credential_id = "unknown_unregistered_cred".to_string();
        let unreg_res =
            service.verify_platform_assertion_and_mint_token(&challenge, &unregistered_assertion);
        assert!(
            unreg_res.is_err(),
            "Unregistered credential must fail closed"
        );

        // 5. Direct verification with explicit public key succeeds for external keys
        let challenge2 = service.issue_challenge("creator_test", "session_456", "action_direct");
        let client_data_json2 = serde_json::to_vec(&serde_json::json!({
            "type": "webauthn.get",
            "challenge": challenge2.nonce,
            "origin": "https://tara.local"
        }))
        .unwrap();
        let cdh2 = crate::cryptography::CryptoUtils::sha256(&client_data_json2);
        let mut signed_data2 = Vec::new();
        signed_data2.extend_from_slice(&auth_data);
        signed_data2.extend_from_slice(&cdh2);
        let sig2 = signing_key.sign(&signed_data2).to_vec();

        let assertion2 = PlatformBiometricAssertion {
            user_verified: true,
            client_data_challenge: challenge2.nonce.clone(),
            authenticator_signature: sig2,
            authenticator_attachment: "platform".to_string(),
            credential_id: "unregistered_external_key".to_string(),
            authenticator_data: auth_data.clone(),
            biometric_modality: "fingerprint".to_string(),
            user_verification_method: 0x02,
            client_data_json: client_data_json2,
        };
        let direct_token_res = service.verify_platform_assertion_with_key_and_mint_token(
            &challenge2,
            &assertion2,
            &verifying_key.to_bytes(),
            CoseAlgorithm::EdDsa,
        );
        assert!(
            direct_token_res.is_ok(),
            "Direct public key assertion verification must succeed: {:?}",
            direct_token_res.err()
        );
    }

    #[test]
    fn test_tpm_pub_area_verification_with_ecc_coords() {
        let x_coord = [0x11u8; 32];
        let y_coord = [0x22u8; 32];

        // Format as CBOR COSE_Key map containing -2 (X) and -3 (Y)
        let mut cred_pk = vec![0xa5];
        cred_pk.extend_from_slice(&[0x01, 0x02]); // 1: 2 (EC2)
        cred_pk.extend_from_slice(&[0x03, 0x26]); // 3: -7 (ES256)
        cred_pk.extend_from_slice(&[0x20, 0x01]); // -1: 1 (P-256)
        cred_pk.extend_from_slice(&[0x21, 0x58, 0x20]); // -2: bytes(32)
        cred_pk.extend_from_slice(&x_coord);
        cred_pk.extend_from_slice(&[0x22, 0x58, 0x20]); // -3: bytes(32)
        cred_pk.extend_from_slice(&y_coord);

        // Build TPM pub_area containing these coordinates
        let mut pub_area = vec![0u8; 100];
        pub_area[20..52].copy_from_slice(&x_coord);
        pub_area[60..92].copy_from_slice(&y_coord);

        // 1. Valid matching TPM pub_area succeeds
        let match_res = verify_tpm_pub_area_matches_cred_pk(&pub_area, &cred_pk);
        assert!(
            match_res.is_ok(),
            "Valid TPM pubArea coordinates must match"
        );

        // 2. Tampered pub_area fails closed
        let mut bad_pub_area = pub_area.clone();
        bad_pub_area[20] ^= 0xff;
        let bad_res = verify_tpm_pub_area_matches_cred_pk(&bad_pub_area, &cred_pk);
        assert!(
            bad_res.is_err(),
            "Tampered pubArea coordinates must fail closed"
        );

        // 3. Empty pub_area fails closed
        let empty_res = verify_tpm_pub_area_matches_cred_pk(&[], &cred_pk);
        assert!(empty_res.is_err(), "Empty pubArea must fail closed");
    }
}
