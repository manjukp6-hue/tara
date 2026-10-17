//! TPM 2.0 Platform Configuration Register (PCR) Quote Attestation Engine.
//!
//! Standards Compliance:
//! - TCG TPM 2.0 Library Specification Version 185 (March 2026).
//! - Verifies TPMS_ATTEST containing TPMS_QUOTE_INFO (TPM_ST_ATTEST_QUOTE = 0x8018).
//! - Confirms PCR selection digest (PCR 0 firmware, PCR 7 Secure Boot, PCR 11 kernel integrity).
//! - Verifies qualifying data freshness (nonce binding) against replay attacks.

use byteorder::{BigEndian, ByteOrder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{CoseAlgorithm, TPM_GENERATED_VALUE};
use crate::cryptography::{
    CryptoUtils, Ed25519Verifier, Es256Verifier, Rs256Verifier, SignatureVerifier,
};

pub const TPM_ST_ATTEST_QUOTE: u16 = 0x8018;

/// A TPM 2.0 Quote statement submitted by a client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TpmPcrQuoteStatement {
    pub attest_data: Vec<u8>,             // Serialized TPMS_ATTEST
    pub quote_signature: Vec<u8>,         // AIK signature over attest_data
    pub aik_public_key: Vec<u8>,          // Attestation Identity Key
    pub pcr_values: HashMap<u32, String>, // Claimed PCR values: index -> hex digest
    pub algorithm: CoseAlgorithm,
}

/// Verification result for PCR measurements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcrQuoteVerificationResult {
    pub is_verified: bool,
    pub pcr_digest: String,
    pub nonce_verified: bool,
    pub verified_pcrs: Vec<u32>,
}

/// Verifies TPM 2.0 PCR Quote against server challenge nonce and trusted baseline digests.
pub struct TpmPcrQuoteVerifier;

impl TpmPcrQuoteVerifier {
    pub fn verify_quote(
        statement: &TpmPcrQuoteStatement,
        trusted_aik_public_key: &[u8],
        expected_nonce: &[u8],
        baseline_pcrs: &HashMap<u32, String>,
    ) -> Result<PcrQuoteVerificationResult, String> {
        let attest = &statement.attest_data;
        if attest.len() < 10 {
            return Err("TPMS_ATTEST structure too short".into());
        }

        // 1. Check Magic (0xFF544347)
        let magic = BigEndian::read_u32(&attest[0..4]);
        if magic != TPM_GENERATED_VALUE {
            return Err(format!(
                "TPM quote magic mismatch: expected 0x{:08X}, got 0x{:08X}",
                TPM_GENERATED_VALUE, magic
            ));
        }

        // 2. Check Type (TPM_ST_ATTEST_QUOTE = 0x8018)
        let struct_type = BigEndian::read_u16(&attest[4..6]);
        if struct_type != TPM_ST_ATTEST_QUOTE {
            return Err(format!("TPM quote structure type mismatch: expected TPM_ST_ATTEST_QUOTE (0x{:04X}), got 0x{:04X}", TPM_ST_ATTEST_QUOTE, struct_type));
        }

        // 3. Verify AIK binding: statement AIK must match trusted enrolled AIK (fixes finding #4)
        if !CryptoUtils::constant_time_eq(&statement.aik_public_key, trusted_aik_public_key) {
            return Err("TpmQuoteAikMismatch: Statement AIK public key does not match trusted attestation key (Quote Rejected)".into());
        }

        // 3b. Verify AIK Signature over TPMS_ATTEST using trusted AIK
        match statement.algorithm {
            CoseAlgorithm::EdDsa => {
                let verifier = Ed25519Verifier;
                verifier.verify(trusted_aik_public_key, attest, &statement.quote_signature)?;
            }
            CoseAlgorithm::Rs256 => {
                let verifier = Rs256Verifier;
                verifier.verify(trusted_aik_public_key, attest, &statement.quote_signature)?;
            }
            CoseAlgorithm::Es256 => {
                let verifier = Es256Verifier;
                verifier.verify(trusted_aik_public_key, attest, &statement.quote_signature)?;
            }
            other => {
                return Err(format!("Unsupported algorithm for TPM quote: {:?}", other));
            }
        }

        // 4. Parse extraData (Qualifying Data / Challenge Nonce)
        let mut offset = 6;
        if attest.len() < offset + 2 {
            return Err("attest truncated before qualifiedSigner len".into());
        }
        let qs_len = BigEndian::read_u16(&attest[offset..offset + 2]) as usize;
        offset += 2 + qs_len;

        if attest.len() < offset + 2 {
            return Err("attest truncated before extraData len".into());
        }
        let extra_len = BigEndian::read_u16(&attest[offset..offset + 2]) as usize;
        offset += 2;

        if attest.len() < offset + extra_len {
            return Err("attest truncated inside extraData payload".into());
        }
        let extra_data = &attest[offset..offset + extra_len];
        offset += extra_len;

        // Verify nonce binding (freshness against replay)
        if !CryptoUtils::constant_time_eq(extra_data, expected_nonce) {
            return Err(
                "TpmQuoteNonceMismatch: Qualifying data in quote does not match server challenge"
                    .into(),
            );
        }

        // Skip clockInfo (17 bytes) + firmwareVersion (8 bytes)
        offset += 17 + 8;

        // 5. Parse TPMS_QUOTE_INFO: pcrSelect + pcrDigest
        if attest.len() < offset + 4 {
            return Err("attest truncated before quoteInfo".into());
        }
        let pcr_select_count = BigEndian::read_u32(&attest[offset..offset + 4]) as usize;
        offset += 4;
        let mut selected_pcr_indices = Vec::new();

        // Each TPMS_PCR_SELECTION is: hashAlg (2 bytes) + sizeofSelect (1 byte) + pcrSelect bytes
        for _ in 0..pcr_select_count {
            if attest.len() < offset + 3 {
                return Err("attest truncated in pcrSelect".into());
            }
            let _hash_alg = BigEndian::read_u16(&attest[offset..offset + 2]);
            let size_of_select = attest[offset + 2] as usize;
            offset += 3;

            if attest.len() < offset + size_of_select {
                return Err("attest truncated inside pcrSelect bytes".into());
            }
            let select_bytes = &attest[offset..offset + size_of_select];
            offset += size_of_select;

            for (byte_idx, &byte_val) in select_bytes.iter().enumerate() {
                for bit_idx in 0..8 {
                    if (byte_val & (1 << bit_idx)) != 0 {
                        selected_pcr_indices.push((byte_idx * 8 + bit_idx) as u32);
                    }
                }
            }
        }

        if attest.len() < offset + 2 {
            return Err("attest truncated before pcrDigest len".into());
        }
        let pcr_digest_len = BigEndian::read_u16(&attest[offset..offset + 2]) as usize;
        offset += 2;

        if pcr_digest_len == 0 {
            return Err("TpmQuotePcrDigestMissing: Signed quote pcrDigest is empty; valid cryptographic PCR digest required (FailClosed)".into());
        }

        if attest.len() < offset + pcr_digest_len {
            return Err("attest truncated inside pcrDigest".into());
        }
        let quote_pcr_digest = &attest[offset..offset + pcr_digest_len];
        let quote_pcr_digest_hex = hex::encode(quote_pcr_digest);

        // 6. Verify claimed PCR values against trusted baseline and quote pcrSelect
        if baseline_pcrs.is_empty() {
            return Err(
                "TpmQuoteBaselineMissing: Baseline PCR map cannot be empty (FailClosed)".into(),
            );
        }
        if selected_pcr_indices.len() != baseline_pcrs.len() {
            return Err(format!(
                "PcrSelectionMismatch: Quote selected {} PCRs but trusted baseline defines {} PCRs",
                selected_pcr_indices.len(),
                baseline_pcrs.len()
            ));
        }
        for sel_idx in &selected_pcr_indices {
            if !baseline_pcrs.contains_key(sel_idx) {
                return Err(format!(
                    "PcrSelectionUnexpected: Quote selected unbaseline PCR {}",
                    sel_idx
                ));
            }
        }

        let mut verified_indices = Vec::new();
        for (idx, baseline_hash) in baseline_pcrs {
            if !selected_pcr_indices.contains(idx) {
                return Err(format!("PcrSelectionMismatch: Mandatory baseline PCR {} was not selected in quote pcrSelect", idx));
            }
            if let Some(claimed_hash) = statement.pcr_values.get(idx) {
                if !CryptoUtils::constant_time_eq(claimed_hash.as_bytes(), baseline_hash.as_bytes())
                {
                    return Err(format!("PcrMeasurementMismatch: PCR {} claimed [{}] does not match trusted baseline [{}] (Host Integrity Breached)", idx, claimed_hash, baseline_hash));
                }
                verified_indices.push(*idx);
            } else {
                return Err(format!(
                    "PcrMeasurementMissing: Mandatory baseline PCR {} was not included in quote",
                    idx
                ));
            }
        }

        // 7. Cryptographically verify quote_pcr_digest matches composite hash in strict ascending order
        verified_indices.sort_unstable();
        let mut composite_preimage = Vec::new();
        for idx in &verified_indices {
            let hex_str = statement
                .pcr_values
                .get(idx)
                .ok_or_else(|| format!("PCR {idx} missing in statement"))?;
            let raw_bytes = hex::decode(hex_str)
                .map_err(|e| format!("Invalid hex encoding for PCR {idx} digest: {e}"))?;
            composite_preimage.extend_from_slice(&raw_bytes);
        }
        let expected_composite_hash = CryptoUtils::sha256(&composite_preimage);
        if !CryptoUtils::constant_time_eq(quote_pcr_digest, &expected_composite_hash) {
            return Err(format!(
                "TpmQuotePcrDigestMismatch: Signed quote pcrDigest [{}] does not match composite hash of verified PCRs [{}] (Quote Forgery Detected)",
                quote_pcr_digest_hex,
                hex::encode(expected_composite_hash)
            ));
        }

        Ok(PcrQuoteVerificationResult {
            is_verified: true,
            pcr_digest: quote_pcr_digest_hex,
            nonce_verified: true,
            verified_pcrs: verified_indices,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{BigEndian, ByteOrder};
    use ed25519_dalek::Signer;

    #[test]
    fn test_tpm_pcr_quote_verification() {
        let (signing_key, verifying_key) = CryptoUtils::generate_ed25519_keypair();
        let nonce = b"tara_secure_quote_challenge_nonce_1234";

        let pcr0_val = hex::encode([0x01u8; 32]);
        let pcr7_val = hex::encode([0x07u8; 32]);

        let mut composite_preimage = Vec::new();
        composite_preimage.extend_from_slice(&[0x01u8; 32]);
        composite_preimage.extend_from_slice(&[0x07u8; 32]);
        let composite_hash = CryptoUtils::sha256(&composite_preimage);

        // Build valid TPMS_ATTEST buffer
        let mut attest = Vec::new();
        // Magic
        let mut u32_buf = [0u8; 4];
        BigEndian::write_u32(&mut u32_buf, TPM_GENERATED_VALUE);
        attest.extend_from_slice(&u32_buf);
        // Type
        let mut u16_buf = [0u8; 2];
        BigEndian::write_u16(&mut u16_buf, TPM_ST_ATTEST_QUOTE);
        attest.extend_from_slice(&u16_buf);
        // qualifiedSigner len = 0
        BigEndian::write_u16(&mut u16_buf, 0);
        attest.extend_from_slice(&u16_buf);
        // extraData len & payload
        BigEndian::write_u16(&mut u16_buf, nonce.len() as u16);
        attest.extend_from_slice(&u16_buf);
        attest.extend_from_slice(nonce);
        // clockInfo (17 bytes)
        attest.extend_from_slice(&[0u8; 17]);
        // firmwareVersion (8 bytes)
        attest.extend_from_slice(&[0u8; 8]);
        // pcrSelect count = 1 (TPMS_PCR_SELECTION for PCR 0 and PCR 7)
        BigEndian::write_u32(&mut u32_buf, 1);
        attest.extend_from_slice(&u32_buf);
        // hashAlg = 0x000B (SHA-256), sizeofSelect = 3, select bitmap: bit 0 (PCR 0) | bit 7 (PCR 7) = 0x81
        BigEndian::write_u16(&mut u16_buf, 0x000B);
        attest.extend_from_slice(&u16_buf);
        attest.push(3); // 3 bytes for 24 PCRs
        attest.extend_from_slice(&[0x81, 0x00, 0x00]);

        // pcrDigest len & payload
        BigEndian::write_u16(&mut u16_buf, composite_hash.len() as u16);
        attest.extend_from_slice(&u16_buf);
        attest.extend_from_slice(&composite_hash);

        let quote_sig = signing_key.sign(&attest).to_vec();
        let trusted_aik = verifying_key.to_bytes();

        let mut pcr_values = HashMap::new();
        pcr_values.insert(0, pcr0_val.clone());
        pcr_values.insert(7, pcr7_val.clone());

        let mut baseline_pcrs = HashMap::new();
        baseline_pcrs.insert(0, pcr0_val);
        baseline_pcrs.insert(7, pcr7_val);

        let statement = TpmPcrQuoteStatement {
            attest_data: attest.clone(),
            quote_signature: quote_sig.clone(),
            aik_public_key: trusted_aik.to_vec(),
            pcr_values: pcr_values.clone(),
            algorithm: CoseAlgorithm::EdDsa,
        };

        // 1. Valid quote succeeds with trusted AIK
        let res =
            TpmPcrQuoteVerifier::verify_quote(&statement, &trusted_aik, nonce, &baseline_pcrs);
        assert!(
            res.is_ok(),
            "Authentic TPM PCR quote must verify: {:?}",
            res.err()
        );
        let outcome = res.unwrap();
        assert!(outcome.is_verified);
        assert_eq!(outcome.verified_pcrs, vec![0, 7]);

        // 2. Untrusted / mismatched AIK fails closed (fixes finding #4)
        let bad_aik_res =
            TpmPcrQuoteVerifier::verify_quote(&statement, &[0xeeu8; 32], nonce, &baseline_pcrs);
        assert!(bad_aik_res.is_err(), "Untrusted AIK must fail closed");

        // 3. Mismatched nonce fails closed
        let bad_nonce_res = TpmPcrQuoteVerifier::verify_quote(
            &statement,
            &trusted_aik,
            b"wrong_nonce",
            &baseline_pcrs,
        );
        assert!(bad_nonce_res.is_err(), "Mismatched nonce must fail closed");

        // 4. Forged PCR baseline fails closed
        let mut tampered_baseline = baseline_pcrs.clone();
        tampered_baseline.insert(0, hex::encode([0xffu8; 32]));
        let bad_pcr_res =
            TpmPcrQuoteVerifier::verify_quote(&statement, &trusted_aik, nonce, &tampered_baseline);
        assert!(
            bad_pcr_res.is_err(),
            "Mismatched PCR measurement must fail closed"
        );

        // 5. Forged composite hash fails closed
        let mut forged_attest = attest;
        let last_idx = forged_attest.len() - 1;
        forged_attest[last_idx] ^= 0x55;
        let forged_sig = signing_key.sign(&forged_attest).to_vec();
        let forged_stmt = TpmPcrQuoteStatement {
            attest_data: forged_attest,
            quote_signature: forged_sig,
            aik_public_key: trusted_aik.to_vec(),
            pcr_values,
            algorithm: CoseAlgorithm::EdDsa,
        };
        let forged_res =
            TpmPcrQuoteVerifier::verify_quote(&forged_stmt, &trusted_aik, nonce, &baseline_pcrs);
        assert!(
            forged_res.is_err(),
            "Forged composite hash must fail closed"
        );
    }
}
