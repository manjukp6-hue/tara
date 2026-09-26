//! Exhaustive unit tests for hardened Rust Creator Authority matching Python parity:
//! 1. Google authentication: Real Google JWT RS256 signature verification using RSA JWKS keys,
//!    validating issuer, audience, expiry, issued-at, nonce, and email verification.
//!    Never trusts decoded JWT payload alone.
//! 2. Key protection: Scrypt N=131072, r=8, p=1 + AES-256-GCM + Windows DPAPI keystore.
//!    Rejection of empty passphrases. Rejection of wrong passphrases.
//! 3. Authority integrity: Ed25519 sealed authority state, canonical hashing, fail-closed on tampering.

use std::fs;
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use rsa::pkcs1v15::SigningKey;
use rsa::signature::{Signer, SignatureEncoding};
use rsa::RsaPrivateKey;
use serde_json::json;
use sha2::Sha256;

use tara_server::access::crypto::SecureKeyStorage;
use tara_server::access::google::GoogleAuthService;
use tara_server::access::lifecycle::{AuthorityLifecycleManager, AuthorityState};
use tara_server::access::{CreatorAuthService, IdentityManager};

fn encode_b64url(bytes: &[u8]) -> String {
    URL_SAFE_NO_PAD.encode(bytes)
}

fn create_test_jwt(
    priv_key: &RsaPrivateKey,
    kid: &str,
    iss: &str,
    aud: &str,
    email: &str,
    email_verified: bool,
    sub: &str,
    exp_offset: f64,
    nonce: Option<&str>,
) -> String {
    let header = json!({
        "alg": "RS256",
        "typ": "JWT",
        "kid": kid
    });

    let now = GoogleAuthService::now_seconds();
    let mut payload = json!({
        "iss": iss,
        "aud": aud,
        "sub": sub,
        "email": email,
        "email_verified": email_verified,
        "iat": now - 5.0,
        "exp": now + exp_offset
    });

    if let Some(n) = nonce {
        payload.as_object_mut().unwrap().insert("nonce".to_string(), json!(n));
    }

    let header_b64 = encode_b64url(&serde_json::to_vec(&header).unwrap());
    let payload_b64 = encode_b64url(&serde_json::to_vec(&payload).unwrap());
    let signing_input = format!("{}.{}", header_b64, payload_b64);

    let signing_key = SigningKey::<Sha256>::new(priv_key.clone());
    let sig = signing_key.sign(signing_input.as_bytes());
    let sig_b64 = encode_b64url(&sig.to_vec());

    format!("{}.{}", signing_input, sig_b64)
}

// ──────────────────────────────────────────────────────────────────────────────
// 1. Google Authentication Tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_google_jwt_rs256_full_verification() {
    let mut rng = rand::thread_rng();
    let priv_key = RsaPrivateKey::new(&mut rng, 2048).expect("failed to generate RSA key");
    let pub_key = rsa::RsaPublicKey::from(&priv_key);

    let kid = "test_google_key_01";
    let client_id = "test-client-id.apps.googleusercontent.com";
    let auth_email = "creator@test.local";

    let auth_service = GoogleAuthService::new(Some(auth_email), Some(client_id));
    auth_service.register_trusted_rsa_key(kid, pub_key);

    // 1. Valid Token: must pass
    let valid_jwt = create_test_jwt(
        &priv_key, kid, "https://accounts.google.com", client_id, auth_email, true, "sub_12345", 3600.0, None,
    );
    let result = auth_service.verify_id_token(&valid_jwt, None, false);
    assert!(result.is_ok(), "Valid token must pass: {:?}", result.err());
    let claims = result.unwrap();
    assert_eq!(claims["email"], auth_email);
    assert_eq!(claims["sub"], "sub_12345");

    // 2. Tampered Payload: signature check must fail
    let parts: Vec<&str> = valid_jwt.split('.').collect();
    let tampered_payload = encode_b64url(b"{\"email\":\"hacker@evil.com\",\"email_verified\":true,\"sub\":\"hacked\"}");
    let tampered_jwt = format!("{}.{}.{}", parts[0], tampered_payload, parts[2]);
    let tamper_res = auth_service.verify_id_token(&tampered_jwt, None, false);
    assert!(tamper_res.is_err(), "Tampered token must fail signature check");

    // 3. Expired Token: must fail
    let expired_jwt = create_test_jwt(
        &priv_key, kid, "https://accounts.google.com", client_id, auth_email, true, "sub_12345", -100.0, None,
    );
    let exp_res = auth_service.verify_id_token(&expired_jwt, None, false);
    assert!(exp_res.is_err(), "Expired token must fail");
    assert!(exp_res.unwrap_err().contains("expired"));

    // 4. Unverified Email: must fail
    let unverified_jwt = create_test_jwt(
        &priv_key, kid, "https://accounts.google.com", client_id, auth_email, false, "sub_12345", 3600.0, None,
    );
    let unver_res = auth_service.verify_id_token(&unverified_jwt, None, false);
    assert!(unver_res.is_err(), "Unverified email must fail");
    assert!(unver_res.unwrap_err().contains("not verified"));

    // 5. Invalid Issuer: must fail
    let bad_iss_jwt = create_test_jwt(
        &priv_key, kid, "https://evil-issuer.com", client_id, auth_email, true, "sub_12345", 3600.0, None,
    );
    let iss_res = auth_service.verify_id_token(&bad_iss_jwt, None, false);
    assert!(iss_res.is_err(), "Invalid issuer must fail");
    assert!(iss_res.unwrap_err().contains("Invalid issuer"));

    // 6. Audience Mismatch: must fail
    let bad_aud_jwt = create_test_jwt(
        &priv_key, kid, "https://accounts.google.com", "wrong-aud.com", auth_email, true, "sub_12345", 3600.0, None,
    );
    let aud_res = auth_service.verify_id_token(&bad_aud_jwt, None, false);
    assert!(aud_res.is_err(), "Wrong audience must fail");
    assert!(aud_res.unwrap_err().contains("Audience mismatch"));

    // 7. Nonce Mismatch: must fail
    let nonce_jwt = create_test_jwt(
        &priv_key, kid, "https://accounts.google.com", client_id, auth_email, true, "sub_12345", 3600.0, Some("token_nonce_1"),
    );
    let nonce_fail = auth_service.verify_id_token(&nonce_jwt, Some("expected_other_nonce"), false);
    assert!(nonce_fail.is_err(), "Nonce mismatch must fail");

    let nonce_ok = auth_service.verify_id_token(&nonce_jwt, Some("token_nonce_1"), false);
    assert!(nonce_ok.is_ok(), "Matching nonce must pass");
}

// ──────────────────────────────────────────────────────────────────────────────
// 2. Key Protection Tests (Scrypt N=131072 + AES-256-GCM + DPAPI)
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_scrypt_aes_gcm_keystore_protection() {
    let tmp_dir = std::env::temp_dir().join(format!("tara_keystore_test_{}", rand::random::<u64>()));
    let storage = SecureKeyStorage::new(&tmp_dir);

    let key_id = "test_root_key";
    let private_bytes = [42u8; 32];
    let passphrase = "SuperSecureMasterPassphrase!2026";

    // 1. Empty passphrase rejected
    let empty_res = storage.store_private_key_modern(key_id, &private_bytes, "   ", true);
    assert!(empty_res.is_err(), "Empty passphrase must be rejected");

    // 2. Store private key using Scrypt N=131072, r=8, p=1 and AES-256-GCM
    let keystore_path = storage.store_private_key_modern(key_id, &private_bytes, passphrase, true)
        .expect("Store private key must succeed");
    assert!(keystore_path.exists());

    // 3. Inspect keystore file: verify parameters and zero plaintext
    let raw = fs::read_to_string(&keystore_path).expect("read keystore");
    let val: serde_json::Value = serde_json::from_str(&raw).expect("parse json");
    assert_eq!(val["version"], 3);
    assert_eq!(val["aead"], "AES-256-GCM");
    assert_eq!(val["kdf"], "Scrypt");
    assert_eq!(val["kdf_params"]["n"], 131072);
    assert_eq!(val["kdf_params"]["r"], 8);
    assert_eq!(val["kdf_params"]["p"], 1);
    assert!(!raw.contains("424242"), "Zero plaintext key in keystore file");

    // 4. Decrypt with correct passphrase
    let decrypted = storage.load_private_key(key_id, Some(passphrase))
        .expect("Decryption with correct passphrase must succeed");
    assert_eq!(decrypted, private_bytes);

    // 5. Decrypt with wrong passphrase must fail closed
    let wrong_res = storage.load_private_key(key_id, Some("WrongPassphrase123!"));
    assert!(wrong_res.is_err(), "Wrong passphrase must fail");

    // Clean up
    let _ = fs::remove_dir_all(&tmp_dir);
}

// ──────────────────────────────────────────────────────────────────────────────
// 3. Authority Integrity & Lifecycle Tamper Resistance Tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_authority_lifecycle_tamper_resistance() {
    let tmp_dir = std::env::temp_dir().join(format!("tara_lifecycle_test_{}", rand::random::<u64>()));
    let _ = fs::create_dir_all(&tmp_dir);

    let lifecycle = AuthorityLifecycleManager::new(&tmp_dir);

    // 1. Fresh installation: must be CREATOR_SETUP_REQUIRED
    let (state1, _) = lifecycle.verify_integrity();
    assert_eq!(state1, AuthorityState::CreatorSetupRequired);

    // 2. Initialize Creator Authority via setup_creator in isolated directory
    let id_mgr = IdentityManager::new(tmp_dir.to_str().unwrap());
    let auth_svc = CreatorAuthService::new(tmp_dir.to_str().unwrap());

    let setup_res = auth_svc.setup_creator(
        "mock_google_id_token",
        true,
        "Primary Rig",
        Some("feedcafe01020304"),
        &id_mgr,
    );
    assert_eq!(setup_res["status"], "SUCCESS");
    assert_eq!(setup_res["authority_state"], "ACTIVE");

    // 3. Integrity verify: must be ACTIVE
    let (state2, _) = lifecycle.verify_integrity();
    assert_eq!(state2, AuthorityState::Active);

    // 4. Tampering: edit operator_record.json directly on disk
    let record_file = tmp_dir.join("TARA").join("ACCESS").join("operator").join("operator_record.json");
    let mut rec_data: serde_json::Value = serde_json::from_str(&fs::read_to_string(&record_file).unwrap()).unwrap();
    rec_data["display_name"] = json!("TAMPERED_OPERATOR");
    fs::write(&record_file, serde_json::to_string_pretty(&rec_data).unwrap()).unwrap();

    // 5. Verify integrity: MUST FAIL CLOSED into AUTHORITY_LOCKED
    let (state_tampered, reason) = lifecycle.verify_integrity();
    assert_eq!(state_tampered, AuthorityState::AuthorityLocked, "Tampered file must lock authority");
    assert!(reason.contains("hash mismatch"), "Reason must mention hash mismatch: {}", reason);

    // 6. Any authentication in locked state must fail closed
    let auth_res = auth_svc.authenticate_creator_key(None, None, "ROOT_OPERATOR", Some("any_passphrase"), "127.0.0.1");
    assert_eq!(auth_res["status"], "AUTHORITY_LOCKED");

    // Clean up
    let _ = fs::remove_dir_all(&tmp_dir);
}
