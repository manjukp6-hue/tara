//! Google OpenID Connect ID token verification layer in Rust.
//! Cryptographically verifies Google ID tokens (JWT with RS256 signatures).
//! Validates:
//! - RS256 signature verification against Google's public JWKS certificates
//! - Issuer ('accounts.google.com' or 'https://accounts.google.com')
//! - Audience (aud matches expected_client_id if configured)
//! - Expiration (exp > current_time)
//! - Issued-at (iat <= current_time)
//! - Nonce matching if present or required
//! - Email verification (email_verified == true)
//! - Non-empty subject ID (sub)
//!
//! Never trusts decoded JWT payload alone.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use rsa::{BigUint, RsaPublicKey, pkcs1v15::VerifyingKey};
use rsa::signature::Verifier;
use serde_json::Value;
use sha2::Sha256;

pub const GOOGLE_ISSUERS: &[&str] = &["accounts.google.com", "https://accounts.google.com"];
pub const GOOGLE_JWKS_URL: &str = "https://www.googleapis.com/oauth2/v3/certs";

fn decode_b64url(s: &str) -> Result<Vec<u8>, String> {
    let clean = s.trim().trim_end_matches('=');
    URL_SAFE_NO_PAD.decode(clean.as_bytes())
        .map_err(|e| format!("Base64URL decode failed: {}", e))
}

pub struct GoogleAuthService {
    pub authorized_email: Option<String>,
    pub expected_client_id: Option<String>,
    pub key_cache: Mutex<HashMap<String, RsaPublicKey>>,
    pub last_key_fetch: Mutex<f64>,
    pub pending_nonces: Mutex<HashMap<String, (f64, bool)>>,
}

impl GoogleAuthService {
    pub fn new(authorized_email: Option<&str>, expected_client_id: Option<&str>) -> Self {
        Self {
            authorized_email: authorized_email.map(|e| e.trim().to_lowercase()),
            expected_client_id: expected_client_id.map(|c| c.trim().to_string()),
            key_cache: Mutex::new(HashMap::new()),
            last_key_fetch: Mutex::new(0.0),
            pending_nonces: Mutex::new(HashMap::new()),
        }
    }

    pub fn now_seconds() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }

    pub fn register_trusted_rsa_key(&self, kid: &str, key: RsaPublicKey) {
        self.key_cache.lock().unwrap().insert(kid.to_string(), key);
    }

    pub fn register_trusted_jwk(&self, kid: &str, n_b64: &str, e_b64: &str) -> Result<(), String> {
        let n_bytes = decode_b64url(n_b64)?;
        let e_bytes = decode_b64url(e_b64)?;
        let n = BigUint::from_bytes_be(&n_bytes);
        let e = BigUint::from_bytes_be(&e_bytes);
        let key = RsaPublicKey::new(n, e)
            .map_err(|err| format!("Invalid RSA key components: {:?}", err))?;
        self.register_trusted_rsa_key(kid, key);
        Ok(())
    }

    pub fn create_auth_nonce(&self, ttl_seconds: f64) -> String {
        let mut bytes = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut bytes);
        let nonce = URL_SAFE_NO_PAD.encode(&bytes);
        let now = Self::now_seconds();

        let mut nonces = self.pending_nonces.lock().unwrap();
        // Purge expired nonces
        nonces.retain(|_, (exp, _)| now <= *exp);
        nonces.insert(nonce.clone(), (now + ttl_seconds, false));
        nonce
    }

    pub fn fetch_google_jwks(&self) -> bool {
        // Attempt fetch via Windows curl or curl
        let output = std::process::Command::new("curl.exe")
            .args(["-s", "-m", "5", GOOGLE_JWKS_URL])
            .output();

        let json_str = match output {
            Ok(out) if out.status.success() => {
                String::from_utf8_lossy(&out.stdout).to_string()
            }
            _ => return false,
        };

        if let Ok(val) = serde_json::from_str::<Value>(&json_str) {
            if let Some(keys_arr) = val.get("keys").and_then(|k| k.as_array()) {
                let mut cache = self.key_cache.lock().unwrap();
                for k in keys_arr {
                    if k.get("kty").and_then(|v| v.as_str()) == Some("RSA")
                        && k.get("alg").and_then(|v| v.as_str()) == Some("RS256")
                    {
                        if let (Some(kid), Some(n_b64), Some(e_b64)) = (
                            k.get("kid").and_then(|v| v.as_str()),
                            k.get("n").and_then(|v| v.as_str()),
                            k.get("e").and_then(|v| v.as_str()),
                        ) {
                            if let (Ok(n_bytes), Ok(e_bytes)) = (decode_b64url(n_b64), decode_b64url(e_b64)) {
                                let n = BigUint::from_bytes_be(&n_bytes);
                                let e = BigUint::from_bytes_be(&e_bytes);
                                if let Ok(rsa_key) = RsaPublicKey::new(n, e) {
                                    cache.insert(kid.to_string(), rsa_key);
                                }
                            }
                        }
                    }
                }
                *self.last_key_fetch.lock().unwrap() = Self::now_seconds();
                return true;
            }
        }
        false
    }

    /// Cryptographically validates a Google OpenID Connect ID token (RS256 JWT).
    /// Returns verified claims dict on success.
    pub fn verify_id_token(
        &self,
        id_token: &str,
        expected_nonce: Option<&str>,
        require_nonce: bool,
    ) -> Result<Value, String> {
        let parts: Vec<&str> = id_token.split('.').collect();
        if parts.len() != 3 {
            return Err("Malformed JWT: expected exactly 3 dot-separated segments".to_string());
        }

        let header_bytes = decode_b64url(parts[0])?;
        let payload_bytes = decode_b64url(parts[1])?;
        let signature_bytes = decode_b64url(parts[2])?;

        let header: Value = serde_json::from_slice(&header_bytes)
            .map_err(|e| format!("Invalid JWT header JSON: {}", e))?;
        let payload: Value = serde_json::from_slice(&payload_bytes)
            .map_err(|e| format!("Invalid JWT payload JSON: {}", e))?;

        // 1. Verify header alg == "RS256"
        let alg = header.get("alg").and_then(|v| v.as_str()).unwrap_or("");
        if alg != "RS256" {
            return Err(format!("Unsupported algorithm '{}', expected RS256", alg));
        }

        let kid = header.get("kid").and_then(|v| v.as_str())
            .ok_or_else(|| "Missing 'kid' in JWT header".to_string())?;

        // 2. Obtain RSA Public Key for kid
        let rsa_key = {
            let cache = self.key_cache.lock().unwrap();
            cache.get(kid).cloned()
        };

        let rsa_key = match rsa_key {
            Some(k) => k,
            None => {
                // Key not in cache: attempt refresh from Google JWKS
                self.fetch_google_jwks();
                let cache = self.key_cache.lock().unwrap();
                cache.get(kid).cloned()
                    .ok_or_else(|| format!("Unknown kid '{}' in JWKS and local cache", kid))?
            }
        };

        // 3. Cryptographically verify RS256 signature
        let signing_input = format!("{}.{}", parts[0], parts[1]);
        let verifying_key = VerifyingKey::<Sha256>::new(rsa_key);
        let rsa_sig = rsa::pkcs1v15::Signature::try_from(signature_bytes.as_slice())
            .map_err(|e| format!("Invalid RSA signature structure: {:?}", e))?;

        verifying_key.verify(signing_input.as_bytes(), &rsa_sig)
            .map_err(|_| "Cryptographic RS256 signature verification failed against Google JWKS".to_string())?;

        // 4. Verify Issuer
        let iss = payload.get("iss").and_then(|v| v.as_str()).unwrap_or("");
        if !GOOGLE_ISSUERS.contains(&iss) {
            return Err(format!("Invalid issuer '{}', expected Google issuer", iss));
        }

        // 5. Verify Audience if configured
        if let Some(ref exp_aud) = self.expected_client_id {
            let aud = payload.get("aud").and_then(|v| v.as_str()).unwrap_or("");
            if aud != exp_aud {
                return Err(format!("Audience mismatch: expected '{}', got '{}'", exp_aud, aud));
            }
        }

        let now = Self::now_seconds();
        let clock_skew = 10.0;

        // 6. Verify Expiry
        let exp = payload.get("exp").and_then(|v| v.as_f64())
            .ok_or_else(|| "Missing 'exp' timestamp".to_string())?;
        if now > (exp + clock_skew) {
            return Err(format!("Token expired: exp={}, now={}", exp, now));
        }

        // 7. Verify Issued-At
        let iat = payload.get("iat").and_then(|v| v.as_f64())
            .ok_or_else(|| "Missing 'iat' timestamp".to_string())?;
        if iat > (now + clock_skew) {
            return Err(format!("Token issued in the future: iat={}, now={}", iat, now));
        }

        // 8. Verify email_verified
        let email_verified = match payload.get("email_verified") {
            Some(Value::Bool(b)) => *b,
            Some(Value::String(s)) => s == "true",
            _ => false,
        };
        if !email_verified {
            return Err("Google account email is not verified (email_verified != true)".to_string());
        }

        // 9. Verify non-empty subject
        let sub = payload.get("sub").and_then(|v| v.as_str()).unwrap_or("");
        if sub.trim().is_empty() {
            return Err("Missing or empty 'sub' subject claim in ID token".to_string());
        }

        // 10. Verify Nonce if provided or required
        let token_nonce = payload.get("nonce").and_then(|v| v.as_str());
        if require_nonce && token_nonce.is_none() {
            return Err("Missing required nonce in ID token".to_string());
        }
        if let Some(exp_n) = expected_nonce {
            match token_nonce {
                Some(tn) if tn == exp_n => {
                    // Mark nonce consumed
                    let mut nonces = self.pending_nonces.lock().unwrap();
                    if let Some((_, consumed)) = nonces.get_mut(exp_n) {
                        *consumed = true;
                    }
                }
                Some(tn) => return Err(format!("Nonce mismatch: expected '{}', got '{}'", exp_n, tn)),
                None => return Err("Expected nonce was not provided in token".to_string()),
            }
        }

        // 11. Verify Email matches authorized email if set
        if let Some(ref auth_email) = self.authorized_email {
            let email = payload.get("email").and_then(|v| v.as_str()).unwrap_or("").trim().to_lowercase();
            if &email != auth_email {
                return Err(format!("Email mismatch: token is for '{}', authorized creator is '{}'", email, auth_email));
            }
        }

        Ok(payload)
    }
}
