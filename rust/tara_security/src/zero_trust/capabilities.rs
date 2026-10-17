//! Cryptographic Capability Tokens (Macaroon-Style Attenuated Delegations).
//!
//! High-Security Guarantee:
//! - Eliminates Ambient Authority: Callers hold explicit cryptographic capability tokens.
//! - Offline Attenuation: Callers can restrict (attenuate) their own capabilities by appending caveats
//!   (e.g., scoping down to a single file, adding an expiry, binding to a specific device) WITHOUT contacting the server.
//! - Anti-Escalation: Caveats are chained via HMAC: `sig_{i+1} = HMAC(sig_i, caveat_{i+1})`.
//!   It is mathematically impossible to remove a caveat or expand permissions without the root secret!

use serde::{Deserialize, Serialize};

use crate::cryptography::CryptoUtils;
use crate::primitives::{DeviceId, SecurityAction, SecurityContext, SecurityResource};

/// Attenuating caveat restricting the scope of a capability token.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilityCaveat {
    /// Valid only before this timestamp (epoch seconds).
    ExpiresBefore(u64),
    /// Valid only for this exact action verb.
    AllowedAction(String),
    /// Valid only for resources starting with this prefix URN.
    ResourcePrefix(String),
    /// Valid only when presented by this bound device.
    BoundDevice(String),
    /// Valid only from this exact IP address.
    BoundClientIp(String),
}

impl CapabilityCaveat {
    pub fn to_predicate_string(&self) -> String {
        match self {
            Self::ExpiresBefore(ts) => format!("exp < {}", ts),
            Self::AllowedAction(act) => format!("action == {}", act),
            Self::ResourcePrefix(res) => format!("res_prefix == {}", res),
            Self::BoundDevice(dev) => format!("device == {}", dev),
            Self::BoundClientIp(ip) => format!("ip == {}", ip),
        }
    }

    /// Evaluates if this caveat is satisfied in the given execution context.
    pub fn evaluate(
        &self,
        action: &SecurityAction,
        resource: &SecurityResource,
        context: &SecurityContext,
        bound_device: Option<&DeviceId>,
    ) -> Result<(), String> {
        match self {
            Self::ExpiresBefore(ts) => {
                if context.request_epoch_secs >= *ts {
                    return Err(format!(
                        "Capability expired: current {} >= expiry {}",
                        context.request_epoch_secs, ts
                    ));
                }
            }
            Self::AllowedAction(expected_action) => {
                if action.canonical_name() != *expected_action && action.verb() != expected_action {
                    return Err(format!(
                        "Action mismatch: capability permits '{}', attempted '{}'",
                        expected_action,
                        action.canonical_name()
                    ));
                }
            }
            Self::ResourcePrefix(expected_prefix) => {
                let res_urn = resource.resource_urn();
                if !matches_resource_prefix(expected_prefix, &res_urn) {
                    return Err(format!("Resource scope violation: capability restricted to prefix hierarchy '{}', attempted '{}'", expected_prefix, res_urn));
                }
            }
            Self::BoundDevice(expected_device) => match bound_device {
                Some(dev) if dev.as_str() == expected_device => {}
                _ => {
                    return Err(format!(
                        "Device binding mismatch: capability requires device '{}'",
                        expected_device
                    ))
                }
            },
            Self::BoundClientIp(expected_ip) => {
                if context.client_ip != *expected_ip {
                    return Err(format!(
                        "Client IP binding mismatch: capability requires IP '{}', caller is '{}'",
                        expected_ip, context.client_ip
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Matches resource URNs strictly along hierarchy segment boundaries (`:`, `/`, `.`).
fn matches_resource_prefix(expected_prefix: &str, resource_urn: &str) -> bool {
    if resource_urn == expected_prefix {
        return true;
    }
    if let Some(rest) = resource_urn.strip_prefix(expected_prefix) {
        rest.starts_with(':') || rest.starts_with('/') || rest.starts_with('.')
    } else {
        false
    }
}

pub const MAX_CAPABILITY_CAVEATS: usize = 32;

/// Attenuated Capability Token holding an identifier, chained caveats, and HMAC signature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityToken {
    pub token_id: String,
    pub caveats: Vec<CapabilityCaveat>,
    pub signature_hex: String,
}

impl CapabilityToken {
    /// Mint a new root capability token using the server's secret root key.
    pub fn mint_root(root_key: &[u8], token_id: impl Into<String>) -> Result<Self, String> {
        let id = token_id.into();
        let initial_sig = CryptoUtils::hmac_sha256(root_key, id.as_bytes())?;
        Ok(Self {
            token_id: id,
            caveats: Vec::new(),
            signature_hex: hex::encode(initial_sig),
        })
    }

    /// Mint a new root capability token bound to a mandatory expiration time.
    pub fn mint_root_with_ttl(
        root_key: &[u8],
        token_id: impl Into<String>,
        ttl_seconds: u64,
        now_epoch: u64,
    ) -> Result<Self, String> {
        let base = Self::mint_root(root_key, token_id)?;
        let expires_at = now_epoch.checked_add(ttl_seconds).ok_or_else(|| {
            "Arithmetic overflow computing capability token expiration".to_string()
        })?;
        base.attenuate(CapabilityCaveat::ExpiresBefore(expires_at))
    }

    /// Attenuate (scope down) this token by appending a new caveat.
    /// Can be performed offline by any caller holding the token.
    pub fn attenuate(&self, caveat: CapabilityCaveat) -> Result<Self, String> {
        if self.caveats.len() >= MAX_CAPABILITY_CAVEATS {
            return Err(format!(
                "MaxCaveatsExceeded: Token cannot exceed {} caveats",
                MAX_CAPABILITY_CAVEATS
            ));
        }

        let current_sig = hex::decode(&self.signature_hex)
            .map_err(|e| format!("Malformed signature in token: {e}"))?;

        // Derive next chained signature: sig_{i+1} = HMAC(sig_i, caveat_{i+1})
        let caveat_bytes = caveat.to_predicate_string();
        let next_sig = CryptoUtils::hmac_sha256(&current_sig, caveat_bytes.as_bytes())?;

        let mut new_caveats = self.caveats.clone();
        new_caveats.push(caveat);

        Ok(Self {
            token_id: self.token_id.clone(),
            caveats: new_caveats,
            signature_hex: hex::encode(next_sig),
        })
    }

    /// Verify the capability token against the server's root key and execution context.
    /// Strictly verifies cryptographic HMAC signature chain FIRST before evaluating caveat details.
    pub fn verify(
        &self,
        root_key: &[u8],
        action: &SecurityAction,
        resource: &SecurityResource,
        context: &SecurityContext,
        bound_device: Option<&DeviceId>,
    ) -> Result<(), String> {
        if self.caveats.len() > MAX_CAPABILITY_CAVEATS {
            return Err("TokenCaveatCountExceeded: Token contains too many caveats".into());
        }

        // 1. Recompute HMAC chain from root key
        let mut running_sig = CryptoUtils::hmac_sha256(root_key, self.token_id.as_bytes())?;

        for caveat in &self.caveats {
            let predicate = caveat.to_predicate_string();
            running_sig = CryptoUtils::hmac_sha256(&running_sig, predicate.as_bytes())?;
        }

        let expected_sig =
            hex::decode(&self.signature_hex).map_err(|e| format!("Invalid signature hex: {e}"))?;

        // Constant-time comparison ensures unforgeability
        if !CryptoUtils::constant_time_eq(&running_sig, &expected_sig) {
            return Err(
                "CapabilityTokenForged: HMAC signature chain validation failed (FailClosed)".into(),
            );
        }

        // 2. Evaluate authenticated caveats against runtime parameters
        let mut has_expiry = false;
        for caveat in &self.caveats {
            if matches!(caveat, CapabilityCaveat::ExpiresBefore(_)) {
                has_expiry = true;
            }
            caveat.evaluate(action, resource, context, bound_device)?;
        }

        if !has_expiry {
            return Err("CapabilityTokenRejected: Token lacks mandatory ExpiresBefore caveat (FailClosed against perpetual tokens)".into());
        }

        Ok(())
    }
}
