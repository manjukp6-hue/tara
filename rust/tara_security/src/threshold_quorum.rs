//! M-of-N Threshold Multi-Signature Quorum Governance Subsystem (Pillar 5).
//!
//! High-Security Guarantee:
//! - Eliminates Single-Point-of-Failure: Prevents a single compromised operator or rogue credential
//!   from unilaterally executing critical operations (e.g. lifting lockdown, rotating authority, re-encrypting vaults).
//! - Enforces M-of-N Cryptographic Signatures from authorized signers.
//! - Validates proposal freshness, action binding, and signer uniqueness.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use crate::cryptography::{CryptoUtils, Ed25519Verifier, SignatureVerifier};
use crate::primitives::{PrincipalId, SecurityAction, SecurityResource};

/// Policy defining quorum thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuorumPolicy {
    pub policy_id: String,
    pub threshold_m: usize,
    pub authorized_signers_n: HashSet<String>, // PrincipalId strings
    #[serde(default)]
    pub authorized_signer_keys: HashMap<String, Vec<u8>>, // Optional pinned PrincipalId -> Ed25519 pubkey
    pub proposal_ttl_secs: u64,
}

/// A proposed critical operation requiring multi-party approval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuorumProposal {
    pub proposal_id: String,
    pub action: SecurityAction,
    pub resource: SecurityResource,
    pub rationale: String,
    pub proposed_by: PrincipalId,
    pub created_epoch_secs: u64,
    pub expires_epoch_secs: u64,
    pub proposal_nonce: String,
}

impl QuorumProposal {
    /// Compute canonical digest of the proposal for cryptographic signing.
    pub fn compute_digest(&self) -> [u8; 32] {
        let preimage = format!(
            "QUORUM_PROPOSAL|{}|{}|{}|{}|{}|{}|{}|{}",
            self.proposal_id,
            self.action.canonical_name(),
            self.resource.resource_urn(),
            self.rationale,
            self.proposed_by.as_str(),
            self.created_epoch_secs,
            self.expires_epoch_secs,
            self.proposal_nonce
        );
        CryptoUtils::sha256(preimage.as_bytes())
    }
}

/// Individual cryptographic signature approval from an authorized signer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuorumSignature {
    pub signer: PrincipalId,
    pub signer_public_key: Vec<u8>,
    pub signature: Vec<u8>,
    pub signed_epoch_secs: u64,
}

/// Final verified quorum authorization token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuorumAuthorizationToken {
    pub proposal: QuorumProposal,
    pub approvals: Vec<QuorumSignature>,
    pub authorized_at_epoch_secs: u64,
}

pub type ProposalStore = HashMap<String, (QuorumProposal, Vec<QuorumSignature>)>;

/// M-of-N Threshold Quorum Engine.
pub struct ThresholdQuorumEngine {
    policies: Arc<RwLock<HashMap<String, QuorumPolicy>>>,
    active_proposals: Arc<RwLock<ProposalStore>>,
}

impl Default for ThresholdQuorumEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ThresholdQuorumEngine {
    pub fn new() -> Self {
        Self {
            policies: Arc::new(RwLock::new(HashMap::new())),
            active_proposals: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a quorum policy for a specific operation scope.
    pub fn register_policy(&self, policy: QuorumPolicy) -> Result<(), String> {
        if policy.threshold_m == 0 || policy.threshold_m > policy.authorized_signers_n.len() {
            return Err("InvalidQuorumThreshold: M must be >= 1 and <= total signers N".into());
        }
        let mut map = self.policies.write().map_err(|_| "Policy lock poisoned")?;
        map.insert(policy.policy_id.clone(), policy);
        Ok(())
    }

    /// Propose a new critical operation.
    pub fn create_proposal(
        &self,
        policy_id: &str,
        action: SecurityAction,
        resource: SecurityResource,
        rationale: String,
        proposer: PrincipalId,
        current_epoch_secs: u64,
    ) -> Result<QuorumProposal, String> {
        let policies = self.policies.read().map_err(|_| "Policy lock poisoned")?;
        let pol = policies.get(policy_id).ok_or_else(|| format!("Quorum policy '{}' not found", policy_id))?;

        let proposal_nonce = hex::encode(rand::random::<[u8; 16]>());
        let proposal_id = format!("prop_{}", hex::encode(rand::random::<[u8; 12]>()));

        let proposal = QuorumProposal {
            proposal_id: proposal_id.clone(),
            action,
            resource,
            rationale,
            proposed_by: proposer,
            created_epoch_secs: current_epoch_secs,
            expires_epoch_secs: current_epoch_secs + pol.proposal_ttl_secs,
            proposal_nonce,
        };

        let mut proposals = self.active_proposals.write().map_err(|_| "Proposals lock poisoned")?;
        proposals.retain(|_, (p, _)| p.expires_epoch_secs > current_epoch_secs);
        proposals.insert(proposal_id, (proposal.clone(), Vec::new()));

        Ok(proposal)
    }

    /// Attach a cryptographic signature from an authorized signer.
    pub fn submit_approval(
        &self,
        policy_id: &str,
        proposal_id: &str,
        approval: QuorumSignature,
        current_epoch_secs: u64,
    ) -> Result<Option<QuorumAuthorizationToken>, String> {
        let policies = self.policies.read().map_err(|_| "Policy lock poisoned")?;
        let pol = policies.get(policy_id).ok_or_else(|| format!("Quorum policy '{}' not found", policy_id))?;

        let signer_id = approval.signer.as_str();
        if !pol.authorized_signers_n.contains(&signer_id) {
            return Err(format!("Signer '{}' is not authorized in quorum policy", signer_id));
        }

        let mut proposals = self.active_proposals.write().map_err(|_| "Proposals lock poisoned")?;
        let (proposal, approvals) = proposals.get_mut(proposal_id).ok_or_else(|| {
            format!("Proposal '{}' not found", proposal_id)
        })?;

        if current_epoch_secs > proposal.expires_epoch_secs {
            return Err("ProposalExpired: Proposal validity window has lapsed".into());
        }

        const MAX_CLOCK_SKEW_SECS: u64 = 60;
        if approval.signed_epoch_secs > current_epoch_secs.saturating_add(MAX_CLOCK_SKEW_SECS) {
            return Err("InvalidSignatureTimestamp: Signature timestamp is in the future beyond acceptable skew".into());
        }
        if approval.signed_epoch_secs < proposal.created_epoch_secs {
            return Err("InvalidSignatureTimestamp: Signature timestamp precedes proposal creation time".into());
        }

        if let Some(pinned_pk) = pol.authorized_signer_keys.get(&signer_id) {
            if !CryptoUtils::constant_time_eq(pinned_pk, &approval.signer_public_key) {
                return Err(format!(
                    "SignerPublicKeyMismatch: Provided public key does not match pinned key for '{}'",
                    signer_id
                ));
            }
        }

        // Also ensure no two approvals reuse the exact same public key across different PrincipalIds!
        if approvals.iter().any(|a| a.signer == approval.signer || CryptoUtils::constant_time_eq(&a.signer_public_key, &approval.signer_public_key)) {
            return Err(format!("DuplicateApproval: Signer '{}' or public key has already signed this proposal", signer_id));
        }

        let digest = proposal.compute_digest();
        let verifier = Ed25519Verifier;
        verifier.verify(&approval.signer_public_key, &digest, &approval.signature)?;

        approvals.push(approval);

        if approvals.len() >= pol.threshold_m {
            let token = QuorumAuthorizationToken {
                proposal: proposal.clone(),
                approvals: approvals.clone(),
                authorized_at_epoch_secs: current_epoch_secs,
            };
            proposals.remove(proposal_id);
            Ok(Some(token))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use rand::rngs::OsRng;

    #[test]
    fn test_quorum_policy_signer_key_pinning_and_duplicate_rejection() {
        let engine = ThresholdQuorumEngine::new();

        let signer1_key = SigningKey::generate(&mut OsRng);
        let signer2_key = SigningKey::generate(&mut OsRng);

        let signer1_id = PrincipalId::new("signer1").unwrap();
        let signer2_id = PrincipalId::new("signer2").unwrap();

        let mut authorized_signers = HashSet::new();
        authorized_signers.insert(signer1_id.as_str());
        authorized_signers.insert(signer2_id.as_str());

        let mut authorized_keys = HashMap::new();
        authorized_keys.insert(signer1_id.as_str(), signer1_key.verifying_key().to_bytes().to_vec());

        let policy = QuorumPolicy {
            policy_id: "sec_ops_policy".into(),
            threshold_m: 2,
            authorized_signers_n: authorized_signers,
            proposal_ttl_secs: 300,
            authorized_signer_keys: authorized_keys,
        };

        engine.register_policy(policy).unwrap();

        let proposer = PrincipalId::new("proposer_admin").unwrap();
        let current_time = 1000u64;
        let proposal = engine
            .create_proposal(
                "sec_ops_policy",
                SecurityAction::lift_lockdown(),
                SecurityResource::AuthorityRoot,
                "Lifting lockdown following recovery audit".into(),
                proposer,
                current_time,
            )
            .unwrap();

        let digest = proposal.compute_digest();

        // 1. Signature timestamp in future beyond skew fails
        let future_sig = signer1_key.sign(&digest);
        let approval_future = QuorumSignature {
            signer: signer1_id.clone(),
            signer_public_key: signer1_key.verifying_key().to_bytes().to_vec(),
            signature: future_sig.to_bytes().to_vec(),
            signed_epoch_secs: current_time + 120, // > 60s skew
        };
        let err = engine.submit_approval("sec_ops_policy", &proposal.proposal_id, approval_future, current_time);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("InvalidSignatureTimestamp"));

        // 2. Signature timestamp preceding proposal creation fails
        let past_sig = signer1_key.sign(&digest);
        let approval_past = QuorumSignature {
            signer: signer1_id.clone(),
            signer_public_key: signer1_key.verifying_key().to_bytes().to_vec(),
            signature: past_sig.to_bytes().to_vec(),
            signed_epoch_secs: current_time - 10,
        };
        let err2 = engine.submit_approval("sec_ops_policy", &proposal.proposal_id, approval_past, current_time);
        assert!(err2.is_err());
        assert!(err2.unwrap_err().contains("InvalidSignatureTimestamp"));

        // 3. Unpinned key mismatch: signer1 submits approval signed by signer2's key
        let mismatched_sig = signer2_key.sign(&digest);
        let approval_mismatched = QuorumSignature {
            signer: signer1_id.clone(),
            signer_public_key: signer2_key.verifying_key().to_bytes().to_vec(),
            signature: mismatched_sig.to_bytes().to_vec(),
            signed_epoch_secs: current_time,
        };
        let err3 = engine.submit_approval("sec_ops_policy", &proposal.proposal_id, approval_mismatched, current_time);
        assert!(err3.is_err());
        assert!(err3.unwrap_err().contains("SignerPublicKeyMismatch"));

        // 4. Valid approval from signer1
        let valid_sig1 = signer1_key.sign(&digest);
        let approval1 = QuorumSignature {
            signer: signer1_id.clone(),
            signer_public_key: signer1_key.verifying_key().to_bytes().to_vec(),
            signature: valid_sig1.to_bytes().to_vec(),
            signed_epoch_secs: current_time,
        };
        let res1 = engine.submit_approval("sec_ops_policy", &proposal.proposal_id, approval1, current_time).unwrap();
        assert!(res1.is_none()); // Threshold is 2, only 1 approval so far

        // 5. Signer2 tries to submit using signer1's public key (duplicate public key check)
        let dup_key_sig = signer1_key.sign(&digest);
        let approval_dup_key = QuorumSignature {
            signer: signer2_id.clone(),
            signer_public_key: signer1_key.verifying_key().to_bytes().to_vec(),
            signature: dup_key_sig.to_bytes().to_vec(),
            signed_epoch_secs: current_time,
        };
        let err4 = engine.submit_approval("sec_ops_policy", &proposal.proposal_id, approval_dup_key, current_time);
        assert!(err4.is_err());
        assert!(err4.unwrap_err().contains("DuplicateApproval"));

        // 6. Signer2 submits legitimate approval and reaches quorum threshold M=2
        let valid_sig2 = signer2_key.sign(&digest);
        let approval2 = QuorumSignature {
            signer: signer2_id.clone(),
            signer_public_key: signer2_key.verifying_key().to_bytes().to_vec(),
            signature: valid_sig2.to_bytes().to_vec(),
            signed_epoch_secs: current_time,
        };
        let res2 = engine.submit_approval("sec_ops_policy", &proposal.proposal_id, approval2, current_time).unwrap();
        assert!(res2.is_some());
        let token = res2.unwrap();
        assert_eq!(token.approvals.len(), 2);
    }

    #[test]
    fn test_expired_proposal_pruning() {
        let engine = ThresholdQuorumEngine::new();
        let mut signers = HashSet::new();
        let signer = PrincipalId::new("signer_node").unwrap();
        signers.insert(signer.as_str());

        let policy = QuorumPolicy {
            policy_id: "ephemeral_policy".into(),
            threshold_m: 1,
            authorized_signers_n: signers,
            proposal_ttl_secs: 50,
            authorized_signer_keys: HashMap::new(),
        };
        engine.register_policy(policy).unwrap();

        let proposer = PrincipalId::new("proposer_admin").unwrap();
        // Create first proposal at t = 100, expires at t = 150
        let prop1 = engine.create_proposal(
            "ephemeral_policy",
            SecurityAction::rotate_authority_key(),
            SecurityResource::SystemVault,
            "Key rotation".into(),
            proposer.clone(),
            100,
        ).unwrap();

        // Create second proposal at t = 200 (prop1 has expired)
        let _prop2 = engine.create_proposal(
            "ephemeral_policy",
            SecurityAction::rotate_authority_key(),
            SecurityResource::SystemVault,
            "Key rotation round 2".into(),
            proposer,
            200,
        ).unwrap();

        let proposals = engine.active_proposals.read().unwrap();
        assert!(!proposals.contains_key(&prop1.proposal_id), "Expired proposal must be pruned on new proposal creation");
        assert_eq!(proposals.len(), 1);
    }
}
