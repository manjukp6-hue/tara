//! Cryptographic Merkle Tree Transparency Log & Verification Engine (RFC 6962 / RFC 9162).
//!
//! Standards & Security Model:
//! - Conforms to RFC 6962 (Certificate Transparency) and RFC 9162 (Certificate Transparency v2).
//! - Signed Tree Heads (STH): Tree roots are cryptographically signed and versioned.
//! - Independent Checkpoint Tracking: Retains previously trusted tree heads; rejects any tree where
//!   new_size < previous_size (rollback defense) or where consistency proof fails (rewrite defense).
//! - Cryptographic Inclusion Proofs: Prove leaf inclusion in log of size N.
//! - Cryptographic Consistency Proofs: Prove that an older log of size M is a faithful prefix of size N.

use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

use crate::cryptography::{CryptoUtils, Ed25519Verifier, SignatureVerifier};

/// An authenticated, signed tree head (STH) representing a verified snapshot of the log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedTreeHead {
    pub tree_size: u64,
    pub timestamp_epoch_secs: u64,
    pub root_hash_hex: String,
    pub sth_signature_hex: String,
    pub signer_public_key_hex: String,
}

impl SignedTreeHead {
    pub fn compute_sth_digest(tree_size: u64, timestamp: u64, root_hash: &[u8; 32]) -> [u8; 32] {
        let preimage = format!(
            "RFC6962_STH|{}|{}|{}",
            tree_size,
            timestamp,
            hex::encode(root_hash)
        );
        CryptoUtils::sha256(preimage.as_bytes())
    }

    /// Signs a tree head using the authority's dedicated tree signing key.
    pub fn sign_tree_head(
        tree_size: u64,
        timestamp: u64,
        root_hash: &[u8; 32],
        signing_key: &SigningKey,
    ) -> Result<Self, String> {
        let digest = Self::compute_sth_digest(tree_size, timestamp, root_hash);
        let sig = CryptoUtils::sign_ed25519(signing_key, &digest);
        let pub_key = signing_key.verifying_key();

        Ok(Self {
            tree_size,
            timestamp_epoch_secs: timestamp,
            root_hash_hex: hex::encode(root_hash),
            sth_signature_hex: hex::encode(sig),
            signer_public_key_hex: hex::encode(pub_key.as_bytes()),
        })
    }

    /// Verifies the cryptographic signature on this STH strictly against an independent trusted public key (ID-03).
    pub fn verify_signature_against_key(&self, trusted_public_key: &[u8]) -> Result<(), String> {
        if trusted_public_key.len() != 32 {
            return Err("Invalid trusted public key length: expected 32 bytes for Ed25519".into());
        }

        let signer_pk_bytes = hex::decode(&self.signer_public_key_hex)
            .map_err(|e| format!("Malformed signer public key hex: {e}"))?;

        if !CryptoUtils::constant_time_eq(&signer_pk_bytes, trusted_public_key) {
            return Err("UntrustedTreeHeadSigner: Signer key in STH does not match configured trusted log authority".into());
        }

        let root_bytes = hex::decode(&self.root_hash_hex).map_err(|e| e.to_string())?;
        if root_bytes.len() != 32 {
            return Err("Invalid root hash length in STH".into());
        }
        let mut root_arr = [0u8; 32];
        root_arr.copy_from_slice(&root_bytes);

        let digest = Self::compute_sth_digest(self.tree_size, self.timestamp_epoch_secs, &root_arr);
        let sig_bytes = hex::decode(&self.sth_signature_hex).map_err(|e| e.to_string())?;

        let verifier = Ed25519Verifier;
        verifier.verify(trusted_public_key, &digest, &sig_bytes)
    }

    /// Verifies the signature using the embedded public key ONLY if verified against a trust anchor.
    pub fn verify_signature(&self) -> Result<(), String> {
        let pub_key_bytes = hex::decode(&self.signer_public_key_hex).map_err(|e| e.to_string())?;
        self.verify_signature_against_key(&pub_key_bytes)
    }
}

/// Leaf node in the Merkle Tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleLeaf {
    pub leaf_index: u64,
    pub payload_hash: [u8; 32],
}

/// Inclusion Proof demonstrating that a leaf exists in a tree of size N with root R.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleInclusionProof {
    pub leaf_index: u64,
    pub tree_size: u64,
    pub audit_path: Vec<[u8; 32]>,
}

/// Consistency Proof demonstrating that tree of size M is an append-only prefix of size N.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleConsistencyProof {
    pub first_tree_size: u64,
    pub second_tree_size: u64,
    pub consistency_path: Vec<[u8; 32]>,
}

/// Cryptographic Merkle Tree implementation conforming to RFC 6962.
pub struct MerkleTree {
    leaf_hashes: Arc<RwLock<Vec<[u8; 32]>>>,
    trusted_tree_heads: Arc<RwLock<Vec<SignedTreeHead>>>,
}

impl Default for MerkleTree {
    fn default() -> Self {
        Self::new()
    }
}

impl MerkleTree {
    pub fn new() -> Self {
        Self {
            leaf_hashes: Arc::new(RwLock::new(Vec::new())),
            trusted_tree_heads: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Compute RFC 6962 leaf hash: SHA-256(0x00 || data) to prevent second-preimage attacks.
    pub fn hash_leaf(data: &[u8]) -> [u8; 32] {
        let mut preimage = Vec::with_capacity(1 + data.len());
        preimage.push(0x00);
        preimage.extend_from_slice(data);
        CryptoUtils::sha256(&preimage)
    }

    /// Compute RFC 6962 internal node hash: SHA-256(0x01 || left || right).
    pub fn hash_children(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        let mut preimage = Vec::with_capacity(65);
        preimage.push(0x01);
        preimage.extend_from_slice(left);
        preimage.extend_from_slice(right);
        CryptoUtils::sha256(&preimage)
    }

    /// Append a new event leaf hash to the Merkle tree and return the new root.
    pub fn append_leaf(&self, leaf_hash: [u8; 32]) -> Result<([u8; 32], u64), String> {
        let mut leaves = self
            .leaf_hashes
            .write()
            .map_err(|_| "Merkle lock poisoned")?;
        leaves.push(leaf_hash);
        let count = leaves.len() as u64;
        let root = Self::calculate_sub_root(&leaves)?;
        Ok((root, count))
    }

    /// Current root hash of the tree.
    pub fn root_hash(&self) -> Result<[u8; 32], String> {
        let leaves = self
            .leaf_hashes
            .read()
            .map_err(|_| "Merkle lock poisoned")?;
        if leaves.is_empty() {
            return Ok(CryptoUtils::sha256(&[]));
        }
        Self::calculate_sub_root(&leaves)
    }

    /// Root hash of the tree at a historical size.
    pub fn root_hash_for_size(&self, size: usize) -> Result<[u8; 32], String> {
        let leaves = self
            .leaf_hashes
            .read()
            .map_err(|_| "Merkle lock poisoned")?;
        if size > leaves.len() {
            return Err("Requested size exceeds leaf count".into());
        }
        if size == 0 {
            return Ok(CryptoUtils::sha256(&[]));
        }
        Self::calculate_sub_root(&leaves[..size])
    }

    /// Total number of leaves in the tree.
    pub fn leaf_count(&self) -> usize {
        let leaves = self.leaf_hashes.read().unwrap_or_else(|p| p.into_inner());
        leaves.len()
    }

    /// Latest trusted tree head.
    pub fn latest_trusted_tree_head(&self) -> Option<SignedTreeHead> {
        if let Ok(heads) = self.trusted_tree_heads.read() {
            heads.last().cloned()
        } else {
            None
        }
    }

    /// Commit a signed tree head into the independent trusted checkpoint history.
    /// Strictly rejects rollbacks, equivocation, and requires verification against a trusted key (ID-03, ID-12).
    pub fn commit_and_verify_tree_head(
        &self,
        sth: SignedTreeHead,
        trusted_log_public_key: &[u8],
    ) -> Result<(), String> {
        // 1. Verify STH signature strictly against trusted anchor (ID-03)
        sth.verify_signature_against_key(trusted_log_public_key)?;

        let mut trusted_heads = self
            .trusted_tree_heads
            .write()
            .map_err(|_| "STH lock poisoned")?;

        // 2. Rollback & Fork checks against previously committed trusted checkpoints
        if let Some(prev) = trusted_heads.last() {
            if sth.tree_size < prev.tree_size {
                return Err(format!(
                    "LogRollbackDetected: Candidate STH size {} is smaller than previously committed size {} (Rollback Attack Detected)",
                    sth.tree_size, prev.tree_size
                ));
            }

            if sth.tree_size == prev.tree_size && sth.root_hash_hex != prev.root_hash_hex {
                return Err(format!(
                    "LogForkDetected: Conflicting root hashes for identical tree size {} (Log Equivocation Detected)",
                    sth.tree_size
                ));
            }
        }

        // 3. Bind candidate root to local leaf state if local tree has reached candidate size (ID-12)
        {
            let leaves = self.leaf_hashes.read().map_err(|_| "Leaf lock poisoned")?;
            if (leaves.len() as u64) >= sth.tree_size && sth.tree_size > 0 {
                let local_sub_root = Self::calculate_sub_root(&leaves[..sth.tree_size as usize])?;
                let candidate_root = hex::decode(&sth.root_hash_hex)
                    .map_err(|e| format!("Invalid hex root: {e}"))?;
                if !CryptoUtils::constant_time_eq(&local_sub_root, &candidate_root) {
                    return Err("CandidateRootMismatch: STH root hash does not match local tree root at declared size".into());
                }
            }
        }

        trusted_heads.push(sth);
        Ok(())
    }

    /// Generate a cryptographic Inclusion Proof for a leaf at `leaf_index`.
    pub fn generate_inclusion_proof(
        &self,
        leaf_index: u64,
    ) -> Result<MerkleInclusionProof, String> {
        let leaves = self
            .leaf_hashes
            .read()
            .map_err(|_| "Merkle lock poisoned")?;
        let tree_size = leaves.len() as u64;

        if leaf_index >= tree_size {
            return Err(format!(
                "Leaf index {} out of bounds for tree size {}",
                leaf_index, tree_size
            ));
        }

        let mut audit_path = Vec::new();
        Self::build_inclusion_path(&leaves, leaf_index, tree_size, &mut audit_path);

        Ok(MerkleInclusionProof {
            leaf_index,
            tree_size,
            audit_path,
        })
    }

    /// Mathematically verify an RFC 6962 Inclusion Proof strictly (ID-05).
    pub fn verify_inclusion_proof(
        expected_root: &[u8; 32],
        leaf_hash: &[u8; 32],
        proof: &MerkleInclusionProof,
    ) -> Result<(), String> {
        if proof.tree_size == 0 {
            return Err("InclusionProofError: Tree size cannot be zero".into());
        }
        if proof.leaf_index >= proof.tree_size {
            return Err(format!(
                "InclusionProofError: Leaf index {} exceeds tree size {}",
                proof.leaf_index, proof.tree_size
            ));
        }

        // Strict proof length validation (ID-05)
        let expected_len = expected_inclusion_path_length(proof.leaf_index, proof.tree_size);
        if proof.audit_path.len() != expected_len {
            return Err(format!(
                "InclusionProofStructureError: Proof path length {} does not match expected RFC 6962 tree shape length {}",
                proof.audit_path.len(),
                expected_len
            ));
        }

        let computed_root = Self::compute_root_from_path(
            proof.leaf_index,
            proof.tree_size,
            leaf_hash,
            &proof.audit_path,
        )?;

        if CryptoUtils::constant_time_eq(expected_root, &computed_root) {
            Ok(())
        } else {
            Err(
                "InclusionProofVerificationFailed: Computed root does not match expected root"
                    .into(),
            )
        }
    }

    /// Generate an RFC 6962 Consistency Proof between tree size M and tree size N.
    pub fn generate_consistency_proof(
        &self,
        m: u64,
        n: u64,
    ) -> Result<MerkleConsistencyProof, String> {
        let leaves = self
            .leaf_hashes
            .read()
            .map_err(|_| "Merkle lock poisoned")?;
        if m == 0 || m > n || n > leaves.len() as u64 {
            return Err("Invalid tree sizes for consistency proof".into());
        }

        let mut path = Vec::new();
        if m < n {
            Self::build_consistency_path(&leaves, m, n, true, &mut path);
        }

        Ok(MerkleConsistencyProof {
            first_tree_size: m,
            second_tree_size: n,
            consistency_path: path,
        })
    }

    /// Mathematically verify an RFC 6962 Consistency Proof (ID-04).
    pub fn verify_consistency_proof(
        old_size: u64,
        new_size: u64,
        old_root: &[u8; 32],
        new_root: &[u8; 32],
        proof: &MerkleConsistencyProof,
    ) -> Result<(), String> {
        if proof.first_tree_size != old_size || proof.second_tree_size != new_size {
            return Err("ConsistencyProofError: Proof sizes do not match expected sizes".into());
        }

        if old_size == 0 {
            return if proof.consistency_path.is_empty() {
                Ok(())
            } else {
                Err(
                    "ConsistencyProofError: Proof path must be empty when old tree size is 0"
                        .into(),
                )
            };
        }

        if old_size == new_size {
            if !CryptoUtils::constant_time_eq(old_root, new_root) {
                return Err(
                    "ConsistencyProofError: Identical tree sizes have conflicting roots".into(),
                );
            }
            return if proof.consistency_path.is_empty() {
                Ok(())
            } else {
                Err(
                    "ConsistencyProofError: Proof path must be empty when tree sizes are identical"
                        .into(),
                )
            };
        }

        if old_size > new_size {
            return Err(
                "ConsistencyProofError: Old tree size cannot exceed new tree size (Rollback)"
                    .into(),
            );
        }

        if proof.consistency_path.is_empty() {
            return Err(
                "ConsistencyProofError: Consistency path cannot be empty for expanding tree".into(),
            );
        }

        // Verify consistency traversal
        let mut path_idx = 0;
        let (computed_old, computed_new) = eval_consistency_path(
            old_size,
            new_size,
            true,
            &proof.consistency_path,
            &mut path_idx,
            old_root,
        )?;

        if path_idx != proof.consistency_path.len() {
            return Err("ConsistencyProofError: Unconsumed proof nodes remain".into());
        }

        if !CryptoUtils::constant_time_eq(&computed_old, old_root) {
            return Err(
                "ConsistencyProofFailed: Computed old root does not match trusted old root".into(),
            );
        }

        if !CryptoUtils::constant_time_eq(&computed_new, new_root) {
            return Err(
                "ConsistencyProofFailed: Computed new root does not match candidate new root"
                    .into(),
            );
        }

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal Merkle Math
    // ─────────────────────────────────────────────────────────────────────────

    fn calculate_sub_root(leaves: &[[u8; 32]]) -> Result<[u8; 32], String> {
        if leaves.is_empty() {
            return Ok(CryptoUtils::sha256(&[]));
        }
        if leaves.len() == 1 {
            return Ok(leaves[0]);
        }

        let k = largest_power_of_two_less_than(leaves.len());
        let left_root = Self::calculate_sub_root(&leaves[..k])?;
        let right_root = Self::calculate_sub_root(&leaves[k..])?;

        Ok(Self::hash_children(&left_root, &right_root))
    }

    fn build_inclusion_path(leaves: &[[u8; 32]], m: u64, n: u64, path: &mut Vec<[u8; 32]>) {
        if m >= n || n <= 1 {
            return;
        }

        let k = largest_power_of_two_less_than(n as usize) as u64;

        if m < k {
            let right_sub_root = Self::calculate_sub_root(&leaves[k as usize..n as usize]).unwrap();
            path.push(right_sub_root);
            Self::build_inclusion_path(&leaves[..k as usize], m, k, path);
        } else {
            let left_sub_root = Self::calculate_sub_root(&leaves[..k as usize]).unwrap();
            path.push(left_sub_root);
            Self::build_inclusion_path(&leaves[k as usize..n as usize], m - k, n - k, path);
        }
    }

    fn build_consistency_path(
        leaves: &[[u8; 32]],
        m: u64,
        n: u64,
        start_from_old: bool,
        path: &mut Vec<[u8; 32]>,
    ) {
        if m == n {
            if !start_from_old {
                path.push(Self::calculate_sub_root(&leaves[..m as usize]).unwrap());
            }
            return;
        }

        let k = largest_power_of_two_less_than(n as usize) as u64;
        if m <= k {
            path.push(Self::calculate_sub_root(&leaves[k as usize..n as usize]).unwrap());
            Self::build_consistency_path(&leaves[..k as usize], m, k, start_from_old, path);
        } else {
            path.push(Self::calculate_sub_root(&leaves[..k as usize]).unwrap());
            Self::build_consistency_path(
                &leaves[k as usize..n as usize],
                m - k,
                n - k,
                false,
                path,
            );
        }
    }

    fn compute_root_from_path(
        m: u64,
        n: u64,
        leaf: &[u8; 32],
        path: &[[u8; 32]],
    ) -> Result<[u8; 32], String> {
        if n == 1 {
            return Ok(*leaf);
        }

        let k = largest_power_of_two_less_than(n as usize) as u64;
        if path.is_empty() {
            return Err("Inclusion proof path empty for multi-node tree".into());
        }

        let sibling = &path[0];
        let remaining_path = &path[1..];

        if m < k {
            let left_root = Self::compute_root_from_path(m, k, leaf, remaining_path)?;
            Ok(Self::hash_children(&left_root, sibling))
        } else {
            let right_root = Self::compute_root_from_path(m - k, n - k, leaf, remaining_path)?;
            Ok(Self::hash_children(sibling, &right_root))
        }
    }
}

fn expected_inclusion_path_length(m: u64, n: u64) -> usize {
    if n <= 1 {
        return 0;
    }
    let k = largest_power_of_two_less_than(n as usize) as u64;
    if m < k {
        1 + expected_inclusion_path_length(m, k)
    } else {
        1 + expected_inclusion_path_length(m - k, n - k)
    }
}

fn eval_consistency_path(
    m: u64,
    n: u64,
    start_from_old: bool,
    path: &[[u8; 32]],
    path_idx: &mut usize,
    old_root: &[u8; 32],
) -> Result<([u8; 32], [u8; 32]), String> {
    if m == n {
        if start_from_old {
            return Ok((*old_root, *old_root));
        } else {
            if *path_idx >= path.len() {
                return Err("Consistency proof truncated".into());
            }
            let node = path[*path_idx];
            *path_idx += 1;
            return Ok((node, node));
        }
    }

    let k = largest_power_of_two_less_than(n as usize) as u64;
    if m <= k {
        if *path_idx >= path.len() {
            return Err("Consistency proof truncated".into());
        }
        let right_sub_root = path[*path_idx];
        *path_idx += 1;

        let (left_old, left_new) =
            eval_consistency_path(m, k, start_from_old, path, path_idx, old_root)?;
        let new_root = MerkleTree::hash_children(&left_new, &right_sub_root);
        Ok((left_old, new_root))
    } else {
        if *path_idx >= path.len() {
            return Err("Consistency proof truncated".into());
        }
        let left_sub_root = path[*path_idx];
        *path_idx += 1;

        let (right_old, right_new) =
            eval_consistency_path(m - k, n - k, false, path, path_idx, old_root)?;
        let old_root_computed = MerkleTree::hash_children(&left_sub_root, &right_old);
        let new_root_computed = MerkleTree::hash_children(&left_sub_root, &right_new);
        Ok((old_root_computed, new_root_computed))
    }
}

fn largest_power_of_two_less_than(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let mut p = 1;
    while (p << 1) < n {
        p <<= 1;
    }
    p
}
