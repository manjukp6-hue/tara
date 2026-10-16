//! Independent Cryptographic Transparency Log Subsystem (Pillar 4).
//!
//! Standards & Guarantees:
//! - Conforms to RFC 6962 and RFC 9162 Merkle tree transparency standards.
//! - Signed Tree Heads (STH) signed by root authority.
//! - Cryptographic inclusion proofs and consistency proofs.
//! - Detects log forks, log rewrites, and rollback attacks across generations.
//! - Authenticated hash-chained audit logging with fail-closed persistence.

pub mod audit;
pub mod merkle;

pub use audit::{AuditRecord, AuditSink, AuthenticatedAuditLogger, FileAuditSink, MemoryAuditSink};
pub use merkle::{
    MerkleConsistencyProof, MerkleInclusionProof, MerkleLeaf, MerkleTree, SignedTreeHead,
};
