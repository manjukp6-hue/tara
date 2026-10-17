//! Source Protector: File-system write-protection on core source trees, architecture maps, and sensitive vaults.
//!
//! Rule compliance:
//! - Pure native Rust, zero outside crate dependencies.
//! - Prevents autonomous agents, untrusted plugins, or unauthorized tasks from modifying
//!   core Rust code, model weights, architecture registries, or security credentials.
//! - Strictly requires verified Creator-Root authorization through the Central Enforcement Gateway.
//! - Strictly rejects path traversal attempts.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::broker::{EnforcementGateway, EnforcementRequest, ProtectedCategory};
use crate::path_security::check_lexical_traversal;
use crate::primitives::{
    AssuranceLevel, SecurityAction, SecurityContext, SecurityResource, VerifiedPrincipal,
};

/// Core Source Protector guarding critical project assets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceProtector {
    protected_roots: Vec<String>,
}

impl Default for SourceProtector {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceProtector {
    pub fn new() -> Self {
        Self {
            protected_roots: vec![
                "rust/tara_core/src/governance.rs".into(),
                "rust/tara_core/src/permissions.rs".into(),
                "rust/tara_security/".into(),
                "rust/tara_server/src/security/".into(),
                "storage/models/".into(),
                "storage/vault/".into(),
                "ARCHITECTURE/".into(),
                "TARA/ACCESS/".into(),
                "TARA/RULES/".into(),
            ],
        }
    }

    pub fn with_protected_roots(roots: Vec<String>) -> Self {
        Self {
            protected_roots: roots,
        }
    }

    pub fn protected_roots(&self) -> &[String] {
        &self.protected_roots
    }

    /// Add a protected root dynamically at runtime.
    pub fn add_protected_root(&mut self, root: impl Into<String>) {
        let r = root.into();
        if !self.protected_roots.contains(&r) {
            self.protected_roots.push(r);
        }
    }

    /// Checks if a relative or normalized path targets a protected system root.
    /// Traversal attempts ('..') are strictly detected and treated as protected/blocked.
    pub fn is_path_protected(&self, rel_path: &str) -> bool {
        let path = Path::new(rel_path);
        if check_lexical_traversal(path).is_err() {
            return true;
        }

        let mut norm = rel_path.replace('\\', "/").to_lowercase();
        while norm.starts_with("./") {
            norm = norm[2..].to_string();
        }
        while norm.starts_with('/') {
            norm = norm[1..].to_string();
        }

        for p in &self.protected_roots {
            let p_norm = p.replace('\\', "/").to_lowercase();
            let p_trimmed = p_norm.trim_matches('/');
            if norm == p_trimmed || norm.starts_with(&format!("{}/", p_trimmed)) {
                return true;
            }
        }

        false
    }

    /// Asserts that a file mutation is authorized by verifying through the Central Enforcement Gateway.
    pub fn assert_mutation_allowed(
        &self,
        rel_path: &str,
        repo_root: &Path,
        principal: &VerifiedPrincipal,
        context: &SecurityContext,
        session_id: Option<String>,
        gateway: &EnforcementGateway,
    ) -> Result<PathBuf, String> {
        let path = Path::new(rel_path);
        check_lexical_traversal(path).map_err(|e| {
            format!(
                "AccessDenied: Path traversal in '{}' is prohibited: {}",
                rel_path, e
            )
        })?;

        let is_protected = self.is_path_protected(rel_path);

        if is_protected {
            // Policy Gate: Modification of protected source roots strictly demands Creator Root assurance!
            if principal.assurance_level().score() < AssuranceLevel::CREATOR_ROOT.score() {
                return Err(format!(
                    "AccessDenied: Path '{}' is in protected core tree and requires verified Creator Root authorization",
                    rel_path
                ));
            }
        }

        let full_path = repo_root.join(rel_path);

        // Enforce write operation through the Central Gateway
        let request = EnforcementRequest {
            principal: principal.clone(),
            session_id,
            action: SecurityAction::write_protected_resource(),
            resource: SecurityResource::custom("source_tree", rel_path.replace('\\', "/"))?,
            context: context.clone(),
            category: ProtectedCategory::Files {
                path: full_path.clone(),
                permitted_root: repo_root.to_path_buf(),
                is_write: true,
            },
        };

        gateway.enforce_protected_operation(request, |_p, _r| Ok(full_path))
    }
}
