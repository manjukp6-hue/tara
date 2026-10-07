//! File-system write-protect enforcement on core source trees.

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
                "rust/tara_server/src/access/".into(),
                "rust/tara_server/src/security/".into(),
                "storage/models/tara/model.safetensors".into(),
            ],
        }
    }

    pub fn is_path_protected(&self, rel_path: &str) -> bool {
        let norm = rel_path.replace('\\', "/");
        for p in &self.protected_roots {
            if norm.starts_with(p) || norm == *p {
                return true;
            }
        }
        false
    }

    pub fn assert_mutation_allowed(
        &self,
        rel_path: &str,
        is_authorized_creator: bool,
    ) -> Result<(), String> {
        if self.is_path_protected(rel_path) && !is_authorized_creator {
            Err(format!("Access Denied: Path '{}' is in protected core tree and requires creator authorization", rel_path))
        } else {
            Ok(())
        }
    }
}
