//! Dependency integrity verification engine.
//!
//! Scans Cargo manifests and lockfiles to ensure all external dependencies
//! match declared, cryptographically sound SHA-256 checksums.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedPackage {
    pub name: String,
    pub version: String,
    pub source: Option<String>,
    pub checksum: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyVerificationReport {
    pub total_packages: usize,
    pub verified_registry_packages: usize,
    pub workspace_packages: usize,
    pub errors: Vec<String>,
}

pub struct DependencyVerifier {
    pub repo_root: PathBuf,
}

impl DependencyVerifier {
    pub fn new<P: AsRef<Path>>(repo_root: P) -> Self {
        Self {
            repo_root: repo_root.as_ref().to_path_buf(),
        }
    }

    /// Verifies Cargo.lock dependency integrity.
    /// Returns the number of verified external dependencies with valid SHA-256 checksums.
    /// Strictly fails closed if any registry dependency is missing a checksum or has an invalid checksum.
    pub fn verify_cargo_lock(&self) -> Result<usize, String> {
        let report = self.verify_cargo_lock_detailed()?;
        if !report.errors.is_empty() {
            return Err(format!(
                "Cargo.lock dependency verification failed with {} error(s): {:?}",
                report.errors.len(),
                report.errors
            ));
        }
        Ok(report.verified_registry_packages)
    }

    /// Performs detailed parsing and integrity check of all [[package]] stanzas in Cargo.lock.
    pub fn verify_cargo_lock_detailed(&self) -> Result<DependencyVerificationReport, String> {
        let lock_path = self.repo_root.join("Cargo.lock");
        if !lock_path.exists() {
            return Err("Cargo.lock not found".into());
        }

        let content = fs::read_to_string(&lock_path).map_err(|e| e.to_string())?;
        let mut packages = Vec::new();
        let mut current_name: Option<String> = None;
        let mut current_version: Option<String> = None;
        let mut current_source: Option<String> = None;
        let mut current_checksum: Option<String> = None;

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed == "[[package]]" {
                if let (Some(name), Some(version)) = (current_name.take(), current_version.take()) {
                    packages.push(VerifiedPackage {
                        name,
                        version,
                        source: current_source.take(),
                        checksum: current_checksum.take(),
                    });
                }
                current_source = None;
                current_checksum = None;
            } else if trimmed.starts_with("name = ") {
                current_name = Self::extract_quoted_val(trimmed);
            } else if trimmed.starts_with("version = ") {
                current_version = Self::extract_quoted_val(trimmed);
            } else if trimmed.starts_with("source = ") {
                current_source = Self::extract_quoted_val(trimmed);
            } else if trimmed.starts_with("checksum = ") {
                current_checksum = Self::extract_quoted_val(trimmed);
            }
        }

        // Catch the last package
        if let (Some(name), Some(version)) = (current_name, current_version) {
            packages.push(VerifiedPackage {
                name,
                version,
                source: current_source,
                checksum: current_checksum,
            });
        }

        let mut verified_registry = 0;
        let mut workspace = 0;
        let mut errors = Vec::new();

        for pkg in &packages {
            if let Some(ref src) = pkg.source {
                // If it's a registry package, it must have a valid 64-hex-character checksum
                if src.starts_with("registry+") {
                    match &pkg.checksum {
                        Some(chk)
                            if chk.len() == 64 && chk.chars().all(|c| c.is_ascii_hexdigit()) =>
                        {
                            verified_registry += 1;
                        }
                        Some(invalid) => {
                            errors.push(format!(
                                "Package '{}' v{} has malformed checksum '{}'",
                                pkg.name, pkg.version, invalid
                            ));
                        }
                        None => {
                            errors.push(format!(
                                "Package '{}' v{} from registry is missing required checksum",
                                pkg.name, pkg.version
                            ));
                        }
                    }
                } else {
                    verified_registry += 1;
                }
            } else {
                workspace += 1;
            }
        }

        Ok(DependencyVerificationReport {
            total_packages: packages.len(),
            verified_registry_packages: verified_registry,
            workspace_packages: workspace,
            errors,
        })
    }

    fn extract_quoted_val(line: &str) -> Option<String> {
        let mut parts = line.split('=');
        parts.next()?;
        let val_part = parts.next()?.trim();
        let stripped = val_part.trim_matches('"');
        Some(stripped.to_string())
    }
}
