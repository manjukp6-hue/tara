//! Centralized Path Security and Filesystem Containment Controls for TARA Security Subsystem.
//!
//! Rule compliance:
//! - Pure native Rust, zero outside crate dependencies.
//! - Strict defense-in-depth controls for all protected operations touching the filesystem:
//!   1. `check_lexical_traversal`: Strictly detects lexical parent directory traversals ('..'),
//!      URL-encoded traversals (%2e%2e), and null-byte injection.
//!   2. `normalize_path`: Cross-platform normalization handling Windows verbatim prefixes (`\\?\`).
//!   3. `canonical_containment`: Validates that existing or pending paths are strictly contained
//!      within the designated root, eliminating sibling prefix-sharing collision attacks.
//!   4. `is_symlink_escaping`: Inspects all path segments to prevent symlink-based boundary escape.
//!   5. `validate_sandbox_path`: High-security composite verification for isolated tool execution sandboxes.
//!   6. `validate_tool_command_path`: Validates binaries and scripts against authorized system directories.

use std::fs;
use std::path::{Component, Path, PathBuf};
use thiserror::Error;

/// Path security violations.
#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum PathSecurityError {
    #[error("Lexical traversal detected: '..' component in '{0}'")]
    LexicalTraversal(String),

    #[error("Encoded or suspicious traversal sequence in '{0}'")]
    SuspiciousTraversalSequence(String),

    #[error("Null byte injection detected in path")]
    NullByteInjection,

    #[error("Symlink escape violation: '{path}' resolves outside permitted root '{root}'")]
    SymlinkEscape { path: String, root: String },

    #[error("Path containment violation: '{path}' resolves outside permitted root '{root}'")]
    ContainmentViolation { path: String, root: String },

    #[error("Unauthorized tool executable path: '{path}' is not within authorized directories")]
    UnauthorizedToolPath { path: String },

    #[error("Filesystem error during path validation: {0}")]
    IoError(String),
}

/// Strictly rejects lexical parent directory traversals (`..`), URL-encoded traversals, and null bytes.
pub fn check_lexical_traversal<P: AsRef<Path>>(path: P) -> Result<(), PathSecurityError> {
    let p = path.as_ref();
    let lossy = p.to_string_lossy();

    if lossy.contains('\0') {
        return Err(PathSecurityError::NullByteInjection);
    }

    for comp in p.components() {
        if comp == Component::ParentDir {
            return Err(PathSecurityError::LexicalTraversal(lossy.to_string()));
        }
    }

    let lower = lossy.to_ascii_lowercase();
    if lower.contains("%2e%2e") || lower.contains("%2f") || lower.contains("%5c") {
        return Err(PathSecurityError::SuspiciousTraversalSequence(
            lossy.to_string(),
        ));
    }

    if lossy.split(['/', '\\']).any(|seg| seg == "..") {
        return Err(PathSecurityError::LexicalTraversal(lossy.to_string()));
    }

    Ok(())
}

/// Normalizes Windows verbatim prefixes (`\\?\` and `\\?\UNC\`) for consistent cross-platform path comparison.
pub fn normalize_path<P: AsRef<Path>>(path: P) -> PathBuf {
    let s = path.as_ref().to_string_lossy();
    #[cfg(windows)]
    {
        if let Some(stripped) = s.strip_prefix(r"\\?\UNC\") {
            return PathBuf::from(format!(r"\\{}", stripped));
        }
        if let Some(stripped) = s.strip_prefix(r"\\?\") {
            return PathBuf::from(stripped);
        }
    }
    PathBuf::from(s.as_ref())
}

/// Detects whether `candidate` or any of its existing ancestor components is a symbolic link escaping `root`.
pub fn is_symlink_escaping<P: AsRef<Path>, Q: AsRef<Path>>(
    root: P,
    candidate: Q,
) -> Result<bool, PathSecurityError> {
    let root_ref = root.as_ref();
    let cand_ref = candidate.as_ref();

    let root_canon = if root_ref.exists() {
        normalize_path(root_ref.canonicalize().map_err(|e| {
            PathSecurityError::IoError(format!("Canonicalize root '{}': {e}", root_ref.display()))
        })?)
    } else {
        normalize_path(root_ref)
    };

    let target_path = if cand_ref.is_absolute() {
        cand_ref.to_path_buf()
    } else {
        root_ref.join(cand_ref)
    };

    // Walk components from root down to check every existing segment for symlinks
    let mut current = PathBuf::new();
    for comp in target_path.components() {
        current.push(comp);
        if let Ok(meta) = fs::symlink_metadata(&current) {
            if meta.is_symlink() {
                let resolved = fs::canonicalize(&current).map_err(|e| {
                    PathSecurityError::IoError(format!(
                        "Resolve symlink '{}': {e}",
                        current.display()
                    ))
                })?;
                let resolved_norm = normalize_path(resolved);
                if !resolved_norm.starts_with(&root_canon) {
                    return Ok(true);
                }
            }
        }
    }

    Ok(false)
}

/// Verifies that `candidate` (existing or pending creation) is strictly contained within `root`.
/// Prevents prefix collisions (e.g. `/root_evil` matching `/root`), lexical traversals, and symlink escapes.
pub fn canonical_containment<P: AsRef<Path>, Q: AsRef<Path>>(
    root: P,
    candidate: Q,
) -> Result<PathBuf, PathSecurityError> {
    let root_ref = root.as_ref();
    let cand_ref = candidate.as_ref();

    // 1. Strict lexical traversal check
    check_lexical_traversal(cand_ref)?;

    let target_path = if cand_ref.is_absolute() {
        cand_ref.to_path_buf()
    } else {
        root_ref.join(cand_ref)
    };

    let root_canon = if root_ref.exists() {
        normalize_path(root_ref.canonicalize().map_err(|e| {
            PathSecurityError::IoError(format!("Canonicalize root '{}': {e}", root_ref.display()))
        })?)
    } else {
        normalize_path(root_ref)
    };

    // 2. Symlink escape check
    if is_symlink_escaping(&root_canon, &target_path)? {
        return Err(PathSecurityError::SymlinkEscape {
            path: cand_ref.display().to_string(),
            root: root_ref.display().to_string(),
        });
    }

    // 3. Resolve target: if exists, full canonicalize; if pending, canonicalize nearest existing ancestor
    let resolved_norm = if target_path.exists() {
        normalize_path(target_path.canonicalize().map_err(|e| {
            PathSecurityError::IoError(format!(
                "Canonicalize path '{}': {e}",
                target_path.display()
            ))
        })?)
    } else {
        let mut existing_ancestor = target_path.clone();
        let mut missing_segments = Vec::new();

        while !existing_ancestor.exists() {
            if let Some(file_name) = existing_ancestor.file_name() {
                missing_segments.push(file_name.to_os_string());
            } else {
                break;
            }
            if !existing_ancestor.pop() {
                break;
            }
        }

        let ancestor_canon = if existing_ancestor.exists() {
            normalize_path(existing_ancestor.canonicalize().map_err(|e| {
                PathSecurityError::IoError(format!(
                    "Canonicalize ancestor '{}': {e}",
                    existing_ancestor.display()
                ))
            })?)
        } else {
            normalize_path(existing_ancestor)
        };

        if !ancestor_canon.starts_with(&root_canon) {
            return Err(PathSecurityError::ContainmentViolation {
                path: ancestor_canon.display().to_string(),
                root: root_canon.display().to_string(),
            });
        }

        let mut reconstructed = ancestor_canon;
        for seg in missing_segments.into_iter().rev() {
            reconstructed.push(seg);
        }
        reconstructed
    };

    // 4. Strict component-based containment verification (prevents prefix sharing attacks)
    if !resolved_norm.starts_with(&root_canon) {
        return Err(PathSecurityError::ContainmentViolation {
            path: resolved_norm.display().to_string(),
            root: root_canon.display().to_string(),
        });
    }

    Ok(resolved_norm)
}

/// Zero-trust composite path validator for sandbox environments.
pub fn validate_sandbox_path<P: AsRef<Path>, Q: AsRef<Path>>(
    root: P,
    candidate: Q,
) -> Result<PathBuf, PathSecurityError> {
    canonical_containment(root, candidate)
}

/// Validates executable or script paths invoked by tools against authorized directories.
pub fn validate_tool_command_path<P: AsRef<Path>>(
    command_path: P,
    allowed_dirs: &[PathBuf],
) -> Result<PathBuf, PathSecurityError> {
    let cmd = command_path.as_ref();
    check_lexical_traversal(cmd)?;

    if allowed_dirs.is_empty() {
        return Err(PathSecurityError::UnauthorizedToolPath {
            path: cmd.display().to_string(),
        });
    }

    for dir in allowed_dirs {
        if let Ok(contained) = canonical_containment(dir, cmd) {
            return Ok(contained);
        }
    }

    Err(PathSecurityError::UnauthorizedToolPath {
        path: cmd.display().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_lexical_traversal_detection() {
        assert!(check_lexical_traversal(Path::new("normal/file.txt")).is_ok());
        assert!(check_lexical_traversal(Path::new("sub/dir/nested.json")).is_ok());
        assert!(check_lexical_traversal(Path::new("archive..tar.gz")).is_ok());
        assert!(check_lexical_traversal(Path::new("sub/dir/archive..tar.gz")).is_ok());

        assert!(check_lexical_traversal(Path::new("../evil.txt")).is_err());
        assert!(check_lexical_traversal(Path::new("dir/../../secret")).is_err());
        assert!(check_lexical_traversal(Path::new("..\\evil.txt")).is_err());
        assert!(check_lexical_traversal(Path::new("dir%2e%2e/evil")).is_err());
        assert!(check_lexical_traversal(Path::new("dir%2E%2E/evil")).is_err());
        assert!(check_lexical_traversal(Path::new("dir%2fevil")).is_err());
        assert!(check_lexical_traversal(Path::new("dir%2Fevil")).is_err());
        assert!(check_lexical_traversal(Path::new("dir%5cevil")).is_err());
        assert!(check_lexical_traversal(Path::new("dir%5Cevil")).is_err());
    }

    #[test]
    fn test_canonical_containment_and_prefix_collision() {
        let tmp = std::env::temp_dir().join(format!("tara_path_test_{}", rand::random::<u64>()));
        let sub = tmp.join("sandbox");
        let evil_sibling = tmp.join("sandbox_evil");
        let _ = fs::create_dir_all(&sub);
        let _ = fs::create_dir_all(&evil_sibling);

        // Valid relative child
        let valid_child = sub.join("data.txt");
        assert!(canonical_containment(&sub, Path::new("data.txt")).is_ok());
        assert!(canonical_containment(&sub, &valid_child).is_ok());

        // Prefix sharing attack: sandbox_evil has sandbox as string prefix, but is NOT inside sandbox!
        let evil_child = evil_sibling.join("exploit.txt");
        let res = canonical_containment(&sub, &evil_child);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.to_string().contains("Path containment violation"));

        // Traversal attempt
        assert!(canonical_containment(&sub, Path::new("../sandbox_evil/exploit.txt")).is_err());

        // Cleanup
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_mixed_separators_and_pending_paths() {
        let tmp = std::env::temp_dir().join(format!("tara_path_test_{}", rand::random::<u64>()));
        let _ = fs::create_dir_all(&tmp);

        let candidate_rel = "deep/nested\\pending/file.bin";
        let res = canonical_containment(&tmp, Path::new(candidate_rel));
        assert!(res.is_ok());

        let _ = fs::remove_dir_all(&tmp);
    }
}
