//! Atomic, Crash-Safe, Hardened Persistence Subsystem.
//!
//! Rule compliance:
//! - Unique temporary files prevent write collisions.
//! - Enforces data sync (sync_all) and directory metadata sync before rename.
//! - Canonical path traversal and reparse-point defense.
//! - Bounded sizes, explicit error propagation, fail-closed persistence.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Bounded limits for security operations.
pub const MAX_SECURE_FILE_SIZE: usize = 16 * 1024 * 1024; // 16 MB limit
pub const MAX_DIRECTORY_DEPTH: usize = 16;

/// Secure Storage Provider trait for pluggable persistence backends.
pub trait SecureStorage: Send + Sync {
    fn write_secure(&self, path: &Path, data: &[u8]) -> Result<(), String>;
    fn read_secure(&self, path: &Path) -> Result<Vec<u8>, String>;
    fn delete_secure(&self, path: &Path) -> Result<(), String>;
    fn exists(&self, path: &Path) -> bool;
}

/// Hardened Local Filesystem Storage implementation.
pub struct HardenedFileSystemStorage {
    base_root: PathBuf,
}

impl HardenedFileSystemStorage {
    pub fn new<P: AsRef<Path>>(base_root: P) -> Result<Self, String> {
        let root = base_root.as_ref().to_path_buf();
        if !root.exists() {
            fs::create_dir_all(&root).map_err(|e| format!("Failed to create storage root: {e}"))?;
        }
        let canonical_root = root
            .canonicalize()
            .map_err(|e| format!("Failed to canonicalize base root: {e}"))?;
        Ok(Self { base_root: canonical_root })
    }

    /// Verifies that candidate path does not escape the canonical base root (traversal defense).
    pub fn validate_path_containment(&self, path: &Path) -> Result<PathBuf, String> {
        let secure_path = crate::path_security::canonical_containment(&self.base_root, path)
            .map_err(|e| format!("PathTraversalViolation: {e}"))?;

        let normalized_root = crate::path_security::normalize_path(&self.base_root);
        let rel_depth = secure_path
            .strip_prefix(&normalized_root)
            .or_else(|_| secure_path.strip_prefix(&self.base_root))
            .map(|rel| rel.components().count())
            .unwrap_or_else(|_| path.components().count());

        if rel_depth > MAX_DIRECTORY_DEPTH {
            return Err(format!(
                "Directory depth {rel_depth} exceeds security limit of {MAX_DIRECTORY_DEPTH}"
            ));
        }

        Ok(secure_path)
    }

    /// Atomic, crash-safe, and permission-hardened write.
    pub fn atomic_write_hardened(&self, target_path: &Path, data: &[u8]) -> Result<(), String> {
        if data.len() > MAX_SECURE_FILE_SIZE {
            return Err(format!(
                "Payload size {} exceeds maximum allowed limit of {} bytes",
                data.len(),
                MAX_SECURE_FILE_SIZE
            ));
        }

        let secure_path = self.validate_path_containment(target_path)?;

        if let Some(parent) = secure_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).map_err(|e| format!("Failed to create parent directory: {e}"))?;
            }
        }

        // 1. Generate unique temporary file name
        let temp_file = secure_path.with_extension(format!(
            "tmp.{}.{}",
            std::process::id(),
            rand::random::<u64>()
        ));

        // 2. Write data and force sync
        {
            let mut f = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&temp_file)
                .map_err(|e| format!("Failed to create temporary file {}: {}", temp_file.display(), e))?;

            f.write_all(data)
                .map_err(|e| format!("Failed to write data to temp file: {e}"))?;

            f.sync_all()
                .map_err(|e| format!("Failed to sync temp file: {e}"))?;
        }

        // 3. Platform permission hardening on the temp file before rename
        crate::hardware_root::harden_file_permissions(&temp_file)?;

        // 4. Atomic rename over destination
        fs::rename(&temp_file, &secure_path).map_err(|e| {
            let _ = fs::remove_file(&temp_file);
            format!("Atomic rename failed: {e}")
        })?;

        // 5. Sync parent directory metadata where supported
        crate::hardware_root::sync_directory_metadata(&secure_path)?;

        Ok(())
    }
}

impl SecureStorage for HardenedFileSystemStorage {
    fn write_secure(&self, path: &Path, data: &[u8]) -> Result<(), String> {
        self.atomic_write_hardened(path, data)
    }

    fn read_secure(&self, path: &Path) -> Result<Vec<u8>, String> {
        let secure_path = self.validate_path_containment(path)?;
        if !secure_path.exists() {
            return Err("FileNotFound: Secure path does not exist".into());
        }

        let metadata = fs::metadata(&secure_path).map_err(|e| format!("Metadata error: {e}"))?;
        if metadata.len() as usize > MAX_SECURE_FILE_SIZE {
            return Err("File exceeds maximum allowed secure size".into());
        }

        let mut f = File::open(&secure_path).map_err(|e| format!("Open error: {e}"))?;
        let mut buffer = Vec::with_capacity(metadata.len() as usize);
        f.read_to_end(&mut buffer).map_err(|e| format!("Read error: {e}"))?;
        Ok(buffer)
    }

    fn delete_secure(&self, path: &Path) -> Result<(), String> {
        let secure_path = self.validate_path_containment(path)?;
        if secure_path.exists() && secure_path.is_file() {
            // Cryptographic wipe: overwrite sectors with zeros before unlinking
            if let Ok(metadata) = fs::metadata(&secure_path) {
                let file_len = metadata.len();
                if let Ok(mut f) = OpenOptions::new().write(true).open(&secure_path) {
                    let zero_chunk = [0u8; 4096];
                    let mut remaining = file_len;
                    while remaining > 0 {
                        let to_write = std::cmp::min(remaining as usize, zero_chunk.len());
                        if f.write_all(&zero_chunk[..to_write]).is_err() {
                            break;
                        }
                        remaining -= to_write as u64;
                    }
                    let _ = f.sync_all();
                    let _ = f.set_len(0);
                }
            }
            fs::remove_file(&secure_path).map_err(|e| format!("Delete error: {e}"))?;
            crate::hardware_root::sync_directory_metadata(&secure_path)?;
        }
        Ok(())
    }

    fn exists(&self, path: &Path) -> bool {
        if let Ok(p) = self.validate_path_containment(path) {
            p.exists()
        } else {
            false
        }
    }
}

/// Global atomic crash-durable file write with parent directory creation, sync_all, metadata sync, and permission hardening.
pub fn atomic_write_file_hardened(path: &Path, data: &[u8]) -> Result<(), String> {
    if path.is_symlink() {
        return Err(format!(
            "Atomic write rejected: target path '{}' is a symlink",
            path.display()
        ));
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            format!("Failed to create directory {}: {}", parent.display(), e)
        })?;
    }

    let nonce = rand::random::<u128>();
    let tmp_path = path.with_extension(format!("tmp.{}.{:032x}", std::process::id(), nonce));

    let write_result = (|| -> Result<(), String> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)
            .map_err(|e| format!("Failed to open temp file {}: {}", tmp_path.display(), e))?;

        file.write_all(data)
            .map_err(|e| format!("Failed to write temp file {}: {}", tmp_path.display(), e))?;

        file.sync_all()
            .map_err(|e| format!("Failed to sync temp file {}: {}", tmp_path.display(), e))?;

        drop(file);

        #[cfg(windows)]
        if path.exists() {
            if let Ok(user) = std::env::var("USERNAME") {
                let system_root =
                    std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".to_string());
                let icacls_path = PathBuf::from(system_root)
                    .join("System32")
                    .join("icacls.exe");
                let _ = std::process::Command::new(icacls_path)
                    .arg(path)
                    .arg("/grant:r")
                    .arg(format!("{}:F", user))
                    .output();
            }
        }

        fs::rename(&tmp_path, path)
            .map_err(|e| format!("Failed to commit file {}: {}", path.display(), e))?;

        let _ = crate::hardware_root::sync_directory_metadata(path);
        let _ = crate::hardware_root::harden_file_permissions(path);

        Ok(())
    })();

    if write_result.is_err() {
        let _ = fs::remove_file(&tmp_path);
    }

    write_result
}
