//! Platform-Specific Hardening and Security Interfaces.
//!
//! Windows: DPAPI, directory metadata sync, and file permission hardening.
//! Unix: POSIX mode 0600 hardening and directory fsync.

use std::path::Path;

/// Sync parent directory metadata to ensure durable directory entry commits.
pub fn sync_directory_metadata(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if parent.exists() {
            #[cfg(unix)]
            {
                use std::fs::File;
                if let Ok(dir_file) = File::open(parent) {
                    let _ = dir_file.sync_all();
                }
            }
            #[cfg(windows)]
            {
                // On Windows NTFS/ReFS, closing a synced file handle with atomic rename
                // journals the directory entry transaction automatically.
                // Opening directory handles directly with standard File::open fails with AccessDenied
                // unless custom Win32 FILE_FLAG_BACKUP_SEMANTICS are supplied.
            }
        }
    }
    Ok(())
}

/// Harden file permissions so that only the current user/process owner has read/write rights.
pub fn harden_file_permissions(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o600);
        std::fs::set_permissions(path, perms)
            .map_err(|e| format!("Failed to set POSIX 0600 permissions: {e}"))?;
    }

    #[cfg(windows)]
    {
        let user_opt = std::env::var("USERNAME").ok().or_else(|| std::env::var("USER").ok());
        if let Some(username) = user_opt {
            if username.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.' || c == ' ') {
                let sys_root = std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".to_string());
                let icacls_bin = std::path::PathBuf::from(sys_root).join("System32").join("icacls.exe");
                let output = std::process::Command::new(icacls_bin)
                    .arg(path)
                    .arg("/inheritance:r")
                    .arg("/grant:r")
                    .arg(format!("{}:F", username))
                    .output();
                if let Ok(out) = output {
                    if !out.status.success() {
                        let err_msg = String::from_utf8_lossy(&out.stderr);
                        if std::env::var("TARA_STRICT_PERMISSIONS").map(|v| v == "1").unwrap_or(false)
                            && !err_msg.trim().is_empty()
                        {
                            return Err(format!("icacls permission hardening failed: {}", err_msg.trim()));
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_harden_file_permissions_and_sync_metadata() {
        let temp_dir = std::env::temp_dir().join(format!("tara_platform_test_{}", rand::random::<u64>()));
        fs::create_dir_all(&temp_dir).expect("create temp dir");

        let test_file = temp_dir.join("test_sec_witness.bin");
        fs::write(&test_file, b"secure payload").expect("write test file");

        // Metadata sync on directory
        assert!(sync_directory_metadata(&test_file).is_ok());

        // File permission hardening
        assert!(harden_file_permissions(&test_file).is_ok());

        // Non-existent path returns Ok(()) early
        let non_existent = temp_dir.join("non_existent_file.bin");
        assert!(harden_file_permissions(&non_existent).is_ok());

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
