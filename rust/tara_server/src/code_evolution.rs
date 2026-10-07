//! Code Evolution Subsystem: Autonomous source code function addition and removal
//! for authorized Creator, protected by immutable security and authority boundaries.
//!
//! Enforces:
//! 1. Creator-only authorization (regular users strictly denied).
//! 2. Immutable security zone (security rules, crypto keys, creator authority cannot be altered).
//! 3. Anti-tampering and fail-closed syntax/integrity verification with automatic rollback.
//! 4. Audit logging of all code mutations.

use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodeEvolutionError {
    #[error("Unauthorized: only verified Creator can modify source code")]
    Unauthorized,

    #[error("Protected security zone: target file '{0}' is immutable")]
    ProtectedSecurityZone(String),

    #[error("Protected core symbol: '{0}' cannot be added, modified, or removed")]
    ProtectedCoreSymbol(String),

    #[error("Dangerous pattern detected: code contains disallowed security bypass pattern")]
    DangerousPatternDetected,

    #[error("File not found: '{0}'")]
    FileNotFound(String),

    #[error("Symbol not found: '{0}' in file '{1}'")]
    SymbolNotFound(String, String),

    #[error("Syntax or integrity check failed: '{0}'")]
    SyntaxVerificationFailed(String),

    #[error("IO error: {0}")]
    IoError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeEvolutionResult {
    pub status: String,
    pub action: String,
    pub file_path: String,
    pub function_name: String,
    pub lines_affected: usize,
    pub timestamp: String,
    pub message: String,
}

pub struct CodeEvolutionEngine {
    repo_root: PathBuf,
    audit_file: PathBuf,
}

impl CodeEvolutionEngine {
    pub fn new(repo_root: &str) -> Self {
        let repo_path = PathBuf::from(repo_root);
        let audit_dir = repo_path.join("storage").join("audit");
        let _ = fs::create_dir_all(&audit_dir);
        let audit_file = audit_dir.join("source_code_evolution.jsonl");

        Self {
            repo_root: repo_path,
            audit_file,
        }
    }

    /// Add a new function to a source file.
    pub fn add_source_function(
        &self,
        target_file_rel: &str,
        function_name: &str,
        function_code: &str,
        creator_verified: bool,
    ) -> Result<CodeEvolutionResult, CodeEvolutionError> {
        // 1. Authorization check: Creator only
        if !creator_verified {
            return Err(CodeEvolutionError::Unauthorized);
        }

        // 2. Protected security zone check
        let normalized_path = target_file_rel.replace('\\', "/");
        if Self::is_protected_file(&normalized_path) {
            return Err(CodeEvolutionError::ProtectedSecurityZone(
                target_file_rel.to_string(),
            ));
        }

        // 3. Protected symbol check
        if Self::is_protected_symbol(function_name) {
            return Err(CodeEvolutionError::ProtectedCoreSymbol(
                function_name.to_string(),
            ));
        }

        // 4. Anti-tampering / dangerous pattern check
        if Self::contains_dangerous_patterns(function_code) {
            return Err(CodeEvolutionError::DangerousPatternDetected);
        }

        // 5. Target file existence check
        let full_path = self.resolve_target(target_file_rel)?;

        // 6. Read existing content & create backup
        let original_content = fs::read_to_string(&full_path)
            .map_err(|e| CodeEvolutionError::IoError(e.to_string()))?;
        let backup_path = full_path.with_extension("bak_evolution");
        fs::write(&backup_path, &original_content)
            .map_err(|e| CodeEvolutionError::IoError(e.to_string()))?;

        // 7. Check if function already exists
        let fn_decl = format!("fn {}", function_name);
        let pub_fn_decl = format!("pub fn {}", function_name);
        if original_content.contains(&fn_decl) || original_content.contains(&pub_fn_decl) {
            let _ = fs::remove_file(&backup_path);
            return Err(CodeEvolutionError::SyntaxVerificationFailed(format!(
                "Function '{}' already exists in target file",
                function_name
            )));
        }

        // 8. Append formatted function
        let mut new_content = original_content.clone();
        if !new_content.ends_with('\n') {
            new_content.push('\n');
        }
        new_content.push_str(&format!("\n// [AUTONOMOUS_EVOLUTION: {}]\n", function_name));
        new_content.push_str(function_code.trim());
        new_content.push('\n');

        // 9. Basic brace/syntax verification
        if let Err(err) = Self::verify_basic_rust_syntax(&new_content) {
            // Rollback immediately
            let _ = fs::write(&full_path, &original_content);
            let _ = fs::remove_file(&backup_path);
            return Err(CodeEvolutionError::SyntaxVerificationFailed(err));
        }

        // 10. Write new content
        if let Err(e) = fs::write(&full_path, &new_content) {
            let _ = fs::write(&full_path, &original_content);
            let _ = fs::remove_file(&backup_path);
            return Err(CodeEvolutionError::IoError(e.to_string()));
        }

        // Clean up backup after successful write
        let _ = fs::remove_file(&backup_path);

        let lines_added = function_code.lines().count();
        let res = CodeEvolutionResult {
            status: "SUCCESS".to_string(),
            action: "ADD_FUNCTION".to_string(),
            file_path: target_file_rel.to_string(),
            function_name: function_name.to_string(),
            lines_affected: lines_added,
            timestamp: crate::now_iso(),
            message: format!(
                "Successfully added function '{}' to {}",
                function_name, target_file_rel
            ),
        };

        self.log_audit(&res);
        Ok(res)
    }

    /// Remove an unneeded or deprecated function from a source file.
    pub fn remove_source_function(
        &self,
        target_file_rel: &str,
        function_name: &str,
        creator_verified: bool,
    ) -> Result<CodeEvolutionResult, CodeEvolutionError> {
        // 1. Authorization check: Creator only
        if !creator_verified {
            return Err(CodeEvolutionError::Unauthorized);
        }

        // 2. Protected security zone check
        let normalized_path = target_file_rel.replace('\\', "/");
        if Self::is_protected_file(&normalized_path) {
            return Err(CodeEvolutionError::ProtectedSecurityZone(
                target_file_rel.to_string(),
            ));
        }

        // 3. Protected core symbol check (cannot remove core engine functions)
        if Self::is_protected_symbol(function_name) {
            return Err(CodeEvolutionError::ProtectedCoreSymbol(
                function_name.to_string(),
            ));
        }

        // 4. Target file check
        let full_path = self.resolve_target(target_file_rel)?;

        let original_content = fs::read_to_string(&full_path)
            .map_err(|e| CodeEvolutionError::IoError(e.to_string()))?;

        // 5. Locate function start
        let fn_patterns = [
            format!("pub fn {}<", function_name),
            format!("pub fn {}(", function_name),
            format!("pub fn {} ", function_name),
            format!("fn {}<", function_name),
            format!("fn {}(", function_name),
            format!("fn {} ", function_name),
        ];

        let mut start_idx = None;
        for pattern in &fn_patterns {
            if let Some(pos) = original_content.find(pattern) {
                start_idx = Some(pos);
                break;
            }
        }

        let start_pos = match start_idx {
            Some(pos) => pos,
            None => {
                return Err(CodeEvolutionError::SymbolNotFound(
                    function_name.to_string(),
                    target_file_rel.to_string(),
                ));
            }
        };

        // Also check if there was an autonomous evolution comment header directly above
        let mut actual_start = start_pos;
        let prefix = &original_content[..start_pos];
        let marker = format!("// [AUTONOMOUS_EVOLUTION: {}]", function_name);
        if let Some(marker_pos) = prefix.rfind(&marker) {
            actual_start = marker_pos;
        }

        // 6. Find end of function block by tracking balanced braces
        let rest = &original_content[start_pos..];
        let brace_start = match rest.find('{') {
            Some(p) => start_pos + p,
            None => {
                return Err(CodeEvolutionError::SyntaxVerificationFailed(
                    "Malformed function: no opening brace found".to_string(),
                ));
            }
        };

        let mut depth = 0;
        let mut end_pos = None;
        for (i, c) in original_content[brace_start..].char_indices() {
            if c == '{' {
                depth += 1;
            } else if c == '}' {
                depth -= 1;
                if depth == 0 {
                    end_pos = Some(brace_start + i + 1);
                    break;
                }
            }
        }

        let function_end = match end_pos {
            Some(p) => p,
            None => {
                return Err(CodeEvolutionError::SyntaxVerificationFailed(
                    "Malformed function: unbalanced braces".to_string(),
                ));
            }
        };

        // 7. Create backup before editing
        let backup_path = full_path.with_extension("bak_remove");
        fs::write(&backup_path, &original_content)
            .map_err(|e| CodeEvolutionError::IoError(e.to_string()))?;

        // 8. Reconstruct content without the removed function
        let mut new_content = String::new();
        new_content.push_str(&original_content[..actual_start]);
        new_content.push_str(&original_content[function_end..]);

        // 9. Verify syntax of the remaining code
        if let Err(err) = Self::verify_basic_rust_syntax(&new_content) {
            let _ = fs::remove_file(&backup_path);
            return Err(CodeEvolutionError::SyntaxVerificationFailed(err));
        }

        // 10. Write modified content
        if let Err(e) = fs::write(&full_path, &new_content) {
            let _ = fs::write(&full_path, &original_content);
            let _ = fs::remove_file(&backup_path);
            return Err(CodeEvolutionError::IoError(e.to_string()));
        }

        let _ = fs::remove_file(&backup_path);

        let lines_removed = original_content[actual_start..function_end].lines().count();
        let res = CodeEvolutionResult {
            status: "SUCCESS".to_string(),
            action: "REMOVE_FUNCTION".to_string(),
            file_path: target_file_rel.to_string(),
            function_name: function_name.to_string(),
            lines_affected: lines_removed,
            timestamp: crate::now_iso(),
            message: format!(
                "Successfully removed function '{}' from {}",
                function_name, target_file_rel
            ),
        };

        self.log_audit(&res);
        Ok(res)
    }

    /// Replace a previously evolved function using an exact source snippet.
    /// The exact-match requirement prevents ambiguous edits to unrelated code.
    pub fn update_source_function(
        &self,
        target_file_rel: &str,
        function_name: &str,
        previous_function_code: &str,
        replacement_function_code: &str,
        creator_verified: bool,
    ) -> Result<CodeEvolutionResult, CodeEvolutionError> {
        if !creator_verified {
            return Err(CodeEvolutionError::Unauthorized);
        }
        let normalized_path = target_file_rel.replace('\\', "/");
        if Self::is_protected_file(&normalized_path) {
            return Err(CodeEvolutionError::ProtectedSecurityZone(
                target_file_rel.to_string(),
            ));
        }
        if Self::is_protected_symbol(function_name) {
            return Err(CodeEvolutionError::ProtectedCoreSymbol(
                function_name.to_string(),
            ));
        }
        if Self::contains_dangerous_patterns(replacement_function_code) {
            return Err(CodeEvolutionError::DangerousPatternDetected);
        }
        if previous_function_code.trim().is_empty()
            || replacement_function_code.trim().is_empty()
            || !replacement_function_code.contains(&format!("fn {function_name}"))
        {
            return Err(CodeEvolutionError::SyntaxVerificationFailed(
                "replacement must be non-empty and declare the same function name".into(),
            ));
        }
        let full_path = self.resolve_target(target_file_rel)?;
        let original = fs::read_to_string(&full_path)
            .map_err(|error| CodeEvolutionError::IoError(error.to_string()))?;
        let marker = format!("// [AUTONOMOUS_EVOLUTION: {function_name}]");
        let code_position = original.find(previous_function_code);
        if original.matches(previous_function_code).count() != 1
            || code_position
                .is_none_or(|position| !original[..position].trim_end().ends_with(&marker))
        {
            return Err(CodeEvolutionError::SymbolNotFound(
                function_name.to_string(),
                target_file_rel.to_string(),
            ));
        }
        let updated =
            original.replacen(previous_function_code, replacement_function_code.trim(), 1);
        Self::verify_basic_rust_syntax(&updated)
            .map_err(CodeEvolutionError::SyntaxVerificationFailed)?;
        let backup_path = full_path.with_extension("bak_update");
        fs::write(&backup_path, &original)
            .map_err(|error| CodeEvolutionError::IoError(error.to_string()))?;
        if let Err(error) = fs::write(&full_path, &updated) {
            let _ = fs::write(&full_path, &original);
            let _ = fs::remove_file(&backup_path);
            return Err(CodeEvolutionError::IoError(error.to_string()));
        }
        fs::remove_file(&backup_path)
            .map_err(|error| CodeEvolutionError::IoError(error.to_string()))?;
        let result = CodeEvolutionResult {
            status: "SUCCESS".into(),
            action: "UPDATE_FUNCTION".into(),
            file_path: target_file_rel.to_string(),
            function_name: function_name.to_string(),
            lines_affected: previous_function_code
                .lines()
                .count()
                .max(replacement_function_code.lines().count()),
            timestamp: crate::now_iso(),
            message: format!(
                "Successfully updated function '{function_name}' in {target_file_rel}"
            ),
        };
        self.log_audit(&result);
        Ok(result)
    }

    /// Check if target file falls within the protected security/authority zone.
    pub fn is_protected_file(normalized_rel_path: &str) -> bool {
        let p = normalized_rel_path.to_lowercase();
        let protected_prefixes = [
            "rust/tara_server/src/rules",
            "rust/tara_server/src/access",
            "tara/rules",
            "tara/access",
            "storage/models",
        ];

        let protected_exact = [
            "rust/tara_server/src/guard.rs",
            "rust/tara_server/src/rules/mod.rs",
            "tara/rules/rulebook.txt",
            "tara/rules/compiled_policy.json",
        ];

        protected_prefixes
            .iter()
            .any(|&pref| p.starts_with(pref) || p.contains(pref))
            || protected_exact.iter().any(|&ex| p.ends_with(ex))
    }

    fn resolve_target(&self, target_file_rel: &str) -> Result<PathBuf, CodeEvolutionError> {
        let root = self
            .repo_root
            .canonicalize()
            .map_err(|e| CodeEvolutionError::IoError(e.to_string()))?;
        let candidate = root.join(target_file_rel);
        let full_path = candidate.canonicalize().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                CodeEvolutionError::FileNotFound(target_file_rel.to_string())
            } else {
                CodeEvolutionError::IoError(e.to_string())
            }
        })?;
        if !full_path.starts_with(&root) || !full_path.is_file() {
            return Err(CodeEvolutionError::FileNotFound(
                target_file_rel.to_string(),
            ));
        }
        let relative = full_path
            .strip_prefix(&root)
            .map_err(|e| CodeEvolutionError::IoError(e.to_string()))?
            .to_string_lossy()
            .replace('\\', "/");
        if Self::is_protected_file(&relative) {
            return Err(CodeEvolutionError::ProtectedSecurityZone(
                target_file_rel.to_string(),
            ));
        }
        Ok(full_path)
    }

    /// Check if a symbol belongs to core immutable authority/security.
    pub fn is_protected_symbol(name: &str) -> bool {
        let n = name.to_lowercase();
        let protected_symbols = [
            "evaluate_action",
            "verify_creator",
            "authenticate",
            "lockdown",
            "setup_creator",
            "verify_integrity",
            "verify_session",
            "verify_jwt",
            "execute_guard",
            "check_conversational_trigger",
            "is_protected_file",
            "is_protected_symbol",
        ];
        protected_symbols.iter().any(|&s| n == s || n.contains(s))
    }

    /// Check for malicious or dangerous bypass patterns.
    pub fn contains_dangerous_patterns(code: &str) -> bool {
        let c = code.to_lowercase();
        let patterns = [
            "bypass_security",
            "creator_override",
            "disable_rules",
            "creator_private_key",
            "override_authority",
            "disable_guard",
            "allow_all_unverified",
            "skip_creator_verification",
        ];
        patterns.iter().any(|&p| c.contains(p))
    }

    /// Parse the complete source file with rustfmt before writing it.
    fn verify_basic_rust_syntax(code: &str) -> Result<(), String> {
        let mut child = Command::new("rustfmt")
            .args(["--edition", "2021", "--emit", "stdout"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Could not start rustfmt parser: {e}"))?;
        child
            .stdin
            .as_mut()
            .ok_or_else(|| "rustfmt stdin unavailable".to_string())?
            .write_all(code.as_bytes())
            .map_err(|e| e.to_string())?;
        let output = child.wait_with_output().map_err(|e| e.to_string())?;
        if output.status.success() {
            Ok(())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
        }
    }

    fn log_audit(&self, res: &CodeEvolutionResult) {
        if let Ok(mut f) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.audit_file)
        {
            let log_line = json!({
                "timestamp": res.timestamp,
                "action": res.action,
                "file": res.file_path,
                "function": res.function_name,
                "lines_affected": res.lines_affected,
                "status": res.status,
            });
            let _ = writeln!(f, "{}", log_line);
        }
    }
}
