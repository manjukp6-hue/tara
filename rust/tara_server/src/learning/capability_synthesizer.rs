//! Capability Synthesizer for TARA.
//!
//! Synthesizes new tool capabilities and skills discovered during research.
//! Strictly validates code and protects governance rules before deployment.
//!
//! The synthesized Rust source file is:
//!   1. Validated against rustfmt (parse-level syntax check).
//!   2. Written to the target path on disk.
//!   3. Compiled with `cargo check` from the workspace root to catch type errors
//!      and borrow-checker violations before the file is accepted.
//!
//! If any step fails the file is removed and an error is returned — no partial writes.

use crate::knowledge::GlobalKnowledgeBase;
use crate::learning::code_evolution::CodeEvolutionEngine;
use serde_json::{json, Value};
use std::fs;
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub struct CapabilitySynthesizer {
    pub repo_root: PathBuf,
}

impl CapabilitySynthesizer {
    pub fn new<P: AsRef<Path>>(repo_root: P) -> Self {
        Self {
            repo_root: repo_root.as_ref().to_path_buf(),
        }
    }

    pub fn synthesize_tool_capability(
        &self,
        kb: &GlobalKnowledgeBase,
        tool_name: &str,
        code_content: &str,
        description: &str,
        is_creator: bool,
    ) -> Result<Value, String> {
        if !is_creator {
            return Err("Unauthorized: Capability synthesis is restricted to ROOT_OPERATOR".into());
        }

        let clean_name = tool_name.trim().replace(['/', '\\', '.', ' ', '-'], "_");
        if clean_name.is_empty()
            || !clean_name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return Err(
                "tool_name contains invalid characters (only ASCII alphanumeric and '_' allowed)"
                    .into(),
            );
        }

        let tools_dir = self
            .repo_root
            .join("rust")
            .join("tara_server")
            .join("src")
            .join("skills");
        let target_path = tools_dir.join(format!("{}.rs", clean_name));

        // Path traversal guard: ensure the resolved path stays inside tools_dir.
        // canonicalize() resolves symlinks — if tools_dir doesn't exist yet we
        // use the pre-creation check (create_dir_all happens below, after this guard).
        let canonical_tools = tools_dir
            .canonicalize()
            .or_else(|_| {
                // tools_dir may not exist yet; construct absolute without canonicalize
                // using the repo_root which must exist.
                self.repo_root.canonicalize().map(|r| {
                    r.join("rust")
                        .join("tara_server")
                        .join("src")
                        .join("skills")
                })
            })
            .map_err(|e| format!("Cannot resolve skills directory: {e}"))?;
        let canonical_target = canonical_tools.join(format!("{}.rs", clean_name));
        if !canonical_target.starts_with(&canonical_tools) {
            return Err(format!(
                "Path traversal rejected: '{}' resolves outside skills directory",
                clean_name
            ));
        }

        // Reject dangerous security bypass patterns
        if CodeEvolutionEngine::contains_dangerous_patterns(code_content) {
            return Err(
                "Synthesized capability rejected: code contains disallowed dangerous patterns"
                    .into(),
            );
        }

        // Reject overwriting existing native skill files
        if target_path.exists() {
            return Err(format!(
                "File '{}' already exists. Use CodeEvolutionEngine to modify existing files.",
                target_path.display()
            ));
        }

        // Require at least one public function or struct definition
        if !code_content.contains("pub fn") && !code_content.contains("pub struct") {
            return Err(
                "Synthesized capability must define public Rust functions or structs".into(),
            );
        }

        // Stage 1: rustfmt parse validation (catches syntax errors)
        self.check_rustfmt_syntax(code_content)?;

        // Stage 2: Write to disk
        fs::create_dir_all(&tools_dir)
            .map_err(|e| format!("Cannot create skills directory: {e}"))?;
        fs::write(&target_path, code_content).map_err(|e| {
            format!(
                "Failed to write capability file '{}': {e}",
                target_path.display()
            )
        })?;

        // Stage 3: cargo check from workspace root to catch type and borrow errors.
        // On failure, remove the file to leave the workspace clean.
        if let Err(compile_err) = self.cargo_check_workspace() {
            let _ = fs::remove_file(&target_path);
            return Err(format!(
                "Capability '{}' rejected: cargo check failed after write. File removed.\n{}",
                clean_name, compile_err
            ));
        }

        // Record knowledge entry only after disk write + compile succeed
        let doc = format!("Synthesized capability: {} - {}", clean_name, description);
        let entry = kb.store_or_update_knowledge("CAPABILITIES", &clean_name, &doc, 0.90);

        Ok(json!({
            "status": "SYNTHESIZED",
            "capability_name": clean_name,
            "file_path": target_path.display().to_string(),
            "description": description,
            "knowledge_entry": entry,
            "timestamp": crate::now_iso()
        }))
    }

    /// Parse-check code via rustfmt. Catches malformed syntax before touching disk.
    fn check_rustfmt_syntax(&self, code: &str) -> Result<(), String> {
        let mut child = Command::new("rustfmt")
            .args(["--edition", "2021", "--emit", "stdout"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Could not start rustfmt: {e}"))?;
        child
            .stdin
            .as_mut()
            .ok_or("rustfmt stdin unavailable")?
            .write_all(code.as_bytes())
            .map_err(|e| e.to_string())?;
        let output = child.wait_with_output().map_err(|e| e.to_string())?;
        if output.status.success() {
            Ok(())
        } else {
            Err(format!(
                "Syntax error (rustfmt): {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ))
        }
    }

    /// Run `cargo check` in the workspace root so the Rust compiler validates
    /// the newly written file's types, imports, and borrow rules.
    fn cargo_check_workspace(&self) -> Result<(), String> {
        let output = Command::new("cargo")
            .args(["check", "-p", "tara_server", "--message-format", "short"])
            .current_dir(&self.repo_root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| format!("Could not run cargo check: {e}"))?;

        if output.status.success() {
            Ok(())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesize_tool_capability_validation() {
        let tmp_dir = std::env::temp_dir().join(format!("tara_synth_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp_dir);
        let synth = CapabilitySynthesizer::new(&tmp_dir);
        let kb = GlobalKnowledgeBase::new(&tmp_dir.to_string_lossy());

        // 1. Unauthorized
        let unauth =
            synth.synthesize_tool_capability(&kb, "my_tool", "pub fn run() {}", "desc", false);
        assert!(unauth.is_err());
        assert!(unauth.unwrap_err().contains("Unauthorized"));

        // 2. Invalid tool name characters
        let invalid_name = synth.synthesize_tool_capability(
            &kb,
            "bad$tool#name!",
            "pub fn run() {}",
            "desc",
            true,
        );
        assert!(invalid_name.is_err());
        assert!(invalid_name.unwrap_err().contains("invalid characters"));

        // 3. Dangerous pattern rejection
        let dangerous = synth.synthesize_tool_capability(
            &kb,
            "bypass_tool",
            "pub fn run() { let _ = bypass_security; }",
            "desc",
            true,
        );
        assert!(dangerous.is_err());
        assert!(dangerous
            .unwrap_err()
            .contains("disallowed dangerous patterns"));
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }
}
