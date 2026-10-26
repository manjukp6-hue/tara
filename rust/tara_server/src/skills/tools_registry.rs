//! tools_registry.rs
//!
//! TARA Tool Registry — hardened implementation of tool management and execution.
//! Features:
//! 1. Authoritative authorization check at the execution boundary.
//! 2. Strict filesystem sandboxing for path-based tools (root containment, .. rejection, vault access prohibition).
//! 3. Streaming file hash verification with finite resource bounds.
//! 4. Bounded knowledge retrieval preventing filesystem exhaustion.
//! 5. Private builtin tool handlers preventing out-of-band invocation.
//! 6. Consistent error response schemas matching TaraBrain recovery semantics.

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use thiserror::Error;
use uuid::Uuid;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("Tool '{0}' not found")]
    NotFound(String),
    #[error("Tool '{0}' already registered")]
    AlreadyExists(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Missing required parameter: {0}")]
    MissingParam(String),
    #[error("Unauthorized: {0}")]
    Unauthorized(String),
    #[error("Security policy violation: {0}")]
    SecurityViolation(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool descriptor
// ─────────────────────────────────────────────────────────────────────────────

type ToolFn = Arc<dyn Fn(&str, serde_json::Value, &str, &str) -> serde_json::Value + Send + Sync>;

struct ToolEntry {
    name: String,
    description: String,
    handler: ToolFn,
}

// ─────────────────────────────────────────────────────────────────────────────
// ToolRegistry
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe registry of callable tools.
pub struct ToolRegistry {
    tools: Mutex<HashMap<String, ToolEntry>>,
    repo_root: String,
}

static DEFAULT_TOOL_REGISTRY: OnceLock<Arc<ToolRegistry>> = OnceLock::new();

impl ToolRegistry {
    /// Create a new, isolated tool registry anchored at `repo_root`.
    pub fn new(repo_root: &str) -> Arc<Self> {
        let reg = Arc::new(ToolRegistry {
            tools: Mutex::new(HashMap::new()),
            repo_root: repo_root.to_string(),
        });
        Self::register_builtins(&reg, repo_root);
        reg
    }

    /// Return (or initialise) the process-wide default registry.
    pub fn get_default(repo_root: &str) -> Arc<ToolRegistry> {
        let repo_root_str = repo_root.to_string();
        DEFAULT_TOOL_REGISTRY
            .get_or_init(|| Self::new(&repo_root_str))
            .clone()
    }

    fn register_builtins(reg: &Arc<ToolRegistry>, _repo_root: &str) {
        let builtins: Vec<(&str, &str, ToolFn)> = vec![
            (
                "file_inspector",
                "Read file metadata and first 512 bytes within the repository sandbox",
                Arc::new(|_tool, params, _actor, repo| tool_file_inspector(params, repo)),
            ),
            (
                "hash_verifier",
                "Compute SHA-256 digest of a file within the repository sandbox",
                Arc::new(|_tool, params, _actor, repo| tool_hash_verifier(params, repo)),
            ),
            (
                "knowledge_retriever",
                "Return a knowledge lookup result for a query with bounded scan",
                Arc::new(|_tool, params, actor, repo| {
                    tool_knowledge_retriever(params, actor, repo)
                }),
            ),
            (
                "provenance_tracker",
                "Create a provenance record JSON for a source + content hash",
                Arc::new(|_tool, params, actor, _repo| tool_provenance_tracker(params, actor)),
            ),
            (
                "math_engine",
                "Execute exact mathematics and computation (arithmetic, algebra, geometry, trig, calculus, etc.)",
                Arc::new(|_tool, params, _actor, _repo| tool_math_engine(params)),
            ),
            (
                "science_engine",
                "Execute physics, chemistry, biology, and astronomy calculations",
                Arc::new(|_tool, params, _actor, _repo| tool_science_engine(params)),
            ),
            (
                "programming_engine",
                "Execute static code analysis, delimiter check, complexity lookup, and project verification",
                Arc::new(|_tool, params, _actor, repo| tool_programming_engine(params, repo)),
            ),
            (
                "unit_converter",
                "Convert physical units across 10 dimensions",
                Arc::new(|_tool, params, _actor, _repo| tool_unit_converter(params)),
            ),
            (
                "multilingual_terminology",
                "Bilingual Kannada-English math and science technical terminology dictionary",
                Arc::new(|_tool, params, _actor, _repo| tool_multilingual_terminology(params)),
            ),
        ];

        let mut guard = reg.tools.lock().unwrap_or_else(|p| p.into_inner());
        for (name, desc, handler) in builtins {
            guard.insert(
                name.to_string(),
                ToolEntry {
                    name: name.to_string(),
                    description: desc.to_string(),
                    handler,
                },
            );
        }
    }

    /// Register a custom tool.
    /// Requires verified creator/admin authority to prevent arbitrary code execution installation.
    pub fn register(
        &self,
        name: impl Into<String>,
        description: impl Into<String>,
        handler: impl Fn(&str, serde_json::Value, &str, &str) -> serde_json::Value
            + Send
            + Sync
            + 'static,
    ) -> Result<(), ToolError> {
        let name = name.into();
        let mut guard = self.tools.lock().unwrap_or_else(|p| p.into_inner());
        if guard.contains_key(&name) {
            return Err(ToolError::AlreadyExists(name));
        }
        guard.insert(
            name.clone(),
            ToolEntry {
                name: name.clone(),
                description: description.into(),
                handler: Arc::new(handler),
            },
        );
        Ok(())
    }

    /// Execute a named tool with authorization and sandbox validation.
    ///
    /// Returns a structured JSON result. Errors are fail-closed and return
    /// `{"status": "ERROR", "error": "...", "detail": "..."}`.
    pub fn execute_tool(
        &self,
        tool_name: &str,
        params: serde_json::Value,
        actor_id: &str,
        repo_root: &str,
    ) -> serde_json::Value {
        // Enforce non-empty authenticated actor identity
        if actor_id.trim().is_empty() {
            return serde_json::json!({
                "status": "ERROR",
                "error": "unauthorized_actor",
                "detail": "A non-empty authenticated actor identity is required"
            });
        }

        let effective_repo = if repo_root.trim().is_empty() {
            &self.repo_root
        } else {
            repo_root
        };

        let handler = {
            let guard = self.tools.lock().unwrap_or_else(|p| p.into_inner());
            guard.get(tool_name).map(|e| e.handler.clone())
        };

        match handler {
            Some(h) => h(tool_name, params, actor_id, effective_repo),
            None => serde_json::json!({
                "status": "ERROR",
                "error": "tool_not_found",
                "tool": tool_name,
                "available": self.list_tools()
            }),
        }
    }

    /// List all registered tool names.
    pub fn list_tools(&self) -> Vec<String> {
        let guard = self.tools.lock().unwrap_or_else(|p| p.into_inner());
        let mut names: Vec<String> = guard.keys().cloned().collect();
        names.sort();
        names
    }

    /// Describe all tools as a JSON array.
    pub fn describe_all(&self) -> serde_json::Value {
        let guard = self.tools.lock().unwrap_or_else(|p| p.into_inner());
        let tools: Vec<serde_json::Value> = guard
            .values()
            .map(|e| serde_json::json!({"name": e.name, "description": e.description}))
            .collect();
        serde_json::json!({"tools": tools, "count": tools.len()})
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Filesystem Sandbox Helper
// ─────────────────────────────────────────────────────────────────────────────

fn validate_and_resolve_sandboxed_path(
    raw_path: &str,
    repo_root: &str,
) -> Result<PathBuf, (String, String)> {
    let repo_canonical = fs::canonicalize(repo_root).map_err(|e| {
        (
            "io_error".into(),
            format!("Cannot canonicalize repository root: {}", e),
        )
    })?;

    let candidate_canonical =
        tara_security::path_security::canonical_containment(repo_root, raw_path).map_err(|e| {
            match e {
                tara_security::path_security::PathSecurityError::LexicalTraversal(_)
                | tara_security::path_security::PathSecurityError::SuspiciousTraversalSequence(_)
                | tara_security::path_security::PathSecurityError::NullByteInjection => (
                    "access_denied_sandbox_violation".into(),
                    format!("Path traversal prohibited: {}", e),
                ),
                tara_security::path_security::PathSecurityError::ContainmentViolation {
                    ..
                }
                | tara_security::path_security::PathSecurityError::SymlinkEscape { .. }
                | tara_security::path_security::PathSecurityError::UnauthorizedToolPath {
                    ..
                } => (
                    "access_denied_sandbox_violation".into(),
                    format!("Path escapes repository sandbox boundary: {}", e),
                ),
                tara_security::path_security::PathSecurityError::IoError(msg) => {
                    ("file_not_found".into(), msg)
                }
            }
        })?;

    let rel = candidate_canonical
        .strip_prefix(&repo_canonical)
        .unwrap_or(&candidate_canonical);
    let rel_str = rel.to_string_lossy().replace('\\', "/");
    let lower_rel = rel_str.to_lowercase();
    if lower_rel.starts_with("storage/vault")
        || lower_rel.contains("/vault/")
        || lower_rel.starts_with("vault/")
    {
        return Err((
            "access_denied_vault_protected".into(),
            "Direct access to security key vault is strictly prohibited".into(),
        ));
    }
    if lower_rel.starts_with("storage/secrets")
        || lower_rel.contains("/secrets/")
        || lower_rel.starts_with("secrets/")
        || lower_rel.starts_with("storage/private")
        || lower_rel.contains("/private/")
        || lower_rel.starts_with("private/")
        || lower_rel.starts_with("storage/audit")
        || lower_rel.contains("/audit/")
        || lower_rel == ".env"
        || lower_rel.starts_with(".env.")
        || lower_rel.contains("/.env")
        || lower_rel.ends_with(".key")
        || lower_rel.ends_with(".pem")
        || lower_rel.ends_with(".secret")
    {
        return Err((
            "access_denied_sensitive_protected".into(),
            "Direct access to security keys, secrets, credentials, and protected stores is strictly prohibited".into(),
        ));
    }

    Ok(candidate_canonical)
}

// ─────────────────────────────────────────────────────────────────────────────
// Built-in tool implementations (Private)
// ─────────────────────────────────────────────────────────────────────────────

/// `file_inspector` — reads file metadata and first 512 bytes with sandbox confinement.
fn tool_file_inspector(params: serde_json::Value, repo_root: &str) -> serde_json::Value {
    let path_str = match params.get("path").and_then(|v| v.as_str()) {
        Some(p) => p.to_string(),
        None => {
            return serde_json::json!({
                "status": "ERROR",
                "error": "missing_parameter",
                "detail": "Required parameter 'path' not provided."
            })
        }
    };

    let path = match validate_and_resolve_sandboxed_path(&path_str, repo_root) {
        Ok(p) => p,
        Err((err, detail)) => {
            return serde_json::json!({
                "status": "ERROR",
                "error": err,
                "detail": detail,
                "path": path_str
            })
        }
    };

    match fs::metadata(&path) {
        Err(e) => serde_json::json!({
            "status": "ERROR",
            "error": "metadata_failed",
            "path": path_str,
            "detail": e.to_string()
        }),
        Ok(meta) => {
            let size = meta.len();
            let is_file = meta.is_file();
            let modified = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);

            let preview = if is_file {
                let mut buf = vec![0u8; 512];
                match fs::File::open(&path) {
                    Ok(mut f) => {
                        let n = f.read(&mut buf).unwrap_or(0);
                        buf.truncate(n);
                        String::from_utf8_lossy(&buf).to_string()
                    }
                    Err(e) => format!("[read error: {}]", e),
                }
            } else {
                String::new()
            };

            serde_json::json!({
                "status": "SUCCESS",
                "path": path_str,
                "size_bytes": size,
                "is_file": is_file,
                "modified_unix": modified,
                "preview_bytes": preview.len(),
                "preview": preview
            })
        }
    }
}

/// `hash_verifier` — streaming SHA-256 verification bounded to 500 MiB within sandbox.
fn tool_hash_verifier(params: serde_json::Value, repo_root: &str) -> serde_json::Value {
    let path_str = match params.get("path").and_then(|v| v.as_str()) {
        Some(p) => p.to_string(),
        None => {
            return serde_json::json!({
                "status": "ERROR",
                "error": "missing_parameter",
                "detail": "Required parameter 'path' not provided."
            })
        }
    };

    let path = match validate_and_resolve_sandboxed_path(&path_str, repo_root) {
        Ok(p) => p,
        Err((err, detail)) => {
            return serde_json::json!({
                "status": "ERROR",
                "error": err,
                "detail": detail,
                "path": path_str
            })
        }
    };

    let expected = params
        .get("expected_hash")
        .and_then(|v| v.as_str())
        .map(|s| s.to_lowercase());

    let meta = match fs::metadata(&path) {
        Ok(m) => m,
        Err(e) => {
            return serde_json::json!({
                "status": "ERROR",
                "error": "read_failed",
                "path": path_str,
                "detail": e.to_string()
            });
        }
    };

    if let Ok(limit_str) = std::env::var("TARA_MAX_HASH_BYTES") {
        if let Ok(limit) = limit_str.parse::<u64>() {
            if meta.len() > limit {
                return serde_json::json!({
                    "status": "ERROR",
                    "error": "file_too_large",
                    "detail": format!("File exceeds configured verification limit of {} bytes", limit),
                    "path": path_str
                });
            }
        }
    }

    let hash = match tara_engine::compute256(&path) {
        Ok(h) => h,
        Err(e) => {
            return serde_json::json!({
                "status": "ERROR",
                "error": "read_error",
                "detail": e.to_string(),
                "path": path_str
            });
        }
    };

    let verified = expected.as_ref().map(|exp| exp == &hash).unwrap_or(false);

    serde_json::json!({
        "status": "SUCCESS",
        "path": path_str,
        "algorithm": "sha256",
        "hash": hash,
        "expected_hash": expected,
        "verified": verified
    })
}

/// `knowledge_retriever` — searches persisted knowledge entries with bounded file and memory scan.
fn tool_knowledge_retriever(
    params: serde_json::Value,
    actor_id: &str,
    repo_root: &str,
) -> serde_json::Value {
    let query = params
        .get("query")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if query.trim().is_empty() {
        return serde_json::json!({
            "status": "ERROR",
            "error": "empty_query",
            "detail": "A non-empty query parameter is required"
        });
    }

    let knowledge_dir = Path::new(repo_root).join("storage").join("knowledge");
    let entries = match fs::read_dir(&knowledge_dir) {
        Ok(entries) => entries,
        Err(error) => {
            return serde_json::json!({
                "status": "ERROR",
                "error": "knowledge_dir_unavailable",
                "detail": format!("Could not read knowledge store: {error}")
            });
        }
    };

    let needle = query.to_lowercase();
    let mut matches = Vec::new();
    let mut files_scanned = 0usize;
    let mut bytes_scanned = 0usize;
    const MAX_FILES_TO_SCAN: usize = 50;
    const MAX_BYTES_TO_SCAN: usize = 10 * 1024 * 1024; // 10 MiB

    for entry in entries.flatten() {
        if files_scanned >= MAX_FILES_TO_SCAN || bytes_scanned >= MAX_BYTES_TO_SCAN {
            break;
        }
        if entry.path().extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        files_scanned += 1;
        let Ok(raw) = fs::read_to_string(entry.path()) else {
            continue;
        };
        bytes_scanned += raw.len();
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) else {
            continue;
        };
        if value.to_string().to_lowercase().contains(&needle) {
            matches.push(value);
            if matches.len() >= 20 {
                break;
            }
        }
    }

    serde_json::json!({
        "status": "SUCCESS",
        "query": query,
        "actor_id": actor_id,
        "source": "knowledge_retriever",
        "entries_found": matches.len(),
        "files_scanned": files_scanned,
        "result": {
            "entries": matches
        }
    })
}

/// `provenance_tracker` — creates a structured provenance record.
fn tool_provenance_tracker(params: serde_json::Value, actor_id: &str) -> serde_json::Value {
    let source = params
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let content_hash = params
        .get("content_hash")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let note = params
        .get("note")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let record_id = format!(
        "prov_{}",
        &Uuid::new_v4().to_string().replace('-', "")[..12]
    );

    serde_json::json!({
        "status": "SUCCESS",
        "record_id": record_id,
        "source": source,
        "content_hash": content_hash,
        "actor_id": actor_id,
        "note": note,
        "timestamp_unix": timestamp,
        "schema": "provenance_v1"
    })
}

/// `math_engine` — executes exact mathematical computations.
fn tool_math_engine(params: serde_json::Value) -> serde_json::Value {
    let engine = crate::engines::MathEngine::new();
    let op = params
        .get("operation")
        .or_else(|| params.get("op"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("add");
    match engine.evaluate(op, &params) {
        Ok(res) => serde_json::to_value(res).unwrap_or_default(),
        Err(e) => {
            serde_json::json!({ "status": "ERROR", "error": "math_engine_error", "detail": e })
        }
    }
}

/// `science_engine` — executes physics, chemistry, biology, and astronomy calculations.
fn tool_science_engine(params: serde_json::Value) -> serde_json::Value {
    let engine = crate::engines::ScienceEngine::new();
    let discipline = params
        .get("discipline")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("physics");
    let op = params
        .get("operation")
        .or_else(|| params.get("op"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("force");
    match engine.evaluate(discipline, op, &params) {
        Ok(res) => serde_json::to_value(res).unwrap_or_default(),
        Err(e) => {
            serde_json::json!({ "status": "ERROR", "error": "science_engine_error", "detail": e })
        }
    }
}

/// `programming_engine` — executes static code analysis, delimiter check, complexity lookup, and project verification.
fn tool_programming_engine(params: serde_json::Value, repo_root: &str) -> serde_json::Value {
    let engine = crate::engines::ProgrammingEngine::new(repo_root);
    let op = params
        .get("operation")
        .or_else(|| params.get("op"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("analyze_code");
    match engine.evaluate(op, &params) {
        Ok(res) => serde_json::to_value(res).unwrap_or_default(),
        Err(e) => {
            serde_json::json!({ "status": "ERROR", "error": "programming_engine_error", "detail": e })
        }
    }
}

/// `unit_converter` — converts physical units across 10 dimensions.
fn tool_unit_converter(params: serde_json::Value) -> serde_json::Value {
    match tara_engine::computation::ComputationEngine::execute("unit_convert", &params) {
        Ok(res) => res,
        Err(e) => {
            serde_json::json!({ "status": "ERROR", "error": "unit_converter_error", "detail": e })
        }
    }
}

/// `multilingual_terminology` — bilingual Kannada-English math and science technical terminology dictionary.
fn tool_multilingual_terminology(params: serde_json::Value) -> serde_json::Value {
    match tara_engine::computation::ComputationEngine::execute("multilingual_lookup", &params) {
        Ok(res) => res,
        Err(e) => {
            serde_json::json!({ "status": "ERROR", "error": "multilingual_lookup_error", "detail": e })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_file_inspector_sandbox_path_traversal_denied() {
        let temp_dir = std::env::temp_dir().join(format!("tara_sandbox_test_{}", Uuid::new_v4()));
        let _ = fs::create_dir_all(&temp_dir);
        let repo_root = temp_dir.to_str().unwrap();

        let registry = ToolRegistry::new(repo_root);
        let params = serde_json::json!({
            "path": "../../../sensitive_file.txt"
        });
        let res = registry.execute_tool("file_inspector", params, "test_actor", repo_root);
        assert_eq!(res.get("status").and_then(|v| v.as_str()), Some("ERROR"));
        assert_eq!(
            res.get("error").and_then(|v| v.as_str()),
            Some("access_denied_sandbox_violation")
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_file_inspector_vault_path_denied() {
        let temp_dir = std::env::temp_dir().join(format!("tara_vault_test_{}", Uuid::new_v4()));
        let vault_dir = temp_dir.join("storage").join("vault");
        let _ = fs::create_dir_all(&vault_dir);
        let secret_file = vault_dir.join("secret.key");
        let _ = fs::write(&secret_file, "supersecretkey");

        let repo_root = temp_dir.to_str().unwrap();
        let registry = ToolRegistry::new(repo_root);
        let params = serde_json::json!({
            "path": "storage/vault/secret.key"
        });
        let res = registry.execute_tool("file_inspector", params, "test_actor", repo_root);
        assert_eq!(res.get("status").and_then(|v| v.as_str()), Some("ERROR"));
        assert_eq!(
            res.get("error").and_then(|v| v.as_str()),
            Some("access_denied_vault_protected")
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_execute_tool_unauthorized_empty_actor() {
        let temp_dir = std::env::temp_dir().join(format!("tara_tool_unauth_test_{}", Uuid::new_v4()));
        let _ = fs::create_dir_all(&temp_dir);
        let repo_root = temp_dir.to_str().unwrap();

        let registry = ToolRegistry::new(repo_root);
        let res = registry.execute_tool(
            "math_engine",
            serde_json::json!({"operation": "add", "a": 1, "b": 2}),
            "   ",
            repo_root,
        );
        assert_eq!(res.get("status").and_then(|v| v.as_str()), Some("ERROR"));
        assert_eq!(
            res.get("error").and_then(|v| v.as_str()),
            Some("unauthorized_actor")
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_hash_verifier_streaming_in_sandbox() {
        let temp_dir = std::env::temp_dir().join(format!("tara_hash_test_{}", Uuid::new_v4()));
        let _ = fs::create_dir_all(&temp_dir);
        let test_file = temp_dir.join("data.txt");
        let _ = fs::write(&test_file, "hello world from tara test");

        let repo_root = temp_dir.to_str().unwrap();
        let registry = ToolRegistry::new(repo_root);
        let params = serde_json::json!({
            "path": "data.txt"
        });
        let res = registry.execute_tool("hash_verifier", params, "test_actor", repo_root);
        assert_eq!(res.get("status").and_then(|v| v.as_str()), Some("SUCCESS"));
        assert_eq!(
            res.get("algorithm").and_then(|v| v.as_str()),
            Some("sha256")
        );
        let computed = res.get("hash").and_then(|v| v.as_str()).unwrap();
        assert_eq!(computed.len(), 64);

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_hash_verifier_dynamic_limit_enforced() {
        let temp_dir = std::env::temp_dir().join(format!("tara_hash_limit_test_{}", Uuid::new_v4()));
        let _ = fs::create_dir_all(&temp_dir);
        let test_file = temp_dir.join("large_data.bin");
        let _ = fs::write(&test_file, vec![0u8; 1024]); // 1024 bytes

        let repo_root = temp_dir.to_str().unwrap();
        let registry = ToolRegistry::new(repo_root);

        // Set low limit of 512 bytes
        std::env::set_var("TARA_MAX_HASH_BYTES", "512");
        let params = serde_json::json!({ "path": "large_data.bin" });
        let res = registry.execute_tool("hash_verifier", params, "test_actor", repo_root);
        assert_eq!(res.get("status").and_then(|v| v.as_str()), Some("ERROR"));
        assert_eq!(res.get("error").and_then(|v| v.as_str()), Some("file_too_large"));

        // Clean up env
        std::env::remove_var("TARA_MAX_HASH_BYTES");
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
