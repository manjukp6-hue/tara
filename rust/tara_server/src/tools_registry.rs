//! tools_registry.rs
//!
//! TARA Tool Registry — ports Python's `tools_registry.py`.
//! Singleton registry of named tools with built-in implementations for
//! file inspection, hash verification, knowledge retrieval, and provenance tracking.

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};
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

/// Thread-safe singleton registry of callable tools.
pub struct ToolRegistry {
    tools: Mutex<HashMap<String, ToolEntry>>,
}

static DEFAULT_TOOL_REGISTRY: OnceLock<Arc<ToolRegistry>> = OnceLock::new();

impl ToolRegistry {
    /// Return (or initialise) the process-wide singleton.
    ///
    /// `repo_root` is used to anchor relative file paths in tool calls.
    pub fn get_default(repo_root: &str) -> Arc<ToolRegistry> {
        let repo_root = repo_root.to_string();
        DEFAULT_TOOL_REGISTRY
            .get_or_init(move || {
                let reg = Arc::new(ToolRegistry {
                    tools: Mutex::new(HashMap::new()),
                });
                Self::register_builtins(&reg, &repo_root);
                reg
            })
            .clone()
    }

    fn register_builtins(reg: &Arc<ToolRegistry>, _repo_root: &str) {
        let builtins: Vec<(&str, &str, ToolFn)> = vec![
            (
                "file_inspector",
                "Read file metadata and first 512 bytes",
                Arc::new(|_tool, params, _actor, _repo| tool_file_inspector(params)),
            ),
            (
                "hash_verifier",
                "Compute SHA-256 digest of a file",
                Arc::new(|_tool, params, _actor, _repo| tool_hash_verifier(params)),
            ),
            (
                "knowledge_retriever",
                "Return a knowledge lookup result for a query",
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

        let mut guard = reg.tools.lock().expect("tool registry lock poisoned");
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

    /// Register a custom tool. Returns error if already registered.
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
        let mut guard = self.tools.lock().expect("tool registry lock poisoned");
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

    /// Execute a named tool.
    ///
    /// Returns a JSON result regardless of success/failure — errors are embedded
    /// as `{"error": "...", "detail": "..."}` in the returned value.
    pub fn execute_tool(
        &self,
        tool_name: &str,
        params: serde_json::Value,
        actor_id: &str,
        repo_root: &str,
    ) -> serde_json::Value {
        let handler = {
            let guard = self.tools.lock().expect("tool registry lock poisoned");
            guard.get(tool_name).map(|e| e.handler.clone())
        };
        match handler {
            Some(h) => h(tool_name, params, actor_id, repo_root),
            None => serde_json::json!({
                "error": "tool_not_found",
                "tool": tool_name,
                "available": self.list_tools()
            }),
        }
    }

    /// List all registered tool names.
    pub fn list_tools(&self) -> Vec<String> {
        let guard = self.tools.lock().expect("tool registry lock poisoned");
        let mut names: Vec<String> = guard.keys().cloned().collect();
        names.sort();
        names
    }

    /// Describe all tools as a JSON array.
    pub fn describe_all(&self) -> serde_json::Value {
        let guard = self.tools.lock().expect("tool registry lock poisoned");
        let tools: Vec<serde_json::Value> = guard
            .values()
            .map(|e| serde_json::json!({"name": e.name, "description": e.description}))
            .collect();
        serde_json::json!({"tools": tools, "count": tools.len()})
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Built-in tool implementations
// ─────────────────────────────────────────────────────────────────────────────

/// `file_inspector` — reads file metadata and first 512 bytes.
pub fn tool_file_inspector(params: serde_json::Value) -> serde_json::Value {
    let path_str = match params.get("path").and_then(|v| v.as_str()) {
        Some(p) => p.to_string(),
        None => {
            return serde_json::json!({
                "error": "missing_parameter",
                "detail": "Required parameter 'path' not provided."
            })
        }
    };

    let path = Path::new(&path_str);
    match fs::metadata(path) {
        Err(e) => serde_json::json!({
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
                match fs::File::open(path) {
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

/// `hash_verifier` — computes SHA-256 of a file and optionally checks against expected hash.
pub fn tool_hash_verifier(params: serde_json::Value) -> serde_json::Value {
    let path_str = match params.get("path").and_then(|v| v.as_str()) {
        Some(p) => p.to_string(),
        None => {
            return serde_json::json!({
                "error": "missing_parameter",
                "detail": "Required parameter 'path' not provided."
            })
        }
    };

    let expected = params
        .get("expected_hash")
        .and_then(|v| v.as_str())
        .map(|s| s.to_lowercase());

    match fs::read(&path_str) {
        Err(e) => serde_json::json!({
            "error": "read_failed",
            "path": path_str,
            "detail": e.to_string()
        }),
        Ok(data) => {
            let mut hasher = Sha256::new();
            hasher.update(&data);
            let hash = hex::encode(hasher.finalize());

            let verified = expected.as_ref().map(|exp| exp == &hash).unwrap_or(false);

            serde_json::json!({
                "path": path_str,
                "algorithm": "sha256",
                "hash": hash,
                "expected_hash": expected,
                "verified": verified
            })
        }
    }
}

/// `knowledge_retriever` — searches persisted global knowledge entries.
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
        return serde_json::json!({"status":"ERROR","error":"A non-empty query is required"});
    }
    let knowledge_dir = Path::new(repo_root).join("storage").join("knowledge");
    let entries = match fs::read_dir(&knowledge_dir) {
        Ok(entries) => entries,
        Err(error) => {
            return serde_json::json!({"status":"ERROR","error":format!("Could not read knowledge store: {error}")})
        }
    };
    let needle = query.to_lowercase();
    let mut matches = Vec::new();
    for entry in entries.flatten() {
        if entry.path().extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let Ok(raw) = fs::read_to_string(entry.path()) else {
            continue;
        };
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
        "query": query,
        "actor_id": actor_id,
        "source": "knowledge_retriever",
        "status": "SUCCESS",
        "entries_found": matches.len(),
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
pub fn tool_math_engine(params: serde_json::Value) -> serde_json::Value {
    let engine = crate::engines::MathEngine::new();
    let op = params
        .get("operation")
        .or_else(|| params.get("op"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("add");
    match engine.evaluate(op, &params) {
        Ok(res) => serde_json::to_value(res).unwrap_or_default(),
        Err(e) => serde_json::json!({ "error": "math_engine_error", "detail": e }),
    }
}

/// `science_engine` — executes physics, chemistry, biology, and astronomy calculations.
pub fn tool_science_engine(params: serde_json::Value) -> serde_json::Value {
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
        Err(e) => serde_json::json!({ "error": "science_engine_error", "detail": e }),
    }
}

/// `programming_engine` — executes static code analysis, delimiter check, complexity lookup, and project verification.
pub fn tool_programming_engine(params: serde_json::Value, repo_root: &str) -> serde_json::Value {
    let engine = crate::engines::ProgrammingEngine::new(repo_root);
    let op = params
        .get("operation")
        .or_else(|| params.get("op"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("analyze_code");
    match engine.evaluate(op, &params) {
        Ok(res) => serde_json::to_value(res).unwrap_or_default(),
        Err(e) => serde_json::json!({ "error": "programming_engine_error", "detail": e }),
    }
}

/// `unit_converter` — converts physical units across 10 dimensions.
pub fn tool_unit_converter(params: serde_json::Value) -> serde_json::Value {
    match tara_engine::computation::ComputationEngine::execute("unit_convert", &params) {
        Ok(res) => res,
        Err(e) => serde_json::json!({ "error": "unit_converter_error", "detail": e }),
    }
}

/// `multilingual_terminology` — bilingual Kannada-English math and science technical terminology dictionary.
pub fn tool_multilingual_terminology(params: serde_json::Value) -> serde_json::Value {
    match tara_engine::computation::ComputationEngine::execute("multilingual_lookup", &params) {
        Ok(res) => res,
        Err(e) => serde_json::json!({ "error": "multilingual_lookup_error", "detail": e }),
    }
}
