//! registry.rs
//!
//! TARA Capability Registry — ports Python's `registry.py`.
//! Provides a thread-safe singleton registry of named capabilities, each with an
//! optional callable handler, category, and description.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors emitted by the capability registry.
#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("Capability '{0}' already registered")]
    AlreadyExists(String),
    #[error("Capability '{0}' not found")]
    NotFound(String),
    #[error("Protected core capability '{0}' cannot be removed or altered")]
    ProtectedCoreCapability(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// CapabilityCategory
// ─────────────────────────────────────────────────────────────────────────────

/// Logical category of a registered capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilityCategory {
    Cognition,
    Memory,
    Reasoning,
    Tools,
    Skills,
    Learning,
    Robotics,
    Security,
    Engineering,
    Custom,
}

impl std::fmt::Display for CapabilityCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            CapabilityCategory::Cognition => "Cognition",
            CapabilityCategory::Memory => "Memory",
            CapabilityCategory::Reasoning => "Reasoning",
            CapabilityCategory::Tools => "Tools",
            CapabilityCategory::Skills => "Skills",
            CapabilityCategory::Learning => "Learning",
            CapabilityCategory::Robotics => "Robotics",
            CapabilityCategory::Security => "Security",
            CapabilityCategory::Engineering => "Engineering",
            CapabilityCategory::Custom => "Custom",
        };
        write!(f, "{}", s)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Capability
// ─────────────────────────────────────────────────────────────────────────────

/// A named, typed capability entry in the registry.
///
/// `handler` is an optional closure that accepts JSON parameters and returns a
/// JSON result. When `executable` is `false` the capability is informational only.
pub struct Capability {
    /// Unique string identifier.
    pub id: String,
    /// Broad category this capability belongs to.
    pub category: CapabilityCategory,
    /// Human-readable description.
    pub description: String,
    /// Whether this capability can be actively invoked.
    pub executable: bool,
    /// Optional callable handler; `None` for informational capabilities.
    pub handler: Option<Arc<dyn Fn(serde_json::Value) -> serde_json::Value + Send + Sync>>,
}

impl std::fmt::Debug for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Capability")
            .field("id", &self.id)
            .field("category", &self.category)
            .field("description", &self.description)
            .field("executable", &self.executable)
            .field("handler", &self.handler.is_some())
            .finish()
    }
}

impl Capability {
    /// Construct a new capability without a handler.
    pub fn new(
        id: impl Into<String>,
        category: CapabilityCategory,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            category,
            description: description.into(),
            executable: false,
            handler: None,
        }
    }

    /// Construct an executable capability with a handler closure.
    pub fn with_handler(
        id: impl Into<String>,
        category: CapabilityCategory,
        description: impl Into<String>,
        handler: impl Fn(serde_json::Value) -> serde_json::Value + Send + Sync + 'static,
    ) -> Self {
        Self {
            id: id.into(),
            category,
            description: description.into(),
            executable: true,
            handler: Some(Arc::new(handler)),
        }
    }

    /// Invoke the capability handler if present.
    pub fn invoke(&self, params: serde_json::Value) -> serde_json::Value {
        match &self.handler {
            Some(h) => h(params),
            None => serde_json::json!({
                "error": "no_handler",
                "capability": self.id,
                "message": "This capability has no executable handler."
            }),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CapabilityRegistry
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe registry of all TARA capabilities.
///
/// Use [`CapabilityRegistry::get_default`] to obtain the process-wide singleton.
pub struct CapabilityRegistry {
    inner: Mutex<HashMap<String, Capability>>,
}

static DEFAULT_REGISTRY: OnceLock<Arc<CapabilityRegistry>> = OnceLock::new();

impl CapabilityRegistry {
    /// Returns the process-wide singleton registry, creating it on first call.
    pub fn get_default() -> Arc<CapabilityRegistry> {
        DEFAULT_REGISTRY
            .get_or_init(|| {
                let reg = Arc::new(CapabilityRegistry {
                    inner: Mutex::new(HashMap::new()),
                });
                // Register built-in capabilities
                Self::register_builtins(&reg);
                reg
            })
            .clone()
    }

    fn register_builtins(reg: &Arc<CapabilityRegistry>) {
        let builtins: Vec<Capability> = vec![
            Capability::new(
                "cognition.respond",
                CapabilityCategory::Cognition,
                "Generate a natural-language response",
            ),
            Capability::new(
                "memory.recall",
                CapabilityCategory::Memory,
                "Retrieve episodic memory",
            ),
            Capability::new(
                "memory.record",
                CapabilityCategory::Memory,
                "Persist an episodic memory entry",
            ),
            Capability::new(
                "reasoning.chain_of_thought",
                CapabilityCategory::Reasoning,
                "Multi-step chain-of-thought reasoning",
            ),
            Capability::with_handler(
                "tools.file_inspector",
                CapabilityCategory::Tools,
                "Inspect file metadata and preview",
                crate::tools_registry::tool_file_inspector,
            ),
            Capability::with_handler(
                "tools.hash_verifier",
                CapabilityCategory::Tools,
                "Verify SHA-256 hash of a file",
                crate::tools_registry::tool_hash_verifier,
            ),
            Capability::with_handler(
                "skills.execute",
                CapabilityCategory::Skills,
                "Execute a named skill",
                |params| {
                    let skill_name = params
                        .get("skill")
                        .or_else(|| params.get("skill_name"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if skill_name.is_empty() {
                        serde_json::json!({ "status": "ERROR", "error": "skill name is required" })
                    } else {
                        let engine =
                            crate::skills::SkillEngine::new("storage/skills", "TARA/SKILLS");
                        engine.execute_skill(&skill_name, params)
                    }
                },
            ),
            Capability::new(
                "learning.search_and_learn",
                CapabilityCategory::Learning,
                "Search web and absorb knowledge",
            ),
            Capability::with_handler(
                "security.lockdown",
                CapabilityCategory::Security,
                "Engage security lockdown",
                |_params| {
                    serde_json::json!({
                        "status": "LOCKDOWN_ENGAGED",
                        "policy": "STRICT_RESTRICTED",
                        "message": "Emergency system security lockdown active"
                    })
                },
            ),
            Capability::new(
                "engineering.code_assist",
                CapabilityCategory::Engineering,
                "Assist with code generation and review",
            ),
        ];
        let mut guard = reg.inner.lock().expect("registry lock poisoned");
        for cap in builtins {
            guard.insert(cap.id.clone(), cap);
        }
    }

    /// Register a new capability. Returns an error if the ID already exists.
    pub fn register_capability(&self, cap: Capability) -> Result<(), RegistryError> {
        let mut guard = self.inner.lock().expect("registry lock poisoned");
        if guard.contains_key(&cap.id) {
            return Err(RegistryError::AlreadyExists(cap.id.clone()));
        }
        guard.insert(cap.id.clone(), cap);
        Ok(())
    }

    /// Register or overwrite a capability (upsert semantics).
    /// Core security and creator authority capabilities cannot be overwritten.
    pub fn upsert_capability(&self, cap: Capability) -> Result<(), RegistryError> {
        if Self::is_protected_capability(&cap.id) {
            return Err(RegistryError::ProtectedCoreCapability(cap.id.clone()));
        }
        let mut guard = self.inner.lock().expect("registry lock poisoned");
        guard.insert(cap.id.clone(), cap);
        Ok(())
    }

    /// Remove an unneeded capability from the registry.
    /// Core security capabilities and creator authority CANNOT be removed.
    pub fn remove_capability(&self, id: &str) -> Result<(), RegistryError> {
        if Self::is_protected_capability(id) {
            return Err(RegistryError::ProtectedCoreCapability(id.to_string()));
        }
        let mut guard = self.inner.lock().expect("registry lock poisoned");
        if guard.remove(id).is_none() {
            return Err(RegistryError::NotFound(id.to_string()));
        }
        Ok(())
    }

    /// Checks if a capability is a protected core security or authority component.
    pub fn is_protected_capability(id: &str) -> bool {
        let id_lower = id.to_lowercase();
        id_lower.starts_with("security.")
            || id_lower.contains("authority")
            || id_lower.contains("creator")
            || id_lower.contains("safety")
            || id_lower.contains("rules")
            || id_lower == "cognition.respond"
    }

    /// Retrieve a snapshot of a capability by its ID.
    pub fn get_capability(&self, id: &str) -> Option<CapabilitySnapshot> {
        let guard = self.inner.lock().expect("registry lock poisoned");
        guard.get(id).map(|c| CapabilitySnapshot {
            id: c.id.clone(),
            category: c.category.clone(),
            description: c.description.clone(),
            executable: c.executable,
        })
    }

    /// Invoke a capability by ID, passing JSON params.
    pub fn invoke(&self, id: &str, params: serde_json::Value) -> serde_json::Value {
        let guard = self.inner.lock().expect("registry lock poisoned");
        match guard.get(id) {
            Some(c) => c.invoke(params),
            None => serde_json::json!({
                "error": "not_found",
                "capability": id
            }),
        }
    }

    /// List all registered capability IDs.
    pub fn list_capabilities(&self) -> Vec<String> {
        let guard = self.inner.lock().expect("registry lock poisoned");
        let mut ids: Vec<String> = guard.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// List capabilities filtered by category.
    pub fn list_by_category(&self, category: &CapabilityCategory) -> Vec<CapabilitySnapshot> {
        let guard = self.inner.lock().expect("registry lock poisoned");
        let mut snaps: Vec<CapabilitySnapshot> = guard
            .values()
            .filter(|c| &c.category == category)
            .map(|c| CapabilitySnapshot {
                id: c.id.clone(),
                category: c.category.clone(),
                description: c.description.clone(),
                executable: c.executable,
            })
            .collect();
        snaps.sort_by(|a, b| a.id.cmp(&b.id));
        snaps
    }

    /// Serialize the full registry to a JSON value for inspection.
    pub fn to_json(&self) -> serde_json::Value {
        let guard = self.inner.lock().expect("registry lock poisoned");
        let entries: Vec<serde_json::Value> = guard
            .values()
            .map(|c| {
                serde_json::json!({
                    "id": c.id,
                    "category": c.category.to_string(),
                    "description": c.description,
                    "executable": c.executable,
                })
            })
            .collect();
        serde_json::json!({ "capabilities": entries, "count": entries.len() })
    }
}

/// Serialisable snapshot of a capability (no handler closure).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilitySnapshot {
    pub id: String,
    pub category: CapabilityCategory,
    pub description: String,
    pub executable: bool,
}
