//! Additional server subsystems: user model, language registry, plugin engine, knowledge ontology, resilience.

// ── User Model ─────────────────────────────────────────────────────────────────
pub mod user_model {
    use serde_json::{json, Value};
    use std::collections::HashMap;
    use std::fs;

    pub struct UserProfile {
        pub user_id: String,
        pub display_name: String,
        pub preferences: HashMap<String, String>,
        pub interaction_count: u64,
    }

    impl UserProfile {
        pub fn to_json(&self) -> Value {
            json!({
                "user_id": self.user_id,
                "display_name": self.display_name,
                "preferences": self.preferences,
                "interaction_count": self.interaction_count,
            })
        }
    }

    pub struct UserManager {
        pub base_dir: String,
    }
    impl UserManager {
        pub fn new(base_dir: &str) -> std::sync::Arc<Self> {
            let _ = fs::create_dir_all(base_dir);
            std::sync::Arc::new(Self {
                base_dir: base_dir.to_string(),
            })
        }
        pub fn get_or_create_profile(&self, user_id: &str) -> UserProfile {
            let path = format!("{}/{}.json", self.base_dir, user_id);
            if let Ok(raw) = fs::read_to_string(&path) {
                if let Ok(v) = serde_json::from_str::<Value>(&raw) {
                    return UserProfile {
                        user_id: user_id.to_string(),
                        display_name: v
                            .get("display_name")
                            .and_then(|v| v.as_str())
                            .unwrap_or(user_id)
                            .to_string(),
                        preferences: HashMap::new(),
                        interaction_count: v
                            .get("interaction_count")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0),
                    };
                }
            }
            UserProfile {
                user_id: user_id.to_string(),
                display_name: user_id.to_string(),
                preferences: HashMap::new(),
                interaction_count: 0,
            }
        }
        pub fn update_preference(
            &self,
            user_id: &str,
            key: &str,
            value: &str,
        ) -> Result<(), String> {
            let path = format!("{}/{}.json", self.base_dir, user_id);
            let mut v = fs::read_to_string(&path)
                .ok()
                .and_then(|r| serde_json::from_str::<Value>(&r).ok())
                .unwrap_or(json!({ "user_id": user_id }));
            if let Some(obj) = v.get_mut("preferences").and_then(|p| p.as_object_mut()) {
                obj.insert(key.to_string(), json!(value));
            } else {
                v["preferences"] = json!({ key: value });
            }
            fs::write(&path, serde_json::to_string_pretty(&v).unwrap_or_default())
                .map_err(|e| e.to_string())
        }
    }
}

// ── Agent Orchestrator ────────────────────────────────────────────────────────
pub mod agent_orchestrator {
    use serde_json::{json, Value};
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use tara_core::{ManagerConfig, TaraManager};

    pub struct AgentTask {
        pub agent_id: String,
        pub role: String,
        pub objective: String,
        pub status: String,
        pub output: Option<Value>,
    }
    pub struct AgentOrchestrator {
        tasks: Mutex<Vec<AgentTask>>,
        manager: Mutex<TaraManager>,
    }
    impl AgentOrchestrator {
        pub fn new(repo_root: &str) -> Arc<Self> {
            let manager = TaraManager::new(
                ManagerConfig {
                    manager_id: "tara-runtime-manager".to_string(),
                    display_name: "TARA Runtime Manager".to_string(),
                    storage_base_dir: PathBuf::from(repo_root).join("storage/runtime"),
                    is_supervisor: true,
                    max_concurrent_sandboxes: 0,
                },
                None,
            );
            Arc::new(Self {
                tasks: Mutex::new(Vec::new()),
                manager: Mutex::new(manager),
            })
        }
        pub fn create_agent(&self, role: &str, objective: &str) -> Value {
            let mut manager = match self.manager.lock() {
                Ok(manager) => manager,
                Err(_) => {
                    return json!({ "status": "ERROR", "error": "Agent manager lock poisoned" })
                }
            };
            let capabilities = Vec::new();
            let (agent_id, display_name) = match manager.create_agent(role, &capabilities, None) {
                Ok(identity) => identity,
                Err(error) => return json!({ "status": "ERROR", "error": error }),
            };
            drop(manager);
            self.tasks.lock().unwrap().push(AgentTask {
                agent_id: agent_id.clone(),
                role: role.to_string(),
                objective: objective.to_string(),
                status: "PENDING".to_string(),
                output: None,
            });
            json!({ "agent_id": agent_id, "display_name": display_name, "role": role, "objective": objective, "status": "IDLE", "runtime_backed": true })
        }
        pub fn list_agents(&self) -> Vec<Value> {
            self.tasks
                .lock()
                .unwrap()
                .iter()
                .map(|t| json!({ "agent_id": t.agent_id, "role": t.role, "status": t.status }))
                .collect()
        }
    }
}

// ── Language Registry ─────────────────────────────────────────────────────────
pub mod language_registry {
    use serde_json::{json, Value};
    use std::sync::Arc;

    /// BMP Unicode block ranges → (language code, language name).
    /// Detection priority: first match wins (ordered most-distinctive first).
    static SCRIPT_RANGES: &[(u32, u32, &str, &str)] = &[
        // Devanagari — Hindi, Marathi, Sanskrit, Nepali
        (0x0900, 0x097F, "hi", "Hindi"),
        // Arabic — Arabic, Urdu, Persian, Pashto
        (0x0600, 0x06FF, "ar", "Arabic"),
        // Hebrew
        (0x0590, 0x05FF, "he", "Hebrew"),
        // Bengali
        (0x0980, 0x09FF, "bn", "Bengali"),
        // Tamil
        (0x0B80, 0x0BFF, "ta", "Tamil"),
        // Telugu
        (0x0C00, 0x0C7F, "te", "Telugu"),
        // Kannada
        (0x0C80, 0x0CFF, "kn", "Kannada"),
        // Malayalam
        (0x0D00, 0x0D7F, "ml", "Malayalam"),
        // Gujarati
        (0x0A80, 0x0AFF, "gu", "Gujarati"),
        // Gurmukhi — Punjabi
        (0x0A00, 0x0A7F, "pa", "Punjabi"),
        // Sinhala
        (0x0D80, 0x0DFF, "si", "Sinhala"),
        // Thai
        (0x0E00, 0x0E7F, "th", "Thai"),
        // CJK Unified Ideographs (Chinese / Japanese kanji / Korean hanja shared range)
        (0x4E00, 0x9FFF, "zh", "Chinese"),
        // Hiragana — Japanese
        (0x3040, 0x309F, "ja", "Japanese"),
        // Katakana — Japanese
        (0x30A0, 0x30FF, "ja", "Japanese"),
        // Hangul Syllables — Korean
        (0xAC00, 0xD7AF, "ko", "Korean"),
        // Cyrillic — Russian, Ukrainian, Bulgarian, etc.
        (0x0400, 0x04FF, "ru", "Russian"),
        // Greek
        (0x0370, 0x03FF, "el", "Greek"),
    ];

    pub struct LanguageRegistry;

    impl LanguageRegistry {
        pub fn new() -> Arc<Self> {
            Arc::new(Self)
        }

        /// Detect the dominant script of `text` by scanning its Unicode code points
        /// against known block ranges. The language whose script characters are most
        /// frequent wins. Falls back to "en" when no non-ASCII script is dominant.
        pub fn detect_language(&self, text: &str) -> String {
            let mut scores: std::collections::HashMap<&str, usize> =
                std::collections::HashMap::new();

            for ch in text.chars() {
                let cp = ch as u32;
                if cp < 0x0080 {
                    // ASCII — not distinctive for non-English detection.
                    continue;
                }
                for &(lo, hi, code, _name) in SCRIPT_RANGES {
                    if cp >= lo && cp <= hi {
                        *scores.entry(code).or_insert(0) += 1;
                        break;
                    }
                }
            }

            if scores.is_empty() {
                return "en".to_string();
            }

            // Pick the language code with the highest character count.
            scores
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(code, _)| code.to_string())
                .unwrap_or_else(|| "en".to_string())
        }

        pub fn list_languages(&self) -> Vec<Value> {
            // Deduplicate by code (ja appears twice for hiragana + katakana).
            let mut seen = std::collections::HashSet::new();
            let mut result =
                vec![json!({ "code": "en", "name": "English", "native_name": "English" })];
            for &(_lo, _hi, code, name) in SCRIPT_RANGES {
                if seen.insert(code) {
                    result.push(json!({ "code": code, "name": name }));
                }
            }
            result
        }
    }
}

// ── Plugin Engine ─────────────────────────────────────────────────────────────
pub mod plugin_engine {
    use serde::{Deserialize, Serialize};
    use serde_json::{json, Value};
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Plugin {
        pub plugin_id: String,
        pub name: String,
        pub version: String,
        pub entrypoint: String,
        pub capabilities: Vec<String>,
        pub enabled: bool,
    }

    pub struct PluginEngine {
        pub plugins_dir: String,
        pub plugins: Arc<Mutex<HashMap<String, Plugin>>>,
    }

    impl PluginEngine {
        pub fn new() -> Arc<Self> {
            Self::new_with_dir("storage/plugins")
        }

        pub fn new_with_dir(plugins_dir: &str) -> Arc<Self> {
            let engine = Arc::new(Self {
                plugins_dir: plugins_dir.to_string(),
                plugins: Arc::new(Mutex::new(HashMap::new())),
            });
            engine.scan_plugins_from_disk();
            engine
        }

        /// Scans disk directory for plugin manifests (manifest.json or plugin.json).
        pub fn scan_plugins_from_disk(&self) {
            let p = Path::new(&self.plugins_dir);
            if !p.exists() {
                let _ = fs::create_dir_all(p);
                return;
            }

            if let Ok(entries) = fs::read_dir(p) {
                let mut map = self.plugins.lock().unwrap();
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        let manifest_cand1 = path.join("plugin.json");
                        let manifest_cand2 = path.join("manifest.json");
                        let manifest_path = if manifest_cand1.exists() {
                            Some(manifest_cand1)
                        } else if manifest_cand2.exists() {
                            Some(manifest_cand2)
                        } else {
                            None
                        };

                        if let Some(mp) = manifest_path {
                            if let Ok(content) = fs::read_to_string(&mp) {
                                if let Ok(val) = serde_json::from_str::<Value>(&content) {
                                    if let Some(id) = val
                                        .get("id")
                                        .or_else(|| val.get("plugin_id"))
                                        .and_then(|v| v.as_str())
                                    {
                                        let name =
                                            val.get("name").and_then(|v| v.as_str()).unwrap_or(id);
                                        let version = val
                                            .get("version")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("1.0.0");
                                        let entrypoint = val
                                            .get("entrypoint")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("index.js");
                                        let caps: Vec<String> = val
                                            .get("capabilities")
                                            .and_then(|v| v.as_array())
                                            .map(|arr| {
                                                arr.iter()
                                                    .filter_map(|c| {
                                                        c.as_str().map(|s| s.to_string())
                                                    })
                                                    .collect()
                                            })
                                            .unwrap_or_default();
                                        let enabled = val
                                            .get("enabled")
                                            .and_then(|v| v.as_bool())
                                            .unwrap_or(true);

                                        map.insert(
                                            id.to_string(),
                                            Plugin {
                                                plugin_id: id.to_string(),
                                                name: name.to_string(),
                                                version: version.to_string(),
                                                entrypoint: entrypoint.to_string(),
                                                capabilities: caps,
                                                enabled,
                                            },
                                        );
                                    }
                                }
                            }
                        }
                    } else if path.extension().and_then(|e| e.to_str()) == Some("json") {
                        if let Ok(content) = fs::read_to_string(&path) {
                            if let Ok(plugin) = serde_json::from_str::<Plugin>(&content) {
                                map.insert(plugin.plugin_id.clone(), plugin);
                            }
                        }
                    }
                }
            }
        }

        pub fn register_plugin(&self, plugin: Plugin) {
            let mut map = self.plugins.lock().unwrap();
            map.insert(plugin.plugin_id.clone(), plugin);
        }

        pub fn list_plugins(&self) -> Vec<Value> {
            let map = self.plugins.lock().unwrap();
            map.values()
                .map(|p| {
                    json!({
                        "plugin_id": p.plugin_id,
                        "name": p.name,
                        "version": p.version,
                        "entrypoint": p.entrypoint,
                        "capabilities": p.capabilities,
                        "enabled": p.enabled
                    })
                })
                .collect()
        }

        pub fn execute_plugin(&self, id: &str, params: Value) -> Value {
            let map = self.plugins.lock().unwrap();
            let plugin = match map.get(id) {
                Some(p) => p.clone(),
                None => {
                    return json!({ "status": "ERROR", "error": format!("Plugin '{}' not found", id) });
                }
            };
            drop(map);

            if !plugin.enabled {
                return json!({
                    "status": "ERROR",
                    "error": format!("Plugin '{}' is disabled", id)
                });
            }

            // 1. Check if entrypoint file exists on disk in plugin directory or plugins root
            let entry_path_1 = Path::new(&self.plugins_dir)
                .join(&plugin.plugin_id)
                .join(&plugin.entrypoint);
            let entry_path_2 = Path::new(&self.plugins_dir).join(&plugin.entrypoint);
            let target_entry = if entry_path_1.exists() {
                Some(entry_path_1)
            } else if entry_path_2.exists() {
                Some(entry_path_2)
            } else {
                None
            };

            if let Some(entry_file) = target_entry {
                let ext = entry_file
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("");
                let output = match ext {
                    "exe" | "bat" | "cmd" => std::process::Command::new(&entry_file)
                        .arg(params.to_string())
                        .output(),
                    "py" => std::process::Command::new("python")
                        .arg(&entry_file)
                        .arg(params.to_string())
                        .output(),
                    "js" => std::process::Command::new("node")
                        .arg(&entry_file)
                        .arg(params.to_string())
                        .output(),
                    _ => std::process::Command::new(&entry_file)
                        .arg(params.to_string())
                        .output(),
                };

                match output {
                    Ok(out) => {
                        let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
                        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
                        if out.status.success() {
                            let parsed: Value = serde_json::from_str(&stdout)
                                .unwrap_or(json!({ "output": stdout }));
                            json!({
                                "status": "SUCCESS",
                                "plugin_id": plugin.plugin_id,
                                "result": parsed
                            })
                        } else {
                            json!({
                                "status": "ERROR",
                                "plugin_id": plugin.plugin_id,
                                "error": if !stderr.is_empty() { stderr } else { format!("Process exited with status {:?}", out.status.code()) }
                            })
                        }
                    }
                    Err(e) => json!({
                        "status": "ERROR",
                        "plugin_id": plugin.plugin_id,
                        "error": format!("Failed to execute plugin entrypoint: {}", e)
                    }),
                }
            } else if plugin
                .capabilities
                .iter()
                .any(|c| c == "arithmetic" || c == "calculator")
            {
                // Real arithmetic computation
                let op = params.get("op").and_then(|v| v.as_str()).unwrap_or("");
                let a = params.get("a").and_then(|v| v.as_f64());
                let b = params.get("b").and_then(|v| v.as_f64());
                match (op, a, b) {
                    ("add", Some(x), Some(y)) => json!({
                        "status": "SUCCESS",
                        "plugin_id": plugin.plugin_id,
                        "result": { "operation": "add", "value": x + y }
                    }),
                    ("sub", Some(x), Some(y)) => json!({
                        "status": "SUCCESS",
                        "plugin_id": plugin.plugin_id,
                        "result": { "operation": "sub", "value": x - y }
                    }),
                    ("mul", Some(x), Some(y)) => json!({
                        "status": "SUCCESS",
                        "plugin_id": plugin.plugin_id,
                        "result": { "operation": "mul", "value": x * y }
                    }),
                    ("div", Some(x), Some(y)) => {
                        if y == 0.0 {
                            json!({ "status": "ERROR", "error": "Division by zero" })
                        } else {
                            json!({
                                "status": "SUCCESS",
                                "plugin_id": plugin.plugin_id,
                                "result": { "operation": "div", "value": x / y }
                            })
                        }
                    }
                    _ => json!({
                        "status": "ERROR",
                        "error": format!("Unsupported arithmetic operation '{}' or missing operands", op)
                    }),
                }
            } else {
                json!({
                    "status": "ERROR",
                    "error": format!("Plugin entrypoint '{}' does not exist on disk", plugin.entrypoint)
                })
            }
        }
    }
}

// ── Knowledge Ontology ────────────────────────────────────────────────────────
pub mod knowledge_ontology {
    pub use crate::cognitive::ontology::{
        ConceptNode, OntologyEdge, OntologyEngine as KnowledgeOntology, OntologyRelation,
        SemanticPath,
    };
}

// ── Resilience Engine ─────────────────────────────────────────────────────────
pub mod resilience {
    use serde_json::{json, Value};
    use std::fs;

    pub struct ResilienceEngine {
        pub repo_root: String,
    }
    impl ResilienceEngine {
        pub fn new(repo_root: &str) -> Self {
            Self {
                repo_root: repo_root.to_string(),
            }
        }
        pub fn run_health_check(&self) -> Value {
            let model_path = format!("{}/storage/models/tara/model.safetensors", self.repo_root);
            let model_exists = fs::metadata(&model_path).is_ok();
            json!({ "model_file_exists": model_exists, "storage_accessible": true, "overall_health": if model_exists { "HEALTHY" } else { "DEGRADED" } })
        }
        pub fn get_resilience_status(&self) -> Value {
            self.run_health_check()
        }
    }
}
