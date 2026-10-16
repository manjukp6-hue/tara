//! Additional server subsystems: user model, language registry, plugin engine, resilience.

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

    fn validate_user_id(user_id: &str) -> Result<String, String> {
        let trimmed = user_id.trim();
        if trimmed.is_empty() || trimmed.len() > 64 {
            return Err("Invalid user_id: length must be between 1 and 64 characters".to_string());
        }
        if !trimmed
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(
                "Invalid user_id: only alphanumeric, underscore, and hyphen characters allowed"
                    .to_string(),
            );
        }
        Ok(trimmed.to_string())
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
            let safe_id = match validate_user_id(user_id) {
                Ok(id) => id,
                Err(_) => "default_user".to_string(),
            };
            let path = format!("{}/{}.json", self.base_dir, safe_id);
            if let Ok(raw) = fs::read_to_string(&path) {
                if let Ok(v) = serde_json::from_str::<Value>(&raw) {
                    let mut prefs = HashMap::new();
                    if let Some(obj) = v.get("preferences").and_then(|p| p.as_object()) {
                        for (k, val) in obj {
                            if let Some(s) = val.as_str() {
                                prefs.insert(k.clone(), s.to_string());
                            }
                        }
                    }
                    return UserProfile {
                        user_id: safe_id.clone(),
                        display_name: v
                            .get("display_name")
                            .and_then(|v| v.as_str())
                            .unwrap_or(&safe_id)
                            .to_string(),
                        preferences: prefs,
                        interaction_count: v
                            .get("interaction_count")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0),
                    };
                }
            }
            UserProfile {
                user_id: safe_id.clone(),
                display_name: safe_id,
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
            let safe_id = validate_user_id(user_id)?;
            let path = format!("{}/{}.json", self.base_dir, safe_id);
            let mut v = fs::read_to_string(&path)
                .ok()
                .and_then(|r| serde_json::from_str::<Value>(&r).ok())
                .unwrap_or(json!({ "user_id": safe_id }));
            if let Some(obj) = v.get_mut("preferences").and_then(|p| p.as_object_mut()) {
                obj.insert(key.to_string(), json!(value));
            } else {
                v["preferences"] = json!({ key: value });
            }
            let tmp_path = format!("{}/{}.json.tmp", self.base_dir, safe_id);
            let json_str = serde_json::to_string_pretty(&v).unwrap_or_default();
            let mut f = fs::File::create(&tmp_path).map_err(|e| e.to_string())?;
            use std::io::Write;
            f.write_all(json_str.as_bytes())
                .map_err(|e| e.to_string())?;
            f.sync_all().map_err(|e| e.to_string())?;
            drop(f);
            fs::rename(&tmp_path, &path).map_err(|e| e.to_string())?;
            Ok(())
        }
    }
}

// ── Agent Orchestrator ────────────────────────────────────────────────────────
pub mod agent_orchestrator {
    use serde_json::{json, Value};
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use tara_core::{ManagerConfig, TaraManager};

    const MAX_AGENT_TASKS: usize = 1000;

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
            let trimmed_role = role.trim();
            let trimmed_obj = objective.trim();
            if trimmed_role.is_empty() || trimmed_role.len() > 128 {
                return json!({ "status": "ERROR", "error": "Invalid role specification" });
            }
            if trimmed_obj.is_empty() || trimmed_obj.len() > 1024 {
                return json!({ "status": "ERROR", "error": "Invalid objective specification" });
            }
            let mut tasks = match self.tasks.lock() {
                Ok(t) => t,
                Err(_) => return json!({ "status": "ERROR", "error": "Agent task lock poisoned" }),
            };
            if tasks.len() >= MAX_AGENT_TASKS {
                return json!({ "status": "ERROR", "error": "Maximum agent task limit reached" });
            }
            let mut manager = match self.manager.lock() {
                Ok(manager) => manager,
                Err(_) => {
                    return json!({ "status": "ERROR", "error": "Agent manager lock poisoned" })
                }
            };
            let capabilities = Vec::new();
            let (agent_id, display_name) =
                match manager.create_agent(trimmed_role, &capabilities, None) {
                    Ok(identity) => identity,
                    Err(error) => return json!({ "status": "ERROR", "error": error }),
                };
            drop(manager);
            tasks.push(AgentTask {
                agent_id: agent_id.clone(),
                role: trimmed_role.to_string(),
                objective: trimmed_obj.to_string(),
                status: "PENDING".to_string(),
                output: None,
            });
            json!({ "agent_id": agent_id, "display_name": display_name, "role": trimmed_role, "objective": trimmed_obj, "status": "IDLE", "runtime_backed": true })
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


// ── Plugin Engine ─────────────────────────────────────────────────────────────
pub mod plugin_engine {
    use serde::{Deserialize, Serialize};
    use serde_json::{json, Value};
    use std::collections::HashMap;
    use std::fs;
    use std::io::Write;
    use std::path::Path;
    use std::process::Stdio;
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
        /// Newly discovered plugins default to `enabled: false` (fail closed).
        pub fn scan_plugins_from_disk(&self) {
            let p = Path::new(&self.plugins_dir);
            if !p.exists() {
                let _ = fs::create_dir_all(p);
                return;
            }

            if let Ok(entries) = fs::read_dir(p) {
                let mut map = self.plugins.lock().unwrap_or_else(|e| e.into_inner());
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
                                        // Reject path traversal in plugin ID
                                        if id.contains('/')
                                            || id.contains('\\')
                                            || id.contains("..")
                                        {
                                            continue;
                                        }

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

                                        // Reject traversal in entrypoint
                                        if entrypoint.contains("..") {
                                            continue;
                                        }

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

                                        // Discovered plugins default to disabled (fail closed; only officially registered plugins can be enabled)
                                        let enabled = false;

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
                            if let Ok(mut plugin) = serde_json::from_str::<Plugin>(&content) {
                                if !plugin.plugin_id.contains("..")
                                    && !plugin.entrypoint.contains("..")
                                {
                                    plugin.enabled = false; // default disabled
                                    map.insert(plugin.plugin_id.clone(), plugin);
                                }
                            }
                        }
                    }
                }
            }
        }

        pub fn register_plugin(&self, plugin: Plugin) {
            if plugin.plugin_id.contains("..") || plugin.entrypoint.contains("..") {
                return;
            }
            let mut map = self.plugins.lock().unwrap_or_else(|e| e.into_inner());
            map.insert(plugin.plugin_id.clone(), plugin);
        }

        pub fn list_plugins(&self) -> Vec<Value> {
            let map = self.plugins.lock().unwrap_or_else(|e| e.into_inner());
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

        pub fn execute_plugin_authorized(
            &self,
            id: &str,
            params: Value,
            principal: &tara_security::primitives::AuthenticatedPrincipal,
        ) -> Value {
            if !principal.is_creator() {
                return json!({
                    "status": "ERROR",
                    "error": "Unauthorized: Plugin execution requires Creator authorization."
                });
            }
            self.execute_plugin_internal(id, params, true)
        }

        pub fn execute_plugin(&self, id: &str, params: Value) -> Value {
            // Unauthenticated callers can NEVER spawn OS processes; only pure in-process arithmetic is allowed
            self.execute_plugin_internal(id, params, false)
        }

        fn execute_plugin_internal(&self, id: &str, params: Value, allow_os_process: bool) -> Value {
            let map = self.plugins.lock().unwrap_or_else(|e| e.into_inner());
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

            // Verify sandboxed containment under canonical plugins_dir
            let canonical_root = match Path::new(&self.plugins_dir).canonicalize() {
                Ok(r) => r,
                Err(_) => {
                    return json!({ "status": "ERROR", "error": "Plugin directory does not exist or cannot be resolved" });
                }
            };

            let entry_cand1 = Path::new(&self.plugins_dir)
                .join(&plugin.plugin_id)
                .join(&plugin.entrypoint);
            let entry_cand2 = Path::new(&self.plugins_dir).join(&plugin.entrypoint);

            let target_entry = if entry_cand1.exists() {
                Some(entry_cand1)
            } else if entry_cand2.exists() {
                Some(entry_cand2)
            } else {
                None
            };

            if let Some(entry_file) = target_entry {
                if !allow_os_process {
                    return json!({
                        "status": "ERROR",
                        "error": "Unauthorized: OS process execution requires execute_plugin_authorized with Creator principal."
                    });
                }
                let canonical_entry = match entry_file.canonicalize() {
                    Ok(c) => c,
                    Err(e) => {
                        return json!({ "status": "ERROR", "error": format!("Cannot canonicalize plugin entrypoint: {e}") });
                    }
                };

                // Assert strict containment
                if !canonical_entry.starts_with(&canonical_root) {
                    return json!({ "status": "ERROR", "error": "Security violation: plugin entrypoint escapes plugin sandbox" });
                }

                let ext = canonical_entry
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("");

                // Pure native workflow: Python execution is prohibited in native runtime
                if ext == "py" {
                    return json!({
                        "status": "ERROR",
                        "error": "Execution of Python scripts is strictly prohibited; runtime operates 100% natively"
                    });
                }

                let mut cmd = match ext {
                    "exe" | "bat" | "cmd" => std::process::Command::new(&canonical_entry),
                    "js" => {
                        let mut c = std::process::Command::new("node");
                        c.arg(&canonical_entry);
                        c
                    }
                    _ => std::process::Command::new(&canonical_entry),
                };

                // Pipe params safely via stdin rather than exposing via command-line arguments
                cmd.stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped());

                let mut child = match cmd.spawn() {
                    Ok(ch) => ch,
                    Err(e) => {
                        return json!({ "status": "ERROR", "error": format!("Failed to spawn plugin process: {e}") });
                    }
                };

                if let Some(mut stdin) = child.stdin.take() {
                    let _ = stdin.write_all(params.to_string().as_bytes());
                }

                let output = match child.wait_with_output() {
                    Ok(out) => out,
                    Err(e) => {
                        return json!({ "status": "ERROR", "error": format!("Plugin process failed: {e}") });
                    }
                };

                let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();

                if output.status.success() {
                    let parsed: Value =
                        serde_json::from_str(&stdout).unwrap_or(json!({ "output": stdout }));
                    json!({
                        "status": "SUCCESS",
                        "plugin_id": plugin.plugin_id,
                        "result": parsed
                    })
                } else {
                    json!({
                        "status": "ERROR",
                        "plugin_id": plugin.plugin_id,
                        "error": if !stderr.is_empty() { stderr } else { format!("Process exited with status {:?}", output.status.code()) }
                    })
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
            // Authentically test storage accessibility
            let storage_dir = format!("{}/storage", self.repo_root);
            let storage_accessible = fs::metadata(&storage_dir)
                .map(|m| m.is_dir())
                .unwrap_or(false);

            // Dynamically query active model artifact from authoritative ModelRegistry
            let active_artifact = crate::learning::model_registry::ModelRegistry::new(&self.repo_root)
                .get_active_version_location();

            let model_path = if let Some(ref rel_path) = active_artifact {
                format!("{}/{}", self.repo_root, rel_path)
            } else {
                format!("{}/storage/models/tara/model.safetensors", self.repo_root)
            };
            let index_path = model_path.replace("model.safetensors", "model.safetensors.index.json");
            let model_exists = fs::metadata(&model_path).is_ok()
                || fs::metadata(&index_path).is_ok()
                || fs::metadata(format!("{}/storage/models/tara/model.safetensors.index.json", self.repo_root)).is_ok();
            let is_healthy = storage_accessible && model_exists;
            json!({
                "model_file_exists": model_exists,
                "model_path": model_path,
                "storage_accessible": storage_accessible,
                "overall_health": if is_healthy { "HEALTHY" } else { "DEGRADED" }
            })
        }
        pub fn get_resilience_status(&self) -> Value {
            self.run_health_check()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::plugin_engine::{Plugin, PluginEngine};
    use super::resilience::ResilienceEngine;
    use serde_json::json;
    use std::fs;

    #[test]
    fn test_plugin_engine_disk_scan_defaults_to_disabled() {
        let temp_dir = std::env::temp_dir().join(format!("tara_plugin_test_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let plugin_dir = temp_dir.join("test_plugin");
        fs::create_dir_all(&plugin_dir).unwrap();

        let manifest = json!({
            "id": "test_plugin",
            "name": "Test Plugin",
            "version": "1.0.0",
            "entrypoint": "run.exe",
            "enabled": true
        });
        fs::write(plugin_dir.join("manifest.json"), manifest.to_string()).unwrap();

        let engine = PluginEngine::new_with_dir(temp_dir.to_str().unwrap());
        let list = engine.list_plugins();
        let p = list.iter().find(|item| item["plugin_id"] == "test_plugin").unwrap();
        assert_eq!(p["enabled"], false);

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_plugin_engine_unauthenticated_blocks_os_processes_and_allows_arithmetic() {
        let temp_dir = std::env::temp_dir().join(format!("tara_plugin_arith_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        fs::create_dir_all(&temp_dir).unwrap();
        let engine = PluginEngine::new_with_dir(temp_dir.to_str().unwrap());

        // Arithmetic plugin (no file on disk, pure in-memory)
        engine.register_plugin(Plugin {
            plugin_id: "calc".to_string(),
            name: "Calculator".to_string(),
            version: "1.0.0".to_string(),
            entrypoint: "calc.exe".to_string(),
            capabilities: vec!["arithmetic".to_string()],
            enabled: true,
        });

        let res = engine.execute_plugin("calc", json!({ "op": "add", "a": 10.0, "b": 25.0 }));
        assert_eq!(res["status"], "SUCCESS");
        assert_eq!(res["result"]["value"], 35.0);

        // If file exists on disk, unauthenticated execute_plugin MUST block OS process spawn
        let dummy_exe = temp_dir.join("calc.exe");
        fs::write(&dummy_exe, b"dummy").unwrap();

        let blocked_res = engine.execute_plugin("calc", json!({ "op": "add", "a": 10.0, "b": 25.0 }));
        assert_eq!(blocked_res["status"], "ERROR");
        assert!(blocked_res["error"].as_str().unwrap().contains("Unauthorized: OS process execution requires execute_plugin_authorized with Creator principal."));

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_resilience_engine_sharded_model_health_check() {
        let temp_dir = std::env::temp_dir().join(format!("tara_resilience_test_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let model_dir = temp_dir.join("storage").join("models").join("tara");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("model.safetensors.index.json"), "{}").unwrap();

        let engine = ResilienceEngine::new(temp_dir.to_str().unwrap());
        let health = engine.run_health_check();
        assert_eq!(health["model_file_exists"], true);
        assert_eq!(health["overall_health"], "HEALTHY");

        let _ = fs::remove_dir_all(temp_dir);
    }
}
