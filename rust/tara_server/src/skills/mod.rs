//! Skills engine: loads and executes native TARA skills across all domains.
//! Completely replaces all legacy Python scripts in TARA/SKILLS/**/scripts/.

pub mod android;
pub mod automation;
pub mod certification;
pub mod coding;
pub mod data;
pub mod multimedia;
pub mod research;
pub mod security;

pub use certification::{
    CertificationState, SkillCertificationAuthority, SkillCertificationRecord, SkillTraceSample,
};

use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// High-speed native dispatch for all 52+ migrated TARA skills.
pub struct SkillEngine {
    pub skills_dir: String,
    pub dynamic_skills_dir: String,
    catalog: HashMap<String, Value>,
}

impl SkillEngine {
    pub fn new(skills_dir: &str, dynamic_skills_dir: &str) -> Self {
        let catalog = Self::load_catalog(skills_dir);
        Self {
            skills_dir: skills_dir.to_string(),
            dynamic_skills_dir: dynamic_skills_dir.to_string(),
            catalog,
        }
    }

    fn load_catalog(skills_dir: &str) -> HashMap<String, Value> {
        let mut catalog = HashMap::new();
        let candidates = [
            format!("{}/CATALOG.json", skills_dir),
            format!("{}/skills_catalog.json", skills_dir),
        ];
        for catalog_path in &candidates {
            if let Ok(raw) = fs::read_to_string(catalog_path) {
                if let Ok(v) = serde_json::from_str::<Value>(&raw) {
                    if let Some(obj) = v.as_object() {
                        if let Some(skills_arr) = obj.get("skills").and_then(Value::as_array) {
                            for item in skills_arr {
                                if let Some(name) = item.get("name").and_then(Value::as_str) {
                                    catalog.insert(name.to_string(), item.clone());
                                }
                            }
                        } else {
                            for (k, v) in obj {
                                catalog.insert(k.clone(), v.clone());
                            }
                        }
                    }
                }
            }
        }
        // Also load individual skill JSON files
        if let Ok(rd) = fs::read_dir(skills_dir) {
            for entry in rd.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("json") {
                    if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                        if name != "skills_catalog" && name != "CATALOG" && name != "IMPORT_MANIFEST" {
                            if let Ok(raw) = fs::read_to_string(&path) {
                                if let Ok(v) = serde_json::from_str::<Value>(&raw) {
                                    catalog.insert(name.to_string(), v);
                                }
                            }
                        }
                    }
                }
            }
        }
        catalog
    }

    /// List all registered and native skill names.
    pub fn list_skills(&self) -> Vec<String> {
        let mut skills: Vec<String> = self.catalog.keys().cloned().collect();

        let native_skills = [
            "android.ui_pick",
            "android.ui_tree_summarize",
            "automation.init_skill",
            "automation.quick_validate",
            "automation.generate_skill_spec",
            "automation.list_skills",
            "coding.gh_address_comments",
            "coding.gh_fix_ci",
            "coding.hatch_pet_compose",
            "coding.migrate_to_codex",
            "coding.mixpanel_auth",
            "coding.create_basic_plugin",
            "coding.sentry_api",
            "data.boltz_crop_radius",
            "data.boltz_detect_disorder",
            "data.boltz_terminus",
            "data.generate_jupyter_notebook",
            "multimedia.generate_image",
            "multimedia.take_screenshot",
            "multimedia.text_to_speech",
            "multimedia.transcribe",
            "research.shift_period",
            "research.calculate_slope",
            "research.generate_preview_report",
            "security.build_ownership_map",
            "security.query_ownership",
        ];

        for s in &native_skills {
            if !skills.contains(&s.to_string()) {
                skills.push(s.to_string());
            }
        }

        // Also scan dynamic directory
        if let Ok(rd) = fs::read_dir(&self.dynamic_skills_dir) {
            for entry in rd.flatten() {
                if let Some(name) = entry.path().file_stem().and_then(|s| s.to_str()) {
                    if !skills.contains(&name.to_string()) {
                        skills.push(name.to_string());
                    }
                }
            }
        }
        skills.sort();
        skills
    }

    /// Execute a named skill with given parameters using 100% native Rust logic.
    pub fn execute_skill(&self, skill_name: &str, params: Value) -> Value {
        if !Self::valid_skill_name(skill_name) {
            return json!({"skill":skill_name,"status":"ERROR","error":"Skill name contains invalid path characters","params":params});
        }
        // Direct dispatch to native Rust implementations
        let parts: Vec<&str> = skill_name.splitn(2, '.').collect();
        if parts.len() == 2 {
            let domain = parts[0];
            let action = parts[1];
            let native_res = Self::dispatch_native(domain, action, params.clone());
            if let Some(res) = native_res {
                let status = res.get("status").and_then(Value::as_str).unwrap_or("ERROR");
                let succeeded = status == "SUCCESS";
                return json!({
                    "skill": skill_name,
                    "mode": "NATIVE_RUST",
                    "status": if succeeded { "SUCCESS" } else { "ERROR" },
                    "result": res,
                    "params": params
                });
            }
        }

        // Check catalog for instruction-only skills
        if let Some(entry) = self.catalog.get(skill_name) {
            return json!({
                "skill": skill_name,
                "mode": "INSTRUCTION_ONLY",
                "instructions": entry,
                "status": "SUCCESS",
                "params": params
            });
        }

        // Dynamic skills are executable only as validated workflows over existing
        // Rust-native handlers; arbitrary instructions are never reported as run.
        let skill_path = format!("{}/{}.json", self.dynamic_skills_dir, skill_name);
        if let Ok(raw) = fs::read_to_string(&skill_path) {
            if let Ok(definition) = serde_json::from_str::<Value>(&raw) {
                if let Err(error) = self.validate_workflow_definition(&definition) {
                    return json!({"skill":skill_name,"status":"ERROR","error":error});
                }
                let steps = definition["steps"]
                    .as_array()
                    .expect("validated workflow steps");
                let mut results = Vec::with_capacity(steps.len());
                for (index, step) in steps.iter().enumerate() {
                    let Some(native_name) = step.get("skill").and_then(Value::as_str) else {
                        return json!({"skill":skill_name,"status":"ERROR","error":format!("step {index} requires a native skill name")});
                    };
                    let Some((domain, action)) = native_name.split_once('.') else {
                        return json!({"skill":skill_name,"status":"ERROR","error":format!("step {index} must name a domain.action Rust skill")});
                    };
                    if domain == "security" {
                        return json!({"skill":skill_name,"status":"ERROR","error":"security actions cannot be embedded in learned workflows"});
                    }
                    let step_params = step.get("params").cloned().unwrap_or_else(|| json!({}));
                    let Some(result) = Self::dispatch_native(domain, action, step_params) else {
                        return json!({"skill":skill_name,"status":"ERROR","error":format!("step {index} does not resolve to a native Rust skill")});
                    };
                    if result.get("status").and_then(Value::as_str) != Some("SUCCESS") {
                        return json!({"skill":skill_name,"status":"ERROR","failed_step":index,"completed_steps":results,"error":result});
                    }
                    results.push(result);
                }
                return json!({"skill":skill_name,"mode":"NATIVE_RUST_WORKFLOW","status":"SUCCESS","steps_completed":results.len(),"results":results,"params":params});
            }
        }

        json!({
            "skill": skill_name,
            "status": "ERROR",
            "error": format!("Skill '{}' not found", skill_name),
            "params": params
        })
    }

    fn dispatch_native(domain: &str, action: &str, params: Value) -> Option<Value> {
        match domain {
            "android" => Some(android::handle_android_skill(action, params)),
            "automation" => Some(automation::handle_automation_skill(action, params)),
            "coding" => Some(coding::handle_coding_skill(action, params)),
            "data" => Some(data::handle_data_skill(action, params)),
            "multimedia" => Some(multimedia::handle_multimedia_skill(action, params)),
            "research" => Some(research::handle_research_skill(action, params)),
            "security" => Some(security::handle_security_skill(action, params)),
            _ => None,
        }
    }

    /// Dynamically add a new function / skill.
    /// Protected core security, creator authority, or root rules cannot be targeted or altered.
    pub fn add_dynamic_skill(&self, skill_name: &str, content: Value) -> Result<String, String> {
        let name_clean = skill_name.trim().to_lowercase().replace(' ', "_");
        if !Self::valid_skill_name(&name_clean) {
            return Err("Invalid skill name".to_string());
        }
        if Self::is_protected_skill_name(&name_clean) {
            return Err(format!(
                "Protected core component: '{}' cannot be added or modified as a dynamic skill",
                skill_name
            ));
        }

        fs::create_dir_all(&self.dynamic_skills_dir).map_err(|e| e.to_string())?;
        let root = Path::new(&self.dynamic_skills_dir)
            .canonicalize()
            .map_err(|e| e.to_string())?;
        let skill_path = root.join(format!("{}.json", name_clean));
        let serialized = serde_json::to_string_pretty(&content).map_err(|e| e.to_string())?;
        fs::write(&skill_path, serialized).map_err(|e| e.to_string())?;
        Ok(format!("Dynamic skill '{}' successfully added", name_clean))
    }

    /// Add a learned skill only when it has executable, supported Rust-native steps.
    pub fn add_executable_dynamic_skill(
        &self,
        skill_name: &str,
        definition: Value,
    ) -> Result<String, String> {
        self.validate_workflow_definition(&definition)?;
        self.add_dynamic_skill(skill_name, definition)
    }

    /// Validate that a skill workflow definition is executable.
    ///
    /// A step is accepted if it satisfies any of:
    ///   1. Its domain resolves via `dispatch_native()` (built-in native Rust domain).
    ///   2. It matches a `.json` workflow file in `dynamic_skills_dir` (runtime-added skills).
    ///
    /// Security-domain steps are always rejected regardless of source.
    pub fn validate_workflow_definition(&self, definition: &Value) -> Result<(), String> {
        let steps = definition
            .get("steps")
            .and_then(Value::as_array)
            .ok_or_else(|| "skill definition must contain a steps array".to_string())?;
        if steps.is_empty() || steps.len() > 32 {
            return Err("a workflow must contain between 1 and 32 steps".into());
        }
        for (index, step) in steps.iter().enumerate() {
            let name = step
                .get("skill")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("step {index} requires a native skill name"))?;

            let Some((domain, _action)) = name.split_once('.') else {
                return Err(format!(
                    "step {index} skill name must be 'domain.action', got '{name}'"
                ));
            };

            // Security domain: always blocked regardless of source
            if domain == "security" {
                return Err(format!(
                    "step {index}: security actions cannot be embedded in learned workflows"
                ));
            }

            // Check 1: is this a built-in native Rust domain?
            let is_native_domain = matches!(
                domain,
                "android" | "automation" | "coding" | "data" | "multimedia" | "research"
            );

            // Check 2: is this a dynamically-added skill (exists on disk)?
            let is_dynamic = {
                let clean = name.replace(['/', '.'], "_");
                let dynamic_path =
                    Path::new(&self.dynamic_skills_dir).join(format!("{clean}.json"));
                dynamic_path.is_file()
            };

            if !is_native_domain && !is_dynamic {
                return Err(format!(
                    "step {index} names an unresolvable skill: '{name}' \
                     (not a built-in domain and not found in dynamic_skills_dir)"
                ));
            }

            if step.get("params").is_some_and(|params| !params.is_object()) {
                return Err(format!("step {index} params must be a JSON object"));
            }
        }
        Ok(())
    }

    /// Dynamically remove an unneeded or deprecated skill.
    /// Protected core components (security, authority, rules) CANNOT be removed.
    pub fn remove_dynamic_skill(&self, skill_name: &str) -> Result<String, String> {
        let name_clean = skill_name.trim().to_lowercase().replace(' ', "_");
        if !Self::valid_skill_name(&name_clean) {
            return Err("Invalid skill name".to_string());
        }
        if Self::is_protected_skill_name(&name_clean) {
            return Err(format!(
                "Protected core component: '{}' cannot be removed",
                skill_name
            ));
        }

        let root = Path::new(&self.dynamic_skills_dir)
            .canonicalize()
            .map_err(|e| e.to_string())?;
        let skill_path = root.join(format!("{}.json", name_clean));
        if skill_path.is_file() {
            fs::remove_file(&skill_path).map_err(|e| e.to_string())?;
            Ok(format!(
                "Dynamic skill '{}' successfully removed",
                name_clean
            ))
        } else {
            Err(format!("Dynamic skill '{}' not found on disk", name_clean))
        }
    }

    fn valid_skill_name(name: &str) -> bool {
        !name.is_empty()
            && !name.contains("..")
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
    }

    /// Checks if a skill name targets protected core security or authority components.
    pub fn is_protected_skill_name(name: &str) -> bool {
        let protected_keywords = [
            "security",
            "authority",
            "creator",
            "rule",
            "guard",
            "root",
            "bypass",
            "auth",
            "permission",
            "crypto",
        ];
        protected_keywords.iter().any(|&k| name.contains(k))
    }
}
