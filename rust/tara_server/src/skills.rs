//! Skills engine: loads and executes named skills.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use serde_json::{json, Value};

/// Loads and dispatches named skills from disk.
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
        let catalog_path = format!("{}/skills_catalog.json", skills_dir);
        if let Ok(raw) = fs::read_to_string(&catalog_path) {
            if let Ok(v) = serde_json::from_str::<Value>(&raw) {
                if let Some(obj) = v.as_object() {
                    for (k, v) in obj {
                        catalog.insert(k.clone(), v.clone());
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
                        if name != "skills_catalog" {
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

    /// List all known skill names.
    pub fn list_skills(&self) -> Vec<String> {
        let mut skills: Vec<String> = self.catalog.keys().cloned().collect();
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

    /// Execute a named skill with given parameters.
    pub fn execute_skill(&self, skill_name: &str, params: Value) -> Value {
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

        // Check for a JSON skill file in dynamic dir
        let skill_path = format!("{}/{}.json", self.dynamic_skills_dir, skill_name);
        if let Ok(raw) = fs::read_to_string(&skill_path) {
            if let Ok(v) = serde_json::from_str::<Value>(&raw) {
                return json!({
                    "skill": skill_name,
                    "mode": "INSTRUCTION_ONLY",
                    "instructions": v,
                    "status": "SUCCESS",
                    "params": params
                });
            }
        }

        json!({
            "skill": skill_name,
            "status": "ERROR",
            "error": format!("Skill '{}' not found", skill_name),
            "params": params
        })
    }
}
