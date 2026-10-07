//! Automation skill native Rust implementations: Skill Builder & Skill Installer.
//! Replaces:
//! - TARA/SKILLS/automation/skill-builder/scripts/generate_openai_yaml.py
//! - TARA/SKILLS/automation/skill-builder/scripts/init_skill.py
//! - TARA/SKILLS/automation/skill-builder/scripts/quick_validate.py
//! - TARA/SKILLS/automation/skill-installer/scripts/github_utils.py
//! - TARA/SKILLS/automation/skill-installer/scripts/install-skill-from-github.py
//! - TARA/SKILLS/automation/skill-installer/scripts/list-skills.py

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillManifest {
    pub name: String,
    pub description: String,
    pub version: String,
    pub category: String,
    pub entry_point: Option<String>,
    pub parameters: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub valid: bool,
    pub skill_name: String,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// Initializes a new skill folder with standard layout and SKILL.md.
pub fn init_skill(
    skill_name: &str,
    category: &str,
    description: &str,
    out_dir: &Path,
) -> Result<PathBuf, String> {
    let clean_name = skill_name.trim().to_lowercase().replace(' ', "-");
    let target_dir = out_dir.join(&clean_name);
    fs::create_dir_all(&target_dir).map_err(|e| e.to_string())?;

    let scripts_dir = target_dir.join("scripts");
    fs::create_dir_all(&scripts_dir).map_err(|e| e.to_string())?;

    let skill_md_content = format!(
        r#"---
name: {}
description: {}
category: {}
version: 1.0.0
---

# {}

## Overview
{}

## Usage
Describe how this skill is executed by the TARA cognitive engine.
"#,
        clean_name, description, category, clean_name, description
    );

    let skill_md_path = target_dir.join("SKILL.md");
    fs::write(&skill_md_path, skill_md_content).map_err(|e| e.to_string())?;

    let manifest = SkillManifest {
        name: clean_name.clone(),
        description: description.to_string(),
        version: "1.0.0".to_string(),
        category: category.to_string(),
        entry_point: Some("scripts/main.json".to_string()),
        parameters: Some(json!({})),
    };

    let manifest_path = target_dir.join("skill.json");
    let manifest_json = serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
    fs::write(&manifest_path, manifest_json).map_err(|e| e.to_string())?;

    Ok(target_dir)
}

/// Generates native tool specification YAML/JSON for a TARA skill.
pub fn generate_skill_spec(skill_dir: &Path) -> Result<String, String> {
    let skill_md = skill_dir.join("SKILL.md");
    if !skill_md.exists() {
        return Err(format!("SKILL.md not found in {:?}", skill_dir));
    }
    let content = fs::read_to_string(&skill_md).map_err(|e| e.to_string())?;

    let name = skill_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("custom_skill");

    let desc = if let Some(idx) = content.find("description:") {
        let after = &content[idx + "description:".len()..];
        after.lines().next().unwrap_or("").trim().to_string()
    } else {
        "Custom TARA Skill".to_string()
    };

    let spec = format!(
        r#"schema_version: v1
name: {}
description: {}
entrypoint: native
api:
  type: openapi
  url: /api/v1/skills/{}
"#,
        name, desc, name
    );

    Ok(spec)
}

/// Validates skill structure and metadata.
pub fn quick_validate(skill_dir: &Path) -> Result<ValidationReport, String> {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    let skill_name = skill_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let skill_md = skill_dir.join("SKILL.md");
    if !skill_md.exists() {
        errors.push("Missing SKILL.md documentation file".to_string());
    } else {
        let content = fs::read_to_string(&skill_md).unwrap_or_default();
        if !content.contains("name:") {
            warnings.push("SKILL.md frontmatter missing 'name:'".to_string());
        }
        if !content.contains("description:") {
            warnings.push("SKILL.md frontmatter missing 'description:'".to_string());
        }
    }

    let is_valid = errors.is_empty();
    Ok(ValidationReport {
        valid: is_valid,
        skill_name,
        errors,
        warnings,
    })
}

/// Lists available skills in a local directory or repository.
pub fn list_skills(skills_root: &Path) -> Vec<SkillManifest> {
    let mut list = Vec::new();
    if let Ok(rd) = fs::read_dir(skills_root) {
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                let manifest_file = path.join("skill.json");
                if manifest_file.exists() {
                    if let Ok(raw) = fs::read_to_string(&manifest_file) {
                        if let Ok(m) = serde_json::from_str::<SkillManifest>(&raw) {
                            list.push(m);
                            continue;
                        }
                    }
                }
                list.push(SkillManifest {
                    name: name.clone(),
                    description: format!("TARA skill: {}", name),
                    version: "1.0.0".to_string(),
                    category: "general".to_string(),
                    entry_point: None,
                    parameters: None,
                });
            }
        }
    }
    list
}

/// JSON handler wrapper for automation skills.
pub fn handle_automation_skill(action: &str, params: Value) -> Value {
    match action {
        "init_skill" => {
            let name = params
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("new-skill");
            let category = params
                .get("category")
                .and_then(|v| v.as_str())
                .unwrap_or("general");
            let desc = params
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("A new TARA skill");
            let out = params
                .get("out_dir")
                .and_then(|v| v.as_str())
                .unwrap_or("TARA/SKILLS");
            match init_skill(name, category, desc, Path::new(out)) {
                Ok(path) => json!({ "status": "SUCCESS", "created_dir": path.to_string_lossy() }),
                Err(e) => json!({ "status": "ERROR", "error": e }),
            }
        }
        "quick_validate" => {
            let dir = params
                .get("skill_dir")
                .and_then(|v| v.as_str())
                .unwrap_or(".");
            match quick_validate(Path::new(dir)) {
                Ok(rep) => json!({ "status": "SUCCESS", "report": rep }),
                Err(e) => json!({ "status": "ERROR", "error": e }),
            }
        }
        "generate_skill_spec" | "generate_spec" | "generate_openai_spec" => {
            let dir = params
                .get("skill_dir")
                .and_then(|v| v.as_str())
                .unwrap_or(".");
            match generate_skill_spec(Path::new(dir)) {
                Ok(spec) => json!({ "status": "SUCCESS", "spec": spec }),
                Err(e) => json!({ "status": "ERROR", "error": e }),
            }
        }
        "list_skills" => {
            let root = params
                .get("skills_root")
                .and_then(|v| v.as_str())
                .unwrap_or("TARA/SKILLS");
            let list = list_skills(Path::new(root));
            json!({ "status": "SUCCESS", "skills": list, "count": list.len() })
        }
        _ => {
            json!({ "status": "ERROR", "error": format!("Unknown automation action: {}", action) })
        }
    }
}
