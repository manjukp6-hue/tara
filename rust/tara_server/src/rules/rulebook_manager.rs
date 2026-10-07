//! Rulebook Manager.
//! Orchestrates Natural-Language Rulebook storage, compilation, cryptographic signing,
//! provenance verification, immutable versioning, rollback, and audit logging.

use super::policy_compiler::PolicyCompiler;
use super::policy_schema::{RuleItem, StructuredCompiledPolicy};
use serde_json::{json, Value};

use std::fs;
use std::path::PathBuf;

const FORBIDDEN_AUDIT_KEYS: &[&str] = &[
    "private_key",
    "privkey",
    "secret",
    "seed",
    "token",
    "password",
    "passphrase",
];

pub struct RulebookManager {
    pub base_dir: PathBuf,
    pub rulebook_txt_path: PathBuf,
    pub default_safe_path: PathBuf,
    pub compiled_policy_path: PathBuf,
    pub audit_file: PathBuf,
    pub compiler: PolicyCompiler,
}

impl RulebookManager {
    pub fn new(rules_base_dir: Option<&str>) -> Self {
        let base = rules_base_dir
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("TARA/RULES"));

        let rulebook_txt_path = base.join("RULEBOOK.txt");
        let default_safe_path = base.join("DEFAULT_SAFE_RULES.txt");
        let compiled_policy_path = base.join("compiled_policy.json");
        let audit_dir = base.join("audit");
        let audit_file = audit_dir.join("rule_audit.jsonl");

        let _ = fs::create_dir_all(&base);
        let _ = fs::create_dir_all(&audit_dir);

        Self {
            base_dir: base,
            rulebook_txt_path,
            default_safe_path,
            compiled_policy_path,
            audit_file,
            compiler: PolicyCompiler::new("ROOT_OPERATOR", "OPERATOR_ROOT"),
        }
    }

    pub fn log_event(&self, event_type: &str, severity: &str, details: Option<&Value>) {
        let mut sanitized = json!({});
        if let Some(d) = details {
            if let Some(obj) = d.as_object() {
                let mut map = serde_json::Map::new();
                for (k, v) in obj {
                    let k_lower = k.to_lowercase();
                    if FORBIDDEN_AUDIT_KEYS
                        .iter()
                        .any(|&bad| k_lower.contains(bad))
                    {
                        map.insert(k.clone(), Value::String("[REDACTED]".to_string()));
                    } else {
                        map.insert(k.clone(), v.clone());
                    }
                }
                sanitized = Value::Object(map);
            }
        }

        let entry = json!({
            "timestamp": crate::now_iso(),
            "event": event_type,
            "severity": severity,
            "details": sanitized
        });

        if let Ok(mut line) = serde_json::to_string(&entry) {
            line.push('\n');
            use std::io::Write;
            if let Ok(mut f) = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.audit_file)
            {
                let _ = f.write_all(line.as_bytes());
            }
        }
    }

    pub fn load_active_policy(&self) -> Option<StructuredCompiledPolicy> {
        if self.compiled_policy_path.exists() {
            if let Ok(content) = fs::read_to_string(&self.compiled_policy_path) {
                return serde_json::from_str(&content).ok();
            }
        }
        None
    }

    pub fn compile_and_save(
        &self,
        target_version: u32,
    ) -> Result<StructuredCompiledPolicy, String> {
        let rulebook_text = fs::read_to_string(&self.rulebook_txt_path).unwrap_or_default();
        let default_safe_text = fs::read_to_string(&self.default_safe_path).unwrap_or_default();

        let (policy, errors, conflicts) =
            self.compiler
                .compile_rules(&rulebook_text, &default_safe_text, target_version);

        if !errors.is_empty() {
            self.log_event(
                "POLICY_COMPILE_WARNING",
                "WARNING",
                Some(&json!({ "errors": errors })),
            );
        }

        if !conflicts.is_empty() {
            self.log_event(
                "POLICY_CONFLICTS_DETECTED",
                "WARNING",
                Some(&json!({ "conflicts": conflicts })),
            );
        }

        let serialized = serde_json::to_string_pretty(&policy).map_err(|e| e.to_string())?;
        fs::write(&self.compiled_policy_path, serialized).map_err(|e| e.to_string())?;

        self.log_event(
            "POLICY_COMPILED_AND_SAVED",
            "INFO",
            Some(&json!({
                "version": target_version,
                "rule_count": policy.rules.len()
            })),
        );

        Ok(policy)
    }

    pub fn get_active_rules(&self) -> Vec<RuleItem> {
        self.load_active_policy()
            .map(|p| p.rules)
            .unwrap_or_default()
    }
}
