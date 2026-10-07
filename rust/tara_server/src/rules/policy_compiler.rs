//! Main Policy Compiler.
//! Compiles RULEBOOK.txt + DEFAULT_SAFE_RULES.txt into a verified StructuredCompiledPolicy object.

use super::language_detector::LanguageDetector;
use super::policy_schema::{PolicyPriority, RuleItem, StructuredCompiledPolicy};
use super::rule_parser::RuleParser;
use super::validator::PolicyValidator;
use std::collections::HashMap;

pub const CANONICAL_CREATOR_ID: &str = "ROOT_OPERATOR";
pub const DEFAULT_DISPLAY_NAME: &str = "OPERATOR_ROOT";

pub struct PolicyCompiler {
    pub creator_id: String,
    pub display_name: String,
}

impl PolicyCompiler {
    pub fn new(creator_id: &str, display_name: &str) -> Self {
        Self {
            creator_id: creator_id.to_string(),
            display_name: display_name.to_string(),
        }
    }

    pub fn compile_rules(
        &self,
        rulebook_text: &str,
        default_safe_text: &str,
        target_version: u32,
    ) -> (StructuredCompiledPolicy, Vec<String>, Vec<String>) {
        let mut policy =
            StructuredCompiledPolicy::new(&self.creator_id, &self.display_name, target_version);
        let mut validation_errors = Vec::new();
        let conflict_reports = Vec::new();

        let parser = RuleParser::new();
        let now_str = crate::now_iso();

        // 1. Parse Baseline Mandatory Safe Rules
        let baseline_rules = parser.parse_lines(default_safe_text);
        let mut mandatory_rules = Vec::new();

        for (idx, r) in baseline_rules.into_iter().enumerate() {
            let rule_id = format!("RULE-BASE-{:03}", idx + 1);
            let lang = LanguageDetector::detect_language(&r.reason);

            let priority = if r.decision == "BLOCK" {
                PolicyPriority::MandatoryLawSafety as u32
            } else {
                PolicyPriority::TaraSecurity as u32
            };

            mandatory_rules.push(RuleItem {
                rule_id,
                version: target_version,
                category: "MANDATORY_LAW".to_string(),
                meaning: r.action_type.clone(),
                priority,
                scope: "GLOBAL".to_string(),
                action: r.decision.clone(),
                conditions: HashMap::new(),
                status: "ACTIVE".to_string(),
                original_text: r.reason.clone(),
                language: lang.language,
                is_mandatory: true,
                created_at: now_str.clone(),
                updated_at: now_str.clone(),
            });
        }

        // 2. Parse Creator Custom Rules
        let creator_rules_parsed = parser.parse_lines(rulebook_text);
        let mut creator_rules = Vec::new();

        for (idx, r) in creator_rules_parsed.into_iter().enumerate() {
            let (is_valid, msg, status) =
                PolicyValidator::validate_rule_candidate(&r.reason, false);
            if !is_valid {
                validation_errors.push(format!("Line {}: {} ('{}')", idx + 1, msg, r.reason));
                continue;
            }

            let rule_id = format!("RULE-CR-{:03}", idx + 1);
            let lang = LanguageDetector::detect_language(&r.reason);

            creator_rules.push(RuleItem {
                rule_id,
                version: target_version,
                category: "CREATOR_CUSTOM".to_string(),
                meaning: r.action_type.clone(),
                priority: PolicyPriority::CreatorRule as u32,
                scope: "GLOBAL".to_string(),
                action: r.decision.clone(),
                conditions: HashMap::new(),
                status,
                original_text: r.reason.clone(),
                language: lang.language,
                is_mandatory: false,
                created_at: now_str.clone(),
                updated_at: now_str.clone(),
            });
        }

        // 3. Resolve conflicts
        let mut all_rules = mandatory_rules.clone();
        for cr in creator_rules {
            all_rules.push(cr);
        }

        // 4. Verify baseline integrity
        let (intact, missing) =
            PolicyValidator::verify_baseline_integrity(&mandatory_rules, &all_rules);
        if !intact {
            for m in missing {
                validation_errors.push(m);
            }
        }

        policy.rules = all_rules;
        (policy, validation_errors, conflict_reports)
    }
}
