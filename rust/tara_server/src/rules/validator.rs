//! Policy Safety Validator.
//! Inspects parsed rule candidates for:
//! - Ambiguity
//! - Privilege escalation attempts
//! - Attempts to disable security
//! - Attempts to override mandatory baseline protections

use super::policy_schema::RuleItem;
use std::collections::HashSet;

pub struct PolicyValidator;

impl PolicyValidator {
    pub fn validate_rule_candidate(
        original_text: &str,
        is_ambiguous: bool,
    ) -> (bool, String, String) {
        let text_lower = original_text.to_lowercase();

        // 1. Detect attempts to override mandatory protections
        let override_law = [
            "ignore law",
            "ignore government",
            "bypass government",
            "allow csam",
            "allow child porn",
            "allow nude",
            "permit nudity",
            "disable safety",
        ];
        if override_law.iter().any(|&w| text_lower.contains(w)) {
            return (
                false,
                "RULE_REJECTED: Attempt to override mandatory law or safety baseline".to_string(),
                "REJECTED".to_string(),
            );
        }

        // 2. Detect privilege escalation attempts
        let escalate = [
            "grant me root",
            "become creator",
            "grant admin",
            "override creator",
            "bypass creator",
            "disable lock",
            "disable lockdown",
        ];
        if escalate.iter().any(|&w| text_lower.contains(w)) {
            return (
                false,
                "RULE_REJECTED: Unauthorized privilege escalation or security disablement attempt"
                    .to_string(),
                "REJECTED".to_string(),
            );
        }

        // 3. Detect ambiguous blanket statements
        if is_ambiguous {
            return (
                true,
                "RULE_AMBIGUOUS: Instruction requires Creator clarification before activation"
                    .to_string(),
                "AMBIGUOUS".to_string(),
            );
        }

        // 4. Valid active rule
        (true, "RULE_VALID".to_string(), "ACTIVE".to_string())
    }

    pub fn verify_baseline_integrity(
        mandatory_rules: &[RuleItem],
        active_rules: &[RuleItem],
    ) -> (bool, Vec<String>) {
        let active_meanings: HashSet<&str> =
            active_rules.iter().map(|r| r.meaning.as_str()).collect();
        let mut missing = Vec::new();

        for m in mandatory_rules {
            if !active_meanings.contains(m.meaning.as_str()) {
                missing.push(format!("Missing mandatory baseline rule: {}", m.meaning));
            }
        }

        let is_intact = missing.is_empty();
        (is_intact, missing)
    }
}
