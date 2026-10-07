//! Natural language rule parser into structured PolicyRule AST.

use crate::rules::semantic_normalizer::SemanticNormalizer;
use crate::rules::PolicyRule;

pub struct RuleParser {
    normalizer: SemanticNormalizer,
}

impl Default for RuleParser {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleParser {
    pub fn new() -> Self {
        Self {
            normalizer: SemanticNormalizer::new(),
        }
    }

    /// Parse natural language rule sentence into structured PolicyRule.
    pub fn parse(&self, raw_rule: &str) -> Result<PolicyRule, String> {
        let trimmed = raw_rule.trim();
        if trimmed.is_empty() {
            return Err("Empty rule string".into());
        }

        let words = self.normalizer.normalize_sentence(trimmed);
        let mut decision = "ALLOW".to_string();
        let mut action_type = "*".to_string();
        let mut conditions = Vec::new();

        // Check decision
        if words.iter().any(|w| w == "BLOCK") {
            decision = "BLOCK".to_string();
        }

        // Check subject / roles
        let is_creator_focused = words
            .iter()
            .any(|w| w == "creator" || w == "root" || w == "admin");
        let is_user_focused = words
            .iter()
            .any(|w| w == "user" || w == "users" || w == "regular");

        if is_creator_focused {
            conditions.push("creator_verified".to_string());
        } else if is_user_focused {
            conditions.push("role:user".to_string());
        }

        // Identify action targets
        for w in &words {
            if [
                "modify",
                "edit",
                "update",
                "delete",
                "execute",
                "read",
                "model_expansion",
                "system_reconfigure",
            ]
            .contains(&w.as_str())
            {
                action_type = w.clone();
                break;
            }
        }

        Ok(PolicyRule {
            action_type,
            decision,
            conditions,
            reason: trimmed.to_string(),
        })
    }

    /// Parse multi-line rulebook text into a list of structured PolicyRule objects.
    pub fn parse_lines(&self, text: &str) -> Vec<PolicyRule> {
        text.lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .filter_map(|l| self.parse(l).ok())
            .collect()
    }
}
