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

        // Identify action targets - specific compound targets take precedence over generic verbs
        let specific_targets = ["model_expansion", "system_reconfigure"];
        let generic_actions = ["modify", "edit", "update", "delete", "execute", "read"];

        if let Some(target) = words.iter().find(|w| specific_targets.contains(&w.as_str())) {
            action_type = target.clone();
        } else if let Some(target) = words.iter().find(|w| generic_actions.contains(&w.as_str())) {
            action_type = target.clone();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_parser_compound_actions_and_synonyms() {
        let parser = RuleParser::new();

        // 1. Compound target model_expansion with "must not" and "trigger"
        let rule1 = parser
            .parse("Users must not trigger model expansion")
            .expect("must parse successfully");
        assert_eq!(rule1.action_type, "model_expansion");
        assert_eq!(rule1.decision, "BLOCK");
        assert!(rule1.conditions.contains(&"role:user".to_string()));

        // 2. Compound target system_reconfigure with "cannot"
        let rule2 = parser
            .parse("Regular users cannot system reconfigure")
            .expect("must parse successfully");
        assert_eq!(rule2.action_type, "system_reconfigure");
        assert_eq!(rule2.decision, "BLOCK");
        assert!(rule2.conditions.contains(&"role:user".to_string()));

        // 3. Creator permission with "admin" and "modify"
        let rule3 = parser
            .parse("Admin can modify system")
            .expect("must parse successfully");
        assert_eq!(rule3.action_type, "modify");
        assert_eq!(rule3.decision, "ALLOW");
        assert!(rule3.conditions.contains(&"creator_verified".to_string()));

        // 4. "never" and "prohibited" synonyms
        let rule4 = parser
            .parse("Users never delete records")
            .expect("must parse successfully");
        assert_eq!(rule4.action_type, "delete");
        assert_eq!(rule4.decision, "BLOCK");

        let rule5 = parser
            .parse("Editing files is prohibited for users")
            .expect("must parse successfully");
        assert_eq!(rule5.action_type, "modify");
        assert_eq!(rule5.decision, "BLOCK");
    }
}
