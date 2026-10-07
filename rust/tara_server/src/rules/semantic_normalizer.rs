//! Semantic Normalizer for Governance and Access Rules.
//!
//! Normalizes natural language synonyms for actions, targets, and permissions.

use std::collections::HashMap;

pub struct SemanticNormalizer {
    synonyms: HashMap<String, String>,
}

impl Default for SemanticNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticNormalizer {
    pub fn new() -> Self {
        let mut map = HashMap::new();
        // Permission verbs
        map.insert("allow".into(), "ALLOW".into());
        map.insert("permit".into(), "ALLOW".into());
        map.insert("grant".into(), "ALLOW".into());
        map.insert("enable".into(), "ALLOW".into());
        map.insert("block".into(), "BLOCK".into());
        map.insert("deny".into(), "BLOCK".into());
        map.insert("prohibit".into(), "BLOCK".into());
        map.insert("forbid".into(), "BLOCK".into());
        map.insert("restrict".into(), "BLOCK".into());

        // Action nouns/verbs
        map.insert("modify".into(), "modify".into());
        map.insert("edit".into(), "modify".into());
        map.insert("update".into(), "modify".into());
        map.insert("change".into(), "modify".into());
        map.insert("delete".into(), "delete".into());
        map.insert("remove".into(), "delete".into());
        map.insert("destroy".into(), "delete".into());
        map.insert("execute".into(), "execute".into());
        map.insert("run".into(), "execute".into());
        map.insert("trigger".into(), "execute".into());
        map.insert("read".into(), "read".into());
        map.insert("view".into(), "read".into());
        map.insert("inspect".into(), "read".into());

        Self { synonyms: map }
    }

    pub fn normalize_token(&self, token: &str) -> String {
        let lower = token.trim().to_lowercase();
        self.synonyms.get(&lower).cloned().unwrap_or(lower)
    }

    pub fn normalize_sentence(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|word| {
                let cleaned = word.trim_matches(|c: char| !c.is_alphanumeric());
                self.normalize_token(cleaned)
            })
            .collect()
    }
}
