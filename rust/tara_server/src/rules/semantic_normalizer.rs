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
        map.insert("never".into(), "BLOCK".into());
        map.insert("cannot".into(), "BLOCK".into());
        map.insert("prohibited".into(), "BLOCK".into());
        map.insert("forbidden".into(), "BLOCK".into());

        map.insert("modify".into(), "modify".into());
        map.insert("modifying".into(), "modify".into());
        map.insert("edit".into(), "modify".into());
        map.insert("editing".into(), "modify".into());
        map.insert("update".into(), "modify".into());
        map.insert("updating".into(), "modify".into());
        map.insert("change".into(), "modify".into());
        map.insert("changing".into(), "modify".into());
        map.insert("delete".into(), "delete".into());
        map.insert("deleting".into(), "delete".into());
        map.insert("remove".into(), "delete".into());
        map.insert("removing".into(), "delete".into());
        map.insert("destroy".into(), "delete".into());
        map.insert("destroying".into(), "delete".into());
        map.insert("execute".into(), "execute".into());
        map.insert("executing".into(), "execute".into());
        map.insert("run".into(), "execute".into());
        map.insert("running".into(), "execute".into());
        map.insert("trigger".into(), "execute".into());
        map.insert("triggering".into(), "execute".into());
        map.insert("read".into(), "read".into());
        map.insert("reading".into(), "read".into());
        map.insert("view".into(), "read".into());
        map.insert("viewing".into(), "read".into());
        map.insert("inspect".into(), "read".into());
        map.insert("inspecting".into(), "read".into());

        Self { synonyms: map }
    }

    pub fn normalize_token(&self, token: &str) -> String {
        let lower = token.trim().to_lowercase();
        self.synonyms.get(&lower).cloned().unwrap_or(lower)
    }

    pub fn normalize_sentence(&self, text: &str) -> Vec<String> {
        let pre_normalized = text
            .to_lowercase()
            .replace("do not", "block")
            .replace("must not", "block")
            .replace("shall not", "block")
            .replace("model expansion", "model_expansion")
            .replace("system reconfigure", "system_reconfigure");
        pre_normalized
            .split_whitespace()
            .map(|word| {
                let cleaned = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
                self.normalize_token(cleaned)
            })
            .filter(|w| !w.is_empty())
            .collect()
    }
}
