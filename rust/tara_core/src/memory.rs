//! memory.rs
//!
//! Structured Memory Subsystem for TARA Core.
//! Preserves short-term, long-term, episodic, procedural, research, and preference memories.
//! ALL memories remain strictly external to neural model weights.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryType {
    ShortTerm,
    LongTerm,
    Task,
    Episodic,
    Procedural,
    Research,
    Preference,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: String,
    pub memory_type: MemoryType,
    pub key: String,
    pub content: String,
    pub timestamp: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Default)]
pub struct MemoryEngine {
    records: HashMap<String, MemoryRecord>,
}

impl MemoryEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn store(&mut self, record: MemoryRecord) {
        self.records.insert(record.key.clone(), record);
    }

    pub fn retrieve(&self, key: &str) -> Option<&MemoryRecord> {
        self.records.get(key)
    }

    pub fn query_by_type(&self, mem_type: MemoryType) -> Vec<&MemoryRecord> {
        self.records
            .values()
            .filter(|r| r.memory_type == mem_type)
            .collect()
    }

    pub fn count(&self) -> usize {
        self.records.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_engine_crud_and_query() {
        let mut engine = MemoryEngine::new();
        assert_eq!(engine.count(), 0);

        engine.store(MemoryRecord {
            id: "rec_1".into(),
            memory_type: MemoryType::Episodic,
            key: "interaction_user_1".into(),
            content: "User requested system audit".into(),
            timestamp: "2026-10-04T12:00:00Z".into(),
            confidence: 0.98,
        });

        engine.store(MemoryRecord {
            id: "rec_2".into(),
            memory_type: MemoryType::Preference,
            key: "pref_response_lang".into(),
            content: "Kannada language preferred".into(),
            timestamp: "2026-10-04T12:01:00Z".into(),
            confidence: 1.0,
        });

        assert_eq!(engine.count(), 2);

        // Retrieve existing key
        let retrieved = engine.retrieve("interaction_user_1").unwrap();
        assert_eq!(retrieved.content, "User requested system audit");
        assert_eq!(retrieved.confidence, 0.98);

        // Retrieve non-existent key
        assert!(engine.retrieve("non_existent_key").is_none());

        // Query by type
        let episodic = engine.query_by_type(MemoryType::Episodic);
        assert_eq!(episodic.len(), 1);
        assert_eq!(episodic[0].key, "interaction_user_1");

        let preferences = engine.query_by_type(MemoryType::Preference);
        assert_eq!(preferences.len(), 1);
        assert_eq!(preferences[0].key, "pref_response_lang");

        let task_mems = engine.query_by_type(MemoryType::Task);
        assert_eq!(task_mems.len(), 0);
    }
}
