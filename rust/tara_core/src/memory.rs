//! memory.rs
//!
//! Structured Memory Subsystem for TARA Core.
//! Preserves short-term, long-term, episodic, procedural, research, and preference memories.
//! ALL memories remain strictly external to neural model weights.
//!
//! # Architecture & Ownership Boundary
//! - **Scope**: In-crate key-value memory records and metadata structures.
//! - **Authoritative Persistent Neural Memory Engine**: Production semantic search, vector storage,
//!   persistent SQLite/RocksDB/file backends, and multi-tier memory synthesis are authoritatively owned
//!   by `tara_server::memory::MemoryEngine`.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

pub const DEFAULT_MAX_MEMORY_RECORDS: usize = 10_000;

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

#[derive(Debug, Clone)]
pub struct CoreMemoryEngine {
    records: HashMap<String, MemoryRecord>,
    order: VecDeque<String>,
    max_records: usize,
}

impl Default for CoreMemoryEngine {
    fn default() -> Self {
        Self {
            records: HashMap::new(),
            order: VecDeque::new(),
            max_records: DEFAULT_MAX_MEMORY_RECORDS,
        }
    }
}

pub type MemoryEngine = CoreMemoryEngine;

impl CoreMemoryEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_records(max_records: usize) -> Self {
        Self {
            records: HashMap::new(),
            order: VecDeque::new(),
            max_records: max_records.max(1),
        }
    }

    pub fn store(&mut self, mut record: MemoryRecord) {
        if record.confidence.is_nan() {
            record.confidence = 0.0;
        } else {
            record.confidence = record.confidence.clamp(0.0, 1.0);
        }

        let key = record.key.clone();
        if !self.records.contains_key(&key) {
            if self.records.len() >= self.max_records {
                while let Some(evicted_key) = self.order.pop_front() {
                    if self.records.remove(&evicted_key).is_some() {
                        break;
                    }
                }
            }
            self.order.push_back(key.clone());
        }
        self.records.insert(key, record);
    }

    pub fn retrieve(&self, key: &str) -> Option<&MemoryRecord> {
        self.records.get(key)
    }

    pub fn remove(&mut self, key: &str) -> Option<MemoryRecord> {
        let removed = self.records.remove(key);
        if removed.is_some() {
            self.order.retain(|k| k != key);
        }
        removed
    }

    pub fn clear(&mut self) {
        self.records.clear();
        self.order.clear();
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

        // Remove key
        let removed = engine.remove("interaction_user_1");
        assert!(removed.is_some());
        assert_eq!(engine.count(), 1);
        assert!(engine.retrieve("interaction_user_1").is_none());

        // Clear all
        engine.clear();
        assert_eq!(engine.count(), 0);
    }

    #[test]
    fn test_memory_engine_fifo_eviction_and_confidence_clamping() {
        let mut engine = MemoryEngine::with_max_records(2);

        // Clamping check: > 1.0 clamped to 1.0, < 0.0 clamped to 0.0, NaN handled
        engine.store(MemoryRecord {
            id: "1".into(),
            memory_type: MemoryType::ShortTerm,
            key: "k1".into(),
            content: "c1".into(),
            timestamp: "2026-10-04T12:00:00Z".into(),
            confidence: 1.5,
        });
        assert_eq!(engine.retrieve("k1").unwrap().confidence, 1.0);

        engine.store(MemoryRecord {
            id: "2".into(),
            memory_type: MemoryType::ShortTerm,
            key: "k2".into(),
            content: "c2".into(),
            timestamp: "2026-10-04T12:00:00Z".into(),
            confidence: f32::NAN,
        });
        assert_eq!(engine.retrieve("k2").unwrap().confidence, 0.0);

        // Exceed capacity: k1 should be evicted first (FIFO)
        engine.store(MemoryRecord {
            id: "3".into(),
            memory_type: MemoryType::ShortTerm,
            key: "k3".into(),
            content: "c3".into(),
            timestamp: "2026-10-04T12:00:00Z".into(),
            confidence: -0.5,
        });
        assert_eq!(engine.count(), 2);
        assert!(
            engine.retrieve("k1").is_none(),
            "k1 should have been evicted via FIFO"
        );
        assert!(engine.retrieve("k2").is_some());
        assert_eq!(engine.retrieve("k3").unwrap().confidence, 0.0);
    }
}
