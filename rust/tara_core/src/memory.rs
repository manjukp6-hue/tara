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

#[derive(Debug, Clone)]
pub struct MemoryEngine {
    records: HashMap<String, MemoryRecord>,
}

impl MemoryEngine {
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
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
