//! Global knowledge base: stores, queries, and manages knowledge entries.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use serde_json::{json, Value};
use thiserror::Error;
use sha2::{Digest, Sha256};

#[derive(Debug, Error)]
pub enum KnowledgeError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Other(String),
}

/// Global knowledge base persisted to disk.
pub struct GlobalKnowledgeBase {
    pub base_dir: String,
    pub candidates_dir: String,
}

impl GlobalKnowledgeBase {
    pub fn new(base_dir: &str) -> Self {
        let candidates_dir = format!("{}/candidates", base_dir);
        let _ = fs::create_dir_all(base_dir);
        let _ = fs::create_dir_all(&candidates_dir);
        Self { base_dir: base_dir.to_string(), candidates_dir }
    }

    /// Query knowledge entries matching a text query.
    pub fn query_knowledge(&self, query: &str, topic: Option<&str>) -> Vec<Value> {
        let mut results = Vec::new();
        let query_lc = query.to_lowercase();

        if let Ok(rd) = fs::read_dir(&self.base_dir) {
            for entry in rd.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("json")
                    && path.file_name().and_then(|n| n.to_str()) != Some("index.json") {
                    if let Ok(raw) = fs::read_to_string(&path) {
                        if let Ok(e) = serde_json::from_str::<Value>(&raw) {
                            if let Some(t) = topic {
                                if e.get("topic").and_then(|v| v.as_str()) != Some(t) {
                                    continue;
                                }
                            }
                            let text = e.to_string().to_lowercase();
                            if text.contains(&query_lc) {
                                results.push(e);
                            }
                        }
                    }
                }
            }
        }
        results
    }

    /// Store or update a knowledge entry.
    pub fn store_or_update_knowledge(
        &self,
        topic: &str,
        subject: &str,
        content: &str,
        confidence: f32,
    ) -> Value {
        let id = hex::encode(Sha256::digest(format!("{}{}", topic, subject).as_bytes()))[..16].to_string();
        let entry = json!({
            "id": id,
            "topic": topic,
            "subject": subject,
            "content": content,
            "confidence": confidence,
            "verification_status": "PENDING",
            "learned_at": crate::now_iso()
        });
        let path = format!("{}/{}.json", self.base_dir, id);
        let _ = fs::write(&path, serde_json::to_string_pretty(&entry).unwrap_or_default());
        entry
    }

    /// List up to `limit` knowledge entries.
    pub fn list_entries(&self, limit: usize) -> Vec<Value> {
        let mut results = Vec::new();
        if let Ok(rd) = fs::read_dir(&self.base_dir) {
            for entry in rd.flatten().take(limit) {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("json")
                    && path.file_name().and_then(|n| n.to_str()) != Some("index.json") {
                    if let Ok(raw) = fs::read_to_string(&path) {
                        if let Ok(e) = serde_json::from_str::<Value>(&raw) {
                            results.push(e);
                        }
                    }
                }
            }
        }
        results
    }

    /// List candidate knowledge entries.
    pub fn list_candidates(&self, status: Option<&str>) -> Vec<Value> {
        let mut results = Vec::new();
        if let Ok(rd) = fs::read_dir(&self.candidates_dir) {
            for entry in rd.flatten() {
                if let Ok(raw) = fs::read_to_string(entry.path()) {
                    if let Ok(e) = serde_json::from_str::<Value>(&raw) {
                        if let Some(s) = status {
                            if e.get("status").and_then(|v| v.as_str()) != Some(s) {
                                continue;
                            }
                        }
                        results.push(e);
                    }
                }
            }
        }
        results
    }

    /// Approve a candidate knowledge entry.
    pub fn approve_candidate(&self, candidate_id: &str, creator_id: &str) -> Result<Value, String> {
        let cand_path = format!("{}/{}.json", self.candidates_dir, candidate_id);
        let raw = fs::read_to_string(&cand_path)
            .map_err(|e| format!("Candidate '{}' not found: {}", candidate_id, e))?;
        let mut entry: Value = serde_json::from_str(&raw)
            .map_err(|e| format!("Parse error: {}", e))?;

        if let Some(obj) = entry.as_object_mut() {
            obj.insert("status".to_string(), json!("APPROVED"));
            obj.insert("approved_by".to_string(), json!(creator_id));
            obj.insert("approved_at".to_string(), json!(crate::now_iso()));
        }

        let approved_path = format!("{}/{}.json", self.base_dir, candidate_id);
        fs::write(&approved_path, serde_json::to_string_pretty(&entry).unwrap_or_default())
            .map_err(|e| e.to_string())?;
        let _ = fs::remove_file(&cand_path);

        Ok(entry)
    }

    /// Get a specific knowledge entry by ID.
    pub fn get_knowledge(&self, id: &str) -> Option<Value> {
        let path = format!("{}/{}.json", self.base_dir, id);
        fs::read_to_string(&path)
            .ok()
            .and_then(|raw| serde_json::from_str(&raw).ok())
    }
}
