//! Memory engine: episodic memory storage, deduplication, and indexed semantic retrieval.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::Digest;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::{Arc, RwLock};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Lightweight index entry for an episode to enable fast semantic/lexical lookup
/// without reading every episode JSON file from disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeIndexEntry {
    pub episode_id: String,
    pub actor_id: String,
    pub intent: String,
    pub action: String,
    pub outcome: String,
    pub timestamp: u128,
    pub relative_path: String,
    pub content_hash: String,
    pub feature_vector: Vec<f32>,
    pub terms: Vec<String>,
}

/// In-memory and persisted index for scalable memory lookup.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MemoryIndex {
    pub entries: Vec<EpisodeIndexEntry>,
    pub content_hashes: HashMap<String, String>, // content_hash -> episode_id
}

/// Compute a 16-dimensional deterministic L2-normalized feature vector from text.
pub fn compute_feature_vector_16(text: &str) -> Vec<f32> {
    let mut vec = vec![0.0f32; 16];
    let tokens: Vec<&str> = text
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    if tokens.is_empty() {
        return vec;
    }

    for token in &tokens {
        let lower = token.to_lowercase();
        let mut hasher = sha2::Sha256::new();
        hasher.update(lower.as_bytes());
        let hash = hasher.finalize();
        let bucket = (hash[0] as usize) % 16;
        vec[bucket] += 1.0;

        // Character 3-grams for morphological/semantic similarity
        if lower.len() >= 3 {
            for window in lower.as_bytes().windows(3) {
                let w_bucket = (window[0] as usize
                    ^ (window[1] as usize).rotate_left(2)
                    ^ (window[2] as usize).rotate_left(4))
                    % 16;
                vec[w_bucket] += 0.3;
            }
        }
    }

    // L2 normalize
    let sum_sq: f32 = vec.iter().map(|x| x * x).sum();
    if sum_sq > 1e-9 {
        let norm = sum_sq.sqrt();
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }

    vec
}

/// Cosine similarity between two 16-dimensional vectors.
pub fn cosine_similarity_16(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != 16 || b.len() != 16 {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot.clamp(0.0, 1.0)
}

/// Parameters for recording a memory episode.
pub struct EpisodeRecordParams<'a> {
    pub actor_id: &'a str,
    pub intent: &'a str,
    pub action: &'a str,
    pub parameters: &'a Value,
    pub outcome: &'a str,
    pub observations: &'a Value,
    pub error: Option<&'a str>,
    pub reflection: &'a str,
}

/// Extract searchable terms from text.
fn extract_terms(text: &str) -> Vec<String> {
    let mut set = HashSet::new();
    for word in text.split(|c: char| !c.is_alphanumeric() && c != '_') {
        let w = word.trim().to_lowercase();
        if w.len() >= 2 {
            set.insert(w);
        }
    }
    set.into_iter().collect()
}

/// Manages episodic memory: records, deduplicates, and semantically queries interaction episodes.
pub struct MemoryEngine {
    pub memory_dir: String,
    index: Arc<RwLock<MemoryIndex>>,
}

impl MemoryEngine {
    pub fn new(memory_dir: &str) -> Self {
        let _ = fs::create_dir_all(format!("{}/episodes", memory_dir));
        let index_path = format!("{}/episode_index.json", memory_dir);

        let initial_index = if Path::new(&index_path).exists() {
            fs::read_to_string(&index_path)
                .ok()
                .and_then(|data| serde_json::from_str::<MemoryIndex>(&data).ok())
                .unwrap_or_else(|| Self::rebuild_index_from_disk(memory_dir))
        } else {
            Self::rebuild_index_from_disk(memory_dir)
        };

        Self {
            memory_dir: memory_dir.to_string(),
            index: Arc::new(RwLock::new(initial_index)),
        }
    }

    /// Rebuild index from existing episodes on disk.
    fn rebuild_index_from_disk(memory_dir: &str) -> MemoryIndex {
        let mut index = MemoryIndex::default();
        let ep_dir = format!("{}/episodes", memory_dir);
        if let Ok(actors) = fs::read_dir(&ep_dir) {
            for actor_entry in actors.flatten() {
                if actor_entry.path().is_dir() {
                    let actor_id = actor_entry.file_name().to_string_lossy().to_string();
                    if let Ok(files) = fs::read_dir(actor_entry.path()) {
                        for file in files.flatten() {
                            if file.path().extension().and_then(|e| e.to_str()) == Some("json") {
                                if let Ok(raw) = fs::read_to_string(file.path()) {
                                    if let Ok(ep) = serde_json::from_str::<Value>(&raw) {
                                        let ep_id = ep
                                            .get("episode_id")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string();
                                        let intent = ep
                                            .get("intent")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string();
                                        let action = ep
                                            .get("action")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string();
                                        let outcome = ep
                                            .get("outcome")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string();
                                        let reflection = ep
                                            .get("reflection")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string();
                                        let ts = ep
                                            .get("timestamp")
                                            .and_then(|v| v.as_u64())
                                            .unwrap_or(0)
                                            as u128;
                                        let params =
                                            ep.get("parameters").cloned().unwrap_or(json!({}));

                                        let content_key = format!(
                                            "{}:{}:{}:{}:{}",
                                            actor_id, intent, action, params, outcome
                                        );
                                        let content_hash = hex::encode(sha2::Sha256::digest(
                                            content_key.as_bytes(),
                                        ));

                                        let text_for_vector = format!(
                                            "{} {} {} {}",
                                            intent, action, outcome, reflection
                                        );
                                        let feature_vector =
                                            compute_feature_vector_16(&text_for_vector);
                                        let terms = extract_terms(&text_for_vector);

                                        let rel_path = format!("{}/{}.json", actor_id, ep_id);

                                        index
                                            .content_hashes
                                            .insert(content_hash.clone(), ep_id.clone());
                                        index.entries.push(EpisodeIndexEntry {
                                            episode_id: ep_id,
                                            actor_id: actor_id.clone(),
                                            intent,
                                            action,
                                            outcome,
                                            timestamp: ts,
                                            relative_path: rel_path,
                                            content_hash,
                                            feature_vector,
                                            terms,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Persist rebuilt index
        let index_path = format!("{}/episode_index.json", memory_dir);
        if let Ok(serialized) = serde_json::to_string_pretty(&index) {
            let _ = fs::write(index_path, serialized);
        }

        index
    }

    /// Save index to disk.
    fn persist_index(&self) {
        if let Ok(index) = self.index.read() {
            let index_path = format!("{}/episode_index.json", self.memory_dir);
            if let Ok(serialized) = serde_json::to_string_pretty(&*index) {
                let _ = fs::write(index_path, serialized);
            }
        }
    }

    /// Record a new episode to disk with content deduplication and semantic indexing.
    pub fn record_episode(&self, p: EpisodeRecordParams<'_>) -> Result<String, MemoryError> {
        // Content deduplication check
        let content_key = format!(
            "{}:{}:{}:{}:{}",
            p.actor_id, p.intent, p.action, p.parameters, p.outcome
        );
        let content_hash = hex::encode(sha2::Sha256::digest(content_key.as_bytes()));

        {
            let index_read = self.index.read().unwrap();
            if let Some(existing_id) = index_read.content_hashes.get(&content_hash) {
                // Duplicate episode content: return existing episode ID without duplicating storage
                return Ok(existing_id.clone());
            }
        }

        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let episode_id = format!("ep_{}", ts);

        let episode = json!({
            "episode_id": episode_id,
            "actor_id": p.actor_id,
            "intent": p.intent,
            "action": p.action,
            "parameters": p.parameters,
            "outcome": p.outcome,
            "observations": p.observations,
            "error": p.error,
            "reflection": p.reflection,
            "timestamp": ts
        });

        let ep_dir = format!("{}/episodes/{}", self.memory_dir, p.actor_id);
        fs::create_dir_all(&ep_dir)?;
        let path = format!("{}/{}.json", ep_dir, episode_id);
        fs::write(&path, serde_json::to_string_pretty(&episode)?)?;

        let rel_path = format!("{}/{}.json", p.actor_id, episode_id);
        let text_for_vector = format!("{} {} {} {}", p.intent, p.action, p.outcome, p.reflection);
        let feature_vector = compute_feature_vector_16(&text_for_vector);
        let terms = extract_terms(&text_for_vector);

        let entry = EpisodeIndexEntry {
            episode_id: episode_id.clone(),
            actor_id: p.actor_id.to_string(),
            intent: p.intent.to_string(),
            action: p.action.to_string(),
            outcome: p.outcome.to_string(),
            timestamp: ts,
            relative_path: rel_path,
            content_hash: content_hash.clone(),
            feature_vector,
            terms,
        };

        {
            let mut index_write = self.index.write().unwrap();
            index_write
                .content_hashes
                .insert(content_hash, episode_id.clone());
            index_write.entries.push(entry);
        }

        self.persist_index();

        Ok(episode_id)
    }

    /// Query episodes using scalable semantic vector and lexical ranking over the in-memory index.
    /// Only the top matching candidate files are loaded from disk.
    pub fn query_episodes(
        &self,
        query: Option<&str>,
        actor_id: Option<&str>,
        outcome_filter: Option<&str>,
        limit: usize,
        _scope: &str,
    ) -> Vec<Value> {
        let index = self.index.read().unwrap();
        if index.entries.is_empty() {
            return Vec::new();
        }

        let mut scored_entries: Vec<(&EpisodeIndexEntry, f32)> = Vec::new();

        if let Some(q) = query {
            let q_trimmed = q.trim();
            if !q_trimmed.is_empty() {
                let q_vec = compute_feature_vector_16(q_trimmed);
                let q_terms = extract_terms(q_trimmed);

                for entry in &index.entries {
                    if let Some(aid) = actor_id {
                        if entry.actor_id != aid {
                            continue;
                        }
                    }
                    if let Some(of) = outcome_filter {
                        if entry.outcome != of {
                            continue;
                        }
                    }

                    let cosine = cosine_similarity_16(&q_vec, &entry.feature_vector);
                    let mut overlap = 0usize;
                    for qt in &q_terms {
                        if entry
                            .terms
                            .iter()
                            .any(|t| t == qt || t.contains(qt) || qt.contains(t))
                        {
                            overlap += 1;
                        }
                    }
                    let lexical_score = if q_terms.is_empty() {
                        0.0
                    } else {
                        overlap as f32 / q_terms.len() as f32
                    };

                    let total_score = 0.6 * cosine + 0.4 * lexical_score;
                    if total_score > 0.05 || (q_terms.is_empty() && cosine > 0.0) {
                        scored_entries.push((entry, total_score));
                    }
                }

                scored_entries
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        }

        // If no query or no query matches found, fallback to filtering by actor/outcome ordered by recency
        if scored_entries.is_empty() && query.is_none() {
            for entry in index.entries.iter().rev() {
                if let Some(aid) = actor_id {
                    if entry.actor_id != aid {
                        continue;
                    }
                }
                if let Some(of) = outcome_filter {
                    if entry.outcome != of {
                        continue;
                    }
                }
                scored_entries.push((entry, 1.0));
                if scored_entries.len() >= limit {
                    break;
                }
            }
        }

        // Load only the top K matched episodes from disk
        let mut results = Vec::new();
        for (entry, _) in scored_entries.into_iter().take(limit) {
            let path = format!("{}/episodes/{}", self.memory_dir, entry.relative_path);
            if let Ok(raw) = fs::read_to_string(&path) {
                if let Ok(val) = serde_json::from_str::<Value>(&raw) {
                    results.push(val);
                }
            }
        }

        results
    }

    /// Store a proven multi-step procedure.
    pub fn store_procedure(
        &self,
        task_key: &str,
        steps: &[Value],
        description: Option<&str>,
    ) -> Result<Value, MemoryError> {
        let proc_id = format!(
            "proc_{}",
            &hex::encode(sha2::Sha256::digest(task_key.as_bytes()))[..12]
        );
        let procedure = json!({
            "proc_id": proc_id,
            "task_key": task_key,
            "description": description.unwrap_or(task_key),
            "steps": steps,
            "updated_at": crate::now_iso()
        });

        let procs_dir = format!("{}/procedures", self.memory_dir);
        fs::create_dir_all(&procs_dir)?;
        let path = format!("{}/{}.json", procs_dir, proc_id);
        fs::write(&path, serde_json::to_string_pretty(&procedure)?)?;

        Ok(procedure)
    }

    /// Retrieve a procedure by task_key.
    pub fn get_procedure(&self, task_key: &str) -> Option<Value> {
        let procs_dir = format!("{}/procedures", self.memory_dir);
        if let Ok(entries) = fs::read_dir(&procs_dir) {
            for entry in entries.flatten() {
                if entry.path().extension().and_then(|e| e.to_str()) == Some("json") {
                    if let Ok(raw) = fs::read_to_string(entry.path()) {
                        if let Ok(p) = serde_json::from_str::<Value>(&raw) {
                            if p.get("task_key").and_then(|v| v.as_str()) == Some(task_key) {
                                return Some(p);
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Auto-migrates legacy flat memory files to structured directories.
    pub fn auto_migrate_legacy_data(&self) -> usize {
        let legacy_file = format!("{}/episodes.jsonl", self.memory_dir);
        let mut count = 0;
        if Path::new(&legacy_file).exists() {
            if let Ok(raw) = fs::read_to_string(&legacy_file) {
                for line in raw.lines() {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
                        let actor = v
                            .get("actor_id")
                            .and_then(Value::as_str)
                            .unwrap_or("default");
                        let ep_id = v
                            .get("episode_id")
                            .and_then(Value::as_str)
                            .unwrap_or("ep_migrated");
                        let target_dir = format!("{}/episodes/{}", self.memory_dir, actor);
                        let _ = fs::create_dir_all(&target_dir);
                        let _ = fs::write(
                            format!("{}/{}.json", target_dir, ep_id),
                            serde_json::to_string_pretty(&v).unwrap_or_default(),
                        );
                        count += 1;
                    }
                }
            }
        }
        if count > 0 {
            let rebuilt = Self::rebuild_index_from_disk(&self.memory_dir);
            *self.index.write().unwrap() = rebuilt;
        }
        count
    }

    /// Return memory statistics in O(1) time using the in-memory index.
    pub fn get_memory_stats(&self) -> Value {
        let total = self.index.read().map(|idx| idx.entries.len()).unwrap_or(0);
        json!({ "total_episodes": total, "memory_dir": self.memory_dir })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_engine_record_and_query() {
        let tmp = std::env::temp_dir().join(format!("tara_mem_test_{}", uuid::Uuid::new_v4()));
        let engine = MemoryEngine::new(tmp.to_str().unwrap());

        let ep_id = engine
            .record_episode(EpisodeRecordParams {
                actor_id: "creator",
                intent: "CODE_INSPECTION",
                action: "AUDIT",
                parameters: &json!({"module": "memory"}),
                outcome: "SUCCESS",
                observations: &json!({"findings": 0}),
                error: None,
                reflection: "Memory engine functioning accurately",
            })
            .expect("Episode must record cleanly");

        assert!(!ep_id.is_empty());

        let results =
            engine.query_episodes(Some("memory"), Some("creator"), Some("SUCCESS"), 10, "all");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["intent"], "CODE_INSPECTION");
        assert_eq!(results[0]["outcome"], "SUCCESS");

        let stats = engine.get_memory_stats();
        assert_eq!(stats["total_episodes"], 1);

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn test_memory_deduplication() {
        let tmp =
            std::env::temp_dir().join(format!("tara_mem_dedup_test_{}", uuid::Uuid::new_v4()));
        let engine = MemoryEngine::new(tmp.to_str().unwrap());

        // Record first time
        let ep_id1 = engine
            .record_episode(EpisodeRecordParams {
                actor_id: "user1",
                intent: "SEARCH",
                action: "QUERY",
                parameters: &json!({"q": "rust async"}),
                outcome: "SUCCESS",
                observations: &json!({"items": 3}),
                error: None,
                reflection: "Executed search",
            })
            .expect("First episode recorded");

        // Record identical episode second time
        let ep_id2 = engine
            .record_episode(EpisodeRecordParams {
                actor_id: "user1",
                intent: "SEARCH",
                action: "QUERY",
                parameters: &json!({"q": "rust async"}),
                outcome: "SUCCESS",
                observations: &json!({"items": 3}),
                error: None,
                reflection: "Executed search duplicate",
            })
            .expect("Second episode recorded");

        // Must return identical episode_id and not duplicate index entry
        assert_eq!(ep_id1, ep_id2);
        assert_eq!(engine.get_memory_stats()["total_episodes"], 1);

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn test_memory_semantic_vector_retrieval() {
        let tmp =
            std::env::temp_dir().join(format!("tara_mem_semantic_test_{}", uuid::Uuid::new_v4()));
        let engine = MemoryEngine::new(tmp.to_str().unwrap());

        engine
            .record_episode(EpisodeRecordParams {
                actor_id: "user2",
                intent: "COMPUTE_FFT",
                action: "SIGNAL_PROCESSING",
                parameters: &json!({"type": "fourier"}),
                outcome: "SUCCESS",
                observations: &json!({}),
                error: None,
                reflection: "Calculated fast fourier transform for spectrum analysis",
            })
            .expect("Recorded signal episode");

        engine
            .record_episode(EpisodeRecordParams {
                actor_id: "user2",
                intent: "RENDER_CANVAS",
                action: "GRAPHICS_DRAW",
                parameters: &json!({"color": "blue"}),
                outcome: "SUCCESS",
                observations: &json!({}),
                error: None,
                reflection: "Rendered interactive UI vector graphic",
            })
            .expect("Recorded graphics episode");

        let results =
            engine.query_episodes(Some("spectrum fourier"), Some("user2"), None, 5, "all");
        assert!(!results.is_empty());
        assert_eq!(results[0]["intent"], "COMPUTE_FFT");

        let _ = fs::remove_dir_all(tmp);
    }
}
