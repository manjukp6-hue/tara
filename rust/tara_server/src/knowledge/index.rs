//! Scalable Inverted Index for Multi-Partition Knowledge Corpus.
//!
//! Maintains compact in-memory metadata (`KnowledgeDocMeta`) and inverted postings (`term -> (doc_id, tf)`).
//! Payloads reside entirely on disk in partitioned directories, preventing memory exhaustion on large corpora.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use super::schema::KnowledgeDocMeta;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ScalableKnowledgeIndex {
    /// Inverted term postings: term -> list of (doc_id, term_frequency)
    pub inverted_index: HashMap<String, Vec<(String, u32)>>,
    /// Document metadata cache: doc_id -> KnowledgeDocMeta (no full content in RAM)
    pub doc_cache: HashMap<String, KnowledgeDocMeta>,
    /// Tag index: tag -> list of doc_ids
    pub tag_index: HashMap<String, Vec<String>>,
    /// Category / Domain index: domain -> list of doc_ids
    pub domain_index: HashMap<String, Vec<String>>,
    /// Content SHA-256 reverse lookup for O(1) duplicate checks: sha256 -> doc_id
    pub sha_index: HashMap<String, String>,
}

impl ScalableKnowledgeIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or update document in the index without loading full payload into memory.
    pub fn index_document(&mut self, meta: KnowledgeDocMeta, full_text_for_tokenization: &str) {
        let doc_id = meta.id.clone();
        let sha256 = meta.content_sha256.clone();
        let domain = meta.category.to_lowercase();
        let tags = meta.tags.clone();

        // 1. Maintain SHA index for O(1) exact dedup checks
        if !sha256.is_empty() {
            self.sha_index.insert(sha256, doc_id.clone());
        }

        // 2. Maintain domain / category index
        let domain_docs = self.domain_index.entry(domain).or_default();
        if !domain_docs.contains(&doc_id) {
            domain_docs.push(doc_id.clone());
        }

        // 3. Maintain tag index
        for tag in &tags {
            let tag_lc = tag.trim().to_lowercase();
            if !tag_lc.is_empty() {
                let tag_docs = self.tag_index.entry(tag_lc).or_default();
                if !tag_docs.contains(&doc_id) {
                    tag_docs.push(doc_id.clone());
                }
            }
        }

        // 4. Tokenize search terms for inverted index
        let mut term_counts: HashMap<String, u32> = HashMap::new();
        let mut add_token = |term: &str, boost: u32| {
            let clean = term
                .trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
                .to_lowercase();
            if clean.len() >= 2 {
                *term_counts.entry(clean.clone()).or_insert(0) += boost;
            }
            if clean.contains('_') || clean.contains('-') {
                for part in clean.split(['_', '-']) {
                    let sub = part.trim_matches(|c: char| !c.is_alphanumeric());
                    if sub.len() >= 2 {
                        *term_counts.entry(sub.to_string()).or_insert(0) += boost;
                    }
                }
            }
        };

        for word in full_text_for_tokenization.split_whitespace() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-');
            if clean.len() >= 2 {
                add_token(clean, 1);
            }
            for part in word.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-') {
                if part.len() >= 2 {
                    add_token(part, 1);
                }
            }
        }

        // Tag boosts in inverted index
        for tag in &tags {
            add_token(tag, 3);
        }

        // Subject words also get high boost in inverted index
        for word in meta.subject.split_whitespace() {
            add_token(word, 4);
        }

        for (term, count) in term_counts {
            let postings = self.inverted_index.entry(term).or_default();
            postings.retain(|(id, _)| id != &doc_id);
            postings.push((doc_id.clone(), count));
        }

        // 5. Store metadata in cache
        self.doc_cache.insert(doc_id, meta);
    }

    /// Incremental document removal from index.
    pub fn remove_document(&mut self, doc_id: &str) {
        if let Some(meta) = self.doc_cache.remove(doc_id) {
            if !meta.content_sha256.is_empty() {
                self.sha_index.remove(&meta.content_sha256);
            }
            if let Some(domain_docs) = self.domain_index.get_mut(&meta.category.to_lowercase()) {
                domain_docs.retain(|id| id != doc_id);
            }
            for tag in &meta.tags {
                if let Some(tag_docs) = self.tag_index.get_mut(&tag.to_lowercase()) {
                    tag_docs.retain(|id| id != doc_id);
                }
            }
            for postings in self.inverted_index.values_mut() {
                postings.retain(|(id, _)| id != doc_id);
            }
        }
    }

    /// Search candidate document IDs by query terms with scoring.
    pub fn search(&self, query: &str, filter_domain: Option<&str>) -> Vec<(String, f32)> {
        let query_lc = query.trim().to_lowercase();
        if query_lc.is_empty() {
            return self
                .doc_cache
                .values()
                .filter(|m| {
                    filter_domain.is_none_or(|d| {
                        m.category.eq_ignore_ascii_case(d) || m.topic.eq_ignore_ascii_case(d)
                    })
                })
                .take(100)
                .map(|m| (m.id.clone(), 1.0))
                .collect();
        }

        let mut query_tokens: Vec<String> = Vec::new();
        for word in query_lc.split_whitespace() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-');
            if clean.len() >= 2 && !query_tokens.iter().any(|s| s == clean) {
                query_tokens.push(clean.to_string());
            }
            for part in word.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-') {
                if part.len() >= 2 && !query_tokens.iter().any(|s| s == part) {
                    query_tokens.push(part.to_string());
                }
            }
        }

        let mut candidate_scores: HashMap<String, f32> = HashMap::new();
        let mut candidate_matched_tokens: HashMap<String, usize> = HashMap::new();

        for token in &query_tokens {
            if let Some(postings) = self.inverted_index.get(token) {
                for (doc_id, tf) in postings {
                    *candidate_scores.entry(doc_id.clone()).or_insert(0.0) +=
                        2.0 + (*tf as f32).min(5.0) * 0.5;
                    *candidate_matched_tokens.entry(doc_id.clone()).or_insert(0) += 1;
                }
            }
        }

        let total_tokens = (query_tokens.len().max(1)) as f32;
        let mut scored_results = Vec::with_capacity(candidate_scores.len());
        for (doc_id, base_score) in candidate_scores {
            if let Some(meta) = self.doc_cache.get(&doc_id) {
                if let Some(d) = filter_domain {
                    if !meta.category.eq_ignore_ascii_case(d) && !meta.topic.eq_ignore_ascii_case(d)
                    {
                        continue;
                    }
                }

                let matched_cnt =
                    candidate_matched_tokens.get(&doc_id).copied().unwrap_or(1) as f32;
                let coverage_factor = matched_cnt / total_tokens;

                // Distinct matched term coordination bonus
                let mut score = base_score + (matched_cnt * 12.0);

                let subj_lc = meta.subject.to_lowercase();
                let top_lc = meta.topic.to_lowercase();
                let prev_lc = meta.content_preview.to_lowercase();

                if subj_lc.contains(&query_lc) || query_lc.contains(&subj_lc) {
                    score += 30.0;
                }
                if top_lc.contains(&query_lc) || query_lc.contains(&top_lc) {
                    score += 15.0;
                }
                if prev_lc.contains(&query_lc) || query_lc.contains(&prev_lc) {
                    score += 5.0;
                }

                for token in &query_tokens {
                    if subj_lc.contains(token.as_str()) {
                        score += 8.0;
                    }
                    if top_lc.contains(token.as_str()) {
                        score += 3.0;
                    }
                }

                for tag in &meta.tags {
                    let tag_lc = tag.to_lowercase();
                    if tag_lc == query_lc {
                        score += 30.0;
                    } else if tag_lc.contains(&query_lc) {
                        score += 18.0;
                    }
                    for token in &query_tokens {
                        if tag_lc == *token {
                            score += 10.0;
                        } else if tag_lc.contains(token.as_str()) {
                            score += 5.0;
                        }
                    }
                }

                // Balance single-term vs multi-term coverage
                score *= 0.4 + 0.6 * coverage_factor;
                score *= meta.confidence.clamp(0.5, 1.0);
                scored_results.push((doc_id, score));
            }
        }

        scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored_results
    }

    /// Full index rebuild from disk storage with corruption detection.
    /// Re-calculates SHA-256 for each persisted file to verify physical data integrity.
    pub fn rebuild_from_storage(base_dir: &Path) -> (Self, Vec<String>) {
        let mut new_idx = Self::new();
        let mut corrupted_files = Vec::new();

        let partitions_dir = base_dir.join("partitions");
        if !partitions_dir.exists() {
            return (new_idx, corrupted_files);
        }

        let mut dirs_to_visit = vec![partitions_dir];
        while let Some(current_dir) = dirs_to_visit.pop() {
            if let Ok(rd) = fs::read_dir(&current_dir) {
                for entry in rd.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        dirs_to_visit.push(path);
                    } else if path.extension().and_then(|e| e.to_str()) == Some("json") {
                        if let Ok(raw_json) = fs::read_to_string(&path) {
                            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw_json) {
                                let id = val
                                    .get("id")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or("");
                                let topic = val
                                    .get("topic")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or("");
                                let subject = val
                                    .get("subject")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or("");
                                let content = val
                                    .get("content")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or("");
                                let declared_sha = val
                                    .get("content_sha256")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or("");
                                let conf =
                                    val.get("confidence")
                                        .and_then(serde_json::Value::as_f64)
                                        .unwrap_or(1.0) as f32;
                                let category = val
                                    .get("category")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or(topic);
                                let partition = val
                                    .get("partition")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or("");
                                let source = val
                                    .get("source")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or("");

                                let tags: Vec<String> = val
                                    .get("tags")
                                    .and_then(serde_json::Value::as_array)
                                    .map(|arr| {
                                        arr.iter()
                                            .filter_map(serde_json::Value::as_str)
                                            .map(String::from)
                                            .collect()
                                    })
                                    .unwrap_or_default();

                                if id.is_empty() || content.is_empty() {
                                    continue;
                                }

                                // Corruption detection: verify SHA-256 against actual file payload
                                let computed_sha =
                                    hex::encode(Sha256::digest(content.trim().as_bytes()));
                                if !declared_sha.is_empty() && declared_sha != computed_sha {
                                    corrupted_files.push(format!(
                                        "{}: SHA mismatch (declared {}, computed {})",
                                        path.display(),
                                        declared_sha,
                                        computed_sha
                                    ));
                                    continue;
                                }

                                let meta = KnowledgeDocMeta {
                                    id: id.to_string(),
                                    topic: topic.to_string(),
                                    subject: subject.to_string(),
                                    tags: tags.clone(),
                                    category: category.to_string(),
                                    partition: partition.to_string(),
                                    source: source.to_string(),
                                    source_type: super::schema::KnowledgeSourceType::ExternalSource,
                                    file_path: path.to_string_lossy().to_string(),
                                    content_preview: content.chars().take(250).collect(),
                                    content_sha256: computed_sha.clone(),
                                    confidence: conf,
                                    originating_record_id: None,
                                };

                                let full_text =
                                    format!("{} {} {} {}", topic, subject, tags.join(" "), content);
                                new_idx.index_document(meta, &full_text);
                            }
                        }
                    }
                }
            }
        }

        (new_idx, corrupted_files)
    }
}
