//! Global Knowledge Base: Extensible production architecture for TARA.
//!
//! Provides partitioned, content-addressable storage, multi-level deduplication,
//! strict license verification, and scalable inverted index retrieval across 12+ domains.

pub mod corpus;
pub mod dedup;
#[path = "../../../tara_training_system/world_model/graph.rs"]
pub mod graph;
pub mod index;
pub mod license;
pub mod partition;
pub mod pipeline;
pub mod schema;
pub mod validation;

pub use corpus::SeedCorpusBuilder;
pub use dedup::{DuplicateCheckResult, DuplicateMatchLevel, KnowledgeDuplicateEngine};
pub use graph::{
    FoundationalKnowledgeGraph, HistoricalContextMeta, KnowledgeGraphEdge, KnowledgeGraphNode,
    KnowledgeNodeType, RelationshipStep, RelationshipType,
};
pub use index::ScalableKnowledgeIndex;
pub use license::{
    ArtifactType, LicenseVerificationEngine, LicenseVerificationReport, ReusePermissionStatus,
};
pub use partition::{DomainPartitionRegistry, PartitionMeta};
pub use pipeline::{KnowledgeImportPipeline, PipelineIngestOutcome, PipelineIngestParams};
pub use schema::{
    IndianMathEntry, KnowledgeDocMeta, KnowledgeProvenance, KnowledgeSourceType, MathFormulaEntry,
    ProgrammingLanguageRef, QuarantineRecord, ScientificConceptEntry,
};
pub use validation::{KnowledgeValidationEngine, ValidationResult};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum KnowledgeError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Other(String),
}

/// Backwards-compatible LicenseValidator for legacy callers.
pub struct LicenseValidator;

impl LicenseValidator {
    pub fn is_permissive_or_open(license: &str) -> bool {
        let rep = LicenseVerificationEngine::verify_license(
            "test://uri",
            license,
            "content",
            None,
            "curator",
        );
        rep.reuse_status == ReusePermissionStatus::Permitted
    }
}

/// Cleans and sanitizes raw ingested text (strips HTML, normalizes unicode spaces/newlines).
pub struct KnowledgeCleaner;

impl KnowledgeCleaner {
    pub fn clean(input: &str) -> String {
        // Strip genuine HTML tags: <tag ...> or </tag>
        let mut stripped = String::with_capacity(input.len());
        let mut chars = input.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '<' {
                // Check if following char is an ASCII letter, '/', or '!' (genuine HTML tag)
                if let Some(&next_ch) = chars.peek() {
                    if next_ch.is_ascii_alphabetic() || next_ch == '/' || next_ch == '!' {
                        let mut closed = false;
                        for inner in chars.by_ref() {
                            if inner == '>' {
                                closed = true;
                                break;
                            }
                        }
                        if closed {
                            stripped.push(' ');
                            continue;
                        }
                    }
                }
            }
            stripped.push(ch);
        }
        // Normalize common HTML entities
        let unescaped = stripped
            .replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'");

        // Normalize multiple spaces/newlines
        let mut result = String::new();
        let mut last_was_space = false;
        let mut consecutive_newlines = 0;
        for ch in unescaped.chars() {
            if ch == '\r' {
                continue;
            }
            if ch == '\n' {
                if consecutive_newlines < 2 {
                    result.push('\n');
                    consecutive_newlines += 1;
                }
                last_was_space = false;
            } else if ch.is_whitespace() {
                if !last_was_space && consecutive_newlines == 0 {
                    result.push(' ');
                    last_was_space = true;
                }
            } else {
                result.push(ch);
                last_was_space = false;
                consecutive_newlines = 0;
            }
        }
        result.trim().to_string()
    }
}

/// Sanitize a topic or domain name into a valid directory partition / category name.
pub fn sanitize_category(name: &str) -> String {
    let clean: String = name
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if clean.is_empty() {
        "general".to_string()
    } else {
        clean
    }
}

/// Global knowledge base persisted to disk with scalable inverted index and foundational relationship graph.
pub struct GlobalKnowledgeBase {
    pub base_dir: String,
    pub candidates_dir: String,
    pub index: Arc<Mutex<ScalableKnowledgeIndex>>,
    pub partition_registry: DomainPartitionRegistry,
    pub relationship_graph: Arc<Mutex<FoundationalKnowledgeGraph>>,
}

impl GlobalKnowledgeBase {
    pub fn new(base_dir: &str) -> Self {
        let candidates_dir = format!("{}/candidates", base_dir);
        let partitions_dir = format!("{}/partitions", base_dir);
        let quarantine_dir = format!("{}/quarantine", base_dir);
        let _ = fs::create_dir_all(base_dir);
        let _ = fs::create_dir_all(&candidates_dir);
        let _ = fs::create_dir_all(&partitions_dir);
        let _ = fs::create_dir_all(&quarantine_dir);

        let index_path = Path::new(base_dir).join("index.json");
        let mut loaded_index = ScalableKnowledgeIndex::default();
        if index_path.exists() {
            if let Ok(raw) = fs::read_to_string(&index_path) {
                if let Ok(idx) = serde_json::from_str::<ScalableKnowledgeIndex>(&raw) {
                    loaded_index = idx;
                }
            }
        }

        if loaded_index.doc_cache.is_empty() {
            let (rebuilt, _corrupted) =
                ScalableKnowledgeIndex::rebuild_from_storage(Path::new(base_dir));
            loaded_index = rebuilt;
            let _ = fs::write(
                &index_path,
                serde_json::to_string(&loaded_index).unwrap_or_default(),
            );
        }

        let registry = DomainPartitionRegistry::new(Path::new(base_dir));

        let graph_path = Path::new(base_dir).join("relationship_graph.json");
        let mut loaded_graph = FoundationalKnowledgeGraph::load_or_create(&graph_path);
        if loaded_graph.nodes.is_empty() {
            let _ = loaded_graph.populate_foundational_axiomatic_graph();
            let _ = loaded_graph.populate_from_knowledge_corpus(Path::new(&partitions_dir));
            let _ = loaded_graph.save_to_disk(Path::new(base_dir));
        } else if loaded_graph.nodes.len() < 50 {
            let _ = loaded_graph.populate_from_knowledge_corpus(Path::new(&partitions_dir));
            let _ = loaded_graph.save_to_disk(Path::new(base_dir));
        }

        Self {
            base_dir: base_dir.to_string(),
            candidates_dir,
            index: Arc::new(Mutex::new(loaded_index)),
            partition_registry: registry,
            relationship_graph: Arc::new(Mutex::new(loaded_graph)),
        }
    }

    /// Persist relationship graph to disk.
    pub fn sync_relationship_graph(&self) -> Result<(), KnowledgeError> {
        let graph = self.relationship_graph.lock().unwrap();
        graph.save_to_disk(Path::new(&self.base_dir))
    }

    /// Retrieve relationship chain for a node up to `max_depth`.
    pub fn get_relationship_chain(&self, node_id: &str, max_depth: usize) -> Vec<RelationshipStep> {
        let graph = self.relationship_graph.lock().unwrap();
        graph.get_relationship_chain(node_id, max_depth)
    }

    /// Query knowledge entries matching a text query with scalable inverted index ranking.
    pub fn query_knowledge(&self, query: &str, topic: Option<&str>) -> Vec<Value> {
        let scored = {
            let idx = self.index.lock().unwrap();
            idx.search(query, topic)
        };

        let mut results = Vec::with_capacity(scored.len().min(50));
        for (doc_id, _score) in scored.into_iter().take(50) {
            if let Some(doc) = self.get_knowledge(&doc_id) {
                results.push(doc);
            }
        }
        results
    }

    /// Explicitly sync in-memory knowledge index to disk.
    pub fn sync_index(&self) {
        if let Ok(idx) = self.index.lock() {
            let index_path = Path::new(&self.base_dir).join("index.json");
            let _ = fs::write(
                &index_path,
                serde_json::to_string(&*idx).unwrap_or_default(),
            );
        }
    }

    /// Query knowledge entries belonging to a specific category or domain.
    pub fn query_by_category(&self, category: &str) -> Vec<Value> {
        let cat_lc = sanitize_category(category);
        let doc_ids: Vec<String> = {
            let idx = self.index.lock().unwrap();
            idx.doc_cache
                .iter()
                .filter(|(_, meta)| meta.category == cat_lc || meta.topic.to_lowercase() == cat_lc)
                .map(|(id, _)| id.clone())
                .collect()
        };

        let mut results = Vec::with_capacity(doc_ids.len());
        for id in doc_ids {
            if let Some(doc) = self.get_knowledge(&id) {
                results.push(doc);
            }
        }
        results
    }

    /// Query knowledge entries matching a specific tag.
    pub fn query_by_tag(&self, tag: &str) -> Vec<Value> {
        let tag_lc = tag.trim().to_lowercase();
        let doc_ids: Vec<String> = {
            let idx = self.index.lock().unwrap();
            idx.doc_cache
                .iter()
                .filter(|(_, meta)| meta.tags.iter().any(|t| t.to_lowercase() == tag_lc))
                .map(|(id, _)| id.clone())
                .collect()
        };

        let mut results = Vec::with_capacity(doc_ids.len());
        for id in doc_ids {
            if let Some(doc) = self.get_knowledge(&id) {
                results.push(doc);
            }
        }
        results
    }

    /// Query formulas specifically by name, expression, or topic within mathematics.
    pub fn query_formulas(&self, query: &str) -> Vec<Value> {
        self.query_knowledge(query, Some("mathematics"))
    }

    /// Store or update a knowledge entry with part-wise / category-wise partitioning, tags, and provenance.
    pub fn store_knowledge_partitioned(
        &self,
        topic: &str,
        subject: &str,
        tags: &[String],
        content: &str,
        confidence: f32,
        provenance: Option<KnowledgeProvenance>,
    ) -> Value {
        if topic.trim().is_empty()
            || subject.trim().is_empty()
            || content.trim().is_empty()
            || !confidence.is_finite()
            || !(0.0..=1.0).contains(&confidence)
        {
            return json!({"status":"ERROR","error":"topic, subject, content, and confidence in [0,1] are required"});
        }

        let source_uri = provenance
            .as_ref()
            .map(|p| p.source_uri.clone())
            .unwrap_or_else(|| "ADMIN_INPUT".to_string());
        let license = provenance
            .as_ref()
            .map(|p| p.license.clone())
            .unwrap_or_else(|| "CC-BY-4.0".to_string());
        let author = provenance
            .as_ref()
            .map(|p| p.author_or_curator.clone())
            .unwrap_or_else(|| "admin".to_string());

        let outcome = KnowledgeImportPipeline::ingest_record(
            self,
            PipelineIngestParams {
                topic,
                subpartition: None,
                subject,
                raw_content: content,
                source_uri: &source_uri,
                declared_license: &license,
                declared_type: None,
                author: &author,
                tags,
                confidence,
                source_type: KnowledgeSourceType::ExternalSource,
                originating_record_id: None,
            },
        );

        if outcome.status == "SUCCESS" || outcome.status == "MERGED" {
            let doc_id = outcome.doc_id.unwrap_or_default();
            self.get_knowledge(&doc_id)
                .unwrap_or_else(|| json!({"status": "SUCCESS", "id": doc_id}))
        } else {
            json!({"status": "ERROR", "error": outcome.error.unwrap_or_else(|| "Ingestion failed".to_string())})
        }
    }

    /// Store or update a knowledge entry with optional provenance metadata.
    pub fn store_knowledge_with_provenance(
        &self,
        topic: &str,
        subject: &str,
        content: &str,
        confidence: f32,
        provenance: Option<KnowledgeProvenance>,
    ) -> Value {
        self.store_knowledge_partitioned(topic, subject, &[], content, confidence, provenance)
    }

    /// Store or update a knowledge entry.
    pub fn store_or_update_knowledge(
        &self,
        topic: &str,
        subject: &str,
        content: &str,
        confidence: f32,
    ) -> Value {
        self.store_knowledge_with_provenance(topic, subject, content, confidence, None)
    }

    /// Get a specific knowledge entry by ID from partitions or legacy storage.
    pub fn get_knowledge(&self, id: &str) -> Option<Value> {
        // 1. Fast in-memory cache lookup for exact file_path
        if let Ok(idx) = self.index.lock() {
            if let Some(meta) = idx.doc_cache.get(id) {
                if !meta.file_path.is_empty() {
                    let p = Path::new(&meta.file_path);
                    if p.exists() {
                        if let Ok(raw) = fs::read_to_string(p) {
                            if let Ok(val) = serde_json::from_str::<Value>(&raw) {
                                return Some(val);
                            }
                        }
                    }
                }
            }
        }

        // 2. Check legacy root location
        let legacy_path = Path::new(&self.base_dir).join(format!("{id}.json"));
        if legacy_path.exists() {
            if let Ok(raw) = fs::read_to_string(&legacy_path) {
                if let Ok(val) = serde_json::from_str::<Value>(&raw) {
                    return Some(val);
                }
            }
        }

        // 3. Search within partition subdirectories recursively
        let partitions_dir = Path::new(&self.base_dir).join("partitions");
        if partitions_dir.exists() {
            let mut dirs_to_visit = vec![partitions_dir];
            while let Some(current) = dirs_to_visit.pop() {
                if let Ok(rd) = fs::read_dir(&current) {
                    for entry in rd.flatten() {
                        let path = entry.path();
                        if path.is_dir() {
                            dirs_to_visit.push(path);
                        } else if path.extension().and_then(|e| e.to_str()) == Some("json") {
                            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                            if stem == id {
                                if let Ok(raw) = fs::read_to_string(&path) {
                                    if let Ok(val) = serde_json::from_str::<Value>(&raw) {
                                        return Some(val);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Ingests open-source knowledge records with license enforcement, cleaning, deduplication, and provenance.
    pub fn import_open_knowledge_batch(
        &self,
        raw_input: &str,
        default_license: Option<&str>,
        default_source: Option<&str>,
    ) -> Value {
        let records: Vec<Value> = if let Ok(val) = serde_json::from_str::<Value>(raw_input) {
            if let Some(arr) = val.as_array() {
                arr.clone()
            } else if val.is_object() {
                vec![val]
            } else {
                Vec::new()
            }
        } else {
            raw_input
                .lines()
                .filter(|line| !line.trim().is_empty())
                .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                .collect()
        };

        let mut total = 0;
        let mut imported = 0;
        let mut duplicates_skipped = 0;
        let mut rejected_license = 0;
        let mut rejected_empty = 0;
        let mut imported_entries = Vec::new();

        for rec in records {
            total += 1;
            let topic = rec
                .get("topic")
                .and_then(|v| v.as_str())
                .unwrap_or("general");
            let subpart = rec.get("subpartition").and_then(|v| v.as_str());
            let subject = rec.get("subject").and_then(|v| v.as_str()).unwrap_or("");
            let raw_content = rec.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let license = rec
                .get("license")
                .and_then(|v| v.as_str())
                .or(default_license)
                .unwrap_or("Unknown");
            let source_uri = rec
                .get("source_uri")
                .or_else(|| rec.get("source"))
                .and_then(|v| v.as_str())
                .or(default_source)
                .unwrap_or("open_source_import");
            let author = rec
                .get("author")
                .and_then(|v| v.as_str())
                .unwrap_or("open_community");

            let tags: Vec<String> = rec
                .get("tags")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(Value::as_str)
                        .map(String::from)
                        .collect()
                })
                .unwrap_or_default();

            let outcome = KnowledgeImportPipeline::ingest_record(
                self,
                PipelineIngestParams {
                    topic,
                    subpartition: subpart,
                    subject,
                    raw_content,
                    source_uri,
                    declared_license: license,
                    declared_type: None,
                    author,
                    tags: &tags,
                    confidence: 1.0,
                    source_type: KnowledgeSourceType::ExternalSource,
                    originating_record_id: None,
                },
            );

            match outcome.action.as_str() {
                "INSERTED" => {
                    imported += 1;
                    if let Some(id) = outcome.doc_id {
                        if let Some(doc) = self.get_knowledge(&id) {
                            imported_entries.push(doc);
                        }
                    }
                }
                "MERGED" => {
                    duplicates_skipped += 1;
                }
                "QUARANTINED" => {
                    if outcome.error.as_deref().unwrap_or("").contains("License") {
                        rejected_license += 1;
                    } else {
                        rejected_empty += 1;
                    }
                }
                _ => {}
            }
        }

        json!({
            "status": "SUCCESS",
            "total_records": total,
            "imported": imported,
            "duplicates_skipped": duplicates_skipped,
            "rejected_license": rejected_license,
            "rejected_empty": rejected_empty,
            "entries": imported_entries
        })
    }

    /// Automatically populate the full Phase 1 knowledge corpus across all 12 domains.
    pub fn populate_seed_corpus(&self) -> Value {
        let mut count_math = 0;
        let mut count_ind_math = 0;
        let mut count_physics = 0;
        let mut count_science = 0;
        let mut count_programming = 0;
        let mut count_langs = 0;
        let mut count_standards = 0;
        let mut count_reasoning = 0;

        // 1. Math formulas
        for f in SeedCorpusBuilder::build_math_formula_corpus() {
            let content = format!(
                "Formula: {}\nExact Expression: {}\nDefinitions: {}\nDerivation/Reference: {}\nExample: {}\nDomain: {}\nAssumptions: {}\nCommon Errors: {}",
                f.formula_name,
                f.exact_expression,
                f.definitions,
                f.derivation_reference,
                f.example,
                f.domain,
                f.assumptions_conditions.join("; "),
                f.common_errors.join("; ")
            );
            let mut tags = f.tags.clone();
            tags.push("formula".to_string());
            tags.push(f.domain.clone());

            let res = KnowledgeImportPipeline::ingest_record(
                self,
                PipelineIngestParams {
                    topic: "mathematics",
                    subpartition: Some(&f.domain),
                    subject: &f.formula_name,
                    raw_content: &content,
                    source_uri: &f.source_url,
                    declared_license: &f.license,
                    declared_type: Some("PAPER_PUBLICATION"),
                    author: &f.author_publisher,
                    tags: &tags,
                    confidence: f.confidence,
                    source_type: KnowledgeSourceType::ExternalSource,
                    originating_record_id: None,
                },
            );
            if res.status == "SUCCESS" || res.status == "MERGED" {
                count_math += 1;
            }
        }

        // 2. Indian Mathematics (Explicit separation of HISTORICAL_FORMULA vs MODERN_EQUIVALENT)
        for im in SeedCorpusBuilder::build_indian_math_corpus() {
            let content = format!(
                "Indian Mathematics Record\nTreatise: {}\nAuthor: {}\nPeriod: {}\nDomain: {}\n\nHISTORICAL_FORMULA:\n{}\n\nMODERN_EQUIVALENT:\n{}\n\nMathematical Context: {}\nHistorical Significance: {}\nProvenance Reference: {}",
                im.treatise_or_sutra,
                im.mathematician_or_school,
                im.historical_period,
                im.topic,
                im.historical_formula,
                im.modern_equivalent,
                im.mathematical_context,
                im.historical_significance,
                im.provenance_reference
            );
            let tags = vec![
                "indian_mathematics".to_string(),
                im.topic.clone(),
                im.mathematician_or_school.to_lowercase().replace(' ', "_"),
                "historical_mathematics".to_string(),
            ];

            let indian_subj = format!(
                "{}_{}",
                im.mathematician_or_school.to_lowercase().replace(' ', "_"),
                im.topic
            );
            let indian_source = format!("tara://historical/{}", im.entry_id);
            let res = KnowledgeImportPipeline::ingest_record(
                self,
                PipelineIngestParams {
                    topic: "mathematics",
                    subpartition: Some("indian_mathematics"),
                    subject: &indian_subj,
                    raw_content: &content,
                    source_uri: &indian_source,
                    declared_license: &im.license,
                    declared_type: Some("HISTORICAL_TREATISE"),
                    author: &im.mathematician_or_school,
                    tags: &tags,
                    confidence: 1.0,
                    source_type: KnowledgeSourceType::ExternalSource,
                    originating_record_id: None,
                },
            );
            if res.status == "SUCCESS" || res.status == "MERGED" {
                count_ind_math += 1;
            }
        }

        // 3. Physics & Science
        for sc in SeedCorpusBuilder::build_physics_and_science_corpus() {
            let content = format!(
                "Scientific Concept: {}\nDomain: {} / {}\nDefinition: {}\n\nPrinciples:\n{}\n\nExperimental Proofs:\n{}\n\nApplications:\n{}",
                sc.concept_name,
                sc.scientific_domain,
                sc.subdiscipline,
                sc.definition,
                sc.fundamental_principles.join("\n"),
                sc.experimental_evidence.join("\n"),
                sc.practical_applications.join("\n")
            );
            let tags = vec![
                sc.scientific_domain.clone(),
                sc.subdiscipline.clone(),
                "scientific_concept".to_string(),
            ];

            let res = KnowledgeImportPipeline::ingest_record(
                self,
                PipelineIngestParams {
                    topic: &sc.scientific_domain,
                    subpartition: Some(&sc.subdiscipline),
                    subject: &sc.concept_name,
                    raw_content: &content,
                    source_uri: &sc.source,
                    declared_license: &sc.license,
                    declared_type: Some("PAPER_PUBLICATION"),
                    author: "Science Reference",
                    tags: &tags,
                    confidence: 1.0,
                    source_type: KnowledgeSourceType::ExternalSource,
                    originating_record_id: None,
                },
            );
            if res.status == "SUCCESS" || res.status == "MERGED" {
                count_physics += 1;
            }
        }

        // 3b. Science Domain Concepts
        for sc in SeedCorpusBuilder::build_science_corpus() {
            let topic = sc["topic"].as_str().unwrap();
            let sub = sc["subpartition"].as_str().unwrap();
            let subj = sc["subject"].as_str().unwrap();
            let cont = sc["content"].as_str().unwrap();
            let src = sc["source"].as_str().unwrap();
            let lic = sc["license"].as_str().unwrap();
            let auth = sc["author"].as_str().unwrap();
            let tags: Vec<String> = sc["tags"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();

            let res = KnowledgeImportPipeline::ingest_record(
                self,
                PipelineIngestParams {
                    topic,
                    subpartition: Some(sub),
                    subject: subj,
                    raw_content: cont,
                    source_uri: src,
                    declared_license: lic,
                    declared_type: Some("EDUCATIONAL_REFERENCE"),
                    author: auth,
                    tags: &tags,
                    confidence: 1.0,
                    source_type: KnowledgeSourceType::ExternalSource,
                    originating_record_id: None,
                },
            );
            if res.status == "SUCCESS" || res.status == "MERGED" {
                count_science += 1;
            }
        }

        // 4. Programming Languages
        for pl in SeedCorpusBuilder::build_languages_corpus() {
            let content = format!(
                "Programming Language Reference: {}\nVersion: {}\nSyntax: {}\nSemantics: {}\nType System: {}\nMemory Model: {}\nConcurrency: {}\nError Handling: {}\nStandard APIs: {}\nSpecification: {}\nOfficial Docs: {}",
                pl.language_name,
                pl.specification_version,
                pl.syntax_overview,
                pl.semantics,
                pl.type_system,
                pl.memory_model,
                pl.concurrency,
                pl.error_handling,
                pl.standard_apis.join(", "),
                pl.language_specification,
                pl.official_documentation
            );
            let tags = vec![
                "programming_languages".to_string(),
                pl.language_name.to_lowercase().replace(' ', "_"),
                "language_specification".to_string(),
            ];

            let lang_sub = pl.language_name.to_lowercase().replace(' ', "_");
            let lang_subj = format!("language_spec_{}", lang_sub);
            let res = KnowledgeImportPipeline::ingest_record(
                self,
                PipelineIngestParams {
                    topic: "languages",
                    subpartition: Some(&lang_sub),
                    subject: &lang_subj,
                    raw_content: &content,
                    source_uri: &pl.official_documentation,
                    declared_license: &pl.license_terms,
                    declared_type: Some("TECHNICAL_MANUAL"),
                    author: &pl.language_name,
                    tags: &tags,
                    confidence: 1.0,
                    source_type: KnowledgeSourceType::ExternalSource,
                    originating_record_id: None,
                },
            );
            if res.status == "SUCCESS" || res.status == "MERGED" {
                count_langs += 1;
            }
        }

        // 4b. Programming Paradigms, Algorithms & Data Structures
        for p in SeedCorpusBuilder::build_programming_corpus() {
            let topic = p["topic"].as_str().unwrap();
            let sub = p["subpartition"].as_str().unwrap();
            let subj = p["subject"].as_str().unwrap();
            let cont = p["content"].as_str().unwrap();
            let src = p["source"].as_str().unwrap();
            let lic = p["license"].as_str().unwrap();
            let auth = p["author"].as_str().unwrap();
            let tags: Vec<String> = p["tags"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();

            let res = KnowledgeImportPipeline::ingest_record(
                self,
                PipelineIngestParams {
                    topic,
                    subpartition: Some(sub),
                    subject: subj,
                    raw_content: cont,
                    source_uri: src,
                    declared_license: lic,
                    declared_type: Some("TECHNICAL_MANUAL"),
                    author: auth,
                    tags: &tags,
                    confidence: 1.0,
                    source_type: KnowledgeSourceType::ExternalSource,
                    originating_record_id: None,
                },
            );
            if res.status == "SUCCESS" || res.status == "MERGED" {
                count_programming += 1;
            }
        }

        // 5. Standards, Manuals, Geography, General
        for smg in SeedCorpusBuilder::build_standards_manuals_geography_corpus() {
            let topic = smg["topic"].as_str().unwrap();
            let sub = smg["subpartition"].as_str().unwrap();
            let subj = smg["subject"].as_str().unwrap();
            let cont = smg["content"].as_str().unwrap();
            let src = smg["source"].as_str().unwrap();
            let lic = smg["license"].as_str().unwrap();
            let auth = smg["author"].as_str().unwrap();
            let tags: Vec<String> = smg["tags"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();

            let res = KnowledgeImportPipeline::ingest_record(
                self,
                PipelineIngestParams {
                    topic,
                    subpartition: Some(sub),
                    subject: subj,
                    raw_content: cont,
                    source_uri: src,
                    declared_license: lic,
                    declared_type: Some("SPECIFICATION_STANDARD"),
                    author: auth,
                    tags: &tags,
                    confidence: 1.0,
                    source_type: KnowledgeSourceType::ExternalSource,
                    originating_record_id: None,
                },
            );
            if res.status == "SUCCESS" || res.status == "MERGED" {
                count_standards += 1;
            }
        }

        // 6. Reasoning
        for r in SeedCorpusBuilder::build_reasoning_corpus() {
            let topic = r["topic"].as_str().unwrap();
            let sub = r["subpartition"].as_str().unwrap();
            let subj = r["subject"].as_str().unwrap();
            let cont = r["content"].as_str().unwrap();
            let src = r["source"].as_str().unwrap();
            let lic = r["license"].as_str().unwrap();
            let auth = r["author"].as_str().unwrap();
            let tags: Vec<String> = r["tags"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();

            let res = KnowledgeImportPipeline::ingest_record(
                self,
                PipelineIngestParams {
                    topic,
                    subpartition: Some(sub),
                    subject: subj,
                    raw_content: cont,
                    source_uri: src,
                    declared_license: lic,
                    declared_type: Some("EDUCATIONAL_REFERENCE"),
                    author: auth,
                    tags: &tags,
                    confidence: 1.0,
                    source_type: KnowledgeSourceType::ExternalSource,
                    originating_record_id: None,
                },
            );
            if res.status == "SUCCESS" || res.status == "MERGED" {
                count_reasoning += 1;
            }
        }

        self.sync_index();

        json!({
            "status": "SUCCESS",
            "imported_math_formulas": count_math,
            "imported_indian_math": count_ind_math,
            "imported_physics_space_particles": count_physics,
            "imported_science": count_science,
            "imported_programming": count_programming,
            "imported_programming_languages": count_langs,
            "imported_standards_and_geography": count_standards,
            "imported_reasoning_models": count_reasoning,
            "total_seed_corpus_items": count_math + count_ind_math + count_physics + count_science + count_programming + count_langs + count_standards + count_reasoning,
        })
    }

    /// List up to `limit` knowledge entries across all partitions.
    pub fn list_entries(&self, limit: usize) -> Vec<Value> {
        let ids: Vec<String> = {
            let idx = self.index.lock().unwrap();
            idx.doc_cache.keys().take(limit).cloned().collect()
        };
        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(doc) = self.get_knowledge(&id) {
                results.push(doc);
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
        let mut entry: Value =
            serde_json::from_str(&raw).map_err(|e| format!("Parse error: {}", e))?;

        if let Some(obj) = entry.as_object_mut() {
            obj.insert("status".to_string(), json!("APPROVED"));
            obj.insert("approved_by".to_string(), json!(creator_id));
            obj.insert("approved_at".to_string(), json!(crate::now_iso()));
        }

        let approved_path = format!("{}/{}.json", self.base_dir, candidate_id);
        fs::write(
            &approved_path,
            serde_json::to_string_pretty(&entry).unwrap_or_default(),
        )
        .map_err(|e| e.to_string())?;
        let _ = fs::remove_file(&cand_path);

        Ok(entry)
    }

    /// Generate deterministic knowledge ID from topic and subject.
    pub fn generate_knowledge_id(topic: &str, subject: &str) -> String {
        hex::encode(Sha256::digest(
            format!(
                "{}:{}",
                topic.trim().to_lowercase(),
                subject.trim().to_lowercase()
            )
            .as_bytes(),
        ))[..16]
            .to_string()
    }

    /// Load index.json from knowledge base directory.
    pub fn load_index(&self) -> Value {
        let index_path = Path::new(&self.base_dir).join("index.json");
        if index_path.exists() {
            fs::read_to_string(&index_path)
                .ok()
                .and_then(|data| serde_json::from_str(&data).ok())
                .unwrap_or_else(|| json!({}))
        } else {
            json!({})
        }
    }

    /// Save index.json to knowledge base directory.
    pub fn save_index(&self, index: &Value) -> Result<(), String> {
        let index_path = Path::new(&self.base_dir).join("index.json");
        fs::write(
            &index_path,
            serde_json::to_string_pretty(index).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())
    }

    /// Store a candidate knowledge item for review.
    pub fn store_candidate(
        &self,
        topic: &str,
        subject: &str,
        content: &str,
        confidence: f32,
    ) -> Value {
        let cand_id = format!("cand_{}", Self::generate_knowledge_id(topic, subject));
        let entry = json!({
            "candidate_id": cand_id,
            "topic": topic,
            "subject": subject,
            "content": content,
            "confidence": confidence,
            "status": "PENDING_REVIEW",
            "created_at": crate::now_iso(),
        });
        let path = Path::new(&self.candidates_dir).join(format!("{}.json", cand_id));
        let _ = fs::write(
            &path,
            serde_json::to_string_pretty(&entry).unwrap_or_default(),
        );
        entry
    }

    /// Get a candidate knowledge item by ID.
    pub fn get_candidate(&self, candidate_id: &str) -> Option<Value> {
        let path = Path::new(&self.candidates_dir).join(format!("{}.json", candidate_id));
        fs::read_to_string(&path)
            .ok()
            .and_then(|raw| serde_json::from_str(&raw).ok())
    }

    /// Save topic dossier.
    pub fn save_topic_dossier(&self, topic: &str, dossier: &Value) -> Result<(), String> {
        let clean_topic = topic.replace(['/', '\\', ':'], "_");
        let dossiers_dir = Path::new(&self.base_dir).join("dossiers");
        let _ = fs::create_dir_all(&dossiers_dir);
        let path = dossiers_dir.join(format!("{}.json", clean_topic));
        fs::write(
            &path,
            serde_json::to_string_pretty(dossier).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())
    }

    /// Get topic dossier.
    pub fn get_topic_dossier(&self, topic: &str) -> Option<Value> {
        let clean_topic = topic.replace(['/', '\\', ':'], "_");
        let path = Path::new(&self.base_dir)
            .join("dossiers")
            .join(format!("{}.json", clean_topic));
        fs::read_to_string(&path)
            .ok()
            .and_then(|raw| serde_json::from_str(&raw).ok())
    }
}
