//! Knowledge schemas for extensible multi-partition knowledge corpus.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Provenance metadata tracking the origin, license, and content digest of ingested knowledge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KnowledgeProvenance {
    pub source_uri: String,
    pub license: String,
    pub author_or_curator: String,
    pub content_sha256: String,
    pub imported_at: String,
}

/// Explicit provenance source type for knowledge origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Default)]
pub enum KnowledgeSourceType {
    #[default]
    ExternalSource,
    ResearchResult,
    ExperimentResult,
    TaraDerived,
    HumanApproved,
    ImportedDataset,
}

impl std::fmt::Display for KnowledgeSourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExternalSource => write!(f, "EXTERNAL_SOURCE"),
            Self::ResearchResult => write!(f, "RESEARCH_RESULT"),
            Self::ExperimentResult => write!(f, "EXPERIMENT_RESULT"),
            Self::TaraDerived => write!(f, "TARA_DERIVED"),
            Self::HumanApproved => write!(f, "HUMAN_APPROVED"),
            Self::ImportedDataset => write!(f, "IMPORTED_DATASET"),
        }
    }
}

/// Standardized schema for all formulas across A-Z mathematics and mathematical sciences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathFormulaEntry {
    pub formula_id: String,
    pub formula_name: String,
    pub exact_expression: String,
    pub alternate_forms: Vec<String>,
    pub variables: HashMap<String, String>,
    pub definitions: String,
    pub units_dimensions: Option<String>,
    pub domain: String,
    pub assumptions_conditions: Vec<String>,
    pub derivation_reference: String,
    pub example: String,
    pub common_errors: Vec<String>,
    pub related_formulas: Vec<String>,
    pub category: String,
    pub topic: String,
    pub tags: Vec<String>,
    pub source: String,
    pub source_url: String,
    pub license: String,
    pub author_publisher: String,
    pub publication_version_date: String,
    pub content_sha256: String,
    pub confidence: f32,
    pub provenance: Option<KnowledgeProvenance>,
}

/// Specialized schema for Indian Mathematics distinguishing historical formulations from modern equivalents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndianMathEntry {
    pub entry_id: String,
    pub mathematician_or_school: String,
    pub treatise_or_sutra: String,
    pub historical_period: String,
    pub topic: String,
    pub historical_formula: String,
    pub modern_equivalent: String,
    pub mathematical_context: String,
    pub historical_significance: String,
    pub provenance_reference: String,
    pub license: String,
}

/// Structured schema for physics and scientific concepts linking to formulas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScientificConceptEntry {
    pub concept_id: String,
    pub concept_name: String,
    pub scientific_domain: String,
    pub subdiscipline: String,
    pub definition: String,
    pub fundamental_principles: Vec<String>,
    pub governing_formula_ids: Vec<String>,
    pub experimental_evidence: Vec<String>,
    pub practical_applications: Vec<String>,
    pub source: String,
    pub license: String,
}

/// Comprehensive programming language specification reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgrammingLanguageRef {
    pub language_id: String,
    pub language_name: String,
    pub specification_version: String,
    pub syntax_overview: String,
    pub semantics: String,
    pub type_system: String,
    pub operators: Vec<String>,
    pub control_flow: Vec<String>,
    pub functions_modules: String,
    pub memory_model: String,
    pub concurrency: String,
    pub error_handling: String,
    pub standard_apis: Vec<String>,
    pub language_specification: String,
    pub official_documentation: String,
    pub license_terms: String,
    pub provenance: Option<KnowledgeProvenance>,
}

/// Compact indexed document metadata for in-memory index without loading full payloads.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KnowledgeDocMeta {
    pub id: String,
    pub topic: String,
    pub subject: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub partition: String,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub source_type: KnowledgeSourceType,
    #[serde(default)]
    pub file_path: String,
    pub content_preview: String,
    pub content_sha256: String,
    pub confidence: f32,
    #[serde(default)]
    pub originating_record_id: Option<String>,
}

/// Quarantine record for rejected or invalid knowledge submissions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarantineRecord {
    pub quarantine_id: String,
    pub candidate_id: String,
    pub rejection_stage: String,
    pub reasons: Vec<String>,
    pub source_uri: String,
    pub license: String,
    pub content_snippet: String,
    pub quarantined_at: String,
}
