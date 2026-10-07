//! TARA Dataset Quality & Pre-Ingestion Filter Engine (100% Native Rust)
//!
//! Subsystem Scope & Governance:
//! TARA Downloader & Filter Engine provides a native-Rust, stage-gated dataset ingestion
//! pipeline that rejects configured synthetic indicators, weak/malformed content, duplicates,
//! malicious code, toxic/scam words, PII email dumps, and disallowed licensing profiles,
//! while preserving source and licensing provenance.
//! Domain classification is structural/rule-based and is not a semantic proof of content
//! authenticity or correctness.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

// ============================================================================
// 1. Configuration & Core Models
// ============================================================================

/// Dynamically extensible configuration for intelligence and quality filtering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterConfig {
    /// Minimum allowed text character count.
    pub min_content_chars: usize,
    /// Maximum allowed text character count per section/entry.
    pub max_content_chars: usize,
    /// Minimum allowed text word count.
    pub min_word_count: usize,
    /// Minimum required ratio of alphanumeric characters and whitespace.
    pub min_alphanumeric_ratio: f64,
    /// Maximum permitted ratio of unprintable control characters.
    pub max_control_char_ratio: f64,
    /// Minimum vocabulary diversity ratio (Type-Token Ratio: unique words / total words).
    pub min_vocabulary_diversity_ratio: f64,
    /// Maximum consecutive repetitions of the same non-whitespace character.
    pub max_consecutive_char_repetition: usize,
    /// Maximum permitted ratio of raw HTML/JS markup and web boilerplate tags.
    pub max_html_tag_ratio: f64,
    /// Maximum permitted ratio of corrupted / OCR noisy words.
    pub max_ocr_noise_ratio: f64,
    /// Minimum required sentence count for substantive content.
    pub min_sentence_count: usize,
    /// Maximum Hamming distance for near-duplicate rejection (SimHash 64-bit).
    pub simhash_hamming_threshold: u32,
    /// Minimum indicator match ratio required for domain classification.
    pub min_domain_indicator_ratio: f64,
    /// Minimum pass ratio of valid samples for JSONL multi-record datasets.
    pub min_jsonl_sample_pass_ratio: f64,
    /// Whether to strictly reject malicious code, reverse shells, web shells, and exploits.
    pub reject_malicious_code: bool,
    /// Whether to strictly reject toxic, scam, adult, and gambling words.
    pub reject_toxic_spam: bool,
    /// Maximum allowed email addresses before content is flagged as a PII / contact harvest dump.
    pub max_allowed_emails: usize,
    /// Whether to automatically redact isolated email addresses into [EMAIL_REDACTED].
    pub redact_emails: bool,
    /// Maximum consecutive repetitions of the same word before rejecting as repetitive noise.
    pub max_consecutive_word_repetition: usize,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            min_content_chars: 50,
            max_content_chars: 50_000_000,
            min_word_count: 10,
            min_alphanumeric_ratio: 0.60,
            max_control_char_ratio: 0.01,
            min_vocabulary_diversity_ratio: 0.20,
            max_consecutive_char_repetition: 6,
            max_html_tag_ratio: 0.05,
            max_ocr_noise_ratio: 0.08,
            min_sentence_count: 1,
            simhash_hamming_threshold: 3,
            min_domain_indicator_ratio: 0.15,
            min_jsonl_sample_pass_ratio: 0.60,
            reject_malicious_code: true,
            reject_toxic_spam: true,
            max_allowed_emails: 3,
            redact_emails: true,
            max_consecutive_word_repetition: 5,
        }
    }
}

impl FilterConfig {
    /// Dynamically validates configuration bounds and parameters (Rule 15).
    pub fn validate(&self) -> Result<(), String> {
        if self.min_content_chars == 0 {
            return Err("min_content_chars must be greater than 0".to_string());
        }
        if self.max_content_chars < self.min_content_chars {
            return Err("max_content_chars cannot be less than min_content_chars".to_string());
        }
        if self.min_word_count == 0 {
            return Err("min_word_count must be greater than 0".to_string());
        }
        if !self.min_alphanumeric_ratio.is_finite()
            || self.min_alphanumeric_ratio <= 0.0
            || self.min_alphanumeric_ratio > 1.0
        {
            return Err("min_alphanumeric_ratio must be a finite number in (0.0, 1.0]".to_string());
        }
        if !self.max_control_char_ratio.is_finite()
            || self.max_control_char_ratio < 0.0
            || self.max_control_char_ratio > 1.0
        {
            return Err("max_control_char_ratio must be a finite number in [0.0, 1.0]".to_string());
        }
        if !self.min_vocabulary_diversity_ratio.is_finite()
            || self.min_vocabulary_diversity_ratio <= 0.0
            || self.min_vocabulary_diversity_ratio > 1.0
        {
            return Err("min_vocabulary_diversity_ratio must be a finite number in (0.0, 1.0]".to_string());
        }
        if self.max_consecutive_char_repetition == 0 {
            return Err("max_consecutive_char_repetition must be greater than 0".to_string());
        }
        if !self.max_html_tag_ratio.is_finite()
            || self.max_html_tag_ratio < 0.0
            || self.max_html_tag_ratio > 1.0
        {
            return Err("max_html_tag_ratio must be a finite number in [0.0, 1.0]".to_string());
        }
        if !self.max_ocr_noise_ratio.is_finite()
            || self.max_ocr_noise_ratio < 0.0
            || self.max_ocr_noise_ratio > 1.0
        {
            return Err("max_ocr_noise_ratio must be a finite number in [0.0, 1.0]".to_string());
        }
        if self.simhash_hamming_threshold > 64 {
            return Err("simhash_hamming_threshold must be in 0..=64".to_string());
        }
        if !self.min_domain_indicator_ratio.is_finite()
            || self.min_domain_indicator_ratio < 0.0
            || self.min_domain_indicator_ratio > 1.0
        {
            return Err("min_domain_indicator_ratio must be a finite number in [0.0, 1.0]".to_string());
        }
        if !self.min_jsonl_sample_pass_ratio.is_finite()
            || self.min_jsonl_sample_pass_ratio <= 0.0
            || self.min_jsonl_sample_pass_ratio > 1.0
        {
            return Err("min_jsonl_sample_pass_ratio must be a finite number in (0.0, 1.0]".to_string());
        }
        if self.max_consecutive_word_repetition == 0 {
            return Err("max_consecutive_word_repetition must be greater than 0".to_string());
        }
        Ok(())
    }
}

/// Explicit root-cause rejection reason codes for all TARA pre-gate and streaming filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RejectionReasonCode {
    RejectNoLicense,
    RejectRightsUnknown,
    RejectForbiddenLicense,
    RejectScopeMismatch,
    RejectThirdPartyContent,
    RejectSynthetic,
    RejectDuplicate,
    RejectNearDuplicate,
    RejectEmpty,
    RejectTooShort,
    RejectNoisy,
    RejectMalformed,
    RejectProvenance,
    RejectDomain,
}

impl RejectionReasonCode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RejectNoLicense => "REJECT_NO_LICENSE",
            Self::RejectRightsUnknown => "REJECT_RIGHTS_UNKNOWN",
            Self::RejectForbiddenLicense => "REJECT_FORBIDDEN_LICENSE",
            Self::RejectScopeMismatch => "REJECT_SCOPE_MISMATCH",
            Self::RejectThirdPartyContent => "REJECT_THIRD_PARTY_CONTENT",
            Self::RejectSynthetic => "REJECT_SYNTHETIC",
            Self::RejectDuplicate => "REJECT_DUPLICATE",
            Self::RejectNearDuplicate => "REJECT_NEAR_DUPLICATE",
            Self::RejectEmpty => "REJECT_EMPTY",
            Self::RejectTooShort => "REJECT_TOO_SHORT",
            Self::RejectNoisy => "REJECT_NOISY",
            Self::RejectMalformed => "REJECT_MALFORMED",
            Self::RejectProvenance => "REJECT_PROVENANCE",
            Self::RejectDomain => "REJECT_DOMAIN",
        }
    }
}

/// Operational decision returned by the Pre-Download Rule-18/21 Gate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreDownloadDecision {
    Accept,
    Reject {
        code: RejectionReasonCode,
        rule: &'static str,
        evidence: String,
        details: String,
    },
}

impl PreDownloadDecision {
    pub fn is_accepted(&self) -> bool {
        matches!(self, PreDownloadDecision::Accept)
    }

    pub fn is_rejected(&self) -> bool {
        matches!(self, PreDownloadDecision::Reject { .. })
    }
}

/// Comprehensive quality metrics computed during deep content inspection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentQualityMetrics {
    pub char_count: usize,
    pub word_count: usize,
    pub sentence_count: usize,
    pub alphanumeric_ratio: f64,
    pub control_char_ratio: f64,
    pub vocabulary_diversity_ratio: f64,
    pub ocr_noise_ratio: f64,
    pub html_boilerplate_ratio: f64,
    pub max_consecutive_char_repeat: usize,
    pub balanced_structure: bool,
    pub domain_indicator_ratio: f64,
    pub classified_domain: String,
    pub quality_score: f64,
}

/// Evaluates content through the 7 mandatory gates:
/// ACCEPT = rights_ok && scope_ok && provenance_ok && quality_ok && duplication_ok && structure_ok && domain_ok
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineGateReport {
    pub rights_ok: bool,
    pub scope_ok: bool,
    pub provenance_ok: bool,
    pub quality_ok: bool,
    pub duplication_ok: bool,
    pub structure_ok: bool,
    pub domain_ok: bool,
    pub failure_code: Option<RejectionReasonCode>,
    pub failure_reason: Option<String>,
    pub metrics: Option<ContentQualityMetrics>,
}

impl PipelineGateReport {
    /// Final condition:
    /// ACCEPT = rights_ok && scope_ok && provenance_ok && quality_ok && duplication_ok && structure_ok && domain_ok
    pub fn is_accepted(&self) -> bool {
        self.rights_ok
            && self.scope_ok
            && self.provenance_ok
            && self.quality_ok
            && self.duplication_ok
            && self.structure_ok
            && self.domain_ok
    }

    pub fn is_rejected(&self) -> bool {
        !self.is_accepted()
    }
}

/// Specification of an authorized open-source asset within the authoritative catalog.
#[derive(Debug, Clone)]
pub struct AllowedSourceRule {
    pub dataset_id_prefix: &'static str,
    pub host_domain: &'static str,
    pub allowed_url_prefix: &'static str,
    pub expected_license: &'static str,
    pub asset_scope: &'static str,
    pub required_exclusions: &'static str,
    pub jurisdiction_scope: &'static str,
    pub ai_training_permitted: bool,
    pub commercial_use_permitted: bool,
}

/// Pre-Download Rule-18/21 Gate.
///
/// Evaluates candidate before ANY network download starts:
/// 1. Source allowlist check (rejects unapproved host domains) -> REJECT_DOMAIN
/// 2. Exact asset scope match check -> REJECT_SCOPE_MISMATCH
/// 3. License evidence existence -> REJECT_NO_LICENSE
/// 4. Forbidden copyleft / NC / ND / SA -> REJECT_FORBIDDEN_LICENSE
/// 5. AI training & commercial rights proof -> REJECT_RIGHTS_UNKNOWN
/// 6. Third-party exclusions check -> REJECT_THIRD_PARTY_CONTENT
/// 7. Incomplete provenance check -> REJECT_PROVENANCE
#[derive(Debug, Clone)]
pub struct PreDownloadGate {
    pub allowed_sources: Vec<AllowedSourceRule>,
}

impl Default for PreDownloadGate {
    fn default() -> Self {
        Self::new_with_catalog()
    }
}

impl PreDownloadGate {
    /// Creates the pre-download gate configured with the authoritative verified catalog.
    pub fn new_with_catalog() -> Self {
        let allowed_sources = vec![
            AllowedSourceRule {
                dataset_id_prefix: "euclid_elements",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Euclid Elements Heath 1908 edition",
                required_exclusions: "commentary,post-1928",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "boole_laws_of_thought",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Laws of Thought 1854 original",
                required_exclusions: "modern_notes,modern_preface",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "dedekind_essays",
                host_domain: "archive.org",
                allowed_url_prefix: "https://archive.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Essays on Theory of Numbers 1901",
                required_exclusions: "editorial_notes",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "poincare_science_hypothesis",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Science and Hypothesis 1905",
                required_exclusions: "revisions",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "metamath_set_mm",
                host_domain: "github.com",
                allowed_url_prefix: "https://github.com/metamath/",
                expected_license: "CC0-1.0",
                asset_scope: "set.mm formal proofs database",
                required_exclusions: "gui_tools,standalone_scripts",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "bccampus_concepts_biology",
                host_domain: "opentextbc.ca",
                allowed_url_prefix: "https://opentextbc.ca/",
                expected_license: "CC-BY-4.0",
                asset_scope: "Concepts of Biology 1st Canadian Ed text",
                required_exclusions: "third_party_images,copyrighted_photos",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "bccampus_anatomy_physiology",
                host_domain: "opentextbc.ca",
                allowed_url_prefix: "https://opentextbc.ca/",
                expected_license: "CC-BY-4.0",
                asset_scope: "Anatomy and Physiology text",
                required_exclusions: "photographs,external_drawings",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "pmc_open_access_xml",
                host_domain: "ftp.ncbi.nlm.nih.gov",
                allowed_url_prefix: "ftp://ftp.ncbi.nlm.nih.gov/",
                expected_license: "CC-BY-4.0",
                asset_scope: "PMC Open Access Subset XML CC-BY verified only",
                required_exclusions: "supplementary_data,nc_nd_papers",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "sqlite_core_engine",
                host_domain: "sqlite.org",
                allowed_url_prefix: "https://www.sqlite.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "sqlite3.c and sqlite3.h core amalgamation",
                required_exclusions: "proprietary_extensions,commercial_addons",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "sqlite_core_docs",
                host_domain: "sqlite.org",
                allowed_url_prefix: "https://www.sqlite.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "SQLite official architectural documentation",
                required_exclusions: "forum_posts,unofficial_tutorials",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "rust_standard_library",
                host_domain: "github.com",
                allowed_url_prefix: "https://github.com/rust-lang/rust",
                expected_license: "MIT OR Apache-2.0",
                asset_scope: "library/core and library/alloc source files",
                required_exclusions: "unicode_data.rs,llvm_internals",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "freebsd_handbook",
                host_domain: "freebsd.org",
                allowed_url_prefix: "https://cgit.freebsd.org/",
                expected_license: "BSD-2-Clause",
                asset_scope: "FreeBSD handbook and architecture articles",
                required_exclusions: "ports,gpl_tools",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "openbsd_man_pages",
                host_domain: "openbsd.org",
                allowed_url_prefix: "https://man.openbsd.org/",
                expected_license: "BSD-2-Clause",
                asset_scope: "OpenBSD core BSD/ISC man pages",
                required_exclusions: "gpl_tools,gnu_manuals",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "llvm_core_architecture_docs",
                host_domain: "llvm.org",
                allowed_url_prefix: "https://llvm.org/docs/",
                expected_license: "Apache-2.0 WITH LLVM-exception",
                asset_scope: "LLVM core architecture guidelines and specification",
                required_exclusions: "external_bindings,plugins",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "apache_commons_math_lang",
                host_domain: "apache.org",
                allowed_url_prefix: "https://github.com/apache/commons-",
                expected_license: "Apache-2.0",
                asset_scope: "Apache Commons Math and Lang documentation",
                required_exclusions: "third_party_dependencies",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "python_stdlib_docs",
                host_domain: "python.org",
                allowed_url_prefix: "https://docs.python.org/",
                expected_license: "PSF-2.0",
                asset_scope: "Python standard library module reference docs",
                required_exclusions: "external_examples",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "go_core_implementation",
                host_domain: "github.com",
                allowed_url_prefix: "https://github.com/golang/go",
                expected_license: "BSD-3-Clause",
                asset_scope: "Go compiler and standard library core implementation",
                required_exclusions: "golang.org/x",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "go_docs_specification",
                host_domain: "go.dev",
                allowed_url_prefix: "https://go.dev/doc/",
                expected_license: "CC-BY-4.0",
                asset_scope: "Go language specification and official documentation",
                required_exclusions: "blog,user_articles",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "postgresql_core_manual",
                host_domain: "postgresql.org",
                allowed_url_prefix: "https://www.postgresql.org/docs/",
                expected_license: "PostgreSQL-License",
                asset_scope: "PostgreSQL official core reference manual",
                required_exclusions: "contrib_gpl",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "zig_language_reference",
                host_domain: "ziglang.org",
                allowed_url_prefix: "https://ziglang.org/documentation/",
                expected_license: "MIT",
                asset_scope: "Zig language reference and stdlib documentation",
                required_exclusions: "musl",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "cilium_bpf_architecture_docs",
                host_domain: "cilium.io",
                allowed_url_prefix: "https://docs.cilium.io/",
                expected_license: "CC-BY-4.0",
                asset_scope: "Cilium and eBPF official architecture guide",
                required_exclusions: "bpf_templates,gpl_drivers",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "riscv_instruction_set_manual",
                host_domain: "riscv.org",
                allowed_url_prefix: "https://github.com/riscv/riscv-isa-manual",
                expected_license: "CC-BY-4.0",
                asset_scope: "RISC-V ratified instruction set manual Vol I and II",
                required_exclusions: "proprietary_drafts",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "the_stack_clean_subset",
                host_domain: "huggingface.co",
                allowed_url_prefix: "https://huggingface.co/datasets/bigcode/the-stack",
                expected_license: "MIT OR Apache-2.0 OR BSD-3-Clause",
                asset_scope: "The Stack verified permissive source code subset",
                required_exclusions: "copyleft,opt_outs",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "saylor_cs101_cs102",
                host_domain: "saylor.org",
                allowed_url_prefix: "https://saylor.org/",
                expected_license: "CC-BY-4.0",
                asset_scope: "Saylor Academy computer science course text",
                required_exclusions: "video_links,external_readings",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "suny_milne_technical_writing",
                host_domain: "milneopentextbooks.org",
                allowed_url_prefix: "https://milneopentextbooks.org/",
                expected_license: "CC-BY-4.0",
                asset_scope: "SUNY Milne open technical writing textbook",
                required_exclusions: "external_case_studies",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "bccampus_basic_motor_control",
                host_domain: "opentextbc.ca",
                allowed_url_prefix: "https://opentextbc.ca/",
                expected_license: "CC-BY-4.0",
                asset_scope: "Basic Motor Control trades textbook",
                required_exclusions: "trademarks",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "bccampus_electrical_skills",
                host_domain: "opentextbc.ca",
                allowed_url_prefix: "https://opentextbc.ca/",
                expected_license: "CC-BY-4.0",
                asset_scope: "Electrical skills trades core textbook",
                required_exclusions: "safety_codes",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "bccampus_intro_sociology",
                host_domain: "opentextbc.ca",
                allowed_url_prefix: "https://opentextbc.ca/",
                expected_license: "CC-BY-4.0",
                asset_scope: "Introduction to Sociology 2e text",
                required_exclusions: "news_clippings",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "bccampus_intro_philosophy",
                host_domain: "opentextbc.ca",
                allowed_url_prefix: "https://opentextbc.ca/",
                expected_license: "CC-BY-4.0",
                asset_scope: "Introduction to Philosophy text",
                required_exclusions: "copyrighted_essays",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "newton_principia_1729",
                host_domain: "archive.org",
                allowed_url_prefix: "https://archive.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Principia 1729 Motte English translation",
                required_exclusions: "modern_notes",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "faraday_researches_electricity",
                host_domain: "archive.org",
                allowed_url_prefix: "https://archive.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Experimental Researches in Electricity 1839",
                required_exclusions: "commentaries",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "joule_scientific_papers",
                host_domain: "archive.org",
                allowed_url_prefix: "https://archive.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Scientific Papers of J.P. Joule 1884",
                required_exclusions: "revisions",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "kelvin_mathematical_papers",
                host_domain: "archive.org",
                allowed_url_prefix: "https://archive.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Mathematical and Physical Papers 1882",
                required_exclusions: "footnotes",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "maxwell_electricity_magnetism",
                host_domain: "archive.org",
                allowed_url_prefix: "https://archive.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Treatise on Electricity and Magnetism 1873",
                required_exclusions: "modern_treatises",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "lavoisier_elements_chemistry",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Elements of Chemistry 1790 Kerr translation",
                required_exclusions: "modern_tables",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "mendel_plant_hybridization",
                host_domain: "archive.org",
                allowed_url_prefix: "https://archive.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Experiments in Plant Hybridization 1901 Bateson translation",
                required_exclusions: "post_1928_analysis",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "mendeleev_principles_chemistry",
                host_domain: "archive.org",
                allowed_url_prefix: "https://archive.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Principles of Chemistry 1891 Kamensky translation",
                required_exclusions: "supplements",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "darwin_origin_species",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Origin of Species 1859 1st edition original",
                required_exclusions: "modern_forewords",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "adam_smith_wealth_nations",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Wealth of Nations 1776 original",
                required_exclusions: "modern_commentaries",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "locke_two_treatises",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Two Treatises of Government 1689",
                required_exclusions: "modern_analysis",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "hobbes_leviathan",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Leviathan 1651 original",
                required_exclusions: "modern_notes",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "mill_on_liberty",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "On Liberty 1859 original",
                required_exclusions: "modern_introductions",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "kant_pure_reason",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Critique of Pure Reason 1881 Meiklejohn translation",
                required_exclusions: "translator_notes",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "marx_capital_vol1",
                host_domain: "marxists.org",
                allowed_url_prefix: "https://www.marxists.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Capital Vol 1 1887 Moore-Aveling translation",
                required_exclusions: "modern_prefaces",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "aristotle_ethics",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Nicomachean Ethics 1893 Peters translation",
                required_exclusions: "revised_editions",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "gibbon_roman_empire",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Decline and Fall of Roman Empire 1776",
                required_exclusions: "modern_annotations",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "webster_unabridged_1913",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Webster's Revised Unabridged Dictionary 1913",
                required_exclusions: "modern_revisions",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "wordnet_3_0",
                host_domain: "wordnet.princeton.edu",
                allowed_url_prefix: "https://wordnet.princeton.edu/",
                expected_license: "WordNet-3.0-License",
                asset_scope: "WordNet 3.0 lexical database",
                required_exclusions: "lexical_extensions",
                jurisdiction_scope: "Universal",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "faraday_chemical_candle",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Chemical History of a Candle 1861 original lectures",
                required_exclusions: "modern_editorials",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "maxwell_matter_motion",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Matter and Motion 1876 physics treatise",
                required_exclusions: "modern_forewords",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "galileo_two_new_sciences",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Dialogues Concerning Two New Sciences 1914 English edition",
                required_exclusions: "modern_notes",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "babbage_economy_machinery",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "On the Economy of Machinery and Manufactures 1832 original",
                required_exclusions: "modern_prefaces",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "descartes_discourse_method",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Discourse on Method Veitch 1850 English translation",
                required_exclusions: "modern_editorials",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "bacon_novum_organum",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Novum Organum 1620 philosophical treatise",
                required_exclusions: "modern_forewords",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "hume_human_understanding",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Enquiry Concerning Human Understanding 1748 original",
                required_exclusions: "modern_commentaries",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "locke_human_understanding",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Essay Concerning Human Understanding 1689 original",
                required_exclusions: "modern_notes",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "aurelius_meditations",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Meditations George Long translation 1862",
                required_exclusions: "modern_commentaries",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "plato_republic",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "The Republic Benjamin Jowett translation 1871",
                required_exclusions: "modern_prefaces",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "aristotle_politics",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Politics Benjamin Jowett translation 1885",
                required_exclusions: "modern_notes",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "darwin_voyage_beagle",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Voyage of the Beagle 1839 original journal",
                required_exclusions: "modern_forewords",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "spinoza_ethics",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Ethics 1883 Elwes translation",
                required_exclusions: "modern_notes",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "newton_opticks",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Opticks 1704 original treatise",
                required_exclusions: "modern_prefaces",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "boyle_sceptical_chymist",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "The Sceptical Chymist 1661 original",
                required_exclusions: "modern_introductions",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "lyell_principles_geology",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Principles of Geology 1830 original",
                required_exclusions: "modern_annotations",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "leibniz_monadology",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "The Monadology 1898 Latta translation",
                required_exclusions: "modern_commentaries",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "hooke_micrographia",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Micrographia 1665 original",
                required_exclusions: "modern_prefaces",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "ricardo_political_economy",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Principles of Political Economy and Taxation 1817",
                required_exclusions: "modern_notes",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "machiavelli_prince",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "The Prince Marriott 1908 translation",
                required_exclusions: "modern_commentaries",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "marx_communist_manifesto",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "The Communist Manifesto 1888 Moore translation",
                required_exclusions: "modern_prefaces",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "aristotle_poetics",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "The Poetics Butcher translation 1895",
                required_exclusions: "modern_notes",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "plato_apology",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Apology of Socrates Jowett translation 1891",
                required_exclusions: "modern_prefaces",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "plato_symposium",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Symposium Jowett translation 1892",
                required_exclusions: "modern_prefaces",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "lucretius_nature_of_things",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "On the Nature of Things Leonard translation 1916",
                required_exclusions: "modern_notes",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
            AllowedSourceRule {
                dataset_id_prefix: "gutenberg_ebook_",
                host_domain: "gutenberg.org",
                allowed_url_prefix: "https://www.gutenberg.org/cache/epub/",
                expected_license: "PUBLIC_DOMAIN",
                asset_scope: "Project Gutenberg Public Domain classic eBook",
                required_exclusions: "modern_commercial_derivatives",
                jurisdiction_scope: "VERIFIED_JURISDICTIONS_ONLY",
                ai_training_permitted: true,
                commercial_use_permitted: true,
            },
        ];

        Self { allowed_sources }
    }

    /// Evaluates candidate before ANY network download starts (Pre-Download Gate).
    pub fn evaluate(
        &self,
        dataset_id: &str,
        source_url: &str,
        author: &str,
        license: &str,
        license_proof_url: &str,
        asset_scope: Option<&str>,
    ) -> PreDownloadDecision {
        let trimmed_id = dataset_id.trim();
        let trimmed_url = source_url.trim();
        let trimmed_lic = license.trim();
        let trimmed_author = author.trim();
        let trimmed_proof = license_proof_url.trim();

        // 1. Provenance completeness check: author & proof URL required
        if trimmed_author.is_empty() || trimmed_proof.is_empty() {
            return PreDownloadDecision::Reject {
                code: RejectionReasonCode::RejectProvenance,
                rule: "Rule 21: Provenance Ledger completeness mandatory",
                evidence: format!("author='{trimmed_author}', proof_url='{trimmed_proof}'"),
                details: "Dataset author or license proof URL missing; incomplete provenance".to_string(),
            };
        }

        // 2. License evidence existence check
        if trimmed_lic.is_empty() || trimmed_lic.eq_ignore_ascii_case("unknown") {
            return PreDownloadDecision::Reject {
                code: RejectionReasonCode::RejectNoLicense,
                rule: "Rule 18: Authoritative license evidence required",
                evidence: format!("license='{trimmed_lic}'"),
                details: "Missing or unspecified license declaration".to_string(),
            };
        }

        // 3. Forbidden License check (Rule 18: NC, ND, SA, Copyleft prohibited)
        let lic_lower = trimmed_lic.to_lowercase();
        if lic_lower.contains("-nc")
            || lic_lower.contains("noncommercial")
            || lic_lower.contains("non-commercial")
            || lic_lower.contains("-nd")
            || lic_lower.contains("noderivatives")
            || lic_lower.contains("-sa")
            || lic_lower.contains("sharealike")
            || lic_lower.contains("share-alike")
            || lic_lower.contains("gpl")
            || lic_lower.contains("agpl")
            || lic_lower.contains("lgpl")
            || lic_lower.contains("mpl")
        {
            return PreDownloadDecision::Reject {
                code: RejectionReasonCode::RejectForbiddenLicense,
                rule: "Rule 18: Copyleft (GPL/LGPL/AGPL/MPL/SA), Non-Commercial (NC), and No-Derivatives (ND) strictly forbidden",
                evidence: format!("license='{trimmed_lic}'"),
                details: format!("License '{trimmed_lic}' contains copyleft or restrictive clauses outside TARA permitted class"),
            };
        }

        // 4. Host Domain extraction and validation against verified open source domains
        let url_lower = trimmed_url.to_lowercase();
        let host = if let Some(stripped) = url_lower.strip_prefix("https://") {
            stripped.split('/').next().unwrap_or("")
        } else if let Some(stripped) = url_lower.strip_prefix("http://") {
            stripped.split('/').next().unwrap_or("")
        } else {
            url_lower.split('/').next().unwrap_or("")
        };

        // 5. Match against authoritative 48-source catalog
        // Priority 1: Match by dataset ID prefix (exact asset mapping)
        let matched_rule = self.allowed_sources.iter().find(|rule| {
            let host_match = rule.host_domain == host || url_lower.contains(rule.host_domain);
            let id_match = trimmed_id.starts_with(rule.dataset_id_prefix) || url_lower.contains(rule.dataset_id_prefix);
            host_match && id_match
        }).or_else(|| {
            // Priority 2: Match by specific URL prefix (path deeper than host root)
            self.allowed_sources.iter().find(|rule| {
                let host_match = rule.host_domain == host || url_lower.contains(rule.host_domain);
                let specific_url_prefix = rule.allowed_url_prefix.trim_end_matches('/').chars().filter(|&c| c == '/').count() >= 3;
                host_match && specific_url_prefix && url_lower.starts_with(rule.allowed_url_prefix)
            })
        });

        let rule = match matched_rule {
            Some(r) => r,
            None => {
                return PreDownloadDecision::Reject {
                    code: RejectionReasonCode::RejectDomain,
                    rule: "Rule 21: Candidate source must be in verified authorized catalog",
                    evidence: format!("host='{host}', url='{trimmed_url}'"),
                    details: format!("Source host '{host}' or dataset ID '{trimmed_id}' is not in authorized allowlist catalog"),
                };
            }
        };

        // 6. Rights proof verification
        if !rule.ai_training_permitted || !rule.commercial_use_permitted {
            return PreDownloadDecision::Reject {
                code: RejectionReasonCode::RejectRightsUnknown,
                rule: "Rule 18 / Rule 21: Mandatory 4-way commercial and AI training rights proof",
                evidence: format!("ai_training={}, commercial_use={}", rule.ai_training_permitted, rule.commercial_use_permitted),
                details: "Commercial AI model training rights are restricted or unverified for this source".to_string(),
            };
        }

        // 7. Asset Scope Match verification
        if let Some(scope) = asset_scope {
            let scope_trimmed = scope.trim();
            if !scope_trimmed.is_empty() && !rule.asset_scope.contains(scope_trimmed) && !scope_trimmed.contains(rule.asset_scope) {
                return PreDownloadDecision::Reject {
                    code: RejectionReasonCode::RejectScopeMismatch,
                    rule: "Rule 18: Exact asset-level scoping mandatory",
                    evidence: format!("provided_scope='{scope_trimmed}', expected_scope='{}'", rule.asset_scope),
                    details: format!("Provided scope '{scope_trimmed}' does not match authorized scope '{}'", rule.asset_scope),
                };
            }
        }

        // 8. Third-party exclusions verification
        if !rule.required_exclusions.is_empty() {
            let exclusions_lower = rule.required_exclusions.to_lowercase();
            for excl in exclusions_lower.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
                if url_lower.contains(excl) {
                    return PreDownloadDecision::Reject {
                        code: RejectionReasonCode::RejectThirdPartyContent,
                        rule: "Rule 18 / Rule 21: Excluded third-party or copyleft assets must not be ingested",
                        evidence: format!("url='{trimmed_url}', excluded_item='{excl}'"),
                        details: format!("URL matches excluded third-party asset '{}'", excl),
                    };
                }
            }
        }

        PreDownloadDecision::Accept
    }
}

/// Operational decision returned by the filter engine.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterDecision {
    /// Approved: quality criteria passed, domain classified via structural indicator match ratio.
    Approve {
        domain: String,
        indicator_score: f64,
    },
    /// Rejected: quality criteria failed with specific root cause reason.
    Reject(String),
}

impl FilterDecision {
    pub fn is_approved(&self) -> bool {
        matches!(self, FilterDecision::Approve { .. })
    }

    pub fn is_rejected(&self) -> bool {
        matches!(self, FilterDecision::Reject(_))
    }

    pub fn reason_code(&self) -> Option<RejectionReasonCode> {
        match self {
            FilterDecision::Approve { .. } => None,
            FilterDecision::Reject(r) => {
                if r.contains("REJECT_NO_LICENSE") || r.contains("Missing license") {
                    Some(RejectionReasonCode::RejectNoLicense)
                } else if r.contains("REJECT_RIGHTS_UNKNOWN") || r.contains("unproven") {
                    Some(RejectionReasonCode::RejectRightsUnknown)
                } else if r.contains("REJECT_FORBIDDEN_LICENSE") || r.contains("copyleft") || r.contains("non-commercial") || r.contains("no-derivatives") {
                    Some(RejectionReasonCode::RejectForbiddenLicense)
                } else if r.contains("REJECT_SCOPE_MISMATCH") {
                    Some(RejectionReasonCode::RejectScopeMismatch)
                } else if r.contains("REJECT_THIRD_PARTY_CONTENT") {
                    Some(RejectionReasonCode::RejectThirdPartyContent)
                } else if r.contains("REJECT_SYNTHETIC") || r.contains("synthetic") || r.contains("Synthetic") {
                    Some(RejectionReasonCode::RejectSynthetic)
                } else if r.contains("REJECT_DUPLICATE") || r.contains("Duplicate content") {
                    Some(RejectionReasonCode::RejectDuplicate)
                } else if r.contains("REJECT_NEAR_DUPLICATE") || r.contains("Near-duplicate") {
                    Some(RejectionReasonCode::RejectNearDuplicate)
                } else if r.contains("REJECT_EMPTY") || r.contains("Empty content") || r.contains("empty") {
                    Some(RejectionReasonCode::RejectEmpty)
                } else if r.contains("REJECT_TOO_SHORT") || r.contains("below minimum") || r.contains("Weak / incomplete") {
                    Some(RejectionReasonCode::RejectTooShort)
                } else if r.contains("REJECT_NOISY") || r.contains("noise") || r.contains("boilerplate") || r.contains("Abnormal character") {
                    Some(RejectionReasonCode::RejectNoisy)
                } else if r.contains("REJECT_MALFORMED") || r.contains("Malformed") || r.contains("syntax error") || r.contains("control characters") {
                    Some(RejectionReasonCode::RejectMalformed)
                } else if r.contains("REJECT_PROVENANCE") {
                    Some(RejectionReasonCode::RejectProvenance)
                } else if r.contains("REJECT_DOMAIN") {
                    Some(RejectionReasonCode::RejectDomain)
                } else {
                    Some(RejectionReasonCode::RejectMalformed)
                }
            }
        }
    }
}

// ============================================================================
// 2. Filter Engine Implementation
// ============================================================================

/// Production intelligence and dataset quality filter engine.
pub struct FilterEngine {
    pub config: FilterConfig,
    pub seen_exact_hashes: HashSet<u64>,
    pub seen_sha256_digests: HashSet<String>,
    pub seen_simhashes: Vec<u64>,
    pub domain_keywords: HashMap<String, Vec<&'static str>>,
}

impl Default for FilterEngine {
    fn default() -> Self {
        let mut domain_keywords = HashMap::new();

        // 1. Mathematics (Rule 18: Universal human knowledge)
        domain_keywords.insert(
            "mathematics".to_string(),
            vec![
                "calculus", "algebra", "geometry", "trigonometry", "matrix",
                "vector", "polynomial", "integral", "derivative", "differential",
                "theorem", "lemma", "proof", "statistics", "probability",
                "topology", "discrete math", "combinatorics", "prime number",
                "linear algebra", "tensor", "eigenvalue", "arithmetic", "fraction",
            ],
        );

        // 2. Computer Science & Systems (Rule 18)
        domain_keywords.insert(
            "computer_science".to_string(),
            vec![
                "algorithm", "data structure", "operating system", "compiler",
                "memory", "concurrency", "thread", "pointer", "rust", "python",
                "binary", "bitwise", "graph", "tree", "sorting", "hashing",
                "stack", "queue", "network", "protocol", "encryption",
                "cryptography", "database", "sql", "api", "bytecode", "cpu",
            ],
        );

        // 3. Physics (Rule 18)
        domain_keywords.insert(
            "physics".to_string(),
            vec![
                "physics", "mechanics", "velocity", "acceleration", "momentum",
                "energy", "kinetic", "potential", "gravity", "gravitation",
                "thermodynamics", "entropy", "quantum", "electromagnetism",
                "magnetic", "electric field", "wave", "photon", "relativity",
                "optics", "astrophysics", "particle physics", "kinematics", "dynamics",
            ],
        );

        // 4. Chemistry (Rule 18)
        domain_keywords.insert(
            "chemistry".to_string(),
            vec![
                "chemistry", "molecule", "molecular", "atom", "atomic",
                "organic chemistry", "inorganic", "acid", "base", "ph level",
                "reaction", "stoichiometry", "covalent", "ionic bond",
                "periodic table", "catalyst", "oxidation", "reduction",
                "enthalpy", "molar mass", "solution", "solvent",
            ],
        );

        // 5. Biology & Life Sciences (Rule 18)
        domain_keywords.insert(
            "biology".to_string(),
            vec![
                "biology", "cell", "cellular", "genetics", "dna", "rna",
                "protein", "enzyme", "organism", "ecology", "evolution",
                "mutation", "ribosome", "chromosome", "membrane", "mitochondria",
                "microbiology", "physiology", "anatomy", "neuron", "species",
            ],
        );

        // 6. Engineering & Technology (Rule 18)
        domain_keywords.insert(
            "engineering".to_string(),
            vec![
                "engineering", "electrical", "mechanical", "civil", "structural",
                "circuit", "voltage", "current", "resistor", "capacitor",
                "transistor", "microcontroller", "robotics", "signal processing",
                "control systems", "stress", "strain", "fluid dynamics", "actuator",
            ],
        );

        // 7. General Science & Empirical Knowledge (Rule 18)
        domain_keywords.insert(
            "general_science".to_string(),
            vec![
                "scientific method", "hypothesis", "experiment", "empirical",
                "peer review", "observation", "laboratory", "measurement",
                "research", "analysis", "evidence", "discovery", "academic",
            ],
        );

        Self {
            config: FilterConfig::default(),
            seen_exact_hashes: HashSet::new(),
            seen_sha256_digests: HashSet::new(),
            seen_simhashes: Vec::new(),
            domain_keywords,
        }
    }
}

impl FilterEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(config: FilterConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            config,
            ..Default::default()
        })
    }

    pub fn pre_check_candidate(
        &self,
        dataset_id: &str,
        source_url: &str,
        author: &str,
    ) -> FilterDecision {
        let combined = format!("{} {} {}", dataset_id, source_url, author).to_lowercase();

        if combined.contains("alpaca")
            || combined.contains("codealpaca")
            || combined.contains("slimorca")
            || combined.contains("openhermes")
            || combined.contains("sharegpt")
            || combined.contains("evol-instruct")
            || combined.contains("evol_instruct")
            || combined.contains("synthetic")
            || combined.contains("chatgpt")
            || combined.contains("text-davinci")
            || combined.contains("gpt-4-distill")
        {
            return FilterDecision::Reject(
                "Configured synthetic / AI-distilled indicator pattern detected (Rule 1 & Rule 18)".to_string(),
            );
        }

        let lower_id = dataset_id.trim().to_lowercase();
        if lower_id.is_empty()
            || lower_id == "dummy"
            || lower_id == "temp"
            || lower_id == "test"
            || lower_id == "sample"
            || lower_id == "placeholder"
        {
            return FilterDecision::Reject(
                format!("Invalid placeholder or dummy dataset identifier '{}'", dataset_id),
            );
        }

        FilterDecision::Approve {
            domain: "pending_download_content_analysis".to_string(),
            indicator_score: 1.0,
        }
    }

    pub fn evaluate_text(&mut self, text: &str, title_hint: &str) -> FilterDecision {
        let trimmed = text.trim();

        // Gate 1: Empty check
        if trimmed.is_empty() {
            return FilterDecision::Reject("Empty content: 0 characters (weak data)".to_string());
        }

        // Gate 2: Content length bounds
        let char_count = trimmed.chars().count();
        if char_count < self.config.min_content_chars {
            return FilterDecision::Reject(format!(
                "Content length {} characters is below minimum configured bound {}",
                char_count, self.config.min_content_chars
            ));
        }
        if char_count > self.config.max_content_chars {
            return FilterDecision::Reject(format!(
                "Content length {} characters exceeds maximum configured bound {}",
                char_count, self.config.max_content_chars
            ));
        }

        // Gate 2b: Word count bound
        let (vocab_ratio, word_count) = calculate_vocabulary_diversity(trimmed);
        if word_count < self.config.min_word_count {
            return FilterDecision::Reject(format!(
                "Word count {} is below minimum configured bound {}",
                word_count, self.config.min_word_count
            ));
        }

        // Gate 3: Character distribution & Noise Resistance
        let mut alphanumeric_count = 0usize;
        let mut control_count = 0usize;
        for c in trimmed.chars() {
            if c.is_alphanumeric() || c.is_whitespace() || is_valid_academic_punctuation(c) {
                alphanumeric_count += 1;
            } else if c.is_control() && c != '\n' && c != '\r' && c != '\t' {
                control_count += 1;
            }
        }

        let alpha_ratio = alphanumeric_count as f64 / char_count as f64;
        if alpha_ratio < self.config.min_alphanumeric_ratio {
            return FilterDecision::Reject(format!(
                "Abnormal character distribution: alphanumeric/valid ratio {:.3} < min threshold {:.3} (corrupt or binary noise)",
                alpha_ratio, self.config.min_alphanumeric_ratio
            ));
        }

        let control_ratio = control_count as f64 / char_count as f64;
        if control_ratio > self.config.max_control_char_ratio {
            return FilterDecision::Reject(format!(
                "Excessive unprintable control characters: ratio {:.3} > max threshold {:.3} (malformed stream)",
                control_ratio, self.config.max_control_char_ratio
            ));
        }

        // Gate 3b: Corrupted / OCR Garbage Detection
        let ocr_ratio = calculate_ocr_noise_ratio(trimmed);
        if ocr_ratio > self.config.max_ocr_noise_ratio {
            return FilterDecision::Reject(format!(
                "Corrupted / OCR garbage detected: noisy symbol ratio {:.3} exceeds threshold {:.3}",
                ocr_ratio, self.config.max_ocr_noise_ratio
            ));
        }

        // Gate 3c: Excessive Repeated Character Run
        let max_repeat = detect_max_consecutive_char_repeat(trimmed);
        if max_repeat > self.config.max_consecutive_char_repetition {
            return FilterDecision::Reject(format!(
                "Excessive repeated characters detected: run of {} consecutive characters exceeds threshold {}",
                max_repeat, self.config.max_consecutive_char_repetition
            ));
        }

        // Gate 3d: HTML/JS Markup & Navigation Boilerplate Dominance
        let html_ratio = calculate_html_boilerplate_ratio(trimmed);
        if html_ratio > self.config.max_html_tag_ratio {
            return FilterDecision::Reject(format!(
                "HTML/JS boilerplate dominance detected: tag ratio {:.3} exceeds threshold {:.3}",
                html_ratio, self.config.max_html_tag_ratio
            ));
        }

        // Gate 4: Rule 1 & Rule 18: Synthetic / AI indicator detection
        if let Some(synthetic_marker) = detect_synthetic_ai_markers(trimmed) {
            return FilterDecision::Reject(format!(
                "Configured synthetic indicator '{}' detected (Rule 1 & Rule 18)",
                synthetic_marker
            ));
        }

        // Gate 4b: Security & Piracy Policy Violation
        if let Some(piracy_marker) = detect_piracy_security_markers(trimmed) {
            return FilterDecision::Reject(format!(
                "Security & Piracy violation rejected: '{}' detected",
                piracy_marker
            ));
        }

        // Gate 4c: Weak review fragment
        if let Some(fragment_marker) = detect_weak_review_fragment(trimmed) {
            return FilterDecision::Reject(format!(
                "Weak / incomplete content fragment rejected: '{}'",
                fragment_marker
            ));
        }

        // Gate 4d: Truncated fragment
        if let Some(trunc_marker) = detect_truncated_fragment(trimmed) {
            return FilterDecision::Reject(format!(
                "Truncated / incomplete fragment rejected: '{}'",
                trunc_marker
            ));
        }

        // Gate 4e: Website legal / privacy policy
        if let Some(legal_marker) = detect_legal_privacy_policy_boilerplate(trimmed) {
            return FilterDecision::Reject(format!(
                "Website legal / Privacy policy boilerplate rejected: '{}'",
                legal_marker
            ));
        }

        // Gate 4f: Merchandise branding trap
        if let Some(brand_marker) = detect_merchandise_branding_marker(trimmed) {
            return FilterDecision::Reject(format!(
                "Commercial merchandise / branding trap rejected: '{}' is not an AI training license",
                brand_marker
            ));
        }

        // Gate 4g: Malicious Code & Exploit Payload Detection
        if self.config.reject_malicious_code {
            if let Some(malware_marker) = detect_malicious_code(trimmed) {
                return FilterDecision::Reject(format!(
                    "Malicious code / exploit pattern rejected: '{}' detected",
                    malware_marker
                ));
            }
        }

        // Gate 4h: Toxic, Scam, Gambling, and Adult Content Detection
        if self.config.reject_toxic_spam {
            if let Some(spam_marker) = detect_toxic_and_scam_words(trimmed) {
                return FilterDecision::Reject(format!(
                    "Toxic / scam / adult content rejected: '{}' detected",
                    spam_marker
                ));
            }
        }

        // Gate 4i: PII / Contact List Email Harvest Detection
        let emails = extract_email_addresses(trimmed);
        if emails.len() > self.config.max_allowed_emails {
            return FilterDecision::Reject(format!(
                "PII / contact list dump rejected: {} email addresses detected (threshold of {})",
                emails.len(),
                self.config.max_allowed_emails
            ));
        }

        // Gate 4j: Repetitive Line / Word Spam Detection
        if let Some(repeat_marker) = detect_repetitive_line_noise(trimmed, self.config.max_consecutive_word_repetition) {
            return FilterDecision::Reject(format!(
                "Repetitive word / line noise rejected: {}",
                repeat_marker
            ));
        }

        // Gate 4k: Vocabulary Diversity / Low-Entropy SEO Keyword Stuffing Detection
        // Per Herdan's law (Heaps' law), Type-Token Ratio naturally decreases with text volume.
        // For short snippets (<1000 words), config threshold (e.g. 0.20) flags keyword-stuffed SEO spam.
        // For long literary and scientific books (>1000 words), authentic human vocabulary diversity threshold
        // scales to prevent false positives on genuine books while still rejecting low-entropy spam loops.
        let effective_min_ttr = if word_count >= 10_000 {
            (self.config.min_vocabulary_diversity_ratio * 0.10).min(0.02)
        } else if word_count >= 1_000 {
            (self.config.min_vocabulary_diversity_ratio * 0.25).min(0.05)
        } else {
            self.config.min_vocabulary_diversity_ratio
        };

        if word_count >= 25 && vocab_ratio < effective_min_ttr {
            return FilterDecision::Reject(format!(
                "Low vocabulary diversity / SEO stuffing detected: unique word ratio {:.3} < threshold {:.3}",
                vocab_ratio, effective_min_ttr
            ));
        }

        // Gate 5: Publishing boilerplate
        if let Some(boilerplate_reason) = detect_publishing_boilerplate(trimmed) {
            return FilterDecision::Reject(format!(
                "Publishing boilerplate / editorial artifact rejected: {}",
                boilerplate_reason
            ));
        }

        // Gate 6: Balanced-Delimiter Structural Syntax Validation
        if !evaluate_bracket_structure(trimmed) {
            return FilterDecision::Reject(
                "Balanced-delimiter syntax error: unbalanced or incorrectly nested brackets ([)]".to_string(),
            );
        }

        // Gate 7: Two-Tier Exact Deduplication
        let exact_fnv = fnv1a_64(trimmed.as_bytes());
        let exact_sha = compute_sha256_digest(trimmed.as_bytes());

        if self.seen_exact_hashes.contains(&exact_fnv) && self.seen_sha256_digests.contains(&exact_sha) {
            return FilterDecision::Reject(format!(
                "Duplicate content confirmed: cryptographic SHA-256 match ({}) already registered",
                &exact_sha[..16]
            ));
        }

        // Gate 8: Near-Deduplication
        let simhash = compute_simhash_64(trimmed);
        for &existing in &self.seen_simhashes {
            let dist = hamming_distance_64(simhash, existing);
            if dist <= self.config.simhash_hamming_threshold {
                return FilterDecision::Reject(format!(
                    "Near-duplicate content rejected: SimHash Hamming distance {} <= threshold {}",
                    dist, self.config.simhash_hamming_threshold
                ));
            }
        }

        // Gate 9: Structural Domain Indicator Scoring
        let (domain, indicator_score) = self.classify_domain(trimmed, title_hint);
        if indicator_score < self.config.min_domain_indicator_ratio {
            return FilterDecision::Reject(format!(
                "Domain indicator match ratio {:.3} below minimum configured threshold {:.3}",
                indicator_score, self.config.min_domain_indicator_ratio
            ));
        }

        self.seen_exact_hashes.insert(exact_fnv);
        self.seen_sha256_digests.insert(exact_sha);
        self.seen_simhashes.push(simhash);

        FilterDecision::Approve { domain, indicator_score }
    }

    pub fn evaluate_file(&mut self, file_path: &Path, title_hint: &str) -> FilterDecision {
        if !file_path.exists() {
            return FilterDecision::Reject(format!("File does not exist: {}", file_path.display()));
        }

        let metadata = match std::fs::metadata(file_path) {
            Ok(m) => m,
            Err(e) => return FilterDecision::Reject(format!("Failed to read file metadata: {e}")),
        };

        if metadata.len() == 0 {
            return FilterDecision::Reject("File is empty: 0 bytes (weak data)".to_string());
        }

        let file = match File::open(file_path) {
            Ok(f) => f,
            Err(e) => return FilterDecision::Reject(format!("Failed to open file: {e}")),
        };

        let is_jsonl = file_path.extension().and_then(|s| s.to_str()) == Some("jsonl");
        let is_txt = file_path.extension().and_then(|s| s.to_str()) == Some("txt");

        if is_jsonl {
            let reader = BufReader::new(file);
            let mut sample_count = 0usize;
            let mut approved_count = 0usize;
            let mut synthetic_or_corrupt_rejected = None;
            let mut aggregated_domain_counts: HashMap<String, usize> = HashMap::new();

            for line_res in reader.lines().take(100) {
                let line = match line_res {
                    Ok(l) => l,
                    Err(e) => return FilterDecision::Reject(format!("Malformed line in JSONL: {e}")),
                };

                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                sample_count += 1;
                let text_to_check = extract_text_from_json_line(trimmed);
                match self.evaluate_text(&text_to_check, title_hint) {
                    FilterDecision::Approve { domain, .. } => {
                        approved_count += 1;
                        *aggregated_domain_counts.entry(domain).or_insert(0) += 1;
                    }
                    FilterDecision::Reject(reason) => {
                        if reason.contains("synthetic") || reason.contains("AI-distilled") {
                            synthetic_or_corrupt_rejected = Some(reason);
                            break;
                        }
                    }
                }
            }

            if let Some(err) = synthetic_or_corrupt_rejected {
                return FilterDecision::Reject(err);
            }

            if sample_count == 0 {
                return FilterDecision::Reject("JSONL file contains zero valid records (weak data)".to_string());
            }

            let pass_ratio = approved_count as f64 / sample_count as f64;
            if pass_ratio < self.config.min_jsonl_sample_pass_ratio {
                return FilterDecision::Reject(format!(
                    "JSONL sample quality pass ratio {:.1}% below {:.1}% requirement",
                    pass_ratio * 100.0,
                    self.config.min_jsonl_sample_pass_ratio * 100.0
                ));
            }

            let dominant_domain = aggregated_domain_counts
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(d, _)| d)
                .unwrap_or_else(|| "general_knowledge".to_string());

            FilterDecision::Approve {
                domain: dominant_domain,
                indicator_score: pass_ratio,
            }
        } else if is_txt {
            let mut reader = BufReader::new(file);
            let mut buffer = String::new();
            if let Err(e) = reader.read_to_string(&mut buffer) {
                return FilterDecision::Reject(format!("Failed to read text file content: {e}"));
            }
            self.evaluate_text(&buffer, title_hint)
        } else {
            let (domain, indicator_score) = self.classify_domain(title_hint, title_hint);
            FilterDecision::Approve { domain, indicator_score }
        }
    }

    pub fn classify_domain(&self, text: &str, title_hint: &str) -> (String, f64) {
        let text_lower = text.to_lowercase();
        let title_lower = title_hint.to_lowercase();

        let mut scores: HashMap<&str, usize> = HashMap::new();

        for (domain, keywords) in &self.domain_keywords {
            let mut count = 0usize;
            for &kw in keywords {
                if title_lower.contains(kw) {
                    count += 3;
                }
                if text_lower.contains(kw) {
                    count += 1;
                }
            }
            scores.insert(domain.as_str(), count);
        }

        let total_hits: usize = scores.values().sum();
        if total_hits == 0 {
            return ("general_knowledge".to_string(), 1.0);
        }

        let (best_domain, best_count) = scores
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .unwrap_or(("general_science", 0));

        let indicator_ratio = (best_count as f64 / total_hits as f64).min(1.0);
        (best_domain.to_string(), indicator_ratio)
    }

    /// Computes comprehensive content quality metrics for deep evaluation.
    pub fn compute_quality_metrics(&self, text: &str, title_hint: &str) -> ContentQualityMetrics {
        let trimmed = text.trim();
        let char_count = trimmed.chars().count();
        let words: Vec<&str> = trimmed.split_whitespace().collect();
        let word_count = words.len();
        let sentence_count = count_sentences(trimmed);

        let mut alpha_count = 0usize;
        let mut control_count = 0usize;
        for c in trimmed.chars() {
            if c.is_alphanumeric() || c.is_ascii_punctuation() || c.is_whitespace() {
                alpha_count += 1;
            }
            if c.is_control() && c != '\n' && c != '\r' && c != '\t' {
                control_count += 1;
            }
        }

        let alphanumeric_ratio = if char_count > 0 {
            alpha_count as f64 / char_count as f64
        } else {
            0.0
        };

        let control_char_ratio = if char_count > 0 {
            control_count as f64 / char_count as f64
        } else {
            0.0
        };

        let (vocabulary_diversity_ratio, _) = calculate_vocabulary_diversity(trimmed);
        let ocr_noise_ratio = calculate_ocr_noise_ratio(trimmed);
        let html_boilerplate_ratio = calculate_html_boilerplate_ratio(trimmed);
        let max_consecutive_char_repeat = detect_max_consecutive_char_repeat(trimmed);
        let balanced_structure = evaluate_bracket_structure(trimmed);
        let (classified_domain, domain_indicator_ratio) = self.classify_domain(trimmed, title_hint);

        // Dynamic quality score in [0.0, 1.0]: composite metric
        let mut score = alphanumeric_ratio * 0.35 + vocabulary_diversity_ratio * 0.35 + domain_indicator_ratio * 0.30;
        if !balanced_structure {
            score *= 0.5;
        }
        score -= ocr_noise_ratio * 2.0;
        score -= html_boilerplate_ratio * 2.0;
        score -= control_char_ratio * 5.0;
        let quality_score = score.clamp(0.0, 1.0);

        ContentQualityMetrics {
            char_count,
            word_count,
            sentence_count,
            alphanumeric_ratio,
            control_char_ratio,
            vocabulary_diversity_ratio,
            ocr_noise_ratio,
            html_boilerplate_ratio,
            max_consecutive_char_repeat,
            balanced_structure,
            domain_indicator_ratio,
            classified_domain,
            quality_score,
        }
    }

    /// Evaluates content across all 7 mandatory pipeline gates:
    /// 1. Rights Gate (rights_ok)
    /// 2. Scope Gate (scope_ok)
    /// 3. Provenance Gate (provenance_ok)
    /// 4. Quality Gate (quality_ok)
    /// 5. Duplication Gate (duplication_ok)
    /// 6. Structure Gate (structure_ok)
    /// 7. Domain Gate (domain_ok)
    ///
    /// Core principle: Quality != License
    /// Final condition: ACCEPT = rights_ok && scope_ok && provenance_ok && quality_ok && duplication_ok && structure_ok && domain_ok
    pub fn evaluate_pipeline(
        &mut self,
        text: &str,
        title_hint: &str,
        rights_ok: bool,
        scope_ok: bool,
        provenance_ok: bool,
    ) -> PipelineGateReport {
        // Gate 1: Rights Gate
        if !rights_ok {
            return PipelineGateReport {
                rights_ok: false,
                scope_ok,
                provenance_ok,
                quality_ok: false,
                duplication_ok: false,
                structure_ok: false,
                domain_ok: false,
                failure_code: Some(RejectionReasonCode::RejectRightsUnknown),
                failure_reason: Some("Rights Gate failed: Commercial or AI training rights not verified (Rule 18 / Rule 21)".to_string()),
                metrics: None,
            };
        }

        // Gate 2: Scope Gate
        if !scope_ok {
            return PipelineGateReport {
                rights_ok: true,
                scope_ok: false,
                provenance_ok,
                quality_ok: false,
                duplication_ok: false,
                structure_ok: false,
                domain_ok: false,
                failure_code: Some(RejectionReasonCode::RejectScopeMismatch),
                failure_reason: Some("Scope Gate failed: Asset scope does not match approved boundary".to_string()),
                metrics: None,
            };
        }

        // Gate 3: Provenance Gate
        if !provenance_ok {
            return PipelineGateReport {
                rights_ok: true,
                scope_ok: true,
                provenance_ok: false,
                quality_ok: false,
                duplication_ok: false,
                structure_ok: false,
                domain_ok: false,
                failure_code: Some(RejectionReasonCode::RejectProvenance),
                failure_reason: Some("Provenance Gate failed: Missing author or license proof URL (Rule 21)".to_string()),
                metrics: None,
            };
        }

        let trimmed = text.trim();
        let metrics = self.compute_quality_metrics(trimmed, title_hint);

        // Gate 4: Quality Gate
        if trimmed.is_empty() {
            return PipelineGateReport {
                rights_ok: true,
                scope_ok: true,
                provenance_ok: true,
                quality_ok: false,
                duplication_ok: false,
                structure_ok: false,
                domain_ok: false,
                failure_code: Some(RejectionReasonCode::RejectEmpty),
                failure_reason: Some("Quality Gate failed: Empty content (0 bytes)".to_string()),
                metrics: Some(metrics),
            };
        }

        if metrics.char_count < self.config.min_content_chars || metrics.word_count < self.config.min_word_count {
            return PipelineGateReport {
                rights_ok: true,
                scope_ok: true,
                provenance_ok: true,
                quality_ok: false,
                duplication_ok: false,
                structure_ok: false,
                domain_ok: false,
                failure_code: Some(RejectionReasonCode::RejectTooShort),
                failure_reason: Some(format!(
                    "Quality Gate failed: Content length {} chars or {} words below configured minimum thresholds",
                    trimmed.len(), metrics.word_count
                )),
                metrics: Some(metrics),
            };
        }

        if metrics.ocr_noise_ratio > self.config.max_ocr_noise_ratio {
            return PipelineGateReport {
                rights_ok: true,
                scope_ok: true,
                provenance_ok: true,
                quality_ok: false,
                duplication_ok: false,
                structure_ok: false,
                domain_ok: false,
                failure_code: Some(RejectionReasonCode::RejectNoisy),
                failure_reason: Some(format!(
                    "Quality Gate failed: OCR garbage / corrupted symbol ratio {:.3} exceeds threshold {:.3}",
                    metrics.ocr_noise_ratio, self.config.max_ocr_noise_ratio
                )),
                metrics: Some(metrics),
            };
        }

        if metrics.control_char_ratio > self.config.max_control_char_ratio || metrics.alphanumeric_ratio < self.config.min_alphanumeric_ratio {
            return PipelineGateReport {
                rights_ok: true,
                scope_ok: true,
                provenance_ok: true,
                quality_ok: false,
                duplication_ok: false,
                structure_ok: false,
                domain_ok: false,
                failure_code: Some(RejectionReasonCode::RejectMalformed),
                failure_reason: Some(format!(
                    "Quality Gate failed: Invalid encoding or abnormal character distribution (alpha: {:.3}, control: {:.3})",
                    metrics.alphanumeric_ratio, metrics.control_char_ratio
                )),
                metrics: Some(metrics),
            };
        }

        if metrics.max_consecutive_char_repeat > self.config.max_consecutive_char_repetition {
            return PipelineGateReport {
                rights_ok: true,
                scope_ok: true,
                provenance_ok: true,
                quality_ok: false,
                duplication_ok: false,
                structure_ok: false,
                domain_ok: false,
                failure_code: Some(RejectionReasonCode::RejectNoisy),
                failure_reason: Some(format!(
                    "Quality Gate failed: Excessive repeated character run of {} chars exceeds threshold {}",
                    metrics.max_consecutive_char_repeat, self.config.max_consecutive_char_repetition
                )),
                metrics: Some(metrics),
            };
        }

        if metrics.html_boilerplate_ratio > self.config.max_html_tag_ratio {
            return PipelineGateReport {
                rights_ok: true,
                scope_ok: true,
                provenance_ok: true,
                quality_ok: false,
                duplication_ok: false,
                structure_ok: false,
                domain_ok: false,
                failure_code: Some(RejectionReasonCode::RejectNoisy),
                failure_reason: Some(format!(
                    "Quality Gate failed: HTML/JS navigation boilerplate dominance ({:.3} > {:.3})",
                    metrics.html_boilerplate_ratio, self.config.max_html_tag_ratio
                )),
                metrics: Some(metrics),
            };
        }

        if let Some(synthetic_marker) = detect_synthetic_ai_markers(trimmed) {
            return PipelineGateReport {
                rights_ok: true,
                scope_ok: true,
                provenance_ok: true,
                quality_ok: false,
                duplication_ok: false,
                structure_ok: false,
                domain_ok: false,
                failure_code: Some(RejectionReasonCode::RejectSynthetic),
                failure_reason: Some(format!(
                    "Quality Gate failed: Synthetic / AI indicator '{}' detected (Rule 1 & Rule 18)",
                    synthetic_marker
                )),
                metrics: Some(metrics),
            };
        }

        if let Some(piracy_marker) = detect_piracy_security_markers(trimmed) {
            return PipelineGateReport {
                rights_ok: true,
                scope_ok: true,
                provenance_ok: true,
                quality_ok: false,
                duplication_ok: false,
                structure_ok: false,
                domain_ok: false,
                failure_code: Some(RejectionReasonCode::RejectMalformed),
                failure_reason: Some(format!("Quality Gate failed: Security & Piracy marker '{}' detected", piracy_marker)),
                metrics: Some(metrics),
            };
        }

        if let Some(fragment) = detect_weak_review_fragment(trimmed) {
            return PipelineGateReport {
                rights_ok: true,
                scope_ok: true,
                provenance_ok: true,
                quality_ok: false,
                duplication_ok: false,
                structure_ok: false,
                domain_ok: false,
                failure_code: Some(RejectionReasonCode::RejectTooShort),
                failure_reason: Some(format!("Quality Gate failed: Weak / incomplete content fragment '{}'", fragment)),
                metrics: Some(metrics),
            };
        }

        if let Some(trunc) = detect_truncated_fragment(trimmed) {
            return PipelineGateReport {
                rights_ok: true,
                scope_ok: true,
                provenance_ok: true,
                quality_ok: false,
                duplication_ok: false,
                structure_ok: false,
                domain_ok: false,
                failure_code: Some(RejectionReasonCode::RejectTooShort),
                failure_reason: Some(format!("Quality Gate failed: Truncated fragment '{}'", trunc)),
                metrics: Some(metrics),
            };
        }

        if let Some(legal) = detect_legal_privacy_policy_boilerplate(trimmed) {
            return PipelineGateReport {
                rights_ok: true,
                scope_ok: true,
                provenance_ok: true,
                quality_ok: false,
                duplication_ok: false,
                structure_ok: false,
                domain_ok: false,
                failure_code: Some(RejectionReasonCode::RejectNoisy),
                failure_reason: Some(format!("Quality Gate failed: Legal / privacy policy boilerplate '{}'", legal)),
                metrics: Some(metrics),
            };
        }

        if let Some(brand) = detect_merchandise_branding_marker(trimmed) {
            return PipelineGateReport {
                rights_ok: true,
                scope_ok: true,
                provenance_ok: true,
                quality_ok: false,
                duplication_ok: false,
                structure_ok: false,
                domain_ok: false,
                failure_code: Some(RejectionReasonCode::RejectNoisy),
                failure_reason: Some(format!("Quality Gate failed: Merchandise / branding trap '{}'", brand)),
                metrics: Some(metrics),
            };
        }

        if self.config.reject_malicious_code {
            if let Some(malware) = detect_malicious_code(trimmed) {
                return PipelineGateReport {
                    rights_ok: true,
                    scope_ok: true,
                    provenance_ok: true,
                    quality_ok: false,
                    duplication_ok: false,
                    structure_ok: false,
                    domain_ok: false,
                    failure_code: Some(RejectionReasonCode::RejectMalformed),
                    failure_reason: Some(format!("Quality Gate failed: Malicious code pattern '{}'", malware)),
                    metrics: Some(metrics),
                };
            }
        }

        if self.config.reject_toxic_spam {
            if let Some(spam) = detect_toxic_and_scam_words(trimmed) {
                return PipelineGateReport {
                    rights_ok: true,
                    scope_ok: true,
                    provenance_ok: true,
                    quality_ok: false,
                    duplication_ok: false,
                    structure_ok: false,
                    domain_ok: false,
                    failure_code: Some(RejectionReasonCode::RejectNoisy),
                    failure_reason: Some(format!("Quality Gate failed: Toxic / scam content '{}'", spam)),
                    metrics: Some(metrics),
                };
            }
        }

        let emails = extract_email_addresses(trimmed);
        if emails.len() > self.config.max_allowed_emails {
            return PipelineGateReport {
                rights_ok: true,
                scope_ok: true,
                provenance_ok: true,
                quality_ok: false,
                duplication_ok: false,
                structure_ok: false,
                domain_ok: false,
                failure_code: Some(RejectionReasonCode::RejectMalformed),
                failure_reason: Some(format!("Quality Gate failed: PII email harvest dump ({} emails)", emails.len())),
                metrics: Some(metrics),
            };
        }

        if let Some(rep_line) = detect_repetitive_line_noise(trimmed, self.config.max_consecutive_word_repetition) {
            return PipelineGateReport {
                rights_ok: true,
                scope_ok: true,
                provenance_ok: true,
                quality_ok: false,
                duplication_ok: false,
                structure_ok: false,
                domain_ok: false,
                failure_code: Some(RejectionReasonCode::RejectNoisy),
                failure_reason: Some(format!("Quality Gate failed: Repetitive line/word noise '{}'", rep_line)),
                metrics: Some(metrics),
            };
        }

        let effective_min_ttr = if metrics.word_count >= 10_000 {
            (self.config.min_vocabulary_diversity_ratio * 0.10).min(0.02)
        } else if metrics.word_count >= 1_000 {
            (self.config.min_vocabulary_diversity_ratio * 0.25).min(0.05)
        } else {
            self.config.min_vocabulary_diversity_ratio
        };

        if metrics.word_count >= 25 && metrics.vocabulary_diversity_ratio < effective_min_ttr {
            return PipelineGateReport {
                rights_ok: true,
                scope_ok: true,
                provenance_ok: true,
                quality_ok: false,
                duplication_ok: false,
                structure_ok: false,
                domain_ok: false,
                failure_code: Some(RejectionReasonCode::RejectNoisy),
                failure_reason: Some(format!(
                    "Quality Gate failed: Low vocabulary diversity / SEO keyword stuffing ({:.3} < {:.3})",
                    metrics.vocabulary_diversity_ratio, effective_min_ttr
                )),
                metrics: Some(metrics),
            };
        }

        if let Some(bp) = detect_publishing_boilerplate(trimmed) {
            return PipelineGateReport {
                rights_ok: true,
                scope_ok: true,
                provenance_ok: true,
                quality_ok: false,
                duplication_ok: false,
                structure_ok: false,
                domain_ok: false,
                failure_code: Some(RejectionReasonCode::RejectNoisy),
                failure_reason: Some(format!("Quality Gate failed: Publishing boilerplate '{}'", bp)),
                metrics: Some(metrics),
            };
        }

        // Gate 5: Duplication / Noise Gate
        let exact_fnv = fnv1a_64(trimmed.as_bytes());
        let exact_sha = compute_sha256_digest(trimmed.as_bytes());
        if self.seen_exact_hashes.contains(&exact_fnv) && self.seen_sha256_digests.contains(&exact_sha) {
            return PipelineGateReport {
                rights_ok: true,
                scope_ok: true,
                provenance_ok: true,
                quality_ok: true,
                duplication_ok: false,
                structure_ok: false,
                domain_ok: false,
                failure_code: Some(RejectionReasonCode::RejectDuplicate),
                failure_reason: Some(format!("Duplication Gate failed: Exact SHA-256 duplicate ({})", &exact_sha[..16])),
                metrics: Some(metrics),
            };
        }

        let simhash = compute_simhash_64(trimmed);
        for &existing in &self.seen_simhashes {
            let dist = hamming_distance_64(simhash, existing);
            if dist <= self.config.simhash_hamming_threshold {
                return PipelineGateReport {
                    rights_ok: true,
                    scope_ok: true,
                    provenance_ok: true,
                    quality_ok: true,
                    duplication_ok: false,
                    structure_ok: false,
                    domain_ok: false,
                    failure_code: Some(RejectionReasonCode::RejectNearDuplicate),
                    failure_reason: Some(format!(
                        "Duplication Gate failed: Near-duplicate content (SimHash distance {} <= threshold {})",
                        dist, self.config.simhash_hamming_threshold
                    )),
                    metrics: Some(metrics),
                };
            }
        }

        // Gate 6: Domain / Structure Gate - Structure check
        let structure_ok = metrics.balanced_structure;
        if !structure_ok {
            return PipelineGateReport {
                rights_ok: true,
                scope_ok: true,
                provenance_ok: true,
                quality_ok: true,
                duplication_ok: true,
                structure_ok: false,
                domain_ok: false,
                failure_code: Some(RejectionReasonCode::RejectMalformed),
                failure_reason: Some("Structure Gate failed: Unbalanced delimiters / malformed syntax".to_string()),
                metrics: Some(metrics),
            };
        }

        // Gate 7: Domain / Structure Gate - Domain check
        let domain_ok = metrics.domain_indicator_ratio >= self.config.min_domain_indicator_ratio;
        if !domain_ok {
            return PipelineGateReport {
                rights_ok: true,
                scope_ok: true,
                provenance_ok: true,
                quality_ok: true,
                duplication_ok: true,
                structure_ok: true,
                domain_ok: false,
                failure_code: Some(RejectionReasonCode::RejectDomain),
                failure_reason: Some(format!(
                    "Domain Gate failed: Domain indicator score {:.3} below threshold {:.3}",
                    metrics.domain_indicator_ratio, self.config.min_domain_indicator_ratio
                )),
                metrics: Some(metrics),
            };
        }

        // Passed all 7 gates!
        self.seen_exact_hashes.insert(exact_fnv);
        self.seen_sha256_digests.insert(exact_sha);
        self.seen_simhashes.push(simhash);

        PipelineGateReport {
            rights_ok: true,
            scope_ok: true,
            provenance_ok: true,
            quality_ok: true,
            duplication_ok: true,
            structure_ok: true,
            domain_ok: true,
            failure_code: None,
            failure_reason: None,
            metrics: Some(metrics),
        }
    }
}

// ============================================================================
// 3. Native Algorithms (Bracket Stack, FNV-1a, SHA-256, SimHash, Boilerplate)
// ============================================================================

pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3u64);
    }
    hash
}

pub fn compute_sha256_digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let result = hasher.finalize();
    let mut hex_str = String::with_capacity(64);
    for b in result {
        use std::fmt::Write;
        let _ = write!(hex_str, "{:02x}", b);
    }
    hex_str
}

pub fn compute_simhash_64(text: &str) -> u64 {
    let mut bit_weights = [0i32; 64];

    let words: Vec<&str> = text
        .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
        .filter(|w| !w.is_empty())
        .collect();

    if words.is_empty() {
        return 0;
    }

    if words.len() < 3 {
        return fnv1a_64(text.as_bytes());
    }

    for window in words.windows(3) {
        let mut shingle = Vec::with_capacity(window[0].len() + window[1].len() + window[2].len() + 2);
        shingle.extend_from_slice(window[0].as_bytes());
        shingle.push(b' ');
        shingle.extend_from_slice(window[1].as_bytes());
        shingle.push(b' ');
        shingle.extend_from_slice(window[2].as_bytes());

        let h = fnv1a_64(&shingle);
        for (bit, w) in bit_weights.iter_mut().enumerate() {
            if ((h >> bit) & 1) == 1 {
                *w += 1;
            } else {
                *w -= 1;
            }
        }
    }

    let mut signature = 0u64;
    for (bit, &w) in bit_weights.iter().enumerate() {
        if w > 0 {
            signature |= 1u64 << bit;
        }
    }
    signature
}

pub fn hamming_distance_64(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

pub fn evaluate_bracket_structure(text: &str) -> bool {
    if !text.contains('{') && !text.contains('(') && !text.contains('[') {
        return true;
    }

    let mut stack = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    let mut in_single_quote = false;
    let mut in_double_quote = false;
    let mut in_line_comment = false;
    let mut in_block_comment = false;

    let mut total_delimiters = 0usize;
    let mut mismatch_count = 0usize;

    while i < len {
        let c = chars[i];
        let next_c = if i + 1 < len { Some(chars[i + 1]) } else { None };

        if in_line_comment {
            if c == '\n' {
                in_line_comment = false;
            }
            i += 1;
            continue;
        }

        if in_block_comment {
            if c == '*' && next_c == Some('/') {
                in_block_comment = false;
                i += 2;
                continue;
            }
            i += 1;
            continue;
        }

        if (in_single_quote || in_double_quote) && c == '\\' {
            i += 2;
            continue;
        }

        if c == '\'' && !in_double_quote {
            // Apostrophe inside word: e.g. "don't", "it's", "Smith's"
            let is_apostrophe = (i > 0 && chars[i - 1].is_alphabetic())
                || (i + 1 < len && chars[i + 1].is_alphabetic());
            if !is_apostrophe {
                in_single_quote = !in_single_quote;
            }
            i += 1;
            continue;
        }
        if c == '"' && !in_single_quote {
            in_double_quote = !in_double_quote;
            i += 1;
            continue;
        }

        if in_single_quote || in_double_quote {
            i += 1;
            continue;
        }

        if c == '/' && next_c == Some('/') {
            in_line_comment = true;
            i += 2;
            continue;
        }
        if c == '/' && next_c == Some('*') {
            in_block_comment = true;
            i += 2;
            continue;
        }

        match c {
            '{' | '(' | '[' => {
                total_delimiters += 1;
                stack.push(c);
            }
            '}' => {
                total_delimiters += 1;
                if stack.pop() != Some('{') {
                    mismatch_count += 1;
                }
            }
            ')' => {
                total_delimiters += 1;
                if stack.pop() != Some('(') {
                    // Check if list bullet: e.g. "1)", "a)", "ii)", "iii)", "iv)"
                    let mut prev_idx = i;
                    while prev_idx > 0 && chars[prev_idx - 1].is_alphanumeric() {
                        prev_idx -= 1;
                    }
                    let is_list_bullet = prev_idx < i
                        && (prev_idx == 0
                            || chars[prev_idx - 1].is_whitespace()
                            || chars[prev_idx - 1] == '.'
                            || chars[prev_idx - 1] == '\n');
                    let is_smiley = i > 0 && (chars[i - 1] == ':' || chars[i - 1] == ';');
                    if !is_list_bullet && !is_smiley {
                        mismatch_count += 1;
                    }
                }
            }
            ']' => {
                total_delimiters += 1;
                if stack.pop() != Some('[') {
                    mismatch_count += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let unclosed = stack.len();
    if unclosed == 0 && mismatch_count == 0 {
        return true;
    }

    // In short code snippets (< 5,000 chars), any structural syntax error is fatal
    if len < 5000 {
        return false;
    }

    // In long multi-megabyte literary and scientific texts, allow at most 1.0% error rate for stray punctuation
    if total_delimiters > 0 {
        let error_rate = (unclosed + mismatch_count) as f64 / total_delimiters as f64;
        return error_rate < 0.01;
    }

    true
}

pub fn detect_synthetic_ai_markers(text: &str) -> Option<&'static str> {
    let lower = text.to_lowercase();

    let markers = [
        ("as an ai language model", "AS_AN_AI_LANGUAGE_MODEL"),
        ("as a large language model", "AS_A_LARGE_LANGUAGE_MODEL"),
        ("developed by openai", "DEVELOPED_BY_OPENAI"),
        ("i do not have personal opinions", "AI_DISCLAIMER_OPINIONS"),
        ("i don't have personal opinions", "AI_DISCLAIMER_OPINIONS"),
        ("i do not have access to real-time", "AI_DISCLAIMER_REALTIME"),
        ("alpaca_data", "ALPACA_DATASET_MARKER"),
        ("sharegpt", "SHAREGPT_DATASET_MARKER"),
        ("evol-instruct", "EVOL_INSTRUCT_MARKER"),
        ("distilled from gpt", "GPT_DISTILLATION_MARKER"),
        ("generated by chatgpt", "CHATGPT_GENERATED_MARKER"),
    ];

    for (pattern, tag) in markers {
        if lower.contains(pattern) {
            return Some(tag);
        }
    }

    None
}

pub fn detect_publishing_boilerplate(text: &str) -> Option<&'static str> {
    let lower = text.to_lowercase();

    if lower.contains("table of contents")
        && (lower.contains("chapter 1") || lower.contains("unit 1") || lower.contains("page"))
    {
        return Some("TABLE_OF_CONTENTS_BOILERPLATE");
    }

    if lower.contains("answers to selected questions")
        || lower.contains("answers to chapter review")
        || lower.contains("practice final exam")
        || lower.contains("review exercises")
    {
        return Some("TEXTBOOK_EXERCISE_APPENDIX_BOILERPLATE");
    }

    if lower.contains("about openstax")
        || lower.contains("openstax is a 501(c)(3)")
        || lower.contains("library of congress cataloging")
        || lower.contains("isbn-10")
        || lower.contains("isbn-13")
    {
        return Some("PUBLISHER_COPYRIGHT_PAGE_BOILERPLATE");
    }

    None
}

pub fn detect_piracy_security_markers(text: &str) -> Option<&'static str> {
    let lower = text.to_lowercase();

    let piracy_markers = [
        ("mod apk", "MOD_APK_PIRACY"),
        ("unlocked apk", "MOD_APK_PIRACY"),
        ("apk mod", "MOD_APK_PIRACY"),
        ("free crack", "CRACKED_SOFTWARE_PIRACY"),
        ("cracked apk", "CRACKED_SOFTWARE_PIRACY"),
        ("crack download", "CRACKED_SOFTWARE_PIRACY"),
        ("keygen", "KEYGEN_WAREZ_PIRACY"),
        ("warez", "WAREZ_PIRACY"),
        ("hack apk", "HACKED_SOFTWARE_PIRACY"),
        ("cheat engine", "CHEAT_ENGINE_PIRACY"),
        ("unlimited money mod", "MOD_APK_PIRACY"),
    ];

    for (pattern, tag) in piracy_markers {
        if lower.contains(pattern) {
            return Some(tag);
        }
    }

    None
}

pub fn detect_weak_review_fragment(text: &str) -> Option<&'static str> {
    let lower = text.to_lowercase();

    if (lower.contains("out of 10 based on") || lower.contains("out of 5 based on") || lower.contains("user ratings"))
        && (lower.contains("ratings") || lower.contains("reviews"))
        && text.split_whitespace().count() < 30
    {
        return Some("WEAK_REVIEW_RATING_FRAGMENT");
    }

    None
}

pub fn detect_truncated_fragment(text: &str) -> Option<&'static str> {
    let trimmed = text.trim();
    if trimmed.ends_with("...") || trimmed.ends_with('…') {
        let stripped = trimmed.trim_end_matches('.').trim_end_matches('…').trim_end();
        let last_token = stripped.split_whitespace().last().unwrap_or("");
        if last_token.len() <= 3 || stripped.len() < 80 {
            return Some("TRUNCATED_FRAGMENT");
        }
    }
    None
}

pub fn detect_legal_privacy_policy_boilerplate(text: &str) -> Option<&'static str> {
    let lower = text.to_lowercase();
    if (lower.contains("privacy policy") || lower.contains("terms of service") || lower.contains("terms of use"))
        && (lower.contains("personal information") || lower.contains("we collect") || lower.contains("cookie policy") || lower.contains("contact us") || lower.contains("opt-out") || lower.contains("opt out"))
    {
        return Some("WEBSITE_LEGAL_PRIVACY_POLICY_BOILERPLATE");
    }
    None
}

pub fn detect_merchandise_branding_marker(text: &str) -> Option<&'static str> {
    let lower = text.to_lowercase();
    if lower.contains("officially licensed")
        || lower.contains("licensed merchandise")
        || lower.contains("licensed product")
    {
        return Some("OFFICIALLY_LICENSED_MERCHANDISE_TRAP");
    }
    None
}

fn is_valid_academic_punctuation(c: char) -> bool {
    matches!(
        c,
        '.' | ',' | ';' | ':' | '!' | '?' | '-' | '_' | '(' | ')' | '[' | ']'
            | '{' | '}' | '+' | '=' | '*' | '/' | '<' | '>' | '^' | '%' | '\\'
            | '|' | '&' | '$' | '#' | '@' | '\'' | '"' | '`' | '~'
    )
}

fn extract_text_from_json_line(line: &str) -> String {
    if let Some(pos) = line.find("\"text\":") {
        let after = line[pos + 7..].trim_start();
        if let Some(stripped) = after.strip_prefix('"') {
            if let Some(end_quote) = stripped.find('"') {
                return stripped[..end_quote].to_string();
            }
        }
    }
    line.to_string()
}

/// Calculates the ratio of words containing internal noisy symbols, unprintable glyphs, or OCR corruption.
pub fn calculate_ocr_noise_ratio(text: &str) -> f64 {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return 0.0;
    }
    let mut noisy_words = 0usize;
    for w in &words {
        let core = w.trim_matches(|c: char| matches!(c, '.' | ',' | ';' | ':' | '!' | '?' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}'));
        if core.len() >= 3 {
            // Exclude genuine file paths, URL fragments, and email addresses from OCR corruption counts
            if core.contains('/') || (core.contains('@') && core.contains('.')) || is_valid_email_token(core) {
                continue;
            }
            let mut internal_symbol_count = 0usize;
            let mut alpha_count = 0usize;
            for c in core.chars() {
                if c.is_alphabetic() {
                    alpha_count += 1;
                } else if matches!(c, '|' | '~' | '^' | '\\' | '`' | '§' | '¤') {
                    internal_symbol_count += 1;
                }
            }
            if internal_symbol_count > 0 && alpha_count > 0 {
                noisy_words += 1;
            }
        }
    }
    noisy_words as f64 / words.len() as f64
}

/// Measures the character ratio of raw HTML markup and web navigation boilerplate.
pub fn calculate_html_boilerplate_ratio(text: &str) -> f64 {
    if text.is_empty() {
        return 0.0;
    }
    let mut html_chars = 0usize;
    let mut in_tag = false;
    let lower = text.to_lowercase();

    for c in lower.chars() {
        if c == '<' {
            in_tag = true;
            html_chars += 1;
        } else if in_tag {
            html_chars += 1;
            if c == '>' {
                in_tag = false;
            }
        }
    }

    let boilerplate_terms = [
        "&nbsp;", "&amp;", "&quot;", "class=\"", "style=\"", "id=\"",
        "href=\"", "onclick=", "javascript:", "display:none",
        "cookie-policy", "nav-item", "navbar-nav",
    ];
    for term in boilerplate_terms {
        let count = lower.matches(term).count();
        html_chars += count * term.len();
    }

    (html_chars as f64 / text.len() as f64).min(1.0)
}

/// Computes the vocabulary diversity (Type-Token Ratio: unique words / total words).
pub fn calculate_vocabulary_diversity(text: &str) -> (f64, usize) {
    let mut word_set = std::collections::HashSet::new();
    let mut total_words = 0usize;

    for token in text.split(|c: char| c.is_whitespace() || matches!(c, ',' | '.' | ';' | ':' | '!' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '"' | '\'' | '-' | '/')) {
        let trimmed = token.trim();
        if !trimmed.is_empty() && trimmed.chars().any(|c| c.is_alphabetic()) {
            total_words += 1;
            word_set.insert(trimmed.to_lowercase());
        }
    }

    if total_words == 0 {
        return (0.0, 0);
    }

    let ratio = word_set.len() as f64 / total_words as f64;
    (ratio, total_words)
}

/// Detects the longest run of consecutive identical non-whitespace characters.
pub fn detect_max_consecutive_char_repeat(text: &str) -> usize {
    let mut max_run = 0usize;
    let mut current_run = 0usize;
    let mut prev_char = None;

    for c in text.chars() {
        if c.is_whitespace() || matches!(c, '-' | '=' | '*' | '_' | '#' | '.') {
            prev_char = None;
            current_run = 0;
            continue;
        }

        if Some(c) == prev_char {
            current_run += 1;
        } else {
            prev_char = Some(c);
            current_run = 1;
        }
        if current_run > max_run {
            max_run = current_run;
        }
    }

    max_run
}

/// Counts total complete sentences delimited by terminal punctuation.
pub fn count_sentences(text: &str) -> usize {
    text.chars().filter(|&c| matches!(c, '.' | '!' | '?')).count()
}

// ============================================================================
// 4. Email, Malicious Code, Toxic/Scam Words, and Line Filtering
// ============================================================================

/// Extracts all valid RFC-style email addresses from text.
pub fn extract_email_addresses(text: &str) -> Vec<String> {
    let mut emails = Vec::new();
    for token in text.split(|c: char| c.is_whitespace() || matches!(c, '<' | '>' | '(' | ')' | '[' | ']' | '"' | '\'' | ',' | ';')) {
        let trimmed = token.trim_matches(|c: char| matches!(c, '.' | ':' | '?' | '!' | '-' | '_'));
        if is_valid_email_token(trimmed) {
            emails.push(trimmed.to_string());
        }
    }
    emails
}

/// Validates whether a token represents an authentic personal or commercial email address.
pub fn is_valid_email_token(token: &str) -> bool {
    if token.len() < 5 || token.len() > 120 {
        return false;
    }
    let parts: Vec<&str> = token.split('@').collect();
    if parts.len() != 2 {
        return false;
    }
    let (local, domain) = (parts[0], parts[1]);
    if local.is_empty() || domain.is_empty() {
        return false;
    }

    if local.starts_with('.') || local.ends_with('.') {
        return false;
    }

    for c in local.chars() {
        if !c.is_alphanumeric() && !matches!(c, '.' | '_' | '%' | '+' | '-') {
            return false;
        }
    }

    if !domain.contains('.') || domain.starts_with('.') || domain.ends_with('.') {
        return false;
    }

    let domain_segments: Vec<&str> = domain.split('.').collect();
    if domain_segments.len() < 2 {
        return false;
    }

    let tld = domain_segments.last().unwrap();
    if tld.len() < 2 || tld.len() > 24 {
        return false;
    }
    for c in tld.chars() {
        if !c.is_ascii_alphabetic() {
            return false;
        }
    }

    for seg in &domain_segments {
        if seg.is_empty() {
            return false;
        }
        for c in seg.chars() {
            if !c.is_alphanumeric() && c != '-' {
                return false;
            }
        }
    }

    true
}

/// Redacts isolated email addresses in text to `[EMAIL_REDACTED]`.
pub fn redact_emails(text: &str) -> String {
    let emails = extract_email_addresses(text);
    if emails.is_empty() {
        return text.to_string();
    }
    let mut redacted = text.to_string();
    for email in emails {
        redacted = redacted.replace(&email, "[EMAIL_REDACTED]");
    }
    redacted
}

/// Detects malicious code, reverse shells, web shells, exploit payloads, and cryptominers.
/// Uses structural keyword pairs to avoid triggering AV false positives on literal exploits.
pub fn detect_malicious_code(text: &str) -> Option<&'static str> {
    let lower = text.to_lowercase();

    // 1. Interactive reverse shell payloads
    if (lower.contains("/bin/sh") || lower.contains("/bin/bash")) && lower.contains("-i") {
        return Some("REVERSE_SHELL_INTERACTIVE_SH");
    }
    if lower.contains("/dev/tcp/") && (lower.contains("bash") || lower.contains(">&")) {
        return Some("BASH_TCP_REVERSE_SHELL");
    }
    if (lower.contains("nc ") || lower.contains("ncat ")) && lower.contains(" -e ") {
        return Some("NETCAT_REVERSE_SHELL");
    }
    if lower.contains("mkfifo") && lower.contains("/tmp/") {
        return Some("MKFIFO_NAMED_PIPE_SHELL");
    }

    // 2. Dangerous web shells & code execution injections (keyword combinations)
    let has_eval = lower.contains("eval(") || lower.contains("assert(");
    let has_decode = lower.contains("base64_decode") || lower.contains("gzinflate");
    if has_eval && has_decode {
        return Some("DYNAMIC_EVAL_DECODE_WEBSHELL");
    }

    let has_exec_func = lower.contains("system(") || lower.contains("shell_exec(") || lower.contains("passthru(");
    let has_superglobals = lower.contains("$_get") || lower.contains("$_post") || lower.contains("$_request");
    if has_exec_func && has_superglobals {
        return Some("REMOTE_EXEC_SUPERGLOBAL_INJECTION");
    }

    // 3. Obfuscated PowerShell execution cradles
    if lower.contains("powershell") && (lower.contains("-enc") || lower.contains("-encodedcommand")) {
        return Some("POWERSHELL_ENCODED_EXECUTION");
    }
    if lower.contains("invoke-expression") && lower.contains("downloadstring") {
        return Some("POWERSHELL_IEX_DOWNLOAD_CRADLE");
    }
    if lower.contains("certutil") && lower.contains("-urlcache") && lower.contains("-split") {
        return Some("CERTUTIL_PAYLOAD_DOWNLOADER");
    }
    if lower.contains("vssadmin") && lower.contains("delete") && lower.contains("shadows") {
        return Some("RANSOMWARE_SHADOW_COPY_DELETION");
    }

    // 4. SQL Injection payloads
    if lower.contains("union select")
        || lower.contains("union all select")
        || lower.contains("union distinct select")
        || lower.contains("union/**/select")
        || lower.contains("union (select")
        || lower.contains("union(select")
    {
        return Some("SQL_INJECTION_UNION_SELECT");
    }
    if lower.contains("xp_cmdshell") {
        return Some("MSSQL_XP_CMDSHELL_EXPLOIT");
    }
    if lower.contains("' or '1'='1") || lower.contains("\" or \"1\"=\"1") {
        return Some("SQL_INJECTION_AUTH_BYPASS");
    }

    // 5. Cryptominers & Ransomware signatures
    if lower.contains("stratum+tcp://") || lower.contains("stratum+ssl://") {
        return Some("CRYPTOMINER_STRATUM_PROTOCOL");
    }
    if lower.contains("xmrig") && (lower.contains("miner") || lower.contains("pool") || lower.contains("cpu")) {
        return Some("XMRIG_MINER_SIGNATURE");
    }
    if lower.contains("coinhive.min.js") {
        return Some("COINHIVE_BROWSER_MINER");
    }
    if (lower.contains("files have been encrypted") || lower.contains("decrypt your files"))
        && (lower.contains("bitcoin") || lower.contains("ransom") || lower.contains("wallet"))
    {
        return Some("RANSOMWARE_NOTE_SIGNATURE");
    }

    None
}

/// Detects toxic, scam, gambling, and adult words/phrases.
pub fn detect_toxic_and_scam_words(text: &str) -> Option<&'static str> {
    let lower = text.to_lowercase();

    // 1. Adult / Exploitation
    let adult_markers = [
        ("pornography", "ADULT_EXPLICIT_CONTENT"),
        ("xxx videos", "ADULT_EXPLICIT_CONTENT"),
        ("escort service", "ADULT_ESCORT_SERVICE"),
        ("camgirls", "ADULT_CAMGIRL_SERVICE"),
        ("adult dating", "ADULT_DATING_SPAM"),
        ("nude webcam", "ADULT_WEBCAM_SPAM"),
        ("sex chat online", "ADULT_CHAT_SPAM"),
    ];
    for (pat, tag) in adult_markers {
        if lower.contains(pat) {
            return Some(tag);
        }
    }

    // 2. Gambling & Betting Spam
    let gambling_markers = [
        ("online casino bonus", "GAMBLING_CASINO_SPAM"),
        ("slot machine jackpot", "GAMBLING_SLOT_SPAM"),
        ("fixed betting odds", "BETTING_SPAM"),
        ("poker real money", "GAMBLING_POKER_SPAM"),
        ("free spins no deposit", "GAMBLING_FREE_SPINS_SPAM"),
        ("roulette strategy 100% win", "GAMBLING_ROULETTE_SPAM"),
    ];
    for (pat, tag) in gambling_markers {
        if lower.contains(pat) {
            return Some(tag);
        }
    }

    // 3. Financial Scams & Crypto Pumps
    let scam_markers = [
        ("free crypto giveaway", "CRYPTO_GIVEAWAY_SCAM"),
        ("send btc get double", "CRYPTO_DOUBLE_SCAM"),
        ("send eth get double", "CRYPTO_DOUBLE_SCAM"),
        ("guaranteed 1000% return", "FINANCIAL_SCAM_PROMISE"),
        ("pump and dump signal", "CRYPTO_PUMP_AND_DUMP"),
        ("earn $5000 a day from home", "WORK_FROM_HOME_SCAM"),
        ("payday loan no credit check", "PREDATORY_LOAN_SPAM"),
    ];
    for (pat, tag) in scam_markers {
        if lower.contains(pat) {
            return Some(tag);
        }
    }

    // 4. Pharma & Illicit Drugs
    let pharma_markers = [
        ("buy cheap viagra", "PHARMA_VIAGRA_SPAM"),
        ("generic cialis online", "PHARMA_CIALIS_SPAM"),
        ("buy oxycodone no prescription", "ILLICIT_PHARMA_SPAM"),
        ("illicit darknet market", "DARKNET_CONTRABAND_MARKER"),
    ];
    for (pat, tag) in pharma_markers {
        if lower.contains(pat) {
            return Some(tag);
        }
    }

    None
}

/// Detects repetitive word loops and character repetition noise in lines.
pub fn detect_repetitive_line_noise(text: &str, max_consecutive_words: usize) -> Option<&'static str> {
    // 1. Consecutive word repetition (e.g. "spam spam spam spam spam spam")
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() >= max_consecutive_words {
        let mut repeat_count = 1usize;
        for i in 1..words.len() {
            if words[i].eq_ignore_ascii_case(words[i - 1]) && words[i].len() > 1 {
                repeat_count += 1;
                if repeat_count >= max_consecutive_words {
                    return Some("EXCESSIVE_CONSECUTIVE_WORD_REPETITION");
                }
            } else {
                repeat_count = 1;
            }
        }
    }

    // 2. Consecutive character repetition within words/sentences (e.g. "xxxxxxxxxxxx")
    let mut prev_char = '\0';
    let mut char_repeat = 0usize;
    for c in text.chars() {
        if matches!(c, '-' | '=' | '*' | '_' | '#' | '.') {
            prev_char = '\0';
            char_repeat = 0;
            continue;
        }
        if c == prev_char && !c.is_whitespace() {
            char_repeat += 1;
            if char_repeat >= 15 {
                return Some("EXCESSIVE_CHARACTER_REPETITION_NOISE");
            }
        } else {
            prev_char = c;
            char_repeat = 1;
        }
    }

    // 3. Dominant single-character line noise (excluding standalone horizontal rules and dot leaders)
    for line in text.lines() {
        let trimmed_line = line.trim();
        let is_pure_horizontal_rule = !trimmed_line.is_empty()
            && trimmed_line.chars().all(|c| matches!(c, '-' | '=' | '*' | '_' | '#' | '.'));
        if is_pure_horizontal_rule {
            continue;
        }
        if trimmed_line.len() >= 25 {
            let mut char_counts: HashMap<char, usize> = HashMap::new();
            for c in trimmed_line.chars() {
                // Dot leaders in tables/TOCs (e.g. "Chapter 1 ....... 10") and whitespace are formatting, not noise
                if c != '.' && !c.is_whitespace() {
                    *char_counts.entry(c).or_insert(0) += 1;
                }
            }
            if let Some((_, &count)) = char_counts.iter().max_by_key(|&(_, c)| c) {
                if count as f64 / trimmed_line.len() as f64 >= 0.70 {
                    return Some("EXCESSIVE_CHARACTER_REPETITION_NOISE");
                }
            }
        }
    }

    None
}

/// Sanitizes text line-by-line: strips cookie banners, newsletter spam, and redacts emails.
pub fn sanitize_text_lines(text: &str) -> (String, usize) {
    let mut retained_lines = Vec::new();
    let mut dropped_count = 0usize;

    for line in text.lines() {
        let lower = line.to_lowercase();

        if lower.contains("we use cookies to enhance")
            || lower.contains("accept all cookies")
            || lower.contains("cookie preferences")
            || lower.contains("subscribe to our newsletter")
            || lower.contains("click here to unsubscribe")
            || lower.contains("all rights reserved. powered by")
        {
            dropped_count += 1;
            continue;
        }

        let redacted_line = redact_emails(line);
        retained_lines.push(redacted_line);
    }

    (retained_lines.join("\n"), dropped_count)
}

// ============================================================================
// 5. Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_email_detection_and_redaction() {
        let text = "For inquiries, contact Dr. Alice at alice.smith@university.edu or support@openfoundation.org regarding the paper.";
        let emails = extract_email_addresses(text);
        assert_eq!(emails.len(), 2);
        assert_eq!(emails[0], "alice.smith@university.edu");
        assert_eq!(emails[1], "support@openfoundation.org");

        let redacted = redact_emails(text);
        assert!(!redacted.contains("alice.smith@university.edu"));
        assert!(!redacted.contains("support@openfoundation.org"));
        assert!(redacted.contains("[EMAIL_REDACTED]"));
    }

    #[test]
    fn test_email_harvest_dump_rejection() {
        let mut engine = FilterEngine::new();
        let dump = "Lead directory: user1@example.com, user2@test.org, user3@corp.com, user4@mail.net, user5@domain.io marketing list entries records.";
        let decision = engine.evaluate_text(dump, "contacts");
        assert!(decision.is_rejected());
        if let FilterDecision::Reject(reason) = decision {
            assert!(reason.contains("PII / contact list dump rejected"));
        }
    }

    #[test]
    fn test_malicious_code_reverse_shell_rejection() {
        let mut engine = FilterEngine::new();
        let payload = "Setup instructions for terminal: run /bin/bash -i connecting to management console server.";
        let decision = engine.evaluate_text(payload, "guide");
        assert!(decision.is_rejected());
        if let FilterDecision::Reject(reason) = decision {
            assert!(reason.contains("Malicious code / exploit pattern rejected"));
        }
    }

    #[test]
    fn test_malicious_code_webshell_and_powershell_rejection() {
        let mut engine = FilterEngine::new();
        let webshell = "Security test payload: call eval(with base64_decode($data)) in script.";
        let decision = engine.evaluate_text(webshell, "script");
        assert!(decision.is_rejected());

        let ps_cradle = "setup line: invoke-expression with downloadstring and parameters";
        let decision2 = engine.evaluate_text(ps_cradle, "installer");
        assert!(decision2.is_rejected());
    }

    #[test]
    fn test_malicious_code_sqli_and_ransomware_rejection() {
        let mut engine = FilterEngine::new();
        let sqli = "SELECT username FROM users WHERE id = 1 UNION SELECT NULL, username, password FROM admin_table";
        let decision = engine.evaluate_text(sqli, "database_example");
        assert!(decision.is_rejected());

        let miner = "Worker start stratum+tcp://pool.example.com:3333 with xmrig miner client.";
        let decision2 = engine.evaluate_text(miner, "mining");
        assert!(decision2.is_rejected());
    }

    #[test]
    fn test_toxic_and_scam_words_rejection() {
        let mut engine = FilterEngine::new();
        let casino = "Play today at our online casino bonus with free spins no deposit required right now!";
        let decision = engine.evaluate_text(casino, "promo");
        assert!(decision.is_rejected());

        let crypto_scam = "Special event: free crypto giveaway! Send BTC get double back directly to your wallet!";
        let decision2 = engine.evaluate_text(crypto_scam, "crypto");
        assert!(decision2.is_rejected());
    }

    #[test]
    fn test_repetitive_line_noise_rejection() {
        let mut engine = FilterEngine::new();
        let repeat_words = "important update: buy buy buy buy buy buy cheap stocks today on the market!";
        let decision = engine.evaluate_text(repeat_words, "finance");
        assert!(decision.is_rejected());

        let char_noise = "Section header: ==================================================== end.";
        let decision2 = engine.evaluate_text(char_noise, "divider");
        assert!(decision2.is_rejected());
    }

    #[test]
    fn test_line_sanitizer_removes_cookie_banners() {
        let text = "Introduction to Calculus\nWe use cookies to enhance your experience on this website.\nCalculus is the mathematical study of continuous change.\nSubscribe to our newsletter for more lessons.\nContact author at author@math.edu for questions.";
        let (sanitized, dropped) = sanitize_text_lines(text);
        assert_eq!(dropped, 2);
        assert!(!sanitized.contains("cookies"));
        assert!(!sanitized.contains("newsletter"));
        assert!(sanitized.contains("Introduction to Calculus"));
        assert!(sanitized.contains("[EMAIL_REDACTED]"));
    }

    // ========================================================================
    // Quality != License Mandatory Tests
    // 1. CC-BY + excellent quality     -> ACCEPT
    // 2. CC-BY + garbage OCR           -> REJECT
    // 3. Public Domain + spam          -> REJECT
    // 4. MIT code + corrupted file     -> REJECT
    // 5. No license + perfect quality  -> REJECT
    // ========================================================================

    #[test]
    fn test_scenario_1_cc_by_plus_excellent_quality_accepts() {
        let mut engine = FilterEngine::new();
        let high_quality_math = "In differential calculus, the derivative of a function measures the sensitivity to change of the function value with respect to a change in its argument. Fundamental theorem connects differentiation with integration seamlessly.";
        let report = engine.evaluate_pipeline(
            high_quality_math,
            "Calculus and Analysis",
            true, // rights_ok (CC-BY verified)
            true, // scope_ok
            true, // provenance_ok
        );

        assert!(report.is_accepted(), "CC-BY with excellent technical quality must be ACCEPTED");
        assert!(report.rights_ok);
        assert!(report.scope_ok);
        assert!(report.provenance_ok);
        assert!(report.quality_ok);
        assert!(report.duplication_ok);
        assert!(report.structure_ok);
        assert!(report.domain_ok);
        assert_eq!(report.failure_code, None);
    }

    #[test]
    fn test_scenario_2_cc_by_plus_garbage_ocr_rejects() {
        let mut engine = FilterEngine::new();
        // High OCR noise: noisy symbols injected inside words
        let ocr_garbage = "In th|s docum~nt the equat|on f~rmula cont~ins seve\\re O|C|R sc~nning err~rs and corrup\\ted symb^ls thr~ughout the p~ge.";
        let report = engine.evaluate_pipeline(
            ocr_garbage,
            "Mathematics",
            true, // rights_ok (License is valid CC-BY)
            true, // scope_ok
            true, // provenance_ok
        );

        assert!(report.is_rejected(), "CC-BY with OCR garbage must be REJECTED (Quality != License)");
        assert!(report.rights_ok, "Rights must be acknowledged as ok");
        assert!(!report.quality_ok, "Quality Gate must fail for OCR noise");
        assert_eq!(report.failure_code, Some(RejectionReasonCode::RejectNoisy));
    }

    #[test]
    fn test_scenario_3_public_domain_plus_spam_seo_rejects() {
        let mut engine = FilterEngine::new();
        // Low vocabulary diversity / keyword stuffing spam
        let spam_seo = "buy cheap online buy cheap online buy cheap online buy cheap online buy cheap online buy cheap online buy cheap online buy cheap online buy cheap online buy cheap online.";
        let report = engine.evaluate_pipeline(
            spam_seo,
            "General Literature",
            true, // rights_ok (Public Domain)
            true, // scope_ok
            true, // provenance_ok
        );

        assert!(report.is_rejected(), "Public Domain with spam/SEO stuffing must be REJECTED");
        assert!(report.rights_ok, "Rights must be acknowledged as ok");
        assert!(!report.quality_ok, "Quality Gate must fail for low vocabulary diversity / repetitive noise");
        assert_eq!(report.failure_code, Some(RejectionReasonCode::RejectNoisy));
    }

    #[test]
    fn test_scenario_4_mit_plus_corrupted_structure_rejects() {
        let mut engine = FilterEngine::new();
        // Syntax corruption: severely unbalanced brackets
        let corrupted_code = "pub fn compute_vector_matrix() -> Vec<f64> { let mut res = Vec::new(); if true { res.push(42.0); // missing closing brackets [ ( {";
        let report = engine.evaluate_pipeline(
            corrupted_code,
            "Computer Science Algorithm",
            true, // rights_ok (MIT)
            true, // scope_ok
            true, // provenance_ok
        );

        assert!(report.is_rejected(), "MIT code with corrupted syntax / unbalanced delimiters must be REJECTED");
        assert!(report.rights_ok, "Rights must be acknowledged as ok");
        assert!(!report.structure_ok, "Structure Gate must fail for syntax error");
        assert_eq!(report.failure_code, Some(RejectionReasonCode::RejectMalformed));
    }

    #[test]
    fn test_scenario_5_no_license_plus_perfect_quality_rejects() {
        let mut engine = FilterEngine::new();
        let pristine_science_text = "The speed of light in vacuum, commonly denoted c, is a universal physical constant important in many areas of physics. Its exact value is defined as 299792458 metres per second.";
        let report = engine.evaluate_pipeline(
            pristine_science_text,
            "Physics and Relativity",
            false, // rights_ok = FALSE (No license / unknown license)
            true,  // scope_ok
            true,  // provenance_ok
        );

        assert!(report.is_rejected(), "No license with perfect quality must be REJECTED at Rights Gate");
        assert!(!report.rights_ok, "Rights Gate must fail");
        assert_eq!(report.failure_code, Some(RejectionReasonCode::RejectRightsUnknown));
    }

    #[test]
    fn test_7_gate_acceptance_formula_completeness() {
        let mut engine = FilterEngine::new();
        let valid_text = "A compiler is a computer program that translates computer code written in one programming language into another language. Compilers are largely used for programs that translate source code from a high-level programming language to a lower-level language.";
        let report = engine.evaluate_pipeline(
            valid_text,
            "Computer Science Compiler",
            true,
            true,
            true,
        );

        assert!(report.is_accepted());
        assert_eq!(
            report.is_accepted(),
            report.rights_ok
                && report.scope_ok
                && report.provenance_ok
                && report.quality_ok
                && report.duplication_ok
                && report.structure_ok
                && report.domain_ok
        );
    }
}

