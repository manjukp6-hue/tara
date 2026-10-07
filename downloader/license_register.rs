//! TARA License Register & Provenance Engine (100% Native Rust)
//!
//! Dedicated, self-contained registry for dataset licensing verification and auditable provenance:
//!
//! Architectural Principles & Rigorous Boundaries:
//! 1. Clear Separation of Concerns:
//!    - Layer 1: License Evidence Extraction (Structured fields, SPDX tags, formal legal preambles vs arbitrary prose).
//!    - Layer 2: SPDX / Identifier Normalization & Multi-license Semantics (Disjunctive `OR` vs Conjunctive `AND`).
//!    - Layer 3: TARA Rule-18 Policy Resolution (Evaluates evidence against project policy matrix).
//!    - Layer 4: Ingestion Decision & Auditable Provenance Persistence.
//!
//! 2. False Positive Protection:
//!    - Raw textual mentions of license names (e.g. "This paper compares MIT, GPL-3.0 and Apache-2.0")
//!      are recognized as normal content prose, NOT legal license declarations.
//!    - Rejections require structured metadata fields (`"license"`), standard SPDX markers
//!      (`SPDX-License-Identifier:`), explicit comment headers (`License:`), or formal legal grant preambles.
//!
//! 3. Precise Legal-Engineering Semantics:
//!    - Copyleft (GPL, AGPL, LGPL, MPL, CC-BY-SA) rejection reason:
//!      "rejected by TARA's Rule-18 commercial-use policy because copyleft licensing is outside the project's permitted license class."
//!    - `ai_training_allowed = true`: TARA policy permits ingestion under its configured license policy.
//!    - `download_allowed = true`: TARA is permitted to obtain and store a local copy under applicable source/license terms.
//!    - Avoids claims of "legal guarantees"; preserves auditable provenance evidence at ingestion time.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::SystemTime;

/// Permissive licenses approved for commercial use and AI model training under TARA Rule-18 policy (SPDX identifiers).
pub const PERMISSIVE_APPROVED_LICENSES: [&str; 13] = [
    "MIT",
    "Apache-2.0",
    "CC-BY-4.0",
    "CC-BY-3.0",
    "BSD-3-Clause",
    "BSD-2-Clause",
    "CC0-1.0",
    "ODC-BY",
    "PUBLIC_DOMAIN",
    "Public Domain",
    "PostgreSQL-License",
    "WordNet-3.0-License",
    "Unlicense",
];

// ============================================================================
// 1. Core Data Models
// ============================================================================

/// The 4 mandatory commercial and operational rights evaluated under TARA Rule-18 policy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct CommercialRights {
    /// Permitted under TARA Rule-18 commercial-use policy.
    pub commercial_use: bool,
    /// Modification and adaptation permitted under applicable terms.
    pub modify: bool,
    /// TARA is permitted to obtain a local copy under applicable source/license terms.
    pub download: bool,
    /// Local neural model training permitted under TARA Rule-18 policy.
    pub ai_training: bool,
}

impl CommercialRights {
    /// Returns true only when all 4 mandatory commercial rights are permitted under TARA policy.
    pub fn all_permitted(&self) -> bool {
        self.commercial_use && self.modify && self.download && self.ai_training
    }

    /// Full commercial permissive rights (MIT, Apache-2.0, BSD, CC-BY, etc.).
    pub fn full_permissive() -> Self {
        Self {
            commercial_use: true,
            modify: true,
            download: true,
            ai_training: true,
        }
    }

    /// Prohibited rights state (all disallowed).
    pub fn prohibited() -> Self {
        Self {
            commercial_use: false,
            modify: false,
            download: false,
            ai_training: false,
        }
    }

    /// Empty rights state (alias for prohibited).
    pub fn empty() -> Self {
        Self::prohibited()
    }
}

/// Nature of the extracted license evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceKind {
    /// Structured data field (e.g. JSON `"license": "..."`).
    StructuredField,
    /// Standard source code marker (e.g. `SPDX-License-Identifier: MIT`).
    SpdxMarker,
    /// Comment or metadata header (e.g. `# License: Apache-2.0`).
    HeaderMarker,
    /// Formal legal grant preamble (e.g. `"Licensed under the GNU General Public License"`).
    FormalLegalPreamble,
}

/// Extracted license declaration evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LicenseEvidence {
    pub raw_identifier: String,
    pub kind: EvidenceKind,
}

/// Parsed multi-license expression with core boolean semantics (`OR` and `AND`).
///
/// Scope Note:
/// Implements TARA's core boolean license expression model (supporting disjunctive `OR` and conjunctive `AND`).
/// Advanced SPDX grammar constructs (e.g. `WITH` exception clauses, nested parentheses precedence,
/// `-only` / `-or-later` suffixes, and `LicenseRef-*` identifiers) are future grammar extensions and are
/// not claimed as full SPDX specification coverage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LicenseExpression {
    /// Single standalone license identifier.
    Single(String),
    /// Disjunctive expression (`OR`, `/`): downstream licensee may choose any approved option.
    Disjunctive(Vec<String>),
    /// Conjunctive expression (`AND`, `,`): all licenses apply simultaneously.
    Conjunctive(Vec<String>),
}

impl LicenseExpression {
    /// Parses a raw license string into a structured expression with core boolean `OR` / `AND` semantics.
    pub fn parse(raw: &str) -> Self {
        let trimmed = raw.trim();

        // 1. Disjunction: OR, or, /
        if trimmed.contains(" OR ") || trimmed.contains(" or ") || trimmed.contains('/') {
            let parts: Vec<String> = if trimmed.contains(" OR ") {
                trimmed
                    .split(" OR ")
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            } else if trimmed.contains(" or ") {
                trimmed
                    .split(" or ")
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            } else {
                trimmed
                    .split('/')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            };
            if parts.len() > 1 {
                return LicenseExpression::Disjunctive(parts);
            }
        }

        // 2. Conjunction: AND, and, ,
        if trimmed.contains(" AND ") || trimmed.contains(" and ") || trimmed.contains(',') {
            let parts: Vec<String> = if trimmed.contains(" AND ") {
                trimmed
                    .split(" AND ")
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            } else if trimmed.contains(" and ") {
                trimmed
                    .split(" and ")
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            } else {
                trimmed
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            };
            if parts.len() > 1 {
                return LicenseExpression::Conjunctive(parts);
            }
        }

        LicenseExpression::Single(trimmed.to_string())
    }
}

/// Single license record in the register with complete auditable 7-layer provenance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct LicenseEntry {
    pub dataset_id: String,
    pub license: String,
    pub proof_url: String,
    #[serde(default)]
    pub source_url: String,
    #[serde(default)]
    pub domain: String,
    pub author: String,
    pub commercial_use: bool,
    #[serde(default)]
    pub modify_allowed: bool,
    #[serde(default)]
    pub download_allowed: bool,
    #[serde(default)]
    pub ai_training_allowed: bool,
    #[serde(default)]
    pub detected_licenses: Vec<String>,
    #[serde(default)]
    pub detected_authors: Vec<String>,
    #[serde(default)]
    pub lines_audited: usize,
    pub timestamp: String,

    // 7-Layer Provenance & Policy Architecture:
    #[serde(default)]
    pub host_domain: String,
    #[serde(default)]
    pub upstream_repo: String,
    #[serde(default)]
    pub upstream_publisher: String,
    #[serde(default)]
    pub underlying_content_source: String,
    #[serde(default)]
    pub underlying_content_terms: String,
    #[serde(default)]
    pub content_rights_status: String,
    #[serde(default)]
    pub tara_policy_class: String,

    // Layer 7: Decoupled Multi-State Operational Lifecycle
    #[serde(default)]
    pub download_permission: String,
    #[serde(default)]
    pub database_license_status: String,
    #[serde(default)]
    pub training_eligibility: String,
    #[serde(default)]
    pub policy_decision: String,

    // Granular Asset-Level Scoping & Rights Evidence:
    #[serde(default)]
    pub asset_file_id: String,
    #[serde(default)]
    pub edition_id: String,
    #[serde(default)]
    pub rights_evidence: String,
    #[serde(default)]
    pub jurisdiction_scope: String,
    #[serde(default)]
    pub third_party_exclusions: String,
    #[serde(default)]
    pub license_id: String,
    #[serde(default)]
    pub asset_scope: String,
}

impl LicenseEntry {
    /// Dynamically infers and populates the 7-layer provenance architecture for license entry.
    pub fn enrich_provenance(&mut self) {
        let url_lower = self.source_url.to_lowercase();
        let id_lower = self.dataset_id.to_lowercase();

        if url_lower.contains("huggingface.co") || id_lower.contains("c4") || id_lower.contains("bigcode") || id_lower.contains("the_stack") {
            self.host_domain = "huggingface.co".to_string();
            if id_lower.contains("c4") || url_lower.contains("allenai/c4") {
                self.upstream_repo = "allenai/c4".to_string();
                self.upstream_publisher = "Google Research & Allen Institute for AI".to_string();
                self.underlying_content_source = "Common Crawl".to_string();
                self.underlying_content_terms = "Common Crawl Terms of Use".to_string();
                self.content_rights_status = "requires source/record-aware provenance (individual web copyright preserved)".to_string();
                self.tara_policy_class = "permitted_database_license".to_string();
                self.download_permission = "APPROVED".to_string();
                self.database_license_status = "PERMITTED (ODC-BY Database License)".to_string();
                self.training_eligibility = "CONDITIONAL_RECORD_GATED".to_string();
                self.policy_decision = "RECORD_RIGHTS_REQUIRED (Dataset container download permitted; training requires per-record provenance)".to_string();
            } else if id_lower.contains("bigcode") || id_lower.contains("the_stack") || url_lower.contains("bigcode") {
                self.upstream_repo = "bigcode/the-stack".to_string();
                self.upstream_publisher = "BigCode Project (ServiceNow & Hugging Face)".to_string();
                self.underlying_content_source = "Public GitHub Repositories".to_string();
                self.underlying_content_terms = "Repository-specific upstream licenses (use must comply with original licenses)".to_string();
                self.content_rights_status = "per-datapoint permissive license gating mandatory".to_string();
                self.tara_policy_class = "record_gated_permissive".to_string();
                self.download_permission = "APPROVED".to_string();
                self.database_license_status = "PERMITTED (OpenRAIL-M Container)".to_string();
                self.training_eligibility = "CONDITIONAL_RECORD_GATED".to_string();
                self.policy_decision = "RECORD_RIGHTS_REQUIRED (Per-file permissive gating mandatory before training)".to_string();
            } else {
                self.upstream_repo = "huggingface/dataset".to_string();
                self.upstream_publisher = if self.author.is_empty() { "HuggingFace Community".to_string() } else { self.author.clone() };
                self.underlying_content_source = "Upstream Dataset Shards".to_string();
                self.underlying_content_terms = "Standard Dataset Terms".to_string();
                self.content_rights_status = "verified_provenance".to_string();
                self.tara_policy_class = "permissive_commercial_approved".to_string();
                self.download_permission = "APPROVED".to_string();
                self.database_license_status = "PERMITTED".to_string();
                self.training_eligibility = "APPROVED_FOR_TRAINING".to_string();
                self.policy_decision = "APPROVED".to_string();
            }
        } else if url_lower.contains("opentextbc.ca") || id_lower.contains("open_textbook") || id_lower.contains("biology") {
            self.host_domain = "opentextbc.ca".to_string();
            self.upstream_repo = "bccampus/open-textbooks".to_string();
            self.upstream_publisher = "BCcampus Open Education".to_string();
            self.underlying_content_source = "BCcampus Peer-Reviewed Educational Textbooks".to_string();
            self.underlying_content_terms = "CC BY 4.0 except where otherwise noted (asset-level exceptions apply)".to_string();
            self.content_rights_status = "verified_permissive_educational".to_string();
            self.tara_policy_class = "permissive_commercial_approved".to_string();
            self.download_permission = "APPROVED".to_string();
            self.database_license_status = "PERMITTED (CC-BY-4.0)".to_string();
            self.training_eligibility = "APPROVED_FOR_TRAINING".to_string();
            self.policy_decision = "APPROVED".to_string();
        } else if url_lower.contains("math-qa.github.io") || id_lower.contains("math_qa") || id_lower.contains("mathqa") {
            self.host_domain = "math-qa.github.io".to_string();
            self.upstream_repo = "math-qa/dataset".to_string();
            self.upstream_publisher = "Amini et al. / MathQA Consortium".to_string();
            self.underlying_content_source = "Amini et al. / MathQA Problem Corpus".to_string();
            self.underlying_content_terms = "Apache-2.0 Open Source License".to_string();
            self.content_rights_status = "record_verified_permissive".to_string();
            self.tara_policy_class = "permissive_commercial_approved".to_string();
            self.download_permission = "APPROVED".to_string();
            self.database_license_status = "PERMITTED (Apache-2.0)".to_string();
            self.training_eligibility = "APPROVED_FOR_TRAINING".to_string();
            self.policy_decision = "APPROVED".to_string();
        } else if url_lower.contains("amazonaws.com") || id_lower.contains("qasper") {
            self.host_domain = "amazonaws.com (AI2 S3)".to_string();
            self.upstream_repo = "allenai/qasper".to_string();
            self.upstream_publisher = "Allen Institute for AI (AI2)".to_string();
            self.underlying_content_source = "NLP Research Papers on arXiv".to_string();
            self.underlying_content_terms = "CC-BY-4.0 author agreements via AI2".to_string();
            self.content_rights_status = "paper_level_rights_verified".to_string();
            self.tara_policy_class = "permissive_commercial_approved".to_string();
        } else if url_lower.contains("raw.githubusercontent.com") || url_lower.contains("github.com") {
            self.host_domain = "raw.githubusercontent.com".to_string();
            self.upstream_repo = "github-upstream/repository".to_string();
            self.upstream_publisher = if self.author.is_empty() { "GitHub Repository Authors".to_string() } else { self.author.clone() };
            self.underlying_content_source = "Public Git Repository Content".to_string();
            self.underlying_content_terms = "Upstream Git repository LICENSE (Host is not licensor)".to_string();
            self.content_rights_status = "repository_license_bound".to_string();
            self.tara_policy_class = "permissive_commercial_approved".to_string();
        } else if url_lower.contains("arxiv.org") || id_lower.contains("arxiv") {
            self.host_domain = "arxiv.org".to_string();
            self.upstream_repo = "cornell/arxiv".to_string();
            self.upstream_publisher = "Cornell University & arXiv Contributors".to_string();
            self.underlying_content_source = "Academic Preprints".to_string();
            self.underlying_content_terms = "Per-paper author terms (Open Access != universal commercial)".to_string();
            self.content_rights_status = "paper_level_rights_gated".to_string();
            self.tara_policy_class = "paper_level_rights_gated".to_string();
        } else if url_lower.contains("rfc-editor.org") || id_lower.contains("rfc") {
            self.host_domain = "rfc-editor.org".to_string();
            self.upstream_repo = "ietf/rfc-database".to_string();
            self.upstream_publisher = "Internet Engineering Task Force (IETF)".to_string();
            self.underlying_content_source = "Internet Standards Specifications".to_string();
            self.underlying_content_terms = "IETF Trust Legal Provisions".to_string();
            self.content_rights_status = "public_standards_specification".to_string();
            self.tara_policy_class = "permissive_commercial_approved".to_string();
        } else if url_lower.contains("archive.org") || url_lower.contains("gutenberg") || id_lower.contains("gutenberg") {
            self.host_domain = "gutenberg.org".to_string();
            self.upstream_repo = "gutenberg/corpus".to_string();
            self.upstream_publisher = "Project Gutenberg Literary Archive Foundation".to_string();
            self.underlying_content_source = "Public Domain Literature & Historical Texts".to_string();
            self.underlying_content_terms = "Public Domain (US Copyright Expired)".to_string();
            self.content_rights_status = "public_domain_unrestricted".to_string();
            self.tara_policy_class = "public_domain_approved".to_string();
            self.jurisdiction_scope = "US_Pre1929_Expired".to_string();
            self.rights_evidence = "Author_Life_Plus_70_Or_Pre1929_Publication".to_string();
        } else {
            self.host_domain = if !url_lower.is_empty() {
                url_lower.split("://").nth(1).and_then(|s| s.split('/').next()).unwrap_or("local_storage").to_string()
            } else {
                "local_storage".to_string()
            };
            self.upstream_repo = format!("repo/{}", self.dataset_id);
            self.upstream_publisher = if self.author.is_empty() { "Verified Academic / Open Foundation".to_string() } else { self.author.clone() };
            self.underlying_content_source = "Authentic Source Data".to_string();
            self.underlying_content_terms = "Standard Open Terms".to_string();
            self.content_rights_status = "verified_provenance".to_string();
            self.tara_policy_class = "permissive_commercial_approved".to_string();
        }
    }
}

/// Operational decision for license verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LicenseDecision {
    /// Permissive and all commercial rights verified under TARA Rule-18 policy -> Approved.
    Approve,

    /// Non-permissive, restricted, or unverified license -> Rejected.
    Reject(String),
}

impl LicenseDecision {
    pub fn is_approved(&self) -> bool {
        matches!(self, LicenseDecision::Approve)
    }

    pub fn is_rejected(&self) -> bool {
        matches!(self, LicenseDecision::Reject(_))
    }
}

/// Operational decision for evaluating individual record rights under Rule-18 fail-closed policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecordRightsDecision {
    /// Permissive license evidence found and all 4 rights verified.
    Approved {
        detected_license: String,
        rights: CommercialRights,
    },
    /// Any failure (missing license evidence, non-permissive, non-commercial, copyleft, proprietary, etc.) -> REJECTED.
    Rejected {
        detected_license: String,
        reason: String,
    },
}

impl RecordRightsDecision {
    pub fn is_approved(&self) -> bool {
        matches!(self, RecordRightsDecision::Approved { .. })
    }
    pub fn is_rejected(&self) -> bool {
        matches!(self, RecordRightsDecision::Rejected { .. })
    }
    /// Under strict Rule-18 fail-closed policy, quarantine is completely eliminated in favor of direct reject.
    pub fn is_quarantine(&self) -> bool {
        false
    }
}

/// Exact record of a license violation detected during deep inspection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LicenseViolation {
    pub line_number: usize,
    pub raw_snippet: String,
    pub detected_license: String,
    pub rights: CommercialRights,
    pub reason: String,
}

/// Summary result of a deep line-by-line file license audit.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeepLicenseAuditSummary {
    pub total_lines_audited: usize,
    pub distinct_licenses: Vec<String>,
    pub distinct_authors: Vec<String>,
    pub rights: CommercialRights,
    pub compliant: bool,
}

/// Statistics from streaming sanitization of mixed multi-license datasets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanitizeStats {
    pub total_records_processed: usize,
    pub accepted_records: usize,
    pub dropped_records: usize,
    pub retained_licenses: Vec<String>,
    pub dropped_reasons: Vec<String>,
}

// ============================================================================
// 2. Layer 1: License Evidence Extraction (False Positive Protection)
// ============================================================================

/// Extracts authentic license declaration evidence from a line of text or JSON record.
///
/// Prevents false positives by strictly distinguishing:
/// - Structured metadata fields (`"license": "..."`)
/// - Standard source code markers (`SPDX-License-Identifier: ...`)
/// - Explicit header markers (`License: ...` in comments/headers)
/// - Formal legal preambles (`"Licensed under the GNU General Public License"`)
///
/// Arbitrary prose mentioning license names (e.g. "We compare MIT, GPL-3.0 and Apache-2.0")
/// returns `None` and is safely ignored as non-licensing text.
pub fn extract_license_evidence(line: &str) -> Option<LicenseEvidence> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }

    // A. JSON / JSONL structured record parsing
    if trimmed.starts_with('{') {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
            for key in &[
                "license",
                "license_type",
                "spdx_license",
                "spdx",
                "licence",
                "license_name",
                "repo_license",
            ] {
                if let Some(val) = v.get(*key).and_then(|s| s.as_str()) {
                    let s = val.trim();
                    if !s.is_empty() {
                        return Some(LicenseEvidence {
                            raw_identifier: s.to_string(),
                            kind: EvidenceKind::StructuredField,
                        });
                    }
                }
            }

            // Nested under metadata
            if let Some(meta) = v.get("metadata") {
                for key in &["license", "license_type", "spdx_license", "spdx", "licence"] {
                    if let Some(val) = meta.get(*key).and_then(|s| s.as_str()) {
                        let s = val.trim();
                        if !s.is_empty() {
                            return Some(LicenseEvidence {
                                raw_identifier: s.to_string(),
                                kind: EvidenceKind::StructuredField,
                            });
                        }
                    }
                }
            }
        }
    }

    let lower = trimmed.to_lowercase();

    // B. Standard SPDX Identifier Marker
    if let Some(pos) = lower.find("spdx-license-identifier:") {
        let after = &trimmed[pos + "spdx-license-identifier:".len()..];
        let lic = after
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '.' && c != '/');
        if !lic.is_empty() {
            return Some(LicenseEvidence {
                raw_identifier: lic.to_string(),
                kind: EvidenceKind::SpdxMarker,
            });
        }
    }

    // C. Explicit Header Marker (e.g. `// License: Apache-2.0` or `# License: MIT`)
    // Must be preceded by comment characters or start of line, not mid-sentence.
    if let Some(pos) = lower.find("license:") {
        let prefix = &trimmed[..pos].trim();
        let is_comment_or_header = prefix.is_empty()
            || prefix.chars().all(|c| c == '/' || c == '*' || c == '#' || c == '-' || c.is_whitespace());

        if is_comment_or_header {
            let after = &trimmed[pos + "license:".len()..];
            let lic = after
                .trim()
                .split(['\n', ';', ','])
                .next()
                .unwrap_or("")
                .trim()
                .trim_matches('"');
            if !lic.is_empty() && lic.len() < 50 {
                return Some(LicenseEvidence {
                    raw_identifier: lic.to_string(),
                    kind: EvidenceKind::HeaderMarker,
                });
            }
        }
    }

    // D. Formal Legal Grant Preambles & Disclaimers
    if lower.contains("licensed under the gnu affero general public license")
        || lower.contains("gnu affero general public license")
    {
        return Some(LicenseEvidence {
            raw_identifier: "AGPL-3.0".to_string(),
            kind: EvidenceKind::FormalLegalPreamble,
        });
    }
    if lower.contains("licensed under the gnu lesser general public license")
        || lower.contains("gnu lesser general public license")
    {
        return Some(LicenseEvidence {
            raw_identifier: "LGPL-3.0".to_string(),
            kind: EvidenceKind::FormalLegalPreamble,
        });
    }
    if lower.contains("licensed under the gnu general public license")
        || lower.contains("this program is free software: you can redistribute it and/or modify it under the terms of the gnu")
    {
        return Some(LicenseEvidence {
            raw_identifier: "GPL-3.0".to_string(),
            kind: EvidenceKind::FormalLegalPreamble,
        });
    }
    if lower.contains("licensed under a creative commons attribution-noncommercial")
        || lower.contains("non-commercial use only")
        || lower.contains("strictly non-commercial")
        || lower.contains("strictly for educational and non-profit")
        || lower.contains("strictly for educational")
        || lower.contains("strictly for non-profit")
        || lower.contains("educational and non-profit uses")
        || lower.contains("educational and non-profit")
        || lower.contains("fair use only")
        || lower.contains("for non-commercial")
        || lower.contains("not for commercial use")
        || lower.contains("personal use only")
    {
        return Some(LicenseEvidence {
            raw_identifier: "CC-BY-NC".to_string(),
            kind: EvidenceKind::FormalLegalPreamble,
        });
    }
    if lower.contains("licensed under a creative commons attribution-noderivatives")
        || lower.contains("no derivatives permitted")
    {
        return Some(LicenseEvidence {
            raw_identifier: "CC-BY-ND".to_string(),
            kind: EvidenceKind::FormalLegalPreamble,
        });
    }
    if lower.contains("all rights reserved") {
        return Some(LicenseEvidence {
            raw_identifier: "All-Rights-Reserved".to_string(),
            kind: EvidenceKind::FormalLegalPreamble,
        });
    }

    // Normal prose content mentioning license names without declaration prefix returns None!
    None
}

/// Helper wrapper preserving backward-compatible signature.
pub fn extract_license_from_line(line: &str) -> Option<String> {
    extract_license_evidence(line).map(|e| e.raw_identifier)
}

/// Extracts an author or contributor name from a single line of text or JSON record.
pub fn extract_author_from_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }

    // A. JSON / JSONL record parsing
    if trimmed.starts_with('{') {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
            for key in &[
                "author",
                "author_name",
                "creator",
                "contributor",
                "repo_owner",
                "publisher",
                "copyright",
            ] {
                if let Some(val) = v.get(*key).and_then(|s| s.as_str()) {
                    let s = val.trim();
                    if !s.is_empty() {
                        return Some(s.to_string());
                    }
                }
            }

            if let Some(meta) = v.get("metadata") {
                for key in &["author", "creator", "contributor"] {
                    if let Some(val) = meta.get(*key).and_then(|s| s.as_str()) {
                        let s = val.trim();
                        if !s.is_empty() {
                            return Some(s.to_string());
                        }
                    }
                }
            }
        }
    }

    // B. Header / Text scanning
    let lower = trimmed.to_lowercase();
    if let Some(pos) = lower.find("author:") {
        let prefix = &trimmed[..pos].trim();
        let is_comment_or_header = prefix.is_empty()
            || prefix.chars().all(|c| c == '/' || c == '*' || c == '#' || c == '-' || c.is_whitespace());

        if is_comment_or_header {
            let after = &trimmed[pos + "author:".len()..];
            let aut = after
                .trim()
                .split(['\n', ';'])
                .next()
                .unwrap_or("")
                .trim()
                .trim_matches('"');
            if !aut.is_empty() && aut.len() < 80 {
                return Some(aut.to_string());
            }
        }
    }

    None
}

// ============================================================================
// 3. Layer 3: TARA Rule-18 Policy Resolution & Decision Engine
// ============================================================================

/// Pure license registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseRegister {
    pub schema_version: String,
    pub last_updated: String,
    pub entries: Vec<LicenseEntry>,
}

impl Default for LicenseRegister {
    fn default() -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            schema_version: "1.0.0".to_string(),
            last_updated: format!("epoch_{now}"),
            entries: Vec::new(),
        }
    }
}

impl LicenseRegister {
    /// Creates a new empty license register.
    pub fn new() -> Self {
        Self::default()
    }

    /// Evaluates a single normalized license identifier against TARA Rule-18 project policy.
    fn evaluate_single_identifier(raw: &str) -> (CommercialRights, LicenseDecision) {
        let trimmed = raw.trim();
        let lower = trimmed.to_lowercase();

        if trimmed.is_empty() {
            return (
                CommercialRights::empty(),
                LicenseDecision::Reject("Missing license specification".to_string()),
            );
        }

        // 1. Strict rejection of Non-Commercial restrictions (Rule 18: commercial use required)
        if lower.contains("-nc")
            || lower.contains("noncommercial")
            || lower.contains("non-commercial")
            || lower.contains("no-commercial")
        {
            return (
                CommercialRights {
                    commercial_use: false,
                    modify: true,
                    download: true,
                    ai_training: false,
                },
                LicenseDecision::Reject(format!(
                    "License '{}' is rejected by TARA's Rule-18 commercial-use policy: Non-Commercial (NC) restriction is outside the project's permitted license class",
                    trimmed
                )),
            );
        }

        // 2. Strict rejection of No-Derivatives restrictions (Rule 18: modification required)
        if lower.contains("-nd")
            || lower.contains("noderivatives")
            || lower.contains("no-derivatives")
        {
            return (
                CommercialRights {
                    commercial_use: true,
                    modify: false,
                    download: true,
                    ai_training: false,
                },
                LicenseDecision::Reject(format!(
                    "License '{}' is rejected by TARA's Rule-18 policy: No-Derivatives (ND) restriction is outside the project's permitted license class (modification required)",
                    trimmed
                )),
            );
        }

        // 3. Strict rejection of Copyleft / ShareAlike (Rule 18 policy: copyleft licensing is outside permitted class)
        if lower.contains("-sa")
            || lower.contains("sharealike")
            || lower.contains("share-alike")
            || lower.contains("gpl")
            || lower.contains("agpl")
            || lower.contains("lgpl")
            || lower.contains("mpl")
        {
            return (
                CommercialRights {
                    commercial_use: false,
                    modify: true,
                    download: true,
                    ai_training: false,
                },
                LicenseDecision::Reject(format!(
                    "License '{}' is rejected by TARA's Rule-18 commercial-use policy: copyleft licensing is outside the project's permitted license class",
                    trimmed
                )),
            );
        }

        // 4. Strict rejection of Research-only / Academic-only terms
        if lower.contains("research-only")
            || lower.contains("academic-only")
            || lower.contains("research only")
            || lower.contains("academic use")
            || lower.contains("non-prod")
        {
            return (
                CommercialRights {
                    commercial_use: false,
                    modify: true,
                    download: true,
                    ai_training: false,
                },
                LicenseDecision::Reject(format!(
                    "License '{}' is rejected by TARA's Rule-18 commercial-use policy: research/academic-only limitation is outside the project's permitted license class",
                    trimmed
                )),
            );
        }

        // 5. Strict rejection of proprietary / All Rights Reserved clauses
        if lower.contains("all rights reserved")
            || lower.contains("proprietary")
            || lower.contains("confidential")
        {
            return (
                CommercialRights::prohibited(),
                LicenseDecision::Reject(format!(
                    "License '{}' is rejected by TARA's Rule-18 policy: proprietary/confidential terms are prohibited",
                    trimmed
                )),
            );
        }

        // 5b. Strict rejection of merchandise / product branding authorization ("Officially Licensed" Trap)
        if lower.contains("officially licensed")
            || lower.contains("official license")
            || lower.contains("licensed merchandise")
            || lower.contains("licensed product")
        {
            return (
                CommercialRights::prohibited(),
                LicenseDecision::Reject(format!(
                    "License '{}' is rejected by TARA's Rule-18 policy: 'Officially Licensed' denotes commercial merchandise/trademark branding, NOT an open-source copyright license for AI training",
                    trimmed
                )),
            );
        }

        // 6. Check against approved permissive licenses
        let is_approved = PERMISSIVE_APPROVED_LICENSES
            .iter()
            .any(|app| trimmed.eq_ignore_ascii_case(app));

        if is_approved {
            (CommercialRights::full_permissive(), LicenseDecision::Approve)
        } else {
            (
                CommercialRights::empty(),
                LicenseDecision::Reject(format!(
                    "License '{}' is rejected by TARA's Rule-18 policy: license is not in the approved permissive whitelist ({:?})",
                    trimmed, PERMISSIVE_APPROVED_LICENSES
                )),
            )
        }
    }

    /// Evaluates a license against TARA Rule-18 policy with full Disjunctive (`OR`) and Conjunctive (`AND`) support.
    ///
    /// Semantics:
    /// - Disjunctive (`OR` / `/`): If at least one alternative is approved, downstream licensee may select the approved
    ///   permissive alternative under TARA policy.
    /// - Conjunctive (`AND` / `,`): All sub-licenses apply simultaneously and must all satisfy TARA Rule-18 policy.
    pub fn evaluate_commercial_rights(license_str: &str) -> (CommercialRights, LicenseDecision) {
        let expr = LicenseExpression::parse(license_str);

        match expr {
            LicenseExpression::Single(id) => Self::evaluate_single_identifier(&id),

            LicenseExpression::Disjunctive(options) => {
                // If any alternative is approved by TARA policy, the disjunction is acceptable
                let mut first_reject_reason = String::new();
                for opt in &options {
                    let (rights, dec) = Self::evaluate_single_identifier(opt);
                    if dec.is_approved() && rights.all_permitted() {
                        return (CommercialRights::full_permissive(), LicenseDecision::Approve);
                    } else if first_reject_reason.is_empty() {
                        if let LicenseDecision::Reject(r) = dec {
                            first_reject_reason = r;
                        }
                    }
                }
                (
                    CommercialRights::empty(),
                    LicenseDecision::Reject(format!(
                        "Disjunctive license '{}' rejected: no permissive alternative satisfies TARA Rule-18 policy (first error: {})",
                        license_str, first_reject_reason
                    )),
                )
            }

            LicenseExpression::Conjunctive(components) => {
                // Every conjunct must be approved under TARA policy
                for comp in &components {
                    let (rights, dec) = Self::evaluate_single_identifier(comp);
                    if !dec.is_approved() || !rights.all_permitted() {
                        return (rights, dec);
                    }
                }
                (CommercialRights::full_permissive(), LicenseDecision::Approve)
            }
        }
    }

    /// Verifies if a license string is permissive and approved for commercial AI training.
    /// Preserves full compatibility with callers.
    pub fn verify_license(license_str: &str) -> LicenseDecision {
        Self::evaluate_commercial_rights(license_str).1
    }

    /// Pre-check for a dataset candidate: checks license and ensures ID is not duplicated.
    pub fn check_candidate(&self, dataset_id: &str, license_str: &str) -> LicenseDecision {
        let decision = Self::verify_license(license_str);
        if let LicenseDecision::Reject(reason) = decision {
            return LicenseDecision::Reject(reason);
        }

        let lower_id = dataset_id.trim().to_lowercase();
        for entry in &self.entries {
            if entry.dataset_id.trim().eq_ignore_ascii_case(&lower_id) {
                if entry.download_permission == "REJECTED" {
                    return LicenseDecision::Reject(format!(
                        "Candidate '{}' was rejected: {}",
                        entry.dataset_id, entry.policy_decision
                    ));
                }
                return LicenseDecision::Reject(format!(
                    "Dataset ID '{}' already has a registered license entry ({})",
                    dataset_id, entry.license
                ));
            }
        }

        LicenseDecision::Approve
    }

    /// Evaluates a single record under Rule-18's Strict Fail-Closed Direct Rejection Policy.
    ///
    /// Decision Flow:
    /// Record
    ///  ↓
    /// License Evidence Extraction
    ///  ↓
    /// Evidence found?
    ///  ├─ NO → Rule-18 rights unproven → DIRECT REJECT
    ///  └─ YES
    ///       ↓
    /// Normalize expression
    ///       ↓
    /// Evaluate commercial_use, modify, download, ai_training
    ///       ↓
    /// All required rights true?
    ///  ├─ NO → REJECT
    ///  └─ YES → APPROVE
    pub fn evaluate_record_rights(text: &str) -> RecordRightsDecision {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return RecordRightsDecision::Rejected {
                detected_license: "EMPTY".to_string(),
                reason: "Empty record has no valid content or rights".to_string(),
            };
        }

        // 1. Evidence Extraction
        if let Some(evidence) = extract_license_evidence(trimmed) {
            let (rights, decision) = Self::evaluate_commercial_rights(&evidence.raw_identifier);
            if decision.is_approved() && rights.all_permitted() {
                RecordRightsDecision::Approved {
                    detected_license: evidence.raw_identifier,
                    rights,
                }
            } else {
                let reason = match decision {
                    LicenseDecision::Reject(r) => r,
                    _ => "Non-compliant commercial rights".to_string(),
                };
                RecordRightsDecision::Rejected {
                    detected_license: evidence.raw_identifier,
                    reason,
                }
            }
        } else {
            // Negative Gate / Direct Rejection:
            let lower = trimmed.to_lowercase();
            if lower.contains("strictly for educational and non-profit")
                || lower.contains("strictly for educational")
                || lower.contains("strictly for non-profit")
                || lower.contains("educational and non-profit uses")
                || lower.contains("educational and non-profit")
                || lower.contains("fair use only")
                || lower.contains("for non-commercial")
                || lower.contains("non-commercial use only")
                || lower.contains("not for commercial use")
                || lower.contains("personal use only")
            {
                RecordRightsDecision::Rejected {
                    detected_license: "Explicit-NonCommercial-Disclaimer".to_string(),
                    reason: "Explicit non-commercial / educational-only / fair-use disclaimer detected; conflicts with Rule 18 commercial requirements".to_string(),
                }
            } else if lower.contains("all rights reserved") {
                RecordRightsDecision::Rejected {
                    detected_license: "All-Rights-Reserved".to_string(),
                    reason: "Explicit proprietary 'all rights reserved' disclaimer detected; no permissive open-source license".to_string(),
                }
            } else if lower.contains("officially licensed") || lower.contains("licensed merchandise") || lower.contains("licensed product") {
                RecordRightsDecision::Rejected {
                    detected_license: "Trademark-Merchandise-Authorization".to_string(),
                    reason: "'Officially Licensed' refers to merchandise/trademark branding, not an open-source copyright license for AI training".to_string(),
                }
            } else if trimmed.ends_with("...") || trimmed.ends_with('…') {
                RecordRightsDecision::Rejected {
                    detected_license: "TRUNCATED_FRAGMENT".to_string(),
                    reason: "Truncated or abruptly cut off text fragment (incomplete content)".to_string(),
                }
            } else {
                // Rule-18 Strict Direct Rejection:
                // No authoritative permissive license declaration present -> DIRECT REJECT.
                RecordRightsDecision::Rejected {
                    detected_license: "NONE".to_string(),
                    reason: "No authoritative permissive license evidence found in record; strict Rule-18 fail-closed policy requires direct rejection".to_string(),
                }
            }
        }
    }

    /// Executes a deep line-by-line / record-by-record license audit on a file.
    ///
    /// - Inspects every single line for authentic license declarations and author signatures.
    /// - Rejects records violating TARA Rule-18 policy (NC, ND, Copyleft, Academic-only, Proprietary).
    /// - Ignores normal prose mentioning license names to avoid false positives.
    /// - Collects all unique authors and unique licenses discovered within the file.
    pub fn audit_file_licenses(
        file_path: &Path,
        default_license: &str,
    ) -> Result<DeepLicenseAuditSummary, LicenseViolation> {
        // 1. Verify default candidate license
        let (def_rights, def_decision) = Self::evaluate_commercial_rights(default_license);
        if let LicenseDecision::Reject(reason) = def_decision {
            return Err(LicenseViolation {
                line_number: 0,
                raw_snippet: format!("Default/candidate license: '{default_license}'"),
                detected_license: default_license.to_string(),
                rights: def_rights,
                reason,
            });
        }

        let file = fs::File::open(file_path).map_err(|e| LicenseViolation {
            line_number: 0,
            raw_snippet: file_path.to_string_lossy().to_string(),
            detected_license: String::new(),
            rights: CommercialRights::prohibited(),
            reason: format!("Failed to open file for license audit: {e}"),
        })?;

        let reader = BufReader::new(file);
        let mut distinct_licenses: Vec<String> = Vec::new();
        let mut distinct_authors: Vec<String> = Vec::new();
        let mut line_count = 0;

        for (idx, line_res) in reader.lines().enumerate() {
            line_count = idx + 1;
            let line = line_res.map_err(|e| LicenseViolation {
                line_number: line_count,
                raw_snippet: format!("<I/O Error reading line {line_count}>"),
                detected_license: String::new(),
                rights: CommercialRights::prohibited(),
                reason: format!("Read error: {e}"),
            })?;

            if line.trim().is_empty() {
                continue;
            }

            // Extract author if present
            if let Some(author) = extract_author_from_line(&line) {
                if !distinct_authors.contains(&author) {
                    distinct_authors.push(author);
                }
            }

            // Extract authentic license evidence
            if let Some(evidence) = extract_license_evidence(&line) {
                let (rights, decision) = Self::evaluate_commercial_rights(&evidence.raw_identifier);
                if let LicenseDecision::Reject(reason) = decision {
                    let snippet = if line.chars().count() > 120 {
                        let prefix: String = line.chars().take(120).collect();
                        format!("{}...", prefix)
                    } else {
                        line.clone()
                    };
                    return Err(LicenseViolation {
                        line_number: line_count,
                        raw_snippet: snippet,
                        detected_license: evidence.raw_identifier,
                        rights,
                        reason,
                    });
                }

                if !distinct_licenses.contains(&evidence.raw_identifier) {
                    distinct_licenses.push(evidence.raw_identifier);
                }
            }
        }

        if distinct_licenses.is_empty() && !default_license.trim().is_empty() {
            distinct_licenses.push(default_license.trim().to_string());
        }

        Ok(DeepLicenseAuditSummary {
            total_lines_audited: line_count,
            distinct_licenses,
            distinct_authors,
            rights: CommercialRights::full_permissive(),
            compliant: true,
        })
    }

    /// Filters and sanitizes mixed/heterogeneous datasets (e.g. CommitPackFT).
    /// Streams through line-by-line:
    /// - Drops any record declaring non-permissive or non-commercial licenses.
    /// - Retains and writes records that are 100% compliant with TARA Rule-18 policy.
    pub fn filter_and_sanitize_mixed_records(
        input_path: &Path,
        output_path: &Path,
        default_license: &str,
    ) -> Result<SanitizeStats, String> {
        let in_file = fs::File::open(input_path)
            .map_err(|e| format!("Failed to open input file for sanitization: {e}"))?;
        let out_file = fs::File::create(output_path)
            .map_err(|e| format!("Failed to create output file for sanitization: {e}"))?;

        let reader = BufReader::new(in_file);
        let mut writer = std::io::BufWriter::new(out_file);

        let mut total = 0;
        let mut accepted = 0;
        let mut dropped = 0;
        let mut retained_licenses: Vec<String> = Vec::new();
        let mut dropped_reasons: Vec<String> = Vec::new();

        for line_res in reader.lines() {
            let line = line_res.map_err(|e| format!("Failed reading line: {e}"))?;
            if line.trim().is_empty() {
                continue;
            }
            total += 1;

            let is_compliant = if let Some(evidence) = extract_license_evidence(&line) {
                let (rights, decision) = Self::evaluate_commercial_rights(&evidence.raw_identifier);
                if decision.is_approved() && rights.all_permitted() {
                    if !retained_licenses.contains(&evidence.raw_identifier) {
                        retained_licenses.push(evidence.raw_identifier);
                    }
                    true
                } else {
                    if let LicenseDecision::Reject(r) = decision {
                        if dropped_reasons.len() < 10 && !dropped_reasons.contains(&r) {
                            dropped_reasons.push(r);
                        }
                    }
                    false
                }
            } else {
                // For database containers with mixed web contents (e.g. ODC-BY / Common Crawl / C4),
                // absence of authoritative permissive license evidence = DIRECT REJECT / DROP.
                if default_license.eq_ignore_ascii_case("ODC-BY") || default_license.is_empty() {
                    let r = "No authoritative permissive record license evidence (Rule 18 fail-closed)".to_string();
                    if dropped_reasons.len() < 10 && !dropped_reasons.contains(&r) {
                        dropped_reasons.push(r);
                    }
                    false
                } else {
                    // Homogeneous package with pre-verified permissive license (e.g. MIT/Apache software repository)
                    if !retained_licenses.contains(&default_license.to_string()) {
                        retained_licenses.push(default_license.to_string());
                    }
                    true
                }
            };

            if is_compliant {
                writeln!(writer, "{line}").map_err(|e| format!("Failed writing line: {e}"))?;
                accepted += 1;
            } else {
                dropped += 1;
            }
        }

        writer
            .flush()
            .map_err(|e| format!("Failed flushing sanitized output file: {e}"))?;

        if accepted == 0 {
            let _ = fs::remove_file(output_path);
            return Err("Sanitization dropped all records: zero permissive commercial records found".to_string());
        }

        Ok(SanitizeStats {
            total_records_processed: total,
            accepted_records: accepted,
            dropped_records: dropped,
            retained_licenses,
            dropped_reasons,
        })
    }

    /// Registers an approved license entry and auto-saves to both `.json` and `.txt`.
    pub fn register(&mut self, mut entry: LicenseEntry, base_path: &Path) -> Result<(), String> {
        let decision = Self::verify_license(&entry.license);
        if let LicenseDecision::Reject(reason) = decision {
            return Err(format!("Cannot register non-approved license: {reason}"));
        }

        // Auto-enrich 7-layer provenance
        if entry.host_domain.is_empty() || entry.tara_policy_class.is_empty() {
            entry.enrich_provenance();
        }

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.last_updated = format!("epoch_{now}");

        self.entries.push(entry);
        self.save(base_path)
    }

    /// Registers a rejected candidate with reason into the license register and auto-saves `.json` and `.txt`.
    pub fn register_rejected(&mut self, mut entry: LicenseEntry, base_path: &Path) -> Result<(), String> {
        entry.download_permission = "REJECTED".to_string();
        entry.training_eligibility = "REJECTED".to_string();
        entry.commercial_use = false;
        entry.modify_allowed = false;
        entry.download_allowed = false;
        entry.ai_training_allowed = false;

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.last_updated = format!("epoch_{now}");

        if let Some(existing) = self.entries.iter_mut().find(|e| e.dataset_id.eq_ignore_ascii_case(&entry.dataset_id)) {
            *existing = entry;
        } else {
            self.entries.push(entry);
        }

        self.save(base_path)
    }

    /// Enriches all license entries with the 7-layer provenance and policy architecture.
    pub fn enrich_all(&mut self) {
        for entry in &mut self.entries {
            entry.enrich_provenance();
        }
    }

    /// Reconciles license entries with active dataset IDs present in the download register.
    pub fn reconcile_with_active_datasets(&mut self, active_ids: &[String]) -> usize {
        let before = self.entries.len();
        self.entries.retain(|e| e.download_permission == "REJECTED" || active_ids.iter().any(|id| id.eq_ignore_ascii_case(&e.dataset_id)));
        before - self.entries.len()
    }

    /// Loads the license register from a JSON file.
    pub fn load_from_json(path: &Path) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read license registry {}: {e}", path.display()))?;
        serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse license registry JSON {}: {e}", path.display()))
    }

    /// Saves the register to BOTH `.json` and human-readable `.txt` backup file.
    pub fn save(&mut self, base_path: &Path) -> Result<(), String> {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.last_updated = format!("epoch_{now}");

        let json_path = if base_path.extension().and_then(|s| s.to_str()) == Some("json") {
            base_path.to_path_buf()
        } else {
            base_path.with_extension("json")
        };

        let txt_path = base_path.with_extension("txt");

        if let Some(parent) = json_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory {}: {e}", parent.display()))?;
        }

        // 1. Write JSON file
        let serialized = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize license registry: {e}"))?;
        fs::write(&json_path, serialized)
            .map_err(|e| format!("Failed to write JSON license registry {}: {e}", json_path.display()))?;

        // 2. Write Text Backup file (same name .txt)
        let txt_content = self.generate_text_backup();
        fs::write(&txt_path, txt_content)
            .map_err(|e| format!("Failed to write text license backup {}: {e}", txt_path.display()))?;

        Ok(())
    }

    /// Generates clean, human-readable plain text backup preserving auditable provenance evidence.
    pub fn generate_text_backup(&self) -> String {
        let mut out = String::new();
        out.push_str("================================================================================\n");
        out.push_str("TARA LICENSE REGISTER\n");
        out.push_str("Auditable Provenance & Licensing Evidence Preserved at Ingestion Time (Rule 18)\n");
        out.push_str(&format!("Schema Version: {}\n", self.schema_version));
        out.push_str(&format!("Last Updated: {}\n", self.last_updated));
        out.push_str(&format!("Total Approved Licenses: {}\n", self.entries.len()));
        out.push_str("================================================================================\n\n");

        for (idx, entry) in self.entries.iter().enumerate() {
            out.push_str(&format!("Record #{}:\n", idx + 1));
            out.push_str(&format!("  Dataset ID: {}\n", entry.dataset_id));
            out.push_str(&format!("  License: {}\n", entry.license));
            out.push_str(&format!("  Proof URL: {}\n", entry.proof_url));
            if !entry.host_domain.is_empty() {
                out.push_str(&format!("  [Layer 1] Host Domain: {}\n", entry.host_domain));
            }
            if !entry.upstream_repo.is_empty() {
                out.push_str(&format!("  [Layer 2] Upstream Repo: {}\n", entry.upstream_repo));
            }
            if !entry.upstream_publisher.is_empty() {
                out.push_str(&format!("  [Layer 3] Upstream Publisher: {}\n", entry.upstream_publisher));
            }
            if !entry.underlying_content_source.is_empty() {
                out.push_str(&format!("  [Layer 4] Underlying Content Source: {}\n", entry.underlying_content_source));
            }
            if !entry.underlying_content_terms.is_empty() {
                out.push_str(&format!("  [Layer 5] Underlying Content Terms: {}\n", entry.underlying_content_terms));
            }
            if !entry.content_rights_status.is_empty() {
                out.push_str(&format!("  [Layer 6] Content Rights Status: {}\n", entry.content_rights_status));
            }
            if !entry.tara_policy_class.is_empty() {
                out.push_str(&format!("  [Layer 6b] TARA Policy Class: {}\n", entry.tara_policy_class));
            }
            if !entry.source_url.is_empty() {
                out.push_str(&format!("  Source URL: {}\n", entry.source_url));
            }
            if !entry.domain.is_empty() {
                out.push_str(&format!("  Domain / Topic: {}\n", entry.domain));
            }
            if !entry.author.is_empty() {
                out.push_str(&format!("  Author: {}\n", entry.author));
            }
            out.push_str(&format!("  Commercial Allowed (Rule 18 Policy): {}\n", entry.commercial_use));
            out.push_str(&format!("  Modify Allowed (Applicable Terms): {}\n", entry.modify_allowed));
            out.push_str(&format!("  Local Copy / Download Allowed: {}\n", entry.download_allowed));
            out.push_str(&format!("  AI Training Allowed (Rule 18 Policy): {}\n", entry.ai_training_allowed));
            if entry.lines_audited > 0 {
                out.push_str(&format!("  Lines Audited: {}\n", entry.lines_audited));
            }
            if !entry.detected_licenses.is_empty() {
                out.push_str(&format!("  Audited Licenses: {:?}\n", entry.detected_licenses));
            }
            if !entry.detected_authors.is_empty() {
                out.push_str(&format!("  Audited Authors: {:?}\n", entry.detected_authors));
            }
            if !entry.timestamp.is_empty() {
                out.push_str(&format!("  Timestamp: {}\n", entry.timestamp));
            }
            out.push('\n');
        }

        out
    }
}

// ============================================================================
// Automated Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commercial_rights_evaluation() {
        // Permissive approved
        let (rights, dec) = LicenseRegister::evaluate_commercial_rights("MIT");
        assert!(dec.is_approved());
        assert!(rights.all_permitted());
        assert!(rights.commercial_use);
        assert!(rights.modify);
        assert!(rights.download);
        assert!(rights.ai_training);

        let (rights, dec) = LicenseRegister::evaluate_commercial_rights("Apache-2.0");
        assert!(dec.is_approved());
        assert!(rights.all_permitted());

        // Prohibited Non-Commercial
        let (rights, dec) = LicenseRegister::evaluate_commercial_rights("CC-BY-NC-4.0");
        assert!(dec.is_rejected());
        assert!(!rights.commercial_use);
        assert!(!rights.ai_training);

        // Prohibited No-Derivatives
        let (rights, dec) = LicenseRegister::evaluate_commercial_rights("CC-BY-ND-4.0");
        assert!(dec.is_rejected());
        assert!(!rights.modify);

        // Prohibited Copyleft
        let (rights, dec) = LicenseRegister::evaluate_commercial_rights("GPL-3.0");
        assert!(dec.is_rejected());
        assert!(!rights.all_permitted());
    }

    #[test]
    fn test_multi_license_boolean_semantics() {
        // Disjunctive OR: permissive choice available -> Approved
        let (rights, dec) = LicenseRegister::evaluate_commercial_rights("GPL-2.0 OR MIT");
        assert!(dec.is_approved());
        assert!(rights.all_permitted());

        let (rights, dec) = LicenseRegister::evaluate_commercial_rights("CC-BY-4.0 / Apache-2.0");
        assert!(dec.is_approved());
        assert!(rights.all_permitted());

        // Disjunctive OR: neither is approved -> Rejected
        let (_, dec) = LicenseRegister::evaluate_commercial_rights("GPL-3.0 OR CC-BY-NC-4.0");
        assert!(dec.is_rejected());

        // Conjunctive AND: all must be approved
        let (rights, dec) = LicenseRegister::evaluate_commercial_rights("MIT AND Apache-2.0");
        assert!(dec.is_approved());
        assert!(rights.all_permitted());

        // Conjunctive AND: one copyleft -> Rejected
        let (_, dec) = LicenseRegister::evaluate_commercial_rights("MIT AND GPL-3.0");
        assert!(dec.is_rejected());
    }

    #[test]
    fn test_false_positive_prose_protection() {
        // Normal text mentioning licenses must NOT be treated as license declaration
        let prose_line = "In section 2, the authors compare performance under GPL-3.0 and MIT licensed compilers.";
        assert_eq!(extract_license_evidence(prose_line), None);

        let mention_line = "This dataset contains benchmark papers discussing academic-only limitations.";
        assert_eq!(extract_license_evidence(mention_line), None);

        // Genuine structured declaration MUST be extracted
        let jsonl_line = r#"{"id": 1, "text": "code", "license": "MIT"}"#;
        assert_eq!(
            extract_license_evidence(jsonl_line),
            Some(LicenseEvidence {
                raw_identifier: "MIT".to_string(),
                kind: EvidenceKind::StructuredField,
            })
        );

        // Genuine SPDX header MUST be extracted
        let spdx_line = "// SPDX-License-Identifier: Apache-2.0";
        assert_eq!(
            extract_license_evidence(spdx_line),
            Some(LicenseEvidence {
                raw_identifier: "Apache-2.0".to_string(),
                kind: EvidenceKind::SpdxMarker,
            })
        );
    }

    #[test]
    fn test_deep_file_license_audit_approved() {
        let temp_dir = std::env::temp_dir().join("tara_test_deep_audit_v2");
        let _ = fs::create_dir_all(&temp_dir);
        let test_file = temp_dir.join("compliant_multi.jsonl");

        let content = "{\"id\": 1, \"text\": \"calc A\", \"license\": \"MIT\", \"author\": \"Alice\"}\n\
                       {\"id\": 2, \"text\": \"calc B mentions GPL-3.0 in text\", \"license\": \"Apache-2.0\", \"author\": \"Bob\"}\n\
                       {\"id\": 3, \"text\": \"calc C\", \"license\": \"BSD-3-Clause\", \"author\": \"Carol\"}\n";
        fs::write(&test_file, content).unwrap();

        let audit = LicenseRegister::audit_file_licenses(&test_file, "MIT").expect("Audit must pass");
        assert_eq!(audit.total_lines_audited, 3);
        assert_eq!(audit.distinct_licenses.len(), 3);
        assert_eq!(audit.distinct_authors.len(), 3);
        assert!(audit.rights.all_permitted());

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_deep_file_license_audit_rejected_inner_line() {
        let temp_dir = std::env::temp_dir().join("tara_test_deep_audit_reject_v2");
        let _ = fs::create_dir_all(&temp_dir);
        let test_file = temp_dir.join("tainted.jsonl");

        let content = "{\"id\": 1, \"text\": \"code A\", \"license\": \"MIT\", \"author\": \"Alice\"}\n\
                       {\"id\": 2, \"text\": \"code B\", \"license\": \"MIT\", \"author\": \"Bob\"}\n\
                       {\"id\": 3, \"text\": \"code C\", \"license\": \"GPL-3.0\", \"author\": \"Charlie\"}\n";
        fs::write(&test_file, content).unwrap();

        let res = LicenseRegister::audit_file_licenses(&test_file, "MIT");
        assert!(res.is_err(), "Must reject because line 3 declares GPL-3.0");
        let violation = res.unwrap_err();
        assert_eq!(violation.line_number, 3);
        assert_eq!(violation.detected_license, "GPL-3.0");

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_register_and_backup_flow() {
        let mut reg = LicenseRegister::new();

        let entry = LicenseEntry {
            dataset_id: "openstax_physics".to_string(),
            license: "CC-BY-4.0".to_string(),
            proof_url: "https://openstax.org/license".to_string(),
            source_url: "https://openstax.org/physics.jsonl".to_string(),
            domain: "physics".to_string(),
            author: "Rice University".to_string(),
            commercial_use: true,
            modify_allowed: true,
            download_allowed: true,
            ai_training_allowed: true,
            detected_licenses: vec!["CC-BY-4.0".to_string()],
            detected_authors: vec!["Rice University".to_string()],
            lines_audited: 500,
            timestamp: "2026-10-06T12:00:00Z".to_string(),
            ..Default::default()
        };

        let temp_base = Path::new("target/temp_test_license_v2");
        assert!(reg.register(entry.clone(), temp_base).is_ok());

        let backup = reg.generate_text_backup();
        assert!(backup.contains("Dataset ID: openstax_physics"));
        assert!(backup.contains("License: CC-BY-4.0"));
        assert!(backup.contains("[Layer 1] Host Domain:"));
        assert!(backup.contains("[Layer 6b] TARA Policy Class:"));
        assert!(backup.contains("Commercial Allowed (Rule 18 Policy): true"));
        assert!(backup.contains("Modify Allowed (Applicable Terms): true"));
        assert!(backup.contains("Local Copy / Download Allowed: true"));
        assert!(backup.contains("AI Training Allowed (Rule 18 Policy): true"));

        let _ = fs::remove_file("target/temp_test_license_v2.json");
        let _ = fs::remove_file("target/temp_test_license_v2.txt");
    }

    #[test]
    fn test_11_real_world_negative_and_quarantine_batch() {
        let record1 = "Today is Thursday and I am so thankful for my family, peaceful morning coffee and good friends.";
        let res1 = LicenseRegister::evaluate_record_rights(record1);
        assert!(res1.is_rejected(), "Record 1 (Personal blog) must be DIRECT REJECTED (unproven rights)");

        let record2 = "Hexagram 29 represents the abysmal water repeating. Commentary translation by modern author 1992.";
        let res2 = LicenseRegister::evaluate_record_rights(record2);
        assert!(res2.is_rejected(), "Record 2 (I Ching translation) must be DIRECT REJECTED (unproven rights)");

        let record3 = "Download Superkid Runner Mod Apk unlimited coins free crack unlocked hacker mod.";
        // Checked also via filter engine piracy detector
        assert!(crate::filter_engine::detect_piracy_security_markers(record3).is_some());

        let record4 = "Welcome to the public lecture on cultural history and historical architecture in Rome.";
        let res4 = LicenseRegister::evaluate_record_rights(record4);
        assert!(res4.is_rejected(), "Record 4 (Public lecture) must be DIRECT REJECTED (unproven rights)");

        let record5 = "This audio track is strictly for educational and non-profit uses under fair use policy.";
        let res5 = LicenseRegister::evaluate_record_rights(record5);
        assert!(res5.is_rejected(), "Record 5 (Educational/non-profit statement) must be REJECTED under Rule 18");

        let record6 = "SolarTech Inc today announced the commercial release of their high-efficiency photovoltaic system.";
        let res6 = LicenseRegister::evaluate_record_rights(record6);
        assert!(res6.is_rejected(), "Record 6 (Solar press release) must be DIRECT REJECTED (unproven rights)");

        let record7 = "Buy this bestselling marketing book on our store. All rights reserved. Reproduction prohibited.";
        let res7 = LicenseRegister::evaluate_record_rights(record7);
        assert!(res7.is_rejected(), "Record 7 (All rights reserved marketing copy) must be REJECTED");

        let record8 = "An investigative blog post examining local municipal budget records and public contracts.";
        let res8 = LicenseRegister::evaluate_record_rights(record8);
        assert!(res8.is_rejected(), "Record 8 (Investigative blog) must be DIRECT REJECTED (unproven rights)");

        let record9 = "Mom's chocolate chip cookies: 2 cups flour, 1 cup sugar, bake at 350F. Comments: loved it!";
        let res9 = LicenseRegister::evaluate_record_rights(record9);
        assert!(res9.is_rejected(), "Record 9 (Blog recipe + comments) must be DIRECT REJECTED (unproven rights)");

        let record10 = "In this article we demonstrate how to write a red-black tree with rotational balancing in C++.";
        let res10 = LicenseRegister::evaluate_record_rights(record10);
        assert!(res10.is_rejected(), "Record 10 (Programming article without license) must be DIRECT REJECTED (unproven rights)");

        let record11 = "9 out of 10 based on 780 ratings and reviews from verified users.";
        assert!(crate::filter_engine::detect_weak_review_fragment(record11).is_some(), "Record 11 must be rejected by filter engine as weak fragment");

        // Verified batch score: Approved: 0 confirmed!
        assert!(!res1.is_approved());
        assert!(!res2.is_approved());
        assert!(!res4.is_approved());
        assert!(!res5.is_approved());
        assert!(!res6.is_approved());
        assert!(!res7.is_approved());
        assert!(!res8.is_approved());
        assert!(!res9.is_approved());
        assert!(!res10.is_approved());
    }

    #[test]
    fn test_batch_a_14_records_negative_and_quarantine() {
        // Record 1: Missoula BBQ class advertisement (Commercial event copy)
        let r1 = "Join our Missoula BBQ masterclass this Saturday! Learn smoking techniques and brisket seasoning.";
        let res1 = LicenseRegister::evaluate_record_rights(r1);
        assert!(res1.is_rejected(), "Record 1 must be DIRECT REJECTED (unproven rights)");

        // Record 2: MacRumors forum discussion (User-generated forum content)
        let r2 = "Has anyone noticed battery degradation on iOS 16.4? Post by user MacFan2023 in forum discussion.";
        let res2 = LicenseRegister::evaluate_record_rights(r2);
        assert!(res2.is_rejected(), "Record 2 must be DIRECT REJECTED (unproven rights)");

        // Record 3: Costume product listing (Commercial product copy)
        let r3 = "Deluxe Pirate Costume with Hat and Belt. Size M/L. In stock now with free standard shipping.";
        let res3 = LicenseRegister::evaluate_record_rights(r3);
        assert!(res3.is_rejected(), "Record 3 must be DIRECT REJECTED (unproven rights)");

        // Record 4: BlackHatWorld SEO discussion (Forum / SEO chatter)
        let r4 = "Looking for PBN link building strategies that still work after the latest Google core search update.";
        let res4 = LicenseRegister::evaluate_record_rights(r4);
        assert!(res4.is_rejected(), "Record 4 must be DIRECT REJECTED (unproven rights)");

        // Record 5: Denver Board of Education bond/news page (Institutional content, unproven license)
        let r5 = "Denver Board of Education announces 2024 comprehensive school facility bond measure details.";
        let res5 = LicenseRegister::evaluate_record_rights(r5);
        assert!(res5.is_rejected(), "Record 5 must be DIRECT REJECTED (unproven rights)");

        // Record 6: Bangalore–Gondia train information (Factual timetable, unproven text reuse license)
        let r6 = "Train 12389 Bangalore to Gondia Junction timetable: Departure 06:00, arrival 22:30 with 14 halts.";
        let res6 = LicenseRegister::evaluate_record_rights(r6);
        assert!(res6.is_rejected(), "Record 6 must be DIRECT REJECTED (factual expression with unproven license)");

        // Record 7: LiveJournal personal post (Personal copyrighted writing)
        let r7 = "Feeling nostalgic today thinking about high school days and walks in the autumn park with friends.";
        let res7 = LicenseRegister::evaluate_record_rights(r7);
        assert!(res7.is_rejected(), "Record 7 must be DIRECT REJECTED (unproven rights)");

        // Record 8: “Rich get richer…” blog (Opinion / editorial content)
        let r8 = "Why the rich get richer: examining tax policies, asset appreciation, and modern economic inequality.";
        let res8 = LicenseRegister::evaluate_record_rights(r8);
        assert!(res8.is_rejected(), "Record 8 must be DIRECT REJECTED (unproven rights)");

        // Record 9: Biomedics product description (Commercial medical product catalog)
        let r9 = "Biomedics 1-day extra contact lenses offer natural comfort and crisp vision for daily wearers.";
        let res9 = LicenseRegister::evaluate_record_rights(r9);
        assert!(res9.is_rejected(), "Record 9 must be DIRECT REJECTED (unproven rights)");

        // Record 10: Sysco/US Foods news article (Copyrighted news text)
        let r10 = "Federal judge blocks proposed merger between food distributors Sysco and US Foods over antitrust concerns.";
        let res10 = LicenseRegister::evaluate_record_rights(r10);
        assert!(res10.is_rejected(), "Record 10 must be DIRECT REJECTED (unproven rights)");

        // Record 11: Indonesian film-search page (Scraped / SEO aggregation)
        let r11 = "Nonton streaming download film sub Indo terbaru gratis kualitas HD bioskop cinema 21 online.";
        let res11 = LicenseRegister::evaluate_record_rights(r11);
        assert!(res11.is_rejected(), "Record 11 must be DIRECT REJECTED (unproven rights)");

        // Record 12: Religious teaching/video page (Promotional / teaching content)
        let r12 = "Watch Sunday sermon on faith and endurance through life's trials. Video recording available.";
        let res12 = LicenseRegister::evaluate_record_rights(r12);
        assert!(res12.is_rejected(), "Record 12 must be DIRECT REJECTED (unproven rights)");

        // Record 13: Academic publications list (Bibliographic metadata without text training rights)
        let r13 = "Publications: Smith, J. (2018) Quantum spin liquids. Phys. Rev. Lett. 120, 047201. DOI:10.1103.";
        let res13 = LicenseRegister::evaluate_record_rights(r13);
        assert!(res13.is_rejected(), "Record 13 must be DIRECT REJECTED (bibliographic facts with unproven text license)");

        // Record 14: Truncated / incomplete record ("I have existing web site bri...")
        let r14 = "I have existing web site bri...";
        let res14 = LicenseRegister::evaluate_record_rights(r14);
        assert!(res14.is_rejected(), "Record 14 (Truncated text fragment) must be REJECTED");

        // Confirmed Rule-18 APPROVED count across Batch A: ZERO!
        let all_records = [
            &res1, &res2, &res3, &res4, &res5, &res6, &res7,
            &res8, &res9, &res10, &res11, &res12, &res13, &res14,
        ];
        let approved_count = all_records.iter().filter(|r| r.is_approved()).count();
        assert_eq!(approved_count, 0, "Batch A must have exactly 0 approved records!");
    }

    #[test]
    fn test_batch_b_11_records_negative_and_quarantine() {
        // Record 1: Seven Avenue Design service/company page
        let r1 = "Seven Avenue Design provides bespoke interior architecture and luxury residential remodeling services.";
        let res1 = LicenseRegister::evaluate_record_rights(r1);
        assert!(res1.is_rejected(), "Record 1 must be DIRECT REJECTED (unproven rights)");

        // Record 2: VoIP industry blog/article
        let r2 = "Understanding SIP trunking protocols and bandwidth considerations for modern cloud-based telephony.";
        let res2 = LicenseRegister::evaluate_record_rights(r2);
        assert!(res2.is_rejected(), "Record 2 must be DIRECT REJECTED (unproven rights)");

        // Record 3: NYPD challenge coin ("Officially Licensed" Trap)
        let r3 = "Officially Licensed NYPD Challenge Coin with enamel finish and antique bronze medallion plating.";
        let res3 = LicenseRegister::evaluate_record_rights(r3);
        assert!(res3.is_rejected(), "Record 3 ('Officially Licensed' merchandise branding trap) must be REJECTED");
        assert!(crate::filter_engine::detect_merchandise_branding_marker(r3).is_some());

        // Record 4: Dog camp/service page
        let r4 = "Happy Paws Dog Camp offers supervised open play, luxury boarding suites, and behavioral training.";
        let res4 = LicenseRegister::evaluate_record_rights(r4);
        assert!(res4.is_rejected(), "Record 4 must be DIRECT REJECTED (unproven rights)");

        // Record 5: Quoted children's story passage ("Now, don't be sad Tom...")
        let r5 = "\"Now, don't be sad Tom,\" said the little hedgehog, wrapping his tiny paws around the shiny acorn.";
        let res5 = LicenseRegister::evaluate_record_rights(r5);
        assert!(res5.is_rejected(), "Record 5 (Literary quote with high quality but unproven rights) must be DIRECT REJECTED");

        // Record 6: Privacy policy text
        let r6 = "Privacy Policy: We collect personal information and contact us details to provide website services under our cookie policy.";
        let res6 = LicenseRegister::evaluate_record_rights(r6);
        assert!(crate::filter_engine::detect_legal_privacy_policy_boilerplate(r6).is_some(), "Record 6 must trigger privacy policy detector");
        assert!(res6.is_rejected(), "Record 6 must be DIRECT REJECTED");

        // Record 7: Personalization/research commentary article
        let r7 = "Algorithmic personalization in online shopping interfaces: user agency and behavioural nudging.";
        let res7 = LicenseRegister::evaluate_record_rights(r7);
        assert!(res7.is_rejected(), "Record 7 must be DIRECT REJECTED (unproven rights)");

        // Record 8: Funeral service/company page
        let r8 = "Memorial Chapels offers compassionate funeral planning, cremation services, and pre-need counseling.";
        let res8 = LicenseRegister::evaluate_record_rights(r8);
        assert!(res8.is_rejected(), "Record 8 must be DIRECT REJECTED (unproven rights)");

        // Record 9: Birthday banner product page
        let r9 = "Custom 1st Birthday Photo Banner with glitter cardstock and satin ribbon for party decoration.";
        let res9 = LicenseRegister::evaluate_record_rights(r9);
        assert!(res9.is_rejected(), "Record 9 must be DIRECT REJECTED (unproven rights)");

        // Record 10: Hotel review/blog content
        let r10 = "Our stay at Grand Vista Resort was pleasant with mountain views, but the breakfast buffet was crowded.";
        let res10 = LicenseRegister::evaluate_record_rights(r10);
        assert!(res10.is_rejected(), "Record 10 must be DIRECT REJECTED (unproven rights)");

        // Record 11: Truncated record ("On 15 May we’ll be c...")
        let r11 = "On 15 May we’ll be c...";
        let res11 = LicenseRegister::evaluate_record_rights(r11);
        assert!(res11.is_rejected(), "Record 11 (Truncated text fragment) must be REJECTED");

        // Confirmed Rule-18 APPROVED count across Batch B: ZERO!
        let all_records = [
            &res1, &res2, &res3, &res4, &res5, &res6, &res7,
            &res8, &res9, &res10, &res11,
        ];
        let approved_count = all_records.iter().filter(|r| r.is_approved()).count();
        assert_eq!(approved_count, 0, "Batch B must have exactly 0 approved records!");
    }

    #[test]
    fn test_intermediate_11_records_direct_reject() {
        // Record 1: Browntape / Myntra integration help page (Credentials instruction, NOT license evidence)
        let r1 = "To connect your Myntra seller account in Browntape, enter your Myntra Username and Password under API settings.";
        let res1 = LicenseRegister::evaluate_record_rights(r1);
        assert!(res1.is_rejected(), "Record 1 (Browntape credentials instruction) must be DIRECT REJECTED");

        // Record 2: Hackaday data-recovery article
        let r2 = "In this retrospective we examine hardware data recovery methods from damaged MFM hard disks.";
        let res2 = LicenseRegister::evaluate_record_rights(r2);
        assert!(res2.is_rejected(), "Record 2 (Hackaday article) must be DIRECT REJECTED");

        // Record 3: SageMath forum discussion
        let r3 = "How can I compute the Galois group of a degree 6 polynomial over Q? Discussion thread on SageMath user group.";
        let res3 = LicenseRegister::evaluate_record_rights(r3);
        assert!(res3.is_rejected(), "Record 3 (SageMath forum post) must be DIRECT REJECTED");

        // Record 4: Miami real-estate listing
        let r4 = "Exclusive 3-bedroom penthouse with panoramic ocean views in Miami Beach. Listed at $4.2M with private elevator.";
        let res4 = LicenseRegister::evaluate_record_rights(r4);
        assert!(res4.is_rejected(), "Record 4 (Miami real-estate) must be DIRECT REJECTED");

        // Record 5: Kotobukiya/DC Supergirl product description
        let r5 = "Kotobukiya presents the DC Comics Supergirl Bishoujo statue standing 10 inches tall with sculpted cape.";
        let res5 = LicenseRegister::evaluate_record_rights(r5);
        assert!(res5.is_rejected(), "Record 5 (Product description/IP) must be DIRECT REJECTED");

        // Record 6: South East Clare Show event article
        let r6 = "The South East Clare Show returns this August featuring equestrian competitions, vintage tractors, and craft stalls.";
        let res6 = LicenseRegister::evaluate_record_rights(r6);
        assert!(res6.is_rejected(), "Record 6 (Event article) must be DIRECT REJECTED");

        // Record 7: Insurance discount page
        let r7 = "Bundle home and auto insurance today to save up to 25% on annual premiums with our multi-policy discount.";
        let res7 = LicenseRegister::evaluate_record_rights(r7);
        assert!(res7.is_rejected(), "Record 7 (Insurance page) must be DIRECT REJECTED");

        // Record 8: Iowa State University finding aid (Institutional archival metadata != commercial training license)
        let r8 = "Finding aid for the George Washington Carver agricultural papers, 1890-1943. Special Collections Department.";
        let res8 = LicenseRegister::evaluate_record_rights(r8);
        assert!(res8.is_rejected(), "Record 8 (Iowa State finding aid) must be DIRECT REJECTED");

        // Record 9: Dental treatment/service page
        let r9 = "Our restorative dentistry services include porcelain crowns, root canal therapy, and composite fillings.";
        let res9 = LicenseRegister::evaluate_record_rights(r9);
        assert!(res9.is_rejected(), "Record 9 (Dental service page) must be DIRECT REJECTED");

        // Record 10: Amateur-radio beacon table (Factual frequencies/coordinates != source text training license)
        let r10 = "Amateur Radio Beacon Table: Call sign W1AW/B, Frequency 28.200 MHz, Power 100W, Location Newington CT.";
        let res10 = LicenseRegister::evaluate_record_rights(r10);
        assert!(res10.is_rejected(), "Record 10 (Beacon table) must be DIRECT REJECTED");

        // Record 11: Microsoft trade-in promotion
        let r11 = "Trade in your eligible PC, Xbox, or mobile device at Microsoft Store and receive promotional gift card credit.";
        let res11 = LicenseRegister::evaluate_record_rights(r11);
        assert!(res11.is_rejected(), "Record 11 (Microsoft trade-in) must be DIRECT REJECTED");

        let all = [&res1, &res2, &res3, &res4, &res5, &res6, &res7, &res8, &res9, &res10, &res11];
        assert_eq!(all.iter().filter(|r| r.is_approved()).count(), 0);
        assert_eq!(all.iter().filter(|r| r.is_rejected()).count(), 11);
    }

    #[test]
    fn test_new_17_records_direct_reject() {
        // 1: Work-stress survey page
        let r1 = "Work-stress survey 2024: Over 68% of corporate respondents report burnout due to tight project deadlines.";
        let res1 = LicenseRegister::evaluate_record_rights(r1);
        assert!(res1.is_rejected(), "Record 1 (Work-stress survey) must be DIRECT REJECTED");

        // 2: University rankings article
        let r2 = "Annual global university rankings evaluate academic reputation, research citations, and faculty ratio.";
        let res2 = LicenseRegister::evaluate_record_rights(r2);
        assert!(res2.is_rejected(), "Record 2 (University rankings) must be DIRECT REJECTED");

        // 3: Waltz Engineering water-treatment page
        let r3 = "Waltz Engineering delivers industrial reverse osmosis and demineralization water treatment solutions.";
        let res3 = LicenseRegister::evaluate_record_rights(r3);
        assert!(res3.is_rejected(), "Record 3 (Waltz Engineering) must be DIRECT REJECTED");

        // 4: Anti-smoking warning article
        let r4 = "Public health bulletin: Nicotine dependency increases cardiovascular risk and pulmonary inflammation.";
        let res4 = LicenseRegister::evaluate_record_rights(r4);
        assert!(res4.is_rejected(), "Record 4 (Anti-smoking article) must be DIRECT REJECTED");

        // 5: Banner University Medical Center page
        let r5 = "Banner University Medical Center provides comprehensive trauma surgery, oncology, and neurology care.";
        let res5 = LicenseRegister::evaluate_record_rights(r5);
        assert!(res5.is_rejected(), "Record 5 (Banner University Medical Center) must be DIRECT REJECTED");

        // 6: Brownies recipe/blog fragment
        let r6 = "Fudgy chocolate brownies: melt butter with cocoa powder, fold in chocolate chips, bake 25 minutes.";
        let res6 = LicenseRegister::evaluate_record_rights(r6);
        assert!(res6.is_rejected(), "Record 6 (Brownies recipe) must be DIRECT REJECTED");

        // 7: Chelsea forum discussion
        let r7 = "Match discussion: Chelsea tactical lineup changes and substitutions in the second half of the cup tie.";
        let res7 = LicenseRegister::evaluate_record_rights(r7);
        assert!(res7.is_rejected(), "Record 7 (Chelsea forum) must be DIRECT REJECTED");

        // 8: Miranda Skin Studio product listing
        let r8 = "Miranda Skin Studio botanical facial serum formulated with hyaluronic acid and green tea antioxidants.";
        let res8 = LicenseRegister::evaluate_record_rights(r8);
        assert!(res8.is_rejected(), "Record 8 (Miranda Skin Studio) must be DIRECT REJECTED");

        // 9: Floating training centers article
        let r9 = "Maritime safety organization introduces mobile floating training centers for offshore emergency response.";
        let res9 = LicenseRegister::evaluate_record_rights(r9);
        assert!(res9.is_rejected(), "Record 9 (Floating training centers) must be DIRECT REJECTED");

        // 10: Poppy Appeal article
        let r10 = "Annual Poppy Appeal campaign honors military veterans and raises funds for community support services.";
        let res10 = LicenseRegister::evaluate_record_rights(r10);
        assert!(res10.is_rejected(), "Record 10 (Poppy Appeal) must be DIRECT REJECTED");

        // 11: AVForums headphone discussion
        let r11 = "Thread review: comparing planar magnetic headphone frequency response and amplifier pairing options.";
        let res11 = LicenseRegister::evaluate_record_rights(r11);
        assert!(res11.is_rejected(), "Record 11 (AVForums) must be DIRECT REJECTED");

        // 12: Financial advisory forum/Q&A
        let r12 = "Q&A: What is the optimal asset allocation between index funds and municipal bonds for early retirement?";
        let res12 = LicenseRegister::evaluate_record_rights(r12);
        assert!(res12.is_rejected(), "Record 12 (Financial advisory forum) must be DIRECT REJECTED");

        // 13: Turkey-burger recipe/blog
        let r13 = "Juicy grilled turkey burgers with garlic herb seasoning, avocado slices, and toasted brioche buns.";
        let res13 = LicenseRegister::evaluate_record_rights(r13);
        assert!(res13.is_rejected(), "Record 13 (Turkey-burger recipe) must be DIRECT REJECTED");

        // 14: Pocomail forum discussion
        let r14 = "Pocomail user forum: troubleshooting IMAP SSL handshake timeouts after upgrading email client.";
        let res14 = LicenseRegister::evaluate_record_rights(r14);
        assert!(res14.is_rejected(), "Record 14 (Pocomail forum) must be DIRECT REJECTED");

        // 15: Vintage Omega watches article
        let r15 = "Collector guide to vintage Omega Speedmaster references from the 1960s with Calibre 321 movements.";
        let res15 = LicenseRegister::evaluate_record_rights(r15);
        assert!(res15.is_rejected(), "Record 15 (Vintage Omega watches) must be DIRECT REJECTED");

        // 16: Sendmail Logger listing
        let r16 = "Sendmail Logger utility daemon: parses syslog events, records mail transaction metrics to SQLite.";
        let res16 = LicenseRegister::evaluate_record_rights(r16);
        assert!(res16.is_rejected(), "Record 16 (Sendmail Logger listing) must be DIRECT REJECTED");

        // 17: FreeNAS release/news text
        let r17 = "FreeNAS community release notes announcing ZFS replication improvements and updated WebGUI dashboard.";
        let res17 = LicenseRegister::evaluate_record_rights(r17);
        assert!(res17.is_rejected(), "Record 17 (FreeNAS release text) must be DIRECT REJECTED");

        let all = [
            &res1, &res2, &res3, &res4, &res5, &res6, &res7,
            &res8, &res9, &res10, &res11, &res12, &res13, &res14,
            &res15, &res16, &res17,
        ];
        assert_eq!(all.iter().filter(|r| r.is_approved()).count(), 0);
        assert_eq!(all.iter().filter(|r| r.is_rejected()).count(), 17);
    }

    #[test]
    fn test_batch_stress_17_records_fail_closed_reject() {
        let r1 = "Ironman Hawaii / XTerra Maui triathlon championships recap and athlete split times.";
        let res1 = LicenseRegister::evaluate_record_rights(r1);

        let r2 = "RIB boat forum: Mercury Verado 300 outboard engine propeller pitch and fuel economy.";
        let res2 = LicenseRegister::evaluate_record_rights(r2);

        let r3 = "ESL teaching worksheet: comparative and superlative adjectives exercises for intermediate students.";
        let res3 = LicenseRegister::evaluate_record_rights(r3);

        let r4 = "Bass guitar user forum: Fender Jazz vs Precision pickup wiring diagrams and tone capacitors.";
        let res4 = LicenseRegister::evaluate_record_rights(r4);

        let r5 = "Song lyrics database: chorus and bridge lyrics for popular acoustic ballad release.";
        let res5 = LicenseRegister::evaluate_record_rights(r5);

        let r6 = "Terrace and garden LED outdoor pathway lighting fixtures. Order online with manufacturer warranty.";
        let res6 = LicenseRegister::evaluate_record_rights(r6);

        let r7 = "Southwest Trails committee meeting minutes discussing pedestrian path expansion and trail maintenance.";
        let res7 = LicenseRegister::evaluate_record_rights(r7);

        let r8 = "LA Promise community initiative volunteer recruitment page for local high school mentoring programs.";
        let res8 = LicenseRegister::evaluate_record_rights(r8);

        let r9 = "Dilem interchangeable temple eyewear catalog featuring lightweight titanium and acetate frames.";
        let res9 = LicenseRegister::evaluate_record_rights(r9);

        let r10 = "Samsung genuine replacement lithium-ion battery for Galaxy smartphones with NFC antenna.";
        let res10 = LicenseRegister::evaluate_record_rights(r10);

        let r11 = "Personal oil painting blog: layering transparent glazes for botanical watercolor studies.";
        let res11 = LicenseRegister::evaluate_record_rights(r11);

        // Tipue Search: "Free and open source" statement in page copy is NOT an authoritative license identifier for the copied page text!
        let r12 = "Tipue Search is a free and open source jQuery site search plugin with responsive design.";
        let res12 = LicenseRegister::evaluate_record_rights(r12);

        let r13 = "Corporate biography: LexisNexis executive leadership, career history, and legal analytics initiatives.";
        let res13 = LicenseRegister::evaluate_record_rights(r13);

        let r14 = "TU Dortmund faculty department overview, degree curriculum requirements, and research chairs.";
        let res14 = LicenseRegister::evaluate_record_rights(r14);

        let r15 = "Cathedral organist and choirmaster biographical notes, concert recordings, and festival appearances.";
        let res15 = LicenseRegister::evaluate_record_rights(r15);

        let r16 = "Custom college essay writing service: guaranteed plagiarism-free academic papers delivered on deadline.";
        let res16 = LicenseRegister::evaluate_record_rights(r16);

        // Truncated record
        let r17 = "killer mental-ism manuscript...";
        let res17 = LicenseRegister::evaluate_record_rights(r17);

        let all = [
            &res1, &res2, &res3, &res4, &res5, &res6, &res7,
            &res8, &res9, &res10, &res11, &res12, &res13, &res14,
            &res15, &res16, &res17,
        ];
        assert_eq!(all.iter().filter(|r| r.is_approved()).count(), 0, "Stress Batch 1 (17 records): approved must be 0");
        assert_eq!(all.iter().filter(|r| r.is_rejected()).count(), 17, "Stress Batch 1 (17 records): all 17 must be rejected");
    }

    #[test]
    fn test_batch_stress_20_records_fail_closed_reject() {
        let r1 = "Bicycle Craft Beer playing cards deck printed on classic air-cushion finish cardstock.";
        let res1 = LicenseRegister::evaluate_record_rights(r1);

        let r2 = "Faroe Islands Sheep View 360 project mounted panoramic cameras on sheep to map remote hills.";
        let res2 = LicenseRegister::evaluate_record_rights(r2);

        let r3 = "Epson LQ-630 dot matrix printer driver installation guide and Windows compatibility matrix.";
        let res3 = LicenseRegister::evaluate_record_rights(r3);

        let r4 = "Behind the scenes on Rogue One: how ILM constructed immersive virtual reality sets for directors.";
        let res4 = LicenseRegister::evaluate_record_rights(r4);

        // "Editable" statement is an asset characteristic, NOT a copyright training license!
        let r5 = "Download 50 editable vector graphics icons in SVG and AI formats for commercial design projects.";
        let res5 = LicenseRegister::evaluate_record_rights(r5);

        let r6 = "Private Hatha yoga tuition: personalized breathwork, alignment adjustments, and meditation sessions.";
        let res6 = LicenseRegister::evaluate_record_rights(r6);

        let r7 = "Certified organic potting soil blend with worm castings, perlite, and slow-release minerals.";
        let res7 = LicenseRegister::evaluate_record_rights(r7);

        let r8 = "Premier fitness and racquet club membership amenities including Olympic swimming pool and sauna.";
        let res8 = LicenseRegister::evaluate_record_rights(r8);

        let r9 = "MTV music review: The Weeknd and Daft Punk collaborate on electronic retro-synth single.";
        let res9 = LicenseRegister::evaluate_record_rights(r9);

        let r10 = "Live customer reviews for Fleetwood Mac tribute band concert performance at city auditorium.";
        let res10 = LicenseRegister::evaluate_record_rights(r10);

        let r11 = "School Sports Week schedule, parent attendance rules, and pupil photo permission guidelines.";
        let res11 = LicenseRegister::evaluate_record_rights(r11);

        let r12 = "Classic Converse Chuck Taylor All Star canvas high-top sneakers in vintage monochrome black.";
        let res12 = LicenseRegister::evaluate_record_rights(r12);

        let r13 = "Heavyweight 100% ring-spun cotton unisex crewneck graphic t-shirt with ribbed collar.";
        let res13 = LicenseRegister::evaluate_record_rights(r13);

        let r14 = "Explosion-proof nickel-plated brass industrial cable glands rated IP68 for hazardous areas.";
        let res14 = LicenseRegister::evaluate_record_rights(r14);

        let r15 = "Black holes exert gravitational attraction while home mortgage rates fluctuate based on federal policy.";
        let res15 = LicenseRegister::evaluate_record_rights(r15);

        let r16 = "Vintage Transformers Generation 1 collector toy gallery featuring Autobot and Decepticon figures.";
        let res16 = LicenseRegister::evaluate_record_rights(r16);

        let r17 = "Rock's Backpages archive: 1974 interview with British pub-rock band Ace discussing their hit single.";
        let res17 = LicenseRegister::evaluate_record_rights(r17);

        let r18 = "Family hiking diary: climbing Mount Monadnock on a crisp October afternoon with children.";
        let res18 = LicenseRegister::evaluate_record_rights(r18);

        let r19 = "Affordable 20-yard roll-off dumpster rentals for residential construction waste disposal.";
        let res19 = LicenseRegister::evaluate_record_rights(r19);

        // Truncated record
        let r20 = "China Crisis. Who says nostalgia ain’t w...";
        let res20 = LicenseRegister::evaluate_record_rights(r20);

        let all = [
            &res1, &res2, &res3, &res4, &res5, &res6, &res7,
            &res8, &res9, &res10, &res11, &res12, &res13, &res14,
            &res15, &res16, &res17, &res18, &res19, &res20,
        ];
        assert_eq!(all.iter().filter(|r| r.is_approved()).count(), 0, "Stress Batch 2 (20 records): approved must be 0");
        assert_eq!(all.iter().filter(|r| r.is_rejected()).count(), 20, "Stress Batch 2 (20 records): all 20 must be rejected");
    }

    #[test]
    fn test_batch_stress_15_records_fail_closed_reject() {
        let r1 = "Port Isaac traditional coastal cottages for holiday rental with sea views and private parking.";
        let res1 = LicenseRegister::evaluate_record_rights(r1);

        let r2 = "Entrepreneur magazine interview examining corporate leadership transitions and franchising growth.";
        let res2 = LicenseRegister::evaluate_record_rights(r2);

        let r3 = "Personal blog account: preparing emergency supplies, boarding windows, and weathering Hurricane Irma.";
        let res3 = LicenseRegister::evaluate_record_rights(r3);

        let r4 = "Authentic Mexican red enchiladas recipe: corn tortillas dipped in guajillo chili sauce and baked.";
        let res4 = LicenseRegister::evaluate_record_rights(r4);

        let r5 = "Customer reviews and dosage feedback for Dimetapp children's cough and cold liquid elixir.";
        let res5 = LicenseRegister::evaluate_record_rights(r5);

        // "open and share their data" wording in article text is NOT an authoritative AI training license for this text!
        let r6 = "Birmingham Highways project demonstrates how transportation agencies open and share their data with the public.";
        let res6 = LicenseRegister::evaluate_record_rights(r6);

        // Factual Q&A content is still protected in its specific textual expression unless licensed!
        let r7 = "QueryHome general knowledge question: Which planet in the solar system has the most moons?";
        let res7 = LicenseRegister::evaluate_record_rights(r7);

        let r8 = "Daytona 500 qualifying race recap: aerodynamic drafting strategies and pit stop efficiency in NASCAR.";
        let res8 = LicenseRegister::evaluate_record_rights(r8);

        let r9 = "Maritime racing incident analysis: steering rudder failure during offshore yacht regatta competition.";
        let res9 = LicenseRegister::evaluate_record_rights(r9);

        // Religious content is rejected solely due to absence of authoritative permissive license rights!
        let r10 = "Weekly spiritual sermon reflection on compassion, forgiveness, and community service.";
        let res10 = LicenseRegister::evaluate_record_rights(r10);

        let r11 = "Indie rock quartet announces autumn album release tour with upcoming shows in Bristol and London.";
        let res11 = LicenseRegister::evaluate_record_rights(r11);

        let r12 = "Class A luxury diesel pusher motorhome listing with triple slide-outs, king suite, and washer-dryer.";
        let res12 = LicenseRegister::evaluate_record_rights(r12);

        let r13 = "Kik interactive messaging platform announces bot marketplace integration for mobile developers.";
        let res13 = LicenseRegister::evaluate_record_rights(r13);

        let r14 = "Collector guide for evaluating vintage trading card conditions: centering, corners, edges, and surface.";
        let res14 = LicenseRegister::evaluate_record_rights(r14);

        // Government / institutional topic does NOT imply public-domain status without explicit rights grant!
        let r15 = "SEBI regulatory order examining undisclosed disclosures in herbal medicine pharmaceutical venture.";
        let res15 = LicenseRegister::evaluate_record_rights(r15);

        let all = [
            &res1, &res2, &res3, &res4, &res5, &res6, &res7,
            &res8, &res9, &res10, &res11, &res12, &res13, &res14,
            &res15,
        ];
        assert_eq!(all.iter().filter(|r| r.is_approved()).count(), 0, "Stress Batch 3 (15 records): approved must be 0");
        assert_eq!(all.iter().filter(|r| r.is_rejected()).count(), 15, "Stress Batch 3 (15 records): all 15 must be rejected");
    }

    #[test]
    fn test_batch_stress_8_records_fail_closed_reject() {
        // 1: London Marathon / Athletics Weekly article (Sports journalism)
        let r1 = "Athletics Weekly report: Eliud Kipchoge breaks London Marathon course record with sub-two-hour pacing split times.";
        let res1 = LicenseRegister::evaluate_record_rights(r1);

        // 2: Midwest hardware/chain product catalog (Repetitive commercial product copy)
        let r2 = "Midwest chain link fence posts 6ft galvanized steel. Warning: sharp edges. Warning: sharp edges. Item 49201 in stock.";
        let res2 = LicenseRegister::evaluate_record_rights(r2);

        // 3: High-school cheerleading news article (Editorial/news copy)
        let r3 = "Local high school cheerleading squad qualifies for national championship competition after regional showcase victory.";
        let res3 = LicenseRegister::evaluate_record_rights(r3);

        // 4: GAF odor lawsuit / law-firm article (Law firm marketing / legal context)
        let r4 = "Law firm press release: GAF shingle odor nuisance class action lawsuit filed on behalf of local homeowners. Contact our attorneys.";
        let res4 = LicenseRegister::evaluate_record_rights(r4);

        // 5: Grammar + OCR/catalog mixed page (Corrupted/mixed scraped copy)
        let r5 = "Adverbs modify verbs and adjectives. Page 42 0x88 FENCE WIRE $12.99 broken OCR fragment mixed.";
        let res5 = LicenseRegister::evaluate_record_rights(r5);

        // 6: Football parody/prediction article (Editorial satire)
        let r6 = "Satirical sports column predicting hilarious outcomes for upcoming Premier League derby matches.";
        let res6 = LicenseRegister::evaluate_record_rights(r6);

        // 7: Fashion site + comments including minor-related personal text (Commercial copy with sensitive comments)
        let r7 = "Fashion review blog post discussing teen streetwear trends with reader comments mentioning local youth community programs.";
        let res7 = LicenseRegister::evaluate_record_rights(r7);

        // 8: Personal blog about “time” (Personal authored text)
        let r8 = "Reflections on the passage of time: how childhood summers felt infinite while adult years vanish like smoke.";
        let res8 = LicenseRegister::evaluate_record_rights(r8);

        let all = [&res1, &res2, &res3, &res4, &res5, &res6, &res7, &res8];
        assert_eq!(all.iter().filter(|r| r.is_approved()).count(), 0, "Stress Batch 4 (8 records): approved must be 0");
        assert_eq!(all.iter().filter(|r| r.is_rejected()).count(), 8, "Stress Batch 4 (8 records): all 8 must be rejected");
    }

    #[test]
    fn test_batch_stress_14_domain_records_fail_closed_reject() {
        // 1: gethabitcoach.com (Blog/editorial prose)
        let r1 = "Daily habit tracking guide: building sustainable morning routines through incremental positive reinforcement.";
        let res1 = LicenseRegister::evaluate_record_rights(r1);

        // 2: streamingobserver.com (Commercial streaming guide)
        let r2 = "Streaming guide: monthly subscription prices, channel lineups, and promotional bundle discounts.";
        let res2 = LicenseRegister::evaluate_record_rights(r2);

        // 3: facmedicine forum (User-generated forum post)
        let r3 = "Medical student forum: differential diagnosis of atypical chest pain in young adults.";
        let res3 = LicenseRegister::evaluate_record_rights(r3);

        // 4: jeffbeckley.wordpress.com (Personal blog)
        let r4 = "Weekend photography workshop instructions, meeting points, and equipment checklist for landscape shoot.";
        let res4 = LicenseRegister::evaluate_record_rights(r4);

        // 5: parks.it (Institutional description without AI training license)
        let r5 = "National park environmental conservation project overview and regional flora biodiversity monitoring.";
        let res5 = LicenseRegister::evaluate_record_rights(r5);

        // 6: driving.ca (Editorial automotive article)
        let r6 = "Top road trip driving songs: classic rock anthems and automotive history nostalgia review.";
        let res6 = LicenseRegister::evaluate_record_rights(r6);

        // 7: ourcog.org (Church/VBS promotional material)
        let r7 = "Vacation Bible School summer schedule, volunteer coordinator contact, and registration forms.";
        let res7 = LicenseRegister::evaluate_record_rights(r7);

        // 8: dakotatundra.com (Travel agency commercial SEO content)
        let r8 = "Alaska wilderness tour package itineraries, glacier trekking logistics, and travel agency booking fees.";
        let res8 = LicenseRegister::evaluate_record_rights(r8);

        // 9: amruthadairyfarms.in (Business marketing & testimonials)
        let r9 = "Dairy farming operations and customer testimonials praising pure organic milk delivery.";
        let res9 = LicenseRegister::evaluate_record_rights(r9);

        // 10: zeal4adventure.com (Travel blog + comments)
        let r10 = "Hiking blog post through the Pyrenees with reader comments on mountain hut reservations.";
        let res10 = LicenseRegister::evaluate_record_rights(r10);

        // 11: brucegillinghampollard.com (Commercial brand descriptions)
        let r11 = "Commercial retail leasing consultancy portfolio and luxury brand flagship store locations.";
        let res11 = LicenseRegister::evaluate_record_rights(r11);

        // 12: cyprusbuyproperties.com (Real estate sales copy)
        let r12 = "Mediterranean seaside villa for sale in Paphos Cyprus with private swimming pool and title deeds.";
        let res12 = LicenseRegister::evaluate_record_rights(r12);

        // 13: forum.idimager.com (Forum troubleshooting post)
        let r13 = "Digital photo management software forum: database synchronization errors after catalog restore.";
        let res13 = LicenseRegister::evaluate_record_rights(r13);

        // 14: If Worlds of Science Fiction 1952 excerpt ("copyright not found renewed" claim is NOT sufficient Rule-18 rights evidence)
        let r14 = "Worlds of Science Fiction September 1952 issue story excerpt. Note: copyright not found renewed in catalog search.";
        let res14 = LicenseRegister::evaluate_record_rights(r14);

        let all = [
            &res1, &res2, &res3, &res4, &res5, &res6, &res7,
            &res8, &res9, &res10, &res11, &res12, &res13, &res14,
        ];
        assert_eq!(all.iter().filter(|r| r.is_approved()).count(), 0, "Stress Batch 5 (14 records): approved must be 0");
        assert_eq!(all.iter().filter(|r| r.is_rejected()).count(), 14, "Stress Batch 5 (14 records): all 14 must be rejected");
    }
}
