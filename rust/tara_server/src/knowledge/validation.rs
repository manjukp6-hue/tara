//! Knowledge Validation and Quarantine Engine.
//!
//! Validates candidates across 11 integrity rules before persistence.
//! Rejects invalid candidates to a dedicated quarantine directory with detailed rejection audit records.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

use super::license::{LicenseVerificationEngine, ReusePermissionStatus};
use super::schema::QuarantineRecord;

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
}

pub struct KnowledgeValidationEngine {
    pub quarantine_dir: String,
}

pub struct ValidationEntryParams<'a> {
    pub topic: &'a str,
    pub subject: &'a str,
    pub content: &'a str,
    pub confidence: f32,
    pub source_uri: &'a str,
    pub declared_license: &'a str,
    pub content_sha256: &'a str,
    pub declared_type: Option<&'a str>,
    pub author: &'a str,
}

impl KnowledgeValidationEngine {
    pub fn new(quarantine_dir: &str) -> Self {
        let _ = fs::create_dir_all(quarantine_dir);
        Self {
            quarantine_dir: quarantine_dir.to_string(),
        }
    }

    /// Validate a candidate document prior to insertion into the Knowledge Base.
    pub fn validate_entry(&self, p: &ValidationEntryParams<'_>) -> ValidationResult {
        let mut errors = Vec::new();

        // 1. Content exists and has meaningful length
        if p.content.trim().is_empty() {
            errors.push("Content is empty".to_string());
        } else if p.content.trim().len() < 5 {
            errors.push("Content is too brief (< 5 characters)".to_string());
        }

        // 2. Subject exists
        if p.subject.trim().is_empty() {
            errors.push("Subject is empty".to_string());
        }

        // 3. Category/topic valid
        if p.topic.trim().is_empty() {
            errors.push("Topic/category is empty".to_string());
        }

        // 4. Source exists
        if p.source_uri.trim().is_empty() {
            errors.push("Source URI is empty (missing source attribution)".to_string());
        }

        // 5. License verification
        let lic_report = LicenseVerificationEngine::verify_license(
            p.source_uri,
            p.declared_license,
            p.content,
            p.declared_type,
            p.author,
        );
        if lic_report.reuse_status != ReusePermissionStatus::Permitted {
            errors.push(format!(
                "License rejected: {}",
                lic_report
                    .rejection_reason
                    .unwrap_or_else(|| "Unverified license".to_string())
            ));
        }

        // 6. SHA-256 integrity verification
        let computed_sha = hex::encode(Sha256::digest(p.content.as_bytes()));
        if !p.content_sha256.is_empty() && p.content_sha256 != computed_sha {
            errors.push(format!(
                "Content SHA-256 digest mismatch! Declared: {}, Computed: {}",
                p.content_sha256, computed_sha
            ));
        }

        // 7. Confidence bounds check
        if !p.confidence.is_finite() || p.confidence < 0.0 || p.confidence > 1.0 {
            errors.push(format!(
                "Confidence score {} is outside valid range [0.0, 1.0]",
                p.confidence
            ));
        }

        // 8. Author / Curator check
        if p.author.trim().is_empty() {
            errors.push("Author / Curator / Publisher attribution is missing".to_string());
        }

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
        }
    }

    /// Move rejected knowledge candidate to quarantine storage with full audit trail.
    pub fn quarantine_rejected_entry(
        &self,
        candidate_id: &str,
        rejection_stage: &str,
        reasons: &[String],
        source_uri: &str,
        license: &str,
        raw_payload: &Value,
    ) -> Result<String, String> {
        let ts = crate::now_iso();
        let q_id = format!(
            "quarantine_{}_{}",
            candidate_id,
            &hex::encode(Sha256::digest(ts.as_bytes()))[..8]
        );
        let record = QuarantineRecord {
            quarantine_id: q_id.clone(),
            candidate_id: candidate_id.to_string(),
            rejection_stage: rejection_stage.to_string(),
            reasons: reasons.to_vec(),
            source_uri: source_uri.to_string(),
            license: license.to_string(),
            content_snippet: raw_payload.to_string().chars().take(300).collect(),
            quarantined_at: ts,
        };

        let file_path = Path::new(&self.quarantine_dir).join(format!("{}.json", q_id));
        let serialized = serde_json::to_string_pretty(&record).map_err(|e| e.to_string())?;
        fs::write(&file_path, serialized).map_err(|e| e.to_string())?;

        Ok(file_path.to_string_lossy().to_string())
    }
}
