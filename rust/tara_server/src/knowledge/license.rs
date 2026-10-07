//! Mandatory License and Reuse Verification Engine.
//!
//! Validates external material before entry into TARA's knowledge base.
//! Ensures no proprietary, restricted, or unverified content enters knowledge storage.

use super::schema::KnowledgeProvenance;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ArtifactType {
    PaperPublication,
    SoftwareLibrary,
    SpecificationStandard,
    TechnicalManual,
    OpenDataset,
    HistoricalTreatise,
    TaraOriginalResearch,
    EducationalReference,
    Unknown,
}

impl ArtifactType {
    pub fn parse(s: &str) -> Self {
        let u = s.trim().to_uppercase();
        if u.contains("PAPER") || u.contains("PUBLICATION") || u.contains("ARTICLE") {
            Self::PaperPublication
        } else if u.contains("SOFTWARE") || u.contains("LIBRARY") || u.contains("CODE") {
            Self::SoftwareLibrary
        } else if u.contains("STANDARD")
            || u.contains("SPEC")
            || u.contains("RFC")
            || u.contains("W3C")
            || u.contains("NIST")
        {
            Self::SpecificationStandard
        } else if u.contains("MANUAL") || u.contains("DOCUMENTATION") || u.contains("DOCS") {
            Self::TechnicalManual
        } else if u.contains("DATASET") || u.contains("CORPUS") {
            Self::OpenDataset
        } else if u.contains("HISTORICAL") || u.contains("TREATISE") || u.contains("SUTRA") {
            Self::HistoricalTreatise
        } else if u.contains("TARA") || u.contains("RESEARCH") || u.contains("INTERNAL") {
            Self::TaraOriginalResearch
        } else if u.contains("EDUCATION") || u.contains("REFERENCE") || u.contains("BOOK") {
            Self::EducationalReference
        } else {
            Self::Unknown
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReusePermissionStatus {
    Permitted,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseVerificationReport {
    pub source_uri: String,
    pub source_license: String,
    pub source_artifact_type: ArtifactType,
    pub reuse_status: ReusePermissionStatus,
    pub evidence_url: String,
    pub verification_timestamp: String,
    pub content_sha256: String,
    pub rejection_reason: Option<String>,
    pub provenance: Option<KnowledgeProvenance>,
}

pub struct LicenseVerificationEngine;

impl LicenseVerificationEngine {
    pub const PERMITTED_LICENSES: &'static [&'static str] = &[
        "MIT",
        "APACHE",
        "BSD",
        "CC0",
        "CC-BY",
        "PUBLIC DOMAIN",
        "OPENDATACOMMONS",
        "W3C",
        "IETF",
        "POSTGRESQL",
        "ISC",
        "OFL",
        "PYTHON",
        "PSF",
        "GPL",
        "PUBLIC ISO STANDARD",
        "ECMA",
        "OPEN-SOURCE",
        "OPEN ACCESS",
        "FREE SOFTWARE",
    ];

    pub const PROHIBITED_KEYWORDS: &'static [&'static str] = &[
        "PROPRIETARY",
        "ALL RIGHTS RESERVED",
        "COMMERCIAL RESTRICTION",
        "NON-COMMERCIAL ONLY",
        "NO REDISTRIBUTION",
        "CONFIDENTIAL",
        "UNKNOWN",
        "INTERNAL USE ONLY",
    ];

    /// Identify artifact type from source uri, subject, and topic hints.
    pub fn identify_artifact_type(
        uri: &str,
        topic: &str,
        declared_type: Option<&str>,
    ) -> ArtifactType {
        if let Some(dt) = declared_type {
            let parsed = ArtifactType::parse(dt);
            if parsed != ArtifactType::Unknown {
                return parsed;
            }
        }
        let uri_lc = uri.to_lowercase();
        let topic_lc = topic.to_lowercase();

        if uri_lc.contains("doi.org")
            || uri_lc.contains("arxiv.org")
            || uri_lc.contains("aclanthology.org")
        {
            ArtifactType::PaperPublication
        } else if uri_lc.contains("github.com")
            || uri_lc.contains("crates.io")
            || uri_lc.contains("pypi.org")
        {
            ArtifactType::SoftwareLibrary
        } else if uri_lc.contains("rfc-editor.org")
            || uri_lc.contains("w3.org")
            || uri_lc.contains("nist.gov")
            || topic_lc.contains("standard")
        {
            ArtifactType::SpecificationStandard
        } else if uri_lc.contains("man.") || uri_lc.contains("docs.") || topic_lc.contains("manual")
        {
            ArtifactType::TechnicalManual
        } else if uri_lc.starts_with("tara://research/") || uri_lc.starts_with("tara://experiment/")
        {
            ArtifactType::TaraOriginalResearch
        } else if topic_lc.contains("indian_mathematics") || topic_lc.contains("historical") {
            ArtifactType::HistoricalTreatise
        } else {
            ArtifactType::EducationalReference
        }
    }

    /// Verify whether material with the declared license and artifact type is legally permitted for storage and redistribution.
    pub fn verify_license(
        source_uri: &str,
        declared_license: &str,
        raw_content: &str,
        declared_type: Option<&str>,
        author_or_publisher: &str,
    ) -> LicenseVerificationReport {
        let trimmed_lic = declared_license.trim();
        let upper_lic = trimmed_lic.to_uppercase();
        let content_sha256 = hex::encode(Sha256::digest(raw_content.as_bytes()));
        let now = crate::now_iso();
        let artifact_type = Self::identify_artifact_type(source_uri, "", declared_type);

        // Check prohibited keywords
        for &proh in Self::PROHIBITED_KEYWORDS {
            if upper_lic.contains(proh) || upper_lic == proh {
                return LicenseVerificationReport {
                    source_uri: source_uri.to_string(),
                    source_license: trimmed_lic.to_string(),
                    source_artifact_type: artifact_type,
                    reuse_status: ReusePermissionStatus::Rejected,
                    evidence_url: source_uri.to_string(),
                    verification_timestamp: now,
                    content_sha256,
                    rejection_reason: Some(format!(
                        "License '{}' contains prohibited term '{}'",
                        trimmed_lic, proh
                    )),
                    provenance: None,
                };
            }
        }

        // Empty license check
        if trimmed_lic.is_empty() {
            return LicenseVerificationReport {
                source_uri: source_uri.to_string(),
                source_license: "UNDECLARED".to_string(),
                source_artifact_type: artifact_type,
                reuse_status: ReusePermissionStatus::Rejected,
                evidence_url: source_uri.to_string(),
                verification_timestamp: now,
                content_sha256,
                rejection_reason: Some("No license declared (unverified source)".to_string()),
                provenance: None,
            };
        }

        // Validate against permitted permissive list
        let mut is_permitted = false;
        for &perm in Self::PERMITTED_LICENSES {
            if upper_lic == perm || upper_lic.starts_with(perm) || upper_lic.contains(perm) {
                is_permitted = true;
                break;
            }
        }

        if !is_permitted {
            return LicenseVerificationReport {
                source_uri: source_uri.to_string(),
                source_license: trimmed_lic.to_string(),
                source_artifact_type: artifact_type,
                reuse_status: ReusePermissionStatus::Rejected,
                evidence_url: source_uri.to_string(),
                verification_timestamp: now,
                content_sha256,
                rejection_reason: Some(format!(
                    "License '{}' is not in permitted open-source / open-access catalog",
                    trimmed_lic
                )),
                provenance: None,
            };
        }

        // Successfully verified
        let prov = KnowledgeProvenance {
            source_uri: source_uri.to_string(),
            license: trimmed_lic.to_string(),
            author_or_curator: author_or_publisher.to_string(),
            content_sha256: content_sha256.clone(),
            imported_at: now.clone(),
        };

        LicenseVerificationReport {
            source_uri: source_uri.to_string(),
            source_license: trimmed_lic.to_string(),
            source_artifact_type: artifact_type,
            reuse_status: ReusePermissionStatus::Permitted,
            evidence_url: source_uri.to_string(),
            verification_timestamp: now,
            content_sha256,
            rejection_reason: None,
            provenance: Some(prov),
        }
    }
}
