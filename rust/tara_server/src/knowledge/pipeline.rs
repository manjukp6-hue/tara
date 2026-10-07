//! Mandatory Knowledge Ingestion and Autonomous Research Pipeline.
//!
//! Enforces the 13-stage ingestion sequence:
//! DOWNLOAD -> LICENSE VERIFY -> CONTENT VERIFY -> CLEAN -> NORMALIZE -> EXTRACT -> TAG -> DEDUPLICATE -> VALIDATE -> HASH -> PARTITION -> INDEX -> PERSIST
//!
//! Provides a dedicated bridge for TARA-derived autonomous research results.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

use super::dedup::{
    CandidateDedupInfo, DuplicateCheckResult, ExistingDedupInfo, KnowledgeDuplicateEngine,
};
use super::license::{LicenseVerificationEngine, ReusePermissionStatus};
use super::schema::{KnowledgeDocMeta, KnowledgeProvenance, KnowledgeSourceType};
use super::validation::{KnowledgeValidationEngine, ValidationEntryParams};
use super::GlobalKnowledgeBase;

#[derive(Debug, Clone)]
pub struct PipelineIngestOutcome {
    pub status: String,
    pub doc_id: Option<String>,
    pub file_path: Option<String>,
    pub action: String, // "INSERTED", "MERGED", "QUARANTINED"
    pub error: Option<String>,
}

pub struct PipelineIngestParams<'a> {
    pub topic: &'a str,
    pub subpartition: Option<&'a str>,
    pub subject: &'a str,
    pub raw_content: &'a str,
    pub source_uri: &'a str,
    pub declared_license: &'a str,
    pub declared_type: Option<&'a str>,
    pub author: &'a str,
    pub tags: &'a [String],
    pub confidence: f32,
    pub source_type: KnowledgeSourceType,
    pub originating_record_id: Option<&'a str>,
}

pub struct KnowledgeImportPipeline;

impl KnowledgeImportPipeline {
    /// Execute the complete 13-stage ingestion pipeline on a single knowledge candidate.
    pub fn ingest_record(
        kb: &GlobalKnowledgeBase,
        p: PipelineIngestParams<'_>,
    ) -> PipelineIngestOutcome {
        let topic = p.topic;
        let subpartition = p.subpartition;
        let subject = p.subject;
        let raw_content = p.raw_content;
        let source_uri = p.source_uri;
        let declared_license = p.declared_license;
        let declared_type = p.declared_type;
        let author = p.author;
        let tags = p.tags;
        let confidence = p.confidence;
        let source_type = p.source_type;
        let originating_record_id = p.originating_record_id;
        // Stage 1 & 2: License Verification
        let lic_report = LicenseVerificationEngine::verify_license(
            source_uri,
            declared_license,
            raw_content,
            declared_type,
            author,
        );

        if lic_report.reuse_status != ReusePermissionStatus::Permitted {
            let quarantine = KnowledgeValidationEngine::new(&format!("{}/quarantine", kb.base_dir));
            let payload = json!({
                "topic": topic,
                "subject": subject,
                "content": raw_content,
                "source": source_uri,
                "license": declared_license,
                "tags": tags,
                "confidence": confidence
            });
            let err_msg = lic_report
                .rejection_reason
                .unwrap_or_else(|| "License rejected".to_string());
            let _ = quarantine.quarantine_rejected_entry(
                subject,
                "LICENSE_VERIFICATION",
                std::slice::from_ref(&err_msg),
                source_uri,
                declared_license,
                &payload,
            );

            return PipelineIngestOutcome {
                status: "QUARANTINED".to_string(),
                doc_id: None,
                file_path: None,
                action: "QUARANTINED".to_string(),
                error: Some(err_msg),
            };
        }

        // Stage 3 & 4: Content Verify & Clean (strip markup, HTML entities)
        let cleaned_content = super::KnowledgeCleaner::clean(raw_content);
        if cleaned_content.trim().is_empty() {
            return PipelineIngestOutcome {
                status: "QUARANTINED".to_string(),
                doc_id: None,
                file_path: None,
                action: "QUARANTINED".to_string(),
                error: Some("Cleaned content is empty".to_string()),
            };
        }

        // Stage 5: Normalize text for comparisons
        let content_sha256 = hex::encode(Sha256::digest(cleaned_content.as_bytes()));

        // Stage 6 & 7: Extract terms and merge tags
        let mut final_tags = tags.to_vec();
        let topic_clean = topic.trim().to_lowercase().replace(' ', "_");
        if !final_tags.iter().any(|t| t == &topic_clean) {
            final_tags.push(topic_clean.clone());
        }
        if let Some(sub) = subpartition {
            let sub_clean = sub.trim().to_lowercase().replace(' ', "_");
            if !final_tags.iter().any(|t| t == &sub_clean) {
                final_tags.push(sub_clean);
            }
        }

        // Stage 8: Deduplicate against existing indexed records
        let doc_id = hex::encode(Sha256::digest(
            format!(
                "{}:{}:{}",
                topic_clean,
                subject.trim().to_lowercase(),
                source_uri
            )
            .as_bytes(),
        ))[..16]
            .to_string();

        let dup_eval = {
            let idx = kb.index.lock().unwrap();
            let mut result = DuplicateCheckResult {
                is_duplicate: false,
                match_level: super::dedup::DuplicateMatchLevel::None,
                existing_doc_id: None,
                explanation: String::new(),
            };

            // Fast Level 1 check via SHA index
            if let Some(existing_id) = idx.sha_index.get(&content_sha256) {
                result = DuplicateCheckResult {
                    is_duplicate: true,
                    match_level: super::dedup::DuplicateMatchLevel::Level1ExactSha256,
                    existing_doc_id: Some(existing_id.clone()),
                    explanation: format!("Level 1 exact SHA-256 match with doc '{}'", existing_id),
                };
            } else {
                // Check across cached documents
                for (cand_id, meta) in &idx.doc_cache {
                    let d = KnowledgeDuplicateEngine::evaluate_duplicate(
                        &CandidateDedupInfo {
                            topic: &topic_clean,
                            subject,
                            content: &cleaned_content,
                            formula_expr: None,
                        },
                        &ExistingDedupInfo {
                            doc_id: cand_id,
                            topic: &meta.category,
                            subject: &meta.subject,
                            content: &meta.content_preview,
                            content_sha256: &meta.content_sha256,
                        },
                    );
                    if d.is_duplicate {
                        result = d;
                        break;
                    }
                }
            }
            result
        };

        if dup_eval.is_duplicate {
            if let Some(existing_id) = dup_eval.existing_doc_id {
                // Merge into existing document without creating duplicate file
                if let Some(mut existing_val) = kb.get_knowledge(&existing_id) {
                    KnowledgeDuplicateEngine::merge_into_existing(
                        &mut existing_val,
                        source_uri,
                        declared_license,
                        author,
                        &final_tags,
                        confidence,
                    );

                    let existing_file_path = existing_val
                        .get("file_path")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    if !existing_file_path.is_empty() {
                        let p = Path::new(existing_file_path);
                        if let Some(parent) = p.parent() {
                            let _ = fs::create_dir_all(parent);
                        }
                        let _ = fs::write(
                            existing_file_path,
                            serde_json::to_string_pretty(&existing_val).unwrap_or_default(),
                        );
                    }

                    if let Ok(mut idx) = kb.index.lock() {
                        if let Some(cached_meta) = idx.doc_cache.get_mut(&existing_id) {
                            for tag in &final_tags {
                                if !cached_meta.tags.contains(tag) {
                                    cached_meta.tags.push(tag.clone());
                                }
                            }
                        }
                    }

                    return PipelineIngestOutcome {
                        status: "MERGED".to_string(),
                        doc_id: Some(existing_id),
                        file_path: if existing_file_path.is_empty() {
                            None
                        } else {
                            Some(existing_file_path.to_string())
                        },
                        action: "MERGED".to_string(),
                        error: None,
                    };
                }
            }
        }

        // Stage 9: Validate against all 11 integrity rules
        let validator = KnowledgeValidationEngine::new(&format!("{}/quarantine", kb.base_dir));
        let val_res = validator.validate_entry(&ValidationEntryParams {
            topic: &topic_clean,
            subject,
            content: &cleaned_content,
            confidence,
            source_uri,
            declared_license,
            content_sha256: &content_sha256,
            declared_type,
            author,
        });

        if !val_res.is_valid {
            let payload = json!({
                "topic": topic,
                "subject": subject,
                "content": cleaned_content,
                "source": source_uri,
                "license": declared_license,
                "tags": final_tags,
                "confidence": confidence
            });
            let _ = validator.quarantine_rejected_entry(
                subject,
                "VALIDATION_ENGINE",
                &val_res.errors,
                source_uri,
                declared_license,
                &payload,
            );

            return PipelineIngestOutcome {
                status: "QUARANTINED".to_string(),
                doc_id: None,
                file_path: None,
                action: "QUARANTINED".to_string(),
                error: Some(val_res.errors.join("; ")),
            };
        }

        // Stage 10, 11, 12, 13: HASH, PARTITION, INDEX, PERSIST
        let sub_part = subpartition.unwrap_or("general");
        let partition_rel = format!("partitions/{}/{}", topic_clean, sub_part);
        let shard = if doc_id.len() >= 2 {
            &doc_id[..2]
        } else {
            "part"
        };
        let partition_dir = Path::new(&kb.base_dir).join(&partition_rel).join(shard);
        if let Err(e) = fs::create_dir_all(&partition_dir) {
            return PipelineIngestOutcome {
                status: "ERROR".to_string(),
                doc_id: None,
                file_path: None,
                action: "ERROR".to_string(),
                error: Some(format!("Failed to create partition dir: {e}")),
            };
        }

        let target_file = partition_dir.join(format!("{}.json", doc_id));
        let target_file_str = target_file.to_string_lossy().to_string();

        let prov = KnowledgeProvenance {
            source_uri: source_uri.to_string(),
            license: declared_license.to_string(),
            author_or_curator: author.to_string(),
            content_sha256: content_sha256.clone(),
            imported_at: crate::now_iso(),
        };

        let entry = json!({
            "status": "SUCCESS",
            "id": doc_id,
            "topic": topic_clean,
            "subpartition": sub_part,
            "subject": subject,
            "tags": final_tags,
            "category": topic_clean,
            "partition": partition_rel,
            "file_path": target_file_str,
            "content": cleaned_content,
            "confidence": confidence,
            "source_type": source_type.to_string(),
            "originating_record_id": originating_record_id,
            "verification_status": "OPEN_SOURCE_VERIFIED",
            "source": source_uri,
            "learned_at": crate::now_iso(),
            "content_sha256": content_sha256,
            "provenance": prov,
        });

        if let Err(e) = fs::write(
            &target_file,
            serde_json::to_string_pretty(&entry).unwrap_or_default(),
        ) {
            return PipelineIngestOutcome {
                status: "ERROR".to_string(),
                doc_id: None,
                file_path: None,
                action: "ERROR".to_string(),
                error: Some(format!("Failed to write file: {e}")),
            };
        }

        // Update Index
        let meta = KnowledgeDocMeta {
            id: doc_id.clone(),
            topic: topic_clean.clone(),
            subject: subject.to_string(),
            tags: final_tags.clone(),
            category: topic_clean.clone(),
            partition: partition_rel,
            source: source_uri.to_string(),
            source_type,
            file_path: target_file_str.clone(),
            content_preview: cleaned_content.chars().take(250).collect(),
            content_sha256: content_sha256.clone(),
            confidence,
            originating_record_id: originating_record_id.map(String::from),
        };

        let full_text = format!(
            "{} {} {} {}",
            topic_clean,
            subject,
            final_tags.join(" "),
            cleaned_content
        );
        {
            let mut idx = kb.index.lock().unwrap();
            idx.index_document(meta, &full_text);
        }

        PipelineIngestOutcome {
            status: "SUCCESS".to_string(),
            doc_id: Some(doc_id),
            file_path: Some(target_file_str),
            action: "INSERTED".to_string(),
            error: None,
        }
    }

    /// Autonomous Research Integration:
    /// Ingest an approved ResearchRecord into the partitioned Knowledge Base as TARA_DERIVED knowledge.
    pub fn ingest_autonomous_research_record(
        kb: &GlobalKnowledgeBase,
        record: &crate::learning::ResearchRecord,
    ) -> Result<Value, String> {
        let topic = &record.goal.domain;
        let subject = format!("empirical_finding_{}", record.goal.goal_id);
        let subpartition = if topic.contains("token") || topic.contains("vocab") {
            "tokenization_algorithms"
        } else if topic.contains("quant") {
            "quantization"
        } else {
            "empirical_research"
        };

        let mut content = format!(
            "Autonomous Scientific Research Finding\nGoal: {}\nHypothesis: {}\nStatus: {}\nPosterior Confidence: {:.4}\nRationale: {}\nConclusion: {}\n\nEmpirical Metrics:\n",
            record.goal.query,
            record.hypotheses.first().map(|h| h.statement.as_str()).unwrap_or(""),
            record.hypothesis_evaluation.status,
            record.hypothesis_evaluation.posterior_confidence,
            record.hypothesis_evaluation.rationale,
            record.conclusion.summary
        );

        for m in &record.execution_result.measured_metrics {
            content.push_str(&format!(
                "- Metric: {}, Measured: {:.4} {}, Delta: {:.4}, p-value: {:.4e}, N: {}\n",
                m.metric_name,
                m.measured_value,
                m.unit,
                m.delta.unwrap_or(0.0),
                m.p_value.unwrap_or(1.0),
                m.sample_size
            ));
        }

        let source_uri = format!("tara://research/{}", record.record_id);
        let license = if record.record_license.is_empty() {
            "Apache-2.0"
        } else {
            &record.record_license
        };

        let author = format!(
            "TARA_AUTONOMOUS_RESEARCH_ENGINE [Record: {}]",
            record.record_id
        );
        let tags = vec![
            "autonomous_research".to_string(),
            topic.to_lowercase(),
            "tara_derived".to_string(),
            "empirical_verification".to_string(),
        ];

        let confidence = (record
            .hypothesis_evaluation
            .posterior_confidence
            .clamp(0.01, 0.999)) as f32;

        let outcome = Self::ingest_record(
            kb,
            PipelineIngestParams {
                topic,
                subpartition: Some(subpartition),
                subject: &subject,
                raw_content: &content,
                source_uri: &source_uri,
                declared_license: license,
                declared_type: Some("TARA_ORIGINAL_RESEARCH"),
                author: &author,
                tags: &tags,
                confidence,
                source_type: KnowledgeSourceType::TaraDerived,
                originating_record_id: Some(&record.record_id),
            },
        );

        if outcome.status == "SUCCESS" || outcome.status == "MERGED" {
            Ok(json!({
                "status": "SUCCESS",
                "doc_id": outcome.doc_id,
                "action": outcome.action,
                "file_path": outcome.file_path,
                "source_type": "TARA_DERIVED",
                "originating_record_id": record.record_id
            }))
        } else {
            Err(format!(
                "Autonomous research ingestion failed: {:?}",
                outcome.error
            ))
        }
    }
}
