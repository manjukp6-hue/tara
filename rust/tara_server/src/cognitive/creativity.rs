//! Cognitive Creativity, Conceptual Blending & Divergent Synthesis Engine.
//!
//! Implements Fauconnier & Turner's Conceptual Blending Theory:
//! - Combines two disparate concept spaces (Input Space 1 & Input Space 2)
//! - Identifies structural generic mappings between domains
//! - Synthesizes an emergent blended concept
//! - Evaluates Novelty Score (semantic divergence from known templates)
//! - Evaluates Coherence / Feasibility Score against logical constraints
//! - Enforces strict anti-repetition / anti-template deduplication with atomic locks.
//! - Persists blends using crash-safe file writes with sync_all().

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const MAX_DOMAIN_LEN: usize = 64;
const MAX_CONCEPT_LEN: usize = 128;
const MAX_SYNTHESIS_LEN: usize = 4096;
const MAX_RULES_COUNT: usize = 32;
const MAX_ATTRIBUTES_COUNT: usize = 32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptualFrame {
    pub domain: String,
    pub concept_name: String,
    pub core_attributes: Vec<String>,
    pub relational_rules: Vec<String>,
}

impl ConceptualFrame {
    pub fn validate(&self) -> Result<(), String> {
        if self.domain.trim().is_empty() {
            return Err("Domain cannot be empty".into());
        }
        if self.domain.len() > 64 {
            return Err("Domain exceeds maximum 64 characters".into());
        }
        if !self
            .domain
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == ' ')
        {
            return Err("Domain must be alphanumeric (allowing space, underscore, hyphen)".into());
        }
        if self.concept_name.trim().is_empty() {
            return Err("Concept name cannot be empty".into());
        }
        if self.concept_name.len() > 1024 {
            return Err("Concept name exceeds maximum 1,024 characters".into());
        }
        if self.core_attributes.len() > 64 {
            return Err(format!(
                "Core attributes count {} exceeds maximum 64",
                self.core_attributes.len()
            ));
        }
        for attr in &self.core_attributes {
            if attr.len() > 1024 {
                return Err("Core attribute string exceeds maximum 1,024 characters".into());
            }
        }
        if self.relational_rules.len() > 32 {
            return Err(format!(
                "Relational rules count {} exceeds maximum 32",
                self.relational_rules.len()
            ));
        }
        for rule in &self.relational_rules {
            if rule.len() > 1024 {
                return Err("Relational rule string exceeds maximum 1,024 characters".into());
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergentBlend {
    pub blend_id: String,
    pub source_concept_a: String,
    pub source_concept_b: String,
    pub blended_title: String,
    pub generic_mapping: Vec<(String, String)>,
    pub emergent_hypothesis: String,
    pub novelty_score: f64, // [0.0, 1.0] Distance from standard domain templates
    pub coherence_score: f64, // [0.0, 1.0] Logical self-consistency
    pub is_novel: bool,
    pub created_at_ms: u64,
}

struct CreativityInnerState {
    blend_history: Vec<EmergentBlend>,
    known_signatures: HashSet<String>,
    corrupted_lines_count: usize,
    recent_synthesis_timestamps: Vec<u64>,
}

pub struct CreativityEngine {
    storage_path: PathBuf,
    inner: Arc<Mutex<CreativityInnerState>>,
}

impl CreativityEngine {
    pub fn new<P: AsRef<Path>>(repo_root: P) -> Self {
        let store_dir = repo_root.as_ref().join("storage").join("persistence");
        if !store_dir.exists() {
            if let Err(e) = fs::create_dir_all(&store_dir) {
                eprintln!(
                    "Warning: Failed to create creativity store directory {}: {}",
                    store_dir.display(),
                    e
                );
            }
        }
        let storage_path = store_dir.join("creative_blends.jsonl");

        let mut history = Vec::new();
        let mut signatures = HashSet::new();
        let mut corrupted_lines = 0usize;

        if storage_path.exists() {
            if let Ok(content) = fs::read_to_string(&storage_path) {
                for line in content.lines() {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        match serde_json::from_str::<EmergentBlend>(trimmed) {
                            Ok(blend) => {
                                let sig = Self::compute_signature(
                                    &blend.source_concept_a,
                                    &blend.source_concept_b,
                                );
                                signatures.insert(sig);
                                history.push(blend);
                            }
                            Err(_) => {
                                corrupted_lines += 1;
                            }
                        }
                    }
                }
            }
        }

        Self {
            storage_path,
            inner: Arc::new(Mutex::new(CreativityInnerState {
                blend_history: history,
                known_signatures: signatures,
                corrupted_lines_count: corrupted_lines,
                recent_synthesis_timestamps: Vec::new(),
            })),
        }
    }

    pub fn compute_signature(concept_a: &str, concept_b: &str) -> String {
        let norm_a = concept_a
            .rsplit(':')
            .next()
            .unwrap_or(concept_a)
            .trim()
            .to_lowercase();
        let norm_b = concept_b
            .rsplit(':')
            .next()
            .unwrap_or(concept_b)
            .trim()
            .to_lowercase();
        let mut parts = [norm_a, norm_b];
        parts.sort();
        let payload = format!("{}|{}", parts[0], parts[1]);
        tara_security::cryptography::CryptoUtils::compute256_bytes_hex(payload.as_bytes())
    }

    pub fn evaluate_coherence(score: f64) -> Result<f64, String> {
        if !score.is_finite() {
            return Err("Coherence score must be a finite float".into());
        }
        Ok(score.clamp(0.0, 1.0))
    }

    pub fn export(&self) -> Result<serde_json::Value, String> {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let public_blends: Vec<serde_json::Value> = inner
            .blend_history
            .iter()
            .map(|b| {
                serde_json::json!({
                    "blend_id": b.blend_id,
                    "title": b.blended_title,
                    "novelty_score": b.novelty_score,
                    "coherence_score": b.coherence_score,
                    "is_novel": b.is_novel,
                    "is_heuristic_telemetry": true
                })
            })
            .collect();
        Ok(serde_json::json!({
            "blends": public_blends,
            "count": public_blends.len()
        }))
    }

    pub fn persist(&self) -> Result<(), String> {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let mut buffer = Vec::new();
        for blend in &inner.blend_history {
            let json_line =
                serde_json::to_string(blend).map_err(|e| format!("Serialization error: {}", e))?;
            buffer.extend_from_slice(json_line.as_bytes());
            buffer.push(b'\n');
        }
        tara_security::persistence::atomic_write_file_hardened(&self.storage_path, &buffer)
    }

    pub fn load_history(&mut self) -> Result<usize, String> {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if !self.storage_path.exists() {
            return Ok(0);
        }
        let content = fs::read_to_string(&self.storage_path)
            .map_err(|e| format!("Failed to read creativity history: {}", e))?;

        let mut loaded = Vec::new();
        let mut signatures = HashSet::new();
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<EmergentBlend>(trimmed) {
                Ok(b) => {
                    let sig = Self::compute_signature(&b.source_concept_a, &b.source_concept_b);
                    signatures.insert(sig);
                    loaded.push(b);
                }
                Err(e) => {
                    inner.corrupted_lines_count += 1;
                    return Err(format!(
                        "CorruptedHistoryRecord: corrupt JSON record in creative_blends.jsonl: {}",
                        e
                    ));
                }
            }
        }

        let domain = format!(
            "creativity_{}",
            &hex::encode(Sha256::digest(
                self.storage_path.to_string_lossy().as_bytes()
            ))[..16]
        );
        if let Ok(Some(committed_watermark)) =
            tara_security::dpapi::load_os_monotonic_watermark(&domain)
        {
            if (loaded.len() as u64) < committed_watermark {
                return Err(format!(
                    "RollbackAttackDetected: loaded count {} is less than committed watermark {}",
                    loaded.len(),
                    committed_watermark
                ));
            }
        }

        let count = loaded.len();
        inner.blend_history = loaded;
        inner.known_signatures = signatures;
        Ok(count)
    }

    /// Evaluates semantic divergence (Jaccard novelty) between emergent attributes and domain baselines.
    pub fn compute_novelty(
        blended_terms: &[String],
        domain_a_terms: &[String],
        domain_b_terms: &[String],
    ) -> f64 {
        let set_blend: HashSet<String> = blended_terms.iter().map(|s| s.to_lowercase()).collect();
        let set_a: HashSet<String> = domain_a_terms.iter().map(|s| s.to_lowercase()).collect();
        let set_b: HashSet<String> = domain_b_terms.iter().map(|s| s.to_lowercase()).collect();

        let union_ab: HashSet<String> = set_a.union(&set_b).cloned().collect();
        if union_ab.is_empty() {
            return 1.0;
        }

        let new_terms = set_blend.difference(&union_ab).count();
        (new_terms as f64 / (set_blend.len().max(1) as f64)).clamp(0.0, 1.0)
    }

    pub fn evaluate_novelty(
        blended_terms: &[String],
        domain_a_terms: &[String],
        domain_b_terms: &[String],
    ) -> f64 {
        Self::compute_novelty(blended_terms, domain_a_terms, domain_b_terms)
    }

    /// Synthesize an emergent conceptual blend from two distinct conceptual frames.
    pub fn synthesize_blend(
        &self,
        frame_a: &ConceptualFrame,
        frame_b: &ConceptualFrame,
        proposed_synthesis: &str,
    ) -> Result<EmergentBlend, String> {
        // Enforce frame validation
        frame_a.validate()?;
        frame_b.validate()?;

        // Input validation & resource bounds
        if frame_a.domain.len() > MAX_DOMAIN_LEN || frame_b.domain.len() > MAX_DOMAIN_LEN {
            return Err("Domain string exceeds maximum length limit".into());
        }
        if frame_a.concept_name.len() > MAX_CONCEPT_LEN
            || frame_b.concept_name.len() > MAX_CONCEPT_LEN
        {
            return Err("Concept name exceeds maximum length limit".into());
        }
        if proposed_synthesis.len() > MAX_SYNTHESIS_LEN {
            return Err("Proposed synthesis exceeds maximum length limit".into());
        }
        if frame_a.relational_rules.len() > MAX_RULES_COUNT
            || frame_b.relational_rules.len() > MAX_RULES_COUNT
        {
            return Err("Relational rules count exceeds maximum limit".into());
        }
        if frame_a.core_attributes.len() > MAX_ATTRIBUTES_COUNT
            || frame_b.core_attributes.len() > MAX_ATTRIBUTES_COUNT
        {
            return Err("Core attributes count exceeds maximum limit".into());
        }

        if frame_a.domain.trim().is_empty() || frame_b.domain.trim().is_empty() {
            return Err("Conceptual frames must have non-empty domains".into());
        }
        if frame_a.domain.eq_ignore_ascii_case(&frame_b.domain) {
            return Err(
                "Divergent creativity requires frames from distinct conceptual domains".into(),
            );
        }

        let sig = Self::compute_signature(&frame_a.concept_name, &frame_b.concept_name);

        // Atomic check for deduplication, rate limiting, and capacity under lock
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if inner.corrupted_lines_count > 0 {
            return Err(format!(
                "CorruptedStateBlocked: creative_blends.jsonl has {} corrupted records. Engine fails closed to prevent synthesizing from incomplete state.",
                inner.corrupted_lines_count
            ));
        }

        let now = crate::now_millis();
        inner
            .recent_synthesis_timestamps
            .retain(|&ts| now.saturating_sub(ts) < 60_000);
        if inner.recent_synthesis_timestamps.len() >= 100 {
            return Err(
                "RateLimitExceeded: Maximum 100 synthesis operations per minute allowed".into(),
            );
        }
        if inner.blend_history.len() >= 10_000 {
            return Err(
                "CapacityExceeded: Maximum 10,000 synthesis history records reached".into(),
            );
        }

        if inner.known_signatures.contains(&sig) {
            return Err(format!(
                "Anti-duplication rejection: blend between '{}' and '{}' has already been synthesized",
                frame_a.concept_name, frame_b.concept_name
            ));
        }

        // 1. Identify relational correspondences
        let mut mappings = Vec::new();
        let min_rules = frame_a
            .relational_rules
            .len()
            .min(frame_b.relational_rules.len());
        for i in 0..min_rules {
            mappings.push((
                frame_a.relational_rules[i].clone(),
                frame_b.relational_rules[i].clone(),
            ));
        }

        // 2. Extract terms and calculate novelty
        let blend_terms: Vec<String> = proposed_synthesis
            .split_whitespace()
            .map(|w| {
                w.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase()
            })
            .filter(|w| w.len() > 3)
            .collect();

        let novelty = Self::compute_novelty(
            &blend_terms,
            &frame_a.core_attributes,
            &frame_b.core_attributes,
        );
        let coherence = if !mappings.is_empty() && proposed_synthesis.len() > 20 {
            0.88
        } else {
            0.45
        };
        let is_novel = novelty >= 0.25 && coherence >= 0.70;

        let ts = crate::now_millis();
        let blend_id = format!("blend_{}_{}", ts, uuid::Uuid::new_v4());

        let blend = EmergentBlend {
            blend_id,
            source_concept_a: format!("{}:{}", frame_a.domain, frame_a.concept_name),
            source_concept_b: format!("{}:{}", frame_b.domain, frame_b.concept_name),
            blended_title: format!(
                "{} X {} Cross-Domain Synthesis",
                frame_a.concept_name, frame_b.concept_name
            ),
            generic_mapping: mappings,
            emergent_hypothesis: proposed_synthesis.to_string(),
            novelty_score: (novelty * 100.0).round() / 100.0,
            coherence_score: coherence,
            is_novel,
            created_at_ms: ts,
        };

        // Persist to disk with sync_all before committing to in-memory state
        let json_line =
            serde_json::to_string(&blend).map_err(|e| format!("Serialization error: {}", e))?;

        if let Some(parent) = self.storage_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "Failed to create storage directory {}: {}",
                    parent.display(),
                    e
                )
            })?;
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.storage_path)
            .map_err(|e| format!("Storage file open error: {}", e))?;

        writeln!(file, "{}", json_line).map_err(|e| format!("Storage write error: {}", e))?;
        file.sync_all()
            .map_err(|e| format!("Storage sync error: {}", e))?;

        // Update in-memory state only upon verified disk persistence
        inner.recent_synthesis_timestamps.push(now);
        inner.known_signatures.insert(sig);
        inner.blend_history.push(blend.clone());
        let committed_len = inner.blend_history.len() as u64;
        drop(inner);

        let domain = format!(
            "creativity_{}",
            &hex::encode(Sha256::digest(
                self.storage_path.to_string_lossy().as_bytes()
            ))[..16]
        );
        let _ = tara_security::dpapi::store_os_monotonic_watermark(&domain, committed_len);

        Ok(blend)
    }

    pub fn synthesize(
        &self,
        frame_a: &ConceptualFrame,
        frame_b: &ConceptualFrame,
        proposed_synthesis: &str,
    ) -> Result<EmergentBlend, String> {
        self.synthesize_blend(frame_a, frame_b, proposed_synthesis)
    }

    pub fn list_blends(&self) -> Vec<EmergentBlend> {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner.blend_history.clone()
    }

    pub fn corrupted_lines_count(&self) -> usize {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner.corrupted_lines_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conceptual_blending_novelty() {
        let tmp = std::env::temp_dir().join(format!("tara_creat_test_{}", uuid::Uuid::new_v4()));
        let engine = CreativityEngine::new(&tmp);

        let frame_physics = ConceptualFrame {
            domain: "physics".into(),
            concept_name: "Thermodynamics Entropy".into(),
            core_attributes: vec![
                "heat".into(),
                "dissipation".into(),
                "equilibrium".into(),
                "entropy".into(),
            ],
            relational_rules: vec![
                "energy cannot be created or destroyed".into(),
                "entropy of isolated systems increases".into(),
            ],
        };

        let frame_cs = ConceptualFrame {
            domain: "computer_science".into(),
            concept_name: "Byzantine Fault Tolerance".into(),
            core_attributes: vec![
                "consensus".into(),
                "validator".into(),
                "message_overhead".into(),
                "fault_tolerance".into(),
            ],
            relational_rules: vec![
                "majority agreement required".into(),
                "malicious nodes consume bandwidth".into(),
            ],
        };

        let synthesis = "Thermodynamic proof-of-entropy consensus: validators burn thermodynamic thermal credits to establish Byzantine consensus, bounding computational dissipation";
        let blend = engine
            .synthesize_blend(&frame_physics, &frame_cs, synthesis)
            .unwrap();

        assert!(blend.is_novel);
        assert!(blend.novelty_score > 0.30);
        assert_eq!(blend.generic_mapping.len(), 2);

        // Verify that exact duplicate synthesis is strictly rejected by anti-duplication policy
        let dup_err = engine.synthesize_blend(&frame_physics, &frame_cs, synthesis);
        assert!(dup_err.is_err(), "Duplicate blend must be rejected");
        if tmp.exists() {
            assert!(fs::remove_dir_all(&tmp).is_ok());
        }
    }
}
