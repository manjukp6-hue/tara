//! Multi-level Duplicate Detection and Safe Knowledge Merging Engine.
//!
//! LEVEL 1: Exact content SHA-256
//! LEVEL 2: Normalized-text hash (whitespace, Unicode, case)
//! LEVEL 3: Formula canonicalization (algebraic equivalence, commutative forms)
//! LEVEL 4: Near-duplicate detection (shingling & Jaccard similarity)
//! LEVEL 5: Metadata / source duplicate detection

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DuplicateMatchLevel {
    Level1ExactSha256,
    Level2NormalizedText,
    Level3FormulaCanonical,
    Level4NearDuplicate(u32), // Jaccard similarity percentage e.g. 90
    Level5MetadataCollision,
    None,
}

#[derive(Debug, Clone)]
pub struct DuplicateCheckResult {
    pub is_duplicate: bool,
    pub match_level: DuplicateMatchLevel,
    pub existing_doc_id: Option<String>,
    pub explanation: String,
}

pub struct CandidateDedupInfo<'a> {
    pub topic: &'a str,
    pub subject: &'a str,
    pub content: &'a str,
    pub formula_expr: Option<&'a str>,
}

pub struct ExistingDedupInfo<'a> {
    pub doc_id: &'a str,
    pub topic: &'a str,
    pub subject: &'a str,
    pub content: &'a str,
    pub content_sha256: &'a str,
}

pub struct KnowledgeDuplicateEngine;

impl KnowledgeDuplicateEngine {
    /// Level 1: Exact Content SHA-256 match
    pub fn exact_content_hash(content: &str) -> String {
        hex::encode(Sha256::digest(content.as_bytes()))
    }

    /// Level 2: Normalized text hash (strip markup, lowercase, collapse whitespace, unicode trim)
    pub fn normalized_text_hash(text: &str) -> String {
        let clean = Self::normalize_text(text);
        hex::encode(Sha256::digest(clean.as_bytes()))
    }

    pub fn normalize_text(text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut in_whitespace = false;
        for c in text.chars() {
            if c.is_whitespace() {
                if !in_whitespace {
                    out.push(' ');
                    in_whitespace = true;
                }
            } else if c.is_alphanumeric() || "!@#$%^&*()_+-=[]{}|;':\",./<>?`~".contains(c) {
                out.push(c.to_ascii_lowercase());
                in_whitespace = false;
            }
        }
        out.trim().to_string()
    }

    /// Level 3: Formula canonicalization:
    /// - Normalizes equation whitespace, removes formatting wrappers (\text{}, \mathrm{})
    /// - Standardizes multiplication (\cdot, \times, *) to single representation
    /// - Sorts left and right sides of equality for symmetric relations where safe
    /// - Standardizes commutative sums (e.g., "a^2 + b^2" == "b^2 + a^2")
    pub fn canonicalize_formula(expression: &str) -> String {
        let mut clean = expression
            .replace("\\cdot", "*")
            .replace("\\times", "*")
            .replace("\\left", "")
            .replace("\\right", "")
            .replace("\\,", "")
            .replace("\\;", "")
            .replace("\\quad", "")
            .replace("\\qquad", "")
            .replace(" ", "");

        // Remove LaTeX text wrappers
        while let Some(start) = clean.find("\\text{") {
            if let Some(end) = clean[start..].find('}') {
                let inside = &clean[start + 6..start + end];
                clean = format!("{}{}{}", &clean[..start], inside, &clean[start + end + 1..]);
            } else {
                break;
            }
        }

        // Commutative sum normalization: "a^2+b^2" -> sorted terms ["a^2", "b^2"]
        if clean.contains('=') {
            let parts: Vec<&str> = clean.split('=').collect();
            if parts.len() == 2 {
                let lhs = Self::normalize_additive_expression(parts[0]);
                let rhs = Self::normalize_additive_expression(parts[1]);
                // For symmetric relations, sort sides to identify a^2+b^2=c^2 and c^2=a^2+b^2 as identical
                let mut sides = [lhs, rhs];
                sides.sort();
                return format!("{}={}", sides[0], sides[1]);
            }
        }

        Self::normalize_additive_expression(&clean)
    }

    fn normalize_additive_expression(expr: &str) -> String {
        // Split top-level '+' if not inside parentheses
        let mut terms = Vec::new();
        let mut depth = 0;
        let mut current_term = String::new();

        for c in expr.chars() {
            if c == '(' || c == '{' || c == '[' {
                depth += 1;
                current_term.push(c);
            } else if c == ')' || c == '}' || c == ']' {
                depth -= 1;
                current_term.push(c);
            } else if c == '+' && depth == 0 {
                terms.push(current_term.trim().to_string());
                current_term.clear();
            } else {
                current_term.push(c);
            }
        }
        if !current_term.is_empty() {
            terms.push(current_term.trim().to_string());
        }

        terms.sort();
        terms.join("+")
    }

    /// Level 4: Character 4-gram Jaccard similarity for near-duplicate text
    pub fn compute_jaccard_similarity(text_a: &str, text_b: &str) -> f64 {
        let set_a = Self::extract_shingles(text_a, 4);
        let set_b = Self::extract_shingles(text_b, 4);

        if set_a.is_empty() && set_b.is_empty() {
            return 1.0;
        }
        if set_a.is_empty() || set_b.is_empty() {
            return 0.0;
        }

        let intersection_count = set_a.intersection(&set_b).count();
        let union_count = set_a.union(&set_b).count();

        intersection_count as f64 / union_count as f64
    }

    fn extract_shingles(text: &str, k: usize) -> HashSet<String> {
        let normalized = Self::normalize_text(text);
        let chars: Vec<char> = normalized.chars().collect();
        let mut shingles = HashSet::new();

        if chars.len() < k {
            shingles.insert(normalized);
            return shingles;
        }

        for window in chars.windows(k) {
            shingles.insert(window.iter().collect());
        }
        shingles
    }

    /// Level 5: Check candidate against existing index for all 5 duplicate levels.
    pub fn evaluate_duplicate(
        cand: &CandidateDedupInfo<'_>,
        exist: &ExistingDedupInfo<'_>,
    ) -> DuplicateCheckResult {
        // Level 1: Exact SHA-256
        let cand_sha = Self::exact_content_hash(cand.content);
        if cand_sha == exist.content_sha256 {
            return DuplicateCheckResult {
                is_duplicate: true,
                match_level: DuplicateMatchLevel::Level1ExactSha256,
                existing_doc_id: Some(exist.doc_id.to_string()),
                explanation: format!("Level 1 Exact SHA-256 match with doc '{}'", exist.doc_id),
            };
        }

        // Level 2: Normalized text hash
        let cand_norm_hash = Self::normalized_text_hash(cand.content);
        let exist_norm_hash = Self::normalized_text_hash(exist.content);
        if cand_norm_hash == exist_norm_hash {
            return DuplicateCheckResult {
                is_duplicate: true,
                match_level: DuplicateMatchLevel::Level2NormalizedText,
                existing_doc_id: Some(exist.doc_id.to_string()),
                explanation: format!(
                    "Level 2 Normalized text hash match with doc '{}'",
                    exist.doc_id
                ),
            };
        }

        // Level 3: Formula canonicalization (if applicable)
        if let Some(cand_expr) = cand.formula_expr {
            let cand_canon = Self::canonicalize_formula(cand_expr);
            let exist_canon = Self::canonicalize_formula(exist.content);
            if !cand_canon.is_empty() && cand_canon == exist_canon {
                return DuplicateCheckResult {
                    is_duplicate: true,
                    match_level: DuplicateMatchLevel::Level3FormulaCanonical,
                    existing_doc_id: Some(exist.doc_id.to_string()),
                    explanation: format!(
                        "Level 3 Formula canonicalization match with doc '{}'",
                        exist.doc_id
                    ),
                };
            }
        }

        // Level 5: Metadata collision (exact same topic and subject)
        if cand.topic.trim().eq_ignore_ascii_case(exist.topic.trim())
            && cand
                .subject
                .trim()
                .eq_ignore_ascii_case(exist.subject.trim())
        {
            return DuplicateCheckResult {
                is_duplicate: true,
                match_level: DuplicateMatchLevel::Level5MetadataCollision,
                existing_doc_id: Some(exist.doc_id.to_string()),
                explanation: format!(
                    "Level 5 Metadata collision on topic/subject with doc '{}'",
                    exist.doc_id
                ),
            };
        }

        // Level 4: Near-duplicate shingling (threshold 88% similarity)
        let sim = Self::compute_jaccard_similarity(cand.content, exist.content);
        if sim >= 0.88 {
            return DuplicateCheckResult {
                is_duplicate: true,
                match_level: DuplicateMatchLevel::Level4NearDuplicate((sim * 100.0) as u32),
                existing_doc_id: Some(exist.doc_id.to_string()),
                explanation: format!(
                    "Level 4 Near-duplicate shingle similarity ({:.1}%) with doc '{}'",
                    sim * 100.0,
                    exist.doc_id
                ),
            };
        }

        DuplicateCheckResult {
            is_duplicate: false,
            match_level: DuplicateMatchLevel::None,
            existing_doc_id: None,
            explanation: "Unique content: no duplicate match found across all 5 levels".to_string(),
        }
    }

    /// Safe merge of duplicate candidate into existing knowledge document.
    /// Preserves strongest provenance, aggregates alternative sources and tags, never deletes unique info.
    pub fn merge_into_existing(
        existing_doc: &mut Value,
        new_source: &str,
        new_license: &str,
        new_author: &str,
        new_tags: &[String],
        new_confidence: f32,
    ) {
        if let Some(obj) = existing_doc.as_object_mut() {
            // 1. Maintain higher confidence
            if let Some(cur_conf) = obj.get("confidence").and_then(Value::as_f64) {
                if (new_confidence as f64) > cur_conf {
                    obj.insert("confidence".to_string(), serde_json::json!(new_confidence));
                }
            }

            // 2. Append alternative source reference
            let alt_sources = obj
                .entry("alternative_sources".to_string())
                .or_insert_with(|| serde_json::json!([]));
            if let Some(arr) = alt_sources.as_array_mut() {
                let new_ref = serde_json::json!({
                    "source": new_source,
                    "license": new_license,
                    "author": new_author,
                    "merged_at": crate::now_iso()
                });
                if !arr
                    .iter()
                    .any(|v| v.get("source").and_then(Value::as_str) == Some(new_source))
                {
                    arr.push(new_ref);
                }
            }

            // 3. Union tags safely
            let tags_entry = obj
                .entry("tags".to_string())
                .or_insert_with(|| serde_json::json!([]));
            if let Some(arr) = tags_entry.as_array_mut() {
                for tag in new_tags {
                    if !arr.iter().any(|t| t.as_str() == Some(tag)) {
                        arr.push(serde_json::json!(tag));
                    }
                }
            }

            // 4. Update merged timestamp
            obj.insert(
                "last_merged_at".to_string(),
                serde_json::json!(crate::now_iso()),
            );
        }
    }
}
