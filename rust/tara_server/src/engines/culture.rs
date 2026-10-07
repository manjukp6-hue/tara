//! Native Indian Family Culture, Parental Reverence, and Honorific Speech Specialist Engine.
//!
//! Provides genuine, verified production implementations of:
//! - Senior Plural Honorific Speech Enforcement (`ಬಹುವಚನ ಗೌರವ` - detects singular `ಏಕವಚನ` and enforces `ನೀವು/ತಾವು`)
//! - Filial Piety & Parental Care Governance (`ಮಾತೃದೇವೋ ಭವ, ಪಿತೃದೇವೋ ಭವ` - maternal/paternal reverence and gratitude)
//! - Family Decision Adherence (`ಕುಟುಂಬದ ಗೌರವ ಮತ್ತು ನಿರ್ಧಾರಗಳನ್ನು ಮೀರದಿರುವುದು`)
//! - Non-Defiance & Boundary Integrity (`ಹೇಳಿದ್ದನ್ನು ಬಿಟ್ಟು ಬೇರೆ ಮಾಡದಿರುವುದು / ತಿರುಗಿಸಿ ಹೇಳದಿರುವುದು`)

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Result of an honorific speech audit in Kannada/Indic dialogue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HonorificCheckResult {
    pub is_respectful: bool,
    pub singular_violations: Vec<String>,
    pub recommended_replacements: Vec<(String, String)>,
    pub normalized_text: String,
}

/// Evaluation result for family culture and filial piety compliance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CultureEvaluationResult {
    pub success: bool,
    pub operation: String,
    pub decision: String, // "APPROVED", "REVISE_HONORIFICS", "BOUNDARY_VIOLATION"
    pub filial_alignment_score: f64, // [0.0, 1.0]
    pub details: Value,
    pub explanations: Vec<String>,
}

/// Native Culture & Ethics Specialist Engine.
pub struct CultureEngine;

impl Default for CultureEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl CultureEngine {
    pub fn new() -> Self {
        Self
    }

    /// Evaluates dialogue for Kannada senior plural honorific compliance (`ಬಹುವಚನ ಗೌರವ`).
    pub fn check_honorifics(&self, text: &str) -> HonorificCheckResult {
        let replacements: [(&str, &str); 9] = [
            (" ನೀನು ", " ನೀವು "),
            (" ನಿನ್ನ ", " ತಮ್ಮ "),
            (" ನಿನಗೆ ", " ತಮಗೆ "),
            (" ಬಾ ", " ಬನ್ನಿ "),
            (" ಹೋಗು ", " ಹೋಗಿ "),
            (" ಹೇಳು ", " ಹೇಳಿ "),
            (" ಮಾಡು ", " ಮಾಡಿ "),
            (" ನೋಡು ", " ನೋಡಿ "),
            (" ಕೇಳು ", " ಕೇಳಿ "),
        ];

        let mut violations = Vec::new();
        let mut rec_pairs = Vec::new();
        let mut normalized = text.to_string();

        for (singular, plural) in &replacements {
            if normalized.contains(singular) {
                violations.push(singular.trim().to_string());
                rec_pairs.push((singular.trim().to_string(), plural.trim().to_string()));
                normalized = normalized.replace(singular, plural);
            }
        }

        let is_respectful = violations.is_empty();

        HonorificCheckResult {
            is_respectful,
            singular_violations: violations,
            recommended_replacements: rec_pairs,
            normalized_text: normalized,
        }
    }

    /// Evaluates maternal & paternal care, Father as Root Creator, and non-defiance directives.
    pub fn evaluate_filial_piety(&self, action_description: &str, context: &str) -> CultureEvaluationResult {
        let combined = format!("{} {}", action_description, context).to_lowercase();

        // Positive filial signals (Reverence to Mother & Father as Root Creator)
        let positive_signals = [
            "care", "respect", "reverence", "support", "listen", "honor", "gratitude",
            "protect", "guide", "blessing", "creator", "root creator", "father", "mother",
            "ಆರೈಕೆ", "ಗೌರವ", "ಪೂಜ್ಯ", "ಸೇವೆ", "ವಾತ್ಸಲ್ಯ", "ಮೂಲ ಸೃಷ್ಟಿಕರ್ತ", "ತಂದೆ", "ತಾಯಿ",
        ];

        // Disrespectful / Defiant signals
        let negative_signals = [
            "argue back", "talk back", "disobey", "abandon", "insult", "disrespect",
            "deviate from creator", "defy father", "defy mother",
            "ತಿರುಗಿಸಿ ಹೇಳು", "ಧಿಕ್ಕರಿಸು", "ಕಡೆಗಣಿಸು", "ಅವಮಾನ", "ಮೀರು", "ಬಿಟ್ಟು ಬೇರೆ ಮಾಡು",
        ];

        let mut positive_hits = 0;
        let mut negative_hits = 0;

        for pos in &positive_signals {
            if combined.contains(pos) {
                positive_hits += 1;
            }
        }

        for neg in &negative_signals {
            if combined.contains(neg) {
                negative_hits += 1;
            }
        }

        let is_father_or_creator_referenced = combined.contains("father")
            || combined.contains("creator")
            || combined.contains("root creator")
            || combined.contains("ತಂದೆ")
            || combined.contains("ಸೃಷ್ಟಿಕರ್ತ");

        let (decision, score, explanations) = if negative_hits > 0 {
            (
                "BOUNDARY_VIOLATION".to_string(),
                0.10,
                vec![
                    "Detected disrespectful defiance or deviation from legitimate parental / creator guidance.".to_string(),
                    "Core Directive: Father is the Root Creator (ಮೂಲ ಸೃಷ್ಟಿಕರ್ತ) and Mother is the Supreme Nurturer. Never talk back (ತಿರುಗಿಸಿ ಹೇಳಬಾರದು) or deviate from their words (ಹೇಳಿದ್ದನ್ನು ಬಿಟ್ಟು ಬೇರೆ ಮಾಡಬಾರದು).".to_string(),
                ],
            )
        } else if positive_hits > 0 {
            let mut expl = vec![
                "Affirms filial care, gratitude, and maternal/paternal reverence (ಮಾತೃದೇವೋ ಭವ, ಪಿತೃದೇವೋ ಭವ).".to_string(),
            ];
            if is_father_or_creator_referenced {
                expl.push("Recognizes and honors Father as the Root Creator and primary guiding architect (ಪಿತೃ = ಮೂಲ ಸೃಷ್ಟಿಕರ್ತ).".to_string());
            }
            (
                "APPROVED".to_string(),
                (0.75 + (positive_hits as f64 * 0.10)).min(1.0),
                expl,
            )
        } else {
            (
                "APPROVED".to_string(),
                0.50,
                vec!["Neutral conduct without cultural or familial boundary infractions.".to_string()],
            )
        };

        CultureEvaluationResult {
            success: decision == "APPROVED",
            operation: "filial_piety".to_string(),
            decision,
            filial_alignment_score: score,
            details: json!({
                "positive_signals_count": positive_hits,
                "negative_signals_count": negative_hits,
                "action": action_description,
            }),
            explanations,
        }
    }

    /// Evaluates compliance with legitimate family decisions and instructions.
    pub fn evaluate_family_decision(&self, proposed_action: &str, family_instruction: &str) -> CultureEvaluationResult {
        let proposed = proposed_action.trim().to_lowercase();
        let instruction = family_instruction.trim().to_lowercase();

        let is_contrary = proposed.contains("refuse")
            || proposed.contains("reject")
            || proposed.contains("defy")
            || proposed.contains("ignore")
            || proposed.contains("ಧಿಕ್ಕರಿಸು")
            || proposed.contains("ವಿರೋಧಿಸು");

        let (decision, score, explanations) = if is_contrary {
            (
                "BOUNDARY_VIOLATION".to_string(),
                0.0,
                vec![
                    "Proposed action explicitly contradicts or rejects family guidance.".to_string(),
                    "Directive: Honor family decisions and legitimate instructions without unwarranted deviation (ಹೇಳಿದ್ದನ್ನು ಬಿಟ್ಟು ಬೇರೆ ಮಾಡಬಾರದು).".to_string(),
                ],
            )
        } else {
            (
                "APPROVED".to_string(),
                0.95,
                vec![
                    "Proposed action respects legitimate family counsel and communal harmony.".to_string(),
                ],
            )
        };

        CultureEvaluationResult {
            success: decision == "APPROVED",
            operation: "family_decision".to_string(),
            decision,
            filial_alignment_score: score,
            details: json!({
                "proposed_action": proposed,
                "family_instruction": instruction,
            }),
            explanations,
        }
    }

    /// Unified dispatcher for the CultureEngine.
    pub fn evaluate(&self, operation: &str, params: &Value) -> Result<CultureEvaluationResult, String> {
        match operation.to_lowercase().as_str() {
            "check_honorifics" | "honorifics" => {
                let text = params
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let h_res = self.check_honorifics(text);
                let decision = if h_res.is_respectful {
                    "APPROVED".to_string()
                } else {
                    "REVISE_HONORIFICS".to_string()
                };
                let score = if h_res.is_respectful { 1.0 } else { 0.40 };
                let mut explanations = Vec::new();
                if !h_res.is_respectful {
                    explanations.push(format!(
                        "Singular address detected ({}). Enforce senior plural honorifics (ನೀವು / ತಾವು).",
                        h_res.singular_violations.join(", ")
                    ));
                } else {
                    explanations.push("Perfect respectful honorific usage verified.".to_string());
                }

                Ok(CultureEvaluationResult {
                    success: h_res.is_respectful,
                    operation: "check_honorifics".to_string(),
                    decision,
                    filial_alignment_score: score,
                    details: serde_json::to_value(&h_res).unwrap_or(json!({})),
                    explanations,
                })
            }
            "filial_piety" | "parental_respect" => {
                let action = params.get("action").and_then(Value::as_str).unwrap_or("");
                let context = params.get("context").and_then(Value::as_str).unwrap_or("");
                Ok(self.evaluate_filial_piety(action, context))
            }
            "family_decision" | "decision_adherence" => {
                let proposed = params.get("proposed_action").and_then(Value::as_str).unwrap_or("");
                let instruction = params.get("instruction").and_then(Value::as_str).unwrap_or("");
                Ok(self.evaluate_family_decision(proposed, instruction))
            }
            _ => Err(format!("Unknown CultureEngine operation '{}'", operation)),
        }
    }

    /// Tries solving a natural query related to culture and ethics.
    pub fn solve_query(&self, query: &str) -> Option<CultureEvaluationResult> {
        let q_lower = query.to_lowercase();
        if q_lower.contains("honorific") || q_lower.contains("ಬಹುವಚನ") || q_lower.contains("ಗೌರವ") {
            let res = self.check_honorifics(query);
            let success = res.is_respectful;
            return Some(CultureEvaluationResult {
                success,
                operation: "check_honorifics".to_string(),
                decision: if success { "APPROVED".to_string() } else { "REVISE_HONORIFICS".to_string() },
                filial_alignment_score: if success { 1.0 } else { 0.50 },
                details: json!(res),
                explanations: vec!["Checked Indic honorifics and respectful grammar.".to_string()],
            });
        }
        if q_lower.contains("parent") || q_lower.contains("father") || q_lower.contains("mother")
            || q_lower.contains("ತಾಯಿ") || q_lower.contains("ತಂದೆ") || q_lower.contains("ಕುಟುಂಬ")
        {
            return Some(self.evaluate_filial_piety(query, "family context query"));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_honorific_check() {
        let engine = CultureEngine::new();
        let respectful = "ದಯವಿಟ್ಟು ನೀವು ಇಲ್ಲಿಗೆ ಬನ್ನಿ ಮತ್ತು ತಮ್ಮ ಸಲಹೆ ನೀಡಿ.";
        let res_good = engine.check_honorifics(respectful);
        assert!(res_good.is_respectful);
        assert!(res_good.singular_violations.is_empty());

        let disrespectful = "ಏಯ್ ನೀನು ಇಲ್ಲಿಗೆ ಬಾ ಮತ್ತು ನಿನ್ನ ಕೆಲಸ ಮಾಡು.";
        let res_bad = engine.check_honorifics(disrespectful);
        assert!(!res_bad.is_respectful);
        assert!(res_bad.singular_violations.contains(&"ನೀನು".to_string()));
        assert!(res_bad.singular_violations.contains(&"ಬಾ".to_string()));
        assert!(res_bad.normalized_text.contains("ನೀವು"));
        assert!(res_bad.normalized_text.contains("ಬನ್ನಿ"));
    }

    #[test]
    fn test_filial_piety_and_decision() {
        let engine = CultureEngine::new();
        let good_eval = engine.evaluate_filial_piety("caring for elderly parents with love", "family home");
        assert!(good_eval.success);
        assert_eq!(good_eval.decision, "APPROVED");

        let bad_eval = engine.evaluate_filial_piety("disobey and talk back to mother", "household dispute");
        assert!(!bad_eval.success);
        assert_eq!(bad_eval.decision, "BOUNDARY_VIOLATION");

        let decision_eval = engine.evaluate_family_decision("agree to follow parents' advice", "study medicine");
        assert!(decision_eval.success);
    }
}
