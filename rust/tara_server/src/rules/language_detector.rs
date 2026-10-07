//! Language detection for TARA Natural-Language Rulebook.
//! Supports English, Kannada (script), Kanglish (phonetic Kannada in Latin script), and Hindi.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageDetectionResult {
    pub language: String,
    pub confidence: f64,
    pub detected_markers: usize,
}

const KANGLISH_PATTERNS: &[&str] = &[
    r"\bmadbeda\b",
    r"\btorisbedi\b",
    r"\bkelbeku\b",
    r"\bkeli\b",
    r"\bkelu\b",
    r"\bmathadu\b",
    r"\bmathadovaga\b",
    r"\bkannadadalli\b",
    r"\bkannada\b",
    r"\bsarkari\b",
    r"\bniyama\b",
    r"\bniyamagalu\b",
    r"\bpalisi\b",
    r"\bmadi\b",
    r"\bmunche\b",
    r"\bnanna\b",
    r"\bnannu\b",
    r"\bhelidange\b",
    r"\bbeda\b",
    r"\bbedve\b",
    r"\bkodbarda\b",
    r"\billade\b",
    r"\bbadalavane\b",
    r"\bmukhyavada\b",
    r"\byavaga\b",
    r"\balla\b",
    r"\bhage\b",
    r"\bhesaru\b",
    r"\bmadodu\b",
];

static KANGLISH_REGEXES: OnceLock<Vec<Regex>> = OnceLock::new();

fn get_kanglish_regexes() -> &'static [Regex] {
    KANGLISH_REGEXES.get_or_init(|| {
        KANGLISH_PATTERNS
            .iter()
            .filter_map(|pat| Regex::new(pat).ok())
            .collect()
    })
}

pub struct LanguageDetector;

impl LanguageDetector {
    pub fn detect_language(text: &str) -> LanguageDetectionResult {
        let cleaned = text.trim();
        if cleaned.is_empty() {
            return LanguageDetectionResult {
                language: "en".to_string(),
                confidence: 1.0,
                detected_markers: 0,
            };
        }

        // 1. Check Kannada script characters (U+0C80..=U+0CFF)
        let kn_chars = cleaned
            .chars()
            .filter(|&c| ('\u{0c80}'..='\u{0cff}').contains(&c))
            .count();
        if kn_chars > 0 {
            let words = cleaned.split_whitespace().count().max(1);
            let confidence = (kn_chars as f64 / words as f64).min(1.0);
            return LanguageDetectionResult {
                language: "kn".to_string(),
                confidence,
                detected_markers: kn_chars,
            };
        }

        // 2. Check Hindi / Devanagari script characters (U+0900..=U+097F)
        let hi_chars = cleaned
            .chars()
            .filter(|&c| ('\u{0900}'..='\u{097f}').contains(&c))
            .count();
        if hi_chars > 0 {
            let words = cleaned.split_whitespace().count().max(1);
            let confidence = (hi_chars as f64 / words as f64).min(1.0);
            return LanguageDetectionResult {
                language: "hi".to_string(),
                confidence,
                detected_markers: hi_chars,
            };
        }

        // 3. Check Kanglish patterns (phonetic Kannada written in Latin)
        let lower = cleaned.to_lowercase();
        let regexes = get_kanglish_regexes();
        let mut kanglish_hits = 0;
        for re in regexes {
            if re.is_match(&lower) {
                kanglish_hits += 1;
            }
        }

        if kanglish_hits >= 1 {
            let confidence = (0.5 + (kanglish_hits as f64 * 0.2)).min(1.0);
            return LanguageDetectionResult {
                language: "kanglish".to_string(),
                confidence,
                detected_markers: kanglish_hits,
            };
        }

        // 4. Default to English
        LanguageDetectionResult {
            language: "en".to_string(),
            confidence: 0.95,
            detected_markers: 0,
        }
    }
}
