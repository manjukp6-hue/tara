//! Canonical LanguageEngine for TARA AI.
//!
//! Provides production multilingual capabilities:
//! - Multi-script identification across 17+ languages (Kannada, English, Hindi, Tamil, Telugu,
//!   Malayalam, Bengali, Marathi, Gujarati, Punjabi, Urdu, Arabic, Hebrew, Chinese, Japanese,
//!   Korean, Russian, Greek)
//! - Unicode script boundary detection
//! - Kannada-English code-switching and Kanglish detection
//! - Technical terminology resolution (bridging to native MultilingualEngine)
//! - Linguistic text metrics and script distribution analysis.

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::OnceLock;
use tara_engine::computation::MultilingualEngine;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScriptType {
    Latin,
    Kannada,
    Devanagari,
    Tamil,
    Telugu,
    Malayalam,
    Bengali,
    Gujarati,
    Gurmukhi,
    Arabic,
    Hebrew,
    Cjk,
    HiraganaKatakana,
    Hangul,
    Cyrillic,
    Greek,
    CommonOrPunctuation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptDistribution {
    pub dominant_script: ScriptType,
    pub script_proportions: HashMap<String, f64>,
    pub total_script_chars: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageAnalysis {
    pub primary_language: String,
    pub language_name: String,
    pub confidence: f64,
    pub script: ScriptType,
    pub is_mixed_or_code_switched: bool,
    pub secondary_languages: Vec<String>,
    pub kanglish_detected: bool,
    pub kanglish_markers: Vec<String>,
    pub char_count: usize,
    pub word_count: usize,
}

const KANGLISH_MARKERS: &[&str] = &[
    "madbeda",
    "torisbedi",
    "kelbeku",
    "keli",
    "kelu",
    "mathadu",
    "mathadovaga",
    "kannadadalli",
    "kannada",
    "sarkari",
    "niyama",
    "niyamagalu",
    "palisi",
    "madi",
    "munche",
    "nanna",
    "nannu",
    "helidange",
    "beda",
    "bedve",
    "kodbarda",
    "illade",
    "badalavane",
    "mukhyavada",
    "yavaga",
    "alla",
    "hage",
    "hesaru",
    "madodu",
    "namaskara",
    "hegiddira",
    "chennagiddira",
    "dayavittu",
    "dhanyavada",
    "oota",
    "aayitha",
    "banni",
    "hogi",
    "baruthe",
    "gotthu",
    "gottilla",
    "beku",
];

static KANGLISH_REGEX: OnceLock<Regex> = OnceLock::new();

fn get_kanglish_regex() -> &'static Regex {
    KANGLISH_REGEX.get_or_init(|| {
        let pattern = format!(r"(?i)\b({})\b", KANGLISH_MARKERS.join("|"));
        Regex::new(&pattern).expect("valid kanglish regex")
    })
}

pub struct LanguageEngine;

impl Default for LanguageEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageEngine {
    pub fn new() -> Self {
        Self
    }

    /// Classify a single Unicode codepoint into its native writing system.
    pub fn classify_codepoint(c: char) -> ScriptType {
        let u = c as u32;
        match u {
            0x0041..=0x005A | 0x0061..=0x007A | 0x00C0..=0x024F => ScriptType::Latin,
            0x0C80..=0x0CFF => ScriptType::Kannada,
            0x0900..=0x097F | 0xA8E0..=0xA8FF => ScriptType::Devanagari,
            0x0B80..=0x0BFF => ScriptType::Tamil,
            0x0C00..=0x0C7F => ScriptType::Telugu,
            0x0D00..=0x0D7F => ScriptType::Malayalam,
            0x0980..=0x09FF => ScriptType::Bengali,
            0x0A80..=0x0AFF => ScriptType::Gujarati,
            0x0A00..=0x0A7F => ScriptType::Gurmukhi,
            0x0600..=0x06FF | 0x0750..=0x077F | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => {
                ScriptType::Arabic
            }
            0x0590..=0x05FF => ScriptType::Hebrew,
            0x4E00..=0x9FFF | 0x3400..=0x4DBF => ScriptType::Cjk,
            0x3040..=0x30FF => ScriptType::HiraganaKatakana,
            0xAC00..=0xD7AF | 0x1100..=0x11FF => ScriptType::Hangul,
            0x0400..=0x052F => ScriptType::Cyrillic,
            0x0370..=0x03FF | 0x1F00..=0x1FFF => ScriptType::Greek,
            _ => ScriptType::CommonOrPunctuation,
        }
    }

    /// Compute script frequency distribution across input text.
    pub fn analyze_script_distribution(text: &str) -> ScriptDistribution {
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut total_meaningful = 0;

        for c in text.chars() {
            let script = Self::classify_codepoint(c);
            if script != ScriptType::CommonOrPunctuation && !c.is_whitespace() {
                total_meaningful += 1;
                let key = format!("{:?}", script);
                *counts.entry(key).or_insert(0) += 1;
            }
        }

        let mut proportions = HashMap::new();
        let mut dominant_script = ScriptType::Latin;
        let mut max_count = 0;

        for (k, v) in &counts {
            let prop = if total_meaningful > 0 {
                *v as f64 / total_meaningful as f64
            } else {
                0.0
            };
            proportions.insert(k.clone(), prop);
            if *v > max_count {
                max_count = *v;
                dominant_script = match k.as_str() {
                    "Kannada" => ScriptType::Kannada,
                    "Devanagari" => ScriptType::Devanagari,
                    "Tamil" => ScriptType::Tamil,
                    "Telugu" => ScriptType::Telugu,
                    "Malayalam" => ScriptType::Malayalam,
                    "Bengali" => ScriptType::Bengali,
                    "Gujarati" => ScriptType::Gujarati,
                    "Gurmukhi" => ScriptType::Gurmukhi,
                    "Arabic" => ScriptType::Arabic,
                    "Hebrew" => ScriptType::Hebrew,
                    "Cjk" => ScriptType::Cjk,
                    "HiraganaKatakana" => ScriptType::HiraganaKatakana,
                    "Hangul" => ScriptType::Hangul,
                    "Cyrillic" => ScriptType::Cyrillic,
                    "Greek" => ScriptType::Greek,
                    _ => ScriptType::Latin,
                };
            }
        }

        ScriptDistribution {
            dominant_script,
            script_proportions: proportions,
            total_script_chars: total_meaningful,
        }
    }

    /// Perform full language identification and script analysis on instance.
    pub fn analyze(&self, text: &str) -> LanguageAnalysis {
        Self::analyze_static(text)
    }

    /// Lookup technical terminology translation.
    pub fn lookup_technical_term(&self, term: &str) -> Option<Value> {
        Self::resolve_terminology(term)
    }

    /// Perform full language identification and script analysis statically.
    pub fn analyze_static(text: &str) -> LanguageAnalysis {
        let cleaned = text.trim();
        let char_count = cleaned.chars().count();
        let words: Vec<&str> = cleaned.split_whitespace().collect();
        let word_count = words.len();

        if cleaned.is_empty() {
            return LanguageAnalysis {
                primary_language: "en".into(),
                language_name: "English".into(),
                confidence: 1.0,
                script: ScriptType::Latin,
                is_mixed_or_code_switched: false,
                secondary_languages: Vec::new(),
                kanglish_detected: false,
                kanglish_markers: Vec::new(),
                char_count: 0,
                word_count: 0,
            };
        }

        let script_dist = Self::analyze_script_distribution(cleaned);

        // Check for Kanglish markers in Latin text
        let mut kanglish_detected = false;
        let mut detected_markers = Vec::new();

        if script_dist.dominant_script == ScriptType::Latin
            || script_dist.script_proportions.contains_key("Latin")
        {
            let re = get_kanglish_regex();
            for mat in re.find_iter(cleaned) {
                kanglish_detected = true;
                let m = mat.as_str().to_lowercase();
                if !detected_markers.contains(&m) {
                    detected_markers.push(m);
                }
            }
        }

        // Determine primary language and name
        let (primary, name, conf) = match script_dist.dominant_script {
            ScriptType::Kannada => (
                "kn",
                "Kannada",
                *script_dist
                    .script_proportions
                    .get("Kannada")
                    .unwrap_or(&1.0),
            ),
            ScriptType::Devanagari => {
                // Disambiguate Hindi vs Marathi if markers present, default Hindi
                let is_marathi =
                    cleaned.contains("आहे") || cleaned.contains("नाही") || cleaned.contains("करणे");
                if is_marathi {
                    (
                        "mr",
                        "Marathi",
                        *script_dist
                            .script_proportions
                            .get("Devanagari")
                            .unwrap_or(&1.0),
                    )
                } else {
                    (
                        "hi",
                        "Hindi",
                        *script_dist
                            .script_proportions
                            .get("Devanagari")
                            .unwrap_or(&1.0),
                    )
                }
            }
            ScriptType::Tamil => (
                "ta",
                "Tamil",
                *script_dist.script_proportions.get("Tamil").unwrap_or(&1.0),
            ),
            ScriptType::Telugu => (
                "te",
                "Telugu",
                *script_dist.script_proportions.get("Telugu").unwrap_or(&1.0),
            ),
            ScriptType::Malayalam => (
                "ml",
                "Malayalam",
                *script_dist
                    .script_proportions
                    .get("Malayalam")
                    .unwrap_or(&1.0),
            ),
            ScriptType::Bengali => (
                "bn",
                "Bengali",
                *script_dist
                    .script_proportions
                    .get("Bengali")
                    .unwrap_or(&1.0),
            ),
            ScriptType::Gujarati => (
                "gu",
                "Gujarati",
                *script_dist
                    .script_proportions
                    .get("Gujarati")
                    .unwrap_or(&1.0),
            ),
            ScriptType::Gurmukhi => (
                "pa",
                "Punjabi",
                *script_dist
                    .script_proportions
                    .get("Gurmukhi")
                    .unwrap_or(&1.0),
            ),
            ScriptType::Arabic => {
                let is_urdu =
                    cleaned.contains("ہے") || cleaned.contains("کی") || cleaned.contains("میں");
                if is_urdu {
                    (
                        "ur",
                        "Urdu",
                        *script_dist.script_proportions.get("Arabic").unwrap_or(&1.0),
                    )
                } else {
                    (
                        "ar",
                        "Arabic",
                        *script_dist.script_proportions.get("Arabic").unwrap_or(&1.0),
                    )
                }
            }
            ScriptType::Hebrew => (
                "he",
                "Hebrew",
                *script_dist.script_proportions.get("Hebrew").unwrap_or(&1.0),
            ),
            ScriptType::Cjk => (
                "zh",
                "Chinese",
                *script_dist.script_proportions.get("Cjk").unwrap_or(&1.0),
            ),
            ScriptType::HiraganaKatakana => (
                "ja",
                "Japanese",
                *script_dist
                    .script_proportions
                    .get("HiraganaKatakana")
                    .unwrap_or(&1.0),
            ),
            ScriptType::Hangul => (
                "ko",
                "Korean",
                *script_dist.script_proportions.get("Hangul").unwrap_or(&1.0),
            ),
            ScriptType::Cyrillic => (
                "ru",
                "Russian",
                *script_dist
                    .script_proportions
                    .get("Cyrillic")
                    .unwrap_or(&1.0),
            ),
            ScriptType::Greek => (
                "el",
                "Greek",
                *script_dist.script_proportions.get("Greek").unwrap_or(&1.0),
            ),
            ScriptType::Latin => {
                if kanglish_detected {
                    ("kn-Latn", "Kanglish (Phonetic Kannada)", 0.85)
                } else {
                    (
                        "en",
                        "English",
                        *script_dist.script_proportions.get("Latin").unwrap_or(&1.0),
                    )
                }
            }
            ScriptType::CommonOrPunctuation => ("en", "English", 1.0),
        };

        let mut secondary = Vec::new();
        let is_mixed = script_dist.script_proportions.len() > 1
            || (script_dist.dominant_script == ScriptType::Kannada
                && cleaned.chars().any(|c| c.is_ascii_alphabetic()));

        if is_mixed {
            for (k, v) in &script_dist.script_proportions {
                if *v >= 0.15 && format!("{:?}", script_dist.dominant_script) != *k {
                    secondary.push(k.clone());
                }
            }
        }

        LanguageAnalysis {
            primary_language: primary.into(),
            language_name: name.into(),
            confidence: conf.clamp(0.1, 1.0),
            script: script_dist.dominant_script,
            is_mixed_or_code_switched: is_mixed,
            secondary_languages: secondary,
            kanglish_detected,
            kanglish_markers: detected_markers,
            char_count,
            word_count,
        }
    }

    /// Resolve technical term across bilingual Kannada and English dictionary.
    pub fn resolve_terminology(term: &str) -> Option<Value> {
        let t = term.trim();
        if let Some(entry) = MultilingualEngine::lookup_english(t) {
            return Some(json!({
                "source": "english",
                "english": entry.english,
                "kannada": entry.kannada,
                "transliteration": entry.kannada_transliteration,
                "symbol": entry.symbol,
                "category": entry.category,
                "definition_en": entry.definition_en,
                "definition_kn": entry.definition_kn,
            }));
        }
        if let Some(entry) = MultilingualEngine::lookup_kannada(t) {
            return Some(json!({
                "source": "kannada",
                "english": entry.english,
                "kannada": entry.kannada,
                "transliteration": entry.kannada_transliteration,
                "symbol": entry.symbol,
                "category": entry.category,
                "definition_en": entry.definition_en,
                "definition_kn": entry.definition_kn,
            }));
        }
        None
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Backward-compatible LanguageDetector adapter for existing callers
// ──────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageDetectionResult {
    pub language: String,
    pub confidence: f64,
    pub detected_markers: usize,
}

pub struct LanguageDetector;

impl LanguageDetector {
    pub fn detect_language(text: &str) -> LanguageDetectionResult {
        let analysis = LanguageEngine::analyze_static(text);
        LanguageDetectionResult {
            language: analysis.primary_language,
            confidence: analysis.confidence,
            detected_markers: analysis.kanglish_markers.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multilingual_script_detection() {
        // Kannada
        let kn = LanguageEngine::analyze_static("ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?");
        assert_eq!(kn.primary_language, "kn");
        assert_eq!(kn.script, ScriptType::Kannada);

        // English
        let en = LanguageEngine::analyze_static("Calculate the determinant of a matrix.");
        assert_eq!(en.primary_language, "en");
        assert_eq!(en.script, ScriptType::Latin);

        // Hindi
        let hi = LanguageEngine::analyze_static("नमस्ते, आप कैसे हैं?");
        assert_eq!(hi.primary_language, "hi");
        assert_eq!(hi.script, ScriptType::Devanagari);

        // Tamil
        let ta = LanguageEngine::analyze_static("வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?");
        assert_eq!(ta.primary_language, "ta");

        // Telugu
        let te = LanguageEngine::analyze_static("నమస్కారం, మీరు ఎలా ఉన్నారు?");
        assert_eq!(te.primary_language, "te");

        // Russian (Cyrillic)
        let ru = LanguageEngine::analyze_static("Здравствуйте, как ваши дела?");
        assert_eq!(ru.primary_language, "ru");

        // Greek
        let el = LanguageEngine::analyze_static("Γειά σας, πώς είστε;");
        assert_eq!(el.primary_language, "el");

        // Chinese
        let zh = LanguageEngine::analyze_static("你好，今天天气怎么样？");
        assert_eq!(zh.primary_language, "zh");

        // Arabic
        let ar = LanguageEngine::analyze_static("مرحبا، كيف حالك؟");
        assert_eq!(ar.primary_language, "ar");
    }

    #[test]
    fn test_kanglish_detection() {
        let kang = LanguageEngine::analyze_static("nanna rule prakara kelbeku madbeda");
        assert_eq!(kang.primary_language, "kn-Latn");
        assert!(kang.kanglish_detected);
        assert!(kang.kanglish_markers.contains(&"kelbeku".to_string()));
        assert!(kang.kanglish_markers.contains(&"madbeda".to_string()));
    }

    #[test]
    fn test_terminology_lookup() {
        let term = LanguageEngine::resolve_terminology("multiplication").unwrap();
        assert_eq!(term["kannada"], "ಗುಣಾಕಾರ");

        let term_kn = LanguageEngine::resolve_terminology("ಸದಿಶ").unwrap();
        assert_eq!(term_kn["english"], "Vector");
    }
}
