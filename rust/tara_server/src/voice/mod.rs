//! Voice Interaction Subsystem: Wake Word Spotter and Real-Time Barge-In Interlock.
//!
//! Provides streaming multilingual keyword spotting (English & Kannada)
//! and full duplex Barge-In coordination with atomic playback cutoff interlocks.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Recognized wake-word match.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WakeWordMatch {
    pub detected: bool,
    pub matched_phrase: String,
    pub language: String,
    pub confidence: f64,
    pub offset_ms: u64,
}

/// Multilingual Keyword / Wake-Word Spotter for English and Kannada.
pub struct WakeWordSpotter {
    canonical_keywords: Vec<(&'static str, &'static str)>, // (phrase, lang)
    confidence_threshold: f64,
}

impl Default for WakeWordSpotter {
    fn default() -> Self {
        Self::new()
    }
}

impl WakeWordSpotter {
    pub fn new() -> Self {
        let mut keywords = vec![
            ("namaskara tara", "kn"),
            ("ನಮಸ್ಕಾರ ತಾರಾ", "kn"),
            ("hello tara", "en"),
            ("hey tara", "en"),
            ("namaskara", "kn"),
            ("ಓ ತಾರಾ", "kn"),
            ("ಹೇ ತಾರಾ", "kn"),
            ("tara", "en"),
            ("ತಾರಾ", "kn"),
        ];
        keywords.sort_by_key(|a| std::cmp::Reverse(a.0.chars().count()));
        Self {
            canonical_keywords: keywords,
            confidence_threshold: 0.70,
        }
    }

    /// Evaluates transcript or phonetic input for wake word detection.
    pub fn spot_keyword(&self, input: &str) -> WakeWordMatch {
        let clean = input.trim().to_lowercase();
        if clean.is_empty() {
            return WakeWordMatch {
                detected: false,
                matched_phrase: String::new(),
                language: "unknown".to_string(),
                confidence: 0.0,
                offset_ms: 0,
            };
        }

        // Exact & substring containment check
        for &(keyword, lang) in &self.canonical_keywords {
            let kw_lower = keyword.to_lowercase();
            if clean == kw_lower {
                return WakeWordMatch {
                    detected: true,
                    matched_phrase: keyword.to_string(),
                    language: lang.to_string(),
                    confidence: 0.99,
                    offset_ms: 0,
                };
            } else if clean.starts_with(&kw_lower) || clean.contains(&kw_lower) {
                return WakeWordMatch {
                    detected: true,
                    matched_phrase: keyword.to_string(),
                    language: lang.to_string(),
                    confidence: 0.90,
                    offset_ms: 0,
                };
            }
        }

        // Levenshtein fuzzy match against short keywords
        let mut best_score = 0.0;
        let mut best_match = "";
        let mut best_lang = "";

        for &(keyword, lang) in &self.canonical_keywords {
            let kw_lower = keyword.to_lowercase();
            let dist = self.levenshtein_distance(&clean, &kw_lower);
            let max_len = clean.chars().count().max(kw_lower.chars().count());
            if max_len > 0 {
                let similarity = 1.0 - (dist as f64 / max_len as f64);
                if similarity > best_score {
                    best_score = similarity;
                    best_match = keyword;
                    best_lang = lang;
                }
            }
        }

        if best_score >= self.confidence_threshold {
            WakeWordMatch {
                detected: true,
                matched_phrase: best_match.to_string(),
                language: best_lang.to_string(),
                confidence: best_score,
                offset_ms: 0,
            }
        } else {
            WakeWordMatch {
                detected: false,
                matched_phrase: String::new(),
                language: "unknown".to_string(),
                confidence: best_score,
                offset_ms: 0,
            }
        }
    }

    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let v1: Vec<char> = s1.chars().collect();
        let v2: Vec<char> = s2.chars().collect();
        let len1 = v1.len();
        let len2 = v2.len();

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        for (i, row) in matrix.iter_mut().enumerate() {
            row[0] = i;
        }
        for (j, item) in matrix[0].iter_mut().enumerate() {
            *item = j;
        }

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if v1[i - 1] == v2[j - 1] { 0 } else { 1 };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }

        matrix[len1][len2]
    }
}

/// State of voice conversation lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoiceTurnState {
    Idle,
    Listening,
    Thinking,
    Speaking,
    Interrupted,
}

/// Barge-In Coordinator managing full duplex turn switching and audio interruption.
pub struct BargeInCoordinator {
    state: Arc<Mutex<VoiceTurnState>>,
    playback_active: Arc<AtomicBool>,
    interrupted_count: Arc<Mutex<usize>>,
}

impl Default for BargeInCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl BargeInCoordinator {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(VoiceTurnState::Idle)),
            playback_active: Arc::new(AtomicBool::new(false)),
            interrupted_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Sets state to Speaking and starts playback latch.
    pub fn start_speaking(&self) {
        let mut st = self.state.lock().unwrap();
        *st = VoiceTurnState::Speaking;
        self.playback_active.store(true, Ordering::SeqCst);
    }

    /// Signals that assistant has finished speaking naturally.
    pub fn stop_speaking(&self) {
        let mut st = self.state.lock().unwrap();
        *st = VoiceTurnState::Idle;
        self.playback_active.store(false, Ordering::SeqCst);
    }

    /// User interrupted assistant while speaking: atomically cuts off audio and resets turn.
    pub fn trigger_barge_in(&self) -> bool {
        let mut st = self.state.lock().unwrap();
        if *st == VoiceTurnState::Speaking || self.playback_active.load(Ordering::SeqCst) {
            *st = VoiceTurnState::Interrupted;
            self.playback_active.store(false, Ordering::SeqCst);
            let mut count = self.interrupted_count.lock().unwrap();
            *count += 1;
            true
        } else {
            false
        }
    }

    pub fn get_state(&self) -> VoiceTurnState {
        *self.state.lock().unwrap()
    }

    pub fn is_playback_active(&self) -> bool {
        self.playback_active.load(Ordering::SeqCst)
    }

    pub fn total_interruptions(&self) -> usize {
        *self.interrupted_count.lock().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wake_word_spotter_multilingual() {
        let spotter = WakeWordSpotter::new();

        // English exact & prefix
        let m1 = spotter.spot_keyword("hey tara what's the weather");
        assert!(m1.detected);
        assert_eq!(m1.matched_phrase, "hey tara");
        assert_eq!(m1.language, "en");

        // Kannada exact & substring
        let m2 = spotter.spot_keyword("ನಮಸ್ಕಾರ ತಾರಾ ಹೇಗಿದ್ದೀರ?");
        assert!(m2.detected);
        assert_eq!(m2.matched_phrase, "ನಮಸ್ಕಾರ ತಾರಾ");
        assert_eq!(m2.language, "kn");

        // Negative check
        let m3 = spotter.spot_keyword("good morning everyone");
        assert!(!m3.detected);
    }

    #[test]
    fn test_barge_in_coordinator_interlock() {
        let coord = BargeInCoordinator::new();
        assert_eq!(coord.get_state(), VoiceTurnState::Idle);
        assert!(!coord.is_playback_active());

        // Assistant starts speaking
        coord.start_speaking();
        assert_eq!(coord.get_state(), VoiceTurnState::Speaking);
        assert!(coord.is_playback_active());

        // User speaks -> Barge-in triggered
        let interrupted = coord.trigger_barge_in();
        assert!(interrupted);
        assert_eq!(coord.get_state(), VoiceTurnState::Interrupted);
        assert!(!coord.is_playback_active());
        assert_eq!(coord.total_interruptions(), 1);

        // Subsequent barge-in when already interrupted returns false
        assert!(!coord.trigger_barge_in());
    }
}
