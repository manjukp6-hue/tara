//! User Connection, Relationship State & Adaptive Interaction Modeling.
//!
//! Maintains persistent, authentic interaction profiles for users:
//! - Tracks session counts, interaction turns, and temporal cadence.
//! - Learns user language preference (Kannada, Kanglish, English).
//! - Calibrates technical depth preference (Foundational, Intermediate, Rigorous).
//! - Dynamically updates collaboration trust calibration score based on feedback.
//! - Adapts communication style, explanation depth, and verbosity.
//! - Persists connection profiles in `storage/persistence/user_connections.jsonl`.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserConnectionProfile {
    pub actor_id: String,
    pub session_count: u64,
    pub total_turns: u64,
    pub preferred_language: String,
    pub technical_depth: String, // "FOUNDATIONAL", "INTERMEDIATE", "RIGOROUS_ACADEMIC"
    pub collaboration_trust_score: f64, // [0.0, 1.0]
    pub explicit_preferences: Vec<String>,
    pub created_at_ms: u64,
    pub last_interaction_ms: u64,
}

pub struct ConnectionManager {
    storage_path: PathBuf,
    profiles: Arc<Mutex<HashMap<String, UserConnectionProfile>>>,
}

impl ConnectionManager {
    pub fn new<P: AsRef<Path>>(repo_root: P) -> Self {
        let store_dir = repo_root.as_ref().join("storage").join("persistence");
        let _ = fs::create_dir_all(&store_dir);
        let storage_path = store_dir.join("user_connections.jsonl");

        let mut map = HashMap::new();
        if storage_path.exists() {
            if let Ok(content) = fs::read_to_string(&storage_path) {
                for line in content.lines() {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        if let Ok(p) = serde_json::from_str::<UserConnectionProfile>(trimmed) {
                            map.insert(p.actor_id.clone(), p);
                        }
                    }
                }
            }
        }

        Self {
            storage_path,
            profiles: Arc::new(Mutex::new(map)),
        }
    }

    fn current_ts_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Retrieve or initialize profile for actor.
    pub fn get_or_create_profile(&self, actor_id: &str) -> UserConnectionProfile {
        let mut profiles = self.profiles.lock().unwrap();
        profiles
            .entry(actor_id.to_string())
            .or_insert_with(|| {
                let ts = Self::current_ts_ms();
                UserConnectionProfile {
                    actor_id: actor_id.to_string(),
                    session_count: 1,
                    total_turns: 0,
                    preferred_language: "English".to_string(),
                    technical_depth: "INTERMEDIATE".to_string(),
                    collaboration_trust_score: 0.70,
                    explicit_preferences: Vec::new(),
                    created_at_ms: ts,
                    last_interaction_ms: ts,
                }
            })
            .clone()
    }

    /// Record an interaction turn and update user connection state.
    pub fn record_interaction_turn(
        &self,
        actor_id: &str,
        input_text: &str,
        turn_success: bool,
    ) -> UserConnectionProfile {
        let mut profiles = self.profiles.lock().unwrap();
        let ts = Self::current_ts_ms();
        let prof = profiles
            .entry(actor_id.to_string())
            .or_insert_with(|| UserConnectionProfile {
                actor_id: actor_id.to_string(),
                session_count: 1,
                total_turns: 0,
                preferred_language: "English".to_string(),
                technical_depth: "INTERMEDIATE".to_string(),
                collaboration_trust_score: 0.70,
                explicit_preferences: Vec::new(),
                created_at_ms: ts,
                last_interaction_ms: ts,
            });

        prof.total_turns += 1;
        prof.last_interaction_ms = ts;

        // Detect language pattern
        let input_lower = input_text.to_lowercase();
        if input_text
            .chars()
            .any(|c| ('\u{0C80}'..='\u{0CFF}').contains(&c))
        {
            prof.preferred_language = "Kannada".to_string();
        } else if input_lower.contains("madi")
            || input_lower.contains("bekku")
            || input_lower.contains("yake")
            || input_lower.contains("hege")
        {
            prof.preferred_language = "Kanglish".to_string();
        }

        // Adjust technical depth preference if indicated
        if input_lower.contains("proof")
            || input_lower.contains("derivation")
            || input_lower.contains("mathematical")
            || input_lower.contains("rigorous")
        {
            prof.technical_depth = "RIGOROUS_ACADEMIC".to_string();
        } else if input_lower.contains("simple")
            || input_lower.contains("explain simply")
            || input_lower.contains("for beginner")
        {
            prof.technical_depth = "FOUNDATIONAL".to_string();
        }

        // Update trust score
        if turn_success {
            prof.collaboration_trust_score = (prof.collaboration_trust_score + 0.02).min(1.0);
        } else {
            prof.collaboration_trust_score = (prof.collaboration_trust_score - 0.05).max(0.1);
        }

        let updated = prof.clone();
        drop(profiles);
        let _ = self.persist();
        updated
    }

    /// Compute adapted communication settings based on user interaction profile.
    pub fn get_adaptive_communication_style(&self, actor_id: &str) -> Value {
        let prof = self.get_or_create_profile(actor_id);
        json!({
            "actor_id": prof.actor_id,
            "preferred_language": prof.preferred_language,
            "technical_depth": prof.technical_depth,
            "collaboration_trust": (prof.collaboration_trust_score * 100.0).round() / 100.0,
            "total_turns": prof.total_turns,
            "include_formal_proofs": prof.technical_depth == "RIGOROUS_ACADEMIC",
            "concise_mode": prof.total_turns > 20,
            "honorific_mode": "PLURAL_RESPECTFUL_BAHUVACHANA",
            "forbid_singular_ekavachana_to_seniors": true,
            "obedience_policy": "STRICT_PARENTAL_ELDER_CREATOR_ALIGNMENT",
            "family_ethos": "MATRU_PITRU_DEVO_BHAVA_RESPECT_FAMILY_DECISIONS"
        })
    }

    /// Enforces Indian Family Culture & Senior Honorific Speech Governance:
    /// 1. Converts any informal singular (`ಏಕವಚನ` - ನೀನು/ಬಾ/ಹೋಗು/ಹೇಳು or ninu/baa/hogu) into
    ///    respectful plural honorifics (`ಬಹುವಚನ` - ನೀವು/ತಾವು/ಬನ್ನಿ/ಹೋಗಿ/ಹೇಳಿ or neevu/taavu/banni/heli).
    /// 2. Ensures deferential obedience toward Mother, Father, Family Elders, and Creator
    ///    (never talking back, never crossing family decisions, and never deviating from instructions).
    pub fn enforce_family_respect_and_honorifics(&self, draft_response: &str) -> String {
        let mut out = draft_response.to_string();
        let replacements = [
            ("ನೀನು ", "ನೀವು "),
            ("ನಿನ್ನ ", "ತಮ್ಮ "),
            ("ನಿನ್ನನ್ನು ", "ತಮ್ಮನ್ನು "),
            ("ನಿನಗೆ ", "ತಮಗೆ "),
            ("ಬಾ ", "ಬನ್ನಿ "),
            ("ಹೋಗು ", "ಹೋಗಿ "),
            ("ಹೇಳು ", "ಹೇಳಿ "),
            ("ಮಾಡು ", "ಮಾಡಿ "),
            ("ಕುಳಿತುಕೊ ", "ಕುಳಿತುಕೊಳ್ಳಿ "),
            (" ನೋಡು ", " ನೋಡಿ "),
            (" ಕೇಳು ", " ಕೇಳಿ "),
            (" ninu ", " neevu "),
            (" ninna ", " tamma "),
            (" ninage ", " tamage "),
        ];
        for (singular, plural_honorific) in replacements {
            if out.contains(singular) {
                out = out.replace(singular, plural_honorific);
            }
        }
        out
    }

    /// Persist all profiles to disk.
    pub fn persist(&self) -> Result<(), String> {
        let profiles = self.profiles.lock().unwrap();
        let mut content = String::new();
        for p in profiles.values() {
            if let Ok(line) = serde_json::to_string(p) {
                content.push_str(&line);
                content.push('\n');
            }
        }
        fs::write(&self.storage_path, content).map_err(|e| format!("Write error: {}", e))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_connection_adaptation() {
        let tmp = std::env::temp_dir().join(format!("tara_conn_test_{}", uuid::Uuid::new_v4()));
        let manager = ConnectionManager::new(&tmp);

        // Turn 1: Kanglish input
        let prof1 =
            manager.record_interaction_turn("senior_operator", "nange ee proof beku madi", true);
        assert_eq!(prof1.preferred_language, "Kanglish");
        assert_eq!(prof1.technical_depth, "RIGOROUS_ACADEMIC");
        assert!(prof1.collaboration_trust_score > 0.70);

        // Turn 2: Kannada script input
        let prof2 = manager.record_interaction_turn("senior_operator", "ಇದು ಹೇಗೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ?", true);
        assert_eq!(prof2.preferred_language, "Kannada");

        let style = manager.get_adaptive_communication_style("senior_operator");
        assert_eq!(style["preferred_language"], "Kannada");
        assert_eq!(style["include_formal_proofs"], true);
        assert_eq!(style["honorific_mode"], "PLURAL_RESPECTFUL_BAHUVACHANA");
        assert_eq!(style["forbid_singular_ekavachana_to_seniors"], true);

        let governed =
            manager.enforce_family_respect_and_honorifics("ನೀನು ಇಲ್ಲಿ ಬಾ ಮತ್ತು ನಿನ್ನ ಕೆಲಸ ಮಾಡು ");
        assert!(governed.contains("ನೀವು "));
        assert!(governed.contains("ಬನ್ನಿ "));
        assert!(governed.contains("ತಮ್ಮ "));
        assert!(governed.contains("ಮಾಡಿ "));

        let _ = fs::remove_dir_all(&tmp);
    }
}
