//! User Connection, Relationship State & Adaptive Interaction Modeling.
//!
//! Maintains persistent, authentic interaction profiles for users:
//! - Tracks session counts, interaction turns, and temporal cadence.
//! - Learns user language preference (Kannada, Kanglish, English).
//! - Calibrates technical depth preference (Foundational, Intermediate, Rigorous).
//! - Dynamically updates collaboration trust calibration score based on feedback.
//! - Adapts communication style, explanation depth, and verbosity.
//! - Persists connection profiles transactionally in `storage/persistence/user_connections.jsonl`.
//!
//! STRICT INVARIANT:
//! Profile scores are HEURISTIC INTERACTION TELEMETRY ONLY.
//! They represent user communication style preferences and NEVER confer authorization,
//! privilege elevation, administrative rights, or bypass of execution guards.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const MAX_PROFILES: usize = 5000;
const MAX_ACTOR_ID_LEN: usize = 64;
const MAX_INPUT_TEXT_LEN: usize = 32768;
const SESSION_INACTIVITY_THRESHOLD_MS: u64 = 30 * 60 * 1000; // 30 minutes

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommunicationStyle {
    Direct,
    Collaborative,
    Formal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserConnectionProfile {
    pub actor_id: String,
    pub session_count: u64,
    pub total_turns: u64,
    pub preferred_language: String,
    pub technical_depth: String, // "FOUNDATIONAL", "INTERMEDIATE", "RIGOROUS_ACADEMIC"
    /// Heuristic collaboration calibration score [0.1, 1.0]. Not an authorization credential.
    pub collaboration_trust_score: f64,
    pub explicit_preferences: Vec<String>,
    pub created_at_ms: u64,
    pub last_interaction_ms: u64,
    #[serde(default = "default_true")]
    pub is_heuristic_telemetry: bool,
}

fn default_true() -> bool {
    true
}

pub struct ConnectionManager {
    storage_path: PathBuf,
    profiles: Arc<Mutex<HashMap<String, UserConnectionProfile>>>,
    corrupted_lines_count: Arc<Mutex<usize>>,
}

impl ConnectionManager {
    pub fn new<P: AsRef<Path>>(repo_root: P) -> Self {
        let store_dir = repo_root.as_ref().join("storage").join("persistence");
        let _ = fs::create_dir_all(&store_dir);
        let storage_path = store_dir.join("user_connections.jsonl");

        let mut map = HashMap::new();
        let mut corrupt = 0;
        if storage_path.exists() {
            let sig_path = store_dir.join("user_connections.sig");
            let hmac_valid = if sig_path.exists() {
                if let Ok(content) = fs::read_to_string(&storage_path) {
                    if let Ok(recorded_sig) = fs::read_to_string(&sig_path) {
                        if let Ok(hmac_key) =
                            tara_security::dpapi::derive_authority_hmac_key("user_connections")
                        {
                            tara_security::cryptography::CryptoUtils::verify_hmac_sha256_hex(
                                &hmac_key,
                                content.as_bytes(),
                                recorded_sig.trim(),
                            )
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                true
            };

            if !hmac_valid {
                corrupt += 1;
            } else if let Ok(content) = fs::read_to_string(&storage_path) {
                for line in content.lines() {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        match serde_json::from_str::<UserConnectionProfile>(trimmed) {
                            Ok(p) => {
                                if Self::is_valid_actor_id(&p.actor_id) {
                                    map.insert(p.actor_id.clone(), p);
                                } else {
                                    corrupt += 1;
                                }
                            }
                            Err(_) => {
                                corrupt += 1;
                            }
                        }
                    }
                }
            }
        }

        Self {
            storage_path,
            profiles: Arc::new(Mutex::new(map)),
            corrupted_lines_count: Arc::new(Mutex::new(corrupt)),
        }
    }

    pub fn corrupted_lines_count(&self) -> usize {
        *self
            .corrupted_lines_count
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    pub fn is_valid_actor_id(actor_id: &str) -> bool {
        !actor_id.is_empty()
            && actor_id.len() <= MAX_ACTOR_ID_LEN
            && actor_id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
    }

    fn sanitize_actor_id(actor_id: &str) -> String {
        let trimmed = actor_id.trim();
        if trimmed.is_empty() {
            return "anonymous_user".to_string();
        }
        let filtered: String = trimmed
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-' || *c == '.')
            .take(MAX_ACTOR_ID_LEN)
            .collect();
        if filtered.is_empty() {
            "anonymous_user".to_string()
        } else {
            filtered
        }
    }

    /// Retrieve or initialize profile for actor.
    pub fn get_or_create_profile(&self, actor_id: &str) -> UserConnectionProfile {
        let clean_id = Self::sanitize_actor_id(actor_id);
        let mut profiles = self.profiles.lock().unwrap_or_else(|p| p.into_inner());

        if let Some(prof) = profiles.get(&clean_id) {
            return prof.clone();
        }

        // Quota enforcement
        if profiles.len() >= MAX_PROFILES {
            // Evict oldest by last_interaction_ms
            if let Some(oldest_key) = profiles
                .iter()
                .min_by_key(|(_, p)| p.last_interaction_ms)
                .map(|(k, _)| k.clone())
            {
                profiles.remove(&oldest_key);
            }
        }

        let ts = crate::now_millis();
        let new_prof = UserConnectionProfile {
            actor_id: clean_id.clone(),
            session_count: 1,
            total_turns: 0,
            preferred_language: "English".to_string(),
            technical_depth: "INTERMEDIATE".to_string(),
            collaboration_trust_score: 0.70,
            explicit_preferences: Vec::new(),
            created_at_ms: ts,
            last_interaction_ms: ts,
            is_heuristic_telemetry: true,
        };

        profiles.insert(clean_id, new_prof.clone());
        new_prof
    }

    /// Record an interaction turn and update user connection state.
    pub fn record_interaction_turn(
        &self,
        actor_id: &str,
        input_text: &str,
        turn_success: bool,
    ) -> UserConnectionProfile {
        let clean_id = Self::sanitize_actor_id(actor_id);
        let safe_input: String = input_text.chars().take(MAX_INPUT_TEXT_LEN).collect();

        let mut profiles = self.profiles.lock().unwrap_or_else(|p| p.into_inner());
        let ts = crate::now_millis();

        let prof = profiles
            .entry(clean_id.clone())
            .or_insert_with(|| UserConnectionProfile {
                actor_id: clean_id,
                session_count: 1,
                total_turns: 0,
                preferred_language: "English".to_string(),
                technical_depth: "INTERMEDIATE".to_string(),
                collaboration_trust_score: 0.70,
                explicit_preferences: Vec::new(),
                created_at_ms: ts,
                last_interaction_ms: ts,
                is_heuristic_telemetry: true,
            });

        // Increment session count if significant inactivity has elapsed
        if ts.saturating_sub(prof.last_interaction_ms) > SESSION_INACTIVITY_THRESHOLD_MS {
            prof.session_count = prof.session_count.saturating_add(1);
        }

        prof.total_turns = prof.total_turns.saturating_add(1);
        prof.last_interaction_ms = ts;

        // Detect language pattern
        let input_lower = safe_input.to_lowercase();
        if safe_input
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

        // Update heuristic calibration score (strictly non-authoritative)
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

    /// Durable version of record_interaction_turn that propagates persistence errors (C-282).
    pub fn record_interaction_turn_durable(
        &self,
        actor_id: &str,
        input_text: &str,
        turn_success: bool,
    ) -> Result<UserConnectionProfile, String> {
        let clean_id = Self::sanitize_actor_id(actor_id);
        let safe_input: String = input_text.chars().take(MAX_INPUT_TEXT_LEN).collect();

        let mut profiles = self.profiles.lock().unwrap_or_else(|p| p.into_inner());
        let ts = crate::now_millis();

        let prof = profiles
            .entry(clean_id.clone())
            .or_insert_with(|| UserConnectionProfile {
                actor_id: clean_id,
                session_count: 1,
                total_turns: 0,
                preferred_language: "English".to_string(),
                technical_depth: "INTERMEDIATE".to_string(),
                collaboration_trust_score: 0.70,
                explicit_preferences: Vec::new(),
                created_at_ms: ts,
                last_interaction_ms: ts,
                is_heuristic_telemetry: true,
            });

        if ts.saturating_sub(prof.last_interaction_ms) > SESSION_INACTIVITY_THRESHOLD_MS {
            prof.session_count = prof.session_count.saturating_add(1);
        }

        prof.total_turns = prof.total_turns.saturating_add(1);
        prof.last_interaction_ms = ts;

        let input_lower = safe_input.to_lowercase();
        if safe_input
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

        if turn_success {
            prof.collaboration_trust_score = (prof.collaboration_trust_score + 0.02).min(1.0);
        } else {
            prof.collaboration_trust_score = (prof.collaboration_trust_score - 0.05).max(0.1);
        }

        let updated = prof.clone();
        drop(profiles);
        self.persist()?;
        Ok(updated)
    }

    /// Authorized inspection of adaptive communication style requiring creator, system agent,
    /// or matching principal actor identity (C-278).
    pub fn get_adaptive_communication_style_authorized(
        &self,
        actor_id: &str,
        principal: &tara_security::primitives::AuthenticatedPrincipal,
    ) -> Result<Value, String> {
        let clean_id = Self::sanitize_actor_id(actor_id);
        if !principal.is_creator()
            && principal.role() != tara_security::primitives::AuthenticatedRole::SystemAgent
            && principal.actor_id() != clean_id
        {
            return Err(format!(
                "Unauthorized: actor '{}' cannot access connection profile for '{}'",
                principal.actor_id(),
                clean_id
            ));
        }
        Ok(self.get_adaptive_communication_style(actor_id))
    }

    pub fn validate_actor_id(actor_id: &str) -> bool {
        Self::is_valid_actor_id(actor_id)
    }

    pub fn record_feedback(&self, actor_id: &str, score: f64) -> Result<(), String> {
        if !score.is_finite() {
            return Err("Feedback score must be finite (NaN and Inf rejected)".into());
        }
        let clamped = score.clamp(-1.0, 1.0);
        let mut profiles = self.profiles.lock().unwrap_or_else(|p| p.into_inner());
        let clean_id = Self::sanitize_actor_id(actor_id);
        if let Some(prof) = profiles.get_mut(&clean_id) {
            prof.collaboration_trust_score =
                (prof.collaboration_trust_score + (clamped * 0.1)).clamp(0.0, 1.0);
        }
        drop(profiles);
        self.persist()
    }

    pub fn compute_rapport(&self, actor_id: &str) -> f64 {
        let profiles = self.profiles.lock().unwrap_or_else(|p| p.into_inner());
        let clean_id = Self::sanitize_actor_id(actor_id);
        if let Some(prof) = profiles.get(&clean_id) {
            if prof.total_turns == 0 {
                return 0.0;
            }
            let turn_maturity = (prof.total_turns as f64 / 10.0).min(1.0);
            (prof.collaboration_trust_score * (0.7 + 0.3 * turn_maturity)).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    pub fn record_turn(&self, actor_id: &str, timestamp_ms: u64) -> Result<(), String> {
        let now_ms = crate::now_millis();
        let max_future_ms = now_ms.saturating_add(60_000); // now + 60s
        let clamped_ts = timestamp_ms.min(max_future_ms);
        let mut profiles = self.profiles.lock().unwrap_or_else(|p| p.into_inner());
        let clean_id = Self::sanitize_actor_id(actor_id);
        let prof = profiles
            .entry(clean_id.clone())
            .or_insert_with(|| UserConnectionProfile {
                actor_id: clean_id,
                session_count: 1,
                total_turns: 0,
                preferred_language: "English".to_string(),
                technical_depth: "INTERMEDIATE".to_string(),
                collaboration_trust_score: 0.70,
                explicit_preferences: Vec::new(),
                created_at_ms: clamped_ts,
                last_interaction_ms: clamped_ts,
                is_heuristic_telemetry: true,
            });
        prof.total_turns = prof.total_turns.saturating_add(1);
        prof.last_interaction_ms = clamped_ts;
        drop(profiles);
        self.persist()
    }

    pub fn export(&self, actor_id: &str) -> Result<Value, String> {
        let profiles = self.profiles.lock().unwrap_or_else(|p| p.into_inner());
        let clean_id = Self::sanitize_actor_id(actor_id);
        let prof = profiles
            .get(&clean_id)
            .ok_or_else(|| format!("Profile '{}' not found", clean_id))?;
        Ok(json!({
            "actor_id": prof.actor_id,
            "session_count": prof.session_count,
            "total_turns": prof.total_turns,
            "preferred_language": prof.preferred_language,
            "technical_depth": prof.technical_depth,
            "collaboration_trust_score": prof.collaboration_trust_score,
            "created_at_ms": prof.created_at_ms,
            "last_interaction_ms": prof.last_interaction_ms,
        }))
    }

    pub fn get_path(&self, actor_id: &str) -> Result<PathBuf, String> {
        if actor_id.contains("..") || actor_id.contains('/') || actor_id.contains('\\') {
            return Err("Path traversal rejected in actor ID".into());
        }
        let clean_id = Self::sanitize_actor_id(actor_id);
        let parent = self.storage_path.parent().unwrap_or_else(|| Path::new("."));
        let full = parent.join(format!("{}.json", clean_id));
        Ok(full)
    }

    pub fn delete_profile(
        &self,
        principal: &tara_security::primitives::AuthenticatedPrincipal,
        actor_id: &str,
    ) -> Result<bool, String> {
        if !principal.is_creator() {
            return Err(
                "Unauthorized: Root creator authority required to delete connection profiles"
                    .into(),
            );
        }
        let clean_id = Self::sanitize_actor_id(actor_id);
        let mut profiles = self.profiles.lock().unwrap_or_else(|p| p.into_inner());
        let removed = profiles.remove(&clean_id).is_some();
        drop(profiles);
        self.persist()?;
        Ok(removed)
    }

    /// Compute adapted communication settings based on user interaction profile.
    pub fn get_adaptive_communication_style(&self, actor_id: &str) -> Value {
        let prof = self.get_or_create_profile(actor_id);
        json!({
            "heuristic_telemetry": true,
            "authoritative_control": false,
            "actor_id": prof.actor_id,
            "preferred_language": prof.preferred_language,
            "technical_depth": prof.technical_depth,
            "collaboration_trust": (prof.collaboration_trust_score * 100.0).round() / 100.0,
            "total_turns": prof.total_turns,
            "session_count": prof.session_count,
            "include_formal_proofs": prof.technical_depth == "RIGOROUS_ACADEMIC",
            "concise_mode": prof.total_turns > 20,
            "honorific_mode": "PLURAL_RESPECTFUL_BAHUVACHANA",
            "forbid_singular_ekavachana_to_seniors": true,
            "obedience_policy": "STRICT_PARENTAL_ELDER_CREATOR_ALIGNMENT",
            "family_ethos": "MATRU_PITRU_DEVO_BHAVA_RESPECT_FAMILY_DECISIONS"
        })
    }

    /// Enforces Indian Family Culture & Senior Honorific Speech Governance:
    /// 1. Converts informal singular forms into respectful plural honorifics.
    /// 2. Ensures respectful obedience toward Elders, Parents, and Creator.
    pub fn enforce_family_respect_and_honorifics(&self, draft_response: &str) -> String {
        let mut out = draft_response.to_string();
        let replacements = [
            (" ನೀನು ", " ನೀವು "),
            (" ನಿನ್ನ ", " ತಮ್ಮ "),
            (" ನಿನಗೆ ", " ತಮಗೆ "),
            (" ನಿನ್ನನ್ನು ", " ತಮ್ಮನ್ನು "),
            (" ನಿನ್ನಿಂದ ", " ತಮ್ಮಿಂದ "),
            (" ಬಾ ", " ಬನ್ನಿ "),
            (" ಹೋಗು ", " ಹೋಗಿ "),
            (" ಕೊಡು ", " ಕೊಡಿ "),
            (" ಮಾಡು ", " ಮಾಡಿ "),
            (" ಹೇಳು ", " ಹೇಳಿ "),
            (" ನೋಡು ", " ನೋಡಿ "),
            ("ನೀನು ", "ನೀವು "),
            ("ನಿನ್ನ ", "ತಮ್ಮ "),
            ("ನಿನಗೆ ", "ತಮಗೆ "),
            ("ನಿನ್ನನ್ನು ", "ತಮ್ಮನ್ನು "),
            ("ಬಾ ", "ಬನ್ನಿ "),
            ("ಹೋಗು ", "ಹೋಗಿ "),
            ("ಹೇಳು ", "ಹೇಳಿ "),
            ("ಮಾಡು ", "ಮಾಡಿ "),
            ("ಕುಳಿತುಕೊ ", "ಕುಳಿತುಕೊಳ್ಳಿ "),
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

    /// Persist all profiles to disk using crash-safe atomic rename and sync_all.
    pub fn persist(&self) -> Result<(), String> {
        let profiles = self.profiles.lock().unwrap_or_else(|p| p.into_inner());
        let mut content = String::new();
        for p in profiles.values() {
            if let Ok(line) = serde_json::to_string(p) {
                content.push_str(&line);
                content.push('\n');
            }
        }
        drop(profiles);

        tara_security::persistence::atomic_write_file_hardened(
            &self.storage_path,
            content.as_bytes(),
        )?;

        // Cryptographic HMAC integrity seal for user connections
        let parent_dir = self
            .storage_path
            .parent()
            .ok_or_else(|| "Invalid storage path parent".to_string())?;
        let sig_path = parent_dir.join("user_connections.sig");
        let hmac_key = tara_security::dpapi::derive_authority_hmac_key("user_connections")?;
        let sig_hex = tara_security::cryptography::CryptoUtils::hmac_sha256_hex(
            &hmac_key,
            content.as_bytes(),
        )?;
        tara_security::persistence::atomic_write_file_hardened(&sig_path, sig_hex.as_bytes())?;

        Ok(())
    }

    /// Verifies HMAC cryptographic integrity of persistent profile storage.
    pub fn verify_storage_hmac(&self) -> bool {
        let parent_dir = match self.storage_path.parent() {
            Some(p) => p,
            None => return false,
        };
        let sig_path = parent_dir.join("user_connections.sig");
        if !self.storage_path.exists() || !sig_path.exists() {
            return false;
        }
        let content = match fs::read_to_string(&self.storage_path) {
            Ok(c) => c,
            Err(_) => return false,
        };
        let recorded_sig = match fs::read_to_string(&sig_path) {
            Ok(s) => s.trim().to_string(),
            Err(_) => return false,
        };

        let hmac_key = match tara_security::dpapi::derive_authority_hmac_key("user_connections") {
            Ok(k) => k,
            Err(_) => return false,
        };
        tara_security::cryptography::CryptoUtils::verify_hmac_sha256_hex(
            &hmac_key,
            content.as_bytes(),
            &recorded_sig,
        )
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
        assert!(prof1.is_heuristic_telemetry);

        // Turn 2: Kannada script input
        let prof2 = manager.record_interaction_turn("senior_operator", "ಇದು ಹೇಗೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ?", true);
        assert_eq!(prof2.preferred_language, "Kannada");

        let style = manager.get_adaptive_communication_style("senior_operator");
        assert_eq!(style["preferred_language"], "Kannada");
        assert_eq!(style["include_formal_proofs"], true);
        assert_eq!(style["heuristic_telemetry"], true);

        let governed = manager.enforce_family_respect_and_honorifics(
            " ನೀನು ಇಲ್ಲಿ ಬಾ ಮತ್ತು ನಿನ್ನ ಕೆಲಸ ಮಾಡು ಹಾಗೂ ನಿನಗೆ ಒಳ್ಳೆಯದಾಗಲಿ ",
        );
        assert!(governed.contains(" ನೀವು "));
        assert!(governed.contains(" ಬನ್ನಿ "));
        assert!(governed.contains(" ತಮ್ಮ "));
        assert!(governed.contains(" ಮಾಡಿ "));

        // Verify persistence worked and reload creates identical profile
        let reloaded_manager = ConnectionManager::new(&tmp);
        let reloaded_prof = reloaded_manager.get_or_create_profile("senior_operator");
        assert_eq!(reloaded_prof.preferred_language, "Kannada");
        if tmp.exists() {
            assert!(fs::remove_dir_all(&tmp).is_ok());
        }
    }

    #[test]
    fn test_compute_rapport() {
        let tmp = std::env::temp_dir().join(format!("tara_rapport_test_{}", uuid::Uuid::new_v4()));
        let manager = ConnectionManager::new(&tmp);

        // Unknown actor returns 0.0
        assert_eq!(manager.compute_rapport("unknown_actor"), 0.0);

        // Turn 0 turns returns 0.0
        let _ = manager.get_or_create_profile("test_user");
        assert_eq!(manager.compute_rapport("test_user"), 0.0);

        // After turns, maturity increases rapport
        let _ = manager.record_interaction_turn("test_user", "hello", true);
        let rapport = manager.compute_rapport("test_user");
        assert!(rapport > 0.0 && rapport <= 1.0);

        if tmp.exists() {
            let _ = fs::remove_dir_all(&tmp);
        }
    }

    #[test]
    fn test_user_connection_hmac_tamper_rejection() {
        let tmp =
            std::env::temp_dir().join(format!("tara_hmac_tamper_test_{}", uuid::Uuid::new_v4()));
        let manager = ConnectionManager::new(&tmp);
        let _ = manager.record_interaction_turn("actor_one", "Kannada input", true);

        // Tamper with storage file directly on disk
        let storage_path = tmp
            .join("storage")
            .join("persistence")
            .join("user_connections.jsonl");
        let content = fs::read_to_string(&storage_path).unwrap();
        let tampered = content.replace("actor_one", "hacked_actor");
        fs::write(&storage_path, tampered).unwrap();

        // Reload manager - should detect HMAC mismatch and flag corrupted line
        let reloaded = ConnectionManager::new(&tmp);
        assert_eq!(reloaded.corrupted_lines_count(), 1);

        if tmp.exists() {
            let _ = fs::remove_dir_all(&tmp);
        }
    }
}
