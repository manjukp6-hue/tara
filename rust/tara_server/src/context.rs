//! context.rs
//!
//! Multi-Turn Context & Clarification Management for TARA Server.
//! Ports Python's context.py fully:
//! - Bounded session history (requests, responses, tool results).
//! - Anaphoric reference resolution ("it", "that file", "previous result").
//! - Missing-parameter detection with halt-and-ask clarification flow.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Session Turn
// ---------------------------------------------------------------------------

/// A single recorded turn in a dialogue session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTurn {
    pub user_input: String,
    pub response: String,
    pub slots: HashMap<String, String>,
    pub tool_result: Option<HashMap<String, serde_json::Value>>,
    pub outcome: String,
    /// Unix timestamp (seconds since epoch, stored as f64 for JSON compat).
    pub timestamp: f64,
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// Encapsulates dialogue state and history for a conversation session.
#[derive(Debug)]
pub struct Session {
    pub session_id: String,
    pub actor_id: String,
    pub max_turns: usize,
    pub history: Vec<SessionTurn>,
    pub pending_task: Option<HashMap<String, serde_json::Value>>,
    pub last_tool_result: Option<HashMap<String, serde_json::Value>>,
    pub last_slots: HashMap<String, String>,
    pub last_user_request: Option<String>,
    pub last_response: Option<String>,
    pub created_at: Instant,
    pub updated_at: Instant,
}

impl Session {
    /// Creates a new empty session.
    pub fn new(
        session_id: impl Into<String>,
        actor_id: impl Into<String>,
        max_turns: usize,
    ) -> Self {
        let now = Instant::now();
        Self {
            session_id: session_id.into(),
            actor_id: actor_id.into(),
            max_turns,
            history: Vec::new(),
            pending_task: None,
            last_tool_result: None,
            last_slots: HashMap::new(),
            last_user_request: None,
            last_response: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Appends a turn to bounded session history and updates state.
    pub fn record_turn(
        &mut self,
        user_input: impl Into<String>,
        response: impl Into<String>,
        slots: Option<HashMap<String, String>>,
        tool_result: Option<HashMap<String, serde_json::Value>>,
        outcome: impl Into<String>,
    ) {
        let user_input = user_input.into();
        let response = response.into();
        let outcome = outcome.into();

        self.last_user_request = Some(user_input.clone());
        self.last_response = Some(response.clone());

        if let Some(ref s) = slots {
            for (k, v) in s {
                if !v.is_empty() {
                    self.last_slots.insert(k.clone(), v.clone());
                }
            }
        }

        if let Some(ref tr) = tool_result {
            // Extract file_path and hash from tool result for slot memory
            if let Some(serde_json::Value::String(fp)) = tr.get("file_path").cloned() {
                if !fp.is_empty() {
                    self.last_slots.insert("file_path".to_owned(), fp);
                }
            }
            if let Some(serde_json::Value::String(sha)) = tr.get("sha256").cloned() {
                if !sha.is_empty() {
                    self.last_slots.insert("hash".to_owned(), sha);
                }
            } else if let Some(serde_json::Value::String(h)) = tr.get("hash").cloned() {
                if !h.is_empty() {
                    self.last_slots.insert("hash".to_owned(), h);
                }
            }
            self.last_tool_result = Some(tr.clone());
        }

        let turn = SessionTurn {
            user_input,
            response,
            slots: self.last_slots.clone(),
            tool_result,
            outcome,
            timestamp: unix_timestamp_f64(),
        };
        self.history.push(turn);
        if self.history.len() > self.max_turns {
            let drain_count = self.history.len() - self.max_turns;
            self.history.drain(0..drain_count);
        }
        self.updated_at = Instant::now();
    }

    /// Returns the pending task if set.
    pub fn get_pending_task(&self) -> Option<&HashMap<String, serde_json::Value>> {
        self.pending_task.as_ref()
    }

    /// Sets a pending task (halted, awaiting clarification).
    pub fn set_pending_task(&mut self, task: HashMap<String, serde_json::Value>) {
        self.pending_task = Some(task);
        self.updated_at = Instant::now();
    }

    /// Clears and returns the pending task.
    pub fn clear_pending_task(&mut self) -> Option<HashMap<String, serde_json::Value>> {
        let task = self.pending_task.take();
        self.updated_at = Instant::now();
        task
    }

    /// Returns approximate age since last update.
    pub fn idle_duration(&self) -> Duration {
        self.updated_at.elapsed()
    }
}

// ---------------------------------------------------------------------------
// SessionContextManager
// ---------------------------------------------------------------------------

/// Thread-safe multi-session context registry with TTL expiration.
pub struct SessionContextManager {
    sessions: Arc<Mutex<HashMap<String, Arc<Mutex<Session>>>>>,
    max_turns: usize,
    ttl: Duration,
    last_cleanup: Mutex<Instant>,
}

/// Default maximum turns retained in session working memory.
pub const DEFAULT_MAX_TURNS_PER_SESSION: usize = 50;
/// Default session TTL in seconds (1 hour).
pub const DEFAULT_SESSION_TTL_SECONDS: u64 = 3600;

impl Default for SessionContextManager {
    /// Constructs manager using dynamic runtime configuration (`TARA_MAX_TURNS_PER_SESSION`, `TARA_SESSION_TTL_SECONDS`),
    /// falling back to the baseline defaults if unspecified.
    fn default() -> Self {
        let max_turns = std::env::var("TARA_MAX_TURNS_PER_SESSION")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MAX_TURNS_PER_SESSION);
        let ttl_secs = std::env::var("TARA_SESSION_TTL_SECONDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_SESSION_TTL_SECONDS);
        Self::new(max_turns, ttl_secs)
    }
}

impl SessionContextManager {
    /// Creates a new manager with bounded turn history and session TTL.
    pub fn new(max_turns_per_session: usize, ttl_seconds: u64) -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            max_turns: max_turns_per_session,
            ttl: Duration::from_secs(ttl_seconds),
            last_cleanup: Mutex::new(Instant::now()),
        }
    }

    /// Retrieves an existing non-expired session or creates a new one.
    pub fn get_or_create(&self, session_id: &str, actor_id: &str) -> Arc<Mutex<Session>> {
        let key = if !session_id.is_empty() {
            format!("{}::{}", actor_id, session_id)
        } else {
            actor_id.to_owned()
        };

        self.maybe_cleanup();

        let mut sessions = self.sessions.lock().expect("session lock poisoned");
        if let Some(existing) = sessions.get(&key) {
            let expired = {
                let sess = existing.lock().expect("session inner lock poisoned");
                sess.idle_duration() > self.ttl
            };
            if !expired {
                return Arc::clone(existing);
            }
            // Expired — recreate
        }

        let sess = Arc::new(Mutex::new(Session::new(
            if session_id.is_empty() {
                "default"
            } else {
                session_id
            },
            actor_id,
            self.max_turns,
        )));
        sessions.insert(key, Arc::clone(&sess));
        sess
    }

    /// Returns the number of currently tracked sessions.
    pub fn active_count(&self) -> usize {
        self.sessions.lock().map(|s| s.len()).unwrap_or(0)
    }

    /// Removes all sessions that have exceeded their TTL.
    pub fn cleanup_expired(&self) -> usize {
        let ttl = self.ttl;
        let mut sessions = self.sessions.lock().expect("session lock poisoned");
        let before = sessions.len();
        sessions.retain(|_, v| v.lock().map(|s| s.idle_duration() <= ttl).unwrap_or(false));
        let removed = before - sessions.len();
        if let Ok(mut last) = self.last_cleanup.lock() {
            *last = Instant::now();
        }
        removed
    }

    fn maybe_cleanup(&self) {
        let should = self
            .last_cleanup
            .lock()
            .map(|ts| ts.elapsed() > Duration::from_secs(60))
            .unwrap_or(false);
        if should {
            self.cleanup_expired();
        }
    }
}

// ---------------------------------------------------------------------------
// CoreferenceResolver
// ---------------------------------------------------------------------------

/// Resolves anaphoric expressions ("it", "that file", "previous result") from session history.
pub struct CoreferenceResolver;

impl CoreferenceResolver {
    const ANAPHORIC_FILE_PATTERNS: &'static [&'static str] =
        &[r"(?i)\b(it|that file|this file|the file|the same file|its hash|the log file)\b"];
    const ANAPHORIC_RESULT_PATTERNS: &'static [&'static str] =
        &[r"(?i)\b(previous result|last result|the result|prior output)\b"];

    /// Resolves anaphora against the session's prior slots and results.
    ///
    /// Returns `(resolved_text, resolved_slots)`.
    pub fn resolve(text: &str, session: Option<&Session>) -> (String, HashMap<String, String>) {
        let mut resolved_slots: HashMap<String, String> = HashMap::new();

        let sess = match session {
            Some(s) if !s.history.is_empty() => s,
            _ => return (text.to_owned(), resolved_slots),
        };

        // File anaphora
        let has_file_ref = Self::ANAPHORIC_FILE_PATTERNS
            .iter()
            .any(|pat| Regex::new(pat).map(|re| re.is_match(text)).unwrap_or(false));
        if has_file_ref {
            if let Some(fp) = sess.last_slots.get("file_path") {
                resolved_slots.insert("file_path".to_owned(), fp.clone());
            }
        }

        // Previous result anaphora
        let has_res_ref = Self::ANAPHORIC_RESULT_PATTERNS
            .iter()
            .any(|pat| Regex::new(pat).map(|re| re.is_match(text)).unwrap_or(false));
        if has_res_ref {
            if let Some(tr) = &sess.last_tool_result {
                // Store serialised result as JSON string in slots
                if let Ok(json_str) = serde_json::to_string(tr) {
                    resolved_slots.insert("previous_result".to_owned(), json_str);
                }
            }
        }

        // Hash anaphora
        let text_lower = text.to_lowercase();
        if (text_lower.contains("hash") || text_lower.contains("checksum"))
            && sess.last_slots.contains_key("hash")
        {
            if let Some(h) = sess.last_slots.get("hash") {
                resolved_slots.insert("expected_hash".to_owned(), h.clone());
            }
        }

        (text.to_owned(), resolved_slots)
    }
}

// ---------------------------------------------------------------------------
// ClarificationManager
// ---------------------------------------------------------------------------

/// A clarification request emitted when a required slot is missing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClarificationResult {
    pub needs_clarification: bool,
    pub target: String,
    pub missing_slot: String,
    pub clarification_question: String,
}

/// Evaluates parameter completeness for intended tools/skills/capabilities.
pub struct ClarificationManager;

impl ClarificationManager {
    /// Returns slot requirements for known tools / skills.
    fn required_slots() -> HashMap<&'static str, Vec<(&'static str, &'static str)>> {
        let mut m: HashMap<&str, Vec<(&str, &str)>> = HashMap::new();
        m.insert(
            "file_inspector",
            vec![("file_path", "Which file would you like me to inspect?")],
        );
        m.insert(
            "hash_verifier",
            vec![(
                "file_path",
                "Which file would you like to calculate or verify the hash for?",
            )],
        );
        m.insert(
            "delete_file",
            vec![(
                "file_path",
                "Which file or directory do you wish to delete?",
            )],
        );
        m.insert(
            "developer",
            vec![(
                "code",
                "Please provide the Python code snippet you want me to check.",
            )],
        );
        m.insert(
            "device",
            vec![(
                "gcode",
                "Please provide the G-code command you want to validate.",
            )],
        );
        m
    }

    /// Checks if required parameters are missing for the given intent.
    ///
    /// Returns `Some(ClarificationResult)` if a required slot is absent, else `None`.
    pub fn evaluate(
        intent: &str,
        tool_or_skill: Option<&str>,
        action_type: Option<&str>,
        params: &HashMap<String, serde_json::Value>,
    ) -> Option<ClarificationResult> {
        let target_name: &str = match intent {
            "EXECUTE_TOOL" => tool_or_skill?,
            "EXECUTE_SKILL" => tool_or_skill?,
            "GUARDED_ACTION" => action_type?,
            _ => return None,
        };

        let req_slots = Self::required_slots();
        let requirements = req_slots.get(target_name)?;

        for (slot_key, question) in requirements {
            let missing = match params.get(*slot_key) {
                None => true,
                Some(serde_json::Value::String(s)) => s.trim().is_empty(),
                Some(serde_json::Value::Null) => true,
                _ => false,
            };
            if missing {
                return Some(ClarificationResult {
                    needs_clarification: true,
                    target: target_name.to_owned(),
                    missing_slot: slot_key.to_string(),
                    clarification_question: question.to_string(),
                });
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the current Unix timestamp as f64 seconds.
fn unix_timestamp_f64() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_turns_bounded_by_max_turns() {
        let mut session = Session::new("sess_1", "user_1", 3);
        assert_eq!(session.history.len(), 0);

        session.record_turn("turn 1", "reply 1", None, None, "ok");
        session.record_turn("turn 2", "reply 2", None, None, "ok");
        session.record_turn("turn 3", "reply 3", None, None, "ok");
        assert_eq!(session.history.len(), 3);
        assert_eq!(session.history[0].user_input, "turn 1");

        // 4th turn pushes out the oldest (turn 1)
        session.record_turn("turn 4", "reply 4", None, None, "ok");
        assert_eq!(session.history.len(), 3);
        assert_eq!(session.history[0].user_input, "turn 2");
        assert_eq!(session.history[2].user_input, "turn 4");
    }

    #[test]
    fn test_session_context_manager_isolation() {
        let mgr = SessionContextManager::new(10, 3600);
        let s1 = mgr.get_or_create("session_a", "alice");
        let s2 = mgr.get_or_create("session_b", "bob");

        s1.lock().unwrap().record_turn("hello from alice", "hi alice", None, None, "ok");
        s2.lock().unwrap().record_turn("hello from bob", "hi bob", None, None, "ok");

        assert_eq!(s1.lock().unwrap().history.len(), 1);
        assert_eq!(s1.lock().unwrap().history[0].user_input, "hello from alice");

        assert_eq!(s2.lock().unwrap().history.len(), 1);
        assert_eq!(s2.lock().unwrap().history[0].user_input, "hello from bob");

        assert_eq!(mgr.active_count(), 2);
    }

    #[test]
    fn test_clarification_manager_missing_parameter() {
        let mut params = HashMap::new();
        params.insert("target_file".to_string(), serde_json::Value::String("src/main.rs".into()));

        // "file_inspector" requires "file_path"
        let result = ClarificationManager::evaluate("EXECUTE_TOOL", Some("file_inspector"), None, &params);
        assert!(result.is_some());
        let clarification = result.unwrap();
        assert!(clarification.needs_clarification);
        assert_eq!(clarification.missing_slot, "file_path");

        // When file_path is provided, no clarification needed
        params.insert("file_path".to_string(), serde_json::Value::String("src/main.rs".into()));
        let result_ok = ClarificationManager::evaluate("EXECUTE_TOOL", Some("file_inspector"), None, &params);
        assert!(result_ok.is_none());
    }
}
