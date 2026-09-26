//! Control token action parser.
//!
//! Parses special control tokens embedded in model output to dispatch
//! tools, skills, and creator-auth flows exactly as the Python implementation.

use serde_json::Value;

/// A parsed action triggered by a control token in model output.
#[derive(Debug, Clone)]
pub struct ControlTokenAction {
    /// "TOOL", "SKILL", "CREATOR_AUTH", or "RULE"
    pub action_type: String,
    /// Target tool/skill name or auth method
    pub target: String,
    /// Optional structured payload parsed from JSON after the control token
    pub payload: Value,
    /// The raw text that was matched
    pub raw_text: String,
}

/// Parses control tokens from model-generated text.
pub struct ControlTokenActionParser;

impl ControlTokenActionParser {
    /// Attempt to parse a [`ControlTokenAction`] from model output text.
    ///
    /// Checks for the following patterns in order:
    /// - `<|tara_exec|>tool_name{...}` → TOOL action
    /// - `<|tara_skill|>skill_name{...}` → SKILL action
    /// - `<|creator_auth|>method` → CREATOR_AUTH action
    /// - `<|tara_rule|>rule_name` → RULE action
    ///
    /// Returns `None` if no control token is found.
    pub fn parse(output_text: &str) -> Option<ControlTokenAction> {
        let text = output_text.trim();

        // TOOL dispatch
        if let Some(rest) = Self::strip_prefix(text, "<|tara_exec|>") {
            let (target, payload) = Self::split_target_payload(rest);
            return Some(ControlTokenAction {
                action_type: "TOOL".to_string(),
                target,
                payload,
                raw_text: text.to_string(),
            });
        }

        // SKILL dispatch
        if let Some(rest) = Self::strip_prefix(text, "<|tara_skill|>") {
            let (target, payload) = Self::split_target_payload(rest);
            return Some(ControlTokenAction {
                action_type: "SKILL".to_string(),
                target,
                payload,
                raw_text: text.to_string(),
            });
        }

        // CREATOR_AUTH flow
        if let Some(rest) = Self::strip_prefix(text, "<|creator_auth|>") {
            let target = rest.split_whitespace().next().unwrap_or("").to_string();
            return Some(ControlTokenAction {
                action_type: "CREATOR_AUTH".to_string(),
                target,
                payload: Value::Null,
                raw_text: text.to_string(),
            });
        }

        // RULE invocation
        if let Some(rest) = Self::strip_prefix(text, "<|tara_rule|>") {
            let target = rest.split_whitespace().next().unwrap_or("").to_string();
            return Some(ControlTokenAction {
                action_type: "RULE".to_string(),
                target,
                payload: Value::Null,
                raw_text: text.to_string(),
            });
        }

        None
    }

    fn strip_prefix<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
        if s.starts_with(prefix) {
            Some(&s[prefix.len()..])
        } else {
            None
        }
    }

    fn split_target_payload(rest: &str) -> (String, Value) {
        // Target name is the first word; payload is optional trailing JSON
        let json_start = rest.find('{');
        let (name_part, json_part) = if let Some(pos) = json_start {
            (&rest[..pos], &rest[pos..])
        } else {
            (rest, "")
        };

        let target = name_part.trim().to_string();
        let payload = if !json_part.is_empty() {
            serde_json::from_str(json_part).unwrap_or(Value::Null)
        } else {
            Value::Null
        };

        (target, payload)
    }
}
