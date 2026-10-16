//! Control token action parser.
//!
//! Parses special control tokens embedded in model output to dispatch
//! tools, skills, and creator-auth flows across the native Rust subsystem.

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

        for (tag, action_type, has_json) in [
            ("<|tara_exec|>", "TOOL", true),
            ("<|tara_skill|>", "SKILL", true),
            ("<|creator_auth|>", "CREATOR_AUTH", false),
            ("<|tara_rule|>", "RULE", false),
        ] {
            if let Some(idx) = text.find(tag) {
                let matched_slice = &text[idx..];
                let rest = &matched_slice[tag.len()..];
                if has_json {
                    let (target, payload) = Self::split_target_payload(rest);
                    if !target.is_empty() {
                        return Some(ControlTokenAction {
                            action_type: action_type.to_string(),
                            target,
                            payload,
                            raw_text: matched_slice.to_string(),
                        });
                    }
                } else {
                    let target = rest
                        .trim_start()
                        .split(|c: char| c.is_whitespace() || c == '<')
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if !target.is_empty() {
                        return Some(ControlTokenAction {
                            action_type: action_type.to_string(),
                            target,
                            payload: Value::Null,
                            raw_text: matched_slice.to_string(),
                        });
                    }
                }
            }
        }

        None
    }

    fn split_target_payload(rest: &str) -> (String, Value) {
        let json_start = rest.find('{');
        let (name_part, json_part) = if let Some(pos) = json_start {
            (&rest[..pos], &rest[pos..])
        } else {
            (rest.split('<').next().unwrap_or(rest), "")
        };

        let target = name_part
            .trim_start()
            .split(|c: char| c.is_whitespace() || c == '<')
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        let payload = if !json_part.is_empty() {
            let mut stream = serde_json::Deserializer::from_str(json_part).into_iter::<Value>();
            stream.next().and_then(|r| r.ok()).unwrap_or(Value::Null)
        } else {
            Value::Null
        };

        (target, payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_with_introductory_text() {
        let text = "Here is my thought process.\n<|tara_exec|>search {\"query\": \"rust\"}";
        let action = ControlTokenActionParser::parse(text).expect("action parsed");
        assert_eq!(action.action_type, "TOOL");
        assert_eq!(action.target, "search");
        assert_eq!(action.payload["query"], "rust");
    }

    #[test]
    fn test_parse_json_payload_with_trailing_im_end() {
        let text = "<|tara_exec|>calculator {\"expression\": \"2 + 2\"}<|im_end|>";
        let action = ControlTokenActionParser::parse(text).expect("action parsed");
        assert_eq!(action.action_type, "TOOL");
        assert_eq!(action.target, "calculator");
        assert_eq!(action.payload["expression"], "2 + 2");
    }

    #[test]
    fn test_parse_skill_with_preceding_text() {
        let text = "Invoking skill now: <|tara_skill|>code_reviewer {\"lang\": \"rs\"}<|im_end|>";
        let action = ControlTokenActionParser::parse(text).expect("action parsed");
        assert_eq!(action.action_type, "SKILL");
        assert_eq!(action.target, "code_reviewer");
        assert_eq!(action.payload["lang"], "rs");
    }

    #[test]
    fn test_parse_creator_auth_and_rule_with_preceding_text() {
        let text_auth = "System authentication check: <|creator_auth|>ed25519";
        let action_auth = ControlTokenActionParser::parse(text_auth).expect("action parsed");
        assert_eq!(action_auth.action_type, "CREATOR_AUTH");
        assert_eq!(action_auth.target, "ed25519");

        let text_rule = "Checking governance policy: <|tara_rule|>deterministic_inference";
        let action_rule = ControlTokenActionParser::parse(text_rule).expect("action parsed");
        assert_eq!(action_rule.action_type, "RULE");
        assert_eq!(action_rule.target, "deterministic_inference");
    }

    #[test]
    fn test_parse_auth_and_rule_with_spaces_and_trailing_tokens() {
        // Space after token tag and trailing <|im_end|>
        let text_auth = "Prefix <|creator_auth|>  ed25519<|im_end|>";
        let action_auth = ControlTokenActionParser::parse(text_auth).expect("action parsed");
        assert_eq!(action_auth.action_type, "CREATOR_AUTH");
        assert_eq!(action_auth.target, "ed25519");
        assert_eq!(action_auth.raw_text, "<|creator_auth|>  ed25519<|im_end|>");

        let text_rule = "Rule check: <|tara_rule|>  safe_mode  <|im_end|>";
        let action_rule = ControlTokenActionParser::parse(text_rule).expect("action parsed");
        assert_eq!(action_rule.action_type, "RULE");
        assert_eq!(action_rule.target, "safe_mode");
        assert_eq!(action_rule.raw_text, "<|tara_rule|>  safe_mode  <|im_end|>");
    }

    #[test]
    fn test_parse_tool_without_json_payload_and_trailing_im_end() {
        let text = "<|tara_exec|> ping_service <|im_end|>";
        let action = ControlTokenActionParser::parse(text).expect("action parsed");
        assert_eq!(action.action_type, "TOOL");
        assert_eq!(action.target, "ping_service");
        assert_eq!(action.payload, Value::Null);
    }

    #[test]
    fn test_parse_empty_target_returns_none() {
        assert!(ControlTokenActionParser::parse("<|creator_auth|>").is_none());
        assert!(ControlTokenActionParser::parse("<|creator_auth|>   <|im_end|>").is_none());
        assert!(ControlTokenActionParser::parse("<|tara_exec|>").is_none());
    }
}
