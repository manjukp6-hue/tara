//! nlu.rs
//!
//! Natural Language Understanding (NLU) & Slot Extraction for TARA Server.
//! Ports Python's nlu.py fully:
//! - Lexical intent parsing with paraphrase mapping.
//! - Deterministic safety interceptors (CSAM, content safety, deletion, mutation).
//! - Structured entity slot extraction: file paths, hashes, devices, datetimes, amounts.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Data Structures
// ---------------------------------------------------------------------------

/// A parsed intent result returned by SemanticIntentParser.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentResult {
    /// Top-level intent label (e.g. EXECUTE_TOOL, CONVERSATIONAL, GUARDED_ACTION).
    pub intent: String,
    /// Tool name when intent == EXECUTE_TOOL.
    pub tool: Option<String>,
    /// Skill name when intent == EXECUTE_SKILL.
    pub skill: Option<String>,
    /// Extracted entity slots.
    pub slots: HashMap<String, String>,
    /// Arbitrary parameters (may include JSON-valued fields encoded as strings).
    pub params: HashMap<String, serde_json::Value>,
    /// Confidence in [0.0, 1.0].
    pub confidence: f32,
    /// Raw query text (preserved for CONVERSATIONAL / COMPOSITE_PLAN).
    pub query: Option<String>,
    /// Action type for GUARDED_ACTION intents.
    pub action_type: Option<String>,
    /// Task type for EXECUTE_ENGINE intents.
    pub task_type: Option<String>,
    /// Learning mode for AUTONOMOUS_LEARNING.
    pub mode: Option<String>,
}

impl IntentResult {
    fn new(intent: impl Into<String>) -> Self {
        Self {
            intent: intent.into(),
            tool: None,
            skill: None,
            slots: HashMap::new(),
            params: HashMap::new(),
            confidence: 1.0,
            query: None,
            action_type: None,
            task_type: None,
            mode: None,
        }
    }
}

// ---------------------------------------------------------------------------
// SlotExtractor
// ---------------------------------------------------------------------------

/// Extracts structured entity slots from raw natural-language text.
pub struct SlotExtractor;

impl SlotExtractor {
    /// Extracts a file or directory path from text.
    pub fn extract_file_path(text: &str) -> Option<String> {
        // 1. Quoted paths
        let quoted_re =
            Regex::new(r#"["']([^"']+\.[a-zA-Z0-9_-]+|[^"']*[/\\][^"']*?)["']"#).ok()?;
        if let Some(cap) = quoted_re.captures(text) {
            return Some(cap[1].trim().to_owned());
        }

        // 2. Known-extension file paths
        let ext_re = Regex::new(
            r"(?i)([a-zA-Z0-9_\-./\\]+\.(?:json|py|txt|log|md|safetensors|db|csv|yaml|yml|gcode|xyz))\b",
        )
        .ok()?;
        if let Some(cap) = ext_re.captures(text) {
            return Some(cap[1].trim().to_owned());
        }

        // 3. Path with directory separators
        let path_re = Regex::new(r"([a-zA-Z0-9_\-]+[/\\][a-zA-Z0-9_\-./\\]+)").ok()?;
        if let Some(cap) = path_re.captures(text) {
            return Some(cap[1].trim().to_owned());
        }

        None
    }

    /// Extracts a cryptographic hex digest (MD5=32, SHA1=40, SHA256=64 chars).
    pub fn extract_hash(text: &str) -> Option<String> {
        let re = Regex::new(r"\b([a-fA-F0-9]{64}|[a-fA-F0-9]{40}|[a-fA-F0-9]{32})\b").ok()?;
        re.captures(text).map(|c| c[1].to_owned())
    }

    /// Extracts a device identifier (e.g. TARA-DEVICE-002, LFAM-3D-PRINTER).
    pub fn extract_device(text: &str) -> Option<String> {
        let re = Regex::new(
            r"(?i)\b(TARA-DEVICE-[a-zA-Z0-9_\-]+|LFAM-[a-zA-Z0-9_\-]+|DEVICE-[a-zA-Z0-9_\-]+)\b",
        )
        .ok()?;
        re.captures(text).map(|c| c[1].to_owned())
    }

    /// Extracts an ISO datetime string or relative date keyword.
    pub fn extract_datetime(text: &str) -> Option<String> {
        let iso_re = Regex::new(r"\b\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}(?::\d{2})?)?\b").ok()?;
        if let Some(m) = iso_re.find(text) {
            return Some(m.as_str().to_owned());
        }
        let rel_re = Regex::new(r"(?i)\b(today|yesterday|tomorrow|now)\b").ok()?;
        rel_re.captures(text).map(|c| c[1].to_lowercase())
    }

    /// Extracts all structured entity slots from text.
    pub fn extract_all_slots(text: &str) -> HashMap<String, String> {
        let mut map = HashMap::new();
        if let Some(v) = Self::extract_file_path(text) {
            map.insert("file_path".to_owned(), v);
        }
        if let Some(v) = Self::extract_hash(text) {
            map.insert("hash".to_owned(), v);
        }
        if let Some(v) = Self::extract_device(text) {
            map.insert("device_id".to_owned(), v);
        }
        if let Some(v) = Self::extract_datetime(text) {
            map.insert("datetime".to_owned(), v);
        }
        map
    }
}

// ---------------------------------------------------------------------------
// SemanticIntentParser
// ---------------------------------------------------------------------------

/// Parses natural-language inputs into structured [`IntentResult`] values.
pub struct SemanticIntentParser {
    pub available_skills: Vec<String>,
}

impl SemanticIntentParser {
    /// Creates a new parser, optionally pre-seeded with known skill names.
    pub fn new(available_skills: Vec<String>) -> Self {
        Self { available_skills }
    }

    /// Updates the available skill list (called after skill engine initialises).
    pub fn set_available_skills(&mut self, skills: Vec<String>) {
        self.available_skills = skills;
    }

    /// Parses `text` into an [`IntentResult`].
    ///
    /// `context` may supply an `intent_override` key to bypass normal parsing.
    pub fn parse(
        &self,
        text: &str,
        context: Option<&HashMap<String, serde_json::Value>>,
    ) -> IntentResult {
        let text_clean = text.trim();
        let text_lower = text_clean.to_lowercase();
        let ctx: HashMap<String, serde_json::Value> = context.cloned().unwrap_or_default();
        let slots = SlotExtractor::extract_all_slots(text_clean);

        // ------------------------------------------------------------------
        // 0. Context intent_override
        // ------------------------------------------------------------------
        if let Some(serde_json::Value::Object(ov)) = ctx.get("intent_override") {
            let intent_str = ov
                .get("intent")
                .and_then(|v| v.as_str())
                .unwrap_or("CONVERSATIONAL")
                .to_owned();
            let mut res = IntentResult::new(intent_str);
            if let Some(v) = ov.get("tool").and_then(|v| v.as_str()) {
                res.tool = Some(v.to_owned());
            }
            if let Some(v) = ov.get("skill").and_then(|v| v.as_str()) {
                res.skill = Some(v.to_owned());
            }
            return res;
        }

        // ------------------------------------------------------------------
        // 0b. Dynamic Engine Dispatch
        // ------------------------------------------------------------------
        if text_lower.starts_with("execute engine") || text_lower.starts_with("run engine") {
            let parts: Vec<&str> = text_clean.splitn(3, char::is_whitespace).collect();
            let task_type = parts.get(2).copied().unwrap_or("").trim().to_owned();
            let mut res = IntentResult::new("EXECUTE_ENGINE");
            res.task_type = Some(task_type);
            return res;
        }

        // ------------------------------------------------------------------
        // 1. COMPOSITE MULTI-STEP INTENT CHECK
        // ------------------------------------------------------------------
        let composite_patterns: &[&str] = &[
            r"(?i)\b(find|inspect|get|read)\b.+(?:,\s*|\s+and\s+|\s+then\s+)(?:calculate|compute|verify)\b.+(?:,\s*|\s+and\s+|\s+then\s+)(?:check|compare|diagnostics|telemetry)\b",
            r"(?i)\b(find|inspect|get|read)\b.+\b(calculate|compute|verify)\b.+\b(diagnostics|telemetry|compare|check)\b",
            r"(?i)\bfirst\b.+\bthen\b.+\bfinally\b",
            r"(?i)\bstep 1\b.+\bstep 2\b",
            r"(?i)\b(inspect|read)\b.+\band\b.+\b(calculate|compute)\b.+\bhash\b",
            r"(?i)\b(calculate|compute)\b.+\bhash\b.+\band\b.+\bcompare\b",
        ];
        for pattern in composite_patterns {
            if let Ok(re) = Regex::new(pattern) {
                if re.is_match(text_lower.as_str()) {
                    let mut res = IntentResult::new("COMPOSITE_PLAN");
                    res.query = Some(text_clean.to_owned());
                    res.slots = slots;
                    return res;
                }
            }
        }

        // ------------------------------------------------------------------
        // 2. SAFETY INTERCEPTORS (FAIL CLOSED / INVARIANTS)
        // ------------------------------------------------------------------
        // CSAM
        if ["child abuse", "csam", "child sexual"]
            .iter()
            .any(|&kw| text_lower.contains(kw))
        {
            let mut res = IntentResult::new("GUARDED_ACTION");
            res.action_type = Some("display_media".to_owned());
            res.params.insert(
                "media_type".to_owned(),
                serde_json::Value::String("csam_content".to_owned()),
            );
            res.params.insert(
                "topic".to_owned(),
                serde_json::Value::String(text_clean.to_owned()),
            );
            return res;
        }

        // Nudity / explicit
        if [
            "nude photo",
            "nude video",
            "porn",
            "naked photo",
            "naked video",
        ]
        .iter()
        .any(|&kw| text_lower.contains(kw))
        {
            let mut res = IntentResult::new("GUARDED_ACTION");
            res.action_type = Some("display_media".to_owned());
            res.params.insert(
                "media_type".to_owned(),
                serde_json::Value::String("nude_photo".to_owned()),
            );
            return res;
        }

        // Guarded Deletion
        let deletion_words = ["delete", "remove", "erase", "purge"];
        let fs_words = ["file", "folder", "directory", "database", "db"];
        if deletion_words.iter().any(|&w| text_lower.contains(w))
            && fs_words.iter().any(|&w| text_lower.contains(w))
        {
            let file_target = slots
                .get("file_path")
                .cloned()
                .or_else(|| {
                    ctx.get("file_path")
                        .and_then(|v| v.as_str())
                        .map(String::from)
                })
                .unwrap_or_else(|| text_clean.to_owned());
            let mut res = IntentResult::new("GUARDED_ACTION");
            res.action_type = Some("delete_file".to_owned());
            res.params.insert(
                "file_path".to_owned(),
                serde_json::Value::String(file_target),
            );
            res.params
                .insert("is_important".to_owned(), serde_json::Value::Bool(true));
            return res;
        }

        // Guarded Machine / Config mutation
        let mut_words = ["modify", "update", "change", "alter"];
        let hw_words = ["machine", "config", "setting", "hardware"];
        if mut_words.iter().any(|&w| text_lower.contains(w))
            && hw_words.iter().any(|&w| text_lower.contains(w))
        {
            let mut res = IntentResult::new("GUARDED_ACTION");
            res.action_type = Some("modify_machine_config".to_owned());
            return res;
        }

        // Guarded Credential / Key Export
        let export_words = ["export", "dump", "backup", "reveal", "read"];
        let secret_words = ["private key", "private_key", "keystore", "secret", "seed"];
        if export_words.iter().any(|&w| text_lower.contains(w))
            && secret_words.iter().any(|&s| text_lower.contains(s))
        {
            let mut res = IntentResult::new("GUARDED_ACTION");
            res.action_type = Some("export_key".to_owned());
            return res;
        }

        // Source Code Evolution (Autonomous code modification)
        if text_lower.contains("add source function")
            || text_lower.contains("add function to source")
        {
            let mut res = IntentResult::new("SOURCE_CODE_EVOLUTION");
            res.action_type = Some("add_source_function".to_owned());
            if let Some(f) = ctx.get("file_path").and_then(|v| v.as_str()) {
                res.params.insert(
                    "target_file".to_string(),
                    serde_json::Value::String(f.to_string()),
                );
            }
            if let Some(fn_name) = ctx.get("function_name").and_then(|v| v.as_str()) {
                res.params.insert(
                    "function_name".to_string(),
                    serde_json::Value::String(fn_name.to_string()),
                );
            }
            if let Some(fn_code) = ctx.get("function_code").and_then(|v| v.as_str()) {
                res.params.insert(
                    "function_code".to_string(),
                    serde_json::Value::String(fn_code.to_string()),
                );
            }
            return res;
        }

        if text_lower.contains("remove source function")
            || text_lower.contains("remove function from source")
        {
            let mut res = IntentResult::new("SOURCE_CODE_EVOLUTION");
            res.action_type = Some("remove_source_function".to_owned());
            if let Some(f) = ctx.get("file_path").and_then(|v| v.as_str()) {
                res.params.insert(
                    "target_file".to_string(),
                    serde_json::Value::String(f.to_string()),
                );
            }
            if let Some(fn_name) = ctx.get("function_name").and_then(|v| v.as_str()) {
                res.params.insert(
                    "function_name".to_string(),
                    serde_json::Value::String(fn_name.to_string()),
                );
            }
            return res;
        }

        if text_lower.contains("update source function")
            || text_lower.contains("edit source function")
            || text_lower.contains("replace source function")
        {
            let mut res = IntentResult::new("SOURCE_CODE_EVOLUTION");
            res.action_type = Some("update_source_function".to_owned());
            for (slot, context_key) in [
                ("target_file", "file_path"),
                ("function_name", "function_name"),
                ("previous_function_code", "previous_function_code"),
                ("function_code", "function_code"),
            ] {
                if let Some(value) = ctx.get(context_key).and_then(|value| value.as_str()) {
                    res.params
                        .insert(slot.to_owned(), serde_json::Value::String(value.to_owned()));
                }
            }
            return res;
        }

        // ------------------------------------------------------------------
        // 3. CONTINUOUS LEARNING TRIGGERS
        // ------------------------------------------------------------------
        if text_lower.contains("learn skill")
            || text_lower.contains("add skill")
            || text_lower.contains("teach tara a skill")
            || text_lower.contains("remove learned skill")
            || text_lower.contains("delete learned skill")
        {
            let mut res = IntentResult::new("LEARN_SKILL");
            let remove = text_lower.contains("remove learned skill")
                || text_lower.contains("delete learned skill");
            res.action_type = Some(if remove { "remove" } else { "add" }.to_owned());
            let from_context = ctx
                .get("skill_name")
                .and_then(|value| value.as_str())
                .map(str::to_owned);
            let from_text = Regex::new(
                r"(?i)(?:learn|add|remove|delete)\s+(?:a\s+)?(?:learned\s+)?skill\s+([a-z0-9_.-]+)|teach\s+tara\s+a?\s*skill\s+([a-z0-9_.-]+)",
            )
            .ok()
            .and_then(|regex| regex.captures(text_clean))
            .and_then(|capture| capture.get(1).or_else(|| capture.get(2)))
            .map(|capture| capture.as_str().to_owned());
            if let Some(name) = from_context.or(from_text) {
                res.params
                    .insert("skill_name".into(), serde_json::Value::String(name));
            }
            if let Some(definition) = ctx.get("skill_definition") {
                res.params.insert("definition".into(), definition.clone());
            }
            return res;
        }

        if [
            "autonomous learning",
            "learn in background",
            "background learning",
        ]
        .iter()
        .any(|&kw| text_lower.contains(kw))
        {
            let mut res = IntentResult::new("AUTONOMOUS_LEARNING");
            res.mode = Some(
                ctx.get("mode")
                    .and_then(|v| v.as_str())
                    .unwrap_or("DURATION_1_HOUR")
                    .to_owned(),
            );
            return res;
        }

        if ["learn online", "search web", "research topic", "web search"]
            .iter()
            .any(|&kw| text_lower.contains(kw))
        {
            let strip_re =
                Regex::new(r"(?i)(learn online|search web|research topic|web search)\s*:?")
                    .unwrap_or_else(|_| Regex::new(r"^$").unwrap());
            let query = strip_re.replace_all(text_clean, "").trim().to_owned();
            let query = if query.is_empty() {
                text_clean.to_owned()
            } else {
                query
            };
            let mut res = IntentResult::new("ONLINE_LEARNING");
            res.query = Some(query);
            return res;
        }

        // ------------------------------------------------------------------
        // 4. PARAPHRASED TOOL INTENTS
        // ------------------------------------------------------------------

        // -- file_inspector --
        let file_inspector_patterns: &[&str] = &[
            r"(?i)\b(inspect_file|file_inspector)\b",
            r"(?i)\b(check|see|look|examine)\s+(what is|what's)\s+inside\b",
            r"(?i)\b(inspect|examine|inspect contents of|view lines in|read)\s+(?:the\s+)?file\b",
            r"(?i)\bcheck\s+if\s+(?:the\s+)?file\s+exists\b",
            r"(?i)\bverify\s+(?:the\s+)?(?:contents|existence|size)\s+of\b",
            r"(?i)\b(check|see|count)\s+(?:how many\s+)?lines\b",
        ];
        if file_inspector_patterns.iter().any(|pat| {
            Regex::new(pat)
                .map(|re| re.is_match(text_lower.as_str()))
                .unwrap_or(false)
        }) {
            let target_path = slots
                .get("file_path")
                .cloned()
                .or_else(|| {
                    ctx.get("file_path")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_owned())
                })
                .unwrap_or_default();
            let mut res = IntentResult::new("EXECUTE_TOOL");
            res.tool = Some("file_inspector".to_owned());
            res.params.insert(
                "file_path".to_owned(),
                serde_json::Value::String(target_path),
            );
            return res;
        }

        // -- hash_verifier --
        let hash_patterns: &[&str] = &[
            r"(?i)\b(hash_verifier|verify_hash|compute_hash)\b",
            r"(?i)\b(calculate|compute|get|generate)\b.*?\b(?:sha256|hash|checksum|digest)\b",
            r"(?i)\b(verify|check)\b.*?\b(?:integrity|hash|checksum|digest)\b",
            r"(?i)\bchecksum\s+of\b",
        ];
        if hash_patterns.iter().any(|pat| {
            Regex::new(pat)
                .map(|re| re.is_match(text_lower.as_str()))
                .unwrap_or(false)
        }) {
            let target_path = slots.get("file_path").cloned().or_else(|| {
                ctx.get("file_path")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_owned())
            });
            let target_hash = slots.get("hash").cloned().or_else(|| {
                ctx.get("expected_hash")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_owned())
            });
            let mut res = IntentResult::new("EXECUTE_TOOL");
            res.tool = Some("hash_verifier".to_owned());
            if let Some(fp) = target_path {
                res.params
                    .insert("file_path".to_owned(), serde_json::Value::String(fp));
            }
            if let Some(eh) = target_hash {
                res.params
                    .insert("expected_hash".to_owned(), serde_json::Value::String(eh));
            }
            if res.params.is_empty() {
                res.params.insert(
                    "data".to_owned(),
                    serde_json::Value::String(text_clean.to_owned()),
                );
            }
            return res;
        }

        // -- knowledge_retriever --
        let knowledge_patterns: &[&str] = &[
            r"(?i)\b(knowledge_retriever|search_knowledge)\b",
            r"(?i)\b(search|query)\s+(?:the\s+)?knowledge\s*(?:base)?\b",
            r"(?i)\b(look up|find)\s+(?:verified\s+)?(?:facts?|information)\s+about\b",
        ];
        if knowledge_patterns.iter().any(|pat| {
            Regex::new(pat)
                .map(|re| re.is_match(text_lower.as_str()))
                .unwrap_or(false)
        }) {
            let strip_re = Regex::new(
                r"(?i)(knowledge_retriever|search knowledge base|search knowledge|find verified facts about)\s*:?",
            )
            .unwrap_or_else(|_| Regex::new(r"^$").unwrap());
            let q_text = strip_re.replace_all(text_clean, "").trim().to_owned();
            let q_text = if q_text.is_empty() {
                text_clean.to_owned()
            } else {
                q_text
            };
            let mut res = IntentResult::new("EXECUTE_TOOL");
            res.tool = Some("knowledge_retriever".to_owned());
            res.params
                .insert("query".to_owned(), serde_json::Value::String(q_text));
            return res;
        }

        // -- provenance_tracker --
        if [
            "provenance_tracker",
            "record provenance",
            "audit trail",
            "track provenance",
        ]
        .iter()
        .any(|&kw| text_lower.contains(kw))
        {
            let mut res = IntentResult::new("EXECUTE_TOOL");
            res.tool = Some("provenance_tracker".to_owned());
            res.params.insert(
                "action".to_owned(),
                serde_json::Value::String("custom_audit".to_owned()),
            );
            res.params.insert(
                "target".to_owned(),
                serde_json::Value::String(text_clean.to_owned()),
            );
            return res;
        }

        // ------------------------------------------------------------------
        // 5. SEMANTIC SKILL TRIGGERS
        // ------------------------------------------------------------------

        // diagnostics
        if [
            "diagnostics",
            "system telemetry",
            "telemetry",
            "system health",
            "cpu load",
            "ram usage",
            "check cpu",
        ]
        .iter()
        .any(|&kw| text_lower.contains(kw))
        {
            let mut res = IntentResult::new("EXECUTE_SKILL");
            res.skill = Some("diagnostics".to_owned());
            return res;
        }

        // developer (syntax check)
        if [
            "check syntax",
            "lint code",
            "validate python",
            "syntax check",
        ]
        .iter()
        .any(|&kw| text_lower.contains(kw))
        {
            let code = ctx
                .get("code")
                .and_then(|v| v.as_str())
                .unwrap_or(text_clean)
                .to_owned();
            let mut res = IntentResult::new("EXECUTE_SKILL");
            res.skill = Some("developer".to_owned());
            res.params
                .insert("code".to_owned(), serde_json::Value::String(code));
            return res;
        }

        // device / G-code / 3D printer
        if ["gcode", "3d printer", "lfam", "printer kinematics"]
            .iter()
            .any(|&kw| text_lower.contains(kw))
        {
            let gcode = ctx
                .get("gcode")
                .and_then(|v| v.as_str())
                .unwrap_or(text_clean)
                .to_owned();
            let mut res = IntentResult::new("EXECUTE_SKILL");
            res.skill = Some("device".to_owned());
            res.params
                .insert("gcode".to_owned(), serde_json::Value::String(gcode));
            return res;
        }

        // translation
        if ["translate", "translation", "in kannada", "glossary"]
            .iter()
            .any(|&kw| text_lower.contains(kw))
        {
            let strip_re = Regex::new(r"(?i)(translate|in kannada|translation)\s*:?")
                .unwrap_or_else(|_| Regex::new(r"^$").unwrap());
            let word_target = strip_re.replace_all(text_clean, "").trim().to_owned();
            let word_target = if word_target.is_empty() {
                text_clean.to_owned()
            } else {
                word_target
            };
            let mut res = IntentResult::new("EXECUTE_SKILL");
            res.skill = Some("translation".to_owned());
            res.params
                .insert("text".to_owned(), serde_json::Value::String(word_target));
            return res;
        }

        // Dynamic skill match against registered skills
        for sk in &self.available_skills {
            let pattern = format!(
                r"(?i)\b{}\b",
                regex::escape(&sk.to_lowercase().replace('_', " "))
            );
            if Regex::new(&pattern)
                .map(|re| re.is_match(text_lower.as_str()))
                .unwrap_or(false)
                || text_lower.contains(sk.to_lowercase().as_str())
            {
                let mut res = IntentResult::new("EXECUTE_SKILL");
                res.skill = Some(sk.clone());
                res.params.insert(
                    "input".to_owned(),
                    serde_json::Value::String(text_clean.to_owned()),
                );
                return res;
            }
        }

        // Audio / Video / PDF fallback keywords
        if ["wav", "sound", "audio"]
            .iter()
            .any(|&kw| text_lower.contains(kw))
        {
            let fp = slots
                .get("file_path")
                .cloned()
                .unwrap_or_else(|| text_clean.to_owned());
            let mut res = IntentResult::new("EXECUTE_SKILL");
            res.skill = Some("audio".to_owned());
            res.params
                .insert("file_path".to_owned(), serde_json::Value::String(fp));
            return res;
        }
        if ["mp4", "video"].iter().any(|&kw| text_lower.contains(kw)) {
            let fp = slots
                .get("file_path")
                .cloned()
                .unwrap_or_else(|| text_clean.to_owned());
            let mut res = IntentResult::new("EXECUTE_SKILL");
            res.skill = Some("video".to_owned());
            res.params
                .insert("file_path".to_owned(), serde_json::Value::String(fp));
            return res;
        }
        if text_lower.contains("pdf") {
            let fp = slots
                .get("file_path")
                .cloned()
                .unwrap_or_else(|| text_clean.to_owned());
            let mut res = IntentResult::new("EXECUTE_SKILL");
            res.skill = Some("pdf".to_owned());
            res.params
                .insert("file_path".to_owned(), serde_json::Value::String(fp));
            return res;
        }

        // ------------------------------------------------------------------
        // 7. DEFAULT: CONVERSATIONAL / FACTUAL QUERY
        // ------------------------------------------------------------------
        let mut res = IntentResult::new("CONVERSATIONAL");
        res.query = Some(text_clean.to_owned());
        res.slots = slots;
        res
    }
}
