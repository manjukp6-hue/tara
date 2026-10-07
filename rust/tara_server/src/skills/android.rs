//! Android Emulator QA skill native Rust implementations.
//! Replaces:
//! - TARA/SKILLS/android/android-emulator-qa/scripts/ui_pick.py
//! - TARA/SKILLS/android/android-emulator-qa/scripts/ui_tree_summarize.py

use serde_json::{json, Value};
use std::collections::HashMap;

/// Clean whitespace in text strings.
pub fn clean_text(s: Option<&str>) -> String {
    match s {
        Some(v) => {
            let parts: Vec<&str> = v.split_whitespace().collect();
            parts.join(" ")
        }
        None => String::new(),
    }
}

/// Represents an Android UI Node parsed from XML hierarchy dump.
#[derive(Debug, Clone)]
pub struct UiNode {
    pub tag: String,
    pub class_name: String,
    pub text: String,
    pub content_desc: String,
    pub resource_id: String,
    pub bounds: String,
    pub clickable: bool,
    pub long_clickable: bool,
    pub scrollable: bool,
    pub focusable: bool,
    pub checked: bool,
    pub selected: bool,
    pub children: Vec<UiNode>,
}

impl UiNode {
    pub fn is_interactive(&self) -> bool {
        self.clickable || self.long_clickable || self.scrollable || self.focusable
    }

    pub fn has_display(&self) -> bool {
        !self.text.is_empty() || !self.content_desc.is_empty() || !self.resource_id.is_empty()
    }

    pub fn should_keep(&self) -> bool {
        self.has_display() || self.is_interactive() || self.scrollable
    }

    pub fn simplify_class(&self) -> String {
        if self.class_name.is_empty() {
            return String::new();
        }
        self.class_name
            .split('.')
            .next_back()
            .unwrap_or("")
            .to_string()
    }

    pub fn simplify_resource_id(&self) -> String {
        if self.resource_id.is_empty() {
            return String::new();
        }
        if let Some(pos) = self.resource_id.find(":id/") {
            let (prefix, rest) = self.resource_id.split_at(pos);
            let id_part = &rest[4..];
            if !prefix.is_empty() && prefix != "android" {
                return format!("id/{}", id_part);
            }
        }
        self.resource_id.clone()
    }

    pub fn format_summary(&self) -> String {
        let mut parts = Vec::new();
        let class_s = self.simplify_class();
        if !class_s.is_empty() {
            parts.push(class_s);
        }
        let res_id = self.simplify_resource_id();
        if !res_id.is_empty() {
            parts.push(format!("id={}", res_id));
        }
        if !self.text.is_empty() {
            parts.push(format!("text=\"{}\"", self.text));
        }
        if !self.content_desc.is_empty() {
            parts.push(format!("desc=\"{}\"", self.content_desc));
        }

        let mut flags = Vec::new();
        if self.clickable {
            flags.push("clickable");
        }
        if self.long_clickable {
            flags.push("long-clickable");
        }
        if self.scrollable {
            flags.push("scrollable");
        }
        if self.focusable {
            flags.push("focusable");
        }
        if self.checked {
            flags.push("checked");
        }
        if self.selected {
            flags.push("selected");
        }

        if !flags.is_empty() {
            parts.push(format!("flags={}", flags.join(",")));
        }

        if !self.bounds.is_empty() && (self.is_interactive() || self.has_display()) {
            parts.push(format!("bounds={}", self.bounds));
        }

        parts.join(" ")
    }
}

/// Simple, robust XML hierarchy parser for Android dumps.
pub fn parse_android_hierarchy(xml_str: &str) -> Result<UiNode, String> {
    let trimmed = if let Some(end_idx) = xml_str.rfind("</hierarchy>") {
        &xml_str[..end_idx + "</hierarchy>".len()]
    } else {
        xml_str.trim()
    };

    // Parse root or mock tree from attributes
    let mut root = UiNode {
        tag: "hierarchy".to_string(),
        class_name: "android.widget.FrameLayout".to_string(),
        text: String::new(),
        content_desc: String::new(),
        resource_id: String::new(),
        bounds: "[0,0][1080,2400]".to_string(),
        clickable: false,
        long_clickable: false,
        scrollable: false,
        focusable: false,
        checked: false,
        selected: false,
        children: Vec::new(),
    };

    // Extract nodes by regex matching `<node ...>` tags
    let re = regex::Regex::new(r#"<node\s+([^>]+)>"#).map_err(|e| e.to_string())?;
    let attr_re = regex::Regex::new(r#"([a-zA-Z0-9_\-]+)="([^"]*)""#).map_err(|e| e.to_string())?;

    for cap in re.captures_iter(trimmed) {
        if let Some(attr_match) = cap.get(1) {
            let attr_str = attr_match.as_str();
            let mut attrs: HashMap<String, String> = HashMap::new();
            for a_cap in attr_re.captures_iter(attr_str) {
                let k = a_cap[1].to_string();
                let v = a_cap[2].to_string();
                attrs.insert(k, v);
            }

            let node = UiNode {
                tag: "node".to_string(),
                class_name: attrs.get("class").cloned().unwrap_or_default(),
                text: clean_text(attrs.get("text").map(|s| s.as_str())),
                content_desc: clean_text(attrs.get("content-desc").map(|s| s.as_str())),
                resource_id: attrs.get("resource-id").cloned().unwrap_or_default(),
                bounds: attrs.get("bounds").cloned().unwrap_or_default(),
                clickable: attrs.get("clickable").map(|s| s == "true").unwrap_or(false),
                long_clickable: attrs
                    .get("long-clickable")
                    .map(|s| s == "true")
                    .unwrap_or(false),
                scrollable: attrs
                    .get("scrollable")
                    .map(|s| s == "true")
                    .unwrap_or(false),
                focusable: attrs.get("focusable").map(|s| s == "true").unwrap_or(false),
                checked: attrs.get("checked").map(|s| s == "true").unwrap_or(false),
                selected: attrs.get("selected").map(|s| s == "true").unwrap_or(false),
                children: Vec::new(),
            };
            root.children.push(node);
        }
    }

    Ok(root)
}

/// Port of `ui_pick.py`: Finds target element center coordinates in XML dump.
pub fn ui_pick(xml_content: &str, target_text: &str) -> Result<(i32, i32), String> {
    let norm_target = clean_text(Some(target_text));
    if norm_target.is_empty() {
        return Err("Target text is empty".to_string());
    }

    let root = parse_android_hierarchy(xml_content)?;

    // Find matching node recursively
    fn find_match<'a>(node: &'a UiNode, target: &str) -> Option<&'a UiNode> {
        if node.text == target || node.content_desc == target {
            return Some(node);
        }
        for child in &node.children {
            if let Some(m) = find_match(child, target) {
                return Some(m);
            }
        }
        None
    }

    let matched = find_match(&root, &norm_target)
        .ok_or_else(|| format!("node not found: {}", norm_target))?;

    let bounds = &matched.bounds;
    let b_re = regex::Regex::new(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]").map_err(|e| e.to_string())?;
    let cap = b_re
        .captures(bounds)
        .ok_or_else(|| "bounds not found or invalid format".to_string())?;

    let x1: i32 = cap[1]
        .parse()
        .map_err(|e: std::num::ParseIntError| e.to_string())?;
    let y1: i32 = cap[2]
        .parse()
        .map_err(|e: std::num::ParseIntError| e.to_string())?;
    let x2: i32 = cap[3]
        .parse()
        .map_err(|e: std::num::ParseIntError| e.to_string())?;
    let y2: i32 = cap[4]
        .parse()
        .map_err(|e: std::num::ParseIntError| e.to_string())?;

    let cx = (x1 + x2) / 2;
    let cy = (y1 + y2) / 2;
    Ok((cx, cy))
}

/// Port of `ui_tree_summarize.py`: Summarizes UI hierarchy into compact indented tree.
pub fn ui_tree_summarize(xml_content: &str, max_depth: usize) -> Result<String, String> {
    let root = parse_android_hierarchy(xml_content)?;
    let mut lines = Vec::new();

    fn build_lines(node: &UiNode, depth: usize, max_d: usize, lines: &mut Vec<String>) {
        if depth > max_d {
            return;
        }
        let include = node.should_keep();
        let child_depth = if include { depth + 1 } else { depth };
        if include {
            let indent = "  ".repeat(depth);
            lines.push(format!("{}{}", indent, node.format_summary()));
        }
        for child in &node.children {
            build_lines(child, child_depth, max_d, lines);
        }
    }

    for child in &root.children {
        build_lines(child, 0, max_depth, &mut lines);
    }

    Ok(lines.join("\n"))
}

/// JSON handler wrapper for Android QA skills.
pub fn handle_android_skill(action: &str, params: Value) -> Value {
    match action {
        "ui_pick" => {
            let xml = params.get("xml").and_then(|v| v.as_str()).unwrap_or("");
            let target = params.get("target").and_then(|v| v.as_str()).unwrap_or("");
            match ui_pick(xml, target) {
                Ok((x, y)) => json!({ "status": "SUCCESS", "x": x, "y": y }),
                Err(e) => json!({ "status": "ERROR", "error": e }),
            }
        }
        "ui_tree_summarize" => {
            let xml = params.get("xml").and_then(|v| v.as_str()).unwrap_or("");
            let max_depth = params
                .get("max_depth")
                .and_then(|v| v.as_u64())
                .unwrap_or(20) as usize;
            match ui_tree_summarize(xml, max_depth) {
                Ok(summary) => json!({ "status": "SUCCESS", "summary": summary }),
                Err(e) => json!({ "status": "ERROR", "error": e }),
            }
        }
        _ => json!({ "status": "ERROR", "error": format!("Unknown android action: {}", action) }),
    }
}
