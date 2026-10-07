//! Learning subsystem: user search-based learning, autonomous online learning, and continual learning governance.

#[path = "../../../tara_training_system/autonomous_training/neural/autonomous_learner.rs"]
pub mod autonomous_learner;
pub mod capability_synthesizer;
#[path = "../../../tara_training_system/autonomous_training/neural/governor.rs"]
pub mod governor;
pub mod self_upgrade;
#[path = "../../../tara_training_system/autonomous_training/world_model/topic_researcher.rs"]
pub mod topic_researcher;

pub use autonomous_learner::{AutonomousOnlineLearner, LearningDurationMode};
pub use capability_synthesizer::CapabilitySynthesizer;
pub use governor::{
    AnchorMetric, ContinualLearningGovernor, CurriculumItem, CurriculumPriorityEngine,
    DegradationEvaluation,
};
pub use self_upgrade::SelfUpgradeEngine;
pub use topic_researcher::{
    AutonomousResearchEngine, ExperimentDesign, ExperimentExecutionResult, HypothesisEvaluation,
    ResearchConclusion, ResearchGoal, ResearchHypothesis, ResearchRecord, ResearchSourceEvidence,
    ScientificMetric, SourceComparisonReport, TopicResearcher,
};

use serde_json::{json, Value};

pub enum UserSearchLearningMode {
    Supervised,
    Autonomous,
    Reinforcement,
}

/// Learns from user search results by calling an HTTP search endpoint.
///
/// The endpoint is configured via the `TARA_SEARCH_ENDPOINT` environment variable.
/// The endpoint must accept a plain HTTP GET request with `?q=<query>` and return
/// a JSON object that contains either a `"results"` array or a `"text"` field.
/// Knowledge is stored in `storage/knowledge/` under the topic `"SEARCH_LEARNED"`.
pub struct UserSearchLearner {
    pub repo_root: String,
}

impl UserSearchLearner {
    pub fn new(repo_root: &str) -> Self {
        Self {
            repo_root: repo_root.to_string(),
        }
    }

    pub fn search_and_learn(&self, query: &str, user_id: &str) -> Value {
        if query.trim().is_empty() {
            return json!({"status":"ERROR","action":"ONLINE_LEARNING","error":"query is required"});
        }

        let endpoint = match std::env::var("TARA_SEARCH_ENDPOINT")
            .ok()
            .filter(|v| !v.trim().is_empty())
        {
            Some(ep) => ep,
            None => {
                return json!({
                    "status": "SKIPPED",
                    "action": "ONLINE_LEARNING",
                    "query": query,
                    "user_id": user_id,
                    "knowledge_updated": false,
                    "message": "TARA_SEARCH_ENDPOINT is not configured; online learning skipped"
                });
            }
        };

        // Strict Core Directive Rule 5 Enforcement: Zero External AI / Zero Cloud LLM / Zero External Agents
        let ep_lower = endpoint.to_lowercase();
        let forbidden_ai_keywords = [
            "openai", "anthropic", "gemini", "claude", "chatgpt", "perplexity",
            "huggingface.co/api", "cohere", "mistral.ai", "deepseek", "groq",
            "ai-agent", "agent-proxy", "llm-api"
        ];
        for kw in forbidden_ai_keywords {
            if ep_lower.contains(kw) {
                return json!({
                    "status": "FORBIDDEN",
                    "action": "ONLINE_LEARNING",
                    "query": query,
                    "user_id": user_id,
                    "error": format!(
                        "External AI provider/agent endpoint ('{}') is strictly forbidden under TARA Core Directive Rule 5.",
                        kw
                    ),
                    "knowledge_updated": false,
                });
            }
        }

        // Perform HTTP GET to the configured search endpoint.
        let response_text = match http_get_sync(&endpoint, query) {
            Ok(body) => body,
            Err(err) => {
                return json!({
                    "status": "ERROR",
                    "action": "ONLINE_LEARNING",
                    "query": query,
                    "user_id": user_id,
                    "provider": endpoint,
                    "knowledge_updated": false,
                    "error": err,
                });
            }
        };

        // Parse the response body as JSON if possible; otherwise treat as plain text.
        let learned_text = if let Ok(parsed) = serde_json::from_str::<Value>(&response_text) {
            // Extract the most useful text field from the JSON.
            parsed
                .get("results")
                .and_then(Value::as_array)
                .and_then(|arr| arr.first())
                .and_then(|first| first.get("snippet").or_else(|| first.get("text")))
                .and_then(Value::as_str)
                .map(|s| s.to_string())
                .or_else(|| {
                    parsed
                        .get("text")
                        .or_else(|| parsed.get("answer"))
                        .or_else(|| parsed.get("summary"))
                        .and_then(Value::as_str)
                        .map(|s| s.to_string())
                })
                .unwrap_or_else(|| response_text.chars().take(4096).collect())
        } else {
            response_text.chars().take(4096).collect::<String>()
        };

        if learned_text.trim().is_empty() {
            return json!({
                "status": "ERROR",
                "action": "ONLINE_LEARNING",
                "query": query,
                "user_id": user_id,
                "provider": endpoint,
                "knowledge_updated": false,
                "error": "Search endpoint returned an empty result",
            });
        }

        // Persist the learned text to the knowledge base directory as a JSON entry.
        let kb_dir = format!("{}/storage/knowledge", self.repo_root);
        let topic = format!("search_{}", sanitize_for_filename(query));
        let entry_path = format!("{}/{}.json", kb_dir, topic);
        let entry = json!({
            "topic": topic,
            "query": query,
            "user_id": user_id,
            "provider": endpoint,
            "learned_text": learned_text,
            "confidence": 0.70,
            "source": "USER_SEARCH",
            "timestamp": crate::now_iso(),
        });

        let write_ok = if let Ok(serialized) = serde_json::to_string_pretty(&entry) {
            let _ = std::fs::create_dir_all(&kb_dir);
            std::fs::write(&entry_path, serialized).is_ok()
        } else {
            false
        };

        json!({
            "status": if write_ok { "SUCCESS" } else { "ERROR" },
            "action": "ONLINE_LEARNING",
            "query": query,
            "user_id": user_id,
            "provider": endpoint,
            "topic": topic,
            "knowledge_updated": write_ok,
            "learned_bytes": learned_text.len(),
        })
    }
}

// ── Internal HTTP Client ──────────────────────────────────────────────────────

/// Minimal synchronous HTTP/1.1 GET using only `std::net::TcpStream`.
/// Sends `GET <path>?q=<query> HTTP/1.1` and reads the response body.
/// Supports http:// URLs only; HTTPS is not supported without a TLS library.
fn http_get_sync(base_url: &str, query: &str) -> Result<String, String> {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::time::Duration;

    // Parse URL using the `url` crate already in the dependency tree.
    let mut parsed =
        url::Url::parse(base_url).map_err(|e| format!("Invalid TARA_SEARCH_ENDPOINT URL: {e}"))?;

    if parsed.scheme() != "http" {
        return Err(format!(
            "TARA_SEARCH_ENDPOINT must use http:// scheme (got '{}'). \
             HTTPS requires a TLS library not included in this build.",
            parsed.scheme()
        ));
    }

    parsed.query_pairs_mut().append_pair("q", query);

    let host = parsed
        .host_str()
        .ok_or("TARA_SEARCH_ENDPOINT URL has no host")?;
    let port = parsed.port().unwrap_or(80);
    let path_and_query = format!(
        "{}{}",
        parsed.path(),
        parsed
            .query()
            .map(|q| format!("?{}", q))
            .unwrap_or_default()
    );

    let addr = format!("{}:{}", host, port);
    let mut stream = TcpStream::connect(&addr)
        .map_err(|e| format!("Cannot connect to search endpoint '{}': {e}", addr))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(15)))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(Duration::from_secs(10)))
        .map_err(|e| e.to_string())?;

    let request = format!(
        "GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nAccept: application/json\r\n\r\n",
        path_and_query, host
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|e| format!("Failed to send HTTP request: {e}"))?;

    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .map_err(|e| format!("Failed to read HTTP response: {e}"))?;

    let response = String::from_utf8_lossy(&raw).into_owned();

    // Split headers from body on the first blank line.
    if let Some(body_start) = response.find("\r\n\r\n") {
        let body = &response[body_start + 4..];
        if body.trim().is_empty() {
            return Err("Search endpoint returned an empty HTTP body".to_string());
        }
        // Check HTTP status line for non-2xx.
        let status_line = response.lines().next().unwrap_or("");
        let status_code: u16 = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        if !(200..300).contains(&status_code) {
            return Err(format!(
                "Search endpoint returned HTTP {}: {}",
                status_code,
                body.chars().take(256).collect::<String>()
            ));
        }
        Ok(body.to_string())
    } else {
        Err("HTTP response missing header/body separator".to_string())
    }
}

fn sanitize_for_filename(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect()
}
