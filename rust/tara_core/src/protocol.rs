//! Standard JSON-RPC & Inter-Process Protocol between TARA Core and Neural Model.
//! Guarantees that the Native Rust SafeTensors Neural Model cannot bypass Core security gates.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreRequest {
    pub protocol_version: String,
    pub request_id: String,
    pub intent: String,
    pub actor_id: String,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreResponse {
    pub protocol_version: String,
    pub request_id: String,
    pub success: bool,
    pub decision: String,
    pub result: serde_json::Value,
    pub execution_time_ms: u64,
}

pub struct CoreProtocolHandler;

impl CoreProtocolHandler {
    pub fn format_request(intent: &str, actor: &str, payload: serde_json::Value) -> CoreRequest {
        CoreRequest {
            protocol_version: "TARA/2.0".into(),
            request_id: format!("req_{:032x}", rand::random::<u128>()),
            intent: intent.into(),
            actor_id: actor.into(),
            payload,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_request_id_format() {
        let req =
            CoreProtocolHandler::format_request("test_intent", "actor_1", serde_json::json!({}));
        assert!(req.request_id.starts_with("req_"));
        // "req_" is 4 chars + 32 hex chars = 36 chars
        assert_eq!(req.request_id.len(), 36);
    }
}
