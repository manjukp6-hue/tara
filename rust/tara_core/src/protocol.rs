//! protocol.rs
//! 
//! Standard JSON-RPC & Inter-Process Protocol between TARA Core and Neural Model.
//! Guarantees that the Python/SafeTensors Model cannot bypass Core security gates.

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
            request_id: format!("req_{}", rand::random::<u32>()),
            intent: intent.into(),
            actor_id: actor.into(),
            payload,
        }
    }
}
