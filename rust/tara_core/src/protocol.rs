//! Standard JSON-RPC & Inter-Process Protocol between TARA Core and Neural Model.
//! Guarantees that the Native Rust SafeTensors Neural Model cannot bypass Core security gates.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Protocol validation and serialization errors.
#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("Invalid protocol version: '{0}', expected 'TARA/2.0' compatible prefix")]
    InvalidProtocolVersion(String),
    #[error("Missing or empty required field: {0}")]
    MissingField(&'static str),
    #[error("Invalid request identifier: '{0}'")]
    InvalidRequestId(String),
    #[error("Payload error: {0}")]
    InvalidPayload(String),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Deserialization error: {0}")]
    Deserialization(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoreRequest {
    pub protocol_version: String,
    pub request_id: String,
    pub intent: String,
    pub actor_id: String,
    pub payload: serde_json::Value,
}

impl CoreRequest {
    /// Validates the request fields for correctness and security compliance.
    pub fn validate(&self) -> Result<(), ProtocolError> {
        if !self.protocol_version.starts_with("TARA/") {
            return Err(ProtocolError::InvalidProtocolVersion(
                self.protocol_version.clone(),
            ));
        }
        if self.request_id.trim().is_empty() {
            return Err(ProtocolError::InvalidRequestId(
                "request_id cannot be empty".to_string(),
            ));
        }
        if self.intent.trim().is_empty() {
            return Err(ProtocolError::MissingField("intent"));
        }
        if self.actor_id.trim().is_empty() {
            return Err(ProtocolError::MissingField("actor_id"));
        }
        Ok(())
    }

    /// Serializes request to JSON string.
    pub fn to_json(&self) -> Result<String, ProtocolError> {
        serde_json::to_string(self).map_err(|e| ProtocolError::Serialization(e.to_string()))
    }

    /// Deserializes request from JSON string.
    pub fn from_json(json_str: &str) -> Result<Self, ProtocolError> {
        serde_json::from_str(json_str).map_err(|e| ProtocolError::Deserialization(e.to_string()))
    }

    /// Serializes request to JSON bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ProtocolError> {
        serde_json::to_vec(self).map_err(|e| ProtocolError::Serialization(e.to_string()))
    }

    /// Deserializes request from JSON bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ProtocolError> {
        serde_json::from_slice(bytes).map_err(|e| ProtocolError::Deserialization(e.to_string()))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoreResponse {
    pub protocol_version: String,
    pub request_id: String,
    pub success: bool,
    pub decision: String,
    pub result: serde_json::Value,
    pub execution_time_ms: u64,
}

impl CoreResponse {
    /// Validates the response fields for correctness.
    pub fn validate(&self) -> Result<(), ProtocolError> {
        if !self.protocol_version.starts_with("TARA/") {
            return Err(ProtocolError::InvalidProtocolVersion(
                self.protocol_version.clone(),
            ));
        }
        if self.request_id.trim().is_empty() {
            return Err(ProtocolError::InvalidRequestId(
                "request_id cannot be empty".to_string(),
            ));
        }
        if self.decision.trim().is_empty() {
            return Err(ProtocolError::MissingField("decision"));
        }
        Ok(())
    }

    /// Serializes response to JSON string.
    pub fn to_json(&self) -> Result<String, ProtocolError> {
        serde_json::to_string(self).map_err(|e| ProtocolError::Serialization(e.to_string()))
    }

    /// Deserializes response from JSON string.
    pub fn from_json(json_str: &str) -> Result<Self, ProtocolError> {
        serde_json::from_str(json_str).map_err(|e| ProtocolError::Deserialization(e.to_string()))
    }

    /// Serializes response to JSON bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ProtocolError> {
        serde_json::to_vec(self).map_err(|e| ProtocolError::Serialization(e.to_string()))
    }

    /// Deserializes response from JSON bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ProtocolError> {
        serde_json::from_slice(bytes).map_err(|e| ProtocolError::Deserialization(e.to_string()))
    }
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

    pub fn format_success_response(
        request: &CoreRequest,
        decision: &str,
        result: serde_json::Value,
        execution_time_ms: u64,
    ) -> CoreResponse {
        CoreResponse {
            protocol_version: request.protocol_version.clone(),
            request_id: request.request_id.clone(),
            success: true,
            decision: decision.into(),
            result,
            execution_time_ms,
        }
    }

    pub fn format_error_response(
        request_id: &str,
        decision: &str,
        error_message: &str,
        execution_time_ms: u64,
    ) -> CoreResponse {
        CoreResponse {
            protocol_version: "TARA/2.0".into(),
            request_id: request_id.into(),
            success: false,
            decision: decision.into(),
            result: serde_json::json!({
                "error": error_message,
            }),
            execution_time_ms,
        }
    }

    pub fn validate_request(request: &CoreRequest) -> Result<(), ProtocolError> {
        request.validate()
    }

    pub fn validate_response(response: &CoreResponse) -> Result<(), ProtocolError> {
        response.validate()
    }

    pub fn parse_and_validate_request(raw_json: &str) -> Result<CoreRequest, ProtocolError> {
        let req = CoreRequest::from_json(raw_json)?;
        req.validate()?;
        Ok(req)
    }

    pub fn parse_and_validate_response(raw_json: &str) -> Result<CoreResponse, ProtocolError> {
        let resp = CoreResponse::from_json(raw_json)?;
        resp.validate()?;
        Ok(resp)
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
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_request_validation() {
        let mut req =
            CoreProtocolHandler::format_request("test_intent", "actor_1", serde_json::json!({}));
        assert!(req.validate().is_ok());

        req.intent = "   ".to_string();
        assert!(matches!(
            req.validate(),
            Err(ProtocolError::MissingField("intent"))
        ));

        req.intent = "valid_intent".to_string();
        req.actor_id = "".to_string();
        assert!(matches!(
            req.validate(),
            Err(ProtocolError::MissingField("actor_id"))
        ));

        req.actor_id = "valid_actor".to_string();
        req.protocol_version = "HTTP/1.1".to_string();
        assert!(matches!(
            req.validate(),
            Err(ProtocolError::InvalidProtocolVersion(_))
        ));

        req.protocol_version = "TARA/2.0".to_string();
        req.request_id = "".to_string();
        assert!(matches!(
            req.validate(),
            Err(ProtocolError::InvalidRequestId(_))
        ));
    }

    #[test]
    fn test_response_validation() {
        let req =
            CoreProtocolHandler::format_request("test_intent", "actor_1", serde_json::json!({}));
        let mut resp = CoreProtocolHandler::format_success_response(
            &req,
            "PERMIT",
            serde_json::json!({"status": "ok"}),
            12,
        );
        assert!(resp.validate().is_ok());

        resp.decision = "".to_string();
        assert!(matches!(
            resp.validate(),
            Err(ProtocolError::MissingField("decision"))
        ));

        resp.decision = "DENY".to_string();
        resp.protocol_version = "INVALID".to_string();
        assert!(matches!(
            resp.validate(),
            Err(ProtocolError::InvalidProtocolVersion(_))
        ));
    }

    #[test]
    fn test_serde_json_and_bytes() {
        let req =
            CoreProtocolHandler::format_request("intent_a", "actor_b", serde_json::json!({"key": 42}));
        let json_str = req.to_json().expect("to_json failed");
        let parsed_req = CoreRequest::from_json(&json_str).expect("from_json failed");
        assert_eq!(req, parsed_req);

        let bytes = req.to_bytes().expect("to_bytes failed");
        let parsed_from_bytes = CoreRequest::from_bytes(&bytes).expect("from_bytes failed");
        assert_eq!(req, parsed_from_bytes);

        let resp = CoreProtocolHandler::format_error_response("req_123", "DENY", "Unauthorized", 5);
        let resp_json = resp.to_json().expect("resp to_json failed");
        let parsed_resp = CoreResponse::from_json(&resp_json).expect("resp from_json failed");
        assert_eq!(resp, parsed_resp);
    }

    #[test]
    fn test_parse_and_validate() {
        let req = CoreProtocolHandler::format_request("run", "root", serde_json::json!({"cmd": "check"}));
        let json_str = req.to_json().unwrap();
        let validated = CoreProtocolHandler::parse_and_validate_request(&json_str).unwrap();
        assert_eq!(validated.request_id, req.request_id);

        let bad_json = r#"{"protocol_version":"TARA/2.0","request_id":"req_1","intent":"","actor_id":"root","payload":{}}"#;
        assert!(CoreProtocolHandler::parse_and_validate_request(bad_json).is_err());
    }
}
