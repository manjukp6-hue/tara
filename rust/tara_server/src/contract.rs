//! Canonical Contract Layer for TARA Server.
//!
//! Provides canonical data structures matching TARA/CONTRACTS/v1/schemas.json.

use serde::{Deserialize, Serialize};

pub const CANONICAL_PROTOCOL_VERSION: &str = "1.0.0";
pub const CANONICAL_MODEL_IDENTITY: &str = "TARA";
// NOTE: CANONICAL_MODEL_SHA256 and CANONICAL_PARAM_COUNT are intentionally removed.
// The 118,080-parameter smoke-test model they referenced has been deleted.
// The production TARA model SHA and parameter count are read at runtime from
// ModelRegistry::get_active_version_sha() and from TaraForCausalLM::total_params().
// Do NOT hardcode these values until a real production model is promoted.
pub const TARA_WORKER_TOKEN_HEADER: &str = "X-Tara-Worker-Token";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuthContext {
    pub actor_id: String,
    pub session_id: String,
    #[serde(default)]
    pub creator_verified: bool,
    #[serde(default)]
    pub client_ip: Option<String>,
}

impl Default for AuthContext {
    fn default() -> Self {
        Self {
            actor_id: "anonymous".to_string(),
            session_id: "default_session".to_string(),
            creator_verified: false,
            client_ip: None,
        }
    }
}

fn default_max_tokens() -> usize {
    150
}
fn default_temperature() -> f32 {
    0.7
}
fn default_top_k() -> usize {
    50
}
fn default_top_p() -> f32 {
    0.9
}
fn default_repetition_penalty() -> f32 {
    1.1
}
fn default_stop_tokens() -> Vec<String> {
    vec!["<|im_end|>".to_string(), "<|pad|>".to_string()]
}
fn default_model_identity() -> String {
    CANONICAL_MODEL_IDENTITY.to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalInferenceRequest {
    pub request_id: String,
    pub prompt: String,
    pub expected_model_checksum: String,
    #[serde(default)]
    pub job_id: Option<String>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    #[serde(default = "default_stop_tokens")]
    pub stop_tokens: Vec<String>,
    #[serde(default = "default_model_identity")]
    pub expected_model_identity: String,
    #[serde(default)]
    pub auth_context: Option<AuthContext>,
}

impl CanonicalInferenceRequest {
    pub fn new(request_id: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            prompt: prompt.into(),
            expected_model_checksum: String::new(),
            job_id: None,
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            top_k: default_top_k(),
            top_p: default_top_p(),
            repetition_penalty: default_repetition_penalty(),
            stop_tokens: default_stop_tokens(),
            expected_model_identity: CANONICAL_MODEL_IDENTITY.to_string(),
            auth_context: None,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.request_id.trim().is_empty() {
            return Err("Missing required request_id".to_string());
        }
        if self.prompt.trim().is_empty() {
            return Err("Missing required prompt".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalInferenceResponse {
    pub request_id: String,
    pub status: String,
    pub text: String,
    pub model_checksum: String,
    pub runtime_engine: String,
    #[serde(default)]
    pub token_count: usize,
    #[serde(default)]
    pub token_ids: Vec<u32>,
    #[serde(default = "default_model_identity")]
    pub model_identity: String,
    #[serde(default)]
    pub first_latency_ms: f64,
    #[serde(default)]
    pub total_latency_ms: f64,
    #[serde(default)]
    pub tokens_per_second: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHealthResponse {
    pub status: String,
    pub ready: bool,
    pub protocol_version: String,
    pub model_checksum: String,
    pub has_gpu: bool,
    #[serde(default)]
    pub worker_identity: String,
    #[serde(default = "default_model_identity")]
    pub model_identity: String,
    #[serde(default)]
    pub parameters: usize,
    #[serde(default)]
    pub gpu_device: Option<String>,
    #[serde(default)]
    pub device_type: String,
    #[serde(default)]
    pub shared_storage_verified: bool,
    #[serde(default)]
    pub queue_depth: usize,
    #[serde(default)]
    pub uptime_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalErrorResponse {
    pub request_id: String,
    pub status: String,
    pub error_code: String,
    pub message: String,
    pub retryable: bool,
    pub failover_advised: bool,
}

impl CanonicalErrorResponse {
    pub fn new(
        request_id: impl Into<String>,
        error_code: impl Into<String>,
        message: impl Into<String>,
        retryable: bool,
        failover_advised: bool,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            status: "ERROR".to_string(),
            error_code: error_code.into(),
            message: message.into(),
            retryable,
            failover_advised,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Voice Subsystem Contracts
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceTranscribeRequest {
    pub audio_data: String,
    #[serde(default = "default_voice_lang")]
    pub language: String,
}

fn default_voice_lang() -> String {
    "en-US".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceTranscribeResponse {
    pub status: String,
    pub transcript: String,
    pub language: String,
    pub confidence: f64,
    pub has_speech: bool,
    #[serde(default)]
    pub duration_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceSynthesizeRequest {
    pub text: String,
    #[serde(default = "default_voice_lang_short")]
    pub language: String,
    #[serde(default)]
    pub rate: Option<i32>,
    #[serde(default)]
    pub volume: Option<i32>,
}

fn default_voice_lang_short() -> String {
    "en".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConverseRequest {
    #[serde(default)]
    pub input_text: Option<String>,
    #[serde(default)]
    pub audio_data: Option<String>,
    #[serde(default = "default_voice_lang")]
    pub language: String,
    #[serde(default)]
    pub context: Option<serde_json::Value>,
    #[serde(default)]
    pub require_wake_word: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConverseResponse {
    pub status: String,
    pub transcript: String,
    pub response_text: String,
    pub audio_base64: String,
    pub language: String,
    pub turn_id: String,
    #[serde(default)]
    pub brain_result: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceBargeInInfo {
    pub interrupted: bool,
    pub total_interruptions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceBargeInResponse {
    pub status: String,
    pub barge_in: VoiceBargeInInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCapabilitiesResponse {
    pub status: String,
    pub stt_available: bool,
    pub tts_available: bool,
    pub languages: Vec<String>,
    pub barge_in_supported: bool,
    pub wake_words: Vec<String>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Creator Setup & Device Management Contracts
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatorSetupRequest {
    pub google_id_token: String,
    #[serde(default = "default_true")]
    pub confirm_identity: bool,
    #[serde(default = "default_primary_device_name")]
    pub device_name: String,
    #[serde(default)]
    pub device_public_key: Option<String>,
}

fn default_true() -> bool {
    true
}
fn default_primary_device_name() -> String {
    "Primary PC".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatorSetupResponse {
    pub status: String,
    pub authority_state: String,
    pub creator_id: String,
    pub display_name: String,
    #[serde(default)]
    pub recovery_code: Option<String>,
    #[serde(default)]
    pub device: Option<serde_json::Value>,
    #[serde(default)]
    pub session: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceRegisterRequest {
    pub device_public_key: String,
    #[serde(default)]
    pub session_token: Option<String>,
    #[serde(default)]
    pub google_id_token: Option<String>,
    #[serde(default = "default_device_name")]
    pub device_name: String,
}

fn default_device_name() -> String {
    "Secondary PC".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceRevokeRequest {
    #[serde(default)]
    pub device_id: Option<String>,
    #[serde(default)]
    pub session_token: Option<String>,
    #[serde(default)]
    pub all: bool,
    #[serde(default = "default_revoke_reason")]
    pub reason: String,
}

fn default_revoke_reason() -> String {
    "manual_revocation".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceListResponse {
    pub status: String,
    pub devices: Vec<serde_json::Value>,
}
