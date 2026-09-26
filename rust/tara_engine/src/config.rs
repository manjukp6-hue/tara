//! Model configuration for TaraForCausalLM.
//!
//! Reads `config.json` from the model directory and exposes all architectural
//! hyper-parameters needed by the inference engine.

use serde::{Deserialize, Serialize};
use std::fs;
use thiserror::Error;

/// Errors that can occur when loading or serialising a [`TaraConfig`].
#[derive(Debug, Error)]
pub enum ConfigError {
    /// I/O failure while reading the config file.
    #[error("I/O error reading config: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parse failure.
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Full model configuration for TaraForCausalLM.
///
/// All fields are serialisable so the struct can be round-tripped to/from
/// `config.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaraConfig {
    /// Size of the token vocabulary.
    pub vocab_size: usize,

    /// Dimension of the hidden state (embedding size).
    pub hidden_size: usize,

    /// Inner dimension of the MLP feed-forward network.
    pub intermediate_size: usize,

    /// Number of transformer decoder layers.
    pub num_hidden_layers: usize,

    /// Number of query attention heads.
    pub num_attention_heads: usize,

    /// Number of key/value attention heads (GQA).
    pub num_key_value_heads: usize,

    /// Dimension per attention head (`hidden_size / num_attention_heads`).
    pub head_dim: usize,

    /// Maximum sequence length the model supports positionally.
    pub max_position_embeddings: usize,

    /// Base frequency for Rotary Position Embeddings.
    pub rope_theta: f64,

    /// Epsilon for RMSNorm numerical stability.
    pub rms_norm_eps: f64,

    /// Standard deviation for weight initialisation.
    pub initializer_range: f64,

    /// Human-readable version string (e.g. `"1.0.0"`).
    pub version: String,

    /// Canonical model type identifier.
    pub model_type: String,
}

impl Default for TaraConfig {
    fn default() -> Self {
        Self {
            vocab_size: 344,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            max_position_embeddings: 2048,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-5,
            initializer_range: 0.02,
            version: "1.0.0".to_string(),
            model_type: "tara".to_string(),
        }
    }
}

impl TaraConfig {
    /// Load configuration from a JSON file on disk.
    ///
    /// # Errors
    /// Returns [`ConfigError::Io`] if the file cannot be read, or
    /// [`ConfigError::Json`] if the JSON is malformed.
    pub fn from_json_file(path: &str) -> Result<Self, ConfigError> {
        let raw = fs::read_to_string(path)?;
        let cfg: Self = serde_json::from_str(&raw)?;
        Ok(cfg)
    }

    /// Serialise the configuration to a [`serde_json::Value`].
    ///
    /// Useful for embedding the config inside API responses.
    pub fn to_json_value(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("TaraConfig is always serialisable")
    }

    /// Derive `head_dim` from `hidden_size / num_attention_heads` if the stored
    /// value is 0, ensuring consistency.
    pub fn effective_head_dim(&self) -> usize {
        if self.head_dim > 0 {
            self.head_dim
        } else {
            self.hidden_size / self.num_attention_heads
        }
    }
}
