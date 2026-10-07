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

    /// A declared dimension or numerical setting cannot form a valid model.
    #[error("invalid model configuration: {0}")]
    Invalid(String),
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
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.vocab_size == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.max_position_embeddings == 0
        {
            return Err(ConfigError::Invalid(
                "all model dimensions must be positive".into(),
            ));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(ConfigError::Invalid(
                "attention head count must be divisible by key/value head count".into(),
            ));
        }
        let head_dim = self.effective_head_dim();
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(ConfigError::Invalid(
                "effective head dimension must be positive and even for rotary embeddings".into(),
            ));
        }
        if !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.initializer_range.is_finite()
            || self.initializer_range <= 0.0
        {
            return Err(ConfigError::Invalid(
                "numerical hyperparameters must be finite and positive".into(),
            ));
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let cfg = TaraConfig::default();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.effective_head_dim(), 16);
    }

    #[test]
    fn test_zero_dimensions_rejected() {
        let cfg = TaraConfig {
            vocab_size: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());

        let cfg = TaraConfig {
            hidden_size: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_gqa_head_divisibility_enforced() {
        let cfg = TaraConfig {
            num_attention_heads: 5,
            num_key_value_heads: 2, // 5 is not divisible by 2
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_odd_head_dim_rejected_for_rope() {
        let cfg = TaraConfig {
            head_dim: 15, // odd head dim cannot support RoPE pair rotation
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_non_finite_hyperparameters_rejected() {
        let cfg = TaraConfig {
            rope_theta: f64::NAN,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());

        let cfg = TaraConfig {
            rms_norm_eps: -1e-5,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_json_roundtrip_preserves_configuration() {
        let cfg = TaraConfig::default();
        let json_val = cfg.to_json_value();
        let decoded: TaraConfig = serde_json::from_value(json_val).unwrap();
        assert_eq!(cfg.vocab_size, decoded.vocab_size);
        assert_eq!(cfg.hidden_size, decoded.hidden_size);
        assert_eq!(cfg.num_hidden_layers, decoded.num_hidden_layers);
    }
}
