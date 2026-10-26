//! Multi-modal processing subsystem for TARA.
//!
//! Provides native cross-modal integration for vision (RGB patches) and audio
//! (Log-Mel Spectrogram features), projecting high-dimensional perception vectors
//! into unified continuous embeddings for `TaraForCausalLM`.

pub mod audio;
pub mod projector;
pub mod vision;

pub use audio::{AudioConfig, AudioEncoder, AudioError, RawAudio};
pub use projector::{CrossModalProjector, ProjectorType};
pub use vision::{RawImage, VisionConfig, VisionEncoder, VisionError};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::model::TaraForCausalLM;
use crate::tokenizer::TaraTokenizer;

/// Errors in multi-modal sequence composition.
#[derive(Debug, Error)]
pub enum MultiModalError {
    #[error("vision error: {0}")]
    Vision(#[from] VisionError),
    #[error("audio error: {0}")]
    Audio(#[from] AudioError),
    #[error("empty input sequence: at least text, image, or audio must be provided")]
    EmptyInput,
    #[error("sequence length ({length}) exceeds maximum limit ({max_len})")]
    SequenceTooLong { length: usize, max_len: usize },
}

/// Unified multi-modal input package.
#[derive(Debug, Clone, Default)]
pub struct MultiModalInput {
    pub text: Option<String>,
    pub image: Option<RawImage>,
    pub audio: Option<RawAudio>,
}

/// Dynamic configuration for the multi-modal engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalConfig {
    pub vision: VisionConfig,
    pub audio: AudioConfig,
    pub max_sequence_length: usize,
}

impl Default for MultiModalConfig {
    fn default() -> Self {
        Self {
            vision: VisionConfig::default(),
            audio: AudioConfig::default(),
            max_sequence_length: 4096,
        }
    }
}

/// Contiguous multi-modal embedding sequence ready for transformer forward passes.
#[derive(Debug, Clone)]
pub struct MultiModalSequence {
    /// Flattened continuous embeddings: shape [seq_len, hidden_size]
    pub embeddings: Vec<f32>,
    /// Number of tokens in the sequence
    pub seq_len: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Token counts per component for auditing
    pub text_token_count: usize,
    pub vision_patch_count: usize,
    pub audio_frame_count: usize,
}

/// Central MultiModal Engine orchestrating vision and audio projection into the neural model.
#[derive(Debug, Clone)]
pub struct MultiModalEngine {
    pub config: MultiModalConfig,
    pub vision_encoder: VisionEncoder,
    pub audio_encoder: AudioEncoder,
}

impl MultiModalEngine {
    /// Construct a new MultiModalEngine with dynamic configurations.
    pub fn new(config: MultiModalConfig) -> Self {
        let vision_encoder = VisionEncoder::new(config.vision.clone());
        let audio_encoder = AudioEncoder::new(config.audio.clone());
        Self {
            config,
            vision_encoder,
            audio_encoder,
        }
    }

    /// Assemble continuous multi-modal embeddings from text, image, and audio.
    ///
    /// Constructs an interleaved sequence:
    /// `[Vision Start Embedding] + [Vision Patches...] + [Vision End Embedding] +`
    /// `[Audio Start Embedding] + [Audio Frames...] + [Audio End Embedding] +`
    /// `[Text Token Embeddings...]`
    pub fn assemble_sequence(
        &self,
        model: &TaraForCausalLM,
        tokenizer: &TaraTokenizer,
        input: &MultiModalInput,
    ) -> Result<MultiModalSequence, MultiModalError> {
        let hs = model.config.hidden_size;
        let mut total_embeddings = Vec::new();
        let mut text_count = 0;
        let mut vision_count = 0;
        let mut audio_count = 0;

        // 1. Process Vision Input if present
        if let Some(ref image) = input.image {
            let (vision_embs, n_patches) = self.vision_encoder.encode_to_embeddings(image)?;
            if n_patches > 0 {
                // Prepend vision boundary start token
                let v_start_id = tokenizer.encode("<|vision_start|>");
                let v_start_token = v_start_id.first().copied().unwrap_or(0);
                let v_start_vec = &model.embed_tokens[(v_start_token as usize % model.config.vocab_size) * hs..((v_start_token as usize % model.config.vocab_size) + 1) * hs];
                total_embeddings.extend_from_slice(v_start_vec);

                // Add visual patch embeddings
                total_embeddings.extend_from_slice(&vision_embs);

                // Append vision boundary end token
                let v_end_id = tokenizer.encode("<|vision_end|>");
                let v_end_token = v_end_id.first().copied().unwrap_or(0);
                let v_end_vec = &model.embed_tokens[(v_end_token as usize % model.config.vocab_size) * hs..((v_end_token as usize % model.config.vocab_size) + 1) * hs];
                total_embeddings.extend_from_slice(v_end_vec);

                vision_count = n_patches + 2;
            }
        }

        // 2. Process Audio Input if present
        if let Some(ref audio) = input.audio {
            let (audio_embs, n_frames) = self.audio_encoder.encode_to_embeddings(audio)?;
            if n_frames > 0 {
                // Prepend audio boundary start token
                let a_start_id = tokenizer.encode("<|audio_start|>");
                let a_start_token = a_start_id.first().copied().unwrap_or(0);
                let a_start_vec = &model.embed_tokens[(a_start_token as usize % model.config.vocab_size) * hs..((a_start_token as usize % model.config.vocab_size) + 1) * hs];
                total_embeddings.extend_from_slice(a_start_vec);

                // Add audio feature embeddings
                total_embeddings.extend_from_slice(&audio_embs);

                // Append audio boundary end token
                let a_end_id = tokenizer.encode("<|audio_end|>");
                let a_end_token = a_end_id.first().copied().unwrap_or(0);
                let a_end_vec = &model.embed_tokens[(a_end_token as usize % model.config.vocab_size) * hs..((a_end_token as usize % model.config.vocab_size) + 1) * hs];
                total_embeddings.extend_from_slice(a_end_vec);

                audio_count = n_frames + 2;
            }
        }

        // 3. Process Text Input if present
        if let Some(ref text) = input.text {
            let token_ids = tokenizer.encode(text);
            for &id in &token_ids {
                let safe_id = (id as usize).min(model.config.vocab_size.saturating_sub(1));
                let token_vec = &model.embed_tokens[safe_id * hs..(safe_id + 1) * hs];
                total_embeddings.extend_from_slice(token_vec);
            }
            text_count = token_ids.len();
        }

        let seq_len = vision_count + audio_count + text_count;
        if seq_len == 0 {
            return Err(MultiModalError::EmptyInput);
        }

        if seq_len > self.config.max_sequence_length {
            return Err(MultiModalError::SequenceTooLong {
                length: seq_len,
                max_len: self.config.max_sequence_length,
            });
        }

        Ok(MultiModalSequence {
            embeddings: total_embeddings,
            seq_len,
            hidden_size: hs,
            text_token_count: text_count,
            vision_patch_count: vision_count,
            audio_frame_count: audio_count,
        })
    }

    /// Run full forward pass across multi-modal sequence returning output logits of shape [seq_len, vocab_size].
    pub fn forward(
        &self,
        model: &TaraForCausalLM,
        tokenizer: &TaraTokenizer,
        input: &MultiModalInput,
    ) -> Result<Vec<f32>, MultiModalError> {
        let sequence = self.assemble_sequence(model, tokenizer, input)?;
        let logits = model.forward_embedded(&sequence.embeddings, sequence.seq_len);
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TaraConfig;
    use std::collections::HashMap;

    #[test]
    fn test_vision_encoder_patchification_and_projection() {
        let config = VisionConfig {
            patch_height: 8,
            patch_width: 8,
            channels: 3,
            hidden_size: 16,
            projector_type: ProjectorType::Mlp,
            intermediate_dim: 32,
            normalize_pixels: true,
            mean: vec![0.5, 0.5, 0.5],
            std: vec![0.5, 0.5, 0.5],
        };
        let encoder = VisionEncoder::new(config);

        // 16x16 RGB image -> 4 patches of size 8x8
        let img_bytes = vec![128u8; 16 * 16 * 3];
        let raw_img = RawImage::from_u8_slice(16, 16, 3, &img_bytes).expect("valid image");

        let (embeddings, num_patches) = encoder.encode_to_embeddings(&raw_img).expect("encode succeeds");
        assert_eq!(num_patches, 4);
        assert_eq!(embeddings.len(), 4 * 16);
    }

    #[test]
    fn test_audio_encoder_stft_and_projection() {
        let config = AudioConfig {
            sample_rate: 16000,
            n_fft: 256,
            hop_length: 128,
            n_mels: 32,
            f_min: 0.0,
            f_max: Some(8000.0),
            preemphasis: 0.97,
            hidden_size: 16,
            intermediate_dim: 32,
            projector_type: ProjectorType::Linear,
        };
        let encoder = AudioEncoder::new(config);

        // 512 samples -> 3 frames with hop 128
        let samples: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin()).collect();
        let raw_audio = RawAudio::from_f32(samples, 16000);

        let (embeddings, num_frames) = encoder.encode_to_embeddings(&raw_audio).expect("encode succeeds");
        assert_eq!(num_frames, 3);
        assert_eq!(embeddings.len(), 3 * 16);
    }

    #[test]
    fn test_multimodal_end_to_end_forward() {
        let tara_config = TaraConfig {
            vocab_size: 16,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            max_position_embeddings: 64,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            initializer_range: 0.02,
            version: "multimodal_test".into(),
            model_type: "tara".into(),
            tie_word_embeddings: None,
        };

        let mut weights = HashMap::<String, Vec<f32>>::new();
        weights.insert("model.embed_tokens.weight".into(), vec![0.1; 16 * 16]);
        weights.insert("model.norm.weight".into(), vec![1.0; 16]);
        weights.insert("lm_head.weight".into(), vec![0.5; 16 * 16]);
        weights.insert("model.layers.0.input_layernorm.weight".into(), vec![1.0; 16]);
        weights.insert("model.layers.0.post_attention_layernorm.weight".into(), vec![1.0; 16]);
        weights.insert("model.layers.0.self_attn.q_proj.weight".into(), vec![0.1; 16 * 16]);
        weights.insert("model.layers.0.self_attn.k_proj.weight".into(), vec![0.1; 8 * 16]);
        weights.insert("model.layers.0.self_attn.v_proj.weight".into(), vec![0.1; 8 * 16]);
        weights.insert("model.layers.0.self_attn.o_proj.weight".into(), vec![0.1; 16 * 16]);
        weights.insert("model.layers.0.mlp.gate_proj.weight".into(), vec![0.1; 32 * 16]);
        weights.insert("model.layers.0.mlp.up_proj.weight".into(), vec![0.1; 32 * 16]);
        weights.insert("model.layers.0.mlp.down_proj.weight".into(), vec![0.1; 16 * 32]);

        let model = TaraForCausalLM::from_weights_move(weights, &tara_config, "test_dir")
            .expect("model initialized");

        let tokenizer_json = r#"{
            "vocab": {
                "<|unk|>": 0,
                "<|vision_start|>": 1,
                "<|vision_end|>": 2,
                "<|audio_start|>": 3,
                "<|audio_end|>": 4,
                "<|im_start|>": 5,
                "<|im_end|>": 6,
                "a": 7,
                "b": 8,
                "c": 9,
                "d": 10,
                "e": 11,
                "f": 12,
                "g": 13,
                "h": 14,
                "i": 15
            }
        }"#;
        let tokenizer = TaraTokenizer::from_json_str(tokenizer_json).expect("valid tokenizer");

        let mm_config = MultiModalConfig {
            vision: VisionConfig {
                patch_height: 8,
                patch_width: 8,
                channels: 3,
                hidden_size: 16,
                projector_type: ProjectorType::Linear,
                intermediate_dim: 16,
                normalize_pixels: false,
                mean: vec![0.0, 0.0, 0.0],
                std: vec![1.0, 1.0, 1.0],
            },
            audio: AudioConfig {
                sample_rate: 16000,
                n_fft: 256,
                hop_length: 128,
                n_mels: 16,
                f_min: 0.0,
                f_max: None,
                preemphasis: 0.97,
                hidden_size: 16,
                intermediate_dim: 16,
                projector_type: ProjectorType::Linear,
            },
            max_sequence_length: 128,
        };

        let engine = MultiModalEngine::new(mm_config);

        let img_bytes = vec![200u8; 8 * 8 * 3]; // 1 patch
        let image = RawImage::from_u8_slice(8, 8, 3, &img_bytes).expect("valid image");

        let audio_samples = vec![0.05f32; 256]; // 1 frame
        let audio = RawAudio::from_f32(audio_samples, 16000);

        let input = MultiModalInput {
            text: Some("describe image and audio".to_string()),
            image: Some(image),
            audio: Some(audio),
        };

        let logits = engine.forward(&model, &tokenizer, &input).expect("multimodal forward pass succeeds");
        assert!(!logits.is_empty());
        assert_eq!(logits.len() % 16, 0); // multiple of vocab_size
    }
}
