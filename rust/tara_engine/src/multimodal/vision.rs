//! Native Vision Processing and Patch Projection Subsystem.
//!
//! Converts raw images (RGB, RGBA, Grayscale) into normalized patches, adds 2D spatial
//! sinusoidal positional embeddings, and projects them into continuous token embeddings.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::projector::{CrossModalProjector, ProjectorType};

/// Errors encountered during vision processing.
#[derive(Debug, Error)]
pub enum VisionError {
    #[error("empty image data")]
    EmptyData,
    #[error("dimension mismatch: buffer size {actual} != expected {expected}")]
    DimensionMismatch { actual: usize, expected: usize },
    #[error("invalid patch dimensions ({patch_h}x{patch_w}) for image ({img_h}x{img_w})")]
    InvalidPatchDimensions {
        patch_h: usize,
        patch_w: usize,
        img_h: usize,
        img_w: usize,
    },
}

/// Dynamic configuration for vision encoding and projection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionConfig {
    pub patch_height: usize,
    pub patch_width: usize,
    pub channels: usize,
    pub hidden_size: usize,
    pub projector_type: ProjectorType,
    pub intermediate_dim: usize,
    pub normalize_pixels: bool,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            patch_height: 16,
            patch_width: 16,
            channels: 3,
            hidden_size: 64,
            projector_type: ProjectorType::Mlp,
            intermediate_dim: 128,
            normalize_pixels: true,
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
        }
    }
}

/// Raw image tensor representation.
#[derive(Debug, Clone)]
pub struct RawImage {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    /// Pixel data stored in row-major order: [H, W, C]
    pub data: Vec<f32>,
}

impl RawImage {
    /// Create image from byte slice [H, W, C] normalized to [0.0, 1.0].
    pub fn from_u8_slice(width: usize, height: usize, channels: usize, bytes: &[u8]) -> Result<Self, VisionError> {
        let expected = width * height * channels;
        if bytes.len() != expected {
            return Err(VisionError::DimensionMismatch {
                actual: bytes.len(),
                expected,
            });
        }
        let data = bytes.iter().map(|&b| b as f32 / 255.0).collect();
        Ok(Self {
            width,
            height,
            channels,
            data,
        })
    }

    /// Create image from existing f32 pixel buffer.
    pub fn from_f32(width: usize, height: usize, channels: usize, data: Vec<f32>) -> Result<Self, VisionError> {
        let expected = width * height * channels;
        if data.len() != expected {
            return Err(VisionError::DimensionMismatch {
                actual: data.len(),
                expected,
            });
        }
        Ok(Self {
            width,
            height,
            channels,
            data,
        })
    }
}

/// Native Vision Encoder converting images to projected transformer tokens.
#[derive(Debug, Clone)]
pub struct VisionEncoder {
    pub config: VisionConfig,
    pub projector: CrossModalProjector,
}

impl VisionEncoder {
    /// Construct a new VisionEncoder with dynamic configuration.
    pub fn new(config: VisionConfig) -> Self {
        let patch_dim = config.patch_height * config.patch_width * config.channels;
        let projector = CrossModalProjector::new(
            config.projector_type,
            patch_dim,
            config.hidden_size,
            config.intermediate_dim,
        );
        Self { config, projector }
    }

    /// Patchify image into flattened patch vectors [num_patches, patch_dim].
    pub fn patchify(&self, image: &RawImage) -> Result<(Vec<f32>, usize, usize), VisionError> {
        if image.data.is_empty() {
            return Err(VisionError::EmptyData);
        }

        let ph = self.config.patch_height;
        let pw = self.config.patch_width;
        let c = self.config.channels;

        if ph == 0 || pw == 0 || image.height < ph || image.width < pw {
            return Err(VisionError::InvalidPatchDimensions {
                patch_h: ph,
                patch_w: pw,
                img_h: image.height,
                img_w: image.width,
            });
        }

        let num_patches_y = image.height / ph;
        let num_patches_x = image.width / pw;
        let num_patches = num_patches_y * num_patches_x;
        let patch_dim = ph * pw * c;

        let mut flattened_patches = vec![0.0f32; num_patches * patch_dim];

        for py in 0..num_patches_y {
            for px in 0..num_patches_x {
                let patch_idx = py * num_patches_x + px;
                let patch_offset = patch_idx * patch_dim;

                let mut p_cursor = 0;
                for dy in 0..ph {
                    let y = py * ph + dy;
                    for dx in 0..pw {
                        let x = px * pw + dx;
                        let src_idx = (y * image.width + x) * c;

                        for ch in 0..c {
                            let mut val = image.data[src_idx + ch];
                            if self.config.normalize_pixels {
                                let mean = self.config.mean.get(ch).copied().unwrap_or(0.5);
                                let std = self.config.std.get(ch).copied().unwrap_or(0.5).max(1e-5);
                                val = (val - mean) / std;
                            }
                            flattened_patches[patch_offset + p_cursor] = val;
                            p_cursor += 1;
                        }
                    }
                }
            }
        }

        Ok((flattened_patches, num_patches_y, num_patches_x))
    }

    /// Generate 2D spatial sinusoidal positional embeddings for a patch grid [num_y, num_x].
    pub fn generate_2d_sinusoidal_pos_embeddings(
        num_y: usize,
        num_x: usize,
        hidden_size: usize,
    ) -> Vec<f32> {
        let num_patches = num_y * num_x;
        let mut pos_emb = vec![0.0f32; num_patches * hidden_size];
        let d_half = hidden_size / 2;

        for py in 0..num_y {
            for px in 0..num_x {
                let idx = py * num_x + px;
                let row = &mut pos_emb[idx * hidden_size..(idx + 1) * hidden_size];

                // Y-coordinate frequencies (first half of dimensions)
                for i in 0..(d_half / 2) {
                    let div_term = (10000.0f32).powf((2 * i) as f32 / d_half as f32);
                    let angle_y = (py as f32 + 1.0) / div_term;
                    row[2 * i] = angle_y.sin();
                    row[2 * i + 1] = angle_y.cos();
                }

                // X-coordinate frequencies (second half of dimensions)
                let x_start = d_half;
                let rem_dims = hidden_size - x_start;
                for i in 0..(rem_dims / 2) {
                    let div_term = (10000.0f32).powf((2 * i) as f32 / rem_dims as f32);
                    let angle_x = (px as f32 + 1.0) / div_term;
                    row[x_start + 2 * i] = angle_x.sin();
                    row[x_start + 2 * i + 1] = angle_x.cos();
                }
            }
        }

        pos_emb
    }

    /// Encode raw image into projected token embeddings: [num_patches, hidden_size].
    pub fn encode_to_embeddings(&self, image: &RawImage) -> Result<(Vec<f32>, usize), VisionError> {
        let (flattened_patches, num_y, num_x) = self.patchify(image)?;
        let num_patches = num_y * num_x;

        // Project patch pixels to hidden dimension
        let mut projected = self.projector.project(&flattened_patches, num_patches);

        // Add 2D sinusoidal spatial embeddings
        let pos_emb = Self::generate_2d_sinusoidal_pos_embeddings(num_y, num_x, self.config.hidden_size);
        for i in 0..projected.len() {
            projected[i] += pos_emb[i];
        }

        Ok((projected, num_patches))
    }
}
