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
    #[error("empty video sequence: at least one frame is required")]
    EmptyVideo,
    #[error("video frame {frame_idx} dimension mismatch: expected {expected:?}, got {actual:?}")]
    FrameDimensionMismatch {
        frame_idx: usize,
        expected: (usize, usize, usize),
        actual: (usize, usize, usize),
    },
}

/// Dynamically resolves the base frequency for spatio-temporal positional embeddings.
fn default_vision_theta_base() -> f32 {
    std::env::var("TARA_VISION_THETA_BASE")
        .or_else(|_| std::env::var("TARA_ROPE_THETA"))
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|&v| v > 1.0 && v.is_finite())
        .unwrap_or(10_000.0)
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
    #[serde(default = "default_vision_theta_base")]
    pub theta_base: f32,
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
            theta_base: default_vision_theta_base(),
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

/// Raw video sequence representation comprising sequential time-ordered frames.
#[derive(Debug, Clone)]
pub struct RawVideo {
    pub frames: Vec<RawImage>,
    pub fps: f32,
}

impl RawVideo {
    /// Create a validated video sequence from sequential frames.
    pub fn new(frames: Vec<RawImage>, fps: f32) -> Result<Self, VisionError> {
        if frames.is_empty() {
            return Err(VisionError::EmptyVideo);
        }
        let w = frames[0].width;
        let h = frames[0].height;
        let c = frames[0].channels;
        for (idx, frame) in frames.iter().enumerate().skip(1) {
            if frame.width != w || frame.height != h || frame.channels != c {
                return Err(VisionError::FrameDimensionMismatch {
                    frame_idx: idx,
                    expected: (w, h, c),
                    actual: (frame.width, frame.height, frame.channels),
                });
            }
        }
        Ok(Self { frames, fps })
    }

    /// Number of frames in this video.
    pub fn num_frames(&self) -> usize {
        self.frames.len()
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

    /// Generate 3D spatio-temporal sinusoidal positional embeddings: [num_frames * num_y * num_x, hidden_size].
    /// Dynamic frequency allocations are partitioned evenly into frequency pairs:
    /// - [0..d_time] for temporal dimension (t = 1..num_frames)
    /// - [d_time..d_time + d_y] for vertical spatial dimension (py = 1..num_y)
    /// - [d_time + d_y..d_time + d_y + d_x] for horizontal spatial dimension (px = 1..num_x)
    pub fn generate_3d_spatiotemporal_pos_embeddings(
        num_frames: usize,
        num_y: usize,
        num_x: usize,
        hidden_size: usize,
        theta_base: f32,
    ) -> Vec<f32> {
        let patches_per_frame = num_y * num_x;
        let total_tokens = num_frames * patches_per_frame;
        let mut pos_emb = vec![0.0f32; total_tokens * hidden_size];

        let safe_theta = if theta_base.is_finite() && theta_base > 1.0 {
            theta_base
        } else {
            default_vision_theta_base()
        };

        // Guarantee even-pair assignment across temporal and spatial dimensions
        let total_pairs = hidden_size / 2;
        let t_pairs = total_pairs / 3;
        let y_pairs = (total_pairs - t_pairs) / 2;
        let x_pairs = total_pairs - t_pairs - y_pairs;

        let d_time = t_pairs * 2;
        let d_y = y_pairs * 2;
        let d_x = x_pairs * 2;

        for t in 0..num_frames {
            for py in 0..num_y {
                for px in 0..num_x {
                    let token_idx = t * patches_per_frame + py * num_x + px;
                    let row = &mut pos_emb[token_idx * hidden_size..(token_idx + 1) * hidden_size];

                    // 1. Temporal frequencies
                    if t_pairs > 0 {
                        let dim_t = d_time.max(2) as f32;
                        for i in 0..t_pairs {
                            let div_term = safe_theta.powf((2 * i) as f32 / dim_t);
                            let angle_t = (t as f32 + 1.0) / div_term;
                            row[2 * i] = angle_t.sin();
                            row[2 * i + 1] = angle_t.cos();
                        }
                    }

                    // 2. Spatial Y frequencies
                    if y_pairs > 0 {
                        let dim_y = d_y.max(2) as f32;
                        for i in 0..y_pairs {
                            let div_term = safe_theta.powf((2 * i) as f32 / dim_y);
                            let angle_y = (py as f32 + 1.0) / div_term;
                            row[d_time + 2 * i] = angle_y.sin();
                            row[d_time + 2 * i + 1] = angle_y.cos();
                        }
                    }

                    // 3. Spatial X frequencies
                    if x_pairs > 0 {
                        let dim_x = d_x.max(2) as f32;
                        let x_start = d_time + d_y;
                        for i in 0..x_pairs {
                            let div_term = safe_theta.powf((2 * i) as f32 / dim_x);
                            let angle_x = (px as f32 + 1.0) / div_term;
                            row[x_start + 2 * i] = angle_x.sin();
                            row[x_start + 2 * i + 1] = angle_x.cos();
                        }
                    }
                }
            }
        }

        pos_emb
    }

    /// Encode raw video sequence into spatiotemporally projected token embeddings:
    /// Returns (embeddings, num_frames, patches_per_frame).
    pub fn encode_video_to_embeddings(
        &self,
        video: &RawVideo,
    ) -> Result<(Vec<f32>, usize, usize), VisionError> {
        if video.frames.is_empty() {
            return Err(VisionError::EmptyVideo);
        }

        let num_frames = video.frames.len();
        let mut all_projected = Vec::new();
        let mut num_y_val = 0;
        let mut num_x_val = 0;

        for (idx, frame) in video.frames.iter().enumerate() {
            let (flattened_patches, num_y, num_x) = self.patchify(frame)?;
            if idx == 0 {
                num_y_val = num_y;
                num_x_val = num_x;
            } else if num_y != num_y_val || num_x != num_x_val {
                return Err(VisionError::FrameDimensionMismatch {
                    frame_idx: idx,
                    expected: (video.frames[0].width, video.frames[0].height, video.frames[0].channels),
                    actual: (frame.width, frame.height, frame.channels),
                });
            }
            let patches = num_y * num_x;
            let projected_frame = self.projector.project(&flattened_patches, patches);
            all_projected.extend(projected_frame);
        }

        let patches_per_frame = num_y_val * num_x_val;
        let pos_emb = Self::generate_3d_spatiotemporal_pos_embeddings(
            num_frames,
            num_y_val,
            num_x_val,
            self.config.hidden_size,
            self.config.theta_base,
        );

        for i in 0..all_projected.len() {
            all_projected[i] += pos_emb[i];
        }

        Ok((all_projected, num_frames, patches_per_frame))
    }
}

/// Native Visual Frame Decoder un-projecting latent representations back into RGB visual frames.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionDecoder {
    pub config: VisionConfig,
    /// Un-projection linear weights: shape [patch_dim, hidden_size]
    pub w_unproj: Vec<f32>,
    pub b_unproj: Vec<f32>,
}

impl VisionDecoder {
    /// Construct a new visual decoder initialized with variance scaling.
    pub fn new(config: VisionConfig) -> Self {
        let patch_dim = config.patch_height * config.patch_width * config.channels;
        let hs = config.hidden_size;
        let scale = (2.0f32 / (hs + patch_dim) as f32).sqrt();
        let total = patch_dim * hs;
        let mut w_unproj = Vec::with_capacity(total);
        for i in 0..total {
            let val = ((i as f32 * 0.283 + 0.3).sin()) * scale;
            w_unproj.push(val);
        }
        let b_unproj = vec![0.0f32; patch_dim];
        Self {
            config,
            w_unproj,
            b_unproj,
        }
    }

    /// Un-patchify flattened patch pixels back into a coherent 2D image.
    pub fn unpatchify(
        &self,
        patch_pixels: &[f32],
        width: usize,
        height: usize,
    ) -> Result<RawImage, VisionError> {
        let ph = self.config.patch_height;
        let pw = self.config.patch_width;
        let c = self.config.channels;

        if ph == 0 || pw == 0 || height < ph || width < pw {
            return Err(VisionError::InvalidPatchDimensions {
                patch_h: ph,
                patch_w: pw,
                img_h: height,
                img_w: width,
            });
        }

        let num_patches_y = height / ph;
        let num_patches_x = width / pw;
        let num_patches = num_patches_y * num_patches_x;
        let patch_dim = ph * pw * c;

        if patch_pixels.len() != num_patches * patch_dim {
            return Err(VisionError::DimensionMismatch {
                actual: patch_pixels.len(),
                expected: num_patches * patch_dim,
            });
        }

        let mut image_data = vec![0.0f32; width * height * c];

        for py in 0..num_patches_y {
            for px in 0..num_patches_x {
                let patch_idx = py * num_patches_x + px;
                let patch_offset = patch_idx * patch_dim;

                let mut p_cursor = 0;
                for dy in 0..ph {
                    let y = py * ph + dy;
                    for dx in 0..pw {
                        let x = px * pw + dx;
                        let dst_idx = (y * width + x) * c;

                        for ch in 0..c {
                            let mut val = patch_pixels[patch_offset + p_cursor];
                            if self.config.normalize_pixels {
                                let mean = self.config.mean.get(ch).copied().unwrap_or(0.5);
                                let std = self.config.std.get(ch).copied().unwrap_or(0.5).max(1e-5);
                                val = val * std + mean;
                            }
                            image_data[dst_idx + ch] = val.clamp(0.0, 1.0);
                            p_cursor += 1;
                        }
                    }
                }
            }
        }

        RawImage::from_f32(width, height, c, image_data)
    }

    /// Decodes a latent representation vector into a full-resolution RGB image.
    pub fn decode_latent_to_image(
        &self,
        latent: &[f32],
        width: usize,
        height: usize,
    ) -> Result<RawImage, VisionError> {
        let ph = self.config.patch_height;
        let pw = self.config.patch_width;
        let c = self.config.channels;
        let patch_dim = ph * pw * c;
        let hs = self.config.hidden_size;

        if latent.len() != hs {
            return Err(VisionError::DimensionMismatch {
                expected: hs,
                actual: latent.len(),
            });
        }

        let num_patches_y = height / ph;
        let num_patches_x = width / pw;
        let num_patches = num_patches_y * num_patches_x;

        // Un-project latent into patch features
        let mut single_patch_features = vec![0.0f32; patch_dim];
        for p in 0..patch_dim {
            let mut sum = self.b_unproj[p];
            let row_offset = p * hs;
            for h in 0..hs {
                sum += self.w_unproj[row_offset + h] * latent[h];
            }
            single_patch_features[p] = sum;
        }

        // Broadcast across patches with spatial modulation
        let mut all_patch_pixels = Vec::with_capacity(num_patches * patch_dim);
        for idx in 0..num_patches {
            let spatial_factor = 1.0 + 0.05 * ((idx as f32 * 0.4).sin());
            for p in 0..patch_dim {
                all_patch_pixels.push((single_patch_features[p] * spatial_factor).clamp(-2.0, 2.0));
            }
        }

        self.unpatchify(&all_patch_pixels, width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_video_validation() {
        let frame1 = RawImage::from_f32(32, 32, 3, vec![0.5f32; 32 * 32 * 3]).unwrap();
        let frame2 = RawImage::from_f32(32, 32, 3, vec![0.8f32; 32 * 32 * 3]).unwrap();
        let video = RawVideo::new(vec![frame1, frame2], 24.0).unwrap();
        assert_eq!(video.num_frames(), 2);

        let mismatch = RawImage::from_f32(16, 16, 3, vec![0.0f32; 16 * 16 * 3]).unwrap();
        let f1 = RawImage::from_f32(32, 32, 3, vec![0.5f32; 32 * 32 * 3]).unwrap();
        let err = RawVideo::new(vec![f1, mismatch], 24.0);
        assert!(err.is_err());
    }

    #[test]
    fn test_3d_spatiotemporal_embeddings_and_video_encoding() {
        let config = VisionConfig {
            patch_height: 16,
            patch_width: 16,
            channels: 3,
            hidden_size: 64,
            projector_type: ProjectorType::Linear,
            intermediate_dim: 128,
            normalize_pixels: true,
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
            theta_base: 10000.0,
        };
        let encoder = VisionEncoder::new(config);

        let frame1 = RawImage::from_f32(32, 32, 3, vec![0.2f32; 32 * 32 * 3]).unwrap();
        let frame2 = RawImage::from_f32(32, 32, 3, vec![0.4f32; 32 * 32 * 3]).unwrap();
        let frame3 = RawImage::from_f32(32, 32, 3, vec![0.6f32; 32 * 32 * 3]).unwrap();
        let video = RawVideo::new(vec![frame1, frame2, frame3], 30.0).unwrap();

        let (embs, n_frames, patches_per_frame) = encoder.encode_video_to_embeddings(&video).unwrap();
        assert_eq!(n_frames, 3);
        assert_eq!(patches_per_frame, 4); // (32/16) * (32/16) = 2 * 2 = 4
        assert_eq!(embs.len(), 3 * 4 * 64);
        assert!(!embs.iter().any(|v| v.is_nan() || v.is_infinite()));
    }
}
