//! Native Audio DSP and Mel Spectrogram Projection Subsystem.
//!
//! Converts raw audio waveforms into Log-Mel Spectrogram representations using
//! an in-house Cooley-Tukey Radix-2 Fast Fourier Transform (FFT) and triangular
//! Mel-filterbank, projecting temporal acoustic features into continuous token embeddings.

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use thiserror::Error;

use super::projector::{CrossModalProjector, ProjectorType};

/// Errors encountered during audio processing.
#[derive(Debug, Error)]
pub enum AudioError {
    #[error("empty audio waveform")]
    EmptyAudio,
    #[error("audio waveform length ({length}) is shorter than FFT window ({n_fft})")]
    WaveformTooShort { length: usize, n_fft: usize },
    #[error("invalid audio parameters: sample_rate={sample_rate}, n_fft={n_fft}, hop_length={hop_length}, n_mels={n_mels}")]
    InvalidParameters {
        sample_rate: usize,
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
    },
}

/// Dynamic configuration for audio encoding and Mel projection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub f_min: f32,
    pub f_max: Option<f32>,
    pub preemphasis: f32,
    pub hidden_size: usize,
    pub intermediate_dim: usize,
    pub projector_type: ProjectorType,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 512,
            hop_length: 160,
            n_mels: 80,
            f_min: 0.0,
            f_max: Some(8000.0),
            preemphasis: 0.97,
            hidden_size: 64,
            intermediate_dim: 128,
            projector_type: ProjectorType::Mlp,
        }
    }
}

/// Raw audio waveform container.
#[derive(Debug, Clone)]
pub struct RawAudio {
    pub sample_rate: usize,
    /// Mono audio samples normalized in range [-1.0, 1.0].
    pub samples: Vec<f32>,
}

impl RawAudio {
    /// Construct from raw 16-bit signed PCM audio.
    pub fn from_i16_pcm(samples: &[i16], sample_rate: usize) -> Self {
        let f_samples = samples.iter().map(|&s| s as f32 / 32768.0).collect();
        Self {
            sample_rate,
            samples: f_samples,
        }
    }

    /// Construct from floating-point samples.
    pub fn from_f32(samples: Vec<f32>, sample_rate: usize) -> Self {
        Self {
            sample_rate,
            samples,
        }
    }
}

/// Native Cooley-Tukey Radix-2 Decimation-In-Time Fast Fourier Transform.
pub fn fft_radix2(real: &mut [f32], imag: &mut [f32]) {
    let n = real.len();
    assert!(n.is_power_of_two(), "FFT size must be a power of two");
    assert_eq!(n, imag.len());

    // Bit reversal permutation
    let mut j = 0;
    for i in 0..n {
        if i < j {
            real.swap(i, j);
            imag.swap(i, j);
        }
        let mut k = n >> 1;
        while k > 0 && j >= k {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    // Cooley-Tukey butterflies
    let mut step = 2;
    while step <= n {
        let half_step = step / 2;
        let angle_step = -2.0 * PI / (step as f32);

        for k in (0..n).step_by(step) {
            for i in 0..half_step {
                let angle = (i as f32) * angle_step;
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                let u_idx = k + i;
                let v_idx = k + i + half_step;

                let vr = real[v_idx];
                let vi = imag[v_idx];

                // Complex multiplication: (vr + i*vi) * (cos + i*sin)
                let tr = vr * cos_val - vi * sin_val;
                let ti = vr * sin_val + vi * cos_val;

                real[v_idx] = real[u_idx] - tr;
                imag[v_idx] = imag[u_idx] - ti;
                real[u_idx] += tr;
                imag[u_idx] += ti;
            }
        }
        step <<= 1;
    }
}

/// Convert frequency in Hertz to Mel scale.
#[inline]
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert Mel scale value back to Hertz.
#[inline]
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// Native Audio Encoder converting audio waveforms to projected transformer tokens.
#[derive(Debug, Clone)]
pub struct AudioEncoder {
    pub config: AudioConfig,
    pub mel_filterbank: Vec<f32>,
    pub num_fft_bins: usize,
    pub projector: CrossModalProjector,
}

impl AudioEncoder {
    /// Construct AudioEncoder and precalculate triangular Mel-filterbank matrix.
    pub fn new(config: AudioConfig) -> Self {
        let n_fft = config.n_fft;
        assert!(n_fft.is_power_of_two(), "n_fft must be a power of two");
        let num_fft_bins = n_fft / 2 + 1;
        let n_mels = config.n_mels;

        let f_min = config.f_min;
        let f_max = config.f_max.unwrap_or(config.sample_rate as f32 / 2.0);

        let mel_min = hz_to_mel(f_min);
        let mel_max = hz_to_mel(f_max);

        // Compute triangular Mel filterbank matrix of shape [n_mels, num_fft_bins]
        let mut mel_filterbank = vec![0.0f32; n_mels * num_fft_bins];

        let mel_points: Vec<f32> = (0..=(n_mels + 1))
            .map(|i| mel_min + (mel_max - mel_min) * (i as f32) / ((n_mels + 1) as f32))
            .collect();
        let hz_points: Vec<f32> = mel_points.into_iter().map(mel_to_hz).collect();
        let bin_points: Vec<f32> = hz_points
            .into_iter()
            .map(|hz| (n_fft as f32 + 1.0) * hz / (config.sample_rate as f32))
            .collect();

        for m in 0..n_mels {
            let left = bin_points[m];
            let center = bin_points[m + 1];
            let right = bin_points[m + 2];

            for k in 0..num_fft_bins {
                let k_f = k as f32;
                let weight = if k_f >= left && k_f <= center && (center - left) > 1e-6 {
                    (k_f - left) / (center - left)
                } else if k_f > center && k_f <= right && (right - center) > 1e-6 {
                    (right - k_f) / (right - center)
                } else {
                    0.0
                };
                mel_filterbank[m * num_fft_bins + k] = weight;
            }
        }

        let projector = CrossModalProjector::new(
            config.projector_type,
            n_mels,
            config.hidden_size,
            config.intermediate_dim,
        );

        Self {
            config,
            mel_filterbank,
            num_fft_bins,
            projector,
        }
    }

    /// Extract Log-Mel Spectrogram features [num_frames, n_mels].
    pub fn extract_mel_spectrogram(&self, audio: &RawAudio) -> Result<(Vec<f32>, usize), AudioError> {
        let samples = &audio.samples;
        if samples.is_empty() {
            return Err(AudioError::EmptyAudio);
        }
        let n_fft = self.config.n_fft;
        let hop = self.config.hop_length;
        let n_mels = self.config.n_mels;

        if samples.len() < n_fft {
            return Err(AudioError::WaveformTooShort {
                length: samples.len(),
                n_fft,
            });
        }

        // Apply pre-emphasis: y[t] = x[t] - alpha * x[t-1]
        let mut preemphasized = Vec::with_capacity(samples.len());
        preemphasized.push(samples[0]);
        for i in 1..samples.len() {
            let val = samples[i] - self.config.preemphasis * samples[i - 1];
            preemphasized.push(val);
        }

        // Construct Hann window
        let window: Vec<f32> = (0..n_fft)
            .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / (n_fft as f32 - 1.0)).cos()))
            .collect();

        let num_frames = (preemphasized.len() - n_fft) / hop + 1;
        let mut mel_spectrogram = vec![0.0f32; num_frames * n_mels];

        let mut real_buf = vec![0.0f32; n_fft];
        let mut imag_buf = vec![0.0f32; n_fft];
        let mut power_spectrum = vec![0.0f32; self.num_fft_bins];

        for f in 0..num_frames {
            let start = f * hop;
            for i in 0..n_fft {
                real_buf[i] = preemphasized[start + i] * window[i];
                imag_buf[i] = 0.0;
            }

            fft_radix2(&mut real_buf, &mut imag_buf);

            // Compute power spectrum for the positive frequency bins
            for k in 0..self.num_fft_bins {
                let r = real_buf[k];
                let im = imag_buf[k];
                power_spectrum[k] = (r * r + im * im) / (n_fft as f32);
            }

            // Apply Mel-filterbank dot product & log energy
            let frame_offset = f * n_mels;
            for m in 0..n_mels {
                let filter_row = &self.mel_filterbank[m * self.num_fft_bins..(m + 1) * self.num_fft_bins];
                let energy: f32 = power_spectrum.iter().zip(filter_row.iter()).map(|(&p, &w)| p * w).sum();
                // Log energy with numerical epsilon floor
                mel_spectrogram[frame_offset + m] = (energy.max(1e-6)).ln();
            }
        }

        Ok((mel_spectrogram, num_frames))
    }

    /// Encode raw audio into projected token embeddings: [num_frames, hidden_size].
    pub fn encode_to_embeddings(&self, audio: &RawAudio) -> Result<(Vec<f32>, usize), AudioError> {
        let (mel_spectrogram, num_frames) = self.extract_mel_spectrogram(audio)?;
        let projected = self.projector.project(&mel_spectrogram, num_frames);
        Ok((projected, num_frames))
    }
}
