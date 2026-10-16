//! Autoregressive generation pipeline.
//!
//! Provides both full-response and streaming token generation with the
//! complete Python-parity sampling pipeline:
//! repetition penalty → temperature → top-k → top-p nucleus → categorical.

use rand::Rng;
use std::time::Instant;

use crate::model::causal_lm::TaraForCausalLM;
use crate::tokenizer::TaraTokenizer;

/// Result returned by [`generate_response`].
#[derive(Debug, Clone)]
pub struct GenerateResult {
    /// The generated text (decoded tokens, excluding prompt).
    pub text: String,
    /// Number of newly generated tokens.
    pub token_count: usize,
    /// Time to first token in milliseconds.
    pub first_latency_ms: f64,
    /// Total generation time in milliseconds.
    pub total_latency_ms: f64,
    /// Tokens per second.
    pub tps: f64,
}

/// Configuration options for text generation.
#[derive(Clone, Debug)]
pub struct GenerateOptions<'a> {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub stop_tokens: Option<&'a [&'a str]>,
}

impl<'a> Default for GenerateOptions<'a> {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            stop_tokens: None,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Sampling core
// ──────────────────────────────────────────────────────────────────────────────

/// Sample the next token given logits for the vocabulary.
///
/// Applies (in order):
/// 1. Repetition penalty on previously generated tokens
/// 2. Temperature scaling
/// 3. Top-k truncation
/// 4. Top-p nucleus truncation
/// 5. Categorical sampling (or argmax if temperature ≤ 0.001)
pub fn sample_next_token(
    logits: &[f32],
    generated_ids: &[u32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    let vocab_size = logits.len();
    let mut scores: Vec<f32> = logits
        .iter()
        .map(|&s| if s.is_finite() { s } else { -1e9 })
        .collect();

    // 1. Repetition penalty (only when finite and > 0.0)
    if repetition_penalty.is_finite() && repetition_penalty > 0.0 && (repetition_penalty - 1.0).abs() > f32::EPSILON {
        for &id in generated_ids {
            let id = id as usize;
            if id < vocab_size {
                if scores[id] > 0.0 {
                    scores[id] /= repetition_penalty;
                } else {
                    scores[id] *= repetition_penalty;
                }
            }
        }
    }

    // 2. Temperature — argmax if effectively deterministic or non-finite
    if !temperature.is_finite() || temperature <= 0.001 {
        return scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    for s in scores.iter_mut() {
        *s /= temperature;
    }

    // 3. Softmax with zero-sum guard
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let prob_sum: f32 = probs.iter().sum();
    if !prob_sum.is_finite() || prob_sum <= 0.0 {
        return 0;
    }
    for p in probs.iter_mut() {
        *p /= prob_sum;
    }

    // 4. Top-k filtering
    let effective_k = if top_k == 0 {
        vocab_size
    } else {
        top_k.min(vocab_size)
    };
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_k_indices: Vec<usize> = indexed[..effective_k].iter().map(|(i, _)| *i).collect();

    // 5. Top-p nucleus filtering
    let mut cumulative = 0.0f32;
    let mut nucleus: Vec<(usize, f32)> = Vec::new();
    for &idx in &top_k_indices {
        cumulative += probs[idx];
        nucleus.push((idx, probs[idx]));
        if cumulative >= top_p {
            break;
        }
    }

    if nucleus.is_empty() {
        nucleus.push((top_k_indices[0], probs[top_k_indices[0]]));
    }

    // Renormalise
    let nucleus_sum: f32 = nucleus.iter().map(|(_, p)| p).sum();
    if nucleus_sum <= 0.0 {
        return nucleus[0].0 as u32;
    }
    for (_, p) in nucleus.iter_mut() {
        *p /= nucleus_sum;
    }

    // Categorical sample
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumul = 0.0f32;
    for (idx, p) in &nucleus {
        cumul += p;
        if r <= cumul {
            return *idx as u32;
        }
    }
    nucleus.last().map(|(i, _)| *i as u32).unwrap_or(0)
}

// ──────────────────────────────────────────────────────────────────────────────
// Full response generation
// ──────────────────────────────────────────────────────────────────────────────

/// Generate a complete response for `prompt`, returning the full [`GenerateResult`].
///
/// # Errors
/// Currently infallible (returns `Ok` always); the signature uses `Result` for
/// forward-compatibility with fallible device backends.
pub fn generate_response(
    model: &TaraForCausalLM,
    tokenizer: &TaraTokenizer,
    prompt: &str,
    options: &GenerateOptions,
) -> Result<GenerateResult, String> {
    let max_new_tokens = options.max_new_tokens;
    let temperature = options.temperature;
    let top_k = options.top_k;
    let top_p = options.top_p;
    let repetition_penalty = options.repetition_penalty;
    let stop_tokens = options.stop_tokens;

    let start = Instant::now();
    let im_end = tokenizer.im_end_id();

    // Encode prompt
    let input_ids: Vec<u32> = tokenizer.encode(prompt);
    let prompt_len = input_ids.len();
    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let mut first_latency_ms = 0.0f64;
    let mut cache = model.create_kv_cache();

    // Prime cache with all prompt tokens except the last one
    if prompt_len > 1 {
        for &id in &input_ids[..prompt_len - 1] {
            let _ = model.forward_token_cached(id, &mut cache);
        }
    }

    let mut current_token = if prompt_len > 0 {
        input_ids[prompt_len - 1]
    } else {
        0
    };

    for step in 0..max_new_tokens {
        let last_logits = model.forward_token_cached(current_token, &mut cache);

        let next_id = sample_next_token(
            &last_logits,
            &generated,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        );

        if step == 0 {
            first_latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        }

        if next_id == im_end {
            break;
        }

        generated.push(next_id);
        current_token = next_id;

        // Stop-token check
        if let Some(stops) = stop_tokens {
            let current_text = tokenizer.decode(&generated);
            if stops.iter().any(|&s| current_text.contains(s)) {
                break;
            }
        }
    }

    let total_latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    let token_count = generated.len();
    let tps = if total_latency_ms > 0.0 {
        token_count as f64 / (total_latency_ms / 1000.0)
    } else {
        0.0
    };

    Ok(GenerateResult {
        text: tokenizer.decode(&generated),
        token_count,
        first_latency_ms,
        total_latency_ms,
        tps,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Streaming generation
// ──────────────────────────────────────────────────────────────────────────────

/// Stream generated tokens via a callback.
///
/// `callback` is called with each newly decoded token string as it is produced.
pub fn generate_stream<F>(
    model: &TaraForCausalLM,
    tokenizer: &TaraTokenizer,
    prompt: &str,
    options: &GenerateOptions,
    mut callback: F,
) -> Result<(), String>
where
    F: FnMut(String),
{
    let max_new_tokens = options.max_new_tokens;
    let temperature = options.temperature;
    let top_k = options.top_k;
    let top_p = options.top_p;
    let repetition_penalty = options.repetition_penalty;
    let stop_tokens = options.stop_tokens;

    let im_end = tokenizer.im_end_id();
    let input_ids: Vec<u32> = tokenizer.encode(prompt);
    let prompt_len = input_ids.len();
    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let mut cache = model.create_kv_cache();

    // Prime cache
    if prompt_len > 1 {
        for &id in &input_ids[..prompt_len - 1] {
            let _ = model.forward_token_cached(id, &mut cache);
        }
    }

    let mut current_token = if prompt_len > 0 {
        input_ids[prompt_len - 1]
    } else {
        0
    };

    for _ in 0..max_new_tokens {
        let last_logits = model.forward_token_cached(current_token, &mut cache);

        let next_id = sample_next_token(
            &last_logits,
            &generated,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        );

        if next_id == im_end {
            break;
        }

        generated.push(next_id);
        current_token = next_id;

        // Decode just this token
        let tok_str = tokenizer
            .id_to_token
            .get(&next_id)
            .cloned()
            .unwrap_or_default();
        callback(tok_str);

        if let Some(stops) = stop_tokens {
            let current = tokenizer.decode(&generated);
            if stops.iter().any(|&s| current.contains(s)) {
                break;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_next_token_empty_and_non_finite_logits() {
        // 1. Empty logits returns 0 without panic
        assert_eq!(sample_next_token(&[], &[], 0.7, 50, 0.9, 1.1), 0);

        // 2. Non-finite logits are sanitized to -1e9 without NaN propagation
        let nan_logits = [f32::NAN, 10.0, f32::INFINITY, -f32::INFINITY];
        let token = sample_next_token(&nan_logits, &[], 0.0, 50, 0.9, 1.1);
        assert_eq!(token, 1); // argmax picks index 1 (10.0)

        // 3. Invalid repetition penalty (negative or NaN) ignored safely
        let token2 = sample_next_token(&[1.0, 2.0], &[1], 0.0, 50, 0.9, f32::NAN);
        assert_eq!(token2, 1);
        let token3 = sample_next_token(&[1.0, 2.0], &[1], 0.0, 50, 0.9, -1.5);
        assert_eq!(token3, 1);

        // 4. Non-finite temperature defaults to argmax safely without panic or NaN
        let token_nan_temp = sample_next_token(&[1.0, 5.0, 2.0], &[], f32::NAN, 50, 0.9, 1.0);
        assert_eq!(token_nan_temp, 1);
        let token_inf_temp = sample_next_token(&[1.0, 5.0, 2.0], &[], f32::INFINITY, 50, 0.9, 1.0);
        assert_eq!(token_inf_temp, 1);

        // 5. Repetition penalty near 1.0 within EPSILON is ignored as no-op
        let token_eps = sample_next_token(&[1.0, 5.0], &[1], 0.0, 50, 0.9, 1.0 + f32::EPSILON / 2.0);
        assert_eq!(token_eps, 1);
    }
}
