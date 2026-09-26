//! Autoregressive generation pipeline.
//!
//! Provides both full-response and streaming token generation with the
//! complete Python-parity sampling pipeline:
//! repetition penalty → temperature → top-k → top-p nucleus → categorical.

use std::time::Instant;
use rand::Rng;

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
    let vocab_size = logits.len();
    let mut scores: Vec<f32> = logits.to_vec();

    // 1. Repetition penalty
    if repetition_penalty != 1.0 {
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

    // 2. Temperature — argmax if effectively deterministic
    if temperature <= 0.001 {
        return scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    for s in scores.iter_mut() {
        *s /= temperature;
    }

    // 3. Softmax
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let prob_sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= prob_sum;
    }

    // 4. Top-k filtering
    let effective_k = if top_k == 0 { vocab_size } else { top_k.min(vocab_size) };
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
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
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    stop_tokens: Option<&[&str]>,
) -> Result<GenerateResult, String> {
    let start = Instant::now();
    let im_end = tokenizer.im_end_id();

    // Encode prompt
    let mut input_ids: Vec<u32> = tokenizer.encode(prompt);
    let prompt_len = input_ids.len();
    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let mut first_latency_ms = 0.0f64;

    for step in 0..max_new_tokens {
        let logits = model.forward(&input_ids);
        let seq_len = input_ids.len();
        // Take logits for the last token: shape [vocab_size]
        let vocab_size = model.config.vocab_size;
        let last_logits = &logits[(seq_len - 1) * vocab_size..seq_len * vocab_size];

        let next_id = sample_next_token(
            last_logits,
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
        input_ids.push(next_id);

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

    let _ = prompt_len; // suppress unused warning
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
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    stop_tokens: Option<&[&str]>,
    mut callback: F,
) -> Result<(), String>
where
    F: FnMut(String),
{
    let im_end = tokenizer.im_end_id();
    let mut input_ids: Vec<u32> = tokenizer.encode(prompt);
    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);

    for _ in 0..max_new_tokens {
        let logits = model.forward(&input_ids);
        let seq_len = input_ids.len();
        let vocab_size = model.config.vocab_size;
        let last_logits = &logits[(seq_len - 1) * vocab_size..seq_len * vocab_size];

        let next_id = sample_next_token(
            last_logits,
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
        input_ids.push(next_id);

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
