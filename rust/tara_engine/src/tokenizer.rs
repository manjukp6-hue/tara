//! TARA tokenizer — greedy longest-match BPE-style tokenizer.
//!
//! Reads `tokenizer.json` with the schema `{"vocab": {"<token>": id, ...}}`
//! and provides encode/decode with full special-token support.

use std::collections::HashMap;
use std::fs;
use thiserror::Error;

/// Special tokens recognised by the tokenizer.
pub const SPECIAL_TOKENS: &[&str] = &[
    "<|pad|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|unk|>",
    "<|creator_auth|>",
    "<|tara_rule|>",
    "<|tara_exec|>",
    "<|tara_skill|>",
    "<|tara_memory|>",
];

/// Errors produced by the tokenizer.
#[derive(Debug, Error)]
pub enum TokenizerError {
    /// I/O failure while reading the vocab file.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Malformed JSON in tokenizer.json.
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    /// The `vocab` key was missing from tokenizer.json.
    #[error("tokenizer.json missing 'vocab' field")]
    MissingVocab,
}

/// Greedy longest-match tokenizer for the TARA model.
///
/// Loads vocabulary from `tokenizer.json` and provides byte-level encode/decode
/// with special-token handling.
pub struct TaraTokenizer {
    /// Maps token string → token id.
    pub token_to_id: HashMap<String, u32>,
    /// Maps token id → token string.
    pub id_to_token: HashMap<u32, String>,
    /// Total number of tokens in the vocabulary.
    pub vocab_size: usize,
}

impl TaraTokenizer {
    /// Load the tokenizer from a `tokenizer.json` file.
    ///
    /// The file must contain a top-level JSON object with a `"vocab"` key
    /// mapping token strings to integer IDs.
    ///
    /// # Errors
    /// Returns [`TokenizerError`] on I/O or parse failures.
    pub fn from_file(path: &str) -> Result<Self, TokenizerError> {
        let raw = fs::read_to_string(path)?;
        let root: serde_json::Value = serde_json::from_str(&raw)?;

        let vocab_obj = root
            .get("vocab")
            .ok_or(TokenizerError::MissingVocab)?;

        let vocab_map = vocab_obj
            .as_object()
            .ok_or(TokenizerError::MissingVocab)?;

        let mut token_to_id: HashMap<String, u32> = HashMap::with_capacity(vocab_map.len());
        let mut id_to_token: HashMap<u32, String> = HashMap::with_capacity(vocab_map.len());

        for (token, id_val) in vocab_map {
            let id = id_val
                .as_u64()
                .ok_or(TokenizerError::MissingVocab)? as u32;
            token_to_id.insert(token.clone(), id);
            id_to_token.insert(id, token.clone());
        }

        let vocab_size = token_to_id.len();

        Ok(Self {
            token_to_id,
            id_to_token,
            vocab_size,
        })
    }

    /// Encode a text string into a sequence of token IDs.
    ///
    /// Uses greedy longest-match: at each byte position it tries the longest
    /// possible token (up to 25 chars) first, with special tokens checked at
    /// every position before regular tokens.  Unrecognised bytes fall back to
    /// the `<|unk|>` token.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let mut i = 0;

        while i < n {
            // --- Special-token priority pass ---
            let mut matched_special = false;
            for &sp in SPECIAL_TOKENS {
                let sp_chars: Vec<char> = sp.chars().collect();
                let sp_len = sp_chars.len();
                if i + sp_len <= n && chars[i..i + sp_len] == sp_chars[..] {
                    if let Some(&id) = self.token_to_id.get(sp) {
                        ids.push(id);
                        i += sp_len;
                        matched_special = true;
                        break;
                    }
                }
            }
            if matched_special {
                continue;
            }

            // --- Greedy longest-match from length 25 down to 1 ---
            let max_len = 25usize.min(n - i);
            let mut found = false;
            for length in (1..=max_len).rev() {
                let slice: String = chars[i..i + length].iter().collect();
                if let Some(&id) = self.token_to_id.get(&slice) {
                    ids.push(id);
                    i += length;
                    found = true;
                    break;
                }
            }

            if !found {
                // Fallback to UNK and advance one character
                ids.push(self.unk_id());
                i += 1;
            }
        }

        ids
    }

    /// Decode a sequence of token IDs into a human-readable string.
    ///
    /// Structural control tokens (`<|pad|>`, `<|im_start|>`, `<|im_end|>`) are
    /// silently skipped.  All other tokens are concatenated.
    pub fn decode(&self, ids: &[u32]) -> String {
        let skip_ids = [self.pad_id(), self.im_start_id(), self.im_end_id()];
        let mut out = String::new();
        for &id in ids {
            if skip_ids.contains(&id) {
                continue;
            }
            if let Some(tok) = self.id_to_token.get(&id) {
                out.push_str(tok);
            }
        }
        out
    }

    /// Returns the ID of `<|unk|>` (unknown) token.
    pub fn unk_id(&self) -> u32 {
        *self.token_to_id.get("<|unk|>").unwrap_or(&3)
    }

    /// Returns the ID of `<|im_end|>` (end-of-turn) token.
    pub fn im_end_id(&self) -> u32 {
        *self.token_to_id.get("<|im_end|>").unwrap_or(&2)
    }

    /// Returns the ID of `<|pad|>` (padding) token.
    pub fn pad_id(&self) -> u32 {
        *self.token_to_id.get("<|pad|>").unwrap_or(&0)
    }

    /// Returns the ID of `<|im_start|>` (start-of-turn) token.
    pub fn im_start_id(&self) -> u32 {
        *self.token_to_id.get("<|im_start|>").unwrap_or(&1)
    }
}
