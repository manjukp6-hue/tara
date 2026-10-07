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
    #[error("invalid tokenizer vocabulary: {0}")]
    InvalidVocab(String),
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
    /// Maximum character length of any token in the vocabulary (dynamically computed at load time).
    pub max_token_len: usize,
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

        let vocab_obj = root.get("vocab").ok_or(TokenizerError::MissingVocab)?;

        let vocab_map = vocab_obj.as_object().ok_or(TokenizerError::MissingVocab)?;

        let mut token_to_id: HashMap<String, u32> = HashMap::with_capacity(vocab_map.len());
        let mut id_to_token: HashMap<u32, String> = HashMap::with_capacity(vocab_map.len());

        for (token, id_val) in vocab_map {
            let raw_id = id_val.as_u64().ok_or(TokenizerError::MissingVocab)?;
            let id = u32::try_from(raw_id).map_err(|_| {
                TokenizerError::InvalidVocab(format!("token id {raw_id} exceeds u32"))
            })?;
            token_to_id.insert(token.clone(), id);
            if id_to_token.insert(id, token.clone()).is_some() {
                return Err(TokenizerError::InvalidVocab(format!(
                    "duplicate token id {id}"
                )));
            }
        }

        let vocab_size = token_to_id.len();
        if vocab_size == 0 || (0..vocab_size as u32).any(|id| !id_to_token.contains_key(&id)) {
            return Err(TokenizerError::InvalidVocab(
                "token ids must be a dense range starting at zero".into(),
            ));
        }

        let max_token_len = token_to_id
            .keys()
            .map(|t| t.chars().count())
            .max()
            .unwrap_or(1);

        Ok(Self {
            token_to_id,
            id_to_token,
            vocab_size,
            max_token_len,
        })
    }

    /// Encode a text string into a sequence of token IDs.
    ///
    /// Uses greedy longest-match: at each byte position it tries the longest
    /// possible token (dynamically up to `max_token_len`) first, with special tokens checked at
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

            // --- Greedy longest-match up to dynamically computed max_token_len ---
            let max_len = self.max_token_len.min(n - i);
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

#[cfg(test)]
mod tests {
    use super::{TaraTokenizer, TokenizerError};

    fn fixture(contents: &str) -> String {
        let path = std::env::temp_dir().join(format!(
            "tara_tokenizer_{}_{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::write(&path, contents).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn rejects_duplicate_and_sparse_token_ids() {
        let duplicate = fixture(r#"{"vocab":{"a":0,"b":0}}"#);
        assert!(matches!(
            TaraTokenizer::from_file(&duplicate),
            Err(TokenizerError::InvalidVocab(_))
        ));
        std::fs::remove_file(duplicate).unwrap();

        let sparse = fixture(r#"{"vocab":{"a":0,"b":2}}"#);
        assert!(matches!(
            TaraTokenizer::from_file(&sparse),
            Err(TokenizerError::InvalidVocab(_))
        ));
        std::fs::remove_file(sparse).unwrap();
    }

    #[test]
    fn test_8192_tokenizer_multi_domain_coverage() {
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let cand_path = manifest_dir.join("../../storage/models/tara_candidate_v1/tokenizer.json");
        if !cand_path.exists() {
            return;
        }

        let tokenizer = TaraTokenizer::from_file(cand_path.to_str().unwrap())
            .expect("8192 tokenizer must load cleanly");
        assert_eq!(
            tokenizer.vocab_size, 8192,
            "Tokenizer must contain exactly 8192 tokens"
        );

        // 1. Kannada
        let kannada_text = "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?";
        let kannada_ids = tokenizer.encode(kannada_text);
        assert!(!kannada_ids.is_empty());
        let kannada_decoded = tokenizer.decode(&kannada_ids);
        assert_eq!(
            kannada_decoded, kannada_text,
            "Kannada text must round-trip exactly"
        );

        // 2. Kanglish
        let kanglish_text = "tara neevu hegiddira oota aitha?";
        let kanglish_ids = tokenizer.encode(kanglish_text);
        assert!(!kanglish_ids.is_empty());
        let kanglish_decoded = tokenizer.decode(&kanglish_ids);
        assert_eq!(
            kanglish_decoded, kanglish_text,
            "Kanglish text must round-trip exactly"
        );

        // 3. English
        let english_text =
            "TARA neural engine scales across multi-capacity transformer architectures.";
        let english_ids = tokenizer.encode(english_text);
        assert!(!english_ids.is_empty());
        let english_decoded = tokenizer.decode(&english_ids);
        assert_eq!(
            english_decoded, english_text,
            "English text must round-trip exactly"
        );

        // 4. Code
        let code_text = "fn compute(x: f32) -> f32 { let y = x * 2.0; y + 1.0 }";
        let code_ids = tokenizer.encode(code_text);
        assert!(!code_ids.is_empty());
        let code_decoded = tokenizer.decode(&code_ids);
        assert_eq!(code_decoded, code_text, "Code text must round-trip exactly");

        // 5. Math
        let math_text =
            "E = mc^2 and \x5cint_0^\x5cinfty e^{-x^2} dx = \x5cfrac{\x5csqrt{\x5cpi}}{2}";
        let math_ids = tokenizer.encode(math_text);
        assert!(!math_ids.is_empty());
        let math_decoded = tokenizer.decode(&math_ids);
        assert_eq!(math_decoded, math_text, "Math text must round-trip exactly");
    }

    #[test]
    fn test_dynamic_max_token_len_handling() {
        let long_token = "very_long_custom_token_that_exceeds_twenty_five_characters";
        let vocab_json = format!(
            r#"{{"vocab":{{"<|pad|>":0,"<|im_start|>":1,"<|im_end|>":2,"<|unk|>":3,"{}":4}}}}"#,
            long_token
        );
        let path = fixture(&vocab_json);
        let tok = TaraTokenizer::from_file(&path).expect("must parse long token vocab");
        assert_eq!(tok.max_token_len, long_token.len());
        let encoded = tok.encode(long_token);
        assert_eq!(encoded, vec![4], "Long token must be matched in single step");
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, long_token);
        std::fs::remove_file(path).unwrap();
    }
}
