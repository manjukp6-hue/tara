//! Native Rust Data-Driven Vocabulary Builder for TARA Engine.
//!
//! Features:
//! - Full dataset streaming: scans all dynamically discovered `.jsonl` shards with zero fixed record limit.
//! - Unicode & script coverage: discovers all unique Unicode characters across all 13 Indian languages,
//!   code syntax, math operators, and punctuation.
//! - Baseline preservation: guarantees baseline token IDs 0..343 remain 100% untouched.
//! - Subword frequency mining & ranking: scores candidates by token length and corpus frequency.
//! - Strict collision avoidance: rejects duplicates, empty strings, and tokens already in baseline.
//! - Deterministic ordering: uses deterministic tie-breaking (score DESC, freq DESC, len DESC, alphabetical ASC).
//! - Direct integration with `ModelExpansionEngine` for atomic candidate model staging.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::config::TaraConfig;
use crate::model_expansion::{GrowthType, ModelExpansionEngine};
use crate::tokenizer::TaraTokenizer;

#[derive(Debug, Error)]
pub enum VocabBuilderError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("expansion error: {0}")]
    Expansion(#[from] crate::model_expansion::ExpansionError),
    #[error("safetensors error: {0}")]
    SafeTensors(#[from] crate::safetensors::SafeTensorsError),
    #[error("invalid configuration: {0}")]
    Config(String),
}

/// Metadata and statistics produced by vocabulary building.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabBuildReport {
    pub dataset_dir: String,
    pub total_records_scanned: usize,
    pub total_chars_scanned: usize,
    pub total_words_scanned: usize,
    pub unique_characters_found: usize,
    pub missing_characters_added: usize,
    pub candidate_subwords_evaluated: usize,
    pub baseline_vocab_size: usize,
    pub final_vocab_size: usize,
    pub added_tokens_count: usize,
    pub candidate_vocab_sha256: String,
    pub shard_hashes: HashMap<String, String>,
}

/// Data-driven vocabulary constructor.
pub struct VocabBuilder {
    pub dataset_dir: PathBuf,
    pub base_model_dir: PathBuf,
    pub target_vocab_size: usize,
    pub min_frequency: usize,
    pub max_token_len: usize,
}

impl VocabBuilder {
    /// Create a new VocabBuilder.
    pub fn new<P: AsRef<Path>, Q: AsRef<Path>>(
        dataset_dir: P,
        base_model_dir: Q,
        target_vocab_size: usize,
    ) -> Self {
        Self {
            dataset_dir: dataset_dir.as_ref().to_path_buf(),
            base_model_dir: base_model_dir.as_ref().to_path_buf(),
            target_vocab_size,
            min_frequency: 10,
            max_token_len: 25,
        }
    }

    /// Set minimum frequency threshold for candidate tokens.
    pub fn with_min_frequency(mut self, min_freq: usize) -> Self {
        self.min_frequency = min_freq;
        self
    }

    /// Set maximum token character length.
    pub fn with_max_token_len(mut self, max_len: usize) -> Self {
        self.max_token_len = max_len;
        self
    }

    /// Discover all `.jsonl` files in dataset_dir dynamically.
    pub fn discover_shards(&self) -> Result<Vec<PathBuf>, VocabBuilderError> {
        let mut shards = Vec::new();
        if !self.dataset_dir.exists() || !self.dataset_dir.is_dir() {
            return Err(VocabBuilderError::Config(format!(
                "Dataset directory does not exist: {}",
                self.dataset_dir.display()
            )));
        }

        for entry in fs::read_dir(&self.dataset_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
                shards.push(path);
            }
        }
        shards.sort();
        Ok(shards)
    }

    /// Execute the complete vocabulary construction from the dataset.
    ///
    /// Returns `(added_tokens, report)`.
    pub fn build_candidate_tokens(
        &self,
    ) -> Result<(Vec<String>, VocabBuildReport), VocabBuilderError> {
        let shards = self.discover_shards()?;
        if shards.is_empty() {
            return Err(VocabBuilderError::Config(
                "No JSONL shards found in dataset directory".into(),
            ));
        }

        // 1. Load baseline tokenizer to extract existing tokens
        let base_tok_path = self.base_model_dir.join("tokenizer.json");
        let base_tokenizer =
            TaraTokenizer::from_file(&base_tok_path.to_string_lossy()).map_err(|e| {
                VocabBuilderError::Config(format!("Failed to load baseline tokenizer: {e}"))
            })?;
        let baseline_vocab_size = base_tokenizer.vocab_size;

        let mut existing_vocab: HashSet<String> =
            base_tokenizer.token_to_id.keys().cloned().collect();

        // 2. Stream ALL dataset records and collect character and subword frequencies
        let mut char_counts: HashMap<char, usize> = HashMap::new();
        let mut token_counts: HashMap<String, usize> = HashMap::new();
        let mut shard_hashes: HashMap<String, String> = HashMap::new();

        let mut total_records = 0usize;
        let mut total_chars = 0usize;
        let mut total_words = 0usize;

        for shard_path in &shards {
            let shard_name = shard_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("shard")
                .to_string();
            let mut hasher = Sha256::new();

            let file = File::open(shard_path)?;
            let reader = BufReader::with_capacity(128 * 1024, file);

            for line_res in reader.lines() {
                let line = line_res?;
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                hasher.update(trimmed.as_bytes());
                hasher.update(b"\n");
                total_records += 1;

                let val: Value = match serde_json::from_str(trimmed) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let input = val
                    .get("input")
                    .or_else(|| val.get("prompt"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let output = val
                    .get("output")
                    .or_else(|| val.get("completion"))
                    .and_then(Value::as_str)
                    .unwrap_or("");

                self.accumulate_text(
                    input,
                    &mut char_counts,
                    &mut token_counts,
                    &mut total_chars,
                    &mut total_words,
                );
                self.accumulate_text(
                    output,
                    &mut char_counts,
                    &mut token_counts,
                    &mut total_chars,
                    &mut total_words,
                );
            }

            let shard_sha = hex::encode(hasher.finalize());
            shard_hashes.insert(shard_name, shard_sha);
        }

        // 3. Mandatory Character Coverage Pass
        // Add all unique characters not present in baseline to guarantee zero single-char <|unk|>
        let mut missing_chars: Vec<char> = char_counts
            .keys()
            .copied()
            .filter(|&ch| {
                let s = ch.to_string();
                !existing_vocab.contains(&s)
                    && !ch.is_control()
                    && ch != '\0'
                    && !s.trim().is_empty()
            })
            .collect();
        missing_chars.sort(); // Deterministic alphabetical/codepoint order

        let mut added_tokens: Vec<String> = Vec::new();
        let slots_for_chars = self.target_vocab_size.saturating_sub(baseline_vocab_size);

        for ch in missing_chars.into_iter().take(slots_for_chars) {
            let s = ch.to_string();
            if !s.trim().is_empty() && !existing_vocab.contains(&s) {
                existing_vocab.insert(s.clone());
                added_tokens.push(s);
            }
        }
        let missing_characters_added = added_tokens.len();

        // 4. Subword / Word Candidate Scoring & Deterministic Ranking
        // Score = frequency * (len - 1).max(1) (representing total character savings)
        struct Candidate {
            token: String,
            freq: usize,
            score: usize,
            len: usize,
        }

        let candidate_subwords_evaluated = token_counts.len();
        let mut candidates: Vec<Candidate> = Vec::with_capacity(candidate_subwords_evaluated);
        for (tok, freq) in token_counts {
            if freq >= self.min_frequency
                && !tok.trim().is_empty()
                && !existing_vocab.contains(&tok)
                && tok.len() <= self.max_token_len
            {
                let char_len = tok.chars().count();
                if char_len > 1 {
                    let score = freq.saturating_mul(char_len.saturating_sub(1));
                    candidates.push(Candidate {
                        token: tok,
                        freq,
                        score,
                        len: char_len,
                    });
                }
            }
        }

        // Deterministic Sort: score DESC, freq DESC, len DESC, token ASC
        candidates.sort_by(|a, b| {
            b.score
                .cmp(&a.score)
                .then_with(|| b.freq.cmp(&a.freq))
                .then_with(|| b.len.cmp(&a.len))
                .then_with(|| a.token.cmp(&b.token))
        });

        // 5. Select candidates up to target_vocab_size
        let max_new_tokens = self.target_vocab_size.saturating_sub(baseline_vocab_size);
        for cand in candidates.into_iter() {
            if added_tokens.len() >= max_new_tokens {
                break;
            }
            if !cand.token.trim().is_empty() && !existing_vocab.contains(&cand.token) {
                existing_vocab.insert(cand.token.clone());
                added_tokens.push(cand.token);
            }
        }

        let final_vocab_size = baseline_vocab_size + added_tokens.len();

        // Compute candidate vocabulary SHA-256
        let mut vocab_hasher = Sha256::new();
        for tok in &added_tokens {
            vocab_hasher.update(tok.as_bytes());
            vocab_hasher.update(b"\n");
        }
        let candidate_vocab_sha256 = hex::encode(vocab_hasher.finalize());

        let report = VocabBuildReport {
            dataset_dir: self.dataset_dir.display().to_string(),
            total_records_scanned: total_records,
            total_chars_scanned: total_chars,
            total_words_scanned: total_words,
            unique_characters_found: char_counts.len(),
            missing_characters_added,
            candidate_subwords_evaluated,
            baseline_vocab_size,
            final_vocab_size,
            added_tokens_count: added_tokens.len(),
            candidate_vocab_sha256,
            shard_hashes,
        };

        Ok((added_tokens, report))
    }

    /// Build vocabulary and immediately hand off to `ModelExpansionEngine`
    /// to stage an atomic candidate model at `candidate_output_dir`.
    pub fn build_and_stage_candidate_model<P: AsRef<Path>>(
        &self,
        candidate_output_dir: P,
    ) -> Result<VocabBuildReport, VocabBuilderError> {
        let dest = candidate_output_dir.as_ref();
        if dest.exists() {
            return Err(VocabBuilderError::Config(format!(
                "Candidate destination already exists: {}",
                dest.display()
            )));
        }

        // 1. Build data-driven candidate tokens
        let (added_tokens, report) = self.build_candidate_tokens()?;

        // 2. Prepare target config
        let source_cfg_path = self.base_model_dir.join("config.json");
        let mut target_cfg = TaraConfig::from_json_file(&source_cfg_path.to_string_lossy())
            .map_err(|e| VocabBuilderError::Config(e.to_string()))?;
        target_cfg.vocab_size = report.final_vocab_size;

        // 3. Expand weights
        let expansion_engine = ModelExpansionEngine::new(&self.base_model_dir.to_string_lossy());
        let (expanded_weights, _growth_meta) =
            expansion_engine.expand_model(&target_cfg, &GrowthType::VocabExpansion)?;

        // 4. Stage candidate model atomically
        expansion_engine.write_expanded_model_with_tokens(
            &expanded_weights,
            &target_cfg,
            &dest.to_string_lossy(),
            &added_tokens,
        )?;

        // 5. Copy helper metadata (tokenizer_config.json, special_tokens_map.json, etc.)
        let src_tok_cfg = self.base_model_dir.join("tokenizer_config.json");
        if src_tok_cfg.exists() {
            let mut tok_cfg_val: Value = serde_json::from_str(&fs::read_to_string(&src_tok_cfg)?)?;
            if let Some(obj) = tok_cfg_val.as_object_mut() {
                obj.insert(
                    "vocab_size".to_string(),
                    serde_json::json!(report.final_vocab_size),
                );
            }
            fs::write(
                dest.join("tokenizer_config.json"),
                serde_json::to_vec_pretty(&tok_cfg_val)?,
            )?;
        }

        let src_sp_map = self.base_model_dir.join("special_tokens_map.json");
        if src_sp_map.exists() {
            fs::copy(&src_sp_map, dest.join("special_tokens_map.json"))?;
        }

        // 6. Write candidate vocabulary manifest & statistics
        let manifest_path = dest.join("vocabulary_manifest.json");
        fs::write(&manifest_path, serde_json::to_vec_pretty(&report)?)?;

        Ok(report)
    }

    fn accumulate_text(
        &self,
        text: &str,
        char_counts: &mut HashMap<char, usize>,
        token_counts: &mut HashMap<String, usize>,
        total_chars: &mut usize,
        total_words: &mut usize,
    ) {
        let chars: Vec<char> = text.chars().collect();
        *total_chars += chars.len();

        for &ch in &chars {
            *char_counts.entry(ch).or_insert(0) += 1;
        }

        // Extract word tokens separated by whitespace or punctuation boundaries
        let mut start = 0;
        let n = chars.len();

        for i in 0..n {
            let ch = chars[i];
            let is_delim = ch.is_whitespace() || is_ascii_punct(ch);

            if is_delim {
                if i > start {
                    let word_len = i - start;
                    if word_len <= self.max_token_len {
                        let word: String = chars[start..i].iter().collect();
                        *token_counts.entry(word).or_insert(0) += 1;
                        *total_words += 1;
                    }
                }
                start = i + 1;
            }
        }
        if n > start {
            let word_len = n - start;
            if word_len <= self.max_token_len {
                let word: String = chars[start..n].iter().collect();
                *token_counts.entry(word).or_insert(0) += 1;
                *total_words += 1;
            }
        }
    }
}

fn is_ascii_punct(ch: char) -> bool {
    matches!(
        ch,
        '!' | '"'
            | '#'
            | '$'
            | '%'
            | '&'
            | '\''
            | '('
            | ')'
            | '*'
            | '+'
            | ','
            | '-'
            | '.'
            | '/'
            | ':'
            | ';'
            | '<'
            | '='
            | '>'
            | '?'
            | '@'
            | '['
            | '\\'
            | ']'
            | '^'
            | '_'
            | '`'
            | '{'
            | '|'
            | '}'
            | '~'
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_vocab_builder_collects_missing_chars_and_tokens() {
        let temp_dir = std::env::temp_dir().join(format!("tara_vocab_test_{}", std::process::id()));
        let ds_dir = temp_dir.join("dataset");
        let model_dir = temp_dir.join("model");

        fs::create_dir_all(&ds_dir).unwrap();
        fs::create_dir_all(&model_dir).unwrap();

        // Create dummy base tokenizer with 4 tokens
        let base_tok = serde_json::json!({
            "version": "1.0.0",
            "vocab_size": 4,
            "vocab": {
                "<|pad|>": 0,
                "<|im_start|>": 1,
                "<|im_end|>": 2,
                "<|unk|>": 3
            }
        });
        fs::write(
            model_dir.join("tokenizer.json"),
            serde_json::to_vec_pretty(&base_tok).unwrap(),
        )
        .unwrap();

        // Create base config
        let cfg = TaraConfig {
            vocab_size: 4,
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
            version: "test".into(),
            model_type: "tara".into(),
        };
        fs::write(
            model_dir.join("config.json"),
            serde_json::to_vec_pretty(&cfg).unwrap(),
        )
        .unwrap();

        // Create a test shard with multilingual text
        let mut shard = File::create(ds_dir.join("test.jsonl")).unwrap();
        for _ in 0..15 {
            writeln!(
                shard,
                "{{\"input\": \"Rust async function\", \"output\": \"ನಮಸ್ಕಾರ ತಾರಾ\"}}"
            )
            .unwrap();
        }

        let builder = VocabBuilder::new(&ds_dir, &model_dir, 50).with_min_frequency(5);
        let (tokens, report) = builder.build_candidate_tokens().unwrap();

        assert_eq!(report.baseline_vocab_size, 4);
        assert!(tokens.len() <= 46); // Up to target 50
        assert!(tokens.iter().any(|t| t == "Rust"));
        assert!(tokens.iter().any(|t| t == "async"));

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
