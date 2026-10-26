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
use crate::inference::model_expansion::{ExpansionError, GrowthType, ModelExpansionEngine};
use crate::tokenizer::TaraTokenizer;

#[derive(Debug, Error)]
pub enum VocabBuilderError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("expansion error: {0}")]
    Expansion(#[from] ExpansionError),
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

    /// Create a new VocabBuilder for building a standalone canonical base vocabulary
    /// from dataset shards without requiring an existing baseline model.
    pub fn new_standalone<P: AsRef<Path>>(dataset_dir: P, target_vocab_size: usize) -> Self {
        Self {
            dataset_dir: dataset_dir.as_ref().to_path_buf(),
            base_model_dir: PathBuf::new(),
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

        let cache_path = self
            .dataset_dir
            .join(format!("candidate_tokens_{}.json", self.target_vocab_size));
        if cache_path.exists() {
            if let Ok(file) = File::open(&cache_path) {
                if let Ok((cached_tokens, cached_report)) =
                    serde_json::from_reader::<_, (Vec<String>, VocabBuildReport)>(file)
                {
                    let required = self.target_vocab_size.saturating_sub(baseline_vocab_size);
                    let has_byte_tokens = cached_tokens.iter().any(|t| t == "<0x00>");
                    if cached_tokens.len() == required
                        && cached_report.baseline_vocab_size == baseline_vocab_size
                        && (required < 256 || has_byte_tokens)
                    {
                        println!("  [VocabBuilder] Loaded {} verified candidate tokens from operational cache '{}'", cached_tokens.len(), cache_path.display());
                        return Ok((cached_tokens, cached_report));
                    }
                }
            }
        }

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
                    .or_else(|| val.get("text"))
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

        // 3. Priority Tier Candidate Allocation Pass
        // Allocate candidate tokens in prioritized tiers up to target_vocab_size without static limits:
        // Tier 1: 256 Raw Byte Fallback Tokens (<0x00> through <0xFF>)
        // Tier 2: Systematic Kannada Unicode Block (U+0C80..=U+0CFF)
        // Tier 3: Systematic Devanagari Unicode Block (U+0900..=U+097F)
        // Tier 4: Canonical Coding Operators, Whitespace, Typography, Math, and Core Keywords
        // Tier 5: Unique Dataset Characters (sorted by script priority tier)
        let mut added_tokens: Vec<String> = Vec::new();
        let max_new_tokens = self.target_vocab_size.saturating_sub(baseline_vocab_size);

        // Priority 1: 256 Raw Byte Fallback Tokens (<0x00> through <0xFF>)
        for byte_tok in canonical_byte_tokens() {
            if added_tokens.len() >= max_new_tokens {
                break;
            }
            if !existing_vocab.contains(&byte_tok) {
                existing_vocab.insert(byte_tok.clone());
                added_tokens.push(byte_tok);
            }
        }

        // Priority 2: Systematic Kannada Unicode Block (U+0C80..=U+0CFF)
        for ch in canonical_kannada_chars() {
            if added_tokens.len() >= max_new_tokens {
                break;
            }
            let s = ch.to_string();
            if !existing_vocab.contains(&s) && !ch.is_control() && ch != '\0' && !s.trim().is_empty() {
                existing_vocab.insert(s.clone());
                added_tokens.push(s);
            }
        }

        // Priority 3: Systematic Devanagari Unicode Block (U+0900..=U+097F)
        for ch in canonical_devanagari_chars() {
            if added_tokens.len() >= max_new_tokens {
                break;
            }
            let s = ch.to_string();
            if !existing_vocab.contains(&s) && !ch.is_control() && ch != '\0' && !s.trim().is_empty() {
                existing_vocab.insert(s.clone());
                added_tokens.push(s);
            }
        }

        // Priority 4: Canonical Coding Operators, Whitespace, Typography, Math, and Core Keywords
        for sym in canonical_coding_and_symbols() {
            if added_tokens.len() >= max_new_tokens {
                break;
            }
            if !sym.is_empty() && !existing_vocab.contains(&sym) {
                existing_vocab.insert(sym.clone());
                added_tokens.push(sym);
            }
        }

        // Priority 5: Dataset Unique Characters (sorted by script priority tier)
        let mut missing_chars_set: HashSet<char> = HashSet::new();
        for &ch in char_counts.keys() {
            let s = ch.to_string();
            if !existing_vocab.contains(&s)
                && !ch.is_control()
                && ch != '\0'
                && !s.trim().is_empty()
            {
                missing_chars_set.insert(ch);
            }
        }

        let mut missing_chars: Vec<char> = missing_chars_set.into_iter().collect();
        missing_chars.sort_by(|&a, &b| {
            let tier_a = char_script_priority_tier(a);
            let tier_b = char_script_priority_tier(b);
            tier_a.cmp(&tier_b).then_with(|| a.cmp(&b))
        });

        for ch in missing_chars {
            if added_tokens.len() >= max_new_tokens {
                break;
            }
            let s = ch.to_string();
            if !existing_vocab.contains(&s) {
                existing_vocab.insert(s.clone());
                added_tokens.push(s);
            }
        }
        let missing_characters_added = added_tokens.len();

        // 4. Subword / Word Candidate Scoring & Script-Prioritized Deterministic Ranking
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
            let char_len = tok.chars().count();
            if freq >= self.min_frequency
                && !tok.trim().is_empty()
                && !existing_vocab.contains(&tok)
                && char_len <= self.max_token_len
                && char_len > 1
            {
                let score = freq.saturating_mul(char_len.saturating_sub(1));
                candidates.push(Candidate {
                    token: tok,
                    freq,
                    score,
                    len: char_len,
                });
            }
        }

        // Script-prioritized deterministic sort:
        // Tier 0 (Kannada Core) > Tier 1 (Devanagari Sanskrit/Hindi) > Tier 2 (Other Non-Latin) > Tier 3 (Latin/English Side-Fallback)
        // Within each script tier: score DESC, freq DESC, len DESC, token ASC
        candidates.sort_by(|a, b| {
            let tier_a = script_priority_tier(&a.token);
            let tier_b = script_priority_tier(&b.token);
            tier_a
                .cmp(&tier_b)
                .then_with(|| b.score.cmp(&a.score))
                .then_with(|| b.freq.cmp(&a.freq))
                .then_with(|| b.len.cmp(&a.len))
                .then_with(|| a.token.cmp(&b.token))
        });

        // 5. Select candidates up to target_vocab_size
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

        let cache_path = self
            .dataset_dir
            .join(format!("candidate_tokens_{}.json", self.target_vocab_size));
        if let Ok(file) = File::create(&cache_path) {
            let _ = serde_json::to_writer_pretty(file, &(&added_tokens, &report));
        }

        Ok((added_tokens, report))
    }

    /// Construct a standalone canonical Indic-first vocabulary directly from dataset shards
    /// without requiring an existing baseline model or `tokenizer.json`.
    ///
    /// The resulting vocabulary contains:
    /// - Canonical special control tokens (<|pad|>, <|im_start|>, <|im_end|>, <|unk|>, <|bos|>, <|eos|>, cognitive control tokens)
    /// - All 256 byte fallback tokens (<0x00> through <0xFF>)
    /// - Full canonical Kannada script and numerals (U+0C80..=U+0CFF)
    /// - Full canonical Devanagari Sanskrit/Hindi script and numerals (U+0900..=U+097F)
    /// - Canonical coding operators, indentations, typography, math symbols, and keywords
    /// - Unique dataset characters
    /// - Mined subwords ranked by script priority tier up to `self.target_vocab_size`
    ///
    /// Returns `(tokenizer_json_string, report)`. The JSON string conforms to the `TaraTokenizer` schema
    /// and guarantees dense contiguous IDs `0..vocab_size`.
    pub fn build_canonical_base_vocab(&self) -> Result<(String, VocabBuildReport), VocabBuilderError> {
        let shards = self.discover_shards()?;
        if shards.is_empty() {
            return Err(VocabBuilderError::Config(
                "No JSONL shards found in dataset directory".into(),
            ));
        }

        // Stream all dataset records and collect character and subword frequencies
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
                    .or_else(|| val.get("text"))
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

        let mut ordered_tokens: Vec<String> = Vec::with_capacity(self.target_vocab_size);
        let mut existing_vocab: HashSet<String> = HashSet::with_capacity(self.target_vocab_size);

        // 1. Canonical Special Control Tokens
        for &sp in canonical_special_tokens().iter() {
            if ordered_tokens.len() >= self.target_vocab_size {
                break;
            }
            let s = sp.to_string();
            if !existing_vocab.contains(&s) {
                existing_vocab.insert(s.clone());
                ordered_tokens.push(s);
            }
        }

        // 2. 256 Raw Byte Fallback Tokens (<0x00> through <0xFF>)
        for byte_tok in canonical_byte_tokens() {
            if ordered_tokens.len() >= self.target_vocab_size {
                break;
            }
            if !existing_vocab.contains(&byte_tok) {
                existing_vocab.insert(byte_tok.clone());
                ordered_tokens.push(byte_tok);
            }
        }

        // 3. Complete Kannada Unicode Block (U+0C80..=U+0CFF)
        for ch in canonical_kannada_chars() {
            if ordered_tokens.len() >= self.target_vocab_size {
                break;
            }
            let s = ch.to_string();
            if !existing_vocab.contains(&s) && !ch.is_control() && ch != '\0' && !s.trim().is_empty() {
                existing_vocab.insert(s.clone());
                ordered_tokens.push(s);
            }
        }

        // 4. Complete Devanagari Unicode Block (U+0900..=U+097F)
        for ch in canonical_devanagari_chars() {
            if ordered_tokens.len() >= self.target_vocab_size {
                break;
            }
            let s = ch.to_string();
            if !existing_vocab.contains(&s) && !ch.is_control() && ch != '\0' && !s.trim().is_empty() {
                existing_vocab.insert(s.clone());
                ordered_tokens.push(s);
            }
        }

        // 5. Canonical Coding Operators, Whitespace, Typography, Math, and Core Keywords
        for sym in canonical_coding_and_symbols() {
            if ordered_tokens.len() >= self.target_vocab_size {
                break;
            }
            if !existing_vocab.contains(&sym) && !sym.is_empty() {
                existing_vocab.insert(sym.clone());
                ordered_tokens.push(sym);
            }
        }

        // 6. Dataset Unique Characters
        let mut missing_chars_set: HashSet<char> = HashSet::new();
        for &ch in char_counts.keys() {
            let s = ch.to_string();
            if !existing_vocab.contains(&s)
                && !ch.is_control()
                && ch != '\0'
                && !s.trim().is_empty()
            {
                missing_chars_set.insert(ch);
            }
        }
        let mut missing_chars: Vec<char> = missing_chars_set.into_iter().collect();
        missing_chars.sort_by(|&a, &b| {
            let tier_a = char_script_priority_tier(a);
            let tier_b = char_script_priority_tier(b);
            tier_a.cmp(&tier_b).then_with(|| a.cmp(&b))
        });
        for ch in missing_chars {
            if ordered_tokens.len() >= self.target_vocab_size {
                break;
            }
            let s = ch.to_string();
            if !existing_vocab.contains(&s) {
                existing_vocab.insert(s.clone());
                ordered_tokens.push(s);
            }
        }
        let missing_characters_added = ordered_tokens.len();

        // 7. Subword Candidates ranked by Script Priority Tier
        struct Candidate {
            token: String,
            freq: usize,
            score: usize,
            len: usize,
        }
        let candidate_subwords_evaluated = token_counts.len();
        let mut candidates: Vec<Candidate> = Vec::with_capacity(candidate_subwords_evaluated);
        for (tok, freq) in token_counts {
            let char_len = tok.chars().count();
            if freq >= self.min_frequency
                && !tok.trim().is_empty()
                && !existing_vocab.contains(&tok)
                && char_len <= self.max_token_len
                && char_len > 1
            {
                let score = freq.saturating_mul(char_len.saturating_sub(1));
                candidates.push(Candidate {
                    token: tok,
                    freq,
                    score,
                    len: char_len,
                });
            }
        }
        candidates.sort_by(|a, b| {
            let tier_a = script_priority_tier(&a.token);
            let tier_b = script_priority_tier(&b.token);
            tier_a
                .cmp(&tier_b)
                .then_with(|| b.score.cmp(&a.score))
                .then_with(|| b.freq.cmp(&a.freq))
                .then_with(|| b.len.cmp(&a.len))
                .then_with(|| a.token.cmp(&b.token))
        });

        for cand in candidates {
            if ordered_tokens.len() >= self.target_vocab_size {
                break;
            }
            if !cand.token.trim().is_empty() && !existing_vocab.contains(&cand.token) {
                existing_vocab.insert(cand.token.clone());
                ordered_tokens.push(cand.token);
            }
        }

        let final_vocab_size = ordered_tokens.len();

        // Construct dense vocab map: token -> id
        let mut vocab_map = serde_json::Map::with_capacity(final_vocab_size);
        for (id, tok) in ordered_tokens.iter().enumerate() {
            vocab_map.insert(tok.clone(), serde_json::json!(id as u32));
        }

        let root = serde_json::json!({
            "version": "1.0.0",
            "model_max_length": 32768,
            "vocab": vocab_map,
            "vocab_size": final_vocab_size
        });

        let json_str = serde_json::to_string_pretty(&root)?;

        // Compute SHA-256
        let mut vocab_hasher = Sha256::new();
        for tok in &ordered_tokens {
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
            baseline_vocab_size: 0,
            final_vocab_size,
            added_tokens_count: final_vocab_size,
            candidate_vocab_sha256,
            shard_hashes,
        };

        Ok((json_str, report))
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
            let is_delim = ch.is_whitespace() || is_punctuation_or_delim(ch);

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

/// Check if a character is ASCII punctuation or delimiter.
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

/// Check if a character is a word delimiter or punctuation (including Indic danda and double danda).
pub fn is_punctuation_or_delim(ch: char) -> bool {
    is_ascii_punct(ch) || ch == '\u{0964}' || ch == '\u{0965}'
}

/// Returns all 256 canonical raw byte fallback tokens `<0x00>` through `<0xFF>`.
pub fn canonical_byte_tokens() -> Vec<String> {
    (0..=255u8).map(|b| format!("<0x{:02X}>", b)).collect()
}

/// Check if a token string matches the canonical raw byte token format `<0x00>`..`<0xFF>`.
pub fn is_byte_token(token: &str) -> bool {
    if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
        u8::from_str_radix(&token[3..5], 16).is_ok()
    } else {
        false
    }
}

/// Canonical special tokens recognized by TARA tokenizer.
pub fn canonical_special_tokens() -> Vec<&'static str> {
    vec![
        "<|pad|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|unk|>",
        "<|bos|>",
        "<|eos|>",
        "<|creator_auth|>",
        "<|tara_rule|>",
        "<|tara_exec|>",
        "<|tara_skill|>",
        "<|tara_memory|>",
    ]
}

/// Returns all canonical Kannada characters, vowel signs, virama, and Kannada numerals
/// by systematically iterating the entire assigned Kannada Unicode block (U+0C80..=U+0CFF).
pub fn canonical_kannada_chars() -> Vec<char> {
    (0x0C80u32..=0x0CFFu32)
        .filter_map(char::from_u32)
        .collect()
}

/// Returns all canonical Devanagari characters, vowel signs, virama, and numerals
/// for Sanskrit & Hindi by systematically iterating the entire Devanagari Unicode block (U+0900..=U+097F).
pub fn canonical_devanagari_chars() -> Vec<char> {
    (0x0900u32..=0x097Fu32)
        .filter_map(char::from_u32)
        .collect()
}

/// Script priority tier for individual characters:
/// - Tier 0: Kannada script and Kannada numerals (U+0C80..=U+0CFF)
/// - Tier 1: Devanagari script and numerals for Sanskrit & Hindi (U+0900..=U+097F)
/// - Tier 2: Other Non-ASCII / Multilingual characters
/// - Tier 3: ASCII characters (single-character fallback)
pub fn char_script_priority_tier(ch: char) -> u8 {
    let cp = ch as u32;
    if (0x0C80..=0x0CFF).contains(&cp) {
        0
    } else if (0x0900..=0x097F).contains(&cp) {
        1
    } else if !ch.is_ascii() {
        2
    } else {
        3
    }
}

/// Returns all canonical coding symbols, multi-character operators, indentations,
/// typography, and core programming keywords to ensure robust code comprehension and generation.
pub fn canonical_coding_and_symbols() -> Vec<String> {
    let tokens = [
        // Indentation & Whitespace
        "  ",       // 2-space indentation
        "    ",     // 4-space indentation
        "        ", // 8-space indentation
        "\r\n",     // Comparison, Logic & Assignment Operators
        "==", "!=", "<=", ">=", "===", "!==", "&&", "||", "+=", "-=", "*=", "/=", "%=", "++", "--",
        "**", ":=", // Scope, Navigation & Pointers
        "::", "->", "=>", "<-", "..", "...", "..=", // Bitwise & Shift Operators
        "<<", ">>", "&=", "|=", "^=", "<<=", ">>=",
        // Code Comments, Markup & Syntax Delimiters
        "//", "/*", "*/", "/**", "///", "//!", "</", "/>", "<!--", "-->", "<?", "?>", "${", "#{",
        // Common Typographic, Currency & Mathematical Symbols
        "“", "”", "‘", "’", "—", "–", "…", "•", "°", "±", "×", "÷", "≠", "≤", "≥", "≈", "∞", "√",
        "∑", "∏", "∫", "∂", "∆", "∇", "∈", "∉", "$", "₹", "€", "£", "¥",
        // Core Programming Keywords
        "fn", "let", "mut", "pub", "struct", "impl", "enum", "trait", "type", "match", "const",
        "async", "await", "self", "Self", "def", "class", "import", "from", "return", "yield",
        "if", "else", "elif", "while", "for", "in", "loop", "break", "continue", "true", "false",
        "null", "None",
    ];

    tokens.iter().map(|&s| s.to_string()).collect()
}

/// Check if a token is a known coding symbol, operator, indentation, or core keyword.
pub fn is_coding_token(token: &str) -> bool {
    matches!(
        token,
        "  " | "    "
            | "        "
            | "\r\n"
            | "=="
            | "!="
            | "<="
            | ">="
            | "==="
            | "!=="
            | "&&"
            | "||"
            | "+="
            | "-="
            | "*="
            | "/="
            | "%="
            | "++"
            | "--"
            | "**"
            | ":="
            | "::"
            | "->"
            | "=>"
            | "<-"
            | ".."
            | "..."
            | "..="
            | "<<"
            | ">>"
            | "&="
            | "|="
            | "^="
            | "<<="
            | ">>="
            | "//"
            | "/*"
            | "*/"
            | "/**"
            | "///"
            | "//!"
            | "</"
            | "/>"
            | "<!--"
            | "-->"
            | "<?"
            | "?>"
            | "${"
            | "#{"
            | "fn"
            | "let"
            | "mut"
            | "pub"
            | "struct"
            | "impl"
            | "enum"
            | "trait"
            | "type"
            | "match"
            | "const"
            | "async"
            | "await"
            | "self"
            | "Self"
            | "def"
            | "class"
            | "import"
            | "from"
            | "return"
            | "yield"
            | "if"
            | "else"
            | "elif"
            | "while"
            | "for"
            | "in"
            | "loop"
            | "break"
            | "continue"
            | "true"
            | "false"
            | "null"
            | "None"
            | "“"
            | "”"
            | "‘"
            | "’"
            | "—"
            | "–"
            | "…"
            | "•"
            | "°"
            | "±"
            | "×"
            | "÷"
            | "≠"
            | "≤"
            | "≥"
            | "≈"
            | "∞"
            | "√"
            | "∑"
            | "∏"
            | "∫"
            | "∂"
            | "∆"
            | "∇"
            | "∈"
            | "∉"
            | "$"
            | "₹"
            | "€"
            | "£"
            | "¥"
    )
}

/// Script priority tier for multi-character candidate tokens:
/// - Tier 0: Kannada (contains at least one Kannada character or Kannada numeral)
/// - Tier 1: Devanagari Sanskrit & Hindi (contains at least one Devanagari character or numeral)
/// - Tier 2: Coding, Programming Symbols & Syntax (Operators, keywords, indentations, symbols)
/// - Tier 3: Other Non-Latin / Multilingual scripts
/// - Tier 4: Pure Latin / General English prose words (secondary side-fallback)
pub fn script_priority_tier(token: &str) -> u8 {
    let mut has_kannada = false;
    let mut has_devanagari = false;
    let mut has_other_non_latin = false;

    for ch in token.chars() {
        let cp = ch as u32;
        if (0x0C80..=0x0CFF).contains(&cp) {
            has_kannada = true;
            break;
        } else if (0x0900..=0x097F).contains(&cp) {
            has_devanagari = true;
        } else if !ch.is_ascii() {
            has_other_non_latin = true;
        }
    }

    if has_kannada {
        0
    } else if has_devanagari {
        1
    } else if is_coding_token(token) {
        2
    } else if has_other_non_latin {
        3
    } else {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_canonical_byte_tokens_generation() {
        let byte_tokens = canonical_byte_tokens();
        assert_eq!(byte_tokens.len(), 256, "Must generate exactly 256 byte tokens");
        assert_eq!(byte_tokens[0], "<0x00>", "First byte token must be <0x00>");
        assert_eq!(byte_tokens[255], "<0xFF>", "Last byte token must be <0xFF>");

        let mut seen = HashSet::new();
        for (i, tok) in byte_tokens.iter().enumerate() {
            let expected = format!("<0x{:02X}>", i);
            assert_eq!(tok, &expected, "Byte token index {i} mismatch");
            assert!(is_byte_token(tok), "is_byte_token must recognize {tok}");
            assert!(seen.insert(tok.clone()), "Duplicate byte token {tok}");
        }

        assert!(!is_byte_token("<0x>"));
        assert!(!is_byte_token("<0x100>"));
        assert!(!is_byte_token("<0xGG>"));
        assert!(!is_byte_token("hello"));
    }

    #[test]
    fn test_canonical_kannada_unicode_iteration() {
        let kannada_chars = canonical_kannada_chars();
        assert_eq!(
            kannada_chars.len(),
            128,
            "Kannada Unicode block U+0C80..=U+0CFF must have 128 scalar values"
        );

        // Verify independent vowels
        let vowels = ['ಅ', 'ಆ', 'ಇ', 'ಈ', 'ಉ', 'ಊ', 'ಋ', 'ಎ', 'ಏ', 'ಐ', 'ಒ', 'ಓ', 'ಔ'];
        for v in vowels {
            assert!(
                kannada_chars.contains(&v),
                "Kannada vowel {v} must be present in canonical set"
            );
        }

        // Verify consonants
        let consonants = [
            'ಕ', 'ಖ', 'ಗ', 'ಘ', 'ಙ', 'ಚ', 'ಛ', 'ಜ', 'ಝ', 'ಞ', 'ಟ', 'ಠ', 'ಡ', 'ಢ', 'ಣ', 'ತ',
            'ಥ', 'ದ', 'ಧ', 'ನ', 'ಪ', 'ಫ', 'ಬ', 'ಭ', 'ಮ', 'ಯ', 'ರ', 'ಱ', 'ಲ', 'ವ', 'ಶ', 'ಷ',
            'ಸ', 'ಹ', 'ಳ', 'ೞ',
        ];
        for c in consonants {
            assert!(
                kannada_chars.contains(&c),
                "Kannada consonant {c} must be present in canonical set"
            );
        }

        // Verify dependent signs and virama
        let signs = ['ಾ', 'ಿ', 'ೀ', 'ು', 'ೂ', 'ೃ', 'ೆ', 'ೇ', 'ೈ', 'ೊ', 'ೋ', 'ೌ', '್'];
        for s in signs {
            assert!(
                kannada_chars.contains(&s),
                "Kannada sign {s} must be present in canonical set"
            );
        }

        // Verify special signs: Anusvara, Visarga, Nukta, Avagraha, Siddham
        assert!(kannada_chars.contains(&'ಂ'), "Anusvara must be present");
        assert!(kannada_chars.contains(&'ಃ'), "Visarga must be present");
        assert!(kannada_chars.contains(&'\u{0CBC}'), "Nukta must be present");
        assert!(kannada_chars.contains(&'\u{0CBD}'), "Avagraha must be present");
        assert!(kannada_chars.contains(&'\u{0C80}'), "Spacing Candrabindu must be present");
        assert!(kannada_chars.contains(&'\u{0C81}'), "Candrabindu must be present");
        assert!(kannada_chars.contains(&'\u{0C84}'), "Siddham must be present");
        assert!(kannada_chars.contains(&'\u{0CF1}'), "Jihvamuliya must be present");
        assert!(kannada_chars.contains(&'\u{0CF2}'), "Upadhmaniya must be present");
        assert!(kannada_chars.contains(&'\u{0CF3}'), "Combining Anusvara must be present");

        // Verify Kannada numerals: ೦, ೧, ೨, ೩, ೪, ೫, ೬, ೭, ೮, ೯ (U+0CE6..=U+0CEF)
        let numerals = ['೦', '೧', '೨', '೩', '೪', '೫', '೬', '೭', '೮', '೯'];
        for n in numerals {
            assert!(
                kannada_chars.contains(&n),
                "Kannada numeral {n} must be present in canonical set"
            );
        }
    }

    #[test]
    fn test_canonical_devanagari_unicode_iteration() {
        let devanagari_chars = canonical_devanagari_chars();
        assert_eq!(
            devanagari_chars.len(),
            128,
            "Devanagari Unicode block U+0900..=U+097F must have 128 scalar values"
        );

        // Verify vowels
        let vowels = ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ'];
        for v in vowels {
            assert!(
                devanagari_chars.contains(&v),
                "Devanagari vowel {v} must be present in canonical set"
            );
        }

        // Verify consonants
        let consonants = [
            'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त',
            'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स',
            'ह',
        ];
        for c in consonants {
            assert!(
                devanagari_chars.contains(&c),
                "Devanagari consonant {c} must be present in canonical set"
            );
        }

        // Verify virama, signs, and punctuation
        assert!(devanagari_chars.contains(&'्'), "Virama must be present");
        assert!(devanagari_chars.contains(&'ं'), "Anusvara must be present");
        assert!(devanagari_chars.contains(&'ः'), "Visarga must be present");
        assert!(devanagari_chars.contains(&'़'), "Nukta must be present");
        assert!(devanagari_chars.contains(&'ऽ'), "Avagraha must be present");
        assert!(devanagari_chars.contains(&'।'), "Danda must be present");
        assert!(devanagari_chars.contains(&'॥'), "Double Danda must be present");
        assert!(devanagari_chars.contains(&'ॐ'), "Om must be present");
        assert!(devanagari_chars.contains(&'\u{0900}'), "Inverted Candrabindu must be present");

        // Verify Devanagari numerals: ०, १, २, ३, ४, ५, ६, ७, ८, ९ (U+0966..=U+096F)
        let numerals = ['०', '१', '२', '३', '४', '५', '६', '७', '८', '९'];
        for n in numerals {
            assert!(
                devanagari_chars.contains(&n),
                "Devanagari numeral {n} must be present in canonical set"
            );
        }
    }

    #[test]
    fn test_canonical_coding_and_symbols() {
        let symbols = canonical_coding_and_symbols();
        assert!(
            symbols.iter().any(|s| s == "  "),
            "2-space indentation must be present"
        );
        assert!(
            symbols.iter().any(|s| s == "    "),
            "4-space indentation must be present"
        );
        assert!(
            symbols.iter().any(|s| s == "=="),
            "== operator must be present"
        );
        assert!(
            symbols.iter().any(|s| s == "!="),
            "!= operator must be present"
        );
        assert!(
            symbols.iter().any(|s| s == "::"),
            ":: operator must be present"
        );
        assert!(
            symbols.iter().any(|s| s == "->"),
            "-> operator must be present"
        );
        assert!(
            symbols.iter().any(|s| s == "=>"),
            "=> operator must be present"
        );
        assert!(
            symbols.iter().any(|s| s == "fn"),
            "fn keyword must be present"
        );
        assert!(
            symbols.iter().any(|s| s == "let"),
            "let keyword must be present"
        );
        assert!(
            symbols.iter().any(|s| s == "struct"),
            "struct keyword must be present"
        );
        assert!(
            symbols.iter().any(|s| s == "async"),
            "async keyword must be present"
        );
        assert!(
            symbols.iter().any(|s| s == "await"),
            "await keyword must be present"
        );
        assert!(
            symbols.iter().any(|s| s == "₹"),
            "Currency symbol ₹ must be present"
        );
        assert!(
            symbols.iter().any(|s| s == "€"),
            "Currency symbol € must be present"
        );
        assert!(
            symbols.iter().any(|s| s == "±"),
            "Math symbol ± must be present"
        );
        assert!(
            symbols.iter().any(|s| s == "×"),
            "Math symbol × must be present"
        );

        for sym in &symbols {
            assert!(
                is_coding_token(sym),
                "is_coding_token must recognize {sym}"
            );
        }
    }

    #[test]
    fn test_vocab_builder_collects_missing_chars_and_tokens() {
        let temp_dir = std::env::temp_dir().join(format!("tara_vocab_test_{}", std::process::id()));
        let ds_dir = temp_dir.join("dataset");
        let model_dir = temp_dir.join("model");

        fs::create_dir_all(&ds_dir).unwrap();
        fs::create_dir_all(&model_dir).unwrap();

        // Create minimal base tokenizer with 4 tokens
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
            tie_word_embeddings: None,
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
                "{{\"input\": \"Rust async function ಗಣಕಯಂತ್ರ\", \"output\": \"ನಮಸ್ಕಾರ ತಾರಾ ಕನ್ನಡ ೦ ೧ ೯\"}}"
            )
            .unwrap();
        }

        let builder = VocabBuilder::new(&ds_dir, &model_dir, 700).with_min_frequency(5);
        let (tokens, report) = builder.build_candidate_tokens().unwrap();

        assert_eq!(report.baseline_vocab_size, 4);
        assert!(tokens.len() <= 696); // Up to target 700 - 4
        // Verify byte tokens are generated and included
        assert!(tokens.iter().any(|t| t == "<0x00>"), "<0x00> must be present");
        assert!(tokens.iter().any(|t| t == "<0xFF>"), "<0xFF> must be present");
        // Verify Kannada numerals and characters
        assert!(tokens.iter().any(|t| t == "೦"), "Kannada numeral ೦ must be present");
        assert!(tokens.iter().any(|t| t == "೧"), "Kannada numeral ೧ must be present");
        assert!(tokens.iter().any(|t| t == "೯"), "Kannada numeral ೯ must be present");
        // Verify mined subwords
        assert!(tokens.iter().any(|t| t == "Rust"));
        assert!(tokens.iter().any(|t| t == "async"));
        assert!(tokens.iter().any(|t| t == "ನಮಸ್ಕಾರ"));
        assert!(tokens.iter().any(|t| t == "ಗಣಕಯಂತ್ರ"));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_indic_priority_and_kannada_numerals() {
        let temp_dir =
            std::env::temp_dir().join(format!("tara_indic_vocab_test_{}", std::process::id()));
        let ds_dir = temp_dir.join("dataset");
        let model_dir = temp_dir.join("model");

        fs::create_dir_all(&ds_dir).unwrap();
        fs::create_dir_all(&model_dir).unwrap();

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

        let mut shard = File::create(ds_dir.join("shard.jsonl")).unwrap();
        // High frequency English words vs lower frequency Kannada & Sanskrit words
        for _ in 0..50 {
            writeln!(
                shard,
                "{{\"input\": \"the system computation framework\", \"output\": \"ಕನ್ನಡ ಸಾಹಿತ್ಯ ಮತ್ತು ಸಂಸ್ಕೃತಿ ಗಣಕಯಂತ್ರ ೦ ೧ ೯ ೧೨೩೪೫\"}}"
            )
            .unwrap();
        }
        for _ in 0..30 {
            writeln!(
                shard,
                "{{\"input\": \"नमस्ते भारत धर्मक्षेत्रे कुरुक्षेत्रे\", \"output\": \"धर्मो रक्षति रक्षितः ० ९ ९೮೭\"}}"
            )
            .unwrap();
        }

        let builder = VocabBuilder::new(&ds_dir, &model_dir, 800).with_min_frequency(5);
        let (tokens, _report) = builder.build_candidate_tokens().unwrap();

        // 1. Kannada numerals must be present
        assert!(
            tokens.iter().any(|t| t == "೧"),
            "Kannada numeral ೧ must be present"
        );
        assert!(
            tokens.iter().any(|t| t == "೨"),
            "Kannada numeral ೨ must be present"
        );
        assert!(
            tokens.iter().any(|t| t == "೦"),
            "Kannada numeral ೦ must be present"
        );
        assert!(
            tokens.iter().any(|t| t == "೯"),
            "Kannada numeral ೯ must be present"
        );
        // 2. Devanagari numerals must be present
        assert!(
            tokens.iter().any(|t| t == "०"),
            "Devanagari numeral ० must be present"
        );
        assert!(
            tokens.iter().any(|t| t == "९"),
            "Devanagari numeral ९ must be present"
        );
        // 3. Kannada candidate subwords must be present
        assert!(
            tokens.iter().any(|t| t == "ಕನ್ನಡ"),
            "Kannada word ಕನ್ನಡ must be present"
        );
        assert!(
            tokens.iter().any(|t| t == "ಗಣಕಯಂತ್ರ"),
            "Kannada word ಗಣಕಯಂತ್ರ must be present"
        );
        // 4. Devanagari subwords must be present
        assert!(
            tokens
                .iter()
                .any(|t| t == "भारत" || t == "नमस्ते" || t == "धर्मक्षेत्रे" || t == "कुरुक्षेत्रे"),
            "Devanagari word must be present"
        );

        // 5. Canonical Coding Operators, Indentations & Core Keywords must be present
        assert!(
            tokens.iter().any(|t| t == "::"),
            "Scope operator :: must be present"
        );
        assert!(
            tokens.iter().any(|t| t == "->"),
            "Arrow operator -> must be present"
        );
        assert!(
            tokens.iter().any(|t| t == "=="),
            "Equality operator == must be present"
        );
        assert!(
            tokens.iter().any(|t| t == "fn"),
            "Rust fn keyword must be present"
        );
        assert!(
            tokens.iter().any(|t| t == "let"),
            "Rust let keyword must be present"
        );
        assert!(
            tokens.iter().any(|t| t == "    "),
            "4-space indentation must be present"
        );
        assert!(
            tokens.iter().any(|t| t == "₹"),
            "Currency symbol ₹ must be present"
        );

        // 6. Script priority test: Kannada subwords rank before pure English words
        let kannada_pos = tokens.iter().position(|t| t == "ಕನ್ನಡ");
        let english_pos = tokens.iter().position(|t| t == "system");
        if let (Some(k_pos), Some(e_pos)) = (kannada_pos, english_pos) {
            assert!(
                k_pos < e_pos,
                "Kannada token must rank ahead of English token (k_pos: {}, e_pos: {})",
                k_pos,
                e_pos
            );
        }

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_build_canonical_base_vocab_parses_cleanly_with_tara_tokenizer() {
        let temp_dir =
            std::env::temp_dir().join(format!("tara_standalone_vocab_test_{}", std::process::id()));
        let ds_dir = temp_dir.join("dataset");
        fs::create_dir_all(&ds_dir).unwrap();

        let mut shard = File::create(ds_dir.join("corpus.jsonl")).unwrap();
        for _ in 0..10 {
            writeln!(
                shard,
                "{{\"input\": \"fn main() {{ let x = ೧ + ०; }}\", \"output\": \"ಕನ್ನಡ ಸಾಹಿತ್ಯ ಮತ್ತು संस्कृतम् ೧೨೩ ०१२\"}}"
            )
            .unwrap();
        }

        let builder = VocabBuilder::new_standalone(&ds_dir, 750).with_min_frequency(2);
        let (json_str, report) = builder.build_canonical_base_vocab().unwrap();

        assert_eq!(report.baseline_vocab_size, 0);
        assert!(report.final_vocab_size <= 750);
        assert!(report.final_vocab_size >= 256 + 128 + 128); // Bytes + Kannada + Devanagari

        // Parse with TaraTokenizer - MUST parse cleanly without error
        let tokenizer = TaraTokenizer::from_json_str(&json_str)
            .expect("TaraTokenizer must parse generated canonical vocabulary JSON cleanly");

        assert_eq!(tokenizer.vocab_size, report.final_vocab_size);

        // Verify all 256 byte tokens are indexed in byte_to_id
        for b in 0..=255u8 {
            assert!(
                tokenizer.byte_to_id[b as usize].is_some(),
                "Byte 0x{:02X} must be mapped to token ID",
                b
            );
            let byte_id = tokenizer.byte_to_id[b as usize].unwrap();
            assert_eq!(tokenizer.id_to_byte.get(&byte_id), Some(&b));
        }

        // Verify Kannada numerals and characters have valid IDs
        for ch in ['೦', '೧', '೨', '೩', '೪', '೫', '೬', '೭', '೮', '೯', 'ಕ', 'ಅ'] {
            let s = ch.to_string();
            assert!(
                tokenizer.token_to_id.contains_key(&s),
                "Kannada char/numeral {s} must be in token_to_id"
            );
        }

        // Verify Devanagari numerals and characters have valid IDs
        for ch in ['०', '१', '२', '३', '४', '५', '६', '७', '८', '९', 'क', 'अ'] {
            let s = ch.to_string();
            assert!(
                tokenizer.token_to_id.contains_key(&s),
                "Devanagari char/numeral {s} must be in token_to_id"
            );
        }

        // Verify coding operators
        for op in ["::", "->", "==", "fn", "let", "    "] {
            assert!(
                tokenizer.token_to_id.contains_key(op),
                "Coding token {op} must be in token_to_id"
            );
        }

        // Verify encode and decode round-trip with zero <|unk|>
        let test_text = "fn main() { let x = ೧ + ०; } // ಕನ್ನಡ & संस्कृतम्";
        let encoded_ids = tokenizer.encode(test_text);
        assert!(
            !encoded_ids.contains(&tokenizer.unk_id()),
            "Zero <|unk|> must be emitted for canonical text"
        );
        let decoded = tokenizer.decode(&encoded_ids);
        assert_eq!(
            decoded, test_text,
            "Lossless round-trip decode must match original text"
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_dynamic_vocab_sizing_respects_target() {
        let temp_dir =
            std::env::temp_dir().join(format!("tara_sizing_test_{}", std::process::id()));
        let ds_dir = temp_dir.join("dataset");
        fs::create_dir_all(&ds_dir).unwrap();

        let mut shard = File::create(ds_dir.join("data.jsonl")).unwrap();
        for _ in 0..5 {
            writeln!(
                shard,
                "{{\"input\": \"a b c d e f g h i j k l m n o p q r s t u v w x y z\", \"output\": \"ಕನ್ನಡ ೦ ೧ ೨\"}}"
            )
            .unwrap();
        }

        // Test with different target sizes
        for target in [100, 300, 500, 750] {
            let builder = VocabBuilder::new_standalone(&ds_dir, target).with_min_frequency(2);
            let (json_str, report) = builder.build_canonical_base_vocab().unwrap();

            assert!(
                report.final_vocab_size <= target,
                "Final vocab size {} must not exceed target {}",
                report.final_vocab_size,
                target
            );
            assert_eq!(report.final_vocab_size, report.added_tokens_count);

            let tokenizer = TaraTokenizer::from_json_str(&json_str)
                .expect("Dense ID validation must pass for all dynamic targets");
            assert_eq!(tokenizer.vocab_size, report.final_vocab_size);
        }

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
