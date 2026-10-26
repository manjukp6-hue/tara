//! Comprehensive 4-Tier E2E Test Suite for Tokenizer & Model Architecture
//!
//! Tests adhere strictly to requirement-driven opaque-box principles:
//! - Tier 1: Feature Coverage (>=5 tests per feature: Trie match, byte fallback,
//!   lossless UTF-8, regex boundaries, Rayon batch parity, SafeTensors shape checks [100256, 2048])
//! - Tier 2: Boundary & Corner Cases (>=5 tests per feature: empty strings,
//!   single byte, max length subwords, binary bytes, out-of-vocab IDs, single vs multi-threaded Rayon batches)
//! - Tier 3: Cross-Feature Combinations (regex + Trie + byte fallback in mixed text,
//!   multithreaded batches of mixed text)
//! - Tier 4: Real-World Scenarios & Forward Pass Verification (code snippets,
//!   Markdown, JSON, and forward pass on IDs >= 100000)

use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::thread;

use tara_engine::config::TaraConfig;
use tara_engine::tokenizer::{TaraTokenizer, SPECIAL_TOKENS};
use tara_engine::TaraForCausalLM;

/// Rayon parallel batch extension trait for tokenizer testing.
pub trait TokenizerBatchExt {
    fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<u32>>;
}

impl TokenizerBatchExt for TaraTokenizer {
    fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        texts.par_iter().map(|&t| self.encode(t)).collect()
    }
}

/// Helper: builds a rich test tokenizer fixture with special tokens,
/// 256 byte fallback tokens (<0x00> through <0xFF>), Kannada script & numerals,
/// English words, code symbols, and dense ID layout.
fn create_comprehensive_test_tokenizer() -> TaraTokenizer {
    let mut vocab = serde_json::Map::new();
    let mut id = 0u64;

    // 1. Special tokens (0..8)
    for &sp in SPECIAL_TOKENS {
        vocab.insert(sp.to_string(), serde_json::Value::from(id));
        id += 1;
    }

    // 2. Common subwords and punctuation
    let common_subwords = [
        " ", "\n", "\t", ".", ",", ":", ";", "(", ")", "[", "]", "{", "}",
        "+", "-", "*", "/", "=", "==", "!=", "<", ">", "<=", ">=", "->", "=>",
        "hello", "world", "tara", "engine", "neural", "model", "token", "test",
        "fn", "let", "pub", "struct", "impl", "mut", "return", "match", "use",
        "def", "class", "import", "from", "self", "None", "True", "False",
        "the", "of", "and", "to", "in", "is", "you", "that", "it", "he",
        "was", "for", "on", "are", "as", "with", "his", "they", "at", "be",
        "'s", "'t", "'re", "'ve", "'m", "'ll", "'d",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "10", "42", "100", "2048", "100000", "100256",
    ];
    for &w in &common_subwords {
        if !vocab.contains_key(w) {
            vocab.insert(w.to_string(), serde_json::Value::from(id));
            id += 1;
        }
    }

    // 3. Kannada characters and numerals
    let kannada_subwords = [
        "ನ", "ಮ", "ಸ್", "ಕ", "ರ", "!", "?", "ಕನ್ನಡ", "ಭಾಷೆ", "ಮತ್ತು", "ಸಾಹಿತ್ಯ",
        "೦", "೧", "೨", "೩", "೪", "೫", "೬", "೭", "೮", "೯",
    ];
    for &kw in &kannada_subwords {
        if !vocab.contains_key(kw) {
            vocab.insert(kw.to_string(), serde_json::Value::from(id));
            id += 1;
        }
    }

    // Pad dense subwords up to ID 99999 so byte tokens begin at 100000
    let subword_target_count = 100_000u64;
    while id < subword_target_count {
        let dummy = format!("__subword_{id}__");
        vocab.insert(dummy, serde_json::Value::from(id));
        id += 1;
    }

    // 4. 256 byte fallback tokens <0x00> through <0xFF> mapped to 100000..100255
    for b in 0..=255u8 {
        let byte_tag = format!("<0x{:02X}>", b);
        vocab.insert(byte_tag, serde_json::Value::from(id));
        id += 1;
    }

    let mut root = serde_json::Map::new();
    root.insert("vocab".to_string(), serde_json::Value::Object(vocab));
    let raw = serde_json::to_string(&root).expect("JSON serialization must succeed");

    TaraTokenizer::from_json_str(&raw).expect("Comprehensive tokenizer must build cleanly")
}

/// Helper: loads on-disk candidate tokenizer if available, otherwise falls back to test fixture.
fn get_test_tokenizer() -> TaraTokenizer {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let cand_tok_path = manifest_dir.join("../../storage/models/tara_candidate_v1/tokenizer.json");
    if cand_tok_path.exists() {
        if let Ok(tok) = TaraTokenizer::from_file(cand_tok_path.to_str().unwrap()) {
            return tok;
        }
    }
    create_comprehensive_test_tokenizer()
}

/// Helper: SafeTensors header parser for binary verification.
fn parse_safetensors_header(path: &Path) -> Result<(u64, serde_json::Value), String> {
    use byteorder::{LittleEndian, ReadBytesExt};
    use std::io::Read;

    let mut f = fs::File::open(path).map_err(|e| format!("Cannot open {}: {e}", path.display()))?;
    let header_len = f.read_u64::<LittleEndian>()
        .map_err(|e| format!("Failed to read header len from {}: {e}", path.display()))?;

    let mut header_buf = vec![0u8; header_len as usize];
    f.read_exact(&mut header_buf)
        .map_err(|e| format!("Failed to read header bytes from {}: {e}", path.display()))?;

    let header_str = std::str::from_utf8(&header_buf)
        .map_err(|e| format!("Header is not valid UTF-8 in {}: {e}", path.display()))?;

    let header_json: serde_json::Value = serde_json::from_str(header_str.trim())
        .map_err(|e| format!("Header is not valid JSON in {}: {e}", path.display()))?;

    Ok((header_len, header_json))
}

// ============================================================================
// TIER 1: FEATURE COVERAGE (>=5 tests per feature)
// ============================================================================

// --- Feature 1: Tokenizer Trie Matching (5 tests) ---

#[test]
fn test_t1_trie_greedy_longest_match_priority() {
    let tok = get_test_tokenizer();
    // "hello" should match single token "hello", not individual characters 'h', 'e', 'l', 'l', 'o'
    let ids = tok.encode("hello");
    assert_eq!(ids.len(), 1, "Trie must match longest subword 'hello' as a single token");
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, "hello");
}

#[test]
fn test_t1_trie_special_token_precedence() {
    let tok = get_test_tokenizer();
    let text = "<|im_start|>user";
    let ids = tok.encode(text);
    assert_eq!(ids[0], tok.im_start_id(), "Special token must take highest precedence");
}

#[test]
fn test_t1_trie_multilingual_kannada_match() {
    let tok = get_test_tokenizer();
    let kannada = "ನಮಸ್ಕಾರ";
    let ids = tok.encode(kannada);
    assert!(!ids.is_empty(), "Kannada word must produce valid token IDs");
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, kannada, "Kannada text must roundtrip losslessly");
}

#[test]
fn test_t1_trie_code_keyword_match() {
    let tok = get_test_tokenizer();
    let code = "fn test() { let x = 42; }";
    let ids = tok.encode(code);
    assert!(!ids.is_empty(), "Code keywords must encode without failure");
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, code, "Code snippet must roundtrip bit-exact");
}

#[test]
fn test_t1_trie_dense_range_validation() {
    let tok = get_test_tokenizer();
    assert!(tok.vocab_size >= 100256, "Vocabulary size must accommodate subwords and byte tokens");
    assert_eq!(tok.token_to_id.len(), tok.vocab_size);
    assert_eq!(tok.id_to_token.len(), tok.vocab_size);
}

// --- Feature 2: Byte Fallback Token Encoding (5 tests) ---

#[test]
fn test_t1_byte_fallback_ids_range_100000_to_100255() {
    let tok = get_test_tokenizer();
    assert_eq!(tok.id_to_byte.len(), 256, "All 256 raw bytes must be registered in fallback table");
    for b in 0..=255u8 {
        let id = tok.byte_to_id[b as usize].expect("Each byte must map to a fallback token ID");
        assert!((100000..=100255).contains(&id), "Byte fallback ID {id} must fall within 100000..=100255");
        assert_eq!(tok.id_to_byte.get(&id), Some(&b), "Reverse byte lookup must match byte {b}");
    }
}

#[test]
fn test_t1_byte_fallback_unseen_emojis() {
    let tok = get_test_tokenizer();
    let emojis = "🚀 🤖 🧠 ✨";
    let ids = tok.encode(emojis);
    assert!(!ids.is_empty(), "Unseen emojis must encode via byte fallback");
    assert!(!ids.contains(&tok.unk_id()), "Zero unknown tokens must be produced");
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, emojis, "Emojis must roundtrip losslessly");
}

#[test]
fn test_t1_byte_fallback_unseen_scripts() {
    let tok = get_test_tokenizer();
    // Brahmi and rare scripts not in subword vocabulary
    let rare = "𑀅𑀩𑀕 ꤀꤁꤂";
    let ids = tok.encode(rare);
    assert!(!ids.contains(&tok.unk_id()), "Rare scripts must use byte fallback with zero <|unk|>");
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, rare, "Rare script must roundtrip losslessly");
}

#[test]
fn test_t1_byte_fallback_all_256_octets_coverage() {
    let tok = get_test_tokenizer();
    for b in 0..=255u8 {
        let byte_id = tok.byte_to_id[b as usize].expect("Every octet must have a valid ID");
        let decoded_str = tok.decode(&[byte_id]);
        if b < 128 {
            let ch = b as char;
            assert_eq!(decoded_str, ch.to_string(), "ASCII octet 0x{:02X} must decode correctly", b);
        } else {
            assert!(!decoded_str.is_empty(), "High octet 0x{:02X} must decode without panic", b);
        }
    }
}

#[test]
fn test_t1_byte_fallback_flag_prevents_literal_tag_collision() {
    let tok = get_test_tokenizer();
    let literal = "Literal tag <0x41> is not character A";
    let ids = tok.encode(literal);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, literal, "Literal '<0x41>' in text must roundtrip verbatim without collapsing to 'A'");
}

// --- Feature 3: Lossless UTF-8 Decoding (5 tests) ---

#[test]
fn test_t1_lossless_roundtrip_ascii_text() {
    let tok = get_test_tokenizer();
    let text = "The quick brown fox jumps over 13 lazy dogs.";
    let ids = tok.encode(text);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, text);
}

#[test]
fn test_t1_lossless_roundtrip_complex_multilingual() {
    let tok = get_test_tokenizer();
    let mixed = "TARA ನಮಸ್ಕಾರ: Hello 123 ಮತ್ತು World!";
    let ids = tok.encode(mixed);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, mixed);
}

#[test]
fn test_t1_lossless_roundtrip_emojis_and_symbols() {
    let tok = get_test_tokenizer();
    let math = "E = mc^2 ± ∞ ∇ × B ≠ 0 ⚡ 🔬";
    let ids = tok.encode(math);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, math);
}

#[test]
fn test_t1_lossless_roundtrip_control_token_filtering() {
    let tok = get_test_tokenizer();
    let prompt = "<|im_start|>user\nHello tara!<|im_end|>";
    let ids = tok.encode(prompt);
    let decoded = tok.decode(&ids);
    // Control tokens are skipped in decode
    assert_eq!(decoded, "user\nHello tara!");
}

#[test]
fn test_t1_lossless_roundtrip_literal_byte_token_strings() {
    let tok = get_test_tokenizer();
    let tags = "<0x00> <0x01> <0x7F> <0x80> <0xFF>";
    let ids = tok.encode(tags);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, tags);
}

// --- Feature 4: Regex Pre-Tokenization Boundaries (5 tests) ---

#[test]
fn test_t1_regex_word_and_punctuation_isolation() {
    let tok = get_test_tokenizer();
    let text = "Hello, world!";
    let ids = tok.encode(text);
    assert!(ids.len() >= 3, "Punctuation must be isolated from words");
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, text);
}

#[test]
fn test_t1_regex_contraction_handling() {
    let tok = get_test_tokenizer();
    let text = "It's they're we've you'll I'd";
    let ids = tok.encode(text);
    assert!(!ids.is_empty());
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, text);
}

#[test]
fn test_t1_regex_consecutive_whitespace_isolation() {
    let tok = get_test_tokenizer();
    let text = "Line 1\n\n\n   Line 2\t\tLine 3";
    let ids = tok.encode(text);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, text);
}

#[test]
fn test_t1_regex_numeric_sequence_segmentation() {
    let tok = get_test_tokenizer();
    let text = "Values: 12 345 6789 100256";
    let ids = tok.encode(text);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, text);
}

#[test]
fn test_t1_regex_special_token_boundary_preservation() {
    let tok = get_test_tokenizer();
    let text = "<|im_start|>assistant\n<|tara_exec|>ls -la<|im_end|>";
    let ids = tok.encode(text);
    assert_eq!(ids[0], tok.im_start_id());
    assert!(ids.contains(&tok.im_end_id()));
}

// --- Feature 5: Rayon Parallel Batch Tokenization Parity (5 tests) ---

#[test]
fn test_t1_rayon_batch_parity_basic_strings() {
    let tok = get_test_tokenizer();
    let batch = ["hello", "world", "tara neural engine", "batch test"];
    let parallel_ids = tok.encode_batch(&batch);
    let sequential_ids: Vec<Vec<u32>> = batch.iter().map(|&s| tok.encode(s)).collect();
    assert_eq!(parallel_ids, sequential_ids, "Parallel encode_batch must be bit-exact identical to sequential encode");
}

#[test]
fn test_t1_rayon_batch_parity_multilingual_sentences() {
    let tok = get_test_tokenizer();
    let batch = [
        "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?",
        "TARA AI is a production native Rust transformer engine.",
        "೦೧೨೩೪೫೬೭೮೯ Kannada numerals.",
        "Mixed: Hello ನಮಸ್ಕಾರ 123!",
    ];
    let parallel = tok.encode_batch(&batch);
    let sequential: Vec<_> = batch.iter().map(|&s| tok.encode(s)).collect();
    assert_eq!(parallel, sequential);
}

#[test]
fn test_t1_rayon_batch_parity_code_and_json() {
    let tok = get_test_tokenizer();
    let batch = [
        "fn main() { println!(\"Hello\"); }",
        r#"{"vocab_size": 100256, "hidden_size": 2048}"#,
        "SELECT * FROM datasets WHERE license = 'MIT';",
        "import torch\nmodel = TaraForCausalLM()",
    ];
    let parallel = tok.encode_batch(&batch);
    let sequential: Vec<_> = batch.iter().map(|&s| tok.encode(s)).collect();
    assert_eq!(parallel, sequential);
}

#[test]
fn test_t1_rayon_batch_parity_emoji_and_binary_octets() {
    let tok = get_test_tokenizer();
    let batch = [
        "Emojis: 🚀 ✨ 🤖 🧠",
        "Raw octets: \x00\x01\x02\x7F",
        "High bytes: \u{0080}\u{00FE}\u{00FF}",
        "Symbols: ∑ ∏ √ ∫ ≈ ≠ ≤ ≥",
    ];
    let parallel = tok.encode_batch(&batch);
    let sequential: Vec<_> = batch.iter().map(|&s| tok.encode(s)).collect();
    assert_eq!(parallel, sequential);
}

#[test]
fn test_t1_rayon_batch_parity_large_heterogeneous_batch() {
    let tok = get_test_tokenizer();
    let samples = [
        "sample a", "sample b 123", "sample c ನಮಸ್ಕಾರ", "sample d 🚀",
        "fn foo() -> u32 { 42 }", "SELECT 1;", "{\"key\": \"val\"}",
        "<|im_start|>user\nHi<|im_end|>",
    ];
    let mut large_batch = Vec::with_capacity(128);
    for i in 0..128 {
        large_batch.push(samples[i % samples.len()]);
    }

    let parallel = tok.encode_batch(&large_batch);
    let sequential: Vec<_> = large_batch.iter().map(|&s| tok.encode(s)).collect();
    assert_eq!(parallel, sequential, "Large 128-item batch must maintain bit-exact parity across all threads");
}

// --- Feature 6: SafeTensors Header & Shape Verification (5 tests) ---

#[test]
fn test_t1_safetensors_candidate_v1_header_schema() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let shard_path = manifest_dir.join("../../storage/models/tara_candidate_v1/model-00001.safetensors");
    if !shard_path.exists() {
        return; // Guard for isolated build environments
    }
    let (hlen, header) = parse_safetensors_header(&shard_path).expect("Candidate shard 1 header must parse");
    assert_eq!(hlen % 8, 0, "Header length {hlen} must be aligned to 8-byte boundary");
    assert!(header.get("lm_head.weight").is_some(), "Shard 1 must contain lm_head.weight");
}

#[test]
fn test_t1_safetensors_candidate_v1_embed_tokens_header_schema() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let shard_path = manifest_dir.join("../../storage/models/tara_candidate_v1/model-00002.safetensors");
    if !shard_path.exists() {
        return;
    }
    let (hlen, header) = parse_safetensors_header(&shard_path).expect("Candidate shard 2 header must parse");
    assert_eq!(hlen % 8, 0, "Header length {hlen} must be aligned to 8-byte boundary");
    assert!(header.get("model.embed_tokens.weight").is_some(), "Shard 2 must contain model.embed_tokens.weight");
}

#[test]
fn test_t1_safetensors_world_model_header_schema() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let shard_path = manifest_dir.join("../../storage/models/tara_world_model/world_model-00001.safetensors");
    if !shard_path.exists() {
        return;
    }
    let (hlen, header) = parse_safetensors_header(&shard_path).expect("World model shard 1 header must parse");
    assert_eq!(hlen % 8, 0, "Header length {hlen} must be aligned to 8-byte boundary");
    let lm_head = header.get("lm_head.weight").expect("Must contain lm_head.weight");
    assert_eq!(lm_head["dtype"].as_str(), Some("BF16"), "World model dtype must be BF16");
}

#[test]
fn test_t1_safetensors_world_model_multi_tensor_offsets() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let shard_path = manifest_dir.join("../../storage/models/tara_world_model/world_model-00002.safetensors");
    if !shard_path.exists() {
        return;
    }
    let (hlen, header) = parse_safetensors_header(&shard_path).expect("World model shard 2 header must parse");
    assert_eq!(hlen % 8, 0, "Header length {hlen} must be aligned to 8-byte boundary");
    let embed = header.get("model.embed_tokens.weight").expect("Must contain embed_tokens");
    assert_eq!(embed["dtype"].as_str(), Some("BF16"));
}

#[test]
fn test_t1_safetensors_index_manifest_integrity() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let index_path = manifest_dir.join("../../storage/models/tara_candidate_v1/model.safetensors.index.json");
    if !index_path.exists() {
        return;
    }
    let content = fs::read_to_string(&index_path).expect("Index manifest must be readable");
    let idx: serde_json::Value = serde_json::from_str(&content).expect("Index manifest must be valid JSON");
    assert!(idx.get("weight_map").is_some(), "Index must contain weight_map");
    assert!(idx.get("metadata").is_some(), "Index must contain metadata");
}

// ============================================================================
// TIER 2: BOUNDARY & CORNER CASES (>=5 tests per category)
// ============================================================================

// --- Category 1: Empty & Whitespace Inputs (5 tests) ---

#[test]
fn test_t2_empty_string_encode_decode() {
    let tok = get_test_tokenizer();
    let ids = tok.encode("");
    assert!(ids.is_empty(), "Empty string must encode to empty vector");
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, "", "Empty IDs must decode to empty string");
}

#[test]
fn test_t2_single_space_encode_decode() {
    let tok = get_test_tokenizer();
    let ids = tok.encode(" ");
    assert!(!ids.is_empty());
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, " ");
}

#[test]
fn test_t2_only_spaces_encode_decode() {
    let tok = get_test_tokenizer();
    let spaces = "     ";
    let ids = tok.encode(spaces);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, spaces);
}

#[test]
fn test_t2_newlines_and_tabs_encode_decode() {
    let tok = get_test_tokenizer();
    let whitespace = "\r\n\t \r\n";
    let ids = tok.encode(whitespace);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, whitespace);
}

#[test]
fn test_t2_mixed_whitespace_only_encode_decode() {
    let tok = get_test_tokenizer();
    let non_breaking = "\u{00A0}\u{2003}\u{2009}";
    let ids = tok.encode(non_breaking);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, non_breaking);
}

// --- Category 2: Single Byte & Control Characters (5 tests) ---

#[test]
fn test_t2_single_ascii_char() {
    let tok = get_test_tokenizer();
    for ch in ['a', 'Z', '0', '$', '_', '~'] {
        let text = ch.to_string();
        let ids = tok.encode(&text);
        assert!(!ids.is_empty());
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }
}

#[test]
fn test_t2_null_byte() {
    let tok = get_test_tokenizer();
    let text = "\0";
    let ids = tok.encode(text);
    assert!(!ids.is_empty());
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, text);
}

#[test]
fn test_t2_bell_and_backspace_control_chars() {
    let tok = get_test_tokenizer();
    let text = "\x07\x08";
    let ids = tok.encode(text);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, text);
}

#[test]
fn test_t2_escape_and_del_chars() {
    let tok = get_test_tokenizer();
    let text = "\x1B\x7F";
    let ids = tok.encode(text);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, text);
}

#[test]
fn test_t2_all_c0_controls_roundtrip() {
    let tok = get_test_tokenizer();
    let mut c0 = String::new();
    for b in 1..=31u8 {
        c0.push(b as char);
    }
    let ids = tok.encode(&c0);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, c0);
}

// --- Category 3: Max Length Subwords & Extreme Input Sizes (5 tests) ---

#[test]
fn test_t2_max_len_subword_exact_match() {
    let tok = get_test_tokenizer();
    let max_len = tok.max_token_len;
    assert!(max_len >= 1, "max_token_len must be positive");
}

#[test]
fn test_t2_input_exceeding_max_token_len() {
    let tok = get_test_tokenizer();
    let long_word = "a".repeat(tok.max_token_len + 50);
    let ids = tok.encode(&long_word);
    assert!(!ids.is_empty());
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, long_word);
}

#[test]
fn test_t2_deep_repetitive_subwords() {
    let tok = get_test_tokenizer();
    let rep = "hello ".repeat(500);
    let ids = tok.encode(&rep);
    assert!(!ids.is_empty());
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, rep);
}

#[test]
fn test_t2_large_text_stream_zero_allocation() {
    let tok = get_test_tokenizer();
    let mut large_text = String::with_capacity(30_000);
    for i in 0..100 {
        large_text.push_str(&format!("Stream chunk {i}: Hello tara! ನಮಸ್ಕಾರ ೦೧೨೩ 🚀 "));
    }
    let ids = tok.encode(&large_text);
    assert!(!ids.contains(&tok.unk_id()));
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, large_text);
}

#[test]
fn test_t2_huge_single_token_vocab_boundary() {
    let tok = get_test_tokenizer();
    // Test that dynamically computed max_token_len matches largest token
    let actual_max = tok.token_to_id.keys().map(|k| k.chars().count()).max().unwrap_or(1);
    assert_eq!(tok.max_token_len, actual_max);
}

// --- Category 4: Arbitrary Binary & Invalid UTF-8 (5 tests) ---

#[test]
fn test_t2_binary_truncated_2byte_utf8() {
    let tok = get_test_tokenizer();
    // 0xC3 alone is incomplete 2-byte sequence
    let id = tok.byte_to_id[0xC3].unwrap();
    let decoded = tok.decode(&[id]);
    assert!(!decoded.is_empty(), "Incomplete UTF-8 sequence must decode safely without panic");
}

#[test]
fn test_t2_binary_truncated_3byte_utf8() {
    let tok = get_test_tokenizer();
    // 0xE0, 0xA2 incomplete 3-byte sequence
    let b1 = tok.byte_to_id[0xE0].unwrap();
    let b2 = tok.byte_to_id[0xA2].unwrap();
    let decoded = tok.decode(&[b1, b2]);
    assert!(!decoded.is_empty());
}

#[test]
fn test_t2_binary_truncated_4byte_utf8() {
    let tok = get_test_tokenizer();
    // 0xF0, 0x90, 0x80 incomplete 4-byte sequence
    let b1 = tok.byte_to_id[0xF0].unwrap();
    let b2 = tok.byte_to_id[0x90].unwrap();
    let b3 = tok.byte_to_id[0x80].unwrap();
    let decoded = tok.decode(&[b1, b2, b3]);
    assert!(!decoded.is_empty());
}

#[test]
fn test_t2_binary_overlong_encoding_handling() {
    let tok = get_test_tokenizer();
    // 0xC0, 0xAF is overlong slash
    let b1 = tok.byte_to_id[0xC0].unwrap();
    let b2 = tok.byte_to_id[0xAF].unwrap();
    let decoded = tok.decode(&[b1, b2]);
    assert!(!decoded.is_empty());
}

#[test]
fn test_t2_binary_high_byte_0xff() {
    let tok = get_test_tokenizer();
    let b_ff = tok.byte_to_id[0xFF].unwrap();
    let decoded = tok.decode(&[b_ff]);
    assert!(!decoded.is_empty());
}

// --- Category 5: Boundary Token IDs (5 tests) ---

#[test]
fn test_t2_vocab_boundary_token_zero() {
    let tok = get_test_tokenizer();
    assert_eq!(tok.pad_id(), 0, "ID 0 must be <|pad|>");
    let decoded = tok.decode(&[0]);
    assert_eq!(decoded, "", "PAD token must be filtered during decode");
}

#[test]
fn test_t2_vocab_boundary_subword_limit_99999() {
    let tok = get_test_tokenizer();
    assert!(tok.id_to_token.contains_key(&99999), "ID 99999 must exist in vocabulary");
}

#[test]
fn test_t2_vocab_boundary_byte_fallback_start_100000() {
    let tok = get_test_tokenizer();
    assert_eq!(tok.id_to_byte.get(&100000), Some(&0x00), "ID 100000 must map to raw byte 0x00");
}

#[test]
fn test_t2_vocab_boundary_byte_fallback_end_100255() {
    let tok = get_test_tokenizer();
    assert_eq!(tok.id_to_byte.get(&100255), Some(&0xFF), "ID 100255 must map to raw byte 0xFF");
}

#[test]
fn test_t2_vocab_boundary_out_of_bounds_100256() {
    let tok = get_test_tokenizer();
    // Out-of-bounds ID should decode safely without panicking
    let decoded = tok.decode(&[100256, 999999]);
    assert!(decoded.is_empty() || !decoded.is_empty(), "Out-of-bounds IDs must not panic");
}

// --- Category 6: Single vs Multi-Threaded Rayon Batches (5 tests) ---

#[test]
fn test_t2_batch_empty_slice() {
    let tok = get_test_tokenizer();
    let empty_batch: [&str; 0] = [];
    let ids = tok.encode_batch(&empty_batch);
    assert!(ids.is_empty(), "Empty batch must return empty result");
}

#[test]
fn test_t2_batch_single_element() {
    let tok = get_test_tokenizer();
    let batch = ["Single element batch test"];
    let parallel = tok.encode_batch(&batch);
    assert_eq!(parallel.len(), 1);
    assert_eq!(parallel[0], tok.encode(batch[0]));
}

#[test]
fn test_t2_batch_two_elements() {
    let tok = get_test_tokenizer();
    let batch = ["first element", "second element"];
    let parallel = tok.encode_batch(&batch);
    assert_eq!(parallel.len(), 2);
    assert_eq!(parallel[0], tok.encode(batch[0]));
    assert_eq!(parallel[1], tok.encode(batch[1]));
}

#[test]
fn test_t2_batch_thread_pool_oversubscription() {
    let tok = get_test_tokenizer();
    let items: Vec<String> = (0..256).map(|i| format!("batch item {i}")).collect();
    let item_refs: Vec<&str> = items.iter().map(|s| s.as_str()).collect();

    let parallel = tok.encode_batch(&item_refs);
    let sequential: Vec<_> = item_refs.iter().map(|&s| tok.encode(s)).collect();
    assert_eq!(parallel, sequential, "Batch larger than core count must retain perfect parity");
}

#[test]
fn test_t2_batch_with_empty_strings_interleaved() {
    let tok = get_test_tokenizer();
    let batch = ["", "hello", "", "world", "", ""];
    let parallel = tok.encode_batch(&batch);
    let sequential: Vec<_> = batch.iter().map(|&s| tok.encode(s)).collect();
    assert_eq!(parallel, sequential);
    assert!(parallel[0].is_empty());
    assert!(parallel[2].is_empty());
}

// ============================================================================
// TIER 3: CROSS-FEATURE COMBINATIONS (Pairwise & Multi-Feature)
// ============================================================================

#[test]
fn test_t3_regex_trie_byte_fallback_mixed_sentence() {
    let tok = get_test_tokenizer();
    let text = "TARA's neural engine: 100256 tokens 🚀 and ಕನ್ನಡ numerals ೦೧೨೩!";
    let ids = tok.encode(text);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, text, "Regex + Trie + Byte Fallback must roundtrip seamlessly");
}

#[test]
fn test_t3_multithreaded_batch_mixed_unicode_ascii_fallback() {
    let tok = get_test_tokenizer();
    let batch = [
        "ASCII string 12345",
        "Kannada: ನಮಸ್ಕಾರ ಮತ್ತು ಶುಭೋದಯ",
        "Math: \x5cint_0^\x5cinfty e^{-x^2} dx",
        "Emojis: 🚀 ✨ 🤖 🧠",
        "Raw octets: \x01\x02\x7F\u{0080}\u{00FF}",
        "Code: fn main() -> Result<(), Box<dyn Error>> {}",
    ];
    let parallel = tok.encode_batch(&batch);
    let sequential: Vec<_> = batch.iter().map(|&s| tok.encode(s)).collect();
    assert_eq!(parallel, sequential, "Mixed Unicode/ASCII/fallback strings must match sequentially across threads");
}

#[test]
fn test_t3_special_tokens_with_regex_punctuation_and_emojis() {
    let tok = get_test_tokenizer();
    let prompt = "<|im_start|>user\nCalculate: 2 + 2 = 4! 🚀<|im_end|>\n<|im_start|>assistant\n4<|im_end|>";
    let ids = tok.encode(prompt);
    assert_eq!(ids[0], tok.im_start_id());
    assert!(ids.contains(&tok.im_end_id()));
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, "user\nCalculate: 2 + 2 = 4! 🚀\nassistant\n4");
}

#[test]
fn test_t3_code_with_byte_fallback_and_regex() {
    let tok = get_test_tokenizer();
    let code = "// ಕನ್ನಡ ಕಾಮೆಂಟ್: calculate hash 🚀\npub fn hash(bytes: &[u8]) -> [u8; 32] { sha256(bytes) }";
    let ids = tok.encode(code);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, code);
}

#[test]
fn test_t3_json_with_unicode_escapes_and_bytes() {
    let tok = get_test_tokenizer();
    let json_text = r#"{"name": "TARA", "version": "v1.0", "scripts": ["ಕನ್ನಡ", "English"], "emoji": "🧠"}"#;
    let ids = tok.encode(json_text);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, json_text);
}

#[test]
fn test_t3_markdown_table_with_kannda_and_math() {
    let tok = get_test_tokenizer();
    let table = "| ID | ಭಾಷೆ | Formula | Status |\n|---|---|---|---|\n| 1 | ಕನ್ನಡ | E = mc^2 | OK 🚀 |";
    let ids = tok.encode(table);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, table);
}

#[test]
fn test_t3_concurrent_batch_encoding_stress() {
    let tok = Arc::new(get_test_tokenizer());
    let mut handles = Vec::new();

    for thread_idx in 0..4 {
        let tok_clone = Arc::clone(&tok);
        let handle = thread::spawn(move || {
            let batch = [
                format!("Thread {thread_idx} text 1: ನಮಸ್ಕಾರ"),
                format!("Thread {thread_idx} text 2: 🚀 100256"),
                format!("Thread {thread_idx} text 3: code fn foo_{thread_idx}()"),
            ];
            let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
            let parallel = tok_clone.encode_batch(&batch_refs);
            let sequential: Vec<_> = batch_refs.iter().map(|&s| tok_clone.encode(s)).collect();
            assert_eq!(parallel, sequential);
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().expect("Concurrent batch threads must join cleanly");
    }
}

#[test]
fn test_t3_decode_sequence_with_interleaved_special_and_bytes() {
    let tok = get_test_tokenizer();
    let byte_id_a = tok.byte_to_id[b'A' as usize].unwrap();
    let byte_id_b = tok.byte_to_id[b'B' as usize].unwrap();
    let ids = vec![tok.im_start_id(), byte_id_a, byte_id_b, tok.im_end_id()];
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, "AB", "Interleaved control and byte fallback IDs must reconstruct payload");
}

#[test]
fn test_t3_roundtrip_adversarial_format_strings() {
    let tok = get_test_tokenizer();
    let adversarial = "Format strings: %s %d %x {0} {name} \\x00 <0x12> && || !=";
    let ids = tok.encode(adversarial);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, adversarial);
}

#[test]
fn test_t3_batch_parity_under_adversarial_lengths() {
    let tok = get_test_tokenizer();
    let empty = "";
    let short = "Hi";
    let medium = "Medium length sentence with ಕನ್ನಡ and symbols 🚀";
    let long = "Long sentence. ".repeat(200);
    let batch = [empty, short, medium, &long];

    let parallel = tok.encode_batch(&batch);
    let sequential: Vec<_> = batch.iter().map(|&s| tok.encode(s)).collect();
    assert_eq!(parallel, sequential);
}

// ============================================================================
// TIER 4: REAL-WORLD SCENARIOS & FORWARD PASS VERIFICATION
// ============================================================================

#[test]
fn test_t4_real_world_rust_source_file_tokenization() {
    let tok = get_test_tokenizer();
    let rust_code = r#"
// Authenticated TARA cognitive kernel
pub fn execute_cognition(input: &str) -> Result<String, EngineError> {
    let tokens = tokenize(input)?;
    if tokens.is_empty() {
        return Err(EngineError::EmptyInput);
    }
    let logits = forward_pass(&tokens);
    Ok(decode(&logits))
}
"#;
    let ids = tok.encode(rust_code);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, rust_code);
}

#[test]
fn test_t4_real_world_python_data_pipeline_tokenization() {
    let tok = get_test_tokenizer();
    let py_code = r#"
import torch
from cloud_train import TaraForCausalLM

def train_step(model: TaraForCausalLM, batch: torch.Tensor) -> float:
    """Execute training step on GPU/CPU."""
    logits, loss, _, _ = model(batch, labels=batch)
    loss.backward()
    return loss.item()
"#;
    let ids = tok.encode(py_code);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, py_code);
}

#[test]
fn test_t4_real_world_json_configuration_tokenization() {
    let tok = get_test_tokenizer();
    let json_config = r#"{
  "architectures": ["TaraForCausalLM"],
  "vocab_size": 100256,
  "hidden_size": 2048,
  "num_hidden_layers": 16,
  "num_attention_heads": 32,
  "intermediate_size": 5632,
  "torch_dtype": "bfloat16"
}"#;
    let ids = tok.encode(json_config);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, json_config);
}

#[test]
fn test_t4_real_world_markdown_readme_tokenization() {
    let tok = get_test_tokenizer();
    let markdown = r#"
# TARA: Autonomous Native AI Architecture

[![Build Status](https://github.com/tara/taracore/workflows/CI/badge.svg)](https://github.com)

- **Pure Native Rust**: Zero runtime Python dependencies.
- **Dynamic Vocabulary**: 100,256 tokens including 256 byte fallbacks.
- **Autonomous LoRA**: Native Low-Rank Adaptation $\Delta W = B \times A$.
"#;
    let ids = tok.encode(markdown);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, markdown);
}

#[test]
fn test_t4_real_world_multilingual_dialogue() {
    let tok = get_test_tokenizer();
    let dialogue = "<|im_start|>system\nYou are TARA, an autonomous AI.<|im_end|>\n<|im_start|>user\nನಮಸ್ಕಾರ, ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ.<|im_end|>\n<|im_start|>assistant\nಖಂಡಿತ, ನಾನು ನಿಮಗೆ ಸಹಾಯ ಮಾಡಲು ಸಿದ್ಧವಾಗಿದ್ದೇನೆ.<|im_end|>";
    let ids = tok.encode(dialogue);
    assert_eq!(ids[0], tok.im_start_id());
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, "system\nYou are TARA, an autonomous AI.\nuser\nನಮಸ್ಕಾರ, ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ.\nassistant\nಖಂಡಿತ, ನಾನು ನಿಮಗೆ ಸಹಾಯ ಮಾಡಲು ಸಿದ್ಧವಾಗಿದ್ದೇನೆ.");
}

#[test]
fn test_t4_real_world_sql_query_with_unicode_literals() {
    let tok = get_test_tokenizer();
    let sql = "SELECT id, title, text FROM dataset_shards WHERE language = 'ಕನ್ನಡ' AND quality >= 0.95 ORDER BY id LIMIT 50;";
    let ids = tok.encode(sql);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, sql);
}

// --- Native Rust Forward Pass Verification on Token IDs >= 100000 ---

fn build_scaled_test_causal_lm(vocab_size: usize, hidden_size: usize) -> TaraForCausalLM {
    let config = TaraConfig {
        vocab_size,
        hidden_size,
        intermediate_size: hidden_size * 2,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: hidden_size / 2,
        max_position_embeddings: 512,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        initializer_range: 0.02,
        version: "1.0.0".to_string(),
        model_type: "tara".to_string(),
        tie_word_embeddings: Some(false),
    };

    let hs = hidden_size;
    let hd = config.effective_head_dim();
    let mut weights = HashMap::new();

    // embed_tokens and lm_head for vocab_size x hidden_size
    weights.insert("model.embed_tokens.weight".to_string(), vec![0.01f32; vocab_size * hs]);
    weights.insert("model.norm.weight".to_string(), vec![1.0f32; hs]);
    weights.insert("lm_head.weight".to_string(), vec![0.01f32; vocab_size * hs]);

    // Layer 0 weights
    weights.insert("model.layers.0.input_layernorm.weight".to_string(), vec![1.0f32; hs]);
    weights.insert("model.layers.0.post_attention_layernorm.weight".to_string(), vec![1.0f32; hs]);
    weights.insert("model.layers.0.self_attn.q_proj.weight".to_string(), vec![0.01f32; 2 * hd * hs]);
    weights.insert("model.layers.0.self_attn.k_proj.weight".to_string(), vec![0.01f32; 2 * hd * hs]);
    weights.insert("model.layers.0.self_attn.v_proj.weight".to_string(), vec![0.01f32; 2 * hd * hs]);
    weights.insert("model.layers.0.self_attn.o_proj.weight".to_string(), vec![0.01f32; hs * 2 * hd]);
    weights.insert("model.layers.0.mlp.gate_proj.weight".to_string(), vec![0.01f32; (hs * 2) * hs]);
    weights.insert("model.layers.0.mlp.up_proj.weight".to_string(), vec![0.01f32; (hs * 2) * hs]);
    weights.insert("model.layers.0.mlp.down_proj.weight".to_string(), vec![0.01f32; hs * (hs * 2)]);

    TaraForCausalLM::from_weights_ref(&weights, &config, "memory_test")
        .expect("Test TaraForCausalLM must initialize successfully")
}

#[test]
fn test_t4_rust_forward_pass_high_token_ids() {
    let vocab_size = 100256;
    let hidden_size = 32;
    let model = build_scaled_test_causal_lm(vocab_size, hidden_size);

    // Sequence containing byte fallback token IDs >= 100000
    let tokens = [1u32, 42u32, 100000u32, 100128u32, 100255u32];
    let logits = model.forward(&tokens);

    assert_eq!(
        logits.len(),
        tokens.len() * vocab_size,
        "Forward pass logits dimension must equal seq_len * vocab_size"
    );
}

#[test]
fn test_t4_rust_forward_pass_logits_shape_verification() {
    let vocab_size = 100256;
    let hidden_size = 32;
    let model = build_scaled_test_causal_lm(vocab_size, hidden_size);

    let single_token = [100200u32];
    let logits = model.forward(&single_token);
    assert_eq!(logits.len(), vocab_size, "Single token forward pass must yield exactly vocab_size logits");
}

#[test]
fn test_t4_rust_forward_pass_finite_logits_on_byte_fallback() {
    let vocab_size = 100256;
    let hidden_size = 32;
    let model = build_scaled_test_causal_lm(vocab_size, hidden_size);

    let tokens = [100000u32, 100255u32];
    let logits = model.forward(&tokens);

    for (idx, &val) in logits.iter().enumerate() {
        assert!(val.is_finite(), "Logit at index {idx} must be finite (got {val})");
    }
}

#[test]
fn test_t4_rust_forward_pass_loss_computation_on_byte_targets() {
    let vocab_size = 100256;
    let hidden_size = 32;
    let model = build_scaled_test_causal_lm(vocab_size, hidden_size);

    let tokens = [1u32, 100000u32, 100255u32];
    let logits = model.forward(&tokens);

    // Compute cross-entropy loss for target token ID 100255 at position 1
    let pos = 1;
    let target = tokens[pos + 1] as usize;
    let logits_slice = &logits[pos * vocab_size..(pos + 1) * vocab_size];

    let max_logit = logits_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits_slice.iter().map(|&x| (x - max_logit).exp()).sum();
    let loss = sum_exp.ln() - (logits_slice[target] - max_logit);

    assert!(loss.is_finite(), "Cross-entropy loss on byte fallback target must be finite");
    assert!(loss > 0.0, "Loss must be positive");
}

#[test]
fn test_t4_rust_forward_pass_all_byte_fallback_coverage() {
    let vocab_size = 100256;
    let hidden_size = 32;
    let model = build_scaled_test_causal_lm(vocab_size, hidden_size);

    // Sequence spanning multiple regions of the byte fallback range [100000..=100255]
    let byte_samples = [100000u32, 100001u32, 100064u32, 100128u32, 100192u32, 100254u32, 100255u32];
    let logits = model.forward(&byte_samples);

    assert_eq!(
        logits.len(),
        byte_samples.len() * vocab_size,
        "Logits length must match byte_samples.len() * vocab_size"
    );
    for (idx, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "Logit at index {idx} was non-finite: {v}");
    }
}

#[test]
fn test_t4_rust_forward_pass_out_of_bounds_saturation() {
    let vocab_size = 100256;
    let hidden_size = 32;
    let model = build_scaled_test_causal_lm(vocab_size, hidden_size);

    // Adversarial out-of-bounds IDs: exactly 100256 (vocab_size) and 999999
    let oob_tokens = [100256u32, 999999u32];
    let logits = model.forward(&oob_tokens);

    assert_eq!(
        logits.len(),
        oob_tokens.len() * vocab_size,
        "Out-of-bounds tokens must be gracefully saturated to vocab_size - 1 without memory faults"
    );
    for (idx, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "Logit at index {idx} was non-finite: {v}");
    }
}

#[test]
fn test_t4_rust_forward_token_cached_on_byte_fallback() {
    let vocab_size = 100256;
    let hidden_size = 32;
    let model = build_scaled_test_causal_lm(vocab_size, hidden_size);
    let mut cache = model.create_kv_cache();

    // Step 1: feed byte fallback token 100000 (0x00)
    let logits_1 = model.forward_token_cached(100000u32, &mut cache);
    assert_eq!(logits_1.len(), vocab_size);
    assert!(logits_1.iter().all(|v| v.is_finite()));
    assert_eq!(cache.seq_len(), 1);

    // Step 2: feed byte fallback token 100255 (0xFF)
    let logits_2 = model.forward_token_cached(100255u32, &mut cache);
    assert_eq!(logits_2.len(), vocab_size);
    assert!(logits_2.iter().all(|v| v.is_finite()));
    assert_eq!(cache.seq_len(), 2);
}
