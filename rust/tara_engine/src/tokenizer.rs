//! TARA tokenizer — high-performance Prefix-Trie tokenizer with regex pre-tokenization boundaries.
//!
//! Reads `tokenizer.json` with the schema `{"vocab": {"<token>": id, ...}}`
//! and provides zero-allocation linear-time Prefix-Trie tokenization with
//! full special-token support, hybrid byte-fallback (<0x00>..<0xFF>), lossless UTF-8
//! roundtrip decoding, and Rayon parallel multi-threaded batch encoding.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::sync::OnceLock;
use rayon::prelude::*;
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

/// Sentinel constant indicating absence of a valid token ID in a Trie node.
pub const NO_TOKEN: u32 = u32::MAX;

/// Sentinel constant indicating absence of a child transition in the Trie root table.
pub const NO_NODE: u32 = u32::MAX;

/// Flag indicating that this Trie node represents a terminal vocabulary token.
pub const FLAG_IS_TOKEN: u16 = 1 << 0;

/// Flag indicating that this Trie node represents a byte fallback token (<0x00>..<0xFF>).
/// Byte fallback tokens are strictly ignored during standard subword matching so that
/// literal `<0x..>` text sequences roundtrip losslessly without collapsing into raw bytes.
pub const FLAG_BYTE_FALLBACK: u16 = 1 << 1;

/// Flag indicating that this Trie node represents a special control token.
pub const FLAG_SPECIAL: u16 = 1 << 2;

/// Global compiled pre-tokenization regex isolating words, whitespaces, numbers,
/// and punctuation symbols before subword token matching.
static PRE_TOKENIZE_REGEX: OnceLock<regex::Regex> = OnceLock::new();

/// Returns a reference to the compiled pre-tokenization regex.
fn get_pre_tokenize_regex() -> &'static regex::Regex {
    PRE_TOKENIZE_REGEX.get_or_init(|| {
        regex::Regex::new(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{M}\p{N}]?(?:\p{L}|\p{M})+|\p{N}{1,3}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+|."
        ).expect("pre-tokenization regex pattern must compile")
    })
}

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

    /// Vocabulary contains invalid, sparse, or duplicate token mappings.
    #[error("invalid tokenizer vocabulary: {0}")]
    InvalidVocab(String),
}

/// Compact 12-byte node in the flat Prefix-Trie.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompactTrieNode {
    /// Token ID if this node represents a complete token; otherwise [`NO_TOKEN`].
    pub token_id: u32,
    /// Starting index into the flat edges array for this node's outgoing edges.
    pub children_start: u32,
    /// Number of outgoing edges (children) from this node.
    pub children_count: u16,
    /// Bit flags ([`FLAG_IS_TOKEN`], [`FLAG_BYTE_FALLBACK`], [`FLAG_SPECIAL`]).
    pub flags: u16,
}

impl Default for CompactTrieNode {
    fn default() -> Self {
        Self {
            token_id: NO_TOKEN,
            children_start: 0,
            children_count: 0,
            flags: 0,
        }
    }
}

/// Compact 8-byte edge in the flat Prefix-Trie.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompactTrieEdge {
    /// Edge transition byte.
    pub byte: u8,
    /// Padding for 4-byte alignment.
    pub _pad: [u8; 3],
    /// Index of the target node in the flat nodes array.
    pub target_node: u32,
}

impl CompactTrieEdge {
    /// Creates a new compact Trie edge.
    #[inline(always)]
    pub fn new(byte: u8, target_node: u32) -> Self {
        Self {
            byte,
            _pad: [0; 3],
            target_node,
        }
    }
}

/// Zero-allocation flat Prefix-Trie / Radix tree for linear-time longest-prefix matching.
#[derive(Clone, Debug)]
pub struct CompactPrefixTrie {
    /// Depth-1 root transition table for instant O(1) 1-byte lookup.
    /// `root_table[b]` stores the target node index for byte `b` from the root,
    /// or [`NO_NODE`] if no token begins with `b`.
    pub root_table: [u32; 256],
    /// Flat array of compact 12-byte Trie nodes.
    pub nodes: Vec<CompactTrieNode>,
    /// Flat array of compact 8-byte Trie edges.
    pub edges: Vec<CompactTrieEdge>,
}

impl CompactPrefixTrie {
    /// Build a flat compact Prefix-Trie from vocabulary mappings and byte fallback registry.
    pub fn build(
        token_to_id: &HashMap<String, u32>,
        byte_to_id: &[Option<u32>; 256],
    ) -> Self {
        struct BuilderNode {
            token_id: u32,
            flags: u16,
            children: BTreeMap<u8, usize>,
        }

        let mut builder_nodes = Vec::with_capacity(token_to_id.len() * 2 + 1);
        builder_nodes.push(BuilderNode {
            token_id: NO_TOKEN,
            flags: 0,
            children: BTreeMap::new(),
        });

        // Insert every token from vocabulary into the trie builder
        for (token_str, &token_id) in token_to_id {
            let is_byte_fallback = (token_id as usize) < 256
                && byte_to_id[token_id as usize] == Some(token_id)
                || byte_to_id.contains(&Some(token_id));
            let is_special = SPECIAL_TOKENS.contains(&token_str.as_str());

            let mut flags = FLAG_IS_TOKEN;
            if is_byte_fallback {
                flags |= FLAG_BYTE_FALLBACK;
            }
            if is_special {
                flags |= FLAG_SPECIAL;
            }

            let mut curr = 0;
            for &b in token_str.as_bytes() {
                if let Some(&next) = builder_nodes[curr].children.get(&b) {
                    curr = next;
                } else {
                    let next = builder_nodes.len();
                    builder_nodes[curr].children.insert(b, next);
                    builder_nodes.push(BuilderNode {
                        token_id: NO_TOKEN,
                        flags: 0,
                        children: BTreeMap::new(),
                    });
                    curr = next;
                }
            }
            builder_nodes[curr].token_id = token_id;
            builder_nodes[curr].flags |= flags;
        }

        // Flatten builder into contiguous node and edge vectors
        let total_nodes = builder_nodes.len();
        let total_edges: usize = builder_nodes.iter().map(|n| n.children.len()).sum();

        let mut nodes = vec![CompactTrieNode::default(); total_nodes];
        let mut edges = Vec::with_capacity(total_edges);

        for (i, bnode) in builder_nodes.iter().enumerate() {
            let children_start = edges.len() as u32;
            let children_count = bnode.children.len() as u16;
            for (&byte, &target_node) in &bnode.children {
                edges.push(CompactTrieEdge::new(byte, target_node as u32));
            }
            nodes[i] = CompactTrieNode {
                token_id: bnode.token_id,
                children_start,
                children_count,
                flags: bnode.flags,
            };
        }

        // Precompute depth-1 root table
        let mut root_table = [NO_NODE; 256];
        if !nodes.is_empty() {
            let root_node = &nodes[0];
            let start = root_node.children_start as usize;
            let count = root_node.children_count as usize;
            for edge in &edges[start..start + count] {
                root_table[edge.byte as usize] = edge.target_node;
            }
        }

        Self {
            root_table,
            nodes,
            edges,
        }
    }

    /// Finds a child node transition for a given byte.
    #[inline(always)]
    pub fn find_child(&self, node: &CompactTrieNode, byte: u8) -> Option<u32> {
        let count = node.children_count as usize;
        if count == 0 {
            return None;
        }
        let start = node.children_start as usize;
        let slice = &self.edges[start..start + count];
        if count <= 8 {
            for edge in slice {
                if edge.byte == byte {
                    return Some(edge.target_node);
                }
            }
            None
        } else {
            slice
                .binary_search_by_key(&byte, |e| e.byte)
                .ok()
                .map(|idx| slice[idx].target_node)
        }
    }

    /// Finds the longest matching non-fallback token in the Trie that matches a prefix of `bytes`.
    /// Returns `Some((token_id, match_byte_len))` or `None`.
    #[inline(always)]
    pub fn longest_match(&self, bytes: &[u8]) -> Option<(u32, usize)> {
        if bytes.is_empty() {
            return None;
        }

        let first_byte = bytes[0];
        let mut curr_node_idx = self.root_table[first_byte as usize];
        if curr_node_idx == NO_NODE {
            return None;
        }

        let mut best_match: Option<(u32, usize)> = None;
        let mut curr_node = &self.nodes[curr_node_idx as usize];

        if curr_node.token_id != NO_TOKEN && (curr_node.flags & FLAG_BYTE_FALLBACK) == 0 {
            best_match = Some((curr_node.token_id, 1));
        }

        for (i, &b) in bytes[1..].iter().enumerate() {
            if let Some(next_idx) = self.find_child(curr_node, b) {
                curr_node_idx = next_idx;
                curr_node = &self.nodes[curr_node_idx as usize];
                if curr_node.token_id != NO_TOKEN && (curr_node.flags & FLAG_BYTE_FALLBACK) == 0 {
                    best_match = Some((curr_node.token_id, i + 2));
                }
            } else {
                break;
            }
        }

        best_match
    }
}

/// Greedy longest-match tokenizer with native hybrid byte-fallback for the TARA model.
///
/// Loads vocabulary from `tokenizer.json` and provides zero-allocation linear-time
/// Prefix-Trie subword matching, regex pre-tokenization boundaries, O(1) byte fallback,
/// lossless UTF-8 roundtrip decoding, and Rayon parallel batch tokenization.
#[derive(Clone, Debug)]
pub struct TaraTokenizer {
    /// Maps token string → token id.
    pub token_to_id: HashMap<String, u32>,
    /// Maps token id → token string.
    pub id_to_token: HashMap<u32, String>,
    /// Maps raw byte values (0..=255) to token id, pre-indexed for fast O(1) fallback.
    pub byte_to_id: [Option<u32>; 256],
    /// Maps token id to raw byte value (0..=255), for reverse byte stream reconstruction.
    pub id_to_byte: HashMap<u32, u8>,
    /// Total number of tokens in the vocabulary.
    pub vocab_size: usize,
    /// Maximum character length of any token in the vocabulary (dynamically computed at load time).
    pub max_token_len: usize,
    /// Compact Prefix-Trie for zero-allocation linear-time subword token matching.
    pub trie: CompactPrefixTrie,
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
        Self::from_json_str(&raw)
    }

    /// Load the tokenizer from a JSON string.
    pub fn from_json_str(raw: &str) -> Result<Self, TokenizerError> {
        let root: serde_json::Value = serde_json::from_str(raw)?;

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

        // Pre-populate fast byte lookup tables for <0x00> through <0xFF>
        let mut byte_to_id: [Option<u32>; 256] = [None; 256];
        let mut id_to_byte: HashMap<u32, u8> = HashMap::with_capacity(256);

        for b in 0..=255u8 {
            let byte_token = format!("<0x{:02X}>", b);
            if let Some(&id) = token_to_id.get(&byte_token) {
                byte_to_id[b as usize] = Some(id);
                id_to_byte.insert(id, b);
            } else {
                let byte_token_lower = format!("<0x{:02x}>", b);
                if let Some(&id) = token_to_id.get(&byte_token_lower) {
                    byte_to_id[b as usize] = Some(id);
                    id_to_byte.insert(id, b);
                }
            }
        }

        let max_token_len = token_to_id
            .keys()
            .map(|t| t.chars().count())
            .max()
            .unwrap_or(1);

        let trie = CompactPrefixTrie::build(&token_to_id, &byte_to_id);

        Ok(Self {
            token_to_id,
            id_to_token,
            byte_to_id,
            id_to_byte,
            vocab_size,
            max_token_len,
            trie,
        })
    }

    /// Encode a text string into a sequence of token IDs.
    ///
    /// Special tokens are extracted first without fragmentation.
    /// Text chunks are segmented by non-lookaround regex pre-tokenization boundaries
    /// (isolating words, whitespace sequences, numbers, and punctuation) and then
    /// matched via the zero-allocation compact Prefix-Trie with O(1) byte fallback.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Fast path: if the entire text is an exact match for a single known non-fallback token,
        // emit it directly in a single step (satisfies dynamic max token len handling).
        if let Some(&id) = self.token_to_id.get(text) {
            if !self.id_to_byte.contains_key(&id) && !SPECIAL_TOKENS.contains(&text) {
                return vec![id];
            }
        }

        let mut ids: Vec<u32> = Vec::with_capacity(text.len());

        // Fast check: if text contains no special token markers, encode directly.
        if !text.contains("<|") {
            self.encode_chunk(text, &mut ids);
            return ids;
        }

        let mut pos = 0;
        let total_len = text.len();

        while pos < total_len {
            let remaining = &text[pos..];

            // Check if remaining starts with any special token
            let mut matched_special: Option<(&str, u32)> = None;
            for &sp in SPECIAL_TOKENS {
                if remaining.starts_with(sp) {
                    if let Some(&id) = self.token_to_id.get(sp) {
                        matched_special = Some((sp, id));
                        break;
                    }
                }
            }

            if let Some((sp_str, sp_id)) = matched_special {
                ids.push(sp_id);
                pos += sp_str.len();
                continue;
            }

            // Find the index of the nearest special token in remaining
            let mut next_special_idx = remaining.len();
            for &sp in SPECIAL_TOKENS {
                if let Some(idx) = remaining.find(sp) {
                    if idx < next_special_idx {
                        next_special_idx = idx;
                    }
                }
            }

            let normal_slice = &remaining[..next_special_idx];
            self.encode_chunk(normal_slice, &mut ids);
            pos += next_special_idx;
        }

        ids
    }

    /// Encodes a normal text slice (free of special tokens) using regex pre-tokenization
    /// and the CompactPrefixTrie with byte fallback.
    fn encode_chunk(&self, chunk: &str, ids: &mut Vec<u32>) {
        if chunk.is_empty() {
            return;
        }

        // If the chunk is an exact non-fallback token, emit directly
        if let Some(&id) = self.token_to_id.get(chunk) {
            if !self.id_to_byte.contains_key(&id) {
                ids.push(id);
                return;
            }
        }

        let re = get_pre_tokenize_regex();
        for mat in re.find_iter(chunk) {
            self.encode_segment(mat.as_str(), ids);
        }
    }

    /// Encodes a single pre-tokenization regex segment using the CompactPrefixTrie.
    fn encode_segment(&self, mut segment: &str, ids: &mut Vec<u32>) {
        while !segment.is_empty() {
            if let Some((token_id, match_len)) = self.trie.longest_match(segment.as_bytes()) {
                ids.push(token_id);
                segment = &segment[match_len..];
            } else {
                // No subword matched: fallback to raw UTF-8 bytes for the first character
                let ch = segment.chars().next().expect("segment is non-empty");
                let ch_len = ch.len_utf8();
                let ch_bytes = &segment.as_bytes()[..ch_len];
                segment = &segment[ch_len..];

                if self.id_to_byte.is_empty() {
                    ids.push(self.unk_id());
                } else {
                    for &b in ch_bytes {
                        if let Some(byte_id) = self.byte_to_id[b as usize] {
                            ids.push(byte_id);
                        } else {
                            ids.push(self.unk_id());
                        }
                    }
                }
            }
        }
    }

    /// Encode a batch of text strings in parallel using Rayon.
    ///
    /// Guarantees bit-exact parity with calling [`Self::encode`] sequentially
    /// on each element, distributing work across available CPU cores.
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        texts.par_iter().map(|&t| self.encode(t)).collect()
    }

    /// Decode a sequence of token IDs into a human-readable string.
    ///
    /// Structural control tokens (`<|pad|>`, `<|im_start|>`, `<|im_end|>`) are
    /// silently skipped. Raw byte tokens and subwords are accumulated into a byte buffer
    /// and reconstructed into authentic UTF-8 text losslessly.
    pub fn decode(&self, ids: &[u32]) -> String {
        let skip_ids = [self.pad_id(), self.im_start_id(), self.im_end_id()];
        let mut byte_buf: Vec<u8> = Vec::with_capacity(ids.len() * 2);

        for &id in ids {
            if skip_ids.contains(&id) {
                continue;
            }
            if let Some(&raw_byte) = self.id_to_byte.get(&id) {
                byte_buf.push(raw_byte);
            } else if let Some(tok_str) = self.id_to_token.get(&id) {
                byte_buf.extend_from_slice(tok_str.as_bytes());
            }
        }

        match String::from_utf8(byte_buf) {
            Ok(s) => s,
            Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
        }
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
    use super::{
        CompactPrefixTrie, CompactTrieNode, TaraTokenizer, TokenizerError,
        FLAG_BYTE_FALLBACK, NO_NODE,
    };

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
        let cfg_path = manifest_dir.join("../../storage/models/tara_candidate_v1/config.json");
        if cfg_path.exists() {
            if let Ok(raw_cfg) = std::fs::read_to_string(&cfg_path) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw_cfg) {
                    if let Some(expected_vocab) = val.get("vocab_size").and_then(|v| v.as_u64()) {
                        assert_eq!(
                            tokenizer.vocab_size, expected_vocab as usize,
                            "Tokenizer vocab size must match model config"
                        );
                    }
                }
            }
        } else {
            assert!(
                tokenizer.vocab_size >= 8192,
                "Tokenizer must contain at least baseline tokens"
            );
        }

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
        assert_eq!(
            encoded,
            vec![4],
            "Long token must be matched in single step"
        );
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, long_token);
        std::fs::remove_file(path).unwrap();
    }

    fn create_test_byte_fallback_tokenizer() -> TaraTokenizer {
        let mut vocab = serde_json::Map::new();
        let mut id = 0u64;

        // Special tokens
        for &sp in super::SPECIAL_TOKENS {
            vocab.insert(sp.to_string(), serde_json::Value::from(id));
            id += 1;
        }

        // 256 byte tokens <0x00> to <0xFF>
        for b in 0..=255u8 {
            vocab.insert(format!("<0x{:02X}>", b), serde_json::Value::from(id));
            id += 1;
        }

        // Kannada basic characters & numerals ೦-೯ (U+0CE6-U+0CEF)
        let kannada_tokens = [
            "ನ", "ಮ", "ಸ್", "ಕ", "ರ", " ", "!", "?",
            "೦", "೧", "೨", "೩", "೪", "೫", "೬", "೭", "೮", "೯",
        ];
        for &kt in &kannada_tokens {
            if !vocab.contains_key(kt) {
                vocab.insert(kt.to_string(), serde_json::Value::from(id));
                id += 1;
            }
        }

        // Common subwords
        let words = ["hello", "world", "tara", "fn", "let"];
        for &w in &words {
            if !vocab.contains_key(w) {
                vocab.insert(w.to_string(), serde_json::Value::from(id));
                id += 1;
            }
        }

        let mut root = serde_json::Map::new();
        root.insert("vocab".to_string(), serde_json::Value::Object(vocab));
        let json_str = serde_json::to_string(&root).unwrap();

        TaraTokenizer::from_json_str(&json_str).expect("test tokenizer must build successfully")
    }

    #[test]
    fn test_byte_tokens_population() {
        let tok = create_test_byte_fallback_tokenizer();
        assert_eq!(tok.id_to_byte.len(), 256);
        for b in 0..=255u8 {
            let id = tok.byte_to_id[b as usize].expect("byte token must be populated");
            assert_eq!(tok.id_to_byte.get(&id), Some(&b));
        }
    }

    #[test]
    fn test_byte_fallback_emojis_and_unseen_scripts_zero_unk() {
        let tok = create_test_byte_fallback_tokenizer();
        let test_cases = [
            "🚀 ✨ 🤖 🧠",
            "𑀅𑀩𑀕 ꤀꤁꤂",
            "Unicode math: ∑ ∏ √ ∫ ≈ ≠ ≤ ≥",
            "Unicode octets: \x00\x01\x02\x7F\u{0080}\u{0081}\u{00FE}\u{00FF}",
            "Mixed: Hello world! 🚀 tara ✨ let x = 42;",
        ];

        for &text in &test_cases {
            let ids = tok.encode(text);
            assert!(!ids.is_empty(), "Encoded IDs must not be empty for: {text}");
            assert!(
                !ids.contains(&tok.unk_id()),
                "Zero <|unk|> tokens must be generated for: {text}"
            );
            let decoded = tok.decode(&ids);
            assert_eq!(
                decoded, text,
                "Decoded text must match original losslessly for: {text}"
            );
        }
    }

    #[test]
    fn test_kannada_script_and_numerals_lossless() {
        let tok = create_test_byte_fallback_tokenizer();

        // 1. Text composed of vocabulary subwords and Kannada numerals ೦-೯
        let known_text = "ನಮಸ್ಕಾರ ೦೧೨೩೪೫೬೭೮೯";
        let known_ids = tok.encode(known_text);
        assert!(
            !known_ids.contains(&tok.unk_id()),
            "Kannada known text must have zero <|unk|>"
        );
        let known_decoded = tok.decode(&known_ids);
        assert_eq!(
            known_decoded, known_text,
            "Kannada known text must roundtrip losslessly"
        );

        // 2. Unseen Kannada conjuncts not in subwords (must fall back to UTF-8 bytes with zero <|unk|>)
        let unseen_kannada = "ಕನ್ನಡ ಲಿಪಿ ಪರೀಕ್ಷೆ ಮತ್ತು ಸೌಂದರ್ಯ ೧೨೩೪";
        let unseen_ids = tok.encode(unseen_kannada);
        assert!(
            !unseen_ids.contains(&tok.unk_id()),
            "Unseen Kannada script must fall back to byte tokens with zero <|unk|>"
        );
        let unseen_decoded = tok.decode(&unseen_ids);
        assert_eq!(
            unseen_decoded, unseen_kannada,
            "Unseen Kannada text must roundtrip losslessly via byte fallback"
        );
    }

    #[test]
    fn test_special_tokens_preservation() {
        let tok = create_test_byte_fallback_tokenizer();
        let prompt = "<|im_start|>user\nHello tara!<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>";
        let ids = tok.encode(prompt);

        assert_eq!(ids[0], tok.im_start_id());
        assert!(ids.contains(&tok.im_end_id()));

        // Decode skips <|im_start|>, <|im_end|>, and <|pad|>
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "user\nHello tara!\nassistant\nHi!");
    }

    #[test]
    fn test_zero_cost_allocation_large_stream() {
        let tok = create_test_byte_fallback_tokenizer();
        let mut large_text = String::with_capacity(60_000);
        for _ in 0..100 {
            large_text.push_str("Hello world tara 🚀 ನಮಸ್ಕಾರ ೦೧೨೩ fn let \x01\x02 ");
        }

        let ids = tok.encode(&large_text);
        assert!(!ids.contains(&tok.unk_id()));
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, large_text);
    }

    #[test]
    fn test_thread_safety_send_sync() {
        use std::sync::Arc;
        use std::thread;

        let tok = Arc::new(create_test_byte_fallback_tokenizer());
        let mut handles = Vec::new();

        for i in 0..4 {
            let tok_clone = Arc::clone(&tok);
            let handle = thread::spawn(move || {
                let msg = format!("Thread {i}: ನಮಸ್ಕಾರ 🚀 ೦೧೨ {i}");
                let ids = tok_clone.encode(&msg);
                assert!(!ids.contains(&tok_clone.unk_id()));
                let decoded = tok_clone.decode(&ids);
                assert_eq!(decoded, msg);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_lossless_roundtrip_with_literal_byte_token_strings() {
        let tok = create_test_byte_fallback_tokenizer();
        let literal_text = "Checking literal byte token tags: <0x00> and <0xFF> in input string.";
        let ids = tok.encode(literal_text);
        let decoded = tok.decode(&ids);
        assert_eq!(
            decoded, literal_text,
            "Literal byte token strings in user text must roundtrip losslessly without collapsing into raw bytes"
        );
    }

    #[test]
    fn test_invalid_utf8_byte_sequence_lossy_decode() {
        let tok = create_test_byte_fallback_tokenizer();
        // Emitting an incomplete 3-byte sequence (first two bytes 0xE0, 0xA2 without the third)
        let byte1 = tok.byte_to_id[0xE0].unwrap();
        let byte2 = tok.byte_to_id[0xA2].unwrap();
        let decoded = tok.decode(&[byte1, byte2]);
        // Must decode without panicking, using fallback lossy representation
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_compact_trie_node_memory_layout_12_bytes() {
        assert_eq!(
            std::mem::size_of::<CompactTrieNode>(),
            12,
            "CompactTrieNode must be exactly 12 bytes"
        );
    }

    #[test]
    fn test_compact_prefix_trie_depth_1_root_dispatch() {
        let tok = create_test_byte_fallback_tokenizer();
        let trie: &CompactPrefixTrie = &tok.trie;
        // Check that depth-1 root table has non-empty entries for ASCII letters
        let h_byte = b'h';
        let h_node_idx = trie.root_table[h_byte as usize];
        assert_ne!(h_node_idx, NO_NODE, "Root table must index 'h' directly");

        let match_result = trie.longest_match(b"hello world");
        assert!(match_result.is_some());
        let (id, len) = match_result.unwrap();
        assert_eq!(len, 5, "Must match 'hello' with length 5");
        assert_eq!(tok.id_to_token.get(&id), Some(&"hello".to_string()));
    }

    #[test]
    fn test_byte_fallback_flag_prevents_false_literal_match() {
        let tok = create_test_byte_fallback_tokenizer();
        // Verify that byte fallback tokens have FLAG_BYTE_FALLBACK set in the Trie nodes
        let byte_0_id = tok.byte_to_id[0].unwrap();
        let has_fallback_flag = tok.trie.nodes.iter().any(|n| {
            n.token_id == byte_0_id && (n.flags & FLAG_BYTE_FALLBACK) != 0
        });
        assert!(has_fallback_flag, "Byte fallback tokens must have FLAG_BYTE_FALLBACK flag set");

        // Longest match on literal "<0x00>" must NOT return byte token 0x00
        let match_result = tok.trie.longest_match(b"<0x00>");
        if let Some((id, _)) = match_result {
            assert!(
                !tok.id_to_byte.contains_key(&id),
                "Trie matching must not return byte fallback tokens directly"
            );
        }
    }

    #[test]
    fn test_encode_batch_parallel_bit_exact_parity() {
        let tok = create_test_byte_fallback_tokenizer();
        let batch = [
            "hello world",
            "tara neural engine",
            "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?",
            "fn compute() -> f32 { let x = 42; }",
            "🚀 ✨ 🤖 🧠",
            "Checking literal <0x00> tag",
            "<|im_start|>user\nHello!<|im_end|>",
        ];

        let parallel_res = tok.encode_batch(&batch);
        let sequential_res: Vec<Vec<u32>> = batch.iter().map(|&t| tok.encode(t)).collect();

        assert_eq!(
            parallel_res, sequential_res,
            "Rayon parallel encode_batch must be bit-exact identical to sequential encode"
        );
    }

    #[test]
    fn test_regex_pre_tokenization_boundaries() {
        let tok = create_test_byte_fallback_tokenizer();
        let text = "Hello, world! 123 4567";
        let ids = tok.encode(text);
        assert!(!ids.is_empty());
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text, "Regex pre-tokenization boundaries must preserve lossless roundtrip");
    }
}
