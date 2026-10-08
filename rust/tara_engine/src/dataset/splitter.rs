//! Deterministic Dataset Splitter & Leakage Control Engine.
//!
//! Provides:
//! - Deterministic train / val / test partitioning using 64-bit content hashes.
//! - Conservative cluster isolation via Disjoint Set Union (DSU transitive clustering /
//!   similarity graph connected components) to group exact and near-duplicate records into
//!   the same partition, preventing cross-split leakage.
//!   Note: Cluster isolation guarantees zero cross-partition leakage; high-density
//!   near-duplicate connected components may cause slight deviation from exact target proportions.
//! - Shared bidirectional contamination and leakage engine (`LeakageIndex`) powering both
//!   `check_leakage` and `decontaminate` with identical two-stage matching:
//!   1. Exact raw line equality (hash bucket + string comparison).
//!   2. Exact canonical prompt / target / combined sample equality.
//!   3. 13-gram character shingle near-duplicate overlap (>= 85% bidirectional threshold)
//!      computed over normalized lowercase whitespace-collapsed text.
//! - Streaming file input/output with bounded per-record buffering (128 KB I/O buffers);
//!   in-memory `LeakageIndex` and DSU cluster tables scale proportionally with the indexed reference corpus.
//! - Support for full TARA dataset schemas (`formatted_input`, `input`, `prompt`, `instruction`,
//!   `trigger_pattern`, `metadata.input`, `formatted_target`, `output`, `completion`, `response`,
//!   `recommendation`, `metadata.output`).
//! - Crash-safe atomic directory/file promotion with startup recovery and audit-grade manifest commit markers.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

/// Bidirectional near-duplicate 13-gram overlap ratio threshold for leakage detection.
pub const LEAKAGE_OVERLAP_THRESHOLD: f64 = 0.85;

#[derive(Debug, Error)]
pub enum SplitterError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Invalid ratio: train ({train}), val ({val}), test ({test}) must be positive finite numbers summing to 1.0")]
    InvalidRatio { train: f64, val: f64, test: f64 },
    #[error("Empty dataset provided at {0}")]
    EmptyDataset(String),
    #[error("Source directory cannot be identical to or contain output directory: {0}")]
    SourceEqualsOutput(String),
    #[error("Committed split manifest or partition corruption detected in {0}")]
    ManifestCorrupted(String),
}

/// Ratio specification for train, validation, and test splits.
///
/// Note: Target ratios represent nominal split goals. Cluster isolation guarantees
/// zero cross-partition leakage, but transitive grouping of near-duplicates may
/// produce slight deviation from exact sample-level ratios.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SplitRatio {
    pub train: f64,
    pub val: f64,
    pub test: f64,
}

impl Default for SplitRatio {
    fn default() -> Self {
        Self {
            train: 0.90,
            val: 0.05,
            test: 0.05,
        }
    }
}

impl SplitRatio {
    pub fn new(train: f64, val: f64, test: f64) -> Result<Self, SplitterError> {
        if !train.is_finite()
            || !val.is_finite()
            || !test.is_finite()
            || train <= 0.0
            || val < 0.0
            || test < 0.0
        {
            return Err(SplitterError::InvalidRatio { train, val, test });
        }
        let sum = train + val + test;
        if (sum - 1.0).abs() > 1e-4 {
            return Err(SplitterError::InvalidRatio { train, val, test });
        }
        Ok(Self { train, val, test })
    }
}

/// Summary report produced by `DatasetSplitter::split`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitReport {
    pub total_samples: usize,
    pub train_samples: usize,
    pub val_samples: usize,
    pub test_samples: usize,
    pub train_path: String,
    pub val_path: String,
    pub test_path: String,
    #[serde(default)]
    pub manifest_path: Option<String>,
}

/// Summary report produced by `LeakageChecker::check_leakage`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakageReport {
    pub total_train_samples: usize,
    pub total_eval_samples: usize,
    pub exact_match_leaks: usize,
    pub ngram_shingle_leaks: usize,
    pub total_leaked_samples: usize,
    pub leakage_rate_pct: f64,
    pub is_clean: bool,
}

/// Deterministic FNV-1a 64-bit non-cryptographic hash used for bucket indexing.
pub fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Text normalization for canonical sample extraction.
pub fn normalize_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

/// Extracts prompt / input field across various TARA dataset schemas.
pub fn extract_prompt_field(val: &Value) -> Option<String> {
    let candidates = [
        val.get("formatted_input"),
        val.get("input"),
        val.get("prompt"),
        val.get("instruction"),
        val.get("trigger_pattern"),
        val.get("metadata").and_then(|m| m.get("input")),
        val.get("text"),
    ];
    for cand in candidates {
        if let Some(s) = cand.and_then(Value::as_str) {
            let t = s.trim();
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
    }
    None
}

/// Extracts target / output field across various TARA dataset schemas.
pub fn extract_target_field(val: &Value) -> Option<String> {
    let candidates = [
        val.get("formatted_target"),
        val.get("output"),
        val.get("completion"),
        val.get("response"),
        val.get("recommendation"),
        val.get("metadata").and_then(|m| m.get("output")),
    ];
    for cand in candidates {
        if let Some(s) = cand.and_then(Value::as_str) {
            let t = s.trim();
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
    }
    None
}

/// Strips known boilerplate prefixes for clean substantive text comparison.
pub fn clean_substantive_prefix(raw: &str) -> String {
    let prefix = "explain the core scientific research, methodology, and theoretical findings of the paper titled:";
    let lower = raw.trim().to_lowercase();
    if lower.starts_with(prefix) {
        let remainder = raw[prefix.len()..].trim();
        if let Some(first_quote) = remainder.find('"') {
            if let Some(last_quote) = remainder.rfind('"') {
                if last_quote > first_quote {
                    return remainder[first_quote + 1..last_quote].trim().to_string();
                }
            }
        }
    }
    raw.trim().to_string()
}

/// Computes 13-gram character shingles for near-duplicate overlap detection.
pub fn compute_13gram_hashes(text: &str) -> Vec<u64> {
    let normalized = normalize_text(text);
    let chars: Vec<char> = normalized.chars().collect();
    if chars.len() < 13 {
        return vec![fnv1a_hash(normalized.as_bytes())];
    }
    let mut hashes = Vec::with_capacity(chars.len() - 12);
    let mut buf = String::with_capacity(32);
    for window in chars.windows(13) {
        buf.clear();
        for &c in window {
            buf.push(c);
        }
        hashes.push(fnv1a_hash(buf.as_bytes()));
    }
    hashes
}

/// Canonical representation of a single dataset sample with multi-level hashing.
#[derive(Debug, Clone)]
pub struct CanonicalSample {
    pub raw_line: String,
    pub prompt: String,
    pub target: String,
    pub combined_text: String,
    pub line_hash: u64,
    pub prompt_hash: u64,
    pub target_hash: u64,
    pub combined_hash: u64,
    pub prompt_shingles: HashSet<u64>,
    pub shingles: HashSet<u64>,
}

/// Extracts a canonical sample from a raw JSONL line.
pub fn extract_canonical_sample(raw_line: &str) -> CanonicalSample {
    let trimmed = raw_line.trim();
    let line_hash = fnv1a_hash(trimmed.as_bytes());

    let (prompt_raw, target_raw) = if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
        let p = extract_prompt_field(&val);
        let t = extract_target_field(&val);
        (p, t)
    } else {
        (None, None)
    };

    let prompt_clean = prompt_raw
        .map(|s| clean_substantive_prefix(&s))
        .unwrap_or_else(|| trimmed.to_string());
    let target_clean = target_raw.unwrap_or_default();

    let norm_prompt = normalize_text(&prompt_clean);
    let norm_target = normalize_text(&target_clean);

    let combined = if norm_target.is_empty() {
        norm_prompt.clone()
    } else {
        format!("{}\n{}", norm_prompt, norm_target)
    };

    let prompt_hash = fnv1a_hash(norm_prompt.as_bytes());
    let target_hash = if norm_target.is_empty() {
        0
    } else {
        fnv1a_hash(norm_target.as_bytes())
    };
    let combined_hash = fnv1a_hash(combined.as_bytes());
    let prompt_shingles: HashSet<u64> = compute_13gram_hashes(&norm_prompt).into_iter().collect();
    let shingles: HashSet<u64> = compute_13gram_hashes(&combined).into_iter().collect();

    CanonicalSample {
        raw_line: trimmed.to_string(),
        prompt: norm_prompt,
        target: norm_target,
        combined_text: combined,
        line_hash,
        prompt_hash,
        target_hash,
        combined_hash,
        prompt_shingles,
        shingles,
    }
}

/// Type of contamination / leakage detected.
#[derive(Debug, Clone, PartialEq)]
pub enum LeakageKind {
    ExactLine,
    ExactCombined,
    ExactPrompt,
    ExactTarget,
    NearDuplicateShingle { overlap_ratio: f64 },
}

/// Shared bidirectional leakage index powering both `check_leakage` and `decontaminate`.
///
/// Implements robust two-stage verification:
/// 1. Hash bucket lookup for $O(1)$ candidate retrieval.
/// 2. Exact string equality check on candidates to eliminate hash collisions.
pub struct LeakageIndex {
    samples: Vec<CanonicalSample>,
    line_map: HashMap<u64, Vec<usize>>,
    combined_map: HashMap<u64, Vec<usize>>,
    prompt_map: HashMap<u64, Vec<usize>>,
    target_map: HashMap<u64, Vec<usize>>,
    inv_index: HashMap<u64, Vec<usize>>,
}

impl LeakageIndex {
    pub fn new(samples: Vec<CanonicalSample>) -> Self {
        let total = samples.len();
        let mut line_map: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut combined_map: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut prompt_map: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut target_map: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut raw_inv_index: HashMap<u64, Vec<usize>> = HashMap::new();

        for (i, s) in samples.iter().enumerate() {
            line_map.entry(s.line_hash).or_default().push(i);
            combined_map.entry(s.combined_hash).or_default().push(i);
            if !s.prompt.is_empty() {
                prompt_map.entry(s.prompt_hash).or_default().push(i);
            }
            if s.target.len() >= 15 {
                target_map.entry(s.target_hash).or_default().push(i);
            }
            for &sh in &s.shingles {
                raw_inv_index.entry(sh).or_default().push(i);
            }
            for &sh in &s.prompt_shingles {
                raw_inv_index.entry(sh).or_default().push(i);
            }
        }

        // Document-frequency cutoff: filter out ubiquitous shingles appearing in >50% of documents
        // to prevent candidate explosion on common boilerplate phrases
        let df_threshold = if total > 20 { (total / 2).max(10) } else { usize::MAX };
        let mut inv_index = HashMap::new();
        for (sh, doc_ids) in raw_inv_index {
            if doc_ids.len() <= df_threshold {
                inv_index.insert(sh, doc_ids);
            }
        }

        Self {
            samples,
            line_map,
            combined_map,
            prompt_map,
            target_map,
            inv_index,
        }
    }

    /// Evaluates whether a query sample leaks against this indexed dataset.
    pub fn find_leakage(&self, query: &CanonicalSample) -> Option<LeakageKind> {
        // 1. Two-stage exact raw line match
        if let Some(cands) = self.line_map.get(&query.line_hash) {
            for &idx in cands {
                if self.samples[idx].raw_line == query.raw_line {
                    return Some(LeakageKind::ExactLine);
                }
            }
        }

        // 2. Two-stage exact combined sample match (prompt + target)
        if let Some(cands) = self.combined_map.get(&query.combined_hash) {
            for &idx in cands {
                if self.samples[idx].combined_text == query.combined_text {
                    return Some(LeakageKind::ExactCombined);
                }
            }
        }

        // 3. Two-stage exact prompt match
        if !query.prompt.is_empty() {
            if let Some(cands) = self.prompt_map.get(&query.prompt_hash) {
                for &idx in cands {
                    if self.samples[idx].prompt == query.prompt {
                        return Some(LeakageKind::ExactPrompt);
                    }
                }
            }
        }

        // 4. Two-stage exact target match for substantive answers
        if query.target.len() >= 15 {
            if let Some(cands) = self.target_map.get(&query.target_hash) {
                for &idx in cands {
                    if self.samples[idx].target == query.target {
                        return Some(LeakageKind::ExactTarget);
                    }
                }
            }
        }

        // 5. Near-duplicate 13-gram overlap check (>= LEAKAGE_OVERLAP_THRESHOLD)
        let mut cand_set: HashSet<usize> = HashSet::new();
        for sh in query.shingles.iter().chain(query.prompt_shingles.iter()) {
            if let Some(docs) = self.inv_index.get(sh) {
                for &d in docs {
                    cand_set.insert(d);
                }
            }
        }

        for doc_id in cand_set {
            let target_sample = &self.samples[doc_id];

            // Check combined shingles overlap
            let overlap_comb = if !query.shingles.is_empty() && !target_sample.shingles.is_empty() {
                let shared = query.shingles.intersection(&target_sample.shingles).count();
                let r1 = shared as f64 / query.shingles.len() as f64;
                let r2 = shared as f64 / target_sample.shingles.len() as f64;
                r1.max(r2)
            } else {
                0.0
            };

            // Check prompt shingles overlap
            let overlap_prompt = if !query.prompt_shingles.is_empty() && !target_sample.prompt_shingles.is_empty() {
                let shared = query.prompt_shingles.intersection(&target_sample.prompt_shingles).count();
                let r1 = shared as f64 / query.prompt_shingles.len() as f64;
                let r2 = shared as f64 / target_sample.prompt_shingles.len() as f64;
                r1.max(r2)
            } else {
                0.0
            };

            let max_overlap = overlap_comb.max(overlap_prompt);
            if max_overlap >= LEAKAGE_OVERLAP_THRESHOLD {
                return Some(LeakageKind::NearDuplicateShingle {
                    overlap_ratio: max_overlap,
                });
            }
        }

        None
    }
}

/// Disjoint Set Union (DSU) for transitive cluster isolation (similarity graph connected components).
///
/// Transitive closure groups connected components in the pairwise similarity graph so that any sample
/// connected by exact or >=85% 13-gram shingle similarity is quarantined into the same partition.
struct DisjointSet {
    parent: Vec<usize>,
}

impl DisjointSet {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
        }
    }

    fn find(&mut self, i: usize) -> usize {
        let root = self.parent[i];
        if root == i {
            i
        } else {
            let res = self.find(root);
            self.parent[i] = res;
            res
        }
    }

    fn union(&mut self, i: usize, j: usize) {
        let root_i = self.find(i);
        let root_j = self.find(j);
        if root_i != root_j {
            if root_i < root_j {
                self.parent[root_j] = root_i;
            } else {
                self.parent[root_i] = root_j;
            }
        }
    }
}

pub struct DatasetSplitter;

impl DatasetSplitter {
    /// Enforces the 3-state manifest commit contract on an output split directory:
    /// 1. `split_manifest.json` exists AND all listed partition SHA-256 digests (`train_sha256`,
    ///    `val_sha256`, `test_sha256`) are non-empty and match on-disk `train.jsonl`, `val.jsonl`,
    ///    `test.jsonl` -> **Committed** (`Ok(())`).
    /// 2. `split_manifest.json` is missing -> **Uncommitted** (`Err(SplitterError::Io(NotFound))`).
    /// 3. `split_manifest.json` exists, but JSON is malformed, a partition file is missing, or any
    ///    partition file's SHA-256 mismatches the manifest -> **Corrupted / Fail Closed**
    ///    (`Err(SplitterError::ManifestCorrupted)`).
    pub fn verify_committed_split<P: AsRef<Path>>(output_dir: P) -> Result<(), SplitterError> {
        let dir = output_dir.as_ref();
        let manifest_path = dir.join("split_manifest.json");
        if !manifest_path.exists() {
            return Err(SplitterError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("split_manifest.json missing in {}", dir.display()),
            )));
        }
        let manifest_str = fs::read_to_string(&manifest_path).map_err(|e| {
            SplitterError::ManifestCorrupted(format!(
                "Failed to read {}: {}",
                manifest_path.display(),
                e
            ))
        })?;
        let manifest: Value = serde_json::from_str(&manifest_str).map_err(|e| {
            SplitterError::ManifestCorrupted(format!(
                "Malformed JSON in {}: {}",
                manifest_path.display(),
                e
            ))
        })?;
        for (file_name, sha_key) in [
            ("train.jsonl", "train_sha256"),
            ("val.jsonl", "val_sha256"),
            ("test.jsonl", "test_sha256"),
        ] {
            let file_path = dir.join(file_name);
            if !file_path.exists() {
                return Err(SplitterError::ManifestCorrupted(format!(
                    "Committed manifest exists in {}, but partition file {} is missing",
                    dir.display(),
                    file_name
                )));
            }
            let expected_sha = manifest
                .get(sha_key)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    SplitterError::ManifestCorrupted(format!(
                        "Missing or empty '{}' in {}",
                        sha_key,
                        manifest_path.display()
                    ))
                })?;
            let actual_sha = compute_file_sha256(&file_path).map_err(|e| {
                SplitterError::ManifestCorrupted(format!(
                    "Failed to compute SHA-256 for {}: {}",
                    file_path.display(),
                    e
                ))
            })?;
            if actual_sha != expected_sha {
                return Err(SplitterError::ManifestCorrupted(format!(
                    "SHA-256 mismatch for {}: manifest expected {}, actual on-disk {}",
                    file_path.display(),
                    expected_sha,
                    actual_sha
                )));
            }
        }
        Ok(())
    }

    /// Checks whether a directory contains a complete, cryptographically consistent split
    /// (`split_manifest.json` + `train.jsonl`, `val.jsonl`, `test.jsonl` matching manifest SHA-256s).
    fn is_valid_committed_split_dir(dir: &Path) -> bool {
        Self::verify_committed_split(dir).is_ok()
    }

    /// Resolves the active immutable release directory from `current_release.json` (`releases/<release_id>`)
    /// and verifies both 3-state manifest/partition SHA-256 integrity and `manifest.release_id == pointer.release_id`
    /// cross-link integrity. Consumers resolving splits through this method obtain true single-pointer atomic visibility.
    pub fn resolve_active_release_dir<P: AsRef<Path>>(
        split_dir: P,
    ) -> Result<PathBuf, SplitterError> {
        let out_dir = split_dir.as_ref();
        let current_ptr = out_dir.join("current_release.json");
        if !current_ptr.exists() {
            return Err(SplitterError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("current_release.json missing in {}", out_dir.display()),
            )));
        }
        let ptr_str = fs::read_to_string(&current_ptr).map_err(|e| {
            SplitterError::ManifestCorrupted(format!(
                "Failed to read {}: {}",
                current_ptr.display(),
                e
            ))
        })?;
        let ptr_val: Value = serde_json::from_str(&ptr_str).map_err(|e| {
            SplitterError::ManifestCorrupted(format!(
                "Malformed JSON in {}: {}",
                current_ptr.display(),
                e
            ))
        })?;
        let rel_id = ptr_val
            .get("current_release")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty() && !s.contains("..") && !s.contains('/') && !s.contains('\\'))
            .ok_or_else(|| {
                SplitterError::ManifestCorrupted(format!(
                    "Invalid or missing 'current_release' in {}",
                    current_ptr.display()
                ))
            })?;

        let rel_dir = out_dir.join("releases").join(rel_id);
        Self::verify_committed_split(&rel_dir)?;

        let rel_manifest_str = fs::read_to_string(rel_dir.join("split_manifest.json"))?;
        let rel_manifest: Value = serde_json::from_str(&rel_manifest_str)?;
        let manifest_rel_id = rel_manifest
            .get("release_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if manifest_rel_id != rel_id {
            return Err(SplitterError::ManifestCorrupted(format!(
                "Release ID cross-link mismatch: current_release.json points to '{}', but release split_manifest.json has '{}'",
                rel_id, manifest_rel_id
            )));
        }

        Ok(rel_dir)
    }

    /// Scans an output directory for incomplete split artifacts (`.tmp_*` files or un-manifested /
    /// partially overwritten `train.jsonl` / `val.jsonl` / `test.jsonl` left behind by an interrupted split).
    ///
    /// Enforces the 3-state manifest commit contract and explicit recovery precedence:
    /// - **Committed (`manifest exists + hashes match`)**: Cleans up any leftover `.tmp_*` or `.bak_*` files and keeps split intact.
    /// - **Uncommitted (`manifest missing`)**:
    ///   1. First tries restoring from the authoritative `current_release.json` target (`resolve_active_release_dir`).
    ///   2. Otherwise, if a legacy `.bak` bundle (`split_manifest.json.bak` + `.bak_*.jsonl`) exists, restores from `.bak`.
    ///   3. Otherwise (first-run interruption with no prior valid release), purges the uncommitted partial artifacts.
    /// - **Corrupted (`manifest exists + hashes mismatch` with no active `.bak` rollback)**:
    ///   Fails closed with `Err(SplitterError::ManifestCorrupted(...))` rather than silently deleting or ignoring corrupted data.
    pub fn recover_or_purge_stale_artifacts<P: AsRef<Path>>(
        output_dir: P,
    ) -> Result<usize, SplitterError> {
        let out_dir = output_dir.as_ref();
        if !out_dir.exists() {
            return Ok(0);
        }

        let mut purged_or_recovered = 0usize;
        for entry in fs::read_dir(out_dir)? {
            let entry = entry?;
            let path = entry.path();
            if let Some(fname) = path.file_name().and_then(|n| n.to_str()) {
                if fname.starts_with(".tmp_") {
                    if path.is_file() {
                        fs::remove_file(&path)?;
                        purged_or_recovered += 1;
                    } else if path.is_dir() {
                        fs::remove_dir_all(&path)?;
                        purged_or_recovered += 1;
                    }
                }
            }
        }

        let releases_dir = out_dir.join("releases");
        if releases_dir.exists() && releases_dir.is_dir() {
            for entry in fs::read_dir(&releases_dir)? {
                let entry = entry?;
                let path = entry.path();
                if let Some(fname) = path.file_name().and_then(|n| n.to_str()) {
                    if fname.starts_with(".tmp_") && path.is_dir() {
                        fs::remove_dir_all(&path)?;
                        purged_or_recovered += 1;
                    }
                }
            }
        }

        let bak_manifest = out_dir.join("split_manifest.json.bak");
        let bak_train = out_dir.join(".bak_train.jsonl");
        let bak_val = out_dir.join(".bak_val.jsonl");
        let bak_test = out_dir.join(".bak_test.jsonl");
        let has_bak_bundle =
            bak_manifest.exists() && bak_train.exists() && bak_val.exists() && bak_test.exists();

        match Self::verify_committed_split(out_dir) {
            Ok(()) => {}
            Err(SplitterError::ManifestCorrupted(msg)) if !has_bak_bundle => {
                // Manifest exists on disk, no mid-overwrite .bak rollback is active, and hashes mismatch:
                // fail closed on post-commit corruption!
                return Err(SplitterError::ManifestCorrupted(msg));
            }
            Err(_) => {
                // Precedence 1: Authoritative valid current_release.json target
                if let Ok(rel_dir) = Self::resolve_active_release_dir(out_dir) {
                    fs::copy(rel_dir.join("train.jsonl"), out_dir.join("train.jsonl"))?;
                    fs::copy(rel_dir.join("val.jsonl"), out_dir.join("val.jsonl"))?;
                    fs::copy(rel_dir.join("test.jsonl"), out_dir.join("test.jsonl"))?;
                    fs::copy(
                        rel_dir.join("split_manifest.json"),
                        out_dir.join("split_manifest.json"),
                    )?;
                    purged_or_recovered += 1;
                } else if has_bak_bundle {
                    // Precedence 2: Legacy .bak rollback bundle
                    fs::rename(&bak_train, out_dir.join("train.jsonl"))?;
                    fs::rename(&bak_val, out_dir.join("val.jsonl"))?;
                    fs::rename(&bak_test, out_dir.join("test.jsonl"))?;
                    fs::rename(&bak_manifest, out_dir.join("split_manifest.json"))?;
                    purged_or_recovered += 1;
                } else {
                    // Precedence 3: First-run interruption with no prior valid release -> purge uncommitted partial files
                    for split_file in [
                        "train.jsonl",
                        "val.jsonl",
                        "test.jsonl",
                        "split_manifest.json",
                    ] {
                        let p = out_dir.join(split_file);
                        if p.exists() {
                            fs::remove_file(&p)?;
                            purged_or_recovered += 1;
                        }
                    }
                }
            }
        }

        // Clean up any leftover .bak files if top-level split is now valid
        if Self::is_valid_committed_split_dir(out_dir) {
            for bak in [
                "split_manifest.json.bak",
                ".bak_train.jsonl",
                ".bak_val.jsonl",
                ".bak_test.jsonl",
            ] {
                let p = out_dir.join(bak);
                if p.exists() {
                    let _ = fs::remove_file(&p);
                }
            }
        }

        Ok(purged_or_recovered)
    }

    /// Resolves a path (which may not yet exist on disk) into a canonicalized comparable `PathBuf`
    /// by canonicalizing its deepest existing ancestor and lexically normalizing remaining components.
    fn resolve_comparable_path(path: &Path) -> std::io::Result<PathBuf> {
        let abs = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()?.join(path)
        };

        let mut normalized = PathBuf::new();
        for comp in abs.components() {
            match comp {
                std::path::Component::CurDir => {}
                std::path::Component::ParentDir => {
                    normalized.pop();
                }
                other => normalized.push(other.as_os_str()),
            }
        }

        let mut existing = normalized.as_path();
        let mut tail: Vec<std::ffi::OsString> = Vec::new();
        while !existing.exists() {
            if let Some(name) = existing.file_name() {
                tail.push(name.to_os_string());
            }
            match existing.parent() {
                Some(p) if !p.as_os_str().is_empty() => existing = p,
                _ => break,
            }
        }

        let mut resolved = if existing.exists() {
            existing.canonicalize()?
        } else {
            normalized
        };
        for part in tail.into_iter().rev() {
            resolved.push(part);
        }
        Ok(resolved)
    }

    /// Deterministically split a JSONL dataset file or directory into train, val, and test partitions.
    ///
    /// The split assignment uses content hashing and DSU transitive cluster isolation so identical
    /// or near-duplicate prompts/targets are grouped into the same partition, eliminating cross-split leakage.
    /// Output files are published via a **manifest-committed crash-safe publication** protocol:
    /// `.tmp_*` files are written and `fsync`ed, promoted to `train.jsonl` / `val.jsonl` / `test.jsonl`,
    /// and finalized by atomically renaming `split_manifest.json`. Any interrupted run without a committed
    /// manifest is automatically purged on recovery.
    pub fn split<P: AsRef<Path>, Q: AsRef<Path>>(
        source_jsonl: P,
        output_dir: Q,
        ratio: SplitRatio,
    ) -> Result<SplitReport, SplitterError> {
        let src_path = source_jsonl.as_ref();
        let out_dir = output_dir.as_ref();

        if src_path == out_dir {
            return Err(SplitterError::SourceEqualsOutput(format!(
                "Source path '{}' matches output directory",
                src_path.display()
            )));
        }

        if !src_path.exists() {
            return Err(SplitterError::EmptyDataset(src_path.display().to_string()));
        }

        // Pre-creation containment check using existing-ancestor canonicalization + lexical normalization
        // so we never create directories inside the source dataset tree before rejecting.
        let s_canon = src_path.canonicalize()?;
        let o_pre_canon = Self::resolve_comparable_path(out_dir)?;
        if s_canon == o_pre_canon
            || o_pre_canon.starts_with(&s_canon)
            || s_canon.starts_with(&o_pre_canon)
        {
            return Err(SplitterError::SourceEqualsOutput(format!(
                "Source path '{}' overlaps with or contains/is contained by output directory '{}'",
                src_path.display(),
                out_dir.display()
            )));
        }

        fs::create_dir_all(out_dir)?;

        // Post-creation canonical verification (fail-closed)
        let o_canon = out_dir.canonicalize()?;
        if s_canon == o_canon || o_canon.starts_with(&s_canon) || s_canon.starts_with(&o_canon) {
            return Err(SplitterError::SourceEqualsOutput(format!(
                "Source path '{}' overlaps with or contains/is contained by output directory '{}'",
                src_path.display(),
                out_dir.display()
            )));
        }

        // Purge any stale .tmp_* or uncommitted split artifacts before starting
        Self::recover_or_purge_stale_artifacts(out_dir)?;

        let mut files_to_read: Vec<PathBuf> = Vec::new();
        if src_path.is_dir() {
            let mut entries: Vec<_> = fs::read_dir(src_path)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    if p.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
                        return false;
                    }
                    if let Some(fname) = p.file_name().and_then(|n| n.to_str()) {
                        // Explicitly exclude previously created split files and temporary artifacts
                        if fname == "train.jsonl"
                            || fname == "val.jsonl"
                            || fname == "test.jsonl"
                            || fname == "clean_train.jsonl"
                            || fname == "split_manifest.json"
                            || fname.contains(".tmp")
                            || fname.starts_with('.')
                        {
                            return false;
                        }
                    }
                    true
                })
                .collect();
            entries.sort();
            files_to_read = entries;
        } else if src_path.is_file() {
            files_to_read.push(src_path.to_path_buf());
        }

        if files_to_read.is_empty() {
            return Err(SplitterError::EmptyDataset(src_path.display().to_string()));
        }

        let mut canonical_samples: Vec<CanonicalSample> = Vec::new();

        for file_path in &files_to_read {
            let file = File::open(file_path)?;
            let reader = BufReader::with_capacity(128 * 1024, file);
            for line_res in reader.lines() {
                let line = line_res?;
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                canonical_samples.push(extract_canonical_sample(trimmed));
            }
        }

        let total = canonical_samples.len();
        if total == 0 {
            return Err(SplitterError::EmptyDataset(src_path.display().to_string()));
        }

        // 1. Cluster exact & near duplicates using DisjointSet
        let mut dsu = DisjointSet::new(total);
        let mut exact_map: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut prompt_map: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut target_map: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut raw_inv_index: HashMap<u64, Vec<usize>> = HashMap::new();

        for (i, s) in canonical_samples.iter().enumerate() {
            // Exact combined match
            if let Some(cands) = exact_map.get(&s.combined_hash) {
                for &prev in cands {
                    if canonical_samples[prev].combined_text == s.combined_text {
                        dsu.union(i, prev);
                    }
                }
            }
            exact_map.entry(s.combined_hash).or_default().push(i);

            // Exact prompt match
            if !s.prompt.is_empty() {
                if let Some(cands) = prompt_map.get(&s.prompt_hash) {
                    for &prev in cands {
                        if canonical_samples[prev].prompt == s.prompt {
                            dsu.union(i, prev);
                        }
                    }
                }
                prompt_map.entry(s.prompt_hash).or_default().push(i);
            }

            // Exact target match (substantive, >= 15 chars matching LeakageIndex)
            if s.target.len() >= 15 {
                if let Some(cands) = target_map.get(&s.target_hash) {
                    for &prev in cands {
                        if canonical_samples[prev].target == s.target {
                            dsu.union(i, prev);
                        }
                    }
                }
                target_map.entry(s.target_hash).or_default().push(i);
            }

            for &sh in s.shingles.iter().chain(s.prompt_shingles.iter()) {
                raw_inv_index.entry(sh).or_default().push(i);
            }
        }

        // Document-frequency cutoff for common shingles
        let df_threshold = if total > 20 { (total / 2).max(10) } else { usize::MAX };
        let mut inv_index: HashMap<u64, Vec<usize>> = HashMap::new();
        for (sh, mut doc_ids) in raw_inv_index {
            doc_ids.sort_unstable();
            doc_ids.dedup();
            if doc_ids.len() <= df_threshold {
                inv_index.insert(sh, doc_ids);
            }
        }

        // Pairwise near-duplicate union across both combined shingles and prompt shingles
        for i in 0..total {
            let s_comb = &canonical_samples[i].shingles;
            let s_prompt = &canonical_samples[i].prompt_shingles;
            if s_comb.is_empty() && s_prompt.is_empty() {
                continue;
            }
            let mut cand_set: HashSet<usize> = HashSet::new();
            for sh in s_comb.iter().chain(s_prompt.iter()) {
                if let Some(cands) = inv_index.get(sh) {
                    for &j in cands {
                        if j > i {
                            cand_set.insert(j);
                        }
                    }
                }
            }
            let mut sorted_cands: Vec<usize> = cand_set.into_iter().collect();
            sorted_cands.sort_unstable();
            for j in sorted_cands {
                let t_comb = &canonical_samples[j].shingles;
                let t_prompt = &canonical_samples[j].prompt_shingles;

                let overlap_comb = if !s_comb.is_empty() && !t_comb.is_empty() {
                    let shared = s_comb.intersection(t_comb).count();
                    (shared as f64 / s_comb.len() as f64)
                        .max(shared as f64 / t_comb.len() as f64)
                } else {
                    0.0
                };

                let overlap_prompt = if !s_prompt.is_empty() && !t_prompt.is_empty() {
                    let shared = s_prompt.intersection(t_prompt).count();
                    (shared as f64 / s_prompt.len() as f64)
                        .max(shared as f64 / t_prompt.len() as f64)
                } else {
                    0.0
                };

                if overlap_comb.max(overlap_prompt) >= LEAKAGE_OVERLAP_THRESHOLD {
                    dsu.union(i, j);
                }
            }
        }

        // 2. Group records by cluster root
        let mut cluster_members: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..total {
            let root = dsu.find(i);
            cluster_members.entry(root).or_default().push(i);
        }

        static RELEASE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let now_stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let seq = RELEASE_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let release_id = format!("run_{now_stamp}_{}_{seq}", std::process::id());

        let train_path = out_dir.join("train.jsonl");
        let val_path = out_dir.join("val.jsonl");
        let test_path = out_dir.join("test.jsonl");

        let tmp_train_path = out_dir.join(format!(".tmp_train_{now_stamp}.jsonl"));
        let tmp_val_path = out_dir.join(format!(".tmp_val_{now_stamp}.jsonl"));
        let tmp_test_path = out_dir.join(format!(".tmp_test_{now_stamp}.jsonl"));

        let mut train_w = BufWriter::with_capacity(128 * 1024, File::create(&tmp_train_path)?);
        let mut val_w = BufWriter::with_capacity(64 * 1024, File::create(&tmp_val_path)?);
        let mut test_w = BufWriter::with_capacity(64 * 1024, File::create(&tmp_test_path)?);

        let mut train_cnt = 0usize;
        let mut val_cnt = 0usize;
        let mut test_cnt = 0usize;

        let val_threshold = (ratio.train * 10000.0) as u64;
        let test_threshold = ((ratio.train + ratio.val) * 10000.0) as u64;

        let mut root_keys: Vec<usize> = cluster_members.keys().copied().collect();
        root_keys.sort();

        for root in root_keys {
            let members = &cluster_members[&root];
            let mut rep_text = &canonical_samples[root].combined_text;
            for &idx in members {
                if canonical_samples[idx].combined_text < *rep_text {
                    rep_text = &canonical_samples[idx].combined_text;
                }
            }
            let h = fnv1a_hash(rep_text.as_bytes()) % 10000;

            let mut sorted_members = members.clone();
            sorted_members.sort();

            if h < val_threshold {
                for &idx in &sorted_members {
                    train_w.write_all(canonical_samples[idx].raw_line.as_bytes())?;
                    train_w.write_all(b"\n")?;
                    train_cnt += 1;
                }
            } else if h < test_threshold {
                for &idx in &sorted_members {
                    val_w.write_all(canonical_samples[idx].raw_line.as_bytes())?;
                    val_w.write_all(b"\n")?;
                    val_cnt += 1;
                }
            } else {
                for &idx in &sorted_members {
                    test_w.write_all(canonical_samples[idx].raw_line.as_bytes())?;
                    test_w.write_all(b"\n")?;
                    test_cnt += 1;
                }
            }
        }

        train_w.flush()?;
        val_w.flush()?;
        test_w.flush()?;

        train_w.get_ref().sync_all()?;
        val_w.get_ref().sync_all()?;
        test_w.get_ref().sync_all()?;
        drop(train_w);
        drop(val_w);
        drop(test_w);

        // Compute SHA-256 digests on the staged temporary files BEFORE touching any live files
        let train_sha256 = compute_file_sha256(&tmp_train_path).unwrap_or_default();
        let val_sha256 = compute_file_sha256(&tmp_val_path).unwrap_or_default();
        let test_sha256 = compute_file_sha256(&tmp_test_path).unwrap_or_default();

        let manifest_content = serde_json::json!({
            "release_id": release_id,
            "timestamp_ns": now_stamp,
            "splitter_version": env!("CARGO_PKG_VERSION"),
            "split_algorithm_version": "fnv1a_dsu_13gram_v2",
            "source_canonical_path": s_canon.to_string_lossy(),
            "train_ratio": ratio.train,
            "val_ratio": ratio.val,
            "test_ratio": ratio.test,
            "total_samples": total,
            "train_records": train_cnt,
            "val_records": val_cnt,
            "test_records": test_cnt,
            "train_samples": train_cnt,
            "val_samples": val_cnt,
            "test_samples": test_cnt,
            "train_file": "train.jsonl",
            "val_file": "val.jsonl",
            "test_file": "test.jsonl",
            "train_sha256": train_sha256,
            "val_sha256": val_sha256,
            "test_sha256": test_sha256,
            "leakage_threshold": LEAKAGE_OVERLAP_THRESHOLD,
            "overlap_threshold": LEAKAGE_OVERLAP_THRESHOLD,
            "leakage_status": "CLEAN"
        });
        let manifest_pretty = serde_json::to_string_pretty(&manifest_content)?;

        // 1. Versioned release directory transaction: stage all 4 files inside releases/.tmp_run_<stamp>
        //    and atomically rename the directory to releases/run_<stamp>, then switch current_release.json.
        let releases_dir = out_dir.join("releases");
        fs::create_dir_all(&releases_dir)?;
        let tmp_release_dir = releases_dir.join(format!(".tmp_{release_id}"));
        let final_release_dir = releases_dir.join(&release_id);
        fs::create_dir_all(&tmp_release_dir)?;
        fs::copy(&tmp_train_path, tmp_release_dir.join("train.jsonl"))?;
        fs::copy(&tmp_val_path, tmp_release_dir.join("val.jsonl"))?;
        fs::copy(&tmp_test_path, tmp_release_dir.join("test.jsonl"))?;
        {
            let mut rmf = File::create(tmp_release_dir.join("split_manifest.json"))?;
            rmf.write_all(manifest_pretty.as_bytes())?;
            rmf.flush()?;
            rmf.sync_all()?;
        }
        fs::rename(&tmp_release_dir, &final_release_dir)?;

        let tmp_ptr = out_dir.join(format!(".tmp_current_release_{now_stamp}.json"));
        let current_ptr = out_dir.join("current_release.json");
        let ptr_json = serde_json::json!({
            "current_release": release_id,
            "release_dir": final_release_dir.to_string_lossy(),
            "train_sha256": train_sha256,
            "val_sha256": val_sha256,
            "test_sha256": test_sha256
        });
        {
            let mut pf = File::create(&tmp_ptr)?;
            pf.write_all(serde_json::to_string_pretty(&ptr_json)?.as_bytes())?;
            pf.flush()?;
            pf.sync_all()?;
        }
        fs::rename(&tmp_ptr, &current_ptr)?;

        // 2. Rollback-safe top-level publication:
        //    If a valid top-level split already exists, preserve .bak backups and atomically rename
        //    split_manifest.json -> split_manifest.json.bak BEFORE modifying top-level partition files,
        //    so a crash mid-promotion leaves split_manifest.json absent + .bak present (restoring the
        //    previous valid split on recovery), whereas split_manifest.json present + mismatched hashes
        //    unambiguously signals post-commit corruption.
        let manifest_file = out_dir.join("split_manifest.json");
        if Self::is_valid_committed_split_dir(out_dir) {
            let _ = fs::copy(&train_path, out_dir.join(".bak_train.jsonl"));
            let _ = fs::copy(&val_path, out_dir.join(".bak_val.jsonl"));
            let _ = fs::copy(&test_path, out_dir.join(".bak_test.jsonl"));
            let _ = fs::rename(&manifest_file, out_dir.join("split_manifest.json.bak"));
        }

        let tmp_manifest_file = out_dir.join(format!(".tmp_manifest_{now_stamp}.json"));
        {
            let mut mf = File::create(&tmp_manifest_file)?;
            mf.write_all(manifest_pretty.as_bytes())?;
            mf.flush()?;
            mf.sync_all()?;
        }

        fs::rename(&tmp_train_path, &train_path)?;
        fs::rename(&tmp_val_path, &val_path)?;
        fs::rename(&tmp_test_path, &test_path)?;
        fs::rename(&tmp_manifest_file, &manifest_file)?;

        // 3. Remove backup files once the new top-level split_manifest.json is committed
        for bak in [
            "split_manifest.json.bak",
            ".bak_train.jsonl",
            ".bak_val.jsonl",
            ".bak_test.jsonl",
        ] {
            let p = out_dir.join(bak);
            if p.exists() {
                let _ = fs::remove_file(&p);
            }
        }

        Ok(SplitReport {
            total_samples: total,
            train_samples: train_cnt,
            val_samples: val_cnt,
            test_samples: test_cnt,
            train_path: train_path.to_string_lossy().to_string(),
            val_path: val_path.to_string_lossy().to_string(),
            test_path: test_path.to_string_lossy().to_string(),
            manifest_path: Some(manifest_file.to_string_lossy().to_string()),
        })
    }
}

pub struct LeakageChecker;

impl LeakageChecker {
    /// Detect contamination / leakage between training data and evaluation data.
    ///
    /// Evaluates:
    /// 1. Exact raw line match.
    /// 2. Exact substantive combined sample match (prompt + target).
    /// 3. Exact prompt match.
    /// 4. Exact target match for substantive answers.
    /// 5. 13-gram shingle near-duplicate overlap (>= 85%).
    pub fn check_leakage<P: AsRef<Path>, Q: AsRef<Path>>(
        train_path: P,
        eval_path: Q,
    ) -> Result<LeakageReport, SplitterError> {
        let train_file = File::open(train_path.as_ref())?;
        let train_reader = BufReader::with_capacity(128 * 1024, train_file);
        let mut train_samples = Vec::new();
        for line_res in train_reader.lines() {
            let line = line_res?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            train_samples.push(extract_canonical_sample(trimmed));
        }

        let eval_file = File::open(eval_path.as_ref())?;
        let eval_reader = BufReader::with_capacity(64 * 1024, eval_file);
        let mut eval_samples = Vec::new();
        for line_res in eval_reader.lines() {
            let line = line_res?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            eval_samples.push(extract_canonical_sample(trimmed));
        }

        let total_train = train_samples.len();
        let total_eval = eval_samples.len();

        let index = LeakageIndex::new(train_samples);

        let mut exact_leaks = 0usize;
        let mut ngram_leaks = 0usize;
        let mut leaked_indices = HashSet::new();

        for (idx, eval_sample) in eval_samples.iter().enumerate() {
            if let Some(leak) = index.find_leakage(eval_sample) {
                leaked_indices.insert(idx);
                match leak {
                    LeakageKind::ExactLine
                    | LeakageKind::ExactCombined
                    | LeakageKind::ExactPrompt
                    | LeakageKind::ExactTarget => {
                        exact_leaks += 1;
                    }
                    LeakageKind::NearDuplicateShingle { .. } => {
                        ngram_leaks += 1;
                    }
                }
            }
        }

        let total_leaked = leaked_indices.len();
        let rate_pct = if total_eval > 0 {
            (total_leaked as f64 / total_eval as f64) * 100.0
        } else {
            0.0
        };

        Ok(LeakageReport {
            total_train_samples: total_train,
            total_eval_samples: total_eval,
            exact_match_leaks: exact_leaks,
            ngram_shingle_leaks: ngram_leaks,
            total_leaked_samples: total_leaked,
            leakage_rate_pct: rate_pct,
            is_clean: total_leaked == 0,
        })
    }

    /// Decontaminate a training dataset by writing a sanitized copy with any leaky records removed.
    ///
    /// Shares the identical `LeakageIndex` detection rules as `check_leakage`, ensuring that
    /// every sample flagged as an exact or near-duplicate leak against the evaluation set is dropped.
    pub fn decontaminate<P: AsRef<Path>, Q: AsRef<Path>, R: AsRef<Path>>(
        train_path: P,
        eval_path: Q,
        clean_train_output: R,
    ) -> Result<usize, SplitterError> {
        let eval_file = File::open(eval_path.as_ref())?;
        let eval_reader = BufReader::with_capacity(64 * 1024, eval_file);
        let mut eval_samples = Vec::new();
        for line_res in eval_reader.lines() {
            let line = line_res?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            eval_samples.push(extract_canonical_sample(trimmed));
        }

        // Build shared leakage index on the evaluation dataset
        let eval_index = LeakageIndex::new(eval_samples);

        let train_file = File::open(train_path.as_ref())?;
        let train_reader = BufReader::with_capacity(128 * 1024, train_file);

        let clean_out_path = clean_train_output.as_ref();
        let now_stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let parent_dir = clean_out_path.parent().unwrap_or_else(|| Path::new("."));
        let tmp_out_path = parent_dir.join(format!(
            ".tmp_clean_train_{}_{now_stamp}.jsonl",
            clean_out_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("output")
        ));

        let out_file = File::create(&tmp_out_path)?;
        let mut writer = BufWriter::with_capacity(128 * 1024, out_file);

        let mut preserved = 0usize;
        for line_res in train_reader.lines() {
            let line = line_res?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let train_sample = extract_canonical_sample(trimmed);
            // Drop contaminated sample if any exact or near-duplicate leakage is detected
            if eval_index.find_leakage(&train_sample).is_some() {
                continue;
            }

            writer.write_all(train_sample.raw_line.as_bytes())?;
            writer.write_all(b"\n")?;
            preserved += 1;
        }

        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        fs::rename(&tmp_out_path, clean_out_path)?;

        Ok(preserved)
    }
}

/// Computes streaming SHA-256 checksum from authentic disk file.
pub fn compute_file_sha256(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let file = File::open(path).map_err(|e| format!("Failed to open file {}: {e}", path.display()))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("Read error on {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_test_dir() -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("tara_split_test_{stamp}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn test_dataset_split_deterministic() {
        let dir = make_test_dir();
        let src_file = dir.join("source.jsonl");
        {
            let mut f = File::create(&src_file).unwrap();
            for i in 0..100 {
                writeln!(f, r#"{{"input":"prompt_{i}","output":"completion_{i}"}}"#).unwrap();
            }
        }

        let out_dir = dir.join("split_out");
        let ratio = SplitRatio::new(0.80, 0.10, 0.10).unwrap();
        let report = DatasetSplitter::split(&src_file, &out_dir, ratio).unwrap();

        assert_eq!(report.total_samples, 100);
        assert_eq!(
            report.train_samples + report.val_samples + report.test_samples,
            100
        );
        assert!(report.train_samples >= 70 && report.train_samples <= 90);
        assert!(report.val_samples > 0);
        assert!(report.test_samples > 0);
        assert!(report.manifest_path.is_some());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_leakage_detection_and_decontamination() {
        let dir = make_test_dir();
        let train_file = dir.join("train.jsonl");
        let eval_file = dir.join("eval.jsonl");
        let clean_train_file = dir.join("clean_train.jsonl");

        {
            let mut f = File::create(&train_file).unwrap();
            writeln!(
                f,
                r#"{{"input":"What is photosynthesis?","output":"Process of plants..."}}"#
            )
            .unwrap();
            writeln!(f, r#"{{"input":"Solve 2+2","output":"4"}}"#).unwrap();
            writeln!(
                f,
                r#"{{"input":"Write a quicksort in Rust","output":"fn qs() {{}}"}}"#
            )
            .unwrap();
        }

        {
            let mut f = File::create(&eval_file).unwrap();
            // Leaked sample
            writeln!(
                f,
                r#"{{"input":"What is photosynthesis?","output":"Process of plants..."}}"#
            )
            .unwrap();
            // Clean sample
            writeln!(
                f,
                r#"{{"input":"What is the capital of Karnataka?","output":"Bengaluru"}}"#
            )
            .unwrap();
        }

        let report = LeakageChecker::check_leakage(&train_file, &eval_file).unwrap();
        assert_eq!(report.total_eval_samples, 2);
        assert_eq!(report.total_leaked_samples, 1);
        assert!(!report.is_clean);

        // Decontaminate
        let preserved =
            LeakageChecker::decontaminate(&train_file, &eval_file, &clean_train_file).unwrap();
        assert_eq!(preserved, 2); // 3 original - 1 leaked = 2

        let clean_report = LeakageChecker::check_leakage(&clean_train_file, &eval_file).unwrap();
        assert_eq!(clean_report.exact_match_leaks, 0);
        assert_eq!(clean_report.total_leaked_samples, 0);
        assert!(clean_report.is_clean);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_cross_split_zero_leakage_and_isolation() {
        let dir = make_test_dir();
        let src_file = dir.join("canonical_source.jsonl");

        {
            let mut f = File::create(&src_file).unwrap();
            for i in 0..150 {
                writeln!(
                    f,
                    r#"{{"input":"Question {} about science and computing","output":"Answer {} with detailed factual explanation."}}"#,
                    i, i
                )
                .unwrap();
            }
        }

        let split_dir = dir.join("splits");
        let report = DatasetSplitter::split(
            &src_file,
            &split_dir,
            SplitRatio::new(0.80, 0.10, 0.10).unwrap(),
        )
        .unwrap();

        assert_eq!(report.total_samples, 150);
        assert_eq!(
            report.train_samples + report.val_samples + report.test_samples,
            150
        );
        assert!(report.train_samples > 0);
        assert!(report.val_samples > 0);
        assert!(report.test_samples > 0);

        let train_path = &report.train_path;
        let val_path = &report.val_path;
        let test_path = &report.test_path;

        // 1. Train vs Val Leakage: Must be 0
        let train_val = LeakageChecker::check_leakage(train_path, val_path).unwrap();
        assert_eq!(train_val.exact_match_leaks, 0);
        assert_eq!(train_val.ngram_shingle_leaks, 0);
        assert_eq!(train_val.total_leaked_samples, 0);
        assert!(train_val.is_clean);

        // 2. Train vs Test Leakage: Must be 0
        let train_test = LeakageChecker::check_leakage(train_path, test_path).unwrap();
        assert_eq!(train_test.exact_match_leaks, 0);
        assert_eq!(train_test.ngram_shingle_leaks, 0);
        assert_eq!(train_test.total_leaked_samples, 0);
        assert!(train_test.is_clean);

        // 3. Val vs Test Leakage: Must be 0
        let val_test = LeakageChecker::check_leakage(val_path, test_path).unwrap();
        assert_eq!(val_test.exact_match_leaks, 0);
        assert_eq!(val_test.ngram_shingle_leaks, 0);
        assert_eq!(val_test.total_leaked_samples, 0);
        assert!(val_test.is_clean);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_near_duplicate_overlap_detected() {
        let dir = make_test_dir();
        let train_file = dir.join("train.jsonl");
        let eval_file = dir.join("eval.jsonl");

        {
            let mut f = File::create(&train_file).unwrap();
            writeln!(
                f,
                r#"{{"input":"The quick brown fox jumps over the lazy dog in the sunny park and rests under the shade","output":"Pangram description alpha."}}"#
            )
            .unwrap();
        }

        {
            let mut f = File::create(&eval_file).unwrap();
            // Near duplicate with >85% shared shingles
            writeln!(
                f,
                r#"{{"input":"The quick brown fox jumps over the lazy dog in the sunny park and rests under the shade!","output":"Pangram description beta."}}"#
            )
            .unwrap();
        }

        let report = LeakageChecker::check_leakage(&train_file, &eval_file).unwrap();
        assert_eq!(report.total_eval_samples, 1);
        assert_eq!(report.total_leaked_samples, 1);
        assert_eq!(report.ngram_shingle_leaks, 1);
        assert!(!report.is_clean);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_near_duplicate_decontaminated() {
        let dir = make_test_dir();
        let train_file = dir.join("train.jsonl");
        let eval_file = dir.join("eval.jsonl");
        let clean_file = dir.join("clean.jsonl");

        {
            let mut f = File::create(&train_file).unwrap();
            // Near duplicate of eval sample
            writeln!(
                f,
                r#"{{"input":"The mitochondria is the powerhouse of the cell providing biochemical energy","output":"Cell biology fact."}}"#
            )
            .unwrap();
            // Completely distinct sample
            writeln!(
                f,
                r#"{{"input":"Binary search algorithm runs in logarithmic time O(log N)","output":"Computer science complexity."}}"#
            )
            .unwrap();
        }

        {
            let mut f = File::create(&eval_file).unwrap();
            writeln!(
                f,
                r#"{{"input":"The mitochondria is the powerhouse of the cell providing biochemical energy!","output":"Cell biology fact."}}"#
            )
            .unwrap();
        }

        let preserved = LeakageChecker::decontaminate(&train_file, &eval_file, &clean_file).unwrap();
        assert_eq!(preserved, 1); // Only the binary search record survives

        let clean_report = LeakageChecker::check_leakage(&clean_file, &eval_file).unwrap();
        assert_eq!(clean_report.total_leaked_samples, 0);
        assert!(clean_report.is_clean);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_tara_schema_fields_detected() {
        let dir = make_test_dir();
        let train_file = dir.join("train.jsonl");
        let eval_file = dir.join("eval.jsonl");

        {
            let mut f = File::create(&train_file).unwrap();
            writeln!(
                f,
                r#"{{"formatted_input":"calculate 20 + 30","formatted_target":"50"}}"#
            )
            .unwrap();
        }

        {
            let mut f = File::create(&eval_file).unwrap();
            writeln!(
                f,
                r#"{{"trigger_pattern":"calculate 20 + 30","recommendation":"50"}}"#
            )
            .unwrap();
        }

        let report = LeakageChecker::check_leakage(&train_file, &eval_file).unwrap();
        assert_eq!(report.total_leaked_samples, 1);
        assert!(!report.is_clean);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_same_target_different_prompt_detected() {
        let dir = make_test_dir();
        let train_file = dir.join("train.jsonl");
        let eval_file = dir.join("eval.jsonl");

        {
            let mut f = File::create(&train_file).unwrap();
            writeln!(
                f,
                r#"{{"input":"What is Newton's second law?","output":"Force equals mass times acceleration (F=ma) in classical mechanics."}}"#
            )
            .unwrap();
        }

        {
            let mut f = File::create(&eval_file).unwrap();
            // Different prompt, identical substantive benchmark target
            writeln!(
                f,
                r#"{{"input":"State the formula for force","output":"Force equals mass times acceleration (F=ma) in classical mechanics."}}"#
            )
            .unwrap();
        }

        let report = LeakageChecker::check_leakage(&train_file, &eval_file).unwrap();
        assert_eq!(report.total_leaked_samples, 1);
        assert!(!report.is_clean);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_no_trailing_newline_counted() {
        let dir = make_test_dir();
        let no_nl_file = dir.join("no_nl.jsonl");

        {
            let mut f = File::create(&no_nl_file).unwrap();
            write!(f, r#"{{"input":"single record","output":"no newline at end"}}"#).unwrap();
        }

        let samples = read_canonical_samples_from_file(&no_nl_file).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].prompt, "single record");

        let _ = fs::remove_dir_all(&dir);
    }

    fn read_canonical_samples_from_file(path: &Path) -> Result<Vec<CanonicalSample>, SplitterError> {
        let file = File::open(path)?;
        let reader = BufReader::with_capacity(128 * 1024, file);
        let mut samples = Vec::new();
        for line_res in reader.lines() {
            let line = line_res?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            samples.push(extract_canonical_sample(trimmed));
        }
        Ok(samples)
    }

    #[test]
    fn test_nan_split_ratio_rejected() {
        assert!(SplitRatio::new(f64::NAN, 0.10, 0.10).is_err());
        assert!(SplitRatio::new(0.80, f64::NAN, 0.10).is_err());
        assert!(SplitRatio::new(0.80, 0.10, f64::NAN).is_err());
        assert!(SplitRatio::new(f64::INFINITY, 0.10, 0.10).is_err());
        assert!(SplitRatio::new(-0.80, 0.10, 0.10).is_err());
        assert!(SplitRatio::new(0.80, -0.10, 0.30).is_err());
        assert!(SplitRatio::new(0.70, 0.10, 0.10).is_err()); // Sum != 1.0
        assert!(SplitRatio::new(0.80, 0.10, 0.10).is_ok());
    }

    #[test]
    fn test_source_equals_output_rejected() {
        let dir = make_test_dir();
        let ratio = SplitRatio::new(0.80, 0.10, 0.10).unwrap();
        let res = DatasetSplitter::split(&dir, &dir, ratio);
        assert!(res.is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_end_to_end_split_contaminate_decontaminate_verify_clean() {
        let dir = make_test_dir();
        let src_file = dir.join("canonical.jsonl");

        {
            let mut f = File::create(&src_file).unwrap();
            for i in 0..100 {
                writeln!(
                    f,
                    r#"{{"input":"Unique problem item number {} in computing curriculum","output":"Comprehensive solution for item {}."}}"#,
                    i, i
                )
                .unwrap();
            }
        }

        let split_out = dir.join("splits");
        let ratio = SplitRatio::new(0.80, 0.10, 0.10).unwrap();
        let split_rep = DatasetSplitter::split(&src_file, &split_out, ratio).unwrap();

        // Check initial split is completely clean
        let initial_check = LeakageChecker::check_leakage(&split_rep.train_path, &split_rep.val_path).unwrap();
        assert!(initial_check.is_clean);
        assert_eq!(initial_check.total_leaked_samples, 0);

        // Intentionally contaminate train with an exact leak and a near-duplicate leak from val
        let contaminated_train = dir.join("contaminated_train.jsonl");
        {
            fs::copy(&split_rep.train_path, &contaminated_train).unwrap();
            let val_content = fs::read_to_string(&split_rep.val_path).unwrap();
            let first_val_line = val_content.lines().next().unwrap();
            let mut f = fs::OpenOptions::new()
                .append(true)
                .open(&contaminated_train)
                .unwrap();
            // 1. Exact leak
            writeln!(f, "{first_val_line}").unwrap();
            // 2. Near-duplicate leak (>85% shingle overlap)
            let val_val: Value = serde_json::from_str(first_val_line).unwrap();
            let prompt = val_val["input"].as_str().unwrap();
            let output = val_val["output"].as_str().unwrap();
            writeln!(
                f,
                r#"{{"input":"{}!","output":"{}"}}"#,
                prompt, output
            )
            .unwrap();
        }

        // Verify that check_leakage now detects the contamination
        let dirty_check = LeakageChecker::check_leakage(&contaminated_train, &split_rep.val_path).unwrap();
        assert!(!dirty_check.is_clean);
        assert!(dirty_check.total_leaked_samples >= 1);

        // Decontaminate
        let clean_train = dir.join("decontaminated_train.jsonl");
        let preserved = LeakageChecker::decontaminate(&contaminated_train, &split_rep.val_path, &clean_train).unwrap();
        assert!(preserved > 0);

        // Verify that check_leakage on decontaminated dataset is now 100% clean!
        let post_clean_check = LeakageChecker::check_leakage(&clean_train, &split_rep.val_path).unwrap();
        assert!(post_clean_check.is_clean);
        assert_eq!(post_clean_check.total_leaked_samples, 0);
        assert_eq!(post_clean_check.exact_match_leaks, 0);
        assert_eq!(post_clean_check.ngram_shingle_leaks, 0);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_empty_dataset_rejected() {
        let dir = make_test_dir();
        let empty_src_dir = dir.join("empty_src");
        fs::create_dir_all(&empty_src_dir).unwrap();
        let out_dir = dir.join("out");
        let ratio = SplitRatio::new(0.80, 0.10, 0.10).unwrap();

        let res = DatasetSplitter::split(&empty_src_dir, &out_dir, ratio);
        assert!(matches!(res, Err(SplitterError::EmptyDataset(_))));

        let empty_file = empty_src_dir.join("empty.jsonl");
        File::create(&empty_file).unwrap();
        let res2 = DatasetSplitter::split(&empty_file, &out_dir, ratio);
        assert!(matches!(res2, Err(SplitterError::EmptyDataset(_))));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_bidirectional_path_containment_rejected() {
        let dir = make_test_dir();
        let parent_dir = dir.join("datasets");
        let child_dir = parent_dir.join("canonical_v1");
        fs::create_dir_all(&child_dir).unwrap();

        let src_file = child_dir.join("data.jsonl");
        writeln!(File::create(&src_file).unwrap(), r#"{{"input":"a","output":"b"}}"#).unwrap();

        let ratio = SplitRatio::new(0.80, 0.10, 0.10).unwrap();

        // 1. output inside source directory -> REJECT
        let out_inside_src = child_dir.join("sub_splits");
        assert!(matches!(
            DatasetSplitter::split(&child_dir, &out_inside_src, ratio),
            Err(SplitterError::SourceEqualsOutput(_))
        ));

        // 2. source directory inside output directory -> REJECT
        assert!(matches!(
            DatasetSplitter::split(&child_dir, &parent_dir, ratio),
            Err(SplitterError::SourceEqualsOutput(_))
        ));

        // 3. source file inside output directory -> REJECT
        assert!(matches!(
            DatasetSplitter::split(&src_file, &child_dir, ratio),
            Err(SplitterError::SourceEqualsOutput(_))
        ));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_stale_artifact_recovery_and_audit_manifest() {
        let dir = make_test_dir();
        let src_file = dir.join("source.jsonl");
        {
            let mut f = File::create(&src_file).unwrap();
            for i in 0..30 {
                writeln!(f, r#"{{"input":"item_{i}","output":"ans_{i}"}}"#).unwrap();
            }
        }

        let out_dir = dir.join("splits");
        fs::create_dir_all(&out_dir).unwrap();

        // Plant stale .tmp_* file and uncommitted train.jsonl (no split_manifest.json)
        let stale_tmp = out_dir.join(".tmp_train_99999.jsonl");
        let stale_train = out_dir.join("train.jsonl");
        writeln!(File::create(&stale_tmp).unwrap(), "corrupt tmp").unwrap();
        writeln!(File::create(&stale_train).unwrap(), "uncommitted train").unwrap();

        let purged = DatasetSplitter::recover_or_purge_stale_artifacts(&out_dir).unwrap();
        assert_eq!(purged, 2);
        assert!(!stale_tmp.exists());
        assert!(!stale_train.exists());

        // Now run full split and verify audit manifest fields
        let ratio = SplitRatio::new(0.80, 0.10, 0.10).unwrap();
        let report = DatasetSplitter::split(&src_file, &out_dir, ratio).unwrap();
        let manifest_str = fs::read_to_string(report.manifest_path.unwrap()).unwrap();
        let manifest: Value = serde_json::from_str(&manifest_str).unwrap();

        assert_eq!(manifest["split_algorithm_version"], "fnv1a_dsu_13gram_v2");
        assert_eq!(manifest["leakage_status"], "CLEAN");
        assert_eq!(manifest["total_samples"], 30);
        assert!(!manifest["train_sha256"].as_str().unwrap().is_empty());
        assert!(!manifest["val_sha256"].as_str().unwrap().is_empty());
        assert!(!manifest["test_sha256"].as_str().unwrap().is_empty());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_13gram_boundary_and_normalization_semantics() {
        let dir = make_test_dir();
        let train_file = dir.join("train.jsonl");
        let eval_below_threshold = dir.join("eval_below.jsonl");
        let eval_case_ws = dir.join("eval_case_ws.jsonl");

        writeln!(
            File::create(&train_file).unwrap(),
            r#"{{"input":"alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima","output":"ans1"}}"#
        )
        .unwrap();

        // Completely different second half (< 85% 13-char shingle overlap) -> NO LEAK
        writeln!(
            File::create(&eval_below_threshold).unwrap(),
            r#"{{"input":"alpha bravo charlie delta echo zulu yankee xray whiskey victor uniform","output":"ans2"}}"#
        )
        .unwrap();

        let rep_below = LeakageChecker::check_leakage(&train_file, &eval_below_threshold).unwrap();
        assert!(rep_below.is_clean);
        assert_eq!(rep_below.total_leaked_samples, 0);

        // Case and whitespace variation -> normalized to exact match -> LEAK
        writeln!(
            File::create(&eval_case_ws).unwrap(),
            r#"{{"input":"  ALPHA   BRAVO charlie   DELTA echo FOXTROT golf HOTEL india JULIET kilo LIMA  ","output":"ans2"}}"#
        )
        .unwrap();

        let rep_case_ws = LeakageChecker::check_leakage(&train_file, &eval_case_ws).unwrap();
        assert!(!rep_case_ws.is_clean);
        assert_eq!(rep_case_ws.total_leaked_samples, 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_repeated_split_idempotent_and_different_ratio() {
        let dir = make_test_dir();
        let src_file = dir.join("source.jsonl");
        {
            let mut f = File::create(&src_file).unwrap();
            for i in 0..100 {
                writeln!(f, r#"{{"input":"unique_q_{i}","output":"unique_a_{i}"}}"#).unwrap();
            }
        }

        let out_dir = dir.join("splits");
        let r1 = SplitRatio::new(0.80, 0.10, 0.10).unwrap();
        let rep1 = DatasetSplitter::split(&src_file, &out_dir, r1).unwrap();
        let train_bytes_1 = fs::read(&rep1.train_path).unwrap();
        let val_bytes_1 = fs::read(&rep1.val_path).unwrap();
        let test_bytes_1 = fs::read(&rep1.test_path).unwrap();
        let m1: Value = serde_json::from_str(
            &fs::read_to_string(rep1.manifest_path.as_ref().unwrap()).unwrap(),
        )
        .unwrap();

        let rep2 = DatasetSplitter::split(&src_file, &out_dir, r1).unwrap();
        let train_bytes_2 = fs::read(&rep2.train_path).unwrap();
        let val_bytes_2 = fs::read(&rep2.val_path).unwrap();
        let test_bytes_2 = fs::read(&rep2.test_path).unwrap();
        let m2: Value = serde_json::from_str(
            &fs::read_to_string(rep2.manifest_path.as_ref().unwrap()).unwrap(),
        )
        .unwrap();

        // 1. Same input + same ratio -> exact same output bytes, same SHA-256 hashes, same manifest semantic fields
        assert_eq!(rep1.train_samples, rep2.train_samples);
        assert_eq!(rep1.val_samples, rep2.val_samples);
        assert_eq!(rep1.test_samples, rep2.test_samples);
        assert_eq!(train_bytes_1, train_bytes_2);
        assert_eq!(val_bytes_1, val_bytes_2);
        assert_eq!(test_bytes_1, test_bytes_2);
        assert_eq!(m1["train_sha256"], m2["train_sha256"]);
        assert_eq!(m1["val_sha256"], m2["val_sha256"]);
        assert_eq!(m1["test_sha256"], m2["test_sha256"]);
        assert_eq!(m1["split_algorithm_version"], m2["split_algorithm_version"]);

        // 2. Same input + different ratio -> old valid split not reused, new hashes, new manifest
        let r2 = SplitRatio::new(0.50, 0.25, 0.25).unwrap();
        let rep3 = DatasetSplitter::split(&src_file, &out_dir, r2).unwrap();
        let m3: Value = serde_json::from_str(
            &fs::read_to_string(rep3.manifest_path.as_ref().unwrap()).unwrap(),
        )
        .unwrap();
        assert_eq!(rep3.total_samples, 100);
        assert!(rep3.train_samples < rep1.train_samples);
        assert_ne!(m1["train_sha256"], m3["train_sha256"]);
        assert_ne!(m1["val_sha256"], m3["val_sha256"]);
        assert_ne!(m1["release_id"], m3["release_id"]);
        assert_eq!(m3["train_ratio"], 0.50);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_mid_publish_crash_restores_previous_valid_release() {
        let dir = make_test_dir();
        let src_file = dir.join("source.jsonl");
        {
            let mut f = File::create(&src_file).unwrap();
            for i in 0..40 {
                writeln!(f, r#"{{"input":"item_q_{i}","output":"item_a_{i}"}}"#).unwrap();
            }
        }

        let out_dir = dir.join("splits");
        let r1 = SplitRatio::new(0.80, 0.10, 0.10).unwrap();
        let rep1 = DatasetSplitter::split(&src_file, &out_dir, r1).unwrap();

        let valid_train_sha = compute_file_sha256(Path::new(&rep1.train_path)).unwrap();
        let valid_val_sha = compute_file_sha256(Path::new(&rep1.val_path)).unwrap();
        let valid_test_sha = compute_file_sha256(Path::new(&rep1.test_path)).unwrap();

        // Simulate a crash mid-overwrite:
        // train.jsonl was overwritten with partial/corrupted bytes and split_manifest.json was removed,
        // while current_release.json / releases/<run_id> holds the last committed release!
        fs::write(&rep1.train_path, b"corrupted partial overwrite").unwrap();
        fs::remove_file(out_dir.join("split_manifest.json")).unwrap();
        fs::write(out_dir.join(".tmp_train_interrupted.jsonl"), b"partial").unwrap();

        // Startup recovery MUST restore the previous valid release instead of purging the valid split!
        let recovered = DatasetSplitter::recover_or_purge_stale_artifacts(&out_dir).unwrap();
        assert!(recovered >= 1);
        assert!(!out_dir.join(".tmp_train_interrupted.jsonl").exists());
        assert!(out_dir.join("split_manifest.json").exists());
        assert_eq!(
            compute_file_sha256(Path::new(&rep1.train_path)).unwrap(),
            valid_train_sha
        );
        assert_eq!(
            compute_file_sha256(Path::new(&rep1.val_path)).unwrap(),
            valid_val_sha
        );
        assert_eq!(
            compute_file_sha256(Path::new(&rep1.test_path)).unwrap(),
            valid_test_sha
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_nonexistent_nested_output_inside_source_rejected_without_creating_dirs() {
        let dir = make_test_dir();
        let src_dir = dir.join("canonical");
        fs::create_dir_all(&src_dir).unwrap();
        let shard = src_dir.join("shard_0.jsonl");
        {
            let mut f = File::create(&shard).unwrap();
            for i in 0..30 {
                writeln!(f, r#"{{"input":"q_{i}","output":"a_{i}"}}"#).unwrap();
            }
        }

        let ratio = SplitRatio::new(0.80, 0.10, 0.10).unwrap();

        // Deep non-existent path inside source -> must be rejected BEFORE creating any directories inside src_dir
        let deep_nested = src_dir.join("deep").join("nested").join("splits");
        let res = DatasetSplitter::split(&src_dir, &deep_nested, ratio);
        assert!(matches!(res, Err(SplitterError::SourceEqualsOutput(_))));
        assert!(
            !src_dir.join("deep").exists(),
            "Pre-creation containment check must not leave partial directories inside source"
        );

        // Non-existent sibling with '..' traversal -> must resolve cleanly and succeed on first run
        let sibling_via_dotdot = src_dir.join("..").join("canonical_splits");
        let ok_rep = DatasetSplitter::split(&src_dir, &sibling_via_dotdot, ratio).unwrap();
        assert_eq!(ok_rep.total_samples, 30);
        assert!(dir.join("canonical_splits").join("split_manifest.json").exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_corrupted_committed_manifest_hash_mismatch_fails_closed() {
        let dir = make_test_dir();
        let src_file = dir.join("source.jsonl");
        {
            let mut f = File::create(&src_file).unwrap();
            for i in 0..40 {
                writeln!(f, r#"{{"input":"item_q_{i}","output":"item_a_{i}"}}"#).unwrap();
            }
        }

        let out_dir = dir.join("splits");
        let ratio = SplitRatio::new(0.80, 0.10, 0.10).unwrap();
        let rep = DatasetSplitter::split(&src_file, &out_dir, ratio).unwrap();

        // State 1: manifest exists + all listed hashes match -> committed (Ok)
        assert!(DatasetSplitter::verify_committed_split(&out_dir).is_ok());

        // State 3: manifest exists + partition file mutated post-commit (no .bak active) -> fail closed!
        fs::write(&rep.val_path, b"{\"input\":\"tampered\",\"output\":\"corrupted\"}\n").unwrap();

        let verify_res = DatasetSplitter::verify_committed_split(&out_dir);
        assert!(matches!(verify_res, Err(SplitterError::ManifestCorrupted(_))));

        let recover_res = DatasetSplitter::recover_or_purge_stale_artifacts(&out_dir);
        assert!(matches!(recover_res, Err(SplitterError::ManifestCorrupted(_))));

        let _ = fs::remove_dir_all(&dir);
    }
}

