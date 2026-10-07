//! Deterministic Dataset Splitter & Leakage Control Engine.
//!
//! Provides:
//! - Deterministic train / val / test partitioning using 64-bit content hashes.
//! - Bidirectional contamination and leakage checking (exact match + 13-gram overlap).
//! - Automatic decontamination filtering to guarantee benchmark integrity.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SplitterError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Invalid ratio: train ({train}) + val ({val}) + test ({test}) must sum to 1.0")]
    InvalidRatio { train: f64, val: f64, test: f64 },
    #[error("Empty dataset provided at {0}")]
    EmptyDataset(String),
}

/// Ratio specification for train, validation, and test splits.
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
        let sum = train + val + test;
        if (sum - 1.0).abs() > 1e-4 || train <= 0.0 || val < 0.0 || test < 0.0 {
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

/// Deterministic FNV-1a 64-bit hash.
fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn extract_text_sample(val: &Value) -> Option<&str> {
    if let Some(inp) = val.get("input").and_then(Value::as_str) {
        Some(inp)
    } else if let Some(p) = val.get("prompt").and_then(Value::as_str) {
        Some(p)
    } else if let Some(t) = val.get("text").and_then(Value::as_str) {
        Some(t)
    } else {
        None
    }
}

fn extract_substantive_text(val: &Value) -> Option<String> {
    let raw = extract_text_sample(val)?;
    let prefix = "explain the core scientific research, methodology, and theoretical findings of the paper titled:";
    let lower = raw.trim().to_lowercase();
    if lower.starts_with(prefix) {
        let remainder = raw[prefix.len()..].trim();
        if let Some(first_quote) = remainder.find('"') {
            if let Some(last_quote) = remainder.rfind('"') {
                if last_quote > first_quote {
                    return Some(remainder[first_quote + 1..last_quote].trim().to_string());
                }
            }
        }
    }
    Some(raw.trim().to_string())
}

/// Computes 13-gram character shingles from a string for near-duplicate overlap detection.
fn compute_13gram_hashes(text: &str) -> Vec<u64> {
    let normalized = text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
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
    /// Deterministically split a JSONL dataset file or directory into train, val, and test partitions.
    ///
    /// The split assignment uses content hashing and cluster isolation so identical
    /// or near-duplicate prompts are grouped into the same partition, eliminating cross-split leakage.
    pub fn split<P: AsRef<Path>, Q: AsRef<Path>>(
        source_jsonl: P,
        output_dir: Q,
        ratio: SplitRatio,
    ) -> Result<SplitReport, SplitterError> {
        let src_path = source_jsonl.as_ref();
        let out_dir = output_dir.as_ref();
        fs::create_dir_all(out_dir)?;

        let mut files_to_read: Vec<std::path::PathBuf> = Vec::new();
        if src_path.is_dir() {
            let mut entries: Vec<_> = fs::read_dir(src_path)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().and_then(|ext| ext.to_str()) == Some("jsonl"))
                .collect();
            entries.sort();
            files_to_read = entries;
        } else if src_path.is_file() {
            files_to_read.push(src_path.to_path_buf());
        }

        if files_to_read.is_empty() {
            return Err(SplitterError::EmptyDataset(src_path.display().to_string()));
        }

        let mut raw_lines: Vec<String> = Vec::new();
        let mut substantive_texts: Vec<String> = Vec::new();

        for file_path in &files_to_read {
            let file = File::open(file_path)?;
            let reader = BufReader::with_capacity(128 * 1024, file);
            for line_res in reader.lines() {
                let line = line_res?;
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let sub = if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
                    extract_substantive_text(&val).unwrap_or_else(|| trimmed.to_string())
                } else {
                    trimmed.to_string()
                };
                raw_lines.push(trimmed.to_string());
                substantive_texts.push(sub);
            }
        }

        let total = raw_lines.len();
        if total == 0 {
            return Err(SplitterError::EmptyDataset(src_path.display().to_string()));
        }

        // 1. Cluster exact & near duplicates using DisjointSet
        let mut dsu = DisjointSet::new(total);
        let mut exact_map: HashMap<u64, usize> = HashMap::new();
        let mut inv_index: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut doc_shingles: Vec<HashSet<u64>> = Vec::with_capacity(total);

        for (i, text) in substantive_texts.iter().enumerate() {
            let text_clean = text.trim().to_lowercase();
            let h = fnv1a_hash(text_clean.as_bytes());

            if let Some(&prev) = exact_map.get(&h) {
                dsu.union(i, prev);
            } else {
                exact_map.insert(h, i);
            }

            let shingles: HashSet<u64> = compute_13gram_hashes(&text_clean).into_iter().collect();
            for &sh in &shingles {
                inv_index.entry(sh).or_default().push(i);
            }
            doc_shingles.push(shingles);
        }

        // Pairwise near-duplicate union
        for i in 0..total {
            let s_set = &doc_shingles[i];
            if s_set.is_empty() {
                continue;
            }
            let mut cand_counts: HashMap<usize, usize> = HashMap::new();
            for sh in s_set {
                if let Some(cands) = inv_index.get(sh) {
                    for &j in cands {
                        if j > i {
                            *cand_counts.entry(j).or_insert(0) += 1;
                        }
                    }
                }
            }
            let mut sorted_cands: Vec<_> = cand_counts.into_iter().collect();
            sorted_cands.sort_by_key(|&(j, _)| j);
            for (j, count) in sorted_cands {
                let overlap_i = count as f64 / s_set.len() as f64;
                let overlap_j = count as f64 / doc_shingles[j].len() as f64;
                if overlap_i >= 0.85 || overlap_j >= 0.85 {
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

        let train_path = out_dir.join("train.jsonl");
        let val_path = out_dir.join("val.jsonl");
        let test_path = out_dir.join("test.jsonl");

        let mut train_w = BufWriter::with_capacity(128 * 1024, File::create(&train_path)?);
        let mut val_w = BufWriter::with_capacity(64 * 1024, File::create(&val_path)?);
        let mut test_w = BufWriter::with_capacity(64 * 1024, File::create(&test_path)?);

        let mut train_cnt = 0usize;
        let mut val_cnt = 0usize;
        let mut test_cnt = 0usize;

        let val_threshold = (ratio.train * 10000.0) as u64;
        let test_threshold = ((ratio.train + ratio.val) * 10000.0) as u64;

        // Deterministically sort clusters by root index
        let mut root_keys: Vec<usize> = cluster_members.keys().copied().collect();
        root_keys.sort();

        for root in root_keys {
            let members = &cluster_members[&root];
            // Find the lexicographically smallest substantive text in the cluster for deterministic bucket hashing
            let mut rep_text = &substantive_texts[root];
            for &idx in members {
                if substantive_texts[idx] < *rep_text {
                    rep_text = &substantive_texts[idx];
                }
            }
            let h = fnv1a_hash(rep_text.as_bytes()) % 10000;

            let mut sorted_members = members.clone();
            sorted_members.sort();

            if h < val_threshold {
                for &idx in &sorted_members {
                    train_w.write_all(raw_lines[idx].as_bytes())?;
                    train_w.write_all(b"\n")?;
                    train_cnt += 1;
                }
            } else if h < test_threshold {
                for &idx in &sorted_members {
                    val_w.write_all(raw_lines[idx].as_bytes())?;
                    val_w.write_all(b"\n")?;
                    val_cnt += 1;
                }
            } else {
                for &idx in &sorted_members {
                    test_w.write_all(raw_lines[idx].as_bytes())?;
                    test_w.write_all(b"\n")?;
                    test_cnt += 1;
                }
            }
        }

        train_w.flush()?;
        val_w.flush()?;
        test_w.flush()?;

        Ok(SplitReport {
            total_samples: total,
            train_samples: train_cnt,
            val_samples: val_cnt,
            test_samples: test_cnt,
            train_path: train_path.to_string_lossy().to_string(),
            val_path: val_path.to_string_lossy().to_string(),
            test_path: test_path.to_string_lossy().to_string(),
        })
    }
}

pub struct LeakageChecker;

impl LeakageChecker {
    /// Detect contamination / leakage between training data and evaluation data.
    ///
    /// Checks:
    /// 1. Exact prompt/target match.
    /// 2. Shingle / 13-gram overlap (>80% shared shingles between eval prompt and any train sample).
    pub fn check_leakage<P: AsRef<Path>, Q: AsRef<Path>>(
        train_path: P,
        eval_path: Q,
    ) -> Result<LeakageReport, SplitterError> {
        let train_file = File::open(train_path.as_ref())?;
        let train_reader = BufReader::with_capacity(128 * 1024, train_file);

        let mut train_exact_hashes: HashSet<u64> = HashSet::new();
        let mut train_prompt_hashes: HashSet<u64> = HashSet::new();
        let mut train_doc_shingles: Vec<HashSet<u64>> = Vec::new();
        let mut train_inv_index: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut total_train = 0usize;

        for line_res in train_reader.lines() {
            let line = line_res?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let doc_id = total_train;
            total_train += 1;
            train_exact_hashes.insert(fnv1a_hash(trimmed.as_bytes()));

            if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
                if let Some(sub) = extract_substantive_text(&val) {
                    let sub_clean = sub.trim().to_lowercase();
                    train_prompt_hashes.insert(fnv1a_hash(sub_clean.as_bytes()));
                    let shingles: HashSet<u64> =
                        compute_13gram_hashes(&sub_clean).into_iter().collect();
                    for &sh in &shingles {
                        train_inv_index.entry(sh).or_default().push(doc_id);
                    }
                    train_doc_shingles.push(shingles);
                    continue;
                }
            }
            train_doc_shingles.push(HashSet::new());
        }

        let eval_file = File::open(eval_path.as_ref())?;
        let eval_reader = BufReader::with_capacity(64 * 1024, eval_file);

        let mut total_eval = 0usize;
        let mut exact_leaks = 0usize;
        let mut ngram_leaks = 0usize;
        let mut leaked_indices = HashSet::new();

        for (idx, line_res) in eval_reader.lines().enumerate() {
            let line = line_res?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            total_eval += 1;

            let line_hash = fnv1a_hash(trimmed.as_bytes());
            if train_exact_hashes.contains(&line_hash) {
                exact_leaks += 1;
                leaked_indices.insert(idx);
                continue;
            }

            if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
                if let Some(sub) = extract_substantive_text(&val) {
                    let sub_clean = sub.trim().to_lowercase();
                    let prompt_h = fnv1a_hash(sub_clean.as_bytes());
                    if train_prompt_hashes.contains(&prompt_h) {
                        exact_leaks += 1;
                        leaked_indices.insert(idx);
                        continue;
                    }

                    // Check pairwise single-document shingle overlap
                    let eval_shingles: HashSet<u64> =
                        compute_13gram_hashes(&sub_clean).into_iter().collect();
                    if !eval_shingles.is_empty() {
                        let mut cand_counts: HashMap<usize, usize> = HashMap::new();
                        for sh in &eval_shingles {
                            if let Some(docs) = train_inv_index.get(sh) {
                                for &d in docs {
                                    *cand_counts.entry(d).or_insert(0) += 1;
                                }
                            }
                        }

                        let mut is_near_leak = false;
                        for (train_doc_id, count) in cand_counts {
                            let ratio_eval = count as f64 / eval_shingles.len() as f64;
                            let train_len = train_doc_shingles[train_doc_id].len();
                            let ratio_train = if train_len > 0 {
                                count as f64 / train_len as f64
                            } else {
                                0.0
                            };
                            if ratio_eval >= 0.85 || ratio_train >= 0.85 {
                                is_near_leak = true;
                                break;
                            }
                        }

                        if is_near_leak {
                            ngram_leaks += 1;
                            leaked_indices.insert(idx);
                        }
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
    pub fn decontaminate<P: AsRef<Path>, Q: AsRef<Path>, R: AsRef<Path>>(
        train_path: P,
        eval_path: Q,
        clean_train_output: R,
    ) -> Result<usize, SplitterError> {
        let eval_file = File::open(eval_path.as_ref())?;
        let eval_reader = BufReader::with_capacity(64 * 1024, eval_file);

        let mut eval_prompt_hashes: HashSet<u64> = HashSet::new();
        for line_res in eval_reader.lines() {
            let line = line_res?;
            if let Ok(val) = serde_json::from_str::<Value>(line.trim()) {
                if let Some(sub) = extract_substantive_text(&val) {
                    eval_prompt_hashes.insert(fnv1a_hash(sub.trim().to_lowercase().as_bytes()));
                }
            }
        }

        let train_file = File::open(train_path.as_ref())?;
        let train_reader = BufReader::with_capacity(128 * 1024, train_file);
        let out_file = File::create(clean_train_output.as_ref())?;
        let mut writer = BufWriter::with_capacity(128 * 1024, out_file);

        let mut preserved = 0usize;
        for line_res in train_reader.lines() {
            let line = line_res?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
                if let Some(sub) = extract_substantive_text(&val) {
                    let h = fnv1a_hash(sub.trim().to_lowercase().as_bytes());
                    if eval_prompt_hashes.contains(&h) {
                        // Drop contaminated sample
                        continue;
                    }
                }
            }
            writer.write_all(trimmed.as_bytes())?;
            writer.write_all(b"\n")?;
            preserved += 1;
        }
        writer.flush()?;

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

    fn make_test_dir() -> std::path::PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
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
}
