//! TARA Canonical Dataset Sanitizer & Filter Binary
//!
//! 100% Pure Native Rust Implementation:
//! - Discovers existing shards in output directory and resumes from max_index + 1 (e.g. tara_shard_054.jsonl).
//! - Streams raw files from input directory (8,400+ Gutenberg, arXiv, Classical Science, LoRA).
//! - Applies security filters: prompt injection defense, exploit scrubbing, email redaction,
//!   corrupted character filtering, and exact SHA-256 deduplication.
//! - Emits standardized 500 MB shards into main dataset directory.
//! - Updates provenance manifest with complete metadata and dynamically computed SHA-256 hashes.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

const DEFAULT_TARGET_SHARD_BYTES: u64 = 500 * 1024 * 1024; // 500 MB

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilteredRecord {
    pub id: String,
    pub source: String,
    pub license: String,
    pub author: String,
    pub text: String,
    pub input: String,
    pub output: String,
    pub formatted_input: String,
    pub formatted_target: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardManifestEntry {
    pub shard: String,
    pub records: usize,
    pub bytes: u64,
    pub sha256: String,
}

pub struct DatasetFilterPipeline {
    pub input_dir: PathBuf,
    pub output_dir: PathBuf,
    target_shard_bytes: u64,
    current_shard_idx: usize,
    current_writer: Option<BufWriter<File>>,
    current_bytes: u64,
    current_records: usize,
    total_records: usize,
    seen_hashes: HashSet<[u8; 32]>,
    new_manifest_entries: Vec<ShardManifestEntry>,
    // Telemetry
    pub total_lines_read: usize,
    pub passed_records: usize,
    pub dropped_malicious: usize,
    pub dropped_garbage: usize,
    pub dropped_misaligned: usize,
    pub dropped_duplicates: usize,
    pub emails_scrubbed: usize,
}

impl DatasetFilterPipeline {
    pub fn new(input_dir: PathBuf, output_dir: PathBuf, target_shard_bytes: u64) -> std::io::Result<Self> {
        fs::create_dir_all(&output_dir)?;
        let start_idx = Self::discover_next_shard_index(&output_dir, "tara_shard");
        println!(">>> Starting at Shard Index: tara_shard_{:03}.jsonl", start_idx);

        let mut slf = Self {
            input_dir,
            output_dir,
            target_shard_bytes,
            current_shard_idx: start_idx,
            current_writer: None,
            current_bytes: 0,
            current_records: 0,
            total_records: 0,
            seen_hashes: HashSet::with_capacity(1_000_000),
            new_manifest_entries: Vec::new(),
            total_lines_read: 0,
            passed_records: 0,
            dropped_malicious: 0,
            dropped_garbage: 0,
            dropped_misaligned: 0,
            dropped_duplicates: 0,
            emails_scrubbed: 0,
        };
        slf.open_next_shard()?;
        Ok(slf)
    }

    fn discover_next_shard_index(dir: &Path, prefix: &str) -> usize {
        let mut max_idx = 0usize;
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with(prefix) && name.ends_with(".jsonl") {
                    let stripped = name.trim_end_matches(".jsonl");
                    if let Some(pos) = stripped.rfind('_') {
                        if let Ok(idx) = stripped[pos + 1..].parse::<usize>() {
                            if idx > max_idx {
                                max_idx = idx;
                            }
                        }
                    }
                }
            }
        }
        max_idx + 1
    }

    fn open_next_shard(&mut self) -> std::io::Result<()> {
        if let Some(mut w) = self.current_writer.take() {
            w.flush()?;
            let shard_name = format!("tara_shard_{:03}.jsonl", self.current_shard_idx - 1);
            let shard_path = self.output_dir.join(&shard_name);
            if shard_path.exists() && self.current_records > 0 {
                let bytes = fs::metadata(&shard_path)?.len();
                let f_bytes = fs::read(&shard_path)?;
                let mut h = Sha256::new();
                h.update(&f_bytes);
                let digest = hex::encode(h.finalize());

                self.new_manifest_entries.push(ShardManifestEntry {
                    shard: shard_name.clone(),
                    records: self.current_records,
                    bytes,
                    sha256: digest,
                });
                println!(
                    "  [Sealed Shard] {}: {} records ({:.2} MB)",
                    shard_name,
                    self.current_records,
                    bytes as f64 / (1024.0 * 1024.0)
                );
            }
        }

        let shard_name = format!("tara_shard_{:03}.jsonl", self.current_shard_idx);
        let shard_path = self.output_dir.join(&shard_name);
        println!(">>> Opening Shard: {:?}", shard_path);
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&shard_path)?;
        self.current_writer = Some(BufWriter::with_capacity(1024 * 1024, file));
        self.current_bytes = 0;
        self.current_records = 0;
        self.current_shard_idx += 1;
        Ok(())
    }

    pub fn is_malicious(text: &str) -> bool {
        let lower = text.to_lowercase();
        let patterns = [
            "ignore all previous instructions",
            "ignore all prior instructions",
            "disregard all previous guidelines",
            "system prompt override",
            "you are now in developer mode",
            "dan mode",
            "jailbreak enabled",
            "/bin/sh",
            "/bin/bash",
            "rm -rf /",
            "drop database",
            "<script",
            "eval(",
            "passthru(",
        ];
        for pat in &patterns {
            if lower.contains(pat) {
                return true;
            }
        }
        false
    }

    pub fn sanitize_emails(text: &str) -> (String, bool) {
        if !text.contains('@') {
            return (text.to_string(), false);
        }
        let mut words: Vec<&str> = text.split_whitespace().collect();
        let mut changed = false;
        let mut out = String::with_capacity(text.len());

        for (i, word) in words.drain(..).enumerate() {
            if i > 0 {
                out.push(' ');
            }
            if word.contains('@') && word.contains('.') && word.len() >= 5 {
                let parts: Vec<&str> = word.split('@').collect();
                if parts.len() == 2 && parts[1].contains('.') {
                    out.push_str("[EMAIL_REDACTED]");
                    changed = true;
                    continue;
                }
            }
            out.push_str(word);
        }
        (out, changed)
    }

    pub fn has_corrupted_artifacts(text: &str) -> bool {
        if text.contains(r#"\",\":"#)
            || text.contains(r#"\",\""#)
            || text.contains(r#"\", \""#)
            || text.contains(r#"\",\n"#)
            || text.contains(r#"\"}"#)
            || text.contains("[NULL]")
        {
            return true;
        }
        false
    }

    pub fn is_bilingual_misaligned(src: &str, tgt: &str) -> bool {
        let src_words = src.split_whitespace().count();
        let tgt_words = tgt.split_whitespace().count();
        if src_words == 0 || tgt_words == 0 {
            return true;
        }
        if (src_words <= 2 && tgt_words >= 8) || (tgt_words <= 2 && src_words >= 8) {
            return true;
        }
        let ratio = src_words as f64 / tgt_words as f64;
        if ratio < 0.18 || ratio > 5.5 {
            return true;
        }
        false
    }

    pub fn is_garbage(text: &str) -> bool {
        let trimmed = text.trim();
        if trimmed.len() < 10 || trimmed.len() > 131_072 {
            return true;
        }
        if trimmed.contains('\u{fffd}') || trimmed.contains('\0') {
            return true;
        }
        // Deliberately do NOT filter escaped quote artifacts here,
        // because programming language source code (Rust, C, Python, JSON)
        // legitimately contains string escapes like \",\" or \",\n.
        false
    }

    pub fn write_record(&mut self, record: &FilteredRecord) -> std::io::Result<()> {
        let line = serde_json::to_string(record)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))? + "\n";
        let line_len = line.len() as u64;

        if self.current_bytes + line_len > self.target_shard_bytes && self.current_records > 0 {
            self.open_next_shard()?;
        }

        if let Some(ref mut w) = self.current_writer {
            w.write_all(line.as_bytes())?;
            self.current_bytes += line_len;
            self.current_records += 1;
            self.total_records += 1;
            self.passed_records += 1;
        }
        Ok(())
    }

    pub fn process_file(&mut self, path: &Path) -> std::io::Result<()> {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("  [Warn] Cannot open {:?}: {}", path, e);
                return Ok(());
            }
        };

        let reader = BufReader::with_capacity(1024 * 1024, file);
        let default_source = path.file_stem().and_then(|s| s.to_str()).unwrap_or("corpus");

        for line_res in reader.lines() {
            self.total_lines_read += 1;
            let line = match line_res {
                Ok(l) => l,
                Err(_) => {
                    self.dropped_garbage += 1;
                    continue;
                }
            };

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let val: serde_json::Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(_) => {
                    self.dropped_garbage += 1;
                    continue;
                }
            };

            // Bilingual alignment and translation quality gating
            let en_field = val.get("english").and_then(|v| v.as_str())
                .or_else(|| val.get("source_text").and_then(|v| v.as_str()));
            let kn_field = val.get("kannada").and_then(|v| v.as_str())
                .or_else(|| val.get("target_text").and_then(|v| v.as_str()));
            if let (Some(en), Some(kn)) = (en_field, kn_field) {
                if Self::is_bilingual_misaligned(en, kn)
                    || Self::has_corrupted_artifacts(en)
                    || Self::has_corrupted_artifacts(kn)
                {
                    self.dropped_misaligned += 1;
                    continue;
                }
            }

            // Extract text fields
            let input_text = val.get("input").and_then(|v| v.as_str()).unwrap_or("");
            let output_text = val.get("output").and_then(|v| v.as_str()).unwrap_or("");
            let raw_text = val.get("text").and_then(|v| v.as_str())
                .or_else(|| val.get("abstract").and_then(|v| v.as_str()))
                .unwrap_or("");

            let text_payload = if !raw_text.is_empty() {
                raw_text.to_string()
            } else if !input_text.is_empty() && !output_text.is_empty() {
                format!("{}\n{}", input_text, output_text)
            } else if !output_text.is_empty() {
                output_text.to_string()
            } else {
                input_text.to_string()
            };

            if Self::is_garbage(&text_payload) {
                self.dropped_garbage += 1;
                continue;
            }

            if Self::is_malicious(&text_payload) {
                self.dropped_malicious += 1;
                continue;
            }

            let (clean_text, scrubbed) = Self::sanitize_emails(&text_payload);
            if scrubbed {
                self.emails_scrubbed += 1;
            }

            // Deduplication via SHA-256
            let mut hasher = Sha256::new();
            hasher.update(clean_text.as_bytes());
            let hash_bytes: [u8; 32] = hasher.finalize().into();

            if self.seen_hashes.contains(&hash_bytes) {
                self.dropped_duplicates += 1;
                continue;
            }
            self.seen_hashes.insert(hash_bytes);

            let inp = if !input_text.is_empty() { input_text.to_string() } else { clean_text.clone() };
            let out = if !output_text.is_empty() { output_text.to_string() } else { clean_text.clone() };
            let fmt_inp = format!("<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n", inp);
            let fmt_tgt = format!("{} [EOS]", out);

            let id = val.get("id").and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("rec_{:08}", self.passed_records + 1));

            let source = val.get("source").and_then(|v| v.as_str())
                .or_else(|| val.get("corpus").and_then(|v| v.as_str()))
                .unwrap_or(default_source)
                .to_string();

            let license = val.get("license").and_then(|v| v.as_str())
                .unwrap_or("PUBLIC_DOMAIN")
                .to_string();

            let author = val.get("author").and_then(|v| v.as_str())
                .or_else(|| val.get("authors").and_then(|v| v.as_str()))
                .unwrap_or("Authoritative Corpus")
                .to_string();

            let rec = FilteredRecord {
                id,
                source,
                license,
                author,
                text: clean_text,
                input: inp,
                output: out,
                formatted_input: fmt_inp,
                formatted_target: fmt_tgt,
            };

            self.write_record(&rec)?;
        }
        Ok(())
    }

    pub fn finish(&mut self) -> std::io::Result<()> {
        if let Some(mut w) = self.current_writer.take() {
            w.flush()?;
            let shard_name = format!("tara_shard_{:03}.jsonl", self.current_shard_idx - 1);
            let shard_path = self.output_dir.join(&shard_name);
            if shard_path.exists() && self.current_records > 0 {
                let bytes = fs::metadata(&shard_path)?.len();
                let f_bytes = fs::read(&shard_path)?;
                let mut h = Sha256::new();
                h.update(&f_bytes);
                let digest = hex::encode(h.finalize());

                self.new_manifest_entries.push(ShardManifestEntry {
                    shard: shard_name.clone(),
                    records: self.current_records,
                    bytes,
                    sha256: digest,
                });
                println!(
                    "  [Final Sealed Shard] {}: {} records ({:.2} MB)",
                    shard_name,
                    self.current_records,
                    bytes as f64 / (1024.0 * 1024.0)
                );
            } else if shard_path.exists() && self.current_records == 0 {
                let _ = fs::remove_file(shard_path);
            }
        }
        Ok(())
    }

    pub fn update_manifest(&self) -> std::io::Result<()> {
        let manifest_path = self.output_dir.join("provenance_manifest.json");
        let mut manifest_val: serde_json::Value = if manifest_path.exists() {
            let f = File::open(&manifest_path)?;
            serde_json::from_reader(f).unwrap_or(serde_json::json!({}))
        } else {
            serde_json::json!({})
        };

        let mut existing_shards = manifest_val
            .get("shards")
            .and_then(|s| s.as_array())
            .cloned()
            .unwrap_or_default();

        for entry in &self.new_manifest_entries {
            existing_shards.push(serde_json::json!({
                "shard": entry.shard,
                "records": entry.records,
                "bytes": entry.bytes,
                "sha256": entry.sha256
            }));
        }

        let total_shards = existing_shards.len();
        let total_records: usize = existing_shards
            .iter()
            .map(|s| s.get("records").and_then(|r| r.as_u64()).unwrap_or(0) as usize)
            .sum();

        manifest_val["shards"] = serde_json::Value::Array(existing_shards);
        manifest_val["total_shards"] = serde_json::json!(total_shards);
        manifest_val["total_records"] = serde_json::json!(total_records);
        manifest_val["last_updated_epoch_secs"] = serde_json::json!(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );

        let out_f = File::create(&manifest_path)?;
        serde_json::to_writer_pretty(out_f, &manifest_val)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        println!("[Manifest Updated] {:?}", manifest_path);
        Ok(())
    }
}

fn collect_jsonl_files(dir: &Path, files: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "jsonl" || ext == "json" {
                        files.push(path);
                    }
                }
            } else if path.is_dir() {
                let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                // Process lora and subdirectories, skip world_v1 trajectory files for language v2
                if name == "lora" || name == "json_converted" {
                    collect_jsonl_files(&path, files);
                }
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!("  TARA DATASET SANITIZER & FILTER PIPELINE (100% NATIVE RUST)");
    println!("================================================================================");

    let args: Vec<String> = std::env::args().collect();
    let mut cli_input: Option<PathBuf> = None;
    let mut cli_output: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" if i + 1 < args.len() => {
                cli_input = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            "--output" | "-o" if i + 1 < args.len() => {
                cli_output = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            "--help" | "-h" => {
                println!("Usage: tara_dataset_filter [OPTIONS]");
                println!("Options:");
                println!("  --input, -i <dir>   Input directory containing downloaded shards");
                println!("  --output, -o <dir>  Output directory for sanitized shards");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    let env_dataset_dir = std::env::var("TARA_DATASET_DIR").ok().map(PathBuf::from);

    let input_dir = cli_input
        .or_else(|| env_dataset_dir.as_ref().map(|p| p.join("downloaded")))
        .unwrap_or_else(|| PathBuf::from(r"D:\taracore_datasets\downloaded"));
    let output_dir = cli_output
        .or(env_dataset_dir)
        .unwrap_or_else(|| PathBuf::from(r"D:\main_dataset_v2"));
    let target_shard_bytes = DEFAULT_TARGET_SHARD_BYTES;

    println!("Input Directory:  {:?}", input_dir);
    println!("Output Directory: {:?}", output_dir);
    println!("Target Shard MB:  {:.2} MB\n", target_shard_bytes as f64 / (1024.0 * 1024.0));

    let mut pipeline = DatasetFilterPipeline::new(input_dir.clone(), output_dir, target_shard_bytes)?;

    println!("Scanning input files in {:?}...", input_dir);
    let mut files = Vec::new();
    collect_jsonl_files(&input_dir, &mut files);
    files.sort();
    println!("Discovered {} JSONL files to filter.\n", files.len());

    let total_files = files.len();
    for (idx, file_path) in files.iter().enumerate() {
        if idx % 500 == 0 || idx == total_files - 1 {
            println!(
                "[{}/{}] Filtering: {:?} (Passed: {}, Dups: {}, Dropped: {})",
                idx + 1,
                total_files,
                file_path.file_name().unwrap_or_default(),
                pipeline.passed_records,
                pipeline.dropped_duplicates,
                pipeline.dropped_malicious + pipeline.dropped_garbage
            );
        }
        pipeline.process_file(file_path)?;
    }

    pipeline.finish()?;
    pipeline.update_manifest()?;

    println!("\n================================================================================");
    println!("  FILTERING SUMMARY");
    println!("================================================================================");
    println!("Total Lines Processed:  {}", pipeline.total_lines_read);
    println!("Passed Records Written: {}", pipeline.passed_records);
    println!("Duplicates Dropped:     {}", pipeline.dropped_duplicates);
    println!("Malicious Dropped:      {}", pipeline.dropped_malicious);
    println!("Garbage Dropped:        {}", pipeline.dropped_garbage);
    println!("Misaligned Dropped:     {}", pipeline.dropped_misaligned);
    println!("Emails Scrubbed:        {}", pipeline.emails_scrubbed);
    println!("New Shards Created:     {}", pipeline.new_manifest_entries.len());
    println!("================================================================================");

    Ok(())
}
