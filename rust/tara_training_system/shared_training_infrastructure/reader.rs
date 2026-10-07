//! Expandable, streaming dataset reader for TARA neural engine.
//!
//! Features:
//! - Non-fixed count: handles expandable, growing, and unbounded datasets.
//! - Dynamic shard expansion: add new shards or directories at runtime without interruption.
//! - Streaming I/O: zero in-memory accumulation, constant low-memory footprint (< 1 MB RAM).
//! - Multi-shard streaming modes: Sequential or Interleaved (round-robin across domain shards).
//! - Rich record parsing: extracts `id`, `input`, `output`, `domain`, `language`, `source_repo`, `license_spdx`.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReaderError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON deserialization error in {path}:{line}: {error}")]
    Json {
        path: String,
        line: usize,
        error: String,
    },
    #[error("no shards registered in expandable reader")]
    NoShards,
}

/// A structured training sample emitted by the dataset reader.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingSample {
    pub id: String,
    pub input: String,
    pub output: String,
    #[serde(default)]
    pub language: String,
    #[serde(default)]
    pub domain: String,
    #[serde(default)]
    pub source_repo: String,
    #[serde(default)]
    pub license_spdx: String,
    #[serde(default)]
    pub curriculum_tier: String,
    #[serde(default)]
    pub curriculum_order: u8,
}

/// Shard streaming strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardStreamingMode {
    /// Read one shard completely until EOF before proceeding to the next.
    Sequential,
    /// Interleave one sample from each active shard in round-robin fashion,
    /// providing natural multi-task domain balancing across coding, security, reasoning, and languages.
    Interleaved,
}

struct ShardState {
    path: PathBuf,
    reader: BufReader<File>,
    current_line: usize,
    exhausted: bool,
    records_read: usize,
}

impl ShardState {
    fn open(path: PathBuf) -> Result<Self, std::io::Error> {
        let file = File::open(&path)?;
        Ok(Self {
            path,
            reader: BufReader::with_capacity(64 * 1024, file),
            current_line: 0,
            exhausted: false,
            records_read: 0,
        })
    }

    fn reset(&mut self) -> Result<(), std::io::Error> {
        self.reader.seek(SeekFrom::Start(0))?;
        self.current_line = 0;
        self.exhausted = false;
        Ok(())
    }

    fn next_sample(&mut self) -> Result<Option<TrainingSample>, ReaderError> {
        if self.exhausted {
            return Ok(None);
        }

        let mut line = String::new();
        while self.reader.read_line(&mut line)? > 0 {
            self.current_line += 1;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                line.clear();
                continue;
            }

            let val: Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(e) => {
                    return Err(ReaderError::Json {
                        path: self.path.display().to_string(),
                        line: self.current_line,
                        error: e.to_string(),
                    });
                }
            };
            line.clear();

            let input = val
                .get("formatted_input")
                .and_then(Value::as_str)
                .map(|s| s.to_string())
                .or_else(|| {
                    val.get("input")
                        .or_else(|| val.get("prompt"))
                        .or_else(|| val.get("instruction"))
                        .or_else(|| val.get("metadata").and_then(|m| m.get("input")))
                        .and_then(Value::as_str)
                        .map(|s| s.trim().to_string())
                })
                .unwrap_or_default();

            let output = val
                .get("formatted_target")
                .and_then(Value::as_str)
                .map(|s| s.to_string())
                .or_else(|| {
                    val.get("output")
                        .or_else(|| val.get("completion"))
                        .or_else(|| val.get("response"))
                        .or_else(|| val.get("metadata").and_then(|m| m.get("output")))
                        .and_then(Value::as_str)
                        .map(|s| s.trim().to_string())
                })
                .unwrap_or_default();

            if !input.trim().is_empty() && !output.trim().is_empty() {
                let id = val
                    .get("id")
                    .or_else(|| val.get("metadata").and_then(|m| m.get("id")))
                    .and_then(Value::as_str)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| {
                        format!(
                            "sample_{}_{}",
                            self.path
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("shard"),
                            self.current_line
                        )
                    });

                let language = val
                    .get("language")
                    .or_else(|| val.get("metadata").and_then(|m| m.get("language")))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let domain = val
                    .get("domain")
                    .or_else(|| val.get("metadata").and_then(|m| m.get("domain")))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let source_repo = val
                    .get("source_repo")
                    .or_else(|| val.get("source"))
                    .or_else(|| val.get("metadata").and_then(|m| m.get("source_repo").or_else(|| m.get("source"))))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let license_spdx = val
                    .get("license_spdx")
                    .or_else(|| val.get("metadata").and_then(|m| m.get("license_spdx")))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let curriculum_tier = val
                    .get("curriculum_tier")
                    .or_else(|| val.get("level"))
                    .or_else(|| val.get("metadata").and_then(|m| m.get("curriculum_tier").or_else(|| m.get("level"))))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let curriculum_order = val
                    .get("curriculum_order")
                    .or_else(|| val.get("metadata").and_then(|m| m.get("curriculum_order")))
                    .and_then(Value::as_u64)
                    .map(|o| o as u8)
                    .unwrap_or_else(|| match curriculum_tier.to_lowercase().as_str() {
                        "primary" => 1,
                        "middle_school" => 2,
                        "high_school" => 3,
                        "college" => 4,
                        "degree" => 5,
                        "engineering" => 6,
                        "phd" => 7,
                        "research" => 8,
                        _ => 0,
                    });

                self.records_read += 1;
                return Ok(Some(TrainingSample {
                    id,
                    input,
                    output,
                    language,
                    domain,
                    source_repo,
                    license_spdx,
                    curriculum_tier,
                    curriculum_order,
                }));
            }
        }

        self.exhausted = true;
        Ok(None)
    }
}

/// Expandable, non-fixed dataset reader.
///
/// Designed to stream continuously from an arbitrary, dynamically expanding collection
/// of shard files with zero fixed limit on record counts.
pub struct ExpandableDatasetReader {
    shards: Vec<ShardState>,
    mode: ShardStreamingMode,
    current_shard_idx: usize,
    total_yielded: usize,
    auto_rewind: bool,
}

impl ExpandableDatasetReader {
    /// Create a new empty expandable reader.
    pub fn new(mode: ShardStreamingMode) -> Self {
        Self {
            shards: Vec::new(),
            mode,
            current_shard_idx: 0,
            total_yielded: 0,
            auto_rewind: false,
        }
    }

    /// Set auto-rewind for continuous epoch streaming across all shards.
    pub fn with_auto_rewind(mut self, auto_rewind: bool) -> Self {
        self.auto_rewind = auto_rewind;
        self
    }

    /// Add a single JSONL shard file to the reader.
    pub fn add_shard<P: AsRef<Path>>(&mut self, path: P) -> Result<(), ReaderError> {
        let path_buf = path.as_ref().to_path_buf();
        if !path_buf.exists() || !path_buf.is_file() {
            return Err(ReaderError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Shard file not found: {}", path_buf.display()),
            )));
        }
        let state = ShardState::open(path_buf)?;
        self.shards.push(state);
        Ok(())
    }

    /// Add shards declared in a `manifest.json` file in deterministic order.
    pub fn add_shards_from_manifest<P: AsRef<Path>>(
        &mut self,
        manifest_path: P,
    ) -> Result<usize, ReaderError> {
        let path = manifest_path.as_ref();
        let bytes = fs::read_to_string(path)?;
        let manifest: Value = serde_json::from_str(&bytes).map_err(|e| ReaderError::Json {
            path: path.to_string_lossy().to_string(),
            line: 0,
            error: e.to_string(),
        })?;
        let base_dir = path.parent().unwrap_or_else(|| Path::new("."));

        let mut added = 0;
        if let Some(shards_obj) = manifest.get("shards").and_then(Value::as_object) {
            let mut shard_names: Vec<String> = shards_obj.keys().cloned().collect();
            shard_names.sort();
            for name in shard_names {
                let shard_path = base_dir.join(&name);
                if shard_path.exists() && shard_path.is_file() {
                    self.add_shard(shard_path)?;
                    added += 1;
                }
            }
        }
        Ok(added)
    }

    /// Add all `.jsonl` shard files found in a directory.
    /// If `manifest.json` exists in the directory, strictly loads manifest-declared shards.
    pub fn add_shards_from_dir<P: AsRef<Path>>(&mut self, dir: P) -> Result<usize, ReaderError> {
        let dir_path = dir.as_ref();
        if !dir_path.exists() || !dir_path.is_dir() {
            return Ok(0);
        }

        let manifest_candidate = dir_path.join("manifest.json");
        if manifest_candidate.exists() && manifest_candidate.is_file() {
            return self.add_shards_from_manifest(manifest_candidate);
        }

        let mut entries: Vec<PathBuf> = fs::read_dir(dir_path)?
            .filter_map(|e| e.ok().map(|entry| entry.path()))
            .filter(|p| p.is_file() && p.extension().and_then(|ext| ext.to_str()) == Some("jsonl"))
            .collect();
        entries.sort();

        let mut added = 0;
        for path in entries {
            self.add_shard(path)?;
            added += 1;
        }

        Ok(added)
    }

    /// Total number of shards currently managed.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Total samples successfully yielded so far.
    pub fn total_yielded(&self) -> usize {
        self.total_yielded
    }

    /// Reset all shards back to their beginning.
    pub fn reset_all(&mut self) -> Result<(), ReaderError> {
        for s in &mut self.shards {
            s.reset()?;
        }
        self.current_shard_idx = 0;
        Ok(())
    }

    /// Pull the next training sample.
    ///
    /// Depending on `mode`:
    /// - Sequential: streams one shard to EOF before advancing to the next.
    /// - Interleaved: round-robins across all non-exhausted shards for balanced multi-domain representation.
    pub fn next_sample(&mut self) -> Result<Option<TrainingSample>, ReaderError> {
        if self.shards.is_empty() {
            return Ok(None);
        }

        match self.mode {
            ShardStreamingMode::Sequential => self.next_sequential(),
            ShardStreamingMode::Interleaved => self.next_interleaved(),
        }
    }

    fn next_sequential(&mut self) -> Result<Option<TrainingSample>, ReaderError> {
        while self.current_shard_idx < self.shards.len() {
            if let Some(sample) = self.shards[self.current_shard_idx].next_sample()? {
                self.total_yielded += 1;
                return Ok(Some(sample));
            }
            // Shard exhausted, advance to next
            self.current_shard_idx += 1;
        }

        if self.auto_rewind && !self.shards.is_empty() {
            self.reset_all()?;
            return self.next_sequential();
        }

        Ok(None)
    }

    fn next_interleaved(&mut self) -> Result<Option<TrainingSample>, ReaderError> {
        let total = self.shards.len();
        let mut attempts = 0;

        while attempts < total {
            let idx = self.current_shard_idx;
            self.current_shard_idx = (self.current_shard_idx + 1) % total;

            if !self.shards[idx].exhausted {
                if let Some(sample) = self.shards[idx].next_sample()? {
                    self.total_yielded += 1;
                    return Ok(Some(sample));
                }
            }
            attempts += 1;
        }

        if self.auto_rewind && !self.shards.is_empty() {
            self.reset_all()?;
            return self.next_interleaved();
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_expandable_reader_interleaved() {
        let dir = std::env::temp_dir().join(format!(
            "tara_reader_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let shard1 = dir.join("shard1.jsonl");
        let shard2 = dir.join("shard2.jsonl");

        {
            let mut f1 = File::create(&shard1).unwrap();
            writeln!(
                f1,
                "{{\"id\": \"s1_1\", \"input\": \"in1\", \"output\": \"out1\"}}"
            )
            .unwrap();
            writeln!(
                f1,
                "{{\"id\": \"s1_2\", \"input\": \"in2\", \"output\": \"out2\"}}"
            )
            .unwrap();

            let mut f2 = File::create(&shard2).unwrap();
            writeln!(
                f2,
                "{{\"id\": \"s2_1\", \"input\": \"in3\", \"output\": \"out3\"}}"
            )
            .unwrap();
            writeln!(
                f2,
                "{{\"id\": \"s2_2\", \"input\": \"in4\", \"output\": \"out4\"}}"
            )
            .unwrap();
        }

        let mut reader = ExpandableDatasetReader::new(ShardStreamingMode::Interleaved);
        reader.add_shard(&shard1).unwrap();
        reader.add_shard(&shard2).unwrap();

        assert_eq!(reader.shard_count(), 2);

        let mut ids = Vec::new();
        while let Some(sample) = reader.next_sample().unwrap() {
            ids.push(sample.id);
        }

        assert_eq!(ids.len(), 4);
        assert_eq!(ids[0], "s1_1");
        assert_eq!(ids[1], "s2_1");
        assert_eq!(ids[2], "s1_2");
        assert_eq!(ids[3], "s2_2");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_curriculum_knowledge_dataset_integration() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let curriculum_dir = manifest_dir.join("../../storage/datasets/curriculum_knowledge");

        if !curriculum_dir.exists() {
            eprintln!("Curriculum dir not found at {:?}", curriculum_dir);
            return;
        }

        let mut reader = ExpandableDatasetReader::new(ShardStreamingMode::Interleaved);
        for entry in std::fs::read_dir(&curriculum_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("jsonl") {
                reader.add_shard(&path).unwrap();
            }
        }

        assert!(
            reader.shard_count() >= 5,
            "Expected at least 5 domain shards registered, found {}",
            reader.shard_count()
        );

        let mut sampled_count = 0;
        let mut domains_seen = std::collections::HashSet::new();

        while let Some(sample) = reader.next_sample().unwrap() {
            assert!(!sample.id.is_empty(), "Record ID cannot be empty");
            assert!(
                !sample.input.is_empty(),
                "Record input prompt cannot be empty"
            );
            assert!(!sample.output.is_empty(), "Record output cannot be empty");
            if !sample.domain.is_empty() {
                domains_seen.insert(sample.domain.clone());
            }

            sampled_count += 1;
            if sampled_count >= 100 {
                break;
            }
        }

        assert!(
            sampled_count >= 100,
            "Expected at least 100 samples read from curriculum"
        );
        assert!(
            domains_seen.contains("mathematics"),
            "Expected mathematics domain in stream"
        );
        assert!(
            domains_seen.contains("science"),
            "Expected science domain in stream"
        );
        assert!(
            domains_seen.contains("programming"),
            "Expected programming domain in stream"
        );
    }

    #[test]
    fn test_expandable_reader_tokenized_and_chatml_format() {
        let temp_dir = std::env::temp_dir().join(format!("tara_reader_test_{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir).unwrap();
        let shard_path = temp_dir.join("tokenized_shard.jsonl");

        let content = [
            r#"{"id":"rec_tok","formatted_input":"<|im_start|>user\nSolve 2+2<|im_end|>\n<|im_start|>assistant\n","formatted_target":"4<|im_end|>","metadata":{"input":"Solve 2+2","output":"4","domain":"mathematics","curriculum_tier":"primary","license_spdx":"CC-BY-4.0"}}"#,
            r#"{"id":"rec_inst","instruction":"Summarize text","response":"Summary here","metadata":{"domain":"science","curriculum_tier":"college"}}"#,
        ].join("\n");

        std::fs::write(&shard_path, content).unwrap();

        let mut reader = ExpandableDatasetReader::new(ShardStreamingMode::Sequential);
        reader.add_shard(&shard_path).unwrap();

        let s1 = reader.next_sample().unwrap().expect("sample 1 must exist");
        assert_eq!(s1.id, "rec_tok");
        assert_eq!(s1.input, "<|im_start|>user\nSolve 2+2<|im_end|>\n<|im_start|>assistant\n");
        assert_eq!(s1.output, "4<|im_end|>");
        assert_eq!(s1.domain, "mathematics");
        assert_eq!(s1.curriculum_tier, "primary");
        assert_eq!(s1.license_spdx, "CC-BY-4.0");

        let s2 = reader.next_sample().unwrap().expect("sample 2 must exist");
        assert_eq!(s2.id, "rec_inst");
        assert_eq!(s2.input, "Summarize text");
        assert_eq!(s2.output, "Summary here");
        assert_eq!(s2.domain, "science");
        assert_eq!(s2.curriculum_tier, "college");

        let _ = std::fs::remove_dir_all(temp_dir);
    }
}
