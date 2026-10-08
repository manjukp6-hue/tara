//! Expandable, streaming dataset reader for TARA neural engine.
//!
//! Features:
//! - Non-fixed count: handles expandable, growing, and unbounded JSONL shard datasets.
//! - Genuine dynamic shard expansion: tracks committed byte offsets for unsealed shards so newly
//!   appended records are read after EOF, retains incomplete partial final lines without
//!   committing offset until a trailing newline (`\n`) arrives, detects truncated/replaced
//!   shards, and rescans watched directories/manifests.
//! - Bounded file-buffer pool: caps concurrent open file handles (`max_open_files`,
//!   default 8 × 64 KiB = ≤512 KiB configured file-buffer budget, excluding per-record
//!   parsing and metadata allocations) with lazy seek-on-reopen.
//! - Producer publication contract: directory scans ignore hidden/temporary files (`.tmp`,
//!   `.part`, leading `.`); producers should publish new shards via write `.tmp` → `fsync` →
//!   atomic rename to `.jsonl`, or append newline-terminated JSONL records to unsealed shards.
//! - Multi-shard streaming modes: Sequential or deterministic round-robin Interleaved, with
//!   explicit `DatasetStreamLifecycle` (`Finite` vs `LiveAppendWait`).
//! - Strict validation: path-containment checks on manifests, canonical path deduplication,
//!   checked `curriculum_order` conversion, and configurable malformed-record policy.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};
use thiserror::Error;

/// Default per-handle buffer size (64 KiB).
const READER_BUF_CAPACITY: usize = 64 * 1024;
/// Default maximum concurrently open file handles (8 × 64 KiB = ≤512 KiB configured file-buffer budget,
/// excluding per-record JSON parsing and metadata allocations).
const DEFAULT_MAX_OPEN_FILES: usize = 8;
/// Maximum retained reusable line buffer capacity before shrinking (64 KiB).
const MAX_RETAINED_LINE_CAPACITY: usize = 64 * 1024;

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
    #[error("invalid record in {path}:{line}: {reason}")]
    InvalidRecord {
        path: String,
        line: usize,
        reason: String,
    },
    #[error("invalid manifest {path}: {reason}")]
    InvalidManifest { path: String, reason: String },
    #[error("path traversal rejected in manifest {manifest}: shard '{shard}' escapes base directory")]
    PathTraversal { manifest: String, shard: String },
    #[error("shard {path} was truncated or replaced beneath active reader (committed offset {offset} > file length {file_len})")]
    ShardTruncatedOrReplaced {
        path: String,
        offset: u64,
        file_len: u64,
    },
    #[error("no shards registered in expandable reader")]
    NoShards,
}

/// Lifecycle state of a registered shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardAvailability {
    /// Shard has unread bytes available for streaming.
    Active,
    /// Shard reached EOF at `current_offset`, but is unsealed and may grow if appended to.
    TemporarilyAtEof,
    /// Shard is permanently sealed and reached EOF; will not be polled for growth unless reset.
    Sealed,
}

/// Policy governing how the reader behaves when all currently registered shards reach EOF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetStreamLifecycle {
    /// Finite dataset mode: when all shards reach EOF, optionally rewind to start a new epoch
    /// if `auto_rewind` is enabled.
    Finite,
    /// Live append-wait mode: unsealed shards in `TemporarilyAtEof` wait for newly appended
    /// data on disk without rewinding already-consumed records. `auto_rewind` will not reset
    /// shards while any unsealed shard is waiting in `TemporarilyAtEof`.
    LiveAppendWait,
}

/// Policy controlling how malformed JSON records or invalid field values are handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MalformedRecordPolicy {
    /// Fail immediately with `ReaderError::Json` or `ReaderError::InvalidRecord` (default).
    Strict,
    /// Increment `records_malformed` counter and continue reading subsequent lines.
    LenientSkip,
}

/// Live telemetry snapshot for `ExpandableDatasetReader`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReaderTelemetry {
    pub total_shards: usize,
    pub active_shards: usize,
    pub temporarily_eof_shards: usize,
    pub sealed_shards: usize,
    pub open_file_handles: usize,
    pub epoch_yielded: u64,
    pub lifetime_yielded: u64,
    pub records_skipped_empty: u64,
    pub records_malformed: u64,
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
    /// Interleave one sample from each active shard in deterministic round-robin fashion.
    Interleaved,
}

struct ShardState {
    path: PathBuf,
    reader: Option<BufReader<File>>,
    current_offset: u64,
    current_line: u64,
    availability: ShardAvailability,
    sealed_by_policy: bool,
    records_read: u64,
    lifetime_records_read: u64,
    records_skipped_empty: u64,
    records_malformed: u64,
    last_used_tick: u64,
}

impl ShardState {
    fn new(path: PathBuf, sealed_by_policy: bool) -> Result<Self, std::io::Error> {
        let _meta = fs::metadata(&path)?;
        Ok(Self {
            path,
            reader: None,
            current_offset: 0,
            current_line: 0,
            availability: ShardAvailability::Active,
            sealed_by_policy,
            records_read: 0,
            lifetime_records_read: 0,
            records_skipped_empty: 0,
            records_malformed: 0,
            last_used_tick: 0,
        })
    }

    fn verify_not_truncated(&self) -> Result<u64, ReaderError> {
        let file_len = fs::metadata(&self.path)?.len();
        if file_len < self.current_offset {
            return Err(ReaderError::ShardTruncatedOrReplaced {
                path: self.path.display().to_string(),
                offset: self.current_offset,
                file_len,
            });
        }
        Ok(file_len)
    }

    fn ensure_open(&mut self, tick: u64) -> Result<(), ReaderError> {
        self.last_used_tick = tick;
        self.verify_not_truncated()?;
        if self.reader.is_none() {
            let mut file = File::open(&self.path)?;
            if self.current_offset > 0 {
                file.seek(SeekFrom::Start(self.current_offset))?;
            }
            self.reader = Some(BufReader::with_capacity(READER_BUF_CAPACITY, file));
        }
        Ok(())
    }

    fn close_handle(&mut self) {
        self.reader = None;
    }

    fn reset(&mut self) -> Result<(), std::io::Error> {
        self.current_offset = 0;
        self.current_line = 0;
        self.records_read = 0;
        self.availability = ShardAvailability::Active;
        if let Some(ref mut reader) = self.reader {
            reader.seek(SeekFrom::Start(0))?;
        }
        Ok(())
    }

    /// Checks whether an unsealed shard at EOF has grown on disk since `current_offset`,
    /// and verifies that the underlying file was not truncated or replaced.
    fn probe_growth(&mut self) -> Result<bool, ReaderError> {
        let file_len = self.verify_not_truncated()?;
        if self.availability != ShardAvailability::TemporarilyAtEof || self.sealed_by_policy {
            return Ok(false);
        }
        if file_len > self.current_offset {
            self.availability = ShardAvailability::Active;
            if let Some(ref mut reader) = self.reader {
                reader.seek(SeekFrom::Start(self.current_offset))?;
            }
            return Ok(true);
        }
        Ok(false)
    }

    fn next_sample(
        &mut self,
        policy: MalformedRecordPolicy,
        line_buf: &mut String,
    ) -> Result<Option<TrainingSample>, ReaderError> {
        if self.availability == ShardAvailability::Sealed {
            return Ok(None);
        }
        if self.availability == ShardAvailability::TemporarilyAtEof && !self.probe_growth()? {
            return Ok(None);
        }

        loop {
            line_buf.clear();
            let bytes_read = {
                let reader = self
                    .reader
                    .as_mut()
                    .expect("ShardState::ensure_open must be called before next_sample");
                reader.read_line(line_buf)?
            };
            if bytes_read == 0 {
                self.verify_not_truncated()?;
                self.availability = if self.sealed_by_policy {
                    ShardAvailability::Sealed
                } else {
                    ShardAvailability::TemporarilyAtEof
                };
                return Ok(None);
            }

            // Invariant for unsealed growing shards:
            // An incomplete final line (missing trailing '\n') must NEVER be committed as a consumed
            // record or rejected as malformed JSON. Rewind the reader to `current_offset` and wait
            // in `TemporarilyAtEof` until the producer finishes writing the line.
            if !self.sealed_by_policy && !line_buf.ends_with('\n') {
                if let Some(ref mut reader) = self.reader {
                    reader.seek(SeekFrom::Start(self.current_offset))?;
                }
                self.availability = ShardAvailability::TemporarilyAtEof;
                return Ok(None);
            }

            self.current_offset = self.current_offset.saturating_add(bytes_read as u64);
            self.current_line = self.current_line.saturating_add(1);
            let line_num_usize = usize::try_from(self.current_line).unwrap_or(usize::MAX);

            let trimmed = line_buf.trim();
            if trimmed.is_empty() {
                self.records_skipped_empty = self.records_skipped_empty.saturating_add(1);
                continue;
            }

            let val: Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(e) => match policy {
                    MalformedRecordPolicy::Strict => {
                        return Err(ReaderError::Json {
                            path: self.path.display().to_string(),
                            line: line_num_usize,
                            error: e.to_string(),
                        });
                    }
                    MalformedRecordPolicy::LenientSkip => {
                        self.records_malformed = self.records_malformed.saturating_add(1);
                        continue;
                    }
                },
            };

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

            if input.trim().is_empty() || output.trim().is_empty() {
                self.records_skipped_empty = self.records_skipped_empty.saturating_add(1);
                continue;
            }

            let curriculum_tier = val
                .get("curriculum_tier")
                .or_else(|| val.get("level"))
                .or_else(|| {
                    val.get("metadata")
                        .and_then(|m| m.get("curriculum_tier").or_else(|| m.get("level")))
                })
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();

            let raw_order_val = val
                .get("curriculum_order")
                .or_else(|| val.get("metadata").and_then(|m| m.get("curriculum_order")));

            let curriculum_order = match raw_order_val {
                Some(v) if !v.is_null() => match v.as_u64().and_then(|o| u8::try_from(o).ok()) {
                    Some(valid_u8) => valid_u8,
                    None => match policy {
                        MalformedRecordPolicy::Strict => {
                            return Err(ReaderError::InvalidRecord {
                                path: self.path.display().to_string(),
                                line: line_num_usize,
                                reason: format!(
                                    "curriculum_order must be an integer in 0..=255, got {}",
                                    v
                                ),
                            });
                        }
                        MalformedRecordPolicy::LenientSkip => {
                            self.records_malformed = self.records_malformed.saturating_add(1);
                            continue;
                        }
                    },
                },
                _ => infer_curriculum_order(&curriculum_tier),
            };

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
                .or_else(|| {
                    val.get("metadata")
                        .and_then(|m| m.get("source_repo").or_else(|| m.get("source")))
                })
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let license_spdx = val
                .get("license_spdx")
                .or_else(|| val.get("metadata").and_then(|m| m.get("license_spdx")))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();

            self.records_read = self.records_read.saturating_add(1);
            self.lifetime_records_read = self.lifetime_records_read.saturating_add(1);

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
}

fn infer_curriculum_order(tier: &str) -> u8 {
    match tier.trim().to_lowercase().as_str() {
        "primary" => 1,
        "middle_school" => 2,
        "high_school" => 3,
        "college" | "undergraduate" => 4,
        "degree" => 5,
        "engineering" => 6,
        "phd" => 7,
        "research" => 8,
        _ => 0,
    }
}

fn validate_relative_shard_path(rel: &Path) -> bool {
    if rel.as_os_str().is_empty() || rel.is_absolute() {
        return false;
    }
    for comp in rel.components() {
        match comp {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => return false,
        }
    }
    true
}

/// Expandable, non-fixed dataset reader.
///
/// Streams continuously from an arbitrary, dynamically expanding collection of JSONL
/// shard files with bounded open-file descriptors and zero fixed limit on record counts.
pub struct ExpandableDatasetReader {
    shards: Vec<ShardState>,
    registered_paths: HashSet<PathBuf>,
    watched_dirs: Vec<PathBuf>,
    watched_manifests: Vec<PathBuf>,
    mode: ShardStreamingMode,
    lifecycle: DatasetStreamLifecycle,
    malformed_policy: MalformedRecordPolicy,
    max_open_files: usize,
    access_tick: u64,
    current_shard_idx: usize,
    total_yielded: u64,
    epoch_yielded: u64,
    auto_rewind: bool,
    line_buf: String,
}

impl ExpandableDatasetReader {
    /// Create a new empty expandable reader with bounded file-handle pooling
    /// (≤512 KiB configured file-buffer budget at default `max_open_files = 8`).
    pub fn new(mode: ShardStreamingMode) -> Self {
        Self {
            shards: Vec::new(),
            registered_paths: HashSet::new(),
            watched_dirs: Vec::new(),
            watched_manifests: Vec::new(),
            mode,
            lifecycle: DatasetStreamLifecycle::Finite,
            malformed_policy: MalformedRecordPolicy::Strict,
            max_open_files: DEFAULT_MAX_OPEN_FILES,
            access_tick: 0,
            current_shard_idx: 0,
            total_yielded: 0,
            epoch_yielded: 0,
            auto_rewind: false,
            line_buf: String::with_capacity(4096),
        }
    }

    /// Set auto-rewind for continuous epoch streaming across all shards.
    pub fn with_auto_rewind(mut self, auto_rewind: bool) -> Self {
        self.auto_rewind = auto_rewind;
        self
    }

    /// Configure the stream lifecycle mode (`Finite` vs `LiveAppendWait`).
    pub fn with_lifecycle(mut self, lifecycle: DatasetStreamLifecycle) -> Self {
        self.lifecycle = lifecycle;
        self
    }

    /// Update the stream lifecycle mode dynamically at runtime.
    pub fn set_lifecycle(&mut self, lifecycle: DatasetStreamLifecycle) {
        self.lifecycle = lifecycle;
    }

    /// Returns the configured stream lifecycle mode.
    pub fn lifecycle(&self) -> DatasetStreamLifecycle {
        self.lifecycle
    }

    /// Configure how malformed JSON lines or out-of-range fields are handled.
    pub fn with_malformed_policy(mut self, policy: MalformedRecordPolicy) -> Self {
        self.malformed_policy = policy;
        self
    }

    /// Configure the maximum number of concurrently open shard file descriptors.
    pub fn with_max_open_files(mut self, max_open_files: usize) -> Self {
        self.max_open_files = max_open_files.max(1);
        self
    }

    /// Add a single JSONL shard file to the reader (unsealed by default so appended lines can be read).
    /// Duplicate registrations of the same canonical path are idempotently ignored.
    pub fn add_shard<P: AsRef<Path>>(&mut self, path: P) -> Result<(), ReaderError> {
        let _ = self.add_shard_with_policy(path, false)?;
        Ok(())
    }

    /// Add a single JSONL shard file with explicit sealing policy.
    /// Returns `Ok(true)` if the shard was newly registered, or `Ok(false)` if already registered.
    pub fn add_shard_with_policy<P: AsRef<Path>>(
        &mut self,
        path: P,
        sealed_by_policy: bool,
    ) -> Result<bool, ReaderError> {
        let path_buf = path.as_ref().to_path_buf();
        if !path_buf.exists() || !path_buf.is_file() {
            return Err(ReaderError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Shard file not found: {}", path_buf.display()),
            )));
        }

        let canonical = fs::canonicalize(&path_buf)?;
        if self.registered_paths.contains(&canonical) {
            return Ok(false);
        }

        let state = ShardState::new(path_buf, sealed_by_policy)?;
        self.registered_paths.insert(canonical);
        self.shards.push(state);
        Ok(true)
    }

    /// Add shards declared in a `manifest.json` file in deterministic order.
    ///
    /// Validates that `"shards"` exists and is non-empty, enforces path containment within
    /// the manifest's directory, and deduplicates already-registered shards.
    pub fn add_shards_from_manifest<P: AsRef<Path>>(
        &mut self,
        manifest_path: P,
    ) -> Result<usize, ReaderError> {
        let path = manifest_path.as_ref();
        if !path.exists() || !path.is_file() {
            return Err(ReaderError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Manifest file not found: {}", path.display()),
            )));
        }

        let canonical_manifest = fs::canonicalize(path)?;
        if !self.watched_manifests.contains(&canonical_manifest) {
            self.watched_manifests.push(canonical_manifest);
        }

        self.load_manifest_entries(path)
    }

    fn load_manifest_entries(&mut self, path: &Path) -> Result<usize, ReaderError> {
        let manifest_display = path.display().to_string();
        let bytes = fs::read_to_string(path)?;
        let manifest: Value = serde_json::from_str(&bytes).map_err(|e| ReaderError::Json {
            path: manifest_display.clone(),
            line: 0,
            error: e.to_string(),
        })?;

        let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
        let canonical_base = fs::canonicalize(base_dir)?;

        let shards_val = manifest.get("shards").ok_or_else(|| ReaderError::InvalidManifest {
            path: manifest_display.clone(),
            reason: "missing required 'shards' field".to_string(),
        })?;

        let mut declared: Vec<(String, bool)> = Vec::new();
        if let Some(shards_obj) = shards_val.as_object() {
            if shards_obj.is_empty() {
                return Err(ReaderError::InvalidManifest {
                    path: manifest_display.clone(),
                    reason: "'shards' object cannot be empty".to_string(),
                });
            }
            let mut keys: Vec<String> = shards_obj.keys().cloned().collect();
            keys.sort();
            for key in keys {
                let entry_val = &shards_obj[&key];
                let sealed = entry_val
                    .get("status")
                    .and_then(Value::as_str)
                    .map(|s| s.eq_ignore_ascii_case("sealed"))
                    .or_else(|| entry_val.get("sealed").and_then(Value::as_bool))
                    .unwrap_or(false);
                declared.push((key, sealed));
            }
        } else if let Some(shards_arr) = shards_val.as_array() {
            if shards_arr.is_empty() {
                return Err(ReaderError::InvalidManifest {
                    path: manifest_display.clone(),
                    reason: "'shards' array cannot be empty".to_string(),
                });
            }
            for item in shards_arr {
                if let Some(name) = item.as_str() {
                    declared.push((name.to_string(), false));
                } else if let Some(obj) = item.as_object() {
                    let name = obj
                        .get("path")
                        .or_else(|| obj.get("name"))
                        .or_else(|| obj.get("file"))
                        .and_then(Value::as_str)
                        .ok_or_else(|| ReaderError::InvalidManifest {
                            path: manifest_display.clone(),
                            reason: "array shard object missing 'path'/'name' string".to_string(),
                        })?;
                    let sealed = obj
                        .get("status")
                        .and_then(Value::as_str)
                        .map(|s| s.eq_ignore_ascii_case("sealed"))
                        .or_else(|| obj.get("sealed").and_then(Value::as_bool))
                        .unwrap_or(false);
                    declared.push((name.to_string(), sealed));
                } else {
                    return Err(ReaderError::InvalidManifest {
                        path: manifest_display.clone(),
                        reason: "invalid shard entry in 'shards' array".to_string(),
                    });
                }
            }
        } else {
            return Err(ReaderError::InvalidManifest {
                path: manifest_display,
                reason: "'shards' field must be a JSON object or array".to_string(),
            });
        }

        let mut added = 0;
        for (name, sealed) in declared {
            let rel_path = Path::new(&name);
            if !validate_relative_shard_path(rel_path) {
                return Err(ReaderError::PathTraversal {
                    manifest: manifest_display,
                    shard: name,
                });
            }

            let shard_path = base_dir.join(rel_path);
            if !shard_path.exists() || !shard_path.is_file() {
                return Err(ReaderError::InvalidManifest {
                    path: manifest_display,
                    reason: format!("declared shard file not found: {}", shard_path.display()),
                });
            }

            let canonical_shard = fs::canonicalize(&shard_path)?;
            if !canonical_shard.starts_with(&canonical_base) {
                return Err(ReaderError::PathTraversal {
                    manifest: manifest_display,
                    shard: name,
                });
            }

            if self.add_shard_with_policy(shard_path, sealed)? {
                added += 1;
            }
        }

        Ok(added)
    }

    /// Add all `.jsonl` shard files found in a directory and register the directory for
    /// dynamic expansion rescans.
    /// If `manifest.json` exists in the directory, strictly loads manifest-declared shards.
    pub fn add_shards_from_dir<P: AsRef<Path>>(&mut self, dir: P) -> Result<usize, ReaderError> {
        let dir_path = dir.as_ref();
        if !dir_path.exists() || !dir_path.is_dir() {
            return Ok(0);
        }

        let canonical_dir = fs::canonicalize(dir_path)?;
        if !self.watched_dirs.contains(&canonical_dir) {
            self.watched_dirs.push(canonical_dir.clone());
        }

        self.scan_dir_entries(&canonical_dir)
    }

    fn scan_dir_entries(&mut self, dir_path: &Path) -> Result<usize, ReaderError> {
        let manifest_candidate = dir_path.join("manifest.json");
        if manifest_candidate.exists() && manifest_candidate.is_file() {
            return self.add_shards_from_manifest(manifest_candidate);
        }

        let mut entries: Vec<PathBuf> = fs::read_dir(dir_path)?
            .filter_map(|e| e.ok().map(|entry| entry.path()))
            .filter(|p| {
                if !p.is_file() || p.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
                    return false;
                }
                if let Some(fname) = p.file_name().and_then(|n| n.to_str()) {
                    // Ignore hidden or in-flight temporary producer files
                    if fname.starts_with('.') || fname.contains(".tmp") || fname.contains(".part") {
                        return false;
                    }
                }
                true
            })
            .collect();
        entries.sort();

        let mut added = 0;
        for path in entries {
            if self.add_shard_with_policy(path, false)? {
                added += 1;
            }
        }

        Ok(added)
    }

    /// Rescans watched directories and manifests for newly created shards and probes
    /// unsealed shards at EOF for newly appended bytes.
    ///
    /// Returns the number of newly discovered or reactivated shards.
    pub fn refresh_dynamic_sources(&mut self) -> Result<usize, ReaderError> {
        let mut newly_active = 0;

        let dirs = self.watched_dirs.clone();
        for dir in dirs {
            if dir.exists() && dir.is_dir() {
                newly_active += self.scan_dir_entries(&dir)?;
            }
        }

        let manifests = self.watched_manifests.clone();
        for manifest in manifests {
            if manifest.exists() && manifest.is_file() {
                newly_active += self.load_manifest_entries(&manifest)?;
            }
        }

        let mut first_reactivated_idx: Option<usize> = None;
        for (idx, shard) in self.shards.iter_mut().enumerate() {
            if shard.probe_growth()? {
                newly_active += 1;
                if first_reactivated_idx.is_none() {
                    first_reactivated_idx = Some(idx);
                }
            }
        }

        if self.mode == ShardStreamingMode::Sequential {
            if let Some(idx) = first_reactivated_idx {
                if idx < self.current_shard_idx {
                    self.current_shard_idx = idx;
                }
            }
        }

        Ok(newly_active)
    }

    /// Total number of shards currently managed.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Total samples successfully yielded across the reader's lifetime (usize for compatibility).
    pub fn total_yielded(&self) -> usize {
        usize::try_from(self.total_yielded).unwrap_or(usize::MAX)
    }

    /// Total samples successfully yielded across the reader's lifetime (`u64`).
    pub fn lifetime_yielded(&self) -> u64 {
        self.total_yielded
    }

    /// Samples yielded in the current pass/epoch (reset on `reset_all()`).
    pub fn epoch_yielded(&self) -> u64 {
        self.epoch_yielded
    }

    /// Number of currently open OS file handles in the bounded reader pool.
    pub fn open_handle_count(&self) -> usize {
        self.shards.iter().filter(|s| s.reader.is_some()).count()
    }

    /// Return a rich telemetry snapshot of shard states and record counters.
    pub fn telemetry(&self) -> ReaderTelemetry {
        let mut active_shards = 0;
        let mut temporarily_eof_shards = 0;
        let mut sealed_shards = 0;
        let mut open_file_handles = 0;
        let mut records_skipped_empty = 0u64;
        let mut records_malformed = 0u64;

        for s in &self.shards {
            match s.availability {
                ShardAvailability::Active => active_shards += 1,
                ShardAvailability::TemporarilyAtEof => temporarily_eof_shards += 1,
                ShardAvailability::Sealed => sealed_shards += 1,
            }
            if s.reader.is_some() {
                open_file_handles += 1;
            }
            records_skipped_empty = records_skipped_empty.saturating_add(s.records_skipped_empty);
            records_malformed = records_malformed.saturating_add(s.records_malformed);
        }

        ReaderTelemetry {
            total_shards: self.shards.len(),
            active_shards,
            temporarily_eof_shards,
            sealed_shards,
            open_file_handles,
            epoch_yielded: self.epoch_yielded,
            lifetime_yielded: self.total_yielded,
            records_skipped_empty,
            records_malformed,
        }
    }

    /// Reset all shards back to byte offset 0 and reset current-epoch counters.
    pub fn reset_all(&mut self) -> Result<(), ReaderError> {
        for s in &mut self.shards {
            s.reset()?;
        }
        self.current_shard_idx = 0;
        self.epoch_yielded = 0;
        Ok(())
    }

    fn acquire_shard_reader(&mut self, target_idx: usize) -> Result<(), ReaderError> {
        self.access_tick = self.access_tick.saturating_add(1);
        let tick = self.access_tick;

        if self.shards[target_idx].reader.is_some() {
            self.shards[target_idx].last_used_tick = tick;
            return Ok(());
        }

        let open_count = self.open_handle_count();
        if open_count >= self.max_open_files {
            if let Some((evict_idx, _)) = self
                .shards
                .iter()
                .enumerate()
                .filter(|(idx, s)| *idx != target_idx && s.reader.is_some())
                .min_by_key(|(_, s)| s.last_used_tick)
            {
                self.shards[evict_idx].close_handle();
            }
        }

        self.shards[target_idx].ensure_open(tick)?;
        Ok(())
    }

    fn maybe_shrink_line_buf(&mut self) {
        if self.line_buf.capacity() > MAX_RETAINED_LINE_CAPACITY {
            self.line_buf = String::with_capacity(4096);
        }
    }

    fn can_auto_rewind(&self) -> bool {
        if !self.auto_rewind || self.shards.is_empty() || self.epoch_yielded == 0 {
            return false;
        }
        if self.lifecycle == DatasetStreamLifecycle::LiveAppendWait {
            // In LiveAppendWait mode, do not rewind while any unsealed shard is waiting at TemporarilyAtEof
            let any_unsealed_waiting = self
                .shards
                .iter()
                .any(|s| s.availability == ShardAvailability::TemporarilyAtEof && !s.sealed_by_policy);
            if any_unsealed_waiting {
                return false;
            }
        }
        true
    }

    /// Pull the next training sample.
    ///
    /// Depending on `mode`:
    /// - Sequential: streams one shard to EOF before advancing to the next.
    /// - Interleaved: round-robins across all active shards for balanced multi-domain representation.
    pub fn next_sample(&mut self) -> Result<Option<TrainingSample>, ReaderError> {
        if self.shards.is_empty() {
            self.refresh_dynamic_sources()?;
            if self.shards.is_empty() {
                return Err(ReaderError::NoShards);
            }
        }

        let res = match self.mode {
            ShardStreamingMode::Sequential => self.next_sequential(),
            ShardStreamingMode::Interleaved => self.next_interleaved(),
        };
        self.maybe_shrink_line_buf();
        res
    }

    fn poll_sequential_pass(&mut self) -> Result<Option<TrainingSample>, ReaderError> {
        while self.current_shard_idx < self.shards.len() {
            let idx = self.current_shard_idx;
            if self.shards[idx].availability == ShardAvailability::Sealed {
                self.current_shard_idx += 1;
                continue;
            }
            if self.shards[idx].availability == ShardAvailability::TemporarilyAtEof
                && !self.shards[idx].probe_growth()?
            {
                self.current_shard_idx += 1;
                continue;
            }

            self.acquire_shard_reader(idx)?;
            let policy = self.malformed_policy;
            let mut line_buf = std::mem::take(&mut self.line_buf);
            let sample_res = self.shards[idx].next_sample(policy, &mut line_buf);
            self.line_buf = line_buf;

            if let Some(sample) = sample_res? {
                self.total_yielded = self.total_yielded.saturating_add(1);
                self.epoch_yielded = self.epoch_yielded.saturating_add(1);
                return Ok(Some(sample));
            }
            self.current_shard_idx += 1;
        }
        Ok(None)
    }

    fn next_sequential(&mut self) -> Result<Option<TrainingSample>, ReaderError> {
        if let Some(sample) = self.poll_sequential_pass()? {
            return Ok(Some(sample));
        }

        // Before rewinding or terminating, check if dynamic directories or growing files added data.
        if self.refresh_dynamic_sources()? > 0 {
            if let Some(sample) = self.poll_sequential_pass()? {
                return Ok(Some(sample));
            }
        }

        // Iterative single-rewind guard: only rewind if allowed by lifecycle policy and epoch yielded > 0.
        if self.can_auto_rewind() {
            self.reset_all()?;
            return self.poll_sequential_pass();
        }

        Ok(None)
    }

    fn poll_interleaved_pass(&mut self) -> Result<Option<TrainingSample>, ReaderError> {
        let total = self.shards.len();
        if total == 0 {
            return Ok(None);
        }
        let mut attempts = 0;

        while attempts < total {
            let idx = self.current_shard_idx % total;
            self.current_shard_idx = (idx + 1) % total;
            attempts += 1;

            if self.shards[idx].availability == ShardAvailability::Sealed {
                continue;
            }
            if self.shards[idx].availability == ShardAvailability::TemporarilyAtEof
                && !self.shards[idx].probe_growth()?
            {
                continue;
            }

            self.acquire_shard_reader(idx)?;
            let policy = self.malformed_policy;
            let mut line_buf = std::mem::take(&mut self.line_buf);
            let sample_res = self.shards[idx].next_sample(policy, &mut line_buf);
            self.line_buf = line_buf;

            if let Some(sample) = sample_res? {
                self.total_yielded = self.total_yielded.saturating_add(1);
                self.epoch_yielded = self.epoch_yielded.saturating_add(1);
                return Ok(Some(sample));
            }
        }
        Ok(None)
    }

    fn next_interleaved(&mut self) -> Result<Option<TrainingSample>, ReaderError> {
        if let Some(sample) = self.poll_interleaved_pass()? {
            return Ok(Some(sample));
        }

        // Before rewinding or terminating, check if dynamic directories or growing files added data.
        if self.refresh_dynamic_sources()? > 0 {
            if let Some(sample) = self.poll_interleaved_pass()? {
                return Ok(Some(sample));
            }
        }

        // Iterative single-rewind guard: only rewind if allowed by lifecycle policy and epoch yielded > 0.
        if self.can_auto_rewind() {
            self.reset_all()?;
            return self.poll_interleaved_pass();
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::io::Write;

    fn unique_test_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "tara_reader_{}_{}_{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_expandable_reader_interleaved() {
        let dir = unique_test_dir("interleaved");
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

        assert_eq!(ids, vec!["s1_1", "s2_1", "s1_2", "s2_2"]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_auto_rewind_empty_and_all_invalid_no_infinite_recursion() {
        let dir = unique_test_dir("auto_rewind_empty");
        let empty_shard = dir.join("empty.jsonl");
        let invalid_shard = dir.join("invalid_empty_fields.jsonl");

        File::create(&empty_shard).unwrap();
        fs::write(
            &invalid_shard,
            "   \n{\"id\":\"bad1\",\"input\":\"\",\"output\":\"\"}\n{\"id\":\"bad2\",\"input\":\"only_in\",\"output\":\"   \"}\n",
        )
        .unwrap();

        for mode in [ShardStreamingMode::Sequential, ShardStreamingMode::Interleaved] {
            let mut reader = ExpandableDatasetReader::new(mode).with_auto_rewind(true);
            reader.add_shard(&empty_shard).unwrap();
            reader.add_shard(&invalid_shard).unwrap();

            // Must return Ok(None) immediately without stack overflow or infinite loop
            let res = reader.next_sample().unwrap();
            assert!(res.is_none());
            assert_eq!(reader.total_yielded(), 0);
            assert_eq!(reader.epoch_yielded(), 0);
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_auto_rewind_restarts_non_empty_and_resets_epoch_counter() {
        let dir = unique_test_dir("auto_rewind_valid");
        let shard = dir.join("shard.jsonl");
        fs::write(
            &shard,
            "{\"id\":\"r1\",\"input\":\"q1\",\"output\":\"a1\"}\n{\"id\":\"r2\",\"input\":\"q2\",\"output\":\"a2\"}\n",
        )
        .unwrap();

        let mut reader =
            ExpandableDatasetReader::new(ShardStreamingMode::Sequential).with_auto_rewind(true);
        reader.add_shard(&shard).unwrap();

        let s1 = reader.next_sample().unwrap().unwrap();
        let s2 = reader.next_sample().unwrap().unwrap();
        assert_eq!(s1.id, "r1");
        assert_eq!(s2.id, "r2");
        assert_eq!(reader.epoch_yielded(), 2);
        assert_eq!(reader.lifetime_yielded(), 2);

        // Next call triggers auto_rewind, resetting epoch_yielded to 0 and then yielding r1 (epoch_yielded = 1)
        let s3 = reader.next_sample().unwrap().unwrap();
        assert_eq!(s3.id, "r1");
        assert_eq!(reader.epoch_yielded(), 1);
        assert_eq!(reader.lifetime_yielded(), 3);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_growing_shard_after_eof_and_dynamic_dir_expansion() {
        let dir = unique_test_dir("growing_shard");
        let shard1 = dir.join("shard1.jsonl");
        fs::write(&shard1, "{\"id\":\"s1_1\",\"input\":\"in1\",\"output\":\"out1\"}\n").unwrap();

        let mut reader = ExpandableDatasetReader::new(ShardStreamingMode::Interleaved);
        assert_eq!(reader.add_shards_from_dir(&dir).unwrap(), 1);

        let first = reader.next_sample().unwrap().unwrap();
        assert_eq!(first.id, "s1_1");
        assert!(reader.next_sample().unwrap().is_none());
        assert_eq!(reader.telemetry().temporarily_eof_shards, 1);

        // Append a new line to shard1 after it hit EOF
        {
            let mut f = OpenOptions::new().append(true).open(&shard1).unwrap();
            writeln!(f, "{{\"id\":\"s1_2\",\"input\":\"in2\",\"output\":\"out2\"}}").unwrap();
            f.flush().unwrap();
        }

        // Also create a brand new shard2.jsonl in the watched directory
        let shard2 = dir.join("shard2.jsonl");
        fs::write(&shard2, "{\"id\":\"s2_1\",\"input\":\"in3\",\"output\":\"out3\"}\n").unwrap();

        // Reader must automatically detect both the appended record in shard1 and the new shard2.jsonl!
        let mut subsequent_ids = Vec::new();
        while let Some(s) = reader.next_sample().unwrap() {
            subsequent_ids.push(s.id);
        }
        subsequent_ids.sort();
        assert_eq!(subsequent_ids, vec!["s1_2", "s2_1"]);
        assert_eq!(reader.shard_count(), 2);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_duplicate_shard_registration_prevented() {
        let dir = unique_test_dir("dedup_shards");
        let shard1 = dir.join("shard1.jsonl");
        fs::write(&shard1, "{\"id\":\"s1\",\"input\":\"in\",\"output\":\"out\"}\n").unwrap();

        let mut reader = ExpandableDatasetReader::new(ShardStreamingMode::Sequential);
        assert_eq!(reader.add_shards_from_dir(&dir).unwrap(), 1);
        assert_eq!(reader.add_shards_from_dir(&dir).unwrap(), 0);
        reader.add_shard(&shard1).unwrap();
        assert_eq!(reader.shard_count(), 1);

        let s = reader.next_sample().unwrap().unwrap();
        assert_eq!(s.id, "s1");
        assert!(reader.next_sample().unwrap().is_none());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_manifest_validation_and_path_traversal_rejection() {
        let dir = unique_test_dir("manifest_security");
        let outside_shard = dir.parent().unwrap().join(format!(
            "outside_secret_{}.jsonl",
            std::process::id()
        ));
        fs::write(&outside_shard, "{\"id\":\"sec\",\"input\":\"a\",\"output\":\"b\"}\n").unwrap();

        // 1. Manifest missing "shards" key must error
        let bad_manifest_missing = dir.join("bad_missing.json");
        fs::write(&bad_manifest_missing, "{\"version\": 1}").unwrap();
        let mut reader = ExpandableDatasetReader::new(ShardStreamingMode::Sequential);
        assert!(matches!(
            reader.add_shards_from_manifest(&bad_manifest_missing),
            Err(ReaderError::InvalidManifest { .. })
        ));

        // 2. Manifest with empty "shards" must error
        let bad_manifest_empty = dir.join("bad_empty.json");
        fs::write(&bad_manifest_empty, "{\"shards\": {}}").unwrap();
        assert!(matches!(
            reader.add_shards_from_manifest(&bad_manifest_empty),
            Err(ReaderError::InvalidManifest { .. })
        ));

        // 3. Manifest attempting "../" path traversal must be rejected
        let traversal_manifest = dir.join("traversal.json");
        let rel_traversal = format!(
            "../{}",
            outside_shard.file_name().unwrap().to_string_lossy()
        );
        fs::write(
            &traversal_manifest,
            format!("{{\"shards\": {{\"{}\": {{}}}}}}", rel_traversal),
        )
        .unwrap();
        assert!(matches!(
            reader.add_shards_from_manifest(&traversal_manifest),
            Err(ReaderError::PathTraversal { .. })
        ));

        let _ = fs::remove_file(&outside_shard);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_curriculum_order_overflow_rejected_and_lenient_policy() {
        let dir = unique_test_dir("curriculum_order_and_policy");
        let shard = dir.join("shard.jsonl");
        fs::write(
            &shard,
            concat!(
                "{\"id\":\"bad_order\",\"input\":\"q1\",\"output\":\"a1\",\"curriculum_order\":300}\n",
                "{malformed json line\n",
                "{\"id\":\"good\",\"input\":\"q2\",\"output\":\"a2\",\"curriculum_order\":6}\n"
            ),
        )
        .unwrap();

        // Strict mode rejects curriculum_order > 255
        let mut strict_reader = ExpandableDatasetReader::new(ShardStreamingMode::Sequential);
        strict_reader.add_shard(&shard).unwrap();
        assert!(matches!(
            strict_reader.next_sample(),
            Err(ReaderError::InvalidRecord { line: 1, .. })
        ));

        // LenientSkip mode skips both invalid curriculum_order and malformed JSON, yielding the good record
        let mut lenient_reader = ExpandableDatasetReader::new(ShardStreamingMode::Sequential)
            .with_malformed_policy(MalformedRecordPolicy::LenientSkip);
        lenient_reader.add_shard(&shard).unwrap();
        let sample = lenient_reader.next_sample().unwrap().unwrap();
        assert_eq!(sample.id, "good");
        assert_eq!(sample.curriculum_order, 6);
        assert!(lenient_reader.next_sample().unwrap().is_none());
        let tel = lenient_reader.telemetry();
        assert_eq!(tel.records_malformed, 2);
        assert_eq!(tel.lifetime_yielded, 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_zero_shard_reader_returns_no_shards_error_and_bounded_fd_pool() {
        let mut empty_reader = ExpandableDatasetReader::new(ShardStreamingMode::Sequential);
        assert!(matches!(empty_reader.next_sample(), Err(ReaderError::NoShards)));

        let dir = unique_test_dir("bounded_fd_pool");
        let mut reader = ExpandableDatasetReader::new(ShardStreamingMode::Interleaved)
            .with_max_open_files(3);

        for i in 0..10 {
            let p = dir.join(format!("shard_{:02}.jsonl", i));
            fs::write(
                &p,
                format!(
                    "{{\"id\":\"s{}_1\",\"input\":\"in\",\"output\":\"out\"}}\n{{\"id\":\"s{}_2\",\"input\":\"in\",\"output\":\"out\"}}\n",
                    i, i
                ),
            )
            .unwrap();
            reader.add_shard(&p).unwrap();
        }

        // Lazy open: 0 handles open before first read
        assert_eq!(reader.open_handle_count(), 0);

        let mut count = 0;
        while let Some(_sample) = reader.next_sample().unwrap() {
            assert!(
                reader.open_handle_count() <= 3,
                "Open file handles ({}) exceeded cap of 3",
                reader.open_handle_count()
            );
            count += 1;
        }
        assert_eq!(count, 20);

        let _ = fs::remove_dir_all(&dir);
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
        let temp_dir = unique_test_dir("tokenized_chatml");
        let shard_path = temp_dir.join("tokenized_shard.jsonl");

        let content = format!(
            "{}\n{}\n",
            r#"{"id":"rec_tok","formatted_input":"<|im_start|>user\nSolve 2+2<|im_end|>\n<|im_start|>assistant\n","formatted_target":"4<|im_end|>","metadata":{"input":"Solve 2+2","output":"4","domain":"mathematics","curriculum_tier":"primary","license_spdx":"CC-BY-4.0"}}"#,
            r#"{"id":"rec_inst","instruction":"Summarize text","response":"Summary here","metadata":{"domain":"science","curriculum_tier":"college"}}"#,
        );

        std::fs::write(&shard_path, content).unwrap();

        let mut reader = ExpandableDatasetReader::new(ShardStreamingMode::Sequential);
        reader.add_shard(&shard_path).unwrap();

        let s1 = reader.next_sample().unwrap().expect("sample 1 must exist");
        assert_eq!(s1.id, "rec_tok");
        assert_eq!(
            s1.input,
            "<|im_start|>user\nSolve 2+2<|im_end|>\n<|im_start|>assistant\n"
        );
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

    #[test]
    fn test_partial_final_jsonl_line_append_completion_yields_exact_record() {
        let dir = unique_test_dir("partial_line_append");
        let shard = dir.join("growing.jsonl");

        // Write 1 complete line + 1 partial incomplete JSON line (no trailing '\n')
        {
            let mut f = File::create(&shard).unwrap();
            write!(
                f,
                "{{\"id\":\"full_1\",\"input\":\"hello\",\"output\":\"world\"}}\n{{\"id\":\"partial_2\",\"input\":\"half\",\"out"
            )
            .unwrap();
            f.flush().unwrap();
        }

        // Even in Strict mode, reader must NOT fail on the incomplete final line or advance offset past it
        let mut reader = ExpandableDatasetReader::new(ShardStreamingMode::Sequential)
            .with_malformed_policy(MalformedRecordPolicy::Strict);
        reader.add_shard(&shard).unwrap();

        let s1 = reader.next_sample().unwrap().unwrap();
        assert_eq!(s1.id, "full_1");

        // Next read hits the incomplete final line -> returns Ok(None) and enters TemporarilyAtEof
        assert!(reader.next_sample().unwrap().is_none());
        assert_eq!(reader.telemetry().temporarily_eof_shards, 1);
        assert_eq!(reader.lifetime_yielded(), 1);

        // Producer now finishes writing the second half of the JSONL line + '\n'
        {
            let mut f = OpenOptions::new().append(true).open(&shard).unwrap();
            write!(f, "put\":\"completed value\"}}\n").unwrap();
            f.flush().unwrap();
        }

        // Reader resumes from the exact uncommitted start offset of line 2 and yields the complete record
        let s2 = reader.next_sample().unwrap().unwrap();
        assert_eq!(s2.id, "partial_2");
        assert_eq!(s2.input, "half");
        assert_eq!(s2.output, "completed value");
        assert_eq!(reader.lifetime_yielded(), 2);
        assert!(reader.next_sample().unwrap().is_none());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_large_100kib_single_record_and_line_buf_shrinking() {
        let dir = unique_test_dir("large_100kib");
        let shard = dir.join("large.jsonl");

        let large_payload = "x".repeat(120 * 1024); // 120 KiB > 64 KiB buffer capacity
        {
            let mut f = File::create(&shard).unwrap();
            writeln!(
                f,
                "{{\"id\":\"big_rec\",\"input\":\"prompt\",\"output\":\"{}\"}}",
                large_payload
            )
            .unwrap();
            writeln!(
                f,
                "{{\"id\":\"small_rec\",\"input\":\"p2\",\"output\":\"o2\"}}"
            )
            .unwrap();
        }

        let mut reader = ExpandableDatasetReader::new(ShardStreamingMode::Sequential);
        reader.add_shard(&shard).unwrap();

        let s1 = reader.next_sample().unwrap().unwrap();
        assert_eq!(s1.id, "big_rec");
        assert_eq!(s1.output.len(), 120 * 1024);
        // After yielding a >64 KiB line, maybe_shrink_line_buf() must reclaim the oversized buffer
        assert!(
            reader.line_buf.capacity() <= MAX_RETAINED_LINE_CAPACITY,
            "Expected line_buf capacity ({}) <= {}",
            reader.line_buf.capacity(),
            MAX_RETAINED_LINE_CAPACITY
        );

        let s2 = reader.next_sample().unwrap().unwrap();
        assert_eq!(s2.id, "small_rec");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_live_open_shard_vs_finite_auto_rewind_policy() {
        let dir = unique_test_dir("live_vs_finite");
        let shard = dir.join("live.jsonl");
        fs::write(
            &shard,
            "{\"id\":\"rec1\",\"input\":\"q1\",\"output\":\"a1\"}\n",
        )
        .unwrap();

        // In LiveAppendWait mode, even if auto_rewind is true, reaching TemporarilyAtEof on an
        // unsealed shard MUST NOT rewind and replay rec1; it must wait for new appends.
        let mut live_reader = ExpandableDatasetReader::new(ShardStreamingMode::Sequential)
            .with_auto_rewind(true)
            .with_lifecycle(DatasetStreamLifecycle::LiveAppendWait);
        live_reader.add_shard(&shard).unwrap();

        let first = live_reader.next_sample().unwrap().unwrap();
        assert_eq!(first.id, "rec1");
        assert!(
            live_reader.next_sample().unwrap().is_none(),
            "LiveAppendWait mode must return None at TemporarilyAtEof instead of replaying old epoch"
        );

        // Append rec2 and verify live_reader yields rec2 without replaying rec1
        {
            let mut f = OpenOptions::new().append(true).open(&shard).unwrap();
            writeln!(f, "{{\"id\":\"rec2\",\"input\":\"q2\",\"output\":\"a2\"}}").unwrap();
            f.flush().unwrap();
        }
        let second = live_reader.next_sample().unwrap().unwrap();
        assert_eq!(second.id, "rec2");
        assert_eq!(live_reader.lifetime_yielded(), 2);

        // Switching lifecycle to Finite allows epoch rewind once all current records are consumed
        live_reader.set_lifecycle(DatasetStreamLifecycle::Finite);
        let rewound = live_reader.next_sample().unwrap().unwrap();
        assert_eq!(rewound.id, "rec1");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_new_shard_discovered_while_producer_writing_tmp_and_partial() {
        let dir = unique_test_dir("producer_race");
        let shard1 = dir.join("shard_01.jsonl");
        fs::write(
            &shard1,
            "{\"id\":\"s1\",\"input\":\"in1\",\"output\":\"out1\"}\n",
        )
        .unwrap();

        let mut reader = ExpandableDatasetReader::new(ShardStreamingMode::Sequential);
        assert_eq!(reader.add_shards_from_dir(&dir).unwrap(), 1);
        assert_eq!(reader.next_sample().unwrap().unwrap().id, "s1");

        // Producer creates a temporary file (.tmp.jsonl) and a newly visible shard_02.jsonl with a partial line
        let tmp_file = dir.join(".tmp_shard_03.jsonl");
        fs::write(&tmp_file, "{corrupt in-flight tmp").unwrap();
        let shard2 = dir.join("shard_02.jsonl");
        fs::write(&shard2, "{\"id\":\"s2\",\"input\":\"in2\"").unwrap(); // No newline yet!

        // Reader scans directory: ignores .tmp_shard_03.jsonl, registers shard_02.jsonl,
        // and holds back the incomplete line without erroring
        assert!(reader.next_sample().unwrap().is_none());
        assert_eq!(reader.shard_count(), 2);

        // Producer completes shard_02.jsonl and atomically renames .tmp_shard_03.jsonl -> shard_03.jsonl
        {
            let mut f = OpenOptions::new().append(true).open(&shard2).unwrap();
            write!(f, ",\"output\":\"out2\"}}\n").unwrap();
            f.flush().unwrap();
        }
        let shard3 = dir.join("shard_03.jsonl");
        fs::write(
            &tmp_file,
            "{\"id\":\"s3\",\"input\":\"in3\",\"output\":\"out3\"}\n",
        )
        .unwrap();
        fs::rename(&tmp_file, &shard3).unwrap();

        let a = reader.next_sample().unwrap().unwrap();
        let b = reader.next_sample().unwrap().unwrap();
        assert_eq!(a.id, "s2");
        assert_eq!(b.id, "s3");
        assert!(reader.next_sample().unwrap().is_none());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_truncated_or_replaced_watched_shard_detected() {
        let dir = unique_test_dir("truncated_shard");
        let shard = dir.join("shard.jsonl");
        fs::write(
            &shard,
            "{\"id\":\"s1\",\"input\":\"long_input_string_1\",\"output\":\"long_output_string_1\"}\n{\"id\":\"s2\",\"input\":\"long_input_string_2\",\"output\":\"long_output_string_2\"}\n",
        )
        .unwrap();

        let mut reader = ExpandableDatasetReader::new(ShardStreamingMode::Sequential);
        reader.add_shard(&shard).unwrap();

        assert_eq!(reader.next_sample().unwrap().unwrap().id, "s1");
        assert_eq!(reader.next_sample().unwrap().unwrap().id, "s2");
        assert!(reader.next_sample().unwrap().is_none());

        // Producer accidentally truncates the file to a smaller size than current_offset
        fs::write(&shard, "{\"id\":\"short\"}\n").unwrap();

        let err = reader.next_sample().unwrap_err();
        assert!(
            matches!(err, ReaderError::ShardTruncatedOrReplaced { .. }),
            "Expected ShardTruncatedOrReplaced error, got: {:?}",
            err
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
