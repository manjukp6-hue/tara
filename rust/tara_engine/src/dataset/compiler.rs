//! Dynamic dataset compiler: scans source directories, scrubs secrets,
//! and compiles training pairs into JSONL using memory-efficient streaming.

use serde_json::Value;
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompilerError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid training record at {path}:{line}: {message}")]
    InvalidRecord {
        path: String,
        line: usize,
        message: String,
    },
}

/// Scrubs sensitive patterns from text before writing to a training dataset.
pub struct SecretScrubber;

impl SecretScrubber {
    const SENSITIVE_PATTERNS: &'static [&'static str] = &[
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "private_key",
        "bearer",
        "authorization",
        "credential",
        "passphrase",
    ];

    /// Return a sanitised copy of `text` with sensitive values redacted.
    pub fn scrub(text: &str) -> String {
        let lower = text.to_ascii_lowercase();
        let mut ranges = Vec::<(usize, usize)>::new();
        for key in Self::SENSITIVE_PATTERNS {
            for (start, _) in lower.match_indices(key) {
                let after_key = start + key.len();
                let tail = text.as_bytes();
                let Some(separator) = tail[after_key..]
                    .iter()
                    .take_while(|byte| **byte != b'\n')
                    .position(|byte| matches!(byte, b'=' | b':'))
                else {
                    continue;
                };
                let sep_pos = after_key + separator;
                let mut value_start = sep_pos + 1;
                while tail.get(value_start).is_some_and(u8::is_ascii_whitespace) {
                    value_start += 1;
                }
                if *key == "authorization" && lower[value_start..].starts_with("bearer ") {
                    value_start += "bearer ".len();
                }
                if value_start >= tail.len() {
                    continue;
                }
                let end = match tail[value_start] {
                    b'"' | b'\'' => tail[value_start + 1..]
                        .iter()
                        .position(|byte| *byte == tail[value_start])
                        .map(|offset| value_start + offset + 2)
                        .unwrap_or(tail.len()),
                    _ => {
                        value_start
                            + tail[value_start..]
                                .iter()
                                .position(|byte| {
                                    byte.is_ascii_whitespace() || matches!(byte, b',' | b';')
                                })
                                .unwrap_or(tail.len() - value_start)
                    }
                };
                ranges.push((value_start, end));
            }
        }
        ranges.sort_unstable();
        let mut output = String::with_capacity(text.len());
        let mut copied = 0;
        for (start, end) in ranges {
            if start < copied || !text.is_char_boundary(start) || !text.is_char_boundary(end) {
                continue;
            }
            output.push_str(&text[copied..start]);
            output.push_str("[REDACTED]");
            copied = end;
        }
        output.push_str(&text[copied..]);
        output
    }
}

/// Compiles training datasets from a source directory using streaming I/O.
pub struct DynamicDatasetCompiler {
    pub source_dir: String,
    pub output_path: String,
}

impl DynamicDatasetCompiler {
    /// Create a compiler for `source_dir` that writes to `output_path`.
    pub fn new(source_dir: &str, output_path: &str) -> Self {
        Self {
            source_dir: source_dir.to_string(),
            output_path: output_path.to_string(),
        }
    }

    /// Compile all JSONL files in `source_dir` into a single output JSONL using streaming.
    ///
    /// Memory consumption is O(number of unique hashes), NOT O(dataset text size).
    /// Returns the number of samples compiled.
    pub fn compile(&self) -> Result<usize, CompilerError> {
        self.compile_streaming()
    }

    /// Streaming compilation: writes each scrubbed record directly to disk.
    pub fn compile_streaming(&self) -> Result<usize, CompilerError> {
        let mut seen: HashSet<u64> = HashSet::new();

        if !Path::new(&self.source_dir).exists() {
            fs::create_dir_all(&self.source_dir)?;
            return Ok(0);
        }

        if let Some(parent) = Path::new(&self.output_path).parent() {
            fs::create_dir_all(parent)?;
        }

        let out_file = File::create(&self.output_path)?;
        let mut writer = BufWriter::with_capacity(128 * 1024, out_file);
        let mut count = 0;

        self.scan_directory_streaming(&self.source_dir, &mut writer, &mut seen, &mut count)?;
        writer.flush()?;

        Ok(count)
    }

    fn scan_directory_streaming(
        &self,
        dir: &str,
        writer: &mut BufWriter<File>,
        seen: &mut HashSet<u64>,
        count: &mut usize,
    ) -> Result<(), CompilerError> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let is_output = path == Path::new(&self.output_path)
                || (path.exists()
                    && Path::new(&self.output_path)
                        .canonicalize()
                        .map(|output| {
                            path.canonicalize()
                                .map(|current| current == output)
                                .unwrap_or(false)
                        })
                        .unwrap_or(false));
            if is_output {
                continue;
            }
            if path.is_dir() {
                let dir_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if dir_name == "raw_sources"
                    || dir_name == "archive_superseded"
                    || dir_name == "target"
                    || dir_name == ".git"
                {
                    continue;
                }
                self.scan_directory_streaming(&path.to_string_lossy(), writer, seen, count)?;
            } else if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                let file = File::open(&path)?;
                let reader = BufReader::with_capacity(64 * 1024, file);

                for (line_number, line_res) in reader.lines().enumerate() {
                    let line = line_res?;
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    // Cheap 64-bit FNV hash for in-memory deduplication set
                    let hash = fnv1a_hash(trimmed.as_bytes());
                    if seen.contains(&hash) {
                        continue;
                    }
                    seen.insert(hash);

                    let val = serde_json::from_str::<Value>(trimmed).map_err(|error| {
                        CompilerError::InvalidRecord {
                            path: path.display().to_string(),
                            line: line_number + 1,
                            message: error.to_string(),
                        }
                    })?;

                    let input = val
                        .get("input")
                        .or_else(|| val.get("prompt"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim();
                    let output = val
                        .get("output")
                        .or_else(|| val.get("completion"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim();

                    if input.is_empty() {
                        return Err(CompilerError::InvalidRecord {
                            path: path.display().to_string(),
                            line: line_number + 1,
                            message: "training input/prompt is required".into(),
                        });
                    }
                    if output.is_empty() {
                        return Err(CompilerError::InvalidRecord {
                            path: path.display().to_string(),
                            line: line_number + 1,
                            message: "training output/completion is required".into(),
                        });
                    }

                    let mut out_obj = val.clone();
                    if let Some(m) = out_obj.as_object_mut() {
                        m.insert(
                            "input".to_string(),
                            Value::String(SecretScrubber::scrub(input)),
                        );
                        m.insert(
                            "output".to_string(),
                            Value::String(SecretScrubber::scrub(output)),
                        );
                    }
                    serde_json::to_writer(&mut *writer, &out_obj)?;
                    writer.write_all(b"\n")?;
                    *count += 1;
                }
            }
        }
        Ok(())
    }
}


fn fnv1a_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Compile all datasets in `repo_root/storage/datasets/` into a unified JSONL.
pub fn compile_unified_dataset(repo_root: &str) -> Value {
    let source_dir = format!("{}/storage/datasets", repo_root);
    let output_path = format!("{}/storage/datasets/unified_training.jsonl", repo_root);

    let compiler = DynamicDatasetCompiler::new(&source_dir, &output_path);
    match compiler.compile() {
        Ok(count) => serde_json::json!({
            "status": "SUCCESS",
            "samples_compiled": count,
            "output_path": output_path
        }),
        Err(e) => serde_json::json!({
            "status": "ERROR",
            "error": e.to_string()
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrub_handles_unicode_and_bearer_credentials() {
        let text = "ನಮಸ್ಕಾರ password='秘密' Authorization: Bearer abc123 api_key=def456";
        let redacted = SecretScrubber::scrub(text);
        assert!(redacted.contains("ನಮಸ್ಕಾರ"));
        assert!(!redacted.contains("秘密"));
        assert!(!redacted.contains("abc123"));
        assert!(!redacted.contains("def456"));
    }
}
