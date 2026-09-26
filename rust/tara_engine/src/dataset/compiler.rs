//! Dynamic dataset compiler: scans source directories, scrubs secrets,
//! and compiles training pairs into JSONL.

use std::collections::HashSet;
use std::fs;
use std::path::Path;
use serde_json::{json, Value};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompilerError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Scrubs sensitive patterns from text before writing to a training dataset.
pub struct SecretScrubber;

impl SecretScrubber {
    const SENSITIVE_PATTERNS: &'static [&'static str] = &[
        "password", "passwd", "secret", "token", "api_key", "private_key",
        "bearer", "authorization", "credential", "passphrase",
    ];

    /// Return a sanitised copy of `text` with sensitive values redacted.
    pub fn scrub(text: &str) -> String {
        let mut out = text.to_string();
        // Redact key=value patterns for sensitive keys
        for pat in Self::SENSITIVE_PATTERNS {
            let pat_lc = pat.to_lowercase();
            let mut search_start = 0;
            while let Some(pos) = out[search_start..].to_lowercase().find(&pat_lc) {
                let abs_pos = search_start + pos;
                // Find the value part after '=' or ':'
                let after = &out[abs_pos + pat.len()..];
                let val_start = after.find(|c: char| c == '=' || c == ':')
                    .map(|p| abs_pos + pat.len() + p + 1);
                if let Some(vs) = val_start {
                    let val_end = out[vs..].find(|c: char| c == ' ' || c == '\n' || c == '"' || c == '\'')
                        .map(|e| vs + e)
                        .unwrap_or(out.len());
                    out.replace_range(vs..val_end, "[REDACTED]");
                    search_start = vs;
                } else {
                    search_start = abs_pos + 1;
                }
            }
        }
        out
    }
}

/// Compiles training datasets from a source directory.
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

    /// Compile all JSONL files in `source_dir` into a single output JSONL.
    ///
    /// Returns the number of samples compiled.
    pub fn compile(&self) -> Result<usize, CompilerError> {
        let mut samples = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        if !Path::new(&self.source_dir).exists() {
            fs::create_dir_all(&self.source_dir)?;
            return Ok(0);
        }

        self.scan_directory(&self.source_dir, &mut samples, &mut seen)?;

        if let Some(parent) = Path::new(&self.output_path).parent() {
            fs::create_dir_all(parent)?;
        }

        let mut output_lines = Vec::new();
        for sample in &samples {
            let scrubbed = json!({
                "input": SecretScrubber::scrub(sample["input"].as_str().unwrap_or("")),
                "output": SecretScrubber::scrub(sample["output"].as_str().unwrap_or(""))
            });
            output_lines.push(serde_json::to_string(&scrubbed)?);
        }

        fs::write(&self.output_path, output_lines.join("\n"))?;
        Ok(samples.len())
    }

    fn scan_directory(
        &self,
        dir: &str,
        samples: &mut Vec<Value>,
        seen: &mut HashSet<String>,
    ) -> Result<(), CompilerError> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                let _ = self.scan_directory(&path.to_string_lossy(), samples, seen);
            } else if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                let raw = fs::read_to_string(&path)?;
                for line in raw.lines() {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if seen.contains(trimmed) {
                        continue;
                    }
                    seen.insert(trimmed.to_string());
                    if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
                        let input = val.get("input").or(val.get("prompt"))
                            .and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let output = val.get("output").or(val.get("completion"))
                            .and_then(|v| v.as_str()).unwrap_or("").to_string();
                        if !input.is_empty() {
                            samples.push(json!({"input": input, "output": output}));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
