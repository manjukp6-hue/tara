//! # TARA Hugging Face Hub & LFS Adapter (100% Native Rust)
//!
//! Authoritative single source of truth for Hugging Face Hub and Git-LFS interactions.
//! Consolidates token resolution, LFS batch negotiations, S3 blob uploads, upload verification,
//! and atomic commit API requests across dataset and model pipelines.
//!
//! 100% Native Rust, zero Python tools, zero synthetics, zero mocks.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread::sleep;
use std::time::Duration;

/// Target repository type on Hugging Face.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HfRepoType {
    Dataset,
    Model,
}

impl HfRepoType {
    /// Return the LFS batch endpoint URL for this repo type.
    pub fn lfs_batch_url(&self, repo_id: &str) -> String {
        match self {
            Self::Dataset => {
                format!(
                    "https://huggingface.co/datasets/{}.git/info/lfs/objects/batch",
                    repo_id
                )
            }
            Self::Model => {
                format!(
                    "https://huggingface.co/{}.git/info/lfs/objects/batch",
                    repo_id
                )
            }
        }
    }

    /// Return the commit API endpoint URL for this repo type.
    pub fn commit_url(&self, repo_id: &str, branch: &str) -> String {
        match self {
            Self::Dataset => {
                format!(
                    "https://huggingface.co/api/datasets/{}/commit/{}",
                    repo_id, branch
                )
            }
            Self::Model => {
                format!(
                    "https://huggingface.co/api/models/{}/commit/{}",
                    repo_id, branch
                )
            }
        }
    }

    /// Return the tree API endpoint URL for this repo type and prefix.
    pub fn tree_url(&self, repo_id: &str, branch: &str, prefix: &str) -> String {
        let clean = prefix.trim_matches('/');
        match self {
            Self::Dataset => {
                if clean.is_empty() {
                    format!(
                        "https://huggingface.co/api/datasets/{}/tree/{}",
                        repo_id, branch
                    )
                } else {
                    format!(
                        "https://huggingface.co/api/datasets/{}/tree/{}/{}",
                        repo_id, branch, clean
                    )
                }
            }
            Self::Model => {
                if clean.is_empty() {
                    format!(
                        "https://huggingface.co/api/models/{}/tree/{}",
                        repo_id, branch
                    )
                } else {
                    format!(
                        "https://huggingface.co/api/models/{}/tree/{}/{}",
                        repo_id, branch, clean
                    )
                }
            }
        }
    }
}

/// Request object specification for Git-LFS batch negotiation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LfsRequestObject {
    pub oid: String,
    pub size: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct LfsBatchRequest {
    operation: String,
    transfers: Vec<String>,
    #[serde(rename = "ref")]
    reference: LfsRef,
    objects: Vec<LfsRequestObject>,
}

#[derive(Debug, Serialize, Deserialize)]
struct LfsRef {
    name: String,
}

/// Response returned by the Hugging Face LFS Batch API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LfsBatchResponse {
    pub transfer: Option<String>,
    pub objects: Vec<LfsResponseObject>,
}

/// Individual object status in the LFS batch response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LfsResponseObject {
    pub oid: String,
    pub size: u64,
    #[serde(default)]
    pub authenticated: bool,
    pub actions: Option<LfsActions>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LfsActions {
    pub upload: Option<LfsActionLink>,
    pub verify: Option<LfsActionLink>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LfsActionLink {
    pub href: String,
    pub header: Option<HashMap<String, String>>,
}

/// Resolves Hugging Face authentication token from environment (`HF_TOKEN`)
/// or queries the Git credential manager for `host=huggingface.co`.
pub fn resolve_hf_token() -> Result<String, String> {
    if let Ok(token) = env::var("HF_TOKEN") {
        let trimmed = token.trim().to_string();
        if !trimmed.is_empty() {
            return Ok(trimmed);
        }
    }

    let mut child = Command::new("git")
        .arg("credential")
        .arg("fill")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Failed to spawn git credential helper: {e}"))?;

    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(b"protocol=https\nhost=huggingface.co\n");
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("Git credential fill wait failed: {e}"))?;

    if output.status.success() {
        let text = String::from_utf8_lossy(&output.stdout);
        for line in text.lines() {
            if let Some(pw) = line.strip_prefix("password=") {
                let trimmed = pw.trim().to_string();
                if !trimmed.is_empty() {
                    return Ok(trimmed);
                }
            }
        }
    }

    Err(
        "Hugging Face API token not found in HF_TOKEN environment variable or Git credential store"
            .into(),
    )
}

/// Computes streaming SHA-256 and byte length of a file on disk.
pub fn compute_sha256_and_size(path: &Path) -> Result<(String, u64), String> {
    let meta = std::fs::metadata(path)
        .map_err(|e| format!("Failed to read metadata for '{}': {e}", path.display()))?;
    let hash = crate::compute256(path)
        .map_err(|e| format!("Failed to compute SHA-256 for '{}': {e}", path.display()))?;
    Ok((hash, meta.len()))
}

/// Cross-platform curl executable binary name.
pub fn curl_bin() -> &'static str {
    if cfg!(windows) {
        "curl.exe"
    } else {
        "curl"
    }
}

/// Queries Hugging Face Git-LFS batch endpoint with configurable retry attempts.
pub fn query_lfs_batch(
    repo_type: HfRepoType,
    repo_id: &str,
    token: &str,
    objects: &[LfsRequestObject],
    max_attempts: usize,
) -> Result<LfsBatchResponse, String> {
    let url = repo_type.lfs_batch_url(repo_id);
    let request_body = LfsBatchRequest {
        operation: "upload".to_string(),
        transfers: vec!["basic".to_string()],
        reference: LfsRef {
            name: "refs/heads/main".to_string(),
        },
        objects: objects.to_vec(),
    };

    let json_bytes = serde_json::to_vec(&request_body)
        .map_err(|e| format!("Failed to serialize LFS batch request: {e}"))?;

    let attempts = max_attempts.max(1);
    for attempt in 1..=attempts {
        let mut cmd = Command::new(curl_bin());
        cmd.args(["-s", "-S", "-f"]);
        #[cfg(windows)]
        cmd.arg("--ssl-no-revoke");
        cmd.args([
            "--retry",
            "5",
            "--retry-delay",
            "3",
            "-X",
            "POST",
            "-H",
            &format!("Authorization: Bearer {}", token),
            "-H",
            "Accept: application/vnd.git-lfs+json",
            "-H",
            "Content-Type: application/vnd.git-lfs+json",
            "--data-binary",
            "@-",
            &url,
        ]);

        let mut child = match cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Failed to spawn curl for LFS batch: {e}. Retrying...");
                sleep(Duration::from_secs(5));
                continue;
            }
        };

        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(&json_bytes);
        }

        if let Ok(output) = child.wait_with_output() {
            if output.status.success() {
                if let Ok(resp) = serde_json::from_slice::<LfsBatchResponse>(&output.stdout) {
                    return Ok(resp);
                }
            }
            let err_msg = String::from_utf8_lossy(&output.stderr);
            eprintln!(
                "LFS batch attempt {}/{} failed: {}. Retrying in 5s...",
                attempt,
                attempts,
                err_msg.trim()
            );
        }
        sleep(Duration::from_secs(5));
    }

    Err(format!(
        "LFS batch request failed after {} attempts",
        attempts
    ))
}

/// Uploads an LFS file to S3 via HTTP PUT with resilient streaming parameters.
pub fn upload_lfs_to_s3(
    file_path: &Path,
    upload_url: &str,
    extra_headers: Option<&HashMap<String, String>>,
) -> Result<(), String> {
    let mut cmd = Command::new(curl_bin());
    cmd.args(["-4", "-f", "-s", "-S"]);
    #[cfg(windows)]
    cmd.arg("--ssl-no-revoke");
    cmd.args([
        "--http1.1",
        "--tcp-nodelay",
        "--keepalive-time",
        "10",
        "--connect-timeout",
        "30",
        "-X",
        "PUT",
        "-T",
    ]);
    cmd.arg(file_path);
    cmd.args(["--speed-limit", "1024", "--speed-time", "60"]);

    if let Some(headers) = extra_headers {
        for (k, v) in headers {
            cmd.arg("-H").arg(format!("{}: {}", k, v));
        }
    }

    cmd.arg(upload_url);

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to execute S3 upload: {e}"))?;

    if !output.status.success() {
        let err_msg = String::from_utf8_lossy(&output.stderr);
        return Err(format!("S3 upload failed for {:?}: {}", file_path, err_msg));
    }

    Ok(())
}

/// Performs completion verification call for an uploaded LFS object.
pub fn verify_lfs_upload(
    verify_url: &str,
    token: &str,
    oid: &str,
    size: u64,
) -> Result<(), String> {
    let body = serde_json::json!({
        "oid": oid,
        "size": size,
    });
    let json_bytes = serde_json::to_vec(&body)
        .map_err(|e| format!("Failed to serialize verify payload: {e}"))?;

    for attempt in 1..=10 {
        let mut cmd = Command::new(curl_bin());
        cmd.args(["-s", "-S", "-f"]);
        #[cfg(windows)]
        cmd.arg("--ssl-no-revoke");
        cmd.args([
            "--retry",
            "3",
            "--retry-delay",
            "2",
            "-X",
            "POST",
            "-H",
            &format!("Authorization: Bearer {}", token),
            "-H",
            "Accept: application/vnd.git-lfs+json",
            "-H",
            "Content-Type: application/vnd.git-lfs+json",
            "--data-binary",
            "@-",
            verify_url,
        ]);
        let mut child = match cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(_) => {
                sleep(Duration::from_secs(3));
                continue;
            }
        };

        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(&json_bytes);
        }

        if let Ok(output) = child.wait_with_output() {
            if output.status.success() {
                return Ok(());
            }
        }
        eprintln!(
            "LFS verification call attempt {}/10 retrying in 3s...",
            attempt
        );
        sleep(Duration::from_secs(3));
    }

    Err("LFS verification call failed after 10 attempts".into())
}

/// Commits LFS objects atomically via Hugging Face Commit API using NDJSON payload.
pub fn commit_lfs_files(
    repo_type: HfRepoType,
    repo_id: &str,
    token: &str,
    summary: &str,
    description: &str,
    files: &[(&str, &str, u64)], // (target_path, oid, size)
) -> Result<(), String> {
    let url = repo_type.commit_url(repo_id, "main");

    let mut ndjson = String::new();
    let header_line = serde_json::json!({
        "key": "header",
        "value": {
            "summary": summary,
            "description": description,
        }
    });
    ndjson.push_str(&header_line.to_string());
    ndjson.push('\n');

    for (target_path, oid, size) in files {
        let lfs_line = serde_json::json!({
            "key": "lfsFile",
            "value": {
                "path": target_path,
                "algo": "sha256",
                "oid": oid,
                "size": size,
            }
        });
        ndjson.push_str(&lfs_line.to_string());
        ndjson.push('\n');
    }

    for attempt in 1..=10 {
        let mut cmd = Command::new(curl_bin());
        cmd.args(["-s", "-S", "-f"]);
        #[cfg(windows)]
        cmd.arg("--ssl-no-revoke");
        cmd.args([
            "--retry",
            "3",
            "--retry-delay",
            "2",
            "-X",
            "POST",
            "-H",
            &format!("Authorization: Bearer {}", token),
            "-H",
            "Content-Type: application/x-ndjson",
            "--data-binary",
            "@-",
            &url,
        ]);
        let mut child = match cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(_) => {
                sleep(Duration::from_secs(3));
                continue;
            }
        };

        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(ndjson.as_bytes());
        }

        if let Ok(output) = child.wait_with_output() {
            if output.status.success() {
                return Ok(());
            }
            let err_msg = String::from_utf8_lossy(&output.stderr);
            eprintln!(
                "Hugging Face commit attempt {}/10 failed: {}. Retrying in 5s...",
                attempt,
                err_msg.trim()
            );
        }
        sleep(Duration::from_secs(5));
    }

    Err("Hugging Face commit failed after 10 attempts".into())
}

/// Remote file entry returned by Hugging Face Tree API.
#[derive(Debug, Clone, Deserialize)]
pub struct RemoteTreeEntry {
    pub path: String,
    pub size: Option<u64>,
}

/// Fetches the remote tree mapping (filename -> size) for a repository at a given prefix.
pub fn fetch_remote_tree(
    repo_type: HfRepoType,
    repo_id: &str,
    token: &str,
    prefix: &str,
) -> Result<HashMap<String, u64>, String> {
    let clean_prefix = prefix.trim_matches('/');
    let url = repo_type.tree_url(repo_id, "main", clean_prefix);

    for attempt in 1..=10 {
        let mut cmd = Command::new(curl_bin());
        cmd.args(["-s", "-S"]);
        #[cfg(windows)]
        cmd.arg("--ssl-no-revoke");
        cmd.args([
            "-w",
            "\n%{http_code}",
            "-H",
            &format!("Authorization: Bearer {}", token),
            &url,
        ]);
        let output = match cmd.output() {
            Ok(o) => o,
            Err(_) => {
                sleep(Duration::from_secs(3));
                continue;
            }
        };

        if output.status.success() {
            let stdout_str = String::from_utf8_lossy(&output.stdout);
            let mut parts = stdout_str.rsplitn(2, '\n');
            let http_code = parts.next().unwrap_or("").trim();
            let body = parts.next().unwrap_or("");

            if http_code == "404" {
                return Ok(HashMap::new());
            }

            if http_code == "200" {
                if let Ok(entries) = serde_json::from_str::<Vec<RemoteTreeEntry>>(body) {
                    let mut map = HashMap::new();
                    for entry in entries {
                        let leaf = Path::new(&entry.path)
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(&entry.path)
                            .to_string();
                        map.insert(leaf, entry.size.unwrap_or(0));
                    }
                    return Ok(map);
                }
            }
        }
        if attempt < 10 {
            sleep(Duration::from_secs(3));
        }
    }

    Ok(HashMap::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curl_bin() {
        let bin = curl_bin();
        #[cfg(windows)]
        assert_eq!(bin, "curl.exe");
        #[cfg(not(windows))]
        assert_eq!(bin, "curl");
    }
}
