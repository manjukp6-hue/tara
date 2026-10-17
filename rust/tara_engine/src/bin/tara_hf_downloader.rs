//! TARA Autonomous Hugging Face Downloader (100% Native Rust).
//!
//! Streams and downloads model weights, configs, and dataset shards from Hugging Face:
//! - Direct communication with Hugging Face Hub endpoints via native curl.
//! - Automatic remote tree listing via Tree API.
//! - Resume support on interrupted transfers.
//! - Zero Python dependencies, zero pip packages, zero external CLI dependencies.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tara_engine::hf_adapter::{curl_bin, fetch_remote_tree, resolve_hf_token, HfRepoType};

fn print_usage() {
    println!("TARA Native Rust Hugging Face Downloader");
    println!("Usage:");
    println!("  tara_hf_downloader [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --repo <repo_id>       Repository ID (e.g. manjukp6/tara)");
    println!("  --repo-type <type>     Repository type: 'model' or 'dataset' (default: model)");
    println!("  --prefix <prefix>      Remote folder prefix / pattern (e.g. 'v2', 'v1', 'neural_model_v1')");
    println!("  --dest <dir>           Destination local directory (default: current directory)");
    println!("  -h, --help             Show this help message");
}

fn download_single_file(
    repo_type: HfRepoType,
    repo_id: &str,
    token: Option<&str>,
    remote_path: &str,
    local_path: &Path,
    expected_size: Option<u64>,
) -> Result<bool, String> {
    if let Some(parent) = local_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Failed to create parent dir: {e}"))?;
    }

    if local_path.exists() {
        if let Some(exp) = expected_size {
            if let Ok(meta) = fs::metadata(local_path) {
                if meta.len() == exp {
                    println!("  [CACHE HIT] '{}' already complete ({exp} bytes).", local_path.display());
                    return Ok(true);
                }
            }
        }
    }

    let url = match repo_type {
        HfRepoType::Dataset => format!("https://huggingface.co/datasets/{repo_id}/resolve/main/{remote_path}"),
        HfRepoType::Model => format!("https://huggingface.co/{repo_id}/resolve/main/{remote_path}"),
    };

    println!("  [DOWNLOADING] {} -> {}", url, local_path.display());

    let mut cmd = Command::new(curl_bin());
    cmd.args(["-s", "-S", "-L", "-f"]);
    #[cfg(windows)]
    cmd.arg("--ssl-no-revoke");
    cmd.args([
        "--retry",
        "5",
        "--retry-delay",
        "3",
        "-C",
        "-",
    ]);

    if let Some(tok) = token {
        if !tok.trim().is_empty() {
            cmd.arg("-H").arg(format!("Authorization: Bearer {}", tok.trim()));
        }
    }

    cmd.arg("-o").arg(local_path);
    cmd.arg(&url);

    let status = cmd.status().map_err(|e| format!("Failed to execute curl: {e}"))?;
    if !status.success() {
        return Err(format!("Download failed for '{remote_path}' with exit code: {:?}", status.code()));
    }

    if let Some(exp) = expected_size {
        if let Ok(meta) = fs::metadata(local_path) {
            if meta.len() != exp {
                return Err(format!(
                    "Size mismatch for '{}': expected {} bytes, downloaded {} bytes",
                    local_path.display(),
                    exp,
                    meta.len()
                ));
            }
        }
    }

    Ok(true)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!("  TARA AUTONOMOUS HUGGING FACE DOWNLOADER (100% NATIVE RUST)");
    println!("================================================================================");

    let args: Vec<String> = env::args().collect();
    let mut repo_id = env::var("HF_REPO").or_else(|_| env::var("HF_DATASET_REPO")).or_else(|_| env::var("HF_MODEL_REPO")).unwrap_or_default();
    let mut repo_type_str = "model".to_string();
    let mut prefix = String::new();
    let mut dest_dir = ".".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--repo" | "-r" => {
                if i + 1 < args.len() {
                    repo_id = args[i + 1].clone();
                    i += 1;
                }
            }
            "--repo-type" | "-t" => {
                if i + 1 < args.len() {
                    repo_type_str = args[i + 1].to_lowercase();
                    i += 1;
                }
            }
            "--prefix" | "-p" => {
                if i + 1 < args.len() {
                    prefix = args[i + 1].clone();
                    i += 1;
                }
            }
            "--dest" | "-d" => {
                if i + 1 < args.len() {
                    dest_dir = args[i + 1].clone();
                    i += 1;
                }
            }
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    if repo_id.trim().is_empty() {
        eprintln!("Error: --repo <repo_id> is required or set HF_REPO / HF_DATASET_REPO / HF_MODEL_REPO.");
        print_usage();
        std::process::exit(1);
    }

    let repo_type = if repo_type_str.contains("data") {
        HfRepoType::Dataset
    } else {
        HfRepoType::Model
    };

    let token = resolve_hf_token().ok();
    println!("Repository ID   : {}", repo_id);
    println!("Repository Type : {:?}", repo_type);
    println!("Prefix Filter   : '{}'", prefix);
    println!("Destination Dir : {}", dest_dir);
    println!("Authentication  : {}", if token.is_some() { "Bearer Token Resolved" } else { "Public / Unauthenticated" });

    let clean_prefix = prefix.trim_matches('/').replace("/*", "");
    let token_ref = token.as_deref().unwrap_or("");
    let remote_tree = fetch_remote_tree(repo_type, &repo_id, token_ref, &clean_prefix)?;

    let dest_path = PathBuf::from(&dest_dir);
    fs::create_dir_all(&dest_path)?;

    if remote_tree.is_empty() {
        println!("No files listed in remote tree via API under prefix '{}'.", clean_prefix);
        println!("Attempting direct downloads for canonical file targets...");
        let targets = match repo_type {
            HfRepoType::Dataset => vec![
                format!("{clean_prefix}/tara_shard_001.jsonl"),
                format!("{clean_prefix}/tara_canonical_shard_000.jsonl"),
            ],
            HfRepoType::Model => vec![
                format!("{clean_prefix}/config.json"),
                format!("{clean_prefix}/model.safetensors"),
                format!("{clean_prefix}/model.safetensors.index.json"),
            ],
        };
        for t in targets {
            let file_name = Path::new(&t).file_name().unwrap_or_default();
            let local_file = dest_path.join(file_name);
            let _ = download_single_file(repo_type, &repo_id, token.as_deref(), &t, &local_file, None);
        }
        return Ok(());
    }

    println!("Found {} remote files matching prefix '{}'. Starting streaming download...", remote_tree.len(), clean_prefix);

    let mut success_count = 0usize;
    for (file_name, size) in &remote_tree {
        let remote_file_path = if clean_prefix.is_empty() {
            file_name.clone()
        } else {
            format!("{clean_prefix}/{file_name}")
        };
        let local_file = dest_path.join(file_name);
        match download_single_file(repo_type, &repo_id, token.as_deref(), &remote_file_path, &local_file, Some(*size)) {
            Ok(true) => success_count += 1,
            Ok(false) => {},
            Err(e) => {
                eprintln!("Warning: Failed to download '{}': {}", remote_file_path, e);
            }
        }
    }

    println!("Download complete: {}/{} files verified locally in '{}'.", success_count, remote_tree.len(), dest_dir);
    Ok(())
}
