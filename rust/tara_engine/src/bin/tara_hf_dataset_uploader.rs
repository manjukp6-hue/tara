//! TARA Autonomous Hugging Face Dataset Uploader (100% Native Rust)
//!
//! Direct-to-Cloud Streaming Uploader:
//! - Discovers canonical Indic pretraining dataset shards in `storage/datasets/main_dataset_v2`.
//! - Strictly excludes download folders and scratch areas.
//! - Dynamically resolves Hugging Face authentication credentials via secure Git credential manager.
//! - Computes runtime cryptographic SHA-256 signatures dynamically from disk.
//! - Communicates directly with Hugging Face Git LFS Batch API and AWS S3 acceleration endpoints.
//! - Streams shard uploads to S3 with verification without local disk duplication.
//! - Performs structured NDJSON commits to the remote Hugging Face repository under the immutable `v2/` hierarchy.
//! - Preserves previous version trees (`v1/`) intact.
//! - Executes exhaustive post-upload verification against the live remote tree.
//! - Indestructible self-healing outer loop: auto-recovers from transient network or DNS drops until 100% complete.

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::thread::sleep;
use std::time::Duration;

use tara_engine::hf_adapter::{
    commit_lfs_files, compute_sha256_and_size, fetch_remote_tree, query_lfs_batch,
    resolve_hf_token, upload_lfs_to_s3, verify_lfs_upload, HfRepoType, LfsRequestObject,
};

fn run_upload_pipeline(repo_id: &str, token: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let root = tara_engine::find_workspace_root()
        .unwrap_or_else(|_| env::current_dir().unwrap_or_default());

    let dataset_dir = root.join("storage").join("datasets").join("main_dataset_v2");
    println!("Target Local Directory : {:?}", dataset_dir);

    if !dataset_dir.exists() {
        return Err(format!("Directory does not exist: {:?}", dataset_dir).into());
    }

    println!("Target HF Repository   : {}", repo_id);
    println!("Target HF Subdirectory : v2/");

    let mut shard_paths = Vec::new();
    for entry in fs::read_dir(&dataset_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "jsonl" || ext == "json" {
                    let filename = path.file_name().unwrap().to_string_lossy().to_string();
                    shard_paths.push((filename, path));
                }
            }
        }
    }

    shard_paths.sort_by(|a, b| a.0.cmp(&b.0));

    let total_bytes: u64 = shard_paths
        .iter()
        .map(|(_, p)| fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .sum();

    println!("Total Canonical Shards : {}", shard_paths.len());
    println!(
        "Total Shard Payload    : {} MB ({:.2} GB)",
        total_bytes / (1024 * 1024),
        total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    if shard_paths.is_empty() {
        return Err("Zero shards found to upload!".into());
    }

    println!("Checking remote repository tree on Hugging Face...");
    let remote_files = fetch_remote_tree(HfRepoType::Dataset, repo_id, token, "v2")
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    println!("Existing Remote Files  : {}", remote_files.len());

    let mut pending_shards = Vec::new();
    for (name, path) in &shard_paths {
        if remote_files.contains_key(name) {
            println!("  [SKIP] Shard already committed on HF: {}", name);
        } else {
            pending_shards.push((name.clone(), path.clone()));
        }
    }

    println!("--------------------------------------------------------------------------------");
    println!("Shards Pending Upload  : {} of {}", pending_shards.len(), shard_paths.len());
    println!("--------------------------------------------------------------------------------");

    if pending_shards.is_empty() {
        println!("All {} shards are already uploaded and committed on Hugging Face!", shard_paths.len());
        return Ok(true);
    }

    let batch_size = 3;
    let total_pending = pending_shards.len();

    for (batch_idx, chunk) in pending_shards.chunks(batch_size).enumerate() {
        let batch_num = batch_idx + 1;
        let total_batches = total_pending.div_ceil(batch_size);
        println!();
        println!("================================================================================");
        println!("Processing Batch {}/{} ({} shards)", batch_num, total_batches, chunk.len());
        println!("================================================================================");

        let mut hash_records: Vec<(String, PathBuf, String, u64)> = Vec::new();
        for (name, path) in chunk {
            print!("Calculating SHA-256 for {} ... ", name);
            std::io::stdout().flush()?;
            let (oid, size) = compute_sha256_and_size(path)
                .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
            println!("OK ({} MB, oid: {:.12}...)", size / (1024 * 1024), oid);
            hash_records.push((name.clone(), path.clone(), oid, size));
        }

        for (name, path, oid, size) in &hash_records {
            let mut attempts = 0;
            let max_attempts = 20;
            loop {
                attempts += 1;
                let single_req = vec![LfsRequestObject {
                    oid: oid.clone(),
                    size: *size,
                }];
                let resp = query_lfs_batch(HfRepoType::Dataset, repo_id, token, &single_req, 20)
                    .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
                let obj = resp.objects.into_iter().next();

                if let Some(obj) = obj {
                    if let Some(ref actions) = obj.actions {
                        if let Some(ref upload_link) = actions.upload {
                            println!("Streaming {} to AWS S3 Acceleration endpoint ({} MB, attempt {}/{})...", name, size / (1024 * 1024), attempts, max_attempts);
                            match upload_lfs_to_s3(path, &upload_link.href, upload_link.header.as_ref()) {
                                Ok(()) => {
                                    println!("  Successfully uploaded {} to S3.", name);
                                    if let Some(ref verify_link) = actions.verify {
                                        println!("Verifying LFS object on Hugging Face...");
                                        verify_lfs_upload(&verify_link.href, token, oid, *size)
                                            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
                                        println!("  LFS object verification confirmed.");
                                    }
                                    break;
                                }
                                Err(e) => {
                                    eprintln!("  Attempt {}/{} failed: {}. Retrying with fresh ticket in 5s...", attempts, max_attempts, e);
                                    if attempts >= max_attempts {
                                        return Err(format!("Max upload attempts exceeded for {}", name).into());
                                    }
                                    sleep(Duration::from_secs(5));
                                }
                            }
                        } else {
                            println!("  Shard {} already exists in LFS store. No upload needed.", name);
                            break;
                        }
                    } else {
                        println!("  Shard {} already exists in LFS store. No upload needed.", name);
                        break;
                    }
                } else {
                    return Err(format!("Empty response from LFS batch endpoint for {}", name).into());
                }
            }
        }

        let commit_items: Vec<(String, &str, u64)> = hash_records
            .iter()
            .map(|(name, _, oid, size)| (format!("v2/{}", name), oid.as_str(), *size))
            .collect();
        let commit_refs: Vec<(&str, &str, u64)> = commit_items
            .iter()
            .map(|(path, oid, size)| (path.as_str(), *oid, *size))
            .collect();

        let first_shard = &hash_records.first().unwrap().0;
        let last_shard = &hash_records.last().unwrap().0;
        let summary = format!("Upload v2 shards: {} to {} (Batch {}/{})", first_shard, last_shard, batch_num, total_batches);

        println!("Committing batch {} to Hugging Face repository...", batch_num);
        commit_lfs_files(
            HfRepoType::Dataset,
            repo_id,
            token,
            &summary,
            "TARA Canonical Indic Dataset v2 Shards",
            &commit_refs,
        ).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        println!("Commit successful: {}", summary);
    }

    println!();
    println!("================================================================================");
    println!("  POST-UPLOAD FINAL REPOSITORY VERIFICATION");
    println!("================================================================================");

    let final_remote = fetch_remote_tree(HfRepoType::Dataset, repo_id, token, "v2")
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    let mut missing = Vec::new();
    for (name, _) in &shard_paths {
        if !final_remote.contains_key(name) {
            missing.push(name.clone());
        }
    }

    if missing.is_empty() {
        println!("VERIFICATION SUCCESS: All {} canonical shards verified live on Hugging Face!", shard_paths.len());
        println!("Repository: https://huggingface.co/datasets/{}", repo_id);
        println!("Directory : v2/");
        println!("100% COMPLETE: Full dataset v2 successfully uploaded to Hugging Face.");
        Ok(true)
    } else {
        println!("Partial batch complete; {} shards remaining.", missing.len());
        Ok(false)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!("  TARA AUTONOMOUS HUGGING FACE DATASET UPLOADER (100% NATIVE RUST)");
    println!("================================================================================");

    let repo_id = match env::var("HF_REPO") {
        Ok(r) if !r.trim().is_empty() => r,
        _ => {
            eprintln!("Error: HF_REPO environment variable must be set (e.g. HF_REPO='organization/dataset-name')");
            std::process::exit(1);
        }
    };
    let token = resolve_hf_token().map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    println!("Authentication         : Verified via Secure Credential Store");

    let root = tara_engine::find_workspace_root()
        .unwrap_or_else(|_| env::current_dir().unwrap_or_default());
    let dataset_dir = root.join("storage").join("datasets").join("main_dataset_v2");
    if !dataset_dir.exists() {
        eprintln!("FATAL ERROR: Local dataset directory does not exist: {:?}", dataset_dir);
        std::process::exit(1);
    }
    let has_files = fs::read_dir(&dataset_dir)
        .map(|entries| {
            entries.flatten().any(|e| {
                e.path().is_file()
                    && e.path()
                        .extension()
                        .is_some_and(|ext| ext == "jsonl" || ext == "json")
            })
        })
        .unwrap_or(false);
    if !has_files {
        eprintln!("FATAL ERROR: Local dataset directory contains 0 files to upload: {:?}", dataset_dir);
        std::process::exit(1);
    }

    const MAX_RECOVERY_CYCLES: usize = 50;
    let mut cycle = 0;
    loop {
        cycle += 1;
        if cycle > MAX_RECOVERY_CYCLES {
            return Err(format!("Exceeded maximum recovery cycles ({MAX_RECOVERY_CYCLES})").into());
        }
        if cycle > 1 {
            println!("\n[CYCLE {}/{}] Resuming autonomous upload pipeline...\n", cycle, MAX_RECOVERY_CYCLES);
        }

        match run_upload_pipeline(&repo_id, &token) {
            Ok(true) => {
                println!("Pipeline reached 100% verified completion.");
                break;
            }
            Ok(false) => {
                println!("Pipeline completed pass. Re-checking remaining shards in 5s...");
                sleep(Duration::from_secs(5));
            }
            Err(e) => {
                eprintln!("\n[RESILIENCE] Temporary error encountered: {}. Auto-recovering in 10s...\n", e);
                sleep(Duration::from_secs(10));
            }
        }
    }

    Ok(())
}
