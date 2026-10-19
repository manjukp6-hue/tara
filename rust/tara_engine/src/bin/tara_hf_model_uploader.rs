//! TARA Autonomous Hugging Face Model Uploader (100% Native Rust).
//!
//! Direct-to-Cloud Streaming Model Uploader:
//! - Uploads Neural Model SafeTensors shards and metadata to remote prefix `neural_model_v1/`.
//! - Uploads World Model SafeTensors shards and metadata to remote prefix `world_model_v1/`.
//! - Dynamically resolves Hugging Face authentication credentials from environment or git store.
//! - Direct communication with Hugging Face Model Git LFS Batch API and AWS S3 acceleration endpoints.
//! - Batched commits via Hugging Face Models Commit API (`/api/models/{repo_id}/commit/main`).
//! - Exhaustive post-upload verification against the live remote tree.
//! - Indestructible self-healing outer loop with automatic retries on transient network drops.

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::Duration;

use tara_engine::hf_adapter::{
    commit_lfs_files, compute_sha256_and_size, fetch_remote_tree, query_lfs_batch,
    resolve_hf_token, upload_lfs_to_s3, verify_lfs_upload, HfRepoType, LfsRequestObject,
};

fn upload_model_directory(
    repo_id: &str,
    token: &str,
    source_dir: &Path,
    remote_prefix: &str,
    label: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    println!();
    println!("================================================================================");
    println!("  UPLOADING: {} -> {}/{}", label, repo_id, remote_prefix);
    println!("================================================================================");
    println!("Local Source Directory : {}", source_dir.display());

    if !source_dir.exists() {
        eprintln!(
            "FATAL ERROR: Source directory '{}' does not exist",
            source_dir.display()
        );
        std::process::exit(1);
    }

    let mut local_files: Vec<(String, PathBuf)> = Vec::new();
    for entry in fs::read_dir(source_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            local_files.push((name, path));
        }
    }
    local_files.sort_by(|a, b| a.0.cmp(&b.0));

    if local_files.is_empty() {
        eprintln!(
            "FATAL ERROR: Source directory '{}' contains 0 files to upload",
            source_dir.display()
        );
        std::process::exit(1);
    }

    let total_bytes: u64 = local_files
        .iter()
        .map(|(_, p)| fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .sum();

    println!("Total Files to Sync    : {}", local_files.len());
    println!(
        "Total Payload Size     : {} MB ({:.2} GB)",
        total_bytes / (1024 * 1024),
        total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    println!(
        "Checking remote tree on Hugging Face at prefix '{}'...",
        remote_prefix
    );
    let remote_files = fetch_remote_tree(HfRepoType::Model, repo_id, token, remote_prefix)
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    println!("Existing Remote Files  : {}", remote_files.len());

    let mut pending_files = Vec::new();
    for (name, path) in &local_files {
        let local_sz = fs::metadata(path)?.len();
        if let Some(&rem_sz) = remote_files.get(name) {
            if rem_sz == local_sz {
                continue;
            }
        }
        pending_files.push((name.clone(), path.clone()));
    }

    println!(
        "Pending Files to Upload: {} of {}",
        pending_files.len(),
        local_files.len()
    );
    if pending_files.is_empty() {
        println!(
            "All {} files already uploaded and verified on Hugging Face!",
            local_files.len()
        );
        return Ok(true);
    }

    let batch_size = 3;
    let total_pending = pending_files.len();
    let clean_prefix = remote_prefix.trim_matches('/');

    for (batch_idx, chunk) in pending_files.chunks(batch_size).enumerate() {
        let batch_num = batch_idx + 1;
        let total_batches = total_pending.div_ceil(batch_size);
        println!();
        println!(
            "--- Processing Batch {}/{} ({} files) ---",
            batch_num,
            total_batches,
            chunk.len()
        );

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
            loop {
                attempts += 1;
                let single_req = [LfsRequestObject {
                    oid: oid.clone(),
                    size: *size,
                }];
                let resp = query_lfs_batch(HfRepoType::Model, repo_id, token, &single_req, 60)
                    .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
                let obj = resp.objects.into_iter().next();

                if let Some(obj) = obj {
                    if let Some(actions) = obj.actions {
                        if let Some(upload_link) = actions.upload {
                            println!("Streaming {} ({} MB) to S3...", name, size / (1024 * 1024));
                            match upload_lfs_to_s3(
                                path,
                                &upload_link.href,
                                upload_link.header.as_ref(),
                            ) {
                                Ok(()) => {
                                    println!("S3 upload finished for {}.", name);
                                }
                                Err(e) => {
                                    eprintln!(
                                        "Upload error for {}: {}. Retrying in 10s...",
                                        name, e
                                    );
                                    sleep(Duration::from_secs(10));
                                    if attempts >= 10 {
                                        return Err(
                                            format!("Max retries reached for {}", name).into()
                                        );
                                    }
                                    continue;
                                }
                            }

                            if let Some(verify_link) = actions.verify {
                                println!("Verifying LFS upload with Hugging Face...");
                                verify_lfs_upload(&verify_link.href, token, oid, *size)
                                    .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
                                println!("LFS upload verified for {}.", name);
                            }
                        } else {
                            println!(
                                "File {} already present in LFS storage (deduplicated).",
                                name
                            );
                        }
                    } else {
                        println!("File {} already exists on Hugging Face.", name);
                    }
                    break;
                }
            }
        }

        let commit_items: Vec<(String, &str, u64)> = hash_records
            .iter()
            .map(|(n, _, oid, sz)| (format!("{}/{}", clean_prefix, n), oid.as_str(), *sz))
            .collect();
        let commit_refs: Vec<(&str, &str, u64)> = commit_items
            .iter()
            .map(|(p, oid, sz)| (p.as_str(), *oid, *sz))
            .collect();

        let first = chunk.first().map(|(n, _)| n.as_str()).unwrap_or("");
        let last = chunk.last().map(|(n, _)| n.as_str()).unwrap_or("");
        let summary = format!(
            "Upload {} files: {} to {} (Batch {}/{})",
            label, first, last, batch_num, total_batches
        );

        println!(
            "Committing batch {} to Hugging Face repository...",
            batch_num
        );
        commit_lfs_files(
            HfRepoType::Model,
            repo_id,
            token,
            &summary,
            &format!("TARA Model Artifacts under prefix '{}'", remote_prefix),
            &commit_refs,
        )
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        println!("Commit successful: {}", summary);
    }

    println!();
    println!(
        "=== VERIFYING REMOTE TREE: {}/{} ===",
        repo_id, remote_prefix
    );
    let final_remote = fetch_remote_tree(HfRepoType::Model, repo_id, token, remote_prefix)
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    let mut missing = Vec::new();
    for (name, path) in &local_files {
        let local_sz = fs::metadata(path)?.len();
        match final_remote.get(name) {
            Some(&rem_sz) if rem_sz == local_sz => {}
            Some(&rem_sz) => {
                missing.push(format!(
                    "{} (size mismatch: local={}, remote={})",
                    name, local_sz, rem_sz
                ));
            }
            None => {
                missing.push(format!("{} (missing on remote)", name));
            }
        }
    }

    if missing.is_empty() {
        println!(
            "VERIFICATION SUCCESS: All {} files verified live in {}/{}!",
            local_files.len(),
            repo_id,
            remote_prefix
        );
        Ok(true)
    } else {
        Err(format!(
            "Verification failed! Missing or mismatched files: {:?}",
            missing
        )
        .into())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!("  TARA AUTONOMOUS HUGGING FACE MODEL UPLOADER (NATIVE RUST)");
    println!("================================================================================");

    let args: Vec<String> = env::args().collect();
    let mut custom_repo = None;
    let mut custom_neural = None;
    let mut custom_world = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--repo" | "-r" if i + 1 < args.len() => {
                custom_repo = Some(args[i + 1].clone());
                i += 1;
            }
            "--neural-dir" | "-n" if i + 1 < args.len() => {
                custom_neural = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            "--world-dir" | "-w" if i + 1 < args.len() => {
                custom_world = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }

    let repo_id = match custom_repo.or_else(|| env::var("HF_MODEL_REPO").ok()).or_else(|| env::var("HF_REPO").ok()) {
        Some(r) if !r.trim().is_empty() => r,
        _ => {
            eprintln!("Error: HF_MODEL_REPO environment variable or --repo flag must be set (e.g. --repo 'organization/model-name')");
            std::process::exit(1);
        }
    };
    let token = resolve_hf_token().map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    println!("Authentication         : Verified via Secure Credential Store");
    println!(
        "Target Model Hub Repo  : https://huggingface.co/{}",
        repo_id
    );

    let neural_dir = custom_neural.unwrap_or_else(|| PathBuf::from("storage/models/tara_candidate_v1"));
    let has_custom_world = custom_world.is_some();
    let world_dir = custom_world.unwrap_or_else(|| PathBuf::from("storage/models/tara_world_model"));

    const MAX_RECOVERY_CYCLES: usize = 50;

    // 1. Upload Neural Model v1 (46 SafeTensors shards + metadata)
    let mut neural_cycle = 0;
    loop {
        neural_cycle += 1;
        if neural_cycle > MAX_RECOVERY_CYCLES {
            return Err(format!("Exceeded maximum recovery cycles ({MAX_RECOVERY_CYCLES})").into());
        }
        if neural_cycle > 1 {
            println!(
                "\n[CYCLE {}/{}] Resuming Neural Model upload pipeline...\n",
                neural_cycle, MAX_RECOVERY_CYCLES
            );
        }

        match upload_model_directory(
            &repo_id,
            &token,
            &neural_dir,
            "neural_model_v1",
            "Neural Model V1",
        ) {
            Ok(true) => break,
            Ok(false) => {
                println!("Neural Model upload incomplete, retrying in 10s...");
                sleep(Duration::from_secs(10));
            }
            Err(e) => {
                eprintln!(
                    "Neural Model upload encountered error: {}. Auto-recovering in 15s...",
                    e
                );
                sleep(Duration::from_secs(15));
            }
        }
    }

    // 2. Upload World Model v1 (only if configured or present)
    let should_upload_world = has_custom_world
        || (world_dir.exists()
            && fs::read_dir(&world_dir)
                .map(|mut d| d.next().is_some())
                .unwrap_or(false));

    if should_upload_world {
        let mut world_cycle = 0;
        loop {
            world_cycle += 1;
            if world_cycle > MAX_RECOVERY_CYCLES {
                return Err(format!("Exceeded maximum recovery cycles ({MAX_RECOVERY_CYCLES})").into());
            }
            if world_cycle > 1 {
                println!(
                    "\n[CYCLE {}/{}] Resuming World Model upload pipeline...\n",
                    world_cycle, MAX_RECOVERY_CYCLES
                );
            }

            match upload_model_directory(
                &repo_id,
                &token,
                &world_dir,
                "world_model_v1",
                "World Model V1",
            ) {
                Ok(true) => break,
                Ok(false) => {
                    println!("World Model upload incomplete, retrying in 10s...");
                    sleep(Duration::from_secs(10));
                }
                Err(e) => {
                    eprintln!(
                        "World Model upload encountered error: {}. Auto-recovering in 15s...",
                        e
                    );
                    sleep(Duration::from_secs(15));
                }
            }
        }
    } else {
        println!("World Model directory not present or empty; skipping World Model sync.");
    }

    println!();
    println!("================================================================================");
    println!("  ALL MODELS SUCCESSFULLY UPLOADED & VERIFIED ON HUGGING FACE!");
    println!("================================================================================");
    println!(
        "Neural Model V1 : https://huggingface.co/{}/tree/main/neural_model_v1",
        repo_id
    );
    println!(
        "World Model V1  : https://huggingface.co/{}/tree/main/world_model_v1",
        repo_id
    );
    println!("100% Complete: Ready for autonomous Google Colab multi-stage training.");
    Ok(())
}
