//! TARA Canonical Dataset Splitter & Leakage Control Engine Binary.
//!
//! Provides deterministic partitioning of canonical datasets into train, val,
//! and test splits, with:
//! - Pre/post split SHA-256 shard snapshot mutation detection (aborts if input mutates during split)
//! - Bidirectional input/output path isolation and sibling default output resolution
//! - 256-bit cryptographic SHA-256 record multiset conservation & cross-partition disjointness verification
//! - Full 3-way pairwise cross-split leakage audit (exact + 13-gram shingle overlap)
//! - Fail-closed gate enforcement (`--skip-leakage` / `--unsafe-skip-leakage` cannot pass Gate 13)
//! - Versioned release staging (`releases/run_<id>`) with atomic `current_release.json` pointer switch
//!   and rollback-safe `split_manifest.json` publication

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tara_engine::dataset::compute_file_sha256;
use tara_engine::dataset::splitter::{
    extract_canonical_sample, DatasetSplitter, LeakageChecker, SplitRatio,
    LEAKAGE_OVERLAP_THRESHOLD,
};

fn print_usage() {
    println!("Usage: tara_dataset_splitter [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --input, --input-dir <PATH>      Path to canonical dataset .jsonl file or directory of shards");
    println!("  --output, --output-dir <PATH>    Output directory for split files (default: <input>_splits)");
    println!("  --train <RATIO>                  Target training split ratio (default: 0.80)");
    println!("  --val <RATIO>                    Target validation split ratio (default: 0.10)");
    println!("  --test <RATIO>                   Target testing split ratio (default: 0.10)");
    println!("  --decontaminate-against <EVAL>   Optional eval dataset to decontaminate input against");
    println!("  --clean-output <PATH>            Output path for decontaminated training dataset");
    println!("  --unsafe-skip-leakage            Skip leakage audit (marks gate NOT VERIFIED and exits non-zero)");
    println!("  --skip-leakage                   Alias for --unsafe-skip-leakage");
    println!("  -h, --help                       Print help information");
    println!();
    println!("Note: DSU transitive cluster isolation guarantees zero cross-partition leakage;");
    println!("      connected near-duplicate components may cause slight deviation from nominal ratios.");
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ShardSnapshot {
    path: PathBuf,
    file_sha256: String,
    bytes: u64,
}

/// Enumerates and hashes all source JSONL shards in deterministic sorted order.
fn capture_input_snapshot(input_path: &Path) -> Result<(String, Vec<ShardSnapshot>), Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    if input_path.is_dir() {
        let mut entries: Vec<PathBuf> = fs::read_dir(input_path)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                if p.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
                    return false;
                }
                if let Some(fname) = p.file_name().and_then(|n| n.to_str()) {
                    if fname == "train.jsonl"
                        || fname == "val.jsonl"
                        || fname == "test.jsonl"
                        || fname == "clean_train.jsonl"
                        || fname == "split_manifest.json"
                        || fname.contains(".tmp")
                        || fname.starts_with('.')
                    {
                        return false;
                    }
                }
                true
            })
            .collect();
        entries.sort();
        files = entries;
    } else if input_path.is_file() {
        files.push(input_path.to_path_buf());
    }

    if files.is_empty() {
        return Err(format!("Input dataset contains zero .jsonl shard files: {}", input_path.display()).into());
    }

    let mut snapshots = Vec::with_capacity(files.len());
    let mut manifest_hasher = Sha256::new();

    for f in files {
        let meta = fs::metadata(&f)?;
        let sha = compute_file_sha256(&f).map_err(std::io::Error::other)?;
        manifest_hasher.update(f.to_string_lossy().as_bytes());
        manifest_hasher.update(b":");
        manifest_hasher.update(sha.as_bytes());
        manifest_hasher.update(b"\n");
        snapshots.push(ShardSnapshot {
            path: f,
            file_sha256: sha,
            bytes: meta.len(),
        });
    }

    let aggregate_sha = hex::encode(manifest_hasher.finalize());
    Ok((aggregate_sha, snapshots))
}

fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Two-stage semantic identity key combining a 256-bit cryptographic SHA-256 digest
/// with exact canonical string equality (mirroring `LeakageIndex`'s collision-free discipline).
type SemanticKey = ([u8; 32], String);

/// Computes multiset of two-stage `(SHA-256, raw_line)` canonical line keys and set of
/// two-stage `(SHA-256, combined_text)` canonical sample keys for a JSONL file.
fn compute_semantic_fingerprints(
    jsonl_path: &Path,
) -> Result<(HashMap<SemanticKey, usize>, HashSet<SemanticKey>, usize), Box<dyn std::error::Error>> {
    let file = File::open(jsonl_path)?;
    let reader = BufReader::with_capacity(128 * 1024, file);
    let mut line_multiset: HashMap<SemanticKey, usize> = HashMap::new();
    let mut combined_set: HashSet<SemanticKey> = HashSet::new();
    let mut count = 0usize;

    for line_res in reader.lines() {
        let line = line_res?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let sample = extract_canonical_sample(trimmed);
        let raw_digest = sha256_bytes(sample.raw_line.as_bytes());
        let comb_digest = sha256_bytes(sample.combined_text.as_bytes());
        *line_multiset
            .entry((raw_digest, sample.raw_line))
            .or_insert(0) += 1;
        combined_set.insert((comb_digest, sample.combined_text));
        count += 1;
    }

    Ok((line_multiset, combined_set, count))
}

/// Computes the union multiset of two-stage `(SHA-256, raw_line)` canonical keys across all input shards.
fn compute_input_multiset(
    shards: &[ShardSnapshot],
) -> Result<(HashMap<SemanticKey, usize>, usize), Box<dyn std::error::Error>> {
    let mut total_multiset: HashMap<SemanticKey, usize> = HashMap::new();
    let mut total_count = 0usize;
    for shard in shards {
        let (ms, _, cnt) = compute_semantic_fingerprints(&shard.path)?;
        for (k, v) in ms {
            *total_multiset.entry(k).or_insert(0) += v;
        }
        total_count += cnt;
    }
    Ok((total_multiset, total_count))
}

pub fn run_splitter_cli(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return Ok(());
    }

    let mut input_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut train_ratio = 0.80f64;
    let mut val_ratio = 0.10f64;
    let mut test_ratio = 0.10f64;
    let mut skip_leakage = false;
    let mut decontaminate_eval: Option<PathBuf> = None;
    let mut clean_output: Option<PathBuf> = None;

    let mut idx = 1;
    while idx < args.len() {
        match args[idx].as_str() {
            "--input-dir" | "--input" => {
                if idx + 1 < args.len() {
                    input_path = Some(PathBuf::from(&args[idx + 1]));
                    idx += 2;
                } else {
                    return Err("Missing argument for --input".into());
                }
            }
            "--output-dir" | "--output" => {
                if idx + 1 < args.len() {
                    output_path = Some(PathBuf::from(&args[idx + 1]));
                    idx += 2;
                } else {
                    return Err("Missing argument for --output".into());
                }
            }
            "--train" => {
                if idx + 1 < args.len() {
                    train_ratio = args[idx + 1].parse::<f64>()?;
                    idx += 2;
                } else {
                    return Err("Missing argument for --train".into());
                }
            }
            "--val" => {
                if idx + 1 < args.len() {
                    val_ratio = args[idx + 1].parse::<f64>()?;
                    idx += 2;
                } else {
                    return Err("Missing argument for --val".into());
                }
            }
            "--test" => {
                if idx + 1 < args.len() {
                    test_ratio = args[idx + 1].parse::<f64>()?;
                    idx += 2;
                } else {
                    return Err("Missing argument for --test".into());
                }
            }
            "--decontaminate-against" => {
                if idx + 1 < args.len() {
                    decontaminate_eval = Some(PathBuf::from(&args[idx + 1]));
                    idx += 2;
                } else {
                    return Err("Missing argument for --decontaminate-against".into());
                }
            }
            "--clean-output" => {
                if idx + 1 < args.len() {
                    clean_output = Some(PathBuf::from(&args[idx + 1]));
                    idx += 2;
                } else {
                    return Err("Missing argument for --clean-output".into());
                }
            }
            "--skip-leakage" | "--unsafe-skip-leakage" => {
                skip_leakage = true;
                idx += 1;
            }
            unknown => {
                return Err(format!("Unknown option: {unknown}").into());
            }
        }
    }

    let input_dir = match input_path {
        Some(p) => p,
        None => {
            print_usage();
            return Err("Error: --input is required".into());
        }
    };

    if !input_dir.exists() {
        return Err(format!("Input path does not exist: {}", input_dir.display()).into());
    }

    if let Some(eval_p) = decontaminate_eval {
        let clean_out = clean_output.unwrap_or_else(|| {
            let fname = input_dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("dataset.jsonl");
            input_dir
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(format!("decontaminated_{fname}"))
        });
        println!("================================================================================");
        println!(" TARA CANONICAL DATASET DECONTAMINATION ENGINE");
        println!("================================================================================");
        println!("Source Train : {}", input_dir.display());
        println!("Target Eval  : {}", eval_p.display());
        println!("Clean Output : {}", clean_out.display());
        println!("--------------------------------------------------------------------------------");
        let start = Instant::now();
        let preserved = LeakageChecker::decontaminate(&input_dir, &eval_p, &clean_out)?;
        let elapsed = start.elapsed();
        println!("[DECONTAMINATION COMPLETE]");
        println!("  Preserved clean samples: {}", preserved);
        println!("  Execution wall time    : {:.2?}", elapsed);
        println!("  Output sanitized file  : {}", clean_out.display());
        println!("================================================================================");
        return Ok(());
    }

    // Default output path is placed outside the input directory (<parent>/<stem>_splits) to prevent self-ingestion.
    // Canonicalizing input_dir first ensures that relative paths like `.`, `./canonical`, or `canonical/`
    // always resolve to a true sibling directory outside the source tree.
    let output_dir = match output_path {
        Some(p) => p,
        None => {
            let canon_in = input_dir.canonicalize()?;
            let parent = canon_in
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .ok_or("Cannot derive sibling default output directory for filesystem root; please pass --output explicitly")?;
            let stem = canon_in
                .file_stem()
                .and_then(|n| n.to_str())
                .unwrap_or("dataset");
            parent.join(format!("{stem}_splits"))
        }
    };

    println!("================================================================================");
    println!(" TARA CANONICAL DATASET SPLITTER & LEAKAGE AUDITOR");
    println!("================================================================================");
    println!("Input Source : {}", input_dir.display());
    println!("Output Target: {}", output_dir.display());
    println!(
        "Split Targets: Train = {:.2} | Val = {:.2} | Test = {:.2}",
        train_ratio, val_ratio, test_ratio
    );
    println!("--------------------------------------------------------------------------------");

    let split_ratio = SplitRatio::new(train_ratio, val_ratio, test_ratio)?;

    // 1. Capture pre-split immutable input snapshot
    let (input_manifest_sha, pre_shards) = capture_input_snapshot(&input_dir)?;
    let (input_multiset, input_record_count) = compute_input_multiset(&pre_shards)?;
    if input_record_count == 0 {
        return Err("Input dataset contains zero valid records.".into());
    }

    // 2. Execute deterministic DSU cluster-isolated split
    let split_start = Instant::now();
    let report = DatasetSplitter::split(&input_dir, &output_dir, split_ratio)?;
    let split_elapsed = split_start.elapsed();

    if report.total_samples == 0 {
        return Err("Input dataset contains zero valid records.".into());
    }

    // 3. Verify input snapshot did not mutate during split execution
    let (post_input_sha, post_shards) = capture_input_snapshot(&input_dir)?;
    if input_manifest_sha != post_input_sha || pre_shards != post_shards {
        return Err("Input dataset mutated during split execution! Aborting for reproducibility.".into());
    }

    let train_pct = (report.train_samples as f64 / report.total_samples as f64) * 100.0;
    let val_pct = (report.val_samples as f64 / report.total_samples as f64) * 100.0;
    let test_pct = (report.test_samples as f64 / report.total_samples as f64) * 100.0;

    let sum_parts = report.train_samples + report.val_samples + report.test_samples;
    let count_conservation_pass = sum_parts == report.total_samples && report.total_samples == input_record_count;

    // 4. Verify two-stage (256-bit SHA-256 + exact canonical string equality) semantic record multiset
    //    conservation and cross-partition disjointness
    let (train_ms, train_comb, _) = compute_semantic_fingerprints(Path::new(&report.train_path))?;
    let (val_ms, val_comb, _) = compute_semantic_fingerprints(Path::new(&report.val_path))?;
    let (test_ms, test_comb, _) = compute_semantic_fingerprints(Path::new(&report.test_path))?;

    let mut union_ms: HashMap<SemanticKey, usize> = HashMap::new();
    for (k, v) in train_ms.iter().chain(val_ms.iter()).chain(test_ms.iter()) {
        *union_ms.entry(k.clone()).or_insert(0) += *v;
    }
    let semantic_conservation_pass = union_ms == input_multiset;

    let tv_disjoint = train_comb.is_disjoint(&val_comb);
    let tt_disjoint = train_comb.is_disjoint(&test_comb);
    let vt_disjoint = val_comb.is_disjoint(&test_comb);
    let disjoint_pass = tv_disjoint && tt_disjoint && vt_disjoint;

    let conservation_pass = count_conservation_pass && semantic_conservation_pass && disjoint_pass;

    println!("[SPLIT EXECUTION COMPLETE]");
    println!("  Input Snapshot SHA-256  : {}", input_manifest_sha);
    println!("  Total Canonical Records : {}", report.total_samples);
    println!(
        "  Train Samples           : {} ({:.2}%) -> {}",
        report.train_samples, train_pct, report.train_path
    );
    println!(
        "  Val Samples             : {} ({:.2}%) -> {}",
        report.val_samples, val_pct, report.val_path
    );
    println!(
        "  Test Samples            : {} ({:.2}%) -> {}",
        report.test_samples, test_pct, report.test_path
    );
    println!(
        "  Conservation Check      : {} = {} + {} + {} (Semantic Multiset & Disjointness) -> {}",
        report.total_samples,
        report.train_samples,
        report.val_samples,
        report.test_samples,
        if conservation_pass {
            "PASS (100% Exact Cardinality & Disjoint Match)"
        } else {
            "FAIL (Record Conservation or Disjointness Mismatch)"
        }
    );
    println!("  Split Wall Time         : {:.2?}", split_elapsed);
    println!("--------------------------------------------------------------------------------");

    if !conservation_pass {
        return Err("Conservation check failed: split outputs do not match input record multiset or disjointness invariant".into());
    }

    println!("[COMPUTING DYNAMIC CRYPTOGRAPHIC FILE SHAS]");
    let train_sha = compute_file_sha256(Path::new(&report.train_path)).map_err(std::io::Error::other)?;
    let val_sha = compute_file_sha256(Path::new(&report.val_path)).map_err(std::io::Error::other)?;
    let test_sha = compute_file_sha256(Path::new(&report.test_path)).map_err(std::io::Error::other)?;
    println!("  TRAIN FILE SHA-256 : {}", train_sha);
    println!("  VAL   FILE SHA-256 : {}", val_sha);
    println!("  TEST  FILE SHA-256 : {}", test_sha);
    println!("--------------------------------------------------------------------------------");

    // 5. Enforce fail-closed gate if leakage check was skipped
    if skip_leakage {
        println!("[LEAKAGE AUDIT SKIPPED]");
        println!("  Leakage Audit: SKIPPED (--unsafe-skip-leakage)");
        println!("  Overall Gate : NOT VERIFIED");
        println!("================================================================================");
        return Err("Strict canonical split gate cannot pass with leakage verification disabled.".into());
    }

    println!("[EXECUTING CROSS-SPLIT LEAKAGE AUDIT]");

    // 1. Train <-> Val
    let tv_start = Instant::now();
    let tv = LeakageChecker::check_leakage(&report.train_path, &report.val_path)?;
    let tv_elapsed = tv_start.elapsed();
    println!(
        "  Train <-> Val  : exact_match = {} | ngram_shingle = {} | total_leaked = {} ({:.2}%) (time: {:.2?}) -> {}",
        tv.exact_match_leaks,
        tv.ngram_shingle_leaks,
        tv.total_leaked_samples,
        tv.leakage_rate_pct,
        tv_elapsed,
        if tv.is_clean { "CLEAN (0.00% Leakage)" } else { "CONTAMINATED" }
    );

    // 2. Train <-> Test
    let tt_start = Instant::now();
    let tt = LeakageChecker::check_leakage(&report.train_path, &report.test_path)?;
    let tt_elapsed = tt_start.elapsed();
    println!(
        "  Train <-> Test : exact_match = {} | ngram_shingle = {} | total_leaked = {} ({:.2}%) (time: {:.2?}) -> {}",
        tt.exact_match_leaks,
        tt.ngram_shingle_leaks,
        tt.total_leaked_samples,
        tt.leakage_rate_pct,
        tt_elapsed,
        if tt.is_clean { "CLEAN (0.00% Leakage)" } else { "CONTAMINATED" }
    );

    // 3. Val <-> Test
    let vt_start = Instant::now();
    let vt = LeakageChecker::check_leakage(&report.val_path, &report.test_path)?;
    let vt_elapsed = vt_start.elapsed();
    println!(
        "  Val   <-> Test : exact_match = {} | ngram_shingle = {} | total_leaked = {} ({:.2}%) (time: {:.2?}) -> {}",
        vt.exact_match_leaks,
        vt.ngram_shingle_leaks,
        vt.total_leaked_samples,
        vt.leakage_rate_pct,
        vt_elapsed,
        if vt.is_clean { "CLEAN (0.00% Leakage)" } else { "CONTAMINATED" }
    );

    let all_clean = tv.is_clean && tt.is_clean && vt.is_clean;
    let leakage_status = if all_clean { "CLEAN" } else { "FAILED" };

    // 6. Persist full audit-grade split_manifest.json including input snapshot and pairwise leakage results
    let now_stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let manifest_path = output_dir.join("split_manifest.json");
    let active_release_id = fs::read_to_string(output_dir.join("current_release.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .and_then(|v| v.get("current_release").and_then(Value::as_str).map(String::from))
        .unwrap_or_else(|| format!("run_{now_stamp}_{}", std::process::id()));

    let source_canon_str = input_dir
        .canonicalize()
        .unwrap_or_else(|_| input_dir.clone())
        .to_string_lossy()
        .to_string();

    let tmp_manifest = output_dir.join(format!(".tmp_audit_manifest_{now_stamp}.json"));
    let shard_json: Vec<Value> = pre_shards
        .iter()
        .map(|s| {
            json!({
                "path": s.path.to_string_lossy(),
                "file_sha256": s.file_sha256,
                "bytes": s.bytes
            })
        })
        .collect();

    let audit_manifest = json!({
        "release_id": active_release_id,
        "timestamp_ns": now_stamp,
        "splitter_version": env!("CARGO_PKG_VERSION"),
        "split_algorithm_version": "fnv1a_dsu_13gram_v2",
        "source_canonical_path": source_canon_str,
        "input_manifest_sha256": input_manifest_sha,
        "input_shards": shard_json,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "total_samples": report.total_samples,
        "train_records": report.train_samples,
        "val_records": report.val_samples,
        "test_records": report.test_samples,
        "train_samples": report.train_samples,
        "val_samples": report.val_samples,
        "test_samples": report.test_samples,
        "train_file": "train.jsonl",
        "val_file": "val.jsonl",
        "test_file": "test.jsonl",
        "train_sha256": train_sha,
        "val_sha256": val_sha,
        "test_sha256": test_sha,
        "semantic_conservation_verified": semantic_conservation_pass,
        "disjoint_partitions_verified": disjoint_pass,
        "leakage_threshold": LEAKAGE_OVERLAP_THRESHOLD,
        "overlap_threshold": LEAKAGE_OVERLAP_THRESHOLD,
        "leakage_status": leakage_status,
        "pairs": {
            "train_val": {
                "exact_matches": tv.exact_match_leaks,
                "shingle_leaks": tv.ngram_shingle_leaks,
                "total_leaked": tv.total_leaked_samples,
                "leakage_rate_pct": tv.leakage_rate_pct,
                "is_clean": tv.is_clean
            },
            "train_test": {
                "exact_matches": tt.exact_match_leaks,
                "shingle_leaks": tt.ngram_shingle_leaks,
                "total_leaked": tt.total_leaked_samples,
                "leakage_rate_pct": tt.leakage_rate_pct,
                "is_clean": tt.is_clean
            },
            "val_test": {
                "exact_matches": vt.exact_match_leaks,
                "shingle_leaks": vt.ngram_shingle_leaks,
                "total_leaked": vt.total_leaked_samples,
                "leakage_rate_pct": vt.leakage_rate_pct,
                "is_clean": vt.is_clean
            }
        }
    });

    let audit_manifest_pretty = serde_json::to_string_pretty(&audit_manifest)?;
    {
        let mut mf = File::create(&tmp_manifest)?;
        mf.write_all(audit_manifest_pretty.as_bytes())?;
        mf.flush()?;
        mf.sync_all()?;
    }
    fs::rename(&tmp_manifest, &manifest_path)?;

    let rel_manifest_path = output_dir
        .join("releases")
        .join(&active_release_id)
        .join("split_manifest.json");
    if let Some(rel_parent) = rel_manifest_path.parent() {
        if rel_parent.exists() {
            let tmp_rel_manifest = rel_parent.join(format!(".tmp_audit_manifest_{now_stamp}.json"));
            if let Ok(mut rmf) = File::create(&tmp_rel_manifest) {
                let _ = rmf.write_all(audit_manifest_pretty.as_bytes());
                let _ = rmf.flush();
                let _ = rmf.sync_all();
                let _ = fs::rename(&tmp_rel_manifest, &rel_manifest_path);
            }
        }
    }

    // Verify 3-state committed split integrity before concluding Gate 13
    DatasetSplitter::verify_committed_split(&output_dir)?;

    println!("  Audit Manifest : {}", manifest_path.display());
    println!("--------------------------------------------------------------------------------");

    if all_clean {
        println!("  Overall Cross-Split Leakage: ZERO CONTAMINATION DETECTED (PASSED)");
    } else {
        return Err("Cross-split contamination detected during leakage audit!".into());
    }

    println!("================================================================================");
    println!(" GATE 13: FULL CANONICAL DATASET SPLIT AUDIT -> VERIFIED SUCCESS");
    println!("================================================================================");

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    run_splitter_cli(&args)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_temp_dir(label: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = env::temp_dir().join(format!("tara_splitter_bin_{label}_{stamp}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_cli_skip_leakage_fails_gate() {
        let dir = make_temp_dir("skip_leak");
        let src_file = dir.join("data.jsonl");
        let out_dir = dir.join("out_splits");
        {
            let mut f = File::create(&src_file).unwrap();
            for i in 0..30 {
                writeln!(f, r#"{{"input":"q_{i}","output":"a_{i}"}}"#).unwrap();
            }
        }

        let args = vec![
            "tara_dataset_splitter".to_string(),
            "--input".to_string(),
            src_file.to_string_lossy().to_string(),
            "--output".to_string(),
            out_dir.to_string_lossy().to_string(),
            "--skip-leakage".to_string(),
        ];

        let res = run_splitter_cli(&args);
        assert!(res.is_err());
        assert!(res
            .unwrap_err()
            .to_string()
            .contains("Strict canonical split gate cannot pass with leakage verification disabled"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_cli_empty_dataset_rejected() {
        let dir = make_temp_dir("empty");
        let empty_file = dir.join("empty.jsonl");
        File::create(&empty_file).unwrap();
        let out_dir = dir.join("out_splits");

        let args = vec![
            "tara_dataset_splitter".to_string(),
            "--input".to_string(),
            empty_file.to_string_lossy().to_string(),
            "--output".to_string(),
            out_dir.to_string_lossy().to_string(),
        ];

        let res = run_splitter_cli(&args);
        assert!(res.is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_cli_path_overlap_rejected_and_default_output_safe() {
        let dir = make_temp_dir("overlap");
        let src_dir = dir.join("canonical_data");
        fs::create_dir_all(&src_dir).unwrap();
        let shard = src_dir.join("shard_0.jsonl");
        {
            let mut f = File::create(&shard).unwrap();
            for i in 0..50 {
                writeln!(f, r#"{{"input":"prompt_item_{i}","output":"target_item_{i}"}}"#).unwrap();
            }
        }

        // 1. Overlapping output inside input directory -> REJECTED
        let bad_out = src_dir.join("nested_splits");
        let bad_args = vec![
            "tara_dataset_splitter".to_string(),
            "--input".to_string(),
            src_dir.to_string_lossy().to_string(),
            "--output".to_string(),
            bad_out.to_string_lossy().to_string(),
        ];
        assert!(run_splitter_cli(&bad_args).is_err());

        // 2. Default output (when --output omitted, including trailing slash on --input) places splits
        //    outside src_dir (<parent>/<stem>_splits) and SUCCEEDS without self-rejection
        let ok_args = vec![
            "tara_dataset_splitter".to_string(),
            "--input".to_string(),
            format!("{}/", src_dir.to_string_lossy()),
        ];
        assert!(run_splitter_cli(&ok_args).is_ok());

        let expected_default_out = dir.join("canonical_data_splits");
        let manifest_path = expected_default_out.join("split_manifest.json");
        assert!(manifest_path.exists());
        assert!(!src_dir.join("splits").exists());

        let manifest: Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
        assert_eq!(manifest["leakage_status"], "CLEAN");
        assert_eq!(manifest["semantic_conservation_verified"], true);
        assert_eq!(manifest["disjoint_partitions_verified"], true);
        assert!(!manifest["input_manifest_sha256"].as_str().unwrap().is_empty());
        assert!(manifest["pairs"]["train_val"]["is_clean"].as_bool().unwrap());
        assert!(manifest["pairs"]["train_test"]["is_clean"].as_bool().unwrap());
        assert!(manifest["pairs"]["val_test"]["is_clean"].as_bool().unwrap());

        let _ = fs::remove_dir_all(&dir);
    }
}
