//! TARA Canonical Indic Dataset Sharder Binary
//!
//! 100% Pure Native Rust:
//! - Consolidates cleaned JSONL datasets into standardized sharded files:
//!   `tara_shard_001.jsonl`, `tara_shard_002.jsonl`, ...
//! - Configurable target shard size: ~500 MB per shard (~100,000 to 500,000 records)
//! - Cleanly writes to `storage/datasets/shards/`
//! - Preserves exact schema, metadata, and licensing intact.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use tara_engine::dataset::ShardWriter;

fn append_jsonl_to_shards(file_path: &Path, sharder: &mut ShardWriter) -> std::io::Result<usize> {
    println!("Sharding file: {:?}", file_path);
    let file = File::open(file_path)?;
    let reader = BufReader::with_capacity(1024 * 1024, file);
    let mut count = 0usize;

    for line_result in reader.lines() {
        let line = line_result?;
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            sharder.write_record(trimmed)?;
            count += 1;
        }
    }
    println!("Added {} records from {:?}", count, file_path);
    Ok(count)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!(" TARA INDIC CANONICAL DATASET V2 — CONTINUOUS SHARD CONSOLIDATOR");
    println!("================================================================================");

    let root = tara_engine::find_workspace_root()
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_default());

    let clean_dir = root.join("storage/datasets/dataset_v2_clean");
    let shards_dir = root.join("storage/datasets/shards");

    println!("Input Clean Directory: {:?}", clean_dir);
    println!("Output Shards Directory: {:?}", shards_dir);

    let mut sharder = ShardWriter::new(shards_dir)?;
    let mut total_sharded_records = 0usize;

    // 1. Classical Epics & Literature (High priority)
    let gita = clean_dir.join("sanskrit_bhagavad_gita_clean.jsonl");
    if gita.exists() {
        total_sharded_records += append_jsonl_to_shards(&gita, &mut sharder)?;
    }

    let ramayan = clean_dir.join("sanskrit_valmiki_ramayan_clean.jsonl");
    if ramayan.exists() {
        total_sharded_records += append_jsonl_to_shards(&ramayan, &mut sharder)?;
    }

    let mahabharat = clean_dir.join("sanskrit_vyasa_mahabharat_clean.jsonl");
    if mahabharat.exists() {
        total_sharded_records += append_jsonl_to_shards(&mahabharat, &mut sharder)?;
    }

    // 2. Parallel Translations
    let trans = clean_dir.join("translation_kannada_english_clean.jsonl");
    if trans.exists() {
        total_sharded_records += append_jsonl_to_shards(&trans, &mut sharder)?;
    }

    // 3. Sanskrit Corpus
    let sa_corp = clean_dir.join("sanskrit_indiccorp_clean.jsonl");
    if sa_corp.exists() {
        total_sharded_records += append_jsonl_to_shards(&sa_corp, &mut sharder)?;
    }

    // 4. Kannada Corpus
    let kn_corp = clean_dir.join("kannada_indiccorp_clean.jsonl");
    if kn_corp.exists() {
        total_sharded_records += append_jsonl_to_shards(&kn_corp, &mut sharder)?;
    }

    if total_sharded_records == 0 {
        return Err("Zero clean JSONL records found to shard!".into());
    }

    sharder.finish()?;
    Ok(())
}
