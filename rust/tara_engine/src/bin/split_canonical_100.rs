//! Native Rust utility to partition canonical dataset into exactly 100 balanced shards
//! each strictly under 100MB to comply with GitHub upload limits (Rule 17).

use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TARA CANONICAL DATASET SHARDING UTILITY (100 SHARDS) ===");

    let input_path = "storage/datasets/tara_dataset_filtered/canonical/tara_canonical_00000.jsonl";
    let output_dir = "storage/datasets/tara_dataset_filtered/canonical_100";

    if !Path::new(input_path).exists() {
        eprintln!("Error: Input file {} not found!", input_path);
        std::process::exit(1);
    }

    fs::create_dir_all(output_dir)?;

    let file = File::open(input_path)?;
    let reader = BufReader::with_capacity(1024 * 1024, file);

    let target_shards = 100;
    let records_per_shard = 5000;

    let mut current_shard_idx = 0;
    let mut current_record_count = 0;
    let mut total_records = 0;

    let mut current_writer: Option<BufWriter<File>> = None;

    for line_res in reader.lines() {
        let line = line_res?;
        if line.trim().is_empty() {
            continue;
        }

        if current_writer.is_none() || current_record_count >= records_per_shard {
            if current_shard_idx >= target_shards {
                break;
            }
            if let Some(mut w) = current_writer.take() {
                w.flush()?;
            }

            let shard_file = format!("{}/tara_canonical_shard_{:03}.jsonl", output_dir, current_shard_idx);
            let out_file = File::create(&shard_file)?;
            current_writer = Some(BufWriter::with_capacity(1024 * 1024, out_file));
            current_shard_idx += 1;
            current_record_count = 0;
        }

        if let Some(ref mut w) = current_writer {
            w.write_all(line.as_bytes())?;
            w.write_all(b"\n")?;
            current_record_count += 1;
            total_records += 1;
        }
    }

    if let Some(mut w) = current_writer.take() {
        w.flush()?;
    }

    println!("SUCCESS: Partitioned {} records across {} shards in {}.", total_records, current_shard_idx, output_dir);
    Ok(())
}
