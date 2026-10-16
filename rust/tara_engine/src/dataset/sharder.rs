//! Canonical JSONL Shard Writer with dynamic byte thresholds.

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

pub fn get_target_shard_bytes() -> u64 {
    std::env::var("TARA_SHARD_SIZE_MB")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&mb| mb > 0)
        .unwrap_or(500) * 1024 * 1024
}

pub struct ShardWriter {
    output_dir: PathBuf,
    current_shard_idx: usize,
    current_writer: Option<BufWriter<File>>,
    current_bytes: u64,
    pub max_shard_bytes: u64,
    pub total_records: usize,
}

impl ShardWriter {
    pub fn new(output_dir: PathBuf) -> std::io::Result<Self> {
        Self::with_start_index(output_dir, 1)
    }

    pub fn with_start_index(output_dir: PathBuf, start_shard_idx: usize) -> std::io::Result<Self> {
        fs::create_dir_all(&output_dir)?;
        let mut slf = Self {
            output_dir,
            current_shard_idx: start_shard_idx,
            current_writer: None,
            current_bytes: 0,
            max_shard_bytes: get_target_shard_bytes(),
            total_records: 0,
        };
        slf.open_next_shard()?;
        Ok(slf)
    }

    fn open_next_shard(&mut self) -> std::io::Result<()> {
        if let Some(mut w) = self.current_writer.take() {
            w.flush()?;
        }
        let shard_name = format!("tara_shard_{:03}.jsonl", self.current_shard_idx);
        let shard_path = self.output_dir.join(&shard_name);
        println!(">>> Opening shard: {:?}", shard_path);
        let (file, existing_bytes) = if shard_path.exists() {
            let metadata = fs::metadata(&shard_path)?;
            let f = OpenOptions::new().append(true).open(&shard_path)?;
            (f, metadata.len())
        } else {
            let f = File::create(&shard_path)?;
            (f, 0)
        };
        self.current_writer = Some(BufWriter::with_capacity(1024 * 1024, file));
        self.current_bytes = existing_bytes;
        self.current_shard_idx += 1;
        Ok(())
    }

    pub fn write_record(&mut self, record_json: &str) -> std::io::Result<()> {
        let line = format!("{}\n", record_json.trim());
        let line_len = line.len() as u64;

        if self.current_bytes + line_len > self.max_shard_bytes {
            self.open_next_shard()?;
        }

        if let Some(ref mut w) = self.current_writer {
            w.write_all(line.as_bytes())?;
            self.current_bytes += line_len;
            self.total_records += 1;
        }

        Ok(())
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        if let Some(ref mut w) = self.current_writer {
            w.flush()?;
        }
        Ok(())
    }

    pub fn finish(&mut self) -> std::io::Result<()> {
        self.flush()?;
        println!(">>> Sharding completed! Total records written: {}", self.total_records);
        Ok(())
    }
}
