//! TARA Core Dataset Sub-Modules (100% Native Rust)
//!
//! Consolidated Engine Architecture:
//! - `compiler`: Dataset compilation and secret scrubbing.
//! - `reader`: Shard streaming and reading for neural training.
//! - `splitter`: Cross-split partitioning with leakage controls.
//! - `tokenizer_stage`: Tokenization with metadata preservation.
//! - `vocab_builder`: Character-frequency dynamic vocabulary builder.

pub mod compiler;
#[path = "../../../tara_training_system/shared_training_infrastructure/reader.rs"]
pub mod reader;
pub mod splitter;
pub mod tokenizer_stage;
pub mod vocab_builder;

pub use compiler::{compile_unified_dataset, DynamicDatasetCompiler};
pub use reader::{
    DatasetStreamLifecycle, ExpandableDatasetReader, MalformedRecordPolicy, ReaderError,
    ReaderTelemetry, ShardAvailability, ShardStreamingMode, TrainingSample,
};
pub use splitter::{
    compute_file_sha256, DatasetSplitter, LeakageChecker, SplitRatio, SplitReport, SplitterError,
};
pub use tokenizer_stage::{TokenizationManifest, TokenizedRecord, TokenizerStage, TrainingFormat};
pub use vocab_builder::VocabBuilder;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;

    #[test]
    fn test_dataset_mod_public_reexports_and_splitter_to_reader_pipeline() {
        let dir = std::env::temp_dir().join(format!(
            "tara_dataset_mod_api_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let src_dir = dir.join("canonical_src");
        let split_dir = dir.join("canonical_splits");
        fs::create_dir_all(&src_dir).unwrap();

        let shard = src_dir.join("shard_0.jsonl");
        {
            let mut f = File::create(&shard).unwrap();
            for i in 0..30 {
                writeln!(
                    f,
                    "{{\"id\":\"id_{i}\",\"input\":\"prompt_{i}\",\"output\":\"answer_content_{i}\",\"curriculum_tier\":\"college\",\"curriculum_order\":4}}"
                )
                .unwrap();
            }
        }

        // 1. Verify DatasetSplitter and SplitRatio via dataset::* re-exports
        let ratio = SplitRatio::new(0.80, 0.10, 0.10).unwrap();
        let split_report: SplitReport = DatasetSplitter::split(&src_dir, &split_dir, ratio).unwrap();
        assert_eq!(split_report.total_samples, 30);

        // 2. Verify LeakageChecker via dataset::* re-exports
        let leak_rep =
            LeakageChecker::check_leakage(&split_report.train_path, &split_report.val_path).unwrap();
        assert!(leak_rep.is_clean);

        // 3. Verify ExpandableDatasetReader, DatasetStreamLifecycle, MalformedRecordPolicy,
        //    ShardStreamingMode, ReaderTelemetry, and ReaderError via dataset::* re-exports
        let mut empty_reader = ExpandableDatasetReader::new(ShardStreamingMode::Sequential);
        assert!(matches!(empty_reader.next_sample(), Err(ReaderError::NoShards)));

        let mut reader = ExpandableDatasetReader::new(ShardStreamingMode::Interleaved)
            .with_lifecycle(DatasetStreamLifecycle::LiveAppendWait)
            .with_malformed_policy(MalformedRecordPolicy::Strict)
            .with_auto_rewind(true);
        assert_eq!(reader.lifecycle(), DatasetStreamLifecycle::LiveAppendWait);

        reader.add_shard(&split_report.train_path).unwrap();
        reader.add_shard(&split_report.val_path).unwrap();
        reader.add_shard(&split_report.test_path).unwrap();

        let mut total_read = 0usize;
        while let Some(sample) = reader.next_sample().unwrap() {
            let _typed: TrainingSample = sample;
            assert_eq!(_typed.curriculum_order, 4);
            total_read += 1;
        }
        assert_eq!(total_read, 30);

        let tel: ReaderTelemetry = reader.telemetry();
        assert_eq!(tel.total_shards, 3);
        assert_eq!(tel.temporarily_eof_shards, 3);
        assert_eq!(tel.lifetime_yielded, 30);

        // 4. Verify TokenizerStage, TrainingFormat, and VocabBuilder re-exports compile cleanly
        let _vb = VocabBuilder::new(&src_dir, &split_dir, 256);
        let _fmt = TrainingFormat::StandardChatml;
        assert_eq!(_fmt, TrainingFormat::default());

        let _ = fs::remove_dir_all(&dir);
    }
}
