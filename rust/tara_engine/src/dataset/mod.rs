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
    ExpandableDatasetReader, MalformedRecordPolicy, ReaderError, ReaderTelemetry,
    ShardAvailability, ShardStreamingMode, TrainingSample,
};
pub use splitter::{compute_file_sha256, DatasetSplitter, LeakageChecker, SplitRatio, SplitReport};
pub use tokenizer_stage::{TokenizerStage, TrainingFormat, TokenizationManifest, TokenizedRecord};
pub use vocab_builder::VocabBuilder;
