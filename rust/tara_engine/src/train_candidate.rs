//! Controlled training for candidate model versions.

use serde_json::{json, Value};
use thiserror::Error;

use crate::trainer::{NativeSelfTrainer, TrainerError};

#[derive(Debug, Error)]
pub enum TrainCandidateError {
    #[error("trainer error: {0}")]
    Trainer(#[from] TrainerError),
}

/// Run a controlled training job for `max_epochs` on `dataset_path`.
pub fn run_controlled_training(
    model_dir: &str,
    repo_root: &str,
    max_epochs: usize,
) -> Result<Value, TrainCandidateError> {
    let trainer = NativeSelfTrainer::new(model_dir, repo_root);
    let result = trainer.run_full_self_learning_cycle(max_epochs, true)?;
    Ok(result)
}
