//! Controlled training for candidate model versions.

use serde_json::Value;
use thiserror::Error;

use crate::trainer::{NativeSelfTrainer, TrainerError};

#[derive(Debug, Error)]
pub enum TrainCandidateError {
    #[error("trainer error: {0}")]
    Trainer(#[from] TrainerError),
}

#[derive(Debug, Clone, Default)]
pub struct TrainingOptions {
    pub dataset_dir: Option<String>,
    pub curriculum_path: Option<String>,
    pub candidate_dir: Option<String>,
    pub learning_rate: Option<f32>,
    pub batch_size: Option<usize>,
    pub max_steps: Option<usize>,
    pub warmup_steps: Option<usize>,
    pub weight_decay: Option<f32>,
    pub grad_clip_norm: Option<f32>,
    pub device: Option<String>,
    pub precision: Option<String>,
    pub checkpoint_dir: Option<String>,
    pub checkpoint_interval: Option<usize>,
    pub resume: bool,
    pub resume_from: Option<String>,
    pub max_memory_mb: Option<f64>,
}

/// Run a controlled training job for `max_epochs` on `dataset_path`.
pub fn run_controlled_training(
    model_dir: &str,
    repo_root: &str,
    max_epochs: usize,
) -> Result<Value, TrainCandidateError> {
    run_controlled_training_with_options(
        model_dir,
        repo_root,
        max_epochs,
        TrainingOptions::default(),
    )
}

/// Run a controlled training job with explicit hyperparameters and paths.
pub fn run_controlled_training_with_options(
    model_dir: &str,
    repo_root: &str,
    max_epochs: usize,
    options: TrainingOptions,
) -> Result<Value, TrainCandidateError> {
    let mut trainer = NativeSelfTrainer::new(model_dir, repo_root);
    if let Some(ds) = options.dataset_dir {
        trainer = trainer.with_dataset_dir(ds);
    }
    if let Some(cp) = options.curriculum_path {
        trainer = trainer.with_curriculum_path(cp);
    }
    if let Some(lr) = options.learning_rate {
        trainer = trainer.with_learning_rate(lr);
    }
    if let Some(bs) = options.batch_size {
        trainer = trainer.with_batch_size(bs);
    }
    if let Some(ms) = options.max_steps {
        trainer = trainer.with_max_steps(ms);
    }
    if let Some(ws) = options.warmup_steps {
        trainer = trainer.with_warmup_steps(ws);
    }
    if let Some(wd) = options.weight_decay {
        trainer = trainer.with_weight_decay(wd);
    }
    if let Some(gcn) = options.grad_clip_norm {
        trainer = trainer.with_grad_clip_norm(gcn);
    }
    if let Some(mb) = options.max_memory_mb {
        trainer = trainer.with_max_memory_mb(mb);
    }
    if let Some(dev_str) = options.device {
        let dev = dev_str
            .parse::<crate::trainer::TrainingDevice>()
            .map_err(TrainerError::Model)?;
        trainer = trainer.with_device(dev);
    }
    if let Some(prec_str) = options.precision {
        let prec = prec_str
            .parse::<crate::cuda::TrainingPrecision>()
            .map_err(TrainerError::Model)?;
        trainer = trainer.with_precision(prec);
    }
    if let Some(ref cp_dir) = options.checkpoint_dir {
        trainer = trainer.with_checkpoint_dir(cp_dir);
    }
    if let Some(cp_interval) = options.checkpoint_interval {
        trainer = trainer.with_checkpoint_interval(cp_interval);
    }
    if options.resume {
        trainer = trainer.with_resume(true);
    }
    if let Some(rf) = options.resume_from {
        trainer = trainer.with_resume_from(rf);
    }
    let result = trainer.run_full_self_learning_cycle_isolated(
        max_epochs,
        true,
        options.candidate_dir.as_deref(),
    )?;
    Ok(result)
}
