//! # TARA Core Curriculum Engine (100% Pure Native Rust)
//!
//! Location: `rust/tara_training_system/curriculum_engine/mod.rs`
//!
//! Mandatory Directives:
//! 1. Auto-Receive: Directly accepts training-ready data from Dataset Engine.
//! 2. Model-Specific Curriculum: Logically manages separate tracks:
//!    - Neural Model Curriculum (Causal LM, prompt/completion sequence training)
//!    - World Model Curriculum (Ontology triples, fact graphs, state transitions)
//! 3. Auto-Ordering: Sorts and arranges samples by difficulty, length, and foundational dependencies.
//! 4. Auto-Progression: Manages dynamic progression across $N$ extensible stages (Rule 15: no hardcoded caps).
//! 5. Auto-Sampling: Generates stratified mini-batches based on active stage focus.
//! 6. Continuous Update: Incorporates newly arrived datasets without restarting training milestones.
//! 7. Zero Hardcoded Limits: Dynamic stage counts, parameter-driven thresholds.

use crate::dataset_engine::{DatasetEngine, NeuralRecord, TargetModel, WorldModelRecord};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::{Arc, RwLock};

/// Dynamic curriculum stage definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CurriculumStage {
    pub stage_index: usize,
    pub name: String,
    pub min_difficulty: f32,
    pub max_difficulty: f32,
    pub target_loss_threshold: f32,
    pub sample_indices: Vec<usize>,
}

/// Logical curriculum track for Neural Model training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralCurriculum {
    pub ordered_records: Vec<NeuralRecord>,
    pub stages: Vec<CurriculumStage>,
    pub active_stage_index: usize,
    pub total_samples: usize,
    pub average_difficulty: f32,
    pub stage_completion_progress: f32,
}

impl Default for NeuralCurriculum {
    fn default() -> Self {
        Self {
            ordered_records: Vec::new(),
            stages: Vec::new(),
            active_stage_index: 0,
            total_samples: 0,
            average_difficulty: 0.0,
            stage_completion_progress: 0.0,
        }
    }
}

/// Logical curriculum track for World Model training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModelCurriculum {
    pub ordered_records: Vec<WorldModelRecord>,
    pub stages: Vec<CurriculumStage>,
    pub active_stage_index: usize,
    pub total_facts: usize,
    pub average_confidence: f32,
    pub stage_completion_progress: f32,
}

impl Default for WorldModelCurriculum {
    fn default() -> Self {
        Self {
            ordered_records: Vec::new(),
            stages: Vec::new(),
            active_stage_index: 0,
            total_facts: 0,
            average_confidence: 0.0,
            stage_completion_progress: 0.0,
        }
    }
}

/// Mini-batch sampled from the curriculum for training steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumBatch {
    pub stage_index: usize,
    pub stage_name: String,
    pub model_type: TargetModel,
    pub neural_samples: Vec<NeuralRecord>,
    pub world_samples: Vec<WorldModelRecord>,
}

/// Summary report returned after curriculum update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumReport {
    pub neural_records_count: usize,
    pub neural_stages_count: usize,
    pub neural_active_stage: usize,
    pub world_records_count: usize,
    pub world_stages_count: usize,
    pub world_active_stage: usize,
    pub duration_ms: u64,
}

/// Configuration for Curriculum Engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumConfig {
    /// Number of dynamic stages to partition data into ($N \ge 1$, default 5).
    pub target_stage_count: usize,
    /// Base loss threshold for advancing stage.
    pub base_stage_loss_threshold: f32,
    /// Sampling temperature: 0.0 = strictly current stage, 1.0 = balanced across all stages.
    pub stage_sampling_mix_rate: f32,
}

impl Default for CurriculumConfig {
    fn default() -> Self {
        Self {
            target_stage_count: 5,
            base_stage_loss_threshold: 1.5,
            stage_sampling_mix_rate: 0.15,
        }
    }
}

/// Main Curriculum Engine.
pub struct CurriculumEngine {
    pub config: CurriculumConfig,
    neural_curriculum: Arc<RwLock<NeuralCurriculum>>,
    world_curriculum: Arc<RwLock<WorldModelCurriculum>>,
}

impl CurriculumEngine {
    pub fn new(config: CurriculumConfig) -> Self {
        Self {
            config,
            neural_curriculum: Arc::new(RwLock::new(NeuralCurriculum::default())),
            world_curriculum: Arc::new(RwLock::new(WorldModelCurriculum::default())),
        }
    }

    /// Auto-receives training-ready data from DatasetEngine and compiles/updates
    /// model-specific curricula.
    pub fn receive_and_update(&self, dataset_engine: &DatasetEngine) -> CurriculumReport {
        let start = std::time::Instant::now();

        let raw_neural = dataset_engine.get_training_ready_neural();
        let raw_world = dataset_engine.get_training_ready_world();

        // 1. Auto-Order and Partition Neural Curriculum
        let (ordered_neural, neural_stages, avg_diff) = self.build_neural_curriculum(raw_neural);
        let (neural_count, neural_stages_len, neural_active) = {
            let mut curr = self.neural_curriculum.write().unwrap();
            let prev_stage = curr.active_stage_index;
            curr.ordered_records = ordered_neural;
            curr.stages = neural_stages;
            curr.total_samples = curr.ordered_records.len();
            curr.average_difficulty = avg_diff;
            // Preserve progression state clamped to valid range
            curr.active_stage_index = prev_stage.min(curr.stages.len().saturating_sub(1));
            (curr.total_samples, curr.stages.len(), curr.active_stage_index)
        };

        // 2. Auto-Order and Partition World Model Curriculum
        let (ordered_world, world_stages, avg_conf) = self.build_world_curriculum(raw_world);
        let (world_count, world_stages_len, world_active) = {
            let mut curr = self.world_curriculum.write().unwrap();
            let prev_stage = curr.active_stage_index;
            curr.ordered_records = ordered_world;
            curr.stages = world_stages;
            curr.total_facts = curr.ordered_records.len();
            curr.average_confidence = avg_conf;
            curr.active_stage_index = prev_stage.min(curr.stages.len().saturating_sub(1));
            (curr.total_facts, curr.stages.len(), curr.active_stage_index)
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        CurriculumReport {
            neural_records_count: neural_count,
            neural_stages_count: neural_stages_len,
            neural_active_stage: neural_active,
            world_records_count: world_count,
            world_stages_count: world_stages_len,
            world_active_stage: world_active,
            duration_ms,
        }
    }

    /// Auto-orders neural records from foundational to advanced difficulty.
    fn build_neural_curriculum(
        &self,
        mut records: Vec<NeuralRecord>,
    ) -> (Vec<NeuralRecord>, Vec<CurriculumStage>, f32) {
        if records.is_empty() {
            return (Vec::new(), Vec::new(), 0.0);
        }

        // Auto-Ordering: Canonical total ordering:
        // 1. difficulty_score ascending (foundational concepts first)
        // 2. tokens_estimate ascending
        // 3. domain ascending
        // 4. prompt ascending
        // 5. id ascending
        records.sort_by(|a, b| {
            a.difficulty_score
                .partial_cmp(&b.difficulty_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.tokens_estimate.cmp(&b.tokens_estimate))
                .then_with(|| a.domain.cmp(&b.domain))
                .then_with(|| a.prompt.cmp(&b.prompt))
                .then_with(|| a.id.cmp(&b.id))
        });

        let total_diff: f32 = records.iter().map(|r| r.difficulty_score).sum();
        let avg_diff = total_diff / records.len() as f32;

        // Auto-Progression: Dynamically partition into $N$ extensible stages
        let stage_count = self.config.target_stage_count.min(records.len()).max(1);

        let mut stages = Vec::new();
        for s in 0..stage_count {
            let start_idx = (s * records.len()) / stage_count;
            let end_idx = ((s + 1) * records.len()) / stage_count;
            if start_idx >= end_idx {
                continue;
            }
            let indices: Vec<usize> = (start_idx..end_idx).collect();

            let min_d = records[start_idx].difficulty_score;
            let max_d = records[end_idx - 1].difficulty_score;

            let stage_name = format!("Neural_Stage_{}: {:.2}-{:.2}", s + 1, min_d, max_d);
            let loss_target = self.config.base_stage_loss_threshold / ((s + 1) as f32).sqrt();

            stages.push(CurriculumStage {
                stage_index: s,
                name: stage_name,
                min_difficulty: min_d,
                max_difficulty: max_d,
                target_loss_threshold: loss_target,
                sample_indices: indices,
            });
        }

        (records, stages, avg_diff)
    }

    /// Auto-orders world model facts by confidence and relational dependencies.
    fn build_world_curriculum(
        &self,
        mut records: Vec<WorldModelRecord>,
    ) -> (Vec<WorldModelRecord>, Vec<CurriculumStage>, f32) {
        if records.is_empty() {
            return (Vec::new(), Vec::new(), 0.0);
        }

        // Auto-Ordering: Canonical total ordering:
        // 1. confidence descending (highest-confidence core facts first)
        // 2. subject ascending
        // 3. predicate ascending
        // 4. object ascending
        // 5. id ascending
        records.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.subject.cmp(&b.subject))
                .then_with(|| a.predicate.cmp(&b.predicate))
                .then_with(|| a.object.cmp(&b.object))
                .then_with(|| a.id.cmp(&b.id))
        });

        let total_conf: f32 = records.iter().map(|r| r.confidence).sum();
        let avg_conf = total_conf / records.len() as f32;

        let stage_count = self.config.target_stage_count.min(records.len()).max(1);

        let mut stages = Vec::new();
        for s in 0..stage_count {
            let start_idx = (s * records.len()) / stage_count;
            let end_idx = ((s + 1) * records.len()) / stage_count;
            if start_idx >= end_idx {
                continue;
            }
            let indices: Vec<usize> = (start_idx..end_idx).collect();

            let min_c = records[end_idx - 1].confidence;
            let max_c = records[start_idx].confidence;

            let stage_name = format!("WorldModel_Stage_{}: Conf {:.2}-{:.2}", s + 1, min_c, max_c);
            let loss_target = self.config.base_stage_loss_threshold / ((s + 1) as f32).sqrt();

            stages.push(CurriculumStage {
                stage_index: s,
                name: stage_name,
                min_difficulty: 1.0 - max_c,
                max_difficulty: 1.0 - min_c,
                target_loss_threshold: loss_target,
                sample_indices: indices,
            });
        }

        (records, stages, avg_conf)
    }

    /// Auto-samples a training mini-batch according to active curriculum stage.
    pub fn sample_batch(&self, batch_size: usize, model_type: TargetModel) -> CurriculumBatch {
        let bsz = batch_size.max(1);

        match model_type {
            TargetModel::NeuralModel => {
                let curr = self.neural_curriculum.read().unwrap();
                let (stage_idx, stage_name, samples) = self.sample_neural_batch(&curr, bsz);
                CurriculumBatch {
                    stage_index: stage_idx,
                    stage_name,
                    model_type: TargetModel::NeuralModel,
                    neural_samples: samples,
                    world_samples: Vec::new(),
                }
            }
            TargetModel::WorldModel => {
                let curr = self.world_curriculum.read().unwrap();
                let (stage_idx, stage_name, samples) = self.sample_world_batch(&curr, bsz);
                CurriculumBatch {
                    stage_index: stage_idx,
                    stage_name,
                    model_type: TargetModel::WorldModel,
                    neural_samples: Vec::new(),
                    world_samples: samples,
                }
            }
            TargetModel::Both => {
                let n_curr = self.neural_curriculum.read().unwrap();
                let w_curr = self.world_curriculum.read().unwrap();
                let half_bsz = (bsz / 2).max(1);
                let (n_stage, n_name, n_samples) = self.sample_neural_batch(&n_curr, half_bsz);
                let (_, _, w_samples) = self.sample_world_batch(&w_curr, half_bsz);
                CurriculumBatch {
                    stage_index: n_stage,
                    stage_name: n_name,
                    model_type: TargetModel::Both,
                    neural_samples: n_samples,
                    world_samples: w_samples,
                }
            }
        }
    }

    fn sample_neural_batch(
        &self,
        curr: &NeuralCurriculum,
        count: usize,
    ) -> (usize, String, Vec<NeuralRecord>) {
        if curr.ordered_records.is_empty() || curr.stages.is_empty() {
            return (0, "Empty".to_string(), Vec::new());
        }

        let stage_idx = curr.active_stage_index.min(curr.stages.len() - 1);
        let stage = &curr.stages[stage_idx];

        let mut samples = Vec::new();
        let mut picked_indices = HashSet::new();

        // 1. Prioritize active stage records first
        if !stage.sample_indices.is_empty() {
            for &rec_idx in &stage.sample_indices {
                if samples.len() >= count {
                    break;
                }
                if rec_idx < curr.ordered_records.len() && picked_indices.insert(rec_idx) {
                    samples.push(curr.ordered_records[rec_idx].clone());
                }
            }
        }

        // 2. Stratified curriculum sampling: supplement with distinct records from all stages (curriculum replay)
        if samples.len() < count && !curr.ordered_records.is_empty() {
            for rec_idx in 0..curr.ordered_records.len() {
                if samples.len() >= count {
                    break;
                }
                if picked_indices.insert(rec_idx) {
                    samples.push(curr.ordered_records[rec_idx].clone());
                }
            }
        }

        // 3. Fallback: only if total records in curriculum < count, cycle to reach count
        if samples.len() < count && !curr.ordered_records.is_empty() {
            let total = curr.ordered_records.len();
            while samples.len() < count {
                let cycle_idx = samples.len() % total;
                samples.push(curr.ordered_records[cycle_idx].clone());
            }
        }

        (stage_idx, stage.name.clone(), samples)
    }

    fn sample_world_batch(
        &self,
        curr: &WorldModelCurriculum,
        count: usize,
    ) -> (usize, String, Vec<WorldModelRecord>) {
        if curr.ordered_records.is_empty() || curr.stages.is_empty() {
            return (0, "Empty".to_string(), Vec::new());
        }

        let stage_idx = curr.active_stage_index.min(curr.stages.len() - 1);
        let stage = &curr.stages[stage_idx];

        let mut samples = Vec::new();
        let mut picked_indices = HashSet::new();

        // 1. Prioritize active stage records first
        if !stage.sample_indices.is_empty() {
            for &rec_idx in &stage.sample_indices {
                if samples.len() >= count {
                    break;
                }
                if rec_idx < curr.ordered_records.len() && picked_indices.insert(rec_idx) {
                    samples.push(curr.ordered_records[rec_idx].clone());
                }
            }
        }

        // 2. Stratified curriculum sampling: supplement with distinct records from all stages (curriculum replay)
        if samples.len() < count && !curr.ordered_records.is_empty() {
            for rec_idx in 0..curr.ordered_records.len() {
                if samples.len() >= count {
                    break;
                }
                if picked_indices.insert(rec_idx) {
                    samples.push(curr.ordered_records[rec_idx].clone());
                }
            }
        }

        // 3. Fallback: only if total records in curriculum < count, cycle to reach count
        if samples.len() < count && !curr.ordered_records.is_empty() {
            let total = curr.ordered_records.len();
            while samples.len() < count {
                let cycle_idx = samples.len() % total;
                samples.push(curr.ordered_records[cycle_idx].clone());
            }
        }

        (stage_idx, stage.name.clone(), samples)
    }

    /// Auto-Progression: Advances active curriculum stage when training milestone is reached.
    pub fn advance_stage(&self, model_type: TargetModel) -> bool {
        match model_type {
            TargetModel::NeuralModel | TargetModel::Both => {
                let mut curr = self.neural_curriculum.write().unwrap();
                if curr.active_stage_index + 1 < curr.stages.len() {
                    curr.active_stage_index += 1;
                    curr.stage_completion_progress = 0.0;
                    true
                } else {
                    false
                }
            }
            TargetModel::WorldModel => {
                let mut curr = self.world_curriculum.write().unwrap();
                if curr.active_stage_index + 1 < curr.stages.len() {
                    curr.active_stage_index += 1;
                    curr.stage_completion_progress = 0.0;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Snapshot of current curriculum state.
    pub fn get_curriculum_snapshots(&self) -> (NeuralCurriculum, WorldModelCurriculum) {
        (
            self.neural_curriculum.read().unwrap().clone(),
            self.world_curriculum.read().unwrap().clone(),
        )
    }
}
