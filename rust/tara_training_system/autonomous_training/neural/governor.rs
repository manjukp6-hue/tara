//! Continual Learning Governor and Curriculum Priority Engine.
//!
//! Provides catastrophic forgetting prevention, degradation ratio evaluation,
//! Elastic Weight Consolidation (EWC) penalty calculation, and curriculum
//! priority scheduling for progressive knowledge acquisition without stubs.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Baseline anchor metrics for a foundational task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnchorMetric {
    pub task_id: String,
    pub baseline_loss: f64,
    pub baseline_accuracy: f64,
    pub importance_weight: f64, // Fisher Information proxy
    pub recorded_at_ms: u64,
}

/// Evaluation result comparing newly evaluated metrics against anchor baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationEvaluation {
    pub task_id: String,
    pub current_loss: f64,
    pub baseline_loss: f64,
    pub loss_degradation_ratio: f64,
    pub accuracy_delta: f64,
    pub catastrophic_forgetting_detected: bool,
    pub ewc_penalty: f64,
    pub action_recommended: String, // "PROCEED", "APPLY_EWC_REGULARIZATION", "ROLLBACK"
}

pub const DEFAULT_MAX_TOLERATED_DEGRADATION: f64 = 0.15;
pub const DEFAULT_EWC_LAMBDA: f64 = 100.0;
pub const DEFAULT_CURRICULUM_WEIGHT_ERROR: f64 = 0.40;
pub const DEFAULT_CURRICULUM_WEIGHT_NOVELTY: f64 = 0.35;
pub const DEFAULT_CURRICULUM_WEIGHT_READINESS: f64 = 0.25;
pub const DEFAULT_CURRICULUM_MASTERY_THRESHOLD: f64 = 0.70;
pub const DEFAULT_CURRICULUM_ELIGIBILITY_THRESHOLD: f64 = 0.50;

pub fn resolve_max_tolerated_degradation() -> f64 {
    std::env::var("TARA_MAX_TOLERATED_DEGRADATION")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_TOLERATED_DEGRADATION)
}

pub fn resolve_ewc_lambda() -> f64 {
    std::env::var("TARA_EWC_LAMBDA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_EWC_LAMBDA)
}

pub fn resolve_curriculum_weight_error() -> f64 {
    std::env::var("TARA_CURRICULUM_WEIGHT_ERROR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_CURRICULUM_WEIGHT_ERROR)
}

pub fn resolve_curriculum_weight_novelty() -> f64 {
    std::env::var("TARA_CURRICULUM_WEIGHT_NOVELTY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_CURRICULUM_WEIGHT_NOVELTY)
}

pub fn resolve_curriculum_weight_readiness() -> f64 {
    std::env::var("TARA_CURRICULUM_WEIGHT_READINESS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_CURRICULUM_WEIGHT_READINESS)
}

pub fn resolve_curriculum_mastery_threshold() -> f64 {
    std::env::var("TARA_CURRICULUM_MASTERY_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_CURRICULUM_MASTERY_THRESHOLD)
}

pub fn resolve_curriculum_eligibility_threshold() -> f64 {
    std::env::var("TARA_CURRICULUM_ELIGIBILITY_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_CURRICULUM_ELIGIBILITY_THRESHOLD)
}

/// Continual Learning Governor protecting against catastrophic forgetting.
pub struct ContinualLearningGovernor {
    anchors: Arc<Mutex<HashMap<String, AnchorMetric>>>,
    max_tolerated_degradation: f64, // Default 0.15 (15%)
    ewc_lambda: f64,                // Regularization coefficient
}

impl Default for ContinualLearningGovernor {
    fn default() -> Self {
        Self::new(
            resolve_max_tolerated_degradation(),
            resolve_ewc_lambda(),
        )
    }
}

impl ContinualLearningGovernor {
    pub fn new(max_tolerated_degradation: f64, ewc_lambda: f64) -> Self {
        Self {
            anchors: Arc::new(Mutex::new(HashMap::new())),
            max_tolerated_degradation,
            ewc_lambda,
        }
    }

    /// Registers or updates an anchor metric for a foundational task.
    pub fn register_anchor(
        &self,
        task_id: &str,
        baseline_loss: f64,
        baseline_accuracy: f64,
        importance_weight: f64,
    ) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let metric = AnchorMetric {
            task_id: task_id.to_string(),
            baseline_loss,
            baseline_accuracy,
            importance_weight,
            recorded_at_ms: now,
        };

        let mut anchors = self.anchors.lock().unwrap();
        anchors.insert(task_id.to_string(), metric);
    }

    /// Evaluates current task performance against baseline anchor.
    pub fn evaluate_task_degradation(
        &self,
        task_id: &str,
        current_loss: f64,
        current_accuracy: f64,
        param_distance_sq: f64,
    ) -> DegradationEvaluation {
        let anchors = self.anchors.lock().unwrap();
        if let Some(anchor) = anchors.get(task_id) {
            let baseline_loss = anchor.baseline_loss.max(1e-9);
            // Degradation ratio = (current_loss - baseline_loss) / baseline_loss
            let loss_degradation_ratio = (current_loss - baseline_loss) / baseline_loss;
            let accuracy_delta = current_accuracy - anchor.baseline_accuracy;

            // EWC Penalty = 0.5 * lambda * F * ||theta - theta*||^2
            let ewc_penalty = 0.5 * self.ewc_lambda * anchor.importance_weight * param_distance_sq;

            let catastrophic = loss_degradation_ratio > self.max_tolerated_degradation;

            let action_recommended =
                if loss_degradation_ratio > (self.max_tolerated_degradation * 2.0) {
                    "ROLLBACK".to_string()
                } else if catastrophic {
                    "APPLY_EWC_REGULARIZATION".to_string()
                } else {
                    "PROCEED".to_string()
                };

            DegradationEvaluation {
                task_id: task_id.to_string(),
                current_loss,
                baseline_loss,
                loss_degradation_ratio,
                accuracy_delta,
                catastrophic_forgetting_detected: catastrophic,
                ewc_penalty,
                action_recommended,
            }
        } else {
            DegradationEvaluation {
                task_id: task_id.to_string(),
                current_loss,
                baseline_loss: current_loss,
                loss_degradation_ratio: 0.0,
                accuracy_delta: 0.0,
                catastrophic_forgetting_detected: false,
                ewc_penalty: 0.0,
                action_recommended: "PROCEED".to_string(),
            }
        }
    }

    /// Returns list of all registered anchor tasks.
    pub fn get_anchors(&self) -> Vec<AnchorMetric> {
        let anchors = self.anchors.lock().unwrap();
        anchors.values().cloned().collect()
    }
}

/// A curriculum learning candidate item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumItem {
    pub item_id: String,
    pub title: String,
    pub domain: String,
    pub difficulty: f64, // 0.0 (easiest) to 1.0 (hardest)
    pub historical_error_rate: f64,
    pub novelty_score: f64,
    pub prerequisites: Vec<String>,
}

/// Curriculum Priority Engine scoring and ranking learning items.
pub struct CurriculumPriorityEngine {
    mastered_concepts: Arc<Mutex<HashMap<String, f64>>>, // concept_id -> mastery_level (0.0 to 1.0)
    pub w_error: f64,
    pub w_novelty: f64,
    pub w_readiness: f64,
    pub mastery_threshold: f64,
    pub eligibility_threshold: f64,
}

impl Default for CurriculumPriorityEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl CurriculumPriorityEngine {
    pub fn new() -> Self {
        Self {
            mastered_concepts: Arc::new(Mutex::new(HashMap::new())),
            w_error: resolve_curriculum_weight_error(),
            w_novelty: resolve_curriculum_weight_novelty(),
            w_readiness: resolve_curriculum_weight_readiness(),
            mastery_threshold: resolve_curriculum_mastery_threshold(),
            eligibility_threshold: resolve_curriculum_eligibility_threshold(),
        }
    }

    /// Sets mastery level for a concept.
    pub fn record_mastery(&self, concept_id: &str, level: f64) {
        let mut mastered = self.mastered_concepts.lock().unwrap();
        mastered.insert(concept_id.to_string(), level.clamp(0.0, 1.0));
    }

    /// Evaluates prerequisite readiness: fraction of prereqs with mastery >= mastery_threshold.
    pub fn evaluate_readiness(&self, prerequisites: &[String]) -> f64 {
        if prerequisites.is_empty() {
            return 1.0;
        }

        let mastered = self.mastered_concepts.lock().unwrap();
        let ready_count = prerequisites
            .iter()
            .filter(|&req| mastered.get(req).copied().unwrap_or(0.0) >= self.mastery_threshold)
            .count();

        (ready_count as f64) / (prerequisites.len() as f64)
    }

    /// Scores a curriculum item using multi-factor priority:
    /// Priority = (w_error * error_rate) + (w_novelty * novelty) + (w_readiness * readiness)
    /// Disqualifies items if prerequisite readiness is below eligibility_threshold (cannot learn advanced before basics).
    pub fn score_item(&self, item: &CurriculumItem) -> (f64, bool) {
        let readiness = self.evaluate_readiness(&item.prerequisites);
        let eligible = readiness >= self.eligibility_threshold;

        let raw_priority = (self.w_error * item.historical_error_rate.clamp(0.0, 1.0))
            + (self.w_novelty * item.novelty_score.clamp(0.0, 1.0))
            + (self.w_readiness * readiness);

        (raw_priority, eligible)
    }

    /// Ranks a candidate list of curriculum items, filtering out ineligible ones.
    pub fn rank_curriculum(&self, items: &[CurriculumItem]) -> Vec<(CurriculumItem, f64)> {
        let mut scored: Vec<(CurriculumItem, f64)> = items
            .iter()
            .filter_map(|item| {
                let (score, eligible) = self.score_item(item);
                if eligible {
                    Some((item.clone(), score))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continual_learning_degradation_and_ewc() {
        let governor = ContinualLearningGovernor::new(0.15, 100.0);
        governor.register_anchor("core_nlu", 0.50, 0.95, 2.0);

        // Case 1: Minimal change -> PROCEED
        let eval_normal = governor.evaluate_task_degradation("core_nlu", 0.52, 0.94, 0.001);
        assert!(!eval_normal.catastrophic_forgetting_detected);
        assert_eq!(eval_normal.action_recommended, "PROCEED");

        // Case 2: 20% loss spike -> APPLY_EWC_REGULARIZATION
        let eval_degraded = governor.evaluate_task_degradation("core_nlu", 0.60, 0.88, 0.05);
        assert!(eval_degraded.catastrophic_forgetting_detected);
        assert_eq!(eval_degraded.action_recommended, "APPLY_EWC_REGULARIZATION");
        assert!(eval_degraded.ewc_penalty > 0.0);

        // Case 3: 50% loss spike -> ROLLBACK
        let eval_severe = governor.evaluate_task_degradation("core_nlu", 0.90, 0.70, 0.1);
        assert!(eval_severe.catastrophic_forgetting_detected);
        assert_eq!(eval_severe.action_recommended, "ROLLBACK");
    }

    #[test]
    fn test_curriculum_priority_ranking() {
        let engine = CurriculumPriorityEngine::new();
        engine.record_mastery("basic_syntax", 0.95);
        engine.record_mastery("functions", 0.85);
        // "concurrency" not mastered

        let items = vec![
            CurriculumItem {
                item_id: "closures".into(),
                title: "Rust Closures".into(),
                domain: "rust".into(),
                difficulty: 0.4,
                historical_error_rate: 0.3,
                novelty_score: 0.7,
                prerequisites: vec!["basic_syntax".into(), "functions".into()],
            },
            CurriculumItem {
                item_id: "async_await".into(),
                title: "Async Programming".into(),
                domain: "rust".into(),
                difficulty: 0.9,
                historical_error_rate: 0.6,
                novelty_score: 0.9,
                prerequisites: vec!["concurrency".into()], // Ineligible!
            },
        ];

        let ranked = engine.rank_curriculum(&items);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].0.item_id, "closures");
    }
}
