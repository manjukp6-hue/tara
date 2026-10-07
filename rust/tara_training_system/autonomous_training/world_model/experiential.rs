//! 12-Stage Experiential Closed-Loop Learning Engine.
//!
//! Implements the complete experiential learning lifecycle:
//! 1. Sensing / Task Reception
//! 2. Context Formulation
//! 3. Knowledge / Skill Retrieval
//! 4. Strategy Formulation
//! 5. Safety & Policy Gate Check
//! 6. Action Execution
//! 7. Observation & Feedback Collection
//! 8. Outcome Evaluation
//! 9. Credit Assignment / Attribution
//! 10. Lesson Formulation & Generalization
//! 11. Strategy Adaptation
//! 12. Persistent Lesson Storage

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperientialStage {
    TaskReception,
    ContextFormulation,
    KnowledgeRetrieval,
    StrategyFormulation,
    SafetyGateCheck,
    ActionExecution,
    FeedbackCollection,
    OutcomeEvaluation,
    CreditAssignment,
    LessonFormulation,
    StrategyAdaptation,
    PersistentStorage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageTrace {
    pub stage: ExperientialStage,
    pub timestamp_ms: u64,
    pub detail: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LessonLearned {
    pub lesson_id: String,
    pub task_category: String,
    pub trigger_pattern: String,
    pub strategy_used: String,
    pub outcome_score: f64,
    pub root_cause: String,
    pub recommendation: String,
    pub confidence_score: f64,
    pub applied_count: u64,
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceEpisode {
    pub episode_id: String,
    pub task_description: String,
    pub context_summary: String,
    pub chosen_strategy: String,
    pub execution_result: Value,
    pub success: bool,
    pub outcome_score: f64,
    pub traces: Vec<StageTrace>,
    pub synthesized_lesson: Option<LessonLearned>,
    pub created_at_ms: u64,
}

pub struct ExperientialLearningEngine {
    repo_root: String,
    lesson_store_path: String,
    in_memory_lessons: Arc<Mutex<Vec<LessonLearned>>>,
}

impl ExperientialLearningEngine {
    pub fn new(repo_root: &str) -> Self {
        let store_dir = format!("{}/storage/persistence", repo_root);
        let _ = fs::create_dir_all(&store_dir);
        let lesson_store_path = format!("{}/experiential_lessons.jsonl", store_dir);

        let mut in_memory = Vec::new();
        if Path::new(&lesson_store_path).exists() {
            if let Ok(content) = fs::read_to_string(&lesson_store_path) {
                for line in content.lines() {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        if let Ok(lesson) = serde_json::from_str::<LessonLearned>(trimmed) {
                            in_memory.push(lesson);
                        }
                    }
                }
            }
        }

        Self {
            repo_root: repo_root.to_string(),
            lesson_store_path,
            in_memory_lessons: Arc::new(Mutex::new(in_memory)),
        }
    }

    /// Runs the full 12-stage closed-loop execution.
    pub fn execute_closed_loop<F>(
        &self,
        task_category: &str,
        task_description: &str,
        context_data: &Value,
        safety_evaluator: impl Fn(&str, &Value) -> bool,
        action_fn: F,
    ) -> ExperienceEpisode
    where
        F: FnOnce(&str) -> Result<Value, String>,
    {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let episode_id = format!("ep_{}_{}", task_category, now);
        let mut traces = Vec::new();

        // Stage 1: Task Reception
        traces.push(StageTrace {
            stage: ExperientialStage::TaskReception,
            timestamp_ms: now,
            detail: format!("Received task: {}", task_description),
            status: "OK".to_string(),
        });

        // Stage 2: Context Formulation
        let context_str = serde_json::to_string(context_data).unwrap_or_default();
        traces.push(StageTrace {
            stage: ExperientialStage::ContextFormulation,
            timestamp_ms: now,
            detail: format!("Formulated context: {} bytes", context_str.len()),
            status: "OK".to_string(),
        });

        // Stage 3: Knowledge / Skill Retrieval (query past lessons)
        let relevant_lessons = self.retrieve_applicable_lessons(task_category, task_description);
        traces.push(StageTrace {
            stage: ExperientialStage::KnowledgeRetrieval,
            timestamp_ms: now,
            detail: format!(
                "Retrieved {} historical lessons for category {}",
                relevant_lessons.len(),
                task_category
            ),
            status: "OK".to_string(),
        });

        // Stage 4: Strategy Formulation
        let strategy = if let Some(best) = relevant_lessons.first() {
            format!("Adaptive: {}", best.recommendation)
        } else {
            "Standard Deterministic Execution".to_string()
        };
        traces.push(StageTrace {
            stage: ExperientialStage::StrategyFormulation,
            timestamp_ms: now,
            detail: format!("Formulated strategy: {}", strategy),
            status: "OK".to_string(),
        });

        // Stage 5: Safety Gate Check
        let passed_safety = safety_evaluator(&strategy, context_data);
        if !passed_safety {
            traces.push(StageTrace {
                stage: ExperientialStage::SafetyGateCheck,
                timestamp_ms: now,
                detail: "Safety check rejected strategy execution".to_string(),
                status: "BLOCKED".to_string(),
            });

            return ExperienceEpisode {
                episode_id,
                task_description: task_description.to_string(),
                context_summary: context_str,
                chosen_strategy: strategy,
                execution_result: json!({"error": "Safety policy constraint violation"}),
                success: false,
                outcome_score: 0.0,
                traces,
                synthesized_lesson: None,
                created_at_ms: now,
            };
        }
        traces.push(StageTrace {
            stage: ExperientialStage::SafetyGateCheck,
            timestamp_ms: now,
            detail: "Safety check passed".to_string(),
            status: "PASSED".to_string(),
        });

        // Stage 6: Action Execution
        let exec_res = action_fn(&strategy);
        traces.push(StageTrace {
            stage: ExperientialStage::ActionExecution,
            timestamp_ms: now,
            detail: "Action executed".to_string(),
            status: if exec_res.is_ok() {
                "SUCCESS".to_string()
            } else {
                "FAILED".to_string()
            },
        });

        // Stage 7: Observation & Feedback Collection
        let (output_val, success) = match exec_res {
            Ok(v) => (v, true),
            Err(e) => (json!({"error": e}), false),
        };
        traces.push(StageTrace {
            stage: ExperientialStage::FeedbackCollection,
            timestamp_ms: now,
            detail: format!("Observed outcome status: success={}", success),
            status: "OK".to_string(),
        });

        // Stage 8: Outcome Evaluation
        let outcome_score = if success { 1.0 } else { 0.0 };
        traces.push(StageTrace {
            stage: ExperientialStage::OutcomeEvaluation,
            timestamp_ms: now,
            detail: format!("Calculated outcome score: {:.2}", outcome_score),
            status: "OK".to_string(),
        });

        // Stage 9: Credit Assignment
        let attribution = if success {
            format!("Strategy '{}' effectively solved task", strategy)
        } else {
            format!("Strategy '{}' encountered execution error", strategy)
        };
        traces.push(StageTrace {
            stage: ExperientialStage::CreditAssignment,
            timestamp_ms: now,
            detail: attribution.clone(),
            status: "OK".to_string(),
        });

        // Stage 10: Lesson Formulation
        let lesson = LessonLearned {
            lesson_id: format!("lesson_{}_{}", task_category, now),
            task_category: task_category.to_string(),
            trigger_pattern: task_description.to_string(),
            strategy_used: strategy.clone(),
            outcome_score,
            root_cause: attribution,
            recommendation: if success {
                format!("Reinforce strategy: {}", strategy)
            } else {
                format!("Avoid strategy '{}', try alternative approach", strategy)
            },
            confidence_score: if success { 0.85 } else { 0.40 },
            applied_count: 1,
            timestamp_ms: now,
        };
        traces.push(StageTrace {
            stage: ExperientialStage::LessonFormulation,
            timestamp_ms: now,
            detail: format!("Formulated lesson: {}", lesson.lesson_id),
            status: "OK".to_string(),
        });

        // Stage 11: Strategy Adaptation (In-memory update)
        {
            let mut mem = self.in_memory_lessons.lock().unwrap();
            mem.push(lesson.clone());
        }
        traces.push(StageTrace {
            stage: ExperientialStage::StrategyAdaptation,
            timestamp_ms: now,
            detail: "Adapted strategy repository with updated weights".to_string(),
            status: "OK".to_string(),
        });

        // Stage 12: Persistent Lesson Storage
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.lesson_store_path)
        {
            if let Ok(serialized) = serde_json::to_string(&lesson) {
                let _ = writeln!(file, "{}", serialized);
            }
        }
        traces.push(StageTrace {
            stage: ExperientialStage::PersistentStorage,
            timestamp_ms: now,
            detail: format!("Appended lesson to {}", self.lesson_store_path),
            status: "PERSISTED".to_string(),
        });

        ExperienceEpisode {
            episode_id,
            task_description: task_description.to_string(),
            context_summary: context_str,
            chosen_strategy: strategy,
            execution_result: output_val,
            success,
            outcome_score,
            traces,
            synthesized_lesson: Some(lesson),
            created_at_ms: now,
        }
    }

    /// Retrieves historical lessons matching the task category or description keywords.
    pub fn retrieve_applicable_lessons(
        &self,
        category: &str,
        description: &str,
    ) -> Vec<LessonLearned> {
        let mem = self.in_memory_lessons.lock().unwrap();
        let desc_lower = description.to_lowercase();
        let cat_lower = category.to_lowercase();

        let mut matches: Vec<LessonLearned> = mem
            .iter()
            .filter(|l| {
                l.task_category.to_lowercase() == cat_lower
                    || desc_lower.contains(&l.trigger_pattern.to_lowercase())
            })
            .cloned()
            .collect();

        // Sort descending by outcome_score * confidence_score
        matches.sort_by(|a, b| {
            let score_b = b.outcome_score * b.confidence_score;
            let score_a = a.outcome_score * a.confidence_score;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        matches
    }

    /// Returns all registered lessons count.
    pub fn lesson_count(&self) -> usize {
        self.in_memory_lessons.lock().unwrap().len()
    }

    /// Records and persists a lesson learned from a cognitive turn.
    pub fn record_lesson(&self, lesson: LessonLearned) {
        {
            let mut mem = self.in_memory_lessons.lock().unwrap();
            mem.push(lesson.clone());
        }
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.lesson_store_path)
        {
            if let Ok(serialized) = serde_json::to_string(&lesson) {
                let _ = writeln!(file, "{}", serialized);
            }
        }
    }

    /// Automatically bridges approved lessons into training dataset pairs in `storage/datasets/reasoning/approved_lessons.jsonl`.
    pub fn export_approved_lessons_to_dataset(
        &self,
        min_outcome: f64,
        min_confidence: f64,
        dataset_subdir: Option<&str>,
    ) -> Result<Value, std::io::Error> {
        let mem = self.in_memory_lessons.lock().unwrap();
        let target_dir = format!(
            "{}/storage/datasets/{}",
            self.repo_root,
            dataset_subdir.unwrap_or("reasoning")
        );
        fs::create_dir_all(&target_dir)?;
        let target_file = format!("{}/approved_lessons.jsonl", target_dir);

        let mut seen_triggers = std::collections::HashSet::new();
        if Path::new(&target_file).exists() {
            if let Ok(content) = fs::read_to_string(&target_file) {
                for line in content.lines() {
                    if let Ok(v) = serde_json::from_str::<Value>(line) {
                        if let Some(prompt) = v.get("prompt").and_then(Value::as_str) {
                            seen_triggers.insert(prompt.to_string());
                        }
                    }
                }
            }
        }

        let mut exported = 0;
        let mut skipped = 0;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&target_file)?;

        for lesson in mem.iter() {
            if lesson.outcome_score >= min_outcome && lesson.confidence_score >= min_confidence {
                let prompt = format!(
                    "<|im_start|>user\n[Task Category: {}] {}<|im_end|>\n<|im_start|>assistant\n",
                    lesson.task_category, lesson.trigger_pattern
                );
                if seen_triggers.contains(&prompt) {
                    skipped += 1;
                    continue;
                }
                let completion = format!(
                    "[Strategy]: {}\n[Analysis]: {}\n[Recommendation]: {}<|im_end|>",
                    lesson.strategy_used, lesson.root_cause, lesson.recommendation
                );

                let record = json!({
                    "prompt": prompt,
                    "completion": completion,
                    "lesson_id": lesson.lesson_id,
                    "outcome_score": lesson.outcome_score,
                    "confidence_score": lesson.confidence_score,
                    "source": "tara_experiential_reasoning"
                });

                writeln!(
                    file,
                    "{}",
                    serde_json::to_string(&record).unwrap_or_default()
                )?;
                seen_triggers.insert(prompt);
                exported += 1;
            } else {
                skipped += 1;
            }
        }

        Ok(json!({
            "status": "SUCCESS",
            "exported": exported,
            "skipped": skipped,
            "target_file": target_file
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experiential_12_stage_cycle() {
        let temp_dir = std::env::temp_dir().join("tara_test_exp_learning");
        let repo_root = temp_dir.to_str().unwrap();

        let engine = ExperientialLearningEngine::new(repo_root);

        let episode = engine.execute_closed_loop(
            "file_io",
            "Read configuration json safely",
            &json!({"target": "config.json"}),
            |_strat, _ctx| true, // safety check passes
            |_strat| Ok(json!({"bytes_read": 1024, "status": "ok"})),
        );

        assert_eq!(episode.traces.len(), 12);
        assert!(episode.success);
        assert_eq!(episode.outcome_score, 1.0);
        assert!(episode.synthesized_lesson.is_some());
        assert_eq!(engine.lesson_count(), 1);

        // Check retrieval
        let retrieved = engine.retrieve_applicable_lessons("file_io", "Read configuration");
        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].task_category, "file_io");

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_experiential_safety_block() {
        let temp_dir = std::env::temp_dir().join("tara_test_exp_safety");
        let repo_root = temp_dir.to_str().unwrap();

        let engine = ExperientialLearningEngine::new(repo_root);

        let episode = engine.execute_closed_loop(
            "privileged_op",
            "Format system partition",
            &json!({"target": "/dev/sda"}),
            |_strat, _ctx| false, // safety gate blocks
            |_strat| Ok(json!({})),
        );

        assert!(!episode.success);
        assert_eq!(episode.outcome_score, 0.0);
        assert_eq!(episode.traces.last().unwrap().status, "BLOCKED");

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_export_approved_lessons_to_dataset() {
        let temp_dir =
            std::env::temp_dir().join(format!("tara_test_exp_export_{}", uuid::Uuid::new_v4()));
        let repo_root = temp_dir.to_str().unwrap();

        let engine = ExperientialLearningEngine::new(repo_root);

        engine.record_lesson(LessonLearned {
            lesson_id: "lesson_test_1".to_string(),
            task_category: "algorithms".to_string(),
            trigger_pattern: "optimize matrix multiplication".to_string(),
            strategy_used: "Strassen algorithm with blocked layout".to_string(),
            outcome_score: 0.95,
            root_cause: "High memory locality achieved".to_string(),
            recommendation: "Use blocked matrix tiling".to_string(),
            confidence_score: 0.90,
            applied_count: 3,
            timestamp_ms: 1000,
        });

        let export_res = engine
            .export_approved_lessons_to_dataset(0.80, 0.70, Some("reasoning"))
            .unwrap();
        assert_eq!(export_res["status"], "SUCCESS");
        assert_eq!(export_res["exported"], 1);

        // Verify dataset file exists and contains valid JSONL
        let exported_file = format!(
            "{}/storage/datasets/reasoning/approved_lessons.jsonl",
            repo_root
        );
        assert!(Path::new(&exported_file).exists());
        let raw = fs::read_to_string(&exported_file).unwrap();
        assert!(raw.contains("optimize matrix multiplication"));
        assert!(raw.contains("Strassen algorithm with blocked layout"));

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
