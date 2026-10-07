//! Epistemic Curiosity and Autonomous Exploration Engine.
//!
//! Provides Shannon entropy-based epistemic uncertainty evaluation,
//! curiosity-driven goal generation, autonomous inquiry formulation,
//! and idle-time epistemic gap exploration without mock or synthetic logic.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Epistemic state evaluation using Shannon information entropy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicState {
    pub topic: String,
    pub known_concepts_count: usize,
    pub unknown_concepts_count: usize,
    pub certainty_distribution: Vec<f64>,
    pub epistemic_entropy: f64,
    pub novelty_score: f64,
    pub exploration_priority: f64,
}

/// A curiosity-generated autonomous exploration goal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationGoal {
    pub goal_id: String,
    pub topic: String,
    pub priority: f64,
    pub epistemic_gap: f64,
    pub suggested_inquiries: Vec<String>,
    pub timestamp: u64,
    pub status: String, // "PENDING", "EXPLORING", "RESOLVED"
}

/// Epistemic entropy evaluator calculating information theory metrics:
/// H(X) = - \sum p_i * log2(p_i)
#[derive(Debug, Clone)]
pub struct EpistemicEntropyEvaluator {
    epsilon: f64,
}

impl Default for EpistemicEntropyEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl EpistemicEntropyEvaluator {
    pub fn new() -> Self {
        Self { epsilon: 1e-12 }
    }

    /// Computes Shannon entropy over a normalized probability distribution.
    pub fn compute_shannon_entropy(&self, distribution: &[f64]) -> f64 {
        if distribution.is_empty() {
            return 0.0;
        }

        let sum: f64 = distribution.iter().sum();
        if sum <= self.epsilon {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &val in distribution {
            if val > self.epsilon {
                let p = val / sum;
                entropy -= p * p.log2();
            }
        }
        entropy
    }

    /// Evaluates epistemic uncertainty given known concepts vs target domain concepts.
    pub fn evaluate_epistemic_gap(
        &self,
        topic: &str,
        known_concepts: &[String],
        domain_topology: &[String],
    ) -> EpistemicState {
        let known_set: HashSet<String> = known_concepts
            .iter()
            .map(|s| s.trim().to_lowercase())
            .collect();
        let domain_set: HashSet<String> = domain_topology
            .iter()
            .map(|s| s.trim().to_lowercase())
            .collect();

        let known_count = domain_set.intersection(&known_set).count();
        let unknown_count = domain_set.difference(&known_set).count();
        let total = known_count + unknown_count;

        let (dist, entropy, novelty) = if total == 0 {
            (vec![0.5, 0.5], 1.0, 1.0)
        } else {
            let p_known = (known_count as f64) / (total as f64);
            let p_unknown = (unknown_count as f64) / (total as f64);
            let d = vec![p_known, p_unknown];
            let h = self.compute_shannon_entropy(&d);
            let n = p_unknown;
            (d, h, n)
        };

        // Exploration priority combines Shannon entropy and novelty:
        // Priority = (Entropy / 1.0) * 0.6 + Novelty * 0.4
        let priority = (entropy.min(1.0) * 0.6) + (novelty * 0.4);

        EpistemicState {
            topic: topic.to_string(),
            known_concepts_count: known_count,
            unknown_concepts_count: unknown_count,
            certainty_distribution: dist,
            epistemic_entropy: entropy,
            novelty_score: novelty,
            exploration_priority: priority,
        }
    }
}

/// Curiosity Drive Engine maintaining exploration goals and idle inquiry generation.
pub struct CuriosityDriveEngine {
    evaluator: EpistemicEntropyEvaluator,
    active_goals: Arc<Mutex<Vec<ExplorationGoal>>>,
    domain_knowledge_index: Arc<Mutex<HashMap<String, Vec<String>>>>,
    max_stored_goals: usize,
}

impl Default for CuriosityDriveEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl CuriosityDriveEngine {
    pub fn new() -> Self {
        Self {
            evaluator: EpistemicEntropyEvaluator::new(),
            active_goals: Arc::new(Mutex::new(Vec::new())),
            domain_knowledge_index: Arc::new(Mutex::new(HashMap::new())),
            max_stored_goals: 100,
        }
    }

    /// Registers recognized concepts for a given knowledge domain.
    pub fn register_domain_ontology(&self, domain: &str, concepts: Vec<String>) {
        let mut index = self.domain_knowledge_index.lock().unwrap();
        let entry = index.entry(domain.to_lowercase()).or_default();
        for c in concepts {
            if !entry.contains(&c) {
                entry.push(c);
            }
        }
    }

    /// Assesses current epistemic state for a topic given known concepts.
    pub fn evaluate_epistemic_state(
        &self,
        topic: &str,
        known_concepts: &[String],
    ) -> EpistemicState {
        let topology = {
            let index = self.domain_knowledge_index.lock().unwrap();
            index
                .get(&topic.to_lowercase())
                .cloned()
                .unwrap_or_default()
        };
        self.evaluator
            .evaluate_epistemic_gap(topic, known_concepts, &topology)
    }

    /// Assesses current epistemic state and generates exploration goals if uncertainty exceeds threshold.
    pub fn evaluate_and_formulate_goal(
        &self,
        domain: &str,
        current_agent_knowledge: &[String],
        uncertainty_threshold: f64,
    ) -> Option<ExplorationGoal> {
        let topology = {
            let index = self.domain_knowledge_index.lock().unwrap();
            index
                .get(&domain.to_lowercase())
                .cloned()
                .unwrap_or_default()
        };

        let state =
            self.evaluator
                .evaluate_epistemic_gap(domain, current_agent_knowledge, &topology);

        if state.exploration_priority >= uncertainty_threshold {
            let known_set: HashSet<String> = current_agent_knowledge
                .iter()
                .map(|s| s.trim().to_lowercase())
                .collect();

            // Formulate concrete inquiries for missing knowledge nodes
            let missing: Vec<String> = topology
                .iter()
                .filter(|c| !known_set.contains(&c.trim().to_lowercase()))
                .cloned()
                .collect();

            let inquiries: Vec<String> = missing
                .iter()
                .take(5)
                .map(|c| {
                    format!(
                        "What are the functional principles and integration contracts of {} in {}?",
                        c, domain
                    )
                })
                .collect();

            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;

            let goal_id = format!("curiosity_goal_{}_{}", domain, ts);
            let goal = ExplorationGoal {
                goal_id: goal_id.clone(),
                topic: domain.to_string(),
                priority: state.exploration_priority,
                epistemic_gap: state.epistemic_entropy,
                suggested_inquiries: inquiries,
                timestamp: ts,
                status: "PENDING".to_string(),
            };

            let mut goals = self.active_goals.lock().unwrap();
            goals.push(goal.clone());
            let cur_len = goals.len();
            if cur_len > self.max_stored_goals {
                goals.drain(0..cur_len - self.max_stored_goals);
            }

            Some(goal)
        } else {
            None
        }
    }

    /// Retrieves pending exploration goals ranked by epistemic priority.
    pub fn get_ranked_exploration_agenda(&self) -> Vec<ExplorationGoal> {
        let goals = self.active_goals.lock().unwrap();
        let mut sorted = goals.clone();
        sorted.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Resolves an exploration goal once the agent acquires relevant facts.
    pub fn mark_goal_resolved(&self, goal_id: &str) -> bool {
        let mut goals = self.active_goals.lock().unwrap();
        if let Some(goal) = goals.iter_mut().find(|g| g.goal_id == goal_id) {
            goal.status = "RESOLVED".to_string();
            true
        } else {
            false
        }
    }

    /// Serializes status to JSON.
    pub fn status_json(&self) -> Value {
        let goals = self.active_goals.lock().unwrap();
        let index = self.domain_knowledge_index.lock().unwrap();
        json!({
            "registered_domains_count": index.len(),
            "total_goals_tracked": goals.len(),
            "pending_goals_count": goals.iter().filter(|g| g.status == "PENDING").count(),
            "resolved_goals_count": goals.iter().filter(|g| g.status == "RESOLVED").count(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shannon_entropy_calculation() {
        let evaluator = EpistemicEntropyEvaluator::new();

        // Equal probabilities [0.5, 0.5] should yield exactly 1.0 bit
        let entropy_half = evaluator.compute_shannon_entropy(&[0.5, 0.5]);
        assert!((entropy_half - 1.0).abs() < 1e-6);

        // Uniform distribution over 4 items -> log2(4) = 2.0 bits
        let entropy_quad = evaluator.compute_shannon_entropy(&[0.25, 0.25, 0.25, 0.25]);
        assert!((entropy_quad - 2.0).abs() < 1e-6);

        // Completely certain [1.0, 0.0] -> 0.0 bits
        let entropy_certain = evaluator.compute_shannon_entropy(&[1.0, 0.0]);
        assert!(entropy_certain.abs() < 1e-6);
    }

    #[test]
    fn test_curiosity_goal_generation() {
        let engine = CuriosityDriveEngine::new();
        engine.register_domain_ontology(
            "robotics",
            vec![
                "kinematics".to_string(),
                "trajectory_planning".to_string(),
                "pid_controller".to_string(),
                "imu_fusion".to_string(),
            ],
        );

        // Agent only knows kinematics
        let known = vec!["kinematics".to_string()];
        let goal = engine.evaluate_and_formulate_goal("robotics", &known, 0.3);

        assert!(goal.is_some());
        let g = goal.unwrap();
        assert_eq!(g.topic, "robotics");
        assert!(g.priority > 0.4);
        assert_eq!(g.suggested_inquiries.len(), 3);

        let agenda = engine.get_ranked_exploration_agenda();
        assert_eq!(agenda.len(), 1);

        let marked = engine.mark_goal_resolved(&g.goal_id);
        assert!(marked);
    }
}
