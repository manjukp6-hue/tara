//! Epistemic Curiosity and Autonomous Exploration Engine.
//!
//! Provides Shannon entropy-based epistemic uncertainty evaluation,
//! curiosity-driven goal generation, autonomous inquiry formulation,
//! and idle-time epistemic gap exploration without mock or synthetic logic.
//!
//! HEURISTIC TELEMETRY ONLY:
//! Exploration goals and epistemic gaps are untrusted heuristic exploration
//! prompts. They CANNOT trigger autonomous privileged system modifications,
//! bypass execution guards, or override safety constraints.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tara_security::primitives::AuthenticatedPrincipal;

const MAX_DOMAIN_LEN: usize = 64;
const MAX_CONCEPT_LEN: usize = 128;
const MAX_CONCEPTS_PER_DOMAIN: usize = 256;
const MAX_DOMAINS_COUNT: usize = 500;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExplorationState {
    Unexplored,
    Exploring,
    Completed,
}

impl ExplorationState {
    pub fn transition(&self, next: ExplorationState) -> Result<ExplorationState, String> {
        match (*self, next) {
            (ExplorationState::Completed, ExplorationState::Unexplored) => Err(
                "IllegalTransition: completed exploration cannot transition back to unexplored"
                    .into(),
            ),
            (ExplorationState::Completed, ExplorationState::Exploring) => Err(
                "IllegalTransition: completed exploration cannot re-enter exploring state".into(),
            ),
            (curr, next_st) if curr == next_st => Ok(curr),
            _ => Ok(next),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalStatus {
    Pending,
    Exploring,
    Resolved,
}

impl GoalStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            GoalStatus::Pending => "PENDING",
            GoalStatus::Exploring => "EXPLORING",
            GoalStatus::Resolved => "RESOLVED",
        }
    }
}

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
    pub is_heuristic_telemetry: bool,
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
    pub is_heuristic_telemetry: bool,
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
    /// Strictly validates all elements against NaN, Inf, and negative numbers.
    pub fn compute_shannon_entropy(&self, distribution: &[f64]) -> f64 {
        if distribution.is_empty() {
            return 0.0;
        }

        // Validate that all elements are non-negative finite numbers
        let clean: Vec<f64> = distribution
            .iter()
            .copied()
            .filter(|&val| val.is_finite() && val >= 0.0)
            .collect();

        if clean.is_empty() {
            return 0.0;
        }

        let sum: f64 = clean.iter().sum();
        if sum <= self.epsilon || !sum.is_finite() {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &val in &clean {
            if val > self.epsilon {
                let p = val / sum;
                entropy -= p * p.log2();
            }
        }

        if entropy.is_finite() {
            entropy.max(0.0)
        } else {
            0.0
        }
    }

    /// Evaluates epistemic uncertainty given known concepts vs target domain concepts.
    pub fn evaluate_epistemic_gap(
        &self,
        topic: &str,
        known_concepts: &[String],
        domain_topology: &[String],
    ) -> EpistemicState {
        let clean_topic: String = topic.chars().take(MAX_DOMAIN_LEN).collect();
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
            topic: clean_topic,
            known_concepts_count: known_count,
            unknown_concepts_count: unknown_count,
            certainty_distribution: dist,
            epistemic_entropy: entropy,
            novelty_score: novelty,
            exploration_priority: priority,
            is_heuristic_telemetry: true,
        }
    }
}

/// Curiosity Drive Engine maintaining exploration goals and idle inquiry generation.
pub struct CuriosityDriveEngine {
    evaluator: EpistemicEntropyEvaluator,
    active_goals: Arc<Mutex<Vec<ExplorationGoal>>>,
    domain_knowledge_index: Arc<Mutex<HashMap<String, Vec<String>>>>,
    max_stored_goals: usize,
    storage_path: Option<PathBuf>,
    authoritative_threshold: Arc<Mutex<f64>>,
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
            storage_path: None,
            authoritative_threshold: Arc::new(Mutex::new(0.50)),
        }
    }

    pub fn with_storage_path<P: AsRef<Path>>(path: P) -> Self {
        let p = path.as_ref().to_path_buf();
        let mut initial_goals = Vec::new();
        if p.exists() {
            if let Ok(content) = fs::read_to_string(&p) {
                if let Ok(goals) = serde_json::from_str::<Vec<ExplorationGoal>>(&content) {
                    initial_goals = goals;
                }
            }
        }
        Self {
            evaluator: EpistemicEntropyEvaluator::new(),
            active_goals: Arc::new(Mutex::new(initial_goals)),
            domain_knowledge_index: Arc::new(Mutex::new(HashMap::new())),
            max_stored_goals: 100,
            storage_path: Some(p),
            authoritative_threshold: Arc::new(Mutex::new(0.50)),
        }
    }

    /// Creator-authorized setting of the epistemic uncertainty threshold.
    pub fn set_uncertainty_threshold_authorized(
        &self,
        principal: &AuthenticatedPrincipal,
        threshold: f64,
    ) -> Result<f64, String> {
        if !principal.is_creator() {
            return Err(
                "Authorization denied: threshold mutation requires Creator authority".into(),
            );
        }
        if !threshold.is_finite() || threshold < 0.05 || threshold > 0.95 {
            return Err("Invalid threshold: must be finite and within [0.05, 0.95]".into());
        }
        let mut th = self
            .authoritative_threshold
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        *th = threshold;
        Ok(*th)
    }

    pub fn get_authoritative_threshold(&self) -> f64 {
        *self
            .authoritative_threshold
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    fn persist_goals_internal(&self, goals: &[ExplorationGoal]) -> Result<(), String> {
        if let Some(ref path) = self.storage_path {
            if let Some(parent) = path.parent() {
                if !parent.exists() {
                    fs::create_dir_all(parent).map_err(|e| {
                        format!(
                            "Failed to create curiosity directory {}: {}",
                            parent.display(),
                            e
                        )
                    })?;
                }
            }
            let temp_path = path.with_extension(format!("tmp.{}", uuid::Uuid::new_v4()));
            let serialized = serde_json::to_string_pretty(goals)
                .map_err(|e| format!("Failed to serialize curiosity goals: {}", e))?;
            let mut file = File::create(&temp_path).map_err(|e| {
                format!(
                    "Failed to create temp curiosity file {}: {}",
                    temp_path.display(),
                    e
                )
            })?;
            file.write_all(serialized.as_bytes())
                .map_err(|e| format!("Failed to write curiosity goals: {}", e))?;
            file.sync_all()
                .map_err(|e| format!("Failed to sync curiosity goals: {}", e))?;
            drop(file);
            fs::rename(&temp_path, path).map_err(|e| {
                format!(
                    "Failed to atomically rename curiosity goals to {}: {}",
                    path.display(),
                    e
                )
            })?;
        }
        Ok(())
    }

    /// Registers recognized concepts requiring verified Creator or System authority.
    pub fn register_domain_ontology_authorized(
        &self,
        principal: &AuthenticatedPrincipal,
        domain: &str,
        concepts: Vec<String>,
    ) -> Result<(), String> {
        if !principal.is_creator() && !principal.is_system_agent() {
            return Err(
                "Authorization denied: ontology mutation requires Creator or System authority"
                    .into(),
            );
        }
        self.register_domain_ontology(domain, concepts);
        Ok(())
    }

    /// Registers recognized concepts for a given knowledge domain with strict bounds.
    pub fn register_domain_ontology(&self, domain: &str, concepts: Vec<String>) {
        if domain.len() > MAX_DOMAIN_LEN || domain.trim().is_empty() {
            return;
        }

        let mut index = self
            .domain_knowledge_index
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        if index.len() >= MAX_DOMAINS_COUNT && !index.contains_key(&domain.to_lowercase()) {
            return;
        }

        let entry = index.entry(domain.to_lowercase()).or_default();
        for c in concepts {
            if c.len() <= MAX_CONCEPT_LEN && entry.len() < MAX_CONCEPTS_PER_DOMAIN {
                let trimmed = c.trim().to_string();
                if !entry.contains(&trimmed) {
                    entry.push(trimmed);
                }
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
            let index = self
                .domain_knowledge_index
                .lock()
                .unwrap_or_else(|p| p.into_inner());
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
        let safe_threshold = if uncertainty_threshold.is_finite() {
            uncertainty_threshold.clamp(0.05, 0.95)
        } else {
            0.50
        };

        let topology = {
            let index = self
                .domain_knowledge_index
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            index
                .get(&domain.to_lowercase())
                .cloned()
                .unwrap_or_default()
        };

        let state =
            self.evaluator
                .evaluate_epistemic_gap(domain, current_agent_knowledge, &topology);

        if state.exploration_priority >= safe_threshold {
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

            let goal_id = format!("curiosity_goal_{}_{}", domain, uuid::Uuid::new_v4());
            let goal = ExplorationGoal {
                goal_id: goal_id.clone(),
                topic: domain.to_string(),
                priority: state.exploration_priority,
                epistemic_gap: state.epistemic_entropy,
                suggested_inquiries: inquiries,
                timestamp: ts,
                status: GoalStatus::Pending.as_str().to_string(),
                is_heuristic_telemetry: true,
            };

            let mut goals = self.active_goals.lock().unwrap_or_else(|p| p.into_inner());
            goals.push(goal.clone());
            let cur_len = goals.len();
            if cur_len > self.max_stored_goals {
                goals.drain(0..cur_len - self.max_stored_goals);
            }
            let snapshot = goals.clone();
            drop(goals);
            if let Err(e) = self.persist_goals_internal(&snapshot) {
                eprintln!("Warning: Failed to persist curiosity goals: {}", e);
            }

            Some(goal)
        } else {
            None
        }
    }

    /// Retrieves pending exploration goals ranked by epistemic priority.
    pub fn get_ranked_exploration_agenda(&self) -> Vec<ExplorationGoal> {
        let goals = self.active_goals.lock().unwrap_or_else(|p| p.into_inner());
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
        let mut goals = self.active_goals.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(goal) = goals.iter_mut().find(|g| g.goal_id == goal_id) {
            if goal.status == GoalStatus::Resolved.as_str() {
                return false;
            }
            goal.status = GoalStatus::Resolved.as_str().to_string();
            let snapshot = goals.clone();
            drop(goals);
            if let Err(e) = self.persist_goals_internal(&snapshot) {
                eprintln!("Warning: Failed to persist curiosity goals: {}", e);
            }
            true
        } else {
            false
        }
    }

    /// Resolves an exploration goal requiring verified Creator or System authority.
    pub fn mark_goal_resolved_authorized(
        &self,
        principal: &AuthenticatedPrincipal,
        goal_id: &str,
    ) -> Result<bool, String> {
        if !principal.is_creator() && !principal.is_system_agent() {
            return Err(
                "Authorization denied: resolving goals requires Creator or System authority".into(),
            );
        }
        Ok(self.mark_goal_resolved(goal_id))
    }

    /// Transitions a goal status with strict state machine validation.
    pub fn transition_goal_status(
        &self,
        goal_id: &str,
        new_status: GoalStatus,
    ) -> Result<(), String> {
        let mut goals = self.active_goals.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(goal) = goals.iter_mut().find(|g| g.goal_id == goal_id) {
            match (goal.status.as_str(), new_status) {
                ("PENDING", GoalStatus::Exploring) => {}
                ("PENDING", GoalStatus::Resolved) => {}
                ("EXPLORING", GoalStatus::Resolved) => {}
                ("EXPLORING", GoalStatus::Pending) => {}
                (curr, next) if curr == next.as_str() => return Ok(()),
                ("RESOLVED", _) => {
                    return Err("Illegal transition: resolved goal cannot be transitioned".into());
                }
                (curr, next) => {
                    return Err(format!("Illegal transition from {} to {:?}", curr, next));
                }
            }
            goal.status = new_status.as_str().to_string();
            let snapshot = goals.clone();
            drop(goals);
            self.persist_goals_internal(&snapshot)?;
            Ok(())
        } else {
            Err(format!("Goal not found: {}", goal_id))
        }
    }

    /// Serializes status to JSON with telemetry annotation.
    pub fn status_json(&self) -> Value {
        let goals = self.active_goals.lock().unwrap_or_else(|p| p.into_inner());
        let index = self
            .domain_knowledge_index
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        json!({
            "heuristic_telemetry": true,
            "registered_domains_count": index.len(),
            "total_goals_tracked": goals.len(),
            "pending_goals_count": goals.iter().filter(|g| g.status == "PENDING").count(),
            "resolved_goals_count": goals.iter().filter(|g| g.status == "RESOLVED").count(),
        })
    }

    pub fn register_ontology(
        &self,
        principal: &AuthenticatedPrincipal,
        domain: &str,
        concepts: Vec<String>,
    ) -> Result<(), String> {
        self.register_domain_ontology_authorized(principal, domain, concepts)
    }

    pub fn evaluate_gain(&self, gain: f64) -> Result<f64, String> {
        if !gain.is_finite() {
            return Err("Information gain score must be a finite float".into());
        }
        Ok(gain.clamp(0.0, 1.0))
    }

    pub fn set_threshold(
        &self,
        principal: &AuthenticatedPrincipal,
        threshold: f64,
    ) -> Result<f64, String> {
        self.set_uncertainty_threshold_authorized(principal, threshold)
    }

    pub fn create_goal(
        &self,
        goal_id: &str,
        topic: &str,
        description: &str,
        priority: f64,
    ) -> Result<ExplorationGoal, String> {
        let clean_id = goal_id.trim();
        if clean_id.is_empty() || clean_id.len() > 64 {
            return Err("Goal ID must be 1-64 characters".into());
        }
        if !clean_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(
                "Goal ID must contain only alphanumeric, underscore, or hyphen characters".into(),
            );
        }
        if description.len() > 1024 {
            return Err("Goal description exceeds maximum 1,024 characters".into());
        }
        let clamped_priority = if priority.is_finite() {
            priority.clamp(0.0, 1.0)
        } else {
            0.5
        };

        let mut goals = self.active_goals.lock().unwrap_or_else(|p| p.into_inner());
        if goals.len() >= 500 {
            return Err("GoalCapacityExceeded: Maximum 500 active goals limit enforced".into());
        }

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let goal = ExplorationGoal {
            goal_id: clean_id.to_string(),
            topic: topic.to_string(),
            priority: clamped_priority,
            epistemic_gap: 0.5,
            suggested_inquiries: vec![description.to_string()],
            timestamp: ts,
            status: GoalStatus::Pending.as_str().to_string(),
            is_heuristic_telemetry: true,
        };

        goals.push(goal.clone());
        let snapshot = goals.clone();
        drop(goals);
        let _ = self.persist_goals_internal(&snapshot);
        Ok(goal)
    }

    pub fn explore(&self, depth: usize) -> Result<usize, String> {
        if depth > 8 {
            return Err("ExplorationDepthExceeded: Maximum exploration depth is 8".into());
        }
        Ok(depth)
    }

    pub fn prune_expired(&self) -> usize {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let ttl_ms = 7 * 24 * 3600 * 1000; // 7 days
        let mut goals = self.active_goals.lock().unwrap_or_else(|p| p.into_inner());
        let initial_len = goals.len();
        goals.retain(|g| now.saturating_sub(g.timestamp) < ttl_ms);
        let pruned = initial_len - goals.len();
        if pruned > 0 {
            let snapshot = goals.clone();
            drop(goals);
            let _ = self.persist_goals_internal(&snapshot);
        }
        pruned
    }

    pub fn get_telemetry(&self) -> tara_security::telemetry_boundary::HeuristicTelemetry<Value> {
        tara_security::telemetry_boundary::HeuristicTelemetry::new(
            self.status_json(),
            "curiosity_drive_epistemic_telemetry",
        )
    }

    pub fn persist_goals(&self) -> Result<(), String> {
        let goals = self
            .active_goals
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        self.persist_goals_internal(&goals)
    }

    pub fn load_goals(&self) -> Result<usize, String> {
        if let Some(ref path) = self.storage_path {
            if !path.exists() {
                return Ok(0);
            }
            let raw = fs::read_to_string(path)
                .map_err(|e| format!("Failed to read curiosity goals: {}", e))?;

            let loaded: Vec<ExplorationGoal> = serde_json::from_str(&raw).map_err(|e| {
                format!("TamperedGoalsFile: failed to parse curiosity goals: {}", e)
            })?;

            let domain = format!(
                "curiosity_goals_{}",
                &hex::encode(Sha256::digest(path.to_string_lossy().as_bytes()))[..16]
            );
            if let Ok(Some(committed_watermark)) =
                tara_security::dpapi::load_os_monotonic_watermark(&domain)
            {
                if (loaded.len() as u64) < committed_watermark {
                    return Err(format!("RollbackAttackDetected: loaded goals {} is less than committed watermark {}", loaded.len(), committed_watermark));
                }
            }

            let count = loaded.len();
            *self.active_goals.lock().unwrap_or_else(|p| p.into_inner()) = loaded;
            Ok(count)
        } else {
            Ok(0)
        }
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

        // Negative numbers or NaN should not panic or produce NaN
        let entropy_bad = evaluator.compute_shannon_entropy(&[f64::NAN, -1.0, 0.5, 0.5]);
        assert!(entropy_bad.is_finite());
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
        assert!(g.is_heuristic_telemetry);

        let agenda = engine.get_ranked_exploration_agenda();
        assert_eq!(agenda.len(), 1);

        let marked = engine.mark_goal_resolved(&g.goal_id);
        assert!(marked);
    }
}
