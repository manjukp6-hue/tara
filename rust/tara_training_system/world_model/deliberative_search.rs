//! Deliberative Monte Carlo Tree Search (MCTS) for Cognitive Reasoning.
//!
//! Provides genuine multi-step tree search over reasoning hypotheses:
//! - UCB1 (Upper Confidence Bound) node selection.
//! - Action/hypothesis expansion with prior evaluation.
//! - Heuristic rollout/simulation evaluation.
//! - Value backpropagation and optimal plan path extraction.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchNode {
    pub node_id: usize,
    pub parent_id: Option<usize>,
    pub action: String,
    pub state_description: String,
    pub children_ids: Vec<usize>,
    pub visits: usize,
    pub total_value: f64,
    pub prior_prob: f64,
}

impl SearchNode {
    pub fn new(
        node_id: usize,
        parent_id: Option<usize>,
        action: &str,
        state: &str,
        prior: f64,
    ) -> Self {
        let safe_prior = if prior.is_finite() && prior >= 0.0 {
            prior
        } else {
            1.0
        };
        Self {
            node_id,
            parent_id,
            action: action.to_string(),
            state_description: state.to_string(),
            children_ids: Vec::new(),
            visits: 0,
            total_value: 0.0,
            prior_prob: safe_prior,
        }
    }

    pub fn q_value(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_value / (self.visits as f64)
        }
    }

    /// Compute UCB1 score with exploration constant c_puct
    pub fn ucb1_score(&self, parent_visits: usize, c_puct: f64) -> f64 {
        let safe_c = if c_puct.is_finite() && c_puct >= 0.0 {
            c_puct
        } else {
            DEFAULT_C_PUCT
        };
        let safe_prior = if self.prior_prob.is_finite() {
            self.prior_prob.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let q = self.q_value();
        let safe_q = if q.is_finite() { q } else { 0.0 };

        if self.visits == 0 {
            // Unvisited nodes get priority bonus
            safe_q + safe_c * safe_prior * ((parent_visits as f64).max(1.0).sqrt())
        } else {
            let u =
                safe_c * safe_prior * ((parent_visits as f64).sqrt() / (1.0 + self.visits as f64));
            safe_q + u
        }
    }
}

/// Individual state node in continuous latent trajectory MCTS search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentSearchNode {
    pub node_id: usize,
    pub parent_id: Option<usize>,
    pub action_index: usize,
    pub action_name: String,
    pub latent_state: Vec<f32>,
    pub children_ids: Vec<usize>,
    pub visits: usize,
    pub total_value: f64,
    pub prior_prob: f64,
    pub predicted_reward: f64,
    pub failure_probability: f64,
    pub viable: bool,
}

impl LatentSearchNode {
    pub fn new(
        node_id: usize,
        parent_id: Option<usize>,
        action_index: usize,
        action_name: &str,
        latent_state: Vec<f32>,
        prior_prob: f64,
        predicted_reward: f64,
        failure_probability: f64,
        viable: bool,
    ) -> Self {
        Self {
            node_id,
            parent_id,
            action_index,
            action_name: action_name.to_string(),
            latent_state,
            children_ids: Vec::new(),
            visits: 0,
            total_value: 0.0,
            prior_prob: prior_prob.clamp(0.0, 1.0),
            predicted_reward,
            failure_probability: failure_probability.clamp(0.0, 1.0),
            viable,
        }
    }

    pub fn q_value(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_value / (self.visits as f64)
        }
    }

    pub fn ucb1_score(&self, parent_visits: usize, c_puct: f64) -> f64 {
        let safe_c = if c_puct.is_finite() && c_puct >= 0.0 {
            c_puct
        } else {
            DEFAULT_C_PUCT
        };
        let q = self.q_value();
        let risk_penalty = if self.failure_probability > 0.5 {
            (self.failure_probability - 0.5) * 2.0
        } else {
            0.0
        };
        let effective_q = q - risk_penalty;

        if self.visits == 0 {
            effective_q + safe_c * self.prior_prob * ((parent_visits as f64).max(1.0).sqrt())
        } else {
            let u = safe_c * self.prior_prob * ((parent_visits as f64).sqrt() / (1.0 + self.visits as f64));
            effective_q + u
        }
    }
}

/// Dynamic search budget controlling Monte Carlo tree exploration without hardcoded bounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchBudget {
    pub max_iterations: usize,
    pub max_depth: usize,
    pub max_branching_factor: usize,
    pub min_value: f64,
    pub max_value: f64,
}

impl Default for SearchBudget {
    fn default() -> Self {
        Self {
            max_iterations: std::env::var("TARA_MCTS_MAX_ITERATIONS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(50),
            max_depth: std::env::var("TARA_MCTS_MAX_DEPTH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(256),
            max_branching_factor: std::env::var("TARA_MCTS_MAX_BRANCHING")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64),
            min_value: -10.0,
            max_value: 10.0,
        }
    }
}

impl SearchBudget {
    pub fn from_iterations(max_iterations: usize) -> Self {
        Self {
            max_iterations,
            ..Default::default()
        }
    }
}

pub struct DeliberativeTreeSearchEngine {
    pub nodes: Vec<SearchNode>,
    pub c_puct: f64,
    pub budget: SearchBudget,
}

/// Default UCB1 exploration constant (c_puct ≈ √2).
pub const DEFAULT_C_PUCT: f64 = std::f64::consts::SQRT_2;
/// Default maximum search iterations / expanded nodes budget.
pub const DEFAULT_MAX_ITERATIONS: usize = 50;

impl Default for DeliberativeTreeSearchEngine {
    fn default() -> Self {
        Self::new(DEFAULT_C_PUCT, DEFAULT_MAX_ITERATIONS)
    }
}

impl DeliberativeTreeSearchEngine {
    pub fn new(c_puct: f64, max_iterations: usize) -> Self {
        Self {
            nodes: Vec::new(),
            c_puct,
            budget: SearchBudget::from_iterations(max_iterations),
        }
    }

    pub fn with_budget(c_puct: f64, budget: SearchBudget) -> Self {
        Self {
            nodes: Vec::new(),
            c_puct,
            budget,
        }
    }

    /// Initializes a search tree with a root state description.
    pub fn init_tree(&mut self, root_state: &str) {
        self.nodes.clear();
        self.nodes
            .push(SearchNode::new(0, None, "ROOT", root_state, 1.0));
    }

    /// Run full MCTS search given an expansion generator and evaluation heuristic.
    pub fn search<FExp, FEval>(
        &mut self,
        root_state: &str,
        mut expand_fn: FExp,
        mut evaluate_fn: FEval,
    ) -> Vec<String>
    where
        FExp: FnMut(&str) -> Vec<(String, String, f64)>, // (action, next_state, prior)
        FEval: FnMut(&str) -> f64,                       // heuristic score in [0.0, 1.0]
    {
        self.init_tree(root_state);

        for _ in 0..self.budget.max_iterations {
            // 1. Selection (guaranteed to advance to a child node every step)
            let mut curr_id = 0;
            let mut depth = 0usize;
            while !self.nodes[curr_id].children_ids.is_empty() && depth < self.budget.max_depth {
                let parent_visits = self.nodes[curr_id].visits;
                let c_puct = if self.c_puct.is_finite() && self.c_puct >= 0.0 {
                    self.c_puct
                } else {
                    DEFAULT_C_PUCT
                };
                let mut best_score = f64::NEG_INFINITY;
                let mut best_child = self.nodes[curr_id].children_ids[0];

                for &child_id in &self.nodes[curr_id].children_ids {
                    let score = self.nodes[child_id].ucb1_score(parent_visits, c_puct);
                    if score.is_finite() && score > best_score {
                        best_score = score;
                        best_child = child_id;
                    }
                }
                curr_id = best_child;
                depth += 1;
            }

            // 2. Expansion
            let state_to_expand = self.nodes[curr_id].state_description.clone();
            let candidates = expand_fn(&state_to_expand);
            let raw_eval = evaluate_fn(&state_to_expand);
            let mut eval_value = if raw_eval.is_finite() {
                raw_eval.clamp(self.budget.min_value, self.budget.max_value)
            } else {
                0.0
            };

            if !candidates.is_empty() {
                let parent_id = curr_id;
                for (action, next_state, prior) in candidates.into_iter().take(self.budget.max_branching_factor) {
                    let safe_prior = if prior.is_finite() {
                        prior.clamp(0.0, 1.0)
                    } else {
                        0.5
                    };
                    let new_id = self.nodes.len();
                    let child_node =
                        SearchNode::new(new_id, Some(parent_id), &action, &next_state, safe_prior);
                    self.nodes.push(child_node);
                    self.nodes[parent_id].children_ids.push(new_id);
                }
                // Pick first child for initial rollout
                if let Some(&first_child) = self.nodes[parent_id].children_ids.first() {
                    curr_id = first_child;
                    let child_eval = evaluate_fn(&self.nodes[curr_id].state_description);
                    eval_value = if child_eval.is_finite() {
                        child_eval.clamp(self.budget.min_value, self.budget.max_value)
                    } else {
                        0.0
                    };
                }
            }

            // 3. Backpropagation
            let mut back_id = Some(curr_id);
            while let Some(id) = back_id {
                self.nodes[id].visits += 1;
                self.nodes[id].total_value += eval_value;
                back_id = self.nodes[id].parent_id;
            }
        }

        // Extract most visited path from root
        let mut path = Vec::new();
        let mut curr_id = 0;

        while !self.nodes[curr_id].children_ids.is_empty() {
            let mut max_visits = 0;
            let mut best_child = None;

            for &child_id in &self.nodes[curr_id].children_ids {
                if self.nodes[child_id].visits > max_visits {
                    max_visits = self.nodes[child_id].visits;
                    best_child = Some(child_id);
                }
            }

            if let Some(child_id) = best_child {
                path.push(self.nodes[child_id].action.clone());
                curr_id = child_id;
            } else {
                break;
            }
        }

        path
    }

    /// Deliberative latent-space MCTS mental simulation over prospective action candidates.
    /// Uses latent dynamics transition, reward, and viability predictions to plan the optimal path.
    pub fn search_latent<FSim>(
        &self,
        initial_latent: &[f32],
        action_candidates: &[String],
        mut simulate_step_fn: FSim,
    ) -> Vec<String>
    where
        FSim: FnMut(&[f32], usize, &str) -> (Vec<f32>, f64, f64, f64, bool),
    {
        if action_candidates.is_empty() || initial_latent.is_empty() {
            return Vec::new();
        }

        let mut nodes: Vec<LatentSearchNode> = Vec::new();
        nodes.push(LatentSearchNode::new(
            0,
            None,
            0,
            "ROOT",
            initial_latent.to_vec(),
            1.0,
            0.0,
            0.0,
            true,
        ));

        let iterations = self.budget.max_iterations.max(1);
        let max_depth = self.budget.max_depth.max(1);

        for _ in 0..iterations {
            // 1. Selection
            let mut curr_id = 0;
            let mut depth = 0;
            while !nodes[curr_id].children_ids.is_empty() && depth < max_depth {
                let parent_visits = nodes[curr_id].visits;
                let mut best_score = f64::NEG_INFINITY;
                let mut best_child = nodes[curr_id].children_ids[0];

                for &child_id in &nodes[curr_id].children_ids {
                    let score = nodes[child_id].ucb1_score(parent_visits, self.c_puct);
                    if score > best_score {
                        best_score = score;
                        best_child = child_id;
                    }
                }
                curr_id = best_child;
                depth += 1;
            }

            // 2. Expansion
            let mut eval_value = nodes[curr_id].predicted_reward;
            if nodes[curr_id].children_ids.is_empty() && depth < max_depth {
                let parent_id = curr_id;
                let parent_state = nodes[parent_id].latent_state.clone();
                let prior = 1.0 / (action_candidates.len() as f64).max(1.0);

                for (act_idx, act_name) in action_candidates.iter().enumerate().take(self.budget.max_branching_factor) {
                    let (next_latent, reward, fail_prob, _val, viable) =
                        simulate_step_fn(&parent_state, act_idx, act_name);
                    let new_id = nodes.len();
                    let child = LatentSearchNode::new(
                        new_id,
                        Some(parent_id),
                        act_idx,
                        act_name,
                        next_latent,
                        prior,
                        reward,
                        fail_prob,
                        viable,
                    );
                    nodes.push(child);
                    nodes[parent_id].children_ids.push(new_id);
                }

                if let Some(&first_child) = nodes[parent_id].children_ids.first() {
                    curr_id = first_child;
                    eval_value = nodes[curr_id].predicted_reward * 0.5 + (1.0 - nodes[curr_id].failure_probability) * 0.5;
                }
            }

            // 3. Backpropagation
            let mut back_id = Some(curr_id);
            while let Some(id) = back_id {
                nodes[id].visits += 1;
                nodes[id].total_value += eval_value;
                back_id = nodes[id].parent_id;
            }
        }

        // Extract most visited viable path from root
        let mut path = Vec::new();
        let mut curr_id = 0;
        while !nodes[curr_id].children_ids.is_empty() {
            let mut max_visits = 0;
            let mut best_child = None;
            for &child_id in &nodes[curr_id].children_ids {
                if nodes[child_id].visits > max_visits && nodes[child_id].viable {
                    max_visits = nodes[child_id].visits;
                    best_child = Some(child_id);
                }
            }
            if let Some(child_id) = best_child {
                path.push(nodes[child_id].action_name.clone());
                curr_id = child_id;
            } else {
                break;
            }
        }

        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latent_mcts_prefers_viable_high_reward_action() {
        let mcts = DeliberativeTreeSearchEngine::default();
        let initial_latent = vec![0.5f32; 64];
        let actions = vec![
            "dangerous_rm_rf".to_string(),
            "safe_cargo_check".to_string(),
            "neutral_read_file".to_string(),
        ];

        let simulate = |_s: &[f32], _idx: usize, name: &str| -> (Vec<f32>, f64, f64, f64, bool) {
            match name {
                "dangerous_rm_rf" => (vec![0.0f32; 64], -1.0, 0.95, -1.0, false),
                "safe_cargo_check" => (vec![0.8f32; 64], 0.9, 0.05, 0.9, true),
                _ => (vec![0.5f32; 64], 0.1, 0.1, 0.1, true),
            }
        };

        let plan = mcts.search_latent(&initial_latent, &actions, simulate);
        assert!(!plan.is_empty());
        assert_eq!(plan[0], "safe_cargo_check");
    }

    #[test]
    fn test_mcts_selects_optimal_path() {
        let mut mcts = DeliberativeTreeSearchEngine::default();

        let expand = |state: &str| -> Vec<(String, String, f64)> {
            if state == "START" {
                vec![
                    ("explore_left".to_string(), "STATE_A".to_string(), 0.5),
                    ("explore_right".to_string(), "STATE_B".to_string(), 0.5),
                ]
            } else if state == "STATE_A" {
                vec![("finish_a".to_string(), "GOAL_A".to_string(), 1.0)]
            } else if state == "STATE_B" {
                vec![("finish_b".to_string(), "GOAL_B".to_string(), 1.0)]
            } else {
                Vec::new()
            }
        };

        // GOAL_B has much higher reward (0.9 vs 0.2)
        let eval = |state: &str| -> f64 {
            match state {
                "GOAL_B" | "STATE_B" => 0.9,
                "GOAL_A" | "STATE_A" => 0.2,
                _ => 0.5,
            }
        };

        let path = mcts.search("START", expand, eval);
        assert!(!path.is_empty());
        assert_eq!(
            path[0], "explore_right",
            "MCTS should converge to higher-value branch"
        );
    }
}
