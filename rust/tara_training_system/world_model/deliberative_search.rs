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
        Self {
            node_id,
            parent_id,
            action: action.to_string(),
            state_description: state.to_string(),
            children_ids: Vec::new(),
            visits: 0,
            total_value: 0.0,
            prior_prob: prior,
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
        if self.visits == 0 {
            // Unvisited nodes get priority bonus
            self.q_value() + c_puct * self.prior_prob * ((parent_visits as f64).max(1.0).sqrt())
        } else {
            let u = c_puct
                * self.prior_prob
                * ((parent_visits as f64).sqrt() / (1.0 + self.visits as f64));
            self.q_value() + u
        }
    }
}

pub struct DeliberativeTreeSearchEngine {
    pub nodes: Vec<SearchNode>,
    pub c_puct: f64,
    pub max_iterations: usize,
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
            max_iterations,
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

        for _ in 0..self.max_iterations {
            // 1. Selection
            let mut curr_id = 0;
            while !self.nodes[curr_id].children_ids.is_empty() {
                let parent_visits = self.nodes[curr_id].visits;
                let c_puct = self.c_puct;
                let mut best_score = f64::NEG_INFINITY;
                let mut best_child = curr_id;

                for &child_id in &self.nodes[curr_id].children_ids {
                    let score = self.nodes[child_id].ucb1_score(parent_visits, c_puct);
                    if score > best_score {
                        best_score = score;
                        best_child = child_id;
                    }
                }
                curr_id = best_child;
            }

            // 2. Expansion
            let state_to_expand = self.nodes[curr_id].state_description.clone();
            let candidates = expand_fn(&state_to_expand);
            let mut eval_value = evaluate_fn(&state_to_expand);

            if !candidates.is_empty() {
                let parent_id = curr_id;
                for (action, next_state, prior) in candidates {
                    let new_id = self.nodes.len();
                    let child_node =
                        SearchNode::new(new_id, Some(parent_id), &action, &next_state, prior);
                    self.nodes.push(child_node);
                    self.nodes[parent_id].children_ids.push(new_id);
                }
                // Pick first child for initial rollout
                if let Some(&first_child) = self.nodes[parent_id].children_ids.first() {
                    curr_id = first_child;
                    eval_value = evaluate_fn(&self.nodes[curr_id].state_description);
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
