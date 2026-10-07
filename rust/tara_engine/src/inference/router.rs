//! Routing trace tracker and access heat management.

use std::collections::HashMap;

pub struct RoutingTracker {
    pub heat_map: HashMap<String, usize>,
    pub access_history: Vec<String>,
    pub transitions: HashMap<String, HashMap<String, usize>>,
    pub step_count: usize,
    pub decay_interval_steps: usize,
}

/// Number of inference steps between routing heat-map decay passes.
const DEFAULT_DECAY_INTERVAL_STEPS: usize = 100;

impl Default for RoutingTracker {
    fn default() -> Self {
        Self::new(DEFAULT_DECAY_INTERVAL_STEPS)
    }
}

impl RoutingTracker {
    pub fn new(decay_interval_steps: usize) -> Self {
        Self {
            heat_map: HashMap::new(),
            access_history: Vec::new(),
            transitions: HashMap::new(),
            step_count: 0,
            decay_interval_steps,
        }
    }

    pub fn record_access(&mut self, tensor_key: &str) {
        self.step_count += 1;
        *self.heat_map.entry(tensor_key.to_string()).or_insert(0) += 1;

        if let Some(prev) = self.access_history.last() {
            let next_map = self.transitions.entry(prev.clone()).or_default();
            *next_map.entry(tensor_key.to_string()).or_insert(0) += 1;
        }

        self.access_history.push(tensor_key.to_string());
        if self.access_history.len() > 1000 {
            self.access_history.drain(..500);
        }

        if self.step_count.is_multiple_of(self.decay_interval_steps) {
            self.decay_all();
        }
    }

    pub fn decay_all(&mut self) {
        let mut to_remove = Vec::new();
        for (k, v) in self.heat_map.iter_mut() {
            *v >>= 1;
            if *v == 0 {
                to_remove.push(k.clone());
            }
        }
        for k in to_remove {
            self.heat_map.remove(&k);
        }
    }

    pub fn predict_next(&self, current_key: &str, top_k: usize) -> Vec<String> {
        if let Some(next_candidates) = self.transitions.get(current_key) {
            let mut list: Vec<(&String, &usize)> = next_candidates.iter().collect();
            list.sort_by(|a, b| b.1.cmp(a.1));
            list.into_iter()
                .take(top_k)
                .map(|(k, _)| k.clone())
                .collect()
        } else {
            Vec::new()
        }
    }

    pub fn get_heat(&self, tensor_key: &str) -> usize {
        self.heat_map.get(tensor_key).copied().unwrap_or(0)
    }
}
