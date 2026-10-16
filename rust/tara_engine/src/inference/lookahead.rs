//! Lookahead and speculative tensor prefetching across neural layers.

use std::collections::{HashMap, HashSet};

pub struct LookaheadPrefetcher {
    pub depth: usize,
    pub enabled: bool,
    pub inflight_prefetches: HashSet<String>,
}

/// Default lookahead depth: number of layers ahead to speculatively prefetch.
const DEFAULT_LOOKAHEAD_DEPTH: usize = 1;

/// Default prefetching state: enabled at startup.
const DEFAULT_LOOKAHEAD_ENABLED: bool = true;

impl Default for LookaheadPrefetcher {
    fn default() -> Self {
        Self::new(DEFAULT_LOOKAHEAD_DEPTH, DEFAULT_LOOKAHEAD_ENABLED)
    }
}

impl LookaheadPrefetcher {
    pub fn new(depth: usize, enabled: bool) -> Self {
        Self {
            depth,
            enabled,
            inflight_prefetches: HashSet::new(),
        }
    }

    pub fn on_layer_begin<F>(
        &mut self,
        current_layer: usize,
        total_layers: usize,
        tensor_names_by_layer: &HashMap<usize, Vec<String>>,
        mut prefetch_fn: F,
    ) where
        F: FnMut(&[String]) -> usize,
    {
        if !self.enabled {
            return;
        }

        let mut keys_to_fetch = Vec::new();
        for step in 1..=self.depth {
            let target_layer = current_layer + step;
            if target_layer < total_layers {
                if let Some(layer_keys) = tensor_names_by_layer.get(&target_layer) {
                    for k in layer_keys {
                        if !self.inflight_prefetches.contains(k) {
                            keys_to_fetch.push(k.clone());
                        }
                    }
                }
            }
        }

        if !keys_to_fetch.is_empty() {
            for k in &keys_to_fetch {
                self.inflight_prefetches.insert(k.clone());
            }
            prefetch_fn(&keys_to_fetch);
        }
    }

    pub fn on_layer_complete(
        &mut self,
        completed_layer: usize,
        tensor_names_by_layer: &HashMap<usize, Vec<String>>,
    ) {
        if let Some(layer_keys) = tensor_names_by_layer.get(&completed_layer) {
            for k in layer_keys {
                self.inflight_prefetches.remove(k);
            }
        }
    }
}
