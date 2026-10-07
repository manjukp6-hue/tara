//! Cache eviction and admission policies.
//!
//! Implements LRU, LFU, and LFRU (Least Frequently and Recently Used)
//! with 25% + 4-unit hysteresis margin to prevent ping-pong thrashing.

use std::collections::HashMap;

/// Admission contract for adaptive resident tiers.
/// threshold = cold + ((cold * margin_pct) / 100) + fixed_margin
pub fn tier_should_promote(
    hot: usize,
    cold: usize,
    margin_pct: usize,
    fixed_margin: usize,
) -> bool {
    let pct_margin = cold.saturating_mul(margin_pct) / 100;
    let threshold = cold
        .saturating_add(pct_margin)
        .saturating_add(fixed_margin);
    hot > threshold
}

/// Periodic half-life decay of heat counter.
pub fn tier_decay_value(heat: usize) -> usize {
    heat >> 1
}

/// LFRU scoring combining access frequency and recency.
pub fn tier_lfru_score(heat: usize, last: usize, clock: usize) -> usize {
    let age = clock.saturating_sub(last);
    let recent = 255_usize.saturating_sub(age);
    (heat << 8) | recent
}

pub trait CachePolicy: Send + Sync {
    fn record_access(&mut self, key: &str);
    fn pick_eviction(&self, resident_keys: &[String]) -> Option<String>;
    fn should_admit(&self, candidate_key: &str, evict_key: &str) -> bool;
    fn decay(&mut self);
}

/// Default hysteresis percentage for tier promotion admission.
/// A candidate is promoted only when hot count exceeds cold + (cold * 25%) + fixed_margin.
pub const DEFAULT_LFRU_MARGIN_PCT: usize = 25;

/// Default fixed-unit hysteresis margin to prevent ping-pong thrashing.
pub const DEFAULT_LFRU_FIXED_MARGIN: usize = 4;

pub struct LFRUCachePolicy {
    pub heat: HashMap<String, usize>,
    pub last_access: HashMap<String, usize>,
    pub clock: usize,
    pub margin_pct: usize,
    pub fixed_margin: usize,
}

impl Default for LFRUCachePolicy {
    fn default() -> Self {
        Self::new(DEFAULT_LFRU_MARGIN_PCT, DEFAULT_LFRU_FIXED_MARGIN)
    }
}

impl LFRUCachePolicy {
    pub fn new(margin_pct: usize, fixed_margin: usize) -> Self {
        Self {
            heat: HashMap::new(),
            last_access: HashMap::new(),
            clock: 0,
            margin_pct,
            fixed_margin,
        }
    }

    pub fn score(&self, key: &str) -> usize {
        let h = self.heat.get(key).copied().unwrap_or(0);
        let last = self.last_access.get(key).copied().unwrap_or(self.clock);
        tier_lfru_score(h, last, self.clock)
    }
}

impl CachePolicy for LFRUCachePolicy {
    fn record_access(&mut self, key: &str) {
        self.clock += 1;
        *self.heat.entry(key.to_string()).or_insert(0) += 1;
        self.last_access.insert(key.to_string(), self.clock);
    }

    fn pick_eviction(&self, resident_keys: &[String]) -> Option<String> {
        if resident_keys.is_empty() {
            return None;
        }
        let mut coldest_key = None;
        let mut min_score = usize::MAX;

        for k in resident_keys {
            let sc = self.score(k);
            if sc < min_score {
                min_score = sc;
                coldest_key = Some(k.clone());
            }
        }
        coldest_key
    }

    fn should_admit(&self, candidate_key: &str, evict_key: &str) -> bool {
        let cand_heat = self.heat.get(candidate_key).copied().unwrap_or(0);
        let evict_heat = self.heat.get(evict_key).copied().unwrap_or(0);
        tier_should_promote(cand_heat, evict_heat, self.margin_pct, self.fixed_margin)
    }

    fn decay(&mut self) {
        for v in self.heat.values_mut() {
            *v = tier_decay_value(*v);
        }
    }
}

#[derive(Default)]
pub struct LRUCachePolicy {
    pub last_access: HashMap<String, usize>,
    pub clock: usize,
}

impl CachePolicy for LRUCachePolicy {
    fn record_access(&mut self, key: &str) {
        self.clock += 1;
        self.last_access.insert(key.to_string(), self.clock);
    }

    fn pick_eviction(&self, resident_keys: &[String]) -> Option<String> {
        resident_keys
            .iter()
            .min_by_key(|k| self.last_access.get(*k).copied().unwrap_or(0))
            .cloned()
    }

    fn should_admit(&self, _candidate_key: &str, _evict_key: &str) -> bool {
        true
    }

    fn decay(&mut self) {}
}

#[derive(Default)]
pub struct LFUCachePolicy {
    pub counts: HashMap<String, usize>,
}

impl CachePolicy for LFUCachePolicy {
    fn record_access(&mut self, key: &str) {
        *self.counts.entry(key.to_string()).or_insert(0) += 1;
    }

    fn pick_eviction(&self, resident_keys: &[String]) -> Option<String> {
        resident_keys
            .iter()
            .min_by_key(|k| self.counts.get(*k).copied().unwrap_or(0))
            .cloned()
    }

    fn should_admit(&self, candidate_key: &str, evict_key: &str) -> bool {
        let c = self.counts.get(candidate_key).copied().unwrap_or(0);
        let e = self.counts.get(evict_key).copied().unwrap_or(0);
        c > e
    }

    fn decay(&mut self) {
        for v in self.counts.values_mut() {
            *v = tier_decay_value(*v);
        }
    }
}
