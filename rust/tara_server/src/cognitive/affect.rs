//! Emotion-Like Computational States & Homeostatic Regulation Engine.
//!
//! STRICT INVARIANT:
//! This module represents NUMERICAL COMPUTATIONAL REGULATION VARIABLES ONLY.
//! It does NOT claim biological sentience, phenomenal feeling, or human emotion.
//!
//! Purpose:
//! Dynamically adjust cognitive resource allocation, search temperature,
//! exploration vs exploitation, and verification stringency using homeostatic
//! numerical feedback loops.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationalAffectState {
    /// Epistemic tension: [0.0, 1.0]. High when encountering high entropy / unknown concepts.
    pub epistemic_tension: f64,
    /// Cognitive load: [0.0, 1.0]. High when working memory is full or search trees are deep.
    pub cognitive_load: f64,
    /// Satisfaction valence: [-1.0, 1.0]. Moving average of recent reward signals.
    pub satisfaction_valence: f64,
    /// Urgency arousal: [0.0, 1.0]. High during errors, timeouts, or security alerts.
    pub urgency_arousal: f64,
    /// Timestamp of last homeostatic update
    pub last_updated_ms: u64,
}

pub struct AffectiveHomeostasisEngine {
    state: Arc<Mutex<ComputationalAffectState>>,
    decay_rate: f64, // Exponential decay toward equilibrium
}

impl Default for AffectiveHomeostasisEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AffectiveHomeostasisEngine {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(ComputationalAffectState {
                epistemic_tension: 0.2,
                cognitive_load: 0.1,
                satisfaction_valence: 0.5,
                urgency_arousal: 0.1,
                last_updated_ms: Self::current_ts_ms(),
            })),
            decay_rate: 0.05, // 5% decay per step toward baseline
        }
    }

    fn current_ts_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Read current computational state snapshot.
    pub fn get_state(&self) -> ComputationalAffectState {
        self.state.lock().unwrap().clone()
    }

    /// Update homeostatic state based on cognitive events:
    /// - `entropy_delta`: Increase in epistemic uncertainty
    /// - `memory_saturation`: [0.0, 1.0] working memory depth ratio
    /// - `reward_delta`: Outcome reward (-1.0 to +1.0)
    /// - `is_error`: Whether an error or security alert occurred
    pub fn update_from_turn_event(
        &self,
        entropy_delta: f64,
        memory_saturation: f64,
        reward_delta: f64,
        is_error: bool,
    ) -> ComputationalAffectState {
        let mut st = self.state.lock().unwrap();

        // 1. Epistemic tension: increases with entropy, decays toward baseline 0.1
        let target_tension = (st.epistemic_tension + entropy_delta * 0.5).clamp(0.0, 1.0);
        st.epistemic_tension =
            st.epistemic_tension * (1.0 - self.decay_rate) + target_tension * self.decay_rate;

        // 2. Cognitive load: tracks memory saturation and search complexity
        st.cognitive_load = (st.cognitive_load * 0.7 + memory_saturation * 0.3).clamp(0.0, 1.0);

        // 3. Satisfaction valence: updated via exponential moving average of reward
        let alpha_reward = 0.2;
        st.satisfaction_valence = ((1.0 - alpha_reward) * st.satisfaction_valence
            + alpha_reward * reward_delta)
            .clamp(-1.0, 1.0);

        // 4. Urgency arousal: spikes on errors/security alerts, decays rapidly toward 0.05
        if is_error {
            st.urgency_arousal = (st.urgency_arousal + 0.4).min(1.0);
        } else {
            st.urgency_arousal = (st.urgency_arousal * 0.85 + 0.05 * 0.15).clamp(0.0, 1.0);
        }

        st.last_updated_ms = Self::current_ts_ms();
        st.clone()
    }

    /// Compute behavioral adaptation parameters from current numerical state:
    /// - `exploration_temperature`: [0.1, 1.5]. Higher tension & low satisfaction -> higher exploration.
    /// - `max_deliberation_depth`: Max steps for tree search. Throttled under high cognitive load or urgent arousal.
    /// - `verification_threshold`: Required confidence threshold. Raised under high urgency or low valence.
    pub fn compute_behavioral_modulation(&self) -> Value {
        let st = self.state.lock().unwrap();

        // Exploration temperature: T = 0.5 + 0.5 * EpistemicTension - 0.25 * SatisfactionValence
        let temp =
            (0.5 + 0.5 * st.epistemic_tension - 0.25 * st.satisfaction_valence).clamp(0.1, 1.5);

        // Max search depth: default 12, throttled down under high cognitive load or urgency
        let max_depth = if st.cognitive_load > 0.70 || st.urgency_arousal > 0.45 {
            4
        } else if st.cognitive_load > 0.35 || st.urgency_arousal > 0.25 {
            7
        } else {
            12
        };

        // Verification threshold: default 0.80, increases to 0.95 during elevated urgency / errors
        let verif_threshold = if st.urgency_arousal > 0.45 || st.satisfaction_valence < 0.35 {
            0.95
        } else {
            0.80
        };

        json!({
            "exploration_temperature": (temp * 100.0).round() / 100.0,
            "max_deliberation_depth": max_depth,
            "verification_threshold": verif_threshold,
            "epistemic_tension": (st.epistemic_tension * 100.0).round() / 100.0,
            "cognitive_load": (st.cognitive_load * 100.0).round() / 100.0,
            "satisfaction_valence": (st.satisfaction_valence * 100.0).round() / 100.0,
            "urgency_arousal": (st.urgency_arousal * 100.0).round() / 100.0
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affective_homeostasis_modulation() {
        let engine = AffectiveHomeostasisEngine::new();
        let initial = engine.get_state();
        assert_eq!(initial.epistemic_tension, 0.2);

        // Simulate high entropy & error
        let st1 = engine.update_from_turn_event(0.8, 0.9, -0.5, true);
        assert!(st1.urgency_arousal > initial.urgency_arousal);
        assert!(st1.cognitive_load > initial.cognitive_load);

        let mod1 = engine.compute_behavioral_modulation();
        // High urgency and cognitive load must reduce max deliberation depth and raise verification threshold
        assert_eq!(mod1["max_deliberation_depth"], 4);
        assert_eq!(mod1["verification_threshold"], 0.95);

        // Simulate recovery with success
        let st2 = engine.update_from_turn_event(0.0, 0.2, 1.0, false);
        assert!(st2.satisfaction_valence > st1.satisfaction_valence);
    }
}
