//! Emotion-Like Computational States & Homeostatic Regulation Engine.
//!
//! STRICT INVARIANT:
//! This module represents NUMERICAL COMPUTATIONAL REGULATION VARIABLES ONLY.
//! It does NOT claim biological sentience, phenomenal feeling, or human emotion.
//!
//! HEURISTIC TELEMETRY ONLY:
//! Output states and modulations are untrusted internal heuristics for search
//! temperature and cognitive budgeting. They CANNOT override security policies,
//! bypass lockdown, or authorize privileged actions.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::Duration;

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
    /// Explicit tag indicating this state is heuristic telemetry
    pub is_heuristic_telemetry: bool,
    /// When true, indicates persistent state was corrupted; fails closed into DEGRADED/BLOCKED mode (C-276)
    #[serde(default)]
    pub is_degraded_blocked: bool,
}

pub struct AffectiveHomeostasisEngine {
    state: Arc<Mutex<ComputationalAffectState>>,
    decay_rate: f64, // Exponential decay toward equilibrium in (0.0, 1.0)
}

impl Default for AffectiveHomeostasisEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AffectiveHomeostasisEngine {
    pub fn new() -> Self {
        Self::with_decay(0.05)
    }

    pub fn with_decay(decay_rate: f64) -> Self {
        let safe_decay = if decay_rate.is_finite() {
            decay_rate.clamp(0.001, 0.999)
        } else {
            0.05
        };

        Self {
            state: Arc::new(Mutex::new(ComputationalAffectState {
                epistemic_tension: 0.2,
                cognitive_load: 0.1,
                satisfaction_valence: 0.5,
                urgency_arousal: 0.1,
                last_updated_ms: crate::now_millis(),
                is_heuristic_telemetry: true,
                is_degraded_blocked: false,
            })),
            decay_rate: safe_decay,
        }
    }

    /// Read current computational state snapshot.
    pub fn get_state(&self) -> ComputationalAffectState {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Loads persisted state from JSON. On corruption or invalid bounds, strictly FAILS CLOSED into
    /// DEGRADED/BLOCKED mode (C-276) rather than silently falling back to nominal baseline.
    pub fn load_state_from_json(&self, raw_json: &str) -> Result<ComputationalAffectState, String> {
        let parsed: Result<ComputationalAffectState, _> = serde_json::from_str(raw_json);
        match parsed {
            Ok(loaded) => {
                // Validate all numerical bounds
                if !loaded.epistemic_tension.is_finite()
                    || loaded.epistemic_tension < 0.0
                    || loaded.epistemic_tension > 1.0
                    || !loaded.cognitive_load.is_finite()
                    || loaded.cognitive_load < 0.0
                    || loaded.cognitive_load > 1.0
                    || !loaded.satisfaction_valence.is_finite()
                    || loaded.satisfaction_valence < -1.0
                    || loaded.satisfaction_valence > 1.0
                    || !loaded.urgency_arousal.is_finite()
                    || loaded.urgency_arousal < 0.0
                    || loaded.urgency_arousal > 1.0
                {
                    let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
                    st.is_degraded_blocked = true;
                    return Err("CorruptedAffectState: numerical bounds invalid; failed closed into DEGRADED/BLOCKED mode".to_string());
                }
                let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
                *st = loaded.clone();
                Ok(loaded)
            }
            Err(e) => {
                let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
                st.is_degraded_blocked = true;
                Err(format!("CorruptedAffectState: JSON parse failed ({}); failed closed into DEGRADED/BLOCKED mode", e))
            }
        }
    }

    /// Strictly validates float inputs and rejects out-of-range, NaN, or Infinity values (C-273, C-274).
    /// Fail-closed input gate: out-of-range values are strictly REJECTED, not silently clamped.
    pub fn try_update_from_turn_event(
        &self,
        entropy_delta: f64,
        memory_saturation: f64,
        reward_delta: f64,
        is_error: bool,
    ) -> Result<ComputationalAffectState, String> {
        let st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if st.is_degraded_blocked {
            return Err("AffectEngineBlocked: system is in DEGRADED/BLOCKED mode due to prior state corruption".into());
        }
        drop(st);

        if !entropy_delta.is_finite() || entropy_delta < -1.0 || entropy_delta > 1.0 {
            return Err("entropy_delta must be in range [-1.0, 1.0] and finite: out-of-range rejected (fail-closed)".into());
        }
        if !memory_saturation.is_finite() || memory_saturation < 0.0 || memory_saturation > 1.0 {
            return Err("memory_saturation must be in range [0.0, 1.0] and finite: out-of-range rejected (fail-closed)".into());
        }
        if !reward_delta.is_finite() || reward_delta < -1.0 || reward_delta > 1.0 {
            return Err("reward_delta must be in range [-1.0, 1.0] and finite: out-of-range rejected (fail-closed)".into());
        }
        Ok(self.update_from_turn_event(entropy_delta, memory_saturation, reward_delta, is_error))
    }

    /// Update homeostatic state based on cognitive events:
    /// - `entropy_delta`: Increase in epistemic uncertainty (sanitized and clamped)
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
        // Sanitize all floating point inputs against NaN and Infinity
        let safe_entropy = if entropy_delta.is_finite() {
            entropy_delta.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let safe_saturation = if memory_saturation.is_finite() {
            memory_saturation.clamp(0.0, 1.0)
        } else {
            0.0
        };

        let safe_reward = if reward_delta.is_finite() {
            reward_delta.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if st.is_degraded_blocked {
            return st.clone();
        }

        // 1. Epistemic tension: decays toward baseline 0.1, influenced by entropy impulse
        let baseline_tension = 0.1;
        let impulse = (safe_entropy * 0.5).clamp(-0.5, 0.5);
        st.epistemic_tension = ((st.epistemic_tension + impulse) * (1.0 - self.decay_rate)
            + baseline_tension * self.decay_rate)
            .clamp(0.0, 1.0);

        // 2. Cognitive load: tracks memory saturation and search complexity
        st.cognitive_load = (st.cognitive_load * 0.7 + safe_saturation * 0.3).clamp(0.0, 1.0);

        // 3. Satisfaction valence: updated via exponential moving average of reward
        let alpha_reward = 0.2;
        st.satisfaction_valence = ((1.0 - alpha_reward) * st.satisfaction_valence
            + alpha_reward * safe_reward)
            .clamp(-1.0, 1.0);

        // 4. Urgency arousal: spikes on errors/security alerts, decays rapidly toward 0.05
        if is_error {
            st.urgency_arousal = (st.urgency_arousal + 0.4).min(1.0);
        } else {
            st.urgency_arousal = (st.urgency_arousal * 0.85 + 0.05 * 0.15).clamp(0.0, 1.0);
        }

        st.last_updated_ms = crate::now_millis();
        st.clone()
    }

    /// Decays internal regulation variables toward baseline homeostasis.
    pub fn decay_step(&self, _duration: Duration) {
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if st.is_degraded_blocked {
            return;
        }
        st.epistemic_tension = (st.epistemic_tension * (1.0 - self.decay_rate)
            + 0.1 * self.decay_rate)
            .clamp(0.0, 1.0);
        st.cognitive_load = (st.cognitive_load * 0.8).clamp(0.0, 1.0);
        st.satisfaction_valence = (st.satisfaction_valence * 0.8).clamp(-1.0, 1.0);
        st.urgency_arousal = (st.urgency_arousal * 0.8 + 0.05 * 0.2).clamp(0.0, 1.0);
        st.last_updated_ms = crate::now_millis();
    }

    /// Compute behavioral adaptation parameters from current numerical state:
    /// - `exploration_temperature`: [0.1, 1.5]. Higher tension & low satisfaction -> higher exploration.
    /// - `max_deliberation_depth`: Max steps for tree search. Throttled under high cognitive load or urgent arousal.
    /// - `verification_threshold`: Required confidence threshold. Raised under high urgency or low valence.
    ///
    /// Note: Output is non-authoritative heuristic guidance only.
    pub fn compute_behavioral_modulation(&self) -> Value {
        let st = self.state.lock().unwrap_or_else(|p| p.into_inner());

        if st.is_degraded_blocked {
            return json!({
                "heuristic_telemetry": true,
                "authoritative_control": false,
                "is_degraded_blocked": true,
                "status": "DEGRADED_BLOCKED",
                "exploration_temperature": 0.1,
                "max_deliberation_depth": 0,
                "verification_threshold": 1.0,
                "security_blocked": true,
                "epistemic_tension": 1.0,
                "cognitive_load": 1.0,
                "satisfaction_valence": -1.0,
                "urgency_arousal": 1.0
            });
        }

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
            "heuristic_telemetry": true,
            "authoritative_control": false,
            "is_degraded_blocked": false,
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
    fn test_affect_input_rejection_fail_closed() {
        let engine = AffectiveHomeostasisEngine::new();

        // Valid inputs pass
        assert!(engine
            .try_update_from_turn_event(0.5, 0.5, 0.5, false)
            .is_ok());

        // Out-of-bounds entropy_delta (> 1.0) must be rejected
        let res_high = engine.try_update_from_turn_event(1.5, 0.5, 0.5, false);
        assert!(
            res_high.is_err(),
            "Out-of-range entropy > 1.0 must be rejected"
        );

        // Out-of-bounds entropy_delta (< -1.0) must be rejected
        let res_low = engine.try_update_from_turn_event(-1.5, 0.5, 0.5, false);
        assert!(
            res_low.is_err(),
            "Out-of-range entropy < -1.0 must be rejected"
        );

        // NaN must be rejected
        let res_nan = engine.try_update_from_turn_event(f64::NAN, 0.5, 0.5, false);
        assert!(res_nan.is_err(), "NaN must be rejected");

        // Out-of-bounds memory_saturation (> 1.0) must be rejected
        let res_sat = engine.try_update_from_turn_event(0.0, 2.0, 0.0, false);
        assert!(
            res_sat.is_err(),
            "Out-of-range memory saturation must be rejected"
        );

        // Out-of-bounds reward_delta (> 1.0) must be rejected
        let res_rew = engine.try_update_from_turn_event(0.0, 0.5, 10.0, false);
        assert!(
            res_rew.is_err(),
            "Out-of-range reward delta must be rejected"
        );
    }

    #[test]
    fn test_affect_corrupted_state_fails_closed_to_degraded_blocked() {
        let engine = AffectiveHomeostasisEngine::new();

        // Corrupted JSON must fail closed to DEGRADED_BLOCKED mode
        let corrupt_json = "{ invalid_json: true }";
        let load_res = engine.load_state_from_json(corrupt_json);
        assert!(load_res.is_err());

        // Engine must now be in DEGRADED/BLOCKED mode
        let st = engine.get_state();
        assert!(
            st.is_degraded_blocked,
            "Corrupted state must flag is_degraded_blocked"
        );

        // Subsequent update attempts must be blocked
        let update_res = engine.try_update_from_turn_event(0.1, 0.1, 0.1, false);
        assert!(update_res.is_err());
        assert!(update_res.unwrap_err().contains("AffectEngineBlocked"));

        // Behavioral modulation must fail closed with max_deliberation_depth = 0 and verification_threshold = 1.0
        let modulation = engine.compute_behavioral_modulation();
        assert_eq!(modulation["status"], "DEGRADED_BLOCKED");
        assert_eq!(modulation["max_deliberation_depth"], 0);
        assert_eq!(modulation["verification_threshold"], 1.0);
        assert_eq!(modulation["security_blocked"], true);
    }

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
        assert_eq!(mod1["max_deliberation_depth"], 4);
        assert_eq!(mod1["verification_threshold"], 0.95);
        assert_eq!(mod1["heuristic_telemetry"], true);

        // Simulate recovery with success
        let st2 = engine.update_from_turn_event(0.0, 0.2, 1.0, false);
        assert!(st2.satisfaction_valence > st1.satisfaction_valence);
    }

    #[test]
    fn test_epistemic_tension_decays_toward_baseline() {
        let engine = AffectiveHomeostasisEngine::with_decay(0.10);
        // Start from high tension
        let mut st = engine.update_from_turn_event(1.0, 0.0, 0.0, false);
        let start_tension = st.epistemic_tension;
        assert!(start_tension > 0.1);

        // With zero entropy, tension must strictly decay towards baseline 0.1
        for _ in 0..10 {
            st = engine.update_from_turn_event(0.0, 0.0, 0.0, false);
        }
        assert!(st.epistemic_tension < start_tension);
        assert!((st.epistemic_tension - 0.1).abs() < 0.2);
    }
}
