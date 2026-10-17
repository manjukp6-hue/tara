//! Continuous Risk-Adaptive Zero-Trust Engine (UEBA & Dynamic Step-Up Enforcement).
//!
//! High-Security Guarantee:
//! - Continuous Authentication: Trust is never static; every operation is evaluated for contextual risk.
//! - Dynamic Step-Up Authentication: Escalates required assurance levels in real-time when anomalous
//!   velocity, IP drift, or suspicious patterns are detected.
//! - Instant Tripwire Lockdown: Automatically locks down the system if risk score exceeds critical ceiling.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::primitives::{AssuranceLevel, PrincipalId, SecurityAction, SecurityContext};

/// Dynamic Risk Assessment Verdict.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskVerdict {
    /// Normal operation; standard policy applies.
    Accept { risk_score: u32 },
    /// Elevated risk; requires Step-Up to Hardware Attestation (WebAuthn / TPM).
    RequireHardwareStepUp { risk_score: u32, reason: String },
    /// High risk; requires Step-Up to Creator Root multi-factor passkey.
    RequireCreatorStepUp { risk_score: u32, reason: String },
    /// Critical threat detected; triggers immediate fail-closed security lockdown!
    TriggerLockdown { risk_score: u32, reason: String },
}

/// Tracking metadata per principal for velocity and anomaly calculations.
#[derive(Debug, Clone)]
struct PrincipalTelemetry {
    recent_requests: Vec<Instant>,
    recent_failures: Vec<Instant>, // Timestamped sliding-window failures (ID-18)
    last_known_ip: String,
}

pub const MAX_TELEMETRY_PRINCIPALS: usize = 10_000;
pub const FAILURE_WINDOW_SECS: u64 = 900; // 15-minute sliding window

/// Continuous Zero-Trust Risk Engine.
pub struct ContinuousRiskEngine {
    telemetry: Arc<Mutex<HashMap<String, PrincipalTelemetry>>>,
    max_requests_per_minute: usize,
}

impl Default for ContinuousRiskEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ContinuousRiskEngine {
    pub fn new() -> Self {
        Self {
            telemetry: Arc::new(Mutex::new(HashMap::new())),
            max_requests_per_minute: 120,
        }
    }

    /// Evaluates current risk for a principal request and returns the risk verdict.
    pub fn evaluate_risk(
        &self,
        principal_id: &PrincipalId,
        action: &SecurityAction,
        context: &SecurityContext,
    ) -> Result<RiskVerdict, String> {
        let mut map = self
            .telemetry
            .lock()
            .map_err(|_| "Risk telemetry lock poisoned")?;
        let now = Instant::now();
        let key = principal_id.as_str();

        if map.len() >= MAX_TELEMETRY_PRINCIPALS && !map.contains_key(&key) {
            // Evict inactive entries
            map.retain(|_, v| !v.recent_requests.is_empty() || !v.recent_failures.is_empty());
            if map.len() >= MAX_TELEMETRY_PRINCIPALS {
                if let Some(victim_key) = map.keys().next().cloned() {
                    map.remove(&victim_key);
                }
            }
        }

        let entry = map
            .entry(key.clone())
            .or_insert_with(|| PrincipalTelemetry {
                recent_requests: Vec::new(),
                recent_failures: Vec::new(),
                last_known_ip: context.client_ip.clone(),
            });

        // 1. Prune requests older than 60 seconds
        entry
            .recent_requests
            .retain(|ts| now.duration_since(*ts).as_secs() < 60);
        entry.recent_requests.push(now);

        // 2. Prune failures older than 15-minute sliding window (ID-18)
        entry
            .recent_failures
            .retain(|ts| now.duration_since(*ts).as_secs() < FAILURE_WINDOW_SECS);

        let mut risk_score: u32 = 0;
        let mut reasons = Vec::new();

        // 3. Velocity evaluation (Request Rate Spikes)
        let count_last_minute = entry.recent_requests.len();
        if count_last_minute > self.max_requests_per_minute {
            let overflow = count_last_minute - self.max_requests_per_minute;
            risk_score += (overflow * 15).min(400) as u32;
            reasons.push(format!(
                "High velocity: {} requests in past 60s",
                count_last_minute
            ));
        }

        // 4. Network Anomaly evaluation (IP Drift)
        if !entry.last_known_ip.is_empty() && entry.last_known_ip != context.client_ip {
            risk_score += 250;
            reasons.push(format!(
                "IP address drift: {} -> {}",
                entry.last_known_ip, context.client_ip
            ));
            entry.last_known_ip = context.client_ip.clone();
        }

        // 5. Action Severity Weight (Critical operations inherently demand Creator-level assurance)
        if action.verb() == "rotate_key" || action.verb() == "lift_lockdown" {
            risk_score += 600;
            reasons.push("Critical security administrative action invoked".to_string());
        }

        // 6. Sliding Window Failed Verification Penalty (ID-18)
        let failure_count = entry.recent_failures.len() as u32;
        if failure_count > 0 {
            let failure_penalty = (failure_count * 200).min(500);
            risk_score += failure_penalty;
            reasons.push(format!(
                "Active failed verification attempts in window: {}",
                failure_count
            ));
        }

        // 7. Compute Verdict based on Dynamic Risk Thresholds
        let reason_summary = reasons.join("; ");

        if risk_score >= 900 && failure_count >= 2 {
            Ok(RiskVerdict::TriggerLockdown {
                risk_score,
                reason: format!("Critical risk threshold breached with repeated verification failures (score {}): {}", risk_score, reason_summary),
            })
        } else if risk_score >= 600 {
            Ok(RiskVerdict::RequireCreatorStepUp {
                risk_score,
                reason: format!(
                    "High risk detected (score {}): {}",
                    risk_score, reason_summary
                ),
            })
        } else if risk_score >= 200 {
            Ok(RiskVerdict::RequireHardwareStepUp {
                risk_score,
                reason: format!(
                    "Medium risk detected (score {}): {}",
                    risk_score, reason_summary
                ),
            })
        } else {
            Ok(RiskVerdict::Accept { risk_score })
        }
    }

    /// Record failed verification to penalize suspicious clients within the sliding window (ID-18).
    pub fn record_failure(&self, principal_id: &PrincipalId) {
        if let Ok(mut map) = self.telemetry.lock() {
            let entry = map
                .entry(principal_id.as_str())
                .or_insert_with(|| PrincipalTelemetry {
                    recent_requests: Vec::new(),
                    recent_failures: Vec::new(),
                    last_known_ip: String::new(),
                });
            let now = Instant::now();
            entry
                .recent_failures
                .retain(|ts| now.duration_since(*ts).as_secs() < FAILURE_WINDOW_SECS);
            entry.recent_failures.push(now);
        }
    }

    /// Determine minimum required assurance level based on current risk verdict.
    pub fn required_assurance_level(verdict: &RiskVerdict) -> AssuranceLevel {
        match verdict {
            RiskVerdict::Accept { .. } => AssuranceLevel::STANDARD_TOKEN,
            RiskVerdict::RequireHardwareStepUp { .. } => AssuranceLevel::HARDWARE_ATTESTED,
            RiskVerdict::RequireCreatorStepUp { .. } => AssuranceLevel::CREATOR_ROOT,
            RiskVerdict::TriggerLockdown { .. } => AssuranceLevel::CREATOR_ROOT,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_telemetry_eviction_when_capacity_reached() {
        let engine = ContinuousRiskEngine::new();
        let action = SecurityAction::new("core", "read").unwrap();
        let context = SecurityContext::new("127.0.0.1", 1_700_000_000);

        // Pre-populate telemetry with MAX_TELEMETRY_PRINCIPALS entries
        {
            let mut map = engine.telemetry.lock().unwrap();
            for i in 0..MAX_TELEMETRY_PRINCIPALS {
                map.insert(
                    format!("principal_{}", i),
                    PrincipalTelemetry {
                        recent_requests: vec![Instant::now()],
                        recent_failures: Vec::new(),
                        last_known_ip: "127.0.0.1".into(),
                    },
                );
            }
            assert_eq!(map.len(), MAX_TELEMETRY_PRINCIPALS);
        }

        // Now evaluate risk for a completely new principal
        let new_pid = PrincipalId::new("principal_new").unwrap();
        let verdict = engine.evaluate_risk(&new_pid, &action, &context);
        assert!(verdict.is_ok());

        // Verify that map size did NOT exceed MAX_TELEMETRY_PRINCIPALS
        let map = engine.telemetry.lock().unwrap();
        assert_eq!(map.len(), MAX_TELEMETRY_PRINCIPALS);
        assert!(map.contains_key(&new_pid.as_str()));
    }
}
