//! agent/mod.rs
//!
//! Independent TARA Agent buddy lifecycle and performance tracking.
//! Agents are long-term collaborative helpers with dynamic names, role definitions,
//! verified performance records, and promotion/demotion pathways.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use crate::runtime::lifecycle::{EntityState, StateMachine, EntityType};

/// Verified performance and reliability metrics for an Agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecord {
    pub total_tasks: u64,
    pub successful_tasks: u64,
    pub failed_tasks: u64,
    pub total_execution_time_ms: u64,
    pub security_violations: u32,
    pub quality_score: f32, // 0.0 - 1.0
    pub rank_tier: u32,     // 1: Standard, 2: Veteran, 3: Elite Buddy
}

impl Default for PerformanceRecord {
    fn default() -> Self {
        Self {
            total_tasks: 0,
            successful_tasks: 0,
            failed_tasks: 0,
            total_execution_time_ms: 0,
            security_violations: 0,
            quality_score: 1.0,
            rank_tier: 1,
        }
    }
}

impl PerformanceRecord {
    pub fn success_rate(&self) -> f32 {
        if self.total_tasks == 0 {
            1.0
        } else {
            self.successful_tasks as f32 / self.total_tasks as f32
        }
    }

    pub fn record_task(&mut self, success: bool, duration_ms: u64, quality: f32, security_violation: bool) {
        self.total_tasks += 1;
        if success {
            self.successful_tasks += 1;
        } else {
            self.failed_tasks += 1;
        }
        self.total_execution_time_ms += duration_ms;
        if security_violation {
            self.security_violations += 1;
        }

        // Rolling exponential smoothing for quality score
        self.quality_score = (self.quality_score * 0.8) + (quality.clamp(0.0, 1.0) * 0.2);
    }
}

/// Agent entity struct representing an autonomous, identifiable assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub internal_id: String,
    pub display_name: String,
    pub role: String,
    pub capabilities: HashSet<String>,
    pub state_machine: StateMachine,
    pub performance: PerformanceRecord,
    pub assigned_sandbox_id: Option<String>,
    pub current_task_id: Option<String>,
    pub team_id: Option<String>,
    pub task_history: Vec<String>,
    pub team_history: Vec<String>,
}

impl Agent {
    pub fn new(internal_id: String, display_name: String, role: String, capabilities: HashSet<String>) -> Self {
        let mut state_machine = StateMachine::new(internal_id.clone(), EntityType::Agent);
        let _ = state_machine.transition_to(EntityState::Ready, "Agent initialized and ready", "MANAGER");
        Self {
            internal_id,
            display_name,
            role,
            capabilities,
            state_machine,
            performance: PerformanceRecord::default(),
            assigned_sandbox_id: None,
            current_task_id: None,
            team_id: None,
            task_history: Vec::new(),
            team_history: Vec::new(),
        }
    }

    pub fn assign_task(&mut self, task_id: &str, sandbox_id: &str) -> Result<(), String> {
        self.state_machine.transition_to(EntityState::Busy, &format!("Assigned to task {}", task_id), "MANAGER")?;
        self.current_task_id = Some(task_id.to_string());
        self.assigned_sandbox_id = Some(sandbox_id.to_string());
        self.task_history.push(task_id.to_string());
        Ok(())
    }

    pub fn complete_task(&mut self, success: bool, duration_ms: u64, quality: f32, security_violation: bool) -> Result<(), String> {
        self.performance.record_task(success, duration_ms, quality, security_violation);
        self.current_task_id = None;
        self.assigned_sandbox_id = None;

        if security_violation {
            self.state_machine.transition_to(EntityState::Quarantined, "Security violation recorded during task execution", "SECURITY_GATE")?;
        } else if success {
            self.state_machine.transition_to(EntityState::Idle, "Task successfully completed", "MANAGER")?;
        } else {
            self.state_machine.transition_to(EntityState::Degraded, "Task failed; placed in degraded observation", "MANAGER")?;
        }
        Ok(())
    }

    /// Evaluates verified metrics for promotion or demotion.
    /// Invariant: Promotion does NOT automatically grant higher security authority.
    pub fn evaluate_promotion(&mut self) -> Result<bool, String> {
        if self.performance.security_violations > 0 {
            // Demote on any security violations
            self.performance.rank_tier = 1;
            return Ok(false);
        }

        if self.performance.total_tasks >= 10
            && self.performance.success_rate() >= 0.90
            && self.performance.quality_score >= 0.85
        {
            if self.performance.rank_tier < 3 {
                self.performance.rank_tier += 1;
                return Ok(true);
            }
        } else if self.performance.total_tasks >= 5 && self.performance.success_rate() < 0.60 {
            if self.performance.rank_tier > 1 {
                self.performance.rank_tier -= 1;
            }
        }
        Ok(false)
    }
}
