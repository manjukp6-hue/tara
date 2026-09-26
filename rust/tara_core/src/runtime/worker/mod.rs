//! worker/mod.rs
//!
//! Independent TARA Worker buddy lifecycle, specialization, and performance.
//! Workers are execution buddies specialized in discrete domains (compute, simulation, parsing, indexing, etc.).

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use crate::runtime::agent::PerformanceRecord;
use crate::runtime::lifecycle::{EntityState, StateMachine, EntityType};

/// Worker entity struct representing an execution-focused buddy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Worker {
    pub internal_id: String,
    pub display_name: String,
    pub specialization: String,
    pub capabilities: HashSet<String>,
    pub state_machine: StateMachine,
    pub performance: PerformanceRecord,
    pub assigned_sandbox_id: Option<String>,
    pub current_task_id: Option<String>,
    pub team_id: Option<String>,
    pub task_history: Vec<String>,
    pub team_history: Vec<String>,
}

impl Worker {
    pub fn new(internal_id: String, display_name: String, specialization: String, capabilities: HashSet<String>) -> Self {
        let mut state_machine = StateMachine::new(internal_id.clone(), EntityType::Worker);
        let _ = state_machine.transition_to(EntityState::Ready, "Worker initialized and ready", "MANAGER");
        Self {
            internal_id,
            display_name,
            specialization,
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
        self.state_machine.transition_to(EntityState::Busy, &format!("Worker assigned to task {}", task_id), "MANAGER")?;
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
            self.state_machine.transition_to(EntityState::Quarantined, "Security violation recorded in worker", "SECURITY_GATE")?;
        } else if success {
            self.state_machine.transition_to(EntityState::Idle, "Workload completed successfully", "MANAGER")?;
        } else {
            self.state_machine.transition_to(EntityState::Degraded, "Workload execution failure", "MANAGER")?;
        }
        Ok(())
    }

    pub fn evaluate_promotion(&mut self) -> Result<bool, String> {
        if self.performance.security_violations > 0 {
            self.performance.rank_tier = 1;
            return Ok(false);
        }

        if self.performance.total_tasks >= 10
            && self.performance.success_rate() >= 0.92
            && self.performance.quality_score >= 0.88
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
