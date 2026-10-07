//! worker/mod.rs
//!
//! Independent TARA Worker buddy lifecycle, specialization, and performance.
//! Workers are execution buddies specialized in discrete domains (compute, simulation, parsing, indexing, etc.).

use crate::runtime::agent::PerformanceRecord;
use crate::runtime::lifecycle::{EntityState, EntityType, StateMachine};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

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

pub const DEFAULT_WORKER_PROMOTION_MIN_TASKS: u64 = 10;
pub const DEFAULT_WORKER_PROMOTION_MIN_SUCCESS: f32 = 0.92;
pub const DEFAULT_WORKER_PROMOTION_MIN_QUALITY: f32 = 0.88;
pub const DEFAULT_WORKER_MAX_RANK_TIER: u32 = 3;
pub const DEFAULT_WORKER_DEMOTION_MIN_TASKS: u64 = 5;
pub const DEFAULT_WORKER_DEMOTION_SUCCESS_RATE: f32 = 0.60;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerPromotionCriteria {
    pub min_tasks_promotion: u64,
    pub min_success_rate: f32,
    pub min_quality_score: f32,
    pub max_rank_tier: u32,
    pub min_tasks_demotion: u64,
    pub demotion_success_rate: f32,
}

impl Default for WorkerPromotionCriteria {
    fn default() -> Self {
        let min_tasks_promotion = std::env::var("TARA_WORKER_PROMOTION_MIN_TASKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_WORKER_PROMOTION_MIN_TASKS);
        let min_success_rate = std::env::var("TARA_WORKER_MIN_SUCCESS_RATE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_WORKER_PROMOTION_MIN_SUCCESS);
        let min_quality_score = std::env::var("TARA_WORKER_MIN_QUALITY_SCORE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_WORKER_PROMOTION_MIN_QUALITY);
        let max_rank_tier = std::env::var("TARA_WORKER_MAX_RANK_TIER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_WORKER_MAX_RANK_TIER);
        let min_tasks_demotion = std::env::var("TARA_WORKER_DEMOTION_MIN_TASKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_WORKER_DEMOTION_MIN_TASKS);
        let demotion_success_rate = std::env::var("TARA_WORKER_DEMOTION_SUCCESS_RATE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_WORKER_DEMOTION_SUCCESS_RATE);

        Self {
            min_tasks_promotion,
            min_success_rate,
            min_quality_score,
            max_rank_tier,
            min_tasks_demotion,
            demotion_success_rate,
        }
    }
}

impl Worker {
    pub fn new(
        internal_id: String,
        display_name: String,
        specialization: String,
        capabilities: HashSet<String>,
    ) -> Self {
        let mut state_machine = StateMachine::new(internal_id.clone(), EntityType::Worker);
        let _ = state_machine.transition_to(
            EntityState::Ready,
            "Worker initialized and ready",
            "MANAGER",
        );
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
        self.state_machine.transition_to(
            EntityState::Busy,
            &format!("Worker assigned to task {}", task_id),
            "MANAGER",
        )?;
        self.current_task_id = Some(task_id.to_string());
        self.assigned_sandbox_id = Some(sandbox_id.to_string());
        self.task_history.push(task_id.to_string());
        Ok(())
    }

    pub fn complete_task(
        &mut self,
        success: bool,
        duration_ms: u64,
        quality: f32,
        security_violation: bool,
    ) -> Result<(), String> {
        self.performance
            .record_task(success, duration_ms, quality, security_violation);
        self.current_task_id = None;
        self.assigned_sandbox_id = None;

        if security_violation {
            self.state_machine.transition_to(
                EntityState::Quarantined,
                "Security violation recorded in worker",
                "SECURITY_GATE",
            )?;
        } else if success {
            self.state_machine.transition_to(
                EntityState::Idle,
                "Workload completed successfully",
                "MANAGER",
            )?;
        } else {
            self.state_machine.transition_to(
                EntityState::Degraded,
                "Workload execution failure",
                "MANAGER",
            )?;
        }
        Ok(())
    }

    pub fn evaluate_promotion_with_criteria(
        &mut self,
        criteria: &WorkerPromotionCriteria,
    ) -> Result<bool, String> {
        if self.performance.security_violations > 0 {
            self.performance.rank_tier = 1;
            return Ok(false);
        }

        if self.performance.total_tasks >= criteria.min_tasks_promotion
            && self.performance.success_rate() >= criteria.min_success_rate
            && self.performance.quality_score >= criteria.min_quality_score
        {
            if self.performance.rank_tier < criteria.max_rank_tier {
                self.performance.rank_tier += 1;
                return Ok(true);
            }
        } else if self.performance.total_tasks >= criteria.min_tasks_demotion
            && self.performance.success_rate() < criteria.demotion_success_rate
            && self.performance.rank_tier > 1
        {
            self.performance.rank_tier -= 1;
        }
        Ok(false)
    }

    pub fn evaluate_promotion(&mut self) -> Result<bool, String> {
        self.evaluate_promotion_with_criteria(&WorkerPromotionCriteria::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_task_lifecycle_and_promotion() {
        let mut worker = Worker::new(
            "wrk_01".to_string(),
            "Worker-Compute".to_string(),
            "LinearAlgebra".to_string(),
            HashSet::new(),
        );

        assert_eq!(worker.state_machine.current_state(), EntityState::Ready);
        assert!(worker.assign_task("tsk_math_01", "sbx_01").is_ok());
        assert_eq!(worker.state_machine.current_state(), EntityState::Busy);

        // Success transition
        assert!(worker.complete_task(true, 150, 0.96, false).is_ok());
        assert_eq!(worker.state_machine.current_state(), EntityState::Idle);

        // Simulate 9 more successful tasks for promotion
        for i in 2..=10 {
            worker.assign_task(&format!("tsk_{}", i), "sbx").unwrap();
            worker.complete_task(true, 100, 0.95, false).unwrap();
        }

        assert_eq!(worker.performance.rank_tier, 1);
        let promoted = worker.evaluate_promotion().unwrap();
        assert!(promoted);
        assert_eq!(worker.performance.rank_tier, 2);
    }

    #[test]
    fn test_worker_failure_degraded() {
        let mut worker = Worker::new(
            "wrk_02".to_string(),
            "Worker-Sim".to_string(),
            "Simulation".to_string(),
            HashSet::new(),
        );
        worker.assign_task("tsk_fail", "sbx").unwrap();

        // Failed task transitions to Degraded
        assert!(worker.complete_task(false, 300, 0.2, false).is_ok());
        assert_eq!(worker.state_machine.current_state(), EntityState::Degraded);
    }
}

