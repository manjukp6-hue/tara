//! agent/mod.rs
//!
//! Independent TARA Agent buddy lifecycle and performance tracking.
//! Agents are long-term collaborative helpers with dynamic names, role definitions,
//! verified performance records, and promotion/demotion pathways.

use crate::runtime::lifecycle::{EntityState, EntityType, StateMachine};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

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

    pub fn record_task(
        &mut self,
        success: bool,
        duration_ms: u64,
        quality: f32,
        security_violation: bool,
    ) {
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

pub const DEFAULT_QUALITY_SMOOTHING_MOMENTUM: f32 = 0.8;
pub const DEFAULT_QUALITY_SMOOTHING_INNOVATION: f32 = 0.2;

pub const DEFAULT_AGENT_PROMOTION_MIN_TASKS: u64 = 10;
pub const DEFAULT_AGENT_PROMOTION_MIN_SUCCESS: f32 = 0.90;
pub const DEFAULT_AGENT_PROMOTION_MIN_QUALITY: f32 = 0.85;
pub const DEFAULT_AGENT_MAX_RANK_TIER: u32 = 3;
pub const DEFAULT_AGENT_DEMOTION_MIN_TASKS: u64 = 5;
pub const DEFAULT_AGENT_DEMOTION_SUCCESS_RATE: f32 = 0.60;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionCriteria {
    pub min_tasks_promotion: u64,
    pub min_success_rate: f32,
    pub min_quality_score: f32,
    pub max_rank_tier: u32,
    pub min_tasks_demotion: u64,
    pub demotion_success_rate: f32,
}

impl Default for PromotionCriteria {
    fn default() -> Self {
        let min_tasks_promotion = std::env::var("TARA_AGENT_PROMOTION_MIN_TASKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_AGENT_PROMOTION_MIN_TASKS);
        let min_success_rate = std::env::var("TARA_AGENT_MIN_SUCCESS_RATE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_AGENT_PROMOTION_MIN_SUCCESS);
        let min_quality_score = std::env::var("TARA_AGENT_MIN_QUALITY_SCORE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_AGENT_PROMOTION_MIN_QUALITY);
        let max_rank_tier = std::env::var("TARA_AGENT_MAX_RANK_TIER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_AGENT_MAX_RANK_TIER);
        let min_tasks_demotion = std::env::var("TARA_AGENT_DEMOTION_MIN_TASKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_AGENT_DEMOTION_MIN_TASKS);
        let demotion_success_rate = std::env::var("TARA_AGENT_DEMOTION_SUCCESS_RATE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_AGENT_DEMOTION_SUCCESS_RATE);

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

impl Agent {
    pub fn new(
        internal_id: String,
        display_name: String,
        role: String,
        capabilities: HashSet<String>,
    ) -> Self {
        let mut state_machine = StateMachine::new(internal_id.clone(), EntityType::Agent);
        let _ = state_machine.transition_to(
            EntityState::Ready,
            "Agent initialized and ready",
            "MANAGER",
        );
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
        self.state_machine.transition_to(
            EntityState::Busy,
            &format!("Assigned to task {}", task_id),
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
                "Security violation recorded during task execution",
                "SECURITY_GATE",
            )?;
        } else if success {
            self.state_machine.transition_to(
                EntityState::Idle,
                "Task successfully completed",
                "MANAGER",
            )?;
        } else {
            self.state_machine.transition_to(
                EntityState::Degraded,
                "Task failed; placed in degraded observation",
                "MANAGER",
            )?;
        }
        Ok(())
    }

    /// Evaluates verified metrics for promotion or demotion using dynamic criteria.
    /// Invariant: Promotion does NOT automatically grant higher security authority.
    pub fn evaluate_promotion_with_criteria(
        &mut self,
        criteria: &PromotionCriteria,
    ) -> Result<bool, String> {
        if self.performance.security_violations > 0 {
            // Demote on any security violations
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
        self.evaluate_promotion_with_criteria(&PromotionCriteria::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_task_lifecycle_success() {
        let mut agent = Agent::new(
            "agt_test_1".to_string(),
            "Buddy-1".to_string(),
            "Researcher".to_string(),
            HashSet::new(),
        );

        assert_eq!(agent.state_machine.current_state(), EntityState::Ready);
        assert!(agent.assign_task("tsk_01", "sbx_01").is_ok());
        assert_eq!(agent.state_machine.current_state(), EntityState::Busy);

        assert!(agent.complete_task(true, 500, 0.95, false).is_ok());
        assert_eq!(agent.state_machine.current_state(), EntityState::Idle);
        assert_eq!(agent.performance.total_tasks, 1);
        assert_eq!(agent.performance.successful_tasks, 1);
    }

    #[test]
    fn test_agent_security_violation_quarantine() {
        let mut agent = Agent::new(
            "agt_test_2".to_string(),
            "Buddy-2".to_string(),
            "Coder".to_string(),
            HashSet::new(),
        );
        agent.assign_task("tsk_02", "sbx_02").unwrap();

        // Completing with security violation forces quarantine
        assert!(agent.complete_task(false, 200, 0.0, true).is_ok());
        assert_eq!(agent.state_machine.current_state(), EntityState::Quarantined);
        assert_eq!(agent.performance.security_violations, 1);

        // Security violations automatically demote rank tier to 1
        agent.performance.rank_tier = 2;
        let promoted = agent.evaluate_promotion().unwrap();
        assert!(!promoted);
        assert_eq!(agent.performance.rank_tier, 1);
    }

    #[test]
    fn test_agent_promotion_and_demotion() {
        let mut agent = Agent::new(
            "agt_test_3".to_string(),
            "Buddy-3".to_string(),
            "Analyst".to_string(),
            HashSet::new(),
        );

        // Simulate 10 high-quality tasks
        for i in 0..10 {
            agent.assign_task(&format!("tsk_{}", i), "sbx").unwrap();
            agent.complete_task(true, 100, 0.95, false).unwrap();
        }

        assert_eq!(agent.performance.rank_tier, 1);
        let promoted = agent.evaluate_promotion().unwrap();
        assert!(promoted);
        assert_eq!(agent.performance.rank_tier, 2);
    }
}

