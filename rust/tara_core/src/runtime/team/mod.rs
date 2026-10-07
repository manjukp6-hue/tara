//! team/mod.rs
//!
//! Temporary Teams for Agent and Worker collaboration.
//! Formed on-demand when composite multi-step tasks require joint execution.
//! Dissolved immediately upon task completion, returning members to their independent pools.

use crate::runtime::lifecycle::{EntityState, EntityType, StateMachine};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TeamType {
    AgentTeam,
    WorkerTeam,
    HybridTeam,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Team {
    pub internal_id: String,
    pub display_name: String,
    pub team_type: TeamType,
    pub objective: String,
    pub member_ids: HashSet<String>,
    pub state_machine: StateMachine,
    pub created_at_ms: u64,
    pub dissolved_at_ms: Option<u64>,
}

impl Team {
    pub fn new(
        internal_id: String,
        display_name: String,
        team_type: TeamType,
        objective: String,
        member_ids: HashSet<String>,
    ) -> Self {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let mut state_machine = StateMachine::new(internal_id.clone(), EntityType::Team);
        let _ = state_machine.transition_to(
            EntityState::Ready,
            "Team initialized with members",
            "MANAGER",
        );
        let _ = state_machine.transition_to(
            EntityState::Active,
            "Team active for objective",
            "MANAGER",
        );

        Self {
            internal_id,
            display_name,
            team_type,
            objective,
            member_ids,
            state_machine,
            created_at_ms: now_ms,
            dissolved_at_ms: None,
        }
    }

    pub fn add_member(&mut self, entity_id: &str) -> Result<(), String> {
        if !self.state_machine.is_operational() {
            return Err("Cannot add members to non-operational team".to_string());
        }
        self.member_ids.insert(entity_id.to_string());
        Ok(())
    }

    pub fn remove_member(&mut self, entity_id: &str) -> Result<(), String> {
        self.member_ids.remove(entity_id);
        Ok(())
    }

    /// Dissolves the team, setting its state to Retired and recording dissolution timestamp.
    pub fn dissolve(&mut self, reason: &str) -> Result<Vec<String>, String> {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        self.dissolved_at_ms = Some(now_ms);
        self.state_machine
            .transition_to(EntityState::Retired, reason, "MANAGER")?;

        let freed_members: Vec<String> = self.member_ids.drain().collect();
        Ok(freed_members)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_team_lifecycle_and_dissolution() {
        let mut members = HashSet::new();
        members.insert("agt_1".to_string());
        members.insert("wrk_1".to_string());

        let mut team = Team::new(
            "team_01".to_string(),
            "Research-Pod".to_string(),
            TeamType::HybridTeam,
            "Analyze dataset".to_string(),
            members,
        );

        assert_eq!(team.state_machine.current_state(), EntityState::Active);
        assert_eq!(team.member_ids.len(), 2);

        // Add member
        assert!(team.add_member("wrk_2").is_ok());
        assert_eq!(team.member_ids.len(), 3);

        // Remove member
        assert!(team.remove_member("agt_1").is_ok());
        assert_eq!(team.member_ids.len(), 2);

        // Dissolve
        let freed = team.dissolve("Goal reached").unwrap();
        assert_eq!(freed.len(), 2);
        assert_eq!(team.state_machine.current_state(), EntityState::Retired);
        assert!(team.dissolved_at_ms.is_some());

        // Cannot add members after retirement
        assert!(team.add_member("agt_new").is_err());
    }
}

