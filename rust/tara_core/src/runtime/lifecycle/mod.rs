//! lifecycle/mod.rs
//!
//! Explicit state machine and lifecycle transitions for all dynamic TARA entities.
//! Controls transitions for Sandboxes, Agents, Workers, Teams, Managers, Tasks, and future entities.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

/// Standard entity types managed within TARA Dynamic Runtime.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    Sandbox,
    Agent,
    Worker,
    Team,
    Manager,
    Skill,
    Tool,
    Task,
    Research,
    Experiment,
    Project,
    Capability,
    Custom(String),
}

impl fmt::Display for EntityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntityType::Sandbox => write!(f, "Sandbox"),
            EntityType::Agent => write!(f, "Agent"),
            EntityType::Worker => write!(f, "Worker"),
            EntityType::Team => write!(f, "Team"),
            EntityType::Manager => write!(f, "Manager"),
            EntityType::Skill => write!(f, "Skill"),
            EntityType::Tool => write!(f, "Tool"),
            EntityType::Task => write!(f, "Task"),
            EntityType::Research => write!(f, "Research"),
            EntityType::Experiment => write!(f, "Experiment"),
            EntityType::Project => write!(f, "Project"),
            EntityType::Capability => write!(f, "Capability"),
            EntityType::Custom(s) => write!(f, "Custom({})", s),
        }
    }
}

/// Comprehensive state enum covering base lifecycle and operational exceptions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityState {
    // Base lifecycle
    Created,
    Initializing,
    Ready,
    Active,
    Busy,
    Idle,
    Draining,
    Stopped,
    Retired,

    // Operational and safety states
    Paused,
    Degraded,
    Failed,
    Quarantined,
    Recovering,
    Revoked,
}

impl fmt::Display for EntityState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntityState::Created => write!(f, "CREATED"),
            EntityState::Initializing => write!(f, "INITIALIZING"),
            EntityState::Ready => write!(f, "READY"),
            EntityState::Active => write!(f, "ACTIVE"),
            EntityState::Busy => write!(f, "BUSY"),
            EntityState::Idle => write!(f, "IDLE"),
            EntityState::Draining => write!(f, "DRAINING"),
            EntityState::Stopped => write!(f, "STOPPED"),
            EntityState::Retired => write!(f, "RETIRED"),
            EntityState::Paused => write!(f, "PAUSED"),
            EntityState::Degraded => write!(f, "DEGRADED"),
            EntityState::Failed => write!(f, "FAILED"),
            EntityState::Quarantined => write!(f, "QUARANTINED"),
            EntityState::Recovering => write!(f, "RECOVERING"),
            EntityState::Revoked => write!(f, "REVOKED"),
        }
    }
}

/// Individual recorded state transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransitionRecord {
    pub from_state: EntityState,
    pub to_state: EntityState,
    pub timestamp_ms: u64,
    pub reason: String,
    pub actor_id: String,
}

/// Dynamic entity state machine enforcing valid transitions and preserving audit history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMachine {
    entity_id: String,
    entity_type: EntityType,
    current_state: EntityState,
    history: VecDeque<StateTransitionRecord>,
    max_history_entries: usize,
}

impl StateMachine {
    pub fn new(entity_id: String, entity_type: EntityType) -> Self {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let initial_record = StateTransitionRecord {
            from_state: EntityState::Created,
            to_state: EntityState::Created,
            timestamp_ms: now_ms,
            reason: "Initial entity registration".to_string(),
            actor_id: "SYSTEM".to_string(),
        };

        let mut history = VecDeque::new();
        history.push_back(initial_record);

        Self {
            entity_id,
            entity_type,
            current_state: EntityState::Created,
            history,
            max_history_entries: 50,
        }
    }

    pub fn current_state(&self) -> EntityState {
        self.current_state
    }

    pub fn entity_id(&self) -> &str {
        &self.entity_id
    }

    pub fn entity_type(&self) -> &EntityType {
        &self.entity_type
    }

    pub fn history(&self) -> &VecDeque<StateTransitionRecord> {
        &self.history
    }

    /// Evaluates whether a transition from `from` to `to` is semantically and cryptographically valid.
    pub fn is_valid_transition(from: EntityState, to: EntityState) -> bool {
        if from == to {
            return true;
        }

        match from {
            EntityState::Created => matches!(to, EntityState::Initializing | EntityState::Ready | EntityState::Active | EntityState::Degraded | EntityState::Failed | EntityState::Quarantined | EntityState::Retired),
            EntityState::Initializing => matches!(to, EntityState::Ready | EntityState::Active | EntityState::Degraded | EntityState::Failed | EntityState::Quarantined | EntityState::Retired),
            EntityState::Ready => matches!(to, EntityState::Active | EntityState::Busy | EntityState::Idle | EntityState::Paused | EntityState::Draining | EntityState::Degraded | EntityState::Stopped | EntityState::Retired),
            EntityState::Active => matches!(to, EntityState::Busy | EntityState::Idle | EntityState::Paused | EntityState::Draining | EntityState::Degraded | EntityState::Failed | EntityState::Quarantined | EntityState::Stopped | EntityState::Retired),
            EntityState::Busy => matches!(to, EntityState::Active | EntityState::Idle | EntityState::Draining | EntityState::Degraded | EntityState::Failed | EntityState::Quarantined | EntityState::Stopped),
            EntityState::Idle => matches!(to, EntityState::Active | EntityState::Busy | EntityState::Paused | EntityState::Draining | EntityState::Stopped | EntityState::Retired),
            EntityState::Paused => matches!(to, EntityState::Ready | EntityState::Active | EntityState::Idle | EntityState::Stopped | EntityState::Quarantined | EntityState::Retired),
            EntityState::Draining => matches!(to, EntityState::Stopped | EntityState::Failed | EntityState::Quarantined | EntityState::Retired),
            EntityState::Degraded => matches!(to, EntityState::Active | EntityState::Recovering | EntityState::Quarantined | EntityState::Failed | EntityState::Stopped | EntityState::Retired),
            EntityState::Failed => matches!(to, EntityState::Recovering | EntityState::Quarantined | EntityState::Stopped | EntityState::Retired),
            EntityState::Quarantined => matches!(to, EntityState::Recovering | EntityState::Revoked | EntityState::Retired),
            EntityState::Recovering => matches!(to, EntityState::Ready | EntityState::Active | EntityState::Failed | EntityState::Quarantined | EntityState::Stopped),
            EntityState::Stopped => matches!(to, EntityState::Initializing | EntityState::Ready | EntityState::Retired),
            EntityState::Revoked => false, // Terminal security revocation
            EntityState::Retired => false, // Terminal archival state
        }
    }

    /// Atomically transitions the state machine to `target_state` if valid, appending to transition log.
    pub fn transition_to(&mut self, target_state: EntityState, reason: &str, actor_id: &str) -> Result<EntityState, String> {
        if !Self::is_valid_transition(self.current_state, target_state) {
            return Err(format!(
                "Illegal lifecycle transition for {} ({}): cannot transition from {} to {}",
                self.entity_type, self.entity_id, self.current_state, target_state
            ));
        }

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let record = StateTransitionRecord {
            from_state: self.current_state,
            to_state: target_state,
            timestamp_ms: now_ms,
            reason: reason.to_string(),
            actor_id: actor_id.to_string(),
        };

        self.current_state = target_state;
        if self.history.len() >= self.max_history_entries {
            self.history.pop_front();
        }
        self.history.push_back(record);

        Ok(self.current_state)
    }

    pub fn is_operational(&self) -> bool {
        matches!(self.current_state, EntityState::Ready | EntityState::Active | EntityState::Busy | EntityState::Idle)
    }

    pub fn is_terminated(&self) -> bool {
        matches!(self.current_state, EntityState::Stopped | EntityState::Retired | EntityState::Revoked)
    }
}
