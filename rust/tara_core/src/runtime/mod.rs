//! runtime/mod.rs
//!
//! TARA Dynamic Runtime Engine: Dynamic Sandboxes, Agents, Workers, Teams, and Manager Capacity.
//! Strict kernel-enforced OS-level isolation, host protection, dynamic naming, and full lifecycle control.

pub mod agent;
pub mod approval;
pub mod authorization;
pub mod health;
pub mod lifecycle;
pub mod manager;
pub mod naming;
pub mod network;
pub mod recovery;
pub mod resource;
pub mod sandbox;
pub mod team;
pub mod tools;
pub mod worker;

pub use agent::{Agent, PerformanceRecord};
pub use approval::{ApprovalGate, ApprovalTicket, RiskLevel};
pub use authorization::{ActionType, AuthorizationGate, AuthorizationResult};
pub use health::{EntityHealthStatus, HealthMonitor};
pub use lifecycle::{EntityState, EntityType, StateMachine, StateTransitionRecord};
pub use manager::{ManagerConfig, RuntimeManagerOps, TaraManager};
pub use naming::{DynamicNamingManager, EntityIdentity, RetiredNameEntry};
pub use network::{NetworkController, NetworkGrant, NetworkPermission};
pub use recovery::{FailureCategory, IncidentReport, RecoveryOrchestrator};
pub use resource::{ResourceGovernor, ResourceQuota, ResourceUsage};
pub use sandbox::{
    BrokeredTransferRequest, BrokeredTransferResponse, IsolatedSandbox, SandboxBroker,
    SandboxConfig, SandboxExecutionResult,
};
pub use team::{Team, TeamType};
pub use tools::{ToolController, ToolDefinition};
pub use worker::Worker;
