//! runtime/mod.rs
//!
//! TARA Dynamic Runtime Engine: Dynamic Sandboxes, Agents, Workers, Teams, and Manager Capacity.
//! Strict kernel-enforced OS-level isolation, host protection, dynamic naming, and full lifecycle control.

pub mod lifecycle;
pub mod naming;
pub mod sandbox;
pub mod agent;
pub mod worker;
pub mod team;
pub mod authorization;
pub mod approval;
pub mod resource;
pub mod tools;
pub mod network;
pub mod recovery;
pub mod health;
pub mod manager;

pub use lifecycle::{EntityState, EntityType, StateMachine, StateTransitionRecord};
pub use naming::{DynamicNamingManager, EntityIdentity, RetiredNameEntry};
pub use sandbox::{
    IsolatedSandbox, SandboxConfig, SandboxExecutionResult,
    SandboxBroker, BrokeredTransferRequest, BrokeredTransferResponse,
};
pub use agent::{Agent, PerformanceRecord};
pub use worker::Worker;
pub use team::{Team, TeamType};
pub use authorization::{AuthorizationGate, ActionType, AuthorizationResult};
pub use approval::{ApprovalGate, ApprovalTicket, RiskLevel};
pub use resource::{ResourceGovernor, ResourceQuota, ResourceUsage};
pub use tools::{ToolController, ToolDefinition};
pub use network::{NetworkController, NetworkGrant, NetworkPermission};
pub use recovery::{RecoveryOrchestrator, IncidentReport, FailureCategory};
pub use health::{HealthMonitor, EntityHealthStatus};
pub use manager::{TaraManager, ManagerConfig, RuntimeManagerOps};
