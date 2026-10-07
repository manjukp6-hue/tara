//! lib.rs
//!
//! TARA AI Core: Hardened, Memory-Safe Security & Control Architecture.
//! Enforces Creator Authority, Rule-First Execution, Capability Permissions, and Sandboxing.

pub mod destruction;
pub mod governance;
pub mod memory;
pub mod permissions;
pub mod protocol;
pub mod robotics;
pub mod rule_engine;
pub mod runtime;
pub mod sandbox;

pub use destruction::{DestructionReport, SecureDestructionEngine};
pub use governance::{CreatorAuthority, Identity, Role};
pub use memory::{MemoryEngine, MemoryRecord, MemoryType};
pub use permissions::{Capability, PermissionEngine};
pub use protocol::{CoreProtocolHandler, CoreRequest, CoreResponse};
pub use robotics::{ActuatorJoint, KalmanStateEstimator, PidController, RoboticsHal};
pub use rule_engine::{Rule, RuleDecision, RuleEngine};
pub use runtime::{
    Agent, DynamicNamingManager, IsolatedSandbox, ManagerConfig, SandboxConfig, TaraManager, Team,
    Worker,
};
pub use sandbox::{ExecutionResult, ExecutionSandbox, SandboxPolicy};
