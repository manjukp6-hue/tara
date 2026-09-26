//! lib.rs
//! 
//! TARA AI Core: Hardened, Memory-Safe Security & Control Architecture.
//! Enforces Creator Authority, Rule-First Execution, Capability Permissions, and Sandboxing.

pub mod governance;
pub mod rule_engine;
pub mod permissions;
pub mod memory;
pub mod sandbox;
pub mod protocol;
pub mod runtime;

pub use governance::{CreatorAuthority, Identity, Role};
pub use rule_engine::{RuleEngine, Rule, RuleDecision};
pub use permissions::{PermissionEngine, Capability};
pub use memory::{MemoryEngine, MemoryRecord, MemoryType};
pub use sandbox::{ExecutionSandbox, SandboxPolicy, ExecutionResult};
pub use protocol::{CoreRequest, CoreResponse, CoreProtocolHandler};
pub use runtime::{TaraManager, ManagerConfig, IsolatedSandbox, SandboxConfig, Agent, Worker, Team, DynamicNamingManager};

