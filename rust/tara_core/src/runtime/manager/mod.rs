//! manager/mod.rs
//!
//! Generic, extensible TARA Manager controlling Sandboxes, Agents, Workers, Teams, and Resources.
//! Implements the complete dynamic lifecycle:
//! TASK -> CREATE -> NAME -> ASSIGN -> WORK -> VERIFY -> RELEASE -> CLEANUP
//! Supports dynamic scaling under load and strict security supervisor separation.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use crate::runtime::lifecycle::{EntityState, EntityType};
use crate::runtime::naming::DynamicNamingManager;
use crate::runtime::sandbox::{IsolatedSandbox, SandboxConfig, SandboxExecutionResult};
use crate::runtime::agent::Agent;
use crate::runtime::worker::Worker;
use crate::runtime::team::{Team, TeamType};
use crate::runtime::authorization::{AuthorizationGate, ActionType};
use crate::runtime::approval::ApprovalGate;
use crate::runtime::resource::{ResourceGovernor, ResourceQuota};
use crate::runtime::tools::ToolController;
use crate::runtime::network::{NetworkController, NetworkPermission};
use crate::runtime::health::{HealthMonitor, EntityHealthStatus};
use crate::runtime::recovery::{RecoveryOrchestrator, FailureCategory};

/// Master configuration for dynamic manager instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagerConfig {
    pub manager_id: String,
    pub display_name: String,
    pub storage_base_dir: PathBuf,
    pub is_supervisor: bool,
    pub max_concurrent_sandboxes: usize,
}

/// Generic extensible Manager control interface.
pub trait RuntimeManagerOps {
    fn create_sandbox(&mut self, sandbox_type: &str, preferred_name: Option<&str>) -> Result<(String, String), String>;
    fn start_sandbox(&mut self, sandbox_id: &str) -> Result<(), String>;
    fn assign_agent(&mut self, agent_id: &str, task_id: &str, sandbox_id: &str) -> Result<(), String>;
    fn reassign_agent(&mut self, agent_id: &str, new_sandbox_id: &str) -> Result<(), String>;
    fn pause(&mut self, entity_id: &str) -> Result<(), String>;
    fn resume(&mut self, entity_id: &str) -> Result<(), String>;
    fn stop(&mut self, entity_id: &str) -> Result<(), String>;
    fn release(&mut self, entity_id: &str) -> Result<(), String>;
    fn promote(&mut self, entity_id: &str) -> Result<bool, String>;
    fn demote(&mut self, entity_id: &str) -> Result<bool, String>;
    fn retire(&mut self, entity_id: &str, reason: &str) -> Result<(), String>;
    fn delete(&mut self, entity_id: &str) -> Result<(), String>;
    fn status(&self, entity_id: &str) -> Result<EntityState, String>;
    fn health(&self, entity_id: &str) -> Result<EntityHealthStatus, String>;
    fn resource_quota(&mut self, entity_id: &str, quota: ResourceQuota) -> Result<(), String>;
    fn tool_assign(&mut self, sandbox_id: &str, tools: &[String]) -> Result<(), String>;
    fn network_grant(&mut self, sandbox_id: &str, task_id: &str, domains: &[String], duration_ms: u64) -> Result<(), String>;
}

/// Complete production TaraManager.
pub struct TaraManager {
    pub config: ManagerConfig,
    pub naming: Arc<DynamicNamingManager>,
    pub sandboxes: HashMap<String, IsolatedSandbox>,
    pub agents: HashMap<String, Agent>,
    pub workers: HashMap<String, Worker>,
    pub teams: HashMap<String, Team>,
    pub approval_gate: ApprovalGate,
    pub resource_governor: ResourceGovernor,
    pub tool_controller: ToolController,
    pub network_controller: NetworkController,
    pub health_monitor: HealthMonitor,
    pub recovery: RecoveryOrchestrator,
    pub is_system_lockdown: bool,
}

impl TaraManager {
    pub fn new(config: ManagerConfig, naming: Option<Arc<DynamicNamingManager>>) -> Self {
        let naming = naming.unwrap_or_else(|| Arc::new(DynamicNamingManager::new()));
        Self {
            config,
            naming,
            sandboxes: HashMap::new(),
            agents: HashMap::new(),
            workers: HashMap::new(),
            teams: HashMap::new(),
            approval_gate: ApprovalGate::new(),
            resource_governor: ResourceGovernor::new(300),
            tool_controller: ToolController::new(),
            network_controller: NetworkController::new(),
            health_monitor: HealthMonitor::new(60),
            recovery: RecoveryOrchestrator::new(),
            is_system_lockdown: false,
        }
    }

    /// Dynamic Task Execution Flow:
    /// TASK -> CREATE -> NAME -> ASSIGN -> WORK -> VERIFY -> RELEASE -> CLEANUP
    pub fn execute_dynamic_task(
        &mut self,
        task_id: &str,
        domain_type: &str,
        required_tools: &[String],
        allow_network_domains: Option<&[String]>,
        workload_cmd: &str,
        workload_args: &[&str],
        stdin: Option<&[u8]>,
    ) -> Result<SandboxExecutionResult, String> {
        // 1. Authorize action
        let caller_caps = HashSet::from(["*".to_string()]);
        let auth = AuthorizationGate::evaluate(
            &self.config.manager_id,
            EntityState::Active,
            &ActionType::CreateSandbox,
            None,
            None,
            &caller_caps,
            self.is_system_lockdown,
        );
        if !auth.is_allowed() {
            return Err("Task authorization denied".to_string());
        }

        // 2. CREATE + NAME Sandbox
        let (sbx_id, _sbx_name) = self.create_sandbox(domain_type, None)?;

        // 3. Assign task-scoped tools
        if !required_tools.is_empty() {
            self.tool_assign(&sbx_id, required_tools)?;
        }

        // 4. Grant temporary network if required
        if let Some(domains) = allow_network_domains {
            if !domains.is_empty() {
                self.network_grant(&sbx_id, task_id, domains, 15_000)?;
            }
        }

        // 5. WORK (Execute inside isolated sandbox)
        let sbx = self.sandboxes.get_mut(&sbx_id).ok_or("Sandbox not found")?;
        let exec_result = sbx.execute(workload_cmd, workload_args, stdin);

        // 6. VERIFY output and failure classification
        let final_result = match exec_result {
            Ok(res) => {
                if res.exit_code != 0 || res.timed_out {
                    let cat = if res.timed_out { FailureCategory::Timeout } else { FailureCategory::ProcessCrash };
                    let _ = self.recovery.handle_failure(&sbx_id, cat, &res.stderr);
                }
                res
            }
            Err(e) => {
                let _ = self.recovery.handle_failure(&sbx_id, FailureCategory::SecurityViolation, &e);
                return Err(format!("Execution failed: {}", e));
            }
        };

        // 7. RELEASE + CLEANUP (Automatic on-demand resource recycling)
        let _ = self.release(&sbx_id);

        Ok(final_result)
    }

    /// Spawns a new Agent buddy on demand.
    pub fn create_agent(&mut self, role: &str, capabilities: &[String], preferred_name: Option<&str>) -> Result<(String, String), String> {
        let identity = self.naming.register_entity(EntityType::Agent, role, preferred_name)?;
        let agent = Agent::new(
            identity.internal_id.clone(),
            identity.display_name.clone(),
            role.to_string(),
            capabilities.iter().cloned().collect(),
        );
        self.agents.insert(identity.internal_id.clone(), agent);
        Ok((identity.internal_id, identity.display_name))
    }

    /// Spawns a new Worker buddy on demand.
    pub fn create_worker(&mut self, specialization: &str, capabilities: &[String], preferred_name: Option<&str>) -> Result<(String, String), String> {
        let identity = self.naming.register_entity(EntityType::Worker, specialization, preferred_name)?;
        let worker = Worker::new(
            identity.internal_id.clone(),
            identity.display_name.clone(),
            specialization.to_string(),
            capabilities.iter().cloned().collect(),
        );
        self.workers.insert(identity.internal_id.clone(), worker);
        Ok((identity.internal_id, identity.display_name))
    }

    /// Forms a temporary collaborative Team.
    pub fn form_team(&mut self, team_type: TeamType, objective: &str, member_ids: &[String], preferred_name: Option<&str>) -> Result<(String, String), String> {
        let identity = self.naming.register_entity(EntityType::Team, objective, preferred_name)?;
        let team = Team::new(
            identity.internal_id.clone(),
            identity.display_name.clone(),
            team_type,
            objective.to_string(),
            member_ids.iter().cloned().collect(),
        );
        self.teams.insert(identity.internal_id.clone(), team);
        Ok((identity.internal_id, identity.display_name))
    }

    /// Dissolves a Team and returns members to their independent pools.
    pub fn dissolve_team(&mut self, team_id: &str) -> Result<Vec<String>, String> {
        let mut team = self.teams.remove(team_id).ok_or_else(|| format!("Team '{}' not found", team_id))?;
        let freed_members = team.dissolve("Team objective concluded; dynamic dissolution")?;
        let _ = self.naming.retire_entity(team_id, "Team dissolved");
        Ok(freed_members)
    }
}

impl RuntimeManagerOps for TaraManager {
    fn create_sandbox(&mut self, sandbox_type: &str, preferred_name: Option<&str>) -> Result<(String, String), String> {
        let identity = self.naming.register_entity(EntityType::Sandbox, sandbox_type, preferred_name)?;
        let sbx_root = self.config.storage_base_dir.join("sandboxes").join(&identity.internal_id);

        let config = SandboxConfig {
            sandbox_id: identity.internal_id.clone(),
            display_name: identity.display_name.clone(),
            sandbox_type: sandbox_type.to_string(),
            root_dir: sbx_root,
            max_memory_bytes: 256 * 1024 * 1024,
            max_processes: 4,
            max_disk_bytes: 512 * 1024 * 1024,
            execution_timeout_ms: 15_000,
            allow_network: false,
            allowed_domains: Vec::new(),
            allowed_tools: HashSet::new(),
            environment_variables: HashMap::new(),
        };

        let sandbox = IsolatedSandbox::new(config)?;
        self.sandboxes.insert(identity.internal_id.clone(), sandbox);
        Ok((identity.internal_id, identity.display_name))
    }

    fn start_sandbox(&mut self, sandbox_id: &str) -> Result<(), String> {
        let sbx = self.sandboxes.get_mut(sandbox_id).ok_or_else(|| format!("Sandbox '{}' not found", sandbox_id))?;
        sbx.state.transition_to(EntityState::Active, "Sandbox started", "MANAGER")?;
        Ok(())
    }

    fn assign_agent(&mut self, agent_id: &str, task_id: &str, sandbox_id: &str) -> Result<(), String> {
        let agent = self.agents.get_mut(agent_id).ok_or_else(|| format!("Agent '{}' not found", agent_id))?;
        agent.assign_task(task_id, sandbox_id)
    }

    fn reassign_agent(&mut self, agent_id: &str, new_sandbox_id: &str) -> Result<(), String> {
        let agent = self.agents.get_mut(agent_id).ok_or_else(|| format!("Agent '{}' not found", agent_id))?;
        agent.assigned_sandbox_id = Some(new_sandbox_id.to_string());
        Ok(())
    }

    fn pause(&mut self, entity_id: &str) -> Result<(), String> {
        if let Some(sbx) = self.sandboxes.get_mut(entity_id) {
            sbx.state.transition_to(EntityState::Paused, "Paused by manager", "MANAGER")?;
            return Ok(());
        }
        if let Some(agent) = self.agents.get_mut(entity_id) {
            agent.state_machine.transition_to(EntityState::Paused, "Paused by manager", "MANAGER")?;
            return Ok(());
        }
        Err(format!("Entity '{}' not found to pause", entity_id))
    }

    fn resume(&mut self, entity_id: &str) -> Result<(), String> {
        if let Some(sbx) = self.sandboxes.get_mut(entity_id) {
            sbx.state.transition_to(EntityState::Active, "Resumed by manager", "MANAGER")?;
            return Ok(());
        }
        if let Some(agent) = self.agents.get_mut(entity_id) {
            agent.state_machine.transition_to(EntityState::Active, "Resumed by manager", "MANAGER")?;
            return Ok(());
        }
        Err(format!("Entity '{}' not found to resume", entity_id))
    }

    fn stop(&mut self, entity_id: &str) -> Result<(), String> {
        if let Some(sbx) = self.sandboxes.get_mut(entity_id) {
            return sbx.stop();
        }
        if let Some(agent) = self.agents.get_mut(entity_id) {
            agent.state_machine.transition_to(EntityState::Stopped, "Stopped by manager", "MANAGER")?;
            return Ok(());
        }
        Err(format!("Entity '{}' not found to stop", entity_id))
    }

    fn release(&mut self, entity_id: &str) -> Result<(), String> {
        if let Some(mut sbx) = self.sandboxes.remove(entity_id) {
            sbx.cleanup()?;
            let _ = self.naming.retire_entity(entity_id, "Sandbox released and cleaned up");
            self.tool_controller.revoke_sandbox_tools(entity_id);
            self.network_controller.revoke_network(entity_id);
            return Ok(());
        }
        Err(format!("Entity '{}' not found to release", entity_id))
    }

    fn promote(&mut self, entity_id: &str) -> Result<bool, String> {
        if let Some(agent) = self.agents.get_mut(entity_id) {
            return agent.evaluate_promotion();
        }
        if let Some(worker) = self.workers.get_mut(entity_id) {
            return worker.evaluate_promotion();
        }
        Err(format!("Buddy '{}' not found for promotion evaluation", entity_id))
    }

    fn demote(&mut self, entity_id: &str) -> Result<bool, String> {
        if let Some(agent) = self.agents.get_mut(entity_id) {
            if agent.performance.rank_tier > 1 {
                agent.performance.rank_tier -= 1;
                return Ok(true);
            }
            return Ok(false);
        }
        if let Some(worker) = self.workers.get_mut(entity_id) {
            if worker.performance.rank_tier > 1 {
                worker.performance.rank_tier -= 1;
                return Ok(true);
            }
            return Ok(false);
        }
        Err(format!("Buddy '{}' not found for demotion", entity_id))
    }

    fn retire(&mut self, entity_id: &str, reason: &str) -> Result<(), String> {
        if let Some(agent) = self.agents.get_mut(entity_id) {
            agent.state_machine.transition_to(EntityState::Retired, reason, "MANAGER")?;
            return self.naming.retire_entity(entity_id, reason);
        }
        if let Some(worker) = self.workers.get_mut(entity_id) {
            worker.state_machine.transition_to(EntityState::Retired, reason, "MANAGER")?;
            return self.naming.retire_entity(entity_id, reason);
        }
        self.release(entity_id)
    }

    fn delete(&mut self, entity_id: &str) -> Result<(), String> {
        self.release(entity_id)
    }

    fn status(&self, entity_id: &str) -> Result<EntityState, String> {
        if let Some(sbx) = self.sandboxes.get(entity_id) {
            return Ok(sbx.state.current_state());
        }
        if let Some(agent) = self.agents.get(entity_id) {
            return Ok(agent.state_machine.current_state());
        }
        if let Some(worker) = self.workers.get(entity_id) {
            return Ok(worker.state_machine.current_state());
        }
        if let Some(team) = self.teams.get(entity_id) {
            return Ok(team.state_machine.current_state());
        }
        Err(format!("Entity '{}' not found", entity_id))
    }

    fn health(&self, entity_id: &str) -> Result<EntityHealthStatus, String> {
        let current_state = self.status(entity_id)?;
        Ok(self.health_monitor.evaluate_health(entity_id, current_state))
    }

    fn resource_quota(&mut self, entity_id: &str, quota: ResourceQuota) -> Result<(), String> {
        self.resource_governor.assign_quota(entity_id, quota);
        Ok(())
    }

    fn tool_assign(&mut self, sandbox_id: &str, tools: &[String]) -> Result<(), String> {
        self.tool_controller.assign_tools_to_sandbox(sandbox_id, tools)
    }

    fn network_grant(&mut self, sandbox_id: &str, task_id: &str, domains: &[String], duration_ms: u64) -> Result<(), String> {
        self.network_controller.grant_network(sandbox_id, task_id, domains, NetworkPermission::ReadOnly, duration_ms);
        Ok(())
    }
}
