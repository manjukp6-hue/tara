//! test_runtime.rs
//!
//! Comprehensive test suite for TARA Dynamic Runtime:
//! Sandboxes, Agents, Workers, Teams, Naming, Isolation, Authorization, Tools, Network, and Recovery.

use std::collections::HashSet;
use std::fs;
use std::path::Path;
use tara_core::runtime::{
    TaraManager, ManagerConfig, RuntimeManagerOps,
    DynamicNamingManager, EntityType, EntityState, StateMachine,
    IsolatedSandbox, SandboxConfig, SandboxBroker, BrokeredTransferRequest,
    Agent, Worker, TeamType,
    AuthorizationGate, ActionType, AuthorizationResult,
    ApprovalGate, RiskLevel,
    ResourceGovernor, ResourceQuota,
    ToolController, ToolDefinition,
    NetworkController, NetworkPermission,
    RecoveryOrchestrator, FailureCategory,
};

#[test]
fn test_01_dynamic_naming_manager() {
    let naming = DynamicNamingManager::new();

    // 1. Contextual selection based on domain
    let math_name = naming.choose_name(&EntityType::Sandbox, "mathematics", None);
    assert!(!math_name.is_empty());

    let med_name = naming.choose_name(&EntityType::Sandbox, "medicine-surgery", None);
    assert!(!med_name.is_empty());
    assert_ne!(math_name, med_name);

    // 2. ID + Name registration
    let entity = naming.register_entity(EntityType::Agent, "calculus-research", None).unwrap();
    assert!(entity.internal_id.starts_with("agt_"));
    assert!(!entity.display_name.is_empty());

    // 3. Collision avoidance
    let duplicate_attempt = naming.register_entity(EntityType::Agent, "calculus-research", Some(&entity.display_name));
    assert!(duplicate_attempt.is_err(), "Should reject duplicate active name");

    // 4. Renaming changes display name while ID remains immutable
    let original_id = entity.internal_id.clone();
    let renamed = naming.rename_entity(&original_id, "Unique-New-Buddy-Name").unwrap();
    assert_eq!(renamed.internal_id, original_id);
    assert_eq!(renamed.display_name, "Unique-New-Buddy-Name");

    // 5. Retired name tracking
    naming.retire_entity(&original_id, "Retiring for test").unwrap();
    let retired = naming.list_retired_names();
    assert!(retired.iter().any(|r| r.name == "Unique-New-Buddy-Name"));
}

#[test]
fn test_02_lifecycle_state_machine() {
    let mut sm = StateMachine::new("sbx_test_123".to_string(), EntityType::Sandbox);
    assert_eq!(sm.current_state(), EntityState::Created);

    // Valid progression
    assert!(sm.transition_to(EntityState::Initializing, "Init", "MGR").is_ok());
    assert!(sm.transition_to(EntityState::Ready, "Ready", "MGR").is_ok());
    assert!(sm.transition_to(EntityState::Active, "Active", "MGR").is_ok());
    assert!(sm.transition_to(EntityState::Busy, "Busy", "MGR").is_ok());
    assert!(sm.transition_to(EntityState::Idle, "Idle", "MGR").is_ok());
    assert!(sm.transition_to(EntityState::Stopped, "Stopped", "MGR").is_ok());

    // Invalid transition blocked
    let invalid = sm.transition_to(EntityState::Busy, "Cannot jump from Stopped to Busy", "MGR");
    assert!(invalid.is_err());
    assert_eq!(sm.current_state(), EntityState::Stopped);
}

#[test]
fn test_03_isolated_sandbox_host_protection() {
    let temp_root = std::env::temp_dir().join(format!("tara_sbx_test_{}", rand::random::<u32>()));
    let config = SandboxConfig {
        sandbox_id: "sbx_sec_01".to_string(),
        display_name: "Security-Test-Box".to_string(),
        sandbox_type: "Code".to_string(),
        root_dir: temp_root.clone(),
        max_memory_bytes: 128 * 1024 * 1024,
        max_processes: 2,
        max_disk_bytes: 512 * 1024 * 1024,
        execution_timeout_ms: 5000,
        allow_network: false,
        allowed_domains: vec![],
        allowed_tools: HashSet::new(),
        environment_variables: Default::default(),
    };

    let sandbox = IsolatedSandbox::new(config).unwrap();

    // 1. Safe file write inside sandbox
    let safe_file = sandbox.write_file(Path::new("workspace/test.txt"), b"Sandbox Safe Data").unwrap();
    assert!(safe_file.exists());
    let read_back = sandbox.read_file(Path::new("workspace/test.txt")).unwrap();
    assert_eq!(read_back, b"Sandbox Safe Data");

    // 2. Path traversal attack blocked
    let traversal_attempt = sandbox.validate_path_safety(Path::new("../../sensitive_host_file.txt"));
    assert!(traversal_attempt.is_err(), "Path traversal must be blocked");

    // Clean up
    let _ = fs::remove_dir_all(&temp_root);
}

#[test]
fn test_04_sandbox_broker_cross_isolation() {
    let root_a = std::env::temp_dir().join(format!("tara_sbx_a_{}", rand::random::<u32>()));
    let root_b = std::env::temp_dir().join(format!("tara_sbx_b_{}", rand::random::<u32>()));

    let mut cfg_a = SandboxConfig::default();
    cfg_a.sandbox_id = "sbx_source".to_string();
    cfg_a.root_dir = root_a.clone();
    let sbx_a = IsolatedSandbox::new(cfg_a).unwrap();

    let mut cfg_b = SandboxConfig::default();
    cfg_b.sandbox_id = "sbx_target".to_string();
    cfg_b.root_dir = root_b.clone();
    let sbx_b = IsolatedSandbox::new(cfg_b).unwrap();

    // Valid transfer
    let req = BrokeredTransferRequest {
        source_sandbox_id: "sbx_source".to_string(),
        target_sandbox_id: "sbx_target".to_string(),
        purpose: "Transfer verified calculation results".to_string(),
        payload_type: "application/json".to_string(),
        payload_bytes: b"{\"result\": 42}".to_vec(),
        max_size_bytes: 1024,
    };

    let resp = SandboxBroker::transfer(&sbx_a, &sbx_b, req);
    assert!(resp.success);
    assert!(!resp.sha256_hash.is_empty());
    assert_eq!(resp.bytes_transferred, 14);

    // Mismatched sender rejected
    let bad_req = BrokeredTransferRequest {
        source_sandbox_id: "sbx_rogue".to_string(),
        target_sandbox_id: "sbx_target".to_string(),
        purpose: "Unauthorized transfer".to_string(),
        payload_type: "text/plain".to_string(),
        payload_bytes: b"Attack".to_vec(),
        max_size_bytes: 1024,
    };
    let bad_resp = SandboxBroker::transfer(&sbx_a, &sbx_b, bad_req);
    assert!(!bad_resp.success);

    let _ = fs::remove_dir_all(&root_a);
    let _ = fs::remove_dir_all(&root_b);
}

#[test]
fn test_05_agent_and_worker_buddy_lifecycle() {
    let mut agent = Agent::new(
        "agt_001".to_string(),
        "Active-Dynamic-Agent".to_string(),
        "Math".to_string(),
        HashSet::from(["tool:math".to_string()]),
    );

    assert_eq!(agent.performance.rank_tier, 1);

    // Assign and complete tasks
    agent.assign_task("task_01", "sbx_01").unwrap();
    assert_eq!(agent.state_machine.current_state(), EntityState::Busy);

    agent.complete_task(true, 120, 0.95, false).unwrap();
    assert_eq!(agent.state_machine.current_state(), EntityState::Idle);

    // Simulate 10 successful tasks for promotion
    for i in 2..=11 {
        agent.assign_task(&format!("task_{}", i), "sbx_01").unwrap();
        agent.complete_task(true, 100, 0.92, false).unwrap();
    }

    let promoted = agent.evaluate_promotion().unwrap();
    assert!(promoted);
    assert_eq!(agent.performance.rank_tier, 2);

    // Security violation demotes immediately
    agent.assign_task("task_bad", "sbx_01").unwrap();
    agent.complete_task(false, 50, 0.0, true).unwrap();
    assert_eq!(agent.state_machine.current_state(), EntityState::Quarantined);
    assert_eq!(agent.performance.security_violations, 1);
    let _ = agent.evaluate_promotion();
    assert_eq!(agent.performance.rank_tier, 1);
}

#[test]
fn test_06_team_lifecycle() {
    let mut team = tara_core::runtime::Team::new(
        "team_alpha".to_string(),
        "Team-Arjuna".to_string(),
        TeamType::AgentTeam,
        "Composite Research & Math".to_string(),
        HashSet::from(["agt_01".to_string(), "agt_02".to_string()]),
    );

    assert_eq!(team.state_machine.current_state(), EntityState::Active);
    assert_eq!(team.member_ids.len(), 2);

    team.add_member("agt_03").unwrap();
    assert_eq!(team.member_ids.len(), 3);

    // Dissolve team
    let freed = team.dissolve("Goal achieved").unwrap();
    assert_eq!(freed.len(), 3);
    assert_eq!(team.state_machine.current_state(), EntityState::Retired);
}

#[test]
fn test_07_authorization_and_approval_gates() {
    let mut caps = HashSet::new();
    caps.insert("tool:file_inspector".to_string());

    // 1. Authorized tool action
    let res = AuthorizationGate::evaluate(
        "agt_01",
        EntityState::Active,
        &ActionType::AccessTool("file_inspector".to_string()),
        None,
        None,
        &caps,
        false,
    );
    assert!(res.is_allowed());

    // 2. Unauthorized tool action
    let res_denied = AuthorizationGate::evaluate(
        "agt_01",
        EntityState::Active,
        &ActionType::AccessTool("code_evaluator".to_string()),
        None,
        None,
        &caps,
        false,
    );
    assert!(!res_denied.is_allowed());

    // 3. Multi-tier approval gate
    let mut approval = ApprovalGate::new();
    // Low risk: auto approved
    assert!(approval.evaluate_execution("agt_01", "read_file", "workspace/data.txt").is_ok());

    // High risk: requires approval ticket
    let high_risk = approval.evaluate_execution("agt_01", "host_tool", "system_tool");
    assert!(high_risk.is_err(), "High risk operation must require approval ticket");
}

#[test]
fn test_08_dynamic_task_manager_full_pipeline() {
    let temp_dir = std::env::temp_dir().join(format!("tara_mgr_test_{}", rand::random::<u32>()));
    fs::create_dir_all(&temp_dir).unwrap();

    let cfg = ManagerConfig {
        manager_id: "mgr_main".to_string(),
        display_name: "Master-Supervisor".to_string(),
        storage_base_dir: temp_dir.clone(),
        is_supervisor: true,
        max_concurrent_sandboxes: 5,
    };

    let mut manager = TaraManager::new(cfg, None);

    // Execute full dynamic task lifecycle:
    // TASK -> CREATE -> NAME -> ASSIGN -> WORK -> VERIFY -> RELEASE -> CLEANUP
    #[cfg(windows)]
    let (cmd, args) = ("cmd.exe", vec!["/C", "echo TARA Dynamic Sandbox Execution Success"]);
    #[cfg(not(windows))]
    let (cmd, args) = ("echo", vec!["TARA Dynamic Sandbox Execution Success"]);

    let result = manager.execute_dynamic_task(
        "task_math_eval_01",
        "mathematics",
        &[],
        None,
        cmd,
        &args,
        None,
    ).unwrap();

    assert_eq!(result.exit_code, 0);
    assert!(result.stdout.contains("TARA Dynamic Sandbox Execution Success"));

    // Verify sandbox resources were released automatically (no lingering permanent count)
    assert_eq!(manager.sandboxes.len(), 0);

    let _ = fs::remove_dir_all(&temp_dir);
}
