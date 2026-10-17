//! Operation Broker & Enforcement Gateway: The Central Protected Gateway for Sensitive Security Operations.
//!
//! Rule compliance:
//! - All privileged operations across TARA must pass through this single authoritative boundary.
//! - Checks global lockdown, verifies authority generation, validates active session non-revocation,
//!   evaluates continuous risk (UEBA / ID-10), enforces central path containment for filesystem/models/tools,
//!   enforces dynamic policy deny-by-default, records pre-execution audit intent, executes operation,
//!   commits post-execution audit completion, and appends to Merkle transparency ledger.
//! - If security persistence, auditing, path security, or risk checks fail, the operation FAILS CLOSED.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

use crate::hardware_root::AuthorityManager;
use crate::path_security::{canonical_containment, validate_tool_command_path};
use crate::primitives::{
    AssuranceLevel, SecurityAction, SecurityContext, SecurityResource, SecurityState,
    VerifiedPrincipal,
};
use crate::transparency_log::{AuthenticatedAuditLogger, MerkleTree};
use crate::zero_trust::{ContinuousRiskEngine, PolicyEngine, RiskVerdict, SessionManager};

/// Supported model lifecycle actions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelAction {
    Load,
    Compile,
    Verify,
    Export,
}

/// The 6 Canonical Protected Operation Categories.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProtectedCategory {
    /// Filesystem access: reads, writes, mutations within a permitted root.
    Files {
        path: PathBuf,
        permitted_root: PathBuf,
        is_write: bool,
    },
    /// Model operations: loading, compilation, verification within models directory.
    Models {
        model_path: PathBuf,
        models_dir: PathBuf,
        action: ModelAction,
    },
    /// Tool / Script execution: sandboxed execution with authorized directory roots.
    Tools {
        command_or_script: PathBuf,
        allowed_dirs: Vec<PathBuf>,
    },
    /// Network operations: inbound or outbound connections to endpoints.
    Network {
        endpoint: String,
        port: u16,
        is_outbound: bool,
    },
    /// Autonomous Goal operations: progression, advancement, task execution.
    AutonomousGoals {
        goal_id: String,
        target_subsystem: String,
    },
    /// Administrative Actions: key rotation, policy changes, quarantine management.
    AdministrativeActions {
        action_type: String,
        requires_quorum: bool,
    },
}

/// Protected Operation Request submitted to the Enforcement Gateway.
#[derive(Debug, Clone)]
pub struct EnforcementRequest {
    pub principal: VerifiedPrincipal,
    pub session_id: Option<String>,
    pub action: SecurityAction,
    pub resource: SecurityResource,
    pub context: SecurityContext,
    pub category: ProtectedCategory,
}

/// Legacy / ergonomic Operation Request submitted to the broker.
#[derive(Debug, Clone)]
pub struct OperationRequest {
    pub principal: VerifiedPrincipal,
    pub action: SecurityAction,
    pub resource: SecurityResource,
    pub context: SecurityContext,
    pub session_id: Option<String>,
}

impl OperationRequest {
    pub fn new(
        principal: VerifiedPrincipal,
        action: SecurityAction,
        resource: SecurityResource,
        context: SecurityContext,
    ) -> Self {
        Self {
            principal,
            action,
            resource,
            context,
            session_id: None,
        }
    }

    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }
}

/// Central Enforcement Gateway coordinating dynamic authorization, global lockdown,
/// central path security, continuous risk evaluation, and tamper-evident auditing.
pub struct EnforcementGateway {
    authority: Arc<AuthorityManager>,
    session_manager: Arc<SessionManager>,
    policy_engine: Arc<PolicyEngine>,
    risk_engine: Arc<ContinuousRiskEngine>,
    audit_logger: Arc<AuthenticatedAuditLogger>,
    merkle_tree: Arc<MerkleTree>,
}

impl EnforcementGateway {
    pub fn new(
        authority: Arc<AuthorityManager>,
        session_manager: Arc<SessionManager>,
        policy_engine: Arc<PolicyEngine>,
        risk_engine: Arc<ContinuousRiskEngine>,
        audit_logger: Arc<AuthenticatedAuditLogger>,
        merkle_tree: Arc<MerkleTree>,
    ) -> Self {
        Self {
            authority,
            session_manager,
            policy_engine,
            risk_engine,
            audit_logger,
            merkle_tree,
        }
    }

    /// Creates a standalone EnforcementGateway with genuine native default components.
    pub fn new_standalone() -> Self {
        let authority = Arc::new(AuthorityManager::new());
        let session_manager = Arc::new(SessionManager::new());
        let policy_engine = Arc::new(PolicyEngine::new());
        let risk_engine = Arc::new(ContinuousRiskEngine::new());
        let audit_key = crate::dpapi::derive_authority_hmac_key("standalone_enforcement_gateway")
            .expect("FAIL CLOSED: Mandatory durable audit HMAC key must be derived from protected DPAPI storage");
        let audit_logger = Arc::new(AuthenticatedAuditLogger::new(
            audit_key,
            Box::new(crate::transparency_log::MemoryAuditSink::new()),
        ));
        let merkle_tree = Arc::new(MerkleTree::new());
        Self::new(
            authority,
            session_manager,
            policy_engine,
            risk_engine,
            audit_logger,
            merkle_tree,
        )
    }

    pub fn authority(&self) -> &Arc<AuthorityManager> {
        &self.authority
    }

    pub fn session_manager(&self) -> &Arc<SessionManager> {
        &self.session_manager
    }

    pub fn policy_engine(&self) -> &Arc<PolicyEngine> {
        &self.policy_engine
    }

    pub fn risk_engine(&self) -> &Arc<ContinuousRiskEngine> {
        &self.risk_engine
    }

    pub fn audit_logger(&self) -> &Arc<AuthenticatedAuditLogger> {
        &self.audit_logger
    }

    pub fn merkle_tree(&self) -> &Arc<MerkleTree> {
        &self.merkle_tree
    }

    /// Evaluates and enforces an OperationRequest through the full 8-stage gateway pipeline.
    pub fn evaluate_and_enforce<T, F>(
        &self,
        request: OperationRequest,
        operation: F,
    ) -> Result<T, String>
    where
        F: FnOnce(&VerifiedPrincipal, &SecurityResource) -> Result<T, String>,
    {
        let category = match &request.resource {
            SecurityResource::FilesystemPath(p) => {
                let permitted_root = request
                    .context
                    .attributes
                    .get("permitted_root")
                    .map(PathBuf::from)
                    .or_else(|| std::env::current_dir().ok())
                    .unwrap_or_else(|| PathBuf::from("."));
                let verb = request.action.verb().to_lowercase();
                let is_write = matches!(
                    verb.as_str(),
                    "write"
                        | "create"
                        | "delete"
                        | "update"
                        | "modify"
                        | "append"
                        | "truncate"
                        | "remove"
                        | "put"
                        | "post"
                        | "patch"
                ) || verb.contains("write")
                    || verb.contains("delete")
                    || verb.contains("create");
                ProtectedCategory::Files {
                    path: p.clone(),
                    permitted_root,
                    is_write,
                }
            }
            SecurityResource::NamedEntity { namespace, id } if namespace == "route" => {
                ProtectedCategory::AdministrativeActions {
                    action_type: format!("http_route:{}:{}", request.action.verb(), id),
                    requires_quorum: false,
                }
            }
            _ => ProtectedCategory::AdministrativeActions {
                action_type: request.action.canonical_name(),
                requires_quorum: false,
            },
        };

        let enf_req = EnforcementRequest {
            principal: request.principal,
            session_id: request.session_id,
            action: request.action,
            resource: request.resource,
            context: request.context,
            category,
        };

        self.enforce_protected_operation(enf_req, operation)
    }

    /// Evaluates an OperationRequest through the full 8-stage gateway verification without executing an inner closure.
    pub fn evaluate_request(&self, request: &OperationRequest) -> Result<(), String> {
        let req_clone = OperationRequest {
            principal: request.principal.clone(),
            action: request.action.clone(),
            resource: request.resource.clone(),
            context: request.context.clone(),
            session_id: request.session_id.clone(),
        };
        self.evaluate_and_enforce(req_clone, |_principal, _resource| Ok(()))
    }

    /// Enforce a protected operation through the unified 8-stage pipeline.
    pub fn enforce_protected_operation<T, F>(
        &self,
        request: EnforcementRequest,
        operation: F,
    ) -> Result<T, String>
    where
        F: FnOnce(&VerifiedPrincipal, &SecurityResource) -> Result<T, String>,
    {
        let epoch_secs = request.context.request_epoch_secs;

        // Stage 1: Global Lockdown & Authority Initialization Check
        if self.authority.is_locked_down() {
            let _ = self.audit_logger.log_event(
                request.principal.id().clone(),
                request.action.clone(),
                request.resource.clone(),
                "DENIED_SYSTEM_LOCKED_DOWN",
                request.principal.authority_generation(),
                epoch_secs,
            );
            return Err("OperationDenied: System is in strict security lockdown".into());
        }

        let current_gen = self.authority.current_generation()?;

        // Verify principal generation alignment
        if request.principal.authority_generation() != current_gen {
            let _ = self.audit_logger.log_event(
                request.principal.id().clone(),
                request.action.clone(),
                request.resource.clone(),
                &format!(
                    "DENIED_AUTHORITY_GENERATION_MISMATCH: principal gen {}, authority gen {}",
                    request.principal.authority_generation().get(),
                    current_gen.get()
                ),
                request.principal.authority_generation(),
                epoch_secs,
            );
            return Err(format!(
                "AuthorityGenerationMismatch: Principal generation {} is stale (current {})",
                request.principal.authority_generation().get(),
                current_gen.get()
            ));
        }

        let current_state = SecurityState::Operational {
            generation: current_gen,
        };

        // Stage 2: Dynamic Principal & Device Revocation Check, and Session Validity
        if self
            .session_manager
            .is_principal_revoked(request.principal.id())
        {
            self.risk_engine.record_failure(request.principal.id());
            let _ = self.audit_logger.log_event(
                request.principal.id().clone(),
                request.action.clone(),
                request.resource.clone(),
                "DENIED_PRINCIPAL_REVOKED",
                request.principal.authority_generation(),
                epoch_secs,
            );
            return Err("PrincipalRevoked: Requesting principal has been revoked".into());
        }

        if let Some(bound_dev) = request.principal.bound_device() {
            if self.session_manager.is_device_revoked(bound_dev) {
                self.risk_engine.record_failure(request.principal.id());
                let _ = self.audit_logger.log_event(
                    request.principal.id().clone(),
                    request.action.clone(),
                    request.resource.clone(),
                    "DENIED_DEVICE_REVOKED",
                    request.principal.authority_generation(),
                    epoch_secs,
                );
                return Err("DeviceRevoked: Bound device has been revoked".into());
            }
        }

        if let Some(ref sess_id) = request.session_id {
            match self
                .session_manager
                .validate_session(sess_id, current_gen, epoch_secs)
            {
                Ok(session_rec) => {
                    if session_rec.principal.id() != request.principal.id() {
                        self.risk_engine.record_failure(request.principal.id());
                        let _ = self.audit_logger.log_event(
                            request.principal.id().clone(),
                            request.action.clone(),
                            request.resource.clone(),
                            "DENIED_PRINCIPAL_SESSION_MISMATCH",
                            request.principal.authority_generation(),
                            epoch_secs,
                        );
                        return Err("PrincipalSessionMismatch: Session does not belong to requesting principal".into());
                    }
                }
                Err(err) => {
                    self.risk_engine.record_failure(request.principal.id());
                    let _ = self.audit_logger.log_event(
                        request.principal.id().clone(),
                        request.action.clone(),
                        request.resource.clone(),
                        &format!("DENIED_INVALID_OR_REVOKED_SESSION: {}", err),
                        request.principal.authority_generation(),
                        epoch_secs,
                    );
                    return Err(format!("SessionVerificationFailed: {}", err));
                }
            }
        }

        // Stage 3: Central Path Security Validation
        match &request.category {
            ProtectedCategory::Files {
                path,
                permitted_root,
                ..
            } => {
                if let Err(e) = canonical_containment(permitted_root, path) {
                    self.risk_engine.record_failure(request.principal.id());
                    let _ = self.audit_logger.log_event(
                        request.principal.id().clone(),
                        request.action.clone(),
                        request.resource.clone(),
                        &format!("DENIED_PATH_SECURITY_VIOLATION: {}", e),
                        request.principal.authority_generation(),
                        epoch_secs,
                    );
                    return Err(format!("PathSecurityViolation: {}", e));
                }
            }
            ProtectedCategory::Models {
                model_path,
                models_dir,
                ..
            } => {
                if let Err(e) = canonical_containment(models_dir, model_path) {
                    self.risk_engine.record_failure(request.principal.id());
                    let _ = self.audit_logger.log_event(
                        request.principal.id().clone(),
                        request.action.clone(),
                        request.resource.clone(),
                        &format!("DENIED_MODEL_PATH_VIOLATION: {}", e),
                        request.principal.authority_generation(),
                        epoch_secs,
                    );
                    return Err(format!("ModelPathSecurityViolation: {}", e));
                }
            }
            ProtectedCategory::Tools {
                command_or_script,
                allowed_dirs,
            } => {
                if let Err(e) = validate_tool_command_path(command_or_script, allowed_dirs) {
                    self.risk_engine.record_failure(request.principal.id());
                    let _ = self.audit_logger.log_event(
                        request.principal.id().clone(),
                        request.action.clone(),
                        request.resource.clone(),
                        &format!("DENIED_TOOL_PATH_VIOLATION: {}", e),
                        request.principal.authority_generation(),
                        epoch_secs,
                    );
                    return Err(format!("ToolPathSecurityViolation: {}", e));
                }
            }
            ProtectedCategory::Network { endpoint, port, .. } => {
                if endpoint.trim().is_empty() || endpoint.contains('\0') || *port == 0 {
                    self.risk_engine.record_failure(request.principal.id());
                    let _ = self.audit_logger.log_event(
                        request.principal.id().clone(),
                        request.action.clone(),
                        request.resource.clone(),
                        "DENIED_INVALID_NETWORK_ENDPOINT",
                        request.principal.authority_generation(),
                        epoch_secs,
                    );
                    return Err("InvalidNetworkEndpoint: Endpoint format or port invalid".into());
                }
            }
            ProtectedCategory::AutonomousGoals {
                goal_id,
                target_subsystem,
            } => {
                if goal_id.trim().is_empty() || target_subsystem.trim().is_empty() {
                    self.risk_engine.record_failure(request.principal.id());
                    return Err("InvalidAutonomousGoalRequest: Goal ID and target subsystem must not be empty".into());
                }
            }
            ProtectedCategory::AdministrativeActions {
                action_type,
                requires_quorum,
            } => {
                if action_type.trim().is_empty() {
                    return Err("InvalidAdministrativeAction: Action type cannot be empty".into());
                }
                if *requires_quorum
                    && request.principal.assurance_level().score()
                        < AssuranceLevel::MULTI_FACTOR_AUTHENTICATED.score()
                {
                    self.risk_engine.record_failure(request.principal.id());
                    let _ = self.audit_logger.log_event(
                        request.principal.id().clone(),
                        request.action.clone(),
                        request.resource.clone(),
                        "DENIED_QUORUM_ASSURANCE_REQUIRED",
                        request.principal.authority_generation(),
                        epoch_secs,
                    );
                    return Err("InsufficientAssurance: Administrative action requires multi-factor quorum assurance".into());
                }
            }
        }

        // Stage 4: Continuous Risk Engine Evaluation (ID-10)
        let risk_verdict = self.risk_engine.evaluate_risk(
            request.principal.id(),
            &request.action,
            &request.context,
        )?;

        match risk_verdict {
            RiskVerdict::TriggerLockdown { risk_score, reason } => {
                self.authority.trigger_lockdown(&reason, epoch_secs)?;
                let _ = self.audit_logger.log_event(
                    request.principal.id().clone(),
                    request.action.clone(),
                    request.resource.clone(),
                    &format!(
                        "CRITICAL_RISK_LOCKDOWN_ENGAGED (score {}): {}",
                        risk_score, reason
                    ),
                    request.principal.authority_generation(),
                    epoch_secs,
                );
                return Err(format!("CriticalRiskLockdown: {}", reason));
            }
            RiskVerdict::RequireCreatorStepUp { risk_score, reason } => {
                if request.principal.assurance_level().score()
                    < AssuranceLevel::CREATOR_ROOT.score()
                {
                    self.risk_engine.record_failure(request.principal.id());
                    let _ = self.audit_logger.log_event(
                        request.principal.id().clone(),
                        request.action.clone(),
                        request.resource.clone(),
                        &format!("DENIED_STEP_UP_REQUIRED (score {}): {}", risk_score, reason),
                        request.principal.authority_generation(),
                        epoch_secs,
                    );
                    return Err(format!(
                        "StepUpAuthenticationRequired: Operation demands Creator Root assurance (score: {})",
                        risk_score
                    ));
                }
            }
            RiskVerdict::RequireHardwareStepUp { risk_score, reason } => {
                if request.principal.assurance_level().score()
                    < AssuranceLevel::HARDWARE_ATTESTED.score()
                {
                    self.risk_engine.record_failure(request.principal.id());
                    let _ = self.audit_logger.log_event(
                        request.principal.id().clone(),
                        request.action.clone(),
                        request.resource.clone(),
                        &format!("DENIED_STEP_UP_REQUIRED (score {}): {}", risk_score, reason),
                        request.principal.authority_generation(),
                        epoch_secs,
                    );
                    return Err(format!(
                        "StepUpAuthenticationRequired: Operation demands Hardware Attestation assurance (score: {})",
                        risk_score
                    ));
                }
            }
            RiskVerdict::Accept { .. } => {}
        }

        // Stage 5: Dynamic Policy Engine Authorization (Deny-by-Default)
        if let Err(denial) = self.policy_engine.authorize(
            &current_state,
            &request.principal,
            &request.action,
            &request.resource,
            &request.context,
        ) {
            let denial_str = denial.to_string();
            self.risk_engine.record_failure(request.principal.id());
            let _ = self.audit_logger.log_event(
                request.principal.id().clone(),
                request.action.clone(),
                request.resource.clone(),
                &format!("DENIED_{}", denial_str),
                request.principal.authority_generation(),
                epoch_secs,
            );
            return Err(format!("AuthorizationDenied: {}", denial_str));
        }

        // Stage 6: Pre-Execution Intent Audit Logging (Fail-Closed)
        self.audit_logger.log_event(
            request.principal.id().clone(),
            request.action.clone(),
            request.resource.clone(),
            "INTENT_START",
            request.principal.authority_generation(),
            epoch_secs,
        )?;

        // Stage 7: Operation Closure Execution
        let op_result = operation(&request.principal, &request.resource);

        // Stage 8: Post-Execution Completion Audit Logging & Merkle Transparency Append
        match op_result {
            Ok(output) => {
                let audit_rec = self.audit_logger.log_event(
                    request.principal.id().clone(),
                    request.action.clone(),
                    request.resource.clone(),
                    "COMPLETION_SUCCESS",
                    request.principal.authority_generation(),
                    epoch_secs,
                )?;

                let leaf_hash = MerkleTree::hash_leaf(audit_rec.record_hmac_signature.as_bytes());
                self.merkle_tree.append_leaf(leaf_hash)?;

                Ok(output)
            }
            Err(err) => {
                self.risk_engine.record_failure(request.principal.id());
                let _ = self.audit_logger.log_event(
                    request.principal.id().clone(),
                    request.action.clone(),
                    request.resource.clone(),
                    &format!("OPERATION_FAILED: {}", err),
                    request.principal.authority_generation(),
                    epoch_secs,
                );
                Err(err)
            }
        }
    }
}

/// Unified Operation Broker coordinating authorization, execution, and audit.
pub struct OperationBroker {
    gateway: Arc<EnforcementGateway>,
}

impl OperationBroker {
    pub fn new(
        authority: Arc<AuthorityManager>,
        policy_engine: Arc<PolicyEngine>,
        risk_engine: Arc<ContinuousRiskEngine>,
        audit_logger: Arc<AuthenticatedAuditLogger>,
        merkle_tree: Arc<MerkleTree>,
    ) -> Self {
        let session_manager = Arc::new(SessionManager::new());
        let gateway = Arc::new(EnforcementGateway::new(
            authority,
            session_manager,
            policy_engine,
            risk_engine,
            audit_logger,
            merkle_tree,
        ));
        Self { gateway }
    }

    pub fn with_session_manager(
        authority: Arc<AuthorityManager>,
        session_manager: Arc<SessionManager>,
        policy_engine: Arc<PolicyEngine>,
        risk_engine: Arc<ContinuousRiskEngine>,
        audit_logger: Arc<AuthenticatedAuditLogger>,
        merkle_tree: Arc<MerkleTree>,
    ) -> Self {
        let gateway = Arc::new(EnforcementGateway::new(
            authority,
            session_manager,
            policy_engine,
            risk_engine,
            audit_logger,
            merkle_tree,
        ));
        Self { gateway }
    }

    pub fn gateway(&self) -> &Arc<EnforcementGateway> {
        &self.gateway
    }

    /// Execute an operation through the broker with automatic category mapping.
    pub fn execute_operation<T, F>(
        &self,
        request: OperationRequest,
        operation: F,
    ) -> Result<T, String>
    where
        F: FnOnce(&VerifiedPrincipal, &SecurityResource) -> Result<T, String>,
    {
        // Derive appropriate category from resource
        let category = match &request.resource {
            SecurityResource::FilesystemPath(p) => {
                let permitted_root = request
                    .context
                    .attributes
                    .get("permitted_root")
                    .map(PathBuf::from)
                    .or_else(|| std::env::current_dir().ok())
                    .unwrap_or_else(|| PathBuf::from("."));
                let verb = request.action.verb().to_lowercase();
                let is_write = matches!(
                    verb.as_str(),
                    "write"
                        | "create"
                        | "delete"
                        | "update"
                        | "modify"
                        | "append"
                        | "truncate"
                        | "remove"
                        | "put"
                        | "post"
                        | "patch"
                ) || verb.contains("write")
                    || verb.contains("delete")
                    || verb.contains("create");
                ProtectedCategory::Files {
                    path: p.clone(),
                    permitted_root,
                    is_write,
                }
            }
            _ => ProtectedCategory::AdministrativeActions {
                action_type: request.action.canonical_name(),
                requires_quorum: false,
            },
        };

        let enf_req = EnforcementRequest {
            principal: request.principal,
            session_id: request.session_id,
            action: request.action,
            resource: request.resource,
            context: request.context,
            category,
        };

        self.gateway.enforce_protected_operation(enf_req, operation)
    }

    /// Execute directly with an explicit EnforcementRequest.
    pub fn enforce_protected_operation<T, F>(
        &self,
        request: EnforcementRequest,
        operation: F,
    ) -> Result<T, String>
    where
        F: FnOnce(&VerifiedPrincipal, &SecurityResource) -> Result<T, String>,
    {
        self.gateway.enforce_protected_operation(request, operation)
    }
}
