//! TARA-Controlled Security Subsystem.
//!
//! Provides the runtime security layers directly controlled and operated by TARA:
//! 1. `AgentExecutionSandbox` (`agent_sandbox`): Autonomous agent isolation, workspace containment,
//!    and tool execution dispatch.
//! 2. `SourceProtector` (`source_protector`): Write protection on core source code, model weights,
//!    and architecture registries against unauthorized mutation.
//! 3. `QuarantineEngine` (`quarantine`): 8-stage artifact inspection, dynamic hash blocklisting,
//!    and threat containment.
//! 4. `JailbreakDefenseEngine` (`jailbreak_defense`): 14-vector deterministic prompt injection and
//!    jailbreak defense with Unicode homoglyph normalization.
//! 5. `SecretLeakScanner` (`secret_scanner`): Hardcoded credentials, private keys, API tokens,
//!    and password leak scanner.
//! 6. `ModelIntegrityGuard` (`model_integrity`): Model weights verification, HMAC seals,
//!    and anti-rollback version checks.
//! 7. `RuntimeTamperSentinel` (`tamper_detector`): Baseline file integrity scanner, dynamic SHA-256
//!    validation, and automatic lockdown signaling.
//! 8. `ProvenanceLedgerGuard` (`provenance`): SPDX license classification (Rigorous Commercial & Permissive Rights compliance),
//!    copyleft taint prevention, and dynamic provenance ledger.
//! 9. `AutonomousGoalGating` (`gated_goals`): Gated autonomous goal lifecycle with TOCTOU defenses.
//!
//! Rule compliance:
//! - Pure native Rust, zero outside crate dependencies.
//! - Deeply integrated with and enforcing the Dynamic Centralized Security System (`EnforcementGateway`,
//!   `AuthorityManager`, `PathSecurity`, and Merkle transparency ledger).

pub mod agent_sandbox;
pub mod dependency_verifier;
pub mod gated_goals;
pub mod jailbreak_defense;
pub mod model_integrity;
pub mod provenance;
pub mod quarantine;
pub mod release_manifest;
pub mod secret_scanner;
pub mod source_protector;
pub mod tamper_detector;

pub use agent_sandbox::{AgentExecutionOutcome, AgentExecutionSandbox, AgentSandboxPolicy};
pub use dependency_verifier::{DependencyVerificationReport, DependencyVerifier, VerifiedPackage};
pub use gated_goals::{AutonomousGoalGating, GatedGoal, GoalState};
pub use jailbreak_defense::{
    AttackVector, HomoglyphNormalizer, JailbreakDefenseEngine, JailbreakDetector, JailbreakVerdict,
};
pub use model_integrity::{
    IntegrityError, ModelIntegrityEngine, ModelIntegrityGuard, ModelManifestSeal, SealResult,
    VerifyResult,
};
pub use provenance::{
    CopyleftConflictReport, LicenseFamily, LicenseProvenanceEngine, ProvenanceLedgerGuard,
    ProvenanceRecord, SpdxLicenseInfo,
};
pub use quarantine::{
    ContainmentManager, QuarantineEngine, QuarantineRecord, QuarantineStage, QuarantineVerdict,
};
pub use release_manifest::{
    canonical_manifest_payload, compute_file_sha256, ReleaseManifest, ReleaseManifestManager,
};
pub use secret_scanner::{SecretFinding, SecretLeakScanner, TokenScanner};
pub use source_protector::SourceProtector;
pub use tamper_detector::{RuntimeTamperSentinel, SourceTamperDetector};

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::broker::EnforcementGateway;
use crate::primitives::{SecurityContext, VerifiedPrincipal};

/// Unified coordinator for TARA-Controlled Security.
pub struct TARAControlledSecurity {
    gateway: Arc<EnforcementGateway>,
    source_protector: Arc<SourceProtector>,
    quarantine: Arc<QuarantineEngine>,
    jailbreak_defense: Arc<JailbreakDefenseEngine>,
    secret_scanner: Arc<SecretLeakScanner>,
    model_integrity: Arc<ModelIntegrityGuard>,
    tamper_sentinel: Arc<RuntimeTamperSentinel>,
    provenance_guard: Arc<ProvenanceLedgerGuard>,
    goal_gating: Arc<AutonomousGoalGating>,
}

impl TARAControlledSecurity {
    pub fn new(
        gateway: Arc<EnforcementGateway>,
        quarantine_root: PathBuf,
        repo_root: PathBuf,
    ) -> Self {
        let authority = gateway.authority().clone();
        let source_protector = Arc::new(SourceProtector::new());
        let quarantine = Arc::new(QuarantineEngine::new(quarantine_root, gateway.clone()));
        let jailbreak_defense = Arc::new(JailbreakDefenseEngine::new());
        let secret_scanner = Arc::new(SecretLeakScanner::new());
        let model_integrity = Arc::new(ModelIntegrityGuard::new());
        let tamper_sentinel =
            Arc::new(RuntimeTamperSentinel::new(repo_root).with_authority(authority.clone()));
        let provenance_guard = Arc::new(ProvenanceLedgerGuard::new());
        let goal_gating = Arc::new(AutonomousGoalGating::new(authority, gateway.clone()));

        Self {
            gateway,
            source_protector,
            quarantine,
            jailbreak_defense,
            secret_scanner,
            model_integrity,
            tamper_sentinel,
            provenance_guard,
            goal_gating,
        }
    }

    pub fn gateway(&self) -> &Arc<EnforcementGateway> {
        &self.gateway
    }

    pub fn source_protector(&self) -> &Arc<SourceProtector> {
        &self.source_protector
    }

    pub fn quarantine(&self) -> &Arc<QuarantineEngine> {
        &self.quarantine
    }

    pub fn jailbreak_defense(&self) -> &Arc<JailbreakDefenseEngine> {
        &self.jailbreak_defense
    }

    pub fn secret_scanner(&self) -> &Arc<SecretLeakScanner> {
        &self.secret_scanner
    }

    pub fn model_integrity(&self) -> &Arc<ModelIntegrityGuard> {
        &self.model_integrity
    }

    pub fn tamper_sentinel(&self) -> &Arc<RuntimeTamperSentinel> {
        &self.tamper_sentinel
    }

    pub fn provenance_guard(&self) -> &Arc<ProvenanceLedgerGuard> {
        &self.provenance_guard
    }

    pub fn goal_gating(&self) -> &Arc<AutonomousGoalGating> {
        &self.goal_gating
    }

    /// Spawns an isolated sandbox for an autonomous agent with workspace containment.
    pub fn spawn_agent_sandbox(
        &self,
        agent_id: impl Into<String>,
        workspace_root: PathBuf,
        policy: AgentSandboxPolicy,
    ) -> Result<AgentExecutionSandbox, String> {
        AgentExecutionSandbox::new(agent_id, workspace_root, policy, self.gateway.clone())
    }

    /// Asserts that a file mutation on the project tree is authorized and not in a protected core root.
    pub fn assert_file_mutation(
        &self,
        rel_path: &str,
        repo_root: &Path,
        principal: &VerifiedPrincipal,
        context: &SecurityContext,
        session_id: Option<String>,
    ) -> Result<PathBuf, String> {
        self.source_protector.assert_mutation_allowed(
            rel_path,
            repo_root,
            principal,
            context,
            session_id,
            &self.gateway,
        )
    }

    /// Ingests and evaluates an incoming artifact through the 8-stage quarantine pipeline.
    pub fn inspect_artifact(
        &self,
        artifact_name: &str,
        artifact_type: &str,
        content: &[u8],
        epoch_secs: u64,
    ) -> Result<QuarantineRecord, String> {
        self.quarantine
            .evaluate_artifact(artifact_name, artifact_type, content, epoch_secs)
    }

    /// Scans a prompt or tool parameter against 14 adversarial prompt injection vectors.
    pub fn inspect_prompt(&self, input: &str) -> JailbreakVerdict {
        self.jailbreak_defense.inspect_prompt(input)
    }

    /// Scans text/code to ensure zero hardcoded credentials or tokens are present.
    pub fn assert_clean_secrets(&self, content: &str) -> Result<(), String> {
        self.secret_scanner.assert_clean(content)
    }

    /// Asserts that a license is fully compliant with commercial and permissive criteria (commercial use, adaptation, redistribution rights).
    pub fn assert_license_compliant(&self, license_text: &str) -> Result<SpdxLicenseInfo, String> {
        self.provenance_guard
            .assert_ingestion_compliant(license_text)
    }
}
