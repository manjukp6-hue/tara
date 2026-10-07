//! Security subsystem: jailbreak defense, quarantine management, and license provenance.

pub mod dependency_verifier;
pub mod gated_goals;
pub mod jailbreak;
pub mod provenance;
pub mod quarantine;
pub mod release_manifest;
pub mod source_protector;
pub mod tamper_detector;
pub mod token_scanner;

pub use dependency_verifier::DependencyVerifier;
pub use gated_goals::{
    ActiveTaskView, FailureRule, GatedGoalSystem, GatedTask, GoalHierarchySpec, MilestoneSpec,
    RetryRule, StageSpec, TaskSpec, TaskStatus, VerificationResult,
};
pub use jailbreak::{AttackVector, HomoglyphNormalizer, JailbreakDetector, JailbreakVerdict};
pub use provenance::{
    CopyleftConflictReport, LicenseFamily, LicenseProvenanceEngine, ProvenanceRecord,
    SpdxLicenseInfo,
};
pub use quarantine::{
    ContainmentManager, QuarantineEngine, QuarantineRecord, QuarantineStage, QuarantineVerdict,
};
pub use release_manifest::{ReleaseManifest, ReleaseManifestManager};
pub use source_protector::SourceProtector;
pub use tamper_detector::SourceTamperDetector;
pub use token_scanner::TokenScanner;
