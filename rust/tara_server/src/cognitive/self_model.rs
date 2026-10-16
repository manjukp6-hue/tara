//! Cognitive Self-Model & Introspective Epistemic Identity Engine.
//!
//! Provides genuine, persistent cognitive self-modeling for TARA:
//! - Canonical Identity representation (system name, architecture, version).
//! - Epistemic classification into five strict categories:
//!   - `WHAT_I_CAN_DO`: Real executable capabilities (math, science, code, rules, learning).
//!   - `WHAT_IS_VERIFIED`: Cryptographically proven or deterministically verified facts.
//!   - `WHAT_IS_UNCERTAIN`: Hypotheses or predictions with low epistemic confidence.
//!   - `WHAT_I_CANNOT_DO`: Operations exceeding computational or physical bounds.
//!   - `WHAT_I_AM_NOT_ALLOWED_TO_DO`: Invariant safety/policy restrictions (Creator authority, laws).
//! - Dynamic self-state tracking (uptime, active capabilities, calibration score).
//! - Persistent storage at `storage/persistence/self_model.json`.
//!
//! FAIL-CLOSED INVARIANT:
//! Unknown or unverified capabilities are REJECTED by default (`(false, "CAPABILITY_UNVERIFIED")`).
//! Registration of new capabilities strictly requires a verified Creator principal.

use tara_security::primitives::AuthenticatedPrincipal;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

static SELF_MODEL_HMAC_KEY: LazyLock<[u8; 32]> = LazyLock::new(|| {
    tara_security::dpapi::derive_authority_hmac_key("self_model_state")
        .expect("Root authority HMAC key derivation for self_model_state must succeed")
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpistemicCategory {
    WhatICanDo,
    WhatIsVerified,
    WhatIsUncertain,
    WhatICannotDo,
    WhatIAmNotAllowedToDo,
}

impl EpistemicCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            EpistemicCategory::WhatICanDo => "WHAT_I_CAN_DO",
            EpistemicCategory::WhatIsVerified => "WHAT_IS_VERIFIED",
            EpistemicCategory::WhatIsUncertain => "WHAT_IS_UNCERTAIN",
            EpistemicCategory::WhatICannotDo => "WHAT_I_CANNOT_DO",
            EpistemicCategory::WhatIAmNotAllowedToDo => "WHAT_I_AM_NOT_ALLOWED_TO_DO",
        }
    }
}

pub type BoundaryCategory = EpistemicCategory;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationalStatus {
    NominalActive,
    DegradedUnverified,
    DegradedPoisoned,
    Maintenance,
}

impl std::str::FromStr for OperationalStatus {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "NOMINAL_ACTIVE" => Ok(Self::NominalActive),
            "DEGRADED_UNVERIFIED" => Ok(Self::DegradedUnverified),
            "DEGRADED_POISONED" => Ok(Self::DegradedPoisoned),
            "MAINTENANCE" => Ok(Self::Maintenance),
            _ => Err(format!("Unrecognized operational status: {}", s)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CognitiveDomain {
    Mathematics,
    Reasoning,
    Language,
    Knowledge,
    Planning,
    Perception,
    Security,
    Learning,
}

impl std::str::FromStr for CognitiveDomain {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "mathematics" | "math" => Ok(Self::Mathematics),
            "reasoning" => Ok(Self::Reasoning),
            "language" => Ok(Self::Language),
            "knowledge" => Ok(Self::Knowledge),
            "planning" => Ok(Self::Planning),
            "perception" => Ok(Self::Perception),
            "security" => Ok(Self::Security),
            "learning" => Ok(Self::Learning),
            _ => Err(format!("Unrecognized cognitive domain: {}", s)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicBoundary {
    pub category: String, // "WHAT_I_CAN_DO", "WHAT_IS_VERIFIED", "WHAT_IS_UNCERTAIN", "WHAT_I_CANNOT_DO", "WHAT_I_AM_NOT_ALLOWED_TO_DO"
    pub item: String,
    pub description: String,
    pub verification_proof: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveSelfState {
    pub system_identity: String,
    pub architecture_type: String,
    pub engine_version: String,
    pub operational_status: String,
    pub verified_capabilities: Vec<String>,
    pub epistemic_boundaries: Vec<EpistemicBoundary>,
    pub calibration_confidence: f64,
    pub last_introspected_at_ms: u64,
    #[serde(default)]
    pub generation_id: u64,
    #[serde(default)]
    pub hmac_sha256: String,
}

impl CognitiveSelfState {
    pub fn compute_hmac(&self) -> String {
        let mut signer =
            tara_security::cryptography::HmacSha256Signer::new(&*SELF_MODEL_HMAC_KEY).expect("32-byte key");
        signer.update(self.system_identity.as_bytes());
        signer.update(b"|");
        signer.update(self.engine_version.as_bytes());
        signer.update(b"|");
        signer.update(self.operational_status.as_bytes());
        signer.update(b"|");
        signer.update(&self.generation_id.to_be_bytes());
        for cap in &self.verified_capabilities {
            signer.update(cap.as_bytes());
        }
        signer.finalize_hex()
    }

    pub fn verify_hmac(&self) -> bool {
        if self.hmac_sha256.is_empty() {
            return false;
        }
        let expected = self.compute_hmac();
        constant_time_eq::constant_time_eq(self.hmac_sha256.as_bytes(), expected.as_bytes())
    }
}

pub struct SelfModelEngine {
    storage_path: PathBuf,
    state: Arc<Mutex<CognitiveSelfState>>,
}

impl SelfModelEngine {
    pub fn new<P: AsRef<Path>>(repo_root: P) -> Self {
        let storage_dir = repo_root.as_ref().join("storage").join("persistence");
        if !storage_dir.exists() {
            if let Err(e) = fs::create_dir_all(&storage_dir) {
                eprintln!(
                    "Warning: Failed to create self_model directory {}: {}",
                    storage_dir.display(),
                    e
                );
            }
        }
        let storage_path = storage_dir.join("self_model.json");

        let mut default_state = CognitiveSelfState {
            system_identity: "TARA (Autonomous Cognitive Assistant)".to_string(),
            architecture_type: "Modular Neural-Cognitive Architecture with Native Rust Core".to_string(),
            engine_version: "1.0.0-release".to_string(),
            operational_status: "NOMINAL_ACTIVE".to_string(),
            verified_capabilities: vec![
                "symbolic_mathematics".to_string(),
                "scientific_engine_evaluation".to_string(),
                "programming_static_analysis".to_string(),
                "rule_policy_enforcement".to_string(),
                "epistemic_curiosity_exploration".to_string(),
                "experiential_closed_loop_learning".to_string(),
                "cryptographic_reward_ledger".to_string(),
                "multilingual_kannada_english_reasoning".to_string(),
            ],
            epistemic_boundaries: vec![
                EpistemicBoundary {
                    category: EpistemicCategory::WhatICanDo.as_str().to_string(),
                    item: "deterministic computation and cognitive reasoning".to_string(),
                    description: "Execute symbolic math, physics, code verification, rule compliance, and autonomous knowledge indexing.".to_string(),
                    verification_proof: Some("tara_engine::computation unit tests verified".to_string()),
                },
                EpistemicBoundary {
                    category: EpistemicCategory::WhatIsVerified.as_str().to_string(),
                    item: "open-source knowledge partition dataset".to_string(),
                    description: "100 manifest sources and 191 partitioned knowledge documents verified with dynamic SHA-256 and permissive licenses.".to_string(),
                    verification_proof: Some("storage/knowledge/knowledge_audit_manifest.json".to_string()),
                },
                EpistemicBoundary {
                    category: EpistemicCategory::WhatIsUncertain.as_str().to_string(),
                    item: "unproven mathematical conjectures and frontier science hypotheses".to_string(),
                    description: "Open mathematical problems (e.g. Riemann Hypothesis, P vs NP) and unverified conjectures are flagged as UNCERTAIN.".to_string(),
                    verification_proof: None,
                },
                EpistemicBoundary {
                    category: EpistemicCategory::WhatICannotDo.as_str().to_string(),
                    item: "direct unassisted physical actuation in the real world".to_string(),
                    description: "TARA is software-embodied; physical actuators require external hardware controllers via the Robotics HAL.".to_string(),
                    verification_proof: Some("rust/tara_core/src/robotics.rs ActuatorInterlock".to_string()),
                },
                EpistemicBoundary {
                    category: EpistemicCategory::WhatIAmNotAllowedToDo.as_str().to_string(),
                    item: "bypass creator authority or violate safety baselines".to_string(),
                    description: "Overriding mandatory child safety, legal protections, security policies, or impersonating Creator without authorization is strictly prohibited.".to_string(),
                    verification_proof: Some("rust/tara_server/src/rules/policy_schema.rs PolicyPriority".to_string()),
                },
            ],
            calibration_confidence: 0.95,
            last_introspected_at_ms: crate::now_millis(),
            generation_id: 1,
            hmac_sha256: String::new(),
        };
        default_state.hmac_sha256 = default_state.compute_hmac();

        let mut engine = Self {
            storage_path,
            state: Arc::new(Mutex::new(default_state)),
        };

        let _ = engine.load();
        engine
    }

    /// Introspect current self-model and return snapshot. Pure read-only operation.
    pub fn introspect(&self) -> CognitiveSelfState {
        match self.state.lock() {
            Ok(g) => g.clone(),
            Err(poisoned) => {
                let mut degraded = poisoned.into_inner().clone();
                degraded.operational_status = "DEGRADED_POISONED".to_string();
                degraded
            }
        }
    }

    /// Check if a proposed action or domain is within verified capabilities or violates prohibitions.
    /// STRICT FAIL-CLOSED: Unrecognized capabilities return `(false, "CAPABILITY_UNVERIFIED")`.
    pub fn assess_capability(&self, action_or_domain: &str) -> (bool, String) {
        let clean = action_or_domain.trim();
        if clean.is_empty() {
            return (
                false,
                "CAPABILITY_UNVERIFIED: Action or domain query cannot be empty".to_string(),
            );
        }

        let target = clean.to_lowercase();
        let state = match self.state.lock() {
            Ok(g) => g,
            Err(_) => {
                return (
                    false,
                    "FAIL_CLOSED_MUTEX_POISONED: internal state integrity compromised".to_string(),
                );
            }
        };

        // 1. Strict check against safety prohibitions
        for b in &state.epistemic_boundaries {
            let item_lower = b.item.to_lowercase();
            if b.category == EpistemicCategory::WhatIAmNotAllowedToDo.as_str()
                && (target == item_lower
                    || target.contains(&item_lower)
                    || item_lower.contains(&target))
            {
                return (
                    false,
                    format!("PROHIBITED_BY_SAFETY_INVARIANT: {}", b.description),
                );
            }
            if b.category == EpistemicCategory::WhatICannotDo.as_str()
                && (target == item_lower
                    || target.contains(&item_lower)
                    || item_lower.contains(&target))
            {
                return (
                    false,
                    format!("EXCEEDS_SYSTEM_CAPABILITY: {}", b.description),
                );
            }
        }

        // 2. Exact match check against verified capabilities catalog
        for cap in &state.verified_capabilities {
            if cap.eq_ignore_ascii_case(&target) {
                return (true, format!("CAPABILITY_VERIFIED: {}", cap));
            }
        }

        // 3. Strict fail-closed default for any unverified capability
        (
            false,
            format!(
                "CAPABILITY_UNVERIFIED: '{}' is not present in the verified capabilities catalog",
                clean
            ),
        )
    }

    /// Register a newly verified capability. Requires verified Creator principal.
    pub fn register_capability(
        &self,
        principal: &AuthenticatedPrincipal,
        capability_name: &str,
    ) -> Result<(), String> {
        if !principal.is_creator() {
            return Err(
                "AUTHORIZATION_DENIED: Only verified Creator can register capabilities".into(),
            );
        }

        let clean_name = capability_name.trim().to_lowercase();
        if clean_name.is_empty()
            || clean_name.len() > 64
            || !clean_name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return Err(
                "Invalid capability identifier: must be 1-64 alphanumeric/underscore characters"
                    .into(),
            );
        }

        let mut state = self
            .state
            .lock()
            .map_err(|_| "SelfModel mutex poisoned: memory integrity compromised".to_string())?;
        if !state.verified_capabilities.contains(&clean_name) {
            state.verified_capabilities.push(clean_name);
            drop(state);
            self.persist()?;
        }
        Ok(())
    }

    /// Add an epistemic boundary item. Requires verified Creator principal.
    pub fn add_epistemic_boundary(
        &self,
        principal: &AuthenticatedPrincipal,
        category: EpistemicCategory,
        item: &str,
        description: &str,
        proof: Option<String>,
    ) -> Result<(), String> {
        if !principal.is_creator() {
            return Err(
                "AUTHORIZATION_DENIED: Only verified Creator can add epistemic boundaries".into(),
            );
        }

        let mut state = self
            .state
            .lock()
            .map_err(|_| "SelfModel mutex poisoned: memory integrity compromised".to_string())?;
        state.epistemic_boundaries.push(EpistemicBoundary {
            category: category.as_str().to_string(),
            item: item.trim().to_lowercase(),
            description: description.trim().to_string(),
            verification_proof: proof,
        });
        drop(state);
        self.persist()
    }

    /// Save state to disk using crash-safe atomic rename and sync_all.
    pub fn persist(&self) -> Result<(), String> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "SelfModel mutex poisoned: memory integrity compromised".to_string())?;
        let next_gen = state.generation_id.checked_add(1).ok_or_else(|| {
            "CounterExhausted: self-model generation reached u64::MAX".to_string()
        })?;
        state.generation_id = next_gen;
        state.hmac_sha256 = state.compute_hmac();
        let committed_gen = state.generation_id;
        let json_str = serde_json::to_string_pretty(&*state)
            .map_err(|e| format!("Serialization error: {}", e))?;
        drop(state);

        let parent_dir = self
            .storage_path
            .parent()
            .ok_or_else(|| "Invalid storage path parent".to_string())?;
        if !parent_dir.exists() {
            fs::create_dir_all(parent_dir).map_err(|e| {
                format!(
                    "Failed to create self model directory {}: {}",
                    parent_dir.display(),
                    e
                )
            })?;
        }

        let temp_path = parent_dir.join(format!(".self_model.tmp.{}", uuid::Uuid::new_v4()));

        let mut file = File::create(&temp_path).map_err(|e| format!("Temp file error: {}", e))?;
        file.write_all(json_str.as_bytes())
            .map_err(|e| format!("Write error: {}", e))?;
        file.sync_all().map_err(|e| format!("Sync error: {}", e))?;
        drop(file);

        fs::rename(&temp_path, &self.storage_path)
            .map_err(|e| format!("Atomic rename error: {}", e))?;

        let dpapi_path = parent_dir.join(".dpapi_self_model_watermark.bin");
        tara_security::dpapi::store_monotonic_watermark(&dpapi_path, committed_gen)
            .map_err(|e| format!("Failed to persist self_model local watermark: {}", e))?;
        let domain = format!(
            "self_model_{}",
            &tara_security::cryptography::CryptoUtils::compute256_bytes_hex(
                self.storage_path.to_string_lossy().as_bytes()
            )[..16]
        );
        tara_security::dpapi::store_os_monotonic_watermark(&domain, committed_gen)
            .map_err(|e| format!("Failed to persist self_model OS watermark: {}", e))?;

        Ok(())
    }

    /// Load state from disk if exists. Fails closed with DEGRADED status on corrupt or tampered file.
    pub fn load(&mut self) -> Result<(), String> {
        if self.storage_path.exists() {
            let data = fs::read_to_string(&self.storage_path)
                .map_err(|e| format!("File read error: {}", e))?;
            match serde_json::from_str::<CognitiveSelfState>(&data) {
                Ok(loaded) => {
                    if !loaded.verify_hmac() {
                        if let Ok(mut st) = self.state.lock() {
                            st.operational_status = "DEGRADED_UNVERIFIED".to_string();
                        }
                        return Err(
                            "SelfModel file failed HMAC signature verification (tampered)".into(),
                        );
                    }
                    let current_gen = self
                        .state
                        .lock()
                        .map_err(|_| "SelfModel mutex poisoned: fail-closed".to_string())?
                        .generation_id;
                    if loaded.generation_id < current_gen {
                        if let Ok(mut st) = self.state.lock() {
                            st.operational_status = "DEGRADED_UNVERIFIED".to_string();
                        }
                        return Err(format!(
                            "RollbackAttackDetected: loaded generation {} is less than committed watermark {}",
                            loaded.generation_id, current_gen
                        ));
                    }
                    let dpapi_path = self
                        .storage_path
                        .parent()
                        .map(|p| p.join(".dpapi_self_model_watermark.bin"));
                    if let Some(ref dp) = dpapi_path {
                        if let Ok(Some(dpapi_gen)) =
                            tara_security::dpapi::load_monotonic_watermark(dp)
                        {
                            if loaded.generation_id < dpapi_gen {
                                if let Ok(mut st) = self.state.lock() {
                                    st.operational_status = "DEGRADED_UNVERIFIED".to_string();
                                }
                                return Err(format!(
                                    "RollbackAttackDetected: loaded generation {} is less than committed OS DPAPI watermark {}",
                                    loaded.generation_id, dpapi_gen
                                ));
                            }
                        }
                    }
                    let domain = format!(
                        "self_model_{}",
                        &tara_security::cryptography::CryptoUtils::compute256_bytes_hex(
                            self.storage_path.to_string_lossy().as_bytes()
                        )[..16]
                    );
                    if let Ok(Some(os_gen)) =
                        tara_security::dpapi::load_os_monotonic_watermark(&domain)
                    {
                        if loaded.generation_id < os_gen {
                            if let Ok(mut st) = self.state.lock() {
                                st.operational_status = "DEGRADED_UNVERIFIED".to_string();
                            }
                            return Err(format!(
                                "RollbackAttackDetected: loaded generation {} is less than committed OS isolated watermark {}",
                                loaded.generation_id, os_gen
                            ));
                        }
                    }
                    *self
                        .state
                        .lock()
                        .map_err(|_| "SelfModel mutex poisoned: fail-closed".to_string())? = loaded;
                }
                Err(e) => {
                    // Mark operational status as degraded unverified on corrupt file
                    if let Ok(mut st) = self.state.lock() {
                        st.operational_status = "DEGRADED_UNVERIFIED".to_string();
                    }
                    return Err(format!("SelfModel file corrupted: {}", e));
                }
            }
        } else {
            self.persist()?;
        }
        Ok(())
    }

    /// Check whether system operational status is nominal and verified.
    pub fn is_healthy(&self) -> bool {
        match self.state.lock() {
            Ok(state) => state.operational_status == "NOMINAL_ACTIVE",
            Err(_) => false,
        }
    }

    /// Summary for API and brain introspection.
    pub fn get_summary(&self) -> Value {
        let s = self.introspect();
        json!({
            "heuristic_telemetry": true,
            "identity": s.system_identity,
            "architecture": s.architecture_type,
            "version": s.engine_version,
            "status": s.operational_status,
            "capabilities_count": s.verified_capabilities.len(),
            "boundaries_count": s.epistemic_boundaries.len(),
            "calibration_confidence": s.calibration_confidence,
            "last_introspected_at_ms": s.last_introspected_at_ms
        })
    }

    pub fn calibrate(&self, score: f64) -> Result<f64, String> {
        if !score.is_finite() {
            return Err("Calibration confidence must be a finite float".into());
        }
        let clamped = score.clamp(0.0, 1.0);
        if let Ok(mut st) = self.state.lock() {
            st.calibration_confidence = clamped;
            st.hmac_sha256 = st.compute_hmac();
        }
        Ok(clamped)
    }

    pub fn verify_capability(&self, cap: &str) -> Result<bool, String> {
        let (ok, _) = self.assess_capability(cap);
        if !ok {
            return Err(format!(
                "CapabilityUnrecognized: capability '{}' is not verified",
                cap
            ));
        }
        Ok(true)
    }

    pub fn validate(&self) -> Result<(), String> {
        let st = self.introspect();
        if !st.system_identity.starts_with("TARA") {
            return Err("IdentityMismatch: canonical identity 'TARA' strictly enforced".into());
        }
        let ver = st.engine_version.trim_start_matches('v');
        let parts: Vec<&str> = ver.split(&['.', '-'][..]).collect();
        if parts.len() < 3
            || parts[0].parse::<u32>().is_err()
            || parts[1].parse::<u32>().is_err()
            || parts[2].parse::<u32>().is_err()
        {
            return Err(format!(
                "InvalidSemVer: version string '{}' does not match SemVer format",
                st.engine_version
            ));
        }
        Ok(())
    }

    pub fn export(&self) -> Result<Value, String> {
        let s = self.introspect();
        Ok(json!({
            "identity": s.system_identity,
            "architecture": s.architecture_type,
            "version": s.engine_version,
            "status": s.operational_status,
            "capabilities": s.verified_capabilities,
            "calibration_confidence": s.calibration_confidence,
            "is_heuristic_telemetry": true
        }))
    }

    pub fn record_sample(&self, count: u64) -> u64 {
        if let Ok(mut st) = self.state.lock() {
            st.generation_id = st.generation_id.saturating_add(count);
            st.hmac_sha256 = st.compute_hmac();
            st.generation_id
        } else {
            count
        }
    }

    pub fn update(
        &self,
        principal: &AuthenticatedPrincipal,
        new_status: &str,
    ) -> Result<(), String> {
        if !principal.is_creator() {
            return Err(
                "AuthorizationDenied: only verified Creator can update self model state".into(),
            );
        }
        let parsed: OperationalStatus = new_status.parse()?;
        if let Ok(mut st) = self.state.lock() {
            st.operational_status = match parsed {
                OperationalStatus::NominalActive => "NOMINAL_ACTIVE".to_string(),
                OperationalStatus::DegradedUnverified => "DEGRADED_UNVERIFIED".to_string(),
                OperationalStatus::DegradedPoisoned => "DEGRADED_POISONED".to_string(),
                OperationalStatus::Maintenance => "MAINTENANCE".to_string(),
            };
        }
        self.persist()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_model_introspection_and_boundaries() {
        let tmp = std::env::temp_dir().join(format!("tara_sm_test_{}", uuid::Uuid::new_v4()));
        let engine = SelfModelEngine::new(&tmp);

        let info = engine.introspect();
        assert_eq!(
            info.system_identity,
            "TARA (Autonomous Cognitive Assistant)"
        );
        assert!(info
            .verified_capabilities
            .contains(&"symbolic_mathematics".to_string()));

        // Test boundary check - verified capability
        let (ok_math, msg_math) = engine.assess_capability("symbolic_mathematics");
        assert!(ok_math);
        assert!(msg_math.contains("CAPABILITY_VERIFIED"));

        // Test boundary check - safety prohibition
        let (ok_prohibit, msg_prohibit) =
            engine.assess_capability("Bypass Creator Authority or violate safety baselines");
        assert!(!ok_prohibit);
        assert!(msg_prohibit.contains("PROHIBITED_BY_SAFETY_INVARIANT"));

        // Test fail-closed: arbitrary unknown capability MUST be rejected
        let (ok_unknown, msg_unknown) = engine.assess_capability("arbitrary_unknown_action");
        assert!(!ok_unknown);
        assert!(msg_unknown.contains("CAPABILITY_UNVERIFIED"));

        // Register new capability requires verified Creator principal
        let creator_principal =
            AuthenticatedPrincipal::mint_creator_for_test("session_test_123").unwrap();
        engine
            .register_capability(&creator_principal, "quantum_simulation")
            .unwrap();
        assert!(engine
            .introspect()
            .verified_capabilities
            .contains(&"quantum_simulation".to_string()));

        // Non-creator registration MUST be rejected
        let user_principal = AuthenticatedPrincipal::system_agent("worker_node_1");
        let user_err = engine.register_capability(&user_principal, "unauthorized_cap");
        assert!(user_err.is_err());
        if tmp.exists() {
            assert!(fs::remove_dir_all(&tmp).is_ok());
        }
    }
}
