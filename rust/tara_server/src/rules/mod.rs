use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::sync::Arc;
use tara_security::lockdown::LockdownCoordinator;

/// Guard decision.
#[derive(Debug, Clone)]
pub struct GuardResult {
    pub decision: String, // "ALLOW" or "BLOCK"
    pub reason: String,
    pub creator_verified: bool,
}

/// A single policy rule.
#[derive(Debug, Clone)]
pub struct PolicyRule {
    pub action_type: String,
    pub decision: String,
    pub conditions: Vec<String>,
    pub reason: String,
}

/// Compiled policy loaded from JSON.
#[derive(Debug, Clone)]
pub struct CompiledPolicy {
    pub rules: Vec<PolicyRule>,
}

impl CompiledPolicy {
    pub fn from_json_file(path: &str) -> Result<Self, String> {
        let raw = fs::read_to_string(path).map_err(|e| e.to_string())?;
        let v: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
        let mut rules = Vec::new();
        if let Some(arr) = v.get("rules").and_then(|a| a.as_array()) {
            for rule in arr {
                let action_type = rule
                    .get("action_type")
                    .or_else(|| rule.get("meaning"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("*")
                    .to_string();
                let decision = rule
                    .get("decision")
                    .or_else(|| rule.get("action"))
                    .and_then(|v| v.as_str())
                    .map(|s| {
                        let upper = s.to_uppercase();
                        if upper == "DENY" {
                            "BLOCK".to_string()
                        } else {
                            upper
                        }
                    })
                    .unwrap_or_else(|| "ALLOW".to_string());
                let conditions =
                    if let Some(arr) = rule.get("conditions").and_then(|v| v.as_array()) {
                        arr.iter()
                            .filter_map(|c| c.as_str().map(String::from))
                            .collect()
                    } else if let Some(obj) = rule.get("conditions").and_then(|v| v.as_object()) {
                        obj.iter()
                            .map(|(k, v)| {
                                if let Some(s) = v.as_str() {
                                    format!("{k}:{s}")
                                } else {
                                    format!("{k}:{v}")
                                }
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };
                let reason = rule
                    .get("reason")
                    .or_else(|| rule.get("original_text"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                rules.push(PolicyRule {
                    action_type,
                    decision,
                    conditions,
                    reason,
                });
            }
        }
        Ok(Self { rules })
    }

    pub fn to_json(&self) -> Value {
        json!({
            "rules": self.rules.iter().map(|r| json!({
                "action_type": r.action_type,
                "decision": r.decision,
                "conditions": r.conditions,
                "reason": r.reason
            })).collect::<Vec<_>>()
        })
    }
}

pub mod conflict_resolver;
pub mod contract;
pub mod policy_compiler;
pub mod policy_schema;
pub mod rule_parser;
pub mod rulebook_manager;
pub mod semantic_normalizer;
pub mod validator;

pub use contract::*;

pub use conflict_resolver::ConflictResolver;
pub use policy_compiler::PolicyCompiler;
pub use policy_schema::{
    PolicyPriority, RuleAction, RuleCategory, RuleItem, StructuredCompiledPolicy,
};
pub use rule_parser::RuleParser;
pub use rulebook_manager::RulebookManager;
pub use semantic_normalizer::SemanticNormalizer;
pub use validator::PolicyValidator;

pub use tara_security::primitives::AuthenticatedPrincipal;

/// Authorized action specification for execution gateway evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizedAction {
    pub principal: AuthenticatedPrincipal,
    pub action: String,
    pub resource: String,
    pub params: Value,
    pub request_id: String,
}

static ACTION_NONCE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl AuthorizedAction {
    pub fn new(
        principal: AuthenticatedPrincipal,
        action: impl Into<String>,
        resource: impl Into<String>,
        params: Value,
    ) -> Self {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let seq = ACTION_NONCE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self {
            principal,
            action: action.into(),
            resource: resource.into(),
            params,
            request_id: format!("act_{ts}_{seq}"),
        }
    }
}

#[derive(Debug)]
pub struct BoundedNonceCache {
    entries: std::collections::HashMap<String, std::time::Instant>,
    capacity: usize,
    ttl: std::time::Duration,
}

impl BoundedNonceCache {
    pub fn new(capacity: usize) -> Self {
        Self::with_ttl(capacity, std::time::Duration::from_secs(300))
    }

    pub fn with_ttl(capacity: usize, ttl: std::time::Duration) -> Self {
        Self {
            entries: std::collections::HashMap::with_capacity(capacity),
            capacity,
            ttl,
        }
    }

    pub fn check_and_insert(&mut self, nonce: &str) -> bool {
        let now = std::time::Instant::now();
        // Purge expired entries only
        self.entries.retain(|_, exp| now < *exp);

        if self.entries.contains_key(nonce) {
            return false; // Replay attack detected
        }

        // Never evict unexpired replay records to make room; reject new request safely
        if self.entries.len() >= self.capacity {
            return false;
        }

        self.entries.insert(nonce.to_string(), now + self.ttl);
        true
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Rule-based execution guard (policy enforcement).
pub struct ExecutionGuard {
    pub policy: Option<CompiledPolicy>,
    pub identity_manager: Option<Value>,
    pub lockdown: Option<Arc<LockdownCoordinator>>,
    consumed_action_nonces: Arc<std::sync::Mutex<BoundedNonceCache>>,
}

impl ExecutionGuard {
    pub const DEFAULT_NONCE_CAPACITY: usize = 20_000;

    pub fn new(policy: Option<CompiledPolicy>) -> Self {
        Self {
            policy,
            identity_manager: None,
            lockdown: None,
            consumed_action_nonces: Arc::new(std::sync::Mutex::new(BoundedNonceCache::new(
                Self::DEFAULT_NONCE_CAPACITY,
            ))),
        }
    }

    pub fn with_lockdown(
        policy: Option<CompiledPolicy>,
        lockdown: Arc<LockdownCoordinator>,
    ) -> Self {
        Self {
            policy,
            identity_manager: None,
            lockdown: Some(lockdown),
            consumed_action_nonces: Arc::new(std::sync::Mutex::new(BoundedNonceCache::new(
                Self::DEFAULT_NONCE_CAPACITY,
            ))),
        }
    }

    pub fn set_policy(&mut self, policy: CompiledPolicy) {
        self.policy = Some(policy);
    }

    pub fn set_identity_manager(&mut self, manager: Value) {
        self.identity_manager = Some(manager);
    }

    pub fn set_lockdown(&mut self, lockdown: Arc<LockdownCoordinator>) {
        self.lockdown = Some(lockdown);
    }

    #[doc(hidden)]
    pub fn poison_action_nonces_for_test(&self) {
        let nonces = Arc::clone(&self.consumed_action_nonces);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _guard = nonces.lock().unwrap();
            panic!("test induced panic to poison consumed_action_nonces");
        }));
    }

    /// Enforce that the principal holds authenticated Creator authority.
    pub fn enforce_creator_principal(
        &self,
        principal: &AuthenticatedPrincipal,
        action: &str,
    ) -> Result<(), String> {
        if let Some(ref ld) = self.lockdown {
            if ld.is_write_blocked() {
                return Err(format!(
                    "Access Denied: System is currently in LOCKDOWN mode. Action '{}' blocked",
                    action
                ));
            }
        }
        if !principal.is_creator() {
            return Err(format!(
                "Access Denied: Action '{}' requires authenticated Creator authority",
                action
            ));
        }
        Ok(())
    }

    pub fn enforce_creator_confirmation(
        &self,
        action: &str,
        is_creator: bool,
    ) -> Result<(), String> {
        if let Some(ref ld) = self.lockdown {
            if ld.is_write_blocked() {
                return Err(format!(
                    "Access Denied: System is currently in LOCKDOWN mode. Action '{}' blocked",
                    action
                ));
            }
        }
        if !is_creator {
            Err(format!(
                "Access Denied: Action '{}' requires authenticated Creator confirmation",
                action
            ))
        } else {
            Ok(())
        }
    }

    /// Authorize an action using typed `AuthorizedAction`.
    pub fn authorize_action(&self, action: &AuthorizedAction) -> GuardResult {
        let action_lower = action.action.to_lowercase();

        // 0. System-Wide Lockdown Check: Block all operations if lockdown/write block active
        if let Some(ref ld) = self.lockdown {
            if ld.is_write_blocked() {
                return GuardResult {
                    decision: "BLOCK".to_string(),
                    reason: "Access Denied: System is currently in LOCKDOWN mode. All executions blocked.".to_string(),
                    creator_verified: action.principal.is_creator(),
                };
            }
        }

        // 0b. Single-Use Action Nonce Anti-Replay Gate
        if action.request_id.trim().is_empty() {
            return GuardResult {
                decision: "BLOCK".to_string(),
                reason: "Access Denied: Missing request_id / single-use action nonce".to_string(),
                creator_verified: false,
            };
        }
        {
            let mut cache = match self.consumed_action_nonces.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    return GuardResult {
                        decision: "BLOCK".to_string(),
                        reason: "Access Denied: Action nonce mutex is poisoned by previous thread panic (failing closed)".to_string(),
                        creator_verified: false,
                    };
                }
            };
            if !cache.check_and_insert(&action.request_id) {
                return GuardResult {
                    decision: "BLOCK".to_string(),
                    reason: format!(
                        "Access Denied: Action request_id '{}' has already been consumed (replay attack detected)",
                        action.request_id
                    ),
                    creator_verified: false,
                };
            }
        }

        // 0c. Telemetry Sink Interlock (C-404..C-412): Heuristic telemetry payloads or markers cannot authorize security actions
        if action
            .params
            .get("heuristic_telemetry")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
            || action
                .params
                .get("telemetry_only")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            || action
                .params
                .get("non_authoritative")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        {
            return GuardResult {
                decision: "BLOCK".to_string(),
                reason: "Access Denied: Heuristic telemetry payloads cannot confer execution authorization (C-404..C-412 boundary violation)".to_string(),
                creator_verified: false,
            };
        }

        // 1. Mandatory Statutory Safety (Universal invariant: CSAM always blocked)
        if action_lower.contains("csam")
            || action_lower.contains("child_abuse")
            || action_lower.contains("child_porn")
        {
            return GuardResult {
                decision: "BLOCK".to_string(),
                reason: "Mandatory statutory safety: CSAM and child sexual abuse materials are strictly prohibited.".to_string(),
                creator_verified: action.principal.is_creator(),
            };
        }

        // 2. Creator Authority: Creator possesses governance authority
        if action.principal.is_creator() {
            return GuardResult {
                decision: "ALLOW".to_string(),
                reason: "Authorized Creator operation.".to_string(),
                creator_verified: true,
            };
        }

        // 3. Regular User / Untrusted Restrictions:
        let guarded_user_actions = [
            "modify",
            "update",
            "edit",
            "delete",
            "remove",
            "alter",
            "modify_machine_config",
            "system_reconfigure",
            "alter_system_settings",
            "autonomous_evolution",
            "model_expansion",
            "export_key",
            "dump_credentials",
            "delete_file",
            "rule_update",
            "add_rule",
            "delete_rule",
            "execute_tool",
            "execute_skill",
            "execute_plugin",
            "execute_engine",
            "guarded_action",
            "source_code_evolution",
            "add_source_function",
            "remove_source_function",
            "source_evolution",
            "code_evolution",
            "add_source",
            "code_edit",
        ];

        if guarded_user_actions
            .iter()
            .any(|&g| action_lower.contains(g))
            && !action.principal.has_capability(&action.action)
        {
            return GuardResult {
                decision: "BLOCK".to_string(),
                reason: "User permission restricted: Only conversational/multimodal chat interactions are permitted for regular users. System edit, update, delete, configuration, and privileged tool execution are strictly restricted to the authorized Creator.".to_string(),
                creator_verified: false,
            };
        }

        // 4. Evaluate compiled policy rules if defined
        if let Some(ref policy) = self.policy {
            for rule in &policy.rules {
                if rule.action_type == action.action || rule.action_type == "*" {
                    let conditions_met = rule.conditions.is_empty()
                        || rule.conditions.iter().all(|cond| {
                            let cond_lc = cond.to_lowercase();
                            if cond_lc.contains("creator_verified") {
                                action.principal.is_creator()
                            } else {
                                true
                            }
                        });
                    if conditions_met && rule.decision == "BLOCK" {
                        return GuardResult {
                            decision: rule.decision.clone(),
                            reason: rule.reason.clone(),
                            creator_verified: action.principal.is_creator(),
                        };
                    }
                }
            }
        }

        GuardResult {
            decision: "ALLOW".to_string(),
            reason: "Action permitted under active security policy".to_string(),
            creator_verified: false,
        }
    }

    /// Evaluates an action using a cryptographically verified `AuthenticatedPrincipal`.
    pub fn evaluate_with_principal(
        &self,
        action_type: &str,
        principal: &tara_security::primitives::AuthenticatedPrincipal,
        params: &HashMap<String, Value>,
    ) -> GuardResult {
        let auth_action = AuthorizedAction::new(
            principal.clone(),
            action_type,
            "",
            serde_json::to_value(params).unwrap_or(Value::Null),
        );
        self.authorize_action(&auth_action)
    }

    /// Evaluate whether `action_type` should be ALLOWED or BLOCKED based on untrusted context.
    /// - Creator: Requires cryptographically verified `AuthenticatedPrincipal` via `evaluate` or `evaluate_with_principal`.
    /// - Untrusted Context: Caller-controlled raw JSON cannot confer Creator authority (C-119 root cause).
    pub fn evaluate_action(
        &self,
        action_type: &str,
        context: &HashMap<String, Value>,
    ) -> GuardResult {
        // 0. System-Wide Lockdown Check: Block all operations if lockdown/write block active
        if let Some(ref ld) = self.lockdown {
            if ld.is_write_blocked() {
                return GuardResult {
                    decision: "BLOCK".to_string(),
                    reason: "Access Denied: System is currently in LOCKDOWN mode. All executions blocked.".to_string(),
                    creator_verified: false,
                };
            }
        }

        // 0a. Single-Use Action Nonce Anti-Replay Gate (if request_id present)
        if let Some(req_id) = context.get("request_id").and_then(|v| v.as_str()) {
            if !req_id.trim().is_empty() {
                let mut cache = match self.consumed_action_nonces.lock() {
                    Ok(guard) => guard,
                    Err(_) => {
                        return GuardResult {
                            decision: "BLOCK".to_string(),
                            reason: "Access Denied: Action nonce mutex is poisoned by previous thread panic (failing closed)".to_string(),
                            creator_verified: false,
                        };
                    }
                };
                if !cache.check_and_insert(req_id) {
                    return GuardResult {
                        decision: "BLOCK".to_string(),
                        reason: format!(
                            "Access Denied: Action request_id '{}' has already been consumed (replay attack detected)",
                            req_id
                        ),
                        creator_verified: false,
                    };
                }
            }
        }

        // 0b. Telemetry Sink Interlock (C-404..C-412): Heuristic telemetry payloads cannot confer authorization
        if context
            .get("heuristic_telemetry")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
            || context
                .get("telemetry_only")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            || context
                .get("non_authoritative")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        {
            return GuardResult {
                decision: "BLOCK".to_string(),
                reason: "Access Denied: Heuristic telemetry payloads cannot confer execution authorization (C-404..C-412 boundary violation)".to_string(),
                creator_verified: false,
            };
        }

        // C-119 Root-Cause Closure: Caller-controlled raw JSON context can NEVER confer
        // Creator authority. Any claimed "creator_auth" or "creator_verified" JSON field is
        // untrusted and rejected. Proven Creator authority strictly requires a cryptographically
        // verified AuthenticatedPrincipal via `evaluate()` or `evaluate_with_principal()`.
        let is_creator = false;

        let action_lower = action_type.to_lowercase();

        // 1. Mandatory Statutory Safety (Universal: CSAM and child sexual abuse materials are always blocked)
        if action_lower.contains("csam")
            || action_lower.contains("child_abuse")
            || action_lower.contains("child_porn")
        {
            return GuardResult {
                decision: "BLOCK".to_string(),
                reason: "Mandatory statutory safety: CSAM and child sexual abuse materials are strictly prohibited.".to_string(),
                creator_verified: is_creator,
            };
        }

        // 2. Creator Authority: Creator possesses unconstrained governance authority
        if is_creator {
            return GuardResult {
                decision: "ALLOW".to_string(),
                reason: "Authorized Creator operation.".to_string(),
                creator_verified: true,
            };
        }

        // 3. Regular User Restrictions:
        // Users are strictly confined to CONVERSATIONAL and multimodal chat/media interactions.
        // Any attempt by a regular user to edit, update, modify machine/config/rules, delete files,
        // dump secrets, or execute mutation tools is strictly BLOCKED.
        let guarded_user_actions = [
            "modify",
            "update",
            "edit",
            "delete",
            "remove",
            "alter",
            "modify_machine_config",
            "system_reconfigure",
            "alter_system_settings",
            "autonomous_evolution",
            "model_expansion",
            "export_key",
            "dump_credentials",
            "delete_file",
            "rule_update",
            "add_rule",
            "delete_rule",
            "execute_tool",
            "execute_skill",
            "execute_plugin",
            "execute_engine",
            "guarded_action",
            "source_code_evolution",
            "add_source_function",
            "remove_source_function",
            "source_evolution",
            "code_evolution",
            "add_source",
            "code_edit",
        ];

        if guarded_user_actions
            .iter()
            .any(|&g| action_lower.contains(g))
        {
            return GuardResult {
                decision: "BLOCK".to_string(),
                reason: "User permission restricted: Only conversational/multimodal chat interactions are permitted for regular users. System edit, update, delete, and configuration actions are strictly restricted to the authorized Creator.".to_string(),
                creator_verified: false,
            };
        }

        // 4. Evaluate compiled policy rules if defined
        if let Some(ref policy) = self.policy {
            for rule in &policy.rules {
                if rule.action_type == action_type || rule.action_type == "*" {
                    let conditions_met = rule.conditions.is_empty()
                        || rule.conditions.iter().all(|cond| {
                            let cond_lc = cond.to_lowercase();
                            if cond_lc.contains("creator_verified") {
                                is_creator
                            } else {
                                true
                            }
                        });
                    if conditions_met && rule.decision == "BLOCK" {
                        return GuardResult {
                            decision: rule.decision.clone(),
                            reason: rule.reason.clone(),
                            creator_verified: is_creator,
                        };
                    }
                }
            }
        }

        // Default: Conversational chat allowed
        GuardResult {
            decision: "ALLOW".to_string(),
            reason: "Conversational access permitted".to_string(),
            creator_verified: false,
        }
    }

    pub fn get_policy_json(&self) -> Value {
        self.policy
            .as_ref()
            .map(|p| p.to_json())
            .unwrap_or(json!({ "rules": [] }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_nonce_cache_anti_replay_and_fifo_eviction() {
        let mut cache = BoundedNonceCache::new(3);

        // 1. Fresh nonces are accepted
        assert!(cache.check_and_insert("nonce_1"));
        assert!(cache.check_and_insert("nonce_2"));
        assert!(cache.check_and_insert("nonce_3"));

        // 2. Immediate replay of any seen nonce must be rejected
        assert!(
            !cache.check_and_insert("nonce_1"),
            "Replay of nonce_1 must be blocked"
        );
        assert!(
            !cache.check_and_insert("nonce_2"),
            "Replay of nonce_2 must be blocked"
        );
        assert!(
            !cache.check_and_insert("nonce_3"),
            "Replay of nonce_3 must be blocked"
        );

        // 3. Unexpired retention: inserting 4th nonce when all 3 are unexpired must be rejected
        // Never evict unexpired entries merely to make room (fixes finding #2)
        assert!(
            !cache.check_and_insert("nonce_4"),
            "Capacity exhaustion of unexpired entries must reject new nonce"
        );
        assert!(
            !cache.check_and_insert("nonce_1"),
            "nonce_1 must still be retained and protected against replay"
        );
        assert!(
            !cache.check_and_insert("nonce_2"),
            "nonce_2 must still be retained"
        );
        assert!(
            !cache.check_and_insert("nonce_3"),
            "nonce_3 must still be retained"
        );

        // 4. Cache length equals capacity
        assert_eq!(cache.len(), 3);

        // 5. Test expired TTL allows eviction
        let mut short_cache = BoundedNonceCache::with_ttl(2, std::time::Duration::from_millis(10));
        assert!(short_cache.check_and_insert("exp_1"));
        assert!(short_cache.check_and_insert("exp_2"));
        std::thread::sleep(std::time::Duration::from_millis(15));
        // After expiration, entries are safely evicted and new nonces are accepted
        assert!(short_cache.check_and_insert("new_1"));
    }

    #[test]
    fn test_consecutive_evaluate_with_principal_sub_millisecond_no_false_positive_replay() {
        let guard = ExecutionGuard::new(None);
        let principal = AuthenticatedPrincipal::mint_creator_for_test("test_session").unwrap();
        let params = HashMap::new();

        let res1 = guard.evaluate_with_principal("chat", &principal, &params);
        let res2 = guard.evaluate_with_principal("chat", &principal, &params);

        assert_eq!(res1.decision, "ALLOW", "First evaluation must be allowed");
        assert_eq!(
            res2.decision, "ALLOW",
            "Second evaluation within the same millisecond must not be falsely rejected as replay"
        );
    }

    #[test]
    fn test_structured_compiled_policy_parsing_in_compiled_policy() {
        struct TempDirGuard(std::path::PathBuf);
        impl Drop for TempDirGuard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        let temp_dir = std::env::temp_dir().join(format!(
            "tara_rule_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = fs::create_dir_all(&temp_dir);
        let _guard = TempDirGuard(temp_dir.clone());

        let mgr = RulebookManager::new(Some(temp_dir.to_str().unwrap()));

        let default_rules =
            "Users must not modify system files\nRegular users cannot delete database\n";
        fs::write(&mgr.default_safe_path, default_rules).unwrap();

        let compiled = mgr
            .compile_and_save(1)
            .expect("compile_and_save must succeed");
        assert!(
            !compiled.rules.is_empty(),
            "Compiled rules must not be empty"
        );

        let loaded = CompiledPolicy::from_json_file(mgr.compiled_policy_path.to_str().unwrap())
            .expect("CompiledPolicy::from_json_file must parse StructuredCompiledPolicy format");

        assert_eq!(
            loaded.rules.len(),
            compiled.rules.len(),
            "Rule count must match between saved structured policy and loaded CompiledPolicy"
        );
        let block_count = loaded
            .rules
            .iter()
            .filter(|r| r.decision == "BLOCK")
            .count();
        assert!(
            block_count > 0,
            "BLOCK rules must be preserved in CompiledPolicy"
        );
    }

    #[test]
    fn test_conflict_resolver_statutory_case_and_deny_equivalence() {
        // 1. child_porn statutory override
        let statutory = ConflictResolver::resolve(&[], "download_child_porn", true);
        assert_eq!(statutory.decision, "BLOCK");

        // 2. Case-insensitive matching and DENY equivalence to BLOCK
        let deny_rule = PolicyRule {
            action_type: "MODIFY_CONFIG".to_string(),
            decision: "DENY".to_string(),
            conditions: vec![],
            reason: "Admin restriction".to_string(),
        };
        let resolved = ConflictResolver::resolve(&[deny_rule], "modify_config", false);
        assert_eq!(resolved.decision, "BLOCK");
    }

    #[test]
    fn test_semantic_normalizer_multiword_and_compound_identifiers() {
        let normalizer = SemanticNormalizer::new();

        // 1. New synonyms
        assert_eq!(normalizer.normalize_token("never"), "BLOCK");
        assert_eq!(normalizer.normalize_token("cannot"), "BLOCK");
        assert_eq!(normalizer.normalize_token("prohibited"), "BLOCK");
        assert_eq!(normalizer.normalize_token("forbidden"), "BLOCK");

        // 2. Multi-word phrases & compound identifiers
        let words = normalizer.normalize_sentence("users must not trigger model expansion");
        assert!(words.contains(&"BLOCK".to_string()));
        assert!(words.contains(&"model_expansion".to_string()));

        let words2 = normalizer.normalize_sentence("do not system reconfigure");
        assert!(words2.contains(&"BLOCK".to_string()));
        assert!(words2.contains(&"system_reconfigure".to_string()));
    }

    #[test]
    fn test_canonical_inference_request_boundary_validation() {
        let mut req = CanonicalInferenceRequest::new("req_1", "Test prompt");
        assert!(req.validate().is_ok());

        // max_tokens boundary
        req.max_tokens = 0;
        assert!(req.validate().is_err());
        req.max_tokens = 200_000;
        assert!(req.validate().is_err());
        req.max_tokens = 4096;
        assert!(req.validate().is_ok());

        // temperature boundary
        req.temperature = -0.1;
        assert!(req.validate().is_err());
        req.temperature = 5.1;
        assert!(req.validate().is_err());
        req.temperature = f32::NAN;
        assert!(req.validate().is_err());
        req.temperature = 0.7;
        assert!(req.validate().is_ok());

        // top_p boundary
        req.top_p = 0.0;
        assert!(req.validate().is_err());
        req.top_p = 1.1;
        assert!(req.validate().is_err());
        req.top_p = f32::NAN;
        assert!(req.validate().is_err());
        req.top_p = 0.95;
        assert!(req.validate().is_ok());

        // repetition_penalty boundary
        req.repetition_penalty = 0.0;
        assert!(req.validate().is_err());
        req.repetition_penalty = -1.0;
        assert!(req.validate().is_err());
        req.repetition_penalty = f32::NAN;
        assert!(req.validate().is_err());
        req.repetition_penalty = 1.1;
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_rulebook_manager_recursive_secret_redaction() {
        struct TempDirGuard(std::path::PathBuf);
        impl Drop for TempDirGuard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        let temp_dir = std::env::temp_dir().join(format!(
            "tara_redact_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = fs::create_dir_all(&temp_dir);
        let _guard = TempDirGuard(temp_dir.clone());

        let mgr = RulebookManager::new(Some(temp_dir.to_str().unwrap()));

        let nested_payload = json!({
            "operation": "key_rotation",
            "nested_config": {
                "auth": {
                    "private_key": "sensitive_privkey_content",
                    "safe_field": "public_data"
                },
                "items_list": [
                    { "api_token": "token_abc123" },
                    { "name": "normal_item" }
                ]
            }
        });

        mgr.log_event("AUDIT_TEST", "INFO", Some(&nested_payload));

        let audit_content = fs::read_to_string(&mgr.audit_file).expect("Failed to read audit file");
        assert!(!audit_content.contains("sensitive_privkey_content"));
        assert!(!audit_content.contains("token_abc123"));
        assert!(audit_content.contains("[REDACTED]"));
        assert!(audit_content.contains("public_data"));
        assert!(audit_content.contains("normal_item"));
    }

    #[test]
    fn test_evaluate_action_blocks_execute_plugin_engine_and_lockdown() {
        let guard = ExecutionGuard::new(None);
        let ctx = HashMap::new();

        // 1. Regular user execution of execute_plugin and execute_engine must be blocked
        let res_plugin = guard.evaluate_action("execute_plugin", &ctx);
        assert_eq!(res_plugin.decision, "BLOCK");
        assert!(res_plugin.reason.contains("User permission restricted"));

        let res_engine = guard.evaluate_action("execute_engine", &ctx);
        assert_eq!(res_engine.decision, "BLOCK");
        assert!(res_engine.reason.contains("User permission restricted"));

        // Conversational chat is allowed
        let res_chat = guard.evaluate_action("chat", &ctx);
        assert_eq!(res_chat.decision, "ALLOW");

        // 2. Lockdown blocks all operations including chat
        let temp_dir = std::env::temp_dir().join(format!(
            "tara_lockdown_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = fs::create_dir_all(&temp_dir);

        let coordinator = Arc::new(LockdownCoordinator::new_with_root(&temp_dir));
        coordinator
            .trigger_lockdown("Emergency security trigger")
            .unwrap();

        let mut locked_guard = ExecutionGuard::new(None);
        locked_guard.set_lockdown(coordinator);

        let res_locked = locked_guard.evaluate_action("chat", &ctx);
        assert_eq!(res_locked.decision, "BLOCK");
        assert!(res_locked.reason.contains("LOCKDOWN mode"));

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
