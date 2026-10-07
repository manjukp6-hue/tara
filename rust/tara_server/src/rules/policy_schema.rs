//! Policy schemas and data models for compiled TARA rules.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PolicyPriority {
    MandatoryLawSafety = 1,
    TaraSecurity = 2,
    CreatorRule = 3,
    UserRequest = 4,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuleAction {
    Allow,
    Deny,
    RequireCreatorConfirmation,
    EnforceSetting,
    Ambiguous,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuleCategory {
    MandatoryLaw,
    ChildSafety,
    ContentSafety,
    Security,
    CredentialProtection,
    Privacy,
    DataMutation,
    SystemIntegrity,
    LanguagePreference,
    CreatorAuthority,
    CreatorCustom,
    Ambiguous,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleItem {
    pub rule_id: String,
    pub version: u32,
    pub category: String,
    pub meaning: String,
    pub priority: u32,
    pub scope: String,
    pub action: String,
    pub conditions: HashMap<String, serde_json::Value>,
    pub status: String, // "ACTIVE", "AMBIGUOUS", "CONFLICTING", "REJECTED"
    pub original_text: String,
    pub language: String,
    pub is_mandatory: bool,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredCompiledPolicy {
    pub policy_version: u32,
    pub creator_id: String,
    pub display_name: String,
    pub compiled_at: String,
    pub signing_key_version: u32,
    pub rules: Vec<RuleItem>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl StructuredCompiledPolicy {
    pub fn new(creator_id: &str, display_name: &str, version: u32) -> Self {
        Self {
            policy_version: version,
            creator_id: creator_id.to_string(),
            display_name: display_name.to_string(),
            compiled_at: crate::now_iso(),
            signing_key_version: 1,
            rules: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}
