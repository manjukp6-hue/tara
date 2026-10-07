//! naming/mod.rs
//!
//! Dynamic Naming Manager for TARA entities.
//! Enforces INTERNAL ID + HUMAN-READABLE NAME separation.
//! Dynamic name generation, domain-context matching, collision prevention, renaming, and retired name tracking.

use crate::runtime::lifecycle::EntityType;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Mapping record for an entity's display identity and internal ID.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityIdentity {
    pub internal_id: String,
    pub display_name: String,
    pub entity_type: EntityType,
    pub domain_context: String,
    pub created_at_ms: u64,
    pub last_renamed_at_ms: Option<u64>,
}

/// Record of an archived or retired name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetiredNameEntry {
    pub internal_id: String,
    pub name: String,
    pub entity_type: EntityType,
    pub retired_at_ms: u64,
    pub reason: String,
}

/// Dynamic Naming Manager coordinates dynamic selection, uniqueness, and resolution.
#[derive(Debug, Clone)]
pub struct DynamicNamingManager {
    inner: Arc<Mutex<NamingStore>>,
}

#[derive(Debug, Default)]
struct NamingStore {
    // internal_id -> EntityIdentity
    entities: HashMap<String, EntityIdentity>,
    // display_name.to_lowercase() -> internal_id (Active names collision prevention)
    active_names: HashMap<String, String>,
    // Historical archive of retired names
    retired_names: Vec<RetiredNameEntry>,
}

impl Default for DynamicNamingManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicNamingManager {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(NamingStore::default())),
        }
    }

    /// Generates a cryptographically strong, unique internal ID for an entity.
    pub fn generate_internal_id(entity_type: &EntityType) -> String {
        let prefix = match entity_type {
            EntityType::Sandbox => "sbx",
            EntityType::Agent => "agt",
            EntityType::Worker => "wrk",
            EntityType::Team => "team",
            EntityType::Manager => "mgr",
            EntityType::Skill => "skl",
            EntityType::Tool => "tool",
            EntityType::Task => "tsk",
            EntityType::Research => "res",
            EntityType::Experiment => "exp",
            EntityType::Project => "prj",
            EntityType::Capability => "cap",
            EntityType::Custom(s) => {
                let cleaned: String = s.chars().filter(|c| c.is_alphanumeric()).take(4).collect();
                if cleaned.is_empty() {
                    "ent"
                } else {
                    return format!(
                        "{}_{:016x}",
                        cleaned.to_lowercase(),
                        rand::thread_rng().next_u64()
                    );
                }
            }
        };

        let mut rng = rand::thread_rng();
        let part1 = rng.next_u64();
        let part2 = rng.next_u64();
        format!(
            "{}_{:08x}{:08x}",
            prefix,
            (part1 & 0xFFFFFFFF) as u32,
            (part2 & 0xFFFFFFFF) as u32
        )
    }

    /// Contextually and dynamically selects a suitable human-readable name for an entity
    /// from domain context, dynamic knowledge sources, and morphological entity descriptors.
    /// Invariant: No fixed or hardcoded name lists exist in production code.
    pub fn choose_name(
        &self,
        entity_type: &EntityType,
        domain_context: &str,
        preferred_prefix: Option<&str>,
    ) -> String {
        let store = self.inner.lock().unwrap();

        // 1. Synthesize clean domain component tokens from context
        let mut domain_tokens: Vec<String> = domain_context
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| {
                let mut c = s.chars();
                match c.next() {
                    None => String::new(),
                    Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                }
            })
            .collect();

        if domain_tokens.is_empty() {
            domain_tokens.push("General".to_string());
        }

        // 2. Derive functional entity role descriptor
        let role_descriptor = match entity_type {
            EntityType::Sandbox => "Sandbox",
            EntityType::Agent => "Agent",
            EntityType::Worker => "Worker",
            EntityType::Team => "Team",
            EntityType::Manager => "Manager",
            EntityType::Skill => "Skill",
            EntityType::Tool => "Tool",
            EntityType::Task => "Task",
            EntityType::Research => "Research",
            EntityType::Experiment => "Experiment",
            EntityType::Project => "Project",
            EntityType::Capability => "Capability",
            EntityType::Custom(ref s) => {
                if s.trim().is_empty() {
                    "Entity"
                } else {
                    s.trim()
                }
            }
        };

        // 3. Query dynamic knowledge base on disk if present
        let mut knowledge_concept: Option<String> = None;
        let knowledge_paths = [
            "storage/knowledge",
            "TARA/KNOWLEDGE",
            "TARA/KNOWLEDGE/entries",
        ];
        for dir in &knowledge_paths {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) == Some("json") {
                        if let Ok(content) = std::fs::read_to_string(&path) {
                            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                                if let Some(topic) = val
                                    .get("topic")
                                    .or_else(|| val.get("title"))
                                    .and_then(|t| t.as_str())
                                {
                                    let topic_lower = topic.to_lowercase();
                                    for dt in &domain_tokens {
                                        if topic_lower.contains(&dt.to_lowercase()) {
                                            let cleaned: String = topic
                                                .chars()
                                                .filter(|c| c.is_alphanumeric() || *c == '-')
                                                .collect();
                                            if !cleaned.is_empty() {
                                                knowledge_concept = Some(cleaned);
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if knowledge_concept.is_some() {
                        break;
                    }
                }
            }
            if knowledge_concept.is_some() {
                break;
            }
        }

        // 4. Construct candidate base name
        let base_core = if let Some(ref kc) = knowledge_concept {
            format!("{}-{}", kc, role_descriptor)
        } else {
            let domain_joined = domain_tokens.join("-");
            format!("{}-{}", domain_joined, role_descriptor)
        };

        let candidate_base = if let Some(pref) = preferred_prefix {
            format!("{}-{}", pref, base_core)
        } else {
            base_core
        };

        // 5. Uniqueness and collision prevention
        let key = candidate_base.to_lowercase();
        if !store.active_names.contains_key(&key) {
            return candidate_base;
        }

        // Sequential collision avoidance
        for counter in 2..10000 {
            let disambiguated = format!("{}-{}", candidate_base, counter);
            let k = disambiguated.to_lowercase();
            if !store.active_names.contains_key(&k) {
                return disambiguated;
            }
        }

        // Salted entropy fallback
        let entropy = rand::thread_rng().next_u64() % 0xFFFF;
        format!("{}-{:04x}", candidate_base, entropy)
    }

    /// Registers a newly created entity with an auto-assigned or specified unique name.
    pub fn register_entity(
        &self,
        entity_type: EntityType,
        domain_context: &str,
        preferred_name: Option<&str>,
    ) -> Result<EntityIdentity, String> {
        let mut store = self.inner.lock().unwrap();
        let internal_id = Self::generate_internal_id(&entity_type);

        let display_name = match preferred_name {
            Some(name) => {
                let key = name.trim().to_lowercase();
                if store.active_names.contains_key(&key) {
                    return Err(format!("Display name '{}' is already in active use", name));
                }
                name.trim().to_string()
            }
            None => {
                drop(store);
                let chosen = self.choose_name(&entity_type, domain_context, None);
                store = self.inner.lock().unwrap();
                chosen
            }
        };

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let identity = EntityIdentity {
            internal_id: internal_id.clone(),
            display_name: display_name.clone(),
            entity_type,
            domain_context: domain_context.to_string(),
            created_at_ms: now_ms,
            last_renamed_at_ms: None,
        };

        store
            .active_names
            .insert(display_name.to_lowercase(), internal_id.clone());
        store.entities.insert(internal_id.clone(), identity.clone());

        Ok(identity)
    }

    /// Renames an entity, releasing the previous display name and updating indexes.
    /// Invariant: Internal ID never changes during rename.
    pub fn rename_entity(
        &self,
        internal_id: &str,
        new_display_name: &str,
    ) -> Result<EntityIdentity, String> {
        let mut store = self.inner.lock().unwrap();
        let new_key = new_display_name.trim().to_lowercase();

        if let Some(existing_id) = store.active_names.get(&new_key) {
            if existing_id != internal_id {
                return Err(format!(
                    "Cannot rename: '{}' is already assigned to another entity",
                    new_display_name
                ));
            }
        }

        let (old_name, entity_type) = {
            let entity = store
                .entities
                .get(internal_id)
                .ok_or_else(|| format!("Entity with ID '{}' not found", internal_id))?;
            (entity.display_name.clone(), entity.entity_type.clone())
        };

        let old_key = old_name.to_lowercase();

        // Release old active name
        store.active_names.remove(&old_key);

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Record retired name entry
        store.retired_names.push(RetiredNameEntry {
            internal_id: internal_id.to_string(),
            name: old_name,
            entity_type,
            retired_at_ms: now_ms,
            reason: format!("Renamed to {}", new_display_name),
        });

        store.active_names.insert(new_key, internal_id.to_string());

        let entity = store.entities.get_mut(internal_id).unwrap();
        entity.display_name = new_display_name.trim().to_string();
        entity.last_renamed_at_ms = Some(now_ms);

        Ok(entity.clone())
    }

    /// Retires an entity name upon entity retirement or termination.
    pub fn retire_entity(&self, internal_id: &str, reason: &str) -> Result<(), String> {
        let mut store = self.inner.lock().unwrap();
        let entity = store
            .entities
            .remove(internal_id)
            .ok_or_else(|| format!("Entity '{}' not found", internal_id))?;

        store
            .active_names
            .remove(&entity.display_name.to_lowercase());

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        store.retired_names.push(RetiredNameEntry {
            internal_id: internal_id.to_string(),
            name: entity.display_name,
            entity_type: entity.entity_type,
            retired_at_ms: now_ms,
            reason: reason.to_string(),
        });

        Ok(())
    }

    /// Looks up display name for a given internal ID.
    pub fn get_display_name(&self, internal_id: &str) -> Option<String> {
        let store = self.inner.lock().unwrap();
        store
            .entities
            .get(internal_id)
            .map(|e| e.display_name.clone())
    }

    /// Looks up internal ID for a given display name.
    pub fn get_internal_id(&self, display_name: &str) -> Option<String> {
        let store = self.inner.lock().unwrap();
        store
            .active_names
            .get(&display_name.trim().to_lowercase())
            .cloned()
    }

    pub fn list_active_entities(&self) -> Vec<EntityIdentity> {
        let store = self.inner.lock().unwrap();
        store.entities.values().cloned().collect()
    }

    pub fn list_retired_names(&self) -> Vec<RetiredNameEntry> {
        let store = self.inner.lock().unwrap();
        store.retired_names.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_internal_id_generation() {
        let id_sbx = DynamicNamingManager::generate_internal_id(&EntityType::Sandbox);
        assert!(id_sbx.starts_with("sbx_"));

        let id_agt = DynamicNamingManager::generate_internal_id(&EntityType::Agent);
        assert!(id_agt.starts_with("agt_"));

        let id_wrk = DynamicNamingManager::generate_internal_id(&EntityType::Worker);
        assert!(id_wrk.starts_with("wrk_"));
    }

    #[test]
    fn test_register_and_collision_prevention() {
        let mgr = DynamicNamingManager::new();

        let e1 = mgr
            .register_entity(EntityType::Agent, "NLP", Some("BERT-Assistant"))
            .unwrap();
        assert_eq!(e1.display_name, "BERT-Assistant");

        // Duplicate preferred name must be rejected
        let dup_res = mgr.register_entity(EntityType::Agent, "NLP", Some("BERT-Assistant"));
        assert!(dup_res.is_err());
        assert!(dup_res.unwrap_err().contains("already in active use"));

        // Auto-chosen name handles collisions automatically
        let chosen1 = mgr.choose_name(&EntityType::Worker, "Compiler", None);
        let e2 = mgr
            .register_entity(EntityType::Worker, "Compiler", Some(&chosen1))
            .unwrap();

        let chosen2 = mgr.choose_name(&EntityType::Worker, "Compiler", None);
        assert_ne!(e2.display_name, chosen2);
    }

    #[test]
    fn test_rename_and_retire() {
        let mgr = DynamicNamingManager::new();

        let e = mgr
            .register_entity(EntityType::Agent, "Vision", Some("VisionBuddy"))
            .unwrap();
        assert_eq!(mgr.get_display_name(&e.internal_id).unwrap(), "VisionBuddy");

        // Rename
        assert!(mgr.rename_entity(&e.internal_id, "VisionHero").is_ok());
        assert_eq!(mgr.get_display_name(&e.internal_id).unwrap(), "VisionHero");
        assert_eq!(mgr.list_retired_names().len(), 1);

        // Retire
        assert!(mgr.retire_entity(&e.internal_id, "Replaced by v2").is_ok());
        assert!(mgr.get_display_name(&e.internal_id).is_none());
        assert_eq!(mgr.list_retired_names().len(), 2);
    }
}

