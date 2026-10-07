//! Access services: Firebase metadata synchronization and multi-backend storage registry.
//!
//! Security invariants:
//! - Firebase/Cloud sync NEVER stores private keys, passwords, or raw secrets.
//! - TARA operates fully offline if cloud services become unreachable.
//! - StorageRegistry strictly tags TARA_OWNED vs USER_CONTENT to prevent data loss.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageType {
    Internal,
    Secondary,
    ExternalSsd,
    Usb,
    SdCard,
    Nas,
    GoogleDrive,
    CloudStorage,
    Custom,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OwnershipTag {
    TaraOwned,
    Ambiguous,
    UserContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageObject {
    pub uri: String,
    pub name: String,
    pub ownership: OwnershipTag,
    pub size_bytes: u64,
}

pub struct StorageRegistry {
    locations: Mutex<HashMap<String, StorageType>>,
    objects: Mutex<Vec<StorageObject>>,
}

impl Default for StorageRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl StorageRegistry {
    pub fn new() -> Self {
        Self {
            locations: Mutex::new(HashMap::new()),
            objects: Mutex::new(Vec::new()),
        }
    }

    pub fn register_location(&self, location_id: &str, storage_type: StorageType) {
        let mut locs = self.locations.lock().unwrap();
        locs.insert(location_id.to_string(), storage_type);
    }

    pub fn track_object(&self, obj: StorageObject) {
        let mut objs = self.objects.lock().unwrap();
        objs.push(obj);
    }

    pub fn list_tara_owned_objects(&self) -> Vec<StorageObject> {
        let objs = self.objects.lock().unwrap();
        objs.iter()
            .filter(|o| o.ownership == OwnershipTag::TaraOwned)
            .cloned()
            .collect()
    }

    pub fn health_check(&self) -> Value {
        let locs = self.locations.lock().unwrap();
        let objs = self.objects.lock().unwrap();
        json!({
            "status": "HEALTHY",
            "registered_locations_count": locs.len(),
            "tracked_objects_count": objs.len()
        })
    }
}

pub struct FirebaseSyncService {
    is_online: Mutex<bool>,
    offline_queue: Mutex<Vec<Value>>,
    firestore_creators: Mutex<HashMap<String, Value>>,
}

impl FirebaseSyncService {
    pub fn new(online: bool) -> Self {
        Self {
            is_online: Mutex::new(online),
            offline_queue: Mutex::new(Vec::new()),
            firestore_creators: Mutex::new(HashMap::new()),
        }
    }

    pub fn set_online_status(&self, online: bool) {
        let mut on = self.is_online.lock().unwrap();
        *on = online;
    }

    pub fn sync_creator_metadata(
        &self,
        creator_id: &str,
        display_name: &str,
        public_key: &str,
        key_version: u32,
        status: &str,
    ) -> bool {
        let record = json!({
            "creator_id": creator_id,
            "display_name": display_name,
            "public_key": public_key,
            "key_version": key_version,
            "status": status,
            "synced_at": crate::now_iso()
        });

        let on = *self.is_online.lock().unwrap();
        if on {
            let mut store = self.firestore_creators.lock().unwrap();
            store.insert(creator_id.to_string(), record);
            true
        } else {
            let mut queue = self.offline_queue.lock().unwrap();
            queue.push(record);
            false
        }
    }

    pub fn get_offline_queue_len(&self) -> usize {
        self.offline_queue.lock().unwrap().len()
    }
}
