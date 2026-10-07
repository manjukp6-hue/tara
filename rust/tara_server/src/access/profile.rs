//! Canonical TARA Root Creator Identity Specification:
//! - Permanent Creator ID: ROOT_OPERATOR
//! - Display Name: OPERATOR_ROOT
//! - Key versioning tracks rotation and revocations.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

pub const CANONICAL_CREATOR_ID: &str = "ROOT_OPERATOR";
pub const DEFAULT_DISPLAY_NAME: &str = "OPERATOR_ROOT";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevokedKey {
    pub key_version: u64,
    pub public_key: String,
    pub revoked_at: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatorIdentityRecord {
    pub creator_id: String,
    pub display_name: String,
    pub identity_version: u64,
    pub key_version: u64,
    pub root_public_key: Option<String>,
    pub status: String,
    pub recovery_enabled: bool,
    pub created_at: Option<String>,
    pub last_key_rotation: Option<String>,
    pub revoked_keys: Vec<RevokedKey>,
}

pub struct CreatorIdentity {
    pub record_path: PathBuf,
    pub record: CreatorIdentityRecord,
}

impl CreatorIdentity {
    pub fn new<P: AsRef<Path>>(record_path: Option<P>) -> Self {
        let p = match record_path {
            Some(path) => path.as_ref().to_path_buf(),
            None => PathBuf::from("TARA/ACCESS/operator/operator_record.json"),
        };

        let mut identity = Self {
            record_path: p,
            record: CreatorIdentityRecord {
                creator_id: CANONICAL_CREATOR_ID.to_string(),
                display_name: DEFAULT_DISPLAY_NAME.to_string(),
                identity_version: 1,
                key_version: 1,
                root_public_key: None,
                status: "CREATOR_SETUP_REQUIRED".to_string(),
                recovery_enabled: false,
                created_at: None,
                last_key_rotation: None,
                revoked_keys: Vec::new(),
            },
        };

        if identity.record_path.exists() {
            let _ = identity.load();
        }

        identity
    }

    pub fn load(&mut self) -> Result<(), String> {
        let data = fs::read_to_string(&self.record_path).map_err(|e| e.to_string())?;
        self.record = serde_json::from_str(&data).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn save(&self) -> Result<(), String> {
        if let Some(parent) = self.record_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let data = serde_json::to_string_pretty(&self.record).map_err(|e| e.to_string())?;
        fs::write(&self.record_path, data).map_err(|e| e.to_string())
    }

    pub fn is_initialized(&self) -> bool {
        self.record.root_public_key.is_some() && self.record.status == "active"
    }

    pub fn initialize_root_creator(
        &mut self,
        public_key_hex: &str,
        display_name: Option<&str>,
    ) -> Result<CreatorIdentityRecord, String> {
        if self.is_initialized() {
            return Err("Creator authority is already initialized. Authentication or authorized recovery is required for changes.".into());
        }

        self.record.creator_id = CANONICAL_CREATOR_ID.to_string();
        self.record.display_name = display_name.unwrap_or(DEFAULT_DISPLAY_NAME).to_string();
        self.record.root_public_key = Some(public_key_hex.to_lowercase());
        self.record.identity_version = 1;
        self.record.key_version = 1;
        self.record.status = "active".to_string();
        self.record.recovery_enabled = true;
        self.record.created_at = Some(crate::now_iso());

        self.save()?;
        Ok(self.record.clone())
    }

    pub fn update_display_name(&mut self, new_name: &str) -> Result<(), String> {
        if new_name.trim().is_empty() {
            return Err("Display name cannot be empty".into());
        }
        self.record.display_name = new_name.trim().to_string();
        self.save()
    }

    pub fn rotate_root_key(
        &mut self,
        new_public_key_hex: &str,
        authorized: bool,
        reason: &str,
    ) -> Result<(), String> {
        if !authorized {
            return Err("Creator key rotation requires authenticated existing creator authorization or valid recovery.".into());
        }
        let current_pk = match self.record.root_public_key.take() {
            Some(pk) => pk,
            None => return Err("Cannot rotate an uninitialized creator key".into()),
        };

        // Archive old key
        self.record.revoked_keys.push(RevokedKey {
            key_version: self.record.key_version,
            public_key: current_pk,
            revoked_at: crate::now_iso(),
            reason: reason.to_string(),
        });

        self.record.key_version += 1;
        self.record.root_public_key = Some(new_public_key_hex.to_lowercase());
        self.record.last_key_rotation = Some(crate::now_iso());
        self.save()
    }

    pub fn is_key_active(&self, pubkey_hex: &str) -> bool {
        if let Some(ref current) = self.record.root_public_key {
            if current.eq_ignore_ascii_case(pubkey_hex) {
                return true;
            }
        }
        false
    }

    pub fn verify_authority(&self, candidate_pubkey_hex: &str) -> bool {
        self.is_initialized() && self.is_key_active(candidate_pubkey_hex)
    }
}
