//! Software License Provenance and Copyleft Conflict Engine.
//!
//! Provides SPDX license classification (MIT, Apache-2.0, BSD, GPL, AGPL, etc.),
//! copyleft taint and compatibility conflict detection, and persistent
//! provenance ledger management in storage/provenance/provenance_registry.json.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LicenseFamily {
    Permissive,
    WeakCopyleft,
    StrongCopyleft,
    NetworkCopyleft,
    Proprietary,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpdxLicenseInfo {
    pub spdx_id: String,
    pub name: String,
    pub family: LicenseFamily,
    pub requires_source_distribution: bool,
    pub allows_commercial: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    pub component_name: String,
    pub file_path: String,
    pub detected_spdx: String,
    pub family: LicenseFamily,
    pub author_or_origin: String,
    pub sha256_checksum: String,
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyleftConflictReport {
    pub has_conflict: bool,
    pub project_target_license: String,
    pub conflicting_components: Vec<String>,
    pub explanation: String,
}

pub struct LicenseProvenanceEngine {
    pub repo_root: String,
    registry_file: String,
    records: Arc<Mutex<HashMap<String, ProvenanceRecord>>>,
}

impl LicenseProvenanceEngine {
    pub fn new(repo_root: &str) -> Self {
        let prov_dir = format!("{}/storage/provenance", repo_root);
        let _ = fs::create_dir_all(&prov_dir);
        let registry_file = format!("{}/provenance_registry.json", prov_dir);

        let mut loaded = HashMap::new();
        if Path::new(&registry_file).exists() {
            if let Ok(content) = fs::read_to_string(&registry_file) {
                if let Ok(map) = serde_json::from_str::<HashMap<String, ProvenanceRecord>>(&content)
                {
                    loaded = map;
                }
            }
        }

        Self {
            repo_root: repo_root.to_string(),
            registry_file,
            records: Arc::new(Mutex::new(loaded)),
        }
    }

    /// Classifies license text or header into an SPDX license identifier.
    pub fn classify_license_text(&self, text: &str) -> SpdxLicenseInfo {
        let lower = text.to_lowercase();

        if lower.contains("affero general public license")
            || lower.contains("agpl-3.0")
            || lower.contains("agplv3")
        {
            SpdxLicenseInfo {
                spdx_id: "AGPL-3.0-only".to_string(),
                name: "GNU Affero General Public License v3.0".to_string(),
                family: LicenseFamily::NetworkCopyleft,
                requires_source_distribution: true,
                allows_commercial: true,
            }
        } else if lower.contains("general public license")
            || lower.contains("gpl-3.0")
            || lower.contains("gplv3")
            || lower.contains("gpl-2.0")
        {
            SpdxLicenseInfo {
                spdx_id: "GPL-3.0-only".to_string(),
                name: "GNU General Public License v3.0".to_string(),
                family: LicenseFamily::StrongCopyleft,
                requires_source_distribution: true,
                allows_commercial: true,
            }
        } else if lower.contains("lesser general public license")
            || lower.contains("lgpl-3.0")
            || lower.contains("lgpl-2.1")
        {
            SpdxLicenseInfo {
                spdx_id: "LGPL-3.0-only".to_string(),
                name: "GNU Lesser General Public License v3.0".to_string(),
                family: LicenseFamily::WeakCopyleft,
                requires_source_distribution: true,
                allows_commercial: true,
            }
        } else if lower.contains("mozilla public license") || lower.contains("mpl-2.0") {
            SpdxLicenseInfo {
                spdx_id: "MPL-2.0".to_string(),
                name: "Mozilla Public License 2.0".to_string(),
                family: LicenseFamily::WeakCopyleft,
                requires_source_distribution: true,
                allows_commercial: true,
            }
        } else if lower.contains("apache license, version 2.0") || lower.contains("apache-2.0") {
            SpdxLicenseInfo {
                spdx_id: "Apache-2.0".to_string(),
                name: "Apache License 2.0".to_string(),
                family: LicenseFamily::Permissive,
                requires_source_distribution: false,
                allows_commercial: true,
            }
        } else if lower == "mit"
            || lower.contains("mit license")
            || lower.contains("permission is hereby granted, free of charge")
        {
            SpdxLicenseInfo {
                spdx_id: "MIT".to_string(),
                name: "MIT License".to_string(),
                family: LicenseFamily::Permissive,
                requires_source_distribution: false,
                allows_commercial: true,
            }
        } else if lower.contains("redistribution and use in source and binary forms")
            || lower.contains("bsd")
        {
            SpdxLicenseInfo {
                spdx_id: "BSD-3-Clause".to_string(),
                name: "BSD 3-Clause License".to_string(),
                family: LicenseFamily::Permissive,
                requires_source_distribution: false,
                allows_commercial: true,
            }
        } else if lower.contains("all rights reserved") || lower.contains("proprietary") {
            SpdxLicenseInfo {
                spdx_id: "Proprietary".to_string(),
                name: "Proprietary Closed Source".to_string(),
                family: LicenseFamily::Proprietary,
                requires_source_distribution: false,
                allows_commercial: false,
            }
        } else {
            SpdxLicenseInfo {
                spdx_id: "Unknown".to_string(),
                name: "Unclassified License".to_string(),
                family: LicenseFamily::Unknown,
                requires_source_distribution: false,
                allows_commercial: false,
            }
        }
    }

    /// Registers a component provenance record and saves the ledger to disk.
    pub fn register_component(
        &self,
        component_name: &str,
        file_path: &str,
        license_text_or_spdx: &str,
        origin: &str,
        sha256_checksum: &str,
    ) -> Result<ProvenanceRecord, std::io::Error> {
        let info = self.classify_license_text(license_text_or_spdx);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let record = ProvenanceRecord {
            component_name: component_name.to_string(),
            file_path: file_path.to_string(),
            detected_spdx: info.spdx_id,
            family: info.family,
            author_or_origin: origin.to_string(),
            sha256_checksum: sha256_checksum.to_string(),
            timestamp_ms: now,
        };

        {
            let mut records = self.records.lock().unwrap();
            records.insert(component_name.to_string(), record.clone());
        }

        self.save_ledger()?;
        Ok(record)
    }

    /// Evaluates if incorporating all registered components into a target license causes a copyleft conflict.
    pub fn check_copyleft_conflicts(&self, project_target_license: &str) -> CopyleftConflictReport {
        let target_info = self.classify_license_text(project_target_license);
        let records = self.records.lock().unwrap();
        let mut conflicts = Vec::new();

        for (comp_name, rec) in records.iter() {
            // If project is Permissive (like MIT/Apache), including Strong or Network Copyleft is a conflict
            if target_info.family == LicenseFamily::Permissive {
                if rec.family == LicenseFamily::StrongCopyleft
                    || rec.family == LicenseFamily::NetworkCopyleft
                {
                    conflicts.push(format!("{}: {}", comp_name, rec.detected_spdx));
                }
            } else if target_info.family == LicenseFamily::StrongCopyleft
                && rec.family == LicenseFamily::NetworkCopyleft
            {
                conflicts.push(format!("{}: {}", comp_name, rec.detected_spdx));
            }
        }

        let has_conflict = !conflicts.is_empty();
        let explanation = if has_conflict {
            format!(
                "Incompatible copyleft contamination: project license '{}' cannot assimilate copyleft components [{}] without re-licensing the entire project under corresponding copyleft terms.",
                project_target_license,
                conflicts.join(", ")
            )
        } else {
            format!(
                "All {} registered components are license-compatible with project target '{}'.",
                records.len(),
                project_target_license
            )
        };

        CopyleftConflictReport {
            has_conflict,
            project_target_license: project_target_license.to_string(),
            conflicting_components: conflicts,
            explanation,
        }
    }

    fn save_ledger(&self) -> Result<(), std::io::Error> {
        let records = self.records.lock().unwrap();
        let json_str = serde_json::to_string_pretty(&*records)?;
        fs::write(&self.registry_file, json_str)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_license_classification() {
        let temp = std::env::temp_dir().join("tara_test_prov_cls");
        let engine = LicenseProvenanceEngine::new(temp.to_str().unwrap());

        let mit = engine.classify_license_text("Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files");
        assert_eq!(mit.spdx_id, "MIT");
        assert_eq!(mit.family, LicenseFamily::Permissive);

        let gpl =
            engine.classify_license_text("GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007");
        assert_eq!(gpl.spdx_id, "GPL-3.0-only");
        assert_eq!(gpl.family, LicenseFamily::StrongCopyleft);

        let agpl = engine.classify_license_text("GNU Affero General Public License Version 3");
        assert_eq!(agpl.spdx_id, "AGPL-3.0-only");
        assert_eq!(agpl.family, LicenseFamily::NetworkCopyleft);

        let _ = fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_copyleft_conflict_detection() {
        let temp = std::env::temp_dir().join("tara_test_prov_conflict");
        let engine = LicenseProvenanceEngine::new(temp.to_str().unwrap());

        engine
            .register_component(
                "lib_utils",
                "src/utils.rs",
                "MIT License",
                "Community",
                "abc1",
            )
            .unwrap();
        engine
            .register_component(
                "gpl_driver",
                "src/driver.rs",
                "GNU General Public License v3",
                "ThirdParty",
                "abc2",
            )
            .unwrap();

        // Testing against MIT project license -> Conflict!
        let report = engine.check_copyleft_conflicts("MIT");
        assert!(report.has_conflict);
        assert_eq!(report.conflicting_components.len(), 1);
        assert!(report.conflicting_components[0].contains("gpl_driver"));

        // Testing against GPL-3.0 project license -> Compatible!
        let report_gpl = engine.check_copyleft_conflicts("GPL-3.0-only");
        assert!(!report_gpl.has_conflict);

        let _ = fs::remove_dir_all(&temp);
    }
}
