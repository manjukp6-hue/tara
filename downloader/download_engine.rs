//! TARA Multi-Stage Dataset Download Engine (100% Native Rust)
//!
//! Replaces legacy download pool with a strict, stage-gated download engine:
//!
//! Workflow:
//! 1. Stage 1 (Download Register):
//!    - Verifies incoming link / dataset ID against `DownloadRegister`.
//!    - If already registered or excluded -> REJECT immediately.
//!    - If approved -> Proceed to Stage 2.
//! 2. Stage 2 (License Register):
//!    - Verifies license against `LicenseRegister` (Rule 18: Permissive, commercial-compliant).
//!    - If non-permissive (NC, ND, Copyleft, Research-only) -> REJECT immediately.
//!    - If approved -> Proceed to Stage 3.
//! 3. Stage 3 (Filter):
//!    - Filter stage placeholder hook (Stage under active design).
//!    - When approved -> Proceed to download.
//! 4. Download Execution & Verification:
//!    - Downloads via native `curl.exe` with retries, timeout, and byte verification.
//!    - Saves to configured `downloaded/` folder.
//!    - Dynamically computes runtime SHA-256 directly from disk (Rule 2).
//!    - Checks hash against `DownloadRegister` (if collision -> deletes temp file & rejects).
//! 5. Registration:
//!    - Records dataset in `DownloadRegister` (auto-saves `.json` and `.txt`).
//!    - Records license in `LicenseRegister` (auto-saves `.json` and `.txt`).

#[path = "download_register.rs"]
pub mod download_register;

#[path = "license_register.rs"]
pub mod license_register;

#[path = "filter_engine.rs"]
pub mod filter_engine;

use download_register::{Decision, DownloadRegister, RegisterEntry};
use filter_engine::{FilterDecision, FilterEngine, PreDownloadDecision, PreDownloadGate, RejectionReasonCode};
use license_register::{LicenseDecision, LicenseEntry, LicenseRegister};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

// ============================================================================
// Core Configuration & Standalone Boundary
// ============================================================================

/// Default folder dedicated solely for saving downloaded datasets.
/// Standalone Boundary Directive:
/// - Strictly ZERO connection to dataset engine (tara_engine, compiler, tokenizer).
/// - Downloader is 100% standalone and isolated.
/// - Approved files are saved ONLY into this `downloaded/` folder.
pub const DEFAULT_DOWNLOADED_DIR: &str = "downloaded";

// ============================================================================
// 1. Core Data Models
// ============================================================================

/// Canonical structured record serialized into JSON Lines (.jsonl) dataset files.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DatasetJsonlRecord {
    pub id: String,
    pub text: String,
    pub domain: String,
    pub author: String,
    pub license: String,
    pub license_proof_url: String,
    pub source_url: String,
    pub line_count: usize,
    pub char_count: usize,
}

/// Incoming candidate request for dataset download.
#[derive(Debug, Clone)]
pub struct DownloadCandidate {
    pub dataset_id: String,
    pub source_url: String,
    pub author: String,
    pub license: String,
    pub license_proof_url: String,
    pub target_filename: Option<String>,
    pub asset_scope: Option<String>,
    pub required_exclusions: Option<String>,
    pub jurisdiction_scope: Option<String>,
}

impl DownloadCandidate {
    /// Creates a new candidate with required fields and defaults for optional fields.
    pub fn new(
        dataset_id: impl Into<String>,
        source_url: impl Into<String>,
        author: impl Into<String>,
        license: impl Into<String>,
        license_proof_url: impl Into<String>,
    ) -> Self {
        Self {
            dataset_id: dataset_id.into(),
            source_url: source_url.into(),
            author: author.into(),
            license: license.into(),
            license_proof_url: license_proof_url.into(),
            target_filename: None,
            asset_scope: None,
            required_exclusions: None,
            jurisdiction_scope: None,
        }
    }
}

/// Result of multi-stage pipeline channel verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageVerificationResult {
    /// Approved through all active stages -> Ready for download.
    Approved,

    /// Rejected at a specific stage with exact reason and optional granular reason code.
    Rejected {
        stage_num: usize,
        stage_name: &'static str,
        reason: String,
        reason_code: Option<RejectionReasonCode>,
    },
}

impl StageVerificationResult {
    pub fn is_approved(&self) -> bool {
        matches!(self, StageVerificationResult::Approved)
    }

    pub fn is_rejected(&self) -> bool {
        matches!(self, StageVerificationResult::Rejected { .. })
    }

    pub fn reason_code(&self) -> Option<RejectionReasonCode> {
        match self {
            StageVerificationResult::Approved => None,
            StageVerificationResult::Rejected { reason_code, .. } => *reason_code,
        }
    }
}

// ============================================================================
// 2. Download Engine
// ============================================================================

/// Multi-stage dataset download engine.
pub struct DownloadEngine {
    pub download_register: DownloadRegister,
    pub license_register: LicenseRegister,
    pub filter_engine: FilterEngine,
    pub pre_download_gate: PreDownloadGate,
}

impl DownloadEngine {
    /// Creates a new engine by loading existing registries from disk.
    pub fn load_from_dir(register_dir: &Path) -> Result<Self, String> {
        let dl_json = register_dir.join("download_register.json");
        let lic_json = register_dir.join("license_register.json");

        let download_register = if dl_json.exists() {
            DownloadRegister::load_from_json(&dl_json)?
        } else {
            DownloadRegister::new()
        };

        let license_register = if lic_json.exists() {
            LicenseRegister::load_from_json(&lic_json)?
        } else {
            LicenseRegister::new()
        };

        let filter_engine = FilterEngine::new();
        let pre_download_gate = PreDownloadGate::new_with_catalog();

        Ok(Self {
            download_register,
            license_register,
            filter_engine,
            pre_download_gate,
        })
    }

    /// Creates an engine with existing in-memory registers.
    pub fn new(download_register: DownloadRegister, license_register: LicenseRegister) -> Self {
        Self {
            download_register,
            license_register,
            filter_engine: FilterEngine::new(),
            pre_download_gate: PreDownloadGate::new_with_catalog(),
        }
    }

    /// Creates an engine with custom filter engine and pre-download gate.
    pub fn with_filter(
        download_register: DownloadRegister,
        license_register: LicenseRegister,
        filter_engine: FilterEngine,
        pre_download_gate: PreDownloadGate,
    ) -> Self {
        Self {
            download_register,
            license_register,
            filter_engine,
            pre_download_gate,
        }
    }

    /// Channels an incoming candidate through strict verification stages before ANY network download starts.
    ///
    /// - Stage 1: Download Register (check if link or ID already registered or excluded) -> REJECT_DUPLICATE
    /// - Stage 2: Pre-Download Rule-18/21 Gate (authoritative catalog allowlist, exact asset scoping,
    ///   forbidden license checking, rights proof, third-party exclusion, provenance)
    /// - Stage 3: License Register (verify license compliance & commercial rights)
    /// - Stage 4: Filter Pre-Check (pre-check for synthetic or distilled datasets per Rules 1 & 18) -> REJECT_SYNTHETIC
    pub fn verify_candidate_stages(&self, candidate: &DownloadCandidate) -> StageVerificationResult {
        // --------------------------------------------------------------------
        // Stage 1: Download Register Check (Existing dataset collision & repeat avoidance)
        // --------------------------------------------------------------------
        let stage1_decision = self.download_register.check_link(&candidate.dataset_id, &candidate.source_url);
        if let Decision::Reject(reason) = stage1_decision {
            return StageVerificationResult::Rejected {
                stage_num: 1,
                stage_name: "Stage 1 (Download Register)",
                reason,
                reason_code: Some(RejectionReasonCode::RejectDuplicate),
            };
        }

        // --------------------------------------------------------------------
        // Stage 2: Pre-Download Rule-18/21 Gate (Catalog, Scoping, Rights, Provenance)
        // --------------------------------------------------------------------
        let pre_gate_decision = self.pre_download_gate.evaluate(
            &candidate.dataset_id,
            &candidate.source_url,
            &candidate.author,
            &candidate.license,
            &candidate.license_proof_url,
            candidate.asset_scope.as_deref(),
        );

        if let PreDownloadDecision::Reject { code, rule, evidence, details } = pre_gate_decision {
            let reason = format!("[{}] {} (Rule: {}, Evidence: {})", code.as_str(), details, rule, evidence);
            return StageVerificationResult::Rejected {
                stage_num: 2,
                stage_name: "Stage 2 (Pre-Download Rule-18/21 Gate)",
                reason,
                reason_code: Some(code),
            };
        }

        // --------------------------------------------------------------------
        // Stage 3: License Register Check
        // --------------------------------------------------------------------
        let stage3_decision = self.license_register.check_candidate(&candidate.dataset_id, &candidate.license);
        if let LicenseDecision::Reject(reason) = stage3_decision {
            return StageVerificationResult::Rejected {
                stage_num: 3,
                stage_name: "Stage 3 (License Register)",
                reason,
                reason_code: Some(RejectionReasonCode::RejectForbiddenLicense),
            };
        }

        // --------------------------------------------------------------------
        // Stage 4: Filter Pre-Check (Rules 1 & 18: Synthetic & Distillation Check)
        // --------------------------------------------------------------------
        let stage4_decision = self.filter_engine.pre_check_candidate(
            &candidate.dataset_id,
            &candidate.source_url,
            &candidate.author,
        );
        if let FilterDecision::Reject(reason) = stage4_decision {
            return StageVerificationResult::Rejected {
                stage_num: 4,
                stage_name: "Stage 4 (Filter Pre-Check)",
                reason,
                reason_code: Some(RejectionReasonCode::RejectSynthetic),
            };
        }

        StageVerificationResult::Approved
    }

    /// Executes the full stage-gated download flow:
    /// 1. Channels candidate through Stage 1, Stage 2, Stage 3 pre-checks.
    /// 2. If approved, downloads file via native `curl.exe` into `downloaded_dir`.
    /// 3. Computes dynamic SHA-256 and verifies hash in download register.
    /// 4. Executes Stage 3 post-download quality filter and domain/topic classification.
    /// 5. Registers in `download_register` (auto-saves `.json` and `.txt`).
    /// 6. Registers in `license_register` with domain & source proof (auto-saves `.json` and `.txt`).
    pub fn download_and_register(
        &mut self,
        candidate: &DownloadCandidate,
        downloaded_dir: &Path,
        register_dir: &Path,
    ) -> Result<RegisterEntry, String> {
        // 1. Channel candidate through Pre-Download Gate and Stage checks
        let stage_result = self.verify_candidate_stages(candidate);
        if let StageVerificationResult::Rejected { stage_num, stage_name, reason, .. } = stage_result {
            // Immediately persist rejection into registers with ZERO bytes downloaded (0 byte network leak)
            let now_epoch = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let now_utc = format!("epoch_{now_epoch}");

            let mut dl_rej = RegisterEntry {
                dataset_id: candidate.dataset_id.clone(),
                source_url: candidate.source_url.clone(),
                author: candidate.author.clone(),
                sha256: String::new(),
                size_bytes: 0,
                record_count: 0,
                local_path: "NONE".to_string(),
                timestamp: now_utc.clone(),
                download_permission: "REJECTED".to_string(),
                training_eligibility: "REJECTED".to_string(),
                policy_decision: format!("REJECTED at {stage_name}: {reason}"),
                ..Default::default()
            };
            dl_rej.enrich_provenance();
            let dl_base = register_dir.join("download_register");
            let _ = self.download_register.register_rejected(dl_rej, &dl_base);

            let mut lic_rej = LicenseEntry {
                dataset_id: candidate.dataset_id.clone(),
                license: candidate.license.clone(),
                proof_url: candidate.license_proof_url.clone(),
                source_url: candidate.source_url.clone(),
                author: candidate.author.clone(),
                commercial_use: false,
                modify_allowed: false,
                download_allowed: false,
                ai_training_allowed: false,
                timestamp: now_utc,
                download_permission: "REJECTED".to_string(),
                training_eligibility: "REJECTED".to_string(),
                policy_decision: format!("REJECTED at {stage_name}: {reason}"),
                asset_scope: candidate.asset_scope.clone().unwrap_or_else(|| "NONE".to_string()),
                third_party_exclusions: candidate.required_exclusions.clone().unwrap_or_else(|| "NONE".to_string()),
                jurisdiction_scope: candidate.jurisdiction_scope.clone().unwrap_or_else(|| "VERIFIED_JURISDICTIONS_ONLY".to_string()),
                ..Default::default()
            };
            lic_rej.enrich_provenance();
            let lic_base = register_dir.join("license_register");
            let _ = self.license_register.register_rejected(lic_rej, &lic_base);

            return Err(format!("Rejected at {stage_name} (Stage {stage_num}): {reason}"));
        }

        // 2. Prepare output filename and directory
        fs::create_dir_all(downloaded_dir)
            .map_err(|e| format!("Failed to create download directory {}: {e}", downloaded_dir.display()))?;

        let default_filename = candidate
            .source_url
            .split('/')
            .next_back()
            .unwrap_or(&candidate.dataset_id);
        let filename = candidate
            .target_filename
            .as_deref()
            .unwrap_or(default_filename);

        // 3. Download via native curl tool (with automatic streaming decompression for .gz archives)
        let is_gzip = candidate.source_url.ends_with(".gz") || filename.ends_with(".gz");
        let (target_path, _download_size) = if is_gzip {
            let temp_gz = downloaded_dir.join(format!("{}.gz_download", candidate.dataset_id));
            let decompressed_name = if filename.ends_with(".json.gz") {
                format!("{}.jsonl", filename.trim_end_matches(".json.gz"))
            } else if filename.ends_with(".gz") {
                filename.trim_end_matches(".gz").to_string()
            } else {
                format!("{}.jsonl", candidate.dataset_id)
            };
            let final_decompressed_path = downloaded_dir.join(&decompressed_name);

            println!("Stage verification PASSED. Downloading compressed archive {} via curl...", candidate.source_url);
            let downloaded_bytes = Self::curl_download(&candidate.source_url, &temp_gz, 1800, 3)?;
            println!("Download complete ({} bytes compressed). Decompressing streamingly on disk...", downloaded_bytes);
            let uncompressed_bytes = Self::decompress_gzip(&temp_gz, &final_decompressed_path)?;
            let _ = fs::remove_file(&temp_gz);
            println!("Decompressed to {} ({} bytes uncompressed).", final_decompressed_path.display(), uncompressed_bytes);
            (final_decompressed_path, uncompressed_bytes)
        } else {
            let target_path = downloaded_dir.join(filename);
            println!("Stage verification PASSED. Downloading {} via curl...", candidate.source_url);
            let byte_size = Self::curl_download(&candidate.source_url, &target_path, 1800, 3)?;
            (target_path, byte_size)
        };

        // 4. Streaming Deep Quality Filter & Domain Classification (Rule 18/21)
        let filter_decision = self.filter_engine.evaluate_file(&target_path, &candidate.dataset_id);
        let domain = match filter_decision {
            FilterDecision::Approve { domain, indicator_score } => {
                println!(
                    "Streaming Deep Filter APPROVED. Classified domain: '{}' (indicator match ratio: {:.1}%).",
                    domain,
                    indicator_score * 100.0
                );
                domain
            }
            FilterDecision::Reject(reason) => {
                let _ = fs::remove_file(&target_path);
                let now_epoch = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                let now_utc = format!("epoch_{now_epoch}");

                let mut dl_rej = RegisterEntry {
                    dataset_id: candidate.dataset_id.clone(),
                    source_url: candidate.source_url.clone(),
                    author: candidate.author.clone(),
                    sha256: String::new(),
                    size_bytes: 0,
                    record_count: 0,
                    local_path: "NONE".to_string(),
                    timestamp: now_utc.clone(),
                    download_permission: "REJECTED".to_string(),
                    training_eligibility: "REJECTED".to_string(),
                    policy_decision: format!("REJECTED by Streaming Deep Filter: {reason}"),
                    ..Default::default()
                };
                dl_rej.enrich_provenance();
                let dl_base = register_dir.join("download_register");
                let _ = self.download_register.register_rejected(dl_rej, &dl_base);

                let mut lic_rej = LicenseEntry {
                    dataset_id: candidate.dataset_id.clone(),
                    license: candidate.license.clone(),
                    proof_url: candidate.license_proof_url.clone(),
                    source_url: candidate.source_url.clone(),
                    author: candidate.author.clone(),
                    commercial_use: false,
                    modify_allowed: false,
                    download_allowed: false,
                    ai_training_allowed: false,
                    timestamp: now_utc,
                    download_permission: "REJECTED".to_string(),
                    training_eligibility: "REJECTED".to_string(),
                    policy_decision: format!("REJECTED by Streaming Deep Filter: {reason}"),
                    asset_scope: candidate.asset_scope.clone().unwrap_or_else(|| "NONE".to_string()),
                    third_party_exclusions: candidate.required_exclusions.clone().unwrap_or_else(|| "NONE".to_string()),
                    jurisdiction_scope: candidate.jurisdiction_scope.clone().unwrap_or_else(|| "VERIFIED_JURISDICTIONS_ONLY".to_string()),
                    ..Default::default()
                };
                lic_rej.enrich_provenance();
                let lic_base = register_dir.join("license_register");
                let _ = self.license_register.register_rejected(lic_rej, &lic_base);

                return Err(format!("Streaming Deep Filter rejected: {reason}"));
            }
        };

        // 5. Deep Line-by-Line License Audit with Rule-18 per-record sanitization (Rule 18)
        println!("Executing Deep Line-by-Line License Audit on {}...", target_path.display());
        let audit_summary = match LicenseRegister::audit_file_licenses(&target_path, &candidate.license) {
            Ok(summary) => {
                println!(
                    "Deep License Audit PASSED: {} lines audited across {} licenses ({:?}) and {} authors.",
                    summary.total_lines_audited,
                    summary.distinct_licenses.len(),
                    summary.distinct_licenses,
                    summary.distinct_authors.len()
                );
                summary
            }
            Err(violation) => {
                println!(
                    "Deep License Audit detected non-permissive record at line {} ({}). Initiating Rule-18 per-record sanitization...",
                    violation.line_number, violation.detected_license
                );
                let sanitized_path = target_path.with_extension("sanitized_tmp");
                match LicenseRegister::filter_and_sanitize_mixed_records(&target_path, &sanitized_path, &candidate.license) {
                    Ok(stats) => {
                        println!(
                            "Per-record sanitization SUCCESS: kept {} records, dropped {} violating records.",
                            stats.accepted_records, stats.dropped_records
                        );
                        let _ = fs::remove_file(&target_path);
                        fs::rename(&sanitized_path, &target_path)
                            .map_err(|e| format!("Failed to replace target with sanitized version: {e}"))?;
                        // Re-audit sanitized file to guarantee 100% compliance
                        LicenseRegister::audit_file_licenses(&target_path, &candidate.license)
                            .map_err(|v| format!("Post-sanitization audit failed: {}", v.reason))?
                    }
                    Err(e) => {
                        let _ = fs::remove_file(&target_path);
                        let _ = fs::remove_file(&sanitized_path);
                        let now_epoch = SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .map(|d| d.as_secs())
                            .unwrap_or(0);
                        let now_utc = format!("epoch_{now_epoch}");

                        let mut dl_rej = RegisterEntry {
                            dataset_id: candidate.dataset_id.clone(),
                            source_url: candidate.source_url.clone(),
                            author: candidate.author.clone(),
                            sha256: String::new(),
                            size_bytes: 0,
                            record_count: 0,
                            local_path: "NONE".to_string(),
                            timestamp: now_utc.clone(),
                            download_permission: "REJECTED".to_string(),
                            training_eligibility: "REJECTED".to_string(),
                            policy_decision: format!("REJECTED by Deep License Audit: {e}"),
                            ..Default::default()
                        };
                        dl_rej.enrich_provenance();
                        let dl_base = register_dir.join("download_register");
                        let _ = self.download_register.register_rejected(dl_rej, &dl_base);

                        let mut lic_rej = LicenseEntry {
                            dataset_id: candidate.dataset_id.clone(),
                            license: candidate.license.clone(),
                            proof_url: candidate.license_proof_url.clone(),
                            source_url: candidate.source_url.clone(),
                            author: candidate.author.clone(),
                            commercial_use: false,
                            modify_allowed: false,
                            download_allowed: false,
                            ai_training_allowed: false,
                            timestamp: now_utc,
                            download_permission: "REJECTED".to_string(),
                            training_eligibility: "REJECTED".to_string(),
                            policy_decision: format!("REJECTED by Deep License Audit: {e}"),
                            asset_scope: candidate.asset_scope.clone().unwrap_or_else(|| "NONE".to_string()),
                            third_party_exclusions: candidate.required_exclusions.clone().unwrap_or_else(|| "NONE".to_string()),
                            jurisdiction_scope: candidate.jurisdiction_scope.clone().unwrap_or_else(|| "VERIFIED_JURISDICTIONS_ONLY".to_string()),
                            ..Default::default()
                        };
                        lic_rej.enrich_provenance();
                        let lic_base = register_dir.join("license_register");
                        let _ = self.license_register.register_rejected(lic_rej, &lic_base);

                        return Err(format!("Per-record license sanitization failed: {e}"));
                    }
                }
            }
        };

        // 5.5 Transform raw text into canonical structured JSONL format
        let is_already_jsonl = target_path.extension().and_then(|e| e.to_str()) == Some("jsonl");
        let (final_path, record_count) = if is_already_jsonl {
            let recs = Self::count_records_if_text(&target_path);
            (target_path, recs)
        } else {
            let jsonl_filename = format!("{}.jsonl", candidate.dataset_id);
            let jsonl_path = downloaded_dir.join(&jsonl_filename);
            println!("Transforming raw text to canonical structured JSONL at {}...", jsonl_path.display());
            let recs = Self::convert_text_to_jsonl(&target_path, &jsonl_path, candidate, &domain)?;
            println!("Transformed {} structured records into JSONL.", recs);
            let _ = fs::remove_file(&target_path); // Remove raw file
            (jsonl_path, recs)
        };

        // 6. Compute dynamic SHA-256 & byte size directly from authentic final file on disk (Rule 2)
        let final_metadata = fs::metadata(&final_path)
            .map_err(|e| format!("Failed to read file metadata for {}: {e}", final_path.display()))?;
        let byte_size = final_metadata.len();
        let dynamic_sha = DownloadRegister::compute_dynamic_sha256(&final_path)
            .map_err(|e| format!("Failed to compute dynamic SHA-256: {e}"))?;

        // 7. Post-download hash collision check
        if let Decision::Reject(reason) = self.download_register.check_sha256(&dynamic_sha) {
            let _ = fs::remove_file(&final_path);
            return Err(format!("Post-download hash collision rejected: {reason}"));
        }

        let now_epoch = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let now_utc = format!("epoch_{now_epoch}");

        // 8. Register in DownloadRegister (auto-saves .json and .txt)
        let mut dl_entry = RegisterEntry {
            dataset_id: candidate.dataset_id.clone(),
            source_url: candidate.source_url.clone(),
            author: candidate.author.clone(),
            sha256: dynamic_sha.clone(),
            size_bytes: byte_size,
            record_count,
            local_path: final_path.to_string_lossy().to_string(),
            timestamp: now_utc.clone(),
            ..Default::default()
        };
        dl_entry.enrich_provenance();

        let dl_base = register_dir.join("download_register");
        self.download_register.register(dl_entry.clone(), &dl_base)?;

        // 9. Register in LicenseRegister with complete auditable provenance evidence & domain (auto-saves .json and .txt)
        let mut lic_entry = LicenseEntry {
            dataset_id: candidate.dataset_id.clone(),
            license: candidate.license.clone(),
            proof_url: candidate.license_proof_url.clone(),
            source_url: candidate.source_url.clone(),
            domain: domain.clone(),
            author: candidate.author.clone(),
            commercial_use: true,
            modify_allowed: true,
            download_allowed: true,
            ai_training_allowed: true,
            detected_licenses: audit_summary.distinct_licenses,
            detected_authors: audit_summary.distinct_authors,
            lines_audited: audit_summary.total_lines_audited,
            timestamp: now_utc,
            asset_scope: candidate.asset_scope.clone().unwrap_or_else(|| "NONE".to_string()),
            third_party_exclusions: candidate.required_exclusions.clone().unwrap_or_else(|| "NONE".to_string()),
            jurisdiction_scope: candidate.jurisdiction_scope.clone().unwrap_or_else(|| "VERIFIED_JURISDICTIONS_ONLY".to_string()),
            ..Default::default()
        };
        lic_entry.enrich_provenance();

        let lic_base = register_dir.join("license_register");
        self.license_register.register(lic_entry, &lic_base)?;

        println!(
            "Successfully registered dataset '{}' (Domain: {}) with auditable provenance into both registries.",
            candidate.dataset_id, domain
        );
        Ok(dl_entry)
    }

    /// Downloads candidate and saves strictly to the standalone `downloaded/` folder.
    /// Strictly NO connection to dataset engine.
    pub fn download_to_downloaded_folder(
        &mut self,
        candidate: &DownloadCandidate,
        register_dir: &Path,
    ) -> Result<RegisterEntry, String> {
        self.download_and_register(candidate, Path::new(DEFAULT_DOWNLOADED_DIR), register_dir)
    }

    // ========================================================================
    // 3. Native Helping Tools
    // ========================================================================

    /// Downloads a URL directly using native curl.exe with timeout, retries, and byte verification.
    pub fn curl_download(
        url: &str,
        target_path: &Path,
        timeout_secs: u64,
        retries: u32,
    ) -> Result<u64, String> {
        let temp_path = target_path.with_extension("download_tmp");

        let timeout_str = timeout_secs.to_string();
        let retry_str = retries.to_string();

        let status = Command::new("curl.exe")
            .args([
                "-f",                        // Fail on HTTP errors (4xx, 5xx)
                "-L",                        // Follow redirects
                "--retry", &retry_str,       // Retry count
                "--retry-delay", "2",        // Retry delay seconds
                "--connect-timeout", "30",   // Connection timeout
                "--max-time", &timeout_str,  // Total maximum time
                "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TARA/1.0",
                "-o", temp_path.to_str().unwrap_or("tmp.download"),
                url,
            ])
            .status()
            .map_err(|e| format!("Failed to execute curl.exe: {e}"))?;

        if !status.success() {
            let _ = fs::remove_file(&temp_path);
            return Err(format!("curl.exe download failed with exit code {:?}", status.code()));
        }

        let metadata = fs::metadata(&temp_path)
            .map_err(|e| format!("Downloaded file has no metadata: {e}"))?;

        let byte_len = metadata.len();
        if byte_len == 0 {
            let _ = fs::remove_file(&temp_path);
            return Err("Downloaded file is empty (0 bytes)".to_string());
        }

        // Atomically rename temporary file to target path
        fs::rename(&temp_path, target_path)
            .map_err(|e| format!("Failed to move downloaded file to {}: {e}", target_path.display()))?;

        Ok(byte_len)
    }

    /// Decompresses a gzip (.gz) archive streamingly to destination path.
    pub fn decompress_gzip(gz_path: &Path, out_path: &Path) -> Result<u64, String> {
        let in_file = File::open(gz_path)
            .map_err(|e| format!("Failed to open gzip archive {}: {e}", gz_path.display()))?;
        let reader = BufReader::new(in_file);
        let mut decoder = flate2::read::GzDecoder::new(reader);
        let mut out_file = File::create(out_path)
            .map_err(|e| format!("Failed to create decompressed file {}: {e}", out_path.display()))?;
        let bytes_written = std::io::copy(&mut decoder, &mut out_file)
            .map_err(|e| format!("Failed to decompress gzip archive {}: {e}", gz_path.display()))?;
        Ok(bytes_written)
    }

    /// Counts lines if the file is text or JSONL.
    pub fn count_records_if_text(path: &Path) -> usize {
        let is_jsonl = path.extension().and_then(|s| s.to_str()) == Some("jsonl");
        let is_txt = path.extension().and_then(|s| s.to_str()) == Some("txt");

        if !is_jsonl && !is_txt {
            return 1; // binary archive or single file defaults to 1 record
        }

        if let Ok(file) = File::open(path) {
            let reader = BufReader::new(file);
            reader.lines().map_while(Result::ok).count()
        } else {
            0
        }
    }

    /// Converts a raw text file into a canonical structured JSONL dataset file.
    /// Breaks coherent text into semantic paragraph records, injecting complete provenance metadata.
    pub fn convert_text_to_jsonl(
        raw_path: &Path,
        jsonl_path: &Path,
        candidate: &DownloadCandidate,
        domain: &str,
    ) -> Result<usize, String> {
        let file = File::open(raw_path)
            .map_err(|e| format!("Failed to open raw file {}: {e}", raw_path.display()))?;
        let reader = BufReader::new(file);

        let out_file = File::create(jsonl_path)
            .map_err(|e| format!("Failed to create JSONL file {}: {e}", jsonl_path.display()))?;
        let mut writer = std::io::BufWriter::new(out_file);

        let mut current_paragraph = String::new();
        let mut current_lines = 0usize;
        let mut record_idx = 0usize;

        for line_res in reader.lines() {
            let line = line_res.map_err(|e| format!("Error reading line: {e}"))?;
            let trimmed = line.trim();

            if trimmed.is_empty() {
                if !current_paragraph.is_empty() {
                    if current_paragraph.len() >= 40 {
                        record_idx += 1;
                        let record = DatasetJsonlRecord {
                            id: format!("{}_{:06}", candidate.dataset_id, record_idx),
                            char_count: current_paragraph.chars().count(),
                            line_count: current_lines,
                            text: current_paragraph.clone(),
                            domain: domain.to_string(),
                            author: candidate.author.clone(),
                            license: candidate.license.clone(),
                            license_proof_url: candidate.license_proof_url.clone(),
                            source_url: candidate.source_url.clone(),
                        };
                        let json_line = serde_json::to_string(&record)
                            .map_err(|e| format!("Failed to serialize record to JSON: {e}"))?;
                        use std::io::Write;
                        writeln!(writer, "{json_line}")
                            .map_err(|e| format!("Failed to write JSON line: {e}"))?;
                    }
                    current_paragraph.clear();
                    current_lines = 0;
                }
            } else {
                if !current_paragraph.is_empty() {
                    current_paragraph.push(' ');
                }
                current_paragraph.push_str(trimmed);
                current_lines += 1;
            }
        }

        if !current_paragraph.is_empty() && current_paragraph.len() >= 40 {
            record_idx += 1;
            let record = DatasetJsonlRecord {
                id: format!("{}_{:06}", candidate.dataset_id, record_idx),
                char_count: current_paragraph.chars().count(),
                line_count: current_lines,
                text: current_paragraph.clone(),
                domain: domain.to_string(),
                author: candidate.author.clone(),
                license: candidate.license.clone(),
                license_proof_url: candidate.license_proof_url.clone(),
                source_url: candidate.source_url.clone(),
            };
            let json_line = serde_json::to_string(&record)
                .map_err(|e| format!("Failed to serialize record to JSON: {e}"))?;
            use std::io::Write;
            writeln!(writer, "{json_line}")
                .map_err(|e| format!("Failed to write JSON line: {e}"))?;
        }

        use std::io::Write;
        writer.flush().map_err(|e| format!("Failed to flush JSONL writer: {e}"))?;
        Ok(record_idx)
    }

    /// Migrates existing raw .txt files in the download directory and registries to canonical .jsonl files.
    pub fn migrate_existing_txt_to_jsonl(&mut self, register_dir: &Path) -> Result<usize, String> {
        let mut migrated = 0;

        for entry in &mut self.download_register.entries {
            if entry.local_path.ends_with(".txt") {
                let txt_path = PathBuf::from(&entry.local_path);
                if txt_path.exists() {
                    let jsonl_path = txt_path.with_extension("jsonl");
                    println!("[MIGRATE] Converting {} -> {}", txt_path.display(), jsonl_path.display());
                    let candidate = DownloadCandidate {
                        dataset_id: entry.dataset_id.clone(),
                        source_url: entry.source_url.clone(),
                        author: entry.author.clone(),
                        license: "PUBLIC_DOMAIN".to_string(),
                        license_proof_url: "https://www.gutenberg.org/license".to_string(),
                        target_filename: None,
                        asset_scope: None,
                        required_exclusions: None,
                        jurisdiction_scope: None,
                    };
                    let recs = Self::convert_text_to_jsonl(&txt_path, &jsonl_path, &candidate, "science_and_philosophy")?;
                    let _ = fs::remove_file(&txt_path);
                    let new_size = fs::metadata(&jsonl_path).map(|m| m.len()).unwrap_or(0);
                    let new_sha = DownloadRegister::compute_dynamic_sha256(&jsonl_path).map_err(|e| e.to_string())?;

                    entry.local_path = jsonl_path.to_string_lossy().to_string();
                    entry.record_count = recs;
                    entry.size_bytes = new_size;
                    entry.sha256 = new_sha;

                    migrated += 1;
                }
            }
        }

        if migrated > 0 {
            let dl_base = register_dir.join("download_register");
            self.download_register.save(&dl_base)?;
            println!("[MIGRATE] Successfully migrated {} files to canonical .jsonl format.", migrated);
        }

        Ok(migrated)
    }
}

// ============================================================================
// Automated Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_channel_verification() {
        let dl_reg = DownloadRegister::new();
        let lic_reg = LicenseRegister::new();
        let engine = DownloadEngine::new(dl_reg, lic_reg);

        // 1. Candidate with non-permissive license -> Rejected at Stage 2 with REJECT_FORBIDDEN_LICENSE
        let bad_candidate = DownloadCandidate {
            dataset_id: "boole_laws_of_thought_bad_lic".to_string(),
            source_url: "https://www.gutenberg.org/cache/epub/15114/pg15114.txt".to_string(),
            author: "George Boole (d. 1864)".to_string(),
            license: "CC-BY-NC-4.0".to_string(),
            license_proof_url: "https://www.gutenberg.org/license".to_string(),
            target_filename: None,
            asset_scope: Some("Laws of Thought 1854 original".to_string()),
            required_exclusions: None,
            jurisdiction_scope: Some("VERIFIED_JURISDICTIONS_ONLY".to_string()),
        };

        let res1 = engine.verify_candidate_stages(&bad_candidate);
        assert!(res1.is_rejected());
        assert_eq!(res1.reason_code(), Some(RejectionReasonCode::RejectForbiddenLicense));
        if let StageVerificationResult::Rejected { stage_num, .. } = res1 {
            assert_eq!(stage_num, 2);
        }

        // 2. Candidate with unapproved domain/source -> Rejected at Stage 2 with REJECT_DOMAIN
        let unapproved_domain = DownloadCandidate {
            dataset_id: "unknown_web_scrape".to_string(),
            source_url: "https://random-blog.com/dump.jsonl".to_string(),
            author: "Anonymous Blogger".to_string(),
            license: "MIT".to_string(),
            license_proof_url: "https://random-blog.com/terms".to_string(),
            target_filename: None,
            asset_scope: None,
            required_exclusions: None,
            jurisdiction_scope: None,
        };

        let res2 = engine.verify_candidate_stages(&unapproved_domain);
        assert!(res2.is_rejected());
        assert_eq!(res2.reason_code(), Some(RejectionReasonCode::RejectDomain));

        // 3. Genuine approved candidate from authoritative 48 catalog -> Approved across all stages
        let good_candidate = DownloadCandidate {
            dataset_id: "boole_laws_of_thought".to_string(),
            source_url: "https://www.gutenberg.org/cache/epub/15114/pg15114.txt".to_string(),
            author: "George Boole (d. 1864)".to_string(),
            license: "PUBLIC_DOMAIN".to_string(),
            license_proof_url: "https://www.gutenberg.org/license".to_string(),
            target_filename: Some("boole_laws_of_thought.txt".to_string()),
            asset_scope: Some("Laws of Thought 1854 original".to_string()),
            required_exclusions: None,
            jurisdiction_scope: Some("VERIFIED_JURISDICTIONS_ONLY".to_string()),
        };

        let res3 = engine.verify_candidate_stages(&good_candidate);
        assert!(res3.is_approved(), "res3 was rejected: {:?}", res3);
    }

    #[test]
    fn test_pre_download_gate_lock_zero_bandwidth_leak() {
        let dl_reg = DownloadRegister::new();
        let lic_reg = LicenseRegister::new();
        let mut engine = DownloadEngine::new(dl_reg, lic_reg);

        let rejected_candidate = DownloadCandidate {
            dataset_id: "valkey_engine_docs".to_string(),
            source_url: "https://github.com/valkey-io/valkey-doc".to_string(),
            author: "Valkey Contributors".to_string(),
            license: "CC-BY-SA-4.0".to_string(),
            license_proof_url: "https://github.com/valkey-io/valkey-doc/blob/main/LICENSE".to_string(),
            target_filename: None,
            asset_scope: Some("Valkey Engine Docs".to_string()),
            required_exclusions: None,
            jurisdiction_scope: None,
        };

        let temp_dl_dir = Path::new("target/temp_test_download_dir");
        let temp_reg_dir = Path::new("target/temp_test_registers");

        // Execution of download_and_register MUST abort immediately
        let res = engine.download_and_register(&rejected_candidate, temp_dl_dir, temp_reg_dir);
        assert!(res.is_err());
        let err_msg = res.unwrap_err();
        assert!(err_msg.contains("REJECT_FORBIDDEN_LICENSE"));

        // Must be recorded as REJECTED in download_register
        assert_eq!(engine.download_register.entries.len(), 1);
        let dl_entry = &engine.download_register.entries[0];
        assert_eq!(dl_entry.dataset_id, "valkey_engine_docs");
        assert_eq!(dl_entry.download_permission, "REJECTED");
        assert_eq!(dl_entry.local_path, "NONE");
        assert_eq!(dl_entry.size_bytes, 0);

        // Must be recorded as REJECTED in license_register
        assert_eq!(engine.license_register.entries.len(), 1);
        let lic_entry = &engine.license_register.entries[0];
        assert_eq!(lic_entry.dataset_id, "valkey_engine_docs");
        assert_eq!(lic_entry.download_permission, "REJECTED");
        assert!(!lic_entry.commercial_use);
        assert!(!lic_entry.ai_training_allowed);

        // Verify zero files created in download directory (0 bandwidth leak)
        assert!(!temp_dl_dir.exists() || fs::read_dir(temp_dl_dir).unwrap().next().is_none());

        // Cleanup temporary test register artifacts (Rule 22)
        let _ = fs::remove_file("target/temp_test_registers/download_register.json");
        let _ = fs::remove_file("target/temp_test_registers/download_register.txt");
        let _ = fs::remove_file("target/temp_test_registers/license_register.json");
        let _ = fs::remove_file("target/temp_test_registers/license_register.txt");
        let _ = fs::remove_dir_all("target/temp_test_registers");
        let _ = fs::remove_dir_all(temp_dl_dir);
    }
}


