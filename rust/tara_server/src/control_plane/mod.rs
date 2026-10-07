//! TARA Dynamic Compute Control Plane
//!
//! Provides the Rust Gateway / Control Plane engine:
//! - Worker Registry & 9-stage lifecycle (DISCOVERED -> READY, QUARANTINED, REVOKED)
//! - Model SHA256 integrity verification against canonical manifest invariant
//! - Distributed Job Engine: intelligent sharding, lease tracking, failure reassignment, result aggregation
//! - Provider Adapter Manager: Cloudflare, ModelScope, HuggingFace, Render, GenericContainer, LocalDevice, FutureProvider
//! - Canonical Manifest Exporter & Differential Sync validator

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

// NOTE: The 118,080-parameter smoke-test model has been cleanly removed.
// These invariants are set to sentinel values until a real production model is trained and registered.
pub const CANONICAL_MODEL_SHA256: &str = "NO_MODEL_REGISTERED";
pub const CANONICAL_PARAM_COUNT: usize = 0;

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ──────────────────────────────────────────────────────────────────────────────
// Worker Registry & Lifecycle
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerNode {
    pub worker_id: String,
    pub name: String,
    pub provider: String,
    pub endpoint: String,
    pub capabilities: Vec<String>,
    pub model_sha256: String,
    pub status: String, // DISCOVERED, AUTHENTICATING, SYNCING, READY, BUSY, DEGRADED, UNAVAILABLE, QUARANTINED, REVOKED
    pub quarantine_reason: Option<String>,
    pub registered_at: u64,
    pub last_heartbeat: u64,
    pub active_jobs: usize,
    pub max_concurrency: usize,
    pub latency_ms: f64,
    pub error_count: u32,
    pub auth_token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRegistryStore {
    pub version: String,
    pub updated_at: u64,
    pub workers: Vec<WorkerNode>,
}

pub struct WorkerRegistry {
    workers: HashMap<String, WorkerNode>,
    persistence_path: Option<String>,
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerRegistry {
    pub fn new() -> Self {
        Self::load_or_new(None)
    }

    pub fn load_or_new(persistence_path: Option<String>) -> Self {
        let mut reg = Self {
            workers: HashMap::new(),
            persistence_path: persistence_path.clone(),
        };

        // Auto-register built-in local engines
        let now = current_timestamp();
        reg.workers.insert(
            "rust_cpu_engine_01".to_string(),
            WorkerNode {
                worker_id: "rust_cpu_engine_01".to_string(),
                name: "Rust Native CPU Engine".to_string(),
                provider: "local_device".to_string(),
                endpoint: "internal://rust_cpu".to_string(),
                capabilities: vec![
                    "chat".to_string(),
                    "inference".to_string(),
                    "cpu".to_string(),
                    "low_latency".to_string(),
                ],
                model_sha256: CANONICAL_MODEL_SHA256.to_string(),
                status: "AWAITING_MODEL".to_string(),
                quarantine_reason: None,
                registered_at: now,
                last_heartbeat: now,
                active_jobs: 0,
                max_concurrency: 8,
                latency_ms: 12.0,
                error_count: 0,
                auth_token: String::new(),
            },
        );

        // Load persisted workers from disk if file exists
        if let Some(ref p) = persistence_path {
            let path = Path::new(p);
            if path.exists() {
                if let Ok(content) = fs::read_to_string(path) {
                    if let Ok(store) = serde_json::from_str::<WorkerRegistryStore>(&content) {
                        for mut w in store.workers {
                            let legacy_python_worker = w.worker_id == "python_gpu_worker_01"
                                || w.name.to_ascii_lowercase().contains("python")
                                || w.endpoint.to_ascii_lowercase().contains("python");
                            if legacy_python_worker {
                                continue;
                            }
                            if w.worker_id != "rust_cpu_engine_01"
                                && w.status != "REVOKED"
                                && w.status != "QUARANTINED"
                            {
                                w.status = "DISCOVERED".to_string();
                            }
                            if w.status != "REVOKED" {
                                reg.workers.insert(w.worker_id.clone(), w);
                            }
                        }
                    }
                }
            }
        }

        reg
    }

    pub fn save(&self) {
        if let Some(ref p) = self.persistence_path {
            let path = Path::new(p);
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            let workers_vec: Vec<WorkerNode> = self
                .workers
                .values()
                .filter(|w| w.status != "REVOKED")
                .cloned()
                .collect();
            let store = WorkerRegistryStore {
                version: "1.0.0".to_string(),
                updated_at: current_timestamp(),
                workers: workers_vec,
            };
            if let Ok(data) = serde_json::to_string_pretty(&store) {
                let _ = fs::write(path, data);
            }
        }
    }

    pub fn validate_worker_token(&self, token: &str) -> Option<String> {
        if token.trim().is_empty() {
            return None;
        }
        let token_hash = hex::encode(Sha256::digest(token.as_bytes()));
        for worker in self.workers.values() {
            if constant_time_eq::constant_time_eq(
                worker.auth_token.as_bytes(),
                token_hash.as_bytes(),
            ) && worker.status != "REVOKED"
            {
                return Some(worker.worker_id.clone());
            }
        }
        None
    }

    pub fn register(&mut self, payload: &Value) -> Result<WorkerNode, String> {
        self.register_with_token(payload)
            .map(|(worker, _token)| worker)
    }

    pub fn register_with_token(&mut self, payload: &Value) -> Result<(WorkerNode, String), String> {
        let worker_id = payload
            .get("worker_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("worker_{}", uuid::Uuid::new_v4()));

        let name = payload
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("Dynamic Worker")
            .to_string();

        let provider = payload
            .get("provider")
            .and_then(|v| v.as_str())
            .unwrap_or("generic_container")
            .to_string();

        let endpoint = payload
            .get("endpoint")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let capabilities = payload
            .get("capabilities")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_else(|| vec!["chat".to_string(), "inference".to_string()]);

        let model_sha256 = payload
            .get("model_sha256")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let now = current_timestamp();
        let auth_token = format!(
            "tok_{:032x}{:032x}",
            rand::random::<u128>(),
            rand::random::<u128>()
        );
        let auth_token_hash = hex::encode(Sha256::digest(auth_token.as_bytes()));

        // Model SHA verification against canonical invariant
        let (status, quarantine_reason) = if model_sha256 != CANONICAL_MODEL_SHA256 {
            (
                "QUARANTINED".to_string(),
                Some(format!(
                    "Model SHA-256 '{}' does not match canonical invariant '{}'",
                    model_sha256, CANONICAL_MODEL_SHA256
                )),
            )
        } else {
            // Registration data is only a claim. A worker must prove possession
            // of its issued token through the authenticated heartbeat path.
            ("DISCOVERED".to_string(), None)
        };

        let node = WorkerNode {
            worker_id: worker_id.clone(),
            name,
            provider,
            endpoint,
            capabilities,
            model_sha256,
            status,
            quarantine_reason,
            registered_at: now,
            last_heartbeat: now,
            active_jobs: 0,
            max_concurrency: payload
                .get("max_concurrency")
                .and_then(|v| v.as_u64())
                .unwrap_or(4) as usize,
            latency_ms: payload
                .get("latency_ms")
                .and_then(|v| v.as_f64())
                .unwrap_or(50.0),
            error_count: 0,
            auth_token: auth_token_hash,
        };

        self.workers.insert(worker_id, node.clone());
        self.save();
        Ok((node, auth_token))
    }

    pub fn quarantine(&mut self, worker_id: &str, reason: &str) -> Result<WorkerNode, String> {
        let result_node = {
            let node = self
                .workers
                .get_mut(worker_id)
                .ok_or_else(|| format!("Worker '{}' not found", worker_id))?;
            node.status = "QUARANTINED".to_string();
            node.quarantine_reason = Some(reason.to_string());
            node.clone()
        };
        self.save();
        Ok(result_node)
    }

    pub fn revoke(&mut self, worker_id: &str, reason: &str) -> Result<WorkerNode, String> {
        let result_node = {
            let node = self
                .workers
                .get_mut(worker_id)
                .ok_or_else(|| format!("Worker '{}' not found", worker_id))?;
            node.status = "REVOKED".to_string();
            node.quarantine_reason = Some(reason.to_string());
            node.clone()
        };
        self.save();
        Ok(result_node)
    }

    pub fn heartbeat(&mut self, worker_id: &str, latency_ms: Option<f64>) -> Result<(), String> {
        {
            let node = self
                .workers
                .get_mut(worker_id)
                .ok_or_else(|| format!("Worker '{}' not found", worker_id))?;
            node.last_heartbeat = current_timestamp();
            if let Some(lat) = latency_ms {
                node.latency_ms = (node.latency_ms * 0.8) + (lat * 0.2);
            }
            if node.status == "DISCOVERED"
                || node.status == "DEGRADED"
                || node.status == "UNAVAILABLE"
            {
                node.status = "READY".to_string();
            }
        }
        self.save();
        Ok(())
    }

    pub fn list_workers(&self) -> Vec<WorkerNode> {
        self.workers.values().cloned().collect()
    }

    pub fn get_worker(&self, worker_id: &str) -> Option<WorkerNode> {
        self.workers.get(worker_id).cloned()
    }

    pub fn select_best_worker(&self, required_cap: &str) -> Option<WorkerNode> {
        let mut candidates: Vec<&WorkerNode> = self
            .workers
            .values()
            .filter(|w| {
                w.status == "READY"
                    && (required_cap.is_empty() || w.capabilities.iter().any(|c| c == required_cap))
            })
            .collect();

        // Sort by multi-factor score: active_jobs (asc), error_count (asc), latency_ms (asc)
        candidates.sort_by(|a, b| {
            a.active_jobs
                .cmp(&b.active_jobs)
                .then(a.error_count.cmp(&b.error_count))
                .then(
                    a.latency_ms
                        .partial_cmp(&b.latency_ms)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });

        candidates.first().map(|w| (*w).clone())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Distributed Job Engine
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobChunkRecord {
    pub chunk_id: String,
    pub chunk_index: usize,
    pub input_data: Value,
    pub assigned_worker_id: Option<String>,
    pub lease_expires_at: u64,
    pub status: String, // PENDING, RUNNING, COMPLETED, FAILED
    pub result: Option<Value>,
    pub retry_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub job_id: String,
    pub task_type: String,
    pub idempotency_key: Option<String>,
    pub status: String, // PENDING, RUNNING, COMPLETED, FAILED
    pub created_at: u64,
    pub completed_at: Option<u64>,
    pub chunks: Vec<JobChunkRecord>,
    pub results: Vec<Value>,
    pub error: Option<String>,
    pub target_provider: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedJobEngineStore {
    pub version: String,
    pub updated_at: u64,
    pub jobs: Vec<JobRecord>,
    pub idempotency_index: HashMap<String, String>,
}

pub struct DistributedJobEngine {
    jobs: HashMap<String, JobRecord>,
    idempotency_index: HashMap<String, String>, // idempotency_key -> job_id
    persistence_path: Option<String>,
}

impl Default for DistributedJobEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DistributedJobEngine {
    pub fn new() -> Self {
        Self::load_or_new(None)
    }

    pub fn load_or_new(persistence_path: Option<String>) -> Self {
        let mut engine = Self {
            jobs: HashMap::new(),
            idempotency_index: HashMap::new(),
            persistence_path: persistence_path.clone(),
        };

        if let Some(ref p) = persistence_path {
            let path = Path::new(p);
            if path.exists() {
                if let Ok(content) = fs::read_to_string(path) {
                    if let Ok(store) = serde_json::from_str::<DistributedJobEngineStore>(&content) {
                        let now = current_timestamp();
                        for mut job in store.jobs {
                            for chunk in job.chunks.iter_mut() {
                                if chunk.status == "RUNNING" && chunk.lease_expires_at < now {
                                    chunk.status = "PENDING".to_string();
                                    chunk.assigned_worker_id = None;
                                }
                            }
                            engine.jobs.insert(job.job_id.clone(), job);
                        }
                        engine.idempotency_index = store.idempotency_index;
                    }
                }
            }
        }

        engine
    }

    pub fn save(&self) {
        if let Some(ref p) = self.persistence_path {
            let path = Path::new(p);
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            let jobs_vec: Vec<JobRecord> = self.jobs.values().cloned().collect();
            let store = DistributedJobEngineStore {
                version: "1.0.0".to_string(),
                updated_at: current_timestamp(),
                jobs: jobs_vec,
                idempotency_index: self.idempotency_index.clone(),
            };
            if let Ok(data) = serde_json::to_string_pretty(&store) {
                let _ = fs::write(path, data);
            }
        }
    }

    pub fn submit_job(
        &mut self,
        task_type: &str,
        items: Vec<Value>,
        chunk_size: usize,
        idempotency_key: Option<&str>,
        target_provider: Option<&str>,
    ) -> JobRecord {
        // Idempotency deduplication
        if let Some(key) = idempotency_key {
            if let Some(existing_id) = self.idempotency_index.get(key) {
                if let Some(existing_job) = self.jobs.get(existing_id) {
                    return existing_job.clone();
                }
            }
        }

        let now = current_timestamp();
        let job_id = format!("job_{}_{}", task_type, now);

        // Intelligent sharding: do not over-shard small workloads
        let effective_chunk_size = if items.len() <= 2 {
            items.len().max(1)
        } else if chunk_size == 0 {
            5
        } else {
            chunk_size
        };

        let mut chunks = Vec::new();
        let mut chunk_idx = 0;
        let mut i = 0;
        while i < items.len() {
            let end = (i + effective_chunk_size).min(items.len());
            let chunk_items = items[i..end].to_vec();
            chunks.push(JobChunkRecord {
                chunk_id: format!("{}_chunk_{}", job_id, chunk_idx),
                chunk_index: chunk_idx,
                input_data: json!(chunk_items),
                assigned_worker_id: None,
                lease_expires_at: 0,
                status: "PENDING".to_string(),
                result: None,
                retry_count: 0,
            });
            chunk_idx += 1;
            i = end;
        }

        if chunks.is_empty() {
            chunks.push(JobChunkRecord {
                chunk_id: format!("{}_chunk_0", job_id),
                chunk_index: 0,
                input_data: json!([]),
                assigned_worker_id: None,
                lease_expires_at: 0,
                status: "PENDING".to_string(),
                result: None,
                retry_count: 0,
            });
        }

        let record = JobRecord {
            job_id: job_id.clone(),
            task_type: task_type.to_string(),
            idempotency_key: idempotency_key.map(|k| k.to_string()),
            status: "PENDING".to_string(),
            created_at: now,
            completed_at: None,
            chunks,
            results: Vec::new(),
            error: None,
            target_provider: target_provider.map(|p| p.to_string()),
        };

        if let Some(key) = idempotency_key {
            self.idempotency_index
                .insert(key.to_string(), job_id.clone());
        }

        self.jobs.insert(job_id, record.clone());
        self.save();
        record
    }

    pub fn get_job(&self, job_id: &str) -> Option<JobRecord> {
        self.jobs.get(job_id).cloned()
    }

    pub fn list_jobs(&self) -> Vec<JobRecord> {
        self.jobs.values().cloned().collect()
    }

    pub fn assign_pending_chunks(&mut self, registry: &WorkerRegistry) {
        let now = current_timestamp();
        for job in self.jobs.values_mut() {
            if job.status == "COMPLETED" || job.status == "FAILED" {
                continue;
            }
            let mut any_running = false;
            for chunk in job.chunks.iter_mut() {
                // Lease check
                if chunk.status == "RUNNING" && chunk.lease_expires_at < now {
                    chunk.status = "PENDING".to_string();
                    chunk.assigned_worker_id = None;
                }

                if chunk.status == "PENDING" {
                    if let Some(worker) = registry.select_best_worker(&job.task_type) {
                        chunk.assigned_worker_id = Some(worker.worker_id.clone());
                        chunk.status = "RUNNING".to_string();
                        chunk.lease_expires_at = now + 60; // 60-second lease
                        any_running = true;
                    }
                } else if chunk.status == "RUNNING" {
                    any_running = true;
                }
            }
            if any_running && job.status == "PENDING" {
                job.status = "RUNNING".to_string();
            }
        }
        self.save();
    }

    pub fn complete_chunk(
        &mut self,
        job_id: &str,
        chunk_id: &str,
        result: Value,
    ) -> Result<JobRecord, String> {
        let result_job = {
            let job = self
                .jobs
                .get_mut(job_id)
                .ok_or_else(|| format!("Job '{}' not found", job_id))?;

            let mut found = false;
            for chunk in job.chunks.iter_mut() {
                if chunk.chunk_id == chunk_id {
                    chunk.status = "COMPLETED".to_string();
                    chunk.result = Some(result.clone());
                    found = true;
                    break;
                }
            }

            if !found {
                return Err(format!(
                    "Chunk '{}' not found in job '{}'",
                    chunk_id, job_id
                ));
            }

            // Check if all chunks completed
            let all_done = job.chunks.iter().all(|c| c.status == "COMPLETED");
            if all_done {
                job.status = "COMPLETED".to_string();
                job.completed_at = Some(current_timestamp());
                // Aggregate results in chunk index order
                let mut sorted_chunks = job.chunks.clone();
                sorted_chunks.sort_by_key(|c| c.chunk_index);
                job.results = sorted_chunks.into_iter().filter_map(|c| c.result).collect();
            }
            job.clone()
        };

        self.save();
        Ok(result_job)
    }

    pub fn fail_chunk(
        &mut self,
        job_id: &str,
        chunk_id: &str,
        error: &str,
    ) -> Result<JobRecord, String> {
        let result_job = {
            let job = self
                .jobs
                .get_mut(job_id)
                .ok_or_else(|| format!("Job '{}' not found", job_id))?;

            let mut found = false;
            for chunk in job.chunks.iter_mut() {
                if chunk.chunk_id == chunk_id {
                    chunk.retry_count += 1;
                    if chunk.retry_count < 3 {
                        // Retry by returning to PENDING
                        chunk.status = "PENDING".to_string();
                        chunk.assigned_worker_id = None;
                        chunk.lease_expires_at = 0;
                    } else {
                        chunk.status = "FAILED".to_string();
                        job.status = "FAILED".to_string();
                        job.error = Some(format!(
                            "Chunk '{}' failed after 3 retries: {}",
                            chunk_id, error
                        ));
                    }
                    found = true;
                    break;
                }
            }

            if !found {
                return Err(format!(
                    "Chunk '{}' not found in job '{}'",
                    chunk_id, job_id
                ));
            }
            job.clone()
        };

        self.save();
        Ok(result_job)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Provider Adapter Manager
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderStatus {
    pub name: String,
    pub adapter_type: String,
    pub is_available: bool,
    pub capability_score: u32,
    pub active_nodes: usize,
    pub description: String,
}

pub struct ProviderManager {
    providers: HashMap<String, ProviderStatus>,
}

impl Default for ProviderManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderManager {
    pub fn new() -> Self {
        let mut m = HashMap::new();
        m.insert(
            "cloudflare".to_string(),
            ProviderStatus {
                name: "cloudflare".to_string(),
                adapter_type: "UnconfiguredProvider".to_string(),
                is_available: false,
                capability_score: 85,
                active_nodes: 0,
                description: "External model and compute deployment is disabled; register a verified TARA worker instead".to_string(),
            },
        );
        m.insert(
            "modelscope".to_string(),
            ProviderStatus {
                name: "modelscope".to_string(),
                adapter_type: "UnconfiguredProvider".to_string(),
                is_available: false,
                capability_score: 90,
                active_nodes: 0,
                description: "External inference-provider connection is disabled".to_string(),
            },
        );
        m.insert(
            "huggingface".to_string(),
            ProviderStatus {
                name: "huggingface".to_string(),
                adapter_type: "UnconfiguredProvider".to_string(),
                is_available: false,
                capability_score: 88,
                active_nodes: 0,
                description: "External inference-provider connection is disabled".to_string(),
            },
        );
        m.insert(
            "render".to_string(),
            ProviderStatus {
                name: "render".to_string(),
                adapter_type: "UnconfiguredProvider".to_string(),
                is_available: false,
                capability_score: 80,
                active_nodes: 0,
                description: "Cloud deployment adapter is not configured".to_string(),
            },
        );
        m.insert(
            "generic_container".to_string(),
            ProviderStatus {
                name: "generic_container".to_string(),
                adapter_type: "UnconfiguredProvider".to_string(),
                is_available: false,
                capability_score: 92,
                active_nodes: 0,
                description: "Container provisioning is not configured; register a verified TARA worker instead".to_string(),
            },
        );
        m.insert(
            "local_device".to_string(),
            ProviderStatus {
                name: "local_device".to_string(),
                adapter_type: "LocalDeviceAdapter".to_string(),
                is_available: false,
                capability_score: 95,
                active_nodes: 0,
                description:
                    "Local workers are registered through the authenticated worker registry"
                        .to_string(),
            },
        );
        Self { providers: m }
    }

    pub fn list_providers(&self) -> Vec<ProviderStatus> {
        self.providers.values().cloned().collect()
    }

    pub fn get_provider(&self, name: &str) -> Option<ProviderStatus> {
        self.providers.get(name).cloned()
    }

    pub fn deploy(&mut self, provider_name: &str, _config: &Value) -> Result<Value, String> {
        let provider = self
            .providers
            .get(provider_name)
            .ok_or_else(|| format!("Unknown provider '{}'", provider_name))?;
        Err(format!(
            "Provider '{}' ({}) has no configured deployment adapter; no deployment was created",
            provider_name, provider.adapter_type
        ))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Canonical Manifest Exporter
// ──────────────────────────────────────────────────────────────────────────────

pub struct CanonicalManifestManager {
    repo_root: String,
}

impl CanonicalManifestManager {
    pub fn new(repo_root: &str) -> Self {
        Self {
            repo_root: repo_root.to_string(),
        }
    }

    pub fn get_manifest(&self) -> Value {
        fn compute_path_sha256(path: &Path) -> String {
            if let Ok(bytes) = fs::read(path) {
                let mut hasher = Sha256::new();
                hasher.update(&bytes);
                format!("{:x}", hasher.finalize())
            } else {
                String::new()
            }
        }

        let tok_path = Path::new(&self.repo_root).join("storage/models/tara/tokenizer.json");
        let schema_path = Path::new(&self.repo_root).join("TARA/CONTRACTS/v1/schemas.json");
        let contract_path = Path::new(&self.repo_root).join("rust/tara_server/src/contract.rs");

        let tok_sha = compute_path_sha256(&tok_path);
        let schema_sha = compute_path_sha256(&schema_path);
        let contract_sha = compute_path_sha256(&contract_path);

        let manifest_path = format!("{}/TARA/MANIFEST/canonical_manifest.json", self.repo_root);
        if Path::new(&manifest_path).exists() {
            if let Ok(content) = fs::read_to_string(&manifest_path) {
                if let Ok(mut parsed) = serde_json::from_str::<Value>(&content) {
                    if let Some(artifacts) = parsed.get_mut("artifacts").and_then(Value::as_object_mut) {
                        if let Some(tok) = artifacts.get_mut("tokenizer").and_then(Value::as_object_mut) {
                            tok.insert("sha256".to_string(), json!(tok_sha));
                        }
                        if let Some(sc) = artifacts.get_mut("contract_schemas").and_then(Value::as_object_mut) {
                            sc.insert("sha256".to_string(), json!(schema_sha));
                        }
                        if let Some(ct) = artifacts.get_mut("rust_contracts").and_then(Value::as_object_mut) {
                            ct.insert("sha256".to_string(), json!(contract_sha));
                        }
                    }
                    return parsed;
                }
            }
        }

        // Authoritative fallback manifest matching canonical production invariant
        json!({
            "manifest_name": "TARA Canonical Artifact Manifest",
            "manifest_version": "1.0.0",
            "model_identity": "TARA",
            "model_version": "none",
            "protocol_version": "1.0.0",
            "canonical_model_sha256": CANONICAL_MODEL_SHA256,
            "canonical_param_count": CANONICAL_PARAM_COUNT,
            "artifacts": {
                "tokenizer": {
                    "path": "storage/models/tara/tokenizer.json",
                    "sha256": tok_sha,
                    "required": true,
                    "format": "json"
                },
                "contract_schemas": {
                    "path": "TARA/CONTRACTS/v1/schemas.json",
                    "sha256": schema_sha,
                    "required": true,
                    "format": "json_schema"
                },
                "rust_contracts": {
                    "path": "rust/tara_server/src/contract.rs",
                    "sha256": contract_sha,
                    "required": true,
                    "format": "rust"
                }
            },
            "runtime_invariants": {
                "allow_unverified_model": false,
                "allow_unauthenticated_workers": false,
                "enforce_zero_trust": true,
                "creator_authority_id": "ROOT_OPERATOR",
                "creator_display_name": "OPERATOR_ROOT"
            }
        })
    }

    pub fn verify_model_sha(&self, sha256: &str) -> bool {
        sha256 == CANONICAL_MODEL_SHA256
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ControlPlane Top-Level Aggregate
// ──────────────────────────────────────────────────────────────────────────────

pub struct ControlPlane {
    pub workers: RwLock<WorkerRegistry>,
    pub jobs: RwLock<DistributedJobEngine>,
    pub providers: RwLock<ProviderManager>,
    pub manifest: CanonicalManifestManager,
    pub repo_root: String,
}

impl ControlPlane {
    pub fn new(repo_root: &str) -> Self {
        let storage_dir = Path::new(repo_root)
            .join("storage")
            .join("persistence")
            .join("control_plane");
        let _ = fs::create_dir_all(&storage_dir);
        let workers_path = storage_dir
            .join("workers.json")
            .to_string_lossy()
            .to_string();
        let jobs_path = storage_dir.join("jobs.json").to_string_lossy().to_string();

        Self {
            workers: RwLock::new(WorkerRegistry::load_or_new(Some(workers_path))),
            jobs: RwLock::new(DistributedJobEngine::load_or_new(Some(jobs_path))),
            providers: RwLock::new(ProviderManager::new()),
            manifest: CanonicalManifestManager::new(repo_root),
            repo_root: repo_root.to_string(),
        }
    }

    pub fn get_status(&self) -> Value {
        let workers = self.workers.read().unwrap();
        let jobs = self.jobs.read().unwrap();
        let providers = self.providers.read().unwrap();

        let active_workers = workers
            .list_workers()
            .into_iter()
            .filter(|w| w.status == "READY")
            .count();
        let quarantined_workers = workers
            .list_workers()
            .into_iter()
            .filter(|w| w.status == "QUARANTINED")
            .count();

        let public_tara_url = std::env::var("PUBLIC_TARA_URL")
            .unwrap_or_else(|_| "https://gateway.tara.local".to_string());
        let control_plane_url = std::env::var("TARA_CONTROL_PLANE_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8765".to_string());
        let safe_mode_active = std::env::var("TARA_SAFE_MODE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        json!({
            "status": "HEALTHY",
            "control_plane": "TARA Dynamic Compute Control Plane",
            "version": "2.0.0",
            "public_tara_url": public_tara_url,
            "control_plane_url": control_plane_url,
            "user_compute_cost": 0.0,
            "enforce_zero_user_cost": true,
            "safe_mode_active": safe_mode_active,
            "canonical_model_sha256": CANONICAL_MODEL_SHA256,
            "canonical_param_count": CANONICAL_PARAM_COUNT,
            "workers": {
                "total": workers.list_workers().len(),
                "ready": active_workers,
                "quarantined": quarantined_workers
            },
            "jobs": {
                "total": jobs.list_jobs().len(),
                "pending": jobs.list_jobs().iter().filter(|j| j.status == "PENDING").count(),
                "running": jobs.list_jobs().iter().filter(|j| j.status == "RUNNING").count(),
                "completed": jobs.list_jobs().iter().filter(|j| j.status == "COMPLETED").count()
            },
            "providers_count": providers.list_providers().len(),
            "timestamp": current_timestamp()
        })
    }
}
