//! # TARA Core Dataset Engine (100% Pure Native Rust)
//!
//! Location: `rust/tara_training_system/dataset_engine/mod.rs`
//!
//! Mandatory Directives:
//! 1. Auto-Detect: Monitors and discovers datasets in `tara_dataset/manual/` and `tara_dataset/autonomous/`.
//! 2. Auto-Register: Computes runtime dynamic SHA-256, record count, byte size, provenance metadata.
//! 3. Auto-Classify: Classifies datasets into `NeuralModel`, `WorldModel`, or `Both`.
//! 4. Auto-Route: Divides dataset into internal `NeuralRecord` and `WorldModelRecord` representations.
//! 5. Manual / Autonomous Separation: Strictly preserves isolation between manual and autonomous sources.
//! 6. Continuous Operation: Incremental discovery and updates without restarting or state loss.
//! 7. Zero Unnecessary Physical Folders: No `neural/`, `world/`, or `shared/` physical directories.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Source origin of the dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetSource {
    Manual,
    Autonomous,
}

impl DatasetSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            DatasetSource::Manual => "manual",
            DatasetSource::Autonomous => "autonomous",
        }
    }
}

/// Logical classification target for training models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetModel {
    NeuralModel,
    WorldModel,
    Both,
}

impl TargetModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            TargetModel::NeuralModel => "neural_model",
            TargetModel::WorldModel => "world_model",
            TargetModel::Both => "both",
        }
    }
}

/// Real metadata and provenance for an indexed dataset.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetMetadata {
    pub dataset_id: String,
    pub name: String,
    pub source: DatasetSource,
    pub target_model: TargetModel,
    pub rel_path: String,
    pub size_bytes: u64,
    pub record_count: usize,
    pub dynamic_sha256: String,
    pub license: String,
    pub detected_at_utc: String,
    pub neural_records_count: usize,
    pub world_records_count: usize,
    pub transition_records_count: usize,
}

/// Internal data representation for Neural Model training.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeuralRecord {
    pub id: String,
    pub prompt: String,
    pub completion: String,
    pub tokens_estimate: usize,
    pub difficulty_score: f32,
    pub domain: String,
    pub source_dataset_id: String,
}

/// Internal data representation for World Model training (facts, relations, state graphs).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorldModelRecord {
    pub id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub context: String,
    pub confidence: f32,
    pub source_dataset_id: String,
}

/// Internal data representation for dynamic World Model state transitions: (s, a, s', r).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StateTransitionRecord {
    pub id: String,
    pub episode_id: String,
    pub step_index: usize,
    pub state_observation: String,
    pub action: String,
    pub next_state_observation: String,
    pub reward: f32,
    pub done: bool,
    pub domain: String,
    pub confidence: f32,
    pub source_dataset_id: String,
}

/// Validates that a state transition record has non-empty identifiers, observations, action,
/// finite reward, and confidence bounded in [0.0, 1.0].
pub fn validate_state_transition(rec: &StateTransitionRecord) -> Result<(), String> {
    if rec.id.trim().is_empty() {
        return Err("StateTransitionRecord id cannot be empty".to_string());
    }
    if rec.episode_id.trim().is_empty() {
        return Err("StateTransitionRecord episode_id cannot be empty".to_string());
    }
    if rec.state_observation.trim().is_empty() {
        return Err("StateTransitionRecord state_observation cannot be empty".to_string());
    }
    if rec.action.trim().is_empty() {
        return Err("StateTransitionRecord action cannot be empty".to_string());
    }
    if rec.next_state_observation.trim().is_empty() {
        return Err("StateTransitionRecord next_state_observation cannot be empty".to_string());
    }
    if !rec.reward.is_finite() {
        return Err("StateTransitionRecord reward must be finite".to_string());
    }
    if !rec.confidence.is_finite() || rec.confidence < 0.0 || rec.confidence > 1.0 {
        return Err(format!(
            "StateTransitionRecord confidence must be in [0.0, 1.0], got {}",
            rec.confidence
        ));
    }
    Ok(())
}

/// Fully parsed and logically routed dataset in memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutedDataset {
    pub metadata: DatasetMetadata,
    pub neural_records: Vec<NeuralRecord>,
    pub world_records: Vec<WorldModelRecord>,
    pub transition_records: Vec<StateTransitionRecord>,
}

/// Configurable epistemic weighting parameters for dynamic ontological confidence computation (parameter-driven, extensible).
///
/// Implements a 3-factor epistemic partition of unity:
/// $\text{Confidence} = \alpha_0 + \alpha_{\text{topo}} \cdot \rho_{\text{degree}} + \alpha_{\text{impl}} \cdot \rho_{\text{struct}}$
/// where:
/// - $\alpha_0$ (`base_prior`): Foundational prior belief assigned to any structurally validated entity (default 0.60).
/// - $\alpha_{\text{topo}}$ (`topological_weight`): Confirmation gain proportional to relative node degree density in CAPABILITY_GRAPH (default 0.24).
/// - $\alpha_{\text{impl}}$ (`implementation_weight`): Grounding gain proportional to physical codebase realization (default 0.14).
/// - $\alpha_0 + \alpha_{\text{topo}} + \alpha_{\text{impl}} \le \text{max\_confidence\_ceiling} \le 1.00$ mathematically guarantees bounded confidence in $(0, 1]$.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TopologicalConfidenceConfig {
    /// Baseline prior for structurally validated entities.
    pub base_prior: f64,
    /// Weight for relative topological node degree connectivity.
    pub topological_weight: f64,
    /// Weight for verified codebase struct/trait realization.
    pub implementation_weight: f64,
    /// Upper ceiling to prevent premature absolute certainty.
    pub max_confidence_ceiling: f64,
}

impl Default for TopologicalConfidenceConfig {
    fn default() -> Self {
        Self {
            base_prior: 0.60,
            topological_weight: 0.24,
            implementation_weight: 0.14,
            max_confidence_ceiling: 0.98,
        }
    }
}

impl TopologicalConfidenceConfig {
    /// Enforces mathematical invariants:
    /// - base_prior >= 0.0
    /// - topological_weight >= 0.0
    /// - implementation_weight >= 0.0
    /// - 0.0 <= max_confidence_ceiling <= 1.0
    /// - base_prior + topological_weight + implementation_weight <= max_confidence_ceiling + 1e-6
    pub fn validate(&self) -> Result<(), String> {
        if !self.base_prior.is_finite() || self.base_prior < 0.0 {
            return Err(format!(
                "base_prior must be non-negative finite, got {}",
                self.base_prior
            ));
        }
        if !self.topological_weight.is_finite() || self.topological_weight < 0.0 {
            return Err(format!(
                "topological_weight must be non-negative finite, got {}",
                self.topological_weight
            ));
        }
        if !self.implementation_weight.is_finite() || self.implementation_weight < 0.0 {
            return Err(format!(
                "implementation_weight must be non-negative finite, got {}",
                self.implementation_weight
            ));
        }
        if !self.max_confidence_ceiling.is_finite()
            || self.max_confidence_ceiling < 0.0
            || self.max_confidence_ceiling > 1.0
        {
            return Err(format!(
                "max_confidence_ceiling must be in [0.0, 1.0], got {}",
                self.max_confidence_ceiling
            ));
        }
        let total_weight = self.base_prior + self.topological_weight + self.implementation_weight;
        if total_weight > self.max_confidence_ceiling + 1e-6 {
            return Err(format!(
                "Sum of epistemic weights ({:.4}) cannot exceed max_confidence_ceiling ({:.4})",
                total_weight, self.max_confidence_ceiling
            ));
        }
        Ok(())
    }

    /// Constructs a custom parameter-driven confidence configuration with validation.
    pub fn new(
        base_prior: f64,
        topological_weight: f64,
        implementation_weight: f64,
        max_confidence_ceiling: f64,
    ) -> Result<Self, String> {
        let cfg = Self {
            base_prior,
            topological_weight,
            implementation_weight,
            max_confidence_ceiling,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Dynamically computes normalized confidence from runtime topological density metrics.
    /// Clamps input ratios to [0.0, 1.0] and raw result to max_confidence_ceiling.
    pub fn compute(&self, incident_ratio: f64, struct_ratio: f64) -> f64 {
        let incident_clamped = incident_ratio.clamp(0.0, 1.0);
        let struct_clamped = struct_ratio.clamp(0.0, 1.0);
        let raw = self.base_prior
            + self.topological_weight * incident_clamped
            + self.implementation_weight * struct_clamped;
        let bounded = raw.min(self.max_confidence_ceiling).clamp(0.0, 1.0);
        ((bounded * 100.0).round()) / 100.0
    }
}

/// Configuration for Dataset Engine.
#[derive(Debug, Clone)]
pub struct DatasetEngineConfig {
    pub dataset_root: PathBuf,
    pub manual_dir: PathBuf,
    pub autonomous_dir: PathBuf,
    pub io_buffer_size: usize,
    pub supported_extensions: Vec<String>,
    pub confidence_config: TopologicalConfidenceConfig,
}

impl DatasetEngineConfig {
    /// Validates configuration and internal confidence invariants.
    pub fn validate(&self) -> Result<(), String> {
        self.confidence_config.validate()
    }
}

impl Default for DatasetEngineConfig {
    fn default() -> Self {
        let ws = find_workspace_root().unwrap_or_else(|_| PathBuf::from("."));
        let root = ws
            .join("rust")
            .join("tara_training_system")
            .join("tara_dataset");
        let manual = root.join("manual");
        let autonomous = root.join("autonomous");

        Self {
            dataset_root: root,
            manual_dir: manual,
            autonomous_dir: autonomous,
            io_buffer_size: 64 * 1024,
            supported_extensions: vec![
                "jsonl".to_string(),
                "json".to_string(),
                "txt".to_string(),
                "csv".to_string(),
                "tsv".to_string(),
            ],
            confidence_config: TopologicalConfidenceConfig::default(),
        }
    }
}

/// Report summarizing synchronization result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSyncReport {
    pub total_manual_datasets: usize,
    pub total_autonomous_datasets: usize,
    pub newly_detected: usize,
    pub updated: usize,
    pub total_neural_records: usize,
    pub total_world_records: usize,
    pub total_transition_records: usize,
    pub duration_ms: u64,
}

/// Main Dataset Engine.
pub struct DatasetEngine {
    pub config: DatasetEngineConfig,
    manual_registry: Arc<RwLock<BTreeMap<String, DatasetMetadata>>>,
    autonomous_registry: Arc<RwLock<BTreeMap<String, DatasetMetadata>>>,
    routed_store: Arc<RwLock<BTreeMap<String, RoutedDataset>>>,
}

impl DatasetEngine {
    pub fn new(config: DatasetEngineConfig) -> Result<Self, Box<dyn std::error::Error>> {
        config
            .validate()
            .map_err(|e| format!("Invalid DatasetEngineConfig: {}", e))?;

        // Enforce physical folder creation only for manual/ and autonomous/
        // Zero physical neural/, world/, or shared/ folders created
        fs::create_dir_all(&config.manual_dir)?;
        fs::create_dir_all(&config.autonomous_dir)?;

        Ok(Self {
            config,
            manual_registry: Arc::new(RwLock::new(BTreeMap::new())),
            autonomous_registry: Arc::new(RwLock::new(BTreeMap::new())),
            routed_store: Arc::new(RwLock::new(BTreeMap::new())),
        })
    }

    /// Auto-detect, auto-register, auto-classify, and auto-route all datasets across
    /// `manual/` and `autonomous/` folders without physical folder duplication.
    pub fn scan_and_sync(&self) -> Result<DatasetSyncReport, Box<dyn std::error::Error>> {
        let start = std::time::Instant::now();
        let mut newly_detected = 0;
        let mut updated = 0;

        // 1. Scan Manual Datasets
        let manual_files = self.discover_dataset_files(&self.config.manual_dir)?;
        for file_path in manual_files {
            let (is_new, is_up) = self.process_dataset_file(&file_path, DatasetSource::Manual)?;
            if is_new {
                newly_detected += 1;
            } else if is_up {
                updated += 1;
            }
        }

        // 2. Scan Autonomous Datasets
        let auto_files = self.discover_dataset_files(&self.config.autonomous_dir)?;
        for file_path in auto_files {
            let (is_new, is_up) =
                self.process_dataset_file(&file_path, DatasetSource::Autonomous)?;
            if is_new {
                newly_detected += 1;
            } else if is_up {
                updated += 1;
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        let manual_count = self
            .manual_registry
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .len();
        let auto_count = self
            .autonomous_registry
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .len();

        let mut total_neural = 0;
        let mut total_world = 0;
        let mut total_transition = 0;
        {
            let store = self.routed_store.read().unwrap_or_else(|p| p.into_inner());
            for routed in store.values() {
                total_neural += routed.neural_records.len();
                total_world += routed.world_records.len();
                total_transition += routed.transition_records.len();
            }
        }

        Ok(DatasetSyncReport {
            total_manual_datasets: manual_count,
            total_autonomous_datasets: auto_count,
            newly_detected,
            updated,
            total_neural_records: total_neural,
            total_world_records: total_world,
            total_transition_records: total_transition,
            duration_ms,
        })
    }

    /// Processes a single dataset file: registers metadata, classifies target model,
    /// routes records into internal representations.
    pub fn process_dataset_file(
        &self,
        path: &Path,
        source: DatasetSource,
    ) -> Result<(bool, bool), Box<dyn std::error::Error>> {
        let dataset_id = self.derive_dataset_id(path, source);
        let dynamic_sha = compute_dynamic_sha256(path, self.config.io_buffer_size)?;
        let size_bytes = path.metadata()?.len();

        // Check if existing and unchanged
        {
            let reg = match source {
                DatasetSource::Manual => self.manual_registry.read().unwrap(),
                DatasetSource::Autonomous => self.autonomous_registry.read().unwrap(),
            };
            if let Some(existing) = reg.get(&dataset_id) {
                if existing.dynamic_sha256 == dynamic_sha && existing.size_bytes == size_bytes {
                    return Ok((false, false)); // No change
                }
            }
        }

        // Parse records from file
        let (raw_records, license_str) = self.parse_raw_records(path)?;
        let record_count = raw_records.len();

        // Auto-classify target model
        let target_model = self.classify_dataset_records(&raw_records);

        // Auto-route records into internal representations
        let (neural_records, world_records, transition_records) =
            self.route_records(&dataset_id, &raw_records);

        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&dataset_id)
            .to_string();

        let rel_path = normalize_rel_path(path);

        let metadata = DatasetMetadata {
            dataset_id: dataset_id.clone(),
            name: file_name,
            source,
            target_model,
            rel_path,
            size_bytes,
            record_count,
            dynamic_sha256: dynamic_sha,
            license: license_str,
            detected_at_utc: now_utc_iso(),
            neural_records_count: neural_records.len(),
            world_records_count: world_records.len(),
            transition_records_count: transition_records.len(),
        };

        let routed = RoutedDataset {
            metadata: metadata.clone(),
            neural_records,
            world_records,
            transition_records,
        };

        // Store into routed store
        self.routed_store
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .insert(dataset_id.clone(), routed);

        // Store into appropriate isolated registry
        let (is_new, is_updated) = match source {
            DatasetSource::Manual => {
                let mut reg = self
                    .manual_registry
                    .write()
                    .unwrap_or_else(|p| p.into_inner());
                let was_present = reg.insert(dataset_id, metadata).is_some();
                (!was_present, was_present)
            }
            DatasetSource::Autonomous => {
                let mut reg = self
                    .autonomous_registry
                    .write()
                    .unwrap_or_else(|p| p.into_inner());
                let was_present = reg.insert(dataset_id, metadata).is_some();
                (!was_present, was_present)
            }
        };

        Ok((is_new, is_updated))
    }

    /// Auto-classifies a collection of records into NeuralModel, WorldModel, or Both.
    pub fn classify_dataset_records(&self, records: &[Value]) -> TargetModel {
        if records.is_empty() {
            return TargetModel::NeuralModel;
        }

        let mut neural_votes = 0;
        let mut world_votes = 0;

        for rec in records {
            match self.classify_single_record(rec) {
                TargetModel::NeuralModel => neural_votes += 1,
                TargetModel::WorldModel => world_votes += 1,
                TargetModel::Both => {
                    neural_votes += 1;
                    world_votes += 1;
                }
            }
        }

        if neural_votes > 0 && world_votes > 0 {
            // Significant presence of both modalities
            let total = records.len() as f32;
            let neural_ratio = neural_votes as f32 / total;
            let world_ratio = world_votes as f32 / total;

            if neural_ratio >= 0.20 && world_ratio >= 0.20 {
                TargetModel::Both
            } else if neural_ratio >= world_ratio {
                TargetModel::NeuralModel
            } else {
                TargetModel::WorldModel
            }
        } else if world_votes > 0 {
            TargetModel::WorldModel
        } else {
            TargetModel::NeuralModel
        }
    }

    /// Classifies an individual record based on structure and semantic field indicators.
    pub fn classify_single_record(&self, rec: &Value) -> TargetModel {
        if let Some(obj) = rec.as_object() {
            let has_transition_fields = obj.contains_key("state_transition")
                || ((obj.contains_key("state")
                    || obj.contains_key("state_observation")
                    || obj.contains_key("observation"))
                    && (obj.contains_key("action") || obj.contains_key("intervention"))
                    && (obj.contains_key("next_state")
                        || obj.contains_key("next_state_observation")
                        || obj.contains_key("subsequent_state")));

            // Check for explicit World Model graph/relational/transition fields
            let has_world_fields = has_transition_fields
                || obj.contains_key("subject")
                || obj.contains_key("predicate")
                || obj.contains_key("relation")
                || obj.contains_key("triples")
                || obj.contains_key("entities")
                || obj.contains_key("ontology_node")
                || obj.contains_key("world_fact");

            // Check for explicit Neural / Language prompt-completion fields
            let has_neural_fields = obj.contains_key("prompt")
                || obj.contains_key("instruction")
                || obj.contains_key("input")
                || obj.contains_key("output")
                || obj.contains_key("completion")
                || obj.contains_key("text")
                || obj.contains_key("conversation")
                || obj.contains_key("messages");

            if has_world_fields && has_neural_fields {
                TargetModel::Both
            } else if has_world_fields {
                TargetModel::WorldModel
            } else {
                TargetModel::NeuralModel
            }
        } else {
            // Fallback for raw text/scalars
            TargetModel::NeuralModel
        }
    }

    /// Routes raw JSON values into internal Neural, World Model, and State Transition records.
    pub fn route_records(
        &self,
        dataset_id: &str,
        records: &[Value],
    ) -> (
        Vec<NeuralRecord>,
        Vec<WorldModelRecord>,
        Vec<StateTransitionRecord>,
    ) {
        let mut neural_out = Vec::new();
        let mut world_out = Vec::new();
        let mut transition_out = Vec::new();

        for (idx, rec) in records.iter().enumerate() {
            let rec_id = format!("{dataset_id}_rec_{idx:06}");

            // 1. Attempt to extract dynamic state transition (s, a, s', r)
            let is_transition =
                if let Some(trans) = self.extract_transition_record(&rec_id, dataset_id, rec) {
                    if validate_state_transition(&trans).is_ok() {
                        transition_out.push(trans);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };

            // 2. Extract knowledge graph world records if knowledge fields are present,
            // or if classified as WorldModel and not already handled as a state transition
            let has_knowledge_fields = rec.as_object().is_some_and(|obj| {
                obj.contains_key("subject")
                    || obj.contains_key("predicate")
                    || obj.contains_key("relation")
                    || obj.contains_key("triples")
                    || obj.contains_key("entities")
                    || obj.contains_key("ontology_node")
                    || obj.contains_key("world_fact")
            });

            let has_neural_fields = rec.as_object().is_some_and(|obj| {
                obj.contains_key("prompt")
                    || obj.contains_key("instruction")
                    || obj.contains_key("input")
                    || obj.contains_key("output")
                    || obj.contains_key("completion")
                    || obj.contains_key("text")
                    || obj.contains_key("conversation")
                    || obj.contains_key("messages")
            });

            if has_knowledge_fields
                || (!is_transition
                    && !has_neural_fields
                    && self.classify_single_record(rec) == TargetModel::WorldModel)
            {
                world_out.push(self.extract_world_record(&rec_id, dataset_id, rec));
            }

            if has_neural_fields
                || (!is_transition
                    && !has_knowledge_fields
                    && self.classify_single_record(rec) == TargetModel::NeuralModel)
            {
                neural_out.push(self.extract_neural_record(&rec_id, dataset_id, rec));
            }
        }

        (neural_out, world_out, transition_out)
    }

    fn extract_transition_record(
        &self,
        id: &str,
        dataset_id: &str,
        rec: &Value,
    ) -> Option<StateTransitionRecord> {
        let obj = rec.as_object()?;

        // If nested inside "state_transition", inspect inner object, else inspect root obj
        let t_obj = if let Some(nested) = obj.get("state_transition").and_then(Value::as_object) {
            nested
        } else {
            obj
        };

        let state_obs = t_obj
            .get("state_observation")
            .or_else(|| t_obj.get("state"))
            .or_else(|| t_obj.get("observation"))
            .and_then(Value::as_str)?;

        let action = t_obj
            .get("action")
            .or_else(|| t_obj.get("intervention"))
            .and_then(Value::as_str)?;

        let next_state_obs = t_obj
            .get("next_state_observation")
            .or_else(|| t_obj.get("next_state"))
            .or_else(|| t_obj.get("subsequent_state"))
            .and_then(Value::as_str)?;

        let episode_id = t_obj
            .get("episode_id")
            .and_then(Value::as_str)
            .unwrap_or(dataset_id)
            .to_string();

        let step_index = t_obj
            .get("step_index")
            .or_else(|| t_obj.get("step"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;

        let reward = t_obj
            .get("reward")
            .or_else(|| t_obj.get("outcome_reward"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0) as f32;

        let done = t_obj
            .get("done")
            .or_else(|| t_obj.get("terminated"))
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let domain = t_obj
            .get("domain")
            .or_else(|| t_obj.get("environment"))
            .and_then(Value::as_str)
            .unwrap_or("general")
            .to_string();

        let confidence = t_obj
            .get("confidence")
            .and_then(Value::as_f64)
            .map(|f| f.clamp(0.0, 1.0) as f32)
            .unwrap_or(1.0);

        Some(StateTransitionRecord {
            id: id.to_string(),
            episode_id,
            step_index,
            state_observation: state_obs.to_string(),
            action: action.to_string(),
            next_state_observation: next_state_obs.to_string(),
            reward,
            done,
            domain,
            confidence,
            source_dataset_id: dataset_id.to_string(),
        })
    }

    fn extract_neural_record(&self, id: &str, dataset_id: &str, rec: &Value) -> NeuralRecord {
        let (prompt, completion, domain) = if let Some(obj) = rec.as_object() {
            let p = obj
                .get("prompt")
                .or_else(|| obj.get("instruction"))
                .or_else(|| obj.get("input"))
                .or_else(|| obj.get("text"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();

            let c = obj
                .get("completion")
                .or_else(|| obj.get("output"))
                .or_else(|| obj.get("response"))
                .or_else(|| obj.get("target"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();

            let d = obj
                .get("domain")
                .or_else(|| obj.get("category"))
                .or_else(|| obj.get("topic"))
                .and_then(Value::as_str)
                .unwrap_or("general")
                .to_string();

            (p, c, d)
        } else if let Some(s) = rec.as_str() {
            (s.to_string(), String::new(), "general".to_string())
        } else {
            (rec.to_string(), String::new(), "general".to_string())
        };

        let tokens_est = (prompt.len() + completion.len()) / 4;
        let diff = calculate_heuristic_difficulty(&prompt, &completion);

        NeuralRecord {
            id: id.to_string(),
            prompt,
            completion,
            tokens_estimate: tokens_est,
            difficulty_score: diff,
            domain,
            source_dataset_id: dataset_id.to_string(),
        }
    }

    fn extract_world_record(&self, id: &str, dataset_id: &str, rec: &Value) -> WorldModelRecord {
        if let Some(obj) = rec.as_object() {
            let subject = obj
                .get("subject")
                .or_else(|| obj.get("entity"))
                .or_else(|| obj.get("concept"))
                .and_then(Value::as_str)
                .unwrap_or("entity_unspecified")
                .to_string();

            let predicate = obj
                .get("predicate")
                .or_else(|| obj.get("relation"))
                .or_else(|| obj.get("property"))
                .and_then(Value::as_str)
                .unwrap_or("relates_to")
                .to_string();

            let object = obj
                .get("object")
                .or_else(|| obj.get("target"))
                .or_else(|| obj.get("value"))
                .and_then(Value::as_str)
                .unwrap_or("target_unspecified")
                .to_string();

            let context = obj
                .get("context")
                .or_else(|| obj.get("description"))
                .or_else(|| obj.get("evidence"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();

            let confidence = obj
                .get("confidence")
                .and_then(Value::as_f64)
                .map(|f| f.clamp(0.0, self.config.confidence_config.max_confidence_ceiling) as f32)
                .unwrap_or_else(|| {
                    // Dynamically derive confidence from structural field completeness via confidence_config (parameter-driven)
                    let present_fields = [
                        !subject.is_empty(),
                        !predicate.is_empty(),
                        !object.is_empty(),
                        !context.is_empty(),
                    ]
                    .iter()
                    .filter(|&&b| b)
                    .count();
                    let field_ratio = present_fields as f64 / 4.0;
                    self.config
                        .confidence_config
                        .compute(field_ratio, field_ratio) as f32
                });

            WorldModelRecord {
                id: id.to_string(),
                subject,
                predicate,
                object,
                context,
                confidence,
                source_dataset_id: dataset_id.to_string(),
            }
        } else {
            let content_len = rec
                .as_str()
                .map(|s| s.len())
                .unwrap_or_else(|| rec.to_string().len());
            let ratio = (content_len as f64 / (content_len as f64 + 32.0)).clamp(0.0, 1.0);
            let scalar_confidence = self.config.confidence_config.compute(ratio, ratio) as f32;
            WorldModelRecord {
                id: id.to_string(),
                subject: "scalar_concept".to_string(),
                predicate: "stated_as".to_string(),
                object: rec.to_string(),
                context: String::new(),
                confidence: scalar_confidence,
                source_dataset_id: dataset_id.to_string(),
            }
        }
    }

    fn parse_raw_records(
        &self,
        path: &Path,
    ) -> Result<(Vec<Value>, String), Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut records = Vec::new();
        let mut detected_license = "Permissive/Verified".to_string();

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        if ext == "jsonl" {
            for line_res in reader.lines() {
                let line = line_res?;
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
                    if let Some(lic) = val.get("license").and_then(Value::as_str) {
                        detected_license = lic.to_string();
                    }
                    records.push(val);
                }
            }
        } else if ext == "json" {
            let full_text = fs::read_to_string(path)?;
            let parsed: Value = serde_json::from_str(full_text.trim_start_matches('\u{feff}'))?;
            if let Some(arr) = parsed.as_array() {
                for item in arr {
                    records.push(item.clone());
                }
            } else if let Some(items) = parsed.get("records").and_then(Value::as_array) {
                for item in items {
                    records.push(item.clone());
                }
            } else {
                records.push(parsed);
            }
        } else {
            // Text or TSV line parsing
            for line_res in reader.lines() {
                let line = line_res?;
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    records.push(Value::String(trimmed.to_string()));
                }
            }
        }

        Ok((records, detected_license))
    }

    fn discover_dataset_files(&self, dir: &Path) -> std::io::Result<Vec<PathBuf>> {
        let mut out = Vec::new();
        if !dir.exists() {
            return Ok(out);
        }

        for entry_res in fs::read_dir(dir)? {
            let entry = entry_res?;
            let p = entry.path();
            if p.is_file() {
                if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                    let ext_lower = ext.to_lowercase();
                    if self.config.supported_extensions.contains(&ext_lower) {
                        out.push(p);
                    }
                }
            }
        }
        out.sort();
        Ok(out)
    }

    fn derive_dataset_id(&self, path: &Path, source: DatasetSource) -> String {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("dataset");
        format!("{}_{}", source.as_str(), stem)
    }

    /// Access all training-ready neural records across all routed datasets.
    pub fn get_training_ready_neural(&self) -> Vec<NeuralRecord> {
        let store = self.routed_store.read().unwrap_or_else(|p| p.into_inner());
        let mut out = Vec::new();
        for routed in store.values() {
            out.extend(routed.neural_records.clone());
        }
        out
    }

    /// Access all training-ready world model records across all routed datasets.
    pub fn get_training_ready_world(&self) -> Vec<WorldModelRecord> {
        let store = self.routed_store.read().unwrap_or_else(|p| p.into_inner());
        let mut out = Vec::new();
        for routed in store.values() {
            out.extend(routed.world_records.clone());
        }
        out
    }

    /// Access all training-ready state transition records across all routed datasets.
    pub fn get_training_ready_transitions(&self) -> Vec<StateTransitionRecord> {
        let store = self.routed_store.read().unwrap_or_else(|p| p.into_inner());
        let mut out = Vec::new();
        for routed in store.values() {
            out.extend(routed.transition_records.clone());
        }
        out
    }

    /// Get snapshot of manual datasets registry.
    pub fn get_manual_registry(&self) -> BTreeMap<String, DatasetMetadata> {
        self.manual_registry
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Get snapshot of autonomous datasets registry.
    pub fn get_autonomous_registry(&self) -> BTreeMap<String, DatasetMetadata> {
        self.autonomous_registry
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }
}

/// Re-export canonical compute_dynamic_sha256 from crate root.
pub use crate::compute_dynamic_sha256;

/// Re-export canonical normalize_rel_path from crate root.
pub use crate::normalize_rel_path;

/// Computes a deterministic heuristic difficulty score [0.0, 1.0].
fn calculate_heuristic_difficulty(prompt: &str, completion: &str) -> f32 {
    let len = (prompt.len() + completion.len()) as f32;
    let length_score = (len / 1500.0).clamp(0.0, 0.5);

    let complex_terms = [
        "theorem",
        "algorithm",
        "differential",
        "matrix",
        "integral",
        "quantum",
        "asymptotic",
        "polynomial",
        "cryptographic",
        "concurrency",
        "distributed",
    ];

    let mut term_hits = 0;
    let lower_p = prompt.to_lowercase();
    let lower_c = completion.to_lowercase();
    for term in &complex_terms {
        if lower_p.contains(term) || lower_c.contains(term) {
            term_hits += 1;
        }
    }
    let term_score = (term_hits as f32 * 0.1).clamp(0.0, 0.5);

    (length_score + term_score).clamp(0.05, 1.0)
}

/// Formats UTC timestamp in ISO 8601 without external crate dependency.
pub fn now_utc_iso() -> String {
    use std::time::SystemTime;
    let secs = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

/// Re-export canonical find_workspace_root from crate root.
pub use crate::find_workspace_root;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topological_confidence_config_rejects_nan() {
        let cfg = TopologicalConfidenceConfig {
            base_prior: f64::NAN,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());

        let cfg2 = TopologicalConfidenceConfig {
            max_confidence_ceiling: f64::NAN,
            ..Default::default()
        };
        assert!(cfg2.validate().is_err());
    }

    #[test]
    fn test_state_transition_schema_validation_and_routing() {
        let valid_trans = StateTransitionRecord {
            id: "trans_001".to_string(),
            episode_id: "ep_42".to_string(),
            step_index: 0,
            state_observation: "motor_speed=120,temp=45".to_string(),
            action: "set_throttle(150)".to_string(),
            next_state_observation: "motor_speed=150,temp=48".to_string(),
            reward: 1.0,
            done: false,
            domain: "robotics".to_string(),
            confidence: 0.95,
            source_dataset_id: "test_source".to_string(),
        };

        assert!(validate_state_transition(&valid_trans).is_ok());

        // Test routing from raw JSON values
        let engine = DatasetEngine::new(DatasetEngineConfig::default()).unwrap();
        let raw_records = vec![
            serde_json::json!({
                "state_observation": "pos=(0,0)",
                "action": "move_forward",
                "next_state_observation": "pos=(0,1)",
                "reward": 0.5,
                "done": false,
                "episode_id": "ep_test",
                "step_index": 1
            }),
            serde_json::json!({
                "prompt": "Explain state transitions",
                "completion": "A state transition moves an environment from state s to s'."
            }),
            serde_json::json!({
                "subject": "WorldModel",
                "predicate": "predicts",
                "object": "EnvironmentDynamics"
            }),
        ];

        let (neural, world, transitions) = engine.route_records("test_dataset", &raw_records);
        assert_eq!(neural.len(), 1);
        assert_eq!(world.len(), 1);
        assert_eq!(transitions.len(), 1);

        assert_eq!(transitions[0].state_observation, "pos=(0,0)");
        assert_eq!(transitions[0].action, "move_forward");
        assert_eq!(transitions[0].next_state_observation, "pos=(0,1)");
        assert_eq!(transitions[0].reward, 0.5);
    }

    #[test]
    fn test_state_transition_schema_rejection_of_invalid_records() {
        let empty_state = StateTransitionRecord {
            id: "trans_bad_state".to_string(),
            episode_id: "ep_1".to_string(),
            step_index: 0,
            state_observation: "".to_string(),
            action: "act".to_string(),
            next_state_observation: "next".to_string(),
            reward: 0.0,
            done: false,
            domain: "sim".to_string(),
            confidence: 1.0,
            source_dataset_id: "source".to_string(),
        };
        assert!(validate_state_transition(&empty_state).is_err());

        let nan_reward = StateTransitionRecord {
            id: "trans_nan".to_string(),
            episode_id: "ep_1".to_string(),
            step_index: 0,
            state_observation: "state".to_string(),
            action: "act".to_string(),
            next_state_observation: "next".to_string(),
            reward: f32::NAN,
            done: false,
            domain: "sim".to_string(),
            confidence: 1.0,
            source_dataset_id: "source".to_string(),
        };
        assert!(validate_state_transition(&nan_reward).is_err());

        let oob_confidence = StateTransitionRecord {
            id: "trans_oob".to_string(),
            episode_id: "ep_1".to_string(),
            step_index: 0,
            state_observation: "state".to_string(),
            action: "act".to_string(),
            next_state_observation: "next".to_string(),
            reward: 1.0,
            done: false,
            domain: "sim".to_string(),
            confidence: 1.5,
            source_dataset_id: "source".to_string(),
        };
        assert!(validate_state_transition(&oob_confidence).is_err());
    }
}
