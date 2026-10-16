//! Fast-Weight Synaptic Plasticity Engine.
//!
//! Implements biologically inspired fast associative memory:
//! - Fast weight matrix A of dimension [dim, dim].
//! - Auto-associative and hetero-associative Hebbian updates with decay:
//!   A_{t+1} = lambda * A_t + eta * (x_t * y_t^T)
//! - Associative retrieval: y_pred = A_t * query

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use tara_security::cryptography::{AeadProvider, Aes256GcmProvider};
use zeroize::Zeroize;

pub const MIN_DIM: usize = 1;
pub const MAX_DIM: usize = 4096;
pub const DEFAULT_DIM: usize = 64;
pub const DEFAULT_LAMBDA: f32 = 0.95;
pub const DEFAULT_ETA: f32 = 0.5;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastWeightPlasticityMatrix {
    dim: usize,
    weights: Vec<f32>, // Flat [dim, dim]
    lambda: f32,       // Decay rate [0.0, 1.0]
    eta: f32,          // Plasticity learning rate [0.0, 10.0]
}

impl Default for FastWeightPlasticityMatrix {
    fn default() -> Self {
        Self::new(DEFAULT_DIM, DEFAULT_LAMBDA, DEFAULT_ETA)
    }
}

impl FastWeightPlasticityMatrix {
    pub fn try_new(dim: usize, lambda: f32, eta: f32) -> Result<Self, String> {
        if !(MIN_DIM..=MAX_DIM).contains(&dim) {
            return Err(format!(
                "Invalid matrix dimension {}: must be between {} and {}",
                dim, MIN_DIM, MAX_DIM
            ));
        }
        if !lambda.is_finite() || lambda < 0.0 || lambda > 1.0 {
            return Err(format!(
                "Invalid lambda {}: must be a finite float in range [0.0, 1.0]",
                lambda
            ));
        }
        if !eta.is_finite() || eta < 0.0 || eta > 10.0 {
            return Err(format!(
                "Invalid eta {}: must be a finite float in range [0.0, 10.0]",
                eta
            ));
        }

        Ok(Self {
            dim,
            weights: vec![0.0f32; dim * dim],
            lambda,
            eta,
        })
    }

    pub fn new(dim: usize, lambda: f32, eta: f32) -> Self {
        Self::try_new(dim, lambda, eta).unwrap_or_else(|_| {
            let safe_dim = dim.clamp(MIN_DIM, MAX_DIM);
            let safe_lambda = if lambda.is_finite() {
                lambda.clamp(0.0, 1.0)
            } else {
                DEFAULT_LAMBDA
            };
            let safe_eta = if eta.is_finite() {
                eta.clamp(0.0, 10.0)
            } else {
                DEFAULT_ETA
            };
            Self {
                dim: safe_dim,
                weights: vec![0.0f32; safe_dim * safe_dim],
                lambda: safe_lambda,
                eta: safe_eta,
            }
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn lambda(&self) -> f32 {
        self.lambda
    }

    pub fn eta(&self) -> f32 {
        self.eta
    }

    /// Reset matrix to zero.
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
    }

    /// Associative Hebbian write step:
    /// A = lambda * A + eta * (key * value^T)
    pub fn write_association(&mut self, key: &[f32], value: &[f32]) -> Result<(), String> {
        let d = self.dim;
        if key.len() != d {
            return Err(format!(
                "Key length mismatch: expected {}, got {}",
                d,
                key.len()
            ));
        }
        if value.len() != d {
            return Err(format!(
                "Value length mismatch: expected {}, got {}",
                d,
                value.len()
            ));
        }

        if !key.iter().all(|x| x.is_finite()) || !value.iter().all(|x| x.is_finite()) {
            return Err("Input vectors contain non-finite numbers (NaN or Inf)".into());
        }

        // Normalize inputs
        let k_norm = safe_norm(key).max(1e-8);
        let v_norm = safe_norm(value).max(1e-8);

        for (i, &k_elem) in key.iter().enumerate() {
            let ki = k_elem / k_norm;
            for (j, &v_elem) in value.iter().enumerate() {
                let vj = v_elem / v_norm;
                let idx = i * d + j;
                self.weights[idx] = self.lambda * self.weights[idx] + self.eta * (ki * vj);
            }
        }

        Ok(())
    }

    /// Associative read step:
    /// output = query^T * A = sum_i query_i * A_{i, j}
    pub fn read_association(&self, query: &[f32]) -> Result<Vec<f32>, String> {
        let d = self.dim;
        if query.len() != d {
            return Err(format!(
                "Query length mismatch: expected {}, got {}",
                d,
                query.len()
            ));
        }

        if !query.iter().all(|x| x.is_finite()) {
            return Err("Query vector contains non-finite numbers (NaN or Inf)".into());
        }

        let q_norm = safe_norm(query).max(1e-8);
        let mut out = vec![0.0f32; d];

        for (i, &q_elem) in query.iter().enumerate() {
            let qi = q_elem / q_norm;
            for (j, out_elem) in out.iter_mut().enumerate() {
                *out_elem += qi * self.weights[i * d + j];
            }
        }

        Ok(out)
    }

    /// Compute cosine similarity between two vectors.
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let d = self.dim;
        if a.len() != d || b.len() != d {
            return 0.0;
        }

        if !a.iter().all(|x| x.is_finite()) || !b.iter().all(|x| x.is_finite()) {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let na = safe_norm(a);
        let nb = safe_norm(b);

        if na <= 1e-8 || nb <= 1e-8 {
            0.0
        } else {
            (dot / (na * nb)).clamp(-1.0, 1.0)
        }
    }
}

fn safe_norm(v: &[f32]) -> f32 {
    let max_abs = v.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    if max_abs <= 1e-12 || !max_abs.is_finite() {
        return 0.0;
    }
    let sum_sq: f32 = v.iter().map(|&x| (x / max_abs).powi(2)).sum();
    max_abs * sum_sq.sqrt()
}

/// Multi-tenant/session isolated fast-weight synaptic memory store.
/// Enforces complete isolation between user sessions and agents to prevent weight cross-contamination.
pub struct FastWeightTenantStore {
    stores: Mutex<HashMap<String, FastWeightPlasticityMatrix>>,
    dim: usize,
    lambda: f32,
    eta: f32,
}

impl Default for FastWeightTenantStore {
    fn default() -> Self {
        Self::new(DEFAULT_DIM, DEFAULT_LAMBDA, DEFAULT_ETA)
    }
}

impl FastWeightTenantStore {
    pub fn new(dim: usize, lambda: f32, eta: f32) -> Self {
        Self {
            stores: Mutex::new(HashMap::new()),
            dim,
            lambda,
            eta,
        }
    }

    pub fn with_tenant_matrix<R>(
        &self,
        tenant_id: &str,
        f: impl FnOnce(&mut FastWeightPlasticityMatrix) -> R,
    ) -> R {
        let mut map = self.stores.lock().unwrap_or_else(|p| p.into_inner());
        let matrix = map
            .entry(tenant_id.to_string())
            .or_insert_with(|| FastWeightPlasticityMatrix::new(self.dim, self.lambda, self.eta));
        f(matrix)
    }

    /// Access tenant associative memory bound strictly to verified AuthenticatedPrincipal identity.
    pub fn with_principal_matrix<R>(
        &self,
        principal: &tara_security::primitives::AuthenticatedPrincipal,
        f: impl FnOnce(&mut FastWeightPlasticityMatrix) -> R,
    ) -> Result<R, String> {
        if !principal.verify_authority() {
            return Err("Authority verification failed for tenant principal".into());
        }
        let tenant_id = principal.actor_id();
        Ok(self.with_tenant_matrix(tenant_id, f))
    }

    pub fn clear_tenant(&self, tenant_id: &str) {
        let mut map = self.stores.lock().unwrap_or_else(|p| p.into_inner());
        map.remove(tenant_id);
    }

    pub fn active_tenants_count(&self) -> usize {
        let map = self.stores.lock().unwrap_or_else(|p| p.into_inner());
        map.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FastWeightMemoryVaultData {
    dim: usize,
    lambda: f32,
    eta: f32,
    weights: Vec<f32>,
    #[serde(default)]
    partitions: HashMap<String, Vec<f32>>,
}

/// Principal-bound, tenant-isolated fast associative memory matrix with gradient clipping,
/// finite float assertions, and zeroization for secure cleanup.
pub struct FastWeightMemory {
    dim: usize,
    weights: Vec<f32>,
    lambda: f32,
    eta: f32,
    partitions: HashMap<String, Vec<f32>>,
}

impl FastWeightMemory {
    pub fn new(dim: usize, lambda: f32, eta: f32) -> Result<Self, String> {
        if dim > 512 {
            return Err("Dimension exceeds maximum limit of 512".into());
        }
        if dim < MIN_DIM {
            return Err("Dimension must be at least 1".into());
        }
        if !lambda.is_finite() || lambda < 0.0 || lambda > 1.0 {
            return Err("Invalid lambda: must be finite and within [0.0, 1.0]".into());
        }
        if !eta.is_finite() || eta < 0.0 || eta > 1.0 {
            return Err("Invalid eta: must be finite and within [0.0, 1.0]".into());
        }
        Ok(Self {
            dim,
            weights: vec![0.0f32; dim * dim],
            lambda,
            eta,
            partitions: HashMap::new(),
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn lambda(&self) -> f32 {
        self.lambda
    }

    pub fn eta(&self) -> f32 {
        self.eta
    }

    pub fn update(&mut self, key: &[f32], value: &[f32]) -> Result<(), String> {
        if key.len() != self.dim || value.len() != self.dim {
            return Err(format!(
                "Dimension mismatch: expected {}, got key {}, val {}",
                self.dim,
                key.len(),
                value.len()
            ));
        }
        if !key.iter().all(|x| x.is_finite()) || !value.iter().all(|x| x.is_finite()) {
            return Err("Input vector contains NaN or Infinity".into());
        }

        let d = self.dim;
        for (i, &k_val) in key.iter().enumerate().take(d) {
            let ki = k_val.clamp(-1.0, 1.0); // gradient clipping <= 1.0
            for (j, &v_val) in value.iter().enumerate().take(d) {
                let vj = v_val.clamp(-1.0, 1.0);
                let idx = i * d + j;
                let updated = self.lambda * self.weights[idx] + self.eta * (ki * vj);
                self.weights[idx] = updated.clamp(-10.0, 10.0); // weight bounds [-10.0, 10.0]
            }
        }
        Ok(())
    }

    pub fn update_with_principal(
        &mut self,
        principal: &tara_security::primitives::AuthenticatedPrincipal,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), String> {
        if !principal.verify_authority() {
            return Err("Authority verification failed for principal".into());
        }
        self.update(key, value)
    }

    pub fn query(&self, query_vec: &[f32]) -> Result<Vec<f32>, String> {
        if query_vec.len() != self.dim {
            return Err(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.dim,
                query_vec.len()
            ));
        }
        if !query_vec.iter().all(|x| x.is_finite()) {
            return Err("Query vector contains NaN or Infinity".into());
        }

        let d = self.dim;
        if self.weights.len() < d * d {
            return Ok(vec![0.0f32; d]);
        }
        let mut out = vec![0.0f32; d];
        for (i, &qi) in query_vec.iter().enumerate().take(d) {
            for (j, out_j) in out.iter_mut().enumerate().take(d) {
                *out_j += qi * self.weights[i * d + j];
            }
        }
        Ok(out)
    }

    pub fn clear(&mut self) {
        self.weights.zeroize();
        self.weights = vec![0.0f32; self.dim * self.dim];
        for part in self.partitions.values_mut() {
            part.zeroize();
        }
        self.partitions.clear();
    }

    pub fn get_partition(
        &mut self,
        tenant_id: &str,
        session_id: &str,
    ) -> Result<&mut Vec<f32>, String> {
        if tenant_id.trim().is_empty() || session_id.trim().is_empty() {
            return Err("Tenant ID and Session ID cannot be empty".into());
        }
        let partition_key = format!("{}:{}", tenant_id, session_id);
        let dim = self.dim;
        let entry = self
            .partitions
            .entry(partition_key)
            .or_insert_with(|| vec![0.0f32; dim * dim]);
        Ok(entry)
    }

    pub fn persist(&self, path: &std::path::Path) -> Result<(), String> {
        let vault_data = FastWeightMemoryVaultData {
            dim: self.dim,
            lambda: self.lambda,
            eta: self.eta,
            weights: self.weights.clone(),
            partitions: self.partitions.clone(),
        };
        let serialized =
            serde_json::to_vec(&vault_data).map_err(|e| format!("Serialization error: {}", e))?;

        let key = tara_security::dpapi::derive_authority_hmac_key("fast_weights_vault")
            .map_err(|e| format!("Key derivation error: {}", e))?;
        let provider = Aes256GcmProvider;
        let encrypted = provider.encrypt(&key, &serialized, b"fast_weights_vault")?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create parent dir: {}", e))?;
        }
        std::fs::write(path, encrypted).map_err(|e| format!("Write error: {}", e))?;
        Ok(())
    }

    /// Loads fast weight memory from encrypted vault file, auto-detecting dimensions and hyperparameters.
    pub fn load_from_path(path: &std::path::Path) -> Result<Self, String> {
        let encrypted = std::fs::read(path).map_err(|e| format!("Read error: {}", e))?;
        let key = tara_security::dpapi::derive_authority_hmac_key("fast_weights_vault")
            .map_err(|e| format!("Key derivation error: {}", e))?;
        let provider = Aes256GcmProvider;
        let decrypted = provider.decrypt(&key, &encrypted, b"fast_weights_vault")?;

        if let Ok(vault_data) = serde_json::from_slice::<FastWeightMemoryVaultData>(&decrypted) {
            if vault_data.weights.len() != vault_data.dim * vault_data.dim {
                return Err(format!(
                    "Dimension mismatch in loaded weights: expected {}, got {}",
                    vault_data.dim * vault_data.dim,
                    vault_data.weights.len()
                ));
            }
            let mut mem = Self::new(vault_data.dim, vault_data.lambda, vault_data.eta)?;
            mem.weights = vault_data.weights;
            mem.partitions = vault_data.partitions;
            Ok(mem)
        } else {
            let weights: Vec<f32> = serde_json::from_slice(&decrypted)
                .map_err(|e| format!("Deserialization error: {}", e))?;
            let len = weights.len();
            let dim = (len as f64).sqrt().round() as usize;
            if dim * dim != len || dim == 0 {
                return Err(format!("Cannot determine square dimension for {} weights", len));
            }
            let mut mem = Self::new(dim, DEFAULT_LAMBDA, DEFAULT_ETA)?;
            mem.weights = weights;
            Ok(mem)
        }
    }

    /// Loads fast weight memory from path and verifies against expected dimensions and bounds.
    pub fn load_from_path_with_dim(
        path: &std::path::Path,
        dim: usize,
        lambda: f32,
        eta: f32,
    ) -> Result<Self, String> {
        let mut mem = Self::load_from_path(path)?;
        if mem.dim != dim {
            return Err(format!(
                "Dimension mismatch in loaded weights: expected {}, got {}",
                dim, mem.dim
            ));
        }
        mem.lambda = lambda;
        mem.eta = eta;
        Ok(mem)
    }

    pub fn load(&mut self, path: &std::path::Path) -> Result<(), String> {
        let mut loaded = Self::load_from_path(path)?;
        if loaded.dim != self.dim {
            return Err(format!(
                "Dimension mismatch in loaded weights: expected {}, got {}",
                self.dim * self.dim,
                loaded.dim * loaded.dim
            ));
        }
        std::mem::swap(&mut self.weights, &mut loaded.weights);
        self.lambda = loaded.lambda;
        self.eta = loaded.eta;
        std::mem::swap(&mut self.partitions, &mut loaded.partitions);
        Ok(())
    }
}

impl Drop for FastWeightMemory {
    fn drop(&mut self) {
        self.weights.zeroize();
        for part in self.partitions.values_mut() {
            part.zeroize();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_weights_associative_recall() {
        let dim = 8;
        let mut fwm = FastWeightPlasticityMatrix::new(dim, 0.95, 1.0);

        let mut key1 = vec![0.0f32; dim];
        key1[0] = 1.0;
        key1[1] = 1.0;

        let mut val1 = vec![0.0f32; dim];
        val1[4] = 1.0;
        val1[5] = 1.0;

        fwm.write_association(&key1, &val1).unwrap();

        // Query with key1
        let recalled = fwm.read_association(&key1).unwrap();
        let sim = fwm.cosine_similarity(&recalled, &val1);
        assert!(
            sim > 0.99,
            "Recalled vector should closely match stored value vector: got {sim}"
        );

        // Query with orthogonal key
        let mut key_ortho = vec![0.0f32; dim];
        key_ortho[2] = 1.0;
        let recalled_ortho = fwm.read_association(&key_ortho).unwrap();
        let sim_ortho = fwm.cosine_similarity(&recalled_ortho, &val1);
        assert!(
            sim_ortho < 0.01,
            "Orthogonal query should yield zero recall: got {sim_ortho}"
        );

        // Mismatched length should return error instead of panicking
        let short_query = vec![1.0f32; 4];
        assert!(fwm.read_association(&short_query).is_err());
    }

    #[test]
    fn test_fast_weight_memory_encrypted_persistence_roundtrip() {
        let temp_dir =
            std::env::temp_dir().join(format!("tara_fwm_test_{}", rand::random::<u64>()));
        let vault_file = temp_dir.join("weights.vault");

        let mut memory = FastWeightMemory::new(4, 0.9, 0.5).unwrap();
        let key = vec![1.0, 0.5, -0.5, 0.0];
        let val = vec![0.2, 0.8, 0.1, -0.9];
        memory.update(&key, &val).unwrap();

        // Persist first time
        memory.persist(&vault_file).unwrap();
        let ciphertext1 = std::fs::read(&vault_file).unwrap();

        // Persist second time: nonces MUST be random and distinct!
        memory.persist(&vault_file).unwrap();
        let ciphertext2 = std::fs::read(&vault_file).unwrap();

        assert_ne!(
            ciphertext1, ciphertext2,
            "Ciphertexts must differ due to random CSPRNG nonces (anti-replay/anti-static-nonce)"
        );

        // Load into new instance via auto-detecting load_from_path
        let loaded = FastWeightMemory::load_from_path(&vault_file).unwrap();
        assert_eq!(loaded.dim(), 4);
        assert_eq!(loaded.lambda(), 0.9);
        assert_eq!(loaded.eta(), 0.5);
        assert_eq!(memory.weights, loaded.weights);

        // Load with explicit dim validation
        let loaded_with_dim =
            FastWeightMemory::load_from_path_with_dim(&vault_file, 4, 0.9, 0.5).unwrap();
        assert_eq!(memory.weights, loaded_with_dim.weights);

        // Incompatible dimension fails closed
        assert!(FastWeightMemory::load_from_path_with_dim(&vault_file, 8, 0.9, 0.5).is_err());

        // Query both to ensure identical associative outputs
        let query_vec = vec![1.0, 0.5, -0.5, 0.0];
        let out_original = memory.query(&query_vec).unwrap();
        let out_loaded = loaded.query(&query_vec).unwrap();
        assert_eq!(out_original, out_loaded);

        // In-place load into another instance of same dim
        let mut target_mem = FastWeightMemory::new(4, 0.9, 0.5).unwrap();
        target_mem.load(&vault_file).unwrap();
        assert_eq!(memory.weights, target_mem.weights);

        // In-place load into mismatching dim fails closed
        let mut mismatch_mem = FastWeightMemory::new(8, 0.9, 0.5).unwrap();
        assert!(mismatch_mem.load(&vault_file).is_err());

        // Tampering with ciphertext should fail decryption (AEAD integrity)
        let mut tampered = ciphertext2.clone();
        let len = tampered.len();
        tampered[len - 1] ^= 0xFF;
        std::fs::write(&vault_file, &tampered).unwrap();
        assert!(
            FastWeightMemory::load_from_path(&vault_file).is_err(),
            "Tampered ciphertext must fail AEAD verification"
        );
    }
}

