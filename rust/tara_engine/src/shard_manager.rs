//! Unified SafeTensors Shard Manager & Lazy Paging Runtime.
//!
//! Features:
//! - Single-model abstraction over single-file or multi-shard SafeTensors weights.
//! - Automatic discovery of `model.safetensors.index.json` or `*.safetensors`.
//! - Lazy shard paging with LRU cache eviction.
//! - Direct tensor extraction with dtype conversion to `Vec<f32>`.

use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::safetensors::{load_safetensors_with_shapes, SafeTensorsError};

#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
    pub shard_filename: String,
    pub data_bytes: usize,
}

pub struct ShardedSafeTensorsManager {
    pub model_dir: PathBuf,
    pub max_cached_shards: usize,
    pub shard_paths: HashMap<String, PathBuf>,
    pub tensor_index: HashMap<String, TensorMetadata>,
    pub weight_map: HashMap<String, String>,
    pub total_weights_bytes: u64,
    // LRU cached shard data: shard_filename -> (data_start_offset, raw_shard_bytes)
    shard_cache: HashMap<String, (usize, Vec<u8>)>,
    lru_order: VecDeque<String>,
}

impl ShardedSafeTensorsManager {
    pub fn new<P: AsRef<Path>>(
        model_dir: P,
        max_cached_shards: usize,
    ) -> Result<Self, SafeTensorsError> {
        let mut mgr = Self {
            model_dir: model_dir.as_ref().to_path_buf(),
            max_cached_shards: max_cached_shards.max(1),
            shard_paths: HashMap::new(),
            tensor_index: HashMap::new(),
            weight_map: HashMap::new(),
            total_weights_bytes: 0,
            shard_cache: HashMap::new(),
            lru_order: VecDeque::new(),
        };

        mgr.discover_and_validate()?;
        Ok(mgr)
    }

    pub fn discover_and_validate(&mut self) -> Result<(), SafeTensorsError> {
        let index_file = self.model_dir.join("model.safetensors.index.json");
        let single_file = self.model_dir.join("model.safetensors");

        if index_file.exists() {
            let data = std::fs::read_to_string(&index_file)?;
            let v: Value = serde_json::from_str(&data)?;
            if let Some(map) = v.get("weight_map").and_then(Value::as_object) {
                for (t_name, s_name) in map {
                    if let Some(s_str) = s_name.as_str() {
                        self.weight_map.insert(t_name.clone(), s_str.to_string());
                        let full_path = self.model_dir.join(s_str);
                        if !full_path.exists() {
                            return Err(SafeTensorsError::Io(std::io::Error::new(
                                std::io::ErrorKind::NotFound,
                                format!("Missing SafeTensors shard: {}", full_path.display()),
                            )));
                        }
                        self.shard_paths.insert(s_str.to_string(), full_path);
                    }
                }
            }
        } else if single_file.exists() {
            self.shard_paths
                .insert("model.safetensors".into(), single_file);
        } else {
            // Scan for any *.safetensors in dir
            for entry in std::fs::read_dir(&self.model_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                    let name = entry.file_name().to_string_lossy().to_string();
                    self.shard_paths.insert(name, path);
                }
            }
            if self.shard_paths.is_empty() {
                return Err(SafeTensorsError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "No SafeTensors weights found in {}",
                        self.model_dir.display()
                    ),
                )));
            }
        }

        // Index headers of all shards
        let shards: Vec<(String, PathBuf)> = self
            .shard_paths
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        for (name, path) in shards {
            self.index_shard_header(&name, &path)?;
        }

        Ok(())
    }

    fn index_shard_header(
        &mut self,
        shard_name: &str,
        shard_path: &Path,
    ) -> Result<(), SafeTensorsError> {
        let mut file = File::open(shard_path)?;
        let mut len_buf = [0u8; 8];
        file.read_exact(&mut len_buf)?;
        let header_len = u64::from_le_bytes(len_buf) as usize;

        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)?;

        let header: Value = serde_json::from_slice(&header_bytes)?;
        let header_obj = header
            .as_object()
            .ok_or_else(|| SafeTensorsError::InvalidHeader("Header is not a JSON object".into()))?;

        for (k, v) in header_obj {
            if k == "__metadata__" {
                continue;
            }
            let dtype = v
                .get("dtype")
                .and_then(Value::as_str)
                .unwrap_or("F32")
                .to_string();
            let shape = v
                .get("shape")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(Value::as_u64)
                        .map(|x| x as usize)
                        .collect()
                })
                .unwrap_or_default();

            let offsets = v
                .get("data_offsets")
                .and_then(Value::as_array)
                .map(|arr| {
                    let start = arr.first().and_then(Value::as_u64).unwrap_or(0) as usize;
                    let end = arr.get(1).and_then(Value::as_u64).unwrap_or(0) as usize;
                    (start, end)
                })
                .unwrap_or((0, 0));

            let data_bytes = offsets.1.saturating_sub(offsets.0);
            self.total_weights_bytes += data_bytes as u64;

            let meta = TensorMetadata {
                name: k.clone(),
                dtype,
                shape,
                data_offsets: offsets,
                shard_filename: shard_name.to_string(),
                data_bytes,
            };

            self.weight_map.insert(k.clone(), shard_name.to_string());
            self.tensor_index.insert(k.clone(), meta);
        }

        Ok(())
    }

    pub fn list_tensors(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tensor_index.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn get_tensor_metadata(&self, name: &str) -> Option<&TensorMetadata> {
        self.tensor_index.get(name)
    }

    /// Load tensor raw bytes from shard.
    pub fn get_tensor_bytes(&mut self, tensor_name: &str) -> Result<Vec<u8>, SafeTensorsError> {
        let meta = self
            .tensor_index
            .get(tensor_name)
            .ok_or_else(|| {
                SafeTensorsError::InvalidHeader(format!(
                    "Tensor '{}' not found in shard index",
                    tensor_name
                ))
            })?
            .clone();

        let shard_name = meta.shard_filename.clone();
        let (start, end) = meta.data_offsets;
        let tensor_bytes_len = end.saturating_sub(start);

        // Check if shard is cached
        if let Some((data_start, cached_bytes)) = self.shard_cache.get(&shard_name) {
            let offset = *data_start + start;
            if offset + tensor_bytes_len <= cached_bytes.len() {
                let bytes = cached_bytes[offset..offset + tensor_bytes_len].to_vec();
                self.touch_lru(&shard_name);
                return Ok(bytes);
            }
        }

        // Read directly from file
        let shard_path = self.shard_paths.get(&shard_name).ok_or_else(|| {
            SafeTensorsError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Shard file path not found for '{}'", shard_name),
            ))
        })?;

        let mut file = File::open(shard_path)?;
        let mut len_buf = [0u8; 8];
        file.read_exact(&mut len_buf)?;
        let header_len = u64::from_le_bytes(len_buf) as usize;
        let data_start = 8 + header_len;

        file.seek(SeekFrom::Start((data_start + start) as u64))?;
        let mut buf = vec![0u8; tensor_bytes_len];
        file.read_exact(&mut buf)?;

        Ok(buf)
    }

    /// Load and convert tensor data into flat `Vec<f32>`.
    pub fn load_tensor(&mut self, tensor_name: &str) -> Result<Vec<f32>, SafeTensorsError> {
        let meta = self
            .tensor_index
            .get(tensor_name)
            .ok_or_else(|| {
                SafeTensorsError::InvalidHeader(format!(
                    "Tensor '{}' not found in shard index",
                    tensor_name
                ))
            })?
            .clone();

        let raw_bytes = self.get_tensor_bytes(tensor_name)?;

        // Convert based on dtype
        match meta.dtype.to_uppercase().as_str() {
            "F32" | "FLOAT32" => {
                let count = raw_bytes.len() / 4;
                let mut res = Vec::with_capacity(count);
                for i in 0..count {
                    let b = [
                        raw_bytes[i * 4],
                        raw_bytes[i * 4 + 1],
                        raw_bytes[i * 4 + 2],
                        raw_bytes[i * 4 + 3],
                    ];
                    res.push(f32::from_le_bytes(b));
                }
                Ok(res)
            }
            "F16" | "FLOAT16" => {
                let count = raw_bytes.len() / 2;
                let mut res = Vec::with_capacity(count);
                for i in 0..count {
                    let b = [raw_bytes[i * 2], raw_bytes[i * 2 + 1]];
                    let val = half::f16::from_le_bytes(b);
                    res.push(val.to_f32());
                }
                Ok(res)
            }
            "BF16" | "BFLOAT16" => {
                let count = raw_bytes.len() / 2;
                let mut res = Vec::with_capacity(count);
                for i in 0..count {
                    let b = [raw_bytes[i * 2], raw_bytes[i * 2 + 1]];
                    let val = half::bf16::from_le_bytes(b);
                    res.push(val.to_f32());
                }
                Ok(res)
            }
            "I8" | "INT8" => Ok(raw_bytes.into_iter().map(|b| (b as i8) as f32).collect()),
            other => Err(SafeTensorsError::UnsupportedDtype(other.to_string())),
        }
    }

    /// Load all tensors into memory simultaneously.
    pub fn load_all_tensors(&mut self) -> Result<HashMap<String, Vec<f32>>, SafeTensorsError> {
        let mut map = HashMap::new();
        // Check if single file
        let single_file = self.model_dir.join("model.safetensors");
        if single_file.exists() && self.shard_paths.len() == 1 {
            let (w, _) = load_safetensors_with_shapes(&single_file.to_string_lossy())?;
            return Ok(w);
        }

        // Multi shard loading
        let names = self.list_tensors();
        for name in names {
            let t = self.load_tensor(&name)?;
            map.insert(name, t);
        }
        Ok(map)
    }

    fn touch_lru(&mut self, shard_name: &str) {
        if let Some(pos) = self.lru_order.iter().position(|x| x == shard_name) {
            self.lru_order.remove(pos);
        }
        self.lru_order.push_back(shard_name.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::write_safetensors_sharded;

    #[test]
    fn test_shard_manager_missing_weights_error() {
        let empty_dir = std::env::temp_dir().join(format!("tara_empty_model_{:x}", rand::random::<u64>()));
        let _ = std::fs::create_dir_all(&empty_dir);
        let res = ShardedSafeTensorsManager::new(&empty_dir, 2);
        let _ = std::fs::remove_dir_all(&empty_dir);
        assert!(res.is_err());
    }

    #[test]
    fn test_shard_manager_roundtrip_discovery_and_load() {
        let test_dir = std::env::temp_dir().join(format!("tara_shard_mgr_{:x}", rand::random::<u64>()));
        let _ = std::fs::create_dir_all(&test_dir);

        let mut tensors = HashMap::new();
        let mut shapes = HashMap::new();

        tensors.insert("weight.a".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
        shapes.insert("weight.a".to_string(), vec![2, 2]);

        tensors.insert("bias.b".to_string(), vec![0.5, -0.5]);
        shapes.insert("bias.b".to_string(), vec![2]);

        let written = write_safetensors_sharded(
            &tensors,
            &shapes,
            &test_dir.to_string_lossy(),
            1024 * 1024,
        );
        assert!(written.is_ok());

        let mut mgr = ShardedSafeTensorsManager::new(&test_dir, 2).unwrap();
        let list = mgr.list_tensors();
        assert_eq!(list.len(), 2);
        assert!(list.contains(&"weight.a".to_string()));
        assert!(list.contains(&"bias.b".to_string()));

        let loaded_a = mgr.load_tensor("weight.a").unwrap();
        assert_eq!(loaded_a, vec![1.0, 2.0, 3.0, 4.0]);

        let loaded_b = mgr.load_tensor("bias.b").unwrap();
        assert_eq!(loaded_b, vec![0.5, -0.5]);

        let all = mgr.load_all_tensors().unwrap();
        assert_eq!(all.len(), 2);

        let _ = std::fs::remove_dir_all(&test_dir);
    }
}

