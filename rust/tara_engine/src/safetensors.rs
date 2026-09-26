//! SafeTensors weight file loader.
//!
//! Reads the binary SafeTensors format:
//! - 8-byte little-endian header length
//! - UTF-8 JSON header describing each tensor's dtype, shape, and data offsets
//! - Raw tensor data
//!
//! Supports F32, F16, BF16, I32, I64, I16, I8, U8 and BOOL dtypes, all
//! converting to `Vec<f32>`.  Also handles sharded models via
//! `model.safetensors.index.json`.

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use thiserror::Error;

use sha2::{Digest, Sha256};

/// Errors produced by SafeTensors loading.
#[derive(Debug, Error)]
pub enum SafeTensorsError {
    /// Underlying I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON header could not be parsed.
    #[error("header JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    /// The file is shorter than the declared header.
    #[error("file truncated: expected {expected} bytes, got {got}")]
    Truncated { expected: usize, got: usize },

    /// An unrecognised tensor dtype was encountered.
    #[error("unsupported dtype '{0}'")]
    UnsupportedDtype(String),

    /// The `data_offsets` field was malformed.
    #[error("malformed data_offsets for tensor '{0}'")]
    BadOffsets(String),
}

// ──────────────────────────────────────────────────────────────
//  Core loader
// ──────────────────────────────────────────────────────────────

/// Load all tensors from a single SafeTensors file.
///
/// Returns a map from tensor name to a flat `Vec<f32>` of values in the
/// natural row-major order they appear in the file.
///
/// # Errors
/// Returns [`SafeTensorsError`] on any I/O or parse failure.
pub fn load_safetensors(path: &str) -> Result<HashMap<String, Vec<f32>>, SafeTensorsError> {
    let bytes = fs::read(path)?;

    if bytes.len() < 8 {
        return Err(SafeTensorsError::Truncated {
            expected: 8,
            got: bytes.len(),
        });
    }

    // 8-byte little-endian header length
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header_end = 8 + header_len;

    if bytes.len() < header_end {
        return Err(SafeTensorsError::Truncated {
            expected: header_end,
            got: bytes.len(),
        });
    }

    let header_str = std::str::from_utf8(&bytes[8..header_end])
        .map_err(|e| SafeTensorsError::Json(serde_json::from_str::<()>(&e.to_string()).unwrap_err()))?;

    let header: serde_json::Value = serde_json::from_str(header_str)?;
    let header_obj = header.as_object().ok_or_else(|| {
        serde_json::from_str::<()>("null").unwrap_err()
    });

    // Best-effort: if we can't get the object, parse as empty map
    let empty_map;
    let obj = match header_obj {
        Ok(o) => o,
        Err(_) => {
            empty_map = serde_json::Map::new();
            &empty_map
        }
    };

    let data_base = header_end;
    let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();

    for (name, meta) in obj {
        // Skip the __metadata__ key
        if name == "__metadata__" {
            continue;
        }

        let dtype_str = meta
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("F32");

        let offsets = meta
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| SafeTensorsError::BadOffsets(name.clone()))?;

        if offsets.len() < 2 {
            return Err(SafeTensorsError::BadOffsets(name.clone()));
        }

        let start = offsets[0].as_u64().ok_or_else(|| SafeTensorsError::BadOffsets(name.clone()))? as usize;
        let end   = offsets[1].as_u64().ok_or_else(|| SafeTensorsError::BadOffsets(name.clone()))? as usize;

        let abs_start = data_base + start;
        let abs_end   = data_base + end;

        if bytes.len() < abs_end {
            return Err(SafeTensorsError::Truncated {
                expected: abs_end,
                got: bytes.len(),
            });
        }

        let raw = &bytes[abs_start..abs_end];
        let floats = convert_to_f32(dtype_str, raw, name)?;
        tensors.insert(name.clone(), floats);
    }

    Ok(tensors)
}

/// Convert raw bytes from a SafeTensors blob to `Vec<f32>`.
fn convert_to_f32(dtype: &str, raw: &[u8], name: &str) -> Result<Vec<f32>, SafeTensorsError> {
    match dtype {
        "F32" => {
            if raw.len() % 4 != 0 {
                return Err(SafeTensorsError::BadOffsets(name.to_string()));
            }
            let mut out = Vec::with_capacity(raw.len() / 4);
            for chunk in raw.chunks_exact(4) {
                out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            Ok(out)
        }
        "F16" => {
            if raw.len() % 2 != 0 {
                return Err(SafeTensorsError::BadOffsets(name.to_string()));
            }
            let mut out = Vec::with_capacity(raw.len() / 2);
            for chunk in raw.chunks_exact(2) {
                let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                out.push(half::f16::from_bits(bits).to_f32());
            }
            Ok(out)
        }
        "BF16" => {
            if raw.len() % 2 != 0 {
                return Err(SafeTensorsError::BadOffsets(name.to_string()));
            }
            let mut out = Vec::with_capacity(raw.len() / 2);
            for chunk in raw.chunks_exact(2) {
                let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                // BF16 occupies the upper 16 bits of an f32
                let f32_bits = (bits as u32) << 16;
                out.push(f32::from_bits(f32_bits));
            }
            Ok(out)
        }
        "I32" => {
            if raw.len() % 4 != 0 {
                return Err(SafeTensorsError::BadOffsets(name.to_string()));
            }
            let mut out = Vec::with_capacity(raw.len() / 4);
            for chunk in raw.chunks_exact(4) {
                out.push(i32::from_le_bytes(chunk.try_into().unwrap()) as f32);
            }
            Ok(out)
        }
        "I64" => {
            if raw.len() % 8 != 0 {
                return Err(SafeTensorsError::BadOffsets(name.to_string()));
            }
            let mut out = Vec::with_capacity(raw.len() / 8);
            for chunk in raw.chunks_exact(8) {
                out.push(i64::from_le_bytes(chunk.try_into().unwrap()) as f32);
            }
            Ok(out)
        }
        "I16" => {
            if raw.len() % 2 != 0 {
                return Err(SafeTensorsError::BadOffsets(name.to_string()));
            }
            let mut out = Vec::with_capacity(raw.len() / 2);
            for chunk in raw.chunks_exact(2) {
                out.push(i16::from_le_bytes(chunk.try_into().unwrap()) as f32);
            }
            Ok(out)
        }
        "I8" => Ok(raw.iter().map(|&b| b as i8 as f32).collect()),
        "U8" => Ok(raw.iter().map(|&b| b as f32).collect()),
        "BOOL" => Ok(raw.iter().map(|&b| if b != 0 { 1.0f32 } else { 0.0f32 }).collect()),
        other => Err(SafeTensorsError::UnsupportedDtype(other.to_string())),
    }
}

// ──────────────────────────────────────────────────────────────
//  Sharded loader
// ──────────────────────────────────────────────────────────────

/// Load weights from either a single `model.safetensors` file or a sharded
/// model described by `model.safetensors.index.json`.
///
/// If `model_dir/model.safetensors.index.json` exists the weight_map is read
/// and every referenced shard file is loaded and merged.  Otherwise
/// `model_dir/model.safetensors` is loaded directly.
///
/// # Errors
/// Returns [`SafeTensorsError`] on any failure.
pub fn load_model_weights(model_dir: &str) -> Result<HashMap<String, Vec<f32>>, SafeTensorsError> {
    let index_path = format!("{}/model.safetensors.index.json", model_dir);
    if std::path::Path::new(&index_path).exists() {
        load_sharded_weights(model_dir, &index_path)
    } else {
        let single_path = format!("{}/model.safetensors", model_dir);
        load_safetensors(&single_path)
    }
}

/// Load a sharded model from its index file.
fn load_sharded_weights(
    model_dir: &str,
    index_path: &str,
) -> Result<HashMap<String, Vec<f32>>, SafeTensorsError> {
    let raw = fs::read_to_string(index_path)?;
    let index: serde_json::Value = serde_json::from_str(&raw)?;

    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| serde_json::from_str::<()>("null").unwrap_err())?;

    // Collect unique shard filenames
    let mut shards: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str())
        .map(|s| s.to_string())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    shards.sort();

    let mut merged: HashMap<String, Vec<f32>> = HashMap::new();
    for shard in shards {
        let shard_path = format!("{}/{}", model_dir, shard);
        let shard_weights = load_safetensors(&shard_path)?;
        merged.extend(shard_weights);
    }

    Ok(merged)
}

// ──────────────────────────────────────────────────────────────
//  SHA-256
// ──────────────────────────────────────────────────────────────

/// Compute the hex-encoded SHA-256 digest of a file.
///
/// # Errors
/// Returns [`SafeTensorsError::Io`] if the file cannot be read.
pub fn compute_sha256(path: &str) -> Result<String, SafeTensorsError> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

// ──────────────────────────────────────────────────────────────
//  SafeTensors writer
// ──────────────────────────────────────────────────────────────

/// Write a map of F32 tensors to a SafeTensors file.
///
/// Tensors are written in sorted key order.  The header is padded to an 8-byte
/// boundary with spaces per the SafeTensors specification.
///
/// # Errors
/// Returns [`SafeTensorsError::Io`] on write failure.
pub fn write_safetensors(
    tensors: &HashMap<String, Vec<f32>>,
    path: &str,
) -> Result<(), SafeTensorsError> {
    use std::io::Write;

    // Build header JSON
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();

    let mut offset: u64 = 0;
    let mut tensor_metas: Vec<serde_json::Value> = Vec::new();
    let mut ordered_keys: Vec<&String> = Vec::new();

    for name in &names {
        let data = &tensors[*name];
        let byte_len = (data.len() * 4) as u64;
        tensor_metas.push(serde_json::json!({
            "dtype": "F32",
            "shape": [data.len()],
            "data_offsets": [offset, offset + byte_len]
        }));
        ordered_keys.push(name);
        offset += byte_len;
    }

    let mut header_obj = serde_json::Map::new();
    for (i, name) in ordered_keys.iter().enumerate() {
        header_obj.insert((*name).clone(), tensor_metas[i].clone());
    }

    let header_json = serde_json::to_string(&header_obj)?;

    // Pad to 8-byte boundary
    let raw_len = header_json.len();
    let padded_len = (raw_len + 7) / 8 * 8;
    let padding = padded_len - raw_len;

    let header_len = padded_len as u64;
    let len_bytes = header_len.to_le_bytes();

    let mut out = fs::File::create(path)?;
    out.write_all(&len_bytes)?;
    out.write_all(header_json.as_bytes())?;
    if padding > 0 {
        out.write_all(&vec![b' '; padding])?;
    }

    // Write data
    for name in &ordered_keys {
        let data = &tensors[*name];
        for &v in data.iter() {
            out.write_all(&v.to_le_bytes())?;
        }
    }

    Ok(())
}
