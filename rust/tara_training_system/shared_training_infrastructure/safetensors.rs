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
    #[error("invalid SafeTensors header: {0}")]
    InvalidHeader(String),
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
/// Weight data map paired with tensor dimension map.
pub type WeightsWithShapes = (HashMap<String, Vec<f32>>, HashMap<String, Vec<usize>>);

/// Returns [`SafeTensorsError`] on any I/O or parse failure.
pub fn load_safetensors(path: &str) -> Result<HashMap<String, Vec<f32>>, SafeTensorsError> {
    Ok(load_safetensors_with_shapes(path)?.0)
}

/// Load tensor values and their original dimensions from a SafeTensors file.
pub fn load_safetensors_with_shapes(path: &str) -> Result<WeightsWithShapes, SafeTensorsError> {
    let bytes = fs::read(path)?;

    if bytes.len() < 8 {
        return Err(SafeTensorsError::Truncated {
            expected: 8,
            got: bytes.len(),
        });
    }

    // 8-byte little-endian header length
    let header_len =
        usize::try_from(u64::from_le_bytes(bytes[..8].try_into().unwrap())).map_err(|_| {
            SafeTensorsError::InvalidHeader("header length overflows address space".into())
        })?;
    let header_end = 8usize
        .checked_add(header_len)
        .ok_or_else(|| SafeTensorsError::InvalidHeader("header length overflow".into()))?;

    if bytes.len() < header_end {
        return Err(SafeTensorsError::Truncated {
            expected: header_end,
            got: bytes.len(),
        });
    }

    let header_str = std::str::from_utf8(&bytes[8..header_end])
        .map_err(|e| SafeTensorsError::InvalidHeader(e.to_string()))?;

    let header: serde_json::Value = serde_json::from_str(header_str)?;
    let obj = header.as_object().ok_or_else(|| {
        SafeTensorsError::InvalidHeader("header root must be a JSON object".into())
    })?;

    let data_base = header_end;
    let data_len = bytes.len() - data_base;
    let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
    let mut ranges = Vec::new();

    for (name, meta) in obj {
        // Skip the __metadata__ key
        if name == "__metadata__" {
            continue;
        }

        let dtype_str = meta.get("dtype").and_then(|v| v.as_str()).ok_or_else(|| {
            SafeTensorsError::InvalidHeader(format!("missing dtype for tensor '{name}'"))
        })?;

        let shape = meta
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                SafeTensorsError::InvalidHeader(format!("missing shape for tensor '{name}'"))
            })?;
        let element_count = shape.iter().try_fold(1usize, |count, dim| {
            let dim = dim
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| {
                    SafeTensorsError::InvalidHeader(format!("invalid shape for tensor '{name}'"))
                })?;
            count.checked_mul(dim).ok_or_else(|| {
                SafeTensorsError::InvalidHeader(format!("shape overflow for tensor '{name}'"))
            })
        })?;
        let shape = shape
            .iter()
            .map(|dim| {
                dim.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| {
                        SafeTensorsError::InvalidHeader(format!(
                            "invalid shape for tensor '{name}'"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let offsets = meta
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| SafeTensorsError::BadOffsets(name.clone()))?;

        if offsets.len() < 2 {
            return Err(SafeTensorsError::BadOffsets(name.clone()));
        }

        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| SafeTensorsError::BadOffsets(name.clone()))?
            as usize;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| SafeTensorsError::BadOffsets(name.clone()))? as usize;

        if start > end || end > data_len {
            return Err(SafeTensorsError::Truncated {
                expected: data_base.saturating_add(end),
                got: bytes.len(),
            });
        }

        ranges.push((start, end, name.clone()));
        let abs_start = data_base + start;
        let abs_end = data_base + end;
        let raw = &bytes[abs_start..abs_end];
        let floats = convert_to_f32(dtype_str, raw, name)?;
        if floats.len() != element_count {
            return Err(SafeTensorsError::BadOffsets(name.clone()));
        }
        tensors.insert(name.clone(), floats);
        shapes.insert(name.clone(), shape);
    }

    ranges.sort_by_key(|(start, _, _)| *start);
    let mut expected_offset = 0usize;
    for (start, end, name) in ranges {
        if start != expected_offset {
            return Err(SafeTensorsError::BadOffsets(name));
        }
        expected_offset = end;
    }
    if expected_offset != data_len {
        return Err(SafeTensorsError::InvalidHeader(
            "unreferenced bytes remain in tensor data".into(),
        ));
    }

    Ok((tensors, shapes))
}

/// Convert raw bytes from a SafeTensors blob to `Vec<f32>`.
fn convert_to_f32(dtype: &str, raw: &[u8], name: &str) -> Result<Vec<f32>, SafeTensorsError> {
    match dtype {
        "F32" => {
            if !raw.len().is_multiple_of(4) {
                return Err(SafeTensorsError::BadOffsets(name.to_string()));
            }
            let mut out = Vec::with_capacity(raw.len() / 4);
            for &chunk in raw.as_chunks::<4>().0 {
                out.push(f32::from_le_bytes(chunk));
            }
            Ok(out)
        }
        "F16" => {
            if !raw.len().is_multiple_of(2) {
                return Err(SafeTensorsError::BadOffsets(name.to_string()));
            }
            let mut out = Vec::with_capacity(raw.len() / 2);
            for &chunk in raw.as_chunks::<2>().0 {
                let bits = u16::from_le_bytes(chunk);
                out.push(half::f16::from_bits(bits).to_f32());
            }
            Ok(out)
        }
        "BF16" => {
            if !raw.len().is_multiple_of(2) {
                return Err(SafeTensorsError::BadOffsets(name.to_string()));
            }
            let mut out = Vec::with_capacity(raw.len() / 2);
            for &chunk in raw.as_chunks::<2>().0 {
                let bits = u16::from_le_bytes(chunk);
                // BF16 occupies the upper 16 bits of an f32
                let f32_bits = (bits as u32) << 16;
                out.push(f32::from_bits(f32_bits));
            }
            Ok(out)
        }
        "I32" => {
            if !raw.len().is_multiple_of(4) {
                return Err(SafeTensorsError::BadOffsets(name.to_string()));
            }
            let mut out = Vec::with_capacity(raw.len() / 4);
            for &chunk in raw.as_chunks::<4>().0 {
                out.push(i32::from_le_bytes(chunk) as f32);
            }
            Ok(out)
        }
        "I64" => {
            if !raw.len().is_multiple_of(8) {
                return Err(SafeTensorsError::BadOffsets(name.to_string()));
            }
            let mut out = Vec::with_capacity(raw.len() / 8);
            for &chunk in raw.as_chunks::<8>().0 {
                out.push(i64::from_le_bytes(chunk) as f32);
            }
            Ok(out)
        }
        "I16" => {
            if !raw.len().is_multiple_of(2) {
                return Err(SafeTensorsError::BadOffsets(name.to_string()));
            }
            let mut out = Vec::with_capacity(raw.len() / 2);
            for &chunk in raw.as_chunks::<2>().0 {
                out.push(i16::from_le_bytes(chunk) as f32);
            }
            Ok(out)
        }
        "I8" => Ok(raw.iter().map(|&b| b as i8 as f32).collect()),
        "U8" => Ok(raw.iter().map(|&b| b as f32).collect()),
        "BOOL" => Ok(raw
            .iter()
            .map(|&b| if b != 0 { 1.0f32 } else { 0.0f32 })
            .collect()),
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
/// and every referenced shard file is loaded and merged. Otherwise
/// `model_dir/model.safetensors` is loaded directly.
///
/// # Errors
/// Returns [`SafeTensorsError`] on any failure.
pub fn load_model_weights(model_dir: &str) -> Result<HashMap<String, Vec<f32>>, SafeTensorsError> {
    Ok(load_model_weights_with_shapes(model_dir)?.0)
}

/// Load model tensors and preserve the dimensions from the source files.
pub fn load_model_weights_with_shapes(
    model_dir: &str,
) -> Result<WeightsWithShapes, SafeTensorsError> {
    let index_path = format!("{}/model.safetensors.index.json", model_dir);
    if std::path::Path::new(&index_path).exists() {
        load_sharded_weights(model_dir, &index_path)
    } else {
        let single_path = format!("{}/model.safetensors", model_dir);
        load_safetensors_with_shapes(&single_path)
    }
}

/// Load a sharded model from its index file.
fn load_sharded_weights(
    model_dir: &str,
    index_path: &str,
) -> Result<WeightsWithShapes, SafeTensorsError> {
    let raw = fs::read_to_string(index_path)?;
    let index: serde_json::Value = serde_json::from_str(&raw)?;

    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| serde_json::from_str::<()>("null").unwrap_err())?;

    let root = std::path::Path::new(model_dir).canonicalize()?;
    let mut merged: HashMap<String, Vec<f32>> = HashMap::new();
    let mut merged_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    let mut shard_cache: HashMap<String, WeightsWithShapes> = HashMap::new();
    for (tensor_name, shard_value) in weight_map {
        let shard = shard_value.as_str().ok_or_else(|| {
            SafeTensorsError::InvalidHeader(format!(
                "invalid shard reference for tensor '{tensor_name}'"
            ))
        })?;
        let shard_path = std::path::Path::new(shard);
        if shard_path.is_absolute()
            || shard_path
                .components()
                .any(|part| !matches!(part, std::path::Component::Normal(_)))
        {
            return Err(SafeTensorsError::InvalidHeader(format!(
                "unsafe shard path '{shard}'"
            )));
        }
        if !shard_cache.contains_key(shard) {
            let full = root.join(shard).canonicalize()?;
            if !full.starts_with(&root) || !full.is_file() {
                return Err(SafeTensorsError::InvalidHeader(format!(
                    "shard path escapes model directory: '{shard}'"
                )));
            }
            let loaded = load_safetensors_with_shapes(&full.to_string_lossy())?;
            shard_cache.insert(shard.to_string(), loaded);
        }
        let tensor = shard_cache
            .get(shard)
            .and_then(|(weights, _)| weights.get(tensor_name))
            .ok_or_else(|| {
                SafeTensorsError::InvalidHeader(format!(
                    "index references missing tensor '{tensor_name}' in '{shard}'"
                ))
            })?;
        if merged.insert(tensor_name.clone(), tensor.clone()).is_some() {
            return Err(SafeTensorsError::InvalidHeader(format!(
                "duplicate tensor '{tensor_name}' in shard index"
            )));
        }
        let shape = shard_cache[shard]
            .1
            .get(tensor_name)
            .cloned()
            .ok_or_else(|| {
                SafeTensorsError::InvalidHeader(format!(
                    "missing dimensions for tensor '{tensor_name}'"
                ))
            })?;
        merged_shapes.insert(tensor_name.clone(), shape);
    }

    Ok((merged, merged_shapes))
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
    let shapes = tensors
        .iter()
        .map(|(name, values)| (name.clone(), vec![values.len()]))
        .collect();
    write_safetensors_with_shapes(tensors, &shapes, path)
}

/// Write F32 tensors while preserving their original shape metadata.
pub fn write_safetensors_with_shapes(
    tensors: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
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
        let shape = shapes.get(*name).ok_or_else(|| {
            SafeTensorsError::InvalidHeader(format!("missing shape for tensor '{name}'"))
        })?;
        let elements = shape
            .iter()
            .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
            .ok_or_else(|| {
                SafeTensorsError::InvalidHeader(format!("shape overflow for tensor '{name}'"))
            })?;
        if elements != data.len() {
            return Err(SafeTensorsError::InvalidHeader(format!(
                "shape for tensor '{name}' does not match its value count"
            )));
        }
        let byte_len = (data.len() * 4) as u64;
        tensor_metas.push(serde_json::json!({
            "dtype": "F32",
            "shape": shape,
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
    let padded_len = raw_len.div_ceil(8) * 8;
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

/// Canonical boundary size for SafeTensors shards (100 MiB).
pub const CANONICAL_MAX_SHARD_SIZE_BYTES: usize = 100 * 1024 * 1024;

/// Resolves the effective maximum shard boundary size dynamically from runtime configuration
/// (`TARA_MAX_SHARD_SIZE_BYTES`), falling back to the 100 MiB baseline if unspecified.
pub fn resolve_max_shard_size_bytes() -> usize {
    std::env::var("TARA_MAX_SHARD_SIZE_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CANONICAL_MAX_SHARD_SIZE_BYTES)
}

/// Write F32 tensors into either a single SafeTensors file or multiple canonical ~100MB shards with an index file.
pub fn write_safetensors_sharded(
    tensors: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    output_dir: &str,
    max_shard_size_bytes: usize,
) -> Result<Vec<String>, SafeTensorsError> {
    fs::create_dir_all(output_dir)?;

    let total_bytes: usize = tensors.values().map(|v| v.len() * 4).sum();
    if total_bytes <= max_shard_size_bytes {
        let single_file = format!("{}/model.safetensors", output_dir);
        write_safetensors_with_shapes(tensors, shapes, &single_file)?;
        return Ok(vec!["model.safetensors".to_string()]);
    }

    // Sort tensor names for deterministic sharding
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();

    let mut shards: Vec<WeightsWithShapes> = Vec::new();
    let mut current_shard_tensors = HashMap::new();
    let mut current_shard_shapes = HashMap::new();
    let mut current_shard_bytes = 0usize;

    for name in names {
        let t_data = tensors[name].clone();
        let t_shape = shapes
            .get(name)
            .cloned()
            .unwrap_or_else(|| vec![t_data.len()]);
        let t_bytes = t_data.len() * 4;

        if current_shard_bytes + t_bytes > max_shard_size_bytes && !current_shard_tensors.is_empty()
        {
            shards.push((current_shard_tensors, current_shard_shapes));
            current_shard_tensors = HashMap::new();
            current_shard_shapes = HashMap::new();
            current_shard_bytes = 0;
        }

        current_shard_bytes += t_bytes;
        current_shard_shapes.insert(name.clone(), t_shape);
        current_shard_tensors.insert(name.clone(), t_data);
    }

    if !current_shard_tensors.is_empty() {
        shards.push((current_shard_tensors, current_shard_shapes));
    }

    let num_shards = shards.len();
    let mut shard_filenames = Vec::new();
    let mut weight_map = serde_json::Map::new();

    for (idx, (shard_tensors, shard_shapes)) in shards.into_iter().enumerate() {
        let filename = format!("model-{:05}-of-{:05}.safetensors", idx + 1, num_shards);
        let shard_path = format!("{}/{}", output_dir, filename);
        write_safetensors_with_shapes(&shard_tensors, &shard_shapes, &shard_path)?;

        for tensor_name in shard_tensors.keys() {
            weight_map.insert(tensor_name.clone(), serde_json::json!(filename));
        }
        shard_filenames.push(filename);
    }

    let index_json = serde_json::json!({
        "metadata": {
            "total_size": total_bytes
        },
        "weight_map": weight_map
    });

    let index_path = format!("{}/model.safetensors.index.json", output_dir);
    fs::write(&index_path, serde_json::to_string_pretty(&index_json)?)?;

    Ok(shard_filenames)
}

#[cfg(test)]
mod tests {
    use super::{
        load_safetensors, load_safetensors_with_shapes, write_safetensors_with_shapes,
        SafeTensorsError,
    };
    use std::collections::HashMap;

    fn temp_file(name: &str) -> String {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir()
            .join(format!(
                "tara_{name}_{}_{}.safetensors",
                std::process::id(),
                stamp
            ))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn safetensors_writer_round_trips_tensor_values() {
        let path = temp_file("roundtrip");
        let mut input = HashMap::new();
        input.insert("weight".to_string(), vec![1.25, -2.0, 0.5]);
        let mut shapes = HashMap::new();
        shapes.insert("weight".to_string(), vec![1, 3]);
        write_safetensors_with_shapes(&input, &shapes, &path).expect("write SafeTensors");
        let output = load_safetensors(&path).expect("load SafeTensors");
        assert_eq!(output.get("weight"), input.get("weight"));
        let (_, loaded_shapes) = load_safetensors_with_shapes(&path).expect("load dimensions");
        assert_eq!(loaded_shapes.get("weight"), Some(&vec![1, 3]));
        std::fs::remove_file(path).expect("remove temporary file");
    }

    #[test]
    fn malformed_safetensors_header_is_rejected() {
        let path = temp_file("badheader");
        let mut bytes = 4u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"null");
        std::fs::write(&path, bytes).expect("write invalid fixture");
        assert!(matches!(
            load_safetensors(&path),
            Err(SafeTensorsError::InvalidHeader(_))
        ));
        std::fs::remove_file(path).expect("remove temporary file");
    }

    #[test]
    fn test_write_safetensors_sharded_round_trip() {
        use super::{load_model_weights_with_shapes, write_safetensors_sharded};
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let out_dir = std::env::temp_dir().join(format!("tara_sharded_{}", stamp));
        let out_dir_str = out_dir.to_string_lossy().to_string();

        let mut tensors = HashMap::new();
        tensors.insert("layer1".to_string(), vec![1.0f32; 100]); // 400 bytes
        tensors.insert("layer2".to_string(), vec![2.0f32; 100]); // 400 bytes
        tensors.insert("layer3".to_string(), vec![3.0f32; 100]); // 400 bytes

        let mut shapes = HashMap::new();
        shapes.insert("layer1".to_string(), vec![10, 10]);
        shapes.insert("layer2".to_string(), vec![10, 10]);
        shapes.insert("layer3".to_string(), vec![10, 10]);

        // Max shard size 500 bytes -> forces multiple shards!
        let shards = write_safetensors_sharded(&tensors, &shapes, &out_dir_str, 500).unwrap();
        assert!(
            shards.len() > 1,
            "Expected multiple shards, got {:?}",
            shards
        );
        assert!(out_dir.join("model.safetensors.index.json").exists());

        // Load using standard sharded loader
        let (loaded, loaded_shapes) = load_model_weights_with_shapes(&out_dir_str).unwrap();
        assert_eq!(loaded["layer1"], tensors["layer1"]);
        assert_eq!(loaded["layer2"], tensors["layer2"]);
        assert_eq!(loaded["layer3"], tensors["layer3"]);
        assert_eq!(loaded_shapes["layer1"], vec![10, 10]);

        let _ = std::fs::remove_dir_all(out_dir);
    }
}
