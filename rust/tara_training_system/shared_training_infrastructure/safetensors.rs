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
use thiserror::Error;

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
    #[error("counter exhausted / generation overflow: {0}")]
    CounterExhausted(String),
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
    use std::io::{Read, Seek, SeekFrom};
    let mut file = fs::File::open(path)?;
    let file_len = file.metadata()?.len() as usize;

    if file_len < 8 {
        return Err(SafeTensorsError::Truncated {
            expected: 8,
            got: file_len,
        });
    }

    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    // 8-byte little-endian header length
    let header_len =
        usize::try_from(u64::from_le_bytes(len_buf)).map_err(|_| {
            SafeTensorsError::InvalidHeader("header length overflows address space".into())
        })?;
    let header_end = 8usize
        .checked_add(header_len)
        .ok_or_else(|| SafeTensorsError::InvalidHeader("header length overflow".into()))?;

    if file_len < header_end {
        return Err(SafeTensorsError::Truncated {
            expected: header_end,
            got: file_len,
        });
    }

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)?;
    let header_str = std::str::from_utf8(&header_bytes)
        .map_err(|e| SafeTensorsError::InvalidHeader(e.to_string()))?;

    let header: serde_json::Value = serde_json::from_str(header_str)?;
    let obj = header.as_object().ok_or_else(|| {
        SafeTensorsError::InvalidHeader("header root must be a JSON object".into())
    })?;

    let data_base = header_end;
    let data_len = file_len - data_base;
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
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| SafeTensorsError::BadOffsets(name.clone()))?;
        let end = offsets[1]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| SafeTensorsError::BadOffsets(name.clone()))?;

        if start > end || end > data_len {
            return Err(SafeTensorsError::Truncated {
                expected: data_base.saturating_add(end),
                got: file_len,
            });
        }

        ranges.push((start, end, name.clone()));
        let abs_start = (data_base + start) as u64;
        let tensor_bytes_len = end - start;
        file.seek(SeekFrom::Start(abs_start))?;
        let mut raw = vec![0u8; tensor_bytes_len];
        file.read_exact(&mut raw)?;
        let floats = convert_to_f32(dtype_str, &raw, name)?;
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

/// Load model tensors and preserve dimensions, tracing the loader path and purpose (INPUT, RESUME, or OUTPUT).
pub fn load_model_weights_with_purpose(
    model_dir: &str,
    purpose: &str,
) -> Result<WeightsWithShapes, SafeTensorsError> {
    println!("[SafeTensors Loader] Path: '{}' | Purpose: {}", model_dir, purpose);
    let index_path = format!("{}/model.safetensors.index.json", model_dir);
    if std::path::Path::new(&index_path).exists() {
        load_sharded_weights_with_purpose(model_dir, &index_path, purpose)
    } else {
        let single_path = format!("{}/model.safetensors", model_dir);
        println!("[SafeTensors Shard Loader] Shard: '{}' | Purpose: {}", single_path, purpose);
        load_safetensors_with_shapes(&single_path)
    }
}

/// Read tensor shapes and dtypes from a SafeTensors file without loading tensor payload data.
pub fn read_safetensors_shapes(path: &str) -> Result<HashMap<String, Vec<usize>>, SafeTensorsError> {
    use std::io::Read;
    let mut file = fs::File::open(path)?;
    let file_len = file.metadata()?.len() as usize;

    if file_len < 8 {
        return Err(SafeTensorsError::Truncated {
            expected: 8,
            got: file_len,
        });
    }

    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = usize::try_from(u64::from_le_bytes(len_buf)).map_err(|_| {
        SafeTensorsError::InvalidHeader("header length overflows address space".into())
    })?;
    let header_end = 8usize
        .checked_add(header_len)
        .ok_or_else(|| SafeTensorsError::InvalidHeader("header length overflow".into()))?;

    if file_len < header_end {
        return Err(SafeTensorsError::Truncated {
            expected: header_end,
            got: file_len,
        });
    }

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)?;
    let header_str = std::str::from_utf8(&header_bytes)
        .map_err(|e| SafeTensorsError::InvalidHeader(e.to_string()))?;

    let header: serde_json::Value = serde_json::from_str(header_str)?;
    let obj = header.as_object().ok_or_else(|| {
        SafeTensorsError::InvalidHeader("header root must be a JSON object".into())
    })?;

    let mut shapes = HashMap::new();
    for (name, meta) in obj {
        if name == "__metadata__" {
            continue;
        }
        let shape_arr = meta
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                SafeTensorsError::InvalidHeader(format!("missing shape for tensor '{name}'"))
            })?;
        let mut shape_vec = Vec::with_capacity(shape_arr.len());
        for dim in shape_arr {
            let d = dim
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| {
                    SafeTensorsError::InvalidHeader(format!("invalid shape for tensor '{name}'"))
                })?;
            shape_vec.push(d);
        }
        shapes.insert(name.clone(), shape_vec);
    }
    Ok(shapes)
}

/// Read tensor shapes for an entire model directory (sharded via index.json or single file)
/// without loading any float payload into memory.
pub fn read_model_metadata_shapes(
    model_dir: &str,
) -> Result<HashMap<String, Vec<usize>>, SafeTensorsError> {
    let index_path = format!("{}/model.safetensors.index.json", model_dir);
    if std::path::Path::new(&index_path).exists() {
        let raw = fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&raw)?;
        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                SafeTensorsError::InvalidHeader(
                    "sharded index is missing 'weight_map' object".into(),
                )
            })?;

        let mut shard_files = std::collections::BTreeSet::new();
        for file_val in weight_map.values() {
            if let Some(f_str) = file_val.as_str() {
                shard_files.insert(f_str.to_string());
            }
        }
        let root = std::path::Path::new(model_dir).canonicalize()?;
        let mut all_shapes = HashMap::new();
        for shard_name in shard_files {
            let shard_path = root.join(&shard_name);
            let shapes = read_safetensors_shapes(&shard_path.to_string_lossy())?;
            all_shapes.extend(shapes);
        }
        Ok(all_shapes)
    } else {
        let single_path = format!("{}/model.safetensors", model_dir);
        read_safetensors_shapes(&single_path)
    }
}

/// Load model tensors and preserve the dimensions from the source files.
pub fn load_model_weights_with_shapes(
    model_dir: &str,
) -> Result<WeightsWithShapes, SafeTensorsError> {
    load_model_weights_with_purpose(model_dir, "INPUT")
}

/// Load a sharded model from its index file with purpose tracing.
fn load_sharded_weights_with_purpose(
    model_dir: &str,
    index_path: &str,
    purpose: &str,
) -> Result<WeightsWithShapes, SafeTensorsError> {
    let raw = fs::read_to_string(index_path)?;
    let index: serde_json::Value = serde_json::from_str(&raw)?;

    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            SafeTensorsError::InvalidHeader(
                "sharded index is missing 'weight_map' object".into(),
            )
        })?;

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
            println!("[SafeTensors Shard Loader] Shard: '{}' | Purpose: {}", full.display(), purpose);
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
    tara_security::cryptography::CryptoUtils::compute256(std::path::Path::new(path))
        .map_err(|e| SafeTensorsError::Io(std::io::Error::other(e)))
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
        let byte_len = (data.len() as u64).checked_mul(4).ok_or_else(|| {
            SafeTensorsError::InvalidHeader(format!("byte length overflow for tensor '{name}'"))
        })?;
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

    let file = fs::File::create(path)?;
    let mut out = std::io::BufWriter::with_capacity(16 * 1024 * 1024, file);
    out.write_all(&len_bytes)?;
    out.write_all(header_json.as_bytes())?;
    if padding > 0 {
        out.write_all(&vec![b' '; padding])?;
    }

    // Write data with chunked buffer for high-throughput sequential I/O
    let mut chunk = Vec::with_capacity(65536);
    for name in &ordered_keys {
        let data = &tensors[*name];
        for &v in data.iter() {
            chunk.extend_from_slice(&v.to_le_bytes());
            if chunk.len() >= 65536 {
                out.write_all(&chunk)?;
                chunk.clear();
            }
        }
    }
    if !chunk.is_empty() {
        out.write_all(&chunk)?;
        chunk.clear();
    }
    out.flush()?;
    out.get_ref().sync_all()?;

    Ok(())
}

/// Write BF16 tensors while preserving their original shape metadata.
pub fn write_safetensors_bf16_with_shapes(
    tensors: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    path: &str,
) -> Result<(), SafeTensorsError> {
    use std::io::Write;

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
        let byte_len = (data.len() as u64).checked_mul(2).ok_or_else(|| {
            SafeTensorsError::InvalidHeader(format!("byte length overflow for tensor '{name}'"))
        })?;
        tensor_metas.push(serde_json::json!({
            "dtype": "BF16",
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

    let file = fs::File::create(path)?;
    let mut out = std::io::BufWriter::with_capacity(16 * 1024 * 1024, file);
    out.write_all(&len_bytes)?;
    out.write_all(header_json.as_bytes())?;
    if padding > 0 {
        out.write_all(&vec![b' '; padding])?;
    }

    // Write BF16 data
    let mut chunk = Vec::with_capacity(65536);
    for name in &ordered_keys {
        let data = &tensors[*name];
        for &v in data.iter() {
            let bf = half::bf16::from_f32(v).to_bits();
            chunk.extend_from_slice(&bf.to_le_bytes());
            if chunk.len() >= 65536 {
                out.write_all(&chunk)?;
                chunk.clear();
            }
        }
    }
    if !chunk.is_empty() {
        out.write_all(&chunk)?;
        chunk.clear();
    }
    out.flush()?;
    out.get_ref().sync_all()?;

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

/// Write tensors into sharded SafeTensors files with configurable prefix, precision, and telemetry callback.
/// Streaming serializer: partitions tensor metadata into shards with ZERO heap cloning of tensor data.
/// Writes directly to disk shard-by-shard with temporary files and atomic rename to guarantee consistency.
pub fn write_safetensors_sharded_with_callback<F>(
    tensors: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    output_dir: &str,
    max_shard_size_bytes: usize,
    prefix: &str,
    is_bf16: bool,
    mut on_diagnostic: F,
) -> Result<Vec<String>, SafeTensorsError>
where
    F: FnMut(&str),
{
    use std::io::Write;

    on_diagnostic("PRE_CHECKPOINT_SERIALIZATION");
    fs::create_dir_all(output_dir)?;

    let bytes_per_elem = if is_bf16 { 2 } else { 4 };
    let total_bytes: usize = tensors.values().map(|v| v.len() * bytes_per_elem).sum();

    // Sort tensor names for deterministic sharding
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();

    // Zero-copy sharding plan: stores only references to tensor names without copying float buffers
    let mut shard_plans: Vec<Vec<&String>> = Vec::new();
    let mut current_shard_names: Vec<&String> = Vec::new();
    let mut current_shard_bytes = 0usize;

    for &name in &names {
        let t_len = tensors[name].len();
        let t_bytes = t_len * bytes_per_elem;

        if current_shard_bytes + t_bytes > max_shard_size_bytes && !current_shard_names.is_empty() {
            shard_plans.push(std::mem::take(&mut current_shard_names));
            current_shard_bytes = 0;
        }

        current_shard_bytes += t_bytes;
        current_shard_names.push(name);
    }

    if !current_shard_names.is_empty() {
        shard_plans.push(current_shard_names);
    }

    let mut shard_filenames = Vec::new();
    let mut weight_map = serde_json::Map::new();

    for (idx, shard_names) in shard_plans.into_iter().enumerate() {
        let op_pre = format!("PRE_SHARD_WRITE_{}", idx + 1);
        on_diagnostic(&op_pre);

        let filename = format!("{}-{:05}.safetensors", prefix, idx + 1);
        let shard_path = format!("{}/{}", output_dir, filename);
        let tmp_shard_path = format!("{}/{}.tmp", output_dir, filename);

        // Build header JSON for this shard from metadata only
        let mut offset: u64 = 0;
        let mut header_obj = serde_json::Map::new();

        for &name in &shard_names {
            let data = &tensors[name];
            let default_shape = vec![data.len()];
            let shape = shapes.get(name).unwrap_or(&default_shape);
            let byte_len = (data.len() as u64)
                .checked_mul(bytes_per_elem as u64)
                .ok_or_else(|| {
                    SafeTensorsError::InvalidHeader(format!("byte length overflow for tensor '{name}'"))
                })?;

            let dtype_str = if is_bf16 { "BF16" } else { "F32" };
            header_obj.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": dtype_str,
                    "shape": shape,
                    "data_offsets": [offset, offset + byte_len]
                }),
            );
            offset += byte_len;
        }

        let header_json = serde_json::to_string(&header_obj)?;

        // Pad to 8-byte boundary per SafeTensors spec
        let raw_len = header_json.len();
        let padded_len = raw_len.div_ceil(8) * 8;
        let padding = padded_len - raw_len;

        let header_len = padded_len as u64;
        let len_bytes = header_len.to_le_bytes();

        let file = fs::File::create(&tmp_shard_path)?;
        let mut out = std::io::BufWriter::with_capacity(4 * 1024 * 1024, file);
        out.write_all(&len_bytes)?;
        out.write_all(header_json.as_bytes())?;
        if padding > 0 {
            out.write_all(&vec![b' '; padding])?;
        }

        // Stream tensor data directly from source references with bounded 64KB buffer
        let mut chunk = Vec::with_capacity(65536);
        for &name in &shard_names {
            let data = &tensors[name];
            if is_bf16 {
                for &v in data.iter() {
                    let bf = half::bf16::from_f32(v).to_bits();
                    chunk.extend_from_slice(&bf.to_le_bytes());
                    if chunk.len() >= 65536 {
                        out.write_all(&chunk)?;
                        chunk.clear();
                    }
                }
            } else {
                for &v in data.iter() {
                    chunk.extend_from_slice(&v.to_le_bytes());
                    if chunk.len() >= 65536 {
                        out.write_all(&chunk)?;
                        chunk.clear();
                    }
                }
            }
        }
        if !chunk.is_empty() {
            out.write_all(&chunk)?;
            drop(chunk);
        }

        out.flush()?;
        out.get_ref().sync_all()?;
        drop(out);

        let op_flush = format!("POST_SHARD_FLUSH_{}", idx + 1);
        on_diagnostic(&op_flush);

        // Atomically rename temporary shard file
        fs::rename(&tmp_shard_path, &shard_path)?;

        for &tensor_name in &shard_names {
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
    let tmp_index_path = format!("{}/model.safetensors.index.json.tmp", output_dir);
    let mut idx_file = fs::File::create(&tmp_index_path)?;
    idx_file.write_all(serde_json::to_string_pretty(&index_json)?.as_bytes())?;
    idx_file.flush()?;
    idx_file.sync_all()?;
    drop(idx_file);
    fs::rename(&tmp_index_path, &index_path)?;

    on_diagnostic("POST_CHECKPOINT_SERIALIZATION");

    Ok(shard_filenames)
}

/// Write tensors into sharded SafeTensors files with configurable prefix and precision.
pub fn write_safetensors_sharded_with_prefix(
    tensors: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    output_dir: &str,
    max_shard_size_bytes: usize,
    prefix: &str,
    is_bf16: bool,
) -> Result<Vec<String>, SafeTensorsError> {
    write_safetensors_sharded_with_callback(
        tensors,
        shapes,
        output_dir,
        max_shard_size_bytes,
        prefix,
        is_bf16,
        |_| {},
    )
}

/// Write F32 tensors into either a single SafeTensors file or multiple canonical shards with an index file.
pub fn write_safetensors_sharded(
    tensors: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    output_dir: &str,
    max_shard_size_bytes: usize,
) -> Result<Vec<String>, SafeTensorsError> {
    write_safetensors_sharded_with_prefix(
        tensors,
        shapes,
        output_dir,
        max_shard_size_bytes,
        "model",
        false,
    )
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
