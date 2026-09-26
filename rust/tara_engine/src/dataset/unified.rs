//! Unified dataset compilation: merges all source datasets.

use serde_json::{json, Value};

use super::compiler::DynamicDatasetCompiler;

/// Compile all datasets in `repo_root/storage/datasets/` into a unified JSONL.
pub fn compile_unified_dataset(repo_root: &str) -> Value {
    let source_dir = format!("{}/storage/datasets", repo_root);
    let output_path = format!("{}/storage/datasets/unified_training.jsonl", repo_root);

    let compiler = DynamicDatasetCompiler::new(&source_dir, &output_path);
    match compiler.compile() {
        Ok(count) => json!({
            "status": "SUCCESS",
            "samples_compiled": count,
            "output_path": output_path
        }),
        Err(e) => json!({
            "status": "ERROR",
            "error": e.to_string()
        }),
    }
}
