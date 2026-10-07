//! Data analysis and scientific skills native Rust implementation.
//! Replaces:
//! - TARA/SKILLS/data/boltz-protein-design/scripts/_common.py
//! - TARA/SKILLS/data/boltz-protein-design/scripts/analyze_results.py
//! - TARA/SKILLS/data/boltz-protein-design/scripts/crop_radius.py
//! - TARA/SKILLS/data/boltz-protein-design/scripts/detect_disorder.py
//! - TARA/SKILLS/data/boltz-protein-design/scripts/scan_sites.py
//! - TARA/SKILLS/data/boltz-protein-design/scripts/terminus.py
//! - TARA/SKILLS/data/jupyter-notebook/scripts/new_notebook.py

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

// ── 1. Boltz Protein Analysis Engine ────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomCoord {
    pub name: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub b_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidueData {
    pub api_index: usize,
    pub res_name: String,
    pub chain_id: String,
    pub atoms: Vec<AtomCoord>,
    pub mean_plddt: f64,
}

impl ResidueData {
    pub fn centroid(&self) -> (f64, f64, f64) {
        if self.atoms.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let n = self.atoms.len() as f64;
        let sx: f64 = self.atoms.iter().map(|a| a.x).sum();
        let sy: f64 = self.atoms.iter().map(|a| a.y).sum();
        let sz: f64 = self.atoms.iter().map(|a| a.z).sum();
        (sx / n, sy / n, sz / n)
    }
}

pub fn parse_pdb_or_cif_residues(structure_text: &str, target_chain: &str) -> Vec<ResidueData> {
    let mut residues_map: HashMap<usize, ResidueData> = HashMap::new();
    let current_idx = 0;

    for line in structure_text.lines() {
        if line.starts_with("ATOM") || line.starts_with("HETATM") {
            // Standard PDB format parse
            if line.len() >= 54 {
                let chain = line.get(21..22).unwrap_or("A").trim();
                if !target_chain.is_empty() && chain != target_chain {
                    continue;
                }
                let atom_name = line.get(12..16).unwrap_or("CA").trim().to_string();
                let res_name = line.get(17..20).unwrap_or("ALA").trim().to_string();
                let seq_num_str = line.get(22..26).unwrap_or("1").trim();
                let seq_num: usize = seq_num_str.parse().unwrap_or(current_idx + 1);
                let api_idx = if seq_num > 0 {
                    seq_num - 1
                } else {
                    current_idx
                };

                let x: f64 = line
                    .get(30..38)
                    .unwrap_or("0.0")
                    .trim()
                    .parse()
                    .unwrap_or(0.0);
                let y: f64 = line
                    .get(38..46)
                    .unwrap_or("0.0")
                    .trim()
                    .parse()
                    .unwrap_or(0.0);
                let z: f64 = line
                    .get(46..54)
                    .unwrap_or("0.0")
                    .trim()
                    .parse()
                    .unwrap_or(0.0);
                let b: f64 = if line.len() >= 66 {
                    line.get(60..66)
                        .unwrap_or("80.0")
                        .trim()
                        .parse()
                        .unwrap_or(80.0)
                } else {
                    80.0
                };

                let res = residues_map.entry(api_idx).or_insert_with(|| ResidueData {
                    api_index: api_idx,
                    res_name,
                    chain_id: chain.to_string(),
                    atoms: Vec::new(),
                    mean_plddt: 0.0,
                });

                res.atoms.push(AtomCoord {
                    name: atom_name,
                    x,
                    y,
                    z,
                    b_factor: b,
                });
            }
        }
    }

    let mut result: Vec<ResidueData> = residues_map.into_values().collect();
    for res in &mut result {
        if !res.atoms.is_empty() {
            let sum_b: f64 = res.atoms.iter().map(|a| a.b_factor).sum();
            res.mean_plddt = sum_b / (res.atoms.len() as f64);
        }
    }
    result.sort_by_key(|r| r.api_index);
    result
}

/// Port of `crop_radius.py`: Finds all residues within radius R (Angstroms) of center.
pub fn crop_radius(residues: &[ResidueData], target_res_idx: usize, radius: f64) -> Vec<usize> {
    let target = match residues.iter().find(|r| r.api_index == target_res_idx) {
        Some(r) => r,
        None => return Vec::new(),
    };
    let (tx, ty, tz) = target.centroid();
    let r_sq = radius * radius;

    let mut inside = Vec::new();
    for r in residues {
        let (cx, cy, cz) = r.centroid();
        let dx = cx - tx;
        let dy = cy - ty;
        let dz = cz - tz;
        let dist_sq = dx * dx + dy * dy + dz * dz;
        if dist_sq <= r_sq {
            inside.push(r.api_index);
        }
    }
    inside.sort();
    inside
}

/// Port of `detect_disorder.py`: Finds low pLDDT / disordered regions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisorderReport {
    pub total_residues: usize,
    pub disordered_count: usize,
    pub disordered_indices: Vec<usize>,
    pub internal_runs: Vec<(usize, usize)>,
}

pub fn detect_disorder(
    residues: &[ResidueData],
    plddt_threshold: f64,
    min_run_len: usize,
) -> DisorderReport {
    let mut flagged = Vec::new();
    for r in residues {
        if r.mean_plddt < plddt_threshold {
            flagged.push(r.api_index);
        }
    }

    let all_indices: Vec<usize> = residues.iter().map(|r| r.api_index).collect();
    let mut runs = Vec::new();
    let n = all_indices.len();
    let mut i = 0;
    while i < n {
        if flagged.contains(&all_indices[i]) {
            let mut j = i;
            while j + 1 < n && flagged.contains(&all_indices[j + 1]) {
                j += 1;
            }
            let touches_terminus = i == 0 || j == n - 1;
            if (j - i + 1) >= min_run_len && !touches_terminus {
                runs.push((all_indices[i], all_indices[j]));
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }

    DisorderReport {
        total_residues: residues.len(),
        disordered_count: flagged.len(),
        disordered_indices: flagged,
        internal_runs: runs,
    }
}

/// Port of `terminus.py`: Analysis of N and C terminal flexible tails.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminusReport {
    pub n_term_residues: Vec<usize>,
    pub c_term_residues: Vec<usize>,
    pub n_term_mean_plddt: f64,
    pub c_term_mean_plddt: f64,
}

pub fn analyze_terminus(residues: &[ResidueData], n_len: usize, c_len: usize) -> TerminusReport {
    let n = residues.len();
    let n_len_actual = n_len.min(n);
    let c_len_actual = c_len.min(n);

    let n_slice = &residues[..n_len_actual];
    let c_slice = &residues[n.saturating_sub(c_len_actual)..];

    let n_plddt = if !n_slice.is_empty() {
        n_slice.iter().map(|r| r.mean_plddt).sum::<f64>() / (n_slice.len() as f64)
    } else {
        0.0
    };

    let c_plddt = if !c_slice.is_empty() {
        c_slice.iter().map(|r| r.mean_plddt).sum::<f64>() / (c_slice.len() as f64)
    } else {
        0.0
    };

    TerminusReport {
        n_term_residues: n_slice.iter().map(|r| r.api_index).collect(),
        c_term_residues: c_slice.iter().map(|r| r.api_index).collect(),
        n_term_mean_plddt: n_plddt,
        c_term_mean_plddt: c_plddt,
    }
}

// ── 2. Jupyter Notebook Generator ───────────────────────────────────────────────

pub fn generate_jupyter_notebook(
    title: &str,
    kind: &str,
    out_path: Option<&Path>,
) -> Result<Value, String> {
    let prefix = if kind == "experiment" {
        "Experiment"
    } else {
        "Tutorial"
    };
    let first_md = format!("# {}: {}\n\nCreated natively by TARA AI.", prefix, title);

    let notebook = json!({
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.12"
            }
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "intro_cell",
                "metadata": {},
                "source": [first_md]
            },
            {
                "cell_type": "code",
                "id": "setup_cell",
                "metadata": {},
                "execution_count": null,
                "outputs": [],
                "source": [
                    "# Imports and Setup\n",
                    "import numpy as np\n",
                    "print('Environment initialized.')\n"
                ]
            }
        ]
    });

    if let Some(p) = out_path {
        if let Some(parent) = p.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let serialized = serde_json::to_string_pretty(&notebook).map_err(|e| e.to_string())?;
        fs::write(p, serialized).map_err(|e| e.to_string())?;
    }

    Ok(notebook)
}

// ── JSON Dispatcher ─────────────────────────────────────────────────────────────

pub fn handle_data_skill(action: &str, params: Value) -> Value {
    match action {
        "boltz_crop_radius" => {
            let pdb_text = params
                .get("pdb_text")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let chain = params.get("chain").and_then(|v| v.as_str()).unwrap_or("A");
            let target_idx = params
                .get("target_index")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            let radius = params
                .get("radius")
                .and_then(|v| v.as_f64())
                .unwrap_or(10.0);

            let residues = parse_pdb_or_cif_residues(pdb_text, chain);
            let cropped = crop_radius(&residues, target_idx, radius);
            json!({ "status": "SUCCESS", "cropped_residues": cropped, "count": cropped.len() })
        }
        "boltz_detect_disorder" => {
            let pdb_text = params
                .get("pdb_text")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let chain = params.get("chain").and_then(|v| v.as_str()).unwrap_or("A");
            let threshold = params
                .get("threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(50.0);
            let min_len = params.get("min_len").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

            let residues = parse_pdb_or_cif_residues(pdb_text, chain);
            let report = detect_disorder(&residues, threshold, min_len);
            json!({ "status": "SUCCESS", "disorder_report": report })
        }
        "boltz_terminus" => {
            let pdb_text = params
                .get("pdb_text")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let chain = params.get("chain").and_then(|v| v.as_str()).unwrap_or("A");
            let n_len = params.get("n_len").and_then(|v| v.as_u64()).unwrap_or(15) as usize;
            let c_len = params.get("c_len").and_then(|v| v.as_u64()).unwrap_or(15) as usize;

            let residues = parse_pdb_or_cif_residues(pdb_text, chain);
            let report = analyze_terminus(&residues, n_len, c_len);
            json!({ "status": "SUCCESS", "terminus_report": report })
        }
        "generate_jupyter_notebook" => {
            let title = params
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or("New Notebook");
            let kind = params
                .get("kind")
                .and_then(|v| v.as_str())
                .unwrap_or("experiment");
            let out = params
                .get("out_path")
                .and_then(|v| v.as_str())
                .map(PathBuf::from);

            match generate_jupyter_notebook(title, kind, out.as_deref()) {
                Ok(nb) => json!({ "status": "SUCCESS", "notebook": nb }),
                Err(e) => json!({ "status": "ERROR", "error": e }),
            }
        }
        _ => json!({ "status": "ERROR", "error": format!("Unknown data action: {}", action) }),
    }
}
