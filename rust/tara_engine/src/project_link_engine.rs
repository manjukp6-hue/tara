//! # Project Link & Reference Engine (100% Native Rust)
//!
//! Authoritative relationship tracking, reconciliation, and synchronization
//! for TARA Core and all workspace crates.
//!
//! ## Core Responsibilities:
//! - Full initial indexing of all Rust files, modules, imports, exports, declarations, and call references.
//! - Inbound (in-link) and Outbound (out-link) dependency graph tracking.
//! - Automatic relationship reconciliation when files are created, modified, moved, or deleted.
//! - Automatic link reconnection when modules move across directory branches.
//! - Native Lexical AST syntax parsing & module-level path dependency resolution without regex crutches.
//! - Graph validation with snapshot-based atomic rollback.
//! - Full synchronization with `ARCHITECTURE/SOURCE_INDEX.json` and `RELATIONSHIP_GRAPH.json`.
//! - Pure dynamic runtime SHA-256 computation (zero static literals).
//! - 100% native Rust, zero Python scripts, zero hardcoded limits.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Kinds of Rust items recognized by the engine.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SymbolKind {
    Struct,
    Enum,
    Trait,
    Function,
    TypeAlias,
    Const,
    Static,
    Module,
}

impl SymbolKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Struct => "struct",
            Self::Enum => "enum",
            Self::Trait => "trait",
            Self::Function => "function",
            Self::TypeAlias => "type",
            Self::Const => "const",
            Self::Static => "static",
            Self::Module => "module",
        }
    }
}

/// Item visibility in Rust source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Visibility {
    Public,
    PubCrate,
    PubSuper,
    Private,
}

/// Declared symbol (struct, enum, function, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolDeclaration {
    pub name: String,
    pub kind: SymbolKind,
    pub visibility: Visibility,
    pub line_number: usize,
}

/// Single import statement (`use ...`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportStatement {
    pub raw_path: String,
    pub is_public_reexport: bool,
    pub imported_symbols: Vec<String>,
    pub line_number: usize,
}

/// Submodule declaration (`mod ...;`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDeclaration {
    pub name: String,
    pub is_public: bool,
    pub custom_path: Option<String>,
    pub line_number: usize,
}

/// Complete AST record for a single Rust source file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustFileNode {
    pub rel_path: String,
    pub crate_name: String,
    pub module_path: String,
    pub lines: usize,
    pub dynamic_sha256: String,
    pub is_test: bool,
    pub modules_declared: Vec<ModuleDeclaration>,
    pub imports: Vec<ImportStatement>,
    pub exports: Vec<SymbolDeclaration>,
    pub internal_symbols: Vec<SymbolDeclaration>,
    pub call_references: BTreeSet<String>,
    pub out_links: BTreeSet<String>, // Paths of files imported/called by this file
    pub in_links: BTreeSet<String>,  // Paths of files that import/call this file
}

/// Change delta reported upon reconciliation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RelationshipDelta {
    pub file_path: String,
    pub action: String,
    pub added_imports: Vec<String>,
    pub removed_imports: Vec<String>,
    pub added_exports: Vec<String>,
    pub removed_exports: Vec<String>,
    pub affected_in_links: Vec<String>,
    pub affected_out_links: Vec<String>,
}

/// Snapshot of the complete relationship graph for validation & rollback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipGraphSnapshot {
    pub snapshot_id: String,
    pub created_at_utc: String,
    pub files: BTreeMap<String, RustFileNode>,
}

/// Comprehensive graph report after indexing or synchronization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipReport {
    pub total_rust_files: usize,
    pub total_modules: usize,
    pub total_symbols: usize,
    pub total_imports: usize,
    pub total_in_out_links: usize,
    pub duration_ms: u64,
}

/// Config for Project Link & Reference Engine.
#[derive(Debug, Clone)]
pub struct ProjectLinkConfig {
    pub workspace_root: PathBuf,
    pub arch_dir: PathBuf,
    pub ignored_dirs: Vec<String>,
    pub io_buffer_size: usize,
}

impl Default for ProjectLinkConfig {
    fn default() -> Self {
        let ws = find_workspace_root().unwrap_or_else(|_| PathBuf::from("."));
        let arch = ws.join("ARCHITECTURE");
        let io_buf = std::env::var("TARA_SYNC_IO_BUF_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64 * 1024);

        Self {
            workspace_root: ws,
            arch_dir: arch,
            ignored_dirs: vec![
                "target".to_string(),
                ".git".to_string(),
                ".gemini".to_string(),
                "brain".to_string(),
                "scratch".to_string(),
                "storage".to_string(),
                ".vscode".to_string(),
                ".idea".to_string(),
            ],
            io_buffer_size: io_buf,
        }
    }
}

/// The Core Project Link & Reference Engine.
pub struct ProjectLinkEngine {
    pub config: ProjectLinkConfig,
    pub files: BTreeMap<String, RustFileNode>,
    pub module_to_file: BTreeMap<String, String>, // module_path -> rel_path
    pub symbol_to_files: BTreeMap<String, BTreeSet<String>>, // symbol_name -> Set<rel_path>
    pub known_crates: BTreeSet<String>,
}

impl ProjectLinkEngine {
    pub fn new(config: ProjectLinkConfig) -> Self {
        Self {
            config,
            files: BTreeMap::new(),
            module_to_file: BTreeMap::new(),
            symbol_to_files: BTreeMap::new(),
            known_crates: BTreeSet::new(),
        }
    }

    /// Finds workspace root containing AGENTS.md and Cargo.toml.
    pub fn workspace_root(&self) -> &Path {
        &self.config.workspace_root
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 1. FULL INITIAL INDEX (RULE 2)
    // ──────────────────────────────────────────────────────────────────────────

    /// Scans, parses, and indexes every Rust file in the project.
    pub fn build_full_index(&mut self) -> Result<RelationshipReport, Box<dyn std::error::Error>> {
        let start = std::time::Instant::now();
        let ws = self.config.workspace_root.clone();

        self.files.clear();
        self.module_to_file.clear();
        self.symbol_to_files.clear();
        self.known_crates.clear();

        // 1. Discover all .rs files
        let mut rs_paths = Vec::new();
        self.discover_rust_files(&ws, &ws, &mut rs_paths)?;

        // 2. Discover workspace crates dynamically from Cargo.toml manifests
        self.discover_workspace_crates(&ws);

        // 3. Parse AST for each file
        for rel_path in &rs_paths {
            let full_path = ws.join(rel_path);
            if let Ok(node) = self.parse_rust_file(&full_path, rel_path) {
                self.known_crates.insert(node.crate_name.clone());
                self.module_to_file.insert(node.module_path.clone(), rel_path.clone());

                for sym in &node.exports {
                    self.symbol_to_files
                        .entry(sym.name.clone())
                        .or_default()
                        .insert(rel_path.clone());
                }

                self.known_crates.insert(node.crate_name.clone());
                self.files.insert(rel_path.clone(), node);
            }
        }

        // 4. Register declared submodules and mount custom paths dynamically
        let declared_submodules: Vec<(String, String, String, String)> = self
            .files
            .values()
            .flat_map(|node| {
                node.modules_declared.iter().map(|m| {
                    (
                        node.rel_path.clone(),
                        node.crate_name.clone(),
                        node.module_path.clone(),
                        m.name.clone(),
                        m.custom_path.clone(),
                    )
                })
            })
            .filter_map(|(src_path, src_crate, src_mod, mod_name, custom_path)| {
                let mod_decl = ModuleDeclaration {
                    name: mod_name.clone(),
                    is_public: true,
                    custom_path: custom_path.clone(),
                    line_number: 0,
                };
                if let Some(src_node) = self.files.get(&src_path) {
                    if let Some(target) = self.resolve_submodule_target(src_node, &mod_decl) {
                        return Some((src_crate, src_mod, mod_name, target));
                    }
                }
                None
            })
            .collect();

        for (src_crate, src_mod, mod_name, target_path) in declared_submodules {
            let logical_mod = if src_mod == src_crate {
                format!("{}::{}", src_crate, mod_name)
            } else {
                format!("{}::{}", src_mod, mod_name)
            };
            self.module_to_file.insert(logical_mod.clone(), target_path.clone());
            self.module_to_file.insert(format!("{}::{}", src_crate, mod_name), target_path.clone());
            self.module_to_file.insert(mod_name.clone(), target_path.clone());
        }

        // 5. Resolve Cross-Module In-Links and Out-Links
        self.resolve_all_links();

        let duration_ms = start.elapsed().as_millis() as u64;

        let mut total_symbols = 0;
        let mut total_imports = 0;
        let mut total_links = 0;

        for node in self.files.values() {
            total_symbols += node.exports.len() + node.internal_symbols.len();
            total_imports += node.imports.len();
            total_links += node.out_links.len() + node.in_links.len();
        }

        Ok(RelationshipReport {
            total_rust_files: self.files.len(),
            total_modules: self.module_to_file.len(),
            total_symbols,
            total_imports,
            total_in_out_links: total_links,
            duration_ms,
        })
    }

    /// Recursively discovers all .rs files in non-ignored directories.
    fn discover_rust_files(
        &self,
        root: &Path,
        current: &Path,
        out: &mut Vec<String>,
    ) -> std::io::Result<()> {
        let rel_folder = if current == root {
            String::new()
        } else {
            let rel = current.strip_prefix(root).unwrap_or(current);
            normalize_rel_path(rel)
        };

        if !rel_folder.is_empty() && self.is_ignored(&rel_folder) {
            return Ok(());
        }

        let entries = match fs::read_dir(current) {
            Ok(e) => e,
            Err(_) => return Ok(()),
        };

        for entry_res in entries {
            let entry = match entry_res {
                Ok(e) => e,
                Err(_) => continue,
            };
            let path = entry.path();
            let ft = match entry.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };

            if ft.is_dir() {
                let name = entry.file_name().to_string_lossy().to_string();
                if !self.is_ignored(&name) {
                    self.discover_rust_files(root, &path, out)?;
                }
            } else if ft.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "rs" {
                        let rel = path.strip_prefix(root).unwrap_or(&path);
                        let norm = normalize_rel_path(rel);
                        if !self.is_ignored(&norm) {
                            out.push(norm);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn is_ignored(&self, path_str: &str) -> bool {
        let norm = path_str.replace('\\', "/");
        let parts: Vec<&str> = norm.split('/').collect();
        for ignored in &self.config.ignored_dirs {
            if parts.contains(&ignored.as_str()) {
                return true;
            }
        }
        false
    }

    /// Dynamically discovers workspace crates by finding Cargo.toml manifests anywhere in the project.
    pub fn discover_workspace_crates(&mut self, root: &Path) {
        if let Ok(entries) = fs::read_dir(root) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    let folder_name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    if self.is_ignored(folder_name) {
                        continue;
                    }
                    self.discover_workspace_crates_recursive(&p);
                }
            }
        }
    }

    fn discover_workspace_crates_recursive(&mut self, dir: &Path) {
        let cargo_toml = dir.join("Cargo.toml");
        if cargo_toml.is_file() {
            if let Ok(content) = fs::read_to_string(&cargo_toml) {
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("name") && trimmed.contains('=') {
                        let parts: Vec<&str> = trimmed.split('=').collect();
                        if parts.len() == 2 {
                            let name = parts[1].trim().trim_matches('"').trim_matches('\'').trim();
                            if !name.is_empty() {
                                self.known_crates.insert(name.to_string());
                                break;
                            }
                        }
                    }
                }
            }
        }

        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    if !self.is_ignored(name) {
                        self.discover_workspace_crates_recursive(&p);
                    }
                }
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 2. NATIVE LEXICAL AST SYNTAX & MODULE PATH RESOLVER (RULE 5)
    // ──────────────────────────────────────────────────────────────────────────

    /// Parses a single Rust file using native lexical and AST extraction.
    pub fn parse_rust_file(
        &self,
        full_path: &Path,
        rel_path: &str,
    ) -> Result<RustFileNode, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(full_path)?;
        let lines_count = content.lines().count();
        let sha256 = compute_dynamic_sha256(full_path, self.config.io_buffer_size)?;

        let crate_name = detect_crate_name(&self.config.workspace_root, rel_path);
        let module_path = derive_module_path(rel_path, &crate_name);
        let is_test = rel_path.contains("/tests/") || rel_path.ends_with("_tests.rs") || content.contains("#[test]");

        let mut parser = NativeRustAstParser::new(&content);
        let parsed = parser.parse();

        // Enforce canonical total ordering on all inner AST vectors (Content-Deterministic Serialization)
        let mut modules_declared = parsed.modules_declared;
        modules_declared.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.line_number.cmp(&b.line_number)));

        let mut imports = parsed.imports;
        imports.sort_by(|a, b| a.raw_path.cmp(&b.raw_path).then_with(|| a.line_number.cmp(&b.line_number)));

        let mut exports = parsed.exports;
        exports.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.line_number.cmp(&b.line_number)));

        let mut internal_symbols = parsed.internal_symbols;
        internal_symbols.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.line_number.cmp(&b.line_number)));

        Ok(RustFileNode {
            rel_path: rel_path.to_string(),
            crate_name,
            module_path,
            lines: lines_count,
            dynamic_sha256: sha256,
            is_test,
            modules_declared,
            imports,
            exports,
            internal_symbols,
            call_references: parsed.call_references,
            out_links: BTreeSet::new(),
            in_links: BTreeSet::new(),
        })
    }

    /// Resolves in-links and out-links across all indexed files.
    pub fn resolve_all_links(&mut self) {
        // Clear existing links
        for node in self.files.values_mut() {
            node.out_links.clear();
            node.in_links.clear();
        }

        // Build list of resolutions (source_path, target_path)
        let mut links: Vec<(String, String)> = Vec::new();

        for (source_path, source_node) in &self.files {
            for import in &source_node.imports {
                if let Some(target_path) = self.resolve_import_target(source_node, &import.raw_path) {
                    if target_path != *source_path {
                        links.push((source_path.clone(), target_path));
                    }
                }
            }

            // Also check declared submodules
            for mod_decl in &source_node.modules_declared {
                if let Some(target_path) = self.resolve_submodule_target(source_node, mod_decl) {
                    if target_path != *source_path {
                        links.push((source_path.clone(), target_path));
                    }
                }
            }
        }

        // Apply links bi-directionally
        for (source, target) in links {
            if let Some(s_node) = self.files.get_mut(&source) {
                s_node.out_links.insert(target.clone());
            }
            if let Some(t_node) = self.files.get_mut(&target) {
                t_node.in_links.insert(source);
            }
        }
    }

    /// Resolves an import statement path to a target file path.
    fn resolve_import_target(&self, source_node: &RustFileNode, raw_import: &str) -> Option<String> {
        let clean = raw_import.trim_start_matches("pub ").trim_start_matches("use ").trim();
        let tokens: Vec<&str> = clean.split("::").collect();
        if tokens.is_empty() {
            return None;
        }

        // 1. crate::... (same crate or mounted module)
        if tokens[0] == "crate" {
            for k in (2..=tokens.len()).rev() {
                let rel_mod = tokens[1..k].join("::");
                let candidate_mod = format!("{}::{}", source_node.crate_name, rel_mod);
                if let Some(target) = self.module_to_file.get(&candidate_mod) {
                    return Some(target.clone());
                }
                if let Some(target) = self.module_to_file.get(&rel_mod) {
                    return Some(target.clone());
                }
            }
            if tokens.len() >= 2 {
                for (mod_name, target) in &self.module_to_file {
                    if mod_name.ends_with(&format!("::{}", tokens[1])) {
                        return Some(target.clone());
                    }
                }
            }
            return None;
        }

        // 2. super::...
        if tokens[0] == "super" {
            let parent_mod = get_parent_module(&source_node.module_path);
            if let Some(parent) = parent_mod {
                if let Some(target) = self.module_to_file.get(&parent) {
                    return Some(target.clone());
                }
            }
            return None;
        }

        // 3. Known workspace crate (dynamically checked against self.known_crates)
        let first = tokens[0];
        if self.known_crates.contains(first) {
            for k in (2..=tokens.len()).rev() {
                let sub_path = tokens[..k].join("::");
                if let Some(target) = self.module_to_file.get(&sub_path) {
                    return Some(target.clone());
                }
            }
            if tokens.len() >= 2 {
                if let Some(target) = self.module_to_file.get(tokens[1]) {
                    return Some(target.clone());
                }
            }
            // Root crate lib.rs / main.rs
            if let Some(target) = self.module_to_file.get(first) {
                return Some(target.clone());
            }
        }

        // 4. Try matching direct module name
        if let Some(target) = self.module_to_file.get(first) {
            return Some(target.clone());
        }

        // 5. Symbol-based import matching: e.g. use foo::Bar; where Bar is declared
        if tokens.len() >= 2 {
            let last_token = tokens[tokens.len() - 1].trim_matches('{').trim_matches('}').trim();
            for sub_token in last_token.split(',') {
                let sub_token = sub_token.trim();
                if let Some(files) = self.symbol_to_files.get(sub_token) {
                    if let Some(f) = files.iter().next() {
                        return Some(f.clone());
                    }
                }
            }
        }

        None
    }

    /// Resolves a submodule declaration target.
    fn resolve_submodule_target(&self, source_node: &RustFileNode, mod_decl: &ModuleDeclaration) -> Option<String> {
        // If custom path provided: #[path = "..."]
        if let Some(custom) = &mod_decl.custom_path {
            let base = Path::new(&source_node.rel_path).parent().unwrap_or(Path::new(""));
            let target_path = normalize_rel_path(&base.join(custom));
            if self.files.contains_key(&target_path) {
                return Some(target_path);
            }
        }

        // Standard Rust module path resolution:
        // Source: rust/crate/src/foo.rs -> Child: rust/crate/src/foo/<name>.rs OR rust/crate/src/foo/<name>/mod.rs
        let base_dir = Path::new(&source_node.rel_path);
        let stem = base_dir.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let parent = base_dir.parent().unwrap_or(Path::new(""));

        let candidate_1 = if stem == "mod" || stem == "lib" || stem == "main" {
            normalize_rel_path(&parent.join(format!("{}.rs", mod_decl.name)))
        } else {
            normalize_rel_path(&parent.join(stem).join(format!("{}.rs", mod_decl.name)))
        };

        if self.files.contains_key(&candidate_1) {
            return Some(candidate_1);
        }

        let candidate_2 = if stem == "mod" || stem == "lib" || stem == "main" {
            normalize_rel_path(&parent.join(&mod_decl.name).join("mod.rs"))
        } else {
            normalize_rel_path(&parent.join(stem).join(&mod_decl.name).join("mod.rs"))
        };

        if self.files.contains_key(&candidate_2) {
            return Some(candidate_2);
        }

        None
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 3. AUTOMATIC RELATIONSHIP RECONCILIATION & SYNC (RULES 3 & 4)
    // ──────────────────────────────────────────────────────────────────────────

    /// Handles file creation event from Live Watcher.
    pub fn on_file_created(&mut self, rel_path: &str) -> Result<RelationshipDelta, Box<dyn std::error::Error>> {
        let full_path = self.config.workspace_root.join(rel_path);
        if !full_path.is_file() || !rel_path.ends_with(".rs") {
            return Ok(RelationshipDelta::default());
        }

        let node = self.parse_rust_file(&full_path, rel_path)?;
        self.module_to_file.insert(node.module_path.clone(), rel_path.to_string());

        for sym in &node.exports {
            self.symbol_to_files
                .entry(sym.name.clone())
                .or_default()
                .insert(rel_path.to_string());
        }

        let mut delta = RelationshipDelta {
            file_path: rel_path.to_string(),
            action: "CREATED".to_string(),
            added_imports: node.imports.iter().map(|i| i.raw_path.clone()).collect(),
            added_exports: node.exports.iter().map(|e| e.name.clone()).collect(),
            ..Default::default()
        };

        self.files.insert(rel_path.to_string(), node);
        self.resolve_all_links();

        if let Some(new_node) = self.files.get(rel_path) {
            delta.affected_out_links = new_node.out_links.iter().cloned().collect();
            delta.affected_in_links = new_node.in_links.iter().cloned().collect();
        }

        Ok(delta)
    }

    /// Handles file modification event from Live Watcher.
    pub fn on_file_modified(&mut self, rel_path: &str) -> Result<RelationshipDelta, Box<dyn std::error::Error>> {
        let full_path = self.config.workspace_root.join(rel_path);
        if !full_path.is_file() || !rel_path.ends_with(".rs") {
            return Ok(RelationshipDelta::default());
        }

        let new_node = self.parse_rust_file(&full_path, rel_path)?;

        let (old_imports, old_exports) = if let Some(old) = self.files.get(rel_path) {
            (
                old.imports.iter().map(|i| i.raw_path.clone()).collect::<BTreeSet<_>>(),
                old.exports.iter().map(|e| e.name.clone()).collect::<BTreeSet<_>>(),
            )
        } else {
            (BTreeSet::new(), BTreeSet::new())
        };

        let new_imports: BTreeSet<String> = new_node.imports.iter().map(|i| i.raw_path.clone()).collect();
        let new_exports: BTreeSet<String> = new_node.exports.iter().map(|e| e.name.clone()).collect();

        let added_imports: Vec<String> = new_imports.difference(&old_imports).cloned().collect();
        let removed_imports: Vec<String> = old_imports.difference(&new_imports).cloned().collect();
        let added_exports: Vec<String> = new_exports.difference(&old_exports).cloned().collect();
        let removed_exports: Vec<String> = old_exports.difference(&new_exports).cloned().collect();

        // Update symbol index
        for rem in &removed_exports {
            if let Some(set) = self.symbol_to_files.get_mut(rem) {
                set.remove(rel_path);
            }
        }
        for add in &added_exports {
            self.symbol_to_files
                .entry(add.clone())
                .or_default()
                .insert(rel_path.to_string());
        }

        self.files.insert(rel_path.to_string(), new_node);
        self.resolve_all_links();

        let mut delta = RelationshipDelta {
            file_path: rel_path.to_string(),
            action: "MODIFIED".to_string(),
            added_imports,
            removed_imports,
            added_exports,
            removed_exports,
            ..Default::default()
        };

        if let Some(updated_node) = self.files.get(rel_path) {
            delta.affected_out_links = updated_node.out_links.iter().cloned().collect();
            delta.affected_in_links = updated_node.in_links.iter().cloned().collect();
        }

        Ok(delta)
    }

    /// Handles file move/relocation event (Automatic Reconnection - Rule 4).
    pub fn on_file_moved(
        &mut self,
        old_path: &str,
        new_path: &str,
    ) -> Result<RelationshipDelta, Box<dyn std::error::Error>> {
        let full_new = self.config.workspace_root.join(new_path);
        if !full_new.is_file() || !new_path.ends_with(".rs") {
            return Ok(RelationshipDelta::default());
        }

        // Clean old mappings
        if let Some(old_node) = self.files.remove(old_path) {
            self.module_to_file.remove(&old_node.module_path);
            for sym in &old_node.exports {
                if let Some(set) = self.symbol_to_files.get_mut(&sym.name) {
                    set.remove(old_path);
                }
            }
        }

        // Parse new file
        let new_node = self.parse_rust_file(&full_new, new_path)?;
        self.module_to_file.insert(new_node.module_path.clone(), new_path.to_string());
        for sym in &new_node.exports {
            self.symbol_to_files
                .entry(sym.name.clone())
                .or_default()
                .insert(new_path.to_string());
        }
        self.files.insert(new_path.to_string(), new_node);

        // Reconnect all links across the graph
        self.resolve_all_links();

        // Automatically update any capability references
        let _ = self.update_capability_file_path(old_path, new_path);

        let mut delta = RelationshipDelta {
            file_path: format!("{old_path} -> {new_path}"),
            action: "MOVED_AND_RECONNECTED".to_string(),
            ..Default::default()
        };

        if let Some(node) = self.files.get(new_path) {
            delta.affected_out_links = node.out_links.iter().cloned().collect();
            delta.affected_in_links = node.in_links.iter().cloned().collect();
        }

        Ok(delta)
    }

    /// Handles file deletion event from Live Watcher.
    pub fn on_file_deleted(&mut self, rel_path: &str) -> Result<RelationshipDelta, Box<dyn std::error::Error>> {
        let mut delta = RelationshipDelta {
            file_path: rel_path.to_string(),
            action: "DELETED".to_string(),
            ..Default::default()
        };

        if let Some(node) = self.files.remove(rel_path) {
            self.module_to_file.remove(&node.module_path);
            for sym in &node.exports {
                if let Some(set) = self.symbol_to_files.get_mut(&sym.name) {
                    set.remove(rel_path);
                }
            }
            delta.affected_in_links = node.in_links.iter().cloned().collect();
            delta.affected_out_links = node.out_links.iter().cloned().collect();
        }

        // Purge references from surviving nodes
        for node in self.files.values_mut() {
            node.out_links.remove(rel_path);
            node.in_links.remove(rel_path);
        }

        Ok(delta)
    }

    /// Handles folder move/rename event.
    /// Reconnects all descendant files, modules, symbols, in-links, out-links,
    /// and capability graph mappings dynamically across any directory.
    pub fn on_folder_moved(
        &mut self,
        old_folder: &str,
        new_folder: &str,
    ) -> Result<Vec<RelationshipDelta>, Box<dyn std::error::Error>> {
        let norm_old = old_folder.replace('\\', "/").trim_end_matches('/').to_string();
        let norm_new = new_folder.replace('\\', "/").trim_end_matches('/').to_string();

        let prefix_old = format!("{}/", norm_old);
        let prefix_new = format!("{}/", norm_new);

        let matching_files: Vec<(String, String)> = self
            .files
            .keys()
            .filter(|p| p.starts_with(&prefix_old))
            .map(|p| (p.clone(), format!("{}{}", prefix_new, &p[prefix_old.len()..])))
            .collect();

        let mut deltas = Vec::new();

        for (old_f, new_f) in matching_files {
            let d = self.on_file_moved(&old_f, &new_f)?;
            deltas.push(d);
        }

        self.resolve_all_links();

        let _ = self.update_capability_folder_path(&norm_old, &norm_new);

        Ok(deltas)
    }

    /// Handles folder deletion event.
    /// Purges all descendant files, symbols, modules, and links across the graph.
    pub fn on_folder_deleted(
        &mut self,
        folder_path: &str,
    ) -> Result<Vec<RelationshipDelta>, Box<dyn std::error::Error>> {
        let norm_folder = folder_path.replace('\\', "/").trim_end_matches('/').to_string();
        let prefix = format!("{}/", norm_folder);

        let matching_files: Vec<String> = self
            .files
            .keys()
            .filter(|p| p.starts_with(&prefix))
            .cloned()
            .collect();

        let mut deltas = Vec::new();
        for f in matching_files {
            let d = self.on_file_deleted(&f)?;
            deltas.push(d);
        }

        self.resolve_all_links();

        let _ = self.purge_capability_folder_path(&norm_folder);

        Ok(deltas)
    }

    /// Handles folder creation event.
    /// Indexes all newly created .rs files inside the folder and resolves links.
    pub fn on_folder_created(
        &mut self,
        folder_path: &str,
    ) -> Result<Vec<RelationshipDelta>, Box<dyn std::error::Error>> {
        let ws = self.config.workspace_root.clone();
        let full_dir = ws.join(folder_path);
        let mut rs_paths = Vec::new();
        if full_dir.is_dir() {
            self.discover_rust_files(&ws, &full_dir, &mut rs_paths)?;
        }

        let mut deltas = Vec::new();
        for rel in rs_paths {
            let d = self.on_file_created(&rel)?;
            deltas.push(d);
        }

        self.resolve_all_links();

        Ok(deltas)
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 4. VALIDATION & ROLLBACK (RULE 6)
    // ──────────────────────────────────────────────────────────────────────────

    /// Takes an in-memory snapshot of the relationship graph before mutations.
    pub fn create_snapshot(&self) -> RelationshipGraphSnapshot {
        let id = format!(
            "snap_{}",
            SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
        );
        RelationshipGraphSnapshot {
            snapshot_id: id,
            created_at_utc: chrono_now_iso(),
            files: self.files.clone(),
        }
    }

    /// Rolls back the engine to a previous snapshot upon validation failure.
    pub fn rollback(&mut self, snapshot: RelationshipGraphSnapshot) {
        self.files = snapshot.files;
        self.module_to_file.clear();
        self.symbol_to_files.clear();

        for (path, node) in &self.files {
            self.module_to_file.insert(node.module_path.clone(), path.clone());
            for sym in &node.exports {
                self.symbol_to_files
                    .entry(sym.name.clone())
                    .or_default()
                    .insert(path.clone());
            }
        }
    }

    /// Validates the relationship graph for broken links or orphaned declarations.
    pub fn validate_graph(&self) -> Result<usize, Vec<String>> {
        let mut issues = Vec::new();

        for (path, node) in &self.files {
            // Check out-links: verify that every out-link target exists in graph
            for out in &node.out_links {
                if !self.files.contains_key(out) {
                    issues.push(format!("[{path}] Broken out-link to non-indexed file: {out}"));
                }
            }

            // Check in-links: verify that every in-link source exists in graph
            for in_l in &node.in_links {
                if !self.files.contains_key(in_l) {
                    issues.push(format!("[{path}] Broken in-link from non-indexed file: {in_l}"));
                }
            }
        }

        if issues.is_empty() {
            Ok(self.files.len())
        } else {
            Err(issues)
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 5. ARCHITECTURE SYNCHRONIZATION (RULE 7)
    // ──────────────────────────────────────────────────────────────────────────

    /// Synchronizes changes into `ARCHITECTURE/SOURCE_INDEX.json` and `RELATIONSHIP_GRAPH.json`.
    pub fn sync_architecture_maps(&self) -> Result<(), Box<dyn std::error::Error>> {
        let arch_dir = &self.config.arch_dir;
        fs::create_dir_all(arch_dir)?;

        // 1. Synchronize ARCHITECTURE/SOURCE_INDEX.json
        let source_index_path = arch_dir.join("SOURCE_INDEX.json");
        let mut source_index: Value = if source_index_path.exists() {
            let raw = fs::read_to_string(&source_index_path)?;
            serde_json::from_str(raw.trim_start_matches('\u{feff}')).unwrap_or(json!({}))
        } else {
            json!({})
        };

        let mut crates_list: Vec<String> = self.known_crates.iter().cloned().collect();
        crates_list.sort();
        source_index["crates"] = json!(crates_list);

        let mut files_map = Map::new();
        for (rel_path, node) in &self.files {
            let mut structs: Vec<String> = node
                .exports
                .iter()
                .chain(node.internal_symbols.iter())
                .filter(|s| s.kind == SymbolKind::Struct)
                .map(|s| s.name.clone())
                .collect();
            structs.sort();
            structs.dedup();

            let mut enums: Vec<String> = node
                .exports
                .iter()
                .chain(node.internal_symbols.iter())
                .filter(|s| s.kind == SymbolKind::Enum)
                .map(|s| s.name.clone())
                .collect();
            enums.sort();
            enums.dedup();

            let mut traits: Vec<String> = node
                .exports
                .iter()
                .chain(node.internal_symbols.iter())
                .filter(|s| s.kind == SymbolKind::Trait)
                .map(|s| s.name.clone())
                .collect();
            traits.sort();
            traits.dedup();

            let mut functions: Vec<String> = node
                .exports
                .iter()
                .chain(node.internal_symbols.iter())
                .filter(|s| s.kind == SymbolKind::Function)
                .map(|s| s.name.clone())
                .collect();
            functions.sort();
            functions.dedup();

            let mut calls: Vec<String> = node.call_references.iter().cloned().collect();
            calls.sort();
            calls.dedup();

            let mut in_links: Vec<String> = node.in_links.iter().cloned().collect();
            in_links.sort();
            in_links.dedup();

            let mut out_links: Vec<String> = node.out_links.iter().cloned().collect();
            out_links.sort();
            out_links.dedup();

            let file_obj = json!({
                "crate": node.crate_name,
                "module": node.module_path,
                "rel_path": rel_path,
                "lines": node.lines,
                "dynamic_sha256": node.dynamic_sha256,
                "is_test": node.is_test,
                "structs": structs,
                "enums": enums,
                "traits": traits,
                "functions": functions,
                "calls": calls,
                "in_links": in_links,
                "out_links": out_links,
            });

            files_map.insert(rel_path.clone(), file_obj);
        }

        source_index["total_files"] = json!(files_map.len());
        source_index["files"] = Value::Object(files_map);

        let formatted_source_index = serde_json::to_string_pretty(&source_index)?.replace("\r\n", "\n") + "\n";
        fs::write(&source_index_path, formatted_source_index)?;

        // 2. Generate ARCHITECTURE/RELATIONSHIP_GRAPH.json
        let graph_path = arch_dir.join("RELATIONSHIP_GRAPH.json");
        let graph_payload = json!({
            "schema_version": "1.0.0",
            "total_files": self.files.len(),
            "total_modules": self.module_to_file.len(),
            "total_symbols": self.symbol_to_files.len(),
            "module_tree": self.module_to_file,
            "files": self.files
        });

        let formatted_graph = serde_json::to_string_pretty(&graph_payload)?.replace("\r\n", "\n") + "\n";
        fs::write(&graph_path, formatted_graph)?;

        // 3. Auto-reconcile CAPABILITY_GRAPH.json and all CAPABILITIES/*.json
        let _ = self.reconcile_capabilities();

        // 4. Keep file_registry.json in sync for architecture maps
        let file_reg_path = arch_dir.join("file_registry.json");
        if file_reg_path.is_file() {
            if let Ok(raw) = fs::read_to_string(&file_reg_path) {
                if let Ok(mut val) = serde_json::from_str::<Value>(raw.trim_start_matches('\u{feff}')) {
                    if let Some(files_map) = val.get_mut("files").and_then(Value::as_object_mut) {
                        let to_sync = [
                            "ARCHITECTURE/SOURCE_INDEX.json",
                            "ARCHITECTURE/RELATIONSHIP_GRAPH.json",
                            "ARCHITECTURE/CAPABILITY_GRAPH.json",
                        ];
                        for rel in to_sync {
                            let p = self.config.workspace_root.join(rel);
                            if p.is_file() {
                                if let Ok(sha) = compute_dynamic_sha256(&p, 64 * 1024) {
                                    if let Some(entry) = files_map.get_mut(rel) {
                                        entry["sha256"] = Value::String(sha);
                                    }
                                }
                            }
                        }

                        let cap_dir = arch_dir.join("CAPABILITIES");
                        if cap_dir.is_dir() {
                            if let Ok(entries) = fs::read_dir(&cap_dir) {
                                for entry in entries.flatten() {
                                    let p = entry.path();
                                    if p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("json") {
                                        let rel = format!("ARCHITECTURE/CAPABILITIES/{}", p.file_name().unwrap().to_string_lossy());
                                        if let Ok(sha) = compute_dynamic_sha256(&p, 64 * 1024) {
                                            if let Some(entry) = files_map.get_mut(&rel) {
                                                entry["sha256"] = Value::String(sha);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if let Ok(updated_raw) = serde_json::to_string_pretty(&val) {
                        let _ = fs::write(&file_reg_path, updated_raw);
                    }
                }
            }
        }

        Ok(())
    }

    /// Locates the new path of a moved or missing implementation file.
    /// Progressive resolution:
    /// 1. Filename basename matching (if distinctive filename, e.g. dataset_engine.rs).
    /// 2. Directory / stem matching (e.g. old foo.rs moved to foo/foo.rs).
    /// 3. Symbol matching against the struct corresponding to the file stem.
    pub fn find_relocated_file(
        &self,
        old_rel_path: &str,
        primary_structs: &[String],
        _cap_id: &str,
        exclude_files: &[String],
    ) -> Option<String> {
        let norm_old = old_rel_path.replace('\\', "/");
        let old_p = Path::new(&norm_old);
        let old_name = old_p.file_name()?.to_str()?;
        let old_stem = old_p.file_stem()?.to_str()?;
        let old_parent = old_p.parent().map(|p| p.to_string_lossy().replace('\\', "/")).unwrap_or_default();

        // 1. Basename matching: If old filename was not generic "mod.rs", look for matching file
        if old_name != "mod.rs" {
            let mut candidates: Vec<&String> = self
                .files
                .keys()
                .filter(|p| {
                    *p != &norm_old
                        && !exclude_files.contains(p)
                        && Path::new(p).file_name().and_then(|n| n.to_str()) == Some(old_name)
                })
                .collect();

            if candidates.len() == 1 {
                return Some(candidates[0].clone());
            } else if candidates.len() > 1 {
                candidates.sort_by_cached_key(|cand| {
                    let common_len = norm_old.chars().zip(cand.chars()).take_while(|(a, b)| a == b).count();
                    std::cmp::Reverse(common_len)
                });
                return Some(candidates[0].clone());
            }
        }

        // 2. Directory / Stem matching:
        // Case A: old was `.../foo.rs` and moved to `.../foo/foo.rs`
        if old_name != "mod.rs" && !old_parent.is_empty() {
            let nested_path = format!("{}/{}/{}", old_parent, old_stem, old_name);
            if self.files.contains_key(&nested_path) && !exclude_files.contains(&nested_path) {
                return Some(nested_path);
            }
        }

        // Case B: old was `.../foo/mod.rs` and moved to `.../foo/foo.rs` or any single `.rs` file in `.../foo/`
        if old_name == "mod.rs" && !old_parent.is_empty() {
            let in_dir: Vec<&String> = self
                .files
                .keys()
                .filter(|p| {
                    *p != &norm_old
                        && !exclude_files.contains(p)
                        && Path::new(p)
                            .parent()
                            .map(|pr| pr.to_string_lossy().replace('\\', "/") == old_parent)
                            .unwrap_or(false)
                })
                .collect();
            if in_dir.len() == 1 {
                return Some(in_dir[0].clone());
            } else if in_dir.len() > 1 {
                let dir_name = Path::new(&old_parent).file_name().and_then(|n| n.to_str()).unwrap_or("");
                if let Some(matched) = in_dir.iter().find(|p| Path::new(p).file_stem().and_then(|s| s.to_str()) == Some(dir_name)) {
                    return Some((*matched).clone());
                }
            }
        }

        // 3. Symbol-based exact resolution: match struct corresponding to old stem
        let pascal_stem = snake_to_pascal_case(old_stem);
        if !pascal_stem.is_empty() {
            if let Some(files) = self.symbol_to_files.get(&pascal_stem) {
                for candidate in files {
                    if candidate != &norm_old && !exclude_files.contains(candidate) {
                        let is_test = self.files.get(candidate).map(|n| n.is_test).unwrap_or(false);
                        if !is_test {
                            return Some(candidate.clone());
                        }
                    }
                }
            }
        }

        // Check primary structs that explicitly match the old stem
        for s in primary_structs {
            let s_lower = s.to_lowercase();
            let stem_lower = old_stem.replace('_', "").to_lowercase();
            if s_lower == stem_lower || s_lower.starts_with(&stem_lower) {
                if let Some(files) = self.symbol_to_files.get(s) {
                    for candidate in files {
                        if candidate != &norm_old && !exclude_files.contains(candidate) {
                            let is_test = self.files.get(candidate).map(|n| n.is_test).unwrap_or(false);
                            if !is_test {
                                return Some(candidate.clone());
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Directly renames an old file path to a new path across `CAPABILITY_GRAPH.json` and all `CAPABILITIES/*.json`.
    pub fn update_capability_file_path(
        &self,
        old_path: &str,
        new_path: &str,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let arch_dir = &self.config.arch_dir;
        let graph_path = arch_dir.join("CAPABILITY_GRAPH.json");
        let cap_dir = arch_dir.join("CAPABILITIES");

        let mut changed = false;
        let mut modified_files = Vec::new();

        if graph_path.is_file() {
            let raw = fs::read_to_string(&graph_path)?;
            if raw.contains(old_path) {
                let mut graph_val: Value = serde_json::from_str(raw.trim_start_matches('\u{feff}'))?;
                if let Some(nodes) = graph_val.get_mut("nodes").and_then(Value::as_array_mut) {
                    for node in nodes {
                        if let Some(impl_files) = node.get_mut("implementation_files").and_then(Value::as_array_mut) {
                            for item in impl_files {
                                if item.as_str() == Some(old_path) {
                                    *item = Value::String(new_path.to_string());
                                    changed = true;
                                }
                            }
                        }
                        if let Some(test_suites) = node.get_mut("test_suites").and_then(Value::as_array_mut) {
                            for item in test_suites {
                                if let Some(s) = item.as_str() {
                                    if s.starts_with(old_path) {
                                        let updated = s.replacen(old_path, new_path, 1);
                                        *item = Value::String(updated);
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }
                }
                if changed {
                    let formatted = serde_json::to_string_pretty(&graph_val)?.replace("\r\n", "\n") + "\n";
                    fs::write(&graph_path, formatted)?;
                    modified_files.push("ARCHITECTURE/CAPABILITY_GRAPH.json".to_string());
                }
            }
        }

        if cap_dir.is_dir() {
            if let Ok(entries) = fs::read_dir(&cap_dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.extension().and_then(|e| e.to_str()) == Some("json") {
                        if let Ok(raw) = fs::read_to_string(&p) {
                            if raw.contains(old_path) {
                                if let Ok(mut val) = serde_json::from_str::<Value>(raw.trim_start_matches('\u{feff}')) {
                                    let mut cap_mod = false;
                                    if let Some(impl_files) = val.get_mut("implementation_files").and_then(Value::as_array_mut) {
                                        for item in impl_files {
                                            if item.as_str() == Some(old_path) {
                                                *item = Value::String(new_path.to_string());
                                                cap_mod = true;
                                            }
                                        }
                                    }
                                    if let Some(test_suites) = val.get_mut("test_suites").and_then(Value::as_array_mut) {
                                        for item in test_suites {
                                            if let Some(s) = item.as_str() {
                                                if s.starts_with(old_path) {
                                                    let updated = s.replacen(old_path, new_path, 1);
                                                    *item = Value::String(updated);
                                                    cap_mod = true;
                                                }
                                            }
                                        }
                                    }
                                    if cap_mod {
                                        let formatted = serde_json::to_string_pretty(&val)?.replace("\r\n", "\n") + "\n";
                                        let _ = fs::write(&p, formatted);
                                        let rel = format!("ARCHITECTURE/CAPABILITIES/{}", p.file_name().unwrap().to_string_lossy());
                                        modified_files.push(rel);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if !modified_files.is_empty() {
            let file_reg_path = arch_dir.join("file_registry.json");
            if file_reg_path.is_file() {
                if let Ok(raw) = fs::read_to_string(&file_reg_path) {
                    if let Ok(mut val) = serde_json::from_str::<Value>(raw.trim_start_matches('\u{feff}')) {
                        if let Some(files_map) = val.get_mut("files").and_then(Value::as_object_mut) {
                            for rel in &modified_files {
                                let p = self.config.workspace_root.join(rel);
                                if p.is_file() {
                                    if let Ok(sha) = compute_dynamic_sha256(&p, self.config.io_buffer_size) {
                                        if let Some(entry) = files_map.get_mut(rel) {
                                            entry["sha256"] = Value::String(sha);
                                        }
                                    }
                                }
                            }
                        }
                        if let Ok(updated_raw) = serde_json::to_string_pretty(&val) {
                            let _ = fs::write(&file_reg_path, updated_raw);
                        }
                    }
                }
            }
        }

        Ok(changed || !modified_files.is_empty())
    }

    /// Updates capability paths when a folder is moved/renamed.
    pub fn update_capability_folder_path(
        &self,
        old_folder: &str,
        new_folder: &str,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let norm_old = old_folder.replace('\\', "/").trim_end_matches('/').to_string();
        let norm_new = new_folder.replace('\\', "/").trim_end_matches('/').to_string();
        let prefix_old = format!("{}/", norm_old);
        let prefix_new = format!("{}/", norm_new);

        let arch_dir = &self.config.arch_dir;
        let graph_path = arch_dir.join("CAPABILITY_GRAPH.json");
        let cap_dir = arch_dir.join("CAPABILITIES");

        let mut changed = false;
        let mut modified_files = Vec::new();

        if graph_path.is_file() {
            let raw = fs::read_to_string(&graph_path)?;
            if raw.contains(&norm_old) {
                let mut graph_val: Value = serde_json::from_str(raw.trim_start_matches('\u{feff}'))?;
                if let Some(nodes) = graph_val.get_mut("nodes").and_then(Value::as_array_mut) {
                    for node in nodes {
                        if let Some(impl_files) = node.get_mut("implementation_files").and_then(Value::as_array_mut) {
                            for item in impl_files {
                                if let Some(s) = item.as_str() {
                                    if s.starts_with(&prefix_old) {
                                        *item = Value::String(format!("{}{}", prefix_new, &s[prefix_old.len()..]));
                                        changed = true;
                                    }
                                }
                            }
                        }
                        if let Some(test_suites) = node.get_mut("test_suites").and_then(Value::as_array_mut) {
                            for item in test_suites {
                                if let Some(s) = item.as_str() {
                                    if s.starts_with(&prefix_old) {
                                        *item = Value::String(format!("{}{}", prefix_new, &s[prefix_old.len()..]));
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }
                }
                if changed {
                    let formatted = serde_json::to_string_pretty(&graph_val)?.replace("\r\n", "\n") + "\n";
                    fs::write(&graph_path, formatted)?;
                    modified_files.push("ARCHITECTURE/CAPABILITY_GRAPH.json".to_string());
                }
            }
        }

        if cap_dir.is_dir() {
            if let Ok(entries) = fs::read_dir(&cap_dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.extension().and_then(|e| e.to_str()) == Some("json") {
                        if let Ok(raw) = fs::read_to_string(&p) {
                            if raw.contains(&norm_old) {
                                if let Ok(mut val) = serde_json::from_str::<Value>(raw.trim_start_matches('\u{feff}')) {
                                    let mut cap_mod = false;
                                    if let Some(impl_files) = val.get_mut("implementation_files").and_then(Value::as_array_mut) {
                                        for item in impl_files {
                                            if let Some(s) = item.as_str() {
                                                if s.starts_with(&prefix_old) {
                                                    *item = Value::String(format!("{}{}", prefix_new, &s[prefix_old.len()..]));
                                                    cap_mod = true;
                                                }
                                            }
                                        }
                                    }
                                    if let Some(test_suites) = val.get_mut("test_suites").and_then(Value::as_array_mut) {
                                        for item in test_suites {
                                            if let Some(s) = item.as_str() {
                                                if s.starts_with(&prefix_old) {
                                                    *item = Value::String(format!("{}{}", prefix_new, &s[prefix_old.len()..]));
                                                    cap_mod = true;
                                                }
                                            }
                                        }
                                    }
                                    if cap_mod {
                                        let formatted = serde_json::to_string_pretty(&val)?.replace("\r\n", "\n") + "\n";
                                        let _ = fs::write(&p, formatted);
                                        let rel = format!("ARCHITECTURE/CAPABILITIES/{}", p.file_name().unwrap().to_string_lossy());
                                        modified_files.push(rel);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if !modified_files.is_empty() {
            let file_reg_path = arch_dir.join("file_registry.json");
            if file_reg_path.is_file() {
                if let Ok(raw) = fs::read_to_string(&file_reg_path) {
                    if let Ok(mut val) = serde_json::from_str::<Value>(raw.trim_start_matches('\u{feff}')) {
                        if let Some(files_map) = val.get_mut("files").and_then(Value::as_object_mut) {
                            for rel in &modified_files {
                                let p = self.config.workspace_root.join(rel);
                                if p.is_file() {
                                    if let Ok(sha) = compute_dynamic_sha256(&p, self.config.io_buffer_size) {
                                        if let Some(entry) = files_map.get_mut(rel) {
                                            entry["sha256"] = Value::String(sha);
                                        }
                                    }
                                }
                            }
                        }
                        if let Ok(updated_raw) = serde_json::to_string_pretty(&val) {
                            let _ = fs::write(&file_reg_path, updated_raw);
                        }
                    }
                }
            }
        }

        Ok(changed || !modified_files.is_empty())
    }

    /// Purges references to a deleted folder from capability files.
    pub fn purge_capability_folder_path(
        &self,
        folder_path: &str,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let norm_folder = folder_path.replace('\\', "/").trim_end_matches('/').to_string();
        let prefix = format!("{}/", norm_folder);

        let arch_dir = &self.config.arch_dir;
        let graph_path = arch_dir.join("CAPABILITY_GRAPH.json");
        let cap_dir = arch_dir.join("CAPABILITIES");

        let mut changed = false;
        let mut modified_files = Vec::new();

        if graph_path.is_file() {
            let raw = fs::read_to_string(&graph_path)?;
            if raw.contains(&norm_folder) {
                let mut graph_val: Value = serde_json::from_str(raw.trim_start_matches('\u{feff}'))?;
                if let Some(nodes) = graph_val.get_mut("nodes").and_then(Value::as_array_mut) {
                    for node in nodes {
                        if let Some(impl_files) = node.get_mut("implementation_files").and_then(Value::as_array_mut) {
                            let before = impl_files.len();
                            impl_files.retain(|v| !v.as_str().map(|s| s.starts_with(&prefix)).unwrap_or(false));
                            if impl_files.len() != before {
                                changed = true;
                            }
                        }
                        if let Some(test_suites) = node.get_mut("test_suites").and_then(Value::as_array_mut) {
                            let before = test_suites.len();
                            test_suites.retain(|v| !v.as_str().map(|s| s.starts_with(&prefix)).unwrap_or(false));
                            if test_suites.len() != before {
                                changed = true;
                            }
                        }
                    }
                }
                if changed {
                    let formatted = serde_json::to_string_pretty(&graph_val)?.replace("\r\n", "\n") + "\n";
                    fs::write(&graph_path, formatted)?;
                    modified_files.push("ARCHITECTURE/CAPABILITY_GRAPH.json".to_string());
                }
            }
        }

        if cap_dir.is_dir() {
            if let Ok(entries) = fs::read_dir(&cap_dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.extension().and_then(|e| e.to_str()) == Some("json") {
                        if let Ok(raw) = fs::read_to_string(&p) {
                            if raw.contains(&norm_folder) {
                                if let Ok(mut val) = serde_json::from_str::<Value>(raw.trim_start_matches('\u{feff}')) {
                                    let mut cap_mod = false;
                                    if let Some(impl_files) = val.get_mut("implementation_files").and_then(Value::as_array_mut) {
                                        let before = impl_files.len();
                                        impl_files.retain(|v| !v.as_str().map(|s| s.starts_with(&prefix)).unwrap_or(false));
                                        if impl_files.len() != before {
                                            cap_mod = true;
                                        }
                                    }
                                    if let Some(test_suites) = val.get_mut("test_suites").and_then(Value::as_array_mut) {
                                        let before = test_suites.len();
                                        test_suites.retain(|v| !v.as_str().map(|s| s.starts_with(&prefix)).unwrap_or(false));
                                        if test_suites.len() != before {
                                            cap_mod = true;
                                        }
                                    }
                                    if cap_mod {
                                        let formatted = serde_json::to_string_pretty(&val)?.replace("\r\n", "\n") + "\n";
                                        let _ = fs::write(&p, formatted);
                                        let rel = format!("ARCHITECTURE/CAPABILITIES/{}", p.file_name().unwrap().to_string_lossy());
                                        modified_files.push(rel);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if !modified_files.is_empty() {
            let file_reg_path = arch_dir.join("file_registry.json");
            if file_reg_path.is_file() {
                if let Ok(raw) = fs::read_to_string(&file_reg_path) {
                    if let Ok(mut val) = serde_json::from_str::<Value>(raw.trim_start_matches('\u{feff}')) {
                        if let Some(files_map) = val.get_mut("files").and_then(Value::as_object_mut) {
                            for rel in &modified_files {
                                let p = self.config.workspace_root.join(rel);
                                if p.is_file() {
                                    if let Ok(sha) = compute_dynamic_sha256(&p, self.config.io_buffer_size) {
                                        if let Some(entry) = files_map.get_mut(rel) {
                                            entry["sha256"] = Value::String(sha);
                                        }
                                    }
                                }
                            }
                        }
                        if let Ok(updated_raw) = serde_json::to_string_pretty(&val) {
                            let _ = fs::write(&file_reg_path, updated_raw);
                        }
                    }
                }
            }
        }

        Ok(changed || !modified_files.is_empty())
    }

    /// Automatically reconciles `ARCHITECTURE/CAPABILITY_GRAPH.json` and all
    /// `ARCHITECTURE/CAPABILITIES/*.json` files against disk truth.
    /// Detects any missing/moved implementation files and test suites, locates their
    /// new paths via AST and structural matching, verifies all primary_structs are covered,
    /// deduplicates entries, and rewrites the capability files automatically.
    pub fn reconcile_capabilities(&self) -> Result<bool, Box<dyn std::error::Error>> {
        let arch_dir = &self.config.arch_dir;
        let graph_path = arch_dir.join("CAPABILITY_GRAPH.json");
        let cap_dir = arch_dir.join("CAPABILITIES");

        if !graph_path.exists() {
            return Ok(false);
        }

        let raw_graph = fs::read_to_string(&graph_path)?;
        let mut graph_val: Value = serde_json::from_str(raw_graph.trim_start_matches('\u{feff}'))?;

        let mut any_graph_changed = false;
        let mut updated_cap_files: Vec<String> = Vec::new();

        if let Some(nodes) = graph_val.get_mut("nodes").and_then(Value::as_array_mut) {
            for node in nodes {
                let cap_id = node.get("id").and_then(Value::as_str).unwrap_or("").to_string();
                if cap_id.is_empty() {
                    continue;
                }

                let primary_structs: Vec<String> = node
                    .get("primary_structs")
                    .and_then(Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .filter_map(Value::as_str)
                            .map(|s| s.to_string())
                            .collect()
                    })
                    .unwrap_or_default();

                // 1. Reconcile implementation_files in CAPABILITY_GRAPH.json node
                if let Some(impl_files) = node.get_mut("implementation_files").and_then(Value::as_array_mut) {
                    let old_list: Vec<String> = impl_files.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect();
                    let mut new_list: Vec<String> = Vec::new();

                    for old_p in &old_list {
                        let full = self.config.workspace_root.join(old_p);
                        if full.is_file() {
                            if !new_list.contains(old_p) {
                                new_list.push(old_p.clone());
                            }
                        } else if let Some(new_p) = self.find_relocated_file(old_p, &primary_structs, &cap_id, &new_list) {
                            if !new_list.contains(&new_p) {
                                new_list.push(new_p);
                            }
                        }
                    }

                    // Self-healing: verify all primary_structs are covered by implementation files
                    for s in &primary_structs {
                        let already_covered = new_list.iter().any(|f| {
                            if let Some(n) = self.files.get(f) {
                                n.exports.iter().any(|sym| sym.name == *s) ||
                                n.internal_symbols.iter().any(|sym| sym.name == *s)
                            } else {
                                false
                            }
                        });

                        if !already_covered {
                            if let Some(declaring) = self.symbol_to_files.get(s) {
                                for cand in declaring {
                                    let is_test = self.files.get(cand).map(|n| n.is_test).unwrap_or(false);
                                    if !is_test && !new_list.contains(cand) {
                                        new_list.push(cand.clone());
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    if new_list != old_list {
                        *impl_files = new_list.into_iter().map(Value::String).collect();
                        any_graph_changed = true;
                    }
                }

                // 2. Reconcile test_suites in CAPABILITY_GRAPH.json node
                if let Some(test_suites) = node.get_mut("test_suites").and_then(Value::as_array_mut) {
                    let old_list: Vec<String> = test_suites.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect();
                    let mut new_list: Vec<String> = Vec::new();

                    for test_entry in &old_list {
                        let (file_part, suffix) = if let Some(idx) = test_entry.find("::") {
                            (&test_entry[..idx], &test_entry[idx..])
                        } else {
                            (test_entry.as_str(), "")
                        };

                        let full = self.config.workspace_root.join(file_part);
                        let candidate_entry = if full.is_file() {
                            test_entry.clone()
                        } else if file_part.ends_with(".rs") {
                            if let Some(new_p) = self.find_relocated_file(file_part, &primary_structs, &cap_id, &[]) {
                                format!("{}{}", new_p, suffix)
                            } else {
                                test_entry.clone()
                            }
                        } else {
                            test_entry.clone()
                        };

                        if !new_list.contains(&candidate_entry) {
                            new_list.push(candidate_entry);
                        }
                    }

                    if new_list != old_list {
                        *test_suites = new_list.into_iter().map(Value::String).collect();
                        any_graph_changed = true;
                    }
                }

                // 3. Reconcile individual ARCHITECTURE/CAPABILITIES/<cap_id>.json
                let cap_file_path = cap_dir.join(format!("{}.json", cap_id));
                if cap_file_path.is_file() {
                    let cap_raw = fs::read_to_string(&cap_file_path)?;
                    let mut cap_val: Value = serde_json::from_str(cap_raw.trim_start_matches('\u{feff}'))?;
                    let mut cap_changed = false;

                    let cap_primary_structs: Vec<String> = cap_val
                        .get("primary_structs")
                        .and_then(Value::as_array)
                        .map(|arr| {
                            arr.iter()
                                .filter_map(Value::as_str)
                                .map(|s| s.to_string())
                                .collect()
                        })
                        .unwrap_or_else(|| primary_structs.clone());

                    if let Some(impl_files) = cap_val.get_mut("implementation_files").and_then(Value::as_array_mut) {
                        let old_list: Vec<String> = impl_files.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect();
                        let mut new_list: Vec<String> = Vec::new();

                        for old_p in &old_list {
                            let full = self.config.workspace_root.join(old_p);
                            if full.is_file() {
                                if !new_list.contains(old_p) {
                                    new_list.push(old_p.clone());
                                }
                            } else if let Some(new_p) = self.find_relocated_file(old_p, &cap_primary_structs, &cap_id, &new_list) {
                                if !new_list.contains(&new_p) {
                                    new_list.push(new_p);
                                }
                            }
                        }

                        // Self-healing: verify all primary_structs
                        for s in &cap_primary_structs {
                            let already_covered = new_list.iter().any(|f| {
                                if let Some(n) = self.files.get(f) {
                                    n.exports.iter().any(|sym| sym.name == *s) ||
                                    n.internal_symbols.iter().any(|sym| sym.name == *s)
                                } else {
                                    false
                                }
                            });

                            if !already_covered {
                                if let Some(declaring) = self.symbol_to_files.get(s) {
                                    for cand in declaring {
                                        let is_test = self.files.get(cand).map(|n| n.is_test).unwrap_or(false);
                                        if !is_test && !new_list.contains(cand) {
                                            new_list.push(cand.clone());
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        if new_list != old_list {
                            *impl_files = new_list.into_iter().map(Value::String).collect();
                            cap_changed = true;
                        }
                    }

                    if let Some(test_suites) = cap_val.get_mut("test_suites").and_then(Value::as_array_mut) {
                        let old_list: Vec<String> = test_suites.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect();
                        let mut new_list: Vec<String> = Vec::new();

                        for test_entry in &old_list {
                            let (file_part, suffix) = if let Some(idx) = test_entry.find("::") {
                                (&test_entry[..idx], &test_entry[idx..])
                            } else {
                                (test_entry.as_str(), "")
                            };

                            let full = self.config.workspace_root.join(file_part);
                            let candidate_entry = if full.is_file() {
                                test_entry.clone()
                            } else if file_part.ends_with(".rs") {
                                if let Some(new_p) = self.find_relocated_file(file_part, &cap_primary_structs, &cap_id, &[]) {
                                    format!("{}{}", new_p, suffix)
                                } else {
                                    test_entry.clone()
                                }
                            } else {
                                test_entry.clone()
                            };

                            if !new_list.contains(&candidate_entry) {
                                new_list.push(candidate_entry);
                            }
                        }

                        if new_list != old_list {
                            *test_suites = new_list.into_iter().map(Value::String).collect();
                            cap_changed = true;
                        }
                    }

                    if cap_changed {
                        let formatted_cap = serde_json::to_string_pretty(&cap_val)?.replace("\r\n", "\n") + "\n";
                        fs::write(&cap_file_path, formatted_cap)?;
                        updated_cap_files.push(format!("ARCHITECTURE/CAPABILITIES/{}.json", cap_id));
                    }
                }
            }
        }

        if any_graph_changed {
            let formatted_graph = serde_json::to_string_pretty(&graph_val)?.replace("\r\n", "\n") + "\n";
            fs::write(&graph_path, formatted_graph)?;
            updated_cap_files.push("ARCHITECTURE/CAPABILITY_GRAPH.json".to_string());
        }

        // 4. Update file_registry.json SHA-256 for all changed capability files
        if !updated_cap_files.is_empty() {
            let file_reg_path = arch_dir.join("file_registry.json");
            if file_reg_path.is_file() {
                if let Ok(raw) = fs::read_to_string(&file_reg_path) {
                    if let Ok(mut val) = serde_json::from_str::<Value>(raw.trim_start_matches('\u{feff}')) {
                        if let Some(files_map) = val.get_mut("files").and_then(Value::as_object_mut) {
                            for rel in &updated_cap_files {
                                let p = self.config.workspace_root.join(rel);
                                if p.is_file() {
                                    if let Ok(sha) = compute_dynamic_sha256(&p, self.config.io_buffer_size) {
                                        if let Some(entry) = files_map.get_mut(rel) {
                                            entry["sha256"] = Value::String(sha);
                                        }
                                    }
                                }
                            }
                        }
                        if let Ok(updated_raw) = serde_json::to_string_pretty(&val) {
                            let _ = fs::write(&file_reg_path, updated_raw);
                        }
                    }
                }
            }
        }

        Ok(any_graph_changed || !updated_cap_files.is_empty())
    }
}

fn snake_to_pascal_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize = true;
    for c in s.chars() {
        if c == '_' {
            capitalize = true;
        } else if capitalize {
            result.extend(c.to_uppercase());
            capitalize = false;
        } else {
            result.push(c);
        }
    }
    result
}

// ──────────────────────────────────────────────────────────────────────────────
// 6. NATIVE RUST LEXICAL & AST PARSER IMPLEMENTATION (RULE 5)
// ──────────────────────────────────────────────────────────────────────────────

struct ParsedAstOutput {
    modules_declared: Vec<ModuleDeclaration>,
    imports: Vec<ImportStatement>,
    exports: Vec<SymbolDeclaration>,
    internal_symbols: Vec<SymbolDeclaration>,
    call_references: BTreeSet<String>,
}

struct NativeRustAstParser<'a> {
    source: &'a str,
}

impl<'a> NativeRustAstParser<'a> {
    fn new(source: &'a str) -> Self {
        Self { source }
    }

    fn parse(&mut self) -> ParsedAstOutput {
        let mut modules_declared = Vec::new();
        let mut imports = Vec::new();
        let mut exports = Vec::new();
        let mut internal_symbols = Vec::new();
        let mut call_references = BTreeSet::new();
        let mut pending_custom_path: Option<String> = None;

        let clean_lines = self.preprocess_clean_lines();

        for (line_idx, line) in clean_lines.iter().enumerate() {
            let line_num = line_idx + 1;
            let trimmed = line.trim();

            if trimmed.is_empty() {
                continue;
            }

            // Check for #[path = "..."] attribute
            if trimmed.starts_with("#[path") || trimmed.contains("#[path") {
                if let Some(start) = trimmed.find("path") {
                    let after = trimmed[start + "path".len()..].trim();
                    let after_eq = after.trim_start_matches(|c: char| c == '=' || c.is_whitespace());
                    if let Some(inner) = after_eq.strip_prefix('"') {
                        if let Some(end) = inner.find('"') {
                            pending_custom_path = Some(inner[..end].to_string());
                        }
                    }
                }
                continue;
            }

            // Extract calls & references: <ident>( or <ident>!(
            Self::extract_calls_from_line(trimmed, &mut call_references);

            // Check Visibility prefix
            let (is_pub, after_vis) = if let Some(stripped) = trimmed.strip_prefix("pub(crate) ") {
                (false, stripped.trim())
            } else if let Some(stripped) = trimmed.strip_prefix("pub(super) ") {
                (false, stripped.trim())
            } else if let Some(stripped) = trimmed.strip_prefix("pub ") {
                (true, stripped.trim())
            } else {
                (false, trimmed)
            };

            // 1. Submodule declaration: [pub] mod <name>;
            if let Some(rest) = after_vis.strip_prefix("mod ") {
                let rest = rest.trim();
                let mod_name = rest
                    .split(|c: char| c.is_whitespace() || c == ';' || c == '{')
                    .next()
                    .unwrap_or("")
                    .to_string();

                if !mod_name.is_empty() {
                    let custom_path = pending_custom_path.take();
                    modules_declared.push(ModuleDeclaration {
                        name: mod_name,
                        is_public: is_pub,
                        custom_path,
                        line_number: line_num,
                    });
                }
                continue;
            }

            if !trimmed.starts_with("#[") {
                pending_custom_path = None;
            }

            // 2. Import statement: [pub] use <path>;
            if let Some(rest) = after_vis.strip_prefix("use ") {
                let rest = rest.trim();
                let clean_use = rest.trim_end_matches(';').trim();
                let imported_symbols = Self::extract_imported_symbols(clean_use);

                imports.push(ImportStatement {
                    raw_path: clean_use.to_string(),
                    is_public_reexport: is_pub,
                    imported_symbols,
                    line_number: line_num,
                });
                continue;
            }

            // 3. Declarations: struct, enum, trait, fn, type, const
            if let Some(sym) = Self::try_parse_item_declaration(after_vis, is_pub, line_num) {
                if sym.visibility == Visibility::Public {
                    exports.push(sym);
                } else {
                    internal_symbols.push(sym);
                }
            }
        }

        ParsedAstOutput {
            modules_declared,
            imports,
            exports,
            internal_symbols,
            call_references,
        }
    }

    /// Strips line and block comments from lines.
    fn preprocess_clean_lines(&self) -> Vec<String> {
        let mut result = Vec::new();
        let mut in_block_comment = false;

        for line in self.source.lines() {
            let mut clean_line = String::new();
            let chars: Vec<char> = line.chars().collect();
            let mut i = 0;

            while i < chars.len() {
                if in_block_comment {
                    if i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == '/' {
                        in_block_comment = false;
                        i += 2;
                    } else {
                        i += 1;
                    }
                } else {
                    if i + 1 < chars.len() && chars[i] == '/' && chars[i + 1] == '*' {
                        in_block_comment = true;
                        i += 2;
                    } else if i + 1 < chars.len() && chars[i] == '/' && chars[i + 1] == '/' {
                        // Line comment, discard remaining line
                        break;
                    } else {
                        clean_line.push(chars[i]);
                        i += 1;
                    }
                }
            }

            result.push(clean_line);
        }

        result
    }

    fn try_parse_item_declaration(
        trimmed: &str,
        is_pub: bool,
        line_num: usize,
    ) -> Option<SymbolDeclaration> {
        let keywords = [
            ("struct ", SymbolKind::Struct),
            ("enum ", SymbolKind::Enum),
            ("trait ", SymbolKind::Trait),
            ("fn ", SymbolKind::Function),
            ("type ", SymbolKind::TypeAlias),
            ("const ", SymbolKind::Const),
            ("static ", SymbolKind::Static),
        ];

        for (kw, kind) in &keywords {
            if let Some(rest) = trimmed.strip_prefix(kw) {
                let rest = rest.trim();
                let name = rest
                    .split(|c: char| c.is_whitespace() || c == '<' || c == '(' || c == ':' || c == '{' || c == ';')
                    .next()
                    .unwrap_or("")
                    .to_string();

                if !name.is_empty() && is_valid_ident(&name) {
                    return Some(SymbolDeclaration {
                        name,
                        kind: kind.clone(),
                        visibility: if is_pub { Visibility::Public } else { Visibility::Private },
                        line_number: line_num,
                    });
                }
            }
        }

        None
    }

    fn extract_imported_symbols(raw: &str) -> Vec<String> {
        let mut symbols = Vec::new();
        if let Some(start_brace) = raw.find('{') {
            if let Some(end_brace) = raw.rfind('}') {
                let inside = &raw[start_brace + 1..end_brace];
                for part in inside.split(',') {
                    let trimmed = part.trim();
                    let final_name = trimmed.split(" as ").last().unwrap_or(trimmed).trim();
                    if !final_name.is_empty() && is_valid_ident(final_name) {
                        symbols.push(final_name.to_string());
                    }
                }
            }
        } else {
            let last_part = raw.split("::").last().unwrap_or("").trim();
            let final_name = last_part.split(" as ").last().unwrap_or(last_part).trim();
            if !final_name.is_empty() && final_name != "*" && is_valid_ident(final_name) {
                symbols.push(final_name.to_string());
            }
        }
        symbols
    }

    fn extract_calls_from_line(line: &str, out: &mut BTreeSet<String>) {
        let words: Vec<&str> = line.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '(' && c != '!').collect();
        for word in words {
            if word.ends_with('(') && word.len() > 1 {
                let call_name = &word[..word.len() - 1];
                if is_valid_ident(call_name) && !is_rust_keyword(call_name) {
                    out.insert(call_name.to_string());
                }
            } else if word.ends_with('!') && word.len() > 1 {
                let macro_name = &word[..word.len() - 1];
                if is_valid_ident(macro_name) {
                    out.insert(macro_name.to_string());
                }
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// 7. HELPER FUNCTIONS & RUNTIME DYNAMIC SHA-256 (RULE 8)
// ──────────────────────────────────────────────────────────────────────────────

fn is_valid_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_alphabetic() || c == '_' => chars.all(|c| c.is_alphanumeric() || c == '_'),
        _ => false,
    }
}

fn is_rust_keyword(s: &str) -> bool {
    matches!(
        s,
        "if" | "else"
            | "while"
            | "for"
            | "loop"
            | "match"
            | "return"
            | "let"
            | "mut"
            | "pub"
            | "fn"
            | "struct"
            | "enum"
            | "trait"
            | "impl"
            | "use"
            | "mod"
            | "where"
            | "as"
            | "in"
            | "break"
            | "continue"
            | "unsafe"
            | "async"
            | "await"
    )
}

fn detect_crate_name(workspace_root: &Path, rel_path: &str) -> String {
    let full = workspace_root.join(rel_path);
    let mut curr = full.parent();
    while let Some(dir) = curr {
        if dir == workspace_root {
            break;
        }
        let cargo_toml = dir.join("Cargo.toml");
        if cargo_toml.is_file() {
            if let Ok(content) = fs::read_to_string(&cargo_toml) {
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("name") && trimmed.contains('=') {
                        let parts: Vec<&str> = trimmed.split('=').collect();
                        if parts.len() == 2 {
                            let name = parts[1].trim().trim_matches('"').trim_matches('\'').trim();
                            if !name.is_empty() {
                                return name.to_string();
                            }
                        }
                    }
                }
            }
            if let Some(folder_name) = dir.file_name().and_then(|n| n.to_str()) {
                return folder_name.to_string();
            }
        }
        curr = dir.parent();
    }

    let norm = rel_path.replace('\\', "/");
    let segments: Vec<&str> = norm.split('/').filter(|s| !s.is_empty()).collect();
    if segments.len() > 1 && segments[0] == "rust" {
        segments[1].to_string()
    } else if !segments.is_empty() {
        segments[0].to_string()
    } else {
        "crate".to_string()
    }
}

fn derive_module_path(rel_path: &str, crate_name: &str) -> String {
    let norm = rel_path.replace('\\', "/");
    let without_prefix = if let Some(idx) = norm.find("/src/") {
        &norm[idx + "/src/".len()..]
    } else {
        let crate_prefix = format!("{}/", crate_name);
        if let Some(idx) = norm.find(&crate_prefix) {
            &norm[idx + crate_prefix.len()..]
        } else {
            &norm
        }
    };

    let without_ext = without_prefix.trim_end_matches(".rs");
    if without_ext == "lib" || without_ext == "main" {
        return crate_name.to_string();
    }

    let without_mod = if let Some(stripped) = without_ext.strip_suffix("/mod") {
        stripped
    } else {
        without_ext
    };

    let module_subpath = without_mod.replace('/', "::");
    if module_subpath.is_empty() {
        crate_name.to_string()
    } else {
        format!("{}::{}", crate_name, module_subpath)
    }
}

fn get_parent_module(mod_path: &str) -> Option<String> {
    let parts: Vec<&str> = mod_path.split("::").collect();
    if parts.len() > 1 {
        Some(parts[..parts.len() - 1].join("::"))
    } else {
        None
    }
}

pub fn normalize_rel_path(p: &Path) -> String {
    let norm = p.to_string_lossy().replace('\\', "/");
    let mut parts = Vec::new();
    for seg in norm.split('/') {
        if seg.is_empty() || seg == "." {
            continue;
        } else if seg == ".." {
            parts.pop();
        } else {
            parts.push(seg);
        }
    }
    parts.join("/")
}

pub fn compute_dynamic_sha256(path: &Path, buffer_size: usize) -> std::io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    // Non-zero progress invariant: buffer must hold at least 1 byte to make progress
    let mut buffer = vec![0u8; buffer_size.max(1)];
    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn chrono_now_iso() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    secs.to_string()
}

pub fn find_workspace_root() -> Result<PathBuf, String> {
    let mut current = std::env::current_dir().map_err(|e| e.to_string())?;
    loop {
        if current.join("AGENTS.md").exists() && current.join("Cargo.toml").exists() {
            return Ok(current);
        }
        if let Some(parent) = current.parent() {
            current = parent.to_path_buf();
        } else {
            break;
        }
    }
    Err("Could not find workspace root (marked by AGENTS.md and Cargo.toml)".to_string())
}

// ──────────────────────────────────────────────────────────────────────────────
// 8. UNIT TESTS
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_native_ast_parser_extracts_symbols_and_imports() {
        let sample = r#"
            // Sample module test
            pub mod child_module;
            use crate::config::TaraConfig;
            pub use crate::tokenizer::TaraTokenizer;

            pub struct EngineState {
                pub ready: bool,
            }

            enum Mode {
                Fast,
                Precise,
            }

            pub trait Computable {
                fn compute(&self) -> u64;
            }

            pub fn initialize() -> EngineState {
                let conf = TaraConfig::default();
                EngineState { ready: true }
            }
        "#;

        let mut parser = NativeRustAstParser::new(sample);
        let res = parser.parse();

        assert_eq!(res.modules_declared.len(), 1);
        assert_eq!(res.modules_declared[0].name, "child_module");

        assert_eq!(res.imports.len(), 2);
        assert!(res.imports[0].imported_symbols.contains(&"TaraConfig".to_string()));
        assert!(res.imports[1].is_public_reexport);

        let export_names: Vec<String> = res.exports.iter().map(|e| e.name.clone()).collect();
        assert!(export_names.contains(&"EngineState".to_string()));
        assert!(export_names.contains(&"Computable".to_string()));
        assert!(export_names.contains(&"initialize".to_string()));

        let internal_names: Vec<String> = res.internal_symbols.iter().map(|e| e.name.clone()).collect();
        assert!(internal_names.contains(&"Mode".to_string()));

        assert!(res.call_references.contains("default"));
    }

    #[test]
    fn test_relationship_engine_full_lifecycle_and_rollback() {
        let temp_dir = std::env::temp_dir().join(format!("tara_link_test_{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        let arch_dir = temp_dir.join("ARCHITECTURE");
        fs::create_dir_all(&arch_dir).unwrap();

        let src_dir = temp_dir.join("rust").join("tara_engine").join("src");
        fs::create_dir_all(&src_dir).unwrap();

        let file_a_path = src_dir.join("mod_a.rs");
        let file_b_path = src_dir.join("mod_b.rs");

        fs::write(&file_a_path, "pub struct StructA;\npub fn helper_a() {}\n").unwrap();
        fs::write(&file_b_path, "use crate::mod_a::StructA;\npub fn use_b() { helper_a(); }\n").unwrap();

        let config = ProjectLinkConfig {
            workspace_root: temp_dir.clone(),
            arch_dir: arch_dir.clone(),
            ignored_dirs: vec!["target".to_string()],
            io_buffer_size: 4096,
        };

        let mut engine = ProjectLinkEngine::new(config);
        let report = engine.build_full_index().unwrap();

        assert!(report.total_rust_files >= 2);
        assert!(report.total_symbols >= 3);

        // Verify out-links and in-links
        let node_b = engine.files.get("rust/tara_engine/src/mod_b.rs").unwrap();
        assert!(node_b.out_links.contains("rust/tara_engine/src/mod_a.rs"));

        let node_a = engine.files.get("rust/tara_engine/src/mod_a.rs").unwrap();
        assert!(node_a.in_links.contains("rust/tara_engine/src/mod_b.rs"));

        // Test snapshot & rollback
        let snap = engine.create_snapshot();
        engine.on_file_deleted("rust/tara_engine/src/mod_a.rs").unwrap();
        assert!(!engine.files.contains_key("rust/tara_engine/src/mod_a.rs"));

        engine.rollback(snap);
        assert!(engine.files.contains_key("rust/tara_engine/src/mod_a.rs"));

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }
}
