//! TARA Architecture Sync Engine (100% Native Rust)
//!
//! Enforces:
//! - Complete project tree, file registry, and folder registry synchronization.
//! - 2 Reliability Layers:
//!   1. LIVE WATCHER: Continuous filesystem events observation via high-efficiency background thread.
//!   2. FULL RECONCILER: Exhaustive disk vs. registry reconciliation on engine start/restart.
//! - STRICT BOUNDARY: Read-only observation of project code. The engine NEVER mutates,
//!   moves, edits, or deletes any project code files. It only updates registry/tree state
//!   in ARCHITECTURE/.
//! - 100% pure Rust, zero Python, zero mocks, dynamic SHA-256 integrity, zero C-toolchain dependencies.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRecord {
    pub rel_path: String,
    pub name: String,
    pub extension: String,
    pub size_bytes: u64,
    pub line_count: usize,
    pub sha256: String,
    pub modified_epoch: u64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub file_id: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolderRecord {
    pub rel_path: String,
    pub name: String,
    pub depth: usize,
    pub direct_files_count: usize,
    pub direct_subdirs_count: usize,
    pub total_descendant_files_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    pub name: String,
    pub rel_path: String,
    pub entry_type: String, // "directory" or "file"
    pub size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub children: Vec<TreeNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureState {
    pub status: String,
    pub engine_name: String,
    pub engine_version: String,
    pub last_sync_utc: String,
    pub last_scan_mode: String,
    pub total_files: usize,
    pub total_folders: usize,
    pub reconciliation_duration_ms: u64,
    pub events_processed: u64,
    pub live_watcher_active: bool,
    pub live_watcher_mode: String,
    pub ignored_directories: Vec<String>,
    pub registries_digest_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconciliationReport {
    pub files_added: Vec<String>,
    pub files_updated: Vec<String>,
    pub files_removed: Vec<String>,
    #[serde(default)]
    pub files_moved: Vec<(String, String)>,
    pub folders_added: Vec<String>,
    pub folders_removed: Vec<String>,
    pub duration_ms: u64,
    pub total_files: usize,
    pub total_folders: usize,
}

#[derive(Debug, Clone)]
pub struct ArchitectureSyncConfig {
    pub workspace_root: PathBuf,
    pub arch_dir: PathBuf,
    pub ignored_dirs: Vec<String>,
    pub poll_interval_ms: u64,
    pub io_buffer_size: usize,
    pub max_text_file_bytes: u64,
}

impl Default for ArchitectureSyncConfig {
    fn default() -> Self {
        let ws = find_workspace_root().unwrap_or_else(|_| PathBuf::from("."));
        let arch = ws.join("ARCHITECTURE");
        let io_buf = std::env::var("TARA_SYNC_IO_BUF_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64 * 1024);
        let max_text = std::env::var("TARA_SYNC_MAX_TEXT_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10 * 1024 * 1024);
        let poll_interval = std::env::var("TARA_SYNC_POLL_INTERVAL_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(500);

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
            poll_interval_ms: poll_interval,
            io_buffer_size: io_buf,
            max_text_file_bytes: max_text,
        }
    }
}

pub struct ArchitectureSyncEngine {
    pub config: ArchitectureSyncConfig,
    file_registry: Arc<RwLock<BTreeMap<String, FileRecord>>>,
    folder_registry: Arc<RwLock<BTreeMap<String, FolderRecord>>>,
    state: Arc<RwLock<ArchitectureState>>,
    events_count: Arc<Mutex<u64>>,
    watcher_shutdown: Arc<Mutex<Option<std::sync::mpsc::Sender<()>>>>,
    link_engine: Arc<RwLock<tara_engine::ProjectLinkEngine>>,
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

/// Normalizes path separators to forward slash ('/') for cross-platform consistency.
pub fn normalize_rel_path(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}

/// Computes SHA-256 dynamically from file bytes with parameter-driven buffer size.
pub fn compute_dynamic_sha256(path: &Path, buffer_size: usize) -> std::io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
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

/// Computes line count for text files; 0 for binary files or files exceeding max_bytes.
pub fn count_lines_if_text(path: &Path, max_bytes: u64) -> usize {
    if let Ok(meta) = path.metadata() {
        if meta.len() > max_bytes {
            return 0; // Skip large files / data shards
        }
    }
    if let Ok(file) = File::open(path) {
        let reader = BufReader::new(file);
        let mut count = 0;
        for line in reader.lines() {
            if line.is_ok() {
                count += 1;
            } else {
                return 0; // Likely binary
            }
        }
        count
    } else {
        0
    }
}

/// Extracts OS-level unique file identity (file index on Windows NTFS/ReFS, inode on Unix)
#[cfg(windows)]
pub fn extract_file_identity_from_path(path: &Path) -> Option<u64> {
    use std::os::windows::io::AsRawHandle;
    let file = fs::File::open(path).ok()?;
    let handle = file.as_raw_handle();

    #[repr(C)]
    struct ByHandleFileInformation {
        dw_file_attributes: u32,
        ft_creation_time: [u32; 2],
        ft_last_access_time: [u32; 2],
        ft_last_write_time: [u32; 2],
        dw_volume_serial_number: u32,
        n_file_size_high: u32,
        n_file_size_low: u32,
        n_number_of_links: u32,
        n_file_index_high: u32,
        n_file_index_low: u32,
    }

    extern "system" {
        fn GetFileInformationByHandle(
            h_file: *mut std::ffi::c_void,
            lp_file_information: *mut ByHandleFileInformation,
        ) -> i32;
    }

    let mut info = ByHandleFileInformation {
        dw_file_attributes: 0,
        ft_creation_time: [0; 2],
        ft_last_access_time: [0; 2],
        ft_last_write_time: [0; 2],
        dw_volume_serial_number: 0,
        n_file_size_high: 0,
        n_file_size_low: 0,
        n_number_of_links: 0,
        n_file_index_high: 0,
        n_file_index_low: 0,
    };

    let ret = unsafe { GetFileInformationByHandle(handle, &mut info) };
    if ret != 0 {
        let index = ((info.n_file_index_high as u64) << 32) | (info.n_file_index_low as u64);
        Some(index)
    } else {
        None
    }
}

#[cfg(unix)]
pub fn extract_file_identity_from_path(path: &Path) -> Option<u64> {
    use std::os::unix::fs::MetadataExt;
    path.metadata().ok().map(|m| m.ino())
}

#[cfg(not(any(windows, unix)))]
pub fn extract_file_identity_from_path(_path: &Path) -> Option<u64> {
    None
}

impl ArchitectureSyncEngine {
    pub fn new(config: ArchitectureSyncConfig) -> Self {
        let initial_state = ArchitectureState {
            status: "INITIALIZING".to_string(),
            engine_name: "TARA Architecture Sync Engine".to_string(),
            engine_version: "1.0.0".to_string(),
            last_sync_utc: chrono_now_iso(),
            last_scan_mode: "UNINITIALIZED".to_string(),
            total_files: 0,
            total_folders: 0,
            reconciliation_duration_ms: 0,
            events_processed: 0,
            live_watcher_active: false,
            live_watcher_mode: "continuous_live_reconciler".to_string(),
            ignored_directories: config.ignored_dirs.clone(),
            registries_digest_sha256: String::new(),
        };

        let link_cfg = tara_engine::ProjectLinkConfig {
            workspace_root: config.workspace_root.clone(),
            arch_dir: config.arch_dir.clone(),
            ..Default::default()
        };
        let link_engine = Arc::new(RwLock::new(tara_engine::ProjectLinkEngine::new(link_cfg)));

        Self {
            config,
            file_registry: Arc::new(RwLock::new(BTreeMap::new())),
            folder_registry: Arc::new(RwLock::new(BTreeMap::new())),
            state: Arc::new(RwLock::new(initial_state)),
            events_count: Arc::new(Mutex::new(0)),
            watcher_shutdown: Arc::new(Mutex::new(None)),
            link_engine,
        }
    }

    /// Checks if a path should be ignored by the engine.
    pub fn is_ignored(&self, rel_path: &str) -> bool {
        let norm = rel_path.replace('\\', "/");
        let segments: Vec<&str> = norm.split('/').collect();
        for ignored in &self.config.ignored_dirs {
            if segments.iter().any(|seg| seg == ignored) {
                return true;
            }
        }
        // Also ignore temporary files or swap files
        if norm.ends_with(".tmp") || norm.ends_with(".swp") || norm.ends_with('~') {
            return true;
        }
        // Ignore internal registry files written by the sync engine itself to prevent loops
        if norm == "ARCHITECTURE/project_tree.json"
            || norm == "ARCHITECTURE/file_registry.json"
            || norm == "ARCHITECTURE/folder_registry.json"
            || norm == "ARCHITECTURE/architecture_state.json"
            || norm == "ARCHITECTURE/SOURCE_INDEX.json"
            || norm == "ARCHITECTURE/RELATIONSHIP_GRAPH.json"
            || norm == "ARCHITECTURE/GITHUB_EXPORT_MANIFEST.json"
            || norm == "architecture/project_tree.json"
            || norm == "architecture/file_registry.json"
            || norm == "architecture/folder_registry.json"
            || norm == "architecture/architecture_state.json"
            || norm == "architecture/SOURCE_INDEX.json"
            || norm == "architecture/RELATIONSHIP_GRAPH.json"
            || norm == "architecture/GITHUB_EXPORT_MANIFEST.json"
        {
            return true;
        }
        false
    }

    // ──────────────────────────────────────────────────────────────────────────
    // LAYER 2: FULL RECONCILER
    // ──────────────────────────────────────────────────────────────────────────

    /// Executes Layer 2: Exhaustive filesystem scan vs. registry reconciliation.
    ///
    /// Observes the entire filesystem state on disk, compares with registry,
    /// repairs any discrepancies, and updates project_tree.json, file_registry.json,
    /// folder_registry.json, and architecture_state.json.
    ///
    /// STRICT BOUNDARY: Never alters project code files. Only reads disk and writes registries.
    pub fn reconcile_full(&self) -> Result<ReconciliationReport, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let ws_root = &self.config.workspace_root;
        fs::create_dir_all(&self.config.arch_dir)?;

        // 1. Load existing registry files if present on disk
        self.load_registries_from_disk()?;

        let mut disk_files: BTreeMap<String, FileRecord> = BTreeMap::new();
        let mut disk_folders: BTreeMap<String, FolderRecord> = BTreeMap::new();

        // 2. Scan entire filesystem recursively
        self.scan_directory_recursive(ws_root, ws_root, &mut disk_files, &mut disk_folders)?;

        // 3. Diff disk vs. current memory registries
        let mut files_added = Vec::new();
        let mut files_updated = Vec::new();
        let mut files_removed = Vec::new();
        let mut folders_added = Vec::new();
        let mut folders_removed = Vec::new();

        let (
            report_files_moved,
            report_files_added,
            report_files_updated,
            report_files_removed,
            report_folders_added,
            report_folders_removed,
        ) = {
            let mut curr_files = self.file_registry.write().unwrap();
            let mut curr_folders = self.folder_registry.write().unwrap();

            // Detect added or updated files
            for (rel_path, disk_rec) in &disk_files {
                match curr_files.get(rel_path) {
                    None => {
                        files_added.push(rel_path.clone());
                    }
                    Some(curr_rec) => {
                        if curr_rec.sha256 != disk_rec.sha256
                            || curr_rec.size_bytes != disk_rec.size_bytes
                            || curr_rec.line_count != disk_rec.line_count
                        {
                            files_updated.push(rel_path.clone());
                        }
                    }
                }
            }

            // Detect removed files
            for curr_path in curr_files.keys() {
                if !disk_files.contains_key(curr_path) {
                    files_removed.push(curr_path.clone());
                }
            }

            // Detect genuine moves/renames:
            // 1. Primary Signal: OS Filesystem File Identity (NTFS file index / Unix inode)
            // 2. Secondary Signal: Candidate Content Identity (dynamic SHA-256 + byte size)
            let mut genuine_moved = Vec::new();
            let mut genuine_added = Vec::new();
            for added_path in files_added {
                if let Some(new_rec) = disk_files.get(&added_path) {
                    let mut matched_pos = None;

                    // Primary Signal: match OS filesystem file-ID where available
                    if let Some(new_fid) = new_rec.file_id {
                        matched_pos = files_removed.iter().position(|r_path| {
                            curr_files.get(r_path).and_then(|r| r.file_id) == Some(new_fid)
                        });
                    }

                    // Secondary Signal: candidate content-identity match (hash + size)
                    if matched_pos.is_none() {
                        matched_pos = files_removed.iter().position(|r_path| {
                            if let Some(old_rec) = curr_files.get(r_path) {
                                old_rec.sha256 == new_rec.sha256 && old_rec.size_bytes == new_rec.size_bytes
                            } else {
                                false
                            }
                        });
                    }

                    if let Some(pos) = matched_pos {
                        let old_path = files_removed.remove(pos);
                        genuine_moved.push((old_path, added_path));
                    } else {
                        genuine_added.push(added_path);
                    }
                } else {
                    genuine_added.push(added_path);
                }
            }
            files_added = genuine_added;

            // Detect added or removed folders
            for rel_folder in disk_folders.keys() {
                if !curr_folders.contains_key(rel_folder) {
                    folders_added.push(rel_folder.clone());
                }
            }
            for curr_folder in curr_folders.keys() {
                if !disk_folders.contains_key(curr_folder) {
                    folders_removed.push(curr_folder.clone());
                }
            }

            // Apply disk truth to registries
            *curr_files = disk_files;
            *curr_folders = disk_folders;

            (genuine_moved, files_added, files_updated, files_removed, folders_added, folders_removed)
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        // 4. Build hierarchical project tree
        let tree = self.build_project_tree()?;

        let registries_missing = !self.config.arch_dir.join("architecture_state.json").exists()
            || !self.config.arch_dir.join("file_registry.json").exists()
            || !self.config.arch_dir.join("folder_registry.json").exists()
            || !self.config.arch_dir.join("project_tree.json").exists()
            || !self.config.arch_dir.join("SOURCE_INDEX.json").exists()
            || !self.config.arch_dir.join("RELATIONSHIP_GRAPH.json").exists();

        let has_changes = registries_missing
            || !report_files_moved.is_empty()
            || !report_files_added.is_empty()
            || !report_files_updated.is_empty()
            || !report_files_removed.is_empty()
            || !report_folders_added.is_empty()
            || !report_folders_removed.is_empty();

        if has_changes {
            // 5. Synchronize Project Link & Reference Engine (AST / imports / in-out links / SOURCE_INDEX.json)
            if let Ok(mut le) = self.link_engine.write() {
                for (old_p, new_p) in &report_files_moved {
                    let _ = le.on_file_moved(old_p, new_p);
                }
                for old_f in &report_folders_removed {
                    if let Some(new_f) = report_folders_added.iter().find(|add_f| {
                        report_files_moved.iter().any(|(o, n)| o.starts_with(old_f.as_str()) && n.starts_with(add_f.as_str()))
                    }) {
                        let _ = le.on_folder_moved(old_f, new_f);
                    }
                }
                for rem_f in &report_folders_removed {
                    let _ = le.on_folder_deleted(rem_f);
                }
                for add_f in &report_folders_added {
                    let _ = le.on_folder_created(add_f);
                }
                for rem_p in &report_files_removed {
                    let _ = le.on_file_deleted(rem_p);
                }
                let _ = le.build_full_index();
                let _ = le.sync_architecture_maps();
            }

            // 6. Persist the 4 required output artifacts to ARCHITECTURE/
            self.persist_registries(&tree, duration_ms, "FULL_RECONCILER")?;
        }

        let (total_f, total_d) = {
            (
                self.file_registry.read().unwrap().len(),
                self.folder_registry.read().unwrap().len(),
            )
        };

        Ok(ReconciliationReport {
            files_added: report_files_added,
            files_updated: report_files_updated,
            files_removed: report_files_removed,
            files_moved: report_files_moved,
            folders_added: report_folders_added,
            folders_removed: report_folders_removed,
            duration_ms,
            total_files: total_f,
            total_folders: total_d,
        })
    }

    fn scan_directory_recursive(
        &self,
        root: &Path,
        current_dir: &Path,
        files_out: &mut BTreeMap<String, FileRecord>,
        folders_out: &mut BTreeMap<String, FolderRecord>,
    ) -> std::io::Result<usize> {
        let rel_folder = if current_dir == root {
            String::new()
        } else {
            let rel = current_dir.strip_prefix(root).unwrap_or(current_dir);
            normalize_rel_path(rel)
        };

        if !rel_folder.is_empty() && self.is_ignored(&rel_folder) {
            return Ok(0);
        }

        let entries = match fs::read_dir(current_dir) {
            Ok(e) => e,
            Err(_) => return Ok(0),
        };

        let mut direct_files = 0;
        let mut direct_subdirs = 0;
        let mut total_subtree_files = 0;

        let mut subdirs = Vec::new();

        for entry_res in entries {
            let entry = match entry_res {
                Ok(e) => e,
                Err(_) => continue,
            };
            let file_type = match entry.file_type() {
                Ok(ft) => ft,
                Err(_) => continue,
            };
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();

            if file_type.is_dir() {
                if !self.is_ignored(&name) {
                    direct_subdirs += 1;
                    subdirs.push(path);
                }
            } else if file_type.is_file() {
                let rel = path.strip_prefix(root).unwrap_or(&path);
                let norm = normalize_rel_path(rel);
                if !self.is_ignored(&norm) {
                    direct_files += 1;
                    total_subtree_files += 1;

                    let metadata = entry.metadata().ok();
                    let size_bytes = metadata.as_ref().map(|m| m.len()).unwrap_or(0);
                    let modified_epoch = metadata
                        .as_ref()
                        .and_then(|m| m.modified().ok())
                        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    let extension = path
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("")
                        .to_string();
                    let sha256 = compute_dynamic_sha256(&path, self.config.io_buffer_size).unwrap_or_default();
                    let line_count = count_lines_if_text(&path, self.config.max_text_file_bytes);

                    let file_id = extract_file_identity_from_path(&path);

                    let record = FileRecord {
                        rel_path: norm.clone(),
                        name,
                        extension,
                        size_bytes,
                        line_count,
                        sha256,
                        modified_epoch,
                        file_id,
                    };
                    files_out.insert(norm, record);
                }
            }
        }

        for sub in subdirs {
            let child_files = self.scan_directory_recursive(root, &sub, files_out, folders_out)?;
            total_subtree_files += child_files;
        }

        if !rel_folder.is_empty() {
            let depth = rel_folder.split('/').count();
            let folder_name = Path::new(&rel_folder)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(&rel_folder)
                .to_string();

            let folder_rec = FolderRecord {
                rel_path: rel_folder.clone(),
                name: folder_name,
                depth,
                direct_files_count: direct_files,
                direct_subdirs_count: direct_subdirs,
                total_descendant_files_count: total_subtree_files,
            };
            folders_out.insert(rel_folder, folder_rec);
        }

        Ok(total_subtree_files)
    }

    /// Builds a hierarchical project tree from the filesystem.
    pub fn build_project_tree(&self) -> Result<TreeNode, Box<dyn std::error::Error>> {
        let ws = &self.config.workspace_root;
        let root_name = ws
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("tara")
            .to_string();

        let files = self.file_registry.read().unwrap();
        let folders = self.folder_registry.read().unwrap();

        fn build_node(
            name: String,
            rel_path: String,
            files: &BTreeMap<String, FileRecord>,
            folders: &BTreeMap<String, FolderRecord>,
        ) -> TreeNode {
            let mut children = Vec::new();

            // Find immediate child folders
            let prefix = if rel_path.is_empty() {
                String::new()
            } else {
                format!("{}/", rel_path)
            };

            let mut immediate_dirs: BTreeMap<String, String> = BTreeMap::new();
            for f_path in folders.keys() {
                if f_path.starts_with(&prefix) {
                    let sub = &f_path[prefix.len()..];
                    if !sub.contains('/') && !sub.is_empty() {
                        immediate_dirs.insert(sub.to_string(), f_path.clone());
                    }
                }
            }

            for (dir_name, full_rel) in immediate_dirs {
                children.push(build_node(dir_name, full_rel, files, folders));
            }

            // Find immediate child files
            for (f_path, f_rec) in files {
                if f_path.starts_with(&prefix) {
                    let sub = &f_path[prefix.len()..];
                    if !sub.contains('/') && !sub.is_empty() {
                        children.push(TreeNode {
                            name: f_rec.name.clone(),
                            rel_path: f_rec.rel_path.clone(),
                            entry_type: "file".to_string(),
                            size_bytes: f_rec.size_bytes,
                            sha256: Some(f_rec.sha256.clone()),
                            children: Vec::new(),
                        });
                    }
                }
            }

            children.sort_by(|a, b| {
                if a.entry_type == b.entry_type {
                    a.name.cmp(&b.name)
                } else if a.entry_type == "directory" {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            });

            TreeNode {
                name,
                rel_path: if rel_path.is_empty() { ".".to_string() } else { rel_path },
                entry_type: "directory".to_string(),
                size_bytes: 0,
                sha256: None,
                children,
            }
        }

        Ok(build_node(root_name, String::new(), &files, &folders))
    }

    /// Persists the 4 output artifacts into ARCHITECTURE/:
    /// 1. project_tree.json
    /// 2. file_registry.json
    /// 3. folder_registry.json
    /// 4. architecture_state.json
    pub fn persist_registries(
        &self,
        tree: &TreeNode,
        duration_ms: u64,
        scan_mode: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let arch_dir = &self.config.arch_dir;
        fs::create_dir_all(arch_dir)?;

        let files = self.file_registry.read().unwrap();
        let folders = self.folder_registry.read().unwrap();

        // 1. project_tree.json
        let tree_json = serde_json::to_string_pretty(tree)?;
        fs::write(arch_dir.join("project_tree.json"), &tree_json)?;

        // 2. file_registry.json
        let file_registry_obj = json!({
            "schema_version": "1.0.0",
            "generated_at_utc": chrono_now_iso(),
            "total_files": files.len(),
            "files": *files
        });
        let file_reg_json = serde_json::to_string_pretty(&file_registry_obj)?;
        fs::write(arch_dir.join("file_registry.json"), &file_reg_json)?;

        // 3. folder_registry.json
        let folder_registry_obj = json!({
            "schema_version": "1.0.0",
            "generated_at_utc": chrono_now_iso(),
            "total_folders": folders.len(),
            "folders": *folders
        });
        let folder_reg_json = serde_json::to_string_pretty(&folder_registry_obj)?;
        fs::write(arch_dir.join("folder_registry.json"), &folder_reg_json)?;

        // Compute combined dynamic SHA-256 of the 3 registries
        let mut hasher = Sha256::new();
        hasher.update(tree_json.as_bytes());
        hasher.update(file_reg_json.as_bytes());
        hasher.update(folder_reg_json.as_bytes());
        let digest_sha = hex::encode(hasher.finalize());

        // 4. architecture_state.json
        let events = *self.events_count.lock().unwrap();
        let state_obj = ArchitectureState {
            status: "SYNCED".to_string(),
            engine_name: "TARA Architecture Sync Engine".to_string(),
            engine_version: "1.0.0".to_string(),
            last_sync_utc: chrono_now_iso(),
            last_scan_mode: scan_mode.to_string(),
            total_files: files.len(),
            total_folders: folders.len(),
            reconciliation_duration_ms: duration_ms,
            events_processed: events,
            live_watcher_active: self.state.read().unwrap().live_watcher_active,
            live_watcher_mode: "continuous_live_reconciler".to_string(),
            ignored_directories: self.config.ignored_dirs.clone(),
            registries_digest_sha256: digest_sha,
        };

        let state_json = serde_json::to_string_pretty(&state_obj)?;
        fs::write(arch_dir.join("architecture_state.json"), &state_json)?;

        // Update in-memory state
        *self.state.write().unwrap() = state_obj;

        Ok(())
    }

    fn load_registries_from_disk(&self) -> Result<(), Box<dyn std::error::Error>> {
        let f_path = self.config.arch_dir.join("file_registry.json");
        if f_path.exists() {
            if let Ok(raw) = fs::read_to_string(&f_path) {
                let clean = raw.trim_start_matches('\u{feff}');
                if let Ok(v) = serde_json::from_str::<Value>(clean) {
                    if let Some(files_obj) = v.get("files") {
                        if let Ok(deserialized) = serde_json::from_value::<BTreeMap<String, FileRecord>>(files_obj.clone()) {
                            *self.file_registry.write().unwrap() = deserialized;
                        }
                    }
                }
            }
        }

        let d_path = self.config.arch_dir.join("folder_registry.json");
        if d_path.exists() {
            if let Ok(raw) = fs::read_to_string(&d_path) {
                let clean = raw.trim_start_matches('\u{feff}');
                if let Ok(v) = serde_json::from_str::<Value>(clean) {
                    if let Some(folders_obj) = v.get("folders") {
                        if let Ok(deserialized) = serde_json::from_value::<BTreeMap<String, FolderRecord>>(folders_obj.clone()) {
                            *self.folder_registry.write().unwrap() = deserialized;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────────────
    // LAYER 1: LIVE WATCHER & EVENT DISPATCH
    // ──────────────────────────────────────────────────────────────────────────

    /// Handles a created file event.
    pub fn on_file_created(&self, abs_path: &Path) -> Result<bool, Box<dyn std::error::Error>> {
        let ws = &self.config.workspace_root;
        let rel = match abs_path.strip_prefix(ws) {
            Ok(r) => normalize_rel_path(r),
            Err(_) => return Ok(false),
        };

        if self.is_ignored(&rel) || !abs_path.is_file() {
            return Ok(false);
        }

        let name = abs_path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
        let extension = abs_path.extension().and_then(|e| e.to_str()).unwrap_or("").to_string();
        let metadata = abs_path.metadata().ok();
        let size_bytes = metadata.as_ref().map(|m| m.len()).unwrap_or(0);
        let modified_epoch = metadata
            .as_ref()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let sha256 = compute_dynamic_sha256(abs_path, self.config.io_buffer_size).unwrap_or_default();
        let line_count = count_lines_if_text(abs_path, self.config.max_text_file_bytes);
        let file_id = extract_file_identity_from_path(abs_path);

        let rec = FileRecord {
            rel_path: rel.clone(),
            name,
            extension,
            size_bytes,
            line_count,
            sha256,
            modified_epoch,
            file_id,
        };

        self.file_registry.write().unwrap().insert(rel, rec);
        *self.events_count.lock().unwrap() += 1;
        self.refresh_state_and_persist("LIVE_WATCHER_FILE_CREATED")?;
        Ok(true)
    }

    /// Handles a modified file event.
    pub fn on_file_modified(&self, abs_path: &Path) -> Result<bool, Box<dyn std::error::Error>> {
        let ws = &self.config.workspace_root;
        let rel = match abs_path.strip_prefix(ws) {
            Ok(r) => normalize_rel_path(r),
            Err(_) => return Ok(false),
        };

        if self.is_ignored(&rel) || !abs_path.is_file() {
            return Ok(false);
        }

        let sha256 = match compute_dynamic_sha256(abs_path, self.config.io_buffer_size) {
            Ok(s) => s,
            Err(_) => return Ok(false),
        };

        // Check if anything actually changed
        {
            let files = self.file_registry.read().unwrap();
            if let Some(existing) = files.get(&rel) {
                if existing.sha256 == sha256 {
                    return Ok(false); // No content change
                }
            }
        }

        let name = abs_path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
        let extension = abs_path.extension().and_then(|e| e.to_str()).unwrap_or("").to_string();
        let metadata = abs_path.metadata().ok();
        let size_bytes = metadata.as_ref().map(|m| m.len()).unwrap_or(0);
        let modified_epoch = metadata
            .as_ref()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let line_count = count_lines_if_text(abs_path, self.config.max_text_file_bytes);
        let file_id = extract_file_identity_from_path(abs_path);

        let rec = FileRecord {
            rel_path: rel.clone(),
            name,
            extension,
            size_bytes,
            line_count,
            sha256,
            modified_epoch,
            file_id,
        };

        self.file_registry.write().unwrap().insert(rel, rec);
        *self.events_count.lock().unwrap() += 1;
        self.refresh_state_and_persist("LIVE_WATCHER_FILE_MODIFIED")?;
        Ok(true)
    }

    /// Handles a deleted file event.
    pub fn on_file_deleted(&self, abs_path: &Path) -> Result<bool, Box<dyn std::error::Error>> {
        let ws = &self.config.workspace_root;
        let rel = match abs_path.strip_prefix(ws) {
            Ok(r) => normalize_rel_path(r),
            Err(_) => return Ok(false),
        };

        let removed = self.file_registry.write().unwrap().remove(&rel).is_some();
        if removed {
            *self.events_count.lock().unwrap() += 1;
            self.refresh_state_and_persist("LIVE_WATCHER_FILE_DELETED")?;
        }
        Ok(removed)
    }

    /// Handles a folder created event.
    pub fn on_folder_created(&self, abs_path: &Path) -> Result<bool, Box<dyn std::error::Error>> {
        let ws = &self.config.workspace_root;
        let rel = match abs_path.strip_prefix(ws) {
            Ok(r) => normalize_rel_path(r),
            Err(_) => return Ok(false),
        };

        if rel.is_empty() || self.is_ignored(&rel) {
            return Ok(false);
        }

        let depth = rel.split('/').count();
        let name = abs_path.file_name().and_then(|n| n.to_str()).unwrap_or(&rel).to_string();

        let rec = FolderRecord {
            rel_path: rel.clone(),
            name,
            depth,
            direct_files_count: 0,
            direct_subdirs_count: 0,
            total_descendant_files_count: 0,
        };

        self.folder_registry.write().unwrap().insert(rel, rec);
        *self.events_count.lock().unwrap() += 1;
        self.refresh_state_and_persist("LIVE_WATCHER_FOLDER_CREATED")?;
        Ok(true)
    }

    /// Handles a folder deleted event.
    pub fn on_folder_deleted(&self, abs_path: &Path) -> Result<bool, Box<dyn std::error::Error>> {
        let ws = &self.config.workspace_root;
        let rel = match abs_path.strip_prefix(ws) {
            Ok(r) => normalize_rel_path(r),
            Err(_) => return Ok(false),
        };

        let prefix = format!("{}/", rel);
        let mut removed_count = 0;

        {
            let mut folders = self.folder_registry.write().unwrap();
            if folders.remove(&rel).is_some() {
                removed_count += 1;
            }
            folders.retain(|k, _| {
                if k.starts_with(&prefix) {
                    removed_count += 1;
                    false
                } else {
                    true
                }
            });
        }

        // Also clean up any files that were in that folder
        {
            let mut files = self.file_registry.write().unwrap();
            files.retain(|k, _| !k.starts_with(&prefix));
        }

        if removed_count > 0 {
            *self.events_count.lock().unwrap() += 1;
            self.refresh_state_and_persist("LIVE_WATCHER_FOLDER_DELETED")?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Handles a renamed/moved event (both files and folders).
    pub fn on_path_renamed(&self, from: &Path, to: &Path) -> Result<bool, Box<dyn std::error::Error>> {
        let ws = &self.config.workspace_root;
        let rel_from = from.strip_prefix(ws).map(normalize_rel_path).ok();
        let rel_to = to.strip_prefix(ws).map(normalize_rel_path).ok();

        if let (Some(rf), Some(_rt)) = (rel_from, rel_to) {
            if to.is_file() {
                let _ = self.on_file_deleted(from);
                let _ = self.on_file_created(to);
                return Ok(true);
            } else if to.is_dir() {
                let _ = self.on_folder_deleted(from);
                let _ = self.on_folder_created(to);
                return Ok(true);
            } else {
                self.file_registry.write().unwrap().remove(&rf);
                self.folder_registry.write().unwrap().remove(&rf);
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn refresh_state_and_persist(&self, mode: &str) -> Result<(), Box<dyn std::error::Error>> {
        let tree = self.build_project_tree()?;
        self.persist_registries(&tree, 0, mode)?;
        Ok(())
    }

    /// Fast incremental tick for the Live Watcher:
    /// Scans filesystem timestamps and sizes in milliseconds.
    /// Emits discrete create, modify, delete, and rename events to the registries.
    pub fn live_watcher_tick(&self) -> Result<usize, Box<dyn std::error::Error>> {
        let ws = &self.config.workspace_root;
        let mut changes = 0;

        // Collect current disk state (fast stat) for files and folders
        let mut disk_stats: BTreeMap<String, (u64, u64, Option<u64>)> = BTreeMap::new();
        let mut disk_folders: BTreeSet<String> = BTreeSet::new();
        self.collect_fast_stats(ws, ws, &mut disk_stats, &mut disk_folders)?;

        // Lock registries for synchronized updates
        let mut curr_files = self.file_registry.write().unwrap();
        let mut curr_folders = self.folder_registry.write().unwrap();

        let mut poll_folders_added: Vec<String> = Vec::new();
        let mut poll_folders_removed: Vec<String> = Vec::new();

        // 1. Folders added
        for rel_folder in &disk_folders {
            if !curr_folders.contains_key(rel_folder) {
                let depth = rel_folder.split('/').count();
                let name = Path::new(rel_folder)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(rel_folder)
                    .to_string();
                let rec = FolderRecord {
                    rel_path: rel_folder.clone(),
                    name,
                    depth,
                    direct_files_count: 0,
                    direct_subdirs_count: 0,
                    total_descendant_files_count: 0,
                };
                curr_folders.insert(rel_folder.clone(), rec);
                poll_folders_added.push(rel_folder.clone());
                changes += 1;
            }
        }

        // 2. Folders removed
        let registered_folder_keys: Vec<String> = curr_folders.keys().cloned().collect();
        for reg_folder in registered_folder_keys {
            if !disk_folders.contains(&reg_folder) {
                curr_folders.remove(&reg_folder);
                poll_folders_removed.push(reg_folder.clone());
                changes += 1;
            }
        }

        // 3. Files removed: collect candidate removals
        let mut candidate_removed: BTreeMap<String, FileRecord> = BTreeMap::new();
        let registered_file_keys: Vec<String> = curr_files.keys().cloned().collect();
        for rel_path in registered_file_keys {
            if !disk_stats.contains_key(&rel_path) {
                if let Some(removed_rec) = curr_files.remove(&rel_path) {
                    candidate_removed.insert(rel_path, removed_rec);
                    changes += 1;
                }
            }
        }

        // 4. Files added: detect genuine moves via:
        //    (a) Primary signal: OS Filesystem File Identity (NTFS file index / Unix inode)
        //    (b) Secondary signal: Candidate dynamic SHA-256 and size matching
        let mut genuine_moves: Vec<(String, String)> = Vec::new();
        for (rel_path, &(disk_mtime, disk_size, disk_fid)) in &disk_stats {
            if !curr_files.contains_key(rel_path) {
                let full_path = ws.join(rel_path);
                if full_path.is_file() {
                    let name = full_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("")
                        .to_string();
                    let extension = full_path
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("")
                        .to_string();
                    let sha256 = compute_dynamic_sha256(&full_path, self.config.io_buffer_size).unwrap_or_default();
                    let line_count = count_lines_if_text(&full_path, self.config.max_text_file_bytes);
                    let file_id = disk_fid.or_else(|| extract_file_identity_from_path(&full_path));

                    // Primary Signal check: OS Filesystem File Identity
                    let move_candidate = if let Some(fid) = file_id {
                        candidate_removed
                            .iter()
                            .find(|(_, rem_rec)| rem_rec.file_id == Some(fid))
                            .map(|(p, _)| (p.clone(), "filesystem file-ID"))
                    } else {
                        None
                    };

                    // Secondary Signal check: Candidate dynamic SHA-256 and size matching
                    let move_candidate = move_candidate.or_else(|| {
                        candidate_removed
                            .iter()
                            .find(|(_, rem_rec)| rem_rec.sha256 == sha256 && rem_rec.size_bytes == disk_size)
                            .map(|(p, _)| (p.clone(), "candidate content-identity (hash+size)"))
                    });

                    if let Some((old_p, _signal_type)) = move_candidate {
                        genuine_moves.push((old_p.clone(), rel_path.clone()));
                        candidate_removed.remove(&old_p);
                    }

                    let rec = FileRecord {
                        rel_path: rel_path.clone(),
                        name,
                        extension,
                        size_bytes: disk_size,
                        line_count,
                        sha256,
                        modified_epoch: disk_mtime,
                        file_id,
                    };
                    curr_files.insert(rel_path.clone(), rec);
                    Self::ensure_ancestor_folders(rel_path, &mut curr_folders);
                    changes += 1;
                }
            }
        }

        // 5. Files modified: detect genuine content change
        for (rel_path, &(disk_mtime, disk_size, disk_fid)) in &disk_stats {
            if let Some(existing) = curr_files.get(rel_path) {
                if existing.modified_epoch != disk_mtime || existing.size_bytes != disk_size {
                    let full_path = ws.join(rel_path);
                    if let Ok(new_sha) = compute_dynamic_sha256(&full_path, self.config.io_buffer_size) {
                        if new_sha != existing.sha256 || disk_size != existing.size_bytes {
                            let line_count = count_lines_if_text(&full_path, self.config.max_text_file_bytes);
                            let file_id = disk_fid.or_else(|| extract_file_identity_from_path(&full_path));
                            let updated_rec = FileRecord {
                                rel_path: rel_path.clone(),
                                name: existing.name.clone(),
                                extension: existing.extension.clone(),
                                size_bytes: disk_size,
                                line_count,
                                sha256: new_sha,
                                modified_epoch: disk_mtime,
                                file_id,
                            };
                            curr_files.insert(rel_path.clone(), updated_rec);
                            changes += 1;
                        }
                    }
                }
            }
        }

        // 6. If any change occurred, recalculate folder metrics and persist all 4 registries
        if changes > 0 {
            Self::recalculate_folder_metrics(&mut curr_folders, &curr_files);

            // Release write locks before persisting (persist acquires read locks)
            drop(curr_files);
            drop(curr_folders);

            // 7. Synchronize code relationships & architecture maps (Project Link & Reference Engine)
            if let Ok(mut le) = self.link_engine.write() {
                for (old_p, new_p) in &genuine_moves {
                    let _ = le.on_file_moved(old_p, new_p);
                }
                for old_f in &poll_folders_removed {
                    if let Some(new_f) = poll_folders_added.iter().find(|add_f| {
                        genuine_moves.iter().any(|(o, n)| o.starts_with(old_f.as_str()) && n.starts_with(add_f.as_str()))
                    }) {
                        let _ = le.on_folder_moved(old_f, new_f);
                    }
                }
                for rem_f in &poll_folders_removed {
                    let _ = le.on_folder_deleted(rem_f);
                }
                for add_f in &poll_folders_added {
                    let _ = le.on_folder_created(add_f);
                }
                for rem_p in candidate_removed.keys() {
                    let _ = le.on_file_deleted(rem_p);
                }
                let _ = le.build_full_index();
                let _ = le.sync_architecture_maps();
            }

            *self.events_count.lock().unwrap() += changes as u64;
            self.refresh_state_and_persist("LIVE_WATCHER_SYNC")?;
        }

        Ok(changes)
    }

    fn collect_fast_stats(
        &self,
        root: &Path,
        dir: &Path,
        files_out: &mut BTreeMap<String, (u64, u64, Option<u64>)>,
        folders_out: &mut BTreeSet<String>,
    ) -> std::io::Result<()> {
        let rel_folder = if dir == root {
            String::new()
        } else {
            let rel = dir.strip_prefix(root).unwrap_or(dir);
            normalize_rel_path(rel)
        };

        if !rel_folder.is_empty() {
            if self.is_ignored(&rel_folder) {
                return Ok(());
            }
            folders_out.insert(rel_folder);
        }

        let entries = match fs::read_dir(dir) {
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
                    let rel = path.strip_prefix(root).unwrap_or(&path);
                    let norm = normalize_rel_path(rel);
                    if !self.is_ignored(&norm) {
                        folders_out.insert(norm);
                    }
                    self.collect_fast_stats(root, &path, files_out, folders_out)?;
                }
            } else if ft.is_file() {
                let rel = path.strip_prefix(root).unwrap_or(&path);
                let norm = normalize_rel_path(rel);
                if !self.is_ignored(&norm) {
                    if let Ok(meta) = entry.metadata() {
                        let size = meta.len();
                        let mtime = meta
                            .modified()
                            .ok()
                            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                            .map(|d| d.as_secs())
                            .unwrap_or(0);
                        let file_id = extract_file_identity_from_path(&path);
                        files_out.insert(norm, (mtime, size, file_id));
                    }
                }
            }
        }

        Ok(())
    }

fn ensure_ancestor_folders(
    rel_path: &str,
    folders: &mut BTreeMap<String, FolderRecord>,
) {
    let mut parts: Vec<&str> = rel_path.split('/').collect();
    if parts.len() > 1 {
        parts.pop(); // remove file name
        let mut curr = String::new();
        for (idx, part) in parts.iter().enumerate() {
            if !curr.is_empty() {
                curr.push('/');
            }
            curr.push_str(part);
            if !folders.contains_key(&curr) {
                let name = part.to_string();
                let depth = idx + 1;
                folders.insert(
                    curr.clone(),
                    FolderRecord {
                        rel_path: curr.clone(),
                        name,
                        depth,
                        direct_files_count: 0,
                        direct_subdirs_count: 0,
                        total_descendant_files_count: 0,
                    },
                );
            }
        }
    }
}

fn recalculate_folder_metrics(
    folders: &mut BTreeMap<String, FolderRecord>,
    files: &BTreeMap<String, FileRecord>,
) {
    let folder_keys: Vec<String> = folders.keys().cloned().collect();
    for f_path in folder_keys {
        let prefix = format!("{}/", f_path);
        let mut direct_files = 0;
        let mut descendant_files = 0;
        for file_path in files.keys() {
            if file_path.starts_with(&prefix) {
                descendant_files += 1;
                let rest = &file_path[prefix.len()..];
                if !rest.contains('/') {
                    direct_files += 1;
                }
            }
        }
        let mut direct_subdirs = 0;
        for other_folder in folders.keys() {
            if other_folder.starts_with(&prefix) {
                let rest = &other_folder[prefix.len()..];
                if !rest.contains('/') && !rest.is_empty() {
                    direct_subdirs += 1;
                }
            }
        }
        if let Some(rec) = folders.get_mut(&f_path) {
            rec.direct_files_count = direct_files;
            rec.direct_subdirs_count = direct_subdirs;
            rec.total_descendant_files_count = descendant_files;
        }
    }
}

    /// Starts Layer 1: The Live Watcher background thread.
    /// Runs continuous ticks without external non-Rust C-dependencies.
    /// Returns a shutdown sender to cleanly stop the watcher when needed.
    pub fn start_live_watcher(self: Arc<Self>) -> Result<std::sync::mpsc::Sender<()>, Box<dyn std::error::Error>> {
        let (shutdown_tx, shutdown_rx) = std::sync::mpsc::channel::<()>();
        *self.watcher_shutdown.lock().unwrap() = Some(shutdown_tx.clone());

        let engine = self.clone();
        let poll_interval = Duration::from_millis(self.config.poll_interval_ms);

        std::thread::Builder::new()
            .name("tara-architecture-live-watcher".to_string())
            .spawn(move || {
                loop {
                    // Check if already signaled to shutdown
                    if shutdown_rx.try_recv().is_ok() {
                        break;
                    }

                    // Mandatory startup/recovery reconciliation before live watching (Rule 20)
                    let _ = engine.reconcile_full();
                    engine.state.write().unwrap().live_watcher_active = true;
                    let _ = engine.refresh_state_and_persist("LIVE_WATCHER_ACTIVE");
                    println!(
                        "[ArchitectureSyncEngine] Live Watcher ACTIVE on workspace: {}",
                        engine.config.workspace_root.display()
                    );

                    let eng_ref = &engine;
                    let rx_ref = &shutdown_rx;
                    let watch_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        loop {
                            match rx_ref.recv_timeout(poll_interval) {
                                Ok(()) => {
                                    println!("[ArchitectureSyncEngine] Live Watcher shutting down.");
                                    return true; // clean shutdown
                                }
                                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                                    let _ = eng_ref.live_watcher_tick();
                                }
                                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                                    return true; // clean shutdown
                                }
                            }
                        }
                    }));

                    match watch_result {
                        Ok(true) => {
                            // Clean shutdown requested
                            break;
                        }
                        Ok(false) => {
                            break;
                        }
                        Err(_) => {
                            eprintln!(
                                "[ArchitectureSyncEngine] RESILIENT RECOVERY: Live Watcher recovered from unexpected panic. Reconciling missed changes..."
                            );
                            std::thread::sleep(Duration::from_millis(1000));
                            // Loop continues: reconcile_full() will execute to recover missed changes
                        }
                    }
                }

                engine.state.write().unwrap().live_watcher_active = false;
                let _ = engine.refresh_state_and_persist("LIVE_WATCHER_STOPPED");
            })?;

        Ok(shutdown_tx)
    }

    /// Stops the live watcher thread if running.
    pub fn stop_live_watcher(&self) {
        if let Some(tx) = self.watcher_shutdown.lock().unwrap().take() {
            let _ = tx.send(());
        }
    }

    /// Returns the current architecture state snapshot.
    pub fn get_state(&self) -> ArchitectureState {
        self.state.read().unwrap().clone()
    }

    /// Returns current file registry as JSON value.
    pub fn get_file_registry_json(&self) -> Value {
        let files = self.file_registry.read().unwrap();
        json!({
            "schema_version": "1.0.0",
            "generated_at_utc": chrono_now_iso(),
            "total_files": files.len(),
            "files": *files
        })
    }

    /// Returns current folder registry as JSON value.
    pub fn get_folder_registry_json(&self) -> Value {
        let folders = self.folder_registry.read().unwrap();
        json!({
            "schema_version": "1.0.0",
            "generated_at_utc": chrono_now_iso(),
            "total_folders": folders.len(),
            "folders": *folders
        })
    }

    /// Returns current project tree as JSON value.
    pub fn get_project_tree_json(&self) -> Result<Value, Box<dyn std::error::Error>> {
        let tree = self.build_project_tree()?;
        Ok(serde_json::to_value(tree)?)
    }
}

impl Drop for ArchitectureSyncEngine {
    fn drop(&mut self) {
        self.stop_live_watcher();
    }
}

fn chrono_now_iso() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{now}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_sync_reconciler_and_artifacts() {
        let root = find_workspace_root().expect("workspace root");
        let temp_arch = root.join("target").join("test_architecture_sync");
        let _ = fs::remove_dir_all(&temp_arch);

        let config = ArchitectureSyncConfig {
            workspace_root: root.clone(),
            arch_dir: temp_arch.clone(),
            poll_interval_ms: 100,
            ..Default::default()
        };

        let engine = ArchitectureSyncEngine::new(config);
        let report = engine.reconcile_full().expect("reconcile_full");

        assert!(report.total_files > 50, "Expected > 50 files in workspace");
        assert!(report.total_folders > 10, "Expected > 10 folders in workspace");

        // Verify the 4 artifacts were created
        assert!(temp_arch.join("project_tree.json").exists());
        assert!(temp_arch.join("file_registry.json").exists());
        assert!(temp_arch.join("folder_registry.json").exists());
        assert!(temp_arch.join("architecture_state.json").exists());

        // Verify JSON readability
        let tree_raw = fs::read_to_string(temp_arch.join("project_tree.json")).unwrap();
        let tree: Value = serde_json::from_str(&tree_raw).unwrap();
        assert_eq!(tree["entry_type"], "directory");

        let state_raw = fs::read_to_string(temp_arch.join("architecture_state.json")).unwrap();
        let state: ArchitectureState = serde_json::from_str(&state_raw).unwrap();
        assert_eq!(state.status, "SYNCED");
        assert_eq!(state.last_scan_mode, "FULL_RECONCILER");

        // Test Live Watcher single tick
        let tick_changes = engine.live_watcher_tick().expect("live_watcher_tick");
        assert_eq!(tick_changes, 0, "No changes expected right after full reconcile");

        // Idempotence Verification: Reconcile pass 2 on unchanged filesystem must produce 0 churn and 100% byte-identical state
        let tree_bytes_pass1 = fs::read(temp_arch.join("project_tree.json")).unwrap();
        let file_bytes_pass1 = fs::read(temp_arch.join("file_registry.json")).unwrap();
        let folder_bytes_pass1 = fs::read(temp_arch.join("folder_registry.json")).unwrap();
        let state_bytes_pass1 = fs::read(temp_arch.join("architecture_state.json")).unwrap();

        let report2 = engine.reconcile_full().expect("reconcile_full pass 2");
        assert_eq!(report2.files_added.len(), 0, "Pass 2 must add 0 files");
        assert_eq!(report2.files_updated.len(), 0, "Pass 2 must update 0 files");
        assert_eq!(report2.files_removed.len(), 0, "Pass 2 must remove 0 files");
        assert_eq!(report2.files_moved.len(), 0, "Pass 2 must move 0 files");
        assert_eq!(report2.folders_added.len(), 0, "Pass 2 must add 0 folders");
        assert_eq!(report2.folders_removed.len(), 0, "Pass 2 must remove 0 folders");

        let tree_bytes_pass2 = fs::read(temp_arch.join("project_tree.json")).unwrap();
        let file_bytes_pass2 = fs::read(temp_arch.join("file_registry.json")).unwrap();
        let folder_bytes_pass2 = fs::read(temp_arch.join("folder_registry.json")).unwrap();
        let state_bytes_pass2 = fs::read(temp_arch.join("architecture_state.json")).unwrap();

        assert_eq!(tree_bytes_pass1, tree_bytes_pass2, "project_tree.json must be 100% byte-identical across passes");
        assert_eq!(file_bytes_pass1, file_bytes_pass2, "file_registry.json must be 100% byte-identical across passes");
        assert_eq!(folder_bytes_pass1, folder_bytes_pass2, "folder_registry.json must be 100% byte-identical across passes");
        assert_eq!(state_bytes_pass1, state_bytes_pass2, "architecture_state.json must be 100% byte-identical across passes");

        let _ = fs::remove_dir_all(&temp_arch);
    }

    #[test]
    fn test_live_watcher_full_lifecycle_create_edit_move_delete() {
        let root = find_workspace_root().expect("workspace root");
        let sandbox = root.join("target").join("test_lifecycle_sandbox");
        let _ = fs::remove_dir_all(&sandbox);
        fs::create_dir_all(&sandbox).expect("create sandbox");

        let arch_dir = sandbox.join("ARCHITECTURE");
        let config = ArchitectureSyncConfig {
            workspace_root: sandbox.clone(),
            arch_dir: arch_dir.clone(),
            ignored_dirs: vec!["target".to_string(), ".git".to_string()],
            poll_interval_ms: 50,
            ..Default::default()
        };

        let engine = ArchitectureSyncEngine::new(config);
        // Initial reconciliation on empty sandbox
        let initial_report = engine.reconcile_full().expect("initial reconcile");
        assert_eq!(initial_report.total_files, 0);

        // -------------------------------------------------------------
        // STEP 1: CREATE 12 files across 4 different locations
        // -------------------------------------------------------------
        let test_files = vec![
            ("loc_a/file_1.txt", "Hello from loc_a file 1\n"),
            ("loc_a/file_2.rs", "fn a2() { println!(\"a2\"); }\n"),
            ("loc_a/file_3.json", "{\"key\": \"value_a3\"}\n"),
            ("loc_b/sub_1/doc_1.md", "# Document 1 in loc_b\nContent here.\n"),
            ("loc_b/sub_1/doc_2.toml", "name = \"doc_2\"\nversion = \"1.0\"\n"),
            ("loc_b/sub_2/doc_3.txt", "Third document in sub_2\n"),
            ("loc_c/alpha/beta/gamma/module.rs", "pub fn gamma() -> i32 { 42 }\n"),
            ("loc_c/alpha/beta/gamma/config.yaml", "mode: test\ngamma: true\n"),
            ("loc_c/alpha/shared.rs", "pub const SHARED: u64 = 100;\n"),
            ("loc_d/service_1.rs", "pub struct S1;\n"),
            ("loc_d/service_2.rs", "pub struct S2;\n"),
            ("loc_d/schema.proto", "syntax = \"proto3\";\nmessage Msg {}\n"),
        ];

        for (rel, content) in &test_files {
            let full = sandbox.join(rel);
            if let Some(parent) = full.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(&full, content).unwrap();
        }

        let changes_step1 = engine.live_watcher_tick().expect("live_watcher_tick create");
        assert!(changes_step1 >= 12, "Expected at least 12 changes detected on create");

        // Verify all 12 files in registry and on disk
        let reg_raw = fs::read_to_string(arch_dir.join("file_registry.json")).unwrap();
        let reg_json: Value = serde_json::from_str(&reg_raw).unwrap();
        assert_eq!(reg_json["total_files"].as_u64().unwrap(), 12);

        for (rel, content) in &test_files {
            let norm = rel.replace('\\', "/");
            let file_entry = &reg_json["files"][&norm];
            assert!(file_entry.is_object(), "File entry must exist: {}", norm);
            let expected_sha = {
                let mut h = Sha256::new();
                h.update(content.as_bytes());
                hex::encode(h.finalize())
            };
            assert_eq!(file_entry["sha256"].as_str().unwrap(), expected_sha);
        }

        // Verify folder registry has all created folders
        let f_reg_raw = fs::read_to_string(arch_dir.join("folder_registry.json")).unwrap();
        let f_reg_json: Value = serde_json::from_str(&f_reg_raw).unwrap();
        assert!(f_reg_json["folders"]["loc_a"].is_object());
        assert!(f_reg_json["folders"]["loc_b/sub_1"].is_object());
        assert!(f_reg_json["folders"]["loc_c/alpha/beta/gamma"].is_object());
        assert!(f_reg_json["folders"]["loc_d"].is_object());

        // Verify project tree contains nodes
        let tree_raw = fs::read_to_string(arch_dir.join("project_tree.json")).unwrap();
        let tree_json: Value = serde_json::from_str(&tree_raw).unwrap();
        assert_eq!(tree_json["entry_type"], "directory");

        // -------------------------------------------------------------
        // STEP 2: EDIT all 12 files with new content
        // -------------------------------------------------------------
        std::thread::sleep(std::time::Duration::from_millis(50));

        let edited_files = vec![
            ("loc_a/file_1.txt", "Hello EDITED from loc_a file 1\nSecond line.\nThird line.\n"),
            ("loc_a/file_2.rs", "fn a2_edited() { println!(\"a2 modified\"); }\n"),
            ("loc_a/file_3.json", "{\"key\": \"value_a3_edited\", \"updated\": true}\n"),
            ("loc_b/sub_1/doc_1.md", "# Document 1 EDITED\nNew section added.\n"),
            ("loc_b/sub_1/doc_2.toml", "name = \"doc_2_edited\"\nversion = \"2.0\"\n"),
            ("loc_b/sub_2/doc_3.txt", "Third document in sub_2 -- EDITED!\n"),
            ("loc_c/alpha/beta/gamma/module.rs", "pub fn gamma_updated() -> i32 { 999 }\n"),
            ("loc_c/alpha/beta/gamma/config.yaml", "mode: production\ngamma: true\nupdated: true\n"),
            ("loc_c/alpha/shared.rs", "pub const SHARED: u64 = 200;\npub const NEW_CONST: i32 = 1;\n"),
            ("loc_d/service_1.rs", "pub struct S1Updated;\n"),
            ("loc_d/service_2.rs", "pub struct S2Updated;\n"),
            ("loc_d/schema.proto", "syntax = \"proto3\";\nmessage MsgUpdated { string id = 1; }\n"),
        ];

        for (rel, content) in &edited_files {
            let full = sandbox.join(rel);
            fs::write(&full, content).unwrap();
        }

        let changes_step2 = engine.live_watcher_tick().expect("live_watcher_tick edit");
        assert!(changes_step2 >= 12, "Expected at least 12 changes detected on edit");

        let reg_raw2 = fs::read_to_string(arch_dir.join("file_registry.json")).unwrap();
        let reg_json2: Value = serde_json::from_str(&reg_raw2).unwrap();
        for (rel, content) in &edited_files {
            let norm = rel.replace('\\', "/");
            let file_entry = &reg_json2["files"][&norm];
            assert!(file_entry.is_object(), "File must still exist: {}", norm);
            let expected_sha = {
                let mut h = Sha256::new();
                h.update(content.as_bytes());
                hex::encode(h.finalize())
            };
            assert_eq!(file_entry["sha256"].as_str().unwrap(), expected_sha, "SHA must be updated for {}", norm);
        }

        // -------------------------------------------------------------
        // STEP 3: MOVE/RENAME files to new locations
        // -------------------------------------------------------------
        std::thread::sleep(std::time::Duration::from_millis(50));

        let moved_mappings = vec![
            ("loc_a/file_1.txt", "relocated/dest_1/moved_file_1.txt"),
            ("loc_a/file_2.rs", "relocated/dest_1/moved_file_2.rs"),
            ("loc_a/file_3.json", "relocated/dest_1/moved_file_3.json"),
            ("loc_b/sub_1/doc_1.md", "relocated/dest_2/moved_doc_1.md"),
            ("loc_b/sub_1/doc_2.toml", "relocated/dest_2/moved_doc_2.toml"),
            ("loc_b/sub_2/doc_3.txt", "relocated/dest_2/moved_doc_3.txt"),
            ("loc_c/alpha/beta/gamma/module.rs", "relocated/dest_3/deep/moved_mod.rs"),
            ("loc_c/alpha/beta/gamma/config.yaml", "relocated/dest_3/deep/moved_config.yaml"),
            ("loc_c/alpha/shared.rs", "relocated/dest_3/moved_shared.rs"),
            ("loc_d/service_1.rs", "relocated/dest_4/moved_s1.rs"),
            ("loc_d/service_2.rs", "relocated/dest_4/moved_s2.rs"),
            ("loc_d/schema.proto", "relocated/dest_4/moved_schema.proto"),
        ];

        for (from, to) in &moved_mappings {
            let from_path = sandbox.join(from);
            let to_path = sandbox.join(to);
            if let Some(parent) = to_path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::rename(&from_path, &to_path).unwrap();
        }

        // Remove now-empty old folders
        let _ = fs::remove_dir_all(sandbox.join("loc_a"));
        let _ = fs::remove_dir_all(sandbox.join("loc_b"));
        let _ = fs::remove_dir_all(sandbox.join("loc_c"));
        let _ = fs::remove_dir_all(sandbox.join("loc_d"));

        let changes_step3 = engine.live_watcher_tick().expect("live_watcher_tick move");
        assert!(changes_step3 > 0, "Expected changes detected on move");

        let reg_raw3 = fs::read_to_string(arch_dir.join("file_registry.json")).unwrap();
        let reg_json3: Value = serde_json::from_str(&reg_raw3).unwrap();
        assert_eq!(reg_json3["total_files"].as_u64().unwrap(), 12);

        // Verify old paths do NOT exist
        for (old_rel, _) in &moved_mappings {
            let norm_old = old_rel.replace('\\', "/");
            assert!(reg_json3["files"][&norm_old].is_null(), "Old path must be removed: {}", norm_old);
        }

        // Verify new paths DO exist
        for (_, new_rel) in &moved_mappings {
            let norm_new = new_rel.replace('\\', "/");
            assert!(reg_json3["files"][&norm_new].is_object(), "New path must exist: {}", norm_new);
        }

        // Verify folder registry has new folders and old folders are removed
        let f_reg_raw3 = fs::read_to_string(arch_dir.join("folder_registry.json")).unwrap();
        let f_reg_json3: Value = serde_json::from_str(&f_reg_raw3).unwrap();
        assert!(f_reg_json3["folders"]["loc_a"].is_null(), "loc_a must be removed");
        assert!(f_reg_json3["folders"]["loc_b/sub_1"].is_null(), "loc_b/sub_1 must be removed");
        assert!(f_reg_json3["folders"]["relocated/dest_1"].is_object(), "relocated/dest_1 must exist");
        assert!(f_reg_json3["folders"]["relocated/dest_3/deep"].is_object(), "relocated/dest_3/deep must exist");

        // -------------------------------------------------------------
        // STEP 4: DELETE all 12 files and folders
        // -------------------------------------------------------------
        std::thread::sleep(std::time::Duration::from_millis(50));
        fs::remove_dir_all(sandbox.join("relocated")).unwrap();

        let changes_step4 = engine.live_watcher_tick().expect("live_watcher_tick delete");
        assert!(changes_step4 > 0, "Expected changes detected on delete");

        let reg_raw4 = fs::read_to_string(arch_dir.join("file_registry.json")).unwrap();
        let reg_json4: Value = serde_json::from_str(&reg_raw4).unwrap();
        assert_eq!(reg_json4["total_files"].as_u64().unwrap(), 0, "All files must be deleted");

        let f_reg_raw4 = fs::read_to_string(arch_dir.join("folder_registry.json")).unwrap();
        let f_reg_json4: Value = serde_json::from_str(&f_reg_raw4).unwrap();
        assert!(f_reg_json4["folders"]["relocated"].is_null(), "relocated folder must be removed");
        assert!(f_reg_json4["folders"]["loc_a"].is_null(), "loc_a must be removed");
        assert_eq!(f_reg_json4["total_folders"].as_u64().unwrap(), 1, "Only ARCHITECTURE folder remains");

        // State must be SYNCED
        let state_raw4 = fs::read_to_string(arch_dir.join("architecture_state.json")).unwrap();
        let state4: ArchitectureState = serde_json::from_str(&state_raw4).unwrap();
        assert_eq!(state4.status, "SYNCED");
        assert_eq!(state4.total_files, 0);
        assert_eq!(state4.total_folders, 1);

        // Cleanup sandbox
        let _ = fs::remove_dir_all(&sandbox);
    }
}
