//! Programming language registry and capability profiles for ProgrammingEngine.

use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanguageCapability {
    pub name: &'static str,
    pub extensions: &'static [&'static str],
    pub paradigm: &'static str,
    pub typing: &'static str,
    pub native_compiler_available: bool,
    pub static_analysis_supported: bool,
    pub execution_sandboxing: bool,
}

static REGISTRY: OnceLock<HashMap<&'static str, LanguageCapability>> = OnceLock::new();

fn init_languages() -> HashMap<&'static str, LanguageCapability> {
    let mut map = HashMap::new();
    map.insert(
        "rust",
        LanguageCapability {
            name: "Rust",
            extensions: &["rs"],
            paradigm: "Multi-paradigm (imperative, functional, concurrent)",
            typing: "Static, strong, affine/linear ownership",
            native_compiler_available: true, // cargo check, cargo test, rustc
            static_analysis_supported: true,
            execution_sandboxing: true,
        },
    );
    map.insert(
        "python",
        LanguageCapability {
            name: "Python",
            extensions: &["py"],
            paradigm: "Multi-paradigm (object-oriented, imperative, functional)",
            typing: "Dynamic, strong",
            native_compiler_available: false,
            static_analysis_supported: true,
            execution_sandboxing: true,
        },
    );
    map.insert(
        "c",
        LanguageCapability {
            name: "C",
            extensions: &["c", "h"],
            paradigm: "Procedural, imperative",
            typing: "Static, weak",
            native_compiler_available: true,
            static_analysis_supported: true,
            execution_sandboxing: true,
        },
    );
    map.insert(
        "cpp",
        LanguageCapability {
            name: "C++",
            extensions: &["cpp", "hpp", "cc", "cxx"],
            paradigm: "Multi-paradigm (procedural, OO, generic, functional)",
            typing: "Static, strong",
            native_compiler_available: true,
            static_analysis_supported: true,
            execution_sandboxing: true,
        },
    );
    map.insert(
        "java",
        LanguageCapability {
            name: "Java",
            extensions: &["java"],
            paradigm: "Object-oriented, class-based",
            typing: "Static, strong",
            native_compiler_available: false,
            static_analysis_supported: true,
            execution_sandboxing: true,
        },
    );
    map.insert(
        "javascript",
        LanguageCapability {
            name: "JavaScript",
            extensions: &["js", "mjs"],
            paradigm: "Event-driven, functional, prototype-based",
            typing: "Dynamic, weak",
            native_compiler_available: false,
            static_analysis_supported: true,
            execution_sandboxing: true,
        },
    );
    map.insert(
        "typescript",
        LanguageCapability {
            name: "TypeScript",
            extensions: &["ts", "tsx"],
            paradigm: "Typed superset of JavaScript",
            typing: "Static, structural",
            native_compiler_available: false,
            static_analysis_supported: true,
            execution_sandboxing: true,
        },
    );
    map.insert(
        "go",
        LanguageCapability {
            name: "Go",
            extensions: &["go"],
            paradigm: "Concurrent, imperative",
            typing: "Static, strong",
            native_compiler_available: false,
            static_analysis_supported: true,
            execution_sandboxing: true,
        },
    );
    map.insert(
        "sql",
        LanguageCapability {
            name: "SQL",
            extensions: &["sql"],
            paradigm: "Declarative, relational query",
            typing: "Static/dynamic depending on dialect",
            native_compiler_available: false,
            static_analysis_supported: true,
            execution_sandboxing: true,
        },
    );
    map.insert(
        "bash",
        LanguageCapability {
            name: "Bash",
            extensions: &["sh", "bash"],
            paradigm: "Shell scripting, command language",
            typing: "Dynamic, string-based",
            native_compiler_available: false,
            static_analysis_supported: true,
            execution_sandboxing: true,
        },
    );
    map.insert(
        "powershell",
        LanguageCapability {
            name: "PowerShell",
            extensions: &["ps1", "psm1"],
            paradigm: "Object-oriented command shell and scripting",
            typing: "Dynamic/optional static, strong",
            native_compiler_available: false,
            static_analysis_supported: true,
            execution_sandboxing: true,
        },
    );
    map
}

pub struct LanguageRegistry;

impl LanguageRegistry {
    pub fn get(lang: &str) -> Option<&'static LanguageCapability> {
        let reg = REGISTRY.get_or_init(init_languages);
        reg.get(lang.trim().to_lowercase().as_str())
    }

    pub fn detect_from_path(path: &str) -> Option<&'static LanguageCapability> {
        let p = std::path::Path::new(path);
        let ext = p.extension()?.to_str()?.to_lowercase();
        let reg = REGISTRY.get_or_init(init_languages);
        reg.values()
            .find(|&cap| cap.extensions.contains(&ext.as_str()))
            .map(|v| v as _)
    }
}
