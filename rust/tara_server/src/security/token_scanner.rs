//! Credentials and token scanner for newly written skill code.
//!
//! Scans code for hardcoded secrets, private keys, and credential patterns.

use regex::Regex;

pub struct TokenScanner {
    patterns: Vec<(&'static str, Regex)>,
}

impl Default for TokenScanner {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenScanner {
    pub fn new() -> Self {
        let regex_defs = [
            ("OPENAI_KEY", Regex::new(r"sk-[a-zA-Z0-9]{32,}").unwrap()),
            ("AWS_ACCESS_KEY", Regex::new(r"AKIA[0-9A-Z]{16}").unwrap()),
            ("GITHUB_TOKEN", Regex::new(r"gh[pous]_[a-zA-Z0-9]{36}").unwrap()),
            ("PRIVATE_KEY_BLOCK", Regex::new(r"-----BEGIN (RSA|EC|DSA|OPENSSH|PGP)? PRIVATE KEY-----").unwrap()),
            ("GENERIC_SECRET", Regex::new(r#"(?i)(api[_-]?key|secret|password|auth[_-]?token)\s*=\s*['"][a-zA-Z0-9_\-.~+=/]{16,}['"]"#).unwrap()),
        ];

        Self {
            patterns: regex_defs.to_vec(),
        }
    }

    /// Scans text/code for secrets. Returns list of detected secret types.
    pub fn scan(&self, content: &str) -> Vec<String> {
        let mut detected = Vec::new();
        for (name, re) in &self.patterns {
            if re.is_match(content) {
                detected.push(name.to_string());
            }
        }
        detected
    }

    pub fn assert_clean(&self, content: &str) -> Result<(), String> {
        let violations = self.scan(content);
        if !violations.is_empty() {
            Err(format!(
                "Credential scan rejected: detected hardcoded secret patterns: {:?}",
                violations
            ))
        } else {
            Ok(())
        }
    }
}
