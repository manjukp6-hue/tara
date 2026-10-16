//! Native Credential, Secret, and Token Leak Scanner.
//!
//! Rule compliance:
//! - Pure native Rust, zero regex crate dependencies.
//! - Detects hardcoded secrets, private keys, API tokens, and credential patterns.
//! - Scans agent inputs, generated code, and prompt responses to prevent secret exfiltration.

use serde::{Deserialize, Serialize};

/// Detailed secret finding identifying the pattern and position.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecretFinding {
    pub pattern_name: String,
    pub line_number: usize,
    pub snippet_preview: String,
}

/// Fast native credential and token scanner.
pub struct SecretLeakScanner;

impl Default for SecretLeakScanner {
    fn default() -> Self {
        Self::new()
    }
}

impl SecretLeakScanner {
    pub fn new() -> Self {
        Self
    }

    /// Scans text/code and returns detailed findings with line numbers and sanitized previews.
    pub fn scan(&self, content: &str) -> Vec<SecretFinding> {
        let mut findings = Vec::new();

        for (line_idx, line) in content.lines().enumerate() {
            let line_num = line_idx + 1;
            let trimmed = line.trim();

            // 1. Private Key Blocks
            if trimmed.contains("-----BEGIN") && trimmed.contains("PRIVATE KEY-----") {
                findings.push(SecretFinding {
                    pattern_name: "PRIVATE_KEY_BLOCK".to_string(),
                    line_number: line_num,
                    snippet_preview: "-----BEGIN...PRIVATE KEY-----".to_string(),
                });
                continue;
            }

            // 2. AWS Access Key ID (AKIA + 16 chars)
            if let Some(pos) = line.find("AKIA") {
                let candidate = &line[pos..];
                if candidate.len() >= 20 {
                    let key_part = &candidate[..20];
                    if key_part.chars().all(|c| c.is_ascii_alphanumeric()) {
                        findings.push(SecretFinding {
                            pattern_name: "AWS_ACCESS_KEY".to_string(),
                            line_number: line_num,
                            snippet_preview: format!("{}...", &key_part[..8]),
                        });
                        continue;
                    }
                }
            }

            // 3. GitHub Tokens (ghp_, gho_, ghu_, ghs_, ghr_ + 36 chars)
            let gh_prefixes = ["ghp_", "gho_", "ghu_", "ghs_", "ghr_"];
            let mut gh_found = false;
            for pfx in gh_prefixes {
                if let Some(pos) = line.find(pfx) {
                    let candidate = &line[pos..];
                    if candidate.len() >= 40 {
                        let token_part = &candidate[..40];
                        if token_part[4..].chars().all(|c| c.is_ascii_alphanumeric()) {
                            findings.push(SecretFinding {
                                pattern_name: "GITHUB_TOKEN".to_string(),
                                line_number: line_num,
                                snippet_preview: format!("{}...", &token_part[..8]),
                            });
                            gh_found = true;
                            break;
                        }
                    }
                }
            }
            if gh_found {
                continue;
            }

            // 4. OpenAI / General LLM API Key (sk- + 32 chars)
            if let Some(pos) = line.find("sk-") {
                let candidate = &line[pos..];
                if candidate.len() >= 35 {
                    let key_part = &candidate[..35];
                    if key_part[3..].chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
                        findings.push(SecretFinding {
                            pattern_name: "OPENAI_API_KEY".to_string(),
                            line_number: line_num,
                            snippet_preview: "sk-...".to_string(),
                        });
                        continue;
                    }
                }
            }

            // 5. Generic Secret Keyword assignments (api_key, secret, password, auth_token)
            let lower = line.to_lowercase();
            let secret_keywords = [
                "api_key", "apikey", "secret", "password", "auth_token",
                "access_token", "passphrase", "credential", "master_key",
                "signing_key", "private_key", "encryption_key", "jwt_secret",
                "hmac_key",
            ];
            for kw in secret_keywords {
                if let Some(kw_pos) = lower.find(kw) {
                    let remainder = &line[kw_pos + kw.len()..];
                    // Look for delimiter = or :
                    if let Some(delim_pos) = remainder.find(['=', ':']) {
                        let val_part = remainder[delim_pos + 1..].trim().trim_matches(['"', '\'', '`', ';']);
                        if val_part.len() >= 16 && val_part.chars().all(|c| c.is_ascii_graphic()) && !val_part.contains(' ') {
                            findings.push(SecretFinding {
                                pattern_name: "GENERIC_SECRET_ASSIGNMENT".to_string(),
                                line_number: line_num,
                                snippet_preview: format!("{}: [REDACTED_SECRET]", kw),
                            });
                            break;
                        }
                    }
                }
            }
        }

        findings
    }

    /// Asserts content contains zero leaked credentials or secret tokens.
    pub fn assert_clean(&self, content: &str) -> Result<(), String> {
        let findings = self.scan(content);
        if findings.is_empty() {
            Ok(())
        } else {
            let details: Vec<String> = findings
                .iter()
                .map(|f| format!("Line {}: {}", f.line_number, f.pattern_name))
                .collect();
            Err(format!(
                "SecretLeakDetected: Content contains hardcoded credentials: [{}]",
                details.join(", ")
            ))
        }
    }

    pub fn scan_details(&self, content: &str) -> Vec<SecretFinding> {
        self.scan(content)
    }
}

pub type TokenScanner = SecretLeakScanner;
