//! Static code analyzer, syntax health verification, and vulnerability scanner.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntaxHealth {
    pub is_balanced: bool,
    pub unclosed_delimiters: Vec<String>,
    pub total_lines: usize,
    pub code_lines: usize,
    pub comment_lines: usize,
    pub blank_lines: usize,
    pub function_count: usize,
    pub cyclomatic_complexity_est: usize,
    pub suspicious_patterns: Vec<SecurityIssue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityIssue {
    pub severity: String,
    pub line: usize,
    pub pattern: String,
    pub description: String,
}

pub struct StaticAnalyzer;

impl StaticAnalyzer {
    /// Perform static analysis across source code.
    pub fn analyze(source: &str, lang: &str) -> SyntaxHealth {
        let lines: Vec<&str> = source.lines().collect();
        let total_lines = lines.len();
        let mut blank_lines = 0;
        let mut comment_lines = 0;
        let mut code_lines = 0;
        let mut function_count = 0;
        let mut cyclomatic = 1; // Base complexity
        let mut issues = Vec::new();

        let is_rust = lang.eq_ignore_ascii_case("rust") || lang.eq_ignore_ascii_case("rs");
        let is_py = lang.eq_ignore_ascii_case("python") || lang.eq_ignore_ascii_case("py");

        for (idx, line) in lines.iter().enumerate() {
            let line_num = idx + 1;
            let trimmed = line.trim();

            if trimmed.is_empty() {
                blank_lines += 1;
                continue;
            }

            if trimmed.starts_with("//") || trimmed.starts_with('#') || trimmed.starts_with("/*") {
                comment_lines += 1;
                continue;
            }

            code_lines += 1;

            // Complexity indicators
            for kw in &["if ", "while ", "for ", "match ", "&&", "||", "case "] {
                if trimmed.contains(kw) {
                    cyclomatic += 1;
                }
            }

            // Function detection
            if (is_rust && (trimmed.starts_with("fn ") || trimmed.contains(" fn ")))
                || (is_py && trimmed.starts_with("def "))
                || trimmed.contains("function ")
                || (trimmed.contains('(')
                    && trimmed.ends_with('{')
                    && !trimmed.starts_with("if")
                    && !trimmed.starts_with("for"))
            {
                function_count += 1;
            }

            // Security pattern audit
            if is_rust && trimmed.contains("unsafe {") {
                issues.push(SecurityIssue {
                    severity: "WARNING".into(),
                    line: line_num,
                    pattern: "unsafe {".into(),
                    description: "Explicit unsafe block in Rust bypasses memory safety guarantees."
                        .into(),
                });
            }

            if is_py && (trimmed.contains("eval(") || trimmed.contains("exec(")) {
                issues.push(SecurityIssue {
                    severity: "HIGH".into(),
                    line: line_num,
                    pattern: "eval/exec".into(),
                    description: "Dynamic code execution via eval/exec introduces remote code execution vulnerability.".into(),
                });
            }

            if trimmed.contains("rm -rf /") || trimmed.contains("Remove-Item -Recurse C:\\") {
                issues.push(SecurityIssue {
                    severity: "CRITICAL".into(),
                    line: line_num,
                    pattern: "root directory deletion".into(),
                    description: "Destructive root filesystem deletion pattern detected.".into(),
                });
            }

            if trimmed.to_uppercase().contains("DROP TABLE")
                || trimmed.to_uppercase().contains("DROP DATABASE")
            {
                issues.push(SecurityIssue {
                    severity: "HIGH".into(),
                    line: line_num,
                    pattern: "DROP TABLE/DATABASE".into(),
                    description: "Destructive DDL query detected.".into(),
                });
            }
        }

        // Bracket balance check
        let (is_balanced, unclosed) = Self::check_delimiter_balance(source);

        SyntaxHealth {
            is_balanced,
            unclosed_delimiters: unclosed,
            total_lines,
            code_lines,
            comment_lines,
            blank_lines,
            function_count,
            cyclomatic_complexity_est: cyclomatic,
            suspicious_patterns: issues,
        }
    }

    /// Check matching of parentheses (), braces {}, and brackets [].
    pub fn check_delimiter_balance(source: &str) -> (bool, Vec<String>) {
        let mut stack = Vec::new();
        let mut unclosed = Vec::new();
        let mut line_num = 1;
        let mut col_num = 0;

        let mut in_string = false;
        let mut string_quote = ' ';
        let mut in_line_comment = false;

        let chars: Vec<char> = source.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            let ch = chars[i];
            col_num += 1;

            if ch == '\n' {
                line_num += 1;
                col_num = 0;
                in_line_comment = false;
                i += 1;
                continue;
            }

            if in_line_comment {
                i += 1;
                continue;
            }

            // Handle string literals
            if !in_string && (ch == '"' || ch == '\'') {
                in_string = true;
                string_quote = ch;
                i += 1;
                continue;
            } else if in_string && ch == string_quote && (i == 0 || chars[i - 1] != '\\') {
                in_string = false;
                i += 1;
                continue;
            }

            if in_string {
                i += 1;
                continue;
            }

            // Comment start
            if ch == '/' && i + 1 < len && chars[i + 1] == '/' {
                in_line_comment = true;
                i += 2;
                continue;
            }

            match ch {
                '(' | '{' | '[' => stack.push((ch, line_num, col_num)),
                ')' => {
                    if let Some(('(', _, _)) = stack.last() {
                        stack.pop();
                    } else {
                        unclosed.push(format!("Unmatched ')' at line {}:{}", line_num, col_num));
                    }
                }
                '}' => {
                    if let Some(('{', _, _)) = stack.last() {
                        stack.pop();
                    } else {
                        unclosed.push(format!("Unmatched '}}' at line {}:{}", line_num, col_num));
                    }
                }
                ']' => {
                    if let Some(('[', _, _)) = stack.last() {
                        stack.pop();
                    } else {
                        unclosed.push(format!("Unmatched ']' at line {}:{}", line_num, col_num));
                    }
                }
                _ => {}
            }
            i += 1;
        }

        for (ch, line, col) in stack {
            unclosed.push(format!("Unclosed '{}' opened at line {}:{}", ch, line, col));
        }

        let balanced = unclosed.is_empty();
        (balanced, unclosed)
    }
}
