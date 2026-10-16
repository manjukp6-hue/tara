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

            // Security pattern audit (spacing-agnostic unsafe block check)
            let compact: String = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
            if is_rust
                && (compact.contains("unsafe{")
                    || trimmed.starts_with("unsafe fn ")
                    || trimmed.contains(" unsafe fn ")
                    || trimmed.contains("unsafe {"))
            {
                issues.push(SecurityIssue {
                    severity: "WARNING".into(),
                    line: line_num,
                    pattern: "unsafe".into(),
                    description: "Explicit unsafe block or function in Rust bypasses memory safety guarantees."
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
    /// Properly handles string literals, escaped backslashes, block/line comments, and Rust lifetimes ('a).
    pub fn check_delimiter_balance(source: &str) -> (bool, Vec<String>) {
        let mut stack = Vec::new();
        let mut unclosed = Vec::new();
        let mut line_num = 1;
        let mut col_num = 0;

        let mut in_double_string = false;
        let mut in_line_comment = false;
        let mut in_block_comment = false;

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

            if in_block_comment {
                if ch == '*' && i + 1 < len && chars[i + 1] == '/' {
                    in_block_comment = false;
                    i += 2;
                    col_num += 1;
                } else {
                    i += 1;
                }
                continue;
            }

            if in_double_string {
                if ch == '\\' {
                    // Skip escaped character (including \" and \\)
                    i += 2;
                    col_num += 1;
                    continue;
                } else if ch == '"' {
                    in_double_string = false;
                }
                i += 1;
                continue;
            }

            // Line comments (//) and Block comments (/* ... */)
            if ch == '/' && i + 1 < len {
                if chars[i + 1] == '/' {
                    in_line_comment = true;
                    i += 2;
                    col_num += 1;
                    continue;
                } else if chars[i + 1] == '*' {
                    in_block_comment = true;
                    i += 2;
                    col_num += 1;
                    continue;
                }
            }

            // Double-quoted string literal start
            if ch == '"' {
                in_double_string = true;
                i += 1;
                continue;
            }

            // Single-quoted character literal (e.g. 'x', '\n', '\'') vs Rust lifetime (e.g. 'a, 'static)
            if ch == '\'' {
                if i + 2 < len && chars[i + 1] != '\\' && chars[i + 2] == '\'' {
                    // Standard char literal like '(' or '}'
                    i += 3;
                    col_num += 2;
                    continue;
                } else if i + 3 < len && chars[i + 1] == '\\' && chars[i + 3] == '\'' {
                    // Escaped char literal like '\'' or '\n'
                    i += 4;
                    col_num += 3;
                    continue;
                } else {
                    // Rust lifetime ('a, 'static) or raw apostrophe — do NOT enter string mode
                    i += 1;
                    continue;
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lifetime_and_escaped_backslash_delimiter_balance() {
        let lifetime_code = "pub fn parse<'a>(input: &'a str) -> &'a str { input }";
        let (balanced, unclosed) = StaticAnalyzer::check_delimiter_balance(lifetime_code);
        assert!(
            balanced,
            "Expected lifetime code to be balanced, got unclosed: {:?}",
            unclosed
        );

        let escaped_path_code = "let path = \"C:\\\\\"; fn main() {}";
        let (balanced_path, unclosed_path) =
            StaticAnalyzer::check_delimiter_balance(escaped_path_code);
        assert!(
            balanced_path,
            "Expected escaped path code to be balanced, got unclosed: {:?}",
            unclosed_path
        );

        let block_comment_code = "/* { ( [ */ fn test() -> i32 { 42 } /* } ) ] */";
        let (balanced_comment, unclosed_comment) =
            StaticAnalyzer::check_delimiter_balance(block_comment_code);
        assert!(
            balanced_comment,
            "Expected block comment code to be balanced, got unclosed: {:?}",
            unclosed_comment
        );
    }

    #[test]
    fn test_unsafe_detection_without_space() {
        let code = "fn main() { unsafe{ std::ptr::null::<i32>().read(); } }";
        let health = StaticAnalyzer::analyze(code, "rust");
        assert!(!health.suspicious_patterns.is_empty());
        assert_eq!(health.suspicious_patterns[0].severity, "WARNING");
    }
}
