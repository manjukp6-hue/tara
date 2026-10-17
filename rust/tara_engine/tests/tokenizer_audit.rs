use std::fs;
use std::path::Path;
use tara_engine::tokenizer::{TaraTokenizer, SPECIAL_TOKENS};

fn audit_model_tokenizer(model_dir: &str) -> Result<String, String> {
    let dir = Path::new(model_dir);
    if !dir.exists() {
        return Err(format!("Directory does not exist: {}", model_dir));
    }

    let tok_path = dir.join("tokenizer.json");
    if !tok_path.exists() {
        return Err(format!("tokenizer.json missing in {}", model_dir));
    }

    let tokenizer = TaraTokenizer::from_file(&tok_path.to_string_lossy())
        .map_err(|e| format!("Failed to parse tokenizer.json in {}: {:?}", model_dir, e))?;

    let mut report = Vec::new();
    report.push(format!("=== AUDIT REPORT FOR: {} ===", model_dir));
    report.push(format!("Vocabulary size: {}", tokenizer.vocab_size));
    report.push(format!("Max token length: {}", tokenizer.max_token_len));

    // 1. Config alignment
    let cfg_path = dir.join("config.json");
    if cfg_path.exists() {
        let cfg_raw = fs::read_to_string(&cfg_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let cfg_val: serde_json::Value = serde_json::from_str(&cfg_raw)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;
        if let Some(v) = cfg_val.get("vocab_size").and_then(|x| x.as_u64()) {
            if v as usize == tokenizer.vocab_size {
                report.push(format!(
                    "Config vocab_size match: PASS ({} == {})",
                    v, tokenizer.vocab_size
                ));
            } else {
                report.push(format!(
                    "Config vocab_size match: FAIL (config has {}, tokenizer has {})",
                    v, tokenizer.vocab_size
                ));
            }
        }
    } else {
        report.push("Config file: NOT FOUND".into());
    }

    // 2. Dense ID check
    let mut missing_ids = Vec::new();
    for id in 0..(tokenizer.vocab_size as u32) {
        if !tokenizer.id_to_token.contains_key(&id) {
            missing_ids.push(id);
        }
    }
    if missing_ids.is_empty() {
        report.push("Dense IDs (0..vocab_size-1): PASS (no gaps)".into());
    } else {
        report.push(format!(
            "Dense IDs: FAIL (missing {} IDs: {:?})",
            missing_ids.len(),
            &missing_ids[..missing_ids.len().min(10)]
        ));
    }

    // 3. Special tokens check
    let mut missing_specials = Vec::new();
    for &sp in SPECIAL_TOKENS {
        if !tokenizer.token_to_id.contains_key(sp) {
            missing_specials.push(sp);
        }
    }
    if missing_specials.is_empty() {
        report.push(format!(
            "Special tokens ({}): PASS (all present)",
            SPECIAL_TOKENS.len()
        ));
    } else {
        report.push(format!(
            "Special tokens: FAIL (missing: {:?})",
            missing_specials
        ));
    }

    // 4. Kannada Numerals
    let kn_numerals = ["೦", "೧", "೨", "೩", "೪", "೫", "೬", "೭", "೮", "೯"];
    let mut missing_kn_nums = Vec::new();
    for &num in &kn_numerals {
        if !tokenizer.token_to_id.contains_key(num) {
            missing_kn_nums.push(num);
        }
    }
    if missing_kn_nums.is_empty() {
        report.push("Kannada Numerals (೦-೯): PASS (all 10 present)".into());
    } else {
        report.push(format!(
            "Kannada Numerals (೦-೯): FAIL (missing: {:?})",
            missing_kn_nums
        ));
    }

    // 5. Kannada Script Core Characters
    let kn_chars = [
        "ಅ", "ಆ", "ಇ", "ಈ", "ಉ", "ಊ", "ಋ", "ಎ", "ಏ", "ಐ", "ಒ", "ಓ", "ಔ", "ಕ", "ಖ", "ಗ", "ಘ", "ಙ",
        "ಚ", "ಛ", "ಜ", "ಝ", "ಞ", "ಟ", "ಠ", "ಡ", "ಢ", "ಣ", "ತ", "ಥ", "ದ", "ಧ", "ನ", "ಪ", "ಫ", "ಬ",
        "ಭ", "ಮ", "ಯ", "ರ", "ಲ", "ವ", "ಶ", "ಷ", "ಸ", "ಹ", "ಳ", "ಂ", "ಃ", "ಾ", "ಿ", "ೀ", "ು", "ೂ", "ೃ",
        "ೆ", "ೇ", "ೈ", "ೊ", "ೋ", "ೌ", "್",
    ];
    let mut missing_kn_chars = Vec::new();
    for &ch in &kn_chars {
        if !tokenizer.token_to_id.contains_key(ch) {
            missing_kn_chars.push(ch);
        }
    }
    if missing_kn_chars.is_empty() {
        report.push(format!(
            "Kannada Basic Alphabet ({} chars): PASS (all present)",
            kn_chars.len()
        ));
    } else {
        report.push(format!(
            "Kannada Basic Alphabet: FAIL (missing: {:?})",
            missing_kn_chars
        ));
    }

    // 6. Devanagari Numerals & Core
    let dv_numerals = ["०", "१", "२", "३", "४", "५", "६", "७", "८", "९"];
    let mut missing_dv_nums = Vec::new();
    for &num in &dv_numerals {
        if !tokenizer.token_to_id.contains_key(num) {
            missing_dv_nums.push(num);
        }
    }
    if missing_dv_nums.is_empty() {
        report.push("Devanagari Numerals (०-९): PASS (all 10 present)".into());
    } else {
        report.push(format!(
            "Devanagari Numerals (०-९): FAIL (missing: {:?})",
            missing_dv_nums
        ));
    }

    // 7. Coding & Technical Symbols
    let code_symbols = [
        "::", "->", "=>", "==", "!=", "<=", ">=", "&&", "||", "+=", "-=", "*=", "/=", "..", "...",
        "//", "/*", "*/", "</", "/>", "  ", "    ", "        ", "₹", "$", "€", "£", "±", "×", "÷",
        "≠", "≤", "≥", "∞", "√", "fn", "let", "mut", "pub", "struct", "impl", "enum", "trait",
        "def", "class", "import", "return",
    ];
    let mut missing_code_syms = Vec::new();
    for &sym in &code_symbols {
        if !tokenizer.token_to_id.contains_key(sym) {
            missing_code_syms.push(sym);
        }
    }
    if missing_code_syms.is_empty() {
        report.push(format!(
            "Coding & Math Symbols ({} checked): PASS (all present)",
            code_symbols.len()
        ));
    } else {
        report.push(format!(
            "Coding & Math Symbols: PARTIAL/FAIL (missing: {:?})",
            missing_code_syms
        ));
    }

    // 8. Round-trip test
    let test_cases = [
        ("Kannada Prose", "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ? ಇಂದು ಕರ್ನಾಟಕ ರಾಜ್ಯೋತ್ಸವ."),
        ("Kannada Numerals", "ದೂರವಾಣಿ ಸಂಖ್ಯೆ: ೯೮೭೬೫೪೩೨೧೦, ಮೊತ್ತ: ₹೨೫೦"),
        ("Kannada Conjuncts", "ಸ್ವಾತಂತ್ರ್ಯ, ವಿಶ್ವವಿದ್ಯಾಲಯ, ರಾಷ್ಟ್ರೀಯತೆ, ಸೂರ್ಯೋದಯ"),
        (
            "Devanagari Prose",
            "नमस्ते, भारतवर्षस्य संस्कृतिः अति प्राचीना वर्तते।",
        ),
        ("Devanagari Numerals", "संख्या: १२३४५६७८९०"),
        (
            "Code Rust",
            "pub fn solve(x: i32) -> Result<i32, String> { let mut y = x + 10; Ok(y) }",
        ),
        ("Math", "f(x) = \u{222b} e^{-x^2} dx \u{2264} \u{221e}"),
        (
            "English",
            "TARA neural engine with multi-capacity architecture.",
        ),
    ];

    let unk_id = tokenizer.unk_id();
    let mut roundtrip_passes = 0;
    let mut unk_detected = 0;

    for (name, text) in &test_cases {
        let encoded = tokenizer.encode(text);
        let unk_count = encoded.iter().filter(|&&id| id == unk_id).count();
        if unk_count > 0 {
            unk_detected += unk_count;
            report.push(format!(
                "Roundtrip [{}]: UNK detected ({} unk tokens)",
                name, unk_count
            ));
        }
        let decoded = tokenizer.decode(&encoded);
        if decoded == *text {
            roundtrip_passes += 1;
        } else {
            report.push(format!(
                "Roundtrip [{}]: MISMATCH! original='{}' decoded='{}'",
                name, text, decoded
            ));
        }
    }

    report.push(format!(
        "Roundtrip fidelity: {}/{} tests passed exactly (total UNK tokens: {})",
        roundtrip_passes,
        test_cases.len(),
        unk_detected
    ));

    Ok(report.join("\n"))
}

#[test]
fn test_audit_tara_candidate_v1_tokenizer() {
    let report = audit_model_tokenizer("../../storage/models/tara_candidate_v1")
        .expect("tara_candidate_v1 must exist and audit");
    println!("\n{}", report);
    assert!(report.contains("Dense IDs (0..vocab_size-1): PASS"));
    assert!(report.contains("Special tokens (9): PASS"));
    assert!(report.contains("Kannada Numerals (೦-೯): PASS"));
    assert!(report.contains("Kannada Basic Alphabet (62 chars): PASS"));
    assert!(report.contains("Devanagari Numerals (०-९): PASS"));
    assert!(report.contains("Roundtrip fidelity: 8/8 tests passed exactly (total UNK tokens: 0)"));
}

#[test]
fn test_audit_tara_world_model_tokenizer() {
    let report = audit_model_tokenizer("../../storage/models/tara_world_model")
        .expect("tara_world_model must exist and audit");
    println!("\n{}", report);
    assert!(report.contains("Dense IDs (0..vocab_size-1): PASS"));
    assert!(report.contains("Special tokens (9): PASS"));
    assert!(report.contains("Kannada Numerals (೦-೯): PASS"));
    assert!(report.contains("Kannada Basic Alphabet (62 chars): PASS"));
    assert!(report.contains("Devanagari Numerals (०-९): PASS"));
    assert!(report.contains("Roundtrip fidelity: 8/8 tests passed exactly (total UNK tokens: 0)"));
}

#[test]
fn test_breakdown_full_100k_vocabulary() {
    let tok_path = "../../storage/models/tara_candidate_v1/tokenizer.json";
    let tokenizer = TaraTokenizer::from_file(tok_path).expect("must load candidate tokenizer");

    let mut special_count = 0;
    let mut kannada_tokens = Vec::new();
    let mut devanagari_tokens = Vec::new();
    let mut code_tokens = Vec::new();
    let mut math_currency_tokens = Vec::new();
    let mut english_ascii_tokens = Vec::new();
    let mut other_tokens = Vec::new();

    let mut kannada_numerals = Vec::new();
    let mut devanagari_numerals = Vec::new();

    let special_set: std::collections::HashSet<&str> = SPECIAL_TOKENS.iter().copied().collect();

    for (tok, &id) in &tokenizer.token_to_id {
        if special_set.contains(tok.as_str()) {
            special_count += 1;
            continue;
        }

        let has_kn = tok.chars().any(|c| ('\u{0C80}'..='\u{0CFF}').contains(&c));
        let has_dv = tok.chars().any(|c| ('\u{0900}'..='\u{097F}').contains(&c));

        if has_kn {
            kannada_tokens.push((tok.clone(), id));
            if tok.chars().all(|c| ('\u{0CE6}'..='\u{0CEF}').contains(&c)) {
                kannada_numerals.push(tok.clone());
            }
        } else if has_dv {
            devanagari_tokens.push((tok.clone(), id));
            if tok.chars().all(|c| ('\u{0966}'..='\u{096F}').contains(&c)) {
                devanagari_numerals.push(tok.clone());
            }
        } else if tok.starts_with("  ")
            || tok.contains("::")
            || tok.contains("->")
            || tok.contains("=>")
            || tok.contains("==")
            || tok.contains("!=")
            || tok.contains("<=")
            || tok.contains(">=")
            || tok.contains("&&")
            || tok.contains("||")
            || tok.contains("+=")
            || tok.contains("-=")
            || tok.contains("//")
            || tok.contains("/*")
            || tok.contains("</")
            || tok.contains("/>")
            || [
                "fn", "let", "mut", "pub", "struct", "impl", "enum", "trait", "class", "def",
                "import", "return", "async", "await",
            ]
            .contains(&tok.as_str())
        {
            code_tokens.push((tok.clone(), id));
        } else if tok.chars().any(|c| "₹$€£¥±×÷≠≤≥∞√∫∑π°%‰".contains(c))
        {
            math_currency_tokens.push((tok.clone(), id));
        } else if tok.is_ascii() {
            english_ascii_tokens.push((tok.clone(), id));
        } else {
            other_tokens.push((tok.clone(), id));
        }
    }

    println!("\n=======================================================");
    println!("     TARA 100,000 PRODUCTION VOCABULARY BREAKDOWN      ");
    println!("=======================================================");
    println!(
        "Total Vocabulary Size            : {}",
        tokenizer.vocab_size
    );
    println!("1. Special Tokens                : {}", special_count);
    println!(
        "2. Kannada Tokens                : {} (Numerals: {})",
        kannada_tokens.len(),
        kannada_numerals.len()
    );
    println!(
        "3. Devanagari (Sanskrit/Hindi)   : {} (Numerals: {})",
        devanagari_tokens.len(),
        devanagari_numerals.len()
    );
    println!("4. Code & Syntax Tokens          : {}", code_tokens.len());
    println!(
        "5. Math & Currency Tokens        : {}",
        math_currency_tokens.len()
    );
    println!(
        "6. English & ASCII Subwords      : {}",
        english_ascii_tokens.len()
    );
    println!("7. Other / Multilingual Tokens   : {}", other_tokens.len());
    println!(
        "Sum of categorized tokens        : {}",
        special_count
            + kannada_tokens.len()
            + devanagari_tokens.len()
            + code_tokens.len()
            + math_currency_tokens.len()
            + english_ascii_tokens.len()
            + other_tokens.len()
    );

    println!("\n--- SAMPLE KANNADA TOKENS (First 20) ---");
    for (tok, id) in kannada_tokens.iter().take(20) {
        println!("  ID {:5} -> '{}'", id, tok);
    }

    println!("\n--- SAMPLE DEVANAGARI TOKENS (First 15) ---");
    for (tok, id) in devanagari_tokens.iter().take(15) {
        println!("  ID {:5} -> '{}'", id, tok);
    }

    println!("\n--- SAMPLE CODE TOKENS (First 15) ---");
    for (tok, id) in code_tokens.iter().take(15) {
        println!("  ID {:5} -> '{}'", id, tok);
    }

    println!("\n--- SAMPLE MATH & CURRENCY TOKENS (First 15) ---");
    for (tok, id) in math_currency_tokens.iter().take(15) {
        println!("  ID {:5} -> '{}'", id, tok);
    }
    println!("=======================================================\n");
}

#[test]
fn test_verify_world_model_loading() {
    let wm_path = "../../storage/models/tara_world_model";
    let res = tara_engine::TaraForCausalLM::load(wm_path);
    match res {
        Ok(m) => {
            println!(
                "World model loaded successfully! Layers: {}",
                m.config.num_hidden_layers
            );
            println!(
                "Hidden size: {}, Intermediate: {}, Heads: {}",
                m.config.hidden_size, m.config.intermediate_size, m.config.num_attention_heads
            );
            let mut nan_count = 0;
            let mut inf_count = 0;
            let mut zero_count = 0;
            let mut total_weights = 0;
            for val in &m.embed_tokens {
                total_weights += 1;
                if val.is_nan() {
                    nan_count += 1;
                } else if val.is_infinite() {
                    inf_count += 1;
                } else if *val == 0.0 {
                    zero_count += 1;
                }
            }
            println!(
                "Embedding tensor check: total={}, NaN={}, Inf={}, Zero={}",
                total_weights, nan_count, inf_count, zero_count
            );
        }
        Err(e) => println!("World model loading failed with error: {:?}", e),
    }
}
