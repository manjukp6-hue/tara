"""
Cross-Language Differential Fuzzing & Parity Verification Suite for TARA.

Compares encoding outputs between:
1. Python TaraNativeTokenizer (cloud_train.py)
2. Rust TaraTokenizer (rust/tara_engine/src/tokenizer.rs via target/debug/examples/tokenizer_fuzzer.exe)

Corpora Tested:
- English text (prose, contractions, CamelCase, snake_case, pangrams)
- Kannada text (vowels, consonants, matras, halanta, conjuncts, numerals ೦–೯, Kanglish)
- Code (Rust, Python, C++, operators, indentation, comments, escapes)
- Math formulas (LaTeX, Unicode math symbols, calculus, equations)
- Mixed unseen characters (Brahmi, Linear B, Hieroglyphs, Runic, Emojis, CJK, Arabic, Cyrillic)
- Boundary & Adversarial edge cases (empty strings, special tokens, literal byte strings, long reps)
- Randomized Unicode fuzz sequences
"""

import json
import os
import random
import subprocess
import sys
import time
from typing import List, Dict, Any, Tuple

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from cloud_train import TaraNativeTokenizer, SPECIAL_TOKENS


def build_adversarial_corpus() -> List[Tuple[str, str]]:
    """Builds a comprehensive, deterministic adversarial test corpus."""
    cases: List[Tuple[str, str]] = []

    # 1. English Text
    english_cases = [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "TARA neural cognition architecture operates with native low-rank adaptation.",
        "Don't worry, it's state-of-the-art; we've verified every single assumption!",
        "CamelCaseIdentifier, snake_case_variable, CONSTANT_NAME, kebab-case-item",
        "Punctuation stress: !@#$%^&*()_+~`-={}|[]\\:\";'<>?,./",
        "Multi-line\nprose with\r\ndifferent newline\rformats and   varying   whitespace.",
        "Hyphenated-compound-words and in-depth multi-layered abstractions.",
    ]
    for text in english_cases:
        cases.append(("English", text))

    # 2. Kannada Text & Numerals
    kannada_cases = [
        "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?",
        "ಕನ್ನಡ ನಾಡು ನುಡಿ ಜಲ ಸಂಸ್ಕೃತಿ ಅತ್ಯಂತ ಶ್ರೀಮಂತವಾಗಿದೆ.",
        "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು ಸಿಲಿಕಾನ್ ಸಿಟಿ ಎಂದೂ ಪ್ರಸಿದ್ಧವಾಗಿದೆ.",
        "೦", "೧", "೨", "೩", "೪", "೫", "೬", "೭", "೮", "೯",
        "೦೧೨೩೪೫೬೭೮೯",
        "ಕನ್ನಡ ಅಂಕೆಗಳು: ೧, ೨, ೩, ೪, ೫, ೬, ೭, ೮, ೯, ೦.",
        "ದಿನಾಂಕ: ೨೪-೧೦-೨೦೨೬, ಸಮಯ: ೨೦:೧೫:೦೦.",
        # Vowels & Consonants
        "ಅ ಆ ಇ ಈ ಉ ಊ ಋ ಎ ಏ ಐ ಒ ಓ ಔ ಅಂ ಅಃ",
        "ಕ ಖ ಗ ಘ ಙ ಚ ಛ ಜ ಝ ಞ ಟ ಠ ಡ ಢ ಣ ತ ಥ ದ ಧ ನ ಪ ಫ ಬ ಭ ಮ ಯ ರ ಲ ವ ಶ ಷ ಸ ಹ ಳ",
        # Matras & Halanta
        "ಕಾ ಕಿ ಕೀ ಕು ಕೂ ಕೃ ಕೆ ಕೇ ಕೈ ಕೊ ಕೋ ಕೌ ಕಂ ಕಃ ಕ್",
        # Conjuncts (ಒತ್ತಕ್ಷರಗಳು)
        "ಸ್ವಾತಂತ್ರ್ಯ, ರಾಷ್ಟ್ರ, ಪ್ರಜ್ಞೆ, ಕೃತ್ರಿಮ, ವಿಜ್ಞಾನ, ಸೃಷ್ಟಿ, ದೃಷ್ಟಿ, ಶಕ್ತಿ, ಮುಕ್ತಿ",
        "ಅರ್ಥ, ಕರ್ಮ, ಧರ್ಮ, ತತ್ವ, ಮಹತ್ವ, ನಾವೀನ್ಯತೆ, ಉತ್ಸಾಹ, ಸಹೋದರತೆ",
        "ಕ್ಕ ಗ್ಗ ಚ್ಚ ಜ್ಜ ಟ್ಟ ಡ್ಡ ಣ್ಣ ತ್ತ ದ್ದ ನ್ನ ಪ್ಪ ಬ್ಬ ಮ್ಮ ಯ್ಯ ರ್ಕ ಲ್ಲ ವ್ವ ಶ್ಶ ಷ್ಷ ಸ್ಸ ಳ್ಳ ಷ್ಟ್ರ ತ್ಸ್ಯ ಜ್ಞ",
        # Kanglish
        "tara neevu hegiddira oota aitha? nanu tumba chennagiddini.",
        "bengaluru weather tumba cool aagide ivathu.",
    ]
    for text in kannada_cases:
        cases.append(("Kannada", text))

    # 3. Code (Multi-language syntax & operators)
    code_cases = [
        "fn main() -> Result<(), Box<dyn std::error::Error>> { println!(\"Hello, TARA!\"); Ok(()) }",
        "pub struct TaraTokenizer<T: Send + Sync> { vocab: HashMap<String, u32>, phantom: PhantomData<T> }",
        "match token_id { 0..=8 => SpecialToken(token_id), 100000..=100255 => ByteFallback(token_id), _ => Subword(token_id) }",
        "def compute_gradients(model: nn.Module, loss: torch.Tensor, scaler: GradScaler) -> None:\n    scaler.scale(loss).backward()\n    scaler.step(optimizer)\n    scaler.update()",
        "#include <iostream>\n#include <vector>\nint main(int argc, char** argv) {\n    std::cout << \"Running test...\" << std::endl;\n    return 0;\n}",
        "if (a && b || !c ^ d) { x += 1; y -= 2; z *= 3; w /= 4; r %= 5; mask &= ~flags; }",
        "/* Multi-line\n   comment block */ // Single line comment\n\"String with \\\"escaped quotes\\\" and \\n escapes\"",
        "    indentation_level_1\n        indentation_level_2\n            indentation_level_3",
        "SELECT u.id, u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id HAVING COUNT(o.id) > 5;",
    ]
    for text in code_cases:
        cases.append(("Code", text))

    # 4. Math Formulas
    math_cases = [
        r"E = mc^2",
        r"\int_{-\infty}^{+\infty} e^{-x^2} dx = \sqrt{\pi}",
        r"\frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m} \nabla^2 \psi + V(\mathbf{r}) \psi",
        r"\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}",
        r"\lim_{x \to 0} \frac{\sin x}{x} = 1",
        r"\nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0 \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t}",
        "∑_{i=1}^n x_i · y_i ≤ (∑_{i=1}^n x_i^2)^{1/2} · (∑_{i=1}^n y_i^2)^{1/2}",
        "∀x ∈ ℝ, ∃y ∈ ℝ : x < y ∧ y - x < ε",
        "α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ, ο, π, ρ, σ, τ, υ, φ, χ, ψ, ω",
        "Α, Β, Γ, Δ, Ε, Ζ, Η, Θ, Ι, Κ, Λ, Μ, Ν, Ξ, Ο, Π, Ρ, Σ, Τ, Υ, Φ, Χ, Ψ, Ω",
        "x₀ + x₁ + x₂ = yⁿ · zᵐ⁺¹",
        "A ⊆ B ∧ B ⊆ C ⇒ A ⊆ C; A ∪ B = B ∪ A; A ∩ ∅ = ∅",
    ]
    for text in math_cases:
        cases.append(("Math", text))

    # 5. Mixed Unseen Characters (Byte Fallback Verification)
    unseen_cases = [
        # Historical scripts
        "𑀅𑀩𑀕 𑀥𑀦 (Brahmi script)",
        "𐀀𐀁𐀂 𐀃𐀄 (Linear B script)",
        "𓀀𓀁𓀂 𓀃𓀄 (Egyptian Hieroglyphs)",
        "ᚠᚢᚦᚨᚱᚲ (Elder Futhark Runes)",
        "ⰀⰁⰂⰃⰄⰅⰆ (Glagolitic)",
        "᚛ᚑᚌᚐᚋ᚜ (Ogham)",
        # Emojis & Modifiers
        "🚀 ✨ 🤖 🧠 🔥 💎 💡 ⚡ 🌟 🌈",
        "👨‍👩‍👧‍👦 (Family ZWJ sequence)",
        "👩‍💻 (Woman technologist ZWJ)",
        "🏳️‍🌈 (Rainbow flag sequence)",
        "👍🏻 👍🏼 👍🏽 👍🏾 👍🏿 (Skin-tone modifiers)",
        "🇮🇳 🇺🇸 🇯🇵 🇩🇪 🇬🇧 🇫🇷 🇨🇦 🇦🇺 (Regional flags)",
        # Diverse Global Scripts
        "深度学习与自然语言处理 (Simplified Chinese)",
        "機械学習とニューラルネットワーク (Japanese)",
        "인공지능 모델 훈련 (Korean)",
        "الذكاء الاصطناعي وتعلم الآلة (Arabic)",
        "בינה מלאכותית ולמידת מכונה (Hebrew)",
        "Нейронные сети и машинное обучение (Cyrillic)",
        "कृत्रिम बुद्धिमत्ता और तंत्रिका नेटवर्क, ०१२३४५६७೮೯ (Devanagari)",
        "செயற்கை நுண்ணறிவு (Tamil)",
        "కృత్రిమ మేధస్సు (Telugu)",
        # Box drawing & Braille & Math alphanumeric
        "┌─┬─┐\n│ │ │\n├─┼─┤\n└─┴─┘",
        "⠓⠑⠇⠇⠕ ⠺⠕⠗⠇⠙ (Braille: hello world)",
        "𝑼𝒏𝒊𝒄𝒐𝒅𝒆 𝑴𝒂𝒕𝒉 𝔉𝔯𝔞𝔨𝔱𝔲𝔯 𝔸𝕝𝕘𝕖𝕓𝕣𝕒 𝕏𝕐ℤ",
    ]
    for text in unseen_cases:
        cases.append(("UnseenUnicode", text))

    # 6. Boundary & Adversarial Edge Cases
    boundary_cases = [
        "",  # Empty string
        "a",  # Single ASCII
        " ",  # Single Space
        "\n", # Single Newline
        "೦",  # Single Kannada zero
        "೯",  # Single Kannada nine
        "🚀", # Single Emoji
        "a" * 1000, # Long repetitive ASCII
        "ನ" * 200,  # Long repetitive Kannada
        "೦" * 300,  # Long repetitive Kannada numerals
        "🚀" * 100, # Long repetitive Emoji
        # Special Tokens in Isolation
        "<|pad|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|unk|>",
        "<|creator_auth|>",
        "<|tara_rule|>",
        "<|tara_exec|>",
        "<|tara_skill|>",
        "<|tara_memory|>",
        # Special Tokens in Context
        "<|im_start|>system\nYou are TARA, an autonomous AI.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n",
        "embedded<|im_start|>token<|im_end|>without_spaces",
        # Literal Byte Token Strings
        "<0x00>",
        "<0x41>",
        "<0xFF>",
        "<0xAA>",
        "Testing literal byte token representation: <0x00> <0x20> <0x41> <0xFF> in text.",
        # Deep nested / multi-script stress
        "ದಿನಾಂಕ ೨೪-೧೦-೨೦೨೬ ರಂದು TARA Core v1.0 ಬಿಡುಗಡೆಯಾಗಲಿದೆ. Model parameters: ೧,೦೪೮,೫೭೬, Vocab: ೧೦೦,೨೫೬. Formula: \\int_0^\\infty e^{-x} dx = ೧. Emojis: 🚀✨🤖. Code: fn init() -> bool { true }",
        "a" * 5000,
        "ಕನ್ನಡ " * 500,
        "೦೧೨೩೪೫೬೭೮೯" * 200,
        "🚀✨🤖" * 300,
    ]
    for text in boundary_cases:
        cases.append(("Boundary", text))

    # 7. Randomized Unicode Fuzz Sequences (Deterministic seed)
    rng = random.Random(1337)
    # Define interesting Unicode ranges
    unicode_ranges = [
        (0x0020, 0x007E),    # ASCII
        (0x0C80, 0x0CFF),    # Kannada
        (0x0900, 0x097F),    # Devanagari
        (0x0370, 0x03FF),    # Greek
        (0x0400, 0x04FF),    # Cyrillic
        (0x2200, 0x22FF),    # Mathematical Operators
        (0x1F600, 0x1F64F),  # Emoticons
        (0x10000, 0x1007F),  # Linear B
        (0x11000, 0x1107F),  # Brahmi
    ]
    for fuzz_idx in range(500):
        length = rng.randint(5, 100)
        chars = []
        for _ in range(length):
            r_start, r_end = rng.choice(unicode_ranges)
            code_point = rng.randint(r_start, r_end)
            chars.append(chr(code_point))
        fuzz_str = "".join(chars)
        cases.append(("RandomFuzz", fuzz_str))

    return cases


def run_differential_fuzzing():
    print("=" * 80)
    print(" TARA CROSS-LANGUAGE DIFFERENTIAL FUZZING & TOKENIZER PARITY SUITE")
    print("=" * 80)

    corpus_cases = build_adversarial_corpus()
    print(f"Generated {len(corpus_cases)} adversarial test cases across 7 domains.")

    # Domain breakdown
    domain_counts: Dict[str, int] = {}
    for dom, _ in corpus_cases:
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
    for dom, count in domain_counts.items():
        print(f"  - {dom:<16}: {count:>4} cases")
    print("-" * 80)

    corpus_texts = [text for _, text in corpus_cases]

    # Save corpus to temporary file for Rust runner
    corpus_file = "target/fuzz_corpus_temp.json"
    os.makedirs("target", exist_ok=True)
    with open(corpus_file, "w", encoding="utf-8") as f:
        json.dump({"cases": corpus_texts}, f, ensure_ascii=False)

    models_to_test = [
        ("Candidate Model V1", "storage/models/tara_candidate_v1/tokenizer.json"),
        ("World Model", "storage/models/tara_world_model/tokenizer.json"),
    ]

    total_assertions = 0
    all_passed = True

    for model_name, model_path in models_to_test:
        print(f"\n>>> TESTING {model_name.upper()} ({model_path})")
        print("=" * 80)

        # 1. Load Python Tokenizer
        t0 = time.time()
        py_tok = TaraNativeTokenizer.from_file(model_path)
        py_load_time = time.time() - t0
        print(f"[Python] Loaded TaraNativeTokenizer (vocab={py_tok.vocab_size:,}) in {py_load_time:.3f}s")

        # 2. Run Rust Fuzzer Example Binary
        rust_out_file = f"target/rust_fuzz_results_{os.path.basename(os.path.dirname(model_path))}.json"
        rust_cmd = [
            "target/debug/examples/tokenizer_fuzzer.exe",
            "--model", model_path,
            "--input", corpus_file,
            "--output", rust_out_file,
        ]
        t0 = time.time()
        res = subprocess.run(rust_cmd, capture_output=True, text=True, encoding="utf-8")
        rust_time = time.time() - t0

        if res.returncode != 0:
            print(f"[ERROR] Rust fuzzer failed with return code {res.returncode}:")
            print(res.stderr)
            all_passed = False
            continue

        print(f"[Rust] Executed tokenizer_fuzzer on {len(corpus_texts)} cases in {rust_time:.3f}s")

        with open(rust_out_file, "r", encoding="utf-8") as f:
            rust_data = json.load(f)

        rust_results = rust_data["results"]
        assert len(rust_results) == len(corpus_cases), f"Result count mismatch: {len(rust_results)} vs {len(corpus_cases)}"

        # 3. Differential Comparison Loop
        mismatches = 0
        lossless_failures = 0
        unexpected_unk = 0
        total_tokens_generated = 0

        for idx, ((dom, text), rust_res) in enumerate(zip(corpus_cases, rust_results)):
            # Python encode and decode
            py_out = py_tok.encode(text)
            py_ids = py_out.ids
            py_decoded = py_tok.decode(py_ids)

            rust_ids = rust_res["ids"]
            rust_decoded = rust_res["decoded"]
            rust_lossless = rust_res["lossless"]
            rust_contains_unk = rust_res["contains_unk"]

            total_tokens_generated += len(py_ids)

            # Assertion 1: Exact Token ID Parity
            total_assertions += 1
            if py_ids != rust_ids:
                mismatches += 1
                all_passed = False
                print(f"[PARITY MISMATCH] Case #{idx} ({dom}):")
                print(f"  Input: {repr(text)}")
                print(f"  Python IDs ({len(py_ids)}): {py_ids}")
                print(f"  Rust IDs   ({len(rust_ids)}): {rust_ids}")

            # Assertion 2: Python Lossless Roundtrip
            total_assertions += 1
            # Note: control tokens (<|pad|>, <|im_start|>, <|im_end|>) are skipped by design during decode
            has_skipped_control_tokens = any(sp in text for sp in ["<|pad|>", "<|im_start|>", "<|im_end|>"])
            if not has_skipped_control_tokens:
                if py_decoded != text:
                    lossless_failures += 1
                    all_passed = False
                    print(f"[PY LOSSLESS FAILURE] Case #{idx} ({dom}):")
                    print(f"  Input:   {repr(text)}")
                    print(f"  Decoded: {repr(py_decoded)}")

                # Assertion 3: Rust Lossless Roundtrip
                total_assertions += 1
                if not rust_lossless or rust_decoded != text:
                    lossless_failures += 1
                    all_passed = False
                    print(f"[RUST LOSSLESS FAILURE] Case #{idx} ({dom}):")
                    print(f"  Input:   {repr(text)}")
                    print(f"  Decoded: {repr(rust_decoded)}")

            # Assertion 4: Zero UNK on Unseen Text
            total_assertions += 1
            input_has_literal_unk = "<|unk|>" in text
            if not input_has_literal_unk:
                if py_tok.unk_id in py_ids:
                    unexpected_unk += 1
                    all_passed = False
                    print(f"[PY UNEXPECTED UNK] Case #{idx} ({dom}): unk token found in {py_ids} for input {repr(text)}")
                if rust_contains_unk:
                    unexpected_unk += 1
                    all_passed = False
                    print(f"[RUST UNEXPECTED UNK] Case #{idx} ({dom}): unk token found in {rust_ids} for input {repr(text)}")

        print("-" * 80)
        print(f"Results for {model_name}:")
        print(f"  Test Cases Processed     : {len(corpus_cases):,}")
        print(f"  Total Tokens Generated   : {total_tokens_generated:,}")
        print(f"  Parity Mismatches        : {mismatches} (Target: 0)")
        print(f"  Lossless Decode Failures : {lossless_failures} (Target: 0)")
        print(f"  Unexpected UNK Tokens    : {unexpected_unk} (Target: 0)")
        if mismatches == 0 and lossless_failures == 0 and unexpected_unk == 0:
            print(f"  Status                   : PASSED (100% Differential Parity Verified)")
        else:
            print(f"  Status                   : FAILED")

    print("\n" + "=" * 80)
    print(f"SUMMARY: Total Assertions Executed = {total_assertions:,}")
    if all_passed:
        print("CROSS-LANGUAGE DIFFERENTIAL FUZZING: 100% PERFECT MATCH!")
        print("VERDICT: ZERO DISCREPANCIES DETECTED.")
    else:
        print("CROSS-LANGUAGE DIFFERENTIAL FUZZING: FAILURES DETECTED!")
    print("=" * 80)

    # Clean up temporary target corpus file
    if os.path.exists(corpus_file):
        os.remove(corpus_file)

    return all_passed


if __name__ == "__main__":
    success = run_differential_fuzzing()
    sys.exit(0 if success else 1)
