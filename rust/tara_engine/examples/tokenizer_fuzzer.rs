use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use tara_engine::tokenizer::TaraTokenizer;

#[derive(Debug, Deserialize)]
struct FuzzInput {
    cases: Vec<String>,
}

#[derive(Debug, Serialize)]
struct TestCaseResult {
    index: usize,
    token_count: usize,
    ids: Vec<u32>,
    decoded: String,
    lossless: bool,
    contains_unk: bool,
}

#[derive(Debug, Serialize)]
struct FuzzOutput {
    model: String,
    vocab_size: usize,
    total_cases: usize,
    results: Vec<TestCaseResult>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let mut model_path = PathBuf::from("storage/models/tara_candidate_v1/tokenizer.json");
    let mut input_path = PathBuf::from("test_corpus.json");
    let mut output_path = PathBuf::from("rust_fuzz_results.json");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                if i + 1 < args.len() {
                    model_path = PathBuf::from(&args[i + 1]);
                    i += 2;
                } else {
                    return Err("Missing value for --model".into());
                }
            }
            "--input" => {
                if i + 1 < args.len() {
                    input_path = PathBuf::from(&args[i + 1]);
                    i += 2;
                } else {
                    return Err("Missing value for --input".into());
                }
            }
            "--output" => {
                if i + 1 < args.len() {
                    output_path = PathBuf::from(&args[i + 1]);
                    i += 2;
                } else {
                    return Err("Missing value for --output".into());
                }
            }
            _ => {
                i += 1;
            }
        }
    }

    println!("[Rust Fuzzer] Loading tokenizer from: {:?}", model_path);
    let tokenizer = TaraTokenizer::from_file(model_path.to_str().ok_or("Invalid model path")?)?;
    let unk_id = tokenizer.unk_id();
    println!(
        "[Rust Fuzzer] Tokenizer loaded: vocab_size={}, max_token_len={}, unk_id={}",
        tokenizer.vocab_size, tokenizer.max_token_len, unk_id
    );

    println!("[Rust Fuzzer] Reading test corpus from: {:?}", input_path);
    let input_file = File::open(&input_path)?;
    let reader = BufReader::new(input_file);
    let input_data: FuzzInput = serde_json::from_reader(reader)?;

    println!(
        "[Rust Fuzzer] Processing {} test cases...",
        input_data.cases.len()
    );
    let mut results = Vec::with_capacity(input_data.cases.len());

    for (idx, text) in input_data.cases.iter().enumerate() {
        let ids = tokenizer.encode(text);
        let decoded = tokenizer.decode(&ids);
        let lossless = &decoded == text;
        let contains_unk = ids.contains(&unk_id);

        results.push(TestCaseResult {
            index: idx,
            token_count: ids.len(),
            ids,
            decoded,
            lossless,
            contains_unk,
        });
    }

    let output_data = FuzzOutput {
        model: model_path.to_string_lossy().into_owned(),
        vocab_size: tokenizer.vocab_size,
        total_cases: results.len(),
        results,
    };

    println!("[Rust Fuzzer] Writing results to: {:?}", output_path);
    let out_file = File::create(&output_path)?;
    let writer = BufWriter::new(out_file);
    serde_json::to_writer(writer, &output_data)?;
    println!("[Rust Fuzzer] Completed successfully.");

    Ok(())
}
