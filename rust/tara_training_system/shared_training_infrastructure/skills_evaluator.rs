//! Skills evaluator: tests model performance on skill-specific prompts.

use serde_json::{json, Value};

use crate::generate::{generate_response, GenerateOptions};
use crate::model::causal_lm::TaraForCausalLM;
use crate::tokenizer::TaraTokenizer;

/// Evaluates model output quality for named skills.
pub struct SkillsEvaluator {
    pub model_dir: String,
}

impl SkillsEvaluator {
    /// Create an evaluator for the given model directory.
    pub fn new(model_dir: &str) -> Self {
        Self {
            model_dir: model_dir.to_string(),
        }
    }

    /// Load model and tokenizer, run each test prompt, return evaluation results.
    pub fn evaluate(&self, skill_name: &str, test_prompts: &[String]) -> Value {
        let model_result = TaraForCausalLM::load(&self.model_dir);
        let tokenizer_path = format!("{}/tokenizer.json", self.model_dir);
        let tokenizer_result = TaraTokenizer::from_file(&tokenizer_path);

        match (model_result, tokenizer_result) {
            (Ok(model), Ok(tokenizer)) => {
                let mut results = Vec::new();
                let mut total_tokens = 0usize;
                let mut total_ms = 0.0f64;

                let options = GenerateOptions {
                    max_new_tokens: 20,
                    temperature: 0.7,
                    top_k: 50,
                    top_p: 0.9,
                    repetition_penalty: 1.1,
                    stop_tokens: None,
                };

                for prompt in test_prompts {
                    match generate_response(&model, &tokenizer, prompt, &options) {
                        Ok(res) => {
                            total_tokens += res.token_count;
                            total_ms += res.total_latency_ms;
                            results.push(json!({
                                "prompt": prompt,
                                "output": res.text,
                                "tokens": res.token_count,
                                "latency_ms": res.total_latency_ms,
                                "tps": res.tps,
                                "pass": !res.text.is_empty()
                            }));
                        }
                        Err(e) => {
                            results.push(json!({
                                "prompt": prompt,
                                "error": e,
                                "pass": false
                            }));
                        }
                    }
                }

                let pass_count = results
                    .iter()
                    .filter(|r| r["pass"].as_bool().unwrap_or(false))
                    .count();
                json!({
                    "skill": skill_name,
                    "total_prompts": test_prompts.len(),
                    "passed": pass_count,
                    "failed": test_prompts.len() - pass_count,
                    "pass_rate": pass_count as f32 / test_prompts.len().max(1) as f32,
                    "avg_tokens": total_tokens as f32 / test_prompts.len().max(1) as f32,
                    "avg_latency_ms": total_ms / test_prompts.len().max(1) as f64,
                    "results": results
                })
            }
            (Err(e), _) => json!({ "error": format!("model load failed: {}", e) }),
            (_, Err(e)) => json!({ "error": format!("tokenizer load failed: {}", e) }),
        }
    }
}
