//! Canonical Ground-Truth Evaluation Benchmark Suite for TARA.
//!
//! Provides genuine, deterministic multi-task probe evaluation across four core capability domains:
//! 1. `Reasoning`: Multi-step arithmetic, deductive logic, and formal proof reasoning (GSM8K-style).
//! 2. `CodeSyntax`: Rust & Python syntax validity, algorithmic logic, and signature verification.
//! 3. `MultilingualQA`: Contemporary Kannada and English factual knowledge, reasoning, and comprehension.
//! 4. `RoboticsCausal`: Quadruped kinematics (Unitree Go2), physical dynamics, and causal state transitions.
//!
//! Metrics computed:
//! - Cross-Entropy Token Loss
//! - Causal Perplexity (PPL: `loss.exp()`)
//! - Top-1 Token Accuracy (`%` of target tokens matching argmax prediction)
//! - Exact Match (`bool` indicating complete sequence token equivalence)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::config::TaraConfig;
use crate::model::TaraForCausalLM;
use crate::tokenizer::TaraTokenizer;

/// Authoritative benchmark category classifications.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BenchmarkCategory {
    Reasoning,
    CodeSyntax,
    MultilingualQA,
    RoboticsCausal,
}

impl BenchmarkCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            BenchmarkCategory::Reasoning => "Reasoning",
            BenchmarkCategory::CodeSyntax => "CodeSyntax",
            BenchmarkCategory::MultilingualQA => "MultilingualQA",
            BenchmarkCategory::RoboticsCausal => "RoboticsCausal",
        }
    }
}

/// A single ground-truth evaluation probe with explicit prompt and target completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkProbe {
    pub id: String,
    pub category: BenchmarkCategory,
    pub prompt: String,
    pub expected_completion: String,
    pub description: String,
}

/// Detailed evaluation outcome for a single probe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeEvaluationResult {
    pub id: String,
    pub category: BenchmarkCategory,
    pub loss: f64,
    pub perplexity: f64,
    pub top1_accuracy: f64,
    pub exact_match: bool,
    pub target_tokens_count: usize,
    pub correct_tokens_count: usize,
}

/// Aggregated metrics for a single benchmark category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryReport {
    pub category: BenchmarkCategory,
    pub probe_count: usize,
    pub avg_loss: f64,
    pub avg_perplexity: f64,
    pub avg_top1_accuracy: f64,
    pub exact_match_count: usize,
    pub exact_match_ratio: f64,
}

/// Comprehensive report emitted after running the benchmark suite against a model checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuiteReport {
    pub total_probes: usize,
    pub overall_loss: f64,
    pub overall_perplexity: f64,
    pub overall_top1_accuracy: f64,
    pub total_exact_matches: usize,
    pub exact_match_ratio: f64,
    pub category_reports: HashMap<String, CategoryReport>,
    pub probe_results: Vec<ProbeEvaluationResult>,
}

impl BenchmarkSuiteReport {
    /// Formats a clean human-readable table of the evaluation results.
    pub fn print_summary(&self) {
        println!("================================================================================");
        println!("               TARA GROUND-TRUTH BENCHMARK EVALUATION REPORT                    ");
        println!("================================================================================");
        println!(
            "  Total Probes Evaluated   : {}",
            self.total_probes
        );
        println!(
            "  Overall Causal Loss      : {:.4}",
            self.overall_loss
        );
        println!(
            "  Overall Perplexity (PPL) : {:.4}",
            self.overall_perplexity
        );
        println!(
            "  Overall Top-1 Accuracy   : {:.2}%",
            self.overall_top1_accuracy
        );
        println!(
            "  Exact Match Rate         : {}/{} ({:.1}%)",
            self.total_exact_matches,
            self.total_probes,
            self.exact_match_ratio * 100.0
        );
        println!("--------------------------------------------------------------------------------");
        println!(" Category Breakdown:");
        println!("  {:<18} | {:<7} | {:<10} | {:<10} | {:<10} | {:<8}",
            "Category", "Probes", "Loss", "PPL", "Accuracy", "Exact");
        println!("  {:-<18}-+-{:-<7}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<8}",
            "", "", "", "", "", "");

        let mut cats: Vec<&CategoryReport> = self.category_reports.values().collect();
        cats.sort_by(|a, b| a.category.as_str().cmp(b.category.as_str()));

        for c in cats {
            println!(
                "  {:<18} | {:<7} | {:<10.4} | {:<10.4} | {:<9.2}% | {}/{}",
                c.category.as_str(),
                c.probe_count,
                c.avg_loss,
                c.avg_perplexity,
                c.avg_top1_accuracy,
                c.exact_match_count,
                c.probe_count
            );
        }
        println!("================================================================================");
    }
}

/// The Ground-Truth Benchmark Suite manager.
#[derive(Debug, Clone)]
pub struct BenchmarkSuite {
    pub probes: Vec<BenchmarkProbe>,
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::canonical_suite()
    }
}

impl BenchmarkSuite {
    /// Constructs a benchmark suite containing the canonical gold-standard probes.
    pub fn canonical_suite() -> Self {
        let probes = vec![
            // ---------------------------------------------------------------------
            // 1. REASONING PROBES (Multi-step logic & arithmetic)
            // ---------------------------------------------------------------------
            BenchmarkProbe {
                id: "reasoning_gsm8k_01".to_string(),
                category: BenchmarkCategory::Reasoning,
                prompt: "Question: Janet buys 3 packs of 12 pencils each. She gives 10 pencils to her brother. How many pencils does Janet have left?\nAnswer: Let's calculate step by step:\n".to_string(),
                expected_completion: "1. Janet starts with 3 packs of 12 pencils: 3 * 12 = 36 pencils.\n2. She gives 10 pencils to her brother: 36 - 10 = 26 pencils.\nTherefore, Janet has 26 pencils left.".to_string(),
                description: "Multi-step Grade-School Math reasoning (multiplication and subtraction).".to_string(),
            },
            BenchmarkProbe {
                id: "reasoning_deductive_01".to_string(),
                category: BenchmarkCategory::Reasoning,
                prompt: "Premise 1: All primates are mammals.\nPremise 2: A chimpanzee is a primate.\nConclusion: Therefore, a chimpanzee is a".to_string(),
                expected_completion: " mammal.".to_string(),
                description: "Categorical syllogism and deductive logical inference.".to_string(),
            },
            BenchmarkProbe {
                id: "reasoning_speed_distance_01".to_string(),
                category: BenchmarkCategory::Reasoning,
                prompt: "Question: If an electric vehicle travels at an average speed of 60 km/h for 2.5 hours, what total distance in kilometers does it cover?\nAnswer:".to_string(),
                expected_completion: " Distance = speed * time = 60 * 2.5 = 150 kilometers.".to_string(),
                description: "Physics arithmetic rate and dimensional computation.".to_string(),
            },

            // ---------------------------------------------------------------------
            // 2. CODE SYNTAX PROBES (Rust & Python functional primitives)
            // ---------------------------------------------------------------------
            BenchmarkProbe {
                id: "code_rust_euclidean_01".to_string(),
                category: BenchmarkCategory::CodeSyntax,
                prompt: "/// Calculate 2D Euclidean distance between two points.\npub fn euclidean_distance(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {\n".to_string(),
                expected_completion: "    let dx = x2 - x1;\n    let dy = y2 - y1;\n    (dx * dx + dy * dy).sqrt()\n}".to_string(),
                description: "Standard Rust function implementation and mathematical expression.".to_string(),
            },
            BenchmarkProbe {
                id: "code_python_list_comp_01".to_string(),
                category: BenchmarkCategory::CodeSyntax,
                prompt: "def filter_even_squares(numbers: list[int]) -> list[int]:\n    \"\"\"Return squares of even numbers in order.\"\"\"\n".to_string(),
                expected_completion: "    return [n * n for n in numbers if n % 2 == 0]".to_string(),
                description: "Python list comprehension with conditional filter predicate.".to_string(),
            },
            BenchmarkProbe {
                id: "code_rust_result_parsing_01".to_string(),
                category: BenchmarkCategory::CodeSyntax,
                prompt: "pub fn parse_positive_i32(s: &str) -> Result<i32, String> {\n".to_string(),
                expected_completion: "    let val = s.trim().parse::<i32>().map_err(|e| e.to_string())?;\n    if val > 0 {\n        Ok(val)\n    } else {\n        Err(\"Value must be strictly positive\".to_string())\n    }\n}".to_string(),
                description: "Idiomatic Rust error propagation with Result and parsing validation.".to_string(),
            },

            // ---------------------------------------------------------------------
            // 3. MULTILINGUAL QA PROBES (Contemporary Kannada & English)
            // ---------------------------------------------------------------------
            BenchmarkProbe {
                id: "multilingual_kn_sunlight_01".to_string(),
                category: BenchmarkCategory::MultilingualQA,
                prompt: "ಪ್ರಶ್ನೆ: ಸೂರ್ಯನ ಬೆಳಕು ಭೂಮಿಗೆ ತಲುಪಲು ಸುಮಾರು ಎಷ್ಟು ಸಮಯ ಬೇಕಾಗುತ್ತದೆ?\nಉತ್ತರ:".to_string(),
                expected_completion: " ಸೂರ್ಯನ ಬೆಳಕು ಭೂಮಿಗೆ ತಲುಪಲು ಸುಮಾರು ೮ ನಿಮಿಷ ೨೦ ಸೆಕೆಂಡುಗಳು (8 minutes 20 seconds) ಬೇಕಾಗುತ್ತದೆ.".to_string(),
                description: "Factual science QA in modern literary Kannada.".to_string(),
            },
            BenchmarkProbe {
                id: "multilingual_kn_binary_01".to_string(),
                category: BenchmarkCategory::MultilingualQA,
                prompt: "ಪ್ರಶ್ನೆ: ಕಂಪ್ಯೂಟರ್ ಪ್ರೋಗ್ರಾಮಿಂಗ್‌ನಲ್ಲಿ ಬೈನರಿ ಸಂಖ್ಯಾ ವ್ಯವಸ್ಥೆ ಯಾವ ಎರಡು ಅಂಕೆಗಳನ್ನು ಬಳಸುತ್ತದೆ?\nಉತ್ತರ:".to_string(),
                expected_completion: " ಬೈನರಿ ಸಂಖ್ಯಾ ವ್ಯವಸ್ಥೆಯು ೦ ಮತ್ತು ೧ (0 ಮತ್ತು 1) ಅಂಕೆಗಳನ್ನು ಬಳಸುತ್ತದೆ.".to_string(),
                description: "Computer science fundamental concepts QA in Kannada.".to_string(),
            },
            BenchmarkProbe {
                id: "multilingual_en_organic_01".to_string(),
                category: BenchmarkCategory::MultilingualQA,
                prompt: "Question: What is the primary chemical element that forms the basis of all organic molecules?\nAnswer:".to_string(),
                expected_completion: " Carbon is the primary chemical element that forms the structural basis of organic molecules.".to_string(),
                description: "Fundamental chemistry and natural sciences QA in English.".to_string(),
            },

            // ---------------------------------------------------------------------
            // 4. ROBOTICS & CAUSAL WORLD MODEL PROBES (Go2 Kinematics & Physics)
            // ---------------------------------------------------------------------
            BenchmarkProbe {
                id: "robotics_go2_pitch_recovery_01".to_string(),
                category: BenchmarkCategory::RoboticsCausal,
                prompt: "State: Quadruped Unitree Go2 trot gait at velocity 1.2 m/s.\nAction: Front left contact lost, pitch angle increased by +0.08 rad.\nExpected stabilization corrective response:".to_string(),
                expected_completion: " Increase rear leg damping, apply downward pitch recovery torque to stabilize body frame.".to_string(),
                description: "Causal stabilization dynamics for quadruped locomotion state recovery.".to_string(),
            },
            BenchmarkProbe {
                id: "robotics_physics_friction_01".to_string(),
                category: BenchmarkCategory::RoboticsCausal,
                prompt: "Causal Physics: A 5kg mass slides on an inclined plane of angle theta with friction coefficient mu.\nNet force parallel to surface:".to_string(),
                expected_completion: " F_net = m * g * sin(theta) - mu * m * g * cos(theta).".to_string(),
                description: "Physical dynamics equation representing gravitational and frictional causal force.".to_string(),
            },
        ];

        Self { probes }
    }

    /// Extends the suite by dynamically ingesting additional JSONL evaluation probes from disk.
    /// Supports parameter-driven extensibility without hardcoded capacity limits.
    pub fn load_probes_from_jsonl(&mut self, path: &Path) -> Result<usize, String> {
        let file = File::open(path)
            .map_err(|e| format!("Failed to open benchmark probe file '{}': {}", path.display(), e))?;
        let reader = BufReader::new(file);
        let mut added = 0;

        for (idx, line_res) in reader.lines().enumerate() {
            let line = line_res.map_err(|e| format!("Error reading line {}: {}", idx + 1, e))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let probe: BenchmarkProbe = serde_json::from_str(trimmed)
                .map_err(|e| format!("Malformed probe JSON at line {}: {}", idx + 1, e))?;
            self.probes.push(probe);
            added += 1;
        }

        Ok(added)
    }

    /// Evaluates the complete benchmark suite against a loaded causal language model.
    pub fn evaluate(
        &self,
        model: &TaraForCausalLM,
        tokenizer: &TaraTokenizer,
        config: &TaraConfig,
    ) -> Result<BenchmarkSuiteReport, String> {
        if self.probes.is_empty() {
            return Err("BenchmarkSuite contains zero probes to evaluate.".to_string());
        }

        let mut probe_results = Vec::with_capacity(self.probes.len());
        let mut category_totals: HashMap<BenchmarkCategory, (usize, f64, f64, f64, usize)> = HashMap::new();

        let mut overall_loss_sum = 0.0f64;
        let mut overall_ppl_sum = 0.0f64;
        let mut overall_acc_sum = 0.0f64;
        let mut total_exact_matches = 0usize;

        for probe in &self.probes {
            let result = Self::evaluate_single_probe(model, tokenizer, config, probe)?;

            overall_loss_sum += result.loss;
            overall_ppl_sum += result.perplexity;
            overall_acc_sum += result.top1_accuracy;
            if result.exact_match {
                total_exact_matches += 1;
            }

            let entry = category_totals.entry(probe.category).or_insert((0, 0.0, 0.0, 0.0, 0));
            entry.0 += 1; // count
            entry.1 += result.loss;
            entry.2 += result.perplexity;
            entry.3 += result.top1_accuracy;
            if result.exact_match {
                entry.4 += 1;
            }

            probe_results.push(result);
        }

        let total_count = self.probes.len() as f64;
        let overall_loss = overall_loss_sum / total_count;
        let overall_perplexity = overall_ppl_sum / total_count;
        let overall_top1_accuracy = overall_acc_sum / total_count;
        let exact_match_ratio = total_exact_matches as f64 / total_count;

        let mut category_reports = HashMap::new();
        for (cat, (count, loss_s, ppl_s, acc_s, em_count)) in category_totals {
            let cnt_f = count as f64;
            category_reports.insert(
                cat.as_str().to_string(),
                CategoryReport {
                    category: cat,
                    probe_count: count,
                    avg_loss: loss_s / cnt_f,
                    avg_perplexity: ppl_s / cnt_f,
                    avg_top1_accuracy: acc_s / cnt_f,
                    exact_match_count: em_count,
                    exact_match_ratio: em_count as f64 / cnt_f,
                },
            );
        }

        Ok(BenchmarkSuiteReport {
            total_probes: self.probes.len(),
            overall_loss,
            overall_perplexity,
            overall_top1_accuracy,
            total_exact_matches,
            exact_match_ratio,
            category_reports,
            probe_results,
        })
    }

    /// Evaluates a single probe forward pass and computes token-level metrics.
    fn evaluate_single_probe(
        model: &TaraForCausalLM,
        tokenizer: &TaraTokenizer,
        config: &TaraConfig,
        probe: &BenchmarkProbe,
    ) -> Result<ProbeEvaluationResult, String> {
        let prompt_tokens = tokenizer.encode(&probe.prompt);
        let completion_tokens = tokenizer.encode(&probe.expected_completion);

        if prompt_tokens.is_empty() {
            return Err(format!("Probe '{}' has empty encoded prompt.", probe.id));
        }
        if completion_tokens.is_empty() {
            return Err(format!("Probe '{}' has empty encoded expected completion.", probe.id));
        }

        // Full sequence: prompt + completion
        let mut full_sequence = Vec::with_capacity(prompt_tokens.len() + completion_tokens.len());
        full_sequence.extend_from_slice(&prompt_tokens);
        full_sequence.extend_from_slice(&completion_tokens);

        let seq_len = full_sequence.len();
        let vs = config.vocab_size;

        let (_, _, logits) = model.forward_with_cache(&full_sequence);
        let expected_logits_len = seq_len * vs;
        if logits.len() != expected_logits_len {
            return Err(format!(
                "Probe '{}' forward pass logits length mismatch: expected {}, got {}",
                probe.id, expected_logits_len, logits.len()
            ));
        }

        let prompt_len = prompt_tokens.len();
        let completion_len = completion_tokens.len();
        let mut probe_loss_sum = 0.0f64;
        let mut correct_tokens_count = 0usize;

        // Evaluate target completion tokens (target index = prompt_len + k, predictor index = target index - 1)
        for (k, &comp_token) in completion_tokens.iter().enumerate() {
            let target_token = comp_token as usize;
            if target_token >= vs {
                return Err(format!(
                    "Target token ID {} in probe '{}' exceeds vocabulary size {}",
                    target_token, probe.id, vs
                ));
            }

            let predictor_idx = prompt_len + k - 1;
            let logits_slice = &logits[predictor_idx * vs..(predictor_idx + 1) * vs];

            // 1. Numerically stable log-softmax cross-entropy loss
            let mut max_val = f32::NEG_INFINITY;
            let mut argmax_token = 0usize;

            for (idx, &val) in logits_slice.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    argmax_token = idx;
                }
            }

            let mut sum_exp = 0.0f64;
            for &val in logits_slice {
                sum_exp += ((val - max_val) as f64).exp();
            }

            let target_val = logits_slice[target_token];
            let log_prob = ((target_val - max_val) as f64) - sum_exp.ln();
            let token_loss = -log_prob;

            if token_loss.is_finite() {
                probe_loss_sum += token_loss;
            }

            // 2. Top-1 Token Accuracy
            if argmax_token == target_token {
                correct_tokens_count += 1;
            }
        }

        let target_tokens_count = completion_len;
        let avg_loss = if target_tokens_count > 0 {
            probe_loss_sum / target_tokens_count as f64
        } else {
            f64::NAN
        };
        let perplexity = if avg_loss.is_finite() {
            avg_loss.exp()
        } else {
            f64::NAN
        };
        let top1_accuracy = if target_tokens_count > 0 {
            (correct_tokens_count as f64 / target_tokens_count as f64) * 100.0
        } else {
            0.0
        };
        let exact_match = correct_tokens_count == target_tokens_count;

        Ok(ProbeEvaluationResult {
            id: probe.id.clone(),
            category: probe.category,
            loss: avg_loss,
            perplexity,
            top1_accuracy,
            exact_match,
            target_tokens_count,
            correct_tokens_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_benchmark_suite_composition() {
        let suite = BenchmarkSuite::canonical_suite();
        assert!(!suite.probes.is_empty(), "Canonical suite must have probes");

        let mut counts: HashMap<BenchmarkCategory, usize> = HashMap::new();
        for probe in &suite.probes {
            *counts.entry(probe.category).or_insert(0) += 1;
            assert!(!probe.prompt.is_empty(), "Probe prompt cannot be empty: {}", probe.id);
            assert!(!probe.expected_completion.is_empty(), "Probe completion cannot be empty: {}", probe.id);
        }

        assert!(counts.contains_key(&BenchmarkCategory::Reasoning));
        assert!(counts.contains_key(&BenchmarkCategory::CodeSyntax));
        assert!(counts.contains_key(&BenchmarkCategory::MultilingualQA));
        assert!(counts.contains_key(&BenchmarkCategory::RoboticsCausal));
    }

    #[test]
    fn test_probe_serialization_roundtrip() {
        let probe = BenchmarkProbe {
            id: "test_probe_01".to_string(),
            category: BenchmarkCategory::Reasoning,
            prompt: "What is 2 + 2?".to_string(),
            expected_completion: " 4".to_string(),
            description: "Basic arithmetic validation probe.".to_string(),
        };

        let json = serde_json::to_string(&probe).expect("Must serialize probe");
        let decoded: BenchmarkProbe = serde_json::from_str(&json).expect("Must deserialize probe");

        assert_eq!(decoded.id, probe.id);
        assert_eq!(decoded.category, probe.category);
        assert_eq!(decoded.prompt, probe.prompt);
        assert_eq!(decoded.expected_completion, probe.expected_completion);
    }
}

