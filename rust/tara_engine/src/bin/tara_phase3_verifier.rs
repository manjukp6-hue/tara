//! TARA Phase 3 Verification Binary
//! Executes Test 2 (Rust / Python inference compatibility) and Test 3 (LoRA integration).

use std::collections::HashMap;
use std::fs;
use tara_engine::config::TaraConfig;
use tara_engine::TaraForCausalLM;
use tara_engine::{LoraAdapterSet, LoraConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    match mode {
        "test2_rust_inference" => {
            let model_dir = args.get(2).expect("model_dir required");
            let tokens_json = args.get(3).expect("tokens json required");
            let tokens: Vec<u32> = serde_json::from_str(tokens_json)?;

            let config_path = format!("{}/config.json", model_dir);
            let config_content = fs::read_to_string(&config_path)?;
            let config: TaraConfig = serde_json::from_str(&config_content)?;

            let model = TaraForCausalLM::load(model_dir)?;
            let logits = model.forward(&tokens);

            let out = serde_json::json!({
                "status": "SUCCESS",
                "vocab_size": config.vocab_size,
                "seq_len": tokens.len(),
                "logits_len": logits.len(),
                "logits_sample": &logits[..logits.len().min(10)],
                "logits_all": logits,
            });
            println!("{}", serde_json::to_string(&out)?);
        }
        "test3_lora_merge_test" => {
            // Verify LoRA layer merge and unmerge with TaraForCausalLM weights
            let hidden_size = 64;
            let num_layers = 1;
            let vocab_size = 100;
            let config = TaraConfig {
                vocab_size,
                hidden_size,
                intermediate_size: 128,
                num_hidden_layers: num_layers,
                num_attention_heads: 4,
                num_key_value_heads: 2,
                head_dim: 16,
                max_position_embeddings: 128,
                rope_theta: 500000.0,
                rms_norm_eps: 1e-5,
                initializer_range: 0.02,
                version: "test".into(),
                model_type: "tara".into(),
                tie_word_embeddings: None,
            };

            // Synthetic base weights with non-uniform values
            let mut weights = HashMap::new();
            let mut embed = Vec::with_capacity(vocab_size * hidden_size);
            for v in 0..vocab_size {
                for h in 0..hidden_size {
                    embed.push(((v * 31 + h * 17) % 100) as f32 * 0.01 - 0.5);
                }
            }
            weights.insert("model.embed_tokens.weight".into(), embed);
            weights.insert("model.norm.weight".into(), vec![1.0; hidden_size]);
            let mut lm_head = Vec::with_capacity(vocab_size * hidden_size);
            for v in 0..vocab_size {
                for h in 0..hidden_size {
                    lm_head.push(((v * 19 + h * 23) % 100) as f32 * 0.01 - 0.5);
                }
            }
            weights.insert("lm_head.weight".into(), lm_head);
            for i in 0..num_layers {
                let p = format!("model.layers.{i}");
                weights.insert(format!("{p}.input_layernorm.weight"), vec![1.0; hidden_size]);
                weights.insert(format!("{p}.post_attention_layernorm.weight"), vec![1.0; hidden_size]);
                weights.insert(format!("{p}.self_attn.q_proj.weight"), vec![0.05; 4 * 16 * hidden_size]);
                weights.insert(format!("{p}.self_attn.k_proj.weight"), vec![0.05; 2 * 16 * hidden_size]);
                weights.insert(format!("{p}.self_attn.v_proj.weight"), vec![0.05; 2 * 16 * hidden_size]);
                weights.insert(format!("{p}.self_attn.o_proj.weight"), vec![0.05; hidden_size * 4 * 16]);
                weights.insert(format!("{p}.mlp.gate_proj.weight"), vec![0.05; 128 * hidden_size]);
                weights.insert(format!("{p}.mlp.up_proj.weight"), vec![0.05; 128 * hidden_size]);
                weights.insert(format!("{p}.mlp.down_proj.weight"), vec![0.05; hidden_size * 128]);
            }

            let base_model = TaraForCausalLM::from_weights_ref(&weights, &config, "mem")?;
            let tokens = vec![1, 5, 9];
            let base_logits = base_model.forward(&tokens);

            // Construct LoRA adapter set
            let lora_cfg = LoraConfig {
                r: 4,
                alpha: 8.0,
                target_modules: vec!["q_proj".into(), "v_proj".into()],
                dropout: 0.0,
                ..Default::default()
            };
            let mut adapter_set = LoraAdapterSet::new("test_ability", lora_cfg);
            let q_target = "model.layers.0.self_attn.q_proj.weight";
            let v_target = "model.layers.0.self_attn.v_proj.weight";
            adapter_set.add_layer(q_target, hidden_size, 4 * 16);
            adapter_set.add_layer(v_target, hidden_size, 2 * 16);

            // Give non-zero weights to B so delta is non-zero
            if let Some(l) = adapter_set.layers.get_mut(q_target) {
                for x in l.b.iter_mut() {
                    *x = 0.02;
                }
            }
            if let Some(l) = adapter_set.layers.get_mut(v_target) {
                for x in l.b.iter_mut() {
                    *x = 0.02;
                }
            }

            // Test merging into base weights
            let mut merged_weights = weights.clone();
            let merged_count = adapter_set.merge_into_weights(&mut merged_weights)?;
            assert_eq!(merged_count, 2, "Expected 2 layers merged");

            // Base weights in `weights` map must remain completely unchanged!
            let q_base_val = weights[q_target][0];
            let q_merged_val = merged_weights[q_target][0];
            assert!(
                (q_merged_val - q_base_val).abs() > 0.0,
                "Merged weights must differ from base"
            );

            let merged_model = TaraForCausalLM::from_weights_ref(&merged_weights, &config, "merged_mem")?;
            let merged_logits = merged_model.forward(&tokens);

            let logit_diff: f32 = merged_logits
                .iter()
                .zip(base_logits.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            println!("DEBUG: merged_count={}, q_base_val={}, q_merged_val={}", merged_count, q_base_val, q_merged_val);
            println!("DEBUG: base_logits[..5]={:?}", &base_logits[..5]);
            println!("DEBUG: merged_logits[..5]={:?}", &merged_logits[..5]);
            println!("DEBUG: logit_diff={}", logit_diff);
            assert!(logit_diff > 0.0, "Merged model must produce adapted logits");

            // Test unmerge
            let mut unmerged_weights = merged_weights.clone();
            let unmerged_count = adapter_set.unmerge_from_weights(&mut unmerged_weights)?;
            assert_eq!(unmerged_count, 2, "Expected 2 layers unmerged");

            // Compare unmerged with original weights
            let mut max_unmerge_err = 0.0f32;
            for (k, v_orig) in &weights {
                let v_un = &unmerged_weights[k];
                for (a, b) in v_orig.iter().zip(v_un.iter()) {
                    max_unmerge_err = max_unmerge_err.max((a - b).abs());
                }
            }
            assert!(
                max_unmerge_err < 1e-6,
                "Unmerged weights must recover original bit-for-bit (err: {})",
                max_unmerge_err
            );

            let out = serde_json::json!({
                "status": "PASS",
                "merged_layers": merged_count,
                "unmerged_layers": unmerged_count,
                "logit_adaptation_diff": logit_diff,
                "max_unmerge_recovery_error": max_unmerge_err,
            });
            println!("{}", serde_json::to_string(&out)?);
        }
        _ => {
            eprintln!("Unknown mode: {mode}");
            std::process::exit(1);
        }
    }
    Ok(())
}
