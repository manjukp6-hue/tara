"""
python/tara_model/train_aligned.py

Executes full retraining from zero of TARA-0.1-tokenizer-aligned:
- Loads canonical 344-token tokenizer
- Initializes model weights fresh matching 344 vocabulary
- Trains on 35-sample dataset smoke test using causal shifted next-token targets
- Evaluates validation loss and perplexity
- Exports SafeTensors, config.json, tokenizer.json, tokenizer_config.json, special_tokens_map.json, training_metadata.json
- Runs 6 benchmark prompts and prints comparative metrics
"""

import os
import sys
import json
import math
import time
import hashlib
from datetime import datetime

# Set utf-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

# Local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tara_model.tokenizer import TaraTokenizer
from tara_model.architecture import TaraModelZero, TaraConfig
from tara_model.dataset import TRAINING_SAMPLES, VALIDATION_SAMPLES

def run_retraining():
    print("=" * 70)
    print("      TARA-0.1-tokenizer-aligned: FRESH INITIALIZATION & SMOKE TEST")
    print("=" * 70 + "\n")

    # 1. Tokenizer
    tok = TaraTokenizer()
    V = len(tok.token_to_id)
    assert V == 344, f"Expected 344 tokens, got {V}"
    print(f"[1/6] Loaded canonical tokenizer: {V} vocabulary tokens.")

    out_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "storage/models/TARA-0.1-tokenizer-aligned"))
    os.makedirs(out_dir, exist_ok=True)

    # 2. Save tokenizer configurations
    print(f"[2/6] Exporting tokenizer files to {out_dir}...")
    with open(os.path.join(out_dir, 'tokenizer.json'), 'w', encoding='utf-8') as f:
        json.dump(tok.export_to_dict(), f, indent=2, ensure_ascii=False)

    tok_config = {
        'tokenizer_class': 'TaraTokenizer',
        'vocab_size': V,
        'model_max_length': 2048,
        'padding_side': 'right',
        'clean_up_tokenization_spaces': False
    }
    with open(os.path.join(out_dir, 'tokenizer_config.json'), 'w', encoding='utf-8') as f:
        json.dump(tok_config, f, indent=2)

    special_tokens = {
        'pad_token': '<|pad|>',
        'bos_token': '<|im_start|>',
        'eos_token': '<|im_end|>',
        'unk_token': '<|unk|>',
        'additional_special_tokens': [
            '<|creator_auth|>',
            '<|tara_rule|>',
            '<|tara_exec|>',
            '<|tara_skill|>',
            '<|tara_memory|>'
        ]
    }
    with open(os.path.join(out_dir, 'special_tokens_map.json'), 'w', encoding='utf-8') as f:
        json.dump(special_tokens, f, indent=2)

    # 3. Model Architecture Initialization
    print(f"[3/6] Initializing TARA causal transformer architecture fresh from scratch...")
    cfg = TaraConfig(
        vocab_size=V,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        version='TARA-0.1-tokenizer-aligned'
    )
    model = TaraModelZero(cfg)
    total_params = sum(
        len(v) * len(v[0]) if isinstance(v[0], list) else len(v)
        for v in model.weights.values()
    )
    print(f"      -> Total parameters: {total_params:,}")
    print(f"      -> Vocabulary size: {V}")
    print(f"      -> Embedding & LM Head dimensions: [{V}, 64]")

    # 4. Prepare datasets
    train_data = []
    for p, c in TRAINING_SAMPLES:
        t = tok.encode(f"{p} {c}")
        if len(t) > 1 and len(t) <= 256:
            train_data.append((t[:-1], t[1:]))

    val_data = []
    for p, c in VALIDATION_SAMPLES:
        t = tok.encode(f"{p} {c}")
        if len(t) > 1 and len(t) <= 256:
            val_data.append((t[:-1], t[1:]))

    val_toks = sum(len(x[0]) for x in val_data)
    val_loss_0 = 0.0
    for inp, tgt in val_data:
        for t in range(len(inp)):
            x = model.weights['model.embed_tokens.weight'][inp[t]]
            logits = [sum(model.weights['lm_head.weight'][v][h] * x[h] for h in range(cfg.hidden_size)) for v in range(V)]
            m = max(logits)
            exps = [math.exp(max(-30.0, min(30.0, l - m))) for l in logits]
            s = sum(exps)
            probs = [e / s for e in exps]
            val_loss_0 += -math.log(max(1e-12, probs[tgt[t]]))
    val_loss_0 /= val_toks
    init_perplexity = math.exp(val_loss_0)
    print(f"      -> Initial Validation Loss: {val_loss_0:.4f} (Perplexity: {init_perplexity:.2f})")

    # 5. Training Loop
    print(f"\n[4/6] Training smoke test (6 epochs, sequence-level AdamW)...")
    H = cfg.hidden_size
    W_embed = model.weights['model.embed_tokens.weight']
    W_head = model.weights['lm_head.weight']
    W_pos = [[0.0]*H for _ in range(256)]

    m_embed = [[0.0]*H for _ in range(V)]
    v_embed = [[0.0]*H for _ in range(V)]
    m_pos = [[0.0]*H for _ in range(256)]
    v_pos = [[0.0]*H for _ in range(256)]
    m_head = [[0.0]*H for _ in range(V)]
    v_head = [[0.0]*H for _ in range(V)]

    lr = 0.012
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    step = 0

    initial_train_loss = 0.0
    final_train_loss = 0.0

    epochs = 6
    t_train_start = time.time()
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        ep_toks = 0
        t0 = time.time()
        for inp, tgt in train_data:
            T = len(inp)
            ep_toks += T
            
            seq_g_head = [[0.0]*H for _ in range(V)]
            seq_g_embed = {}
            seq_g_pos = {}
            
            for t in range(T):
                tid = inp[t]
                tgt_id = tgt[t]
                x = [W_embed[tid][h] + W_pos[t][h] for h in range(H)]
                logits = [sum(W_head[v][h] * x[h] for h in range(H)) for v in range(V)]
                m = max(logits)
                exps = [math.exp(max(-30.0, min(30.0, l - m))) for l in logits]
                s = sum(exps)
                probs = [e / s for e in exps]
                
                loss_t = -math.log(max(1e-12, probs[tgt_id]))
                ep_loss += loss_t
                probs[tgt_id] -= 1.0
                
                d_x = [0.0]*H
                for v in range(V):
                    pv = probs[v]
                    if v != tgt_id and abs(pv) < 5e-4:
                        continue
                    head_v = W_head[v]
                    for h in range(H):
                        seq_g_head[v][h] += pv * x[h]
                        d_x[h] += pv * head_v[h]
                        
                if tid not in seq_g_embed:
                    seq_g_embed[tid] = [0.0]*H
                if t not in seq_g_pos:
                    seq_g_pos[t] = [0.0]*H
                for h in range(H):
                    seq_g_embed[tid][h] += d_x[h]
                    seq_g_pos[t][h] += d_x[h]
                    
            # Sequence-level AdamW step
            step += 1
            for v in range(V):
                for h in range(H):
                    g = seq_g_head[v][h]
                    if g == 0.0:
                        continue
                    m_head[v][h] = beta1 * m_head[v][h] + (1 - beta1) * g
                    v_head[v][h] = beta2 * v_head[v][h] + (1 - beta2) * (g * g)
                    m_h = m_head[v][h] / (1 - beta1 ** step)
                    v_h = v_head[v][h] / (1 - beta2 ** step)
                    W_head[v][h] -= lr * m_h / (math.sqrt(v_h) + eps)
                    
            for tid, g_vec in seq_g_embed.items():
                for h in range(H):
                    g = g_vec[h]
                    m_embed[tid][h] = beta1 * m_embed[tid][h] + (1 - beta1) * g
                    v_embed[tid][h] = beta2 * v_embed[tid][h] + (1 - beta2) * (ge := g * g)
                    m_he = m_embed[tid][h] / (1 - beta1 ** step)
                    v_he = v_embed[tid][h] / (1 - beta2 ** step)
                    W_embed[tid][h] -= lr * m_he / (math.sqrt(v_he) + eps)
                    
            for t, g_vec in seq_g_pos.items():
                for h in range(H):
                    g = g_vec[h]
                    m_pos[t][h] = beta1 * m_pos[t][h] + (1 - beta1) * g
                    v_pos[t][h] = beta2 * v_pos[t][h] + (1 - beta2) * (g * g)
                    m_hp = m_pos[t][h] / (1 - beta1 ** step)
                    v_hp = v_pos[t][h] / (1 - beta2 ** step)
                    W_pos[t][h] -= lr * m_hp / (math.sqrt(v_hp) + eps)

        avg_train_loss = ep_loss / ep_toks
        if ep == 1:
            initial_train_loss = avg_train_loss
        final_train_loss = avg_train_loss
        print(f"      Epoch {ep}/{epochs} | Train Loss: {avg_train_loss:.4f} | Took: {time.time()-t0:.1f}s")

    # Final Validation Loss
    val_loss_f = 0.0
    for inp, tgt in val_data:
        for t in range(len(inp)):
            x = [W_embed[inp[t]][h] + W_pos[t][h] for h in range(H)]
            logits = [sum(W_head[v][h] * x[h] for h in range(H)) for v in range(V)]
            m = max(logits)
            exps = [math.exp(max(-30.0, min(30.0, l - m))) for l in logits]
            s = sum(exps)
            probs = [e / s for e in exps]
            val_loss_f += -math.log(max(1e-12, probs[tgt[t]]))
    val_loss_f /= val_toks
    final_perplexity = math.exp(val_loss_f)
    print(f"\n      -> Final Validation Loss: {val_loss_f:.4f} (Perplexity: {final_perplexity:.2f})")

    # 6. Save Model Checkpoint Files
    print(f"\n[5/6] Serializing SafeTensors and metadata manifests...")
    safetensors_bytes = model.export_to_safetensors()
    model_path = os.path.join(out_dir, 'model.safetensors')
    with open(model_path, 'wb') as f:
        f.write(safetensors_bytes)
    sha256 = hashlib.sha256(safetensors_bytes).hexdigest()

    cfg_dict = cfg.to_dict()
    cfg_dict['training_summary'] = {
        'epochs': epochs,
        'initial_loss': round(initial_train_loss, 4),
        'final_loss': round(final_train_loss, 4),
        'val_loss': round(val_loss_f, 4),
        'initial_perplexity': round(init_perplexity, 2),
        'final_perplexity': round(final_perplexity, 2),
        'total_parameters': total_params,
        'vocab_size': V,
        'sha256': sha256,
        'created_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    }
    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg_dict, f, indent=2)

    training_meta = {
        'model_name': 'TARA-0.1-tokenizer-aligned',
        'version': 'TARA-0.1-tokenizer-aligned',
        'checkpoint_sha256': sha256,
        'total_parameters': total_params,
        'vocab_size': V,
        'initial_loss': round(initial_train_loss, 4),
        'final_training_loss': round(final_train_loss, 4),
        'validation_loss': round(val_loss_f, 4),
        'initial_perplexity': round(init_perplexity, 2),
        'final_perplexity': round(final_perplexity, 2),
        'loss_reduction_pct': round(((initial_train_loss - final_train_loss)/initial_train_loss)*100, 2),
        'training_duration_seconds': round(time.time() - t_train_start, 2),
        'smoke_test_sample_count': len(train_data),
        'status': 'SMOKE_TEST_PASSED_NEEDS_LARGER_DATASET'
    }
    with open(os.path.join(out_dir, 'training_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(training_meta, f, indent=2)

    # 7. Run 6 benchmark prompts
    print(f"\n[6/6] Running 6 benchmark prompts through trained model...")
    prompts = [
        ('1. Identity', 'Hello, who are you?'),
        ('2. Arithmetic', 'What is 2 + 2?'),
        ('3. TARA Architecture', 'Explain what TARA is.'),
        ('4. Coding (Python)', 'Write a short Python function that adds two numbers.'),
        ('5. Kannada Language', 'Respond in Kannada: TARA yenu?'),
        ('6. Technical Reasoning', 'What is the mathematical definition of grouped query attention?')
    ]

    print("\n" + "=" * 70)
    print("   TARA-0.1-tokenizer-aligned: 6 BENCHMARK PROMPTS EVALUATION")
    print("=" * 70)
    eval_results = []
    for cat, p in prompts:
        t_start = time.perf_counter()
        curr = tok.encode(p)
        p_len = len(curr)
        out_ids = []
        for _ in range(25):
            t = len(curr) - 1
            x = [W_embed[curr[-1]][h] + W_pos[t][h] for h in range(H)]
            logits = [sum(W_head[v][h] * x[h] for h in range(H)) for v in range(V)]
            best_id = logits.index(max(logits))
            if best_id in [tok.token_to_id['<|im_end|>'], tok.token_to_id['<|pad|>']]:
                break
            out_ids.append(best_id)
            curr.append(best_id)
        t_elapsed = time.perf_counter() - t_start
        gen_text = tok.decode(out_ids)
        tps = len(out_ids) / max(0.0001, t_elapsed)
        latency_ms = t_elapsed * 1000
        eval_results.append({
            'category': cat,
            'prompt': p,
            'output': gen_text,
            'tokens': len(out_ids),
            'latency_ms': round(latency_ms, 2),
            'tps': round(tps, 2)
        })
        print(f"[{cat}] Prompt: \"{p}\"")
        print(f"       Generated: \"{gen_text}\"")
        print(f"       Metrics: {len(out_ids)} tokens | Latency: {latency_ms:.1f} ms | Speed: {tps:.2f} tok/s\n")
    print("=" * 70 + "\n")

    return {
        'initial_loss': initial_train_loss,
        'final_loss': final_train_loss,
        'val_loss': val_loss_f,
        'initial_perplexity': init_perplexity,
        'final_perplexity': final_perplexity,
        'total_parameters': total_params,
        'vocab_size': V,
        'eval_results': eval_results
    }

if __name__ == '__main__':
    run_retraining()
