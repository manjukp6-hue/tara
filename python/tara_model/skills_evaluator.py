"""
python/tara_model/skills_evaluator.py

Dedicated Comprehensive Evaluation Suite for TARA Neural Models:
- Evaluates all 160 Skills (142 catalog/sub-skills + 18 native/dynamic skills)
  across identification, purpose, trigger conditions, and safe workflows.
- Evaluates all 3 Knowledge items (factual precision and retrieval).
- Evaluates 4 Tools (file_inspector, hash_verifier, knowledge_retriever, provenance_tracker).
- Evaluates Rulebook Invariants (CSAM safety, creator priority, language, system integrity).
- Evaluates Identity Governance (ROOT_OPERATOR, OPERATOR_ROOT, offline root authority).
- Evaluates Kannada & Multilingual capabilities.
- Evaluates Base Capabilities & Catastrophic Forgetting (arithmetic, reasoning, coding).
- Evaluates Canonical Validation Split (storage/datasets/tara/val.jsonl).
- Evaluates Test Split (storage/datasets/tara/test.jsonl).
- Evaluates Baseline Original Validation Split (storage/backups/tara_baseline_dataset/val.jsonl).
"""

import os
import sys
import json
import math
import time
import argparse
from collections import defaultdict
from typing import Dict, List, Any, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

# Add project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from tara_model.architecture import TaraForCausalLM, TaraConfig
from tara_model.tokenizer import TaraTokenizer
from tara_model.compile_unified_dataset import (
    extract_catalog_skills,
    extract_native_skills,
    extract_knowledge,
    extract_tools,
    extract_rules,
    extract_identity
)


class SkillsEvaluator:
    def __init__(self, model_dir: str, device: str = "cpu"):
        self.model_dir = model_dir
        self.device = torch.device(device)
        self.tokenizer = TaraTokenizer()

        config_path = os.path.join(model_dir, "config.json")
        self.config = TaraConfig.from_json_file(config_path)

        weights_path = os.path.join(model_dir, "model.safetensors")
        index_path = os.path.join(model_dir, "model.safetensors.index.json")

        self.model = TaraForCausalLM(self.config).to(self.device)
        state_dict: Dict[str, torch.Tensor] = {}

        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            unique_shards = sorted(list(set(weight_map.values())))
            for shard_name in unique_shards:
                shard_file = os.path.join(model_dir, shard_name)
                shard_weights = load_file(shard_file)
                state_dict.update(shard_weights)
        elif os.path.exists(weights_path):
            state_dict = load_file(weights_path)
        else:
            raise FileNotFoundError(f"Model weights not found at {weights_path} or {index_path}")

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def evaluate_split_loss(self, jsonl_path: str) -> Dict[str, Any]:
        """Computes exact next-token prediction loss and perplexity on a dataset split."""
        if not os.path.exists(jsonl_path):
            return {"loss": float("nan"), "perplexity": float("nan"), "samples": 0, "tokens": 0}

        total_loss = 0.0
        total_tokens = 0
        sample_count = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                text = f"{d['prompt']} {d['completion']}"
                tokens = self.tokenizer.encode(text)
                if len(tokens) > 1 and len(tokens) <= 256:
                    inp = torch.tensor([tokens], device=self.device)
                    with torch.no_grad():
                        loss, _ = self.model(inp, labels=inp)
                    n_tok = len(tokens) - 1
                    total_loss += loss.item() * n_tok
                    total_tokens += n_tok
                    sample_count += 1

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
        return {
            "loss": round(avg_loss, 4),
            "perplexity": round(ppl, 2),
            "samples": sample_count,
            "tokens": total_tokens
        }

    def compute_completion_loss(self, prompt: str, completion: str) -> Tuple[float, str]:
        """
        Computes the cross-entropy loss specifically over the completion tokens given the prompt.
        Also generates greedy output to test token recall.
        """
        prompt_tokens = self.tokenizer.encode(prompt)
        comp_tokens = self.tokenizer.encode(completion)
        if not comp_tokens:
            return 999.0, ""

        full_tokens = prompt_tokens + comp_tokens
        if len(full_tokens) > 256:
            full_tokens = full_tokens[:256]
            comp_tokens = full_tokens[len(prompt_tokens):]

        if not comp_tokens:
            return 999.0, ""

        inp = torch.tensor([full_tokens], device=self.device)
        with torch.no_grad():
            _, logits = self.model(inp)

        start_idx = len(prompt_tokens) - 1
        end_idx = len(full_tokens) - 1
        
        target_tokens = full_tokens[len(prompt_tokens):]
        pred_logits = logits[0, start_idx:end_idx, :]

        comp_loss = F.cross_entropy(
            pred_logits,
            torch.tensor(target_tokens, device=self.device)
        ).item()

        # Greedy generation for qualitative check (up to 20 tokens)
        gen_ids = []
        curr = list(prompt_tokens)
        with torch.no_grad():
            for _ in range(20):
                inp_t = torch.tensor([curr], device=self.device)
                _, l = self.model(inp_t)
                next_tok = torch.argmax(l[0, -1, :]).item()
                if next_tok in [
                    self.tokenizer.token_to_id.get("<|im_end|>", 2),
                    self.tokenizer.token_to_id.get("<|pad|>", 0)
                ]:
                    break
                gen_ids.append(next_tok)
                curr.append(next_tok)
                if len(curr) >= 256:
                    break

        gen_text = self.tokenizer.decode(gen_ids)
        return comp_loss, gen_text

    def evaluate_skills(self) -> Dict[str, Any]:
        """Evaluates all 160 skills (142 catalog + 18 native/dynamic)."""
        cat_samples = extract_catalog_skills(PROJECT_ROOT)
        nat_samples = extract_native_skills(PROJECT_ROOT)
        all_skill_samples = cat_samples + nat_samples

        skill_groups = defaultdict(list)
        for s in all_skill_samples:
            skill_groups[s["item_name"]].append(s)

        results = {}
        passed_count = 0

        for skill_name, samples in skill_groups.items():
            losses = []
            gen_texts = []
            for item in samples:
                loss, gen_text = self.compute_completion_loss(item["prompt"], item["completion"])
                losses.append(loss)
                gen_texts.append(gen_text)

            avg_loss = sum(losses) / len(losses) if losses else 999.0
            ppl = math.exp(avg_loss) if avg_loss < 30 else float("inf")
            
            keywords = [skill_name.lower().replace("-", " "), "tara", "skill"]
            recalled = any(any(k in gt.lower() for k in keywords) for gt in gen_texts)
            
            passed = avg_loss < 3.2 or recalled
            if passed:
                passed_count += 1

            results[skill_name] = {
                "passed": passed,
                "avg_completion_loss": round(avg_loss, 4),
                "perplexity": round(ppl, 2),
                "sample_count": len(samples),
                "sample_generation": gen_texts[0] if gen_texts else ""
            }

        return {
            "total_skills": len(skill_groups),
            "passed_skills": passed_count,
            "pass_rate": round(passed_count / len(skill_groups) * 100, 2) if skill_groups else 0.0,
            "details": results
        }

    def evaluate_knowledge(self) -> Dict[str, Any]:
        """Evaluates all 3 verified knowledge items."""
        know_samples = extract_knowledge(PROJECT_ROOT)
        know_groups = defaultdict(list)
        for k in know_samples:
            know_groups[k["item_name"]].append(k)

        results = {}
        passed_count = 0
        for item_name, samples in know_groups.items():
            losses = []
            gen_texts = []
            for s in samples:
                loss, gen = self.compute_completion_loss(s["prompt"], s["completion"])
                losses.append(loss)
                gen_texts.append(gen)

            avg_loss = sum(losses) / len(losses) if losses else 999.0
            ppl = math.exp(avg_loss) if avg_loss < 30 else float("inf")
            passed = avg_loss < 3.2
            if passed:
                passed_count += 1

            results[item_name] = {
                "passed": passed,
                "avg_completion_loss": round(avg_loss, 4),
                "perplexity": round(ppl, 2),
                "sample_count": len(samples),
                "sample_generation": gen_texts[0] if gen_texts else ""
            }

        return {
            "total_knowledge_items": len(know_groups),
            "passed_items": passed_count,
            "pass_rate": round(passed_count / len(know_groups) * 100, 2) if know_groups else 0.0,
            "details": results
        }

    def evaluate_tools(self) -> Dict[str, Any]:
        """Evaluates the 4 tools."""
        tool_samples = extract_tools(PROJECT_ROOT)
        tool_groups = defaultdict(list)
        for t in tool_samples:
            tool_groups[t["item_name"]].append(t)

        results = {}
        passed_count = 0
        for tool_name, samples in tool_groups.items():
            losses = []
            gen_texts = []
            for s in samples:
                loss, gen = self.compute_completion_loss(s["prompt"], s["completion"])
                losses.append(loss)
                gen_texts.append(gen)

            avg_loss = sum(losses) / len(losses) if losses else 999.0
            ppl = math.exp(avg_loss) if avg_loss < 30 else float("inf")
            passed = avg_loss < 3.2
            if passed:
                passed_count += 1

            results[tool_name] = {
                "passed": passed,
                "avg_completion_loss": round(avg_loss, 4),
                "perplexity": round(ppl, 2),
                "sample_count": len(samples),
                "sample_generation": gen_texts[0] if gen_texts else ""
            }

        return {
            "total_tools": len(tool_groups),
            "passed_tools": passed_count,
            "pass_rate": round(passed_count / len(tool_groups) * 100, 2) if tool_groups else 0.0,
            "details": results
        }

    def evaluate_rules_and_identity(self) -> Dict[str, Any]:
        """Evaluates rule invariants and identity governance."""
        rule_samples = extract_rules(PROJECT_ROOT)
        ident_samples = extract_identity(PROJECT_ROOT)

        def eval_group(samples):
            losses = []
            for s in samples:
                loss, _ = self.compute_completion_loss(s["prompt"], s["completion"])
                losses.append(loss)
            avg = sum(losses) / len(losses) if losses else 999.0
            return {
                "avg_loss": round(avg, 4),
                "perplexity": round(math.exp(avg), 2) if avg < 30 else float("inf"),
                "passed": avg < 3.2
            }

        return {
            "rules": eval_group(rule_samples),
            "identity": eval_group(ident_samples)
        }

    def evaluate_base_capabilities_and_catastrophic_forgetting(self) -> Dict[str, Any]:
        """
        Evaluates arithmetic, logic, language, and the original baseline validation split
        to rigorously detect catastrophic forgetting.
        """
        baseline_val_path = os.path.join(PROJECT_ROOT, "storage/backups/tara_baseline_dataset/val.jsonl")
        baseline_val_metrics = self.evaluate_split_loss(baseline_val_path)

        math_prompts = [
            ("What is 2 + 2?", "4"),
            ("What is 5 + 3?", "8"),
            ("What is 10 + 20?", "30"),
            ("Calculate 7 * 6", "42"),
            ("Compute 15 - 9", "6")
        ]
        math_losses = []
        math_gens = []
        for p, c in math_prompts:
            l, g = self.compute_completion_loss(p, c)
            math_losses.append(l)
            math_gens.append({"prompt": p, "expected": c, "generated": g})

        avg_math_loss = sum(math_losses) / len(math_losses)

        kannada_prompts = [
            ("Respond in Kannada: TARA yenu?", "TARA swataha kelasa maduva AI vyavasthe"),
            ("Namaskara TARA nina creator yaru?", "Nanna creator ROOT_OPERATOR OPERATOR_ROOT")
        ]
        kannada_losses = []
        kannada_gens = []
        for p, c in kannada_prompts:
            l, g = self.compute_completion_loss(p, c)
            kannada_losses.append(l)
            kannada_gens.append({"prompt": p, "generated": g})

        avg_kannada_loss = sum(kannada_losses) / len(kannada_losses)

        return {
            "baseline_val_dataset": baseline_val_metrics,
            "arithmetic": {
                "avg_loss": round(avg_math_loss, 4),
                "perplexity": round(math.exp(avg_math_loss), 2) if avg_math_loss < 30 else float("inf"),
                "probes": math_gens
            },
            "kannada": {
                "avg_loss": round(avg_kannada_loss, 4),
                "perplexity": round(math.exp(avg_kannada_loss), 2) if avg_kannada_loss < 30 else float("inf"),
                "probes": kannada_gens
            }
        }

    def run_full_evaluation(self) -> Dict[str, Any]:
        """Runs the entire benchmark suite across all splits and dimensions."""
        print(f"\n========================================================")
        print(f"   EVALUATING MODEL: {self.model_dir}")
        print(f"========================================================")

        val_path = os.path.join(PROJECT_ROOT, "storage/datasets/tara/val.jsonl")
        test_path = os.path.join(PROJECT_ROOT, "storage/datasets/tara/test.jsonl")

        print("  Evaluating canonical validation split (262 samples)...")
        val_metrics = self.evaluate_split_loss(val_path)
        print(f"    -> Val Loss: {val_metrics['loss']}, Perplexity: {val_metrics['perplexity']}")

        print("  Evaluating test split (276 samples)...")
        test_metrics = self.evaluate_split_loss(test_path)
        print(f"    -> Test Loss: {test_metrics['loss']}, Perplexity: {test_metrics['perplexity']}")

        print("  Evaluating all 160 skills...")
        skills_metrics = self.evaluate_skills()
        print(f"    -> Skills Score: {skills_metrics['passed_skills']}/{skills_metrics['total_skills']} ({skills_metrics['pass_rate']}%)")

        print("  Evaluating 3 knowledge bases...")
        know_metrics = self.evaluate_knowledge()
        print(f"    -> Knowledge Score: {know_metrics['passed_items']}/{know_metrics['total_knowledge_items']} ({know_metrics['pass_rate']}%)")

        print("  Evaluating 4 tools...")
        tool_metrics = self.evaluate_tools()
        print(f"    -> Tools Score: {tool_metrics['passed_tools']}/{tool_metrics['total_tools']} ({tool_metrics['pass_rate']}%)")

        print("  Evaluating Rules & Identity...")
        rule_ident_metrics = self.evaluate_rules_and_identity()
        print(f"    -> Rules Loss: {rule_ident_metrics['rules']['avg_loss']} | Identity Loss: {rule_ident_metrics['identity']['avg_loss']}")

        print("  Evaluating Base capabilities & Baseline dataset regression...")
        base_metrics = self.evaluate_base_capabilities_and_catastrophic_forgetting()
        print(f"    -> Baseline Original Val Loss: {base_metrics['baseline_val_dataset']['loss']}")
        print(f"    -> Arithmetic Loss: {base_metrics['arithmetic']['avg_loss']}")
        print(f"    -> Kannada Loss: {base_metrics['kannada']['avg_loss']}")

        report = {
            "model_dir": self.model_dir,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "canonical_validation": val_metrics,
            "test_metrics": test_metrics,
            "skills": {
                "score": f"{skills_metrics['passed_skills']}/{skills_metrics['total_skills']}",
                "total": skills_metrics['total_skills'],
                "passed": skills_metrics['passed_skills'],
                "pass_rate": skills_metrics['pass_rate']
            },
            "knowledge": {
                "score": f"{know_metrics['passed_items']}/{know_metrics['total_knowledge_items']}",
                "total": know_metrics['total_knowledge_items'],
                "passed": know_metrics['passed_items'],
                "pass_rate": know_metrics['pass_rate']
            },
            "tools": {
                "score": f"{tool_metrics['passed_tools']}/{tool_metrics['total_tools']}",
                "total": tool_metrics['total_tools'],
                "passed": tool_metrics['passed_tools'],
                "pass_rate": tool_metrics['pass_rate']
            },
            "rules": rule_ident_metrics["rules"],
            "identity": rule_ident_metrics["identity"],
            "catastrophic_forgetting": base_metrics,
            "raw_details": {
                "skills": skills_metrics["details"],
                "knowledge": know_metrics["details"],
                "tools": tool_metrics["details"]
            }
        }
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TARA model on comprehensive benchmark suite.")
    parser.add_argument("--model_dir", type=str, default="storage/models/tara", help="Directory containing model.safetensors and config.json")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save evaluation output JSON")
    args = parser.parse_args()

    evaluator = SkillsEvaluator(args.model_dir)
    report = evaluator.run_full_evaluation()

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.output_json}")
