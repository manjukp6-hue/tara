#!/usr/bin/env python3
"""
scripts/server_continuous_evolution.py

TARA Continuous Model Evolution Pipeline
Coordinates the ongoing deployment-independent model lifecycle:
Observe -> Research -> Learn -> Train -> Evaluate -> Compress -> Version -> Deploy

Runs on any authorized TARA host/runtime (user device, desktop, private server, or cloud host).
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

# Local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python.tara_model.train import train_tara_model
from python.tara_model.evaluator import ModelEvaluator

def run_server_evolution_cycle(target_version="TARA-0.2", epochs=5, repo_id=None, token=None):
    print("=" * 65)
    print(f"      STARTING TARA CONTINUOUS EVOLUTION: {target_version}")
    print("=" * 65 + "\n")

    output_dir = f"storage/models/{target_version.lower()}"

    # Step 1: Execute server-side training on updated domain dataset
    print(f"[1/4] Training next-generation model weights ({target_version})...")
    train_res = train_tara_model(output_dir=output_dir, epochs=epochs, lr=0.008)

    # Step 2: Run 7-dimension quality evaluation benchmark
    print(f"\n[2/4] Running rigorous benchmark evaluation...")
    evaluator = ModelEvaluator(model_dir=output_dir)
    eval_results = evaluator.run_benchmark_suite()

    # Step 3: Enforce strict regression gate
    print(f"\n[3/4] Enforcing Quality & Safety Acceptance Gate...")
    if eval_results["decision"] != "ACCEPT_FOR_PRODUCTION":
        print(f"  [REJECTED] Model failed validation gate: {eval_results['decision']}. Aborting release.")
        return False

    print(f"  [ACCEPTED] Model passed all benchmarks (Score: {eval_results['composite_score'] * 100:.2f}%)")

    # Step 4: Optional cloud push to Hugging Face
    if token and repo_id:
        print(f"\n[4/4] Deploying {target_version} to Hugging Face Hub ({repo_id})...")
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            api.upload_folder(
                folder_path=output_dir,
                repo_id=repo_id,
                path_in_repo=f"versions/{target_version}",
                commit_message=f"Deploy verified TARA AI model version {target_version}"
            )
            print(f"  [✓] Successfully deployed to https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"  Hugging Face upload skipped or error: {e}")
    else:
        print("\n[4/4] Local cloud build verified. Pass --token and --repo-id to publish live.")

    print("\n====================================================")
    print(f" [COMPLETE] {target_version} EVOLUTION CYCLE SUCCESSFUL!")
    print("====================================================\n")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TARA Server-Side Continuous Evolution Pipeline")
    parser.add_argument("--version", default="TARA-0.2", help="Target model version tag")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--repo-id", default=os.getenv("HF_REPO_ID"), help="Hugging Face Repository ID")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"), help="Hugging Face API Token")
    args = parser.parse_args()

    run_server_evolution_cycle(target_version=args.version, epochs=args.epochs, repo_id=args.repo_id, token=args.token)
