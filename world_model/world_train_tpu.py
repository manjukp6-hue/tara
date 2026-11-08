#!/usr/bin/env python3
"""
Dedicated Standalone Neural World Model TPU Training Pipeline.

Optimized for Google Cloud TPU / Kaggle TPU v2/v3/v4 pods using PyTorch/XLA.
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# PyTorch / XLA integration for TPU accelerators
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

try:
    from safetensors.torch import save_file as save_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

# Import model definition from shared GPU module
from world_train_gpu import NeuralWorldModel, StateTransitionDataset


def train_world_model_tpu(args):
    if HAS_XLA:
        device = xm.xla_device()
        print(f"[World Model TPU Training] Connected to Google Cloud TPU Device: {device}")
    else:
        device = torch.device("cpu")
        print("[World Model TPU Training] PyTorch/XLA not detected; falling back to CPU mode.")

    model = NeuralWorldModel(
        latent_dim=args.latent_dim,
        action_dim=args.action_dim,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        head_dim=args.head_dim,
        learning_rate=args.lr,
    ).to(device)

    dataset = StateTransitionDataset(args.data_path, latent_dim=args.latent_dim, action_dim=args.action_dim)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    lambda_dyn = args.lambda_dyn
    lambda_rew = args.lambda_rew
    lambda_fail = args.lambda_fail
    lambda_val = args.lambda_val

    print(f"[World Model TPU] Starting {args.epochs} epochs with multi-objective supervision...")

    best_loss = float("inf")
    best_bm_dir = os.path.join(args.output_dir, "best_milestone")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            s = batch["state"].to(device)
            a = batch["action"].to(device)
            ns = batch["next_state"].to(device)
            r = batch["reward"].to(device)
            failed = batch["failed"].to(device)
            ret = batch["return"].to(device)

            pred_ns, pred_r, pred_fail, pred_val = model(s, a)

            loss = (
                lambda_dyn * F.mse_loss(pred_ns, ns)
                + lambda_rew * F.mse_loss(pred_r, r)
                + lambda_fail * F.binary_cross_entropy(pred_fail, failed)
                + lambda_val * F.mse_loss(pred_val, ret)
            )

            optimizer.zero_grad()
            loss.backward()

            if HAS_XLA:
                xm.optimizer_step(optimizer)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()

        n = max(len(dataloader), 1)
        avg_epoch_loss = total_loss / n
        print(f"  [TPU Epoch {epoch+1}/{args.epochs}] Total Multi-Loss: {avg_epoch_loss:.4f}")

        if HAS_XLA:
            xm.mark_step()

        # Best Milestone Checkpoint Tracking
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            os.makedirs(best_bm_dir, exist_ok=True)
            model.save_checkpoint(best_bm_dir)
            meta = {"epoch": epoch + 1, "loss": avg_epoch_loss, "best_loss": best_loss, "status": "BEST_MILESTONE"}
            with open(os.path.join(best_bm_dir, "STAGE_METADATA.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(f"[World Model TPU] >>> Committing Best Milestone Checkpoint: {best_loss:.4f} <<<", flush=True)

        # Regular Checkpoint Save with Loss Spike Guard
        if best_loss < float("inf") and avg_epoch_loss > (best_loss * 1.5):
            print(
                f"[World Model TPU Loss Spike Guard] Epoch {epoch+1}: Loss {avg_epoch_loss:.4f} > 1.5x best loss ({best_loss:.4f}). "
                f"Skipping regular checkpoint save to protect normal backups from divergence pollution!",
                flush=True,
            )
        else:
            model.save_checkpoint(args.output_dir)
            meta = {"epoch": epoch + 1, "loss": avg_epoch_loss, "best_loss": best_loss, "status": "NORMAL_CHECKPOINT"}
            with open(os.path.join(args.output_dir, "STAGE_METADATA.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(f"[World Model TPU] Checkpoint persisted to: {args.output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="World Model TPU Trainer")
    parser.add_argument("--data-path", type=str, default="storage/datasets/world_model")
    parser.add_argument("--output-dir", type=str, default="storage/models/tara_world_model")
    parser.add_argument("--latent-dim", type=int, default=2048)
    parser.add_argument("--action-dim", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=5632)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--lambda-dyn", type=float, default=2.0)
    parser.add_argument("--lambda-rew", type=float, default=1.0)
    parser.add_argument("--lambda-fail", type=float, default=1.0)
    parser.add_argument("--lambda-val", type=float, default=0.5)
    args = parser.parse_args()

    train_world_model_tpu(args)


if __name__ == "__main__":
    main()
