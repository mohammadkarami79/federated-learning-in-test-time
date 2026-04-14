#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stronger Baseline Comparison for MedFedPure (Table C)
======================================================
Compares MedFedPure against an expanded set of baselines including:

Test-time defense baselines:
  1. Input smoothing (Gaussian blur)
  2. Bit-depth reduction
  3. DiffPure-only (purify all without detection)
  4. MAE-detect-only  (detect but no purification)

Federated robustness baselines:
  5. FedAvg (no defense)
  6. Local Adversarial Training (each client trains with PGD)
  7. pFedDef (adversarial training in FL)
  8. FedAvg + DiffPure-only
  9. FedAvg + MAE-only

Ours:
  10. MedFedPure (full pipeline)

Usage:
  python experiments/baseline_comparison.py --dataset br35h --seed 42
  python experiments/baseline_comparison.py --dataset cifar10 --seed 42
"""

import os
import sys
import json
import copy
import random
import logging
import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.ablation_study import (
    AblationConfig, get_logger, set_seed,
    dirichlet_split,
    MoEClient, DiffusionUNet, SimpleMAE,
    train_diffusion, train_mae, train_personalized_fl,
    pgd_attack, ddpm_purify, adaptive_t,
    evaluate_clean, VARIANT_LABELS,
    make_beta_schedule,
    create_resnet18,
)
from experiments.additional_dataset import (
    PFedDefModel, train_pfeddef, train_fedavg_baseline,
)

# ══════════════════════════════════════════════════════════════════════════════
# Simple baseline defense implementations
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def gaussian_smoothing(imgs: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Gaussian blur defense (input transformation)."""
    import math
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    # Create Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32, device=imgs.device)
    x -= kernel_size // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss /= gauss.sum()
    kernel = gauss.outer(gauss).unsqueeze(0).unsqueeze(0)  # [1,1,k,k]
    kernel = kernel.expand(imgs.size(1), 1, -1, -1)       # [C,1,k,k]
    pad = kernel_size // 2
    smoothed = F.conv2d(imgs, kernel, padding=pad, groups=imgs.size(1))
    return smoothed.clamp(0, 1)


@torch.no_grad()
def bit_depth_reduction(imgs: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Bit-depth reduction defense."""
    levels = 2 ** bits
    return (imgs * levels).floor() / levels


# ══════════════════════════════════════════════════════════════════════════════
# Local adversarial training baseline
# ══════════════════════════════════════════════════════════════════════════════

def train_local_adversarial(cfg: AblationConfig, client_subsets: List[Subset],
                              ckpt_path: Path, logger: logging.Logger) -> nn.Module:
    """FedAvg where each client uses PGD adversarial training locally."""
    if ckpt_path.exists():
        logger.info(f"Loading Local-AT FedAvg from {ckpt_path}")
        model = create_resnet18(cfg.NUM_CLASSES, pretrained=False).to(cfg.DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=cfg.DEVICE))
        model.eval()
        return model

    logger.info("Training FedAvg + Local Adversarial Training...")
    global_model = create_resnet18(cfg.NUM_CLASSES, pretrained=True).to(cfg.DEVICE)

    for rnd in range(cfg.NUM_ROUNDS):
        local_states = []
        for subset in client_subsets:
            local = copy.deepcopy(global_model)
            local.train()
            loader = DataLoader(subset, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                num_workers=cfg.NUM_WORKERS,
                                pin_memory=(cfg.DEVICE == "cuda"))
            opt = optim.SGD(local.parameters(), lr=cfg.LEARNING_RATE,
                            momentum=0.9, weight_decay=5e-4)
            for _ in range(cfg.LOCAL_EPOCHS):
                for imgs, labels in loader:
                    if imgs.size(0) < 2:
                        continue
                    imgs, labels = imgs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                    adv = pgd_attack(local, imgs, labels,
                                     cfg.PGD_EPS, cfg.PGD_ALPHA,
                                     min(cfg.PGD_STEPS, 5), cfg.DEVICE)
                    loss = 0.5 * F.cross_entropy(local(imgs), labels) + \
                           0.5 * F.cross_entropy(local(adv), labels)
                    opt.zero_grad(); loss.backward(); opt.step()
            local_states.append(local.state_dict())

        avg = {k: torch.stack([s[k].float() for s in local_states]).mean(0)
               for k in local_states[0]}
        global_model.load_state_dict(avg)
        if (rnd + 1) % 5 == 0:
            logger.info(f"  Local-AT round {rnd+1}/{cfg.NUM_ROUNDS}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(global_model.state_dict(), ckpt_path)
    logger.info(f"Saved Local-AT → {ckpt_path}")
    global_model.eval()
    return global_model


# ══════════════════════════════════════════════════════════════════════════════
# Unified adversarial evaluation with any baseline defense
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_with_defense(
        model: nn.Module,
        loader: DataLoader,
        cfg: AblationConfig,
        defense_fn,       # callable(adv_imgs) -> processed_imgs  (or None)
        logger: logging.Logger,
        defense_name: str = "",
) -> Dict:
    """Generic adversarial evaluation with pluggable defense."""
    model.eval()
    device = cfg.DEVICE
    correct = total = 0
    total_latency = 0.0
    batches = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        B = imgs.size(0)

        adv = pgd_attack(model, imgs, labels,
                         cfg.PGD_EPS, cfg.PGD_ALPHA, cfg.PGD_STEPS, device)

        t0 = time.perf_counter()
        processed = defense_fn(adv) if defense_fn else adv
        total_latency += (time.perf_counter() - t0) * 1000

        with torch.no_grad():
            preds = model(processed).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += B
        batches += 1

    return {
        "adv_acc":       round(100.0 * correct / total, 2) if total else 0.0,
        "avg_latency_ms": round(total_latency / batches, 2) if batches else 0.0,
    }


def evaluate_medfedpure(
        model: nn.Module,
        loader: DataLoader,
        cfg: AblationConfig,
        mae_model: SimpleMAE,
        diffuser: DiffusionUNet,
        logger: logging.Logger,
) -> Dict:
    """Full MedFedPure evaluation (reuse ablation_study implementation)."""
    from experiments.ablation_study import evaluate_adversarial
    return evaluate_adversarial(
        model, loader, cfg, mae_model, diffuser, "full_medfedpure", logger)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline_comparison(cfg: AblationConfig, logger: logging.Logger) -> List[Dict]:
    set_seed(cfg.SEED)
    device = cfg.DEVICE

    from experiments.ablation_study import get_dataset
    logger.info("Loading dataset...")
    train_ds, test_ds = get_dataset(cfg)
    logger.info(f"  Train: {len(train_ds)}  Test: {len(test_ds)}")

    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                             num_workers=cfg.NUM_WORKERS,
                             pin_memory=(device == "cuda"))

    client_indices = dirichlet_split(train_ds, cfg.NUM_CLIENTS,
                                     cfg.DIRICHLET_ALPHA, cfg.SEED)
    client_subsets = [Subset(train_ds, idx) for idx in client_indices]

    full_loader = DataLoader(train_ds, batch_size=cfg.DIFF_BATCH, shuffle=True,
                             num_workers=cfg.NUM_WORKERS,
                             pin_memory=(device == "cuda"))

    ckpt_dir = Path("checkpoints") / f"baseline_{cfg.DATASET.lower()}_seed{cfg.SEED}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Train shared defense components ───────────────────────────────────────
    diffuser  = train_diffusion(cfg, full_loader, ckpt_dir / "diffusion.pt", logger)
    mae_model = train_mae(cfg, full_loader, ckpt_dir / "mae_detector.pt", logger)

    # ── Train models ──────────────────────────────────────────────────────────
    logger.info("Training all baseline models...")
    fedavg_model  = train_fedavg_baseline(cfg, client_subsets,
                                           ckpt_dir / "fedavg.pt", logger)
    localat_model = train_local_adversarial(cfg, client_subsets,
                                             ckpt_dir / "local_at.pt", logger)
    pfeddef_model = train_pfeddef(cfg, client_subsets,
                                   ckpt_dir / "pfeddef.pt", logger)
    moe_clients   = train_personalized_fl(cfg, client_subsets,
                                           ckpt_dir / "moe_clients.pt", logger)
    moe_model = moe_clients[0]; moe_model.eval()

    # ── Define defense functions ───────────────────────────────────────────────
    def diffpure_all(adv):
        return ddpm_purify(diffuser, adv, cfg.DIFF_T_FIXED, cfg)

    def mae_detect_only(adv):
        # Detect but DON'T purify (pass-through)
        return adv

    def smooth_defense(adv):
        return gaussian_smoothing(adv, sigma=1.0)

    def bitdepth_defense(adv):
        return bit_depth_reduction(adv, bits=5)

    # ── Evaluate all baselines ─────────────────────────────────────────────────
    results = []
    configs = [
        # (label, model, defense_fn or None, defense_type, notes)
        ("FedAvg",                 fedavg_model,  None,             "No defense",       ""),
        ("FedAvg + Smoothing",     fedavg_model,  smooth_defense,   "Input transform",  "σ=1.0"),
        ("FedAvg + BitDepth",      fedavg_model,  bitdepth_defense, "Input transform",  "5-bit"),
        ("FedAvg + DiffPure-only", fedavg_model,  diffpure_all,     "Purification",     "all samples"),
        ("FedAvg + MAE-only",      fedavg_model,  mae_detect_only,  "Detection only",   "no purify"),
        ("Local Adv. Training",    localat_model, None,             "AT",               ""),
        ("pFedDef",                pfeddef_model, None,             "FL + AT",          "Kim2023"),
        ("MedFedPure (ours)",      moe_model,     None,             "Full pipeline",    "MAE+Diff+MoE"),
    ]

    for label, model, defense_fn, def_type, notes in configs:
        logger.info(f"\n[Baseline] {label}")
        clean_acc = evaluate_clean(model, test_loader, device)
        logger.info(f"  Clean Acc: {clean_acc:.2f}%")

        if label == "MedFedPure (ours)":
            adv_metrics = evaluate_medfedpure(
                model, test_loader, cfg, mae_model, diffuser, logger)
            adv_acc = adv_metrics["adv_acc"]
            latency = adv_metrics["avg_latency_ms"]
        else:
            adv_metrics = evaluate_with_defense(
                model, test_loader, cfg, defense_fn, logger, label)
            adv_acc = adv_metrics["adv_acc"]
            latency = adv_metrics["avg_latency_ms"]

        logger.info(f"  Adv Acc: {adv_acc:.2f}%  Latency: {latency:.1f}ms/batch")

        results.append({
            "dataset":      cfg.DATASET.upper(),
            "seed":         cfg.SEED,
            "method":       label,
            "defense_type": def_type,
            "clean_acc":    round(clean_acc, 2),
            "adv_acc":      round(adv_acc, 2),
            "latency_ms":   round(latency, 2),
            "notes":        notes,
        })

    # Compute relative gain over FedAvg (no defense)
    fedavg_adv = next(r["adv_acc"] for r in results if r["method"] == "FedAvg")
    for r in results:
        r["rel_gain_over_fedavg"] = round(r["adv_acc"] - fedavg_adv, 2)

    return results


def save_baseline_results(results, out_dir: Path, dataset: str, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{dataset.lower()}_seed{seed}"
    json_path = out_dir / f"baseline_comparison_{tag}.json"
    csv_path  = out_dir / f"baseline_comparison_{tag}.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    fields = ["method", "defense_type", "clean_acc", "adv_acc",
              "rel_gain_over_fedavg", "latency_ms", "notes"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in results:
            w.writerow({k: row[k] for k in fields})

    return json_path, csv_path


def parse_args():
    p = argparse.ArgumentParser(description="MedFedPure Baseline Comparison")
    p.add_argument("--dataset",  type=str, default="br35h",
                   choices=["br35h", "cifar10"])
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--rounds",  type=int, default=None)
    p.add_argument("--epochs",  type=int, default=None)
    p.add_argument("--clients", type=int, default=None)
    p.add_argument("--out-dir", type=str,
                   default="experiment_results/baseline_comparison")
    p.add_argument("--fast", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = AblationConfig()
    cfg.DATASET   = args.dataset
    cfg.DATA_ROOT = args.data_root
    cfg.SEED      = args.seed
    if args.rounds:  cfg.NUM_ROUNDS   = args.rounds
    if args.epochs:  cfg.LOCAL_EPOCHS = args.epochs
    if args.clients: cfg.NUM_CLIENTS  = args.clients
    if args.fast:
        cfg.NUM_ROUNDS   = 3
        cfg.LOCAL_EPOCHS = 2
        cfg.DIFF_EPOCHS  = 5
        cfg.MAE_EPOCHS   = 5
        cfg.NUM_CLIENTS  = 3
    cfg.finalize()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"baseline_{cfg.DATASET.lower()}_seed{cfg.SEED}.log"
    logger = get_logger("baseline_comparison", str(log_file))

    logger.info("=" * 60)
    logger.info(f"MedFedPure – Baseline Comparison on {cfg.DATASET.upper()}")
    logger.info("=" * 60)

    results = run_baseline_comparison(cfg, logger)

    json_p, csv_p = save_baseline_results(results, out_dir, cfg.DATASET, cfg.SEED)
    logger.info(f"\nResults saved: {json_p}  |  {csv_p}")

    # Print table
    logger.info("\n{:<32} {:>10} {:>10} {:>10} {:>10}".format(
        "Method", "Clean%", "Adv%", "Gain", "ms/B"))
    logger.info("-" * 74)
    for r in results:
        logger.info("{:<32} {:>10.2f} {:>10.2f} {:>+10.2f} {:>10.1f}".format(
            r["method"], r["clean_acc"], r["adv_acc"],
            r["rel_gain_over_fedavg"], r["latency_ms"]))


if __name__ == "__main__":
    main()
