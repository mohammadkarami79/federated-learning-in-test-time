#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Efficiency / Overhead Analysis for MedFedPure (Table D)
========================================================
Measures inference-time cost of each component:

  1. Classifier only  (baseline)
  2. MAE detection only
  3. DiffPure only (fixed steps)
  4. MAE + DiffPure (selective – our default)
  5. Full MedFedPure pipeline

Reported metrics per method:
  - Avg inference time per sample (ms)
  - Avg inference time per batch  (ms)
  - Fraction of samples sent to purification
  - Peak GPU memory (MB)
  - Approximate overhead vs classifier-only

Usage:
  python experiments/efficiency_analysis.py --dataset br35h --seed 42
"""

import os
import sys
import json
import copy
import time
import logging
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.ablation_study import (
    AblationConfig, get_logger, set_seed,
    get_dataset, dirichlet_split,
    MoEClient, DiffusionUNet, SimpleMAE,
    train_diffusion, train_mae, train_personalized_fl,
    ddpm_purify, adaptive_t, pgd_attack,
    VARIANT_LABELS,
)

# ══════════════════════════════════════════════════════════════════════════════
# Timing helpers
# ══════════════════════════════════════════════════════════════════════════════

def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    def __init__(self):
        self.elapsed = 0.0
        self._start = None

    def start(self):
        sync_cuda()
        self._start = time.perf_counter()

    def stop(self):
        sync_cuda()
        self.elapsed += time.perf_counter() - self._start

    def reset(self):
        self.elapsed = 0.0


def gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def reset_gpu_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ══════════════════════════════════════════════════════════════════════════════
# Efficiency benchmark
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def benchmark_classifier_only(model: nn.Module, loader: DataLoader,
                               cfg: AblationConfig) -> Dict:
    """Measure pure classifier inference time."""
    model.eval()
    device = cfg.DEVICE
    timer = Timer()
    total_samples = 0

    reset_gpu_stats()
    for imgs, _ in loader:
        imgs = imgs.to(device)
        B = imgs.size(0)
        timer.start()
        _ = model(imgs)
        timer.stop()
        total_samples += B

    return {
        "total_time_s": timer.elapsed,
        "samples": total_samples,
        "ms_per_sample": 1000 * timer.elapsed / total_samples,
        "gpu_peak_mb": gpu_memory_mb(),
        "frac_purified": 0.0,
    }


def benchmark_mae_only(model: nn.Module, mae: SimpleMAE,
                       loader: DataLoader, cfg: AblationConfig) -> Dict:
    """MAE detection + classifier (no purification)."""
    model.eval(); mae.eval()
    device = cfg.DEVICE
    timer_detect = Timer(); timer_classify = Timer()
    total_samples = 0; flagged = 0

    reset_gpu_stats()
    for imgs, _ in loader:
        imgs = imgs.to(device)
        B = imgs.size(0)

        # Simulate adversarial input
        with torch.enable_grad():
            adv = pgd_attack(model, imgs,
                             torch.zeros(B, dtype=torch.long, device=device),
                             cfg.PGD_EPS, cfg.PGD_ALPHA, 3, device)

        timer_detect.start()
        errs = mae.reconstruction_error(adv)
        thresh = float(torch.quantile(errs, 1 - cfg.MAE_KAPPA / 100.0))
        flags = errs > thresh
        timer_detect.stop()

        timer_classify.start()
        with torch.no_grad():
            _ = model(adv)
        timer_classify.stop()

        total_samples += B
        flagged += flags.sum().item()

    total = timer_detect.elapsed + timer_classify.elapsed
    return {
        "total_time_s": total,
        "samples": total_samples,
        "ms_per_sample": 1000 * total / total_samples,
        "ms_detect_per_sample": 1000 * timer_detect.elapsed / total_samples,
        "ms_classify_per_sample": 1000 * timer_classify.elapsed / total_samples,
        "gpu_peak_mb": gpu_memory_mb(),
        "frac_purified": flagged / total_samples,
    }


def benchmark_diffpure_only(model: nn.Module, diffuser: DiffusionUNet,
                             loader: DataLoader, cfg: AblationConfig) -> Dict:
    """Purify all + classify (no detection)."""
    model.eval(); diffuser.eval()
    device = cfg.DEVICE
    timer_purify = Timer(); timer_classify = Timer()
    total_samples = 0

    reset_gpu_stats()
    for imgs, _ in loader:
        imgs = imgs.to(device)
        B = imgs.size(0)

        with torch.enable_grad():
            adv = pgd_attack(model, imgs,
                             torch.zeros(B, dtype=torch.long, device=device),
                             cfg.PGD_EPS, cfg.PGD_ALPHA, 3, device)

        timer_purify.start()
        purified = ddpm_purify(diffuser, adv, cfg.DIFF_T_FIXED, cfg)
        timer_purify.stop()

        timer_classify.start()
        with torch.no_grad():
            _ = model(purified)
        timer_classify.stop()

        total_samples += B

    total = timer_purify.elapsed + timer_classify.elapsed
    return {
        "total_time_s": total,
        "samples": total_samples,
        "ms_per_sample": 1000 * total / total_samples,
        "ms_purify_per_sample": 1000 * timer_purify.elapsed / total_samples,
        "ms_classify_per_sample": 1000 * timer_classify.elapsed / total_samples,
        "gpu_peak_mb": gpu_memory_mb(),
        "frac_purified": 1.0,
    }


def benchmark_full_medfedpure(model: nn.Module, mae: SimpleMAE,
                               diffuser: DiffusionUNet,
                               loader: DataLoader, cfg: AblationConfig) -> Dict:
    """Full MedFedPure: MAE detect → adaptive diffusion → classify."""
    model.eval(); mae.eval(); diffuser.eval()
    device = cfg.DEVICE
    timer_detect = Timer(); timer_purify = Timer(); timer_classify = Timer()
    total_samples = 0; flagged = 0

    reset_gpu_stats()
    for imgs, _ in loader:
        imgs = imgs.to(device)
        B = imgs.size(0)

        with torch.enable_grad():
            adv = pgd_attack(model, imgs,
                             torch.zeros(B, dtype=torch.long, device=device),
                             cfg.PGD_EPS, cfg.PGD_ALPHA, 3, device)

        timer_detect.start()
        errs = mae.reconstruction_error(adv)
        thresh = float(torch.quantile(errs, 1 - cfg.MAE_KAPPA / 100.0))
        flags = errs > thresh
        timer_detect.stop()

        processed = adv.clone()
        if flags.any():
            timer_purify.start()
            t_vals = adaptive_t(errs[flags], cfg)
            for flag_idx, t_val in zip(flags.nonzero(as_tuple=True)[0], t_vals):
                single = adv[flag_idx].unsqueeze(0)
                processed[flag_idx] = ddpm_purify(
                    diffuser, single, t_val.item(), cfg).squeeze(0)
            timer_purify.stop()

        timer_classify.start()
        with torch.no_grad():
            _ = model(processed)
        timer_classify.stop()

        total_samples += B
        flagged += flags.sum().item()

    total = timer_detect.elapsed + timer_purify.elapsed + timer_classify.elapsed
    return {
        "total_time_s": total,
        "samples": total_samples,
        "ms_per_sample": 1000 * total / total_samples,
        "ms_detect_per_sample": 1000 * timer_detect.elapsed / total_samples,
        "ms_purify_per_sample": 1000 * timer_purify.elapsed / total_samples,
        "ms_classify_per_sample": 1000 * timer_classify.elapsed / total_samples,
        "gpu_peak_mb": gpu_memory_mb(),
        "frac_purified": flagged / total_samples,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

BENCHMARK_CONFIGS = [
    "classifier_only",
    "mae_only",
    "diffpure_only",
    "full_medfedpure",
]

BENCHMARK_LABELS = {
    "classifier_only": "Classifier Only",
    "mae_only":        "MAE Detection + Classifier",
    "diffpure_only":   "DiffPure-All + Classifier",
    "full_medfedpure": "Full MedFedPure (ours)",
}


def run_efficiency_analysis(cfg: AblationConfig,
                             logger: logging.Logger) -> List[Dict]:
    set_seed(cfg.SEED)
    device = cfg.DEVICE

    logger.info("Loading dataset...")
    train_ds, test_ds = get_dataset(cfg)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                             num_workers=cfg.NUM_WORKERS,
                             pin_memory=(device == "cuda"))
    full_loader = DataLoader(train_ds, batch_size=cfg.DIFF_BATCH, shuffle=True,
                             num_workers=cfg.NUM_WORKERS,
                             pin_memory=(device == "cuda"))

    client_indices = dirichlet_split(train_ds, cfg.NUM_CLIENTS,
                                     cfg.DIRICHLET_ALPHA, cfg.SEED)
    from torch.utils.data import Subset
    client_subsets = [Subset(train_ds, idx) for idx in client_indices]

    ckpt_dir = Path("checkpoints") / f"eff_{cfg.DATASET.lower()}_seed{cfg.SEED}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    diffuser  = train_diffusion(cfg, full_loader, ckpt_dir / "diffusion.pt", logger)
    mae_model = train_mae(cfg, full_loader, ckpt_dir / "mae_detector.pt", logger)
    moe_clients = train_personalized_fl(cfg, client_subsets,
                                         ckpt_dir / "moe_clients.pt", logger)
    model = moe_clients[0]; model.eval()

    logger.info("\nRunning efficiency benchmarks (N_batches on test set)...")
    results = []

    # Classifier only
    logger.info("[1/4] Classifier only...")
    m = benchmark_classifier_only(model, test_loader, cfg)
    baseline_ms = m["ms_per_sample"]
    results.append({"method": "classifier_only", "label": BENCHMARK_LABELS["classifier_only"], **m,
                     "overhead_vs_baseline": 0.0})

    # MAE only
    logger.info("[2/4] MAE detect + classifier...")
    m = benchmark_mae_only(model, mae_model, test_loader, cfg)
    results.append({"method": "mae_only", "label": BENCHMARK_LABELS["mae_only"], **m,
                     "overhead_vs_baseline": round(m["ms_per_sample"] / baseline_ms - 1, 3)})

    # DiffPure only
    logger.info("[3/4] DiffPure-all + classifier...")
    m = benchmark_diffpure_only(model, diffuser, test_loader, cfg)
    results.append({"method": "diffpure_only", "label": BENCHMARK_LABELS["diffpure_only"], **m,
                     "overhead_vs_baseline": round(m["ms_per_sample"] / baseline_ms - 1, 3)})

    # Full MedFedPure
    logger.info("[4/4] Full MedFedPure...")
    m = benchmark_full_medfedpure(model, mae_model, diffuser, test_loader, cfg)
    results.append({"method": "full_medfedpure", "label": BENCHMARK_LABELS["full_medfedpure"], **m,
                     "overhead_vs_baseline": round(m["ms_per_sample"] / baseline_ms - 1, 3)})

    return results


def save_efficiency_results(results, out_dir: Path, dataset: str, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{dataset.lower()}_seed{seed}"
    json_path = out_dir / f"efficiency_{tag}.json"
    csv_path  = out_dir / f"efficiency_{tag}.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    fields = ["label", "ms_per_sample", "frac_purified",
              "gpu_peak_mb", "overhead_vs_baseline"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in results:
            w.writerow({k: row.get(k, "") for k in fields})

    return json_path, csv_path


def parse_args():
    p = argparse.ArgumentParser(description="MedFedPure Efficiency Analysis")
    p.add_argument("--dataset",  type=str, default="br35h",
                   choices=["br35h", "cifar10"])
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--rounds",  type=int, default=None)
    p.add_argument("--epochs",  type=int, default=None)
    p.add_argument("--clients", type=int, default=None)
    p.add_argument("--out-dir", type=str,
                   default="experiment_results/efficiency")
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
    log_file = out_dir / f"efficiency_{cfg.DATASET.lower()}_seed{cfg.SEED}.log"
    logger = get_logger("efficiency", str(log_file))

    logger.info("=" * 60)
    logger.info(f"MedFedPure Efficiency Analysis – {cfg.DATASET.upper()}")
    logger.info("=" * 60)

    results = run_efficiency_analysis(cfg, logger)

    json_p, csv_p = save_efficiency_results(
        results, out_dir, cfg.DATASET, cfg.SEED)
    logger.info(f"\nResults saved: {json_p}  |  {csv_p}")

    # Print table
    logger.info("\n{:<35} {:>12} {:>12} {:>12} {:>12}".format(
        "Method", "ms/sample", "FracPur%", "GPU(MB)", "Overhead"))
    logger.info("-" * 83)
    for r in results:
        logger.info("{:<35} {:>12.3f} {:>12.1f} {:>12.1f} {:>+12.1%}".format(
            r["label"],
            r["ms_per_sample"],
            r["frac_purified"] * 100,
            r.get("gpu_peak_mb", 0.0),
            r["overhead_vs_baseline"],
        ))


if __name__ == "__main__":
    main()
