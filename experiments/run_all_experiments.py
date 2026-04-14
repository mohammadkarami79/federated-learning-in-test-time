#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Runner for All New MedFedPure Experiments
==================================================
Runs all four experiment groups sequentially (or selectively) and
produces a final consolidated report.

Usage examples:
  # Run everything on Br35H with 3 seeds:
  python experiments/run_all_experiments.py --dataset br35h --seeds 42 43 44

  # Run ablation only:
  python experiments/run_all_experiments.py --dataset br35h --parts ablation

  # Fast debug run (reduced epochs):
  python experiments/run_all_experiments.py --dataset br35h --fast --seeds 42

  # Full run (Br35H + additional MedMNIST dataset):
  python experiments/run_all_experiments.py --dataset br35h \\
      --additional-dataset breastmnist --seeds 42 43 44
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.ablation_study import (
    AblationConfig, get_logger, run_ablation, save_results as save_ablation
)
from experiments.additional_dataset import (
    run_additional_dataset, save_additional_results
)
from experiments.baseline_comparison import (
    run_baseline_comparison, save_baseline_results
)
from experiments.efficiency_analysis import (
    run_efficiency_analysis, save_efficiency_results
)


# ══════════════════════════════════════════════════════════════════════════════
# Multi-seed aggregation
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_seeds(results_by_seed: List[List[dict]], key_field: str = "variant"
                    ) -> List[dict]:
    """Average numeric metrics across seeds; keep std."""
    from collections import defaultdict
    import numpy as np

    grouped = defaultdict(list)
    for seed_results in results_by_seed:
        for row in seed_results:
            grouped[row[key_field]].append(row)

    aggregated = []
    for key, rows in grouped.items():
        base = {k: v for k, v in rows[0].items()
                if not isinstance(v, (int, float)) or k == "seed"}
        numeric_keys = [k for k, v in rows[0].items()
                        if isinstance(v, (int, float)) and k != "seed"]
        for nk in numeric_keys:
            vals = [r[nk] for r in rows]
            base[f"{nk}_mean"] = round(float(np.mean(vals)), 3)
            base[f"{nk}_std"]  = round(float(np.std(vals)), 3)
        aggregated.append(base)
    return aggregated


def save_aggregated(agg_results: List[dict], out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"aggregated_{tag}.json"
    with open(path, "w") as f:
        json.dump(agg_results, f, indent=2)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="MedFedPure – Run All New Experiments")
    p.add_argument("--dataset",  type=str, default="br35h",
                   choices=["br35h", "cifar10"],
                   help="Primary dataset")
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="Random seeds (run each experiment per seed)")
    p.add_argument("--rounds",  type=int, default=None,
                   help="Override NUM_ROUNDS")
    p.add_argument("--epochs",  type=int, default=None,
                   help="Override LOCAL_EPOCHS")
    p.add_argument("--clients", type=int, default=None,
                   help="Override NUM_CLIENTS")
    p.add_argument("--parts", type=str, nargs="+",
                   default=["ablation", "additional", "baseline", "efficiency"],
                   choices=["ablation", "additional", "baseline", "efficiency"],
                   help="Which experiment parts to run")
    p.add_argument("--additional-dataset", type=str, default="breastmnist",
                   choices=["brats", "breastmnist", "pathmnist", "octmnist"],
                   help="Additional medical dataset to use in Part 2")
    p.add_argument("--additional-img-size", type=int, default=64,
                   help="Image size for additional dataset (64 for MedMNIST)")
    p.add_argument("--out-dir", type=str, default="experiment_results",
                   help="Root output directory")
    p.add_argument("--fast", action="store_true",
                   help="Quick test mode (reduced epochs)")
    return p.parse_args()


def build_cfg(args, seed: int) -> AblationConfig:
    cfg = AblationConfig()
    cfg.DATASET   = args.dataset
    cfg.DATA_ROOT = args.data_root
    cfg.SEED      = seed
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
    return cfg


def main():
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    log_file = out_root / f"run_all_{args.dataset}.log"
    logger = get_logger("run_all", str(log_file))

    logger.info("=" * 70)
    logger.info("MedFedPure – Full Experiment Suite")
    logger.info(f"Dataset: {args.dataset.upper()}  Seeds: {args.seeds}  "
                f"Parts: {args.parts}")
    logger.info("=" * 70)

    wall_start = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # Part 1: Ablation Study
    # ══════════════════════════════════════════════════════════════════════════
    if "ablation" in args.parts:
        logger.info("\n" + "─" * 60)
        logger.info("PART 1 — Ablation Study")
        logger.info("─" * 60)
        ablation_all_seeds = []
        abl_out = out_root / "ablation"
        abl_out.mkdir(parents=True, exist_ok=True)

        for seed in args.seeds:
            logger.info(f"\n  [Seed {seed}]")
            cfg = build_cfg(args, seed)
            results = run_ablation(cfg, logger)
            save_ablation(results, abl_out, cfg.DATASET, seed)
            ablation_all_seeds.append(results)

        if len(args.seeds) > 1:
            agg = aggregate_seeds(ablation_all_seeds, key_field="variant")
            agg_path = save_aggregated(agg, abl_out, f"{args.dataset}_aggregated")
            logger.info(f"  Aggregated ablation results → {agg_path}")
            # Print mean ± std table
            logger.info("\n{:<30} {:>12} {:>12}".format(
                "Variant", "CleanAcc", "AdvAcc"))
            logger.info("-" * 54)
            for row in agg:
                logger.info("{:<30} {:>9.2f}±{:.2f} {:>9.2f}±{:.2f}".format(
                    row.get("label", row.get("variant", "?")),
                    row.get("clean_acc_mean", 0), row.get("clean_acc_std", 0),
                    row.get("adv_acc_mean", 0),   row.get("adv_acc_std", 0),
                ))

    # ══════════════════════════════════════════════════════════════════════════
    # Part 2: Additional Medical Dataset
    # ══════════════════════════════════════════════════════════════════════════
    if "additional" in args.parts:
        logger.info("\n" + "─" * 60)
        logger.info(f"PART 2 — Additional Dataset: {args.additional_dataset.upper()}")
        logger.info("─" * 60)
        addl_all_seeds = []
        addl_out = out_root / "additional_dataset"
        addl_out.mkdir(parents=True, exist_ok=True)

        for seed in args.seeds:
            logger.info(f"\n  [Seed {seed}]")
            cfg = build_cfg(args, seed)
            cfg.IMG_SIZE  = args.additional_img_size
            cfg.IMG_CHANNELS = 3
            try:
                results = run_additional_dataset(cfg, args.additional_dataset, logger)
                save_additional_results(results, addl_out,
                                        f"{args.additional_dataset}_seed{seed}")
                addl_all_seeds.append(results)
            except (FileNotFoundError, ImportError) as e:
                logger.warning(f"  Skipping additional dataset: {e}")
                break

        if len(addl_all_seeds) > 1:
            agg = aggregate_seeds(addl_all_seeds, key_field="method")
            save_aggregated(agg, addl_out,
                            f"{args.additional_dataset}_aggregated")

    # ══════════════════════════════════════════════════════════════════════════
    # Part 3: Baseline Comparison
    # ══════════════════════════════════════════════════════════════════════════
    if "baseline" in args.parts:
        logger.info("\n" + "─" * 60)
        logger.info("PART 3 — Baseline Comparison")
        logger.info("─" * 60)
        base_all_seeds = []
        base_out = out_root / "baseline_comparison"
        base_out.mkdir(parents=True, exist_ok=True)

        for seed in args.seeds:
            logger.info(f"\n  [Seed {seed}]")
            cfg = build_cfg(args, seed)
            results = run_baseline_comparison(cfg, logger)
            save_baseline_results(results, base_out, cfg.DATASET, seed)
            base_all_seeds.append(results)

        if len(base_all_seeds) > 1:
            agg = aggregate_seeds(base_all_seeds, key_field="method")
            save_aggregated(agg, base_out,
                            f"{args.dataset}_aggregated")

    # ══════════════════════════════════════════════════════════════════════════
    # Part 4: Efficiency Analysis
    # ══════════════════════════════════════════════════════════════════════════
    if "efficiency" in args.parts:
        logger.info("\n" + "─" * 60)
        logger.info("PART 4 — Efficiency Analysis")
        logger.info("─" * 60)
        eff_out = out_root / "efficiency"
        eff_out.mkdir(parents=True, exist_ok=True)

        # Efficiency is deterministic; run once with first seed
        seed = args.seeds[0]
        cfg  = build_cfg(args, seed)
        results = run_efficiency_analysis(cfg, logger)
        save_efficiency_results(results, eff_out, cfg.DATASET, seed)

    # ══════════════════════════════════════════════════════════════════════════
    # Final summary
    # ══════════════════════════════════════════════════════════════════════════
    wall_elapsed = time.time() - wall_start
    logger.info("\n" + "=" * 70)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info(f"Total wall time: {wall_elapsed/3600:.2f} h  "
                f"({wall_elapsed:.0f} s)")
    logger.info(f"All results saved under: {out_root.resolve()}")
    logger.info("=" * 70)

    # Write a compact summary JSON for the writing team
    summary = {
        "dataset": args.dataset,
        "seeds": args.seeds,
        "parts_run": args.parts,
        "additional_dataset": args.additional_dataset,
        "output_dir": str(out_root.resolve()),
        "wall_time_hours": round(wall_elapsed / 3600, 3),
        "config": {
            "NUM_ROUNDS":   args.rounds or "default",
            "LOCAL_EPOCHS": args.epochs or "default",
            "NUM_CLIENTS":  args.clients or "default",
        }
    }
    with open(out_root / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary → {out_root / 'experiment_summary.json'}")


if __name__ == "__main__":
    main()
