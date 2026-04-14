#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Additional Medical Dataset Experiment for MedFedPure
=====================================================
Validates that MedFedPure generalises beyond Br35H.

Dataset priority (feedback.txt Part 2):
  A. BraTS 2020 – brain MRI tumour/non-tumour classification
     (if data/brats/ exists with the expected folder structure)
  B. BreastMNIST – breast ultrasound malignancy detection
     (via medmnist package; auto-downloads)
  C. PathMNIST – colon pathology (fallback)

For each dataset we compare:
  1. Classifier-only federated baseline (FedAvg)
  2. pFedDef (current strongest federated baseline in the paper)
  3. Full MedFedPure

Usage:
  pip install medmnist          # for BreastMNIST fallback
  python experiments/additional_dataset.py --dataset breastmnist --seed 42
  python experiments/additional_dataset.py --dataset brats       --seed 42
"""

import os
import sys
import time
import json
import copy
import random
import logging
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse helpers from ablation_study
from experiments.ablation_study import (
    AblationConfig, get_logger, set_seed,
    dirichlet_split,
    DiffusionUNet, SimpleMAE,
    train_diffusion, train_mae,
    pgd_attack, ddpm_purify, adaptive_t,
    evaluate_clean, evaluate_adversarial,
    VARIANT_LABELS,
)

# ══════════════════════════════════════════════════════════════════════════════
# Dataset Loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_brats_classification(data_root: str, img_size: int = 224
                               ) -> Tuple[Dataset, Dataset]:
    """
    Load BraTS 2020 as binary tumour/non-tumour classification.

    Expected folder structure:
      data/brats/train/tumor/   (axial MRI slices with tumour)
      data/brats/train/healthy/ (healthy axial slices)
      data/brats/test/tumor/
      data/brats/test/healthy/

    Pre-processing BraTS:
      - Convert .nii.gz volumes to 2-D axial PNG slices
      - Keep only slices that contain at least 1 % tumour mask (positive)
      - Sample equal number of healthy (central) slices (negative)
    """
    brats_dir = Path(data_root) / "brats"
    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    tf_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    if not (brats_dir / "train").exists():
        raise FileNotFoundError(
            f"BraTS data not found at {brats_dir}.\n"
            "See README for BraTS preprocessing instructions.\n"
            "Use --dataset breastmnist as a practical fallback."
        )
    train_ds = torchvision.datasets.ImageFolder(str(brats_dir / "train"), transform=tf_train)
    test_ds  = torchvision.datasets.ImageFolder(str(brats_dir / "test"),  transform=tf_test)
    return train_ds, test_ds


def load_medmnist_dataset(name: str, data_root: str, img_size: int = 224
                           ) -> Tuple[Dataset, Dataset, int]:
    """
    Load a MedMNIST dataset.
    Returns (train_dataset, test_dataset, num_classes).

    Supported names: breastmnist, pathmnist, dermamnist, octmnist, etc.
    Install via: pip install medmnist
    """
    try:
        import medmnist
        from medmnist import INFO
    except ImportError:
        raise ImportError(
            "medmnist not installed.\n"
            "Run: pip install medmnist\n"
            "Or use --dataset brats if you have the BraTS data."
        )

    info = INFO[name]
    n_channels = info["n_channels"]
    n_classes  = len(info["label"])
    DataClass  = getattr(medmnist, info["python_class"])

    # MedMNIST images are 28×28; resize to img_size
    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    tf_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    download_root = str(Path(data_root) / "medmnist")
    Path(download_root).mkdir(parents=True, exist_ok=True)

    train_ds = DataClass(split="train", transform=tf_train,
                         download=True, root=download_root)
    test_ds  = DataClass(split="test",  transform=tf_test,
                         download=True, root=download_root)

    # MedMNIST targets come as (N,1) arrays; wrap dataset to return int labels
    class FlatLabelDataset(Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            img, lbl = self.ds[idx]
            return img, int(lbl[0]) if hasattr(lbl, "__len__") else int(lbl)

    return FlatLabelDataset(train_ds), FlatLabelDataset(test_ds), n_classes


# ══════════════════════════════════════════════════════════════════════════════
# pFedDef baseline model
# ══════════════════════════════════════════════════════════════════════════════

class PFedDefModel(nn.Module):
    """
    Simplified pFedDef: K=3 ResNet-18 experts with shared backbone,
    mixture weights learned globally (not per-input).
    """

    def __init__(self, num_classes: int, k: int = 3):
        super().__init__()
        self.k = k
        # Shared feature extractor
        backbone = torchvision.models.resnet18(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        # K classifier heads
        self.heads = nn.ModuleList([
            nn.Linear(feat_dim, num_classes) for _ in range(k)
        ])
        # Learnable mixture weights
        self.mix_weights = nn.Parameter(torch.ones(k) / k)

    def forward(self, x):
        feat = self.backbone(x)
        w = torch.softmax(self.mix_weights, dim=0)
        out = sum(w[i] * self.heads[i](feat) for i in range(self.k))
        return out


def train_pfeddef(cfg, client_subsets, ckpt_path, logger):
    """Train pFedDef model (adversarial training + FedAvg)."""
    if ckpt_path.exists():
        logger.info(f"Loading pFedDef model from {ckpt_path}")
        model = PFedDefModel(cfg.NUM_CLASSES).to(cfg.DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=cfg.DEVICE))
        model.eval()
        return model

    logger.info("Training pFedDef baseline...")
    global_model = PFedDefModel(cfg.NUM_CLASSES).to(cfg.DEVICE)

    for rnd in range(cfg.NUM_ROUNDS):
        local_states = []
        for subset in client_subsets:
            local = copy.deepcopy(global_model)
            local.train()
            loader = DataLoader(subset, batch_size=cfg.BATCH_SIZE,
                                shuffle=True, num_workers=cfg.NUM_WORKERS,
                                pin_memory=(cfg.DEVICE == "cuda"))
            opt = optim.SGD(local.parameters(), lr=cfg.LEARNING_RATE,
                            momentum=0.9, weight_decay=5e-4)
            for _ in range(cfg.LOCAL_EPOCHS):
                for imgs, labels in loader:
                    if imgs.size(0) < 2:
                        continue
                    imgs, labels = imgs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                    # Adversarial training (L∞ PGD)
                    adv = pgd_attack(local, imgs, labels,
                                     cfg.PGD_EPS, cfg.PGD_ALPHA, cfg.PGD_STEPS,
                                     cfg.DEVICE)
                    loss = F.cross_entropy(local(adv), labels) + \
                           0.5 * F.cross_entropy(local(imgs), labels)
                    opt.zero_grad(); loss.backward(); opt.step()
            local_states.append(local.state_dict())

        avg = {}
        for key in local_states[0]:
            avg[key] = torch.stack([s[key].float() for s in local_states]).mean(0)
        global_model.load_state_dict(avg)
        if (rnd + 1) % 5 == 0:
            logger.info(f"  pFedDef round {rnd+1}/{cfg.NUM_ROUNDS}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(global_model.state_dict(), ckpt_path)
    logger.info(f"Saved pFedDef → {ckpt_path}")
    global_model.eval()
    return global_model


def train_fedavg_baseline(cfg, client_subsets, ckpt_path, logger):
    """Train standard FedAvg without any defense (classifier-only baseline)."""
    if ckpt_path.exists():
        logger.info(f"Loading FedAvg baseline from {ckpt_path}")
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, cfg.NUM_CLASSES)
        model = model.to(cfg.DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=cfg.DEVICE))
        model.eval()
        return model

    logger.info("Training FedAvg baseline (classifier-only)...")
    global_model = torchvision.models.resnet18(weights=None)
    global_model.fc = nn.Linear(global_model.fc.in_features, cfg.NUM_CLASSES)
    global_model = global_model.to(cfg.DEVICE)

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
                    loss = F.cross_entropy(local(imgs), labels)
                    opt.zero_grad(); loss.backward(); opt.step()
            local_states.append(local.state_dict())

        avg = {}
        for key in local_states[0]:
            avg[key] = torch.stack([s[key].float() for s in local_states]).mean(0)
        global_model.load_state_dict(avg)
        if (rnd + 1) % 5 == 0:
            logger.info(f"  FedAvg round {rnd+1}/{cfg.NUM_ROUNDS}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(global_model.state_dict(), ckpt_path)
    logger.info(f"Saved FedAvg baseline → {ckpt_path}")
    global_model.eval()
    return global_model


# ══════════════════════════════════════════════════════════════════════════════
# MedFedPure for additional dataset (uses ablation MoE from ablation_study.py)
# ══════════════════════════════════════════════════════════════════════════════

from experiments.ablation_study import MoEClient, train_personalized_fl


# ══════════════════════════════════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════════════════════════════════

def run_additional_dataset(cfg: AblationConfig, dataset_name: str,
                           logger: logging.Logger) -> List[Dict]:
    set_seed(cfg.SEED)
    device = cfg.DEVICE

    # ── Load dataset ─────────────────────────────────────────────────────────
    logger.info(f"Loading additional dataset: {dataset_name}")
    if dataset_name == "brats":
        train_ds, test_ds = load_brats_classification(cfg.DATA_ROOT, cfg.IMG_SIZE)
        cfg.NUM_CLASSES = 2
    else:
        train_ds, test_ds, cfg.NUM_CLASSES = load_medmnist_dataset(
            dataset_name, cfg.DATA_ROOT, cfg.IMG_SIZE)

    logger.info(f"  Train: {len(train_ds)}  Test: {len(test_ds)}  "
                f"Classes: {cfg.NUM_CLASSES}")

    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                             num_workers=cfg.NUM_WORKERS,
                             pin_memory=(device == "cuda"))

    # ── Non-IID split ─────────────────────────────────────────────────────────
    client_indices = dirichlet_split(train_ds, cfg.NUM_CLIENTS,
                                     cfg.DIRICHLET_ALPHA, cfg.SEED)
    client_subsets = [Subset(train_ds, idx) for idx in client_indices]

    full_loader = DataLoader(train_ds, batch_size=cfg.DIFF_BATCH, shuffle=True,
                             num_workers=cfg.NUM_WORKERS,
                             pin_memory=(device == "cuda"))

    # ── Checkpoint paths ─────────────────────────────────────────────────────
    tag = f"{dataset_name}_seed{cfg.SEED}"
    ckpt_dir = Path("checkpoints") / f"addl_{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Train shared components ───────────────────────────────────────────────
    diffuser  = train_diffusion(cfg, full_loader, ckpt_dir / "diffusion.pt", logger)
    mae_model = train_mae(cfg, full_loader, ckpt_dir / "mae_detector.pt", logger)

    # ── Train baseline models ─────────────────────────────────────────────────
    fedavg_model  = train_fedavg_baseline(cfg, client_subsets,
                                           ckpt_dir / "fedavg.pt", logger)
    pfeddef_model = train_pfeddef(cfg, client_subsets,
                                   ckpt_dir / "pfeddef.pt", logger)
    moe_clients   = train_personalized_fl(cfg, client_subsets,
                                           ckpt_dir / "moe_clients.pt", logger)
    moe_model = moe_clients[0]
    moe_model.eval()

    # ── Evaluate three methods ────────────────────────────────────────────────
    results = []
    methods = [
        ("FedAvg (Classifier-only)", fedavg_model,  None,      None,     "classifier_only"),
        ("pFedDef",                   pfeddef_model, None,      None,     "classifier_only"),
        ("MedFedPure (ours)",         moe_model,     mae_model, diffuser, "full_medfedpure"),
    ]

    for label, model, mae, diff, variant in methods:
        logger.info(f"\n[Method] {label}")
        clean_acc = evaluate_clean(model, test_loader, device)
        logger.info(f"  Clean Acc: {clean_acc:.2f}%")

        adv = evaluate_adversarial(model, test_loader, cfg, mae, diff, variant, logger)
        logger.info(f"  Adv Acc: {adv['adv_acc']:.2f}%")

        results.append({
            "dataset":       dataset_name,
            "seed":          cfg.SEED,
            "method":        label,
            "clean_acc":     round(clean_acc, 2),
            "adv_acc":       round(adv["adv_acc"], 2),
            "detection_f1":  round(adv["detection_f1"], 4),
            "frac_purified": round(adv["frac_purified"], 4),
            "latency_ms":    round(adv["avg_latency_ms"], 2),
            "notes":         f"eps={cfg.PGD_EPS}, steps={cfg.PGD_STEPS}",
        })

    return results


def save_additional_results(results, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"additional_{tag}.json"
    csv_path  = out_dir / f"additional_{tag}.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    fields = ["method", "clean_acc", "adv_acc", "detection_f1",
              "frac_purified", "latency_ms", "notes"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in results:
            w.writerow({k: row[k] for k in fields})

    return json_path, csv_path


def parse_args():
    p = argparse.ArgumentParser(description="MedFedPure – Additional Dataset")
    p.add_argument("--dataset",  type=str, default="breastmnist",
                   choices=["brats", "breastmnist", "pathmnist", "octmnist",
                            "dermamnist"],
                   help="Additional medical dataset to evaluate on")
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--img-size",  type=int, default=64,
                   help="Image size (64 recommended for MedMNIST, 224 for BraTS)")
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--rounds",  type=int, default=None)
    p.add_argument("--epochs",  type=int, default=None)
    p.add_argument("--clients", type=int, default=None)
    p.add_argument("--out-dir", type=str,
                   default="experiment_results/additional_dataset")
    p.add_argument("--fast", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = AblationConfig()
    cfg.DATASET   = args.dataset
    cfg.DATA_ROOT = args.data_root
    cfg.IMG_SIZE  = args.img_size
    cfg.IMG_CHANNELS = 3
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_seed{args.seed}"
    log_file = out_dir / f"additional_{tag}.log"
    logger = get_logger("additional_dataset", str(log_file))

    logger.info("=" * 60)
    logger.info(f"Additional Dataset Experiment: {args.dataset.upper()}")
    logger.info("=" * 60)

    results = run_additional_dataset(cfg, args.dataset, logger)

    json_p, csv_p = save_additional_results(results, out_dir, tag)
    logger.info(f"\nResults saved: {json_p}  |  {csv_p}")

    # Print table
    logger.info("\n{:<30} {:>10} {:>10} {:>10}".format(
        "Method", "Clean%", "Adv%", "Det-F1"))
    logger.info("-" * 62)
    for r in results:
        logger.info("{:<30} {:>10.2f} {:>10.2f} {:>10.4f}".format(
            r["method"], r["clean_acc"], r["adv_acc"], r["detection_f1"]))


if __name__ == "__main__":
    main()
