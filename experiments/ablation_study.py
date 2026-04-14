#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study for MedFedPure
==============================
Evaluates contribution of each component on Br35H and optionally CIFAR-10.

Variants:
  1. Full MedFedPure (all components active)
  2. No MAE detector, purify ALL inputs
  3. No MAE detector, NO purification
  4. No diffusion purification (MAE detection only)
  5. No personalization (standard FedAvg backbone)
  6. No adaptive purification (fixed denoising steps)
  7. Classifier only (no defense)

Usage (on server):
  python experiments/ablation_study.py --dataset br35h --seed 42
  python experiments/ablation_study.py --dataset cifar10 --seed 42
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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AblationConfig:
    # ── dataset ──────────────────────────────────────────────────────────────
    DATASET: str = "br35h"
    DATA_ROOT: str = "data"

    # Br35H
    BR35H_IMG_SIZE: int = 224
    BR35H_CHANNELS: int = 3
    BR35H_CLASSES: int = 2

    # CIFAR-10
    CIFAR10_IMG_SIZE: int = 32
    CIFAR10_CHANNELS: int = 3
    CIFAR10_CLASSES: int = 10

    # ── federated learning ────────────────────────────────────────────────────
    NUM_CLIENTS: int = 10
    NUM_ROUNDS: int = 20
    LOCAL_EPOCHS: int = 15
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    DIRICHLET_ALPHA: float = 0.4       # non-IID heterogeneity

    # Mixture-of-experts (personalized)
    N_EXPERTS: int = 3
    ENTROPY_REG: float = 0.1
    L2_REG: float = 1e-4

    # ── adversarial attack (L∞ PGD) ──────────────────────────────────────────
    PGD_EPS: float = 0.015            # ε  (matches paper Table 2)
    PGD_ALPHA: float = 0.003          # step size
    PGD_STEPS: int = 7                # K
    PGD_NORM: str = "linf"

    # ── diffusion purification ────────────────────────────────────────────────
    DIFF_T_MAX: int = 1000
    DIFF_BETA_START: float = 1e-4
    DIFF_BETA_END: float = 0.02
    DIFF_HIDDEN: int = 128
    DIFF_EPOCHS: int = 50
    DIFF_LR: float = 1e-4
    DIFF_BATCH: int = 32

    # adaptive purification schedule
    DIFF_T_CLEAN: int = 50            # steps for low-error (less purification)
    DIFF_T_ADV: int = 150             # steps for high-error (more purification)

    # fixed purification (ablation variant 6)
    DIFF_T_FIXED: int = 100

    # ── MAE detector ─────────────────────────────────────────────────────────
    MAE_EPOCHS: int = 30
    MAE_LR: float = 1e-3
    MAE_BATCH: int = 64
    MAE_EMBED_DIM: int = 512
    MAE_DEPTH: int = 12
    MAE_HEADS: int = 8
    MAE_MASK_RATIO: float = 0.75
    MAE_KAPPA_BR35H: float = 18.0    # top-κ% flagged as adversarial
    MAE_KAPPA_CIFAR: float = 5.0

    # ── reproducibility ───────────────────────────────────────────────────────
    SEED: int = 42

    # ── hardware ──────────────────────────────────────────────────────────────
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS: int = 2

    # ── derived (set after init) ──────────────────────────────────────────────
    IMG_SIZE: int = 224
    IMG_CHANNELS: int = 3
    NUM_CLASSES: int = 2
    MAE_PATCH_SIZE: int = 16
    MAE_KAPPA: float = 18.0

    def finalize(self):
        if self.DATASET.lower() == "br35h":
            self.IMG_SIZE = self.BR35H_IMG_SIZE
            self.IMG_CHANNELS = self.BR35H_CHANNELS
            self.NUM_CLASSES = self.BR35H_CLASSES
            self.MAE_PATCH_SIZE = 16
            self.MAE_KAPPA = self.MAE_KAPPA_BR35H
        else:
            self.IMG_SIZE = self.CIFAR10_IMG_SIZE
            self.IMG_CHANNELS = self.CIFAR10_CHANNELS
            self.NUM_CLASSES = self.CIFAR10_CLASSES
            self.MAE_PATCH_SIZE = 4
            self.MAE_KAPPA = self.MAE_KAPPA_CIFAR
        return self


# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════════
# Dataset helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_br35h(data_root: str, img_size: int) -> Tuple[Dataset, Dataset]:
    """Load Br35H from local folder or Kaggle download."""
    br35h_dir = Path(data_root) / "br35h"

    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dir = br35h_dir / "train"
    test_dir  = br35h_dir / "test"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Br35H dataset not found at {br35h_dir}.\n"
            "Expected structure: data/br35h/train/{yes,no}/ and data/br35h/test/{yes,no}/"
        )

    train_dataset = torchvision.datasets.ImageFolder(str(train_dir), transform=transform_train)
    test_dataset  = torchvision.datasets.ImageFolder(str(test_dir),  transform=transform_test)
    return train_dataset, test_dataset


def load_cifar10(data_root: str) -> Tuple[Dataset, Dataset]:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010]),
    ])
    train = torchvision.datasets.CIFAR10(data_root, train=True,  download=True, transform=transform_train)
    test  = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=transform_test)
    return train, test


def get_dataset(cfg: AblationConfig) -> Tuple[Dataset, Dataset]:
    if cfg.DATASET.lower() == "br35h":
        return load_br35h(cfg.DATA_ROOT, cfg.IMG_SIZE)
    elif cfg.DATASET.lower() == "cifar10":
        return load_cifar10(cfg.DATA_ROOT)
    else:
        raise ValueError(f"Unknown dataset: {cfg.DATASET}")


# ── Non-IID Dirichlet split ───────────────────────────────────────────────────

def dirichlet_split(dataset: Dataset, num_clients: int, alpha: float,
                    seed: int = 42) -> List[List[int]]:
    """Split dataset indices non-IID using Dirichlet distribution."""
    rng = np.random.default_rng(seed)

    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = int(labels.max()) + 1
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(cls_idx)).astype(int)
        # Adjust rounding
        proportions[-1] = len(cls_idx) - proportions[:-1].sum()
        start = 0
        for cid, n in enumerate(proportions):
            client_indices[cid].extend(cls_idx[start:start + n].tolist())
            start += n

    return client_indices


# ══════════════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════════════

def create_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    """ResNet-18 backbone (paper backbone for Br35H)."""
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class MoEClient(nn.Module):
    """
    Mixture-of-Experts personalized FL model.
    K=3 ResNet-18 experts + per-input attention network.
    """

    def __init__(self, num_classes: int, k: int = 3, pretrained: bool = True):
        super().__init__()
        self.k = k

        # Shared feature extractor (ResNet-18 without final FC)
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feat_dim = feat_dim

        # K expert classifier heads
        self.expert_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes),
            ) for _ in range(k)
        ])

        # K attention networks (produce scalar attention score per expert)
        self.attention_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            ) for _ in range(k)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)                        # [B, feat_dim]

        # Attention weights via softmax
        scores = torch.cat(
            [net(feat) for net in self.attention_nets], dim=1
        )                                               # [B, K]
        weights = torch.softmax(scores, dim=1)         # [B, K]

        # Weighted sum of expert predictions
        logits = torch.stack(
            [head(feat) for head in self.expert_heads], dim=1
        )                                               # [B, K, C]
        out = (weights.unsqueeze(-1) * logits).sum(dim=1)  # [B, C]
        return out

    def entropy_regularization(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        scores = torch.cat(
            [net(feat) for net in self.attention_nets], dim=1
        )
        weights = torch.softmax(scores, dim=1)
        entropy = -(weights * (weights + 1e-9).log()).sum(dim=1).mean()
        return entropy

    def attention_l2(self) -> torch.Tensor:
        l2 = sum(p.pow(2).sum() for net in self.attention_nets
                 for p in net.parameters())
        return l2


# ══════════════════════════════════════════════════════════════════════════════
# Diffusion model (simplified U-Net, same architecture as integrated_diffusion_training.py)
# ══════════════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class DiffusionUNet(nn.Module):
    """Lightweight U-Net for diffusion purification."""

    def __init__(self, in_channels: int = 3, hidden: int = 128):
        super().__init__()
        h = hidden

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, h),
            nn.SiLU(),
            nn.Linear(h, h),
        )

        self.enc1 = DoubleConv(in_channels, h)
        self.enc2 = DoubleConv(h, h * 2)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(h * 2, h * 4)

        self.up1 = nn.ConvTranspose2d(h * 4, h * 2, 2, stride=2)
        self.dec1 = DoubleConv(h * 4, h * 2)

        self.up2 = nn.ConvTranspose2d(h * 2, h, 2, stride=2)
        self.dec2 = DoubleConv(h * 2, h)

        self.out = nn.Conv2d(h, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t: [B] float
        t_emb = self.time_mlp(t.view(-1, 1).float())  # [B, h]

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b  = self.bottleneck(self.pool(e2))

        d1 = self.up1(b)
        if d1.shape[-2:] != e2.shape[-2:]:
            d1 = F.interpolate(d1, size=e2.shape[-2:])
        d1 = self.dec1(torch.cat([d1, e2], dim=1))

        d2 = self.up2(d1)
        if d2.shape[-2:] != e1.shape[-2:]:
            d2 = F.interpolate(d2, size=e1.shape[-2:])
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        return self.out(d2)


def make_beta_schedule(T: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


def train_diffusion(cfg: AblationConfig, train_loader: DataLoader,
                    ckpt_path: Path, logger: logging.Logger) -> DiffusionUNet:
    """Train diffusion model (Phase 1 of MedFedPure)."""
    if ckpt_path.exists():
        logger.info(f"Loading diffusion model from {ckpt_path}")
        model = DiffusionUNet(cfg.IMG_CHANNELS, cfg.DIFF_HIDDEN).to(cfg.DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=cfg.DEVICE))
        model.eval()
        return model

    logger.info("Training diffusion model (Phase 1)...")
    model = DiffusionUNet(cfg.IMG_CHANNELS, cfg.DIFF_HIDDEN).to(cfg.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.DIFF_LR)

    betas = make_beta_schedule(cfg.DIFF_T_MAX, cfg.DIFF_BETA_START, cfg.DIFF_BETA_END)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0).to(cfg.DEVICE)

    model.train()
    for epoch in range(cfg.DIFF_EPOCHS):
        total_loss = 0.0
        batches = 0
        for imgs, _ in train_loader:
            imgs = imgs.to(cfg.DEVICE)
            B = imgs.size(0)

            # Sample random timestep
            t = torch.randint(1, cfg.DIFF_T_MAX + 1, (B,), device=cfg.DEVICE)
            ab = alpha_bar[t - 1].view(B, 1, 1, 1)

            # Forward diffusion
            noise = torch.randn_like(imgs)
            noisy = ab.sqrt() * imgs + (1 - ab).sqrt() * noise

            # Predict noise
            pred_noise = model(noisy, t.float())
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Diffusion epoch {epoch+1}/{cfg.DIFF_EPOCHS}  loss={total_loss/batches:.4f}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Saved diffusion model → {ckpt_path}")
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Simple ViT-MAE (for adversarial detection)
# ══════════════════════════════════════════════════════════════════════════════

class SimpleMAE(nn.Module):
    """Lightweight convolutional MAE for detection (fast training)."""

    def __init__(self, in_channels: int = 3, img_size: int = 224):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64,  3, stride=2, padding=1),  # /2
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),            # /4
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),           # /8
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(64, in_channels, 2, stride=2),
            nn.Sigmoid(),
        )
        self.img_size = img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        recon = self.decoder(feat)
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[-2:], mode="bilinear",
                                  align_corners=False)
        return recon

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE between input and reconstruction."""
        with torch.no_grad():
            self.eval()
            recon = self(x)
            err = F.mse_loss(recon, x.clamp(0, 1), reduction="none")
            return err.view(x.size(0), -1).mean(dim=1)


def train_mae(cfg: AblationConfig, train_loader: DataLoader,
              ckpt_path: Path, logger: logging.Logger) -> SimpleMAE:
    """Train MAE detector (Phase 2 of MedFedPure)."""
    if ckpt_path.exists():
        logger.info(f"Loading MAE detector from {ckpt_path}")
        model = SimpleMAE(cfg.IMG_CHANNELS, cfg.IMG_SIZE).to(cfg.DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=cfg.DEVICE))
        model.eval()
        return model

    logger.info("Training MAE detector (Phase 2)...")
    model = SimpleMAE(cfg.IMG_CHANNELS, cfg.IMG_SIZE).to(cfg.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.MAE_LR)

    model.train()
    for epoch in range(cfg.MAE_EPOCHS):
        total_loss = 0.0
        batches = 0
        for imgs, _ in train_loader:
            imgs = imgs.to(cfg.DEVICE).clamp(0, 1)
            # Random masking: zero out random patches
            B, C, H, W = imgs.shape
            mask = torch.ones_like(imgs)
            p = 16 if H >= 64 else 4
            for bi in range(B):
                for hi in range(0, H, p):
                    for wi in range(0, W, p):
                        if random.random() < cfg.MAE_MASK_RATIO:
                            mask[bi, :, hi:hi+p, wi:wi+p] = 0
            masked = imgs * mask

            recon = model(masked)
            # Only compute loss on masked patches
            inv_mask = 1 - mask
            denom = inv_mask.sum() + 1e-6
            loss = ((recon - imgs).pow(2) * inv_mask).sum() / denom

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  MAE epoch {epoch+1}/{cfg.MAE_EPOCHS}  loss={total_loss/batches:.4f}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Saved MAE detector → {ckpt_path}")
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# PGD Attack
# ══════════════════════════════════════════════════════════════════════════════

def pgd_attack(model: nn.Module, imgs: torch.Tensor, labels: torch.Tensor,
               eps: float, alpha: float, steps: int,
               device: str) -> torch.Tensor:
    """L∞ PGD white-box attack."""
    model.eval()
    imgs_orig = imgs.clone().detach().to(device)
    labels = labels.to(device)

    x_adv = imgs_orig.clone().detach()
    # Random start
    x_adv = x_adv + torch.zeros_like(x_adv).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0, 1)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = (x_adv - imgs_orig).clamp(-eps, eps)
        x_adv = (imgs_orig + delta).clamp(0, 1)

    return x_adv.detach()


# ══════════════════════════════════════════════════════════════════════════════
# Diffusion purification helpers
# ══════════════════════════════════════════════════════════════════════════════

def ddpm_purify(diffuser: DiffusionUNet, imgs: torch.Tensor,
                t_start: int, cfg: AblationConfig) -> torch.Tensor:
    """
    Partial reverse diffusion (purification).
    Forward: add noise up to timestep t_start.
    Reverse: denoise back.
    """
    if diffuser is None or t_start <= 0:
        return imgs

    betas = make_beta_schedule(cfg.DIFF_T_MAX, cfg.DIFF_BETA_START, cfg.DIFF_BETA_END).to(imgs.device)
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    with torch.no_grad():
        # Forward diffusion up to t_start
        ab = alpha_bar[t_start - 1]
        noise = torch.randn_like(imgs)
        x_t = ab.sqrt() * imgs + (1 - ab).sqrt() * noise

        # Reverse diffusion
        x = x_t.clone()
        for t_step in range(t_start, 0, -1):
            t_tensor = torch.full((imgs.size(0),), t_step,
                                  device=imgs.device, dtype=torch.float32)
            pred_noise = diffuser(x, t_tensor)
            beta_t = betas[t_step - 1]
            alpha_t = alphas[t_step - 1]
            ab_t = alpha_bar[t_step - 1]

            # DDPM reverse step (mean)
            x0_pred = (x - (1 - ab_t).sqrt() * pred_noise) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)

            if t_step > 1:
                ab_prev = alpha_bar[t_step - 2]
                mean = (ab_prev.sqrt() * beta_t / (1 - ab_t)) * x0_pred + \
                       (alpha_t.sqrt() * (1 - ab_prev) / (1 - ab_t)) * x
                var = beta_t * (1 - ab_prev) / (1 - ab_t)
                x = mean + var.sqrt() * torch.randn_like(mean)
            else:
                x = x0_pred

        return x.clamp(0, 1)


def adaptive_t(recon_errors: torch.Tensor, cfg: AblationConfig) -> torch.Tensor:
    """Map reconstruction error to purification timestep adaptively."""
    err_min = recon_errors.min()
    err_max = recon_errors.max() + 1e-8
    normalized = (recon_errors - err_min) / (err_max - err_min)
    t_values = (cfg.DIFF_T_CLEAN + normalized * (cfg.DIFF_T_ADV - cfg.DIFF_T_CLEAN)).long()
    return t_values.clamp(cfg.DIFF_T_CLEAN, cfg.DIFF_T_ADV)


# ══════════════════════════════════════════════════════════════════════════════
# Federated Training
# ══════════════════════════════════════════════════════════════════════════════

def train_personalized_fl(cfg: AblationConfig, client_subsets: List[Subset],
                          ckpt_path: Path, logger: logging.Logger,
                          ) -> List[MoEClient]:
    """
    Phase 3: Train personalized federated classifier (MoE).
    Returns list of K client models.
    """
    if ckpt_path.exists():
        logger.info(f"Loading personalized FL model from {ckpt_path}")
        # Create models and load
        clients = [MoEClient(cfg.NUM_CLASSES, cfg.N_EXPERTS, pretrained=False).to(cfg.DEVICE)
                   for _ in range(cfg.NUM_CLIENTS)]
        states = torch.load(ckpt_path, map_location=cfg.DEVICE)
        for i, state in enumerate(states):
            clients[i].load_state_dict(state)
            clients[i].eval()
        return clients

    logger.info("Training personalized FL model (Phase 3 - MoE)...")

    # Initialize global model
    global_model = MoEClient(cfg.NUM_CLASSES, cfg.N_EXPERTS, pretrained=True).to(cfg.DEVICE)

    # Per-client local models
    client_models = [
        copy.deepcopy(global_model) for _ in range(cfg.NUM_CLIENTS)
    ]

    for round_num in range(cfg.NUM_ROUNDS):
        local_states = []

        for cid, (client_model, subset) in enumerate(zip(client_models, client_subsets)):
            client_model.train()
            loader = DataLoader(subset, batch_size=cfg.BATCH_SIZE,
                                shuffle=True, num_workers=cfg.NUM_WORKERS,
                                pin_memory=(cfg.DEVICE == "cuda"))
            optimizer = optim.Adam(client_model.parameters(), lr=cfg.LEARNING_RATE,
                                   weight_decay=cfg.L2_REG)

            for _ in range(cfg.LOCAL_EPOCHS):
                for imgs, labels in loader:
                    if imgs.size(0) < 2:
                        continue
                    imgs, labels = imgs.to(cfg.DEVICE), labels.to(cfg.DEVICE)

                    logits = client_model(imgs)
                    ce_loss = F.cross_entropy(logits, labels)
                    ent_reg = client_model.entropy_regularization(imgs)
                    l2_reg  = client_model.attention_l2()
                    loss = ce_loss + cfg.ENTROPY_REG * ent_reg + cfg.L2_REG * l2_reg

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            local_states.append(copy.deepcopy(client_model.state_dict()))

        # FedAvg aggregation
        avg_state = {}
        for key in local_states[0].keys():
            tensors = [s[key].float() for s in local_states]
            avg_state[key] = torch.stack(tensors).mean(dim=0)

        global_model.load_state_dict(avg_state)

        # Broadcast to clients
        for cm in client_models:
            cm.load_state_dict(copy.deepcopy(avg_state))

        if (round_num + 1) % 5 == 0:
            logger.info(f"  FL round {round_num+1}/{cfg.NUM_ROUNDS} complete")

    # Save all client models
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save([cm.state_dict() for cm in client_models], ckpt_path)
    logger.info(f"Saved personalized FL models → {ckpt_path}")

    for cm in client_models:
        cm.eval()
    return client_models


def train_standard_fedavg(cfg: AblationConfig, client_subsets: List[Subset],
                           ckpt_path: Path, logger: logging.Logger) -> nn.Module:
    """Train standard (non-personalized) FedAvg model."""
    if ckpt_path.exists():
        logger.info(f"Loading FedAvg model from {ckpt_path}")
        model = create_resnet18(cfg.NUM_CLASSES, pretrained=False).to(cfg.DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=cfg.DEVICE))
        model.eval()
        return model

    logger.info("Training standard FedAvg model (non-personalized)...")
    global_model = create_resnet18(cfg.NUM_CLASSES, pretrained=True).to(cfg.DEVICE)

    for round_num in range(cfg.NUM_ROUNDS):
        local_states = []

        for subset in client_subsets:
            local_model = copy.deepcopy(global_model)
            local_model.train()
            loader = DataLoader(subset, batch_size=cfg.BATCH_SIZE,
                                shuffle=True, num_workers=cfg.NUM_WORKERS,
                                pin_memory=(cfg.DEVICE == "cuda"))
            optimizer = optim.SGD(local_model.parameters(), lr=cfg.LEARNING_RATE,
                                  momentum=0.9, weight_decay=5e-4)

            for _ in range(cfg.LOCAL_EPOCHS):
                for imgs, labels in loader:
                    if imgs.size(0) < 2:
                        continue
                    imgs, labels = imgs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                    loss = F.cross_entropy(local_model(imgs), labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            local_states.append(local_model.state_dict())

        # FedAvg
        avg_state = {}
        for key in local_states[0].keys():
            tensors = [s[key].float() for s in local_states]
            avg_state[key] = torch.stack(tensors).mean(dim=0)
        global_model.load_state_dict(avg_state)

        if (round_num + 1) % 5 == 0:
            logger.info(f"  FedAvg round {round_num+1}/{cfg.NUM_ROUNDS} complete")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(global_model.state_dict(), ckpt_path)
    logger.info(f"Saved FedAvg model → {ckpt_path}")
    global_model.eval()
    return global_model


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_clean(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def evaluate_adversarial(
        model: nn.Module,
        loader: DataLoader,
        cfg: AblationConfig,
        mae_model: Optional[SimpleMAE],
        diffuser: Optional[DiffusionUNet],
        variant: str,
        logger: logging.Logger,
) -> Dict:
    """
    Evaluate adversarial accuracy under the given defense variant.

    Returns dict with keys:
        adv_acc, detection_precision, detection_recall, detection_f1,
        frac_purified, avg_latency_ms
    """
    model.eval()
    device = cfg.DEVICE

    correct = total = 0
    true_pos = false_pos = false_neg = true_neg = 0
    total_flagged = 0
    total_latency = 0.0
    n_batches = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        B = imgs.size(0)

        # Generate adversarial examples
        adv_imgs = pgd_attack(model, imgs, labels,
                              cfg.PGD_EPS, cfg.PGD_ALPHA, cfg.PGD_STEPS, device)

        t0 = time.perf_counter()

        # ── Apply defense variant ──────────────────────────────────────────
        if variant == "full_medfedpure":
            # Step 1: MAE detection
            errs = mae_model.reconstruction_error(adv_imgs)
            kappa = cfg.MAE_KAPPA
            thresh = float(torch.quantile(errs, 1 - kappa / 100.0))
            flagged = errs > thresh                              # bool [B]
            total_flagged += flagged.sum().item()

            # Step 2: Adaptive diffusion purification on flagged samples
            processed = adv_imgs.clone()
            if flagged.any():
                t_vals = adaptive_t(errs[flagged], cfg)
                for idx, (flag_idx, t_val) in enumerate(
                        zip(flagged.nonzero(as_tuple=True)[0], t_vals)):
                    single = adv_imgs[flag_idx].unsqueeze(0)
                    processed[flag_idx] = ddpm_purify(diffuser, single, t_val.item(), cfg).squeeze(0)

        elif variant == "no_mae_purify_all":
            # Purify every sample with fixed steps
            processed = ddpm_purify(diffuser, adv_imgs, cfg.DIFF_T_FIXED, cfg)
            flagged = torch.ones(B, dtype=torch.bool, device=device)
            total_flagged += B

        elif variant == "no_mae_no_purify":
            # No defense
            processed = adv_imgs
            flagged = torch.zeros(B, dtype=torch.bool, device=device)

        elif variant == "no_diffusion_mae_only":
            # Detect but do not purify
            errs = mae_model.reconstruction_error(adv_imgs)
            kappa = cfg.MAE_KAPPA
            thresh = float(torch.quantile(errs, 1 - kappa / 100.0))
            flagged = errs > thresh
            total_flagged += flagged.sum().item()
            processed = adv_imgs   # pass-through without purification

        elif variant == "no_personalization":
            # Same defense as full, but model is standard FedAvg (handled outside)
            errs = mae_model.reconstruction_error(adv_imgs)
            kappa = cfg.MAE_KAPPA
            thresh = float(torch.quantile(errs, 1 - kappa / 100.0))
            flagged = errs > thresh
            total_flagged += flagged.sum().item()

            processed = adv_imgs.clone()
            if flagged.any():
                t_vals = adaptive_t(errs[flagged], cfg)
                for flag_idx, t_val in zip(
                        flagged.nonzero(as_tuple=True)[0], t_vals):
                    single = adv_imgs[flag_idx].unsqueeze(0)
                    processed[flag_idx] = ddpm_purify(diffuser, single, t_val.item(), cfg).squeeze(0)

        elif variant == "fixed_purification":
            # MAE detection + fixed steps (no adaptive)
            errs = mae_model.reconstruction_error(adv_imgs)
            kappa = cfg.MAE_KAPPA
            thresh = float(torch.quantile(errs, 1 - kappa / 100.0))
            flagged = errs > thresh
            total_flagged += flagged.sum().item()

            processed = adv_imgs.clone()
            if flagged.any():
                adv_sub = adv_imgs[flagged]
                purified_sub = ddpm_purify(diffuser, adv_sub, cfg.DIFF_T_FIXED, cfg)
                processed[flagged] = purified_sub

        elif variant == "classifier_only":
            # No defense at all
            processed = adv_imgs
            flagged = torch.zeros(B, dtype=torch.bool, device=device)

        else:
            raise ValueError(f"Unknown variant: {variant}")

        t1 = time.perf_counter()
        total_latency += (t1 - t0) * 1000  # ms
        n_batches += 1

        # Classify
        with torch.no_grad():
            preds = model(processed).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += B

        # Detection metrics (ground truth: all adv_imgs are adversarial)
        gt_adv = torch.ones(B, dtype=torch.bool, device=device)
        true_pos  += (flagged & gt_adv).sum().item()
        false_pos += (flagged & ~gt_adv).sum().item()
        false_neg += (~flagged & gt_adv).sum().item()
        true_neg  += (~flagged & ~gt_adv).sum().item()

    adv_acc = 100.0 * correct / total if total > 0 else 0.0
    precision = true_pos / (true_pos + false_pos + 1e-9)
    recall    = true_pos / (true_pos + false_neg + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    frac_pur  = total_flagged / total if total > 0 else 0.0
    avg_lat   = total_latency / n_batches if n_batches > 0 else 0.0

    return {
        "adv_acc": adv_acc,
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": f1,
        "frac_purified": frac_pur,
        "avg_latency_ms": avg_lat,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main ablation runner
# ══════════════════════════════════════════════════════════════════════════════

VARIANTS = [
    "full_medfedpure",
    "no_mae_purify_all",
    "no_mae_no_purify",
    "no_diffusion_mae_only",
    "no_personalization",
    "fixed_purification",
    "classifier_only",
]

VARIANT_LABELS = {
    "full_medfedpure":       "Full MedFedPure",
    "no_mae_purify_all":     "No MAE (purify all)",
    "no_mae_no_purify":      "No MAE, No Purify",
    "no_diffusion_mae_only": "No Diffusion (MAE only)",
    "no_personalization":    "No Personalization",
    "fixed_purification":    "Fixed Purification",
    "classifier_only":       "Classifier Only",
}

# Which components are active per variant (for table display)
VARIANT_COMPONENTS = {
    "full_medfedpure":       dict(personalization=True,  mae=True,  diffusion=True,  adaptive=True),
    "no_mae_purify_all":     dict(personalization=True,  mae=False, diffusion=True,  adaptive=False),
    "no_mae_no_purify":      dict(personalization=True,  mae=False, diffusion=False, adaptive=False),
    "no_diffusion_mae_only": dict(personalization=True,  mae=True,  diffusion=False, adaptive=False),
    "no_personalization":    dict(personalization=False, mae=True,  diffusion=True,  adaptive=True),
    "fixed_purification":    dict(personalization=True,  mae=True,  diffusion=True,  adaptive=False),
    "classifier_only":       dict(personalization=True,  mae=False, diffusion=False, adaptive=False),
}


def run_ablation(cfg: AblationConfig, logger: logging.Logger) -> List[Dict]:
    set_seed(cfg.SEED)
    device = cfg.DEVICE
    logger.info(f"Dataset: {cfg.DATASET.upper()}  |  Device: {device}")

    # ── Load dataset ─────────────────────────────────────────────────────────
    logger.info("Loading dataset...")
    train_dataset, test_dataset = get_dataset(cfg)
    logger.info(f"  Train: {len(train_dataset)}  Test: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE,
                             shuffle=False, num_workers=cfg.NUM_WORKERS,
                             pin_memory=(device == "cuda"))

    # ── Non-IID split ─────────────────────────────────────────────────────────
    client_indices = dirichlet_split(train_dataset, cfg.NUM_CLIENTS,
                                     cfg.DIRICHLET_ALPHA, cfg.SEED)
    client_subsets = [Subset(train_dataset, idx) for idx in client_indices]

    # Full dataset loader for diffusion / MAE training
    full_loader = DataLoader(train_dataset, batch_size=cfg.DIFF_BATCH,
                             shuffle=True, num_workers=cfg.NUM_WORKERS,
                             pin_memory=(device == "cuda"))

    # ── Checkpoint paths ─────────────────────────────────────────────────────
    ckpt_dir = Path("checkpoints") / f"ablation_{cfg.DATASET.lower()}_seed{cfg.SEED}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    diff_ckpt = ckpt_dir / "diffusion.pt"
    mae_ckpt  = ckpt_dir / "mae_detector.pt"
    moe_ckpt  = ckpt_dir / "fl_moe_clients.pt"
    fedavg_ckpt = ckpt_dir / "fl_fedavg.pt"

    # ── Phase 1: Train diffusion model ───────────────────────────────────────
    diffuser = train_diffusion(cfg, full_loader, diff_ckpt, logger)

    # ── Phase 2: Train MAE detector ──────────────────────────────────────────
    mae_model = train_mae(cfg, full_loader, mae_ckpt, logger)

    # ── Phase 3a: Train personalized FL model (MoE) ──────────────────────────
    moe_clients = train_personalized_fl(cfg, client_subsets, moe_ckpt, logger)
    # Use first client model (or global) as representative for evaluation
    moe_model = moe_clients[0]
    moe_model.eval()

    # ── Phase 3b: Train standard FedAvg model ────────────────────────────────
    fedavg_model = train_standard_fedavg(cfg, client_subsets, fedavg_ckpt, logger)

    # ── Evaluation loop ───────────────────────────────────────────────────────
    results = []
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION EVALUATION")
    logger.info("=" * 70)

    for variant in VARIANTS:
        logger.info(f"\n[Variant] {VARIANT_LABELS[variant]}")

        # Choose which model to use
        if variant == "no_personalization":
            eval_model = fedavg_model
        else:
            eval_model = moe_model

        # Clean accuracy
        clean_acc = evaluate_clean(eval_model, test_loader, device)
        logger.info(f"  Clean Acc: {clean_acc:.2f}%")

        # Which defense components to pass
        use_mae  = VARIANT_COMPONENTS[variant]["mae"]
        use_diff = VARIANT_COMPONENTS[variant]["diffusion"]

        mae_arg  = mae_model  if use_mae  else None
        diff_arg = diffuser   if use_diff else None

        # Adversarial accuracy + metrics
        adv_metrics = evaluate_adversarial(
            eval_model, test_loader, cfg,
            mae_arg, diff_arg, variant, logger
        )
        logger.info(f"  Adv  Acc: {adv_metrics['adv_acc']:.2f}%  |  "
                    f"Det-F1: {adv_metrics['detection_f1']:.3f}  |  "
                    f"Purified: {adv_metrics['frac_purified']*100:.1f}%  |  "
                    f"Latency: {adv_metrics['avg_latency_ms']:.1f}ms/batch")

        comp = VARIANT_COMPONENTS[variant]
        row = {
            "dataset":          cfg.DATASET.upper(),
            "seed":             cfg.SEED,
            "variant":          variant,
            "label":            VARIANT_LABELS[variant],
            "personalization":  comp["personalization"],
            "mae_detector":     comp["mae"],
            "diffusion":        comp["diffusion"],
            "adaptive":         comp["adaptive"],
            "clean_acc":        round(clean_acc, 2),
            "adv_acc":          round(adv_metrics["adv_acc"], 2),
            "detection_f1":     round(adv_metrics["detection_f1"], 4),
            "frac_purified":    round(adv_metrics["frac_purified"], 4),
            "avg_latency_ms":   round(adv_metrics["avg_latency_ms"], 2),
        }
        results.append(row)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════════════════════

def save_results(results: List[Dict], out_dir: Path, dataset: str, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out_dir / f"ablation_{dataset.lower()}_seed{seed}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # CSV (Table A format from feedback.txt)
    csv_path = out_dir / f"ablation_{dataset.lower()}_seed{seed}.csv"
    fieldnames = [
        "label", "personalization", "mae_detector", "diffusion", "adaptive",
        "clean_acc", "adv_acc", "detection_f1", "frac_purified", "avg_latency_ms"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})

    # Human-readable summary
    summary_path = out_dir / f"ablation_{dataset.lower()}_seed{seed}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Ablation Study – {dataset.upper()} (seed={seed})\n")
        f.write("=" * 80 + "\n\n")
        header = f"{'Method':<30} {'P':>2} {'M':>2} {'D':>2} {'A':>2} "
        header += f"{'Clean%':>8} {'Adv%':>8} {'Det-F1':>8} {'Frac%':>7} {'ms/B':>8}"
        f.write(header + "\n")
        f.write("-" * 80 + "\n")
        for row in results:
            line = (
                f"{row['label']:<30} "
                f"{'Y' if row['personalization'] else 'N':>2} "
                f"{'Y' if row['mae_detector'] else 'N':>2} "
                f"{'Y' if row['diffusion'] else 'N':>2} "
                f"{'Y' if row['adaptive'] else 'N':>2} "
                f"{row['clean_acc']:>8.2f} "
                f"{row['adv_acc']:>8.2f} "
                f"{row['detection_f1']:>8.4f} "
                f"{row['frac_purified']*100:>7.1f} "
                f"{row['avg_latency_ms']:>8.1f}"
            )
            f.write(line + "\n")
        f.write("\nP=Personalization, M=MAE-Detector, D=Diffusion, A=Adaptive\n")

    return json_path, csv_path, summary_path


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="MedFedPure Ablation Study")
    p.add_argument("--dataset",  type=str, default="br35h",
                   choices=["br35h", "cifar10"])
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--rounds",   type=int, default=None,
                   help="Override NUM_ROUNDS")
    p.add_argument("--epochs",   type=int, default=None,
                   help="Override LOCAL_EPOCHS")
    p.add_argument("--clients",  type=int, default=None,
                   help="Override NUM_CLIENTS")
    p.add_argument("--out-dir",  type=str, default="experiment_results/ablation",
                   help="Output directory for results")
    p.add_argument("--fast",     action="store_true",
                   help="Quick test run (reduced epochs for CI)")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = AblationConfig()
    cfg.DATASET  = args.dataset
    cfg.DATA_ROOT = args.data_root
    cfg.SEED     = args.seed

    if args.rounds:
        cfg.NUM_ROUNDS = args.rounds
    if args.epochs:
        cfg.LOCAL_EPOCHS = args.epochs
    if args.clients:
        cfg.NUM_CLIENTS = args.clients

    if args.fast:
        cfg.NUM_ROUNDS   = 3
        cfg.LOCAL_EPOCHS = 2
        cfg.DIFF_EPOCHS  = 5
        cfg.MAE_EPOCHS   = 5
        cfg.NUM_CLIENTS  = 3

    cfg.finalize()

    out_dir = Path(args.out_dir)
    log_file = out_dir / f"ablation_{cfg.DATASET.lower()}_seed{cfg.SEED}.log"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("ablation", str(log_file))
    logger.info("=" * 70)
    logger.info("MedFedPure Ablation Study")
    logger.info(f"Dataset={cfg.DATASET}  Rounds={cfg.NUM_ROUNDS}  "
                f"Epochs={cfg.LOCAL_EPOCHS}  Clients={cfg.NUM_CLIENTS}")
    logger.info("=" * 70)

    results = run_ablation(cfg, logger)

    json_p, csv_p, sum_p = save_results(results, out_dir, cfg.DATASET, cfg.SEED)
    logger.info(f"\nResults saved:")
    logger.info(f"  JSON:    {json_p}")
    logger.info(f"  CSV:     {csv_p}")
    logger.info(f"  Summary: {sum_p}")

    # Print summary table
    logger.info("\n" + open(sum_p).read())


if __name__ == "__main__":
    main()
