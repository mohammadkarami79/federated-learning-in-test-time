"""
MAE‑style detector aligned with the paper
"""
import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -------------------------------------------------------------
# ----------------------  CORE ARCHITECTURE  ------------------
# -------------------------------------------------------------
class PatchEmbed(nn.Module):
    """Split image into patches and embed (like ViT/MAE).
    Args:
        img_size: int – assumed square input
        patch_size: int – side length of each square patch
        in_chans: number of input channels
        embed_dim: token dimension
    Output: (B, N, D) where N = num_patches
    """
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # B C H W → B N D
        x = self.proj(x)                      # B D H/P W/P
        x = x.flatten(2).transpose(1, 2)      # B N D
        return x


class MAEEncoder(nn.Module):
    """Transformer encoder that sees only visible tokens"""
    def __init__(self, embed_dim: int = 256, depth: int = 6, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=int(embed_dim * mlp_ratio),
                                                   activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # B N_vis D
        return self.encoder(x)


class MAEDecoder(nn.Module):
    """Transformer decoder that reconstructs all tokens (visible + mask tokens)"""
    def __init__(self, embed_dim: int = 256, dec_dim: int = 128, depth: int = 4, num_heads: int = 8,
                 patch_dim: int = 3 * 4 * 4):
        super().__init__()
        self.proj_vis = nn.Linear(embed_dim, dec_dim) if embed_dim != dec_dim else nn.Identity()
        decoder_layer = nn.TransformerEncoderLayer(d_model=dec_dim, nhead=num_heads,
                                                   dim_feedforward=int(dec_dim * 4),
                                                   activation="gelu", batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        self.pred = nn.Linear(dec_dim, patch_dim)  # project tokens → pixel values

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # B N D → B N P
        x = self.proj_vis(x)
        x = self.decoder(x)
        x = self.pred(x)
        return x


class MAE(nn.Module):
    """Masked Autoencoder (vision‑transformer style) for small images (e.g., 32×32)."""
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3,
                 embed_dim: int = 256, depth: int = 6, num_heads: int = 8,
                 decoder_dim: int = 128, decoder_depth: int = 4, mask_ratio: float = 0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.patch_dim = in_chans * patch_size * patch_size
        self.embed_dim = embed_dim

        # modules
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.encoder = MAEEncoder(embed_dim, depth, num_heads)

        # mask token & decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.decoder = MAEDecoder(embed_dim, decoder_dim, decoder_depth, num_heads, self.patch_dim)

    # utility functions ------------------------------------------------------
    def random_mask(self, N: int, mask_ratio: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random per‑sample masks returning (visible_idx, mask_idx)."""
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(N, device=device)  # 1‑D noise per token → permute
        ids_shuffle = torch.argsort(noise)    # ascending
        ids_keep = ids_shuffle[:len_keep]
        ids_mask = ids_shuffle[len_keep:]
        return ids_keep, ids_mask

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:  # B C H W → B N P
        p = self.patch_size
        B, C, H, W = imgs.shape
        imgs = imgs.view(B, C, H // p, p, W // p, p)
        imgs = imgs.permute(0, 2, 4, 3, 5, 1).contiguous()  # B h w p p C
        return imgs.view(B, -1, self.patch_dim)

    def unpatchify(self, x: torch.Tensor, img_size: int) -> torch.Tensor:  # B N P → B C H W
        p = self.patch_size
        B, N, _ = x.shape
        h = w = int(math.sqrt(N))
        x = x.view(B, h, w, p, p, -1)  # B h w p p C
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(B, -1, img_size, img_size)

    # ------------------------------------------------------------------------
    def forward(self, imgs: torch.Tensor, mask_ratio: float = None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        device = imgs.device
        B = imgs.size(0)

        # 1) Patchify & embed
        x = self.patch_embed(imgs)  # B N D
        x = x + self.pos_embed

        # 2) Generate per‑sample mask & select visible tokens
        ids_keep_list, ids_mask_list = [], []
        x_vis_list = []
        for b in range(B):
            ids_keep, ids_mask = self.random_mask(self.num_patches, mask_ratio, device)
            ids_keep_list.append(ids_keep)
            ids_mask_list.append(ids_mask)
            x_vis_list.append(x[b, ids_keep])
        x_vis = torch.stack([F.pad(t, (0, 0, 0, self.num_patches - t.size(0)), "constant", 0) for t in x_vis_list])

        # 3) Encode visible tokens
        enc_out = []
        for b in range(B):
            enc_out.append(self.encoder(x_vis_list[b].unsqueeze(0)))
        enc_out = torch.cat(enc_out, dim=0)  # B len_keep D

        # 4) Prepare decoder input (visible + mask tokens) per sample
        dec_in = []
        for b in range(B):
            ids_keep, ids_mask = ids_keep_list[b], ids_mask_list[b]
            # start with all mask tokens
            tokens = self.mask_token.repeat(self.num_patches, 1)  # N D
            tokens[ids_keep] = enc_out[b]
            dec_in.append(tokens.unsqueeze(0))
        dec_in = torch.cat(dec_in, dim=0)  # B N D
        dec_in = dec_in + self.decoder_pos_embed

        # 5) Decode & predict
        pred = self.decoder(dec_in)  # B N P

        # 6) Compute per‑sample loss on masked patches (if target given later)
        return pred, ids_mask_list

    # helpers for external usage --------------------------------------------
    def reconstruct(self, imgs: torch.Tensor, mask_ratio: float = 0.75) -> torch.Tensor:
        pred, _ = self.forward(imgs, mask_ratio)
        rec_imgs = self.unpatchify(pred, imgs.size(2))
        return rec_imgs

    def reconstruction_error(self, imgs: torch.Tensor, mask_ratio: float = 0.0) -> torch.Tensor:
        """Return MSE per sample (no masking → full input)."""
        rec = self.reconstruct(imgs, mask_ratio)
        err = (rec - imgs).pow(2).mean(dim=(1, 2, 3))
        return err

# -------------------------------------------------------------
# ------------------------  DETECTOR  -------------------------
# -------------------------------------------------------------
class MAEDetector:
    """MAE‑based adversarial detector using reconstruction error."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.DEVICE)
        # architecture hyper‑params chosen for CIFAR‑like images (32×32)
        self.model = MAE(img_size=cfg.IMG_SIZE, patch_size=cfg.PATCH_SIZE,
                         embed_dim=cfg.MAE_DIM, depth=cfg.MAE_DEPTH, num_heads=cfg.MAE_HEADS,
                         decoder_dim=cfg.MAE_DEC_DIM, decoder_depth=cfg.MAE_DEC_DEPTH,
                         mask_ratio=cfg.MAE_MASK_RATIO).to(self.device)
        self.threshold = cfg.MAE_THRESHOLD
        self.ckpt = Path("checkpoints/mae_detector.pt")
        self.ckpt.parent.mkdir(exist_ok=True)
        self._try_load()

    # ---------------------------------------------------------
    def _try_load(self):
        if self.ckpt.exists():
            data = torch.load(self.ckpt, map_location=self.device)
            self.model.load_state_dict(data["state"])
            self.threshold = data.get("thr", self.threshold)

    def save(self):
        torch.save({"state": self.model.state_dict(), "thr": self.threshold}, self.ckpt)

    # ---------------------------------------------------------
    def train(self, train_loader, epochs: int = 100):
        self.model.train()
        opt = optim.AdamW(self.model.parameters(), lr=self.cfg.LR, betas=(0.9, 0.95), weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for ep in range(epochs):
            running = 0.0
            seen = 0
            for imgs, _ in train_loader:
                imgs = imgs.to(self.device)
                pred, ids_mask_list = self.model(imgs, mask_ratio=self.cfg.MAE_MASK_RATIO)
                target = self.model.patchify(imgs)

                loss = 0.0
                for b in range(imgs.size(0)):
                    ids_mask = ids_mask_list[b]
                    loss += F.mse_loss(pred[b, ids_mask], target[b, ids_mask])
                loss = loss / imgs.size(0)

                opt.zero_grad(); loss.backward(); opt.step()
                running += loss.item() * imgs.size(0)
                seen += imgs.size(0)
            sched.step()
            print(f"[MAE] Epoch {ep+1}/{epochs}  loss={running/seen:.4f}")
            self.save()
        self.calibrate_threshold(train_loader)

    # ---------------------------------------------------------
    @torch.no_grad()
    def calibrate_threshold(self, loader):
        self.model.eval()
        errs = []
        for imgs, _ in loader:
            imgs = imgs.to(self.device)
            errs.append(self.model.reconstruction_error(imgs))
        errs = torch.cat(errs)
        self.threshold = errs.mean().item() + 2 * errs.std().item()
        print(f"[MAE] Calibrated threshold -> {self.threshold:.6f}")
        self.save()

    # ---------------------------------------------------------
    @torch.no_grad()
    def detect(self, imgs: torch.Tensor) -> torch.Tensor:
        """Return 1 for adversarial, 0 for clean."""
        self.model.eval()
        errs = self.model.reconstruction_error(imgs.to(self.device))
        return (errs > self.threshold).int()
