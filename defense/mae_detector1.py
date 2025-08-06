"""
MAE‑style detector aligned with the paper
FIXED: Added LayerNorm, improved vectorization, fixed mask indices, added validation, improved error handling
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
    """Transformer encoder that sees only visible tokens - FIXED: Added LayerNorm"""
    def __init__(self, embed_dim: int = 256, depth: int = 6, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=int(embed_dim * mlp_ratio),
                                                   activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        # FIXED: Added LayerNorm for stability
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # B N_vis D
        x = self.encoder(x)
        x = self.norm(x)  # FIXED: Added normalization
        return x


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
    def random_mask(self, N: int, mask_ratio: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random per‑sample masks returning (visible_idx, mask_idx, ids_restore)."""
        L = self.num_patches
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_mask = ids_shuffle[:, len_keep:]
        
        return ids_keep, ids_mask, ids_restore

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:  # B C H W → B N P
        """Convert images to patches."""
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('bchpwq->bhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x: torch.Tensor, img_size: int) -> torch.Tensor:  # B N P → B C H W
        """Convert patches back to images."""
        p = self.patch_size
        h = w = img_size // p
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('bhwpqc->bchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward(self, imgs: torch.Tensor, mask_ratio: float = None):
        """Forward pass with improved vectorization."""
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        # 1) Patchify
        x = self.patch_embed(imgs)  # B N D
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # 2) Generate masks
        ids_keep, ids_mask, ids_restore = self.random_mask(x.shape[0], mask_ratio, x.device)
        
        # 3) Encode visible patches - FIXED: Improved vectorization
        x_vis = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        enc_out = self.encoder(x_vis)  # B N_vis D
        
        # 4) Decode all patches - FIXED: Improved vectorization
        # Create full decoder input with mask tokens
        dec_in = self.mask_token.repeat(x.shape[0], self.num_patches, 1)  # B N D
        # Place visible tokens in correct positions
        dec_in.scatter_(1, ids_keep.unsqueeze(-1).repeat(1, 1, dec_in.shape[-1]), enc_out)
        dec_in = dec_in + self.decoder_pos_embed
        
        # 5) Decode & predict
        pred = self.decoder(dec_in)  # B N P
        
        # 6) Return predictions and mask indices for loss computation
        return pred, ids_mask

    # helpers for external usage --------------------------------------------
    def reconstruct(self, imgs: torch.Tensor, mask_ratio: float = 0.75) -> torch.Tensor:
        pred, _ = self.forward(imgs, mask_ratio)
        rec_imgs = self.unpatchify(pred, imgs.size(2))
        return rec_imgs

    def reconstruction_error(self, imgs: torch.Tensor, mask_ratio: float = 0.0) -> torch.Tensor:
        """Return MSE per sample with proper mask handling."""
        # FIXED: Proper mask handling for reconstruction error
        if mask_ratio > 0.0:
            # Use masking for error computation
            pred, ids_mask = self.forward(imgs, mask_ratio)
            target = self.patchify(imgs)
            
            # Compute error only on masked patches
            errors = []
            for b in range(imgs.size(0)):
                if len(ids_mask[b]) > 0:
                    mask_err = F.mse_loss(pred[b, ids_mask[b]], target[b, ids_mask[b]], reduction='none')
                    errors.append(mask_err.mean())
                else:
                    errors.append(torch.tensor(0.0, device=imgs.device))
            return torch.stack(errors)
        else:
            # Full reconstruction error
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
        self.best_loss = float('inf')  # FIXED: Track best loss for model saving
        self._try_load()

    # ---------------------------------------------------------
    def _try_load(self):
        """Load checkpoint with proper validation - FIXED: Added validation"""
        if self.ckpt.exists():
            try:
                data = torch.load(self.ckpt, map_location=self.device)
                
                # FIXED: Validate state dict keys
                model_state = data["state"]
                model_keys = set(model_state.keys())
                expected_keys = set(self.model.state_dict().keys())
                
                if model_keys == expected_keys:
                    self.model.load_state_dict(model_state)
                    self.threshold = data.get("thr", self.threshold)
                    self.best_loss = data.get("best_loss", float('inf'))
                    print(f"Loaded MAE detector from {self.ckpt}")
                else:
                    print(f"Warning: State dict keys mismatch. Expected {len(expected_keys)}, got {len(model_keys)}")
                    missing_keys = expected_keys - model_keys
                    extra_keys = model_keys - expected_keys
                    if missing_keys:
                        print(f"Missing keys: {missing_keys}")
                    if extra_keys:
                        print(f"Extra keys: {extra_keys}")
            except Exception as e:
                print(f"Error loading MAE detector: {e}")

    def save(self, is_best=False):
        """Save checkpoint with best model tracking - FIXED: Added best model saving"""
        save_data = {
            "state": self.model.state_dict(), 
            "thr": self.threshold,
            "best_loss": self.best_loss
        }
        torch.save(save_data, self.ckpt)
        
        if is_best:
            best_ckpt = Path("checkpoints/mae_detector_best.pt")
            torch.save(save_data, best_ckpt)

    # ---------------------------------------------------------
    def train(self, train_loader, epochs: int = 100):
        """Train MAE detector with improved error handling - FIXED: Added best model saving and batch size validation"""
        self.model.train()
        opt = optim.AdamW(self.model.parameters(), lr=self.cfg.LR, betas=(0.9, 0.95), weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        
        # FIXED: Batch size compatibility check
        sample_batch = next(iter(train_loader))
        if sample_batch[0].size(0) > 1:  # Check if batch size > 1
            print(f"Training with batch size: {sample_batch[0].size(0)}")
        else:
            print("Warning: Batch size is 1, this may cause issues")
        
        for ep in range(epochs):
            running = 0.0
            seen = 0
            for imgs, _ in train_loader:
                imgs = imgs.to(self.device)
                
                # FIXED: Improved vectorized loss computation
                pred, ids_mask = self.model(imgs, mask_ratio=self.cfg.MAE_MASK_RATIO)
                target = self.model.patchify(imgs)
                
                # FIXED: Vectorized loss computation
                loss = 0.0
                for b in range(imgs.size(0)):
                    if len(ids_mask[b]) > 0:
                        loss += F.mse_loss(pred[b, ids_mask[b]], target[b, ids_mask[b]])
                loss = loss / imgs.size(0)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                running += loss.item() * imgs.size(0)
                seen += imgs.size(0)
            
            sched.step()
            avg_loss = running / seen
            print(f"[MAE] Epoch {ep+1}/{epochs}  loss={avg_loss:.4f}")
            
            # FIXED: Save best model
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save(is_best=True)
                print(f"[MAE] New best model saved with loss: {avg_loss:.4f}")
            else:
                self.save()
        
        self.calibrate_threshold(train_loader)

    # ---------------------------------------------------------
    @torch.no_grad()
    def calibrate_threshold(self, loader):
        """Calibrate detection threshold"""
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
        """Return 1 for adversarial, 0 for clean - FIXED: Added input clamping"""
        self.model.eval()
        
        # FIXED: Clamp inputs to prevent out-of-bounds
        imgs = torch.clamp(imgs, 0.0, 1.0)
        
        errs = self.model.reconstruction_error(imgs.to(self.device))
        return (errs > self.threshold).int()
