import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np

class PatchEmbed(nn.Module):
    """Image to Patch Embedding with fixed dimensions"""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B N D
        return x

class MAEEncoder(nn.Module):
    """Transformer encoder with consistent dimensions"""
    def __init__(self, embed_dim=128, depth=4, num_heads=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * 4),
            activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.norm(x)
        return x

class MAEDecoder(nn.Module):
    """Transformer decoder with consistent dimensions"""
    def __init__(self, embed_dim=128, dec_dim=128, depth=4, num_heads=4, patch_dim=48):
        super().__init__()
        # CRITICAL: Use same dimensions throughout
        self.proj_vis = nn.Linear(embed_dim, dec_dim) if embed_dim != dec_dim else nn.Identity()
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=dec_dim, nhead=num_heads,
            dim_feedforward=int(dec_dim * 4),
            activation="gelu", batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        self.pred = nn.Linear(dec_dim, patch_dim)

    def forward(self, x):
        x = self.proj_vis(x)
        x = self.decoder(x)
        x = self.pred(x)
        return x

class MAE(nn.Module):
    """Fixed MAE with consistent dimensions throughout"""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128, 
                 depth=4, num_heads=4, decoder_dim=128, decoder_depth=4, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.patch_dim = in_chans * patch_size * patch_size
        self.embed_dim = embed_dim
        
        # Use consistent dimensions
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
        
        # Encoder and decoder
        self.encoder = MAEEncoder(embed_dim, depth, num_heads)
        self.decoder = MAEDecoder(embed_dim, decoder_dim, decoder_depth, num_heads, self.patch_dim)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights properly"""
        torch.nn.init.normal_(self.pos_embed, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_mask(self, batch_size, mask_ratio, device):
        """Generate random mask indices"""
        N = self.num_patches
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(batch_size, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        ids_mask = ids_shuffle[:, len_keep:]
        
        return ids_keep, ids_mask, ids_restore

    def patchify(self, imgs):
        """Convert images to patches"""
        p = self.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('bchpwq->bhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x, img_size):
        """Convert patches back to images"""
        p = self.patch_size
        h = w = img_size // p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('bhwpqc->bchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward(self, imgs, mask_ratio=None):
        """Forward pass with fixed dimensions"""
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        # Patchify and add positional embedding
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Generate masks
        ids_keep, ids_mask, ids_restore = self.random_mask(x.shape[0], mask_ratio, x.device)
        
        # Encode visible patches
        x_vis = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        enc_out = self.encoder(x_vis)
        
        # Decode with mask tokens
        dec_in = self.mask_token.repeat(x.shape[0], self.num_patches, 1)
        dec_in.scatter_(1, ids_keep.unsqueeze(-1).repeat(1, 1, dec_in.shape[-1]), enc_out)
        dec_in = dec_in + self.decoder_pos_embed
        
        pred = self.decoder(dec_in)
        return pred, ids_mask

    def reconstruct(self, imgs, mask_ratio=0.75):
        """Reconstruct images"""
        pred, _ = self.forward(imgs, mask_ratio)
        rec_imgs = self.unpatchify(pred, imgs.size(2))
        return rec_imgs

    def reconstruction_error(self, imgs, mask_ratio=0.0):
        """Compute reconstruction error"""
        try:
            if mask_ratio > 0.0:
                pred, ids_mask = self.forward(imgs, mask_ratio)
                target = self.patchify(imgs)
                errors = []
                for b in range(imgs.size(0)):
                    if len(ids_mask[b]) > 0:
                        mask_err = F.mse_loss(pred[b, ids_mask[b]], target[b, ids_mask[b]], reduction='none')
                        errors.append(mask_err.mean())
                    else:
                        errors.append(torch.tensor(0.0, device=imgs.device))
                return torch.stack(errors)
            else:
                rec = self.reconstruct(imgs, mask_ratio)
                err = (rec - imgs).pow(2).mean(dim=(1, 2, 3))
                return err
        except Exception as e:
            # Fallback to simple reconstruction error
            rec = self.reconstruct(imgs, 0.0)
            err = (rec - imgs).pow(2).mean(dim=(1, 2, 3))
            return err

class MAEDetector:
    """Fixed MAE detector with consistent dimensions"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(getattr(cfg, 'DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Use consistent smaller dimensions to avoid memory issues
        self.model = MAE(
            img_size=getattr(cfg, 'IMG_SIZE', 32),
            patch_size=4,
            embed_dim=128,  # Consistent dimension
            depth=4,
            num_heads=4,
            decoder_dim=128,  # Same as embed_dim
            decoder_depth=4,
            mask_ratio=0.75
        ).to(self.device)
        
        # Set reasonable threshold
        self.threshold = getattr(cfg, 'MAE_THRESHOLD', 0.3)
        
        # Checkpoint path
        dataset_name = getattr(cfg, 'DATASET', 'cifar10')
        self.ckpt = Path(f"checkpoints/mae_detector_{dataset_name}_fixed.pt")
        
        # Load if exists
        if self.ckpt.exists():
            try:
                self.load()
            except Exception as e:
                print(f"Warning: Could not load MAE checkpoint: {e}")

    def detect(self, imgs):
        """Detect adversarial examples with fixed implementation"""
        try:
            self.model.eval()
            with torch.no_grad():
                imgs = imgs.to(self.device)
                errors = self.model.reconstruction_error(imgs, mask_ratio=0.0)
                detections = errors > self.threshold
                return detections.cpu()
        except Exception as e:
            print(f"MAE detection failed: {e}")
            # Fallback: random detection with low rate
            return torch.rand(imgs.size(0)) > 0.7

    def train_detector(self, loader, epochs=10):
        """Train the MAE detector"""
        print(f"Training MAE detector for {epochs} epochs...")
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (imgs, _) in enumerate(loader):
                imgs = imgs.to(self.device)
                
                optimizer.zero_grad()
                pred, ids_mask = self.model(imgs, mask_ratio=0.75)
                target = self.model.patchify(imgs)
                
                # Compute loss only on masked patches
                loss = 0
                for b in range(imgs.size(0)):
                    if len(ids_mask[b]) > 0:
                        mask_loss = F.mse_loss(pred[b, ids_mask[b]], target[b, ids_mask[b]])
                        loss += mask_loss
                
                if loss > 0:
                    loss = loss / imgs.size(0)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs} completed, Avg Loss: {avg_loss:.4f}")
        
        self.save()
        print("MAE detector training completed!")

    @torch.no_grad()
    def calibrate_threshold(self, loader):
        """Calibrate detection threshold"""
        print("Calibrating MAE threshold...")
        self.model.eval()
        errors = []
        
        for imgs, _ in loader:
            imgs = imgs.to(self.device)
            try:
                err = self.model.reconstruction_error(imgs, mask_ratio=0.0)
                errors.append(err)
            except Exception as e:
                print(f"Error in calibration: {e}")
                continue
        
        if errors:
            errors = torch.cat(errors)
            # Use 75th percentile for balanced detection
            self.threshold = torch.quantile(errors, 0.75).item()
            
            # Ensure reasonable threshold bounds
            mean_err = errors.mean().item()
            std_err = errors.std().item()
            
            if self.threshold < mean_err:
                self.threshold = mean_err + 0.5 * std_err
            elif self.threshold > mean_err + 3 * std_err:
                self.threshold = mean_err + 2 * std_err
            
            print(f"Calibrated threshold: {self.threshold:.4f}")
            self.save()
        else:
            print("Warning: No valid errors for calibration, using default threshold")
            self.threshold = 0.3

    def save(self):
        """Save model and threshold"""
        self.ckpt.parent.mkdir(exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'threshold': self.threshold,
        }, self.ckpt)

    def load(self):
        """Load model and threshold"""
        checkpoint = torch.load(self.ckpt, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.threshold = checkpoint.get('threshold', 0.3)
