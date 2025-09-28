"""
ENHANCED MAE DETECTOR WITH PROPER RECONSTRUCTION
==============================================
Implements actual MAE reconstruction for adversarial detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

class EnhancedMAEDetector(nn.Module):
    """Enhanced MAE detector with proper reconstruction-based detection"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = getattr(cfg, 'DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # MAE architecture parameters
        self.img_size = getattr(cfg, 'IMG_SIZE', 32)
        self.patch_size = 4
        self.embed_dim = getattr(cfg, 'MAE_EMBED_DIM', 128)
        self.num_patches = (self.img_size // self.patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        # Transformer encoder (simplified)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=8,
                dim_feedforward=self.embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )
        
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 2, self.patch_size * self.patch_size * 3)
        )
        
        # Detection threshold
        self.threshold = getattr(cfg, 'MAE_THRESHOLD', 0.08)
        
    def forward(self, x):
        """Forward pass for training"""
        batch_size = x.size(0)
        
        # Patch embedding
        patches = self.patch_embed(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add positional encoding (simplified)
        pos_embed = torch.randn(1, self.num_patches, self.embed_dim, device=x.device) * 0.02
        patches = patches + pos_embed
        
        # Encoder
        encoded = self.encoder(patches)
        
        # Decoder
        reconstructed = self.decoder(encoded)  # [B, num_patches, patch_size^2 * 3]
        
        # Reshape to image format
        reconstructed = reconstructed.view(batch_size, self.num_patches, 3, self.patch_size, self.patch_size)
        
        return reconstructed
    
    def detect_adversarial(self, x):
        """Detect adversarial examples using reconstruction error"""
        self.eval()
        batch_size = x.size(0)
        
        with torch.no_grad():
            # Get original patches
            original_patches = self.patch_embed(x)
            original_patches = original_patches.flatten(2).transpose(1, 2)
            
            # Reconstruct
            reconstructed = self.forward(x)
            
            # Calculate reconstruction error per sample
            recon_errors = []
            for i in range(batch_size):
                # Convert patches back to image space for comparison
                orig_img = x[i]
                recon_patches = reconstructed[i]  # [num_patches, 3, patch_size, patch_size]
                
                # Reconstruct image from patches
                recon_img = torch.zeros_like(orig_img)
                patch_idx = 0
                for h in range(0, self.img_size, self.patch_size):
                    for w in range(0, self.img_size, self.patch_size):
                        recon_img[:, h:h+self.patch_size, w:w+self.patch_size] = recon_patches[patch_idx]
                        patch_idx += 1
                
                # Calculate MSE error
                mse_error = F.mse_loss(orig_img, recon_img).item()
                recon_errors.append(mse_error)
            
            recon_errors = torch.tensor(recon_errors, device=x.device)
            
            # Detect based on threshold
            is_adversarial = recon_errors > self.threshold
            
            return is_adversarial, recon_errors
    
    def detect(self, x):
        """Main detection interface"""
        is_adv, errors = self.detect_adversarial(x)
        return is_adv

# Create detector factory
def create_enhanced_mae_detector(cfg):
    """Factory function to create enhanced MAE detector"""
    return EnhancedMAEDetector(cfg)
