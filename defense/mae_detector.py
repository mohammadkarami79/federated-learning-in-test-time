"""
MAE Detector for adversarial detection
Compatible with user's mae_detector1.py implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MAEDetector:
    """MAE-based adversarial detector"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(getattr(cfg, 'DEVICE', 'cuda'))
        self.threshold = getattr(cfg, 'MAE_THRESHOLD', 0.1)
        
        # Try to use user's implementation
        self.user_detector = None
        try:
            from mae_detector1 import MAEDetector as UserMAEDetector
            # Create a compatible config for user's detector
            user_cfg = self._create_compatible_config(cfg)
            self.user_detector = UserMAEDetector(user_cfg)
            logger.info("Using user's MAE detector implementation")
        except ImportError:
            logger.info("User's MAE detector not available, using simple implementation")
        except Exception as e:
            logger.warning(f"Error loading user's MAE detector: {e}")
        
        # Fallback simple model
        self.model = SimpleMAEModel(cfg).to(self.device)
    
    def _create_compatible_config(self, cfg):
        """Create a config compatible with user's MAE detector"""
        import types
        user_cfg = types.SimpleNamespace()
        
        # Map our config attributes to what user's detector expects
        user_cfg.DEVICE = getattr(cfg, 'DEVICE', 'cuda')
        user_cfg.PATCH_SIZE = getattr(cfg, 'MAE_PATCH_SIZE', 4)
        user_cfg.EMBED_DIM = getattr(cfg, 'MAE_EMBED_DIM', 128)
        user_cfg.MAE_DIM = getattr(cfg, 'MAE_EMBED_DIM', 128)  # Alias for EMBED_DIM
        user_cfg.DEPTH = getattr(cfg, 'MAE_DEPTH', 4)
        user_cfg.MAE_DEPTH = getattr(cfg, 'MAE_DEPTH', 4)  # Alias for DEPTH
        user_cfg.MAE_DEC_DEPTH = getattr(cfg, 'MAE_DEPTH', 4)  # Decoder depth
        user_cfg.NUM_HEADS = getattr(cfg, 'MAE_NUM_HEADS', 4)
        user_cfg.MAE_HEADS = getattr(cfg, 'MAE_NUM_HEADS', 4)  # Alias for NUM_HEADS
        user_cfg.MASK_RATIO = getattr(cfg, 'MAE_MASK_RATIO', 0.5)
        user_cfg.MAE_MASK_RATIO = getattr(cfg, 'MAE_MASK_RATIO', 0.5)  # Alias for mask ratio
        user_cfg.DECODER_EMBED_DIM = getattr(cfg, 'MAE_DECODER_EMBED_DIM', 64)
        user_cfg.MAE_DEC_DIM = getattr(cfg, 'MAE_DECODER_EMBED_DIM', 64)  # Alias for decoder dim
        user_cfg.THRESHOLD = getattr(cfg, 'MAE_THRESHOLD', 0.1)
        user_cfg.MAE_THRESHOLD = getattr(cfg, 'MAE_THRESHOLD', 0.1)  # Alias for threshold
        user_cfg.LEARNING_RATE = getattr(cfg, 'LEARNING_RATE', 0.001)
        user_cfg.N_CLASSES = getattr(cfg, 'N_CLASSES', 10)
        user_cfg.IMG_SIZE = getattr(cfg, 'IMG_SIZE', 32)
        user_cfg.IMG_CHANNELS = getattr(cfg, 'IMG_CHANNELS', 3)
        
        # Additional attributes that might be expected
        user_cfg.BATCH_SIZE = getattr(cfg, 'BATCH_SIZE', 32)
        user_cfg.EPOCHS = getattr(cfg, 'N_ROUNDS', 10)
        user_cfg.DATASET = getattr(cfg, 'DATASET', 'cifar10')
        
        return user_cfg
        
    def detect(self, images: torch.Tensor) -> torch.Tensor:
        """Detect adversarial examples
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            Binary tensor [B] where 1 = adversarial, 0 = clean
        """
        # Try user's detector first
        if self.user_detector is not None:
            try:
                return self.user_detector.detect(images)
            except Exception as e:
                logger.warning(f"User's MAE detector failed: {e}")
        
        # Fallback to simple detection
        try:
            with torch.no_grad():
                self.model.eval()
                recon, _ = self.model(images)
                mse = F.mse_loss(recon, images, reduction='none')
                mse = mse.view(images.size(0), -1).mean(dim=1)
                return (mse > self.threshold).int()
        except Exception as e:
            logger.warning(f"MAE detection failed: {e}")
            # Return all clean as safe fallback
            return torch.zeros(images.size(0), dtype=torch.int, device=images.device)
    
    def train(self, train_loader, epochs: int = 1):
        """Train the MAE detector"""
        if self.user_detector is not None:
            try:
                self.user_detector.train(train_loader, epochs)
                return
            except Exception as e:
                logger.warning(f"User's MAE detector training failed: {e}")
        
        # Simple training for fallback model
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                if batch_idx >= 10:  # Limit training for speed
                    break
                data = data.to(self.device)
                optimizer.zero_grad()
                recon, _ = self.model(data)
                loss = F.mse_loss(recon, data)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(f"MAE Detector - Epoch {epoch+1}/{epochs}: Loss = {total_loss:.4f}")
    
    def save(self):
        """Save the detector"""
        if self.user_detector is not None and hasattr(self.user_detector, 'save'):
            try:
                self.user_detector.save()
            except Exception as e:
                logger.warning(f"Failed to save user's MAE detector: {e}")

class SimpleMAEModel(nn.Module):
    """Simple MAE model for testing purposes"""
    
    def __init__(self, cfg):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4)
        )
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, mask_ratio=0.5):
        """Forward pass with optional masking"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        if decoded.shape != x.shape:
            decoded = F.interpolate(decoded, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        mask = torch.ones(x.size(0), 1, device=x.device)
        return decoded, mask

class MAEModel(SimpleMAEModel):
    """Alias for backward compatibility"""
    pass

# For backward compatibility with existing imports
mae_detector = MAEDetector 