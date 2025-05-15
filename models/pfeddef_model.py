"""
pFedDef model implementation
"""

import torch
import torch.nn as nn
from typing import List, Optional

class pFedDefModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_learners = cfg.N_LEARNERS
        
        # Get width multiplier from config
        self.width_multiplier = getattr(cfg, 'RESNET_WIDTH', 1.0)
        
        # Create ensemble of learners
        self.learners = nn.ModuleList([
            self._create_learner() for _ in range(self.n_learners)
        ])
        
        # Initialize mixture weights
        self.mixture_weights = nn.Parameter(
            torch.ones(self.n_learners) / self.n_learners
        )
        
    def _create_learner(self) -> nn.Module:
        """Create a single learner model with configurable width."""
        # Scale channel dimensions by width multiplier
        base_channels = int(64 * self.width_multiplier)
        mid_channels = int(128 * self.width_multiplier)
        hidden_dim = int(512 * self.width_multiplier)
        
        # Ensure minimum channel counts
        base_channels = max(16, base_channels)
        mid_channels = max(32, mid_channels)
        hidden_dim = max(64, hidden_dim)
        
        return nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(mid_channels * 8 * 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )
        
    def forward(self, x: torch.Tensor, client_id: Optional[int] = None) -> torch.Tensor:
        """Forward pass through ensemble.
        
        Args:
            x: Input tensor
            client_id: Client ID (optional)
            
        Returns:
            Ensemble predictions
        """
        # Get predictions from each learner
        predictions = []
        for learner in self.learners:
            pred = learner(x)
            predictions.append(pred)
            
        # Stack predictions
        stacked_preds = torch.stack(predictions, dim=0)  # [n_learners, batch_size, n_classes]
        
        # Apply mixture weights
        weights = torch.softmax(self.mixture_weights, dim=0)
        weighted_preds = (stacked_preds * weights.view(-1, 1, 1)).sum(dim=0)
        
        return weighted_preds 