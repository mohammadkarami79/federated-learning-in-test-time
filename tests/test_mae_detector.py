"""
Unit tests for MAE detector
"""

import pytest
import torch
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from defense.mae_detector import MAEDetector, MAEModel
from utils.data_utils import get_dataloader
from attacks.pgd import PGDAttack

class TestMAEDetector:
    @pytest.fixture
    def cfg(self):
        """Get debug configuration"""
        return get_config('debug')
    
    @pytest.fixture
    def detector(self, cfg):
        """Initialize MAE detector"""
        return MAEDetector(cfg)
    
    @pytest.fixture
    def dataloader(self, cfg):
        """Get test dataloader"""
        return get_dataloader(cfg, split="test")
    
    @pytest.fixture
    def sample_images(self, dataloader):
        """Get sample images from dataloader"""
        data, _ = next(iter(dataloader))
        return data[:4]  # Get first 4 images
    
    def test_mae_model_initializes(self):
        """Test MAE model initialization"""
        model = MAEModel(in_channels=3, latent_dim=64)
        assert isinstance(model, torch.nn.Module)
        
        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        recon, mask = model(x, mask_ratio=0.5)
        
        # Check output shape
        assert recon.shape == x.shape
        assert mask.shape[0] == x.shape[0]
        assert mask.shape[1] == 1
    
    def test_mae_detector_initializes(self, detector):
        """Test MAE detector initialization"""
        assert detector.model is not None
        assert detector.threshold == detector.cfg.MAE_THRESHOLD
    
    def test_reconstruction_error(self, detector, sample_images):
        """Test reconstruction error computation"""
        # Get reconstruction error for sample images
        error = detector.model.compute_reconstruction_error(sample_images.to(detector.device))
        
        # Check error shape and basic properties
        assert error.shape[0] == sample_images.shape[0]
        assert torch.all(error >= 0)  # MSE is non-negative
        
        print(f"Reconstruction errors: {error}")
    
    def test_detection(self, detector, sample_images):
        """Test detection functionality"""
        # Create adversarial examples
        attack = PGDAttack(epsilon=8/255, steps=5, step_size=2/255)
        model = detector.model.encoder  # Use encoder as target model
        
        # Create adversarial versions of sample images
        sample_clean = sample_images.to(detector.device)
        sample_adv = attack.perturb(model, sample_clean, torch.zeros(sample_clean.shape[0]).long().to(detector.device))
        
        # Detect adversarial examples
        is_adv_clean = detector.detect(sample_clean)
        is_adv_perturbed = detector.detect(sample_adv)
        
        # Print detection results
        print(f"Clean detection: {is_adv_clean}")
        print(f"Adversarial detection: {is_adv_perturbed}")
        
        # There should be more adversarial detections in perturbed images
        assert is_adv_perturbed.sum() >= is_adv_clean.sum()
    
    @pytest.mark.slow
    def test_training(self, detector, cfg):
        """Test training function (marked as slow)"""
        # Train for just 1 step to verify it works
        detector.train(epochs=1)
        
        # Verify checkpoint was saved
        assert detector.checkpoint_path.exists() 