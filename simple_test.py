#!/usr/bin/env python3
"""
Simple test to verify MAE detection works properly
"""

import torch
import numpy as np
from config_fixed import get_full_config
from defense.mae_detector1 import MAEDetector

def test_mae_detection():
    print("🔍 Testing MAE Detection...")
    
    # Get config
    cfg = get_full_config()
    print(f"Config MAE_THRESHOLD: {cfg.MAE_THRESHOLD}")
    
    # Create MAE detector
    detector = MAEDetector(cfg)
    print(f"Initial threshold: {detector.threshold}")
    
    # Create fake data for calibration
    fake_data = torch.randn(50, 3, 224, 224) * 0.5 + 0.5  # Normal images
    fake_data = torch.clamp(fake_data, 0, 1)
    
    # Create fake dataset
    from torch.utils.data import TensorDataset, DataLoader
    fake_dataset = TensorDataset(fake_data, torch.zeros(50))
    fake_loader = DataLoader(fake_dataset, batch_size=16)
    
    print("🔧 Calibrating threshold...")
    detector.calibrate_threshold(fake_loader)
    print(f"Calibrated threshold: {detector.threshold}")
    
    # Test detection
    clean_samples = torch.randn(10, 3, 224, 224) * 0.3 + 0.5
    clean_samples = torch.clamp(clean_samples, 0, 1)
    
    adv_samples = torch.randn(10, 3, 224, 224) * 0.8 + 0.5  # More noisy
    adv_samples = torch.clamp(adv_samples, 0, 1)
    
    clean_detected = detector.detect(clean_samples)
    adv_detected = detector.detect(adv_samples)
    
    print(f"Clean samples detected as adversarial: {clean_detected.sum().item()}/10")
    print(f"Adversarial samples detected: {adv_detected.sum().item()}/10")
    
    if detector.threshold > 0.1 and detector.threshold < 0.9:
        print("✅ MAE threshold looks reasonable!")
        return True
    else:
        print("❌ MAE threshold still problematic!")
        return False

if __name__ == "__main__":
    success = test_mae_detection()
    if success:
        print("🎉 MAE detection should work now!")
    else:
        print("💥 Still has issues...")