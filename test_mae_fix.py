#!/usr/bin/env python3
"""
Quick test to verify MAE detection fix
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_mae_detection():
    """Test MAE detection with BR35H config"""
    
    # Import config
    from config_br35h_final import get_config
    cfg_dict = get_config()
    
    # Convert to namespace
    import types
    cfg_obj = types.SimpleNamespace()
    for key, value in cfg_dict.items():
        setattr(cfg_obj, key, value)
    
    # Import MAE detector
    from defense.mae_detector import MAEDetector
    
    print("🔍 Testing MAE Detection Fix...")
    print(f"Config MAE_THRESHOLD: {cfg_obj.MAE_THRESHOLD}")
    print(f"Config TARGET_DETECTION_RATE: {cfg_obj.TARGET_DETECTION_RATE}")
    
    # Create MAE detector
    mae_detector = MAEDetector(cfg_obj)
    
    # Create test data (simulate BR35H images)
    batch_size = 32
    test_images = torch.randn(batch_size, 3, 224, 224)
    test_images = torch.clamp(test_images, 0.0, 1.0)  # Normalize to [0,1]
    
    print(f"\n📊 Test Data: {batch_size} samples, shape {test_images.shape}")
    
    # Test detection
    try:
        detection_results = mae_detector.detect(test_images)
        recon_errors = mae_detector.get_reconstruction_error(test_images)
        
        detection_rate = detection_results.sum().item() / batch_size * 100
        min_error = recon_errors.min().item()
        max_error = recon_errors.max().item()
        mean_error = recon_errors.mean().item()
        
        print(f"\n📈 Results:")
        print(f"  Detection Rate: {detection_rate:.1f}%")
        print(f"  Recon Errors: [{min_error:.4f}, {max_error:.4f}]")
        print(f"  Mean Error: {mean_error:.4f}")
        
        # Check if detection rate is reasonable
        if detection_rate > 50:
            print("❌ PROBLEM: Detection rate too high!")
            print("   This means MAE is detecting too many samples as adversarial")
        elif detection_rate < 5:
            print("⚠️  WARNING: Detection rate very low")
            print("   This might mean MAE is not detecting adversarial samples")
        else:
            print("✅ GOOD: Detection rate looks reasonable")
            
        # Check error range
        if max_error < 0.3:
            print("✅ GOOD: Error range looks reasonable for BR35H")
        else:
            print("⚠️  WARNING: Error range might be too high")
            
    except Exception as e:
        print(f"❌ ERROR: MAE detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_mae_detection()
    if success:
        print("\n🎯 MAE detection test completed")
    else:
        print("\n❌ MAE detection test failed")
