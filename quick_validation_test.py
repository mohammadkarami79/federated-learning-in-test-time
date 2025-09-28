#!/usr/bin/env python3
"""
Quick validation test for the federated learning components
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_diffpure_function():
    """Test the DiffPure purification function"""
    print("🔍 Testing DiffPure function...")
    
    try:
        from main import diffpure_purify
        from config_fixed import get_debug_config
        from diffusion.diffuser import UNet
        
        cfg = get_debug_config()
        
        # Create dummy data
        batch_size = 4
        channels = cfg.IMG_CHANNELS
        img_size = cfg.IMG_SIZE
        
        adv_data = torch.randn(batch_size, channels, img_size, img_size)
        
        # Create dummy diffuser
        diffuser = UNet(in_channels=channels, hidden_channels=256)
        
        # Test purification
        purified = diffpure_purify(diffuser, adv_data, cfg)
        
        # Validate output
        assert purified.shape == adv_data.shape, f"Shape mismatch: {purified.shape} vs {adv_data.shape}"
        assert torch.all(purified >= 0) and torch.all(purified <= 1), "Output not in [0,1] range"
        
        print("✅ DiffPure function test passed")
        print(f"   - Input shape: {adv_data.shape}")
        print(f"   - Output shape: {purified.shape}")
        print(f"   - Output range: [{purified.min():.3f}, {purified.max():.3f}]")
        return True
        
    except Exception as e:
        print(f"❌ DiffPure function test failed: {e}")
        return False

def test_mae_detector():
    """Test the MAE detector"""
    print("\n🔍 Testing MAE detector...")
    
    try:
        from defense.mae_detector1 import MAEDetector
        from config_fixed import get_debug_config
        
        cfg = get_debug_config()
        
        # Create MAE detector
        mae_detector = MAEDetector(cfg)
        
        # Create dummy data
        batch_size = 4
        channels = cfg.IMG_CHANNELS
        img_size = cfg.IMG_SIZE
        
        test_data = torch.randn(batch_size, channels, img_size, img_size)
        
        # Test detection
        detections = mae_detector.detect(test_data)
        
        # Validate output
        assert detections.shape[0] == batch_size, f"Detection shape mismatch: {detections.shape}"
        assert detections.dtype == torch.bool, f"Detection should be boolean, got {detections.dtype}"
        
        print("✅ MAE detector test passed")
        print(f"   - Input shape: {test_data.shape}")
        print(f"   - Detection shape: {detections.shape}")
        print(f"   - Detections: {detections.sum()}/{batch_size}")
        return True
        
    except Exception as e:
        print(f"❌ MAE detector test failed: {e}")
        return False

def test_config_settings():
    """Test configuration settings"""
    print("\n🔍 Testing configuration settings...")
    
    try:
        from config_fixed import get_debug_config, get_full_config
        
        test_cfg = get_debug_config()
        full_cfg = get_full_config()
        
        # Check critical parameters
        print(f"Test config MAE_THRESHOLD: {test_cfg.MAE_THRESHOLD}")
        print(f"Full config MAE_THRESHOLD: {full_cfg.MAE_THRESHOLD}")
        
        print(f"Test config DIFFPURE_SIGMA: {getattr(test_cfg, 'DIFFPURE_SIGMA', 'NOT_SET')}")
        print(f"Full config DIFFPURE_SIGMA: {getattr(full_cfg, 'DIFFPURE_SIGMA', 'NOT_SET')}")
        
        # Check if configs are valid
        required_attrs = ['DATASET', 'NUM_CLASSES', 'IMG_SIZE', 'IMG_CHANNELS', 'BATCH_SIZE']
        
        for attr in required_attrs:
            assert hasattr(test_cfg, attr), f"Test config missing {attr}"
            assert hasattr(full_cfg, attr), f"Full config missing {attr}"
        
        print("✅ Configuration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_import_dependencies():
    """Test if all required dependencies can be imported"""
    print("\n🔍 Testing import dependencies...")
    
    try:
        # Test core imports
        import torch
        import torchvision
        print(f"   - PyTorch: {torch.__version__}")
        print(f"   - TorchVision: {torchvision.__version__}")
        
        # Test project imports
        from utils.data_utils import get_dataset
        from attacks.pgd import PGDAttack
        from federated.client import Client
        from federated.server import Server
        
        print("✅ Import dependencies test passed")
        return True
        
    except Exception as e:
        print(f"❌ Import dependencies test failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\n🔍 Testing data loading...")
    
    try:
        from utils.data_utils import get_dataset
        from config_fixed import get_debug_config
        
        cfg = get_debug_config()
        
        # Try to load dataset
        result = get_dataset(cfg.DATASET, cfg.DATA_ROOT)
        
        if isinstance(result, tuple):
            train_dataset, test_dataset = result
            print(f"   - Train dataset size: {len(train_dataset)}")
            print(f"   - Test dataset size: {len(test_dataset)}")
            dataset_to_test = test_dataset
        else:
            print(f"   - Single dataset returned: {type(result)}")
            dataset_to_test = result
        
        # Test a sample
        if hasattr(dataset_to_test, '__len__') and len(dataset_to_test) > 0:
            sample = dataset_to_test[0]
            print(f"   - Sample shape: {sample[0].shape}")
            print(f"   - Sample label: {sample[1]}")
        
        print("✅ Data loading test passed")
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🚀 Quick Validation Test Suite")
    print("=" * 50)
    
    tests = [
        test_import_dependencies,
        test_config_settings,
        test_data_loading,
        test_diffpure_function,
        test_mae_detector,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "="*50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
