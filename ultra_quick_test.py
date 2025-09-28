#!/usr/bin/env python3
"""
Ultra Quick Test - Just test component initialization without training
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_initialization():
    """Test if all components can be initialized without errors"""
    print("🔧 Testing basic component initialization...")
    
    try:
        # Get config
        from config_fixed import get_debug_config
        cfg = get_debug_config()
        print(f"✅ Config loaded: {cfg.DATASET} dataset")
        
        # Test imports
        from utils.data_utils import get_dataset, create_federated_datasets
        from federated.client import Client
        from federated.server import FedServer
        from attacks.pgd import PGDAttack
        from diffusion.diffuser import UNet
        from defense.mae_detector1 import MAEDetector
        print("✅ All imports successful")
        
        # Test dataset loading
        result = get_dataset(cfg.DATASET, cfg.DATA_ROOT)
        if isinstance(result, tuple):
            train_dataset, test_dataset = result
            print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
        else:
            train_dataset = result
            print(f"✅ Dataset loaded: {len(train_dataset)} samples")
        
        # Test diffusion model
        diffuser = UNet(in_channels=cfg.IMG_CHANNELS, hidden_channels=64)
        diffuser = diffuser.to(cfg.DEVICE)
        print("✅ Diffusion model created")
        
        # Test client initialization
        client = Client(0, cfg, diffuser)
        print("✅ Client initialized")
        
        # Test server initialization
        server = FedServer(cfg)
        print("✅ Server initialized")
        
        # Test attacker initialization
        attacker = PGDAttack(cfg)
        print("✅ PGD attacker initialized")
        
        # Test MAE detector initialization
        mae_detector = MAEDetector(cfg)
        print("✅ MAE detector initialized")
        
        print("\n🎉 ALL COMPONENTS INITIALIZED SUCCESSFULLY!")
        print("✅ System is ready for federated training")
        return True
        
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_diffpure_functionality():
    """Test DiffPure function specifically"""
    print("\n🧪 Testing DiffPure functionality...")
    
    try:
        from main import diffpure_purify
        from config_fixed import get_debug_config
        from diffusion.diffuser import UNet
        
        cfg = get_debug_config()
        
        # Create test data
        test_data = torch.randn(2, cfg.IMG_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)
        
        # Create diffuser
        diffuser = UNet(in_channels=cfg.IMG_CHANNELS, hidden_channels=64)
        
        # Test purification
        purified = diffpure_purify(diffuser, test_data, cfg)
        
        # Validate
        assert purified.shape == test_data.shape, "Shape mismatch"
        assert torch.all(purified >= 0) and torch.all(purified <= 1), "Range invalid"
        
        print("✅ DiffPure function works correctly")
        return True
        
    except Exception as e:
        print(f"❌ DiffPure test failed: {e}")
        return False

def test_mae_detection():
    """Test MAE detection functionality"""
    print("\n🔍 Testing MAE detection...")
    
    try:
        from defense.mae_detector1 import MAEDetector
        from config_fixed import get_debug_config
        
        cfg = get_debug_config()
        
        # Create MAE detector
        mae_detector = MAEDetector(cfg)
        
        # Create test data
        test_data = torch.randn(2, cfg.IMG_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)
        
        # Test detection
        detections = mae_detector.detect(test_data)
        
        # Validate
        assert detections.dtype == torch.bool, f"Expected bool, got {detections.dtype}"
        assert detections.shape[0] == 2, f"Expected 2 detections, got {detections.shape[0]}"
        
        print(f"✅ MAE detection works: {detections.sum()}/2 detected as adversarial")
        return True
        
    except Exception as e:
        print(f"❌ MAE detection test failed: {e}")
        return False

def main():
    """Run ultra quick tests"""
    print("⚡ Ultra Quick Component Test")
    print("=" * 40)
    
    tests = [
        test_basic_initialization,
        test_diffpure_functionality,
        test_mae_detection,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "="*40)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🚀 READY FOR FEDERATED TRAINING!")
        print("💡 You can now run: python main.py --dataset cifar10 --mode debug")
        return True
    else:
        print("⚠️ Some components need fixes before training")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
