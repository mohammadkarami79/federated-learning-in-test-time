#!/usr/bin/env python3
"""
Debug script to test main components step by step
"""

import logging
import sys
import traceback
from pathlib import Path

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def test_imports():
    """Test basic imports"""
    logger = logging.getLogger(__name__)
    logger.info("Testing basic imports...")
    
    try:
        import torch
        logger.info(f"✅ PyTorch {torch.__version__}")
    except Exception as e:
        logger.error(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        logger.info(f"✅ TorchVision {torchvision.__version__}")
    except Exception as e:
        logger.error(f"❌ TorchVision import failed: {e}")
        return False
    
    try:
        import numpy as np
        logger.info(f"✅ NumPy {np.__version__}")
    except Exception as e:
        logger.error(f"❌ NumPy import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading"""
    logger = logging.getLogger(__name__)
    logger.info("Testing configuration...")
    
    try:
        from config_fixed import get_debug_config
        cfg = get_debug_config()
        logger.info(f"✅ Config loaded: {cfg.DATASET_NAME}")
        return True
    except Exception as e:
        logger.error(f"❌ Config loading failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_dataset_loading():
    """Test dataset loading"""
    logger = logging.getLogger(__name__)
    logger.info("Testing dataset loading...")
    
    try:
        from config_fixed import get_debug_config
        from utils.data_utils import get_dataset
        
        cfg = get_debug_config()
        cfg.DATASET = 'br35h'
        cfg.DATASET_NAME = 'BR35H'
        
        logger.info("Loading BR35H dataset...")
        train_dataset, test_dataset = get_dataset(cfg)
        
        logger.info(f"✅ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
        return True
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_model_creation():
    """Test model creation"""
    logger = logging.getLogger(__name__)
    logger.info("Testing model creation...")
    
    try:
        from config_fixed import get_debug_config
        from models import get_model
        
        cfg = get_debug_config()
        cfg.DATASET = 'br35h'
        cfg.DATASET_NAME = 'BR35H'
        cfg.NUM_CLASSES = 2
        
        logger.info("Creating model...")
        model = get_model(cfg)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Model created: {total_params:,} parameters")
        
        # Test forward pass
        import torch
        dummy_input = torch.randn(1, 3, 224, 224)
        if torch.cuda.is_available():
            model = model.cuda()
            dummy_input = dummy_input.cuda()
        
        with torch.no_grad():
            output = model(dummy_input)
            logger.info(f"✅ Forward pass successful: output shape {output.shape}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_system_setup():
    """Test system setup"""
    logger = logging.getLogger(__name__)
    logger.info("Testing system setup...")
    
    try:
        from setup_system import run_comprehensive_check
        success = run_comprehensive_check()
        if success:
            logger.info("✅ System setup passed")
            return True
        else:
            logger.error("❌ System setup failed")
            return False
    except Exception as e:
        logger.error(f"❌ System setup failed: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main debug function"""
    logger = setup_logging()
    
    print("=" * 60)
    print("DEBUG TEST SCRIPT")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_imports),
        ("Configuration", test_config),
        ("Dataset Loading", test_dataset_loading),
        ("Model Creation", test_model_creation),
        ("System Setup", test_system_setup),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("DEBUG RESULTS")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The system should work.")
    else:
        print("⚠️ Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main() 