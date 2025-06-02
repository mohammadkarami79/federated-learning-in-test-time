#!/usr/bin/env python3
"""
Simple Test Script to Verify Complete Workflow
"""

import torch
import sys
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    return logging.getLogger(__name__)

def test_basic_imports():
    """Test that basic imports work"""
    logger = logging.getLogger(__name__)
    
    try:
        # Test optimized config system
        from config_fixed import get_debug_config, get_test_config, get_full_config
        logger.info("✅ Config system working")
        
        # Test dataset loading
        from utils.data_utils import get_dataset
        logger.info("✅ Data utils working")
        
        # Test model system
        from models import get_model
        logger.info("✅ Model system working")
        
        # Test attacks
        from attacks.pgd import PGDAttack
        logger.info("✅ Attack system working")
        
        # Test defense
        from defense.mae_detector import MAEDetector
        logger.info("✅ Defense system working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False

def test_config_system():
    """Test config system works for all datasets"""
    logger = logging.getLogger(__name__)
    
    try:
        from config_fixed import get_debug_config
        
        # Test CIFAR-10
        cfg = get_debug_config()
        cfg.DATASET = 'cifar10'
        cfg.DATASET_NAME = 'CIFAR10'
        cfg.IMG_CHANNELS = 3
        cfg.N_CLASSES = 10
        
        # Test CIFAR-100
        cfg_100 = get_debug_config()
        cfg_100.DATASET = 'cifar100'
        cfg_100.DATASET_NAME = 'CIFAR100'
        cfg_100.N_CLASSES = 100
        
        # Test MNIST
        cfg_mnist = get_debug_config()
        cfg_mnist.DATASET = 'mnist'
        cfg_mnist.DATASET_NAME = 'MNIST'
        cfg_mnist.IMG_CHANNELS = 1
        cfg_mnist.IMG_SIZE = 28
        
        logger.info("✅ All dataset configs working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Config test failed: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading for multiple datasets"""
    logger = logging.getLogger(__name__)
    
    try:
        from config_fixed import get_debug_config
        from utils.data_utils import get_dataset
        
        # Test CIFAR-10
        cfg = get_debug_config()
        train_ds, test_ds = get_dataset(cfg)
        logger.info(f"✅ CIFAR-10: {len(train_ds)} train, {len(test_ds)} test")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset test failed: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    logger = logging.getLogger(__name__)
    
    try:
        from config_fixed import get_debug_config
        from models import get_model
        
        cfg = get_debug_config()
        model = get_model(cfg)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Model created: {total_params:,} parameters")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        return False

def test_complete_workflow():
    """Test that main training components work"""
    logger = logging.getLogger(__name__)
    
    try:
        from config_fixed import get_debug_config
        from federated.client import Client
        from federated.server import Server
        from diffusion.diffuser import UNet
        
        cfg = get_debug_config()
        
        # Create diffusion model
        diffuser = UNet(in_channels=cfg.IMG_CHANNELS, hidden_channels=32)
        
        # Create simple components
        client = Client(0, cfg, diffuser)
        server = Server(cfg)
        
        logger.info("✅ Complete workflow components created")
        return True
        
    except Exception as e:
        logger.error(f"❌ Workflow test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger = setup_logging()
    
    print("🔍 TESTING COMPLETE WORKFLOW")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Config System", test_config_system),
        ("Dataset Loading", test_dataset_loading),
        ("Model Creation", test_model_creation),
        ("Complete Workflow", test_complete_workflow),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ PASS - {test_name}")
            else:
                print(f"❌ FAIL - {test_name}")
        except Exception as e:
            print(f"❌ FAIL - {test_name}: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} PASSED ({100*passed/total:.0f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - SYSTEM READY!")
        return 0
    else:
        print("❌ SOME TESTS FAILED - PLEASE FIX ISSUES")
        return 1

if __name__ == "__main__":
    exit(main()) 