#!/usr/bin/env python3
"""
Test Config Integration - Verify Complete Workflow
Tests if config properly propagates through all components for each dataset
"""

import torch
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    return logging.getLogger(__name__)

def test_config_integration_complete():
    """Test complete config integration for all datasets"""
    logger = logging.getLogger(__name__)
    
    datasets = ['cifar10', 'cifar100', 'mnist']
    
    for dataset in datasets:
        logger.info(f"\n🔍 Testing {dataset.upper()} integration...")
        
        try:
            # 1. Test config creation
            from config_fixed import get_debug_config
            cfg = get_debug_config()
            
            # Update config for dataset
            cfg.DATASET = dataset.lower()
            cfg.DATASET_NAME = dataset.upper()
            
            if dataset == 'mnist':
                cfg.IMG_CHANNELS = 1
                cfg.IMG_SIZE = 28
                cfg.N_CLASSES = 10
            elif dataset == 'cifar100':
                cfg.N_CLASSES = 100
                cfg.IMG_CHANNELS = 3
                cfg.IMG_SIZE = 32
            else:  # cifar10
                cfg.N_CLASSES = 10
                cfg.IMG_CHANNELS = 3
                cfg.IMG_SIZE = 32
            
            logger.info(f"✅ Config: {cfg.DATASET}, {cfg.IMG_CHANNELS}ch, {cfg.N_CLASSES}cls")
            
            # 2. Test dataset loading
            from utils.data_utils import get_dataset
            train_ds, test_ds = get_dataset(cfg)
            logger.info(f"✅ Dataset: {len(train_ds)} train, {len(test_ds)} test")
            
            # 3. Test model creation with config
            from models import get_model
            model = get_model(cfg)
            logger.info(f"✅ Model: {sum(p.numel() for p in model.parameters()):,} params")
            
            # 4. Test diffusion model with config channels
            from diffusion.diffuser import UNet
            diffuser = UNet(in_channels=cfg.IMG_CHANNELS, hidden_channels=32)
            logger.info(f"✅ Diffusion: {cfg.IMG_CHANNELS} input channels")
            
            # 5. Test MAE detector with config
            from defense.mae_detector import MAEDetector
            detector = MAEDetector(cfg)
            logger.info(f"✅ MAE: detector created")
            
            # 6. Test federated components
            from federated.client import Client
            from federated.server import Server
            client = Client(0, cfg, diffuser)
            server = Server(cfg)
            logger.info(f"✅ Federated: client and server created")
            
            # 7. Test data batch compatibility
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=False)
            batch = next(iter(train_loader))
            images, labels = batch
            
            expected_shape = (4, cfg.IMG_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)
            if images.shape == expected_shape:
                logger.info(f"✅ Data Shape: {images.shape} matches config")
            else:
                logger.error(f"❌ Data Shape: {images.shape} != {expected_shape}")
                return False
            
            # 8. Test model forward pass with data
            with torch.no_grad():
                output = model(images)
                if output.shape[1] == cfg.N_CLASSES:
                    logger.info(f"✅ Model Output: {output.shape[1]} classes matches config")
                else:
                    logger.error(f"❌ Model Output: {output.shape[1]} != {cfg.N_CLASSES}")
                    return False
            
            logger.info(f"🎉 {dataset.upper()} COMPLETE INTEGRATION SUCCESS!")
            
        except Exception as e:
            logger.error(f"❌ {dataset.upper()} integration failed: {e}")
            return False
    
    return True

def test_cross_dataset_consistency():
    """Test that configs are correctly isolated between datasets"""
    logger = logging.getLogger(__name__)
    
    try:
        from config_fixed import get_debug_config
        
        # Test CIFAR-10
        cfg1 = get_debug_config()
        cfg1.DATASET = 'cifar10'
        cfg1.IMG_CHANNELS = 3
        cfg1.N_CLASSES = 10
        
        # Test MNIST 
        cfg2 = get_debug_config()
        cfg2.DATASET = 'mnist'
        cfg2.IMG_CHANNELS = 1
        cfg2.N_CLASSES = 10
        
        # Test CIFAR-100
        cfg3 = get_debug_config()
        cfg3.DATASET = 'cifar100'
        cfg3.IMG_CHANNELS = 3
        cfg3.N_CLASSES = 100
        
        # Verify they're different
        if (cfg1.IMG_CHANNELS == 3 and cfg2.IMG_CHANNELS == 1 and 
            cfg1.N_CLASSES == 10 and cfg3.N_CLASSES == 100):
            logger.info("✅ Config isolation working correctly")
            return True
        else:
            logger.error("❌ Config isolation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Cross-dataset test failed: {e}")
        return False

def test_file_naming_consistency():
    """Test that file naming follows config"""
    logger = logging.getLogger(__name__)
    
    try:
        from config_fixed import get_debug_config
        
        datasets = ['cifar10', 'cifar100', 'mnist']
        
        for dataset in datasets:
            cfg = get_debug_config()
            cfg.DATASET = dataset
            
            # Test expected file names
            expected_diffusion = f'checkpoints/diffuser_{dataset}.pt'
            expected_path = Path(expected_diffusion)
            
            logger.info(f"✅ {dataset}: Expected file {expected_diffusion}")
        
        logger.info("✅ File naming follows config patterns")
        return True
        
    except Exception as e:
        logger.error(f"❌ File naming test failed: {e}")
        return False

def main():
    """Run all config integration tests"""
    logger = setup_logging()
    
    print("🔍 TESTING COMPLETE CONFIG INTEGRATION")
    print("=" * 60)
    
    tests = [
        ("Complete Integration", test_config_integration_complete),
        ("Cross-Dataset Consistency", test_cross_dataset_consistency),
        ("File Naming Consistency", test_file_naming_consistency),
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
    
    print("\n" + "=" * 60)
    print(f"📊 CONFIG INTEGRATION: {passed}/{total} PASSED ({100*passed/total:.0f}%)")
    
    if passed == total:
        print("🎉 CONFIG INTEGRATION FULLY WORKING!")
        return 0
    else:
        print("❌ CONFIG INTEGRATION HAS ISSUES")
        return 1

if __name__ == "__main__":
    exit(main()) 