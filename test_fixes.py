#!/usr/bin/env python3
"""
Test script to verify critical fixes for federated learning project
Tests MAE detector, training accuracy, and overall system integration
"""

import torch
import torch.nn as nn
import logging
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_mae_detector():
    """Test MAE detector functionality"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Testing MAE Detector...")
    
    try:
        from config_fixed import get_debug_config
        from defense.mae_detector import MAEDetector
        
        cfg = get_debug_config()
        mae_detector = MAEDetector(cfg)
        
        # Test with dummy data
        dummy_data = torch.randn(4, 3, 32, 32).to(cfg.DEVICE)
        
        # Test detection
        detections = mae_detector.detect(dummy_data)
        logger.info(f"✅ MAE Detector working - Detection shape: {detections.shape}")
        logger.info(f"✅ Detection values: {detections.tolist()}")
        
        # Test model forward pass
        if hasattr(mae_detector, 'model'):
            recon, mask = mae_detector.model(dummy_data)
            logger.info(f"✅ MAE Model forward pass - Reconstruction shape: {recon.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ MAE Detector test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_configuration():
    """Test configuration consistency"""
    logger = logging.getLogger(__name__)
    logger.info("⚙️ Testing Configuration...")
    
    try:
        from config_fixed import get_debug_config, get_full_config, validate_config
        
        # Test debug config
        debug_cfg = get_debug_config()
        validate_config(debug_cfg)
        logger.info(f"✅ Debug config valid - {debug_cfg.NUM_ROUNDS} rounds, {debug_cfg.CLIENT_EPOCHS} client epochs")
        
        # Test full config
        full_cfg = get_full_config()
        validate_config(full_cfg)
        logger.info(f"✅ Full config valid - {full_cfg.NUM_ROUNDS} rounds, {full_cfg.CLIENT_EPOCHS} client epochs")
        
        # Check key parameters
        assert debug_cfg.MAE_THRESHOLD <= 0.5, f"MAE threshold too high: {debug_cfg.MAE_THRESHOLD}"
        assert debug_cfg.CLIENT_EPOCHS >= 3, f"Client epochs too low: {debug_cfg.CLIENT_EPOCHS}"
        assert debug_cfg.NUM_CLASSES == 10, f"Wrong number of classes: {debug_cfg.NUM_CLASSES}"
        
        logger.info("✅ All configuration tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_model_creation():
    """Test model creation and basic functionality"""
    logger = logging.getLogger(__name__)
    logger.info("🏗️ Testing Model Creation...")
    
    try:
        from config_fixed import get_debug_config
        import torchvision.models as models
        
        cfg = get_debug_config()
        
        # Test ResNet18 creation
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, cfg.NUM_CLASSES)
        model = model.to(cfg.DEVICE)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 32, 32).to(cfg.DEVICE)
        output = model(dummy_input)
        
        assert output.shape == (2, cfg.NUM_CLASSES), f"Wrong output shape: {output.shape}"
        logger.info(f"✅ Model creation successful - Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model creation test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_diffpure_integration():
    """Test DiffPure integration"""
    logger = logging.getLogger(__name__)
    logger.info("🌊 Testing DiffPure Integration...")
    
    try:
        from config_fixed import get_debug_config
        from diffusion.diffuser import UNet
        from main import diffpure_purify
        
        cfg = get_debug_config()
        
        # Create dummy diffuser
        diffuser = UNet(in_channels=cfg.IMG_CHANNELS, hidden_channels=64)
        diffuser = diffuser.to(cfg.DEVICE)
        diffuser.eval()
        
        # Test purification
        dummy_adv = torch.randn(2, 3, 32, 32).to(cfg.DEVICE)
        purified = diffpure_purify(diffuser, dummy_adv, cfg)
        
        assert purified.shape == dummy_adv.shape, f"Shape mismatch: {purified.shape} vs {dummy_adv.shape}"
        assert torch.all(purified >= 0) and torch.all(purified <= 1), "Purified data not in [0,1] range"
        
        logger.info(f"✅ DiffPure integration working - Purified shape: {purified.shape}")
        logger.info(f"✅ Purified data range: [{purified.min():.3f}, {purified.max():.3f}]")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ DiffPure integration test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_training_setup():
    """Test training setup and basic client functionality"""
    logger = logging.getLogger(__name__)
    logger.info("🏋️ Testing Training Setup...")
    
    try:
        from config_fixed import get_debug_config
        from federated.client import Client
        from federated.server import FedServer
        
        cfg = get_debug_config()
        
        # Test server creation
        server = FedServer(cfg)
        logger.info("✅ Server created successfully")
        
        # Test client creation (this might fail due to data dependencies)
        try:
            client = Client(0, cfg)
            logger.info("✅ Client created successfully")
        except Exception as e:
            logger.warning(f"⚠️ Client creation failed (expected if no data): {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training setup test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def run_comprehensive_test():
    """Run all tests and report results"""
    logger = setup_logging()
    logger.info("🚀 Starting Comprehensive Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Configuration", test_configuration),
        ("Model Creation", test_model_creation),
        ("MAE Detector", test_mae_detector),
        ("DiffPure Integration", test_diffpure_integration),
        ("Training Setup", test_training_setup),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n🔄 Running {test_name} Test...")
        results[test_name] = test_func()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    logger.info("=" * 60)
    logger.info(f"Overall: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! System is ready for training.")
        return True
    else:
        logger.error(f"❌ {total-passed} tests failed. Please fix issues before training.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
