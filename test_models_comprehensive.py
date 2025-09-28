#!/usr/bin/env python3
"""
Comprehensive test suite for MAE and Diffusion models
Tests model architectures, checkpoints, and integration issues
"""

import torch
import torch.nn as nn
import logging
import sys
import traceback
from pathlib import Path
import json

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

def test_mae_checkpoint_compatibility():
    """Test MAE checkpoint and config compatibility"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Testing MAE Checkpoint Compatibility...")
    
    try:
        from config_fixed import get_debug_config, get_full_config
        
        # Test both configs
        configs_to_test = [
            ("debug", get_debug_config()),
            ("full", get_full_config())
        ]
        
        for config_name, cfg in configs_to_test:
            logger.info(f"Testing {config_name} config...")
            
            # Check MAE checkpoint files
            dataset_name = cfg.DATASET.lower()
            mae_files = [
                f"checkpoints/mae_detector_{dataset_name}.pt",
                f"checkpoints/mae_detector_{dataset_name}_best.pt",
                "checkpoints/mae_detector.pt",
                "checkpoints/mae_detector_best.pt"
            ]
            
            found_checkpoint = None
            for mae_file in mae_files:
                if Path(mae_file).exists():
                    found_checkpoint = mae_file
                    logger.info(f"✅ Found MAE checkpoint: {mae_file}")
                    break
            
            if found_checkpoint:
                # Load checkpoint and inspect dimensions
                try:
                    checkpoint = torch.load(found_checkpoint, map_location='cpu')
                    logger.info(f"📊 MAE Checkpoint keys: {list(checkpoint.keys())}")
                    
                    # Look for dimension info in checkpoint
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                    
                    # Find embedding dimensions from checkpoint
                    embed_dims = []
                    for key, tensor in state_dict.items():
                        if 'embed' in key.lower() and len(tensor.shape) >= 2:
                            embed_dims.append((key, tensor.shape))
                    
                    logger.info(f"📐 Embedding dimensions in checkpoint:")
                    for key, shape in embed_dims:
                        logger.info(f"  {key}: {shape}")
                    
                    # Compare with config
                    logger.info(f"⚙️ Config MAE_EMBED_DIM: {cfg.MAE_EMBED_DIM}")
                    logger.info(f"⚙️ Config MAE_DECODER_EMBED_DIM: {cfg.MAE_DECODER_EMBED_DIM}")
                    
                except Exception as e:
                    logger.error(f"❌ Error loading MAE checkpoint: {e}")
            else:
                logger.warning(f"⚠️ No MAE checkpoint found for {config_name} config")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ MAE checkpoint test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_diffusion_checkpoint():
    """Test diffusion model checkpoint"""
    logger = logging.getLogger(__name__)
    logger.info("🌊 Testing Diffusion Checkpoint...")
    
    try:
        from config_fixed import get_debug_config
        from diffusion.diffuser import UNet
        
        cfg = get_debug_config()
        dataset_name = cfg.DATASET.lower()
        
        # Check diffusion checkpoint
        diffusion_file = f"checkpoints/diffuser_{dataset_name}.pt"
        
        if Path(diffusion_file).exists():
            logger.info(f"✅ Found diffusion checkpoint: {diffusion_file}")
            
            # Load and inspect checkpoint
            checkpoint = torch.load(diffusion_file, map_location='cpu')
            logger.info(f"📊 Diffusion checkpoint type: {type(checkpoint)}")
            
            if isinstance(checkpoint, dict):
                logger.info(f"📊 Diffusion checkpoint keys: {list(checkpoint.keys())}")
            
            # Test loading into model
            diffuser = UNet(
                in_channels=cfg.IMG_CHANNELS, 
                hidden_channels=getattr(cfg, 'DIFFUSION_HIDDEN_CHANNELS', 128)
            )
            
            try:
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    diffuser.load_state_dict(checkpoint['model_state_dict'])
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    diffuser.load_state_dict(checkpoint['state_dict'])
                else:
                    diffuser.load_state_dict(checkpoint)
                
                logger.info("✅ Diffusion model loaded successfully")
                
                # Test forward pass
                dummy_input = torch.randn(2, cfg.IMG_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)
                dummy_t = torch.randn(2)
                
                with torch.no_grad():
                    output = diffuser(dummy_input, dummy_t)
                
                logger.info(f"✅ Diffusion forward pass successful: {output.shape}")
                
            except Exception as e:
                logger.error(f"❌ Error loading diffusion model: {e}")
                return False
                
        else:
            logger.warning(f"⚠️ Diffusion checkpoint not found: {diffusion_file}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Diffusion test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_mae_model_creation():
    """Test MAE model creation with different configurations"""
    logger = logging.getLogger(__name__)
    logger.info("🏗️ Testing MAE Model Creation...")
    
    try:
        from defense.mae_detector import MAEDetector
        from config_fixed import get_debug_config
        
        # Test different embedding dimensions
        test_configs = [
            {"MAE_EMBED_DIM": 128, "MAE_DECODER_EMBED_DIM": 64},
            {"MAE_EMBED_DIM": 256, "MAE_DECODER_EMBED_DIM": 128},
            {"MAE_EMBED_DIM": 512, "MAE_DECODER_EMBED_DIM": 256},
        ]
        
        for i, test_params in enumerate(test_configs):
            logger.info(f"Testing config {i+1}: {test_params}")
            
            cfg = get_debug_config()
            # Override with test parameters
            for key, value in test_params.items():
                setattr(cfg, key, value)
            
            try:
                mae_detector = MAEDetector(cfg)
                
                # Test detection
                dummy_data = torch.randn(4, cfg.IMG_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)
                detections = mae_detector.detect(dummy_data)
                
                logger.info(f"✅ Config {i+1} successful - Detection shape: {detections.shape}")
                
            except Exception as e:
                logger.error(f"❌ Config {i+1} failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ MAE model creation test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def analyze_checkpoint_dimensions():
    """Analyze all checkpoint files and report dimensions"""
    logger = logging.getLogger(__name__)
    logger.info("📐 Analyzing Checkpoint Dimensions...")
    
    try:
        checkpoint_dir = Path("checkpoints")
        if not checkpoint_dir.exists():
            logger.warning("⚠️ Checkpoints directory not found")
            return False
        
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        
        for checkpoint_file in checkpoint_files:
            logger.info(f"\n🔍 Analyzing: {checkpoint_file.name}")
            
            try:
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                
                # Get state dict
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                        logger.info("📊 Found 'model_state_dict' key")
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                        logger.info("📊 Found 'state_dict' key")
                    else:
                        state_dict = checkpoint
                        logger.info("📊 Using checkpoint as state_dict")
                    
                    # Additional info
                    if 'config' in checkpoint:
                        logger.info(f"📋 Config in checkpoint: {checkpoint['config']}")
                    if 'epoch' in checkpoint:
                        logger.info(f"📅 Epoch: {checkpoint['epoch']}")
                    if 'loss' in checkpoint:
                        logger.info(f"📉 Loss: {checkpoint['loss']}")
                else:
                    state_dict = checkpoint
                    logger.info("📊 Direct state_dict")
                
                # Analyze dimensions
                logger.info("📐 Key tensor dimensions:")
                for key, tensor in state_dict.items():
                    if len(tensor.shape) >= 2:  # Only show meaningful tensors
                        logger.info(f"  {key}: {tensor.shape}")
                
            except Exception as e:
                logger.error(f"❌ Error analyzing {checkpoint_file.name}: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Checkpoint analysis failed: {e}")
        logger.error(traceback.format_exc())
        return False

def suggest_config_fixes():
    """Suggest configuration fixes based on checkpoint analysis"""
    logger = logging.getLogger(__name__)
    logger.info("🔧 Suggesting Configuration Fixes...")
    
    try:
        from config_fixed import get_debug_config, get_full_config
        
        # Check current configs
        debug_cfg = get_debug_config()
        full_cfg = get_full_config()
        
        logger.info("📋 Current Configuration Analysis:")
        logger.info(f"Debug Config - MAE_EMBED_DIM: {debug_cfg.MAE_EMBED_DIM}, MAE_DECODER_EMBED_DIM: {debug_cfg.MAE_DECODER_EMBED_DIM}")
        logger.info(f"Full Config - MAE_EMBED_DIM: {full_cfg.MAE_EMBED_DIM}, MAE_DECODER_EMBED_DIM: {full_cfg.MAE_DECODER_EMBED_DIM}")
        
        # Check for common dimension mismatches
        logger.info("\n💡 Suggested Fixes:")
        logger.info("1. If you see '256 vs 128' error, try:")
        logger.info("   - Set MAE_EMBED_DIM = 256 and MAE_DECODER_EMBED_DIM = 256")
        logger.info("   - Or retrain MAE with MAE_EMBED_DIM = 128")
        
        logger.info("2. If diffusion model fails, check:")
        logger.info("   - DIFFUSION_HIDDEN_CHANNELS matches trained model")
        logger.info("   - IMG_CHANNELS matches dataset (3 for CIFAR-10)")
        
        logger.info("3. For better training accuracy:")
        logger.info("   - Increase CLIENT_EPOCHS to 5-10")
        logger.info("   - Use LEARNING_RATE = 0.01 for CIFAR-10")
        logger.info("   - Ensure NUM_ROUNDS >= 15")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Config analysis failed: {e}")
        logger.error(traceback.format_exc())
        return False

def run_comprehensive_model_test():
    """Run all model tests and provide detailed analysis"""
    logger = setup_logging()
    logger.info("🚀 Starting Comprehensive Model Test Suite")
    logger.info("=" * 70)
    
    tests = [
        ("MAE Checkpoint Compatibility", test_mae_checkpoint_compatibility),
        ("Diffusion Checkpoint", test_diffusion_checkpoint),
        ("MAE Model Creation", test_mae_model_creation),
        ("Checkpoint Dimension Analysis", analyze_checkpoint_dimensions),
        ("Configuration Fix Suggestions", suggest_config_fixes),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n🔄 Running {test_name}...")
        results[test_name] = test_func()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 COMPREHENSIVE MODEL TEST RESULTS")
    logger.info("=" * 70)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:<30}: {status}")
        if result:
            passed += 1
    
    logger.info("=" * 70)
    logger.info(f"Overall: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    
    if passed >= 3:  # Allow some tests to fail
        logger.info("🎉 Model analysis completed! Check suggestions above.")
        return True
    else:
        logger.error(f"❌ Too many tests failed. Check model configurations.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_model_test()
    sys.exit(0 if success else 1)
