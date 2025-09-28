#!/usr/bin/env python3
"""
Quick fix for MAE dimension mismatch and CLIENT_EPOCHS issues
Run this to fix the immediate problems shown in your log
"""

import torch
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def fix_mae_config_mismatch():
    """Fix MAE configuration to match trained model dimensions"""
    logger = logging.getLogger(__name__)
    logger.info("🔧 Fixing MAE Configuration Mismatch...")
    
    try:
        # Read current config file
        config_file = Path("config_fixed.py")
        if not config_file.exists():
            logger.error("❌ config_fixed.py not found")
            return False
        
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Fix debug config MAE dimensions
        content = content.replace(
            'cfg.MAE_EMBED_DIM = 256\n    cfg.MAE_DECODER_EMBED_DIM = 256',
            'cfg.MAE_EMBED_DIM = 256\n    cfg.MAE_DECODER_EMBED_DIM = 128'
        )
        
        # Also fix full config if needed
        content = content.replace(
            'cfg.MAE_EMBED_DIM = 256   # Larger for CIFAR-10\n    cfg.MAE_DECODER_EMBED_DIM = 128',
            'cfg.MAE_EMBED_DIM = 256   # FIXED: Match trained model\n    cfg.MAE_DECODER_EMBED_DIM = 128'
        )
        
        with open(config_file, 'w') as f:
            f.write(content)
        
        logger.info("✅ Fixed MAE dimensions in config")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to fix config: {e}")
        return False

def test_mae_with_fixed_config():
    """Test MAE detector with fixed configuration"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing MAE with Fixed Config...")
    
    try:
        from config_fixed import get_debug_config
        from defense.mae_detector import MAEDetector
        
        cfg = get_debug_config()
        logger.info(f"Config MAE_EMBED_DIM: {cfg.MAE_EMBED_DIM}")
        logger.info(f"Config MAE_DECODER_EMBED_DIM: {cfg.MAE_DECODER_EMBED_DIM}")
        
        # Test MAE detector creation
        mae_detector = MAEDetector(cfg)
        
        # Test with dummy data
        dummy_data = torch.randn(4, 3, 32, 32).to(cfg.DEVICE)
        detections = mae_detector.detect(dummy_data)
        
        logger.info(f"✅ MAE detector working - Detection shape: {detections.shape}")
        logger.info(f"✅ Detection values: {detections.tolist()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ MAE test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def verify_client_epochs_fix():
    """Verify CLIENT_EPOCHS configuration"""
    logger = logging.getLogger(__name__)
    logger.info("⚙️ Verifying CLIENT_EPOCHS Fix...")
    
    try:
        from config_fixed import get_debug_config, get_full_config
        
        debug_cfg = get_debug_config()
        full_cfg = get_full_config()
        
        logger.info(f"Debug config CLIENT_EPOCHS: {debug_cfg.CLIENT_EPOCHS}")
        logger.info(f"Full config CLIENT_EPOCHS: {full_cfg.CLIENT_EPOCHS}")
        
        # Check if they're reasonable values
        if debug_cfg.CLIENT_EPOCHS >= 5 and full_cfg.CLIENT_EPOCHS >= 3:
            logger.info("✅ CLIENT_EPOCHS values look good")
            return True
        else:
            logger.warning(f"⚠️ CLIENT_EPOCHS may be too low")
            return False
        
    except Exception as e:
        logger.error(f"❌ CLIENT_EPOCHS verification failed: {e}")
        return False

def main():
    """Run all quick fixes"""
    logger = setup_logging()
    logger.info("🚀 Running Quick Fixes for MAE and Training Issues")
    logger.info("=" * 60)
    
    fixes = [
        ("MAE Config Mismatch", fix_mae_config_mismatch),
        ("MAE Test with Fixed Config", test_mae_with_fixed_config),
        ("CLIENT_EPOCHS Verification", verify_client_epochs_fix),
    ]
    
    results = {}
    for fix_name, fix_func in fixes:
        logger.info(f"\n🔄 Running {fix_name}...")
        results[fix_name] = fix_func()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 QUICK FIX RESULTS")
    logger.info("=" * 60)
    
    passed = 0
    total = len(fixes)
    
    for fix_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{fix_name:<25}: {status}")
        if result:
            passed += 1
    
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("🎉 All fixes applied successfully!")
        logger.info("💡 Now copy the updated files to your server and restart training")
        return True
    else:
        logger.error(f"❌ {total-passed} fixes failed. Manual intervention needed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
