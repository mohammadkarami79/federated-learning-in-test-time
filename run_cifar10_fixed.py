#!/usr/bin/env python3
"""
CIFAR-10 Optimized Training Script
Fixes MAE detector and improves adversarial accuracy
"""

import sys
import logging
from pathlib import Path

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Run optimized CIFAR-10 training"""
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Starting OPTIMIZED CIFAR-10 Training...")
    
    try:
        # Import optimized config
        from CIFAR10_CRITICAL_FIX import create_cifar10_optimized_config
        cfg = create_cifar10_optimized_config()
        
        if cfg is None:
            logger.error("Failed to create optimized config")
            return False
            
        # Override default config
        import config_fixed
        config_fixed.get_config = lambda: cfg
        
        logger.info("Expected improvements:")
        logger.info("   - Clean accuracy: 70%+ (was 20.94%)")
        logger.info("   - MAE detection: 25-40% (was 100%)")
        logger.info("   - Adversarial accuracy: 40-60% (was 10.62%)")
        
        # Force CIFAR-10 dataset
        import sys
        sys.argv = ['main.py', '--dataset', 'cifar10', '--mode', 'full']
        
        # Import and run main training
        from main import main as run_training
        return run_training()
        
    except Exception as e:
        logger.error(f"CIFAR-10 training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("CIFAR-10 training completed successfully!")
    else:
        print("CIFAR-10 training failed!")
        sys.exit(1)
