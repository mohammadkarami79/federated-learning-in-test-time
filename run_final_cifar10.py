#!/usr/bin/env python3
"""
FINAL CIFAR-10 Training Script - All Issues Fixed
"""

import sys
import logging
from pathlib import Path

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Run final CIFAR-10 training with all fixes"""
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Starting FINAL CIFAR-10 Training (All Issues Fixed)...")
    
    try:
        # Apply final fixes
        from FINAL_COMPLETE_MAE_FIX import create_final_cifar10_config, delete_broken_mae_checkpoint
        
        # Delete broken checkpoints
        delete_broken_mae_checkpoint()
        
        # Create optimized config
        cfg = create_final_cifar10_config()
        if cfg is None:
            logger.error("Failed to create final config")
            return False
            
        # Override default config
        import config_fixed
        config_fixed.get_config = lambda: cfg
        
        # Force CIFAR-10 dataset
        sys.argv = ['main.py', '--dataset', 'cifar10', '--mode', 'full', '--train-mae']
        
        logger.info("Expected results:")
        logger.info("   - Clean accuracy: 70%+ (progressive improvement)")
        logger.info("   - MAE detection: 25-40% (balanced detection)")
        logger.info("   - Adversarial accuracy: 40-60% (strong defense)")
        logger.info("   - No dimension errors")
        
        # Import and run main training
        from main import main as run_training
        return run_training()
        
    except Exception as e:
        logger.error(f"Final CIFAR-10 training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nFINAL CIFAR-10 training completed successfully!")
        print("All issues should now be resolved.")
    else:
        print("\nFinal CIFAR-10 training failed!")
        sys.exit(1)
