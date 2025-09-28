#!/usr/bin/env python3
"""
ULTIMATE CIFAR-10 TRAINING SCRIPT
================================
Training script with fixed MAE detector and optimized parameters
"""

import sys
import os
import torch
import logging
from datetime import datetime

# Force CIFAR-10 dataset
sys.argv = ['run_ultimate_cifar10.py', '--dataset', 'cifar10', '--mode', 'full']

def setup_logging():
    """Setup logging configuration"""
    log_filename = f"logs/ultimate_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return log_filename

def patch_mae_detector():
    """Patch the MAE detector to use fixed implementation"""
    try:
        # Replace the MAE detector import in the defense module
        import defense.mae_detector as mae_module
        from defense.mae_detector_fixed import MAEDetector as FixedMAEDetector
        
        # Monkey patch the MAE detector class
        mae_module.MAEDetector = FixedMAEDetector
        print("MAE detector patched with fixed implementation")
        
        # Also patch the main module if it imports MAE detector directly
        import sys
        if 'defense.mae_detector1' in sys.modules:
            import defense.mae_detector1 as mae1_module
            mae1_module.MAEDetector = FixedMAEDetector
            
    except Exception as e:
        print(f"Could not patch MAE detector: {e}")

def patch_config():
    """Patch the configuration to use ultimate config"""
    try:
        from config_ultimate import get_config
        import utils.args as args_module
        
        # Get ultimate config
        ultimate_config = get_config()
        
        # Patch the get_config function in args module
        def patched_get_config():
            return ultimate_config
            
        args_module.get_config = patched_get_config
        print("Configuration patched with ultimate config")
        
    except Exception as e:
        print(f"Could not patch config: {e}")

def main_training():
    """Main training function with fixes"""
    print("Starting Ultimate CIFAR-10 Training")
    print("=" * 50)
    
    # Setup logging
    log_file = setup_logging()
    logging.info("Ultimate CIFAR-10 training started")
    logging.info(f"Log file: {log_file}")
    
    # Apply patches
    patch_mae_detector()
    patch_config()
    
    # Import main after patching
    from main import main
    
    try:
        # Run main training
        success = main()
        logging.info("Training completed successfully!")
        print("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        print(f"Training failed: {e}")
        success = False
        raise
    
    return success

if __name__ == "__main__":
    success = main_training()
    if success:
        print("\nULTIMATE CIFAR-10 training completed!")
    else:
        print("\nUltimate training failed!")
        sys.exit(1)
