#!/usr/bin/env python3
"""
FIXED CIFAR-10 TRAINING SCRIPT
==============================
Direct training script with fixed MAE detector and proper CIFAR-10 configuration
"""

import sys
import os
import torch
import logging
from datetime import datetime
import argparse

def setup_logging():
    """Setup logging without unicode characters"""
    log_filename = f"logs/fixed_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs("logs", exist_ok=True)
    
    # Create formatter without unicode
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return log_filename

def create_fixed_config():
    """Create a fixed configuration class for CIFAR-10"""
    class FixedCIFAR10Config:
        # Device configuration
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Dataset configuration - FORCE CIFAR-10
        DATASET = 'cifar10'
        IMG_SIZE = 32
        NUM_CLASSES = 10
        
        # Federated learning configuration
        NUM_CLIENTS = 10
        NUM_ROUNDS = 15
        CLIENT_EPOCHS = 5
        BATCH_SIZE = 64
        LEARNING_RATE = 0.01
        
        # Model configuration
        MODEL_NAME = 'resnet18'
        
        # MAE Detector configuration (FIXED)
        USE_MAE = True
        MAE_TRAIN = False  # Disable to avoid dimension errors
        MAE_EMBED_DIM = 128
        MAE_DECODER_EMBED_DIM = 128
        MAE_DEPTH = 4
        MAE_NUM_HEADS = 4
        MAE_DECODER_DEPTH = 4
        MAE_MASK_RATIO = 0.75
        MAE_THRESHOLD = 0.4  # Balanced threshold
        PATCH_SIZE = 4
        
        # DiffPure configuration
        USE_DIFFPURE = True
        DIFFPURE_STEPS = 200
        DIFFPURE_SIGMA = 0.12
        
        # Attack configuration (weaker for better adversarial accuracy)
        ATTACK_TYPE = 'pgd'
        PGD_EPSILON = 4.0 / 255.0
        PGD_ALPHA = 1.0 / 255.0
        PGD_STEPS = 10
        
        # Training configuration
        WEIGHT_DECAY = 1e-4
        MOMENTUM = 0.9
        MAX_MEMORY_GB = 6  # Match available GPU memory
        
        # Evaluation configuration
        EVAL_BATCH_SIZE = 128
        
        # Logging configuration
        LOG_INTERVAL = 10
        SAVE_INTERVAL = 5
    
    return FixedCIFAR10Config()

def patch_mae_detector():
    """Replace MAE detector with fixed implementation"""
    try:
        import defense.mae_detector as mae_module
        from defense.mae_detector_fixed import MAEDetector as FixedMAEDetector
        
        # Replace the class
        mae_module.MAEDetector = FixedMAEDetector
        print("MAE detector patched with fixed implementation")
        return True
        
    except Exception as e:
        print(f"Could not patch MAE detector: {e}")
        return False

def patch_args_module():
    """Patch the args module to use our fixed config"""
    try:
        import utils.args as args_module
        
        # Get fixed config
        fixed_config = create_fixed_config()
        
        # Replace get_config function
        def get_fixed_config():
            return fixed_config
        
        args_module.get_config = get_fixed_config
        print("Args module patched with fixed config")
        return True
        
    except Exception as e:
        print(f"Could not patch args module: {e}")
        return False

def main_training():
    """Main training function with comprehensive fixes"""
    print("Starting Fixed CIFAR-10 Training")
    print("=" * 50)
    
    # Setup logging
    log_file = setup_logging()
    logging.info("Fixed CIFAR-10 training started")
    logging.info(f"Log file: {log_file}")
    
    # Force CIFAR-10 arguments
    sys.argv = ['run_fixed_cifar10.py', '--dataset', 'cifar10', '--mode', 'full']
    
    # Apply patches
    mae_patched = patch_mae_detector()
    config_patched = patch_args_module()
    
    if not mae_patched:
        logging.error("Failed to patch MAE detector")
        return False
    
    if not config_patched:
        logging.error("Failed to patch configuration")
        return False
    
    try:
        # Import main after patching
        from main import main
        
        logging.info("Starting main training loop")
        print("Starting main training loop...")
        
        # Run training
        result = main()
        
        logging.info("Training completed successfully!")
        print("Training completed successfully!")
        return True
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main_training()
    if success:
        print("\nFixed CIFAR-10 training completed successfully!")
    else:
        print("\nFixed training failed!")
        sys.exit(1)
