#!/usr/bin/env python3
"""
Fixed MAE training script
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import logging
from pathlib import Path

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train MAE Detector')
    parser.add_argument('--dataset', type=str, default='br35h', choices=['mnist', 'cifar10', 'cifar100', 'br35h'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    return parser.parse_args()

def get_config_for_dataset(dataset):
    """Get configuration for dataset"""
    from config_fixed import get_debug_config
    cfg = get_debug_config()
    
    if dataset == 'mnist':
        cfg.IMG_CHANNELS = 1
        cfg.IMG_SIZE = 28
        cfg.NUM_CLASSES = 10
    elif dataset == 'cifar100':
        cfg.NUM_CLASSES = 100
        cfg.IMG_CHANNELS = 3
        cfg.IMG_SIZE = 32
    elif dataset == 'br35h':
        cfg.IMG_SIZE = 224
        cfg.IMG_CHANNELS = 3
        cfg.NUM_CLASSES = 2
    else:  # cifar10
        cfg.NUM_CLASSES = 10
        cfg.IMG_CHANNELS = 3
        cfg.IMG_SIZE = 32
    
    cfg.DATASET = dataset.upper()
    cfg.DATASET_NAME = dataset.upper()
    
    # Add missing attributes for MAE detector
    cfg.LR = getattr(cfg, 'LEARNING_RATE', 0.001)
    cfg.EPOCHS = getattr(cfg, 'NUM_EPOCHS', 10)
    
    return cfg

def train_mae_detector_fixed(cfg, epochs=30):
    """Train MAE detector with proper dataset handling"""
    logger = logging.getLogger(__name__)
    
    try:
        from mae_detector_fixed import MAEDetector
        from utils.datasets.br35h import Br35HDataset
        from utils.data_utils import get_data_transforms
        import torch.utils.data as data_utils
        import os
        
        # Load dataset PROPERLY for br35h
        if cfg.DATASET.lower() == 'br35h':
            transform = get_data_transforms('br35h', train=True)
            dataset = Br35HDataset(
                root_dir=os.path.join(getattr(cfg, 'DATA_ROOT', 'data'), 'Br35H'),
                transform=transform
            )
            logger.info(f"✅ BR35H dataset loaded: {len(dataset)} samples")
        else:
            # For other datasets, use the original method
            result = get_dataset(cfg.DATASET, getattr(cfg, 'DATA_ROOT', 'data'))
            if isinstance(result, tuple):
                dataset, _ = result
            else:
                dataset = result
                
        train_loader = data_utils.DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(f"✅ DataLoader created: {len(train_loader)} batches")
        
        # Create MAE detector
        detector = MAEDetector(cfg)
        logger.info("✅ MAE detector created")
        
        # Train the detector
        detector.train(train_loader, epochs=epochs)
        
        # Save the detector
        detector.save()
        
        logger.info(f"✅ MAE detector training completed for {cfg.DATASET}")
        return True
        
    except Exception as e:
        import traceback
        logger.error(f"❌ MAE detector training failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def main():
    """Main function"""
    logger = setup_logging()
    args = parse_args()
    
    logger.info(f"🔍 Training MAE detector for {args.dataset.upper()}")
    logger.info(f"⚙️ Settings: {args.epochs} epochs, batch size {args.batch_size}")
    
    cfg = get_config_for_dataset(args.dataset)
    
    success = train_mae_detector_fixed(cfg, epochs=args.epochs)
    
    if success:
        logger.info("✅ MAE detector training completed successfully!")
    else:
        logger.error("❌ MAE detector training failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
