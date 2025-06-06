#!/usr/bin/env python3
"""
Train MAE Detector for Different Datasets
"""
import sys
sys.path.append('/teamspace/studios/this_studio/federated-learning-in-test-time')

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
    """Parse arguments"""
    parser = argparse.ArgumentParser(description='Train MAE Detector')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100', 'mnist'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    return parser.parse_args()

def get_config_for_dataset(dataset):
    """Get configuration for dataset"""
    from config_fixed import get_debug_config
    cfg = get_debug_config()
    
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
    
    cfg.DATASET = dataset
    cfg.DATASET_NAME = dataset.upper()
    
    return cfg

def train_mae_detector(cfg, epochs=5):
    """Train MAE detector for the given dataset"""
    logger = logging.getLogger(__name__)
    
    try:
        from defense.mae_detector import MAEDetector
        from utils.data_utils import get_dataset
        import torch.utils.data as data_utils
        
        # Load dataset
        train_dataset, _ = get_dataset(cfg)
        train_loader = data_utils.DataLoader(
            train_dataset, 
            batch_size=32, 
            shuffle=True
        )
        
        # Create MAE detector
        detector = MAEDetector(cfg)
        
        # Train the detector
        detector.train(train_loader, epochs=epochs)
        
        # Save the detector
        detector.save()
        
        logger.info(f"✅ MAE detector training completed for {cfg.DATASET_NAME}")
        return True
        
    except Exception as e:
        logger.error(f"❌ MAE detector training failed: {e}")
        return False

def main():
    """Main function"""
    logger = setup_logging()
    args = parse_args()
    
    logger.info(f"🔍 Training MAE detector for {args.dataset.upper()}")
    logger.info(f"⚙️ Settings: {args.epochs} epochs, batch size {args.batch_size}")
    
    # Get configuration
    cfg = get_config_for_dataset(args.dataset)
    
    # Train detector
    success = train_mae_detector(cfg, args.epochs)
    
    if success:
        logger.info("🎉 MAE detector training completed successfully!")
        return 0
    else:
        logger.error("❌ MAE detector training failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 