#!/usr/bin/env python
"""
Train MAE detector for adversarial example detection
"""

import os
import sys
import argparse
import logging
import torch
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config, parse_args, DEVICE
from defense.mae_detector import MAEDetector
from utils.data_utils import get_dataloader

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'mae_detector_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments for MAE detector training."""
    parser = argparse.ArgumentParser(description='Train MAE detector for adversarial example detection')
    parser.add_argument('--preset', type=str, choices=['debug', 'full'], default='debug',
                       help='Configuration preset (debug or full)')
    return parser.parse_args()

def main():
    """Main function to train and evaluate the MAE detector"""
    # Parse arguments
    args = parse_args()
    
    # Get configuration based on preset
    cfg = get_config(args.preset)
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting MAE detector training with {args.preset} preset")
    logger.info(f"Using dataset: {cfg.DATASET_NAME} from {cfg.DATA_PATH}")
    
    # Initialize MAE detector
    detector = MAEDetector(cfg)
    
    # Train detector
    logger.info(f"Training for {cfg.EPOCHS_DETECTOR} epochs")
    detector.train(epochs=cfg.EPOCHS_DETECTOR)
    
    # Evaluate on test data
    logger.info("Evaluating on test data")
    detector.evaluate()
    
    # Save final checkpoint
    detector.save_checkpoint()
    
    logger.info("MAE detector training complete")

if __name__ == "__main__":
    main() 