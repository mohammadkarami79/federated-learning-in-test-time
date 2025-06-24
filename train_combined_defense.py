"""
Main script for training combined DiffPure + pFedDef defense
"""

import argparse
import logging
import torch
from pathlib import Path

from config import Config, get_config
from federated.trainer import run_federated
from utils.data_utils import create_data_loaders
from utils.logger import MetricsLogger

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', type=str, default='debug',
                      choices=['debug', 'full'],
                      help='Configuration preset to use')
    parser.add_argument('--demo', action='store_true',
                      help='Run in demo mode with minimal settings')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Get configuration
    cfg = get_config(args.preset)
    if args.demo:
        cfg.N_ROUNDS = 1
        cfg.LOCAL_EPOCHS = 1
        cfg.BATCH_SIZE = 8
        cfg.N_CLIENTS = 2
        cfg.N_LEARNERS = 2
        cfg.DIFFUSER_STEPS = 4
        cfg.DIFFUSER_SIGMA = 0.04
        cfg.PGD_STEPS = 10
        
    # Set experiment name
    cfg.experiment_name = f"combined_defense_{args.preset}"
    if args.demo:
        cfg.experiment_name += "_demo"
        
    # Create output directories
    output_dir = Path('results') / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(output_dir)
    
    # Run training
    logging.info(f"Starting training with preset: {args.preset}")
    if args.demo:
        logging.info("Running in demo mode")
        
    run_federated(cfg)
    
    logging.info("Training complete!")
    
if __name__ == '__main__':
    main() 