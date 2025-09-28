#!/usr/bin/env python3
"""
Safe Mode CIFAR-10 Training with Memory Management
"""

import os
import sys
import torch
import logging
import gc

# Force single-threaded to avoid deadlocks
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)

# Import main after setting environment
sys.path.append('.')
from main import main
import argparse

def safe_main():
    """Run main with memory management"""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("🛡️  SAFE MODE CIFAR-10 TRAINING")
    logger.info("=" * 50)
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("🧹 GPU cache cleared")
    
    # Force garbage collection
    gc.collect()
    logger.info("🗑️  Garbage collection completed")
    
    # Set safe arguments
    sys.argv = [
        'safe_cifar10_run.py',
        '--dataset', 'cifar10',
        '--mode', 'full',
        '--skip-setup'
    ]
    
    try:
        logger.info("🚀 Starting safe training...")
        result = main()
        logger.info("✅ Training completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(safe_main())
