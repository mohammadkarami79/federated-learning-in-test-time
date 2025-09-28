#!/usr/bin/env python3
"""
ADVERSARIAL ACCURACY OPTIMIZED TRAINING
======================================
Training script specifically optimized for 40-60% adversarial accuracy
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    
    print("ADVERSARIAL ACCURACY OPTIMIZED TRAINING")
    print("=" * 50)
    
    try:
        # Use adversarial optimized config
        from config_adversarial_optimized import get_adversarial_optimized_config
        
        # Patch the config in utils.args
        import utils.args as args_module
        args_module.get_ultimate_config = get_adversarial_optimized_config
        logger.info("Adversarial optimized config loaded")
        
        # Run main training
        from main import main as run_main
        logger.info("Starting adversarial accuracy optimized training...")
        
        return run_main()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
