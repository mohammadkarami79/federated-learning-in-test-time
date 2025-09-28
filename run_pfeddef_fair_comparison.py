#!/usr/bin/env python3
"""
PFEDDEF FAIR COMPARISON TRAINING
===============================
Training script that maintains attack strength for fair PFedDef comparison
while optimizing defense pipeline for better adversarial accuracy
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
    
    print("PFEDDEF FAIR COMPARISON TRAINING - CIFAR10")
    print("=" * 50)
    print("Maintains standard attack strength for fair comparison")
    print("Optimizes defense pipeline for better adversarial accuracy")
    print()
    
    try:
        # Force CIFAR10 arguments
        sys.argv = ['run_pfeddef_fair_comparison.py', '--dataset', 'cifar10', '--mode', 'full']
        
        # Use fair comparison config
        from config_pfeddef_fair_comparison import get_pfeddef_fair_config
        
        # Patch the config in utils.args
        import utils.args as args_module
        args_module.get_ultimate_config = get_pfeddef_fair_config
        logger.info("PFedDef fair comparison config loaded")
        logger.info("Attack strength: PGD eps=0.031, steps=10 (standard)")
        logger.info("Defense: Optimized MAE + DiffPure pipeline")
        
        # Run main training
        from main import main as run_main
        logger.info("Starting PFedDef fair comparison training...")
        
        return run_main()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
