#!/usr/bin/env python3
"""
BR35H Emergency Fix - Stop the 100% detection problem
"""

import sys
import os
import torch
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup logging configuration"""
    import logging
    
    # Create logs directory if it doesn't exist
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Setup logging
    log_file = log_dir / "br35h_emergency_fix.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def main():
    """Main training function"""
    logger = setup_logging()
    logger.info("🚨 Starting BR35H Emergency Fix - Stop 100% Detection Problem")
    
    try:
        # Import emergency fix configuration
        from config_br35h_emergency_fix import get_config
        
        # Get emergency fix configuration
        cfg_dict = get_config()
        logger.info(f"Loaded BR35H emergency fix configuration: {len(cfg_dict)} parameters")
        logger.info(f"MAE_THRESHOLD: {cfg_dict['MAE_THRESHOLD']}")
        logger.info(f"ADAPTIVE_THRESHOLD: {cfg_dict['ADAPTIVE_THRESHOLD']}")
        logger.info(f"TARGET_DETECTION_RATE: {cfg_dict['TARGET_DETECTION_RATE']}")
        
        # Convert to namespace object for compatibility
        import types
        cfg_obj = types.SimpleNamespace()
        for key, value in cfg_dict.items():
            setattr(cfg_obj, key, value)
        
        # Import data utilities
        from utils.data_utils import get_dataset
        
        # Load test dataset for evaluation
        logger.info("Loading BR35H dataset...")
        _, test_dataset = get_dataset(cfg_obj, train=False)
        
        # Create test loader
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg_obj.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(f"Test dataset loaded: {len(test_dataset)} samples")
        
        # Import and run federated training
        from main import run_federated_training
        
        logger.info("Starting federated training with emergency fix...")
        success = run_federated_training(cfg_obj, test_loader)
        
        if success:
            logger.info("✅ BR35H emergency fix training completed!")
            
            # Check results
            results_file = project_root / "experiment_results" / "latest_results_br35h.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    
                clean_acc = results['final_metrics']['clean_accuracy']
                adv_acc = results['final_metrics']['adversarial_accuracy']
                det_rate = results['final_metrics']['mae_detection_rate']
                
                logger.info(f"🎯 Emergency Fix Results:")
                logger.info(f"  Clean Accuracy: {clean_acc:.2f}%")
                logger.info(f"  Adversarial Accuracy: {adv_acc:.2f}%")
                logger.info(f"  Detection Rate: {det_rate:.2f}%")
                
                # Calculate gap
                gap = clean_acc - adv_acc
                logger.info(f"  Accuracy Gap: {gap:.2f}%")
                
                # Check if fix worked
                if det_rate < 30:
                    logger.info("✅ SUCCESS: Detection rate fixed!")
                else:
                    logger.warning("⚠️ Detection rate still too high")
                    
                if adv_acc >= 80:
                    logger.info("🎉 EXCELLENT: High adversarial accuracy achieved!")
                elif adv_acc >= 70:
                    logger.info("✅ GOOD: Adversarial accuracy is acceptable")
                else:
                    logger.warning("⚠️ Adversarial accuracy still needs improvement")
                    
                if gap <= 15:
                    logger.info("✅ EXCELLENT: Small accuracy gap achieved!")
                elif gap <= 25:
                    logger.info("✅ GOOD: Reasonable accuracy gap")
                else:
                    logger.warning("⚠️ Large accuracy gap - needs more tuning")
        else:
            logger.error("❌ Emergency fix training failed")
            
    except Exception as e:
        logger.error(f"Emergency fix failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
