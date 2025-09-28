#!/usr/bin/env python3
"""
BR35H Ultra Gentle - Final Solution
MAE detection works, DiffPure is ultra gentle to preserve adversarial accuracy
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
    log_file = log_dir / "br35h_ultra_gentle.log"
    
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
    logger.info("🎯 Starting BR35H Ultra Gentle - Final Solution")
    logger.info("MAE detection: 9.33% (working) | DiffPure: Ultra gentle")
    
    try:
        # Import ultra gentle configuration
        from config_br35h_ultra_gentle import get_config
        
        # Get ultra gentle configuration
        cfg_dict = get_config()
        logger.info(f"Loaded BR35H ultra gentle configuration: {len(cfg_dict)} parameters")
        logger.info(f"MAE_THRESHOLD: {cfg_dict['MAE_THRESHOLD']} (working)")
        logger.info(f"DIFFUSER_STEPS: {cfg_dict['DIFFUSER_STEPS']} (ultra gentle)")
        logger.info(f"DIFFUSER_SIGMA: {cfg_dict['DIFFUSER_SIGMA']} (ultra gentle)")
        
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
        
        logger.info("Starting federated training with ultra gentle DiffPure...")
        success = run_federated_training(cfg_obj, test_loader)
        
        if success:
            logger.info("✅ BR35H ultra gentle training completed!")
            
            # Check results
            results_file = project_root / "experiment_results" / "latest_results_br35h.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    
                clean_acc = results['final_metrics']['clean_accuracy']
                adv_acc = results['final_metrics']['adversarial_accuracy']
                det_rate = results['final_metrics']['mae_detection_rate']
                
                logger.info(f"🎯 Ultra Gentle Results:")
                logger.info(f"  Clean Accuracy: {clean_acc:.2f}%")
                logger.info(f"  Adversarial Accuracy: {adv_acc:.2f}%")
                logger.info(f"  Detection Rate: {det_rate:.2f}%")
                
                # Calculate gap
                gap = clean_acc - adv_acc
                logger.info(f"  Accuracy Gap: {gap:.2f}%")
                
                # Check if solution worked
                if det_rate < 20:
                    logger.info("✅ SUCCESS: MAE detection rate is good!")
                else:
                    logger.warning("⚠️ MAE detection rate still too high")
                    
                if adv_acc >= 80:
                    logger.info("🎉 EXCELLENT: High adversarial accuracy achieved!")
                elif adv_acc >= 70:
                    logger.info("✅ GOOD: Adversarial accuracy is acceptable")
                elif adv_acc >= 60:
                    logger.info("⚠️ IMPROVEMENT: Adversarial accuracy is better")
                else:
                    logger.warning("❌ Adversarial accuracy still needs work")
                    
                if gap <= 15:
                    logger.info("🎉 EXCELLENT: Small accuracy gap achieved!")
                elif gap <= 25:
                    logger.info("✅ GOOD: Reasonable accuracy gap")
                else:
                    logger.warning("⚠️ Large accuracy gap - may need more tuning")
                    
                # Final assessment
                if adv_acc >= 75 and gap <= 20:
                    logger.info("🏆 SUCCESS: BR35H defense is working well!")
                elif adv_acc >= 65 and gap <= 30:
                    logger.info("✅ GOOD: BR35H defense is acceptable")
                else:
                    logger.info("⚠️ NEEDS WORK: BR35H defense needs more tuning")
        else:
            logger.error("❌ Ultra gentle training failed")
            
    except Exception as e:
        logger.error(f"Ultra gentle training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
