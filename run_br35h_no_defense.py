#!/usr/bin/env python3
"""
BR35H No Defense Test
Test: Disable all defense mechanisms to isolate the problem
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
    log_file = log_dir / "br35h_no_defense.log"
    
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
    logger.info("🧪 Starting BR35H No Defense Test")
    logger.info("Goal: Test if the problem is with defense mechanisms")
    
    try:
        # Import no defense configuration
        from config_br35h_no_defense import get_config
        
        # Get no defense configuration
        cfg_dict = get_config()
        logger.info(f"Loaded BR35H no defense configuration: {len(cfg_dict)} parameters")
        logger.info(f"ENABLE_MAE_DETECTOR: {cfg_dict['ENABLE_MAE_DETECTOR']} (DISABLED)")
        logger.info(f"ENABLE_DIFFPURE: {cfg_dict['ENABLE_DIFFPURE']} (DISABLED)")
        logger.info(f"SELECTIVE_DEFENSE: {cfg_dict['SELECTIVE_DEFENSE']} (DISABLED)")
        
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
        
        logger.info("Starting federated training without any defense...")
        success = run_federated_training(cfg_obj, test_loader)
        
        if success:
            logger.info("✅ BR35H no defense test completed!")
            
            # Check results
            results_file = project_root / "experiment_results" / "latest_results_br35h.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    
                clean_acc = results['final_metrics']['clean_accuracy']
                adv_acc = results['final_metrics']['adversarial_accuracy']
                det_rate = results['final_metrics']['mae_detection_rate']
                
                logger.info(f"🧪 No Defense Test Results:")
                logger.info(f"  Clean Accuracy: {clean_acc:.2f}%")
                logger.info(f"  Adversarial Accuracy: {adv_acc:.2f}%")
                logger.info(f"  Detection Rate: {det_rate:.2f}%")
                
                # Calculate gap
                gap = clean_acc - adv_acc
                logger.info(f"  Accuracy Gap: {gap:.2f}%")
                
                # Analysis
                if adv_acc >= 85:
                    logger.info("🎉 EXCELLENT: High adversarial accuracy without defense!")
                    logger.info("   → Problem is with defense mechanisms")
                    logger.info("   → Need to fix DiffPure or MAE implementation")
                elif adv_acc >= 70:
                    logger.info("✅ GOOD: Good adversarial accuracy without defense!")
                    logger.info("   → Problem is partially with defense mechanisms")
                    logger.info("   → Need to improve defense implementation")
                elif adv_acc >= 60:
                    logger.info("⚠️ MODERATE: Moderate adversarial accuracy without defense")
                    logger.info("   → Problem is partially with defense mechanisms")
                    logger.info("   → Need to investigate both defense and model")
                else:
                    logger.warning("❌ LOW: Low adversarial accuracy even without defense")
                    logger.info("   → Problem is NOT with defense mechanisms")
                    logger.info("   → Need to investigate model robustness")
                    
                if gap <= 10:
                    logger.info("🎉 EXCELLENT: Very small accuracy gap!")
                elif gap <= 15:
                    logger.info("✅ VERY GOOD: Small accuracy gap!")
                elif gap <= 20:
                    logger.info("✅ GOOD: Reasonable accuracy gap")
                elif gap <= 30:
                    logger.info("⚠️ MODERATE: Acceptable accuracy gap")
                else:
                    logger.warning("❌ LARGE: Accuracy gap is too large")
                    
                # Final assessment
                if adv_acc >= 85 and gap <= 15:
                    logger.info("🏆 SUCCESS: Model is robust without defense!")
                    logger.info("   → Problem is definitely with defense mechanisms")
                    logger.info("   → Need to fix DiffPure or MAE implementation")
                elif adv_acc >= 70 and gap <= 25:
                    logger.info("🎉 EXCELLENT: Model is good without defense!")
                    logger.info("   → Problem is mostly with defense mechanisms")
                    logger.info("   → Need to improve defense implementation")
                elif adv_acc >= 60 and gap <= 30:
                    logger.info("✅ GOOD: Model is acceptable without defense!")
                    logger.info("   → Problem is partially with defense mechanisms")
                    logger.info("   → Need to investigate both defense and model")
                else:
                    logger.info("❌ PROBLEM: Model has fundamental robustness issues")
                    logger.info("   → Problem is NOT with defense mechanisms")
                    logger.info("   → Need to investigate model training and architecture")
                    
                # Specific recommendations
                if adv_acc >= 80:
                    logger.info("💡 RECOMMENDATIONS:")
                    logger.info("   → Model is robust - problem is with defense")
                    logger.info("   → Fix DiffPure implementation")
                    logger.info("   → Fix MAE detection logic")
                elif adv_acc >= 60:
                    logger.info("💡 RECOMMENDATIONS:")
                    logger.info("   → Model is partially robust")
                    logger.info("   → Improve model training")
                    logger.info("   → Fix defense implementation")
                else:
                    logger.info("💡 RECOMMENDATIONS:")
                    logger.info("   → Model has fundamental issues")
                    logger.info("   → Improve model architecture")
                    logger.info("   → Improve training procedure")
        else:
            logger.error("❌ No defense test failed")
            
    except Exception as e:
        logger.error(f"No defense test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
