#!/usr/bin/env python3
"""
BR35H Minimal DiffPure Test
Test: Minimal DiffPure that does almost nothing
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
    log_file = log_dir / "br35h_minimal_diffpure.log"
    
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
    logger.info("🔬 Starting BR35H Minimal DiffPure Test")
    logger.info("Goal: Test if minimal DiffPure preserves adversarial accuracy")
    
    try:
        # Import minimal DiffPure configuration
        from config_br35h_minimal_diffpure import get_config
        
        # Get minimal DiffPure configuration
        cfg_dict = get_config()
        logger.info(f"Loaded BR35H minimal DiffPure configuration: {len(cfg_dict)} parameters")
        logger.info(f"DIFFUSER_STEPS: {cfg_dict['DIFFUSER_STEPS']} (minimal)")
        logger.info(f"DIFFUSER_SIGMA: {cfg_dict['DIFFUSER_SIGMA']} (minimal)")
        
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
        
        logger.info("Starting federated training with minimal DiffPure...")
        success = run_federated_training(cfg_obj, test_loader)
        
        if success:
            logger.info("✅ BR35H minimal DiffPure test completed!")
            
            # Check results
            results_file = project_root / "experiment_results" / "latest_results_br35h.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    
                clean_acc = results['final_metrics']['clean_accuracy']
                adv_acc = results['final_metrics']['adversarial_accuracy']
                det_rate = results['final_metrics']['mae_detection_rate']
                
                logger.info(f"🔬 Minimal DiffPure Test Results:")
                logger.info(f"  Clean Accuracy: {clean_acc:.2f}%")
                logger.info(f"  Adversarial Accuracy: {adv_acc:.2f}%")
                logger.info(f"  Detection Rate: {det_rate:.2f}%")
                
                # Calculate gap
                gap = clean_acc - adv_acc
                logger.info(f"  Accuracy Gap: {gap:.2f}%")
                
                # Analysis
                if adv_acc >= 80:
                    logger.info("🎉 SUCCESS: High adversarial accuracy with minimal DiffPure!")
                    logger.info("   → Minimal DiffPure works - can increase gradually")
                elif adv_acc >= 60:
                    logger.info("✅ IMPROVEMENT: Better adversarial accuracy with minimal DiffPure")
                    logger.info("   → Minimal DiffPure helps - need to tune carefully")
                else:
                    logger.warning("❌ STILL BAD: Low adversarial accuracy even with minimal DiffPure")
                    logger.info("   → Problem is NOT with DiffPure strength - it's deeper")
                    
                if gap <= 15:
                    logger.info("🎉 EXCELLENT: Small accuracy gap with minimal DiffPure!")
                elif gap <= 25:
                    logger.info("✅ GOOD: Reasonable accuracy gap with minimal DiffPure")
                else:
                    logger.warning("⚠️ Large accuracy gap even with minimal DiffPure")
                    
                # Final assessment
                if adv_acc >= 75 and gap <= 20:
                    logger.info("🏆 SUCCESS: BR35H works well with minimal DiffPure!")
                    logger.info("   → Can gradually increase DiffPure strength")
                elif adv_acc >= 65 and gap <= 30:
                    logger.info("✅ GOOD: BR35H is acceptable with minimal DiffPure")
                    logger.info("   → Need to tune DiffPure parameters carefully")
                else:
                    logger.info("❌ PROBLEM: BR35H has fundamental issues")
                    logger.info("   → Need to investigate deeper problems")
        else:
            logger.error("❌ Minimal DiffPure test failed")
            
    except Exception as e:
        logger.error(f"Minimal DiffPure test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
