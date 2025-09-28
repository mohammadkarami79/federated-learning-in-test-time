#!/usr/bin/env python3
"""
BR35H Ultra Mild Defense
MAE Detection + Ultra Mild DiffPure for 85-90% adversarial accuracy
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
    log_file = log_dir / "br35h_ultra_mild.log"
    
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
    logger.info("🛡️ Starting BR35H Ultra Mild Defense")
    logger.info("Goal: MAE Detection + Ultra Mild DiffPure for 85-90% adversarial accuracy")
    
    try:
        # Import ultra mild configuration
        from config_br35h_ultra_mild import get_config
        
        # Get ultra mild configuration
        cfg_dict = get_config()
        logger.info(f"Loaded BR35H ultra mild configuration: {len(cfg_dict)} parameters")
        logger.info(f"MAE_THRESHOLD: {cfg_dict['MAE_THRESHOLD']} (working)")
        logger.info(f"DIFFUSER_STEPS: {cfg_dict['DIFFUSER_STEPS']} (only 1 step)")
        logger.info(f"DIFFUSER_SIGMA: {cfg_dict['DIFFUSER_SIGMA']} (very low)")
        logger.info(f"DIFFPURE_STRENGTH: {cfg_dict['DIFFPURE_STRENGTH']} (very gentle - 5%)")
        logger.info(f"TARGET_ADV_ACC: {cfg_dict['TARGET_ADV_ACC']}% (85-90%)")
        
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
        
        logger.info("Starting federated training with ultra mild defense...")
        success = run_federated_training(cfg_obj, test_loader)
        
        if success:
            logger.info("✅ BR35H ultra mild defense completed!")
            
            # Check results
            results_file = project_root / "experiment_results" / "latest_results_br35h.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    
                clean_acc = results['final_metrics']['clean_accuracy']
                adv_acc = results['final_metrics']['adversarial_accuracy']
                det_rate = results['final_metrics']['mae_detection_rate']
                
                logger.info(f"🛡️ Ultra Mild Defense Results:")
                logger.info(f"  Clean Accuracy: {clean_acc:.2f}%")
                logger.info(f"  Adversarial Accuracy: {adv_acc:.2f}%")
                logger.info(f"  Detection Rate: {det_rate:.2f}%")
                
                # Calculate gap
                gap = clean_acc - adv_acc
                logger.info(f"  Accuracy Gap: {gap:.2f}%")
                
                # Analysis
                if adv_acc >= 85:
                    logger.info("🎉 EXCELLENT: Target adversarial accuracy achieved!")
                    logger.info("   → Ultra mild defense is working perfectly")
                elif adv_acc >= 80:
                    logger.info("✅ VERY GOOD: High adversarial accuracy achieved!")
                    logger.info("   → Ultra mild defense is working well")
                elif adv_acc >= 70:
                    logger.info("✅ GOOD: Good adversarial accuracy achieved!")
                    logger.info("   → Ultra mild defense is working")
                elif adv_acc >= 60:
                    logger.info("⚠️ IMPROVEMENT: Better adversarial accuracy")
                    logger.info("   → Ultra mild defense needs fine-tuning")
                else:
                    logger.warning("❌ STILL LOW: Adversarial accuracy needs work")
                    logger.info("   → Need to make DiffPure even more gentle")
                    
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
                    logger.info("🏆 SUCCESS: Perfect ultra mild defense!")
                    logger.info("   → Ready for production use")
                elif adv_acc >= 80 and gap <= 20:
                    logger.info("🎉 EXCELLENT: Very good ultra mild defense!")
                    logger.info("   → High quality defense achieved")
                elif adv_acc >= 70 and gap <= 25:
                    logger.info("✅ GOOD: Good ultra mild defense!")
                    logger.info("   → Solid defense performance")
                elif adv_acc >= 60 and gap <= 30:
                    logger.info("⚠️ ACCEPTABLE: Ultra mild defense needs tuning")
                    logger.info("   → Can be improved with parameter adjustment")
                else:
                    logger.info("❌ NEEDS WORK: Ultra mild defense needs improvement")
                    logger.info("   → Need to investigate and adjust parameters")
                    
                # Specific recommendations
                if adv_acc < 70:
                    logger.info("💡 RECOMMENDATIONS:")
                    logger.info("   → Reduce DIFFPURE_STRENGTH further (currently 0.05)")
                    logger.info("   → Reduce DIFFUSER_SIGMA further (currently 0.05)")
                    logger.info("   → Consider single-step purification only")
                elif adv_acc < 80:
                    logger.info("💡 RECOMMENDATIONS:")
                    logger.info("   → Fine-tune DIFFPURE_STRENGTH")
                    logger.info("   → Adjust DIFFUSER_SIGMA")
                    logger.info("   → Consider adaptive purification")
                else:
                    logger.info("💡 RECOMMENDATIONS:")
                    logger.info("   → Current parameters are working well!")
                    logger.info("   → Consider increasing purification for better defense")
        else:
            logger.error("❌ Ultra mild defense failed")
            
    except Exception as e:
        logger.error(f"Ultra mild defense failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
