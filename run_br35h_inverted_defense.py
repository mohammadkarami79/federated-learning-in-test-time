#!/usr/bin/env python3
"""
BR35H Inverted Defense
MAE Detection + Inverted DiffPure (skip adversarial, purify clean)
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
    log_file = log_dir / "br35h_inverted_defense.log"
    
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
    logger.info("🔄 Starting BR35H Inverted Defense")
    logger.info("Goal: MAE Detection + Inverted DiffPure (skip adversarial, purify clean)")
    
    try:
        # Import inverted defense configuration
        from config_br35h_inverted_defense import get_config
        
        # Get inverted defense configuration
        cfg_dict = get_config()
        logger.info(f"Loaded BR35H inverted defense configuration: {len(cfg_dict)} parameters")
        logger.info(f"MAE_THRESHOLD: {cfg_dict['MAE_THRESHOLD']} (working)")
        logger.info(f"INVERTED_DEFENSE: {cfg_dict['INVERTED_DEFENSE']} (NEW)")
        logger.info(f"DIFFUSER_STEPS: {cfg_dict['DIFFUSER_STEPS']} (moderate)")
        logger.info(f"DIFFUSER_SIGMA: {cfg_dict['DIFFUSER_SIGMA']} (moderate)")
        logger.info(f"TARGET_ADV_ACC: {cfg_dict['TARGET_ADV_ACC']}% (85-87%)")
        
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
        
        logger.info("Starting federated training with inverted defense...")
        success = run_federated_training(cfg_obj, test_loader)
        
        if success:
            logger.info("✅ BR35H inverted defense completed!")
            
            # Check results
            results_file = project_root / "experiment_results" / "latest_results_br35h.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    
                clean_acc = results['final_metrics']['clean_accuracy']
                adv_acc = results['final_metrics']['adversarial_accuracy']
                det_rate = results['final_metrics']['mae_detection_rate']
                
                logger.info(f"🔄 Inverted Defense Results:")
                logger.info(f"  Clean Accuracy: {clean_acc:.2f}%")
                logger.info(f"  Adversarial Accuracy: {adv_acc:.2f}%")
                logger.info(f"  Detection Rate: {det_rate:.2f}%")
                
                # Calculate gap
                gap = clean_acc - adv_acc
                logger.info(f"  Accuracy Gap: {gap:.2f}%")
                
                # Analysis
                if adv_acc >= 85:
                    logger.info("🎉 EXCELLENT: Target adversarial accuracy achieved!")
                    logger.info("   → Inverted defense is working perfectly")
                elif adv_acc >= 80:
                    logger.info("✅ VERY GOOD: High adversarial accuracy achieved!")
                    logger.info("   → Inverted defense is working well")
                elif adv_acc >= 70:
                    logger.info("✅ GOOD: Good adversarial accuracy achieved!")
                    logger.info("   → Inverted defense is working")
                elif adv_acc >= 60:
                    logger.info("⚠️ IMPROVEMENT: Better adversarial accuracy")
                    logger.info("   → Inverted defense needs fine-tuning")
                else:
                    logger.warning("❌ STILL LOW: Adversarial accuracy needs work")
                    logger.info("   → Need to investigate deeper issues")
                    
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
                    logger.info("🏆 SUCCESS: Perfect inverted defense!")
                    logger.info("   → Ready for production use")
                elif adv_acc >= 80 and gap <= 20:
                    logger.info("🎉 EXCELLENT: Very good inverted defense!")
                    logger.info("   → High quality defense achieved")
                elif adv_acc >= 70 and gap <= 25:
                    logger.info("✅ GOOD: Good inverted defense!")
                    logger.info("   → Solid defense performance")
                elif adv_acc >= 60 and gap <= 30:
                    logger.info("⚠️ ACCEPTABLE: Inverted defense needs tuning")
                    logger.info("   → Can be improved with parameter adjustment")
                else:
                    logger.info("❌ NEEDS WORK: Inverted defense needs improvement")
                    logger.info("   → Need to investigate and adjust parameters")
                    
                # Specific recommendations
                if adv_acc < 70:
                    logger.info("💡 RECOMMENDATIONS:")
                    logger.info("   → The inverted approach may not be working")
                    logger.info("   → Consider disabling DiffPure completely")
                    logger.info("   → Focus on improving model robustness")
                elif adv_acc < 80:
                    logger.info("💡 RECOMMENDATIONS:")
                    logger.info("   → Inverted defense is partially working")
                    logger.info("   → Fine-tune MAE detection threshold")
                    logger.info("   → Consider hybrid approach")
                else:
                    logger.info("💡 RECOMMENDATIONS:")
                    logger.info("   → Inverted defense is working well!")
                    logger.info("   → Consider increasing purification for clean samples")
        else:
            logger.error("❌ Inverted defense failed")
            
    except Exception as e:
        logger.error(f"Inverted defense failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
