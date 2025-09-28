#!/usr/bin/env python3
"""
Run Selective Defense for BR35H dataset.
Keeps CIFAR-10 pipeline untouched; this launcher loads BR35H config.
"""

import sys
from pathlib import Path
import torch
import json

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    import logging
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "selective_defense_br35h.log"
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
    logger = setup_logging()
    logger.info("Starting BR35H Selective Defense Training")
    try:
        from config_br35h_selective_defense import get_config
        cfg_dict = get_config()
        logger.info(f"Loaded BR35H configuration: {len(cfg_dict)} parameters")

        import types
        cfg_obj = types.SimpleNamespace()
        for k, v in cfg_dict.items():
            setattr(cfg_obj, k, v)

        from utils.data_utils import get_dataset
        logger.info("Loading BR35H dataset...")
        _, test_dataset = get_dataset(cfg_obj, train=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg_obj.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        from main import run_federated_training
        logger.info("Starting federated training with selective defense (BR35H)...")
        success = run_federated_training(cfg_obj, test_loader)
        if success:
            logger.info("✅ BR35H selective defense training completed successfully!")
            results_file = project_root / "experiment_results" / "latest_results_br35h.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                logger.info(f"Final Results:")
                logger.info(f"  Clean Accuracy: {results['final_metrics'].get('clean_accuracy','-')}%")
                logger.info(f"  Adversarial Accuracy: {results['final_metrics'].get('adversarial_accuracy','-')}%")
                logger.info(f"  Detection Rate: {results['final_metrics'].get('mae_detection_rate','-')}%")
        else:
            logger.error("❌ Training failed")
    except Exception as e:
        logger = setup_logging()
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True

if __name__ == "__main__":
    main()


