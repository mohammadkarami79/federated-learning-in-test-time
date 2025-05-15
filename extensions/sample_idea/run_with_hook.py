#!/usr/bin/env python
"""
Run script for testing the sample defense hook extension.
This script enables the new defense hook and runs a test evaluation.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import project modules
from config import get_config, DEVICE
from diffusion.diffuser import UNet
from models.pfeddef_model import pFedDefModel
from attacks.pgd import PGDAttack
from extensions.sample_idea.defense_hook import NewDefenseHook  # Import our custom hook

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the test with custom defense hook."""
    # Get configuration with debug preset
    cfg = get_config('debug')
    
    # Enable the extension hook
    setattr(cfg, 'ENABLE_NEW_HOOK', True)
    logger.info("New defense hook enabled")
    
    # Create minimal models for testing
    logger.info("Initializing models...")
    diffuser = UNet(in_channels=3, hidden_channels=8).to(DEVICE)
    pfeddef_model = pFedDefModel(cfg).to(DEVICE)
    
    # Create PGD attack for evaluation
    attack = PGDAttack(
        epsilon=8/255,
        steps=5,
        step_size=2/255
    )
    
    # Create our custom classifier with the extension hook
    logger.info("Creating classifier with defense hook...")
    classifier = NewDefenseHook(
        diffuser=diffuser,
        pfeddef_model=pfeddef_model,
        cfg=cfg
    ).to(DEVICE)
    
    # Generate random test data
    logger.info("Generating test data...")
    test_images = torch.randn(2, 3, 32, 32).to(DEVICE)
    test_images = torch.clamp(test_images * 0.1 + 0.5, 0, 1)
    test_labels = torch.randint(0, 10, (2,)).to(DEVICE)
    
    # Run evaluation
    try:
        logger.info("Generating adversarial examples...")
        adv_images = attack.perturb(pfeddef_model, test_images, test_labels)
        
        # Evaluate with the extension enabled
        logger.info("Evaluating with defense hook...")
        classifier.eval()
        with torch.no_grad():
            output = classifier(adv_images)
            pred = output.argmax(dim=1)
            accuracy = (pred == test_labels).float().mean().item()
            attack_success_rate = 1.0 - accuracy
        
        # Print results
        logger.info("\n===== Defense Hook Evaluation =====")
        logger.info(f"Hook enabled: {classifier.enable_hook}")
        logger.info(f"Accuracy against adversarial examples: {accuracy:.4f}")
        logger.info(f"Attack success rate: {attack_success_rate:.4f}")
        logger.info("===================================\n")
        
        # Check if defense meets requirements
        if attack_success_rate <= 0.40:
            logger.info("✅ Defense meets requirements (attack success rate <= 0.40)")
        else:
            logger.info("❌ Defense does NOT meet requirements (attack success rate > 0.40)")
            
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        
if __name__ == "__main__":
    main() 