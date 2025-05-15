"""
Quick test for the combined defense
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import gc
from pathlib import Path

# Setup paths
sys.path.append('.')

from config import get_config, DEVICE
from diffusion.diffuser import UNet
from models.pfeddef_model import pFedDefModel
from attacks.pgd import PGDAttack
from defense.combined_defense import CombinedClassifier

# Set PyTorch memory optimizations
def optimize_memory():
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Collect garbage
    gc.collect()
    
    # Set PyTorch memory optimization flags
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def main():
    # Load configuration
    cfg = get_config('debug')
    
    # Apply memory optimizations
    optimize_memory()
    
    # Override config for smaller memory footprint
    cfg.BATCH_SIZE = 2
    cfg.DIFFUSER_STEPS = 3
    
    # Create models with smaller hidden dimensions
    diffuser = UNet(in_channels=3, hidden_channels=8).to(DEVICE)
    pfeddef_model = pFedDefModel(cfg).to(DEVICE)
    
    # Create attack with fewer steps
    attack = PGDAttack(
        epsilon=8/255,
        steps=5,
        step_size=2/255
    )
    
    # Create defense
    classifier = CombinedClassifier(
        diffuser=diffuser,
        pfeddef_model=pfeddef_model,
        cfg=cfg
    ).to(DEVICE)
    
    # Generate random data for testing (smaller batch size)
    test_images = torch.randn(2, 3, 32, 32).to(DEVICE)
    test_images = torch.clamp(test_images * 0.1 + 0.5, 0, 1)  # Make it more like real images
    test_labels = torch.randint(0, 10, (2,)).to(DEVICE)
    
    try:
        # Create adversarial examples
        print("Generating adversarial examples...")
        adv_images = attack.perturb(pfeddef_model, test_images, test_labels)
        
        # Clear memory
        optimize_memory()
        
        # Evaluate with and without defense
        classifier.eval()
        
        # Test without defense
        print("Evaluating without defense...")
        classifier.disable_defense = True
        with torch.no_grad():
            vanilla_output = classifier(adv_images)
            vanilla_pred = vanilla_output.argmax(dim=1)
            no_defense_accuracy = (vanilla_pred == test_labels).float().mean().item()
        
        # Free memory
        optimize_memory()
        
        # Test with defense
        print("Evaluating with defense...")
        classifier.disable_defense = False
        with torch.no_grad():
            defended_output = classifier(adv_images)
            defended_pred = defended_output.argmax(dim=1)
            defense_accuracy = (defended_pred == test_labels).float().mean().item()
        
        # Calculate attack success rate (lower is better for defense)
        attack_success_rate = 1.0 - defense_accuracy
        
        # Print results
        print("\n===== Combined Defense Test Results =====")
        print(f"Accuracy without defense: {no_defense_accuracy:.4f}")
        print(f"Accuracy with defense: {defense_accuracy:.4f}")
        print(f"Attack success rate: {attack_success_rate:.4f}")
        print("========================================\n")
        
        print("Sample predictions:")
        for i in range(len(test_labels)):
            print(f"Sample {i+1}:")
            print(f"  True label: {test_labels[i].item()}")
            print(f"  Vanilla prediction: {vanilla_pred[i].item()}")
            print(f"  Defended prediction: {defended_pred[i].item()}")
        
        # Verify that defense meets required threshold
        threshold = 0.40
        if attack_success_rate <= threshold:
            print(f"\n✅ Defense meets requirement (attack success rate <= {threshold})")
        else:
            print(f"\n❌ Defense does NOT meet requirement (attack success rate > {threshold})")
            
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        # Try a simpler test
        try:
            print("\nRetrying with simpler test setup...")
            classifier.disable_defense = True
            
            # Simple forward pass test
            with torch.no_grad():
                output = pfeddef_model(test_images)
                vanilla_acc = (output.argmax(dim=1) == test_labels).float().mean().item()
                print(f"Basic model accuracy: {vanilla_acc:.4f}")
                
            print("Basic model test successful")
        except Exception as e2:
            print(f"Basic test also failed: {str(e2)}")
    
if __name__ == "__main__":
    main() 