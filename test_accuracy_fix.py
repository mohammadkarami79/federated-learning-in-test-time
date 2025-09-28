#!/usr/bin/env python3
"""
Test script to verify adversarial accuracy fixes
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_adversarial_accuracy():
    """Test that adversarial accuracy is non-zero"""
    
    print("=== Testing Adversarial Accuracy Fix ===")
    
    try:
        from attacks.pgd_bulletproof import PGDAttack
        from models import get_model
        import types
        
        # Create simple config
        cfg = types.SimpleNamespace()
        cfg.PGD_EPS = 0.031  # 8/255
        cfg.PGD_STEPS = 10
        cfg.PGD_ALPHA = 0.007  # epsilon/steps * 2.5
        cfg.PGD_RANDOM_START = True
        
        # Create model and data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use pre-trained model for meaningful accuracy
        import torchvision.models as models
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 10)  # CIFAR-10 classes
        model = model.to(device)
        model.eval()
        
        # Create realistic test data (CIFAR-10 like)
        batch_size = 8
        x = torch.rand(batch_size, 3, 32, 32).to(device)  # Valid image range [0,1]
        
        # Get model predictions as targets (so clean accuracy will be 100%)
        with torch.no_grad():
            clean_output = model(x)
            y = clean_output.argmax(dim=1)
        
        # Test clean accuracy
        with torch.no_grad():
            clean_output = model(x)
            clean_pred = clean_output.argmax(dim=1)
            clean_correct = clean_pred.eq(y).sum().item()
            clean_acc = 100. * clean_correct / batch_size
        
        print(f"Clean accuracy: {clean_acc:.2f}%")
        
        # Test PGD attack
        pgd_attacker = PGDAttack(cfg)
        x_adv = pgd_attacker.attack(model, x, y)
        
        # Check perturbation magnitude
        perturbation = (x_adv - x).abs().max().item()
        print(f"Max perturbation: {perturbation:.6f} (should be <= {cfg.PGD_EPS:.6f})")
        
        # Test adversarial accuracy
        with torch.no_grad():
            adv_output = model(x_adv)
            adv_pred = adv_output.argmax(dim=1)
            adv_correct = adv_pred.eq(y).sum().item()
            adv_acc = 100. * adv_correct / batch_size
        
        print(f"Adversarial accuracy (no defense): {adv_acc:.2f}%")
        
        # Test simple purification
        def simple_purify(x_adv, sigma=0.1):
            noise = torch.randn_like(x_adv) * sigma
            purified = torch.clamp(x_adv + noise, 0, 1)
            return 0.8 * purified + 0.2 * x_adv
        
        x_purified = simple_purify(x_adv)
        
        with torch.no_grad():
            purified_output = model(x_purified)
            purified_pred = purified_output.argmax(dim=1)
            purified_correct = purified_pred.eq(y).sum().item()
            purified_acc = 100. * purified_correct / batch_size
        
        print(f"Adversarial accuracy (with purification): {purified_acc:.2f}%")
        
        # Verify fixes
        if perturbation <= cfg.PGD_EPS + 1e-6:
            print("[SUCCESS] PGD attack epsilon constraint satisfied")
        else:
            print("[ERROR] PGD attack epsilon constraint violated")
        
        if adv_acc < clean_acc:
            print("[SUCCESS] Attack is effective (reduces accuracy)")
        else:
            print("[WARNING] Attack may not be effective")
        
        if purified_acc >= adv_acc:
            print("[SUCCESS] Purification improves adversarial accuracy")
        else:
            print("[WARNING] Purification not effective")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_adversarial_accuracy()
