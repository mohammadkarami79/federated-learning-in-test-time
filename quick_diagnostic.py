#!/usr/bin/env python3
"""
Quick diagnostic to identify why adversarial accuracy is still 0.0%
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def diagnose_adversarial_accuracy():
    """Diagnose the adversarial accuracy calculation issue"""
    
    print("=== Adversarial Accuracy Diagnostic ===")
    
    try:
        # Import required modules
        from config_selective_defense import get_selective_defense_config
        from utils.data_utils import get_dataset
        from torch.utils.data import DataLoader
        from attacks.pgd import PGDAttack
        from models import get_model
        import types
        
        # Get config
        cfg_dict = get_selective_defense_config()
        cfg = types.SimpleNamespace()
        for key, value in cfg_dict.items():
            setattr(cfg, key, value)
        
        # Map attack parameters
        cfg.PGD_EPS = cfg_dict['ATTACK_EPSILON']
        cfg.PGD_STEPS = cfg_dict['ATTACK_STEPS'] 
        cfg.PGD_ALPHA = cfg_dict['ATTACK_ALPHA']
        cfg.PGD_RANDOM_START = cfg_dict['ATTACK_RANDOM_START']
        cfg.DATA_ROOT = './data'
        
        print(f"[CONFIG] Attack epsilon: {cfg.PGD_EPS}")
        print(f"[CONFIG] Attack steps: {cfg.PGD_STEPS}")
        print(f"[CONFIG] MAE threshold: {cfg.MAE_THRESHOLD}")
        
        # Get test data
        result = get_dataset(cfg.DATASET, cfg.DATA_ROOT)
        if isinstance(result, tuple):
            train_dataset, test_dataset = result
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        else:
            test_loader = result
        
        # Get a small batch for testing
        data_iter = iter(test_loader)
        data, target = next(data_iter)
        data, target = data[:8], target[:8]  # Small batch
        
        print(f"[DATA] Test batch shape: {data.shape}")
        print(f"[DATA] Target shape: {target.shape}")
        
        # Create a simple model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_model(cfg.MODEL_NAME).to(device)
        data, target = data.to(device), target.to(device)
        
        # Test clean prediction
        with torch.no_grad():
            clean_output = model(data)
            clean_pred = clean_output.argmax(dim=1)
            clean_correct = clean_pred.eq(target).sum().item()
            clean_acc = 100. * clean_correct / len(target)
        
        print(f"[CLEAN] Clean accuracy on test batch: {clean_acc:.2f}%")
        
        # Test adversarial attack
        pgd_attacker = PGDAttack(cfg)
        adv_data = pgd_attacker.attack(model, data, target)
        
        print(f"[ATTACK] Adversarial data shape: {adv_data.shape}")
        print(f"[ATTACK] Perturbation magnitude: {(adv_data - data).abs().max().item():.6f}")
        
        # Test adversarial prediction WITHOUT purification
        with torch.no_grad():
            adv_output = model(adv_data)
            adv_pred = adv_output.argmax(dim=1)
            adv_correct = adv_pred.eq(target).sum().item()
            adv_acc = 100. * adv_correct / len(target)
        
        print(f"[ADV] Adversarial accuracy (no defense): {adv_acc:.2f}%")
        
        # Test MAE detection
        try:
            from defense.mae_detector import MAEDetector
            mae_detector = MAEDetector(cfg)
            
            # Test reconstruction error calculation
            recon_errors = mae_detector.get_reconstruction_error(adv_data)
            threshold = cfg.MAE_THRESHOLD
            detected_mask = recon_errors > threshold
            
            print(f"[MAE] Reconstruction errors: {recon_errors}")
            print(f"[MAE] Threshold: {threshold}")
            print(f"[MAE] Detected: {detected_mask.sum().item()}/{len(detected_mask)}")
            
        except Exception as e:
            print(f"[MAE] MAE detection failed: {e}")
            detected_mask = torch.zeros(len(adv_data), dtype=torch.bool, device=device)
        
        # Test selective purification
        try:
            from diffusion.diffuser import DiffusionPurifier
            diffuser = DiffusionPurifier()
            
            purified_data = adv_data.clone()
            if detected_mask.sum() > 0:
                detected_samples = adv_data[detected_mask]
                purified_samples = diffuser.purify(detected_samples, steps=cfg.DIFFUSER_STEPS, sigma=cfg.DIFFUSER_SIGMA)
                purified_data[detected_mask] = purified_samples
                print(f"[PURIFY] Applied DiffPure to {detected_mask.sum().item()} samples")
            else:
                print(f"[PURIFY] No samples detected for purification")
            
            # Test final adversarial accuracy with selective defense
            with torch.no_grad():
                final_output = model(purified_data)
                final_pred = final_output.argmax(dim=1)
                final_correct = final_pred.eq(target).sum().item()
                final_acc = 100. * final_correct / len(target)
            
            print(f"[FINAL] Adversarial accuracy (with selective defense): {final_acc:.2f}%")
            
        except Exception as e:
            print(f"[PURIFY] DiffPure failed: {e}")
            final_acc = adv_acc
        
        # Summary
        print("\n=== DIAGNOSTIC SUMMARY ===")
        print(f"Clean Accuracy: {clean_acc:.2f}%")
        print(f"Adversarial Accuracy (no defense): {adv_acc:.2f}%")
        print(f"Adversarial Accuracy (selective defense): {final_acc:.2f}%")
        print(f"Detection Rate: {100. * detected_mask.sum().item() / len(detected_mask):.2f}%")
        
        if final_acc > adv_acc:
            print("[SUCCESS] Selective defense is working!")
        else:
            print("[ISSUE] Selective defense not improving adversarial accuracy")
            
        if final_acc == 0.0:
            print("[CRITICAL] Zero adversarial accuracy indicates a calculation bug")
            
    except Exception as e:
        print(f"[ERROR] Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_adversarial_accuracy()
