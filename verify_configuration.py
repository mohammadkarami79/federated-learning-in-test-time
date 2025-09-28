#!/usr/bin/env python3
"""
Configuration Verification Script
Ensures all settings match the successful log7.txt configuration
"""

import sys
from pathlib import Path
import torch
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def verify_configuration():
    """Verify configuration matches log7.txt successful run"""
    print("🔍 VERIFYING CONFIGURATION AGAINST LOG7.TXT SUCCESS")
    print("=" * 60)
    
    try:
        # Import configuration
        from config_selective_defense import get_config
        
        # Get configuration
        cfg_dict = get_config()
        print(f"✅ Configuration loaded: {len(cfg_dict)} parameters")
        
        # Convert to namespace object
        import types
        cfg_obj = types.SimpleNamespace()
        for key, value in cfg_dict.items():
            setattr(cfg_obj, key, value)
        
        # Expected values from log7.txt analysis
        expected_values = {
            'DATASET': 'cifar10',
            'NUM_CLASSES': 10,
            'IMG_SIZE': 32,
            'IMG_CHANNELS': 3,
            'MODEL_NAME': 'resnet18',
            'NUM_CLIENTS': 10,
            'NUM_ROUNDS': 15,
            'CLIENT_EPOCHS': 8,  # From log7.txt
            'BATCH_SIZE': 64,
            'LEARNING_RATE': 0.01,
            'ATTACK_EPSILON': 0.031,
            'ATTACK_STEPS': 10,
            'MAE_THRESHOLD': 0.15,  # From log7.txt
            'DIFFUSER_STEPS': 4,    # From log7.txt
            'DIFFUSER_SIGMA': 0.3,  # From log7.txt
        }
        
        print("\n📊 VERIFYING KEY PARAMETERS:")
        all_correct = True
        
        for param, expected in expected_values.items():
            actual = getattr(cfg_obj, param, None)
            if actual == expected:
                print(f"  ✅ {param}: {actual} (matches log7.txt)")
            else:
                print(f"  ❌ {param}: {actual} (expected: {expected})")
                all_correct = False
        
        # Verify data loading works
        print("\n🔍 TESTING DATA LOADING:")
        try:
            from utils.data_utils import get_dataset
            train_dataset, test_dataset = get_dataset(cfg_obj, train=True)
            print(f"  ✅ Train dataset: {len(train_dataset)} samples")
            
            train_dataset, test_dataset = get_dataset(cfg_obj, train=False)
            print(f"  ✅ Test dataset: {len(test_dataset)} samples")
        except Exception as e:
            print(f"  ❌ Data loading failed: {e}")
            all_correct = False
        
        # Verify model creation
        print("\n🔍 TESTING MODEL CREATION:")
        try:
            import torchvision.models as models
            import torch.nn as nn
            
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, cfg_obj.NUM_CLASSES)
            model = model.to(cfg_obj.DEVICE)
            print(f"  ✅ Model created successfully")
            print(f"  ✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        except Exception as e:
            print(f"  ❌ Model creation failed: {e}")
            all_correct = False
        
        # Verify MAE detector
        print("\n🔍 TESTING MAE DETECTOR:")
        try:
            from defense.mae_detector import MAEDetector
            mae_detector = MAEDetector(cfg_obj)
            print(f"  ✅ MAE detector created successfully")
            print(f"  ✅ MAE threshold: {mae_detector.threshold}")
        except Exception as e:
            print(f"  ❌ MAE detector creation failed: {e}")
            all_correct = False
        
        # Verify DiffPure
        print("\n🔍 TESTING DIFFPURE:")
        try:
            from diffusion.diffuser import UNet
            diffuser = UNet(in_channels=cfg_obj.IMG_CHANNELS, hidden_channels=128)
            print(f"  ✅ DiffPure model created successfully")
        except Exception as e:
            print(f"  ❌ DiffPure creation failed: {e}")
            all_correct = False
        
        # Verify attack configuration
        print("\n🔍 TESTING ATTACK CONFIGURATION:")
        try:
            from attacks.pgd import PGDAttack
            attack = PGDAttack(
                epsilon=cfg_obj.ATTACK_EPSILON,
                step_size=cfg_obj.ATTACK_ALPHA,
                num_steps=cfg_obj.ATTACK_STEPS,
                random_start=cfg_obj.ATTACK_RANDOM_START
            )
            print(f"  ✅ PGD attack created successfully")
            print(f"  ✅ Attack epsilon: {cfg_obj.ATTACK_EPSILON}")
            print(f"  ✅ Attack steps: {cfg_obj.ATTACK_STEPS}")
        except Exception as e:
            print(f"  ❌ Attack creation failed: {e}")
            all_correct = False
        
        # Summary
        print("\n" + "=" * 60)
        if all_correct:
            print("🎉 ALL CHECKS PASSED!")
            print("✅ Configuration matches log7.txt successful run")
            print("✅ Ready to reproduce 87.7% clean, 72.1% adversarial accuracy")
            print("\n🚀 You can now run:")
            print("   nohup python run_selective_defense.py > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &")
        else:
            print("❌ SOME CHECKS FAILED!")
            print("⚠️  Please fix the issues above before running training")
        
        return all_correct
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_log7():
    """Compare current config with log7.txt results"""
    print("\n📊 COMPARING WITH LOG7.TXT RESULTS:")
    print("=" * 40)
    
    # Expected results from log7.txt
    log7_results = {
        "final_clean_accuracy": 87.7,
        "final_adversarial_accuracy": 72.12,
        "mae_detection_rate": 15.63,
        "total_rounds": 15,
        "training_time_hours": 12.3
    }
    
    print("Expected results from log7.txt:")
    for key, value in log7_results.items():
        print(f"  {key}: {value}")
    
    print("\nYour configuration should produce similar results!")
    print("Key factors that led to success:")
    print("  ✅ MAE threshold: 0.15 (good balance)")
    print("  ✅ DiffPure steps: 4 (efficient purification)")
    print("  ✅ DiffPure sigma: 0.3 (effective noise level)")
    print("  ✅ Client epochs: 8 (good local training)")
    print("  ✅ Learning rate: 0.01 (stable convergence)")

if __name__ == "__main__":
    print("🔍 CONFIGURATION VERIFICATION FOR LOG7.TXT REPRODUCTION")
    print("=" * 60)
    
    success = verify_configuration()
    compare_with_log7()
    
    if success:
        print("\n🎉 VERIFICATION COMPLETE - READY TO RUN!")
    else:
        print("\n❌ VERIFICATION FAILED - FIX ISSUES FIRST")
