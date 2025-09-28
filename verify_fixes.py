#!/usr/bin/env python3
"""
Verification script to ensure all critical fixes are applied correctly.
Run this script to verify that the publication-ready changes are in place.
"""

import sys
from pathlib import Path

def check_config_fixes():
    """Check if config_fixed.py has the correct parameters"""
    print("🔍 Checking config_fixed.py...")
    
    config_file = Path("config_fixed.py")
    if not config_file.exists():
        print("❌ config_fixed.py not found!")
        return False
    
    content = config_file.read_text()
    
    checks = [
        ("MAE_THRESHOLD = 0.001", "MAE threshold set to ultra-low value"),
        ("BATCH_SIZE = 64", "Batch size increased for stability"),
        ("LEARNING_RATE = 0.001", "Learning rate reduced to prevent overfitting"),
        ("WEIGHT_DECAY = 5e-3", "Weight decay increased for regularization"),
        ("NUM_EPOCHS = 8", "Epochs reduced to prevent overfitting"),
        ("CLIENT_EPOCHS = 2", "Client epochs reduced to prevent overfitting"),
        ("NUM_ROUNDS = 8", "Rounds reduced to prevent overfitting"),
        ("DROPOUT_RATE = 0.5", "Dropout increased for stronger regularization"),
        ("MAX_GRAD_NORM = 0.5", "Gradient clipping strengthened"),
        ("LABEL_SMOOTHING_FACTOR = 0.2", "Label smoothing strengthened"),
        ("MIXUP_ALPHA = 0.4", "Mixup strengthened"),
    ]
    
    all_good = True
    for check, description in checks:
        if check in content:
            print(f"  ✅ {description}")
        else:
            print(f"  ❌ MISSING: {description}")
            all_good = False
    
    return all_good

def check_mae_detector_fixes():
    """Check if MAE detector has the improved threshold calibration"""
    print("\n🔍 Checking defense/mae_detector1.py...")
    
    mae_file = Path("defense/mae_detector1.py")
    if not mae_file.exists():
        print("❌ defense/mae_detector1.py not found!")
        return False
    
    content = mae_file.read_text()
    
    checks = [
        ("torch.quantile(errs, 0.95)", "95th percentile threshold calibration"),
        ("mean_err = errs.mean().item()", "Enhanced threshold calculation"),
        ("std_err = errs.std().item()", "Standard deviation tracking"),
    ]
    
    all_good = True
    for check, description in checks:
        if check in content:
            print(f"  ✅ {description}")
        else:
            print(f"  ❌ MISSING: {description}")
            all_good = False
    
    return all_good

def check_main_fixes():
    """Check if main.py has enhanced MAE debugging"""
    print("\n🔍 Checking main.py...")
    
    main_file = Path("main.py")
    if not main_file.exists():
        print("❌ main.py not found!")
        return False
    
    content = main_file.read_text()
    
    checks = [
        ("recon_errors = mae_detector.model.reconstruction_error", "Reconstruction error analysis"),
        ("Reconstruction errors: min=", "Enhanced debugging output"),
        ("dynamic_threshold = recon_errors.mean().item()", "Dynamic threshold fallback"),
        ("Threshold too high! Using dynamic:", "Threshold validation"),
    ]
    
    all_good = True
    for check, description in checks:
        if check in content:
            print(f"  ✅ {description}")
        else:
            print(f"  ❌ MISSING: {description}")
            all_good = False
    
    return all_good

def main():
    """Run all verification checks"""
    print("🚨 PUBLICATION-READY FIXES VERIFICATION")
    print("=" * 50)
    
    config_ok = check_config_fixes()
    mae_ok = check_mae_detector_fixes()
    main_ok = check_main_fixes()
    
    print("\n📊 VERIFICATION SUMMARY")
    print("=" * 30)
    
    if config_ok and mae_ok and main_ok:
        print("✅ ALL FIXES VERIFIED SUCCESSFULLY!")
        print("🎯 Ready for publication-quality training!")
        print("\n🚀 You can now run:")
        print("   python main.py --dataset br35h --mode full --skip-setup")
        return 0
    else:
        print("❌ SOME FIXES ARE MISSING!")
        print("⚠️ Please follow the CRITICAL_SERVER_UPDATE_GUIDE.md")
        print("   to apply all necessary changes.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
