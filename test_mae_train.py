#!/usr/bin/env python3
"""
Test MAE training script
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from scripts.train_mae_detector import get_config_for_dataset, train_mae_detector
    print("✅ Imports successful")
    
    cfg = get_config_for_dataset('br35h')
    print("✅ Config loaded")
    print(f"Dataset: {cfg.DATASET}, IMG_SIZE: {cfg.IMG_SIZE}")
    
    # Try training with 1 epoch
    success = train_mae_detector(cfg, epochs=1)
    if success:
        print("✅ MAE training successful")
    else:
        print("❌ MAE training failed")
        
except Exception as e:
    import traceback
    print(f"❌ Error: {e}")
    print("Full traceback:")
    traceback.print_exc()
