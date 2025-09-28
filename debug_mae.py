#!/usr/bin/env python3
"""
Debug MAE training issue
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("1. Testing imports...")
    from scripts.train_mae_detector import get_config_for_dataset
    from defense.mae_detector1 import MAEDetector
    from utils.data_utils import get_dataset
    import torch.utils.data as data_utils
    print("✅ All imports successful")
    
    print("\n2. Testing config...")
    cfg = get_config_for_dataset('br35h')
    print(f"✅ Config: {cfg.DATASET}, IMG_SIZE: {cfg.IMG_SIZE}")
    print(f"   LR: {getattr(cfg, 'LR', 'MISSING')}")
    print(f"   MAE_THRESHOLD: {getattr(cfg, 'MAE_THRESHOLD', 'MISSING')}")
    
    print("\n3. Testing dataset loading...")
    result = get_dataset(cfg.DATASET, getattr(cfg, 'DATA_ROOT', 'data'))
    if isinstance(result, tuple):
        train_dataset, _ = result
    else:
        train_dataset = result
    print(f"✅ Dataset loaded: {len(train_dataset)} samples")
    
    print("\n4. Testing data loader...")
    train_loader = data_utils.DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=True
    )
    print(f"✅ DataLoader created: {len(train_loader)} batches")
    
    print("\n5. Testing MAE detector creation...")
    detector = MAEDetector(cfg)
    print("✅ MAE detector created")
    
    print("\n6. Testing batch iteration...")
    sample_batch = next(iter(train_loader))
    print(f"✅ Sample batch: {type(sample_batch)}, len: {len(sample_batch) if hasattr(sample_batch, '__len__') else 'no len'}")
    if isinstance(sample_batch, (list, tuple)):
        print(f"   Batch[0] shape: {sample_batch[0].shape}")
        print(f"   Batch[1] shape: {sample_batch[1].shape if len(sample_batch) > 1 else 'no batch[1]'}")
    
    print("\n7. Testing MAE train method...")
    # Test with very small training
    detector.train(train_loader, epochs=1)
    print("✅ MAE training completed successfully!")
    
except Exception as e:
    import traceback
    print(f"\n❌ Error at step: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
