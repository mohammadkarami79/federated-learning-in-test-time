#!/usr/bin/env python3
"""
Quick System Test - Identify exact issues
"""

import sys
import traceback

def test_basic_imports():
    """Test basic imports"""
    print("Testing basic imports...")
    try:
        import torch
        import torchvision
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ TorchVision: {torchvision.__version__}")
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_config():
    """Test config loading"""
    print("\nTesting config...")
    try:
        from config_fixed import get_full_config
        cfg = get_full_config()
        print(f"✅ Config loaded")
        print(f"   DATASET: {cfg.DATASET}")
        print(f"   BATCH_SIZE: {cfg.BATCH_SIZE}")
        print(f"   NUM_CLIENTS: {cfg.NUM_CLIENTS}")
        return cfg
    except Exception as e:
        print(f"❌ Config failed: {e}")
        traceback.print_exc()
        return None

def test_dataset_function():
    """Test the exact dataset function behavior"""
    print("\nTesting dataset function...")
    try:
        from utils.data_utils import get_dataset
        from config_fixed import get_full_config
        
        cfg = get_full_config()
        print(f"Config type: {type(cfg)}")
        print(f"Config has DATASET: {hasattr(cfg, 'DATASET')}")
        print(f"Config DATASET value: {getattr(cfg, 'DATASET', 'NOT_FOUND')}")
        
        # Test the actual call
        result = get_dataset(cfg, cfg.DATA_ROOT)
        print(f"Result type: {type(result)}")
        
        if isinstance(result, tuple):
            train_ds, test_ds = result
            print(f"✅ Got tuple: train={len(train_ds)}, test={len(test_ds)}")
            print(f"Train type: {type(train_ds)}")
            print(f"Test type: {type(test_ds)}")
            
            # Test if we can iterate
            try:
                sample_train = next(iter(train_ds))
                print(f"Train sample: {type(sample_train)}, len={len(sample_train) if hasattr(sample_train, '__len__') else 'no len'}")
            except Exception as e:
                print(f"❌ Cannot iterate train dataset: {e}")
                
            try:
                sample_test = next(iter(test_ds))
                print(f"Test sample: {type(sample_test)}, len={len(sample_test) if hasattr(sample_test, '__len__') else 'no len'}")
            except Exception as e:
                print(f"❌ Cannot iterate test dataset: {e}")
                
            return train_ds, test_ds
        else:
            print(f"❌ Got single object: {type(result)}")
            return None, None
            
    except Exception as e:
        print(f"❌ Dataset function failed: {e}")
        traceback.print_exc()
        return None, None

def test_dataloader():
    """Test DataLoader creation"""
    print("\nTesting DataLoader...")
    try:
        from torch.utils.data import DataLoader
        from config_fixed import get_full_config
        from utils.data_utils import get_dataset
        
        cfg = get_full_config()
        train_ds, test_ds = get_dataset(cfg, cfg.DATA_ROOT)
        
        if train_ds is None or test_ds is None:
            print("❌ Cannot test DataLoader - datasets are None")
            return
            
        # Try creating DataLoader
        test_loader = DataLoader(
            test_ds,
            batch_size=4,  # Small batch size
            shuffle=False,
            num_workers=0,  # No multiprocessing
            pin_memory=False
        )
        
        print(f"✅ DataLoader created: {len(test_loader)} batches")
        
        # Try getting one batch
        first_batch = next(iter(test_loader))
        print(f"✅ First batch: {type(first_batch)}, len={len(first_batch)}")
        
        if len(first_batch) == 2:
            data, target = first_batch
            print(f"✅ Data shape: {data.shape}, Target shape: {target.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ DataLoader failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔍 QUICK SYSTEM TEST")
    print("=" * 50)
    
    # Test 1: Basic imports
    if not test_basic_imports():
        print("\n❌ STOP: Fix basic imports first")
        return
    
    # Test 2: Config
    cfg = test_config()
    if cfg is None:
        print("\n❌ STOP: Fix config loading first")
        return
    
    # Test 3: Dataset function
    train_ds, test_ds = test_dataset_function()
    if train_ds is None or test_ds is None:
        print("\n❌ STOP: Fix dataset loading first")
        return
    
    # Test 4: DataLoader
    if not test_dataloader():
        print("\n❌ STOP: Fix DataLoader creation first")
        return
    
    print("\n" + "=" * 50)
    print("🎉 ALL BASIC TESTS PASSED!")
    print("The system should work now. Try running:")
    print("python comprehensive_diagnostic.py")

if __name__ == "__main__":
    main()
