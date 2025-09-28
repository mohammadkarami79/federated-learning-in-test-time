#!/usr/bin/env python3
"""
Simple Configuration Verification
"""

def verify_config():
    """Verify configuration matches log7.txt"""
    print("🔍 VERIFYING CONFIGURATION")
    print("=" * 40)
    
    try:
        # Test 1: Import configuration
        print("1. Testing configuration import...")
        from config_selective_defense import get_config
        cfg = get_config()
        print(f"   ✅ Config loaded: {len(cfg)} parameters")
        
        # Test 2: Check critical parameters
        print("\n2. Checking critical parameters...")
        critical_params = {
            'DATASET': 'cifar10',
            'NUM_CLASSES': 10,
            'NUM_CLIENTS': 10,
            'NUM_ROUNDS': 15,
            'CLIENT_EPOCHS': 8,
            'BATCH_SIZE': 64,
            'LEARNING_RATE': 0.01,
            'MAE_THRESHOLD': 0.15,
            'DIFFUSER_STEPS': 4,
            'DIFFUSER_SIGMA': 0.3,
            'ATTACK_EPSILON': 0.031,
            'ATTACK_STEPS': 10,
        }
        
        all_good = True
        for param, expected in critical_params.items():
            actual = cfg.get(param)
            if actual == expected:
                print(f"   ✅ {param}: {actual}")
            else:
                print(f"   ❌ {param}: {actual} (expected: {expected})")
                all_good = False
        
        # Test 3: Check required attributes
        print("\n3. Checking required attributes...")
        required_attrs = ['DATA_ROOT', 'DATA_PATH', 'MODE', 'EVAL_BATCH_SIZE']
        for attr in required_attrs:
            if attr in cfg:
                print(f"   ✅ {attr}: {cfg[attr]}")
            else:
                print(f"   ❌ Missing: {attr}")
                all_good = False
        
        # Test 4: Test data loading
        print("\n4. Testing data loading...")
        try:
            import types
            cfg_obj = types.SimpleNamespace(**cfg)
            from utils.data_utils import get_dataset
            train_dataset, test_dataset = get_dataset(cfg_obj, train=True)
            print(f"   ✅ Train dataset: {len(train_dataset)} samples")
            train_dataset, test_dataset = get_dataset(cfg_obj, train=False)
            print(f"   ✅ Test dataset: {len(test_dataset)} samples")
        except Exception as e:
            print(f"   ❌ Data loading failed: {e}")
            all_good = False
        
        # Summary
        print("\n" + "=" * 40)
        if all_good:
            print("🎉 ALL CHECKS PASSED!")
            print("✅ Configuration matches log7.txt")
            print("✅ Ready to reproduce 87.7% clean, 72.1% adversarial accuracy")
            print("\n🚀 Run this on your server:")
            print("   nohup python run_selective_defense.py > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &")
        else:
            print("❌ SOME CHECKS FAILED!")
            print("⚠️  Fix the issues above before running")
        
        return all_good
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    verify_config()
