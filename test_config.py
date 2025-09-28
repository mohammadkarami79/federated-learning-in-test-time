#!/usr/bin/env python3
"""
Test Configuration - Verify all required attributes are present
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_config():
    """Test the configuration for missing attributes"""
    print("🔍 Testing Configuration...")
    
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
        
        # Test data loading
        from utils.data_utils import get_dataset
        
        print("🔍 Testing dataset loading...")
        try:
            train_dataset, test_dataset = get_dataset(cfg_obj, train=True)
            print(f"✅ Train dataset loaded: {len(train_dataset)} samples")
            
            train_dataset, test_dataset = get_dataset(cfg_obj, train=False)
            print(f"✅ Test dataset loaded: {len(test_dataset)} samples")
        except Exception as e:
            print(f"❌ Dataset loading failed: {e}")
            return False
        
        # Test required attributes
        required_attrs = [
            'DATASET', 'DATA_ROOT', 'DATA_PATH', 'NUM_CLASSES', 'IMG_SIZE', 'IMG_CHANNELS',
            'NUM_CLIENTS', 'NUM_ROUNDS', 'CLIENT_EPOCHS', 'BATCH_SIZE', 'LEARNING_RATE',
            'EVAL_BATCH_SIZE', 'MODE', 'DATA_DISTRIBUTION'
        ]
        
        print("🔍 Checking required attributes...")
        missing_attrs = []
        for attr in required_attrs:
            if not hasattr(cfg_obj, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            print(f"❌ Missing attributes: {missing_attrs}")
            return False
        else:
            print("✅ All required attributes present")
        
        print("🎉 Configuration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config()
    if success:
        print("\n✅ Ready to run training!")
    else:
        print("\n❌ Configuration needs fixing")
