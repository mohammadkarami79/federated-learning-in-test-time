#!/usr/bin/env python3
"""
Quick Fix for Configuration Issues
Run this on your server to fix the missing attributes
"""

def fix_config():
    """Fix the configuration file"""
    config_file = "config_selective_defense.py"
    
    print("🔧 Fixing configuration file...")
    
    # Read current config
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Add missing attributes if not present
    if 'DATA_ROOT' not in content:
        content = content.replace(
            "'DATASET': 'cifar10',",
            "'DATASET': 'cifar10',\n        'DATA_ROOT': 'data',  # Add missing DATA_ROOT\n        'DATA_PATH': 'data',  # Add DATA_PATH for compatibility"
        )
        print("✅ Added DATA_ROOT and DATA_PATH")
    
    if 'MODE' not in content:
        # Find the end of the config dict and add missing attributes
        if 'TARGET_DETECTION_RATE' in content:
            content = content.replace(
                "'TARGET_DETECTION_RATE': 25.0,  # Target detection rate",
                "'TARGET_DETECTION_RATE': 25.0,  # Target detection rate\n        \n        # Additional attributes for compatibility\n        'MODE': 'full',\n        'DATA_DISTRIBUTION': 'iid',\n        'DIFFUSION_HIDDEN_CHANNELS': 128,\n        'MAE_EPOCHS': 10,\n        'DIFFUSION_EPOCHS': 50,\n        'DIFFPURE_STEPS': 50,\n        'DIFFPURE_SIGMA': 0.1,"
            )
            print("✅ Added additional compatibility attributes")
    
    # Write fixed config
    with open(config_file, 'w') as f:
        f.write(content)
    
    print("✅ Configuration file fixed!")
    print("\nNow you can run:")
    print("nohup python run_selective_defense.py > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &")

if __name__ == "__main__":
    fix_config()
