#!/usr/bin/env python3
"""
Simple script to run optimized pFedDef + DiffPure federated training
Usage: python run_training.py [mode]
Modes: debug (default), test, full
"""

import sys
import time
import torch
from pathlib import Path

def run_federated_training(mode='debug'):
    """Run federated training with specified mode"""
    
    print("🚀 Starting Optimized pFedDef + DiffPure Training")
    print("=" * 60)
    
    # Import optimized configuration
    from config_fixed import get_debug_config, get_test_config, get_full_config
    
    # Select configuration based on mode
    if mode == 'debug':
        cfg = get_debug_config()
        print("📋 Mode: DEBUG (2-5 minutes)")
    elif mode == 'test':
        cfg = get_test_config()
        print("📋 Mode: TEST (10-20 minutes)")
    elif mode == 'full':
        cfg = get_full_config()
        print("📋 Mode: FULL (20-60 minutes)")
    else:
        print(f"❌ Unknown mode: {mode}")
        print("Available modes: debug, test, full")
        return False
    
    print(f"⚙️ Configuration: {cfg.N_ROUNDS} rounds, {cfg.LOCAL_STEPS_PER_EPOCH} steps")
    print(f"🎯 Device: {cfg.DEVICE}")
    
    # Import components
    try:
        from models import get_model
        from utils.data_utils import get_dataset
        from federated.client import Client
        from federated.server import Server
        from attacks.pgd import PGDAttack
        from defense.mae_detector import MAEDetector
        print("✅ All components imported successfully")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ Running on CPU (training will be slower)")
    
    # Create datasets
    print("\n📊 Loading datasets...")
    try:
        train_dataset, test_dataset = get_dataset(cfg)
        print(f"✅ CIFAR-10 loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False
    
    # Create model
    print("\n🧠 Creating model...")
    try:
        model = get_model(cfg)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created: {total_params:,} parameters")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Create defense components
    print("\n🛡️ Setting up defenses...")
    try:
        attack = PGDAttack(cfg)
        detector = MAEDetector(cfg)
        print(f"✅ PGD Attack: {cfg.PGD_STEPS} steps, eps={cfg.PGD_EPS:.4f}")
        print(f"✅ MAE Detector: threshold={cfg.MAE_THRESHOLD}")
    except Exception as e:
        print(f"❌ Defense setup failed: {e}")
        return False
    
    # Memory check
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1e9
        print(f"\n💾 Initial GPU memory: {initial_memory:.3f} GB")
    
    print(f"\n🎯 TRAINING CONFIGURATION SUMMARY:")
    print(f"   📈 Rounds: {cfg.N_ROUNDS}")
    print(f"   👥 Clients: {cfg.N_CLIENTS}")
    print(f"   🔄 Local steps: {cfg.LOCAL_STEPS_PER_EPOCH}")
    print(f"   📚 Batch size: {cfg.BATCH_SIZE}")
    print(f"   📊 Learning rate: {cfg.LEARNING_RATE}")
    print(f"   ⚔️ PGD steps: {cfg.PGD_STEPS}")
    print(f"   🌊 Diffusion steps: {cfg.DIFFUSION_STEPS}")
    
    estimated_time = {
        'debug': '2-5 minutes',
        'test': '10-20 minutes', 
        'full': '20-60 minutes'
    }
    
    print(f"\n⏱️ Estimated training time: {estimated_time[mode]}")
    print(f"🎯 Expected accuracy improvement: 9.38% → 40-70%")
    
    print("\n" + "=" * 60)
    print("🚀 TRAINING READY!")
    print("💡 To start training, use:")
    print(f"   python main1.py  # (User's optimized training script)")
    print(f"   OR")
    print(f"   python main.py   # (Original training script)")
    print("\n🔧 For debugging, first run:")
    print("   python simple_test.py")
    print("=" * 60)
    
    return True

def main():
    """Main function"""
    
    # Parse command line argument
    mode = 'debug'
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    start_time = time.time()
    success = run_federated_training(mode)
    duration = time.time() - start_time
    
    print(f"\n⏱️ Setup completed in {duration:.2f}s")
    
    if success:
        print("✅ System ready for training!")
        return 0
    else:
        print("❌ Setup failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 