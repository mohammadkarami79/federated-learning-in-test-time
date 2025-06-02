#!/usr/bin/env python3
"""
Simple comprehensive test for the optimized pFedDef + DiffPure system
Tests all components without running full training
"""

import sys
import torch
import torch.nn as nn
import traceback
import time
from pathlib import Path

def test_imports():
    """Test all critical imports"""
    print("=" * 50)
    print("TESTING IMPORTS")
    print("=" * 50)
    
    try:
        # Core imports
        import torch
        import torchvision
        print(f"✓ PyTorch {torch.__version__}")
        print(f"✓ TorchVision {torchvision.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Configuration
        from config_fixed import get_debug_config, validate_config
        cfg = get_debug_config()
        validate_config(cfg)
        print(f"✓ Optimized configuration loaded")
        
        # MAE Detector (critical fix)
        from defense.mae_detector import MAEDetector
        detector = MAEDetector(cfg)
        print(f"✓ MAE Detector initialized")
        
        # User's implementations
        try:
            import mae_detector1
            print(f"✓ User's MAE detector available")
        except ImportError:
            print("! User's MAE detector not found (optional)")
        
        try:
            import main1
            print(f"✓ User's main1.py available")
        except ImportError:
            print("! User's main1.py not found")
            
        try:
            import server1
            print(f"✓ User's server1.py available")
        except ImportError:
            print("! User's server1.py not found")
        
        # Federated components
        from federated.client import Client
        from federated.server import Server
        print(f"✓ Federated learning components")
        
        # Attacks
        from attacks.pgd import PGDAttack
        print(f"✓ PGD Attack")
        
        # Defense
        from defense.combined_defense import CombinedDefense
        print(f"✓ Combined Defense")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_models():
    """Test model creation and basic operations"""
    print("\n" + "=" * 50)
    print("TESTING MODELS")
    print("=" * 50)
    
    try:
        from config_fixed import get_debug_config
        from models import get_model
        
        cfg = get_debug_config()
        device = torch.device(cfg.DEVICE)
        
        # Test model creation
        model = get_model(cfg)
        model = model.to(device)
        print(f"✓ Model created: {type(model).__name__}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 32, 32).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Forward pass: {dummy_input.shape} -> {output.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading"""
    print("\n" + "=" * 50)
    print("TESTING DATA LOADING")
    print("=" * 50)
    
    try:
        from config_fixed import get_debug_config
        from utils.data_utils import get_dataset
        
        cfg = get_debug_config()
        
        # Test dataset loading
        train_dataset, test_dataset = get_dataset(cfg)
        print(f"✓ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
        
        # Test data loader
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues in tests
        )
        
        # Test one batch
        batch = next(iter(train_loader))
        images, labels = batch
        print(f"✓ Data batch: {images.shape}, {labels.shape}")
        print(f"✓ Value ranges: images [{images.min():.3f}, {images.max():.3f}], labels {labels.unique()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        traceback.print_exc()
        return False

def test_attacks():
    """Test adversarial attacks"""
    print("\n" + "=" * 50)
    print("TESTING ATTACKS")
    print("=" * 50)
    
    try:
        from config_fixed import get_debug_config
        from models import get_model
        from attacks.pgd import PGDAttack
        
        cfg = get_debug_config()
        device = torch.device(cfg.DEVICE)
        
        # Create model and attack
        model = get_model(cfg).to(device)
        attack = PGDAttack(cfg)
        
        # Test attack
        dummy_input = torch.randn(2, 3, 32, 32).to(device)
        dummy_target = torch.randint(0, 10, (2,)).to(device)
        
        model.eval()
        start_time = time.time()
        adv_images = attack.generate(model, dummy_input, dummy_target)
        attack_time = time.time() - start_time
        
        print(f"✓ PGD attack: {dummy_input.shape} -> {adv_images.shape}")
        print(f"✓ Attack time: {attack_time:.3f}s for {cfg.PGD_STEPS} steps")
        
        # Check perturbation
        perturbation = (adv_images - dummy_input).abs().max().item()
        print(f"✓ Max perturbation: {perturbation:.6f} (eps={cfg.PGD_EPS:.6f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Attack test failed: {e}")
        traceback.print_exc()
        return False

def test_mae_detector():
    """Test MAE detector"""
    print("\n" + "=" * 50)
    print("TESTING MAE DETECTOR")
    print("=" * 50)
    
    try:
        from config_fixed import get_debug_config
        from defense.mae_detector import MAEDetector
        
        cfg = get_debug_config()
        device = torch.device(cfg.DEVICE)
        
        # Create detector
        detector = MAEDetector(cfg)
        print(f"✓ MAE Detector created")
        
        # Test detection
        dummy_images = torch.randn(4, 3, 32, 32).to(device)
        
        start_time = time.time()
        predictions = detector.detect(dummy_images)
        detection_time = time.time() - start_time
        
        print(f"✓ Detection: {dummy_images.shape} -> {predictions.shape}")
        print(f"✓ Detection time: {detection_time:.3f}s")
        print(f"✓ Predictions: {predictions}")
        
        return True
        
    except Exception as e:
        print(f"✗ MAE detector test failed: {e}")
        traceback.print_exc()
        return False

def test_memory():
    """Test memory usage"""
    print("\n" + "=" * 50)
    print("TESTING MEMORY USAGE")
    print("=" * 50)
    
    try:
        if not torch.cuda.is_available():
            print("! CUDA not available, skipping memory test")
            return True
            
        # Clear cache
        torch.cuda.empty_cache()
        
        from config_fixed import get_debug_config
        from models import get_model
        
        cfg = get_debug_config()
        device = torch.device('cuda')
        
        # Initial memory
        initial_memory = torch.cuda.memory_allocated() / 1e9
        print(f"Initial GPU memory: {initial_memory:.3f} GB")
        
        # Create model
        model = get_model(cfg).to(device)
        model_memory = torch.cuda.memory_allocated() / 1e9
        print(f"After model: {model_memory:.3f} GB (+{model_memory - initial_memory:.3f} GB)")
        
        # Forward pass
        dummy_input = torch.randn(cfg.BATCH_SIZE, 3, 32, 32).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        forward_memory = torch.cuda.memory_allocated() / 1e9
        print(f"After forward: {forward_memory:.3f} GB (+{forward_memory - model_memory:.3f} GB)")
        
        # Memory limit check
        max_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak memory: {max_memory:.3f} GB")
        
        if max_memory > cfg.MAX_MEMORY_GB:
            print(f"⚠ Memory usage ({max_memory:.1f} GB) exceeds target ({cfg.MAX_MEMORY_GB} GB)")
        else:
            print(f"✓ Memory usage within target ({cfg.MAX_MEMORY_GB} GB)")
        
        return True
        
    except Exception as e:
        print(f"✗ Memory test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("COMPREHENSIVE SYSTEM TEST")
    print("Testing optimized pFedDef + DiffPure system")
    print("=" * 80)
    
    tests = [
        ("Imports", test_imports),
        ("Models", test_models), 
        ("Data Loading", test_data_loading),
        ("Attacks", test_attacks),
        ("MAE Detector", test_mae_detector),
        ("Memory", test_memory)
    ]
    
    results = []
    total_start = time.time()
    
    for test_name, test_func in tests:
        print(f"\n[TEST {len(results) + 1}/{len(tests)}] {test_name}")
        start = time.time()
        success = test_func()
        duration = time.time() - start
        results.append((test_name, success, duration))
        print(f"Duration: {duration:.2f}s")
    
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, duration in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} {test_name:20} ({duration:.2f}s)")
    
    print("-" * 40)
    print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"Total time: {total_time:.2f}s")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for training.")
        return True
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please fix before training.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 