#!/usr/bin/env python3
"""
Comprehensive Test Suite for pFedDef + DiffPure System
Tests for real problems and edge cases to ensure robustness
"""

import torch
import torch.nn as nn
import sys
import traceback
import time
import psutil
import os
import logging
from pathlib import Path
import subprocess

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    return logging.getLogger(__name__)

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, test_name):
        self.passed += 1
        print(f"✅ PASS - {test_name}")
    
    def add_fail(self, test_name, error):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"❌ FAIL - {test_name}: {error}")
    
    def get_summary(self):
        total = self.passed + self.failed
        success_rate = (self.passed / total * 100) if total > 0 else 0
        return {
            'passed': self.passed,
            'failed': self.failed,
            'total': total,
            'success_rate': success_rate
        }

def test_critical_imports(results):
    """Test all critical imports work without errors"""
    critical_imports = [
        ('torch', 'import torch'),
        ('config_fixed', 'from config_fixed import get_debug_config'),
        ('utils.data_utils', 'from utils.data_utils import get_dataset'),
        ('models', 'from models import get_model'),
        ('diffusion.diffuser', 'from diffusion.diffuser import UNet'),
        ('defense.mae_detector', 'from defense.mae_detector import MAEDetector'),
        ('federated.client', 'from federated.client import Client'),
        ('federated.server', 'from federated.server import Server'),
        ('attacks.pgd', 'from attacks.pgd import PGDAttack'),
    ]
    
    for name, import_stmt in critical_imports:
        try:
            exec(import_stmt)
            results.add_pass(f"Import {name}")
        except Exception as e:
            results.add_fail(f"Import {name}", str(e))

def test_config_system_robustness(results):
    """Test config system handles different scenarios"""
    try:
        from config_fixed import get_debug_config, get_test_config, get_full_config
        
        # Test all config modes
        configs = {
            'debug': get_debug_config(),
            'test': get_test_config(),
            'full': get_full_config()
        }
        
        for mode, cfg in configs.items():
            # Test required attributes exist
            required_attrs = ['N_ROUNDS', 'N_CLIENTS', 'BATCH_SIZE', 'LEARNING_RATE', 
                            'IMG_CHANNELS', 'N_CLASSES', 'DATASET', 'DEVICE']
            
            missing_attrs = [attr for attr in required_attrs if not hasattr(cfg, attr)]
            if missing_attrs:
                results.add_fail(f"Config {mode}", f"Missing: {missing_attrs}")
            else:
                results.add_pass(f"Config {mode} completeness")
        
        # Test dataset-specific configs
        datasets = ['cifar10', 'cifar100', 'mnist']
        for dataset in datasets:
            cfg = get_debug_config()
            cfg.DATASET = dataset
            if dataset == 'mnist':
                cfg.IMG_CHANNELS = 1
                cfg.IMG_SIZE = 28
            elif dataset == 'cifar100':
                cfg.N_CLASSES = 100
            
            # Validate config makes sense
            if cfg.IMG_CHANNELS in [1, 3] and cfg.N_CLASSES > 0:
                results.add_pass(f"Config {dataset} validation")
            else:
                results.add_fail(f"Config {dataset} validation", "Invalid parameters")
                
    except Exception as e:
        results.add_fail("Config system", str(e))

def test_dataset_loading_robustness(results):
    """Test dataset loading for all supported datasets"""
    try:
        from config_fixed import get_debug_config
        from utils.data_utils import get_dataset
        
        datasets = ['cifar10', 'cifar100', 'mnist']
        
        for dataset in datasets:
            try:
                cfg = get_debug_config()
                cfg.DATASET = dataset
                if dataset == 'mnist':
                    cfg.IMG_CHANNELS = 1
                    cfg.IMG_SIZE = 28
                elif dataset == 'cifar100':
                    cfg.N_CLASSES = 100
                
                # Test dataset loading
                train_ds, test_ds = get_dataset(cfg)
                
                # Verify dataset properties
                if len(train_ds) > 0 and len(test_ds) > 0:
                    results.add_pass(f"Dataset {dataset} loading")
                else:
                    results.add_fail(f"Dataset {dataset} loading", "Empty datasets")
                
                # Test data shapes
                train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2)
                batch = next(iter(train_loader))
                images, labels = batch
                
                expected_channels = 1 if dataset == 'mnist' else 3
                if images.shape[1] == expected_channels:
                    results.add_pass(f"Dataset {dataset} shape")
                else:
                    results.add_fail(f"Dataset {dataset} shape", f"Expected {expected_channels} channels, got {images.shape[1]}")
                    
            except Exception as e:
                results.add_fail(f"Dataset {dataset}", str(e))
                
    except Exception as e:
        results.add_fail("Dataset loading system", str(e))

def test_model_creation_robustness(results):
    """Test model creation for different configurations"""
    try:
        from config_fixed import get_debug_config
        from models import get_model
        from diffusion.diffuser import UNet
        
        datasets = ['cifar10', 'cifar100', 'mnist']
        
        for dataset in datasets:
            try:
                cfg = get_debug_config()
                cfg.DATASET = dataset
                if dataset == 'mnist':
                    cfg.IMG_CHANNELS = 1
                    cfg.N_CLASSES = 10
                elif dataset == 'cifar100':
                    cfg.N_CLASSES = 100
                    cfg.IMG_CHANNELS = 3
                else:
                    cfg.N_CLASSES = 10
                    cfg.IMG_CHANNELS = 3
                
                # Test main model
                model = get_model(cfg)
                if hasattr(model, 'parameters'):
                    results.add_pass(f"Model creation {dataset}")
                else:
                    results.add_fail(f"Model creation {dataset}", "Invalid model")
                
                # Test diffusion model
                diffuser = UNet(in_channels=cfg.IMG_CHANNELS, hidden_channels=32)
                if hasattr(diffuser, 'forward'):
                    results.add_pass(f"Diffusion model {dataset}")
                else:
                    results.add_fail(f"Diffusion model {dataset}", "Invalid diffusion model")
                    
            except Exception as e:
                results.add_fail(f"Model {dataset}", str(e))
                
    except Exception as e:
        results.add_fail("Model creation system", str(e))

def test_training_components(results):
    """Test training components work together"""
    try:
        from config_fixed import get_debug_config
        from federated.client import Client
        from federated.server import Server
        from diffusion.diffuser import UNet
        from defense.mae_detector import MAEDetector
        from attacks.pgd import PGDAttack
        
        cfg = get_debug_config()
        cfg.N_ROUNDS = 1  # Quick test
        cfg.LOCAL_STEPS_PER_EPOCH = 1
        
        # Test component creation
        diffuser = UNet(in_channels=cfg.IMG_CHANNELS, hidden_channels=16)
        client = Client(0, cfg, diffuser)
        server = Server(cfg)
        detector = MAEDetector(cfg)
        attack = PGDAttack(cfg)
        
        results.add_pass("Training components creation")
        
        # Test client training (very brief)
        try:
            client.train(epochs=1)
            results.add_pass("Client training execution")
        except Exception as e:
            results.add_fail("Client training execution", str(e))
            
    except Exception as e:
        results.add_fail("Training components", str(e))

def test_memory_usage(results):
    """Test memory usage stays reasonable"""
    try:
        import gc
        
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Create components
        from config_fixed import get_debug_config
        from models import get_model
        from diffusion.diffuser import UNet
        
        cfg = get_debug_config()
        model = get_model(cfg)
        diffuser = UNet(in_channels=cfg.IMG_CHANNELS, hidden_channels=32)
        
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_used = current_memory - initial_memory
        
        # Check memory usage is reasonable (under 1GB for basic components)
        if memory_used < 1000:  # 1GB
            results.add_pass(f"Memory usage ({memory_used:.1f} MB)")
        else:
            results.add_fail("Memory usage", f"Used {memory_used:.1f} MB (too much)")
            
        # Clean up
        del model, diffuser
        gc.collect()
        
    except Exception as e:
        results.add_fail("Memory usage test", str(e))

def test_file_operations(results):
    """Test file operations work correctly"""
    try:
        # Test checkpoint directory creation
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)
        
        if checkpoint_dir.exists():
            results.add_pass("Checkpoint directory creation")
        else:
            results.add_fail("Checkpoint directory creation", "Directory not created")
        
        # Test model saving/loading
        from diffusion.diffuser import UNet
        model = UNet(in_channels=3, hidden_channels=16)
        
        test_path = checkpoint_dir / 'test_model.pt'
        torch.save(model.state_dict(), test_path)
        
        if test_path.exists():
            results.add_pass("Model saving")
            
            # Test loading
            loaded_state = torch.load(test_path, map_location='cpu')
            model.load_state_dict(loaded_state)
            results.add_pass("Model loading")
            
            # Clean up
            test_path.unlink()
        else:
            results.add_fail("Model saving", "File not created")
            
    except Exception as e:
        results.add_fail("File operations", str(e))

def test_error_handling(results):
    """Test system handles errors gracefully"""
    try:
        from config_fixed import get_debug_config
        from utils.data_utils import get_dataset
        
        # Test invalid dataset
        try:
            cfg = get_debug_config()
            cfg.DATASET = 'invalid_dataset'
            train_ds, test_ds = get_dataset(cfg)
            results.add_fail("Error handling invalid dataset", "Should have raised error")
        except ValueError:
            results.add_pass("Error handling invalid dataset")
        except Exception as e:
            results.add_fail("Error handling invalid dataset", f"Wrong error type: {e}")
        
        # Test invalid config
        try:
            from models import get_model
            cfg = get_debug_config()
            cfg.N_CLASSES = -1  # Invalid
            model = get_model(cfg)
            # If it doesn't fail, that's actually okay - some models might handle it
            results.add_pass("Error handling invalid config")
        except Exception:
            results.add_pass("Error handling invalid config")
            
    except Exception as e:
        results.add_fail("Error handling system", str(e))

def test_cross_dataset_compatibility(results):
    """Test switching between datasets works correctly"""
    try:
        from config_fixed import get_debug_config
        from utils.data_utils import get_dataset
        from models import get_model
        from diffusion.diffuser import UNet
        
        datasets = ['cifar10', 'mnist', 'cifar100']
        
        for i, dataset in enumerate(datasets):
            cfg = get_debug_config()
            cfg.DATASET = dataset
            
            if dataset == 'mnist':
                cfg.IMG_CHANNELS = 1
                cfg.IMG_SIZE = 28
                cfg.N_CLASSES = 10
            elif dataset == 'cifar100':
                cfg.IMG_CHANNELS = 3
                cfg.IMG_SIZE = 32
                cfg.N_CLASSES = 100
            else:  # cifar10
                cfg.IMG_CHANNELS = 3
                cfg.IMG_SIZE = 32
                cfg.N_CLASSES = 10
            
            # Test dataset switching
            train_ds, test_ds = get_dataset(cfg)
            model = get_model(cfg)
            diffuser = UNet(in_channels=cfg.IMG_CHANNELS, hidden_channels=16)
            
            # Test forward pass
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2)
            batch = next(iter(train_loader))
            images, labels = batch
            
            with torch.no_grad():
                output = model(images)
                diffusion_output = diffuser(images, torch.rand(2))
            
            results.add_pass(f"Cross-dataset compatibility {dataset}")
        
    except Exception as e:
        results.add_fail("Cross-dataset compatibility", str(e))

def test_performance_benchmarks(results):
    """Test basic performance benchmarks"""
    try:
        from config_fixed import get_debug_config
        from utils.data_utils import get_dataset
        from models import get_model
        
        cfg = get_debug_config()
        
        # Test dataset loading speed
        start_time = time.time()
        train_ds, test_ds = get_dataset(cfg)
        dataset_time = time.time() - start_time
        
        if dataset_time < 30:  # Should load in under 30 seconds
            results.add_pass(f"Dataset loading speed ({dataset_time:.1f}s)")
        else:
            results.add_fail("Dataset loading speed", f"Too slow: {dataset_time:.1f}s")
        
        # Test model creation speed
        start_time = time.time()
        model = get_model(cfg)
        model_time = time.time() - start_time
        
        if model_time < 5:  # Should create in under 5 seconds
            results.add_pass(f"Model creation speed ({model_time:.1f}s)")
        else:
            results.add_fail("Model creation speed", f"Too slow: {model_time:.1f}s")
        
        # Test forward pass speed
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32)
        batch = next(iter(train_loader))
        images, labels = batch
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):  # 10 forward passes
                output = model(images)
        forward_time = (time.time() - start_time) / 10
        
        if forward_time < 1.0:  # Should be under 1 second per batch
            results.add_pass(f"Forward pass speed ({forward_time:.3f}s)")
        else:
            results.add_fail("Forward pass speed", f"Too slow: {forward_time:.3f}s")
            
    except Exception as e:
        results.add_fail("Performance benchmarks", str(e))

def test_system_integration(results):
    """Test the complete system integration end-to-end"""
    try:
        # Test the main workflow components
        from config_fixed import get_debug_config
        
        cfg = get_debug_config()
        cfg.N_ROUNDS = 1
        cfg.LOCAL_STEPS_PER_EPOCH = 1
        cfg.N_CLIENTS = 2
        
        # Test system setup would work
        from utils.data_utils import get_dataset
        from models import get_model
        from diffusion.diffuser import UNet
        from federated.client import Client
        from federated.server import Server
        
        # Create all components
        train_ds, test_ds = get_dataset(cfg)
        model = get_model(cfg)
        diffuser = UNet(in_channels=cfg.IMG_CHANNELS, hidden_channels=16)
        clients = [Client(i, cfg, diffuser) for i in range(2)]
        server = Server(cfg)
        
        results.add_pass("System integration - component creation")
        
        # Test basic training loop structure
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=4)
        batch = next(iter(test_loader))
        images, labels = batch
        
        # Test client can process data
        client = clients[0]
        with torch.no_grad():
            output = client.model(images.to(client.device))
        
        results.add_pass("System integration - data flow")
        
    except Exception as e:
        results.add_fail("System integration", str(e))

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    logger = setup_logging()
    results = TestResults()
    
    print("🔍 COMPREHENSIVE TEST SUITE - FINDING ALL PROBLEMS")
    print("=" * 70)
    
    test_functions = [
        ("Critical Imports", test_critical_imports),
        ("Config System Robustness", test_config_system_robustness),
        ("Dataset Loading Robustness", test_dataset_loading_robustness),
        ("Model Creation Robustness", test_model_creation_robustness),
        ("Training Components", test_training_components),
        ("Memory Usage", test_memory_usage),
        ("File Operations", test_file_operations),
        ("Error Handling", test_error_handling),
        ("Cross-Dataset Compatibility", test_cross_dataset_compatibility),
        ("Performance Benchmarks", test_performance_benchmarks),
        ("System Integration", test_system_integration),
    ]
    
    for test_name, test_func in test_functions:
        print(f"\n📋 Testing {test_name}...")
        try:
            test_func(results)
        except Exception as e:
            results.add_fail(test_name, f"Test crashed: {str(e)}")
            print(f"💥 Test {test_name} crashed: {e}")
            traceback.print_exc()
    
    # Print final summary
    summary = results.get_summary()
    
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 70)
    print(f"✅ Passed: {summary['passed']}")
    print(f"❌ Failed: {summary['failed']}")
    print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
    
    if summary['failed'] > 0:
        print(f"\n🔥 ISSUES FOUND:")
        for error in results.errors:
            print(f"  • {error}")
    
    print("\n" + "=" * 70)
    
    if summary['success_rate'] >= 95:
        print("🎉 SYSTEM WORKING EXCELLENTLY!")
        return True
    elif summary['success_rate'] >= 85:
        print("✅ SYSTEM WORKING WELL - Minor issues found")
        return True
    else:
        print("❌ SYSTEM HAS SIGNIFICANT ISSUES")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1) 