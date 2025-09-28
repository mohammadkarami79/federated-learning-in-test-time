#!/usr/bin/env python3
"""
Comprehensive Diagnostic Test for Federated Learning System
This will identify ALL issues before we fix them
"""

import sys
import os
import logging
import traceback
import torch
import torchvision
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test all critical imports"""
    print("=" * 60)
    print("🔍 TESTING IMPORTS")
    print("=" * 60)
    
    imports_to_test = [
        ('torch', 'import torch'),
        ('torchvision', 'import torchvision'),
        ('config_fixed', 'from config_fixed import get_full_config'),
        ('data_utils', 'from utils.data_utils import get_dataset, create_federated_datasets'),
        ('client', 'from federated.client import Client'),
        ('server', 'from federated.server import FedServer'),
        ('pgd', 'from attacks.pgd import PGDAttack'),
        ('diffuser', 'from diffusion.diffuser import UNet'),
        ('mae_detector', 'from defense.mae_detector import MAEDetector'),
    ]
    
    results = {}
    for name, import_stmt in imports_to_test:
        try:
            exec(import_stmt)
            print(f"✅ {name}: SUCCESS")
            results[name] = True
        except Exception as e:
            print(f"❌ {name}: FAILED - {e}")
            results[name] = False
    
    return results

def test_config():
    """Test configuration loading"""
    print("\n" + "=" * 60)
    print("🔍 TESTING CONFIGURATION")
    print("=" * 60)
    
    try:
        from config_fixed import get_full_config
        cfg = get_full_config()
        
        required_attrs = [
            'DATASET', 'DATA_ROOT', 'BATCH_SIZE', 'NUM_CLIENTS', 
            'NUM_ROUNDS', 'CLIENT_EPOCHS', 'DEVICE', 'IMG_CHANNELS'
        ]
        
        print(f"✅ Config loaded successfully")
        for attr in required_attrs:
            if hasattr(cfg, attr):
                value = getattr(cfg, attr)
                print(f"✅ {attr}: {value}")
            else:
                print(f"❌ {attr}: MISSING")
                
        return cfg
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        traceback.print_exc()
        return None

def test_dataset_loading(cfg):
    """Test dataset loading with different methods"""
    print("\n" + "=" * 60)
    print("🔍 TESTING DATASET LOADING")
    print("=" * 60)
    
    if cfg is None:
        print("❌ Cannot test datasets - config is None")
        return None, None
    
    try:
        from utils.data_utils import get_dataset
        
        # Method 1: Pass config object
        print("📝 Method 1: Passing config object...")
        result1 = get_dataset(cfg, cfg.DATA_ROOT)
        print(f"Result type: {type(result1)}")
        
        if isinstance(result1, tuple):
            train_ds, test_ds = result1
            print(f"✅ Got tuple: train={len(train_ds)}, test={len(test_ds)}")
            print(f"Train dataset type: {type(train_ds)}")
            print(f"Test dataset type: {type(test_ds)}")
            return train_ds, test_ds
        else:
            print(f"❌ Got single object: {type(result1)}")
            if hasattr(result1, '__len__'):
                print(f"Length: {len(result1)}")
            return result1, None
            
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        traceback.print_exc()
        return None, None

def test_dataloader_creation(train_dataset, test_dataset, cfg):
    """Test DataLoader creation"""
    print("\n" + "=" * 60)
    print("🔍 TESTING DATALOADER CREATION")
    print("=" * 60)
    
    if train_dataset is None or test_dataset is None:
        print("❌ Cannot test DataLoaders - datasets are None")
        return None, None
    
    try:
        from torch.utils.data import DataLoader
        
        # Test train loader
        print("📝 Creating train DataLoader...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid worker issues
            pin_memory=False
        )
        print(f"✅ Train DataLoader created: {len(train_loader)} batches")
        
        # Test test loader
        print("📝 Creating test DataLoader...")
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid worker issues
            pin_memory=False
        )
        print(f"✅ Test DataLoader created: {len(test_loader)} batches")
        
        # Test actual data loading
        print("📝 Testing actual data loading...")
        for i, (data, target) in enumerate(train_loader):
            print(f"✅ Train batch {i}: data={data.shape}, target={target.shape}")
            if i >= 2:  # Test first 3 batches
                break
                
        for i, (data, target) in enumerate(test_loader):
            print(f"✅ Test batch {i}: data={data.shape}, target={target.shape}")
            if i >= 2:  # Test first 3 batches
                break
                
        return train_loader, test_loader
        
    except Exception as e:
        print(f"❌ DataLoader creation failed: {e}")
        traceback.print_exc()
        return None, None

def test_model_creation(cfg):
    """Test model creation"""
    print("\n" + "=" * 60)
    print("🔍 TESTING MODEL CREATION")
    print("=" * 60)
    
    try:
        import torchvision.models as models
        
        # Test ResNet18 creation
        print("📝 Creating ResNet18...")
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, getattr(cfg, 'NUM_CLASSES', 10))
        model = model.to(cfg.DEVICE)
        print(f"✅ ResNet18 created and moved to {cfg.DEVICE}")
        
        # Test forward pass
        print("📝 Testing forward pass...")
        dummy_input = torch.randn(2, 3, 32, 32).to(cfg.DEVICE)
        output = model(dummy_input)
        print(f"✅ Forward pass successful: input={dummy_input.shape}, output={output.shape}")
        
        return model
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        traceback.print_exc()
        return None

def test_federated_components(cfg):
    """Test federated learning components"""
    print("\n" + "=" * 60)
    print("🔍 TESTING FEDERATED COMPONENTS")
    print("=" * 60)
    
    try:
        from federated.client import Client
        from federated.server import FedServer
        
        # Test server creation
        print("📝 Creating FedServer...")
        server = FedServer(cfg)
        print("✅ FedServer created successfully")
        
        # Test client creation
        print("📝 Creating Client...")
        client = Client(0, cfg)
        print("✅ Client created successfully")
        print(f"Client model type: {type(client.model)}")
        
        return server, client
        
    except Exception as e:
        print(f"❌ Federated components failed: {e}")
        traceback.print_exc()
        return None, None

def test_mae_detector(cfg):
    """Test MAE detector"""
    print("\n" + "=" * 60)
    print("🔍 TESTING MAE DETECTOR")
    print("=" * 60)
    
    try:
        from defense.mae_detector import MAEDetector
        
        print("📝 Creating MAE Detector...")
        mae_detector = MAEDetector(cfg)
        print("✅ MAE Detector created successfully")
        
        # Test detection on dummy data
        print("📝 Testing MAE detection...")
        dummy_images = torch.randn(4, 3, 32, 32).to(cfg.DEVICE)
        detections = mae_detector.detect(dummy_images)
        print(f"✅ MAE detection successful: {detections.shape}")
        
        return mae_detector
        
    except Exception as e:
        print(f"❌ MAE Detector failed: {e}")
        traceback.print_exc()
        return None

def test_complete_pipeline():
    """Test the complete pipeline"""
    print("\n" + "=" * 60)
    print("🔍 TESTING COMPLETE PIPELINE")
    print("=" * 60)
    
    # Test all components
    import_results = test_imports()
    cfg = test_config()
    train_dataset, test_dataset = test_dataset_loading(cfg)
    train_loader, test_loader = test_dataloader_creation(train_dataset, test_dataset, cfg)
    model = test_model_creation(cfg)
    server, client = test_federated_components(cfg)
    mae_detector = test_mae_detector(cfg)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    components = [
        ("Imports", all(import_results.values())),
        ("Config", cfg is not None),
        ("Datasets", train_dataset is not None and test_dataset is not None),
        ("DataLoaders", train_loader is not None and test_loader is not None),
        ("Model", model is not None),
        ("Federated Components", server is not None and client is not None),
        ("MAE Detector", mae_detector is not None),
    ]
    
    all_good = True
    for name, status in components:
        if status:
            print(f"✅ {name}: WORKING")
        else:
            print(f"❌ {name}: BROKEN")
            all_good = False
    
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 ALL COMPONENTS WORKING! System should run successfully.")
        
        # Create a minimal working config
        print("\n📝 Creating minimal working main.py...")
        create_minimal_main(cfg, train_dataset, test_dataset)
        
    else:
        print("❌ SOME COMPONENTS BROKEN! Need to fix before running main.py")
        
        # Create fixes for broken components
        create_fixes_for_broken_components(components)
    
    return all_good

def create_minimal_main(cfg, train_dataset, test_dataset):
    """Create a minimal working main.py"""
    minimal_main = '''#!/usr/bin/env python3
"""
MINIMAL WORKING MAIN.PY - Generated by diagnostic
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import time
import traceback
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Minimal working main function"""
    try:
        logger.info("🚀 Starting minimal federated learning...")
        
        # Load config
        from config_fixed import get_full_config
        cfg = get_full_config()
        logger.info(f"✅ Config loaded: {cfg.DATASET} dataset")
        
        # Load datasets
        from utils.data_utils import get_dataset
        train_dataset, test_dataset = get_dataset(cfg, cfg.DATA_ROOT)
        logger.info(f"✅ Datasets loaded: {len(train_dataset)} train, {len(test_dataset)} test")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
        logger.info(f"✅ DataLoaders created")
        
        # Create model
        import torchvision.models as models
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, getattr(cfg, 'NUM_CLASSES', 10))
        model = model.to(cfg.DEVICE)
        logger.info(f"✅ Model created and moved to {cfg.DEVICE}")
        
        # Simple training loop (1 round, 1 epoch)
        logger.info("🔄 Starting simple training...")
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 10:  # Only 10 batches for quick test
                break
                
            data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
        train_acc = 100. * correct / total
        logger.info(f"✅ Training completed: Loss={running_loss/10:.4f}, Acc={train_acc:.2f}%")
        
        # Simple evaluation
        logger.info("🔄 Starting evaluation...")
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                if batch_idx >= 10:  # Only 10 batches for quick test
                    break
                    
                data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        test_acc = 100. * correct / total
        logger.info(f"✅ Evaluation completed: Loss={test_loss/10:.4f}, Acc={test_acc:.2f}%")
        
        # Save results
        results = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_loss': running_loss / 10,
            'test_loss': test_loss / 10
        }
        
        logger.info("🎉 MINIMAL TEST SUCCESSFUL!")
        logger.info(f"📊 Results: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Minimal test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\\n🎉 SUCCESS! The system is working. You can now run the full federated learning.")
    else:
        print("\\n❌ FAILURE! Please check the errors above and fix them first.")
'''
    
    with open('minimal_working_main.py', 'w') as f:
        f.write(minimal_main)
    
    print("✅ Created minimal_working_main.py - test this first!")

def create_fixes_for_broken_components(components):
    """Create specific fixes for broken components"""
    print("\n📝 CREATING FIXES FOR BROKEN COMPONENTS...")
    
    fixes = []
    
    for name, status in components:
        if not status:
            if name == "Datasets":
                fixes.append("Fix dataset loading in utils/data_utils.py")
            elif name == "DataLoaders":
                fixes.append("Fix DataLoader creation - use num_workers=0")
            elif name == "MAE Detector":
                fixes.append("Fix MAE detector imports and implementation")
            elif name == "Federated Components":
                fixes.append("Fix federated client/server imports")
    
    if fixes:
        print("🔧 REQUIRED FIXES:")
        for i, fix in enumerate(fixes, 1):
            print(f"{i}. {fix}")

if __name__ == "__main__":
    print("🔍 COMPREHENSIVE FEDERATED LEARNING DIAGNOSTIC")
    print("This will identify ALL issues in your system...")
    print()
    
    success = test_complete_pipeline()
    
    if success:
        print("\n🎯 NEXT STEPS:")
        print("1. Run: python minimal_working_main.py")
        print("2. If that works, the system is ready!")
        print("3. Then you can run the full main.py")
    else:
        print("\n🎯 NEXT STEPS:")
        print("1. Fix the broken components listed above")
        print("2. Run this diagnostic again")
        print("3. Only proceed when all components are working")
