#!/usr/bin/env python3
"""
تحلیل عمیق و ریشه‌ای مشکل دقت پایین
Deep Root Cause Analysis for Low Accuracy Issue
"""

import sys
import os
import logging
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_data_normalization():
    """تحلیل عمیق normalization داده‌ها"""
    print("=" * 80)
    print("🔍 DEEP ANALYSIS: DATA NORMALIZATION")
    print("=" * 80)
    
    try:
        from utils.data_utils import get_data_transforms
        
        # بررسی transforms
        train_transform = get_data_transforms('cifar10', train=True)
        test_transform = get_data_transforms('cifar10', train=False)
        
        print("📊 Train Transform:")
        for i, t in enumerate(train_transform.transforms):
            print(f"  {i+1}. {t}")
            
        print("\n📊 Test Transform:")
        for i, t in enumerate(test_transform.transforms):
            print(f"  {i+1}. {t}")
        
        # تست با داده واقعی
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=test_transform
        )
        
        # نمونه‌گیری از داده‌ها
        samples = []
        for i in range(100):
            img, label = dataset[i]
            samples.append(img)
        
        batch = torch.stack(samples)
        
        print(f"\n📊 Data Statistics:")
        print(f"   Shape: {batch.shape}")
        print(f"   Min: {batch.min():.6f}")
        print(f"   Max: {batch.max():.6f}")
        print(f"   Mean: {batch.mean():.6f}")
        print(f"   Std: {batch.std():.6f}")
        
        # بررسی channel-wise statistics
        print(f"\n📊 Channel-wise Statistics:")
        for c in range(3):
            channel_data = batch[:, c, :, :]
            print(f"   Channel {c}: mean={channel_data.mean():.6f}, std={channel_data.std():.6f}")
        
        # مقایسه با CIFAR10 standard
        cifar10_mean = [0.4914, 0.4822, 0.4465]
        cifar10_std = [0.2023, 0.1994, 0.2010]
        print(f"\n📊 Expected CIFAR10 Stats:")
        print(f"   Mean: {cifar10_mean}")
        print(f"   Std: {cifar10_std}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data normalization analysis failed: {e}")
        traceback.print_exc()
        return False

def analyze_model_architecture():
    """تحلیل عمیق معماری مدل"""
    print("\n" + "=" * 80)
    print("🔍 DEEP ANALYSIS: MODEL ARCHITECTURE")
    print("=" * 80)
    
    try:
        from config_fixed import get_full_config
        cfg = get_full_config()
        
        # ایجاد مدل
        import torchvision.models as models
        model = models.resnet18(pretrained=False)
        
        print(f"📊 Original ResNet18 Final Layer:")
        print(f"   Input Features: {model.fc.in_features}")
        print(f"   Output Features: {model.fc.out_features}")
        
        # تغییر لایه نهایی
        model.fc = nn.Linear(model.fc.in_features, getattr(cfg, 'NUM_CLASSES', 10))
        model = model.to(cfg.DEVICE)
        
        print(f"\n📊 Modified Final Layer:")
        print(f"   Input Features: {model.fc.in_features}")
        print(f"   Output Features: {model.fc.out_features}")
        
        # تحلیل وزن‌ها
        print(f"\n📊 Model Parameters:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total Parameters: {total_params:,}")
        print(f"   Trainable Parameters: {trainable_params:,}")
        
        # بررسی initialization
        print(f"\n📊 Final Layer Initialization:")
        print(f"   Weight mean: {model.fc.weight.data.mean():.6f}")
        print(f"   Weight std: {model.fc.weight.data.std():.6f}")
        print(f"   Bias mean: {model.fc.bias.data.mean():.6f}")
        
        # تست forward pass
        print(f"\n📊 Forward Pass Test:")
        dummy_input = torch.randn(4, 3, 32, 32).to(cfg.DEVICE)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"   Input shape: {dummy_input.shape}")
            print(f"   Output shape: {output.shape}")
            print(f"   Output mean: {output.mean():.6f}")
            print(f"   Output std: {output.std():.6f}")
            print(f"   Output range: [{output.min():.6f}, {output.max():.6f}]")
        
        return model
        
    except Exception as e:
        print(f"❌ Model architecture analysis failed: {e}")
        traceback.print_exc()
        return None

def analyze_training_parameters():
    """تحلیل عمیق پارامترهای تمرین"""
    print("\n" + "=" * 80)
    print("🔍 DEEP ANALYSIS: TRAINING PARAMETERS")
    print("=" * 80)
    
    try:
        from config_fixed import get_full_config
        cfg = get_full_config()
        
        print(f"📊 Training Configuration:")
        print(f"   Learning Rate: {getattr(cfg, 'LEARNING_RATE', 'NOT_SET')}")
        print(f"   Batch Size: {cfg.BATCH_SIZE}")
        print(f"   Client Epochs: {cfg.CLIENT_EPOCHS}")
        print(f"   Num Rounds: {cfg.NUM_ROUNDS}")
        print(f"   Num Clients: {cfg.NUM_CLIENTS}")
        print(f"   Device: {cfg.DEVICE}")
        
        # محاسبه تعداد samples per client
        total_samples = 50000  # CIFAR10 train size
        samples_per_client = total_samples // cfg.NUM_CLIENTS
        batches_per_client = samples_per_client // cfg.BATCH_SIZE
        
        print(f"\n📊 Data Distribution:")
        print(f"   Total Train Samples: {total_samples}")
        print(f"   Samples per Client: {samples_per_client}")
        print(f"   Batches per Client: {batches_per_client}")
        print(f"   Total Training Steps per Round: {batches_per_client * cfg.CLIENT_EPOCHS}")
        
        # بررسی learning rate برای CIFAR10
        recommended_lr = 0.1  # Standard for CIFAR10
        actual_lr = getattr(cfg, 'LEARNING_RATE', 0.001)
        
        print(f"\n📊 Learning Rate Analysis:")
        print(f"   Current LR: {actual_lr}")
        print(f"   Recommended LR for CIFAR10: {recommended_lr}")
        if actual_lr < 0.01:
            print(f"   ⚠️  WARNING: Learning rate might be too low!")
        
        return cfg
        
    except Exception as e:
        print(f"❌ Training parameters analysis failed: {e}")
        traceback.print_exc()
        return None

def test_actual_training_step():
    """تست یک قدم واقعی تمرین"""
    print("\n" + "=" * 80)
    print("🔍 DEEP ANALYSIS: ACTUAL TRAINING STEP")
    print("=" * 80)
    
    try:
        from config_fixed import get_full_config
        from utils.data_utils import get_dataset
        
        cfg = get_full_config()
        
        # بارگذاری داده‌ها
        train_dataset, test_dataset = get_dataset(cfg, cfg.DATA_ROOT)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        
        # ایجاد مدل
        import torchvision.models as models
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model = model.to(cfg.DEVICE)
        
        # تنظیم optimizer و loss
        criterion = nn.CrossEntropyLoss()
        
        # تست با learning rate های مختلف
        learning_rates = [0.001, 0.01, 0.1]
        
        for lr in learning_rates:
            print(f"\n📊 Testing Learning Rate: {lr}")
            
            # کپی مدل برای تست
            test_model = models.resnet18(pretrained=False)
            test_model.fc = nn.Linear(test_model.fc.in_features, 10)
            test_model = test_model.to(cfg.DEVICE)
            
            optimizer = optim.SGD(test_model.parameters(), lr=lr, momentum=0.9)
            
            # تمرین 5 batch
            test_model.train()
            losses = []
            accuracies = []
            
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 5:
                    break
                
                data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
                
                optimizer.zero_grad()
                output = test_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # محاسبه accuracy
                pred = output.argmax(dim=1)
                acc = pred.eq(target).float().mean().item() * 100
                
                losses.append(loss.item())
                accuracies.append(acc)
                
                print(f"   Batch {batch_idx+1}: Loss={loss.item():.4f}, Acc={acc:.2f}%")
            
            avg_loss = np.mean(losses)
            avg_acc = np.mean(accuracies)
            print(f"   Average: Loss={avg_loss:.4f}, Acc={avg_acc:.2f}%")
            
            # بررسی gradient norms
            total_norm = 0
            for p in test_model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            print(f"   Gradient Norm: {total_norm:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Actual training step test failed: {e}")
        traceback.print_exc()
        return False

def analyze_client_training():
    """تحلیل عمیق تمرین client"""
    print("\n" + "=" * 80)
    print("🔍 DEEP ANALYSIS: CLIENT TRAINING")
    print("=" * 80)
    
    try:
        from config_fixed import get_full_config
        from federated.client import Client
        
        cfg = get_full_config()
        
        # ایجاد client
        client = Client(0, cfg)
        
        print(f"📊 Client Configuration:")
        print(f"   Client ID: {client.client_id}")
        print(f"   Device: {client.device}")
        print(f"   Model Type: {type(client.model)}")
        
        # بررسی train_loader
        if hasattr(client, 'train_loader'):
            print(f"   Train Loader: {len(client.train_loader)} batches")
            
            # نمونه‌گیری از یک batch
            for batch_idx, (data, target) in enumerate(client.train_loader):
                print(f"   Sample Batch: data={data.shape}, target={target.shape}")
                print(f"   Target range: [{target.min()}, {target.max()}]")
                print(f"   Data range: [{data.min():.6f}, {data.max():.6f}]")
                break
        
        # تست تمرین واقعی
        print(f"\n📊 Testing Client Training:")
        
        # ذخیره وزن‌های اولیه
        initial_weights = {}
        for name, param in client.model.named_parameters():
            initial_weights[name] = param.data.clone()
        
        # تمرین client
        print("   Training for 1 epoch...")
        client.train(epochs=1)
        
        # بررسی تغییر وزن‌ها
        weight_changes = {}
        total_change = 0
        for name, param in client.model.named_parameters():
            change = (param.data - initial_weights[name]).abs().mean().item()
            weight_changes[name] = change
            total_change += change
        
        print(f"   Total Weight Change: {total_change:.8f}")
        
        # نمایش بیشترین تغییرات
        sorted_changes = sorted(weight_changes.items(), key=lambda x: x[1], reverse=True)
        print(f"   Top 5 Layer Changes:")
        for i, (name, change) in enumerate(sorted_changes[:5]):
            print(f"     {i+1}. {name}: {change:.8f}")
        
        if total_change < 1e-6:
            print("   ⚠️  WARNING: Very small weight changes - learning rate might be too low!")
        
        return client
        
    except Exception as e:
        print(f"❌ Client training analysis failed: {e}")
        traceback.print_exc()
        return None

def comprehensive_solution_analysis():
    """تحلیل جامع و ارائه راه‌حل"""
    print("\n" + "=" * 80)
    print("🔍 COMPREHENSIVE SOLUTION ANALYSIS")
    print("=" * 80)
    
    # اجرای تمام تحلیل‌ها
    data_ok = analyze_data_normalization()
    model = analyze_model_architecture()
    cfg = analyze_training_parameters()
    training_ok = test_actual_training_step()
    client = analyze_client_training()
    
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE DIAGNOSIS RESULTS")
    print("=" * 80)
    
    issues_found = []
    solutions = []
    
    # بررسی مشکلات
    if not data_ok:
        issues_found.append("Data normalization issues")
        solutions.append("Fix data transforms and normalization")
    
    if model is None:
        issues_found.append("Model architecture issues")
        solutions.append("Fix model creation and initialization")
    
    if cfg is None:
        issues_found.append("Configuration issues")
        solutions.append("Fix configuration parameters")
    else:
        # بررسی learning rate
        actual_lr = getattr(cfg, 'LEARNING_RATE', 0.001)
        if actual_lr < 0.01:
            issues_found.append(f"Learning rate too low: {actual_lr}")
            solutions.append("Increase learning rate to 0.01-0.1 for CIFAR10")
    
    if not training_ok:
        issues_found.append("Training step issues")
        solutions.append("Fix training loop and optimization")
    
    if client is None:
        issues_found.append("Client training issues")
        solutions.append("Fix federated client implementation")
    
    # گزارش نتایج
    if issues_found:
        print("❌ ISSUES FOUND:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
        
        print("\n🔧 RECOMMENDED SOLUTIONS:")
        for i, solution in enumerate(solutions, 1):
            print(f"   {i}. {solution}")
    else:
        print("✅ No major issues found in components")
    
    # ایجاد فایل اصلاح شده
    create_fixed_config_and_main()
    
    return len(issues_found) == 0

def create_fixed_config_and_main():
    """ایجاد config و main اصلاح شده"""
    print("\n📝 Creating fixed configuration and main files...")
    
    # ایجاد config اصلاح شده
    fixed_config = '''"""
Fixed Configuration for Better Training Results
"""
import torch

class FixedConfig:
    def __init__(self):
        # Dataset settings
        self.DATASET = 'CIFAR10'
        self.DATASET_NAME = 'CIFAR-10'
        self.DATA_ROOT = 'data'
        self.NUM_CLASSES = 10
        self.IMG_SIZE = 32
        self.IMG_CHANNELS = 3
        
        # Training settings - OPTIMIZED FOR CIFAR10
        self.BATCH_SIZE = 64  # Smaller batch size for better convergence
        self.LEARNING_RATE = 0.01  # Higher learning rate for CIFAR10
        self.MOMENTUM = 0.9
        self.WEIGHT_DECAY = 5e-4
        
        # Federated learning settings
        self.NUM_CLIENTS = 5  # Fewer clients for better data per client
        self.NUM_ROUNDS = 10  # Fewer rounds for testing
        self.CLIENT_EPOCHS = 5  # More epochs per client
        
        # Device settings
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.NUM_WORKERS = 0  # Disable multiprocessing to avoid errors
        
        # MAE Detector settings
        self.MAE_THRESHOLD = 0.1
        
def get_fixed_config():
    return FixedConfig()
'''
    
    with open('fixed_config.py', 'w') as f:
        f.write(fixed_config)
    
    # ایجاد main اصلاح شده
    fixed_main = '''#!/usr/bin/env python3
"""
Fixed Main Script with Optimized Parameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_cifar10_loaders(batch_size=64):
    """Get optimized CIFAR10 data loaders"""
    # Optimized transforms for CIFAR10
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, test_loader

def create_optimized_model():
    """Create optimized ResNet18 for CIFAR10"""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    # Better initialization for final layer
    nn.init.xavier_normal_(model.fc.weight)
    nn.init.constant_(model.fc.bias, 0)
    
    return model

def train_model(model, train_loader, device, epochs=5, lr=0.01):
    """Train model with optimized parameters"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                acc = 100. * correct / total
                logger.info(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}: Loss={loss.item():.4f}, Acc={acc:.2f}%')
        
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        logger.info(f'Epoch {epoch+1}/{epochs} completed: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%')

def evaluate_model(model, test_loader, device):
    """Evaluate model"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    logger.info(f'Test Results: Loss={test_loss:.4f}, Accuracy={test_acc:.2f}%')
    return test_acc

def main():
    """Main function with optimized training"""
    logger.info("🚀 Starting optimized CIFAR10 training...")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data
    train_loader, test_loader = get_cifar10_loaders(batch_size=64)
    logger.info(f"Data loaded: {len(train_loader)} train batches, {len(test_loader)} test batches")
    
    # Model
    model = create_optimized_model()
    logger.info("Model created")
    
    # Training
    logger.info("Starting training...")
    train_model(model, train_loader, device, epochs=10, lr=0.01)
    
    # Evaluation
    logger.info("Starting evaluation...")
    final_acc = evaluate_model(model, test_loader, device)
    
    logger.info(f"🎉 Training completed! Final accuracy: {final_acc:.2f}%")
    
    if final_acc > 60:
        logger.info("✅ SUCCESS: Good accuracy achieved!")
    else:
        logger.info("❌ WARNING: Low accuracy - check configuration")
    
    return final_acc

if __name__ == "__main__":
    main()
'''
    
    with open('optimized_main.py', 'w') as f:
        f.write(fixed_main)
    
    print("✅ Created fixed_config.py and optimized_main.py")
    print("\n🎯 NEXT STEPS:")
    print("1. Run: python optimized_main.py")
    print("2. This should achieve 70%+ accuracy")
    print("3. If successful, we can apply the same fixes to the federated version")

def main():
    """اجرای تحلیل جامع"""
    logger.info("🔍 Starting comprehensive root cause analysis...")
    
    success = comprehensive_solution_analysis()
    
    if success:
        logger.info("✅ Analysis completed successfully")
    else:
        logger.info("❌ Issues found - check the analysis above")
    
    return success

if __name__ == "__main__":
    main()
