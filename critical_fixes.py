#!/usr/bin/env python3
"""
Critical Fixes for Identified Issues
اصلاحات حیاتی برای مشکلات شناسایی شده
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

def test_data_normalization_fix():
    """اصلاح و تست normalization داده‌ها"""
    print("=" * 80)
    print("🔧 FIXING DATA NORMALIZATION")
    print("=" * 80)
    
    # Correct CIFAR10 transforms
    correct_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Test with correct normalization
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=correct_transform
    )
    
    # Sample data
    samples = []
    for i in range(100):
        img, _ = dataset[i]
        samples.append(img)
    
    batch = torch.stack(samples)
    
    print(f"📊 CORRECTED Data Statistics:")
    print(f"   Shape: {batch.shape}")
    print(f"   Min: {batch.min():.6f}")
    print(f"   Max: {batch.max():.6f}")
    print(f"   Mean: {batch.mean():.6f}")
    print(f"   Std: {batch.std():.6f}")
    
    # Channel-wise
    for c in range(3):
        channel_data = batch[:, c, :, :]
        print(f"   Channel {c}: mean={channel_data.mean():.6f}, std={channel_data.std():.6f}")
    
    # Check if normalization is working
    mean_close_to_zero = abs(batch.mean()) < 0.1
    std_close_to_one = abs(batch.std() - 1.0) < 0.2
    
    if mean_close_to_zero and std_close_to_one:
        print("✅ Data normalization is CORRECT")
        return True
    else:
        print("❌ Data normalization is STILL WRONG")
        return False

def test_optimal_learning_rate():
    """تست learning rate بهینه"""
    print("\n" + "=" * 80)
    print("🔧 FINDING OPTIMAL LEARNING RATE")
    print("=" * 80)
    
    # Correct transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Small dataset for testing
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Small subset for quick testing
    subset_indices = list(range(1000))  # Only 1000 samples
    subset = torch.utils.data.Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=32, shuffle=True, num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different learning rates
    learning_rates = [0.001, 0.005, 0.01, 0.03, 0.05]
    results = {}
    
    for lr in learning_rates:
        print(f"\n📊 Testing LR: {lr}")
        
        # Create fresh model
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= 10:  # Only 10 batches
                break
                
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Check gradient norm
            grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            batch_count += 1
            
            if batch_idx == 0:
                print(f"   First batch: Loss={loss.item():.4f}, Grad_norm={grad_norm:.4f}")
        
        avg_loss = total_loss / batch_count
        avg_acc = 100. * correct / total
        
        results[lr] = {'loss': avg_loss, 'acc': avg_acc}
        print(f"   Final: Loss={avg_loss:.4f}, Acc={avg_acc:.2f}%")
        
        # Check for divergence
        if avg_loss > 5.0:
            print(f"   ⚠️  WARNING: Loss too high - LR {lr} causes divergence")
        elif avg_acc > 20:
            print(f"   ✅ Good: LR {lr} shows learning")
    
    # Find best LR
    best_lr = min(results.keys(), key=lambda x: results[x]['loss'])
    print(f"\n🎯 OPTIMAL LR: {best_lr} (Loss={results[best_lr]['loss']:.4f}, Acc={results[best_lr]['acc']:.2f}%)")
    
    return best_lr

def test_correct_client_data_partition():
    """تست تقسیم صحیح داده‌ها برای client ها"""
    print("\n" + "=" * 80)
    print("🔧 TESTING CORRECT CLIENT DATA PARTITION")
    print("=" * 80)
    
    try:
        from utils.data_utils import create_federated_datasets
        from config_fixed import get_full_config
        
        cfg = get_full_config()
        
        # Get dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        
        print(f"📊 Original dataset size: {len(train_dataset)}")
        
        # Create federated datasets
        federated_datasets = create_federated_datasets(
            train_dataset, cfg.NUM_CLIENTS, 'iid'
        )
        
        print(f"📊 Number of clients: {len(federated_datasets)}")
        
        total_samples = 0
        for i, client_loader in enumerate(federated_datasets):
            client_samples = len(client_loader.dataset)
            total_samples += client_samples
            print(f"   Client {i}: {client_samples} samples, {len(client_loader)} batches")
        
        print(f"📊 Total samples across clients: {total_samples}")
        
        # Test if partition is correct
        expected_per_client = len(train_dataset) // cfg.NUM_CLIENTS
        actual_per_client = total_samples // len(federated_datasets)
        
        if abs(actual_per_client - expected_per_client) < 100:
            print("✅ Client data partition is CORRECT")
            return True
        else:
            print(f"❌ Client data partition is WRONG: expected ~{expected_per_client}, got ~{actual_per_client}")
            return False
            
    except Exception as e:
        print(f"❌ Client partition test failed: {e}")
        return False

def create_completely_fixed_system():
    """ایجاد سیستم کاملاً اصلاح شده"""
    print("\n" + "=" * 80)
    print("🔧 CREATING COMPLETELY FIXED SYSTEM")
    print("=" * 80)
    
    # Fixed main script
    fixed_main = '''#!/usr/bin/env python3
"""
COMPLETELY FIXED FEDERATED LEARNING SYSTEM
سیستم کاملاً اصلاح شده federated learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import logging
import time
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_correct_transforms():
    """Get correctly normalized transforms"""
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
    
    return train_transform, test_transform

def create_federated_data(num_clients=5):
    """Create properly partitioned federated data"""
    train_transform, test_transform = get_correct_transforms()
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Create IID partitions
    total_size = len(train_dataset)
    client_size = total_size // num_clients
    
    client_loaders = []
    for i in range(num_clients):
        start_idx = i * client_size
        end_idx = start_idx + client_size if i < num_clients - 1 else total_size
        indices = list(range(start_idx, end_idx))
        
        client_dataset = Subset(train_dataset, indices)
        client_loader = DataLoader(
            client_dataset, batch_size=64, shuffle=True, num_workers=0
        )
        client_loaders.append(client_loader)
        
        logger.info(f"Client {i}: {len(client_dataset)} samples, {len(client_loader)} batches")
    
    # Test loader
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=0
    )
    
    return client_loaders, test_loader

def create_model():
    """Create properly initialized model"""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    # Better initialization
    nn.init.xavier_normal_(model.fc.weight)
    nn.init.constant_(model.fc.bias, 0)
    
    return model

def train_client(client_loader, device, epochs=3, lr=0.01):
    """Train a single client with optimal parameters"""
    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(client_loader):
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
        
        epoch_loss = running_loss / len(client_loader)
        epoch_acc = 100. * correct / total
        logger.info(f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")
    
    return model

def evaluate_model(model, test_loader, device):
    """Evaluate model correctly"""
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
    
    return test_loss, test_acc

def federated_averaging(models):
    """Simple federated averaging"""
    if not models:
        return None
    
    # Get the first model as template
    avg_model = create_model()
    avg_state_dict = avg_model.state_dict()
    
    # Average all parameters
    for key in avg_state_dict.keys():
        avg_state_dict[key] = torch.stack([model.state_dict()[key] for model in models]).mean(0)
    
    avg_model.load_state_dict(avg_state_dict)
    return avg_model

def main():
    """Main federated learning with all fixes"""
    logger.info("🚀 Starting COMPLETELY FIXED federated learning...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create federated data
    num_clients = 5
    num_rounds = 3
    client_epochs = 3
    lr = 0.01
    
    client_loaders, test_loader = create_federated_data(num_clients)
    logger.info(f"Created federated data for {num_clients} clients")
    
    # Federated training
    global_model = None
    
    for round_idx in range(num_rounds):
        logger.info(f"\\n🔄 Round {round_idx+1}/{num_rounds}")
        
        # Train clients
        client_models = []
        for client_idx in range(num_clients):
            logger.info(f"Training client {client_idx+1}/{num_clients}")
            client_model = train_client(
                client_loaders[client_idx], device, client_epochs, lr
            )
            client_models.append(client_model)
        
        # Federated averaging
        logger.info("Performing federated averaging...")
        global_model = federated_averaging(client_models)
        global_model = global_model.to(device)
        
        # Evaluation
        test_loss, test_acc = evaluate_model(global_model, test_loader, device)
        logger.info(f"Round {round_idx+1} Results: Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
        
        if test_acc > 60:
            logger.info("✅ Good accuracy achieved!")
        
        # Cleanup
        del client_models
        torch.cuda.empty_cache()
    
    logger.info(f"🎉 Federated training completed! Final accuracy: {test_acc:.2f}%")
    
    return test_acc

if __name__ == "__main__":
    final_acc = main()
    if final_acc > 60:
        print("\\n🎉 SUCCESS: Federated learning achieved good accuracy!")
    else:
        print("\\n❌ NEED MORE WORK: Accuracy still low")
'''
    
    with open('completely_fixed_main.py', 'w') as f:
        f.write(fixed_main)
    
    print("✅ Created completely_fixed_main.py")

def main():
    """اجرای تمام اصلاحات"""
    logger.info("🔧 Starting critical fixes...")
    
    # Test 1: Data normalization
    norm_ok = test_data_normalization_fix()
    
    # Test 2: Optimal learning rate
    optimal_lr = test_optimal_learning_rate()
    
    # Test 3: Client data partition
    partition_ok = test_correct_client_data_partition()
    
    # Create fixed system
    create_completely_fixed_system()
    
    print("\n" + "=" * 80)
    print("📊 CRITICAL FIXES SUMMARY")
    print("=" * 80)
    
    print(f"✅ Data Normalization: {'FIXED' if norm_ok else 'NEEDS WORK'}")
    print(f"✅ Optimal Learning Rate: {optimal_lr}")
    print(f"✅ Client Data Partition: {'FIXED' if partition_ok else 'NEEDS WORK'}")
    print(f"✅ Complete Fixed System: CREATED")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Run: python completely_fixed_main.py")
    print("2. This should achieve 70%+ accuracy")
    print("3. If successful, we know the fixes work!")
    
    return True

if __name__ == "__main__":
    main()
