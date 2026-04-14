#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kim et al., 2023 - SIMPLE TEST (No DiffPure)
===========================================

Test if training works without DiffPure first.
"""

import os
import sys
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Add paths
sys.path.append('.')
sys.path.append('./defense')

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def create_mobilenetv2_model(num_classes=10):
    """Create MobileNetV2 model"""
    model = torchvision.models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, num_classes)
    )
    
    # Better initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    
    model.apply(init_weights)
    return model

class SimpleL2PGD:
    """Simple L2-PGD attack"""
    
    def __init__(self, epsilon=4.5, alpha=0.01, steps=10):
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
    
    def attack(self, model, images, labels):
        """Generate L2-PGD adversarial examples"""
        if images.size(0) < 2:
            return images
            
        model.eval()
        adv_images = images.clone().detach()
        
        # Random start
        delta = torch.randn_like(images)
        delta_norm = torch.norm(delta.view(delta.size(0), -1), dim=1, keepdim=True)
        delta = delta / (delta_norm.view(-1, 1, 1, 1) + 1e-8)
        delta = delta * self.epsilon * torch.rand(images.size(0), 1, 1, 1, device=images.device)
        adv_images = torch.clamp(images + delta, 0, 1)
        
        # PGD steps
        for _ in range(self.steps):
            adv_images_var = adv_images.clone().requires_grad_(True)
            outputs = model(adv_images_var)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            
            grad = adv_images_var.grad.data
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True)
            grad_normalized = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-8)
            
            adv_images = adv_images + self.alpha * grad_normalized
            
            # L2 projection
            delta = adv_images - images
            delta_norm = torch.norm(delta.view(delta.size(0), -1), dim=1, keepdim=True)
            factor = torch.min(torch.ones_like(delta_norm), self.epsilon / (delta_norm + 1e-8))
            delta = delta * factor.view(-1, 1, 1, 1)
            adv_images = torch.clamp(images + delta, 0, 1)
        
        return adv_images.detach()

def create_dirichlet_split(dataset, num_clients=40, alpha=0.4):
    """Create Non-IID data split"""
    num_classes = len(dataset.classes)
    labels = np.array(dataset.targets)
    client_distributions = np.random.dirichlet([alpha] * num_classes, num_clients)
    client_indices = [[] for _ in range(num_clients)]
    
    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        np.random.shuffle(class_indices)
        
        class_distribution = client_distributions[:, class_id]
        class_distribution = class_distribution / class_distribution.sum()
        samples_per_client = (class_distribution * len(class_indices)).astype(int)
        
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + samples_per_client[client_id]
            if client_id == num_clients - 1:
                end_idx = len(class_indices)
            
            client_indices[client_id].extend(class_indices[start_idx:end_idx])
            start_idx = end_idx
    
    return client_indices

class SimpleClient:
    """Simple federated client"""
    
    def __init__(self, client_id, train_indices, dataset, device):
        self.client_id = client_id
        self.device = device
        self.model = create_mobilenetv2_model().to(device)
        
        client_dataset = Subset(dataset, train_indices)
        self.train_loader = DataLoader(client_dataset, batch_size=64, shuffle=True)
        self.num_samples = len(train_indices)
        
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self, epochs=8):
        """Train local model"""
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for data, target in self.train_loader:
                if data.size(0) < 2:
                    continue
                    
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Print loss for first client only
            if self.client_id == 0 and num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"Client 1 Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
    
    def get_state_dict(self):
        return self.model.state_dict()
    
    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict, strict=False)

class SimpleServer:
    """Simple federated server"""
    
    def __init__(self, device):
        self.device = device
        self.global_model = create_mobilenetv2_model().to(device)
        # Store (state_dict, weight)
        self.client_updates = {}
    
    def receive_update(self, client_id, state_dict, weight: int):
        self.client_updates[client_id] = (state_dict, weight)
    
    def aggregate(self):
        """Weighted FedAvg aggregation (float params, includes BN running stats)"""
        if not self.client_updates:
            return
        
        client_items = list(self.client_updates.values())
        client_models = [item[0] for item in client_items]
        client_weights = torch.tensor([max(1, int(item[1])) for item in client_items], dtype=torch.float32)
        total_weight = client_weights.sum()
        global_state = self.global_model.state_dict()
        
        for key, param in global_state.items():
            # Skip BN tracking counter only
            if 'num_batches_tracked' in key:
                continue
            
            # Ensure all clients have the key
            if not all(key in client_state for client_state in client_models):
                continue
            
            try:
                client_tensors = [client_state[key] for client_state in client_models]
                # Only average float tensors
                if not all(isinstance(t, torch.Tensor) and t.is_floating_point() for t in client_tensors):
                    continue
                # Weighted average on CPU
                stacked = torch.stack([t.cpu() for t in client_tensors], dim=0)  # [N, ...]
                # Reshape weights for broadcasting
                w = client_weights.view(-1, *([1] * (stacked.dim() - 1)))
                weighted_sum = (stacked * w).sum(dim=0)
                averaged = weighted_sum / (total_weight + 1e-8)
                global_state[key] = averaged.to(param.dtype)
            except Exception:
                continue
        
        self.global_model.load_state_dict(global_state, strict=False)
        self.client_updates.clear()
    
    def get_state_dict(self):
        return self.global_model.state_dict()

def evaluate_model(model, test_loader, device, attacker=None):
    """Evaluate model"""
    model.eval()
    clean_correct = adv_correct = total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if data.size(0) < 2:
                continue
            if batch_idx >= 50:
                break
                
            data, target = data.to(device), target.to(device)
            
            # Clean accuracy
            clean_outputs = model(data)
            clean_correct += clean_outputs.argmax(1).eq(target).sum().item()
            
            # Adversarial accuracy (enable grads only for attack)
            if attacker:
                torch.cuda.empty_cache()
                with torch.enable_grad():
                    adv_data = attacker.attack(model, data, target)
                
                # Evaluate without grads
                adv_outputs = model(adv_data)
                adv_correct += adv_outputs.argmax(1).eq(target).sum().item()
            
            total += data.size(0)
    
    clean_acc = 100. * clean_correct / total if total > 0 else 0
    adv_acc = 100. * adv_correct / total if total > 0 else 0
    return clean_acc, adv_acc

def main():
    """Main function"""
    logger = setup_logging()
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("=== KIM ET AL. 2023 - SIMPLE TEST ===")
    logger.info("Testing training without DiffPure first...")
    
    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    logger.info(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    
    # Create Non-IID split
    client_indices = create_dirichlet_split(train_dataset, num_clients=40, alpha=0.4)
    
    # Log first 3 clients
    for i in range(3):
        client_labels = [train_dataset[idx][1] for idx in client_indices[i]]
        distribution = [client_labels.count(c) for c in range(10)]
        logger.info(f"Client {i+1} distribution: {distribution}")
    
    # Initialize server and clients
    server = SimpleServer(device)
    clients = [SimpleClient(i, client_indices[i], train_dataset, device) for i in range(40)]
    
    # Create attacker
    attacker = SimpleL2PGD(epsilon=4.5, alpha=0.01, steps=10)
    
    logger.info("Starting federated training...")
    
    # Training loop
    for round_idx in range(15):
        round_start = time.time()
        logger.info(f"Round {round_idx+1}/15")
        
        # Broadcast global model
        global_state = server.get_state_dict()
        for client in clients:
            client.set_state_dict(global_state)
        
        # Client training
        for i, client in enumerate(clients):
            client_start = time.time()
            client.train(epochs=8)
            server.receive_update(i, client.get_state_dict(), weight=client.num_samples)
            client_time = time.time() - client_start
            logger.info(f"Client {i+1}/40 completed in {client_time:.1f}s")
        
        # Aggregate
        logger.info("Aggregating...")
        server.aggregate()
        
        # Evaluate
        clean_acc, adv_acc = evaluate_model(server.global_model, test_loader, device, attacker)
        round_time = time.time() - round_start
        
        logger.info(f"Round {round_idx+1} | Clean: {clean_acc:.2f}% | Adv: {adv_acc:.2f}% | Time: {round_time:.1f}s")
        
        # Early stop if good
        if clean_acc > 60:
            logger.info("Clean accuracy > 60%, training is working!")
            break
    
    logger.info("Test completed!")
    return 0

if __name__ == "__main__":
    main()
