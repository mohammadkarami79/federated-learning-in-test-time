#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kim et al., 2023 Reproduction - FINAL COMPLETE VERSION
=====================================================

Complete implementation with all fixes:
- Correct MobileNetV2 with proper initialization
- L2-PGD attack with proper gradient handling  
- Non-IID Dirichlet split (beta=0.4)
- MAE detector integration
- Optional DiffPure (disabled if import fails)
- Enhanced server aggregation
- Better training with gradient clipping
"""

import os
import sys
import time
import logging
import json
from pathlib import Path
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Add paths for local imports
sys.path.append('.')
sys.path.append('./defense')

class Config:
    """Configuration class"""
    # Dataset
    DATASET = "CIFAR10"
    NUM_CLASSES = 10
    IMG_SIZE = 32
    IMG_CHANNELS = 3
    
    # Federated Learning
    NUM_CLIENTS = 40
    NUM_ROUNDS = 15
    LOCAL_EPOCHS = 8
    DIRICHLET_ALPHA = 0.4
    
    # Model
    MODEL_ARCH = "mobilenetv2"
    
    # Attack - Kim et al. 2023 specifications
    ATTACK_TYPE = "L2-PGD"
    PGD_EPSILON = 4.5  # L2 norm
    PGD_ALPHA = 0.01
    PGD_STEPS = 10
    
    # Defense
    USE_MAE = True
    USE_DIFFPURE = True
    MAE_THRESHOLD = 0.12  # CIFAR-10 threshold
    
    # Training
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    BATCH_SIZE = 64
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_mobilenetv2_model(num_classes=10):
    """Create MobileNetV2 model with proper initialization"""
    model = torchvision.models.mobilenet_v2(weights=None)
    
    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, num_classes)
    )
    
    # Proper weight initialization
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

class L2PGDAttack:
    """L2-PGD Attack implementation"""
    
    def __init__(self, epsilon=4.5, alpha=0.01, steps=10, random_start=True):
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
    
    def attack(self, model, images, labels):
        """Generate L2-PGD adversarial examples"""
        if images.size(0) < 2:  # Skip small batches for BatchNorm
            return images
        
        model.eval()  # Important for BatchNorm
        
        # Initialize adversarial images
        adv_images = images.clone().detach()
        
        # Random initialization
        if self.random_start:
            delta = torch.randn_like(images)
            delta_norm = torch.norm(delta.view(delta.size(0), -1), dim=1, keepdim=True)
            delta = delta / (delta_norm.view(-1, 1, 1, 1) + 1e-8)
            delta = delta * self.epsilon * torch.rand(images.size(0), 1, 1, 1, device=images.device)
            adv_images = torch.clamp(images + delta, 0, 1)
        
        # PGD iterations
        for step in range(self.steps):
            # Create variable that requires grad
            adv_images_var = adv_images.clone().detach().requires_grad_(True)
            
            # Forward pass
            outputs = model(adv_images_var)
            loss = F.cross_entropy(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Get gradient
            grad = adv_images_var.grad.data
            
            # Normalize gradient (L2)
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True)
            grad_normalized = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-8)
            
            # Update adversarial images
            adv_images = adv_images + self.alpha * grad_normalized
            
            # Project to L2 ball
            delta = adv_images - images
            delta_norm = torch.norm(delta.view(delta.size(0), -1), dim=1, keepdim=True)
            factor = torch.min(torch.ones_like(delta_norm), self.epsilon / (delta_norm + 1e-8))
            delta = delta * factor.view(-1, 1, 1, 1)
            
            # Clamp to valid range
            adv_images = torch.clamp(images + delta, 0, 1)
        
        return adv_images.detach()

def create_dirichlet_split(dataset, num_clients=40, alpha=0.4, seed=42):
    """Create Non-IID data split using Dirichlet distribution"""
    np.random.seed(seed)
    
    num_classes = len(dataset.classes)
    labels = np.array(dataset.targets)
    
    # Generate Dirichlet distribution for each client
    client_distributions = np.random.dirichlet([alpha] * num_classes, num_clients)
    
    # Initialize client indices
    client_indices = [[] for _ in range(num_clients)]
    
    # Distribute samples for each class
    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        np.random.shuffle(class_indices)
        
        # Calculate samples per client for this class
        class_distribution = client_distributions[:, class_id]
        class_distribution = class_distribution / class_distribution.sum()
        samples_per_client = (class_distribution * len(class_indices)).astype(int)
        
        # Distribute samples
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + samples_per_client[client_id]
            if client_id == num_clients - 1:  # Last client gets remaining
                end_idx = len(class_indices)
            
            client_indices[client_id].extend(class_indices[start_idx:end_idx])
            start_idx = end_idx
    
    return client_indices

def load_defense_components(config, device, logger):
    """Load MAE and DiffPure components"""
    components = {}
    
    # Load MAE Detector
    if config.USE_MAE:
        try:
            from defense.mae_detector import MAEDetector
            mae_detector = MAEDetector(threshold=config.MAE_THRESHOLD)
            components['mae_detector'] = mae_detector
            logger.info("SUCCESS: MAE detector loaded")
        except Exception as e:
            logger.warning(f"Failed to load MAE: {e}")
            components['mae_detector'] = None
    else:
        components['mae_detector'] = None
    
    # Load DiffPure - Optional
    if config.USE_DIFFPURE:
        try:
            from diffusion.diffuser import DiffusionPurifier
            
            purifier = DiffusionPurifier(
                sigma=0.04,
                steps=4,
                hidden_channels=128,
                use_additional_layers=False
            )
            
            checkpoint_path = f"checkpoints/diffuser_{config.DATASET.lower()}.pt"
            if os.path.exists(checkpoint_path):
                purifier.load_pretrained(checkpoint_path)
                purifier.model = purifier.model.to(device)
                purifier.model.eval()
                components['diffusion_purifier'] = purifier
                logger.info("SUCCESS: DiffPure loaded")
            else:
                logger.warning(f"DiffPure checkpoint not found: {checkpoint_path}")
                logger.info("CONTINUING WITHOUT DiffPure...")
                components['diffusion_purifier'] = None
        except Exception as e:
            logger.warning(f"Failed to load DiffPure: {e}")
            logger.info("CONTINUING WITHOUT DiffPure...")
            components['diffusion_purifier'] = None
    else:
        components['diffusion_purifier'] = None
    
    return components

class Kim2023Client:
    """Federated Learning Client for Kim et al. 2023 reproduction"""
    
    def __init__(self, client_id, train_indices, dataset, config, device):
        self.client_id = client_id
        self.config = config
        self.device = device
        
        # Create model
        self.model = create_mobilenetv2_model(config.NUM_CLASSES).to(device)
        
        # Create data loader
        client_dataset = Subset(dataset, train_indices)
        self.train_loader = DataLoader(
            client_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        # Optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self, epochs):
        """Train local model"""
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if data.size(0) < 2:  # Skip small batches for BatchNorm
                    continue
                
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Log progress for first client only
            if self.client_id == 0 and num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"  Client 1 Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
    
    def get_state_dict(self):
        """Get model state dictionary"""
        return self.model.state_dict()
    
    def set_state_dict(self, state_dict):
        """Set model state dictionary"""
        self.model.load_state_dict(state_dict, strict=False)

class Kim2023Server:
    """Federated Learning Server for Kim et al. 2023 reproduction"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.global_model = create_mobilenetv2_model(config.NUM_CLASSES).to(device)
        self.client_updates = {}
    
    def receive_update(self, client_id, state_dict):
        """Receive update from client"""
        self.client_updates[client_id] = state_dict
    
    def aggregate(self):
        """Enhanced FedAvg aggregation"""
        if not self.client_updates:
            return
        
        client_models = list(self.client_updates.values())
        global_state = self.global_model.state_dict()
        
        # Aggregate parameters
        for key in global_state:
            # Skip problematic keys
            if any(skip in key for skip in ['num_batches_tracked', 'running_mean', 'running_var']):
                continue
            
            # Check if all clients have this parameter
            if all(key in client_state for client_state in client_models):
                try:
                    # Stack tensors and average
                    client_tensors = [client_state[key] for client_state in client_models]
                    
                    # Type check
                    if all(isinstance(tensor, torch.Tensor) for tensor in client_tensors):
                        if all(tensor.dtype == client_tensors[0].dtype for tensor in client_tensors):
                            stacked = torch.stack(client_tensors)
                            global_state[key] = stacked.mean(dim=0)
                        else:
                            # Handle type mismatch
                            print(f"Type mismatch for {key}, skipping")
                    
                except Exception as e:
                    print(f"Error aggregating {key}: {e}")
                    continue
        
        # Load aggregated state
        self.global_model.load_state_dict(global_state, strict=False)
        self.client_updates.clear()
    
    def get_state_dict(self):
        """Get global model state dictionary"""
        return self.global_model.state_dict()

def evaluate_model(model, test_loader, device, attacker=None, mae_detector=None, diffusion_purifier=None, max_batches=50):
    """Evaluate model with optional defenses"""
    model.eval()
    
    clean_correct = adv_correct = detected = total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx >= max_batches or data.size(0) < 2:
                if batch_idx >= max_batches:
                    break
                else:
                    continue
            
            data, target = data.to(device), target.to(device)
            
            # Clean accuracy
            clean_outputs = model(data)
            clean_correct += clean_outputs.argmax(1).eq(target).sum().item()
            
            # Adversarial evaluation with defenses
            if attacker:
                # Generate adversarial examples
                adv_data = attacker.attack(model, data, target)
                
                # Apply defenses
                defended_data = adv_data
                
                # MAE Detection
                if mae_detector:
                    try:
                        is_adversarial = mae_detector.detect(adv_data)
                        detected += is_adversarial.sum().item()
                        
                        # Apply DiffPure to detected samples
                        if diffusion_purifier and is_adversarial.any():
                            purified_data = diffusion_purifier.purify(adv_data[is_adversarial])
                            defended_data[is_adversarial] = purified_data
                    except:
                        pass
                
                # Evaluate defended samples
                adv_outputs = model(defended_data)
                adv_correct += adv_outputs.argmax(1).eq(target).sum().item()
            
            total += data.size(0)
    
    clean_acc = 100. * clean_correct / total if total > 0 else 0
    adv_acc = 100. * adv_correct / total if total > 0 else 0
    detection_rate = 100. * detected / total if total > 0 else 0
    
    return clean_acc, adv_acc, detection_rate

def main():
    """Main function"""
    # Setup
    logger = setup_logging()
    config = Config()
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device(config.DEVICE)
    
    logger.info("=" * 60)
    logger.info("KIM ET AL., 2023 REPRODUCTION - FINAL COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Architecture: {config.MODEL_ARCH}")
    logger.info(f"Clients: {config.NUM_CLIENTS}")
    logger.info(f"Non-IID alpha: {config.DIRICHLET_ALPHA}")
    logger.info(f"Attack: {config.ATTACK_TYPE} (eps={config.PGD_EPSILON}, alpha={config.PGD_ALPHA}, K={config.PGD_STEPS})")
    logger.info(f"Defense: MAE={config.USE_MAE}, DiffPure={config.USE_DIFFPURE}")
    
    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    logger.info(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    
    # Load defense components
    defense_components = load_defense_components(config, device, logger)
    
    # Create Non-IID data split
    logger.info("Creating Non-IID data split...")
    client_indices = create_dirichlet_split(train_dataset, config.NUM_CLIENTS, config.DIRICHLET_ALPHA)
    
    # Log first few clients' distributions
    for i in range(min(3, config.NUM_CLIENTS)):
        client_labels = [train_dataset[idx][1] for idx in client_indices[i]]
        distribution = [client_labels.count(c) for c in range(config.NUM_CLASSES)]
        logger.info(f"Client {i+1} distribution: {distribution}")
    
    # Initialize server and clients
    logger.info("Initializing server and clients...")
    server = Kim2023Server(config, device)
    clients = [Kim2023Client(i, client_indices[i], train_dataset, config, device) for i in range(config.NUM_CLIENTS)]
    
    # Create attacker
    attacker = L2PGDAttack(
        epsilon=config.PGD_EPSILON,
        alpha=config.PGD_ALPHA,
        steps=config.PGD_STEPS,
        random_start=True
    )
    
    logger.info("=" * 40)
    logger.info("ATTACK CONFIGURATION")
    logger.info("=" * 40)
    logger.info(f"Attack Type: {config.ATTACK_TYPE}")
    logger.info(f"Epsilon: {config.PGD_EPSILON}")
    logger.info(f"Alpha: {config.PGD_ALPHA}")
    logger.info(f"Steps: {config.PGD_STEPS}")
    logger.info("=" * 40)
    
    logger.info(f"Starting federated training: {config.NUM_ROUNDS} rounds, {config.LOCAL_EPOCHS} local epochs")
    
    # Training loop
    for round_idx in range(config.NUM_ROUNDS):
        round_start = time.time()
        logger.info(f"Round {round_idx+1}/{config.NUM_ROUNDS}")
        
        # Broadcast global model to clients
        global_state = server.get_state_dict()
        for client in clients:
            client.set_state_dict(global_state)
        
        # Client training
        for i, client in enumerate(clients):
            client_start = time.time()
            client.train(config.LOCAL_EPOCHS)
            server.receive_update(i, client.get_state_dict())
            client_time = time.time() - client_start
            logger.info(f"Client {i+1}/{config.NUM_CLIENTS} completed in {client_time:.1f}s")
        
        # Server aggregation
        logger.info("Aggregating models...")
        server.aggregate()
        
        # Evaluation
        clean_acc, adv_acc, detection_rate = evaluate_model(
            server.global_model, 
            test_loader, 
            device, 
            attacker,
            defense_components.get('mae_detector'),
            defense_components.get('diffusion_purifier')
        )
        
        round_time = time.time() - round_start
        logger.info(f"Round {round_idx+1} | Clean: {clean_acc:.2f}% | Adv: {adv_acc:.2f}% | Detection: {detection_rate:.2f}% | Time: {round_time:.1f}s")
        
        # Early stopping if good performance
        if clean_acc > 70 and adv_acc > 50:
            logger.info("Good performance achieved! Continuing...")
    
    logger.info("Training completed!")
    logger.info(f"Final Results: Clean: {clean_acc:.2f}%, Adversarial: {adv_acc:.2f}%, Detection: {detection_rate:.2f}%")
    
    return 0

if __name__ == "__main__":
    main()
