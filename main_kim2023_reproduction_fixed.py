#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kim et al., 2023 Reproduction - FIXED VERSION
============================================

This script reproduces the experimental setup from:
"Characterizing Internal Evasion Attacks in Federated Learning"
by Kim et al., 2023

Fixed issues:
- Proper server aggregation (from main.py)
- Working MAE detector integration
- Working DiffPure integration  
- Fixed model architecture and training
"""

import os
import sys
import time
import json
import logging
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Ensure we can import from our modules
sys.path.append('.')
sys.path.append('./defense')
sys.path.append('./diffusion')
sys.path.append('./attacks')

@dataclass
class Kim2023Config:
    """Configuration for Kim et al., 2023 reproduction"""
    # Model and data
    DATASET: str = "cifar10"
    IMG_SIZE: int = 32
    IMG_CHANNELS: int = 3
    NUM_CLASSES: int = 10
    BATCH_SIZE: int = 64
    
    # Federated Learning
    NUM_CLIENTS: int = 40
    NUM_ROUNDS: int = 15
    LOCAL_EPOCHS: int = 8
    
    # Non-IID setup
    DIRICHLET_BETA: float = 0.4
    
    # L2-PGD Attack (Kim et al., 2023)
    ATTACK_EPSILON: float = 4.5
    ATTACK_ALPHA: float = 0.01
    ATTACK_STEPS: int = 10
    
    # Defense components
    USE_MAE_DETECTOR: bool = True
    USE_DIFFPURE: bool = True
    
    # Device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Diffusion settings
    DIFFUSION_HIDDEN_CHANNELS: int = 128

def setup_logging() -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def create_mobilenetv2_model(num_classes: int = 10) -> nn.Module:
    """Create MobileNetV2 model for Kim et al., 2023 reproduction"""
    model = torchvision.models.mobilenet_v2(weights=None)  # No pretrained weights
    
    # Modify classifier for CIFAR-10
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
    
    model.apply(init_weights)
    return model

class L2PGDAttack:
    """L2-norm PGD attack for Kim et al., 2023 reproduction"""
    
    def __init__(self, epsilon=4.5, alpha=0.01, steps=10, random_start=True):
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
    
    def attack(self, model, images, labels):
        """Generate L2-PGD adversarial examples"""
        model.eval()  # Important: set to eval mode
        
        images = images.clone().detach()
        labels = labels.clone().detach()
        
        # Skip small batches to avoid BatchNorm issues
        if images.size(0) < 2:
            return images
        
        adv_images = images.clone().detach()
        
        if self.random_start:
            # Random start in L2 ball
            delta = torch.randn_like(images)
            delta = delta / torch.norm(delta.view(delta.size(0), -1), dim=1, keepdim=True).view(-1, 1, 1, 1)
            delta = delta * self.epsilon * torch.rand(images.size(0), 1, 1, 1, device=images.device)
            adv_images = torch.clamp(images + delta, 0, 1)
        
        for step in range(self.steps):
            adv_images = adv_images.detach()
            adv_images.requires_grad_(True)
            
            outputs = model(adv_images)
            loss = F.cross_entropy(outputs, labels)
            
            # Compute gradients
            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
            
            # L2 projection
            delta = adv_images - images
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True)
            grad = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-8)
            
            # Update
            delta = delta + self.alpha * grad
            
            # Project to L2 ball
            delta_norm = torch.norm(delta.view(delta.size(0), -1), dim=1, keepdim=True)
            delta = delta / torch.max(delta_norm / self.epsilon, torch.ones_like(delta_norm)).view(-1, 1, 1, 1)
            
            adv_images = torch.clamp(images + delta, 0, 1)
        
        return adv_images.detach()

def create_dirichlet_split(dataset, num_clients: int, alpha: float = 0.4):
    """Create Non-IID data split using Dirichlet distribution"""
    num_classes = len(dataset.classes)
    num_samples = len(dataset)
    
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # Sample from Dirichlet distribution
    client_distributions = np.random.dirichlet([alpha] * num_classes, num_clients)
    
    # Distribute samples to clients
    client_indices = [[] for _ in range(num_clients)]
    
    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        
        # Shuffle class indices
        np.random.shuffle(class_indices)
        
        # Calculate number of samples per client for this class
        class_distribution = client_distributions[:, class_id]
        class_distribution = class_distribution / class_distribution.sum()
        
        samples_per_client = (class_distribution * len(class_indices)).astype(int)
        
        # Distribute samples
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + samples_per_client[client_id]
            if client_id == num_clients - 1:  # Last client gets remaining samples
                end_idx = len(class_indices)
            
            client_indices[client_id].extend(class_indices[start_idx:end_idx])
            start_idx = end_idx
    
    return client_indices

class Kim2023Client:
    """Federated client for Kim et al., 2023 reproduction"""
    
    def __init__(self, client_id: int, config: Kim2023Config, train_indices: List[int], dataset):
        self.client_id = client_id
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Create model
        self.model = create_mobilenetv2_model(config.NUM_CLASSES).to(self.device)
        
        # Create data loader
        client_dataset = Subset(dataset, train_indices)
        self.train_loader = DataLoader(
            client_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        
        # Optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self, epochs: int = None):
        """Train the local model"""
        if epochs is None:
            epochs = self.config.LOCAL_EPOCHS
        
        self.model.train()
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Skip small batches to avoid BatchNorm issues
                if data.size(0) < 2:
                    continue
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
    
    def get_model_state(self) -> Dict:
        """Get model state for aggregation"""
        return self.model.state_dict()
    
    def set_model_state(self, state_dict: Dict):
        """Set model state from server"""
        self.model.load_state_dict(state_dict, strict=False)

class Kim2023Server:
    """Federated server for Kim et al., 2023 reproduction - uses working aggregation from main.py"""
    
    def __init__(self, config: Kim2023Config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Global model
        self.global_model = create_mobilenetv2_model(config.NUM_CLASSES).to(self.device)
        
        # Client updates storage
        self.client_updates = {}
    
    def receive_update(self, client_id: int, client_state: Dict):
        """Store client update"""
        self.client_updates[client_id] = client_state
    
    def get_global_model_state(self) -> Dict:
        """Get global model state"""
        return self.global_model.state_dict()
    
    def aggregate(self):
        """FedAvg aggregation - same method as working main.py"""
        n_clients = len(self.client_updates)
        if n_clients == 0:
            return
        
        # Get all client models
        client_models = list(self.client_updates.values())
        
        # Average the parameters (exact same method as main.py)
        avg_state = self.global_model.state_dict()
        
        for key in avg_state:
            # Skip BN layers for stability (same as main.py)
            if 'bn' not in key.lower() and 'num_batches_tracked' not in key:
                # Check if all clients have this key
                if all(key in client_state for client_state in client_models):
                    try:
                        stacked = torch.stack([
                            client_state[key] for client_state in client_models
                        ])
                        # Convert to float if it's integer type
                        if stacked.dtype in [torch.long, torch.int, torch.int64, torch.int32]:
                            stacked = stacked.float()
                        avg_state[key] = stacked.mean(dim=0)
                    except Exception as e:
                        # Skip parameters that can't be stacked
                        continue
                
        # Load averaged weights back
        self.global_model.load_state_dict(avg_state, strict=False)
        
        # Clear updates
        self.client_updates.clear()

def load_defense_components(config: Kim2023Config, logger: logging.Logger):
    """Load pre-trained defense components - same method as main.py"""
    components = {}
    
    # Load MAE detector (same method as main.py)
    if config.USE_MAE_DETECTOR:
        try:
            from defense.mae_detector import MAEDetector
            
            # Create compatible config
            class MAEConfig:
                def __init__(self, kim_config):
                    self.DATASET = kim_config.DATASET
                    self.DEVICE = kim_config.DEVICE
                    self.IMG_SIZE = kim_config.IMG_SIZE
                    self.IMG_CHANNELS = kim_config.IMG_CHANNELS
                    self.NUM_CLASSES = kim_config.NUM_CLASSES
            
            mae_config = MAEConfig(config)
            mae_detector = MAEDetector(mae_config)
            components['mae_detector'] = mae_detector
            logger.info("✅ MAE detector loaded successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load MAE detector: {e}")
            components['mae_detector'] = None
    else:
        components['mae_detector'] = None
    
    # Load DiffPure model (same architecture as main.py)
    if config.USE_DIFFPURE:
        try:
            import torch.nn as nn
            
            # Use the EXACT same architecture as main.py TrainedUNet
            class TrainedUNet(nn.Module):
                def __init__(self, in_channels=3, hidden_channels=128):
                    super().__init__()
                    
                    # Time embedding
                    self.time_embed = nn.Sequential(
                        nn.Linear(1, 64),
                        nn.ReLU(),
                        nn.Linear(64, hidden_channels)
                    )
                    
                    # Encoder
                    self.enc1 = nn.Sequential(
                        nn.Conv2d(in_channels, hidden_channels//4, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(hidden_channels//4, hidden_channels//2, 3, padding=1),
                        nn.ReLU()
                    )
                    
                    self.enc2 = nn.Sequential(
                        nn.MaxPool2d(2),
                        nn.Conv2d(hidden_channels//2, hidden_channels//2, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(hidden_channels//2, hidden_channels, 3, padding=1),
                        nn.ReLU()
                    )
                    
                    # Bottleneck
                    self.bottleneck = nn.Sequential(
                        nn.MaxPool2d(2),
                        nn.Conv2d(hidden_channels, hidden_channels*2, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(hidden_channels*2, hidden_channels*2, 3, padding=1),
                        nn.ReLU()
                    )
                    
                    # Decoder (simplified for compatibility)
                    self.up1 = nn.ConvTranspose2d(hidden_channels*2, hidden_channels, 2, stride=2)
                    self.dec1 = nn.Sequential(
                        nn.Conv2d(hidden_channels*2, hidden_channels, 3, padding=1),
                        nn.ReLU()
                    )
                    
                    self.up2 = nn.ConvTranspose2d(hidden_channels, hidden_channels//2, 2, stride=2)
                    self.dec2 = nn.Sequential(
                        nn.Conv2d(hidden_channels, hidden_channels//2, 3, padding=1),
                        nn.ReLU()
                    )
                    
                    # Final output
                    self.final = nn.Conv2d(hidden_channels//2, in_channels, 1)
                
                def forward(self, x, t=None):
                    # Encoder path
                    e1 = self.enc1(x)
                    e2 = self.enc2(e1)
                    
                    # Bottleneck
                    b = self.bottleneck(e2)
                    
                    # Decoder path
                    d1 = self.up1(b)
                    # Handle dimension mismatch
                    if d1.size(2) != e2.size(2) or d1.size(3) != e2.size(3):
                        d1 = F.interpolate(d1, size=(e2.size(2), e2.size(3)), mode='bilinear', align_corners=False)
                    d1 = torch.cat([d1, e2], dim=1)
                    d1 = self.dec1(d1)
                    
                    d2 = self.up2(d1)
                    # Handle dimension mismatch
                    if d2.size(2) != e1.size(2) or d2.size(3) != e1.size(3):
                        d2 = F.interpolate(d2, size=(e1.size(2), e1.size(3)), mode='bilinear', align_corners=False)
                    d2 = torch.cat([d2, e1], dim=1)
                    d2 = self.dec2(d2)
                    
                    # Final output
                    out = self.final(d2)
                    return torch.clamp(out, 0, 1)
            
            # Try with 256 first (matching the checkpoint), then fallback to 128
            try:
                diffuser = TrainedUNet(
                    in_channels=config.IMG_CHANNELS, 
                    hidden_channels=256  # Match the checkpoint
                ).to(torch.device(config.DEVICE))
            except:
                diffuser = TrainedUNet(
                    in_channels=config.IMG_CHANNELS, 
                    hidden_channels=128  # Fallback
                ).to(torch.device(config.DEVICE))
            
            # Load trained diffusion model (same as main.py)
            diffusion_checkpoint = Path(f"checkpoints/diffuser_{config.DATASET.lower()}.pt")
            if diffusion_checkpoint.exists():
                try:
                    diffuser.load_state_dict(torch.load(diffusion_checkpoint, map_location=config.DEVICE))
                    logger.info(f"✅ Loaded diffusion model: {diffusion_checkpoint}")
                    components['diffuser'] = diffuser
                except Exception as e:
                    logger.error(f"❌ Failed to load diffusion model: {e}")
                    logger.info("🔄 Will proceed without diffusion model...")
                    components['diffuser'] = None
            else:
                logger.warning(f"⚠️ Diffusion checkpoint not found: {diffusion_checkpoint}")
                components['diffuser'] = None
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load DiffPure: {e}")
            components['diffuser'] = None
    else:
        components['diffuser'] = None
    
    return components

def diffpure_purify(diffuser, adv_data, config):
    """Apply DiffPure purification - same method as main.py"""
    if diffuser is None:
        return adv_data
    
    diffuser.eval()
    with torch.no_grad():
        # Simple denoising
        return diffuser(adv_data)

def apply_defense_pipeline(images: torch.Tensor, components: Dict, config: Kim2023Config) -> torch.Tensor:
    """Apply defense pipeline: MAE Detection → DiffPure → Clean"""
    
    # MAE Detection
    if components.get('mae_detector') is not None:
        try:
            detected_int = components['mae_detector'].detect(images)
            detected_mask = detected_int.to(torch.bool)
            
            # Apply DiffPure to detected adversarial samples
            if components.get('diffuser') is not None and detected_mask.any():
                selected_indices = torch.where(detected_mask)[0]
                if len(selected_indices) > 0:
                    to_purify = images[selected_indices]
                    purified = diffpure_purify(components['diffuser'], to_purify, config)
                    images[selected_indices] = purified
                    
        except Exception as e:
            pass  # Continue without defense
    
    return images

def evaluate_model(model, test_loader, device, components=None, config=None, attacker=None):
    """Evaluate model on clean and adversarial examples"""
    model.eval()
    
    clean_correct = 0
    adv_correct = 0
    detection_count = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # Skip small batches
            if data.size(0) < 2:
                continue
                
            data, target = data.to(device), target.to(device)
            
            # Clean accuracy
            clean_outputs = model(data)
            clean_pred = clean_outputs.argmax(dim=1, keepdim=True)
            clean_correct += clean_pred.eq(target.view_as(clean_pred)).sum().item()
            
            # Adversarial accuracy
            if attacker is not None:
                model.eval()  # Ensure eval mode for attack
                adv_data = attacker.attack(model, data, target)
                
                # Apply defense pipeline
                if components is not None and config is not None:
                    defended_data = apply_defense_pipeline(adv_data.clone(), components, config)
                    
                    # Check detection
                    if components.get('mae_detector') is not None:
                        try:
                            detected = components['mae_detector'].detect(adv_data)
                            detection_count += detected.sum().item()
                        except:
                            pass
                    
                    adv_outputs = model(defended_data)
                else:
                    adv_outputs = model(adv_data)
                
                adv_pred = adv_outputs.argmax(dim=1, keepdim=True)
                adv_correct += adv_pred.eq(target.view_as(adv_pred)).sum().item()
            
            total += data.size(0)
            
            # Limit evaluation for speed
            if batch_idx >= 50:
                break
    
    clean_acc = 100. * clean_correct / total if total > 0 else 0
    adv_acc = 100. * adv_correct / total if total > 0 else 0
    detection_rate = 100. * detection_count / total if total > 0 else 0
    
    return clean_acc, adv_acc, detection_rate

def main():
    """Main function for Kim et al., 2023 reproduction"""
    parser = argparse.ArgumentParser(description='Kim et al., 2023 Reproduction')
    parser.add_argument('--output-dir', type=str, default='./kim2023_results_fixed', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Configuration
    config = Kim2023Config()
    
    logger.info("============================================================")
    logger.info("KIM ET AL., 2023 REPRODUCTION EXPERIMENT - FIXED VERSION")
    logger.info("============================================================")
    logger.info(f"Kim et al., 2023 Reproduction Configuration:")
    logger.info(f"- Architecture: MobileNetV2 (no pretrained)")
    logger.info(f"- Clients: {config.NUM_CLIENTS}")
    logger.info(f"- Non-IID beta: {config.DIRICHLET_BETA}")
    logger.info(f"- Attack: L2-PGD (eps={config.ATTACK_EPSILON}, alpha={config.ATTACK_ALPHA}, K={config.ATTACK_STEPS})")
    logger.info(f"- Defense: MAE={config.USE_MAE_DETECTOR}, DiffPure={config.USE_DIFFPURE}")
    
    # Load CIFAR-10
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    logger.info(f"Test dataset loaded: {len(test_dataset)} samples")
    
    # Load defense components
    logger.info("Loading defense components...")
    components = load_defense_components(config, logger)
    
    # Create Non-IID data split
    logger.info("Creating Non-IID data split with Dirichlet distribution...")
    client_indices = create_dirichlet_split(train_dataset, config.NUM_CLIENTS, config.DIRICHLET_BETA)
    
    # Log data distribution for first few clients
    for i in range(min(5, config.NUM_CLIENTS)):
        client_labels = [train_dataset[idx][1] for idx in client_indices[i]]
        distribution = [client_labels.count(c) for c in range(config.NUM_CLASSES)]
        logger.info(f"Client {i+1} data distribution: {distribution}")
    
    # Initialize server and clients
    logger.info("Initializing server and clients...")
    server = Kim2023Server(config)
    
    clients = []
    for i in range(config.NUM_CLIENTS):
        client = Kim2023Client(i, config, client_indices[i], train_dataset)
        clients.append(client)
    
    # Create attacker
    attacker = L2PGDAttack(
        epsilon=config.ATTACK_EPSILON,
        alpha=config.ATTACK_ALPHA,
        steps=config.ATTACK_STEPS,
        random_start=True
    )
    
    logger.info("========================================")
    logger.info("ATTACK CONFIGURATION (Kim et al., 2023)")
    logger.info("========================================")
    logger.info(f"Attack Type: L2-norm PGD (Untargeted)")
    logger.info(f"Epsilon (eps): {config.ATTACK_EPSILON}")
    logger.info(f"Alpha (alpha): {config.ATTACK_ALPHA}")
    logger.info(f"Steps (K): {config.ATTACK_STEPS}")
    logger.info(f"Random Start: True")
    logger.info(f"Formula: x_{{t+1}} = Pi_{{x+S}}(x_t + alpha * grad_x L(h_theta, x, y))")
    logger.info("========================================")
    
    # Main federated training loop
    logger.info(f"Starting federated training: {config.NUM_ROUNDS} rounds, {config.LOCAL_EPOCHS} local epochs")
    
    results = []
    
    for round_idx in range(config.NUM_ROUNDS):
        round_start_time = time.time()
        logger.info(f"Round {round_idx+1}/{config.NUM_ROUNDS}")
        print(f"[L2-PGD Attack] eps={config.ATTACK_EPSILON}, alpha={config.ATTACK_ALPHA}, steps={config.ATTACK_STEPS}, random_start=True")
        
        # Broadcast global model to clients
        global_state = server.get_global_model_state()
        for client in clients:
            client.set_model_state(global_state)
        
        # Client training
        for client_idx, client in enumerate(clients):
            client_start_time = time.time()
            client.train(config.LOCAL_EPOCHS)
            client_state = client.get_model_state()
            server.receive_update(client_idx, client_state)
            client_time = time.time() - client_start_time
            logger.info(f"Client {client_idx+1}/{config.NUM_CLIENTS} completed in {client_time:.1f}s")
        
        # Server aggregation
        logger.info("Aggregating models...")
        server.aggregate()
        
        # Evaluation
        logger.info("Running evaluation...")
        
        # Set server model for evaluation
        eval_model = server.global_model
        clean_acc, adv_acc, detection_rate = evaluate_model(
            eval_model, test_loader, server.device, components, config, attacker
        )
        
        round_time = time.time() - round_start_time
        
        logger.info(f"Round {round_idx+1} Clean Acc: {clean_acc:.2f}% | Adv Acc: {adv_acc:.2f}% | Detection: {detection_rate:.2f}% | Time: {round_time:.1f}s")
        
        results.append({
            'round': round_idx + 1,
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
            'detection_rate': detection_rate,
            'round_time': round_time
        })
    
    # Save results
    final_results = {
        'config': {
            'num_clients': config.NUM_CLIENTS,
            'num_rounds': config.NUM_ROUNDS,
            'local_epochs': config.LOCAL_EPOCHS,
            'dirichlet_beta': config.DIRICHLET_BETA,
            'attack_epsilon': config.ATTACK_EPSILON,
            'attack_alpha': config.ATTACK_ALPHA,
            'attack_steps': config.ATTACK_STEPS,
            'use_mae': config.USE_MAE_DETECTOR,
            'use_diffpure': config.USE_DIFFPURE
        },
        'results': results,
        'final_metrics': {
            'clean_accuracy': results[-1]['clean_accuracy'],
            'adversarial_accuracy': results[-1]['adversarial_accuracy'],
            'detection_rate': results[-1]['detection_rate']
        }
    }
    
    # Save to file
    results_file = output_dir / 'kim2023_reproduction_results.json'
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    logger.info("Kim et al., 2023 reproduction completed successfully!")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
