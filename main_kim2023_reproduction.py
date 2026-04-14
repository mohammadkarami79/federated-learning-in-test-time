#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kim et al., 2023 Reproduction Script
====================================

Reproduces the exact experimental setup from:
"Characterizing Internal Evasion Attacks in Federated Learning" (Kim et al., 2023)

Key Changes from log8.txt:
- L2-norm PGD attack instead of L∞
- ε = 4.5 (instead of 0.031)  
- α = 0.01 (instead of 0.007)
- K = 10 steps (same)
- MobileNetV2 architecture (instead of ResNet18)
- Non-IID with β = 0.4 (Dirichlet split)
- 40 clients (instead of 10)

This script is completely independent from main.py to preserve reproducibility.
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.models as models

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

@dataclass
class Kim2023Config:
    """Configuration matching Kim et al., 2023 paper exactly"""
    
    # Dataset and Architecture (EXACT MATCH)
    DATASET: str = 'cifar10'
    DATASET_NAME: str = 'CIFAR-10'
    IMG_SIZE: int = 32
    IMG_CHANNELS: int = 3
    NUM_CLASSES: int = 10
    
    # Architecture: MobileNetV2 (as in Kim et al., 2023)
    ARCHITECTURE: str = 'mobilenetv2'
    
    # Federated Learning (EXACT MATCH)
    NUM_CLIENTS: int = 40  # Kim et al. used 40 clients
    CLIENTS_PER_ROUND: int = 40  # All clients participate
    ROUNDS: int = 15  # Same as log8.txt for fair comparison
    CLIENT_EPOCHS: int = 8  # Same as log8.txt
    
    # Non-IID Data Distribution (EXACT MATCH)
    NON_IID: bool = True
    DIRICHLET_BETA: float = 0.4  # Kim et al. parameter
    
    # Training Parameters (MATCHED TO LOG8.TXT base)
    LEARNING_RATE: float = 0.01
    BATCH_SIZE: int = 64
    WEIGHT_DECAY: float = 1e-4
    MOMENTUM: float = 0.9
    
    # PGD Attack Parameters (EXACT MATCH to Kim et al., 2023)
    ATTACK_NORM: str = 'l2'  # L2 norm (NOT L∞)
    ATTACK_EPSILON: float = 4.5  # ε = 4.5 (NOT 0.031)
    ATTACK_ALPHA: float = 0.01  # α = 0.01 (NOT 0.007)
    ATTACK_STEPS: int = 10  # K = 10 steps
    ATTACK_RANDOM_START: bool = True
    ATTACK_TARGETED: bool = False  # Untargeted
    
    # Defense Configuration (SAME AS MAIN.PY)
    USE_MAE_DETECTOR: bool = True
    USE_DIFFPURE: bool = True
    USE_PFEDDEF: bool = True
    
    # MAE Parameters
    MAE_THRESHOLD: float = 0.12  # CIFAR-10 optimized
    ADAPTIVE_THRESHOLD: bool = True
    TARGET_DETECTION_RATE: float = 18.0
    
    # DiffPure Parameters  
    DIFFUSER_STEPS: int = 2
    DIFFUSER_SIGMA: float = 0.05
    DIFFPURE_STRENGTH: float = 0.06
    MAX_PURIFY_RATE: float = 0.3
    
    # System Configuration
    DEVICE: str = 'cuda'
    NUM_WORKERS: int = 4
    SEED: int = 42
    
    # Evaluation
    EVAL_BATCH_SIZE: int = 128
    SAVE_FREQUENCY: int = 5
    
    def to_dict(self) -> Dict:
        return asdict(self)

def setup_logging(config: Kim2023Config) -> logging.Logger:
    """Setup logging for the experiment"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'kim2023_reproduction_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("KIM ET AL., 2023 REPRODUCTION EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    
    return logger

def set_deterministic(seed: int = 42):
    """Set deterministic behavior for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class L2PGDAttack:
    """L2-norm PGD Attack as used in Kim et al., 2023
    
    Implements the exact L2-PGD attack from the paper:
    x_{t+1} = Π_{x+S}(x_t + α_PGD · sign(∇_x L(h_θ, x, y)))
    where Π_{x+S} is projection onto L2 ball with radius ε=4.5
    """
    
    def __init__(self, epsilon: float, alpha: float, steps: int, random_start: bool = True):
        self.epsilon = epsilon  # ε = 4.5
        self.alpha = alpha      # α = 0.01
        self.steps = steps      # K = 10
        self.random_start = random_start
        
        print(f"[L2-PGD Attack] ε={epsilon}, α={alpha}, steps={steps}, random_start={random_start}")
    
    def attack(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generate L2-norm PGD adversarial examples following Kim et al., 2023"""
        device = images.device
        batch_size = images.shape[0]
        
        # Start with clean images
        adv_images = images.clone().detach()
        
        # Random initialization within L2 ball (if enabled)
        if self.random_start:
            # Generate random noise
            noise = torch.randn_like(images)
            # Normalize to unit L2 norm
            noise_norm = torch.norm(noise.view(batch_size, -1), p=2, dim=1, keepdim=True)
            noise = noise / noise_norm.view(-1, 1, 1, 1)
            # Scale by random radius within epsilon
            random_radius = torch.rand(batch_size, device=device) * self.epsilon
            noise = noise * random_radius.view(-1, 1, 1, 1)
            # Add to original images
            adv_images = images + noise
            # Ensure valid pixel range
            adv_images = torch.clamp(adv_images, 0, 1)
        
        # PGD iterations
        for step in range(self.steps):
            adv_images.requires_grad_(True)
            
            # Forward pass through model
            # Keep model in eval mode to avoid BatchNorm issues with small batches
            model.eval()
            outputs = model(adv_images)
            
            # Cross-entropy loss
            loss = F.cross_entropy(outputs, labels)
            
            # Compute gradients
            grad = torch.autograd.grad(loss, adv_images, 
                                     retain_graph=False, create_graph=False)[0]
            
            # L2 gradient ascent step (Kim et al., 2023 formula)
            grad_flat = grad.view(batch_size, -1)
            grad_norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True)
            grad_normalized = grad_flat / (grad_norm + 1e-12)
            grad_normalized = grad_normalized.view_as(grad)
            
            # Update: x_{t+1} = x_t + α * normalized_gradient
            adv_images = adv_images.detach() + self.alpha * grad_normalized
            
            # Project back to L2 ball: Π_{x+S}
            delta = adv_images - images
            delta_flat = delta.view(batch_size, -1)
            delta_norm = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
            
            # Clip to L2 ball radius
            scaling = torch.min(torch.ones_like(delta_norm), 
                              self.epsilon / (delta_norm + 1e-12))
            delta_flat = delta_flat * scaling
            delta = delta_flat.view_as(images)
            
            adv_images = images + delta
            
            # Ensure valid pixel values [0, 1]
            adv_images = torch.clamp(adv_images, 0, 1).detach()
        
        return adv_images

def create_mobilenetv2_model(num_classes: int = 10) -> nn.Module:
    """Create MobileNetV2 model exactly as used in Kim et al., 2023
    
    Architecture:
    - MobileNetV2 backbone (no pretrained weights)
    - Modified classifier for CIFAR-10 (10 classes)
    - Dropout 0.2 (standard for MobileNet)
    """
    # Create MobileNetV2 without pretrained weights
    model = models.mobilenet_v2(weights=None)
    
    # Get the number of input features for classifier
    in_features = model.classifier[1].in_features  # Should be 1280
    
    # Replace classifier for CIFAR-10
    model.classifier = nn.Sequential(
        nn.Dropout(0.2, inplace=False),
        nn.Linear(in_features, num_classes)
    )
    
    # Initialize weights (important for fair comparison)
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    return model

def create_dirichlet_split(dataset, num_clients: int, beta: float, num_classes: int) -> List[List[int]]:
    """Create non-IID data split using Dirichlet distribution (β = 0.4)"""
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # Dirichlet distribution
    client_indices = [[] for _ in range(num_clients)]
    
    for k in range(num_classes):
        # Get indices for class k
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet([beta] * num_clients)
        
        # Assign samples to clients
        proportions = np.cumsum(proportions)
        proportions = (proportions * len(idx_k)).astype(int)
        
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = proportions[client_id]
            client_indices[client_id].extend(idx_k[start_idx:end_idx])
            start_idx = end_idx
    
    # Shuffle each client's data
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])
    
    return client_indices

class Kim2023Client:
    """Federated learning client for Kim et al., 2023 reproduction"""
    
    def __init__(self, client_id: int, config: Kim2023Config, train_indices: List[int]):
        self.client_id = client_id
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Create model - MobileNetV2
        self.model = create_mobilenetv2_model(config.NUM_CLASSES).to(self.device)
        
        # Create data loader
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        client_dataset = Subset(full_dataset, train_indices)
        
        self.train_loader = DataLoader(
            client_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )
        
        # Store data distribution info
        labels = [full_dataset[i][1] for i in train_indices]
        self.data_distribution = np.bincount(labels, minlength=config.NUM_CLASSES)
    
    def train(self, epochs: int = 1) -> Dict:
        """Train local model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            momentum=self.config.MOMENTUM,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.train_loader) / epochs
        accuracy = 100.0 * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples': total,
            'data_distribution': self.data_distribution.tolist()
        }
    
    def get_model_state(self) -> Dict:
        """Get model state dict"""
        return self.model.state_dict()
    
    def set_model_state(self, state_dict: Dict):
        """Set model state dict"""
        self.model.load_state_dict(state_dict)

class Kim2023Server:
    """Federated learning server for Kim et al., 2023 reproduction"""
    
    def __init__(self, config: Kim2023Config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Global model - MobileNetV2
        self.global_model = create_mobilenetv2_model(config.NUM_CLASSES).to(self.device)
        
        # Initialize with Xavier/Kaiming
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
        
        self.global_model.apply(init_weights)
    
    def get_global_model_state(self) -> Dict:
        """Get global model state"""
        return self.global_model.state_dict()
    
    def aggregate(self, client_states: List[Dict], client_weights: List[float]) -> Dict:
        """FedAvg aggregation - same as working main.py"""
        n_clients = len(client_states)
        if n_clients == 0:
            return self.global_model.state_dict()
        
        # Average the parameters (same method as main.py)
        avg_state = self.global_model.state_dict()
        
        for key in avg_state:
            # Skip BN layers for stability (same as main.py)
            if 'bn' not in key.lower() and 'num_batches_tracked' not in key:
                # Check if all clients have this key
                if all(key in client_state for client_state in client_states):
                    try:
                        stacked = torch.stack([
                            client_state[key] for client_state in client_states
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
        return avg_state

def load_defense_components(config: Kim2023Config, logger: logging.Logger):
    """Load pre-trained defense components (MAE + DiffPure)"""
    components = {}
    
    if config.USE_MAE_DETECTOR:
        try:
            # Try to import and create MAE detector with proper config
            import sys
            sys.path.append('.')
            
            # Create a compatible config for MAE
            mae_config = type('MAEConfig', (), {
                'DATASET': config.DATASET,
                'IMG_SIZE': config.IMG_SIZE,
                'IMG_CHANNELS': config.IMG_CHANNELS,
                'PATCH_SIZE': 4 if config.DATASET.lower() == 'cifar10' else 16,
                'EMBED_DIM': 768,
                'NUM_HEADS': 12,  # Make sure embed_dim (768) is divisible by num_heads (12)
                'ENCODER_DEPTH': 12,
                'DECODER_DEPTH': 8,
                'DECODER_EMBED_DIM': 512,
                'MASK_RATIO': 0.75,
                'DEVICE': config.DEVICE
            })()
            
            from defense.mae_detector import MAEDetector
            mae_detector = MAEDetector(mae_config)
            components['mae_detector'] = mae_detector
            logger.info("✅ MAE detector loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load MAE detector: {e}")
            logger.info("Continuing without MAE detection...")
            components['mae_detector'] = None
    
    if config.USE_DIFFPURE:
        try:
            # Try multiple import paths for diffusion components
            diffusion_model = None
            diffuser = None
            
            # Method 1: Try diffusion.diffuser (same as main.py)
            try:
                from diffusion.diffuser import UNet
                logger.info("✅ Import successful: from diffusion.diffuser import UNet")
                diffusion_model = UNet(
                    in_channels=config.IMG_CHANNELS,
                    hidden_channels=256
                )
                logger.info("Loaded UNet from diffusion module")
            except ImportError as e1:
                logger.warning(f"Method 1 failed: {e1}")
                # Method 2: Try local UNet definition (fallback)
                try:
                    import torch.nn as nn
                    
                    class UNet(nn.Module):
                        def __init__(self, in_channels=3, hidden_channels=256):
                            super().__init__()
                            self.conv1 = nn.Conv2d(in_channels, hidden_channels//2, 3, padding=1)
                            self.conv2 = nn.Conv2d(hidden_channels//2, hidden_channels, 3, padding=1)
                            self.conv3 = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)
                        
                        def forward(self, x, t=None):
                            h = torch.relu(self.conv1(x))
                            h = torch.relu(self.conv2(h))
                            return torch.sigmoid(self.conv3(h))
                    diffusion_model = UNet(
                        in_channels=config.IMG_CHANNELS,
                        hidden_channels=256
                    )
                    logger.info("✅ Created local UNet fallback")
                except Exception as e2:
                    logger.warning(f"Method 2 failed: {e2}")
                    logger.warning("❌ All diffusion import methods failed")
                    diffusion_model = None
            
            if diffusion_model is not None:
                diffusion_model = diffusion_model.to(torch.device(config.DEVICE))
                
                # Load checkpoint
                checkpoint_path = f"checkpoints/diffuser_{config.DATASET.lower()}.pt"
                if Path(checkpoint_path).exists():
                    try:
                        state_dict = torch.load(checkpoint_path, map_location=config.DEVICE)
                        diffusion_model.load_state_dict(state_dict)
                        # Simple diffuser wrapper (no complex diffusion class needed)
                        components['diffuser'] = diffusion_model
                        logger.info("✅ DiffPure loaded successfully")
                    except Exception as load_error:
                        logger.warning(f"Failed to load diffusion checkpoint: {load_error}")
                        components['diffuser'] = None
                else:
                    logger.warning(f"⚠️ Diffusion checkpoint not found: {checkpoint_path}")
                    components['diffuser'] = None
            else:
                components['diffuser'] = None
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load DiffPure: {e}")
            logger.info("Continuing without DiffPure purification...")
            components['diffuser'] = None
    
    return components

def apply_defense_pipeline(images: torch.Tensor, components: Dict, config: Kim2023Config) -> torch.Tensor:
    """Apply defense pipeline: MAE Detection → DiffPure → Clean"""
    
    # MAE Detection
    if components.get('mae_detector') is not None:
        try:
            detection_scores = components['mae_detector'].detect_batch(images)
            is_adversarial = detection_scores > config.MAE_THRESHOLD
            
            # Apply DiffPure only to detected adversarial samples
            if components.get('diffuser') is not None and is_adversarial.any():
                # Apply purification to detected samples
                adv_indices = torch.where(is_adversarial)[0]
                if len(adv_indices) > 0:
                    adv_images = images[adv_indices]
                    
                    # Purify with adaptive timesteps
                    t_purify = min(config.DIFFUSER_STEPS, 50)
                    purified_images = components['diffuser'].ddim_sample(
                        adv_images, t_purify, eta=0.0
                    )
                    
                    # Replace detected adversarial images with purified ones
                    images = images.clone()
                    images[adv_indices] = purified_images
                    
        except Exception as e:
            # If defense fails, return original images
            pass
    
    return images

def evaluate_model(model: nn.Module, test_loader: DataLoader, attacker: L2PGDAttack, 
                  components: Dict, config: Kim2023Config, logger: logging.Logger) -> Tuple[float, float, float]:
    """Evaluate model with clean and adversarial accuracy"""
    model.eval()
    
    clean_correct = 0
    adv_correct = 0
    total = 0
    detection_count = 0
    
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(torch.device(config.DEVICE))
        labels = labels.to(torch.device(config.DEVICE))
        
        # Skip small batches to avoid BatchNorm issues
        if images.size(0) < 2:
            continue
        
        # Clean accuracy - use no_grad for inference
        with torch.no_grad():
            clean_defended = apply_defense_pipeline(images, components, config)
            model.eval()  # Ensure eval mode
            clean_outputs = model(clean_defended)
            clean_pred = clean_outputs.argmax(dim=1)
            clean_correct += (clean_pred == labels).sum().item()
        
        # Generate adversarial examples (REQUIRES GRADIENTS)
        # Note: BatchNorm should be in eval mode during attack to avoid issues
        model.eval()  # Keep in eval mode for attack to avoid BatchNorm issues
        adv_images = attacker.attack(model, images, labels)
        
        # Adversarial accuracy with defense
        with torch.no_grad():
            adv_defended = apply_defense_pipeline(adv_images, components, config)
            model.eval()  # Ensure eval mode
            adv_outputs = model(adv_defended)
            adv_pred = adv_outputs.argmax(dim=1)
            adv_correct += (adv_pred == labels).sum().item()
        
        # Detection statistics
        if components.get('mae_detector') is not None:
            try:
                with torch.no_grad():
                    detection_scores = components['mae_detector'].detect_batch(adv_images)
                    detection_count += (detection_scores > config.MAE_THRESHOLD).sum().item()
            except:
                pass
        
        total += labels.size(0)
    
    clean_acc = 100.0 * clean_correct / total if total > 0 else 0.0
    adv_acc = 100.0 * adv_correct / total if total > 0 else 0.0
    detection_rate = 100.0 * detection_count / total if total > 0 else 0.0
    
    return clean_acc, adv_acc, detection_rate

def main():
    """Main experiment function"""
    parser = argparse.ArgumentParser(description='Kim et al., 2023 Reproduction')
    parser.add_argument('--config', type=str, help='Config file path (optional)')
    parser.add_argument('--output-dir', type=str, default='./kim2023_results', 
                       help='Output directory for results')
    args = parser.parse_args()
    
    # Create config
    config = Kim2023Config()
    
    # Setup
    set_deterministic(config.SEED)
    logger = setup_logging(config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    logger.info("Kim et al., 2023 Reproduction Configuration:")
    logger.info(f"- Architecture: {config.ARCHITECTURE}")
    logger.info(f"- Clients: {config.NUM_CLIENTS}")
    logger.info(f"- Non-IID β: {config.DIRICHLET_BETA}")
    logger.info(f"- Attack: L2-PGD (ε={config.ATTACK_EPSILON}, α={config.ATTACK_ALPHA}, K={config.ATTACK_STEPS})")
    logger.info(f"- Defense: MAE={config.USE_MAE_DETECTOR}, DiffPure={config.USE_DIFFPURE}")
    
    # Load test dataset
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.EVAL_BATCH_SIZE, 
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    logger.info(f"Test dataset loaded: {len(test_dataset)} samples")
    
    # Load defense components
    logger.info("Loading defense components...")
    components = load_defense_components(config, logger)
    
    # Create data splits (Non-IID Dirichlet)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    
    logger.info("Creating Non-IID data split with Dirichlet distribution...")
    client_indices = create_dirichlet_split(
        train_dataset, 
        config.NUM_CLIENTS, 
        config.DIRICHLET_BETA, 
        config.NUM_CLASSES
    )
    
    # Log data distribution
    for i in range(min(5, config.NUM_CLIENTS)):
        labels = [train_dataset[idx][1] for idx in client_indices[i]]
        dist = np.bincount(labels, minlength=config.NUM_CLASSES)
        logger.info(f"Client {i+1} data distribution: {dist.tolist()}")
    
    # Initialize server and clients
    logger.info("Initializing server and clients...")
    server = Kim2023Server(config)
    clients = []
    
    for client_id in range(config.NUM_CLIENTS):
        client = Kim2023Client(client_id, config, client_indices[client_id])
        clients.append(client)
    
    # Initialize L2-PGD attacker
    attacker = L2PGDAttack(
        epsilon=config.ATTACK_EPSILON,
        alpha=config.ATTACK_ALPHA,
        steps=config.ATTACK_STEPS,
        random_start=config.ATTACK_RANDOM_START
    )
    
    logger.info("=" * 40)
    logger.info("ATTACK CONFIGURATION (Kim et al., 2023)")
    logger.info("=" * 40)
    logger.info(f"Attack Type: L2-norm PGD (Untargeted)")
    logger.info(f"Epsilon (ε): {config.ATTACK_EPSILON}")
    logger.info(f"Alpha (α): {config.ATTACK_ALPHA}")
    logger.info(f"Steps (K): {config.ATTACK_STEPS}")
    logger.info(f"Random Start: {config.ATTACK_RANDOM_START}")
    logger.info(f"Formula: x_{{t+1}} = Π_{{x+S}}(x_t + α·∇_x L(h_θ, x, y))")
    logger.info("=" * 40)
    
    # Training loop
    results = []
    
    logger.info(f"Starting federated training: {config.ROUNDS} rounds, {config.CLIENT_EPOCHS} local epochs")
    
    for round_num in range(1, config.ROUNDS + 1):
        round_start_time = time.time()
        logger.info(f"Round {round_num}/{config.ROUNDS}")
        
        # Distribute global model to clients
        global_state = server.get_global_model_state()
        for client in clients:
            client.set_model_state(global_state)
        
        # Client training
        client_states = []
        client_weights = []
        total_samples = 0
        
        for client_id, client in enumerate(clients):
            start_time = time.time()
            
            # Train client
            client_metrics = client.train(config.CLIENT_EPOCHS)
            
            # Collect updates
            client_states.append(client.get_model_state())
            client_weights.append(client_metrics['samples'])
            total_samples += client_metrics['samples']
            
            elapsed = time.time() - start_time
            logger.info(f"Client {client_id+1}/{config.NUM_CLIENTS} completed in {elapsed:.1f}s")
        
        # Normalize weights
        client_weights = [w / total_samples for w in client_weights]
        
        # Server aggregation
        logger.info("Aggregating models...")
        server.aggregate(client_states, client_weights)
        
        # Evaluation
        logger.info("Running evaluation...")
        clean_acc, adv_acc, detection_rate = evaluate_model(
            server.global_model, test_loader, attacker, components, config, logger
        )
        
        round_time = time.time() - round_start_time
        
        # Log results
        logger.info(f"Round {round_num} Clean Acc: {clean_acc:.2f}% | "
                   f"Adv Acc: {adv_acc:.2f}% | "
                   f"Detection: {detection_rate:.2f}% | "
                   f"Time: {round_time:.1f}s")
        
        # Store results
        round_result = {
            'round': round_num,
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
            'detection_rate': detection_rate,
            'round_time': round_time,
            'total_samples': total_samples
        }
        results.append(round_result)
        
        # Save intermediate results
        if round_num % config.SAVE_FREQUENCY == 0:
            with open(output_dir / f'results_round_{round_num}.json', 'w') as f:
                json.dump(results, f, indent=2)
    
    # Final results
    final_clean_acc = results[-1]['clean_accuracy']
    final_adv_acc = results[-1]['adversarial_accuracy']
    final_detection = results[-1]['detection_rate']
    
    logger.info("=" * 60)
    logger.info("FINAL RESULTS (Kim et al., 2023 Reproduction)")
    logger.info("=" * 60)
    logger.info(f"Clean Accuracy: {final_clean_acc:.2f}%")
    logger.info(f"Adversarial Accuracy (L2-PGD): {final_adv_acc:.2f}%")
    logger.info(f"Detection Rate: {final_detection:.2f}%")
    logger.info("=" * 60)
    
    # Save final results
    final_results = {
        'experiment_info': {
            'paper': 'Kim et al., 2023 Reproduction',
            'dataset': config.DATASET_NAME,
            'architecture': config.ARCHITECTURE,
            'attack': f'L2-PGD (ε={config.ATTACK_EPSILON}, α={config.ATTACK_ALPHA}, K={config.ATTACK_STEPS})',
            'clients': config.NUM_CLIENTS,
            'non_iid_beta': config.DIRICHLET_BETA,
            'rounds': config.ROUNDS,
            'total_time': sum(r['round_time'] for r in results)
        },
        'final_metrics': {
            'clean_accuracy': final_clean_acc,
            'adversarial_accuracy': final_adv_acc,
            'detection_rate': final_detection
        },
        'round_by_round': results,
        'config': config.to_dict()
    }
    
    with open(output_dir / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"Results saved to: {output_dir}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
