#!/usr/bin/env python3
"""
Train diffusion model for input purification
Updated to support different datasets with new config system
FIXED: Added checkpoint resume, final save, configurable parameters, and pre-trained support
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from pathlib import Path
import time
from tqdm import tqdm
import logging
import random
import numpy as np
import torch.nn.functional as F

from diffusion.diffuser import UNet
from utils.data_utils import get_dataset
import torch.utils.data as data_utils

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train diffusion Model')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100', 'mnist', 'br35h'])
    parser.add_argument('--epochs', type=int, default=50)  # PROFESSIONAL: Increased for paper quality
    parser.add_argument('--batch-size', type=int, default=64)  # REDUCED: For memory compatibility
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sigma', type=float, default=0.04)
    parser.add_argument('--hidden-channels', type=int, default=256,  # REDUCED: For memory compatibility 
                       help='Number of hidden channels in UNet')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pre-trained model to load')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--save-config', action='store_true',
                       help='Save training configuration to disk')
    return parser.parse_args()

def get_config_for_dataset(dataset):
    """Get configuration for dataset"""
    from config_fixed import get_debug_config
    cfg = get_debug_config()
    
    if dataset == 'mnist':
        cfg.IMG_CHANNELS = 1
        cfg.IMG_SIZE = 28
        cfg.N_CLASSES = 10
    elif dataset == 'cifar100':
        cfg.N_CLASSES = 100
        cfg.IMG_CHANNELS = 3
        cfg.IMG_SIZE = 32
    elif dataset == 'br35h':
        cfg.IMG_SIZE = 224
        cfg.IMG_CHANNELS = 3
        cfg.N_CLASSES = 2
    else:  # cifar10
        cfg.N_CLASSES = 10
        cfg.IMG_CHANNELS = 3
        cfg.IMG_SIZE = 32
    
    cfg.DATASET = dataset
    cfg.DATASET_NAME = dataset.upper()
    
    return cfg

def get_medical_config():
    """Get configuration optimized for medical images"""
    cfg = get_config_for_dataset('br35h')
    cfg.LINEAR_SCHEDULING = True
    cfg.RICIAN_NOISE = True
    cfg.ANATOMICAL_CONSTRAINTS = True
    cfg.MEDICAL_AUGMENTATION = True
    return cfg

def add_medical_noise(x, rician=True):
    """Add noise suitable for medical images"""
    if rician:
        # Rician noise for medical images
        noise_real = torch.randn_like(x)
        noise_imag = torch.randn_like(x)
        noise = torch.sqrt(noise_real**2 + noise_imag**2)
        return x + noise * 0.1
    else:
        # Standard Gaussian noise
        return x + torch.randn_like(x) * 0.1

def apply_anatomical_constraints(x, mask=None):
    """Apply anatomical constraints for medical images"""
    if mask is not None:
        # Apply mask to preserve anatomical structure
        x = x * mask
    return x

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_config(args, cfg, save_path):
    """Save training configuration to disk for experiment reproducibility"""
    import json
    import sys
    
    config_dict = {
        'args': vars(args),
        'config': {
            'DATASET': cfg.DATASET,
            'DATASET_NAME': cfg.DATASET_NAME,
            'IMG_CHANNELS': cfg.IMG_CHANNELS,
            'IMG_SIZE': cfg.IMG_SIZE,
            'N_CLASSES': cfg.N_CLASSES,
            'HIDDEN_CHANNELS': args.hidden_channels
        },
        'system_info': {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        },
        'timestamp': time.time()
    }
    
    # Create directory if it doesn't exist
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    print(f"Configuration saved to {save_path}")
    print(f"Configuration includes: {len(config_dict['args'])} args, {len(config_dict['config'])} config params")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load checkpoint for resuming training"""
    if not checkpoint_path.exists():
        return 0, float('inf')
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    
    return start_epoch, best_loss

def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, save_path):
    """Save checkpoint for resuming training"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss
    }
    torch.save(checkpoint, save_path)

def train_epoch(model, train_loader, optimizer, device, cfg, epoch):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        batch_size = data.shape[0]
        t = torch.rand(batch_size, device=device)
        
        # Add noise based on configuration
        if getattr(cfg, 'RICIAN_NOISE', False):
            noisy_data = add_medical_noise(data, rician=True)
        else:
            noise = torch.randn_like(data) * 0.1
            noisy_data = data + noise
        
        # Apply anatomical constraints if enabled
        if getattr(cfg, 'ANATOMICAL_CONSTRAINTS', False):
            noisy_data = apply_anatomical_constraints(noisy_data)
        
        # Predict noise
        predicted_noise = model(noisy_data, t)
        
        # Compute loss
        if getattr(cfg, 'RICIAN_NOISE', False):
            # Rician noise loss
            loss = F.mse_loss(predicted_noise, (data - noisy_data))
        else:
            # Standard MSE loss
            loss = F.mse_loss(predicted_noise, (data - noisy_data))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
    
    return total_loss / len(train_loader)

def evaluate(model, test_loader, device, cfg):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.shape[0]
            t = torch.rand(batch_size, device=device)
            
            # Add noise based on configuration
            if getattr(cfg, 'RICIAN_NOISE', False):
                noisy_data = add_medical_noise(data, rician=True)
            else:
                noise = torch.randn_like(data) * 0.1
                noisy_data = data + noise
            
            # Apply anatomical constraints if enabled
            if getattr(cfg, 'ANATOMICAL_CONSTRAINTS', False):
                noisy_data = apply_anatomical_constraints(noisy_data)
            
            predicted_noise = model(noisy_data, t)
            
            if getattr(cfg, 'RICIAN_NOISE', False):
                loss = F.mse_loss(predicted_noise, (data - noisy_data))
            else:
                loss = F.mse_loss(predicted_noise, (data - noisy_data))
            
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

def load_pretrained_model(model, pretrained_path, enable_fine_tuning=False):
    """Load a pretrained model for fine-tuning"""
    if pretrained_path and Path(pretrained_path).exists():
        logger.info(f"Loading pretrained model from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        
        if enable_fine_tuning:
            model.enable_fine_tuning()
            logger.info("Enabled fine-tuning mode - early layers frozen")
        else:
            model.disable_fine_tuning()
            logger.info("Disabled fine-tuning mode - all layers trainable")
            
        return True
    else:
        logger.warning(f"Pretrained model not found at {pretrained_path}")
        return False

def main():
    """Main training function"""
    logger = setup_logging()
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Get configuration
    if args.dataset.lower() == 'br35h':
        cfg = get_medical_config()
    else:
        cfg = get_config_for_dataset(args.dataset)
    
    logger.info(f"Training diffusion model for {cfg.DATASET_NAME}")
    logger.info(f"Settings: {args.epochs} epochs, batch size {args.batch_size}, lr {args.lr}, sigma {args.sigma}")
    logger.info(f"Hidden channels: {args.hidden_channels}, Seed: {args.seed}")
    
    # Create directories
    checkpoints_dir = Path('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Save configuration if requested
    if args.save_config:
        config_save_path = checkpoints_dir / f'diffuser_{cfg.DATASET.lower()}_config.json'
        save_config(args, cfg, config_save_path)
        logger.info(f"Configuration saved to {config_save_path}")
    
    # Setup data using new dataset loading
    train_dataset, test_dataset = get_dataset(cfg)
    train_loader = data_utils.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    test_loader = data_utils.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    # Create model with configurable hidden channels
    model = UNet(in_channels=cfg.IMG_CHANNELS, 
                 hidden_channels=args.hidden_channels,
                 use_additional_layers=getattr(cfg, 'USE_ADDITIONAL_LAYERS', False))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load pre-trained model if specified
    if args.pretrained:
        load_pretrained_model(model, args.pretrained, enable_fine_tuning=True)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Training parameters
    n_epochs = args.epochs
    sigma = args.sigma
    best_loss = float('inf')
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            start_epoch, best_loss = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
            logger.info(f"Resumed training from epoch {start_epoch} with best loss {best_loss:.6f}")
        else:
            logger.warning(f"Checkpoint not found: {args.resume}")
    
    logger.info(f"Dataset: {cfg.DATASET_NAME}, Device: {device}")
    logger.info(f"Model: UNet with {cfg.IMG_CHANNELS} input channels, {args.hidden_channels} hidden channels")
    
    # Training loop
    for epoch in range(start_epoch, n_epochs):
        logger.info(f"Epoch {epoch + 1}/{n_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, cfg, epoch)
        logger.info(f"Training Loss: {train_loss:.6f}")
        
        # Evaluate
        val_loss = evaluate(model, test_loader, device, cfg)
        logger.info(f"Validation Loss: {val_loss:.6f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save checkpoint for resuming
        checkpoint_path = checkpoints_dir / f'diffuser_{cfg.DATASET.lower()}_checkpoint.pt'
        save_checkpoint(model, optimizer, scheduler, epoch + 1, best_loss, checkpoint_path)
        
        # Save best model with dataset-specific name
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_path = checkpoints_dir / f'diffuser_{cfg.DATASET.lower()}.pt'
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model with loss: {best_loss:.6f} to {best_model_path}")
    
    # Save final model (FIXED: Issue 8)
    final_model_path = checkpoints_dir / f'diffuser_{cfg.DATASET.lower()}_final.pt'
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_loss:.6f}")
    
    return 0

if __name__ == '__main__':
    exit(main()) 