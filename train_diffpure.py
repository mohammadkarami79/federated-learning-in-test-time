#!/usr/bin/env python3
"""
Train diffusion model for input purification
Updated to support different datasets with new config system
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
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sigma', type=float, default=0.04)
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

def train_epoch(model, train_loader, optimizer, device, sigma, scale_noise_by_t=True):
    """Train one epoch of diffusion purification model"""
    model.train()
    total_loss = 0
    loss_fn = nn.MSELoss()
    
    for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc='Training')):
        data = data.to(device)
        batch_size = data.size(0)
        
        # Sample random timesteps and reshape
        t = torch.rand(batch_size, device=device).view(-1, 1)
        
        # Add noise
        noise = torch.randn_like(data)
        if scale_noise_by_t:
            noisy_data = data + sigma * t.view(-1, 1, 1, 1) * noise
        else:
            noisy_data = data + sigma * noise
        
        # Predict noise
        pred_noise = model(noisy_data, t)
        loss = loss_fn(pred_noise, noise)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, test_loader, device, sigma, scale_noise_by_t=True):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    loss_fn = nn.MSELoss()
    
    with torch.no_grad():
        for data, _ in tqdm(test_loader, desc='Evaluating'):
            data = data.to(device)
            batch_size = data.size(0)
            
            # Random timestep
            t = torch.rand(batch_size, device=device).view(-1, 1)
            
            # Add noise
            noise = torch.randn_like(data)
            if scale_noise_by_t:
                noisy_data = data + sigma * t.view(-1, 1, 1, 1) * noise
            else:
                noisy_data = data + sigma * noise
            
            # Predict
            pred_noise = model(noisy_data, t)
            loss = loss_fn(pred_noise, noise)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)


def main():
    """Main training function"""
    logger = setup_logging()
    args = parse_args()
    cfg = get_config_for_dataset(args.dataset)
    logger.info(f"🌊 Training diffusion model for {cfg.DATASET_NAME}")
    logger.info(f"⚙️ Settings: {args.epochs} epochs, batch size {args.batch_size}, lr {args.lr}, sigma {args.sigma}")
    
    # Create directories
    checkpoints_dir = Path('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)
    
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
    
    # Create model with proper channels for dataset
    model = UNet(in_channels=cfg.IMG_CHANNELS, hidden_channels=64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
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
    
    logger.info(f"📊 Dataset: {cfg.DATASET_NAME}, Device: {device}")
    logger.info(f"🏗️ Model: UNet with {cfg.IMG_CHANNELS} input channels")
    
    # Training loop
    for epoch in range(n_epochs):
        logger.info(f"📈 Epoch {epoch + 1}/{n_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, sigma)
        logger.info(f"🏋️ Training Loss: {train_loss:.6f}")
        
        # Evaluate
        val_loss = evaluate(model, test_loader, device, sigma)
        logger.info(f"📊 Validation Loss: {val_loss:.6f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model with dataset-specific name
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = checkpoints_dir / f'diffuser_{cfg.DATASET.lower()}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"💾 Saved best model with loss: {best_loss:.6f} to {checkpoint_path}")
    
    logger.info("🎉 Training complete!")
    logger.info(f"🏆 Best validation loss: {best_loss:.6f}")
    
    return 0

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    set_seed(42)  # Set seed for reproducibility
    exit(main()) 