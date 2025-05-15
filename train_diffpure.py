"""
Train DiffPure model for input purification
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import time
from tqdm import tqdm

from config import get_config, DEVICE, parse_args
from diffusion.diffuser import UNet
from utils.data_utils import get_dataloader

def train_epoch(model, train_loader, optimizer, device, sigma):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc='Training')):
        data = data.to(device)
        batch_size = data.shape[0]
        
        # Sample random timesteps
        t = torch.rand(batch_size, device=device)
        
        # Add noise
        noise = torch.randn_like(data) * sigma
        noisy_data = data + noise
        
        # Forward pass
        pred_noise = model(noisy_data, t)
        loss = nn.MSELoss()(pred_noise, noise)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, test_loader, device, sigma):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, _ in tqdm(test_loader, desc='Evaluating'):
            data = data.to(device)
            batch_size = data.shape[0]
            
            # Sample random timesteps
            t = torch.rand(batch_size, device=device)
            
            # Add noise
            noise = torch.randn_like(data) * sigma
            noisy_data = data + noise
            
            # Forward pass
            pred_noise = model(noisy_data, t)
            loss = nn.MSELoss()(pred_noise, noise)
            
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

def main():
    # Parse arguments and get configuration
    args = parse_args()
    cfg = get_config(args.preset or 'debug')
    
    # Create directories
    checkpoints_dir = Path('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Setup data using unified dataloader
    train_loader = get_dataloader(cfg, split="train")
    test_loader = get_dataloader(cfg, split="test")
    
    # Create model
    model = UNet(in_channels=3, hidden_channels=64).to(DEVICE)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training parameters
    n_epochs = 50  # Adjust based on convergence
    sigma = cfg.DIFFUSER_SIGMA
    best_loss = float('inf')
    
    print(f"Training DiffPure model with sigma={sigma}")
    print(f"Dataset: {cfg.DATASET_NAME}")
    print(f"Device: {DEVICE}")
    
    # Training loop
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE, sigma)
        print(f"Training Loss: {train_loss:.6f}")
        
        # Evaluate
        val_loss = evaluate(model, test_loader, DEVICE, sigma)
        print(f"Validation Loss: {val_loss:.6f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = checkpoints_dir / 'diffuser.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model with loss: {best_loss:.6f}")
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Model saved to: {checkpoints_dir / 'diffuser.pt'}")

if __name__ == '__main__':
    main() 