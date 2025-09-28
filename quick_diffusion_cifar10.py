#!/usr/bin/env python3
"""
Quick Diffusion Model Training for CIFAR-10 (Fast Version)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import logging
from tqdm import tqdm
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleDiffusionUNet(nn.Module):
    """Lightweight U-Net for fast diffusion training"""
    def __init__(self, in_channels=3, hidden_channels=64):  # Reduced from 128
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck (simplified)
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels * 2, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(hidden_channels, hidden_channels, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Conv2d(hidden_channels, in_channels, 1)
        
    def forward(self, x, t):
        # Simple time embedding
        t = t.view(-1, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3)) / 1000.0
        x = torch.cat([x, t], dim=1)
        
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc2)
        
        # Decoder
        up1 = self.up1(bottleneck)
        dec1 = self.dec1(torch.cat([up1, enc2], dim=1))
        
        up2 = self.up2(dec1)
        dec2 = self.dec2(torch.cat([up2, enc1], dim=1))
        
        return self.final(dec2)

def quick_train_diffusion():
    """Quick diffusion training with optimized parameters"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # FAST parameters
    epochs = 50          # Reduced from 150
    batch_size = 64      # Increased from 32
    learning_rate = 2e-4 # Increased learning rate
    noise_steps = 500    # Reduced from 1000
    
    logger.info(f"🚀 FAST Training: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load CIFAR-10 dataset (subset for speed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    full_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Use subset for faster training (optional)
    subset_size = min(25000, len(full_dataset))  # Use half dataset
    subset_indices = torch.randperm(len(full_dataset))[:subset_size]
    dataset = torch.utils.data.Subset(full_dataset, subset_indices)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    logger.info(f"Dataset loaded: {len(dataset)} samples (subset for speed)")
    
    # Initialize lightweight model
    model = SimpleDiffusionUNet(in_channels=3, hidden_channels=64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    logger.info("Lightweight model initialized")
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(device)
            batch_size_current = data.size(0)
            
            # Sample random noise levels
            t = torch.randint(0, noise_steps, (batch_size_current,), device=device).float()
            
            # Add noise to images (simplified noise schedule)
            noise = torch.randn_like(data)
            alpha = (1 - t / noise_steps).view(-1, 1, 1, 1)
            noisy_data = torch.sqrt(alpha) * data + torch.sqrt(1 - alpha) * noise
            
            # Predict noise
            optimizer.zero_grad()
            predicted_noise = model(noisy_data, t)
            loss = criterion(predicted_noise, noise)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'LR': f'{scheduler.get_last_lr()[0]:.6f}'})
        
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            temp_path = f'checkpoints/diffuser_cifar10_quick_epoch_{epoch+1}.pt'
            torch.save(model.state_dict(), temp_path)
            logger.info(f"Quick checkpoint saved: {temp_path}")
    
    # Save final model
    final_path = 'checkpoints/diffuser_cifar10.pt'
    torch.save(model.state_dict(), final_path)
    logger.info(f"✅ Quick diffusion model completed: {final_path}")
    
    return True

if __name__ == "__main__":
    try:
        logger.info("🚀 Starting QUICK diffusion training for CIFAR-10...")
        success = quick_train_diffusion()
        if success:
            logger.info("🎉 Quick diffusion training completed successfully!")
        else:
            logger.error("❌ Quick diffusion training failed!")
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        raise
