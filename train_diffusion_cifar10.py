#!/usr/bin/env python3
"""
Standalone Diffusion Model Training for CIFAR-10
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

class UNet(nn.Module):
    """Simple U-Net for diffusion"""
    def __init__(self, in_channels=3, hidden_channels=128):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden_channels, 3, padding=1),  # +1 for time embedding
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU()
        )
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1),
            nn.ReLU()
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, 3, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1),
            nn.ReLU()
        )
        
        self.up2 = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU()
        )
        
        self.final = nn.Conv2d(hidden_channels, in_channels, 1)
        
    def forward(self, x, t):
        # Simple time embedding
        t = t.view(-1, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))
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

def train_diffusion_model():
    """Train diffusion model for CIFAR-10"""
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # CIFAR-10 optimized parameters (faster training)
    epochs = 100  # Reduced from 150 for faster training
    batch_size = 64  # Increased for faster training
    learning_rate = 2e-4  # Slightly higher LR
    noise_steps = 1000
    
    logger.info(f"Training parameters: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Check if model already exists
    checkpoint_path = Path('checkpoints/diffuser_cifar10.pt')
    if checkpoint_path.exists():
        logger.info(f"Diffusion model already exists: {checkpoint_path}")
        return True
    
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Initialize model
    model = UNet(in_channels=3, hidden_channels=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    logger.info("Model initialized")
    
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
            
            # Add noise to images
            noise = torch.randn_like(data)
            alpha = (1 - t / noise_steps).view(-1, 1, 1, 1)
            noisy_data = alpha * data + (1 - alpha) * noise
            
            # Predict noise
            optimizer.zero_grad()
            predicted_noise = model(noisy_data, t)
            loss = criterion(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            temp_path = f'checkpoints/diffuser_cifar10_epoch_{epoch+1}.pt'
            torch.save(model.state_dict(), temp_path)
            logger.info(f"Checkpoint saved: {temp_path}")
    
    # Save final model
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"✅ Diffusion model training completed and saved: {checkpoint_path}")
    
    return True

if __name__ == "__main__":
    try:
        success = train_diffusion_model()
        if success:
            logger.info("🎉 Diffusion training completed successfully!")
        else:
            logger.error("❌ Diffusion training failed!")
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        raise
