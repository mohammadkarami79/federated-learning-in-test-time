"""
DiffPure purification using DPM-Solver-2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels//2 * 2, out_channels)  # *2 because of concatenation

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handling different sizes
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64):
        super().__init__()
        self.inc = DoubleConv(in_channels, hidden_channels)
        self.down1 = Down(hidden_channels, hidden_channels*2)
        self.down2 = Down(hidden_channels*2, hidden_channels*4)
        self.down3 = Down(hidden_channels*4, hidden_channels*8)
        self.down4 = Down(hidden_channels*8, hidden_channels*16)
        # self.down5 = Down(hidden_channels*16, hidden_channels*32)
        # self.up0 = Up(hidden_channels*32, hidden_channels*16)
        self.up1 = Up(hidden_channels*16, hidden_channels*8)
        self.up2 = Up(hidden_channels*8, hidden_channels*4)
        self.up3 = Up(hidden_channels*4, hidden_channels*2)
        self.up4 = Up(hidden_channels*2, hidden_channels)
        self.outc = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

    def forward(self, x, t):
        # Time embedding
        t = t.view(-1, 1)
        t = self.time_embed(t)
        t = t.view(-1, t.shape[1], 1, 1)
        
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1 + t)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        
        return x
        
    def purify(self, x, steps=10, sigma=0.1):
        """Purify images using diffusion."""
        batch_size = x.shape[0]
        device = x.device
        
        # Add noise
        noise = torch.randn_like(x) * sigma
        noisy_x = x + noise
        
        # Denoise
        for i in range(steps):
            t = torch.ones(batch_size, device=device) * (steps - i) / steps
            pred_noise = self(noisy_x, t)
            noisy_x = noisy_x - sigma * pred_noise / steps
            
        return torch.clamp(noisy_x, 0, 1)

class DiffusionPurifier:
    """DPM-Solver-2 based diffusion purifier"""
    def __init__(self, sigma=0.04, steps=4):
        self.sigma = sigma
        self.steps = steps
        self.model = UNet(in_channels=3)  # Only image channels
    
    @torch.no_grad()
    def purify(self, x, steps=None, sigma=None):
        """
        Purify input images using DPM-Solver-2
        
        Args:
            x: Input images [B, C, H, W]
            steps: Number of denoising steps (optional)
            sigma: Noise level (optional)
        """
        steps = steps or self.steps
        sigma = sigma or self.sigma
        
        # Initialize
        x = x.clone()
        t = torch.linspace(1, 0, steps + 1, device=x.device)[:-1]
        h = 1.0 / steps
        
        # DPM-Solver-2 steps
        for i in range(steps):
            # Predict noise
            t_batch = t[i].expand(x.shape[0])  # Expand time step to batch size
            noise = self.model(x, t_batch)
            
            if i == steps - 1:
                # Last step: direct prediction
                x = x - noise * h
            else:
                # DPM-Solver-2 update
                x_tmp = x - noise * h
                t_batch_next = t[i+1].expand(x.shape[0])
                noise_tmp = self.model(x_tmp, t_batch_next)
                x = x - 0.5 * h * (noise + noise_tmp)
        
        return x

# Create global diffuser instance
diffuser = DiffusionPurifier() 