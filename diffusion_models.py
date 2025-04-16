import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class DiffusionModel(nn.Module):
    """
    A simplified diffusion model for adversarial purification based on VP-SDE as described in DiffPure paper.
    
    This model adds noise to adversarial images following the forward diffusion process, 
    and then recovers clean images through the reverse generative process.
    """
    def __init__(self, 
                 model, 
                 beta_min=0.1, 
                 beta_max=20.0, 
                 device='cpu'):
        """
        Initialize the diffusion model.
        
        Args:
            model: The neural network model that predicts the noise (score function)
            beta_min: Minimum noise level
            beta_max: Maximum noise level
            device: Device to run the model on ('cpu' or 'cuda')
        """
        super().__init__()
        self.model = model  # Score model (predicts noise)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.device = device
        
    def beta_t(self, t):
        """Linear noise schedule"""
        return self.beta_min + (self.beta_max - self.beta_min) * t
    
    def alpha_t(self, t):
        """Compute alpha_t from beta_t"""
        return torch.exp(-0.5 * (t * self.beta_min + 0.5 * t**2 * (self.beta_max - self.beta_min)))
    
    def forward_diffusion(self, x_0, t_star):
        """
        Forward diffusion process: add noise to x_0 up to timestep t_star.
        
        Args:
            x_0: Original image (adversarial example in our case)
            t_star: Target diffusion timestep
            
        Returns:
            x_t: Diffused image at timestep t_star
        """
        # Calculate alpha at timestep t_star
        alpha_t_star = self.alpha_t(t_star)
        
        # Sample noise
        epsilon = torch.randn_like(x_0)
        
        # Forward process (diffuse the image)
        x_t = alpha_t_star * x_0 + torch.sqrt(1 - alpha_t_star**2) * epsilon
        
        return x_t
    
    def reverse_diffusion(self, x_t, t_star, n_steps=100):
        """
        Reverse diffusion process: recover original image from x_t.
        
        Args:
            x_t: Diffused image at timestep t_star
            t_star: Starting diffusion timestep
            n_steps: Number of discretization steps
            
        Returns:
            x_0: Recovered (purified) image
        """
        # Initialize x
        x = x_t.clone()
        
        # Create discretized timesteps
        timesteps = torch.linspace(t_star, 0, n_steps+1, device=self.device)
        
        # Use Euler-Maruyama SDE solver
        with torch.no_grad():
            for i in range(n_steps):
                t = timesteps[i]
                dt = timesteps[i] - timesteps[i+1]
                
                # Predict noise
                predicted_noise = self.model(x, t)
                
                # Calculate drift coefficient
                beta_t = self.beta_t(t)
                drift = -0.5 * beta_t * (x + 2 * predicted_noise)
                
                # Calculate diffusion coefficient
                diffusion = torch.sqrt(beta_t)
                
                # Sample Wiener process increment
                dw = torch.randn_like(x) * torch.sqrt(torch.abs(dt))
                
                # Update x using the SDE
                x = x + drift * dt + diffusion * dw
                
                # Optionally apply clipping to maintain image bounds
                x = torch.clamp(x, -1.0, 1.0)
        
        return x
    
    def purify(self, x_adv, t_star=0.1, n_steps=100):
        """
        Purify adversarial examples using diffusion.
        
        Args:
            x_adv: Adversarial examples
            t_star: Diffusion timestep (controls noise level)
            n_steps: Number of discretization steps for reverse diffusion
            
        Returns:
            x_purified: Purified images
        """
        # Forward diffusion process to add noise
        x_t = self.forward_diffusion(x_adv, t_star)
        
        # Reverse diffusion process to recover clean image
        x_purified = self.reverse_diffusion(x_t, t_star, n_steps)
        
        return x_purified


class ScoreNetwork(nn.Module):
    """
    U-Net style network for predicting the score (gradient of log likelihood).
    This is a simplified version of the score network used in diffusion models.
    """
    def __init__(self, in_channels=3, hidden_channels=128, out_channels=3):
        super().__init__()
        
        # Time embedding layer
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, stride=2)
        
        # Middle
        self.middle1 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.middle2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(hidden_channels*2, hidden_channels, 4, padding=1, stride=2)
        self.conv4 = nn.Conv2d(hidden_channels*2, hidden_channels, 3, padding=1)
        
        self.up2 = nn.ConvTranspose2d(hidden_channels*2, hidden_channels, 4, padding=1, stride=2)
        self.conv5 = nn.Conv2d(hidden_channels*2, hidden_channels, 3, padding=1)
        
        # Output
        self.out = nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
        
    def forward(self, x, t):
        """
        Forward pass of the score network.
        
        Args:
            x: Input image tensor [B, C, H, W]
            t: Timestep tensor [B]
            
        Returns:
            Score prediction (noise)
        """
        # Time embedding
        t = t.view(-1, 1)
        t_emb = self.time_embed(t)
        t_emb = t_emb.view(-1, t_emb.shape[1], 1, 1)
        
        # Encoder
        h1 = F.silu(self.conv1(x))
        h1 = h1 + t_emb
        
        h2 = F.silu(self.conv2(h1))
        h2 = h2 + t_emb
        
        h3 = F.silu(self.conv3(h2))
        h3 = h3 + t_emb
        
        # Middle
        h3 = F.silu(self.middle1(h3))
        h3 = F.silu(self.middle2(h3))
        
        # Decoder with skip connections
        h = torch.cat([h3, h3], dim=1)
        h = F.silu(self.up1(h))
        
        h = torch.cat([h, h2], dim=1)
        h = F.silu(self.conv4(h))
        
        h = torch.cat([h, h], dim=1)
        h = F.silu(self.up2(h))
        
        h = torch.cat([h, h1], dim=1)
        h = F.silu(self.conv5(h))
        
        # Output
        output = self.out(h)
        
        return output


def train_diffusion_model(score_model, dataloader, epochs=10, lr=1e-4, device='cpu'):
    """
    Train the diffusion model's score network using denoising score matching.
    
    Args:
        score_model: Score network model
        dataloader: DataLoader for training data
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        
    Returns:
        Trained score model
    """
    score_model = score_model.to(device)
    optimizer = torch.optim.Adam(score_model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for images, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            batch_size = images.shape[0]
            
            # Sample random timesteps
            t = torch.rand(batch_size, device=device)
            
            # Compute alpha_t
            alpha_t = torch.exp(-0.5 * (t * 0.1 + 0.5 * t**2 * (20.0 - 0.1)))
            
            # Sample noise
            epsilon = torch.randn_like(images)
            
            # Create noisy images
            x_t = alpha_t.view(-1, 1, 1, 1) * images + torch.sqrt(1 - alpha_t.view(-1, 1, 1, 1)**2) * epsilon
            
            # Predict noise
            predicted_noise = score_model(x_t, t)
            
            # Compute loss (denoising score matching)
            loss = F.mse_loss(predicted_noise, epsilon)
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return score_model 