#!/usr/bin/env python3
"""
Ultra Gentle DiffPure Implementation
Minimal purification to preserve adversarial accuracy
"""

import torch
import torch.nn.functional as F

def ultra_gentle_diffpure_purify(diffuser, adv_data, cfg):
    """Ultra gentle DiffPure purification - minimal changes"""
    
    # Ultra gentle parameters
    num_steps = getattr(cfg, 'DIFFUSER_STEPS', 2)  # Very few steps
    sigma = getattr(cfg, 'DIFFUSER_SIGMA', 0.1)    # Very low noise
    
    device = adv_data.device
    batch_size = adv_data.size(0)
    
    # Ensure inputs are in [0,1] range
    adv_data = torch.clamp(adv_data, 0.0, 1.0)
    purified_data = adv_data.clone()
    
    with torch.no_grad():
        # Single gentle purification step
        for step in range(num_steps):
            # Very low noise
            current_sigma = sigma * 0.5  # Even lower noise
            noise = torch.randn_like(purified_data) * current_sigma
            noisy_data = purified_data + noise
            
            # Create time steps for diffusion model
            t = torch.full((batch_size,), num_steps - step, device=device, dtype=torch.float32)
            
            # Predict and remove noise with very gentle scaling
            predicted_noise = diffuser(noisy_data, t)
            
            # Very gentle denoising - only 20% of predicted noise
            denoising_strength = 0.2  # Very gentle
            purified_data = noisy_data - predicted_noise * denoising_strength
            
            # Clamp to valid range
            purified_data = torch.clamp(purified_data, 0.0, 1.0)
    
    return purified_data

def minimal_diffpure_purify(diffuser, adv_data, cfg):
    """Minimal DiffPure - almost no changes"""
    
    device = adv_data.device
    batch_size = adv_data.size(0)
    
    # Ensure inputs are in [0,1] range
    adv_data = torch.clamp(adv_data, 0.0, 1.0)
    purified_data = adv_data.clone()
    
    with torch.no_grad():
        # Single very gentle step
        noise = torch.randn_like(purified_data) * 0.05  # Very low noise
        noisy_data = purified_data + noise
        
        # Create time step
        t = torch.full((batch_size,), 1, device=device, dtype=torch.float32)
        
        # Predict and remove noise with minimal scaling
        predicted_noise = diffuser(noisy_data, t)
        
        # Minimal denoising - only 10% of predicted noise
        purified_data = noisy_data - predicted_noise * 0.1
        
        # Clamp to valid range
        purified_data = torch.clamp(purified_data, 0.0, 1.0)
    
    return purified_data

if __name__ == "__main__":
    print("Ultra Gentle DiffPure Functions:")
    print("1. ultra_gentle_diffpure_purify - 2 steps, sigma=0.1, 20% denoising")
    print("2. minimal_diffpure_purify - 1 step, sigma=0.05, 10% denoising")
    print("These should preserve adversarial accuracy while still providing some defense.")
