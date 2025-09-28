#!/usr/bin/env python3
"""
Conservative DiffPure Implementation
Very gentle purification to preserve adversarial accuracy
"""

import torch
import torch.nn.functional as F

def conservative_diffpure_purify(diffuser, adv_data, cfg):
    """Conservative DiffPure purification - very gentle"""
    
    # Conservative parameters
    num_steps = getattr(cfg, 'DIFFUSER_STEPS', 2)  # Few steps
    sigma = getattr(cfg, 'DIFFUSER_SIGMA', 0.1)    # Low noise
    strength = getattr(cfg, 'DIFFPURE_STRENGTH', 0.15)  # Very gentle strength
    
    device = adv_data.device
    batch_size = adv_data.size(0)
    
    # Ensure inputs are in [0,1] range
    adv_data = torch.clamp(adv_data, 0.0, 1.0)
    purified_data = adv_data.clone()
    
    with torch.no_grad():
        # Conservative purification - very gentle
        for step in range(num_steps):
            # Very low noise
            current_sigma = sigma * (0.5 ** step)  # Even lower noise
            noise = torch.randn_like(purified_data) * current_sigma
            noisy_data = purified_data + noise
            
            # Create time steps for diffusion model
            t = torch.full((batch_size,), num_steps - step, device=device, dtype=torch.float32)
            
            # Predict and remove noise
            predicted_noise = diffuser(noisy_data, t)
            
            # Very gentle denoising - only 15% of predicted noise
            denoising_strength = strength  # Very gentle
            purified_data = noisy_data - predicted_noise * denoising_strength
            
            # Clamp to valid range
            purified_data = torch.clamp(purified_data, 0.0, 1.0)
    
    return purified_data

def ultra_conservative_diffpure_purify(diffuser, adv_data, cfg):
    """Ultra conservative DiffPure - minimal changes"""
    
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
        
        # Ultra minimal denoising - only 10% of predicted noise
        purified_data = noisy_data - predicted_noise * 0.1
        
        # Clamp to valid range
        purified_data = torch.clamp(purified_data, 0.0, 1.0)
    
    return purified_data

if __name__ == "__main__":
    print("Conservative DiffPure Functions:")
    print("1. conservative_diffpure_purify - 2 steps, sigma=0.1, 15% denoising")
    print("2. ultra_conservative_diffpure_purify - 1 step, sigma=0.05, 10% denoising")
    print("These should preserve adversarial accuracy while providing gentle defense.")
