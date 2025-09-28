"""
SELECTIVE DEFENSE PIPELINE
=========================
Integrates MAE detection with selective DiffPure purification
"""

import torch
import torch.nn as nn
import logging

class SelectiveDefensePipeline(nn.Module):
    """Selective defense using MAE detection + DiffPure purification"""
    
    def __init__(self, mae_detector, diffuser, cfg):
        super().__init__()
        self.mae_detector = mae_detector
        self.diffuser = diffuser
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        
        # Defense parameters
        self.diffpure_steps = getattr(cfg, 'DIFFPURE_STEPS', 15)
        self.diffpure_sigma = getattr(cfg, 'DIFFPURE_SIGMA', 0.04)
        
    def forward(self, x, model):
        """Apply selective defense pipeline"""
        batch_size = x.size(0)
        device = x.device
        
        # Step 1: MAE Detection
        is_adversarial, recon_errors = self.mae_detector.detect_adversarial(x)
        
        # Step 2: Selective DiffPure
        processed_x = x.clone()
        adv_indices = torch.where(is_adversarial)[0]
        
        if len(adv_indices) > 0:
            # Apply DiffPure only to detected adversarial samples
            adv_samples = x[adv_indices]
            purified_samples = self.apply_diffpure(adv_samples)
            processed_x[adv_indices] = purified_samples
            
            self.logger.info(f"Selective Defense: {len(adv_indices)}/{batch_size} samples purified")
        
        # Step 3: Model prediction
        with torch.no_grad():
            outputs = model(processed_x)
        
        return outputs, is_adversarial
    
    def apply_diffpure(self, x):
        """Apply DiffPure purification"""
        batch_size = x.size(0)
        device = x.device
        
        # Add noise
        noise = torch.randn_like(x) * self.diffpure_sigma
        noisy_x = torch.clamp(x + noise, 0, 1)
        
        # Denoise using diffusion model
        with torch.no_grad():
            for step in range(self.diffpure_steps):
                # Create timestep
                t = torch.full((batch_size,), step, device=device, dtype=torch.float32)
                
                # Predict noise
                predicted_noise = self.diffuser(noisy_x, t)
                
                # Remove noise (simplified)
                noisy_x = noisy_x - predicted_noise * 0.1
                noisy_x = torch.clamp(noisy_x, 0, 1)
        
        return noisy_x

def create_selective_pipeline(mae_detector, diffuser, cfg):
    """Factory function"""
    return SelectiveDefensePipeline(mae_detector, diffuser, cfg)
