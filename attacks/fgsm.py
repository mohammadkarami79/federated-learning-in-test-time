"""
FGSM attack implementation
"""

import torch
import torch.nn as nn

class FGSMAttack:
    def __init__(self, epsilon=8/255):
        self.epsilon = epsilon
        
    def perturb(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples using FGSM.
        
        Args:
            model: Target model
            x: Clean images
            y: True labels
            
        Returns:
            Adversarial examples
        """
        x.requires_grad = True
        
        # Forward pass
        outputs = model(x)
        loss = torch.nn.functional.cross_entropy(outputs, y)
        
        # Backward pass
        loss.backward()
        
        # Generate perturbation
        perturbation = self.epsilon * x.grad.sign()
        
        # Create adversarial examples
        x_adv = x + perturbation
        x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv.detach() 