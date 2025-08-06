import torch
import torch.nn as nn
import torch.nn.functional as F

class PGDAttack(nn.Module):
    """PGD Attack implementation compatible with configuration objects"""
    
    def __init__(self, cfg_or_epsilon=8/255, step_size=None, steps=None, random_start=None):
        super().__init__()
        
        # Handle both config object and direct parameters
        if hasattr(cfg_or_epsilon, 'PGD_EPS'):
            # Config object passed
            cfg = cfg_or_epsilon
            self.epsilon = float(getattr(cfg, 'PGD_EPS', 8/255))
            self.steps = getattr(cfg, 'PGD_STEPS', 10)
            self.step_size = float(getattr(cfg, 'PGD_ALPHA', self.epsilon/4))
            self.random_start = getattr(cfg, 'PGD_RANDOM_START', True)
        else:
            # Direct parameters passed
            self.epsilon = float(cfg_or_epsilon)
            self.steps = steps or 10
            self.step_size = float(step_size or (self.epsilon/4))
            self.random_start = random_start if random_start is not None else True
        
        self.criterion = nn.CrossEntropyLoss()
    
    def generate(self, model, x, y):
        """Generate adversarial examples using PGD - main interface"""
        return self.forward(model, x, y)
    
    def forward(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples using PGD.
        
        Args:
            model: Target model
            x: Clean images
            y: True labels
            
        Returns:
            Adversarial examples
        """
        model.eval()  # Ensure model is in eval mode
        
        # Clone and detach input, then explicitly enable gradients
        x_adv = x.clone().detach().requires_grad_(True)
        
        # Random initialization
        if self.random_start:
            with torch.no_grad():
                noise = torch.zeros_like(x_adv).uniform_(-self.epsilon, self.epsilon)
                x_adv = torch.clamp(x_adv + noise, 0, 1).requires_grad_(True)
        
        for step in range(self.steps):
            # Clear gradients
            if x_adv.grad is not None:
                x_adv.grad.zero_()
            
            # Forward pass
            outputs = model(x_adv)
            loss = F.cross_entropy(outputs, y)
            
            # Backward pass
            loss.backward()
            
            # Update
            with torch.no_grad():
                grad = x_adv.grad
                if grad is None:
                    # If gradient is None, break
                    break
                    
                # Update adversarial examples
                adv = x_adv + self.step_size * grad.sign()
                
                # Project back to epsilon ball
                delta = torch.clamp(adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, 0, 1)
            
            # Re-enable gradients for next iteration (except last)
            if step < self.steps - 1:
                x_adv = x_adv.detach().requires_grad_(True)
            
        return x_adv.detach()
        
    def perturb(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Alias for forward method."""
        return self.forward(model, x, y)

    def attack(self, model, images, labels):
        """
        Generate adversarial examples using PGD attack
        
        Args:
            model: Target model
            images: Clean images
            labels: True labels
            
        Returns:
            Adversarial examples
        """
        return self.forward(model, images, labels) 