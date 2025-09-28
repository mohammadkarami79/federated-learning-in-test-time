import torch
import torch.nn as nn
import torch.nn.functional as F

class PGDAttack(nn.Module):
    """Bulletproof PGD Attack with guaranteed epsilon constraint"""
    
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
        
        # Ensure reasonable step size
        self.step_size = min(self.step_size, self.epsilon / 2)
            
        print(f"[PGD] Epsilon: {self.epsilon:.6f}, Step size: {self.step_size:.6f}, Steps: {self.steps}")
        
        self.criterion = nn.CrossEntropyLoss()
    
    def attack(self, model, x, y):
        """Generate adversarial examples with GUARANTEED epsilon constraint"""
        model.eval()
        
        # Ensure inputs are on same device as model
        device = next(model.parameters()).device
        x, y = x.to(device), y.to(device)
        
        # Store original images for projection
        x_orig = x.clone().detach()
        
        # Initialize adversarial examples
        x_adv = x.clone().detach()
        
        # Random initialization within epsilon ball
        if self.random_start:
            noise = torch.zeros_like(x_adv).uniform_(-self.epsilon, self.epsilon)
            x_adv = x_adv + noise
            # CRITICAL: Project to valid range AND epsilon ball
            x_adv = torch.clamp(x_adv, 0, 1)
            delta = x_adv - x_orig
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            x_adv = x_orig + delta
        
        # PGD iterations
        for step in range(self.steps):
            x_adv.requires_grad_(True)
            
            # Forward pass
            outputs = model(x_adv)
            loss = F.cross_entropy(outputs, y)
            
            # Backward pass
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
            
            # Update adversarial examples
            with torch.no_grad():
                # Gradient ascent step
                x_adv = x_adv + self.step_size * grad.sign()
                
                # CRITICAL: Project to epsilon ball first, then valid range
                delta = x_adv - x_orig
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                x_adv = x_orig + delta
                x_adv = torch.clamp(x_adv, 0, 1)
                
                # Double-check: ensure we're still in epsilon ball after clamping
                delta = x_adv - x_orig
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                x_adv = x_orig + delta
        
        # Final verification
        final_delta = x_adv - x_orig
        max_perturbation = final_delta.abs().max().item()
        if max_perturbation > self.epsilon + 1e-6:
            print(f"[WARNING] Perturbation {max_perturbation:.6f} exceeds epsilon {self.epsilon:.6f}")
            # Force projection
            final_delta = torch.clamp(final_delta, -self.epsilon, self.epsilon)
            x_adv = x_orig + final_delta
            x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv.detach()
    
    def forward(self, model, x, y):
        """Forward method for compatibility"""
        return self.attack(model, x, y)
    
    def generate(self, model, x, y):
        """Generate method for compatibility"""
        return self.attack(model, x, y)
