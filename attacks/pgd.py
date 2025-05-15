import torch
import torch.nn as nn
import torch.nn.functional as F

class PGDAttack(nn.Module):
    """PGD Attack implementation"""
    def __init__(self, epsilon=8/255, step_size=None, steps=40, random_start=True):
        super().__init__()
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size or epsilon/4
        self.random_start = random_start
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples using PGD.
        
        Args:
            model: Target model
            x: Clean images
            y: True labels
            
        Returns:
            Adversarial examples
        """
        # Clone and detach input, then explicitly enable gradients
        x_adv = x.clone().detach().requires_grad_(True)
        
        # Random initialization
        if self.random_start:
            with torch.no_grad():
                noise = torch.zeros_like(x_adv).uniform_(-self.epsilon, self.epsilon)
                x_adv = torch.clamp(x_adv + noise, 0, 1).requires_grad_(True)
        
        for _ in range(self.steps):
            # Forward pass
            outputs = model(x_adv)
            loss = torch.nn.functional.cross_entropy(outputs, y)
            
            # Backward pass
            loss.backward()
            
            # Update
            with torch.no_grad():
                grad = x_adv.grad
                if grad is None:
                    # If gradient is None, try a different approach
                    break
                    
                # Update adversarial examples
                adv = x_adv + self.step_size * grad.sign()
                
                # Project back to epsilon ball
                delta = torch.clamp(adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, 0, 1)
            
            # Reset gradients and re-enable for next iteration
            if _ < self.steps - 1:  # No need to retain grad for last iteration
                x_adv = x_adv.detach().requires_grad_(True)
            
        return x_adv.detach()
        
    def perturb(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Alias for forward method."""
        return self.forward(model, x, y)

    def attack(self, images, labels):
        """
        Generate adversarial examples using PGD attack
        
        Args:
            images: Clean images
            labels: True labels
            
        Returns:
            Adversarial examples
        """
        images = images.clone().detach()
        labels = labels.clone().detach()
        
        # Move to device if needed
        if next(self.model.parameters()).is_cuda:
            images = images.cuda()
            labels = labels.cuda()
        
        # Initialize adversarial examples with gradients enabled
        adv_images = images.clone().detach().requires_grad_(True)
        
        if self.random_start:
            # Random initialization
            with torch.no_grad():
                noise = torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
                adv_images = torch.clamp(adv_images + noise, min=0, max=1)
            adv_images = adv_images.detach().requires_grad_(True)
        
        # PGD iterations
        for _ in range(self.steps):
            # Forward pass
            with torch.enable_grad():
                outputs = self.model(adv_images, is_training=False)
                cost = self.criterion(outputs, labels)
            
            # Backward pass
            grad = torch.autograd.grad(cost, adv_images, 
                                     retain_graph=False, 
                                     create_graph=False)[0]
            
            # Update adversarial images
            with torch.no_grad():
                adv_images = adv_images.detach() + self.step_size * grad.sign()
                
                # Project back to epsilon ball
                delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
                adv_images = torch.clamp(images + delta, min=0, max=1)
            
            # Re-enable gradients for next iteration
            if _ < self.steps - 1:  # No need for last iteration
                adv_images = adv_images.requires_grad_(True)
        
        return adv_images.detach() 