"""
Attack utilities for generating adversarial examples
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Callable
import logging

class PGD:
    """
    Projected Gradient Descent (PGD) attack
    """
    def __init__(self, epsilon=0.1, alpha=0.01, num_steps=20):
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        
    def generate(self, x, y, model):
        """
        Generate adversarial examples
        
        Args:
            x (torch.Tensor): Input images
            y (torch.Tensor): Target labels
            model (nn.Module): Model to attack
            
        Returns:
            torch.Tensor: Adversarial examples
        """
        x_adv = x.clone().detach().requires_grad_(True)
        
        for _ in range(self.num_steps):
            output = model(x_adv)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            
            with torch.no_grad():
                x_adv = x_adv + self.alpha * x_adv.grad.sign()
                x_adv = torch.clamp(x_adv, x - self.epsilon, x + self.epsilon)
                x_adv = torch.clamp(x_adv, 0, 1)
                x_adv.grad.zero_()
        
        return x_adv.detach()

class FGSM:
    """
    Fast Gradient Sign Method (FGSM) attack
    """
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        
    def generate(self, x, y, model):
        """
        Generate adversarial examples
        
        Args:
            x (torch.Tensor): Input images
            y (torch.Tensor): Target labels
            model (nn.Module): Model to attack
            
        Returns:
            torch.Tensor: Adversarial examples
        """
        x_adv = x.clone().detach().requires_grad_(True)
        
        output = model(x_adv)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        with torch.no_grad():
            x_adv = x_adv + self.epsilon * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv.detach()

def create_attack(config: Dict[str, Any]) -> Callable:
    """
    Create an attack function based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Callable: Attack function
    """
    attack_type = config['attack']['type']
    epsilon = config['attack']['epsilon']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if attack_type == 'pgd':
        def pgd_attack(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            """
            PGD attack
            
            Args:
                images: Input images
                labels: True labels
                
            Returns:
                torch.Tensor: Adversarial images
            """
            images = images.clone().detach().to(device)
            labels = labels.clone().detach().to(device)
            
            # Initialize perturbation
            delta = torch.zeros_like(images).uniform_(-epsilon, epsilon)
            delta = torch.clamp(delta, -epsilon, epsilon)
            
            # Get model from config
            model = config['models']['target']['model']
            model.eval()
            
            # PGD iterations
            for _ in range(config['attack']['num_steps']):
                delta.requires_grad = True
                outputs = model(images + delta)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                
                model.zero_grad()
                loss.backward()
                
                # Update perturbation
                grad = delta.grad.detach()
                delta = delta + config['attack']['step_size'] * grad.sign()
                delta = torch.clamp(delta, -epsilon, epsilon)
                delta = delta.detach()
            
            # Return adversarial images
            return torch.clamp(images + delta, 0, 1)
        
        return pgd_attack
    
    elif attack_type == 'fgsm':
        def fgsm_attack(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            """
            FGSM attack
            
            Args:
                images: Input images
                labels: True labels
                
            Returns:
                torch.Tensor: Adversarial images
            """
            images = images.clone().detach().to(device)
            labels = labels.clone().detach().to(device)
            
            # Get model from config
            model = config['models']['target']['model']
            model.eval()
            
            # Compute gradient
            images.requires_grad = True
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            model.zero_grad()
            loss.backward()
            
            # Create adversarial images
            grad = images.grad.detach()
            delta = epsilon * grad.sign()
            return torch.clamp(images + delta, 0, 1)
        
        return fgsm_attack
    
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}") 