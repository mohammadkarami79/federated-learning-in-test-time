"""
Transfer attacks module for testing diffusion purification defenses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Callable, Dict, List
import torchvision

class FGSM:
    """
    Fast Gradient Sign Method (FGSM) attack
    """
    def __init__(self, model: nn.Module, epsilon: float = 8/255):
        self.model = model
        self.epsilon = epsilon
        self.name = "FGSM"
    
    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial examples using FGSM
        
        Args:
            images: Clean images
            labels: True labels
            
        Returns:
            Adversarial examples
        """
        images.requires_grad = True
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)
        self.model.zero_grad()
        loss.backward()
        
        perturbed_images = images + self.epsilon * images.grad.sign()
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        return perturbed_images.detach()

class PGD:
    """
    Projected Gradient Descent (PGD) attack
    """
    def __init__(self, 
                 model: nn.Module, 
                 epsilon: float = 8/255, 
                 alpha: float = 2/255, 
                 steps: int = 10,
                 random_start: bool = True):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.name = "PGD"
    
    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial examples using PGD
        
        Args:
            images: Clean images
            labels: True labels
            
        Returns:
            Adversarial examples
        """
        perturbed_images = images.clone().detach()
        
        if self.random_start:
            # Add uniform random noise to the images
            noise = torch.FloatTensor(images.shape).uniform_(-self.epsilon, self.epsilon).to(images.device)
            perturbed_images = perturbed_images + noise
            # Clip the images to ensure valid pixel range [0, 1]
            perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        for _ in range(self.steps):
            perturbed_images.requires_grad = True
            outputs = self.model(perturbed_images)
            loss = F.cross_entropy(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            # Update adversarial images
            adv_images = perturbed_images + self.alpha * perturbed_images.grad.sign()
            # Project back to epsilon ball
            eta = torch.clamp(adv_images - images, -self.epsilon, self.epsilon)
            perturbed_images = torch.clamp(images + eta, 0, 1).detach()
        
        return perturbed_images

class CW:
    """
    Carlini & Wagner (C&W) L2 attack
    """
    def __init__(self, model: nn.Module, c: float = 1.0, steps: int = 100, lr: float = 0.01):
        self.model = model
        self.c = c
        self.steps = steps
        self.lr = lr
        self.name = "CW"
    
    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial examples using C&W L2 attack
        
        Args:
            images: Clean images
            labels: True labels
            
        Returns:
            Adversarial examples
        """
        batch_size = images.size(0)
        # Initialize delta
        delta = torch.zeros_like(images, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.lr)
        
        for _ in range(self.steps):
            adv_images = torch.clamp(images + delta, 0, 1)
            outputs = self.model(adv_images)
            
            # Calculate f(x+δ)_t
            target_logits = outputs.gather(1, labels.unsqueeze(1)).squeeze()
            # Calculate max{f(x+δ)_i : i≠t}
            max_other_logits = torch.zeros_like(target_logits)
            for i in range(batch_size):
                other_logits = torch.cat([outputs[i, :labels[i]], outputs[i, labels[i]+1:]])
                max_other_logits[i] = torch.max(other_logits)
            
            # Calculate loss
            l2_loss = torch.sum(delta ** 2)
            margin_loss = torch.clamp(target_logits - max_other_logits, min=-50.0)
            loss = l2_loss + self.c * margin_loss.sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return torch.clamp(images + delta.detach(), 0, 1)

def create_transfer_model(model_name: str = 'resnet18', dataset_name: str = 'cifar10', device=None) -> nn.Module:
    """
    Create a transfer model for generating adversarial examples
    
    Args:
        model_name: Name of the model architecture
        dataset_name: Name of the dataset
        device: Device to put the model on
        
    Returns:
        Pretrained model
    """
    if model_name == 'resnet18':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    elif model_name == 'vgg16':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    elif model_name == 'densenet121':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Get number of classes based on dataset
    num_classes = 10  # Default for CIFAR-10, MNIST
    if dataset_name == 'cifar100':
        num_classes = 100
    
    # Adjust input channels for grayscale datasets
    in_channels = 1 if dataset_name == 'mnist' else 3
    
    # Adjust the model for the dataset
    if model_name == 'resnet18':
        if in_channels == 1:
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg16':
        if in_channels == 1:
            model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'densenet121':
        if in_channels == 1:
            model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    # Move to device if specified
    if device is not None:
        model = model.to(device)
    
    model.eval()
    return model

def get_attack(attack_type, model, **kwargs):
    """
    Get attack instance based on type
    
    Args:
        attack_type: Type of attack ('fgsm', 'pgd', 'cw')
        model: Target model
        **kwargs: Attack parameters
        
    Returns:
        Attack instance
    """
    if attack_type == 'fgsm':
        return FGSM(model, **kwargs)
    elif attack_type == 'pgd':
        return PGD(model, **kwargs)
    elif attack_type == 'cw':
        return CW(model, **kwargs)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

def evaluate_transfer_attack(clean_model: nn.Module, 
                            transfer_model: nn.Module,
                            attack_type: str,
                            test_loader: torch.utils.data.DataLoader,
                            device: torch.device,
                            **attack_kwargs) -> Tuple[float, float, torch.Tensor]:
    """
    Evaluate transfer attack performance
    
    Args:
        clean_model: Model to evaluate robustness
        transfer_model: Model to generate adversarial examples
        attack_type: Type of attack
        test_loader: Test data loader
        device: Device to use
        **attack_kwargs: Additional arguments for the attack
        
    Returns:
        Tuple of (clean accuracy, adversarial accuracy, adversarial examples)
    """
    transfer_model.eval()
    clean_model.eval()
    
    attack = get_attack(attack_type, transfer_model, **attack_kwargs)
    
    total = 0
    clean_correct = 0
    adv_correct = 0
    adv_examples = []
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        total += labels.size(0)
        
        # Evaluate on clean examples
        with torch.no_grad():
            outputs = clean_model(images)
            _, predicted = torch.max(outputs.data, 1)
            clean_correct += (predicted == labels).sum().item()
        
        # Generate adversarial examples
        adv_images = attack.attack(images, labels)
        
        # Evaluate on adversarial examples
        with torch.no_grad():
            adv_outputs = clean_model(adv_images)
            _, adv_predicted = torch.max(adv_outputs.data, 1)
            adv_correct += (adv_predicted == labels).sum().item()
        
        # Save some adversarial examples
        if len(adv_examples) < 10:
            adv_examples.append((images[0].cpu(), labels[0].item(), 
                                adv_images[0].cpu(), adv_predicted[0].item()))
    
    clean_acc = clean_correct / total
    adv_acc = adv_correct / total
    
    return clean_acc, adv_acc, adv_examples 