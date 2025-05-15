"""
Internal grey-box PGD attack implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from federated.client import Client
from config import Config, DEVICE

class InternalPGDAttack(nn.Module):
    """Internal PGD Attack that generates PGD on purified inputs"""
    def __init__(self, epsilon=8/255, step_size=None, steps=40, random_start=True):
        super().__init__()
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size or epsilon/4
        self.random_start = random_start
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, model: nn.Module, diffuser: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples using PGD on purified inputs.
        
        Args:
            model: Target model
            diffuser: Diffusion purifier
            x: Clean images
            y: True labels
            
        Returns:
            Adversarial examples
        """
        x = x.detach().clone()
        
        # First purify the input
        x_clean_pur = diffuser.purify(x)
        
        # Random initialization
        if self.random_start:
            x_adv = x_clean_pur + torch.zeros_like(x_clean_pur).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
        else:
            x_adv = x_clean_pur.clone()
        
        for _ in range(self.steps):
            x_adv.requires_grad = True
            
            # Forward pass
            outputs = model(x_adv)
            loss = torch.nn.functional.cross_entropy(outputs, y)
            
            # Backward pass
            loss.backward()
            
            # Update
            grad = x_adv.grad.detach()
            x_adv = x_adv.detach()
            x_adv = x_adv + self.step_size * grad.sign()
            
            # Project to epsilon ball around the purified image
            x_adv = torch.min(torch.max(x_adv, x_clean_pur - self.epsilon), x_clean_pur + self.epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
            
        return x_adv.detach()
        
    def perturb(self, model: nn.Module, diffuser: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Alias for forward method."""
        return self.forward(model, diffuser, x, y)
        
    def attack_combined_defense(self, model: nn.Module, diffuser: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Attack the combined defense by generating examples that are adversarial post-purification.
        
        Args:
            model: Target model (pFedDef)
            diffuser: Diffusion purifier
            x: Clean images
            y: True labels
            
        Returns:
            Adversarial examples
        """
        # Purify the clean input first
        x_clean_pur = diffuser.purify(x)
        
        # Initialize adversarial examples from the purified inputs
        x_adv = x_clean_pur.clone().detach()
        if self.random_start:
            noise = torch.zeros_like(x_adv).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv + noise, 0, 1)
        
        for _ in range(self.steps):
            x_adv.requires_grad_(True)
            
            # Forward pass through the model
            outputs = model(x_adv)
            loss = F.cross_entropy(outputs, y)
            
            # Compute gradients
            loss.backward()
            
            # Update with gradient sign
            with torch.no_grad():
                grad = x_adv.grad.sign()
                x_adv = x_adv + self.step_size * grad
                
                # Project back to epsilon ball around purified input
                delta = torch.clamp(x_adv - x_clean_pur, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x_clean_pur + delta, 0, 1)
            
            # Reset gradients
            x_adv.grad = None
            
        return x_adv.detach()

def evaluate_internal_attack(attacker_client, victim_clients, test_loader):
    """
    Evaluate internal grey-box attack success
    
    Args:
        attacker_client: Client 0 (attacker)
        victim_clients: List of other clients
        test_loader: Test data loader
        
    Returns:
        dict: Attack success metrics
    """
    attack = InternalPGDAttack(
        epsilon=CONFIG['PGD_EPS'],
        steps=CONFIG['PGD_STEPS']
    )
    
    success_rates = []
    for victim in victim_clients:
        correct = 0
        total = 0
        
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # Generate adversarial examples
            adv_data = attack.attack(attacker_client.model, attacker_client.diffuser, data, target)
            
            # Purify adversarial examples before evaluation
            with torch.no_grad():
                adv_data = victim.diffuser.purify(
                    adv_data,
                    steps=CONFIG['DIFFUSER_STEPS'],
                    sigma=CONFIG['DIFFUSER_SIGMA']
                )
            
            # Evaluate on victim
            with torch.no_grad():
                outputs = victim.model(adv_data, is_training=False)
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        # Calculate success rate (misclassification rate)
        success_rate = 100. * (1 - correct / total)
        success_rates.append(success_rate)
    
    return {
        'mean_success_rate': sum(success_rates) / len(success_rates),
        'std_success_rate': torch.tensor(success_rates).std().item(),
        'per_client_rates': success_rates
    }

def internal_attack(clients: List[Client], cfg: Config) -> float:
    """Evaluate internal (grey-box) attack success rate.
    
    Each client acts as an attacker, crafting adversarial examples on their test set
    and evaluating success rate against other clients' models.
    """
    total_success_rate = 0
    n_pairs = 0
    
    for attacker_idx, attacker in enumerate(clients):
        # Freeze attacker's model
        for param in attacker.model.parameters():
            param.requires_grad = False
            
        test_loader = torch.utils.data.DataLoader(
            attacker.test_data,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False
        )
        
        # For each victim
        for victim_idx, victim in enumerate(clients):
            if attacker_idx == victim_idx:
                continue
                
            success = 0
            total = 0
            
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                
                # Generate adversarial examples using PGD
                adv_data = attack.attack(attacker.model, attacker.diffuser, data, target)
                
                # Purify adversarial examples before evaluation
                with torch.no_grad():
                    adv_data = victim.diffuser.purify(
                        adv_data,
                        steps=cfg.DIFFUSER_STEPS,
                        sigma=cfg.DIFFUSER_SIGMA
                    )
                
                # Evaluate on victim model
                with torch.no_grad():
                    output = victim.classifier(adv_data)
                    pred = output.argmax(dim=1)
                    success += (pred != target).sum().item()
                    total += target.size(0)
                    
            success_rate = success / total
            total_success_rate += success_rate
            n_pairs += 1
            
        # Unfreeze attacker's model
        for param in attacker.model.parameters():
            param.requires_grad = True
            
    return total_success_rate / n_pairs

def pgd_attack(model, diffuser, data, target, epsilon, alpha, steps):
    """PGD attack implementation with purification."""
    # First purify the input
    with torch.no_grad():
        data = diffuser.purify(
            data,
            steps=steps,
            sigma=CONFIG['DIFFUSER_SIGMA']
        )
    
    perturbed_data = data.clone().detach()
    
    if steps > 0:
        perturbed_data = perturbed_data + torch.zeros_like(perturbed_data).uniform_(-epsilon, epsilon)
        
    for step in range(steps):
        perturbed_data.requires_grad = True
        output = model(perturbed_data)
        
        model.zero_grad()
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        
        with torch.no_grad():
            perturbed_data = perturbed_data.detach() + alpha * perturbed_data.grad.sign()
            delta = torch.clamp(perturbed_data - data, -epsilon, epsilon)
            perturbed_data = torch.clamp(data + delta, 0, 1).detach()
            
    return perturbed_data 