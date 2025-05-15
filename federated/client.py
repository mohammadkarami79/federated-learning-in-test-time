"""
Federated client implementation with local training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import gc

from config import Config, DEVICE
from defense.combined_defense import CombinedClassifier
from models.pfeddef_model import pFedDefModel
from attacks.pgd import PGDAttack
from utils.data_utils import get_dataloader

class Client:
    def __init__(self, client_id: int, cfg: Config, diffuser: Optional[nn.Module] = None):
        self.client_id = client_id
        self.cfg = cfg
        self.diffuser = diffuser
        
        # Initialize models
        self.model = self._create_model()
        self.classifier = None  # Will be set during training
        
        # Get data loaders using unified function
        self.train_loader = get_dataloader(cfg, split="train")
        self.test_loader = get_dataloader(cfg, split="test")
        
    def _create_model(self) -> nn.Module:
        """Create local model."""
        return pFedDefModel(self.cfg).to(DEVICE)
        
    def train(self, epochs: int = 1):
        """Train local model with purification and adversarial training.
        
        Args:
            epochs: Number of local epochs
        """
        # Create combined classifier
        self.classifier = CombinedClassifier(
            diffuser=self.diffuser,
            pfeddef_model=self.model,
            cfg=self.cfg
        ).to(DEVICE)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.cfg.LEARNING_RATE,
            momentum=0.9
        )
        
        # Initialize PGD attack
        pgd_attack = PGDAttack(
            epsilon=8/255,
            steps=self.cfg.PGD_STEPS,
            step_size=2/255
        )
        
        # Training loop
        self.classifier.train()
        for epoch in range(epochs):
            running_loss = 0.0
            batch_count = 0
            
            # Iterate over the entire DataLoader
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # Check if we've reached the maximum steps per epoch
                if batch_count >= self.cfg.LOCAL_STEPS_PER_EPOCH:
                    break
                    
                # Move data to device
                data, target = data.to(DEVICE), target.to(DEVICE)
                
                # Use mixed precision if enabled
                with torch.amp.autocast('cuda', enabled=getattr(self.cfg, 'USE_AMP', False)):
                    # Purify clean inputs and detach to break gradient flow
                    with torch.no_grad():
                        x_clean_pur = self.diffuser.purify(
                            data,
                            steps=self.cfg.DIFFUSER_STEPS,
                            sigma=self.cfg.DIFFUSER_SIGMA
                        ).detach()
                    
                    # Re-enable gradients for the purified inputs
                    x_clean_pur.requires_grad_(True)
                    
                    # Generate adversarial examples on purified inputs
                    x_adv_raw = pgd_attack.perturb(
                        self.model,
                        x_clean_pur,
                        target
                    )
                    
                    # Purify adversarial examples and detach
                    with torch.no_grad():
                        x_adv_pur = self.diffuser.purify(
                            x_adv_raw,
                            steps=self.cfg.DIFFUSER_STEPS,
                            sigma=self.cfg.DIFFUSER_SIGMA
                        ).detach()
                    
                    # Re-enable gradients for adversarial purified inputs
                    x_adv_pur.requires_grad_(True)
                    
                    # Forward passes
                    clean_output = self.model(x_clean_pur)
                    adv_output = self.model(x_adv_pur)
                    
                    # Compute losses
                    clean_loss = criterion(clean_output, target)
                    adv_loss = criterion(adv_output, target)
                    
                    # Compute KL divergence between learners
                    kl_loss = 0
                    if self.cfg.N_LEARNERS > 1:
                        logits_list = []
                        for learner in self.model.learners:
                            logits = learner(x_clean_pur)
                            logits_list.append(F.log_softmax(logits, dim=1))
                        
                        # Compute pairwise KL divergence
                        for i in range(len(logits_list)):
                            for j in range(i+1, len(logits_list)):
                                kl_loss += F.kl_div(
                                    logits_list[i],
                                    logits_list[j],
                                    reduction='batchmean'
                                )
                    
                    # Combined loss
                    loss = 0.5 * (clean_loss + adv_loss) + self.cfg.LAMBDA_KL * kl_loss
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update counters
                running_loss += loss.item()
                batch_count += 1
                
                # Free memory
                del x_clean_pur, x_adv_raw, x_adv_pur, clean_output, adv_output
                torch.cuda.empty_cache()
                gc.collect()
                    
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate model on test data.
        
        Returns:
            Tuple of (loss, accuracy)
        """
        self.classifier.eval()
        test_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = self.classifier(data, self.client_id)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        test_loss /= len(self.test_loader)
        accuracy = 100. * correct / total
        
        return test_loss, accuracy 