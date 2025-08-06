"""
Federated client implementation with local training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import List, Optional, Tuple
import numpy as np
from pathlib import Path

# Import after other modules to avoid circular imports
from models import get_model
from utils.data_utils import get_dataset
from attacks.pgd import PGDAttack
from defense.mae_detector import MAEDetector
from config_fixed import get_debug_config
from defense.combined_defense import CombinedClassifier

class Client:
    def __init__(self, client_id: int, cfg, diffuser: Optional[nn.Module] = None):
        self.client_id = client_id
        self.cfg = cfg
        self.diffuser = diffuser
        
        # Get device from cfg or use cuda if available
        self.device = getattr(cfg, 'DEVICE', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize models
        self.model = self._create_model()
        self.classifier = None  # Will be set during training
        
        # Get data using new dataset function
        train_dataset, test_dataset = get_dataset(cfg)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False
        )
        
    def _create_model(self) -> nn.Module:
        """Create local model."""
        try:
            # Use simple ResNet18 instead of pFedDef to avoid state_dict issues
            import torchvision.models as models
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, getattr(self.cfg, 'NUM_CLASSES', 10))
            return model.to(self.device)
        except:
            # Fallback to basic model
            model = get_model(self.cfg)
            return model.to(self.device)
    
    def get_models_for_server(self) -> List[nn.Module]:
        """Get models to send to server. For pFedDef, extract individual learners."""
        if hasattr(self.model, 'learners'):
            # pFedDef model - return individual learners
            return list(self.model.learners)
        else:
            # Regular model - return as single item list
            return [self.model]
        
    def train(self, epochs: int = 1):
        """Train local model with improved training for better accuracy.
        
        Args:
            epochs: Number of local epochs
        """        
        # Setup training with improved parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.cfg.LEARNING_RATE,
            momentum=0.9,
            weight_decay=1e-4  # Added weight decay for regularization
        )
        
        # Add learning rate scheduler for better convergence
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        
        # Training loop with improved monitoring
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            batch_count = 0
            correct = 0
            total = 0
            
            # Iterate over the training data
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # Check if we've reached the maximum steps per epoch (if configured)
                max_steps = getattr(self.cfg, 'LOCAL_STEPS_PER_EPOCH', None)
                if max_steps is not None and batch_count >= max_steps:
                    break
                    
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                num_classes = getattr(self.cfg, 'NUM_CLASSES', 10)

                if target.min() < 0 or target.max() >= num_classes:
                    print(f"Skipping batch {batch_idx} due to {num_classes} invalid labels.", target.min(), target.max())
                    continue
                
                # Forward pass
                output = self.model(data)
                loss = criterion(output, target)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Update counters and accuracy
                running_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                batch_count += 1
                
                # Free memory
                torch.cuda.empty_cache()
            
            # Step the scheduler
            scheduler.step()
            
            # Print epoch summary
            epoch_loss = running_loss / batch_count if batch_count > 0 else 0
            epoch_acc = 100. * correct / total if total > 0 else 0
            print(f"Client {self.client_id} Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")
                    
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate model on test data.
        
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        test_loss /= len(self.test_loader)
        accuracy = 100. * correct / total
        
        return test_loss, accuracy 