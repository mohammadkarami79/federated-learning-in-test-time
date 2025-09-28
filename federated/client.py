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
        """Create local model - use simple ResNet18."""
        try:
            # Use simple ResNet18 for stability
            import torchvision.models as models
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, getattr(self.cfg, 'NUM_CLASSES', 10))
            return model.to(self.device)
        except Exception as e:
            # Fallback to basic model
            from models import get_model
            model = get_model(self.cfg)
            return model.to(self.device)
    
    def get_models_for_server(self) -> List[nn.Module]:
        """Get models to send to server - simple ResNet18."""
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
            weight_decay=5e-4
        )
        
        # Add learning rate scheduler for better convergence
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        
        # Training loop with improved monitoring
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Iterate over the training data (no artificial limits)
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                num_classes = getattr(self.cfg, 'NUM_CLASSES', 10)
                if target.min() < 0 or target.max() >= num_classes:
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
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
            
            # Step the scheduler
            scheduler.step()
            
            # Print epoch summary
            epoch_loss = running_loss / max(1, len(self.train_loader))
            epoch_acc = 100. * correct / max(1, total)
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
        
        # Toggle back to train
        self.model.train()
        
        test_loss /= len(self.test_loader)
        accuracy = 100. * correct / total
        
        return test_loss, accuracy 