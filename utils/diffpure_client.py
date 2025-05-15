import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

class DiffPureClient:
    def __init__(self, dataset_name='cifar10', defense_params=None):
        self.dataset_name = dataset_name
        self.defense_params = defense_params or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.model = self._create_diffusion_model()
        self.defense_model = None
        
    def _create_diffusion_model(self):
        """Create the diffusion model architecture."""
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        ).to(self.device)
        return model
    
    def train(self, epochs=3, learning_rate=0.001):
        """Train the diffusion model."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Get training data
        train_loader = self._get_data_loader(train=True)
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(self.device)
                
                # Add noise to data
                noisy_data = self._add_noise(data)
                
                # Forward pass
                optimizer.zero_grad()
                output = self.model(noisy_data)
                loss = criterion(output, data)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    logger.info(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
            
            logger.info(f"Epoch {epoch} completed. Average loss: {epoch_loss/len(train_loader):.4f}")
    
    def evaluate(self, target_model):
        """Evaluate the defense on the target model."""
        self.model.eval()
        target_model.eval()
        
        test_loader = self._get_data_loader(train=False)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Purify data using diffusion model
                purified_data = self.model(data)
                
                # Get predictions from target model
                outputs = target_model(purified_data)
                _, predicted = outputs.max(1)
                
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def _get_data_loader(self, train=True):
        """Get data loader for the specified dataset."""
        if self.dataset_name == 'cifar10':
            from torchvision import datasets, transforms
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            
            dataset = datasets.CIFAR10(
                root='./data',
                train=train,
                download=True,
                transform=transform
            )
            
            return DataLoader(
                dataset,
                batch_size=64,
                shuffle=True,
                num_workers=2
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def _add_noise(self, data, sigma=0.1):
        """Add noise to the data."""
        noise = torch.randn_like(data) * sigma
        return data + noise
    
    def purify(self, images):
        """Purify images using the diffusion model."""
        self.model.eval()
        with torch.no_grad():
            purified = self.model(images)
        return purified 