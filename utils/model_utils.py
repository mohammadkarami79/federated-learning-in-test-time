"""
Model utilities for creating and training different models
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from pathlib import Path
import logging
from torchvision import transforms
from typing import Dict, Any, Optional
import torchvision.models as models
from config import DEVICE, CONFIG
import copy
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def create_model(
    model_name: str,
    num_classes: Optional[int] = None,
    pretrained: bool = False,
    **kwargs
) -> nn.Module:
    """
    Create a model
    
    Args:
        model_name: Name of the model
        num_classes: Number of classes (for classification models)
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments for model creation
        
    Returns:
        nn.Module: Created model
    """
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        if num_classes is not None:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'ddpm':
        model = create_diffusion_model(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(DEVICE)

def create_diffusion_model(
    sigma: float = 0.1,
    steps: int = 100,
    input_channels: int = 3,
    hidden_channels: int = 128,
    **kwargs
) -> nn.Module:
    """
    Create a diffusion model
    
    Args:
        sigma: Noise level
        steps: Number of diffusion steps
        input_channels: Number of input channels
        hidden_channels: Number of hidden channels
        **kwargs: Additional arguments
        
    Returns:
        nn.Module: Diffusion model
    """
    class DiffusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.sigma = sigma
            self.steps = steps
            
            # U-Net architecture
            self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            
            self.middle = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels * 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
                nn.ReLU()
            )
            
            self.decoder = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, input_channels, 3, padding=1)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass"""
            # Add noise
            noise = torch.randn_like(x) * self.sigma
            noisy_x = x + noise
            
            # Encode
            h = self.encoder(noisy_x)
            
            # Middle
            h = self.middle(h)
            
            # Decode
            output = self.decoder(h)
            
            return output
        
        def purify(self, x: torch.Tensor, steps: Optional[int] = None, sigma: Optional[float] = None) -> torch.Tensor:
            """
            Purify input images
            
            Args:
                x: Input images
                steps: Number of steps (optional)
                sigma: Noise level (optional)
                
            Returns:
                torch.Tensor: Purified images
            """
            steps = steps or self.steps
            sigma = sigma or self.sigma
            
            x = x.clone()
            for _ in range(steps):
                # Add noise
                noise = torch.randn_like(x) * sigma
                noisy_x = x + noise
                
                # Denoise
                x = self.forward(noisy_x)
            
            return x
    
    return DiffusionModel()

def load_model(path: str, model: Optional[nn.Module] = None) -> nn.Module:
    """
    Load model weights
    
    Args:
        path: Path to model weights
        model: Model to load weights into (optional)
        
    Returns:
        nn.Module: Model with loaded weights
    """
    state_dict = torch.load(path)
    
    if model is None:
        # Try to determine model type from state dict
        if any('fc' in key for key in state_dict.keys()):
            model = models.resnet18(pretrained=False)
            if state_dict['fc.weight'].shape[0] != model.fc.out_features:
                model.fc = nn.Linear(model.fc.in_features, state_dict['fc.weight'].shape[0])
        else:
            raise ValueError("Could not determine model type from state dict")
    
    model.load_state_dict(state_dict)
    return model

def save_model(model: nn.Module, path: str) -> None:
    """
    Save model weights
    
    Args:
        model: Model to save
        path: Path to save to
    """
    torch.save(model.state_dict(), path)

def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get model information
    
    Args:
        model: Model to get information for
        
    Returns:
        Dict[str, Any]: Model information
    """
    info = {
        'name': model.__class__.__name__,
        'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    # Try to get more specific information
    if isinstance(model, models.ResNet):
        info.update({
            'type': 'classification',
            'num_classes': model.fc.out_features
        })
    elif hasattr(model, 'sigma') and hasattr(model, 'steps'):
        info.update({
            'type': 'diffusion',
            'sigma': model.sigma,
            'steps': model.steps
        })
    
    return info

def verify_model_performance(model, device, num_samples=1000):
    """
    Verify model performance on a small test set
    
    Args:
        model: Model to verify
        device: Device to run on
        num_samples: Number of samples to test
        
    Returns:
        float: Model accuracy
    """
    # Load test data
    test_loader = create_test_loader(num_samples)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = correct / total
    return accuracy

def create_test_loader(num_samples):
    """
    Create a test loader with specified number of samples
    
    Args:
        num_samples: Number of samples to include
        
    Returns:
        DataLoader: Test data loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Take subset of test data
    indices = torch.randperm(len(test_dataset))[:num_samples]
    test_subset = torch.utils.data.Subset(test_dataset, indices)
    
    return torch.utils.data.DataLoader(
        test_subset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

def get_target_model(name='resnet18', pretrained=True):
    """
    Get target model
    """
    if name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 has 10 classes
    else:
        raise ValueError(f"Unknown model: {name}")
    return model

def get_diffusion_model(in_channels=3, hidden_channels=32, num_blocks=2, sigma=0.1):
    """
    Get diffusion model
    
    Args:
        in_channels: Number of input channels
        hidden_channels: Number of hidden channels
        num_blocks: Number of residual blocks
        sigma: Noise level for diffusion
    """
    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            identity = x
            out = self.relu(self.conv1(x))
            out = self.conv2(out)
            out += identity
            out = self.relu(out)
            return out
    
    class DiffusionModel(nn.Module):
        def __init__(self, in_channels, hidden_channels, num_blocks, sigma):
            super().__init__()
            self.in_channels = in_channels
            self.hidden_channels = hidden_channels
            self.num_blocks = num_blocks
            self.sigma = sigma
            
            # Initial convolution
            self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
            
            # Residual blocks
            self.blocks = nn.ModuleList([
                ResidualBlock(hidden_channels) for _ in range(num_blocks)
            ])
            
            # Final convolution
            self.conv2 = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)
            
        def forward(self, x):
            # Initial convolution
            x = self.conv1(x)
            
            # Residual blocks
            for block in self.blocks:
                x = block(x)
            
            # Final convolution
            x = self.conv2(x)
            
            return x
        
        def purify(self, x, steps=20, sigma=None):
            """
            Purify input images
            """
            sigma = sigma or self.sigma
            x = x.clone()
            for _ in range(steps):
                # Add noise
                noise = torch.randn_like(x) * sigma
                noisy_x = x + noise
                
                # Denoise
                x = self.forward(noisy_x)
            
            return x
    
    return DiffusionModel(in_channels, hidden_channels, num_blocks, sigma)

def get_pfeddef_model(name='resnet18', n_learners=2, num_classes=10, weights=None):
    """
    Create a pFedDef model
    
    Args:
        name: Base model name
        n_learners: Number of learners
        num_classes: Number of output classes
        weights: Pre-trained weights to use
        
    Returns:
        nn.Module: pFedDef model
    """
    class pFedDefModel(nn.Module):
        def __init__(self, n_learners=2, num_classes=10, dropout_rate=0.3, attention_dropout=0.2):
            super(pFedDefModel, self).__init__()
            self.n_learners = n_learners
            self.num_classes = num_classes
            self.dropout_rate = dropout_rate
            self.attention_dropout = attention_dropout
            
            # Create base model
            if name == 'resnet18':
                self.base_model = models.resnet18(weights=weights)
                self.feature_dim = self.base_model.fc.in_features
                self.base_model.fc = nn.Identity()
            else:
                raise ValueError(f"Unsupported model name: {name}")
            
            # Create learners with consistent architecture
            self.learners = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.feature_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(256, num_classes)
                ) for _ in range(n_learners)
            ])
            
            # Create attention mechanisms with consistent architecture
            self.attention = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.feature_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(attention_dropout),
                    nn.Linear(256, 1)
                ) for _ in range(n_learners)
            ])
            
            # Initialize mixture weights
            self.mixture_weights = nn.Parameter(torch.ones(n_learners) / n_learners)
        
        def get_params(self):
            """Get model parameters for initialization"""
            return {
                'n_learners': self.n_learners,
                'num_classes': self.num_classes,
                'dropout_rate': self.dropout_rate,
                'attention_dropout': self.attention_dropout
            }
        
        def forward(self, x, is_training=False, client_id=None):
            """
            Forward pass
            
            Args:
                x: Input tensor
                is_training: Whether in training mode
                client_id: Specific learner ID to use (if None, use mixture)
                
            Returns:
                torch.Tensor: Model output
            """
            # Get features from base model
            features = self.base_model(x)  # [B, feature_dim]
            
            if client_id is not None:
                # Use specific learner
                output = self.learners[client_id](features)  # [B, num_classes]
                attention = torch.sigmoid(self.attention[client_id](features))  # [B, 1]
                return output * attention
            
            # Get predictions from all learners
            outputs = []
            attentions = []
            
            for i in range(self.n_learners):
                output = self.learners[i](features)  # [B, num_classes]
                attention = torch.sigmoid(self.attention[i](features))  # [B, 1]
                outputs.append(output)
                attentions.append(attention)
            
            # Stack outputs and attentions
            outputs = torch.stack(outputs, dim=0)  # [n_learners, B, num_classes]
            attentions = torch.stack(attentions, dim=0)  # [n_learners, B, 1]
            
            # Compute weighted output
            if is_training:
                # During training, use mixture weights
                weights = F.softmax(self.mixture_weights, dim=0)  # [n_learners]
                weighted_output = (outputs * weights.view(-1, 1, 1)).sum(dim=0)  # [B, num_classes]
            else:
                # During inference, use attention weights
                weights = F.softmax(attentions.squeeze(-1), dim=0)  # [n_learners, B]
                weighted_output = (outputs * weights.unsqueeze(-1)).sum(dim=0)  # [B, num_classes]
            
            return weighted_output
    
    return pFedDefModel(n_learners=n_learners, num_classes=num_classes)

def train_pfeddef_model(model, train_loader, test_loader, attack, global_epochs=1, local_epochs=1, learning_rate=0.001, device='cuda'):
    """
    Train pFedDef model with improved training process
    
    Args:
        model: pFedDef model
        train_loader: Training data loader
        test_loader: Test data loader
        attack: Adversarial attack object
        global_epochs: Number of global aggregation rounds
        local_epochs: Number of local training epochs
        learning_rate: Learning rate
        device: Device to train on
    """
    model = model.to(device)
    model.train()
    
    # Initialize optimizer with weight decay
    optimizer = torch.optim.SGD([
        {'params': model.learners[0].parameters(), 'lr': learning_rate},
        {'params': model.attention[0].parameters(), 'lr': learning_rate * 0.1},
        {'params': model.mixture_weights, 'lr': learning_rate * 0.01},
        {'params': model.normalize[0].parameters(), 'lr': learning_rate * 0.1},
        {'params': model.feature_extractors[0].parameters(), 'lr': learning_rate * 0.1}
    ], momentum=0.9, weight_decay=CONFIG['model']['pfeddef']['weight_decay'])
    
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
        eta_min=CONFIG['training']['scheduler']['min_lr']
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=CONFIG['defense']['patience'],
        min_delta=CONFIG['defense']['min_delta'],
        mode='max'
    )
    
    # Loss functions
    criterion = nn.CrossEntropyLoss()
    kl_div = nn.KLDivLoss(reduction='batchmean')
    
    # Training metrics
    best_acc = 0.0
    train_metrics = {'loss': [], 'acc': [], 'rob_acc': []}
    val_metrics = {'loss': [], 'acc': [], 'rob_acc': []}
    
    # Global training loop
    for global_epoch in range(global_epochs):
        print(f"\nGlobal Epoch: {global_epoch + 1}/{global_epochs}")
        
        # Local training loop
        for local_epoch in range(local_epochs):
            print(f"Local Epoch: {local_epoch + 1}/{local_epochs}")
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Use tqdm for progress bar
            pbar = tqdm(train_loader, desc=f"Training")
            for batch_idx, (data, target) in enumerate(pbar):
                try:
                    data, target = data.to(device), target.to(device)
                    
                    # Generate adversarial examples
                    if attack is not None:
                        adv_data = attack.attack(data, target)
                    else:
                        adv_data = data
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass with clean and adversarial data
                    outputs_clean = model(data, is_training=True, client_id=0)
                    outputs_adv = model(adv_data, is_training=True, client_id=0)
                    
                    # Calculate losses
                    loss_clean = criterion(outputs_clean, target)
                    loss_adv = criterion(outputs_adv, target)
                    
                    # KL divergence between clean and adversarial predictions
                    kl_clean_adv = kl_div(
                        F.log_softmax(outputs_adv, dim=1),
                        F.softmax(outputs_clean, dim=1)
                    )
                    
                    # Total loss with regularization
                    loss = (
                        0.5 * (loss_clean + loss_adv) +  # Classification loss
                        0.1 * kl_clean_adv +  # Consistency regularization
                        0.01 * torch.norm(model.mixture_weights, p=2)  # L2 regularization on mixture weights
                    )
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        CONFIG['model']['pfeddef']['gradient_clip_norm']
                    )
                    
                    # Optimize
                    optimizer.step()
                    
                    # Update statistics
                    running_loss += loss.item()
                    _, predicted = outputs_clean.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'Loss': f"{running_loss/(batch_idx+1):.4f}",
                        'Acc': f"{100.*correct/total:.2f}%"
                    })
                    
                except RuntimeError as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    continue
            
            # Calculate epoch metrics
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100. * correct / total
            
            # Update learning rate
            scheduler.step()
            
            # Print epoch statistics
            print(f"\nEpoch {local_epoch + 1}/{local_epochs}")
            print(f"Training Loss: {epoch_loss:.4f}")
            print(f"Training Accuracy: {epoch_acc:.2f}%")
            
            # Validate
            val_metrics = validate_model(
                model=model,
                test_loader=test_loader,
                criterion=criterion,
                attack=attack,
                device=device
            )
            
            print(f"Validation Loss: {val_metrics['loss']:.4f}")
            print(f"Validation Accuracy: {val_metrics['acc']:.2f}%")
            print(f"Validation Robustness: {val_metrics['rob_acc']:.2f}%")
            
            # Save best model
            if val_metrics['acc'] > best_acc:
                best_acc = val_metrics['acc']
                torch.save({
                    'epoch': global_epoch * local_epochs + local_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc,
                    'val_metrics': val_metrics
                }, 'checkpoints/pfeddef_best.pth')
                print(f"Saved best model with accuracy: {best_acc:.2f}%")
            
            # Early stopping check
            if early_stopping(val_metrics['acc']):
                print("Early stopping triggered")
                return model
            
            model.train()
    
    return model

class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
    
    def __call__(self, value):
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'max':
            if value > self.best_value + self.min_delta:
                self.best_value = value
                self.counter = 0
            else:
                self.counter += 1
        else:
            if value < self.best_value - self.min_delta:
                self.best_value = value
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            return True
        return False

def validate_model(model, test_loader, criterion, attack, device):
    """Validate model with comprehensive metrics"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    rob_correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Clean accuracy
            outputs = model(data, is_training=False)
            val_loss += criterion(outputs, target).item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Robustness accuracy
            if attack is not None:
                adv_data = attack.attack(data, target)
                adv_outputs = model(adv_data, is_training=False)
                _, adv_predicted = adv_outputs.max(1)
                rob_correct += adv_predicted.eq(target).sum().item()
    
    metrics = {
        'loss': val_loss / len(test_loader),
        'acc': 100. * correct / total,
        'rob_acc': 100. * rob_correct / total if attack is not None else 0.0
    }
    
    return metrics

def compute_model_similarities(learners):
    """
    Compute pairwise similarities between learners
    """
    n_learners = len(learners)
    similarities = torch.zeros((n_learners, n_learners))
    
    for i in range(n_learners):
        for j in range(n_learners):
            if i != j:
                # Compute cosine similarity of model parameters
                params_i = torch.cat([p.view(-1) for p in learners[i].parameters()])
                params_j = torch.cat([p.view(-1) for p in learners[j].parameters()])
                similarities[i,j] = F.cosine_similarity(params_i.unsqueeze(0), params_j.unsqueeze(0))
    
    return F.softmax(similarities, dim=1)

def evaluate_ensemble(model, test_loader, device, diffusion_model=None, defense_type='pfeddef', return_metrics=False):
    """
    Evaluate ensemble model on test data with comprehensive metrics
    
    Args:
        model: pFedDef model
        test_loader: Test data loader
        device: Device to evaluate on
        diffusion_model: DiffPure model for purification (optional)
        defense_type: Type of defense ('pfeddef', 'diffpure', or 'combined')
        return_metrics: Whether to return detailed metrics
        
    Returns:
        If return_metrics is False:
            float: Overall accuracy
        If return_metrics is True:
            dict: Dictionary containing various metrics
    """
    model.eval()
    if diffusion_model is not None:
        diffusion_model.eval()
    
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    all_confidences = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Apply DiffPure purification if using combined defense
            if defense_type == 'combined' and diffusion_model is not None:
                data = diffusion_model.purify(data)
            
            # Get predictions from each learner
            outputs = []
            for i in range(model.n_learners):
                output = model(data, is_training=False, client_id=i)
                outputs.append(output)
            
            # Stack outputs [n_learners, batch_size, num_classes]
            outputs = torch.stack(outputs, dim=0)
            
            # Get ensemble prediction
            probs = F.softmax(outputs, dim=2)
            weights = F.softmax(model.mixture_weights, dim=0)
            weighted_probs = (probs * weights.view(-1, 1, 1)).sum(dim=0)
            
            # Get predictions and confidences
            confidences, predicted = weighted_probs.max(1)
            
            # Update statistics
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Store predictions and targets for metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    # Calculate overall accuracy
    accuracy = 100. * correct / total
    
    if not return_metrics:
        return accuracy
    
    # Calculate additional metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_confidences = np.array(all_confidences)
    
    # Calculate per-class accuracy
    num_classes = model.learners[0].fc.out_features
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    
    for i in range(num_classes):
        mask = all_targets == i
        class_total[i] = np.sum(mask)
        if class_total[i] > 0:
            class_correct[i] = np.sum((all_predictions == i) & mask)
    
    class_accuracy = 100. * class_correct / (class_total + 1e-8)
    
    # Calculate confidence metrics
    correct_mask = all_predictions == all_targets
    correct_confidences = all_confidences[correct_mask]
    incorrect_confidences = all_confidences[~correct_mask]
    
    metrics = {
        'overall_accuracy': accuracy,
        'class_accuracy': class_accuracy.tolist(),
        'mean_confidence': np.mean(all_confidences),
        'correct_confidence': np.mean(correct_confidences) if len(correct_confidences) > 0 else 0,
        'incorrect_confidence': np.mean(incorrect_confidences) if len(incorrect_confidences) > 0 else 0,
        'confidence_gap': np.mean(correct_confidences) - np.mean(incorrect_confidences) if len(correct_confidences) > 0 and len(incorrect_confidences) > 0 else 0
    }
    
    return metrics

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate a single model on test data
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        float: Average loss
    """
    model.eval()
    total_loss = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * target.size(0)
            total += target.size(0)
    
    return total_loss / total

def train_diffusion_model(model, train_loader, test_loader=None, epochs=10, learning_rate=1e-4, device='cuda'):
    """
    Train the diffusion model using denoising score matching.
    
    Args:
        model: Diffusion model to train
        train_loader: Training data loader
        test_loader: Test data loader (optional)
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        Trained model
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_loss = float('inf')
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            data = data.to(device)
            batch_size = data.shape[0]
            
            # Sample random timesteps
            t = torch.rand(batch_size, device=device)
            
            # Compute alpha_t
            alpha_t = torch.exp(-0.5 * (t * model.sigma + 0.5 * t**2 * (20.0 - model.sigma)))
            
            # Sample noise
            epsilon = torch.randn_like(data)
            
            # Create noisy images
            x_t = alpha_t.view(-1, 1, 1, 1) * data + torch.sqrt(1 - alpha_t.view(-1, 1, 1, 1)**2) * epsilon
            
            # Predict noise
            predicted_noise = model(x_t)
            
            # Compute loss (denoising score matching)
            loss = F.mse_loss(predicted_noise, epsilon)
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
        
        # Validation
        if test_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, _ in test_loader:
                    data = data.to(device)
                    batch_size = data.shape[0]
                    
                    # Sample random timesteps
                    t = torch.rand(batch_size, device=device)
                    
                    # Compute alpha_t
                    alpha_t = torch.exp(-0.5 * (t * model.sigma + 0.5 * t**2 * (20.0 - model.sigma)))
                    
                    # Sample noise
                    epsilon = torch.randn_like(data)
                    
                    # Create noisy images
                    x_t = alpha_t.view(-1, 1, 1, 1) * data + torch.sqrt(1 - alpha_t.view(-1, 1, 1, 1)**2) * epsilon
                    
                    # Predict noise
                    predicted_noise = model(x_t)
                    
                    # Compute loss
                    loss = F.mse_loss(predicted_noise, epsilon)
                    val_loss += loss.item()
            
            val_loss /= len(test_loader)
            print(f"Validation Loss: {val_loss:.6f}")
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = model.state_dict()
    
    # Load best model if validation was performed
    if test_loader is not None:
        model.load_state_dict(best_state)
    
    return model