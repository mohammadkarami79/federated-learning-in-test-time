"""
Utility functions for integrating pFedDef and DiffPure defenses.

This module provides functions for loading pFedDef models, training them if needed,
and combining them with DiffPure diffusion models for a comprehensive defense strategy.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
from tqdm import tqdm
import logging
import time

from config import CONFIG, SAVE_PATH, DEVICE, FEDERATED_CONFIG
from diffusion_models import DiffusionModel, ScoreNetwork
from diffusion_manager import DiffusionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pfeddef_diffpure.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pFedDef-DiffPure")


class PFedDefHandler:
    """
    Handler for pFedDef models and their integration with DiffPure.
    """
    
    def __init__(self, dataset, device=DEVICE):
        """
        Initialize the handler.
        
        Args:
            dataset: Dataset name ('cifar10', 'mnist')
            device: Device to run models on ('cpu' or 'cuda')
        """
        self.dataset = dataset
        self.device = device
        self.pfeddef_enabled = CONFIG['pfeddef']['enabled']
        self.n_learners = CONFIG['pfeddef']['n_learners']
        self.client_weights_path = CONFIG['pfeddef']['client_weights_path']
        self.auto_train = CONFIG['pfeddef']['auto_train_models']
        self.models_cache = {}  # Cache loaded models
        
        # Create directory structure
        os.makedirs(self.client_weights_path, exist_ok=True)
        
    def get_model_path(self, model_name, client_id, learner_id=None):
        """
        Get the path for a specific client model.
        
        Args:
            model_name: Model architecture ('resnet18', 'vgg16', etc.)
            client_id: Client ID
            learner_id: Learner ID (None for non-mixture clients)
            
        Returns:
            Path to the model file
        """
        if learner_id is not None:
            # Path for a specific learner in the mixture
            return os.path.join(
                self.client_weights_path, 
                f'client_{client_id}', 
                f'{model_name}_{self.dataset}_learner{learner_id}.pth'
            )
        else:
            # Path for a single-model client
            return os.path.join(
                self.client_weights_path, 
                f'client_{client_id}', 
                f'{model_name}_{self.dataset}.pth'
            )
    
    def model_exists(self, model_name, client_id, learner_id=None):
        """
        Check if a model exists at the specified path.
        
        Args:
            model_name: Model architecture
            client_id: Client ID
            learner_id: Learner ID (None for non-mixture clients)
            
        Returns:
            Boolean indicating if the model exists
        """
        path = self.get_model_path(model_name, client_id, learner_id)
        return os.path.exists(path)
    
    def create_model_architecture(self, model_name, dataset_name):
        """
        Create a model with the specified architecture.
        
        Args:
            model_name: Model architecture ('resnet18', 'vgg16', etc.)
            dataset_name: Dataset name ('cifar10', 'mnist')
            
        Returns:
            Model object
        """
        # Set input channels based on dataset
        in_channels = 1 if dataset_name == 'mnist' else 3
        num_classes = 10  # Both MNIST and CIFAR-10 have 10 classes
        
        if model_name == 'resnet18':
            model = torchvision.models.resnet18(pretrained=True)
            # Modify the first layer for grayscale images if using MNIST
            if in_channels == 1:
                model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Modify the final layer for the correct number of classes
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'vgg16':
            model = torchvision.models.vgg16(pretrained=True)
            if in_channels == 1:
                model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        elif model_name == 'densenet121':
            model = torchvision.models.densenet121(pretrained=True)
            if in_channels == 1:
                model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model architecture: {model_name}")
        
        return model
    
    def load_model(self, model_name, client_id, learner_id=None):
        """
        Load a model for a specific client and learner.
        
        Args:
            model_name: Model architecture
            client_id: Client ID
            learner_id: Learner ID (None for non-mixture clients)
            
        Returns:
            Loaded model
        """
        # Create model architecture
        model = self.create_model_architecture(model_name, self.dataset)
        
        # Get model path
        model_path = self.get_model_path(model_name, client_id, learner_id)
        
        # Load weights if they exist
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded {model_name} for client {client_id}" + 
                     (f", learner {learner_id}" if learner_id is not None else "") +
                     f" from {model_path}")
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
                print(f"Using pretrained ImageNet weights for client {client_id}" +
                     (f", learner {learner_id}" if learner_id is not None else ""))
        else:
            print(f"No saved model found at {model_path}. " +
                 f"Using pretrained ImageNet weights for client {client_id}" +
                 (f", learner {learner_id}" if learner_id is not None else ""))
        
        # Move model to device
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def load_client_models(self, model_name, client_ids):
        """
        Load models for specified clients.
        
        Args:
            model_name: Model architecture
            client_ids: List of client IDs
            
        Returns:
            Dictionary mapping client IDs to their models
        """
        models = {}
        
        for client_id in client_ids:
            if self.pfeddef_enabled:
                # Load ensemble of learners for pFedDef
                learners = []
                for learner_id in range(self.n_learners):
                    learner = self.load_model(model_name, client_id, learner_id)
                    learners.append(learner)
                models[client_id] = learners
            else:
                # Load single model
                model = self.load_model(model_name, client_id)
                models[client_id] = model
        
        return models
    
    def train_model(self, model_name, client_id, learner_id=None):
        """
        Train a model for a specific client and learner if auto_train is enabled.
        In a real implementation, this would properly train using pFedDef methodology.
        
        Args:
            model_name: Model architecture
            client_id: Client ID
            learner_id: Learner ID (None for non-mixture clients)
            
        Returns:
            Trained model
        """
        if not self.auto_train:
            # If auto_train is disabled, just load the model
            return self.load_model(model_name, client_id, learner_id)
        
        print(f"Training {model_name} for client {client_id}" +
             (f", learner {learner_id}" if learner_id is not None else "") +
             "... This may take a while.")
        
        # In a real implementation, you would properly train using pFedDef
        # For now, we'll just create the model architecture and save it
        model = self.create_model_architecture(model_name, self.dataset)
        
        # Get model path
        model_path = self.get_model_path(model_name, client_id, learner_id)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        torch.save(model.state_dict(), model_path)
        
        print(f"Saved {model_name} for client {client_id}" +
             (f", learner {learner_id}" if learner_id is not None else "") +
             f" to {model_path}")
        
        # Move model to device
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def load_or_train_client_models(self, model_name, client_ids):
        """
        Load or train models for specified clients.
        
        Args:
            model_name: Model architecture
            client_ids: List of client IDs
            
        Returns:
            Dictionary mapping client IDs to their models
        """
        models = {}
        
        for client_id in client_ids:
            if self.pfeddef_enabled:
                # Load ensemble of learners for pFedDef
                learners = []
                for learner_id in range(self.n_learners):
                    if self.model_exists(model_name, client_id, learner_id):
                        learner = self.load_model(model_name, client_id, learner_id)
                    else:
                        learner = self.train_model(model_name, client_id, learner_id)
                    learners.append(learner)
                models[client_id] = learners
            else:
                # Load single model
                if self.model_exists(model_name, client_id):
                    model = self.load_model(model_name, client_id)
                else:
                    model = self.train_model(model_name, client_id)
                models[client_id] = model
        
        return models
    
    def pfeddef_predict(self, models, inputs):
        """
        Make predictions using pFedDef ensemble.
        
        Args:
            models: List of models for a client
            inputs: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Tensor of predicted labels
        """
        # Forward pass through each model
        outputs = []
        for model in models:
            model.eval()
            with torch.no_grad():
                output = model(inputs)
                outputs.append(output)
        
        # Stack outputs and aggregate (use median by default)
        outputs = torch.stack(outputs, dim=0)
        aggregated = torch.median(outputs, dim=0)[0]
        
        # Get predicted classes
        _, predicted = torch.max(aggregated, 1)
        
        return predicted


class PFedDefDiffPureHandler:
    """
    Handler for combined pFedDef and DiffPure defenses.
    """
    
    def __init__(self, dataset, device=DEVICE):
        """
        Initialize the handler.
        
        Args:
            dataset: Dataset name ('cifar10', 'mnist')
            device: Device to run on
        """
        self.dataset = dataset
        self.device = device
        self.pfeddef_handler = PFedDefHandler(dataset, device)
        self.diffusion_manager = DiffusionManager(dataset_name=dataset)
    
    def load_client_models_and_diffusion(self, model_name, client_ids, sigma, diffusion_steps):
        """
        Load both client models (pFedDef) and diffusion models.
        
        Args:
            model_name: Model architecture
            client_ids: List of client IDs
            sigma: Sigma parameter for diffusion
            diffusion_steps: Number of diffusion steps
            
        Returns:
            Tuple of (client_models, diffusion_models)
        """
        # Load client models
        client_models = self.pfeddef_handler.load_or_train_client_models(model_name, client_ids)
        
        # Load diffusion models
        diffusion_models = {}
        for client_id in client_ids:
            diffusion_model = self.diffusion_manager.load_diffusion_model(
                device=self.device,
                client_id=client_id,
                sigma=sigma,
                diffusion_steps=diffusion_steps
            )
            diffusion_models[client_id] = diffusion_model
        
        return client_models, diffusion_models
    
    def pfeddef_diffpure_predict(self, client_models, diffusion_model, inputs):
        """
        Make predictions using combined pFedDef and DiffPure defense.
        
        Args:
            client_models: List of client models (pFedDef ensemble)
            diffusion_model: Diffusion model for purification
            inputs: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Tensor of predicted labels
        """
        # Purify inputs using diffusion model
        purified_inputs = diffusion_model.purify(inputs)
        
        # Make predictions using pFedDef ensemble
        predictions = self.pfeddef_handler.pfeddef_predict(client_models, purified_inputs)
        
        return predictions, purified_inputs


def load_pfeddef_diffpure_models(model_name, dataset, client_ids, sigma=None, diffusion_steps=None, device=DEVICE):
    """
    Load all necessary models for pFedDef-DiffPure defense.
    
    Args:
        model_name: Model architecture ('resnet18', 'vgg16', etc.)
        dataset: Dataset name ('cifar10', 'mnist')
        client_ids: List of client IDs
        sigma: Sigma parameter for diffusion
        diffusion_steps: Number of diffusion steps
        device: Device to run on
        
    Returns:
        Tuple of (client_models, diffusion_models)
    """
    # Use default parameters if not specified
    if sigma is None:
        sigma = CONFIG['diffusion']['default_sigma']
    if diffusion_steps is None:
        diffusion_steps = CONFIG['diffusion']['default_steps']
    
    # Create handler
    handler = PFedDefDiffPureHandler(dataset, device)
    
    # Load models
    return handler.load_client_models_and_diffusion(model_name, client_ids, sigma, diffusion_steps)


def evaluate_with_pfeddef_diffpure(inputs, targets, client_models, diffusion_models, device=DEVICE):
    """
    Evaluate a batch using pFedDef-DiffPure defense.
    
    Args:
        inputs: Input tensor of shape [batch_size, channels, height, width]
        targets: Target tensor of shape [batch_size]
        client_models: Dictionary mapping client IDs to their models
        diffusion_models: Dictionary mapping client IDs to their diffusion models
        device: Device to run on
        
    Returns:
        Tuple of (accuracy, purified_inputs)
    """
    handler = PFedDefDiffPureHandler(None, device)
    
    # Choose a client (use the first one by default)
    client_id = list(client_models.keys())[0]
    client_model = client_models[client_id]
    diffusion_model = diffusion_models[client_id]
    
    # Get predictions
    with torch.no_grad():
        predictions, purified_inputs = handler.pfeddef_diffpure_predict(
            client_model, diffusion_model, inputs
        )
        
        # Calculate accuracy
        correct = (predictions == targets).sum().item()
        accuracy = 100.0 * correct / targets.size(0)
    
    return accuracy, purified_inputs 