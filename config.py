"""
Configuration file for federated DiffPure + pFedDef defense
"""

import os
import torch
import argparse
from pathlib import Path
from dataclasses import dataclass

# Base paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
LOGS_DIR = BASE_DIR / 'logs'
CHECKPOINTS_DIR = BASE_DIR / 'checkpoints'

# Device configuration
USE_CUDA = True  # Set this to False to force CPU usage
DEVICE = torch.device('cuda' if (torch.cuda.is_available() and USE_CUDA) else 'cpu')

@dataclass
class Config:
    # Federated settings
    N_CLIENTS: int = 3
    N_LEARNERS: int = 3  # Default to 3 learners for full robustness
    DIRICHLET_ALPHA: float = 0.3
    
    # Training settings
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    N_ROUNDS: int = 100
    LOCAL_EPOCHS: int = 1
    LOCAL_STEPS_PER_EPOCH: int = 100  # Number of batches per epoch
    MAX_STEPS: int = 100  # Maximum number of steps for training
    USE_AMP: bool = False  # Use automatic mixed precision
    
    # Model settings
    RESNET_WIDTH: float = 1.0  # Width multiplier for ResNet models
    
    # DiffPure settings
    DIFFUSER_STEPS: int = 10
    DIFFUSER_SIGMA: float = 0.1
    
    # Attack settings
    PGD_STEPS: int = 40  # Default to 40 steps for full robustness
    PGD_EPS: float = 8/255
    PGD_ALPHA: float = 2/255
    
    # Defense settings
    LAMBDA_KL: float = 1.0  # Weight for KL divergence loss
    ENABLE_NEW_HOOK: bool = False  # Enable sample extension hook
    
    # Device
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    output_dir: str = 'results'
    model_dir: str = 'models'
    
    # Logging
    verbose: bool = True
    
    # Dataset setting
    dataset: str = 'cifar10'
    
    # Dataset settings
    DATASET_NAME: str = 'CIFAR10'
    DATA_PATH: str = './data'
    
    # MAE Detector settings
    ENABLE_MAE_DETECTOR: bool = False
    MAE_THRESHOLD: float = 0.1  
    DETECTOR_BATCH_SIZE: int = 32
    EPOCHS_DETECTOR: int = 1
    
    # Experiment name 
    experiment_name: str = 'default_experiment'

# Preset configurations exactly as specified
PRESETS = {
    'debug': {
        'N_CLIENTS': 2,
        'N_LEARNERS': 2,
        'N_ROUNDS': 2,
        'LOCAL_EPOCHS': 1,
        'BATCH_SIZE': 32,  # Updated from 8 for better batch statistics
        'DIFFUSER_STEPS': 4,  # Reduced steps for memory optimization
        'DIFFUSER_SIGMA': 0.04,
        'PGD_STEPS': 10,
        'PGD_EPS': 8/255,
        'PGD_ALPHA': 2/255,
        'LEARNING_RATE': 0.001,
        'LAMBDA_KL': 1.0,
        'DIRICHLET_ALPHA': 0.3,
        'LOCAL_STEPS_PER_EPOCH': 100,
        'MAX_STEPS': 10,
        'verbose': True,
        'output_dir': 'results',
        'model_dir': 'models',
        'dataset': 'cifar10',
        'experiment_name': 'debug_experiment',
        'ENABLE_NEW_HOOK': False,
        'USE_AMP': True,  # Enable automatic mixed precision for memory optimization
        'RESNET_WIDTH': 0.5,  # Use half-width ResNet models for memory optimization
        'DATASET_NAME': 'CIFAR10',   # options: "CIFAR10", "MedMNIST", etc.
        'DATA_PATH': './data',       # root folder for datasets
        
        # MAE Detector
        'ENABLE_MAE_DETECTOR': True,        # turn on/off MAE gatekeeper
        'MAE_THRESHOLD': 0.1,               # MSE cutoff for adversarial detection
        'DETECTOR_BATCH_SIZE': 32,          # batch size for detector training & inference
        'EPOCHS_DETECTOR': 1                # debug=1 / full=5
    },
    'full': {
        'N_CLIENTS': 5,
        'N_LEARNERS': 3,
        'N_ROUNDS': 100,
        'LOCAL_EPOCHS': 2,
        'BATCH_SIZE': 64,
        'DIFFUSER_STEPS': 10,
        'DIFFUSER_SIGMA': 0.1,
        'PGD_STEPS': 40,
        'PGD_EPS': 8/255,
        'PGD_ALPHA': 2/255,
        'LEARNING_RATE': 0.001,
        'LAMBDA_KL': 1.0,
        'DIRICHLET_ALPHA': 0.3,
        'LOCAL_STEPS_PER_EPOCH': 100,
        'MAX_STEPS': 100,
        'verbose': True,
        'output_dir': 'results',
        'model_dir': 'models',
        'dataset': 'cifar10',
        'experiment_name': 'full_experiment',
        'ENABLE_NEW_HOOK': False,
        'USE_AMP': False,
        'RESNET_WIDTH': 1.0,  # Use full-width ResNet models
        'DATASET_NAME': 'CIFAR10',   # options: "CIFAR10", "MedMNIST", etc.
        'DATA_PATH': './data',       # root folder for datasets
        
        # MAE Detector
        'ENABLE_MAE_DETECTOR': True,        # turn on/off MAE gatekeeper
        'MAE_THRESHOLD': 0.1,               # MSE cutoff for adversarial detection
        'DETECTOR_BATCH_SIZE': 32,          # batch size for detector training & inference
        'EPOCHS_DETECTOR': 5                # debug=1 / full=5
    }
}

def get_config(preset: str = None) -> Config:
    """Get configuration based on preset name."""
    cfg = Config()
    
    if preset and preset in PRESETS:
        # Apply preset values to the config
        preset_values = PRESETS[preset]
        for key, value in preset_values.items():
            setattr(cfg, key, value)
    
    return cfg

def parse_args():
    """Parse command-line arguments for configuration."""
    parser = argparse.ArgumentParser(description='Federated DiffPure + pFedDef defense')
    parser.add_argument('--preset', type=str, choices=['debug', 'full'], 
                      help='Configuration preset (debug or full)')
    
    return parser.parse_args()

# Print device information
if DEVICE.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("Using CPU")

# Create necessary directories
for path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
    if isinstance(path, Path):
        path.mkdir(parents=True, exist_ok=True)
    else:
        Path(path).mkdir(parents=True, exist_ok=True)

# Create model-specific directories
(MODELS_DIR / 'pfeddef').mkdir(exist_ok=True)
(MODELS_DIR / 'diffusion').mkdir(exist_ok=True)
(RESULTS_DIR / 'metrics').mkdir(exist_ok=True)
(RESULTS_DIR / 'plots').mkdir(exist_ok=True)
(LOGS_DIR / 'training').mkdir(exist_ok=True)