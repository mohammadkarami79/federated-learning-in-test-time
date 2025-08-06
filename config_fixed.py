"""
OPTIMIZED CONFIGURATION FOR PFEDDEF + DIFFPURE SYSTEM
Designed for 10x performance improvement while maintaining functionality
FIXED: Improved hyperparameters, fixed full config, added proper training parameters
"""

import torch
import os
from types import SimpleNamespace

# ============================================================================
# OPTIMIZED PARAMETERS FOR PERFORMANCE
# ============================================================================

# Core experiment settings
N_CLIENTS = 10
N_TASKS = 10  # Reduced from 80 tasks for faster testing
N_ROUNDS = 10  # Start with 10 rounds for testing
LOCAL_STEPS_PER_EPOCH = 10  # Reduced from 100 (10x speedup)
EVAL_INTERVAL = 2  # Evaluate every 2 rounds instead of every round

# Dataset settings
DATASET = "cifar10"
N_CLASSES = 10
IMG_SIZE = 32
IMG_CHANNELS = 3
DATA_PATH = "data"

# Device and performance settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_ENABLED = True  # Mixed precision for memory efficiency
NUM_WORKERS = 2  # Parallel data loading
PIN_MEMORY = True

# Learning settings - OPTIMIZED
LEARNING_RATE = 0.005  
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
GRADIENT_CLIPPING = 1.0  # Prevent gradient explosion

# Model architecture - OPTIMIZED  
MODEL_TYPE = "resnet18"
RESNET_WIDTH = 0.25  # Reduced model size (4x speedup)
DROPOUT_RATE = 0.1

# PGD Attack settings - OPTIMIZED
PGD_STEPS = 5  # Reduced from 10 (2x speedup)
PGD_EPS = 8/255
PGD_ALPHA = 2/255
PGD_RANDOM_START = True

# DiffPure settings - OPTIMIZED
DIFFUSION_STEPS = 2  # Reduced from 4 (2x speedup)
DIFFUSION_NOISE_SCHEDULE = "linear"
DIFFUSION_TIMESTEPS = 100  # Reduced for speed
DIFFUSION_BETA_START = 1e-4
DIFFUSION_BETA_END = 2e-2

# MAE Detector settings - NEW
MAE_ENABLED = True
MAE_THRESHOLD = 0.1
MAE_PATCH_SIZE = 4
MAE_EMBED_DIM = 128  # Reduced for speed
MAE_DEPTH = 4  # Reduced from typical 12
MAE_NUM_HEADS = 4
MAE_MASK_RATIO = 0.5
MAE_DECODER_EMBED_DIM = 64

# FedEM aggregation settings
FEDEM_N_LEARNERS = 3
FEDEM_SAMPLING_RATE = 1.0

# Logging and saving
VERBOSE = True
SAVE_PATH = "results/optimized_pfeddef"
LOG_LEVEL = "INFO"
SAVE_FREQ = 5  # Save every 5 rounds

# Checkpointing
CHECKPOINT_DIR = "checkpoints/optimized"
RESUME_TRAINING = False
CHECKPOINT_FREQ = 5

# Memory optimization
MAX_MEMORY_GB = 4.0  # Target max memory usage
CLEANUP_CACHE_FREQ = 2  # Clear cache every 2 rounds

# Simplified training mode for debugging
SIMPLIFIED_TRAINING = True  # Skip some complex operations for testing
FAST_MODE = True  # Enable all speed optimizations

# Training parameters - FIXED: Added proper training parameters
DIFFUSION_EPOCHS = 10  # Configurable diffusion training epochs
MAE_EPOCHS = 10        # Configurable MAE training epochs
CLIENT_EPOCHS = 3      # Configurable client training epochs
DIFFUSION_HIDDEN_CHANNELS = 64  # Configurable diffusion model size

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

def get_debug_config():
    """Get debug configuration with improved parameters for better accuracy"""
    cfg = type('Config', (), {})()
    
    # Basic settings
    cfg.MODE = 'debug'
    cfg.DATASET = 'CIFAR10'
    cfg.DATASET_NAME = 'CIFAR-10'
    cfg.IMG_CHANNELS = 3
    cfg.IMG_SIZE = 32
    cfg.NUM_CLASSES = 10
    
    # Training parameters - IMPROVED for better accuracy
    cfg.BATCH_SIZE = 32  # Reduced for better gradient estimates
    cfg.LEARNING_RATE = 0.01  # INCREASED from 0.001 to 0.01
    cfg.EPOCHS = 10  # INCREASED from 5 to 10
    cfg.N_CLIENTS = 5
    cfg.N_ROUNDS = 5  # INCREASED from 3 to 5
    cfg.CLIENT_EPOCHS = 3  # INCREASED from 1 to 3
    cfg.DIFFUSION_EPOCHS = 10  # INCREASED from 5 to 10
    cfg.MAE_EPOCHS = 10  # INCREASED from 5 to 10
    cfg.DIFFUSION_HIDDEN_CHANNELS = 64
    
    # Model parameters
    cfg.MODEL_WIDTH = 1.0
    cfg.USE_ADDITIONAL_LAYERS = False
    cfg.N_LEARNERS = 2  # Added for pFedDef compatibility
    
    # Defense parameters
    cfg.MAE_EMBED_DIM = 128
    cfg.MAE_DECODER_EMBED_DIM = 128
    cfg.MAE_MASK_RATIO = 0.75
    cfg.MAE_PATCH_SIZE = 4
    cfg.MAE_THRESHOLD = 0.1
    
    # Attack parameters
    cfg.PGD_EPS = 8/255
    cfg.PGD_ALPHA = 2/255
    cfg.PGD_STEPS = 10
    cfg.PGD_RANDOM_START = True
    cfg.ATTACK_EPSILON = 0.3
    cfg.ATTACK_ALPHA = 0.01
    cfg.ATTACK_STEPS = 10
    
    # Memory optimization
    cfg.MAX_MEMORY_GB = 2.0
    cfg.USE_AMP = True
    
    # Device
    cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return cfg

def get_test_config():
    """Get test configuration with moderate parameters"""
    cfg = type('Config', (), {})()
    
    # Basic settings
    cfg.MODE = 'test'
    cfg.DATASET = 'CIFAR100'
    cfg.DATASET_NAME = 'CIFAR-100'
    cfg.IMG_CHANNELS = 3
    cfg.IMG_SIZE = 32
    cfg.NUM_CLASSES = 100
    
    # Training parameters
    cfg.BATCH_SIZE = 128
    cfg.LEARNING_RATE = 0.0005
    cfg.EPOCHS = 10
    cfg.N_CLIENTS = 10
    cfg.N_ROUNDS = 5
    cfg.CLIENT_EPOCHS = 2
    cfg.DIFFUSION_EPOCHS = 10
    cfg.MAE_EPOCHS = 10
    cfg.DIFFUSION_HIDDEN_CHANNELS = 128
    
    # Model parameters
    cfg.MODEL_WIDTH = 1.0
    cfg.USE_ADDITIONAL_LAYERS = False
    cfg.N_LEARNERS = 2  # Added for pFedDef compatibility
    
    # Defense parameters
    cfg.MAE_EMBED_DIM = 256
    cfg.MAE_DECODER_EMBED_DIM = 256
    cfg.MAE_MASK_RATIO = 0.75
    cfg.MAE_PATCH_SIZE = 4
    
    # Attack parameters
    cfg.PGD_EPS = 8/255
    cfg.PGD_ALPHA = 2/255
    cfg.PGD_STEPS = 20
    cfg.PGD_RANDOM_START = True
    cfg.ATTACK_EPSILON = 0.3
    cfg.ATTACK_ALPHA = 0.01
    cfg.ATTACK_STEPS = 20
    
    # Memory optimization
    cfg.MAX_MEMORY_GB = 4.0
    cfg.USE_AMP = True
    
    # Device
    cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return cfg

def get_full_config():
    """Get full configuration with production parameters"""
    cfg = type('Config', (), {})()
    
    # Basic settings
    cfg.MODE = 'full'
    cfg.DATASET = 'MNIST'
    cfg.DATASET_NAME = 'MNIST'
    cfg.IMG_CHANNELS = 1
    cfg.IMG_SIZE = 28
    cfg.NUM_CLASSES = 10
    
    # Training parameters - PROFESSIONAL VALUES FOR PAPER QUALITY (STANDARD)
    cfg.BATCH_SIZE = 128  # Standard for papers (not too large, not too small)
    cfg.LEARNING_RATE = 0.001  # Standard learning rate for federated learning
    cfg.EPOCHS = 30
    cfg.N_CLIENTS = 15
    cfg.N_ROUNDS = 20
    cfg.CLIENT_EPOCHS = 15
    cfg.DIFFUSION_EPOCHS = 50
    cfg.MAE_EPOCHS = 30
    cfg.DIFFUSION_HIDDEN_CHANNELS = 256  # REDUCED: For memory compatibility
    
    # Model parameters
    cfg.MODEL_WIDTH = 1.0
    cfg.USE_ADDITIONAL_LAYERS = True
    cfg.N_LEARNERS = 3  # Added for pFedDef compatibility
    
    # Defense parameters - PROFESSIONAL VALUES FOR PAPER QUALITY (STANDARD)
    cfg.MAE_EMBED_DIM = 512  # Balanced between quality and speed
    cfg.MAE_DECODER_EMBED_DIM = 512  # Balanced between quality and speed
    cfg.MAE_MASK_RATIO = 0.75
    cfg.MAE_PATCH_SIZE = 4
    
    # Attack parameters - PROFESSIONAL VALUES FOR PAPER QUALITY
    cfg.PGD_EPS = 8/255
    cfg.PGD_ALPHA = 2/255
    cfg.PGD_STEPS = 100
    cfg.PGD_RANDOM_START = True
    cfg.ATTACK_EPSILON = 0.3
    cfg.ATTACK_ALPHA = 0.01
    cfg.ATTACK_STEPS = 100
    
    # Memory optimization
    cfg.MAX_MEMORY_GB = 8.0
    cfg.USE_AMP = True
    
    # Device
    cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return cfg

# ============================================================================
# VALIDATION AND HELPERS
# ============================================================================

def validate_config(cfg):
    """Validate configuration parameters"""
    required_fields = [
        'MODE', 'DATASET', 'DATASET_NAME', 'IMG_CHANNELS', 'IMG_SIZE', 'NUM_CLASSES',
        'BATCH_SIZE', 'LEARNING_RATE', 'EPOCHS', 'N_CLIENTS', 'N_ROUNDS', 'CLIENT_EPOCHS',
        'DIFFUSION_EPOCHS', 'MAE_EPOCHS', 'DIFFUSION_HIDDEN_CHANNELS', 'DEVICE'
    ]
    
    for field in required_fields:
        if not hasattr(cfg, field):
            raise ValueError(f"Missing required config field: {field}")
    
    # Validate numeric ranges
    if cfg.BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be positive")
    if cfg.LEARNING_RATE <= 0:
        raise ValueError("LEARNING_RATE must be positive")
    if cfg.EPOCHS <= 0:
        raise ValueError("EPOCHS must be positive")
    if cfg.N_CLIENTS <= 0:
        raise ValueError("N_CLIENTS must be positive")
    if cfg.N_ROUNDS <= 0:
        raise ValueError("N_ROUNDS must be positive")
    if cfg.CLIENT_EPOCHS <= 0:
        raise ValueError("CLIENT_EPOCHS must be positive")
    if cfg.DIFFUSION_EPOCHS <= 0:
        raise ValueError("DIFFUSION_EPOCHS must be positive")
    if cfg.MAE_EPOCHS <= 0:
        raise ValueError("MAE_EPOCHS must be positive")
    if cfg.DIFFUSION_HIDDEN_CHANNELS <= 0:
        raise ValueError("DIFFUSION_HIDDEN_CHANNELS must be positive")
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if hasattr(cfg, 'MAX_MEMORY_GB') and cfg.MAX_MEMORY_GB > gpu_memory:
            print(f"Warning: MAX_MEMORY_GB ({cfg.MAX_MEMORY_GB}) exceeds GPU memory ({gpu_memory:.1f}GB)")
    
    return True

def get_memory_optimized_config(max_memory_gb=3.0):
    """Get config optimized for specific memory constraints - FIXED: Improved memory optimization"""
    if max_memory_gb < 4:
        cfg = get_debug_config()
    elif max_memory_gb < 6:
        cfg = get_test_config()
    else:
        cfg = get_full_config()
    
    # Adjust based on memory constraints
    if max_memory_gb < 3:
        cfg.BATCH_SIZE = 16
        cfg.RESNET_WIDTH = 0.125
        cfg.MAE_EMBED_DIM = 64
        cfg.MAE_DEPTH = 2
        cfg.DIFFUSION_HIDDEN_CHANNELS = 32
    elif max_memory_gb < 4:
        cfg.BATCH_SIZE = 24
        cfg.RESNET_WIDTH = 0.25
        cfg.MAE_EMBED_DIM = 96
        cfg.MAE_DEPTH = 3
        cfg.DIFFUSION_HIDDEN_CHANNELS = 48
    elif max_memory_gb < 6:
        cfg.BATCH_SIZE = 32
        cfg.RESNET_WIDTH = 0.5
        cfg.MAE_EMBED_DIM = 192
        cfg.MAE_DEPTH = 4
        cfg.DIFFUSION_HIDDEN_CHANNELS = 64
    
    cfg.MAX_MEMORY_GB = max_memory_gb
    return cfg

# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

def get_config():
    """Get default configuration based on environment"""
    # Check available GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory < 4:
            return get_memory_optimized_config(gpu_memory)
        elif gpu_memory < 6:
            return get_test_config()
        else:
            return get_full_config()
    else:
        return get_debug_config()  # CPU fallback 