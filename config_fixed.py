"""
OPTIMIZED CONFIGURATION FOR PFEDDEF + DIFFPURE SYSTEM
Designed for 10x performance improvement while maintaining functionality
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
LEARNING_RATE = 0.01  # Increased from 0.001 (10x faster learning)
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

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

def get_debug_config():
    """Get debug configuration for quick testing (2-5 minutes)"""
    cfg = SimpleNamespace()
    
    # Dataset configuration
    cfg.DATASET = 'cifar10'
    cfg.DATASET_NAME = 'CIFAR10'
    cfg.DATA_PATH = './data'
    cfg.IMG_SIZE = 32
    cfg.IMG_CHANNELS = 3
    cfg.N_CLASSES = 10
    
    # Training configuration (optimized for 20x speedup)
    cfg.N_ROUNDS = 3           # 10 → 3 (3x speedup)
    cfg.N_CLIENTS = 10         # Standard federated setup
    cfg.N_LEARNERS = 2         # Number of learners per client (for pFedDef)
    cfg.LOCAL_EPOCHS = 1       # Epochs per client per round
    cfg.LOCAL_STEPS_PER_EPOCH = 5  # 100 → 5 (20x speedup)
    cfg.BATCH_SIZE = 32
    cfg.LEARNING_RATE = 0.01   # 0.001 → 0.01 (10x faster convergence)
    
    # Model configuration
    cfg.MODEL = 'resnet18'
    cfg.MODEL_NAME = 'ResNet18'
    cfg.RESNET_WIDTH = 0.25    # Smaller model for speed
    cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Attack configuration (optimized for 5x speedup)
    cfg.PGD_EPS = 8/255
    cfg.PGD_ALPHA = 2/255
    cfg.PGD_STEPS = 2          # 10 → 2 (5x speedup)
    cfg.PGD_RESTARTS = 1
    
    # Defense configuration
    cfg.MAE_THRESHOLD = 0.1
    cfg.MAE_PATCH_SIZE = 4
    cfg.MAE_EMBED_DIM = 128
    cfg.MAE_DEPTH = 4
    cfg.MAE_NUM_HEADS = 4
    cfg.MAE_MASK_RATIO = 0.5
    cfg.MAE_DECODER_EMBED_DIM = 64
    
    # Diffusion configuration (optimized for 4x speedup)
    cfg.DIFFUSION_STEPS = 1    # 4 → 1 (4x speedup)
    cfg.DIFFUSER_STEPS = 1     # Alias for compatibility
    cfg.DIFFUSER_SIGMA = 0.04
    
    # Training optimization
    cfg.USE_AMP = True         # Mixed precision for memory efficiency
    cfg.LAMBDA_KL = 0.01      # KL divergence regularization
    cfg.MAX_MEMORY_GB = 4.0   # Memory target for testing
    
    return cfg

def get_test_config():
    """Get test configuration for validation (10-20 minutes)"""
    cfg = get_debug_config()  # Start with debug config
    
    # Increase parameters for better accuracy
    cfg.N_ROUNDS = 5
    cfg.LOCAL_STEPS_PER_EPOCH = 8  # 5 → 8 for better convergence
    cfg.PGD_STEPS = 3              # 2 → 3 for stronger attacks
    cfg.DIFFUSION_STEPS = 2        # 1 → 2 for better purification
    cfg.DIFFUSER_STEPS = 2
    
    return cfg

def get_full_config():
    """Get full configuration for complete training (20-60 minutes)"""
    cfg = get_debug_config()  # Start with debug config
    
    # Full parameters for research-quality results
    cfg.N_ROUNDS = 50
    cfg.LOCAL_STEPS_PER_EPOCH = 10  # 5 → 10 for full convergence
    cfg.PGD_STEPS = 5               # 2 → 5 for research-quality attacks
    cfg.DIFFUSION_STEPS = 2         # 1 → 2 for better purification
    cfg.DIFFUSER_STEPS = 2
    cfg.LEARNING_RATE = 0.005       # Slightly lower for stability
    
    return cfg

# ============================================================================
# VALIDATION AND HELPERS
# ============================================================================

def validate_config(cfg):
    """Validate configuration settings"""
    assert cfg.N_CLIENTS > 0, "N_CLIENTS must be positive"
    assert cfg.N_ROUNDS > 0, "N_ROUNDS must be positive"
    assert cfg.LOCAL_STEPS_PER_EPOCH > 0, "LOCAL_STEPS_PER_EPOCH must be positive"
    assert cfg.BATCH_SIZE > 0, "BATCH_SIZE must be positive"
    assert 0 < cfg.LEARNING_RATE < 1, "LEARNING_RATE must be in (0, 1)"
    assert cfg.PGD_STEPS > 0, "PGD_STEPS must be positive"
    assert cfg.DIFFUSION_STEPS > 0, "DIFFUSION_STEPS must be positive"
    
    print(f"✓ Configuration validated successfully")
    print(f"  - Device: {cfg.DEVICE}")
    print(f"  - Clients: {cfg.N_CLIENTS}")
    print(f"  - Rounds: {cfg.N_ROUNDS}")
    print(f"  - Local steps: {cfg.LOCAL_STEPS_PER_EPOCH}")
    print(f"  - Learning rate: {cfg.LEARNING_RATE}")
    print(f"  - PGD steps: {cfg.PGD_STEPS}")
    print(f"  - Diffusion steps: {cfg.DIFFUSION_STEPS}")

def get_memory_optimized_config(max_memory_gb=3.0):
    """Get config optimized for specific memory constraints"""
    cfg = get_debug_config() if max_memory_gb < 4 else get_test_config()
    
    if max_memory_gb < 3:
        cfg.BATCH_SIZE = 16
        cfg.RESNET_WIDTH = 0.125
        cfg.MAE_EMBED_DIM = 64
        cfg.MAE_DEPTH = 2
    elif max_memory_gb < 4:
        cfg.BATCH_SIZE = 24
        cfg.RESNET_WIDTH = 0.25
        cfg.MAE_EMBED_DIM = 96
        cfg.MAE_DEPTH = 3
    
    return cfg

# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

# Default config for imports
def get_config():
    """Get the default optimized configuration"""
    return get_test_config()

# Auto-create config on import
CONFIG = get_config()

if __name__ == "__main__":
    print("=== OPTIMIZED PFEDDEF + DIFFPURE CONFIGURATION ===")
    
    configs = {
        "Debug (2-5 min)": get_debug_config(),
        "Test (10-20 min)": get_test_config(), 
        "Full experiment": get_full_config(),
        "Memory constrained": get_memory_optimized_config(2.5)
    }
    
    for name, cfg in configs.items():
        print(f"\n{name}:")
        print(f"  Rounds: {cfg.N_ROUNDS}, Tasks: {cfg.N_TASKS}")
        print(f"  Local steps: {cfg.LOCAL_STEPS_PER_EPOCH}")
        print(f"  PGD: {cfg.PGD_STEPS}, Diffusion: {cfg.DIFFUSION_STEPS}")
        print(f"  Batch size: {cfg.BATCH_SIZE}, LR: {cfg.LEARNING_RATE}")
    
    print(f"\nGPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB") 