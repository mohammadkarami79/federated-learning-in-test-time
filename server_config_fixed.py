"""
CRITICAL SERVER CONFIGURATION FIX
This file contains the EXACT configuration needed to fix your server issues.
Copy this ENTIRE file to your server as config_fixed.py
"""

import torch
import os
from types import SimpleNamespace

# ============================================================================
# OPTIMIZED PARAMETERS FOR PERFORMANCE
# ============================================================================

class Config:
    """Configuration class for storing all parameters"""
    pass

# Core experiment settings
N_CLIENTS = 10
N_TASKS = 10  # Reduced from 80 tasks for faster testing
N_ROUNDS = 15  # FIXED: Increased for better convergence

def get_full_config():
    """Get full configuration with all fixes applied"""
    cfg = Config()
    
    # Dataset and basic parameters
    cfg.DATASET = 'cifar10'
    cfg.NUM_CLASSES = 10
    cfg.IMG_SIZE = 32
    cfg.IMG_CHANNELS = 3
    cfg.BATCH_SIZE = 128
    cfg.NUM_WORKERS = 4
    
    # Training parameters - FIXED: Proper values for CIFAR-10
    cfg.EPOCHS = 200
    cfg.LEARNING_RATE = 0.01  # FIXED: Proper LR for CIFAR-10
    cfg.MOMENTUM = 0.9
    cfg.WEIGHT_DECAY = 5e-4
    cfg.SCHEDULER = 'cosine'
    cfg.WARMUP_EPOCHS = 5
    
    # Federated learning parameters - FIXED: Proper training setup
    cfg.NUM_CLIENTS = 10
    cfg.NUM_ROUNDS = 15  # FIXED: Increased for better convergence
    cfg.CLIENT_EPOCHS = 5  # FIXED: Increased from 3 to 5 for better local learning
    cfg.DIFFUSION_EPOCHS = 15  # Good for CIFAR-10
    cfg.MAE_EPOCHS = 15   # Good for CIFAR-10
    cfg.DIFFUSION_HIDDEN_CHANNELS = 128
    
    # Model parameters
    cfg.MODEL_WIDTH = 1.0
    cfg.USE_ADDITIONAL_LAYERS = False
    cfg.N_LEARNERS = 2  # Added for pFedDef compatibility
    
    # Defense parameters - CRITICAL FIX: Match your trained MAE model
    cfg.MAE_EMBED_DIM = 256
    cfg.MAE_DECODER_EMBED_DIM = 128  # CRITICAL: This MUST be 128 to match your checkpoint
    cfg.MAE_MASK_RATIO = 0.75
    cfg.MAE_PATCH_SIZE = 4
    cfg.MAE_THRESHOLD = 0.25    # FIXED: Reasonable threshold (was 0.1, too low)
    
    # Attack parameters - OPTIMIZED
    cfg.PGD_EPS = 8/255
    cfg.PGD_ALPHA = 1/255     # REDUCED for more realistic attacks
    cfg.PGD_STEPS = 20        # INCREASED for stronger attacks
    cfg.PGD_RANDOM_START = True
    cfg.ATTACK_EPSILON = 0.3
    
    # DiffPure parameters - FIXED: Better balance
    cfg.DIFFUSION_STEPS = 100  # REDUCED from 1000 for speed
    cfg.DIFFUSION_NOISE_SCALE = 0.5  # REDUCED for better image quality
    cfg.DIFFUSION_BETA_START = 0.0001
    cfg.DIFFUSION_BETA_END = 0.02
    cfg.DIFFUSION_SCHEDULE = 'linear'
    
    # Model architecture - FIXED: Proper ResNet setup
    cfg.RESNET_DEPTH = 18
    cfg.RESNET_WIDTH = 1.0    # FIXED: Full width for better accuracy
    cfg.DROPOUT_RATE = 0.1
    
    # MAE architecture - CRITICAL: Match your trained model
    cfg.MAE_DEPTH = 6
    cfg.MAE_NUM_HEADS = 8
    cfg.MAE_DECODER_DEPTH = 4
    cfg.MAE_DECODER_NUM_HEADS = 8
    cfg.MAE_MLP_RATIO = 4.0
    
    # Optimization settings
    cfg.OPTIMIZER = 'sgd'
    cfg.BETA1 = 0.9
    cfg.BETA2 = 0.999
    cfg.EPS = 1e-8
    cfg.GRAD_CLIP = 1.0
    
    # Device and memory
    cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.MIXED_PRECISION = True
    cfg.PIN_MEMORY = True
    cfg.NON_BLOCKING = True
    
    # Logging and checkpoints
    cfg.LOG_INTERVAL = 10
    cfg.SAVE_INTERVAL = 5
    cfg.EVAL_INTERVAL = 1
    cfg.CHECKPOINT_DIR = 'checkpoints'
    cfg.LOG_DIR = 'logs'
    
    # Paths - FIXED: Proper checkpoint paths
    cfg.MAE_CHECKPOINT_PATH = 'checkpoints/mae_detector_cifar10.pt'
    cfg.DIFFUSION_CHECKPOINT_PATH = 'checkpoints/diffuser_cifar10.pt'
    cfg.MODEL_CHECKPOINT_PATH = 'checkpoints/model_best.pt'
    
    # Data augmentation
    cfg.USE_AUGMENTATION = True
    cfg.AUGMENTATION_STRENGTH = 0.5
    cfg.CUTOUT_LENGTH = 16
    cfg.MIXUP_ALPHA = 0.2
    
    # Validation parameters
    cfg.VAL_BATCH_SIZE = 256
    cfg.TEST_BATCH_SIZE = 256
    cfg.EVAL_CLEAN = True
    cfg.EVAL_ADV = True
    cfg.EVAL_MAE = True
    
    # Advanced settings
    cfg.SEED = 42
    cfg.DETERMINISTIC = False  # Set to True for reproducibility
    cfg.BENCHMARK = True
    
    return cfg

def get_debug_config():
    """Get debug configuration for testing"""
    cfg = get_full_config()
    
    # Reduce parameters for debugging
    cfg.NUM_CLIENTS = 2
    cfg.NUM_ROUNDS = 3
    cfg.CLIENT_EPOCHS = 2
    cfg.BATCH_SIZE = 32
    cfg.EPOCHS = 10
    
    # Keep MAE dimensions correct
    cfg.MAE_EMBED_DIM = 256
    cfg.MAE_DECODER_EMBED_DIM = 128  # CRITICAL: Still 128 for debug
    cfg.MAE_THRESHOLD = 0.25
    
    return cfg

def get_test_config():
    """Get test configuration"""
    cfg = get_full_config()
    
    # Moderate parameters for testing
    cfg.NUM_CLIENTS = 5
    cfg.NUM_ROUNDS = 10
    cfg.CLIENT_EPOCHS = 3
    cfg.BATCH_SIZE = 64
    
    # Keep MAE dimensions correct
    cfg.MAE_EMBED_DIM = 256
    cfg.MAE_DECODER_EMBED_DIM = 128  # CRITICAL: Still 128 for test
    cfg.MAE_THRESHOLD = 0.25
    
    return cfg

def validate_config(cfg):
    """Validate configuration parameters"""
    if cfg.EPOCHS <= 0:
        raise ValueError("EPOCHS must be positive")
    if cfg.NUM_CLIENTS <= 0:
        raise ValueError("NUM_CLIENTS must be positive")
    if cfg.NUM_ROUNDS <= 0:
        raise ValueError("NUM_ROUNDS must be positive")
    if cfg.CLIENT_EPOCHS <= 0:
        raise ValueError("CLIENT_EPOCHS must be positive")
    if cfg.DIFFUSION_EPOCHS <= 0:
        raise ValueError("DIFFUSION_EPOCHS must be positive")
    if cfg.MAE_EPOCHS <= 0:
        raise ValueError("MAE_EPOCHS must be positive")
    if cfg.DIFFUSION_HIDDEN_CHANNELS <= 0:
        raise ValueError("DIFFUSION_HIDDEN_CHANNELS must be positive")
    
    # CRITICAL: Validate MAE dimensions
    if cfg.MAE_DECODER_EMBED_DIM != 128:
        print(f"WARNING: MAE_DECODER_EMBED_DIM is {cfg.MAE_DECODER_EMBED_DIM}, should be 128")
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if hasattr(cfg, 'MAX_MEMORY_GB') and cfg.MAX_MEMORY_GB > gpu_memory:
            print(f"Warning: MAX_MEMORY_GB ({cfg.MAX_MEMORY_GB}) exceeds GPU memory ({gpu_memory:.1f}GB)")
    
    return True

def get_memory_optimized_config(max_memory_gb=3.0):
    """Get config optimized for specific memory constraints"""
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
        cfg.MAE_DECODER_EMBED_DIM = 32  # Keep ratio consistent
        cfg.MAE_DEPTH = 2
        cfg.DIFFUSION_HIDDEN_CHANNELS = 32
    elif max_memory_gb < 4:
        cfg.BATCH_SIZE = 24
        cfg.RESNET_WIDTH = 0.25
        cfg.MAE_EMBED_DIM = 96
        cfg.MAE_DECODER_EMBED_DIM = 48  # Keep ratio consistent
        cfg.MAE_DEPTH = 3
        cfg.DIFFUSION_HIDDEN_CHANNELS = 48
    elif max_memory_gb < 6:
        cfg.BATCH_SIZE = 32
        cfg.RESNET_WIDTH = 0.5
        cfg.MAE_EMBED_DIM = 192
        cfg.MAE_DECODER_EMBED_DIM = 96  # Keep ratio consistent
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

# Test the configuration
if __name__ == "__main__":
    print("Testing configuration...")
    cfg = get_full_config()
    validate_config(cfg)
    print(f"✅ Configuration validated successfully!")
    print(f"📊 Key parameters:")
    print(f"   CLIENT_EPOCHS: {cfg.CLIENT_EPOCHS}")
    print(f"   MAE_EMBED_DIM: {cfg.MAE_EMBED_DIM}")
    print(f"   MAE_DECODER_EMBED_DIM: {cfg.MAE_DECODER_EMBED_DIM}")
    print(f"   MAE_THRESHOLD: {cfg.MAE_THRESHOLD}")
    print(f"   NUM_ROUNDS: {cfg.NUM_ROUNDS}")
