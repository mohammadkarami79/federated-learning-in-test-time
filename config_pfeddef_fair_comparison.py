"""
PFEDDEF FAIR COMPARISON CONFIG
=============================
Configuration that maintains attack strength for fair comparison with PFedDef baseline
while optimizing the defense pipeline for better adversarial accuracy
"""

import torch

def get_pfeddef_fair_config():
    """Configuration optimized for fair PFedDef comparison"""
    
    class PFedDefFairConfig:
        # Dataset settings
        DATASET = 'CIFAR10'
        DATASET_NAME = 'Cifar10'
        DATA_ROOT = './data'
        IMG_SIZE = 32
        IMG_CHANNELS = 3
        NUM_CLASSES = 10
        
        # Federated learning settings
        NUM_CLIENTS = 10
        NUM_ROUNDS = 15
        CLIENT_EPOCHS = 8  # More training for robustness
        
        # Training settings - OPTIMIZED FOR ROBUSTNESS
        BATCH_SIZE = 64
        LEARNING_RATE = 0.01
        WEIGHT_DECAY = 1e-4
        MOMENTUM = 0.9
        
        # MAINTAIN STANDARD ATTACK STRENGTH for fair comparison
        PGD_STEPS = 10  # Standard attack strength
        PGD_ALPHA = 0.01  # Standard step size
        PGD_EPSILON = 0.031  # Standard perturbation budget (8/255)
        
        # OPTIMIZED DiffPure settings for better defense
        DIFFPURE_STEPS = 15  # Balanced purification
        DIFFPURE_SIGMA = 0.03  # Optimized noise level
        
        # OPTIMIZED MAE settings for efficient detection
        MAE_THRESHOLD = 0.12  # Lower threshold for better detection
        MAE_EMBED_DIM = 128
        ENABLE_MAE_DETECTOR = True  # Enable MAE-based selective defense
        
        # Defense optimization settings
        DEFENSE_ENSEMBLE_SIZE = 3  # Multiple defense passes
        CONFIDENCE_THRESHOLD = 0.8  # High confidence threshold
        
        # System settings
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        NUM_WORKERS = 2
        PIN_MEMORY = True
        
        # Advanced training settings
        SCHEDULER_STEP_SIZE = 3
        SCHEDULER_GAMMA = 0.9
        WARMUP_EPOCHS = 2
        
        # Diffusion model settings
        DIFFUSION_HIDDEN_CHANNELS = 128
        DIFFUSION_EPOCHS = 30
        
        # MAE detector settings
        MAE_EPOCHS = 15
        
    return PFedDefFairConfig()

# Export the config
get_ultimate_config = get_pfeddef_fair_config
