"""
FINAL OPTIMIZED CONFIG FOR PFEDDEF + DIFFPURE + MAE
==================================================
Complete integration with proper adversarial accuracy optimization
"""

import torch

def get_ultimate_config():
    """Final optimized configuration"""
    
    class FinalOptimizedConfig:
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
        CLIENT_EPOCHS = 8  # Increased for better robustness
        
        # Training settings - OPTIMIZED FOR ROBUSTNESS
        BATCH_SIZE = 64
        LEARNING_RATE = 0.01
        WEIGHT_DECAY = 1e-4
        MOMENTUM = 0.9
        
        # ATTACK SETTINGS - MAINTAIN STANDARD STRENGTH FOR FAIR COMPARISON
        PGD_STEPS = 10  # Standard attack
        PGD_ALPHA = 0.01  # Standard step size  
        PGD_EPSILON = 0.031  # Standard perturbation (8/255)
        PGD_EPS = 0.031  # Alternative name
        
        # OPTIMIZED MAE DETECTOR SETTINGS
        MAE_THRESHOLD = 0.08  # Lower threshold for better detection
        MAE_EMBED_DIM = 128
        ENABLE_MAE_DETECTOR = True
        USE_ACTUAL_MAE_RECONSTRUCTION = True  # Enable real MAE detection
        
        # OPTIMIZED DIFFPURE SETTINGS - SELECTIVE APPLICATION
        DIFFPURE_STEPS = 15  # Reduced for efficiency
        DIFFPURE_SIGMA = 0.04  # Gentler purification
        SELECTIVE_DIFFPURE = True  # Only apply to detected adversarial samples
        
        # PFEDDEF INTEGRATION SETTINGS
        USE_PFEDDEF_ENSEMBLE = True
        N_LEARNERS = 3  # Multi-learner ensemble
        FEDEM_N_LEARNERS = 3
        
        # DEFENSE OPTIMIZATION
        CONFIDENCE_THRESHOLD = 0.7
        ENSEMBLE_TEMPERATURE = 0.8
        
        # System settings
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        NUM_WORKERS = 2
        PIN_MEMORY = True
        
        # Advanced training settings
        SCHEDULER_STEP_SIZE = 3
        SCHEDULER_GAMMA = 0.9
        WARMUP_EPOCHS = 2
        
    return FinalOptimizedConfig()

# Export the config
get_debug_config = get_ultimate_config
get_test_config = get_ultimate_config
get_full_config = get_ultimate_config
