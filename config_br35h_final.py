"""
BR35H Final Configuration - Optimized for High Adversarial Accuracy
Based on analysis of defense mechanism issues
"""

import torch

def get_config():
    """Get optimized BR35H configuration for high adversarial accuracy"""
    config = {
        # Dataset and model settings
        'DATASET': 'br35h',
        'DATA_ROOT': 'data',
        'DATA_PATH': 'data',
        'NUM_CLASSES': 2,
        'IMG_SIZE': 224,
        'IMG_CHANNELS': 3,
        'MODEL_NAME': 'resnet34',

        # Training settings - prevent overfitting
        'NUM_CLIENTS': 5,
        'NUM_ROUNDS': 15,
        'CLIENT_EPOCHS': 8,  # Reduced to prevent overfitting
        'BATCH_SIZE': 32,
        'LEARNING_RATE': 0.001,  # Reduced learning rate
        'MOMENTUM': 0.9,
        'WEIGHT_DECAY': 1e-4,  # Added weight decay

        # Device settings
        'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
        'NUM_WORKERS': 4,

        # Attack settings
        'ATTACK_TYPE': 'pgd',
        'ATTACK_EPSILON': 0.031,
        'ATTACK_STEPS': 10,
        'ATTACK_ALPHA': 0.007,
        'ATTACK_RANDOM_START': True,

        # MAE Detector settings - optimized for BR35H
        'ENABLE_MAE_DETECTOR': True,
        'MAE_THRESHOLD': 0.15,  # Base only; adaptive will override on BR35H
        'ADAPTIVE_THRESHOLD': True,
        'TARGET_DETECTION_RATE': 35.0,  # Raise to engage purification (~30–35% detection)
        'MAE_PATCH_SIZE': 16,  # Appropriate for 224x224
        'MAE_DEPTH': 6,
        'MAE_NUM_HEADS': 8,
        'MAE_MASK_RATIO': 0.75,
        'MAE_EMBED_DIM': 256,
        'MAE_DECODER_EMBED_DIM': 256,

        # DiffPure settings - enhanced for better purification
        'ENABLE_DIFFPURE': True,
        'DIFFUSER_STEPS': 3,   # Gentle but effective
        'DIFFUSER_SIGMA': 0.2, # Gentle noise level
        'DIFFUSER_SCHEDULE': 'linear',
        'DIFFPURE_STRENGTH': 0.30,  # Purification strength (used in main)

        # Combined defense settings
        'ENABLE_COMBINED_DEFENSE': True,
        'SELECTIVE_DEFENSE': True,
        'DETECTION_FALLBACK': 'variance',
        'MIN_DETECTION_RATE': 0.1,
        'MAX_DETECTION_RATE': 0.3,

        # Training optimization
        'USE_MIXED_PRECISION': True,
        'GRADIENT_CLIPPING': 1.0,
        'SCHEDULER_TYPE': 'cosine',
        'WARMUP_EPOCHS': 2,

        # Evaluation settings
        'EVAL_BATCH_SIZE': 32,
        'EVAL_FREQUENCY': 1,
        'SAVE_CHECKPOINTS': True,

        # Logging and debugging
        'LOG_LEVEL': 'INFO',
        'DETAILED_LOGGING': True,
        'SAVE_RESULTS': True,
        'RESULTS_DIR': 'experiment_results',

        # Performance targets
        'TARGET_CLEAN_ACC': 95.0,
        'TARGET_ADV_ACC': 85.0,  # Target adversarial accuracy
        'TARGET_DETECTION_RATE': 35.0,

        # Additional attributes for compatibility
        'MODE': 'full',
        'DATA_DISTRIBUTION': 'iid',
        'DIFFUSION_HIDDEN_CHANNELS': 128,
        'MAE_EPOCHS': 10,
        'DIFFUSION_EPOCHS': 30,
        'DIFFPURE_STEPS': 3,   # Match DIFFUSER_STEPS
        'DIFFPURE_SIGMA': 0.2, # Match DIFFUSER_SIGMA
    }
    return config

def validate_config(config):
    """Validate configuration parameters"""
    if not (0.0 < config['MAE_THRESHOLD'] < 1.0):
        raise ValueError("MAE_THRESHOLD must be between 0 and 1")
    if config['DIFFUSER_STEPS'] < 1:
        raise ValueError("DIFFUSER_STEPS must be at least 1")
    return True

if __name__ == "__main__":
    cfg = get_config()
    validate_config(cfg)
    print("BR35H Final Configuration:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
