"""
BR35H Selective Defense Configuration - MAE + DiffPure
Keeps CIFAR-10 config untouched. Use this for BR35H runs.
"""

import torch

def get_config():
    """Get BR35H selective defense configuration"""
    config = {
        # Dataset and model settings
        'DATASET': 'br35h',
        'DATA_ROOT': 'data/br35h',
        'DATA_PATH': 'data/br35h',
        'NUM_CLASSES': 2,
        'IMG_SIZE': 224,
        'IMG_CHANNELS': 3,
        'MODEL_NAME': 'resnet34',

        # Federated learning settings (slower, more epochs)
        'NUM_CLIENTS': 5,
        'NUM_ROUNDS': 15,
        'CLIENT_EPOCHS': 10,  # Reduced to prevent overfitting
        'BATCH_SIZE': 32,
        'LEARNING_RATE': 0.001,  # Reduced learning rate
        'MOMENTUM': 0.9,
        'WEIGHT_DECAY': 5e-4,

        # Device settings
        'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
        'NUM_WORKERS': 4,

        # Attack settings (same scale as CIFAR-10 by default)
        'ATTACK_TYPE': 'pgd',
        'ATTACK_EPSILON': 0.031,
        'ATTACK_STEPS': 10,
        'ATTACK_ALPHA': 0.007,
        'ATTACK_RANDOM_START': True,

        # MAE Detector settings
        'ENABLE_MAE_DETECTOR': True,
        # Use a higher base threshold so non-adaptive case won't mark all as adv
        'MAE_THRESHOLD': 0.15,  # Reduced from 0.22
        'ADAPTIVE_THRESHOLD': True,
        'TARGET_DETECTION_RATE': 18.0,  # Reduced from 69.67%
        'MAE_PATCH_SIZE': 16,   # larger image → larger patch
        'MAE_DEPTH': 6,
        'MAE_NUM_HEADS': 8,
        'MAE_MASK_RATIO': 0.75,
        'MAE_EMBED_DIM': 256,
        # Match decoder to encoder to avoid shape mismatch (256 % 8 heads OK)
        'MAE_DECODER_EMBED_DIM': 256,

        # DiffPure settings
        'ENABLE_DIFFPURE': True,
        # Reduce aggressiveness for 224x224 to avoid OOM
        'DIFFUSER_STEPS': 6,  # Increased from 3
        'DIFFUSER_SIGMA': 0.6,  # Increased from 0.25
        'DIFFUSER_SCHEDULE': 'linear',

        # Combined defense
        'ENABLE_COMBINED_DEFENSE': True,
        'SELECTIVE_DEFENSE': True,
        'DETECTION_FALLBACK': 'variance',

        # Training optimization
        'USE_MIXED_PRECISION': True,
        'GRADIENT_CLIPPING': 1.0,
        'SCHEDULER_TYPE': 'cosine',
        'WARMUP_EPOCHS': 2,

        # Evaluation
        # Smaller eval batch to reduce memory footprint during DiffPure
        'EVAL_BATCH_SIZE': 32,
        'EVAL_FREQUENCY': 1,
        'SAVE_CHECKPOINTS': True,

        # Logging
        'LOG_LEVEL': 'INFO',
        'DETAILED_LOGGING': True,
        'SAVE_RESULTS': True,
        'RESULTS_DIR': 'experiment_results',

        # Compatibility
        'MODE': 'full',
        'DATA_DISTRIBUTION': 'iid',
        'DIFFUSION_HIDDEN_CHANNELS': 128,
        'MAE_EPOCHS': 10,
        'DIFFUSION_EPOCHS': 30,
        'DIFFPURE_STEPS': 50,
        'DIFFPURE_SIGMA': 0.1,
    }
    return config

def validate_config(config):
    if not (0.0 < config['MAE_THRESHOLD'] < 1.0):
        raise ValueError("MAE_THRESHOLD must be between 0 and 1")
    if config['DIFFUSER_STEPS'] < 1:
        raise ValueError("DIFFUSER_STEPS must be at least 1")
    return True

if __name__ == "__main__":
    cfg = get_config()
    validate_config(cfg)
    print("BR35H Selective Defense Configuration:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")


