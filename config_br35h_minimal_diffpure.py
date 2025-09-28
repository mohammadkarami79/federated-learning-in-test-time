"""
BR35H Minimal DiffPure Configuration
Test: Minimal DiffPure that does almost nothing
"""

import torch

def get_config():
    """Get configuration with minimal DiffPure"""
    config = {
        # Dataset and model settings
        'DATASET': 'br35h',
        'DATA_ROOT': 'data',
        'DATA_PATH': 'data',
        'NUM_CLASSES': 2,
        'IMG_SIZE': 224,
        'IMG_CHANNELS': 3,
        'MODEL_NAME': 'resnet34',

        # Training settings
        'NUM_CLIENTS': 5,
        'NUM_ROUNDS': 15,
        'CLIENT_EPOCHS': 8,
        'BATCH_SIZE': 32,
        'LEARNING_RATE': 0.001,
        'MOMENTUM': 0.9,
        'WEIGHT_DECAY': 1e-4,

        # Device settings
        'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
        'NUM_WORKERS': 4,

        # Attack settings
        'ATTACK_TYPE': 'pgd',
        'ATTACK_EPSILON': 0.031,
        'ATTACK_STEPS': 10,
        'ATTACK_ALPHA': 0.007,
        'ATTACK_RANDOM_START': True,

        # MAE Detector settings - keep working settings
        'ENABLE_MAE_DETECTOR': True,
        'MAE_THRESHOLD': 0.3,  # Keep this - it's working
        'ADAPTIVE_THRESHOLD': False,  # Keep this - it's working
        'TARGET_DETECTION_RATE': 10.0,  # Keep this - it's working
        'MAE_PATCH_SIZE': 16,
        'MAE_DEPTH': 6,
        'MAE_NUM_HEADS': 8,
        'MAE_MASK_RATIO': 0.75,
        'MAE_EMBED_DIM': 256,
        'MAE_DECODER_EMBED_DIM': 256,

        # DiffPure settings - MINIMAL
        'ENABLE_DIFFPURE': True,
        'DIFFUSER_STEPS': 1,  # Only 1 step
        'DIFFUSER_SIGMA': 0.01,  # Very low noise
        'DIFFUSER_SCHEDULE': 'linear',

        # Combined defense settings
        'ENABLE_COMBINED_DEFENSE': True,
        'SELECTIVE_DEFENSE': True,
        'DETECTION_FALLBACK': 'variance',
        'MIN_DETECTION_RATE': 0.05,
        'MAX_DETECTION_RATE': 0.15,

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
        'TARGET_ADV_ACC': 85.0,
        'TARGET_DETECTION_RATE': 10.0,

        # Additional attributes for compatibility
        'MODE': 'full',
        'DATA_DISTRIBUTION': 'iid',
        'DIFFUSION_HIDDEN_CHANNELS': 128,
        'MAE_EPOCHS': 10,
        'DIFFUSION_EPOCHS': 30,
        'DIFFPURE_STEPS': 1,
        'DIFFPURE_SIGMA': 0.01,
    }
    return config

if __name__ == "__main__":
    cfg = get_config()
    print("BR35H Minimal DiffPure Configuration:")
    print(f"DIFFUSER_STEPS: {cfg['DIFFUSER_STEPS']} (minimal)")
    print(f"DIFFUSER_SIGMA: {cfg['DIFFUSER_SIGMA']} (minimal)")
    print("Expected: High adversarial accuracy with minimal purification")
