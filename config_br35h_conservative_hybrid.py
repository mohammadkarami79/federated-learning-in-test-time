"""
BR35H Conservative Hybrid Defense Configuration
MAE Detection + Conservative DiffPure for 85-87% adversarial accuracy
"""

import torch

def get_config():
    """Get conservative hybrid defense configuration"""
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

        # Conservative DiffPure settings - very gentle
        'ENABLE_DIFFPURE': True,
        'DIFFUSER_STEPS': 2,  # Few steps
        'DIFFUSER_SIGMA': 0.1,  # Low noise
        'DIFFUSER_SCHEDULE': 'linear',
        'DIFFPURE_STRENGTH': 0.15,  # Very gentle purification strength
        'DIFFPURE_ADAPTIVE': True,  # Adaptive purification

        # Combined defense settings
        'ENABLE_COMBINED_DEFENSE': True,
        'SELECTIVE_DEFENSE': True,
        'DETECTION_FALLBACK': 'variance',
        'MIN_DETECTION_RATE': 0.05,
        'MAX_DETECTION_RATE': 0.15,
        'HYBRID_DEFENSE_MODE': 'conservative',  # Conservative hybrid defense

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
        'TARGET_ADV_ACC': 85.0,  # Target: 85-87%
        'TARGET_DETECTION_RATE': 10.0,

        # Additional attributes for compatibility
        'MODE': 'full',
        'DATA_DISTRIBUTION': 'iid',
        'DIFFUSION_HIDDEN_CHANNELS': 128,
        'MAE_EPOCHS': 10,
        'DIFFUSION_EPOCHS': 30,
        'DIFFPURE_STEPS': 2,
        'DIFFPURE_SIGMA': 0.1,
    }
    return config

if __name__ == "__main__":
    cfg = get_config()
    print("BR35H Conservative Hybrid Defense Configuration:")
    print(f"MAE_THRESHOLD: {cfg['MAE_THRESHOLD']} (working)")
    print(f"DIFFUSER_STEPS: {cfg['DIFFUSER_STEPS']} (few)")
    print(f"DIFFUSER_SIGMA: {cfg['DIFFUSER_SIGMA']} (low)")
    print(f"DIFFPURE_STRENGTH: {cfg['DIFFPURE_STRENGTH']} (very gentle)")
    print(f"TARGET_ADV_ACC: {cfg['TARGET_ADV_ACC']}% (85-87%)")
    print("Expected: High adversarial accuracy with very gentle purification")
