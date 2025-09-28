"""
Selective Defense Configuration - Optimized for MAE+DiffPure Integration
This configuration implements selective defense where MAE detection determines
which samples need DiffPure purification, achieving better adversarial accuracy.
"""

import torch

def get_config():
    """Get optimized selective defense configuration"""
    
    # Base configuration
    config = {
        # Dataset and model settings
        'DATASET': 'cifar10',
        'DATA_ROOT': 'data',  # Add missing DATA_ROOT
        'DATA_PATH': 'data',  # Add DATA_PATH for compatibility
        'NUM_CLASSES': 10,
        'IMG_SIZE': 32,
        'IMG_CHANNELS': 3,
        'MODEL_NAME': 'resnet18',
        
        # Federated learning settings
        'NUM_CLIENTS': 10,
        'NUM_ROUNDS': 15,
        'CLIENT_EPOCHS': 8,  # Increased from 5 for better convergence
        'BATCH_SIZE': 64,
        'LEARNING_RATE': 0.01,
        'MOMENTUM': 0.9,
        'WEIGHT_DECAY': 1e-4,
        
        # Device settings
        'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
        'NUM_WORKERS': 4,
        
        # Attack settings - Standard PGD for fair comparison
        'ATTACK_TYPE': 'pgd',
        'ATTACK_EPSILON': 0.031,  # 8/255
        'ATTACK_STEPS': 10,
        'ATTACK_ALPHA': 0.007,
        'ATTACK_RANDOM_START': True,
        
        # MAE Detector settings - Optimized for selective defense
        'ENABLE_MAE_DETECTOR': True,
        'MAE_THRESHOLD': 0.15,  # Match log7 baseline; adaptive will refine
        'ADAPTIVE_THRESHOLD': True,  # Ensure adaptive is enabled in base config
        'MAE_PATCH_SIZE': 4,
        'MAE_DEPTH': 4,
        'MAE_NUM_HEADS': 8,
        'MAE_MASK_RATIO': 0.75,
        'MAE_EMBED_DIM': 256,
        'MAE_DECODER_EMBED_DIM': 128,
        
        # DiffPure settings - Gentler for selective application
        'ENABLE_DIFFPURE': True,
        'DIFFUSER_STEPS': 4,   # Match log7
        'DIFFUSER_SIGMA': 0.3, # Match log7
        'DIFFUSER_SCHEDULE': 'linear',
        
        # Combined defense settings
        'ENABLE_COMBINED_DEFENSE': True,
        'SELECTIVE_DEFENSE': True,  # Key: only purify detected samples
        'DETECTION_FALLBACK': 'variance',  # Use variance if MAE fails
        'MIN_DETECTION_RATE': 0.1,  # Minimum 10% detection for robustness
        'MAX_DETECTION_RATE': 0.4,  # Maximum 40% to avoid over-purification
        
        # Training optimization
        'USE_MIXED_PRECISION': True,
        'GRADIENT_CLIPPING': 1.0,
        'SCHEDULER_TYPE': 'cosine',
        'WARMUP_EPOCHS': 2,
        
        # Evaluation settings
        'EVAL_BATCH_SIZE': 128,
        'EVAL_FREQUENCY': 1,  # Evaluate every round
        'SAVE_CHECKPOINTS': True,
        
        # Logging and debugging
        'LOG_LEVEL': 'INFO',
        'DETAILED_LOGGING': True,
        'SAVE_RESULTS': True,
        'RESULTS_DIR': 'experiment_results',
        
        # Performance targets
        'TARGET_CLEAN_ACC': 85.0,  # Target clean accuracy
        'TARGET_ADV_ACC': 70.0,   # Target adversarial accuracy with selective defense
        'TARGET_DETECTION_RATE': 15.63,  # Match log7
        
        # Additional attributes for compatibility
        'MODE': 'full',
        'DATA_DISTRIBUTION': 'iid',
        'DIFFUSION_HIDDEN_CHANNELS': 128,
        'MAE_EPOCHS': 10,
        'DIFFUSION_EPOCHS': 50,
        'DIFFPURE_STEPS': 50,
        'DIFFPURE_SIGMA': 0.1,
    }
    
    return config

def get_selective_defense_config():
    """Get configuration specifically optimized for selective defense"""
    config = get_config()
    
    # Selective defense specific optimizations
    config.update({
        # Keep training like log7
        'MAE_THRESHOLD': 0.15,
        'DIFFUSER_STEPS': 4,
        'DIFFUSER_SIGMA': 0.3,
        'CLIENT_EPOCHS': 8,
        'LEARNING_RATE': 0.01,

        # Adaptive thresholding to keep detection ~15.6%
        'ADAPTIVE_THRESHOLD': True,
        'THRESHOLD_UPDATE_FREQUENCY': 1,
        'THRESHOLD_ADAPTATION_RATE': 0.1,

        # Evaluation
        'EVAL_CLEAN_AND_ADV': True,
        'EVAL_DETECTION_METRICS': True,
        'EVAL_SELECTIVE_DEFENSE': True,
    })
    
    return config

def validate_config(config):
    """Validate configuration parameters"""
    
    # Check required parameters
    required_params = [
        'DATASET', 'NUM_CLASSES', 'NUM_CLIENTS', 'NUM_ROUNDS',
        'ATTACK_EPSILON', 'MAE_THRESHOLD', 'DIFFUSER_STEPS'
    ]
    
    for param in required_params:
        if param not in config:
            raise ValueError(f"Missing required parameter: {param}")
    
    # Validate ranges
    if not (0.0 < config['MAE_THRESHOLD'] < 1.0):
        raise ValueError("MAE_THRESHOLD must be between 0 and 1")
    
    if not (0.0 < config['ATTACK_EPSILON'] < 1.0):
        raise ValueError("ATTACK_EPSILON must be between 0 and 1")
    
    if config['DIFFUSER_STEPS'] < 1:
        raise ValueError("DIFFUSER_STEPS must be at least 1")
    
    print("[OK] Configuration validation passed")
    return True

# Export the main configuration function
if __name__ == "__main__":
    cfg = get_selective_defense_config()
    validate_config(cfg)
    print("Selective Defense Configuration:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")
