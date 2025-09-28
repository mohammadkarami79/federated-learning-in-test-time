"""
Optimized Configuration for Better Performance
Based on analysis of current results - targeting 90%+ clean accuracy and 75%+ adversarial accuracy
"""

import torch

def get_optimized_config():
    """Get optimized configuration for better performance"""
    
    config = {
        # Dataset and model settings
        'DATASET': 'cifar10',
        'NUM_CLASSES': 10,
        'IMG_SIZE': 32,
        'IMG_CHANNELS': 3,
        'MODEL_NAME': 'resnet18',
        
        # Federated learning settings - OPTIMIZED
        'NUM_CLIENTS': 10,
        'NUM_ROUNDS': 20,  # Increased for better convergence
        'CLIENT_EPOCHS': 12,  # Increased for better local training
        'BATCH_SIZE': 64,
        'LEARNING_RATE': 0.02,  # Increased for faster convergence
        'MOMENTUM': 0.9,
        'WEIGHT_DECAY': 1e-4,
        
        # Learning rate scheduling - NEW
        'USE_SCHEDULER': True,
        'SCHEDULER_TYPE': 'cosine',
        'WARMUP_EPOCHS': 3,
        'MIN_LR': 0.001,
        
        # Device settings
        'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
        'NUM_WORKERS': 4,
        
        # Attack settings - Standard PGD
        'ATTACK_TYPE': 'pgd',
        'ATTACK_EPSILON': 0.031,
        'ATTACK_STEPS': 10,
        'ATTACK_ALPHA': 0.007,
        'ATTACK_RANDOM_START': True,
        
        # MAE Detector settings - OPTIMIZED
        'ENABLE_MAE_DETECTOR': True,
        'MAE_THRESHOLD': 0.12,  # Slightly lower for more detections
        'MAE_PATCH_SIZE': 4,
        'MAE_DEPTH': 4,
        'MAE_NUM_HEADS': 8,
        'MAE_MASK_RATIO': 0.75,
        'MAE_EMBED_DIM': 256,
        'MAE_DECODER_EMBED_DIM': 128,
        
        # DiffPure settings - STRONGER for better purification
        'ENABLE_DIFFPURE': True,
        'DIFFUSER_STEPS': 5,  # Increased for stronger purification
        'DIFFUSER_SIGMA': 0.35,  # Increased for better purification
        'DIFFUSER_SCHEDULE': 'linear',
        
        # Combined defense settings
        'ENABLE_COMBINED_DEFENSE': True,
        'SELECTIVE_DEFENSE': True,
        'DETECTION_FALLBACK': 'variance',
        'MIN_DETECTION_RATE': 0.15,  # Higher minimum detection
        'MAX_DETECTION_RATE': 0.35,  # Allow more detections
        
        # Training optimization - ENHANCED
        'USE_MIXED_PRECISION': True,
        'GRADIENT_CLIPPING': 1.0,
        'USE_GRADIENT_ACCUMULATION': True,
        'ACCUMULATION_STEPS': 2,
        
        # Data augmentation - ENHANCED
        'USE_CUTMIX': True,
        'CUTMIX_ALPHA': 1.0,
        'USE_MIXUP': True,
        'MIXUP_ALPHA': 0.2,
        
        # Evaluation settings
        'EVAL_BATCH_SIZE': 128,
        'EVAL_FREQUENCY': 1,
        'SAVE_CHECKPOINTS': True,
        
        # Logging and debugging
        'LOG_LEVEL': 'INFO',
        'DETAILED_LOGGING': True,
        'SAVE_RESULTS': True,
        'RESULTS_DIR': 'experiment_results',
        
        # Performance targets - UPDATED
        'TARGET_CLEAN_ACC': 90.0,  # Higher target
        'TARGET_ADV_ACC': 75.0,   # Higher target
        'TARGET_DETECTION_RATE': 20.0,  # Higher detection rate
        
        # Early stopping
        'USE_EARLY_STOPPING': True,
        'PATIENCE': 5,
        'MIN_DELTA': 0.001,
    }
    
    return config

def get_br35h_config():
    """Get configuration optimized for BR35H dataset"""
    
    config = get_optimized_config()
    
    # BR35H specific modifications
    config.update({
        'DATASET': 'br35h',
        'NUM_CLASSES': 2,  # Binary classification
        'IMG_SIZE': 224,   # Larger images for medical data
        'IMG_CHANNELS': 3,
        'MODEL_NAME': 'resnet34',  # Larger model for medical data
        
        # Medical data specific settings
        'BATCH_SIZE': 32,  # Smaller batch for larger images
        'LEARNING_RATE': 0.01,  # Lower LR for medical data
        'CLIENT_EPOCHS': 15,  # More epochs for medical data
        
        # Medical data augmentation
        'USE_MEDICAL_AUGMENTATION': True,
        'ROTATION_RANGE': 15,
        'BRIGHTNESS_RANGE': 0.2,
        'CONTRAST_RANGE': 0.2,
        
        # BR35H specific paths
        'DATA_ROOT': 'data/br35h',
        'TRAIN_CSV': 'data/br35h/train.csv',
        'TEST_CSV': 'data/br35h/test.csv',
    })
    
    return config

def validate_config(config):
    """Validate configuration parameters"""
    
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
    
    print("[OK] Optimized configuration validation passed")
    return True

# Export the main configuration function
if __name__ == "__main__":
    cfg = get_optimized_config()
    validate_config(cfg)
    print("Optimized Configuration:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")
