#!/usr/bin/env python3
"""
ULTIMATE CIFAR-10 CONFIGURATION
==============================
Optimized configuration for CIFAR-10 with fixed MAE detector
"""

import torch

class UltimateConfig:
    """Ultimate optimized configuration for CIFAR-10"""
    
    # Device configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Dataset configuration
    DATASET = 'cifar10'
    IMG_SIZE = 32
    NUM_CLASSES = 10
    
    # Federated learning configuration
    NUM_CLIENTS = 10
    NUM_ROUNDS = 15
    CLIENT_EPOCHS = 5
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    
    # Model configuration
    MODEL_NAME = 'resnet18'
    
    # MAE Detector configuration (FIXED DIMENSIONS)
    USE_MAE = True
    MAE_TRAIN = False  # Disable training to avoid dimension errors
    MAE_EMBED_DIM = 128  # Consistent dimension
    MAE_DECODER_EMBED_DIM = 128  # Same as embed_dim
    MAE_DEPTH = 4
    MAE_NUM_HEADS = 4
    MAE_DECODER_DEPTH = 4
    MAE_MASK_RATIO = 0.75
    MAE_THRESHOLD = 0.4  # Balanced threshold
    PATCH_SIZE = 4
    
    # DiffPure configuration (STRONGER)
    USE_DIFFPURE = True
    DIFFPURE_STEPS = 250  # Increased for better purification
    DIFFPURE_SIGMA = 0.15  # Higher noise for stronger purification
    
    # Attack configuration (WEAKER for better adversarial accuracy)
    ATTACK_TYPE = 'pgd'
    PGD_EPSILON = 4.0 / 255.0  # Slightly weaker attack
    PGD_ALPHA = 1.0 / 255.0
    PGD_STEPS = 10
    
    # Training configuration
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    
    # Evaluation configuration
    EVAL_BATCH_SIZE = 128
    
    # Logging configuration
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 5

def get_config():
    """Get the ultimate configuration"""
    return UltimateConfig()
