# BR35H Selective Defense Configuration - Improved
# Based on analysis of current results

# Dataset Configuration
DATASET = 'br35h'
IMG_SIZE = 224
NUM_CLASSES = 2
MODEL_NAME = 'resnet34'

# Training Configuration
NUM_ROUNDS = 15
CLIENT_EPOCHS = 10  # Reduced to prevent overfitting
BATCH_SIZE = 32
LEARNING_RATE = 0.001  # Reduced learning rate
WEIGHT_DECAY = 1e-4  # Added weight decay
MOMENTUM = 0.9

# Attack Configuration
ATTACK_EPSILON = 0.031
ATTACK_STEPS = 10
ATTACK_ALPHA = 0.007

# MAE Detector Configuration
MAE_THRESHOLD = 0.15  # Reduced threshold
ADAPTIVE_THRESHOLD = True
TARGET_DETECTION_RATE = 18.0  # Reduced target rate
MAE_PATCH_SIZE = 16
MAE_DEPTH = 6
MAE_NUM_HEADS = 8
MAE_EMBED_DIM = 256
MAE_DECODER_EMBED_DIM = 256

# DiffPure Configuration
DIFFUSER_STEPS = 6  # Increased steps
DIFFUSER_SIGMA = 0.6  # Increased sigma
EVAL_BATCH_SIZE = 32

# Other Configuration
NUM_CLIENTS = 5
DATA_DISTRIBUTION = 'iid'
DEVICE = 'cuda'
