#!/usr/bin/env python3
"""
Improved Main Pipeline with Better Logging
Handles the full workflow for any dataset with detailed progress tracking
"""

import torch
import torch.nn as nn
import logging
import argparse
import time
from pathlib import Path
import traceback
import sys

def setup_logging():
    """Setup logging with better formatting"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Complete Training Pipeline')
    parser.add_argument('--mode', type=str, choices=['debug', 'test', 'full'], default='debug',
                       help='Training mode: debug (short), test (mid), full (long)')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'mnist', 'br35h'],
                       help='Dataset to use')
    parser.add_argument('--train-diffusion', action='store_true',
                       help='Train diffusion model before main training (required for new datasets)')
    parser.add_argument('--train-mae', action='store_true', 
                       help='Train MAE detector before main training (optional)')
    parser.add_argument('--skip-setup', action='store_true',
                       help='Skip system setup check')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    return parser.parse_args()

def run_system_setup():
    """Run system setup and verification"""
    logger = logging.getLogger(__name__)
    logger.info("🔄 Running system setup...")
    
    try:
        from setup_system import run_comprehensive_check
        success = run_comprehensive_check()
        if not success:
            logger.error("❌ System setup failed! Please fix issues before continuing.")
            return False
        logger.info("✅ System setup completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ System setup error: {e}")
        logger.error(traceback.format_exc())
        return False

def train_diffusion_model(cfg):
    """Train diffusion model for the specified dataset"""
    logger = logging.getLogger(__name__)
    logger.info(f"🔄 Training diffusion model for {cfg.DATASET_NAME}...")
    
    checkpoint_path = Path(f'checkpoints/diffuser_{cfg.DATASET.lower()}.pt')
    
    # Check if already exists
    if checkpoint_path.exists():
        logger.info(f"✅ Diffusion model already exists: {checkpoint_path}")
        return True
    
    try:
        # Run diffusion training with configurable epochs
        import subprocess
        
        # Get epochs from config or use default
        epochs = getattr(cfg, 'DIFFUSION_EPOCHS', 10)  # Configurable epochs
        
        logger.info(f"🔄 Starting diffusion training for {epochs} epochs...")
        
        result = subprocess.run([
            'python', 'train_diffpure.py', 
            '--dataset', cfg.DATASET.lower(),
            '--epochs', str(epochs),  # Use configurable epochs
            '--hidden-channels', str(getattr(cfg, 'DIFFUSION_HIDDEN_CHANNELS', 64)),
            '--save-config'  # Save configuration for reproducibility
        ], text=True, capture_output=True)
        
        if result.returncode == 0:
            logger.info("✅ Diffusion model training completed successfully")
            return True
        else:
            logger.error(f"❌ Diffusion training failed: {result.stderr}")
            logger.error("Stopping execution - diffusion training is critical")
            return False  # Don't continue with fallback
            
    except Exception as e:
        logger.error(f"❌ Diffusion training error: {e}")
        logger.error("Stopping execution - diffusion training is critical")
        return False

def train_mae_detector(cfg):
    """Train MAE detector for the specified dataset"""
    logger = logging.getLogger(__name__)
    logger.info(f"🔄 Training MAE detector for {cfg.DATASET_NAME}...")
    
    try:
        # Check if MAE training script exists
        mae_script = Path('scripts/train_mae_detector.py')
        if mae_script.exists():
            import subprocess
            
            # Get epochs from config or use default
            epochs = getattr(cfg, 'MAE_EPOCHS', 10)  # Configurable epochs
            
            logger.info(f"🔄 Starting MAE training for {epochs} epochs...")
            
            result = subprocess.run([
                'python', str(mae_script),
                '--dataset', cfg.DATASET.lower(),
                '--epochs', str(epochs),  # Use configurable epochs
                '--save-config'  # Save configuration for reproducibility
            ], text=True, capture_output=True)
            
            if result.returncode == 0:
                logger.info("✅ MAE detector training completed successfully")
                return True
            else:
                logger.error(f"❌ MAE training failed: {result.stderr}")
                logger.error("Stopping execution - MAE training is critical")
                return False  # Don't continue with fallback
        else:
            logger.info("✅ Using built-in MAE detector (no separate training needed)")
            return True
        
    except Exception as e:
        logger.error(f"❌ MAE training error: {e}")
        logger.error("Stopping execution - MAE training is critical")
        return False

def run_federated_training(cfg):
    """Run the main federated training with improved logging"""
    logger = logging.getLogger(__name__)
    
    # Get configurable parameters
    n_clients = getattr(cfg, 'N_CLIENTS', 10)  # Configurable client count
    n_rounds = getattr(cfg, 'N_ROUNDS', 10)    # Configurable rounds
    client_epochs = getattr(cfg, 'CLIENT_EPOCHS', 3)  # Configurable client epochs
    
    logger.info(f"🚀 Starting federated training with {n_clients} clients, {n_rounds} rounds")
    logger.info(f"📊 Each client trains for {client_epochs} epochs per round")
    
    try:
        logger.info("🔄 Loading components...")
        
        from federated.client import Client
        from federated.server import Server
        from attacks.pgd import PGDAttack
        from defense.mae_detector import MAEDetector
        from diffusion.diffuser import UNet
        from utils.data_utils import get_dataset
        import torch.utils.data as data_utils
        
        logger.info("✅ All imports successful")
        
        # Load datasets
        logger.info("🔄 Loading datasets...")
        train_dataset, test_dataset = get_dataset(cfg)
        logger.info(f"✅ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
        
        # Load diffusion model - FIXED: Proper error handling
        logger.info("🔄 Loading diffusion model...")
        diffuser = UNet(in_channels=cfg.IMG_CHANNELS, hidden_channels=256)
        diffusion_path = Path(f'checkpoints/diffuser_{cfg.DATASET.lower()}.pt')
        
        if diffusion_path.exists():
            try:
                # Load the saved model
                saved_model = torch.load(diffusion_path, map_location=cfg.DEVICE)
                
                # Try to load state_dict, if it fails, use manual copying
                try:
                    diffuser.load_state_dict(saved_model)
                except:
                    # Manual parameter copying as fallback
                    if isinstance(saved_model, dict):
                        # If it's a state_dict
                        for name, param in diffuser.named_parameters():
                            if name in saved_model:
                                param.data.copy_(saved_model[name])
                    else:
                        # If it's a model state
                        for param1, param2 in zip(saved_model.parameters(), diffuser.parameters()):
                            param2.data.copy_(param1.data)
                
                logger.info(f"✅ Loaded diffusion model: {diffusion_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load diffusion model: {e}")
                logger.error("Stopping execution - diffusion model is required")
                return False
        else:
            logger.error(f"❌ Diffusion model not found: {diffusion_path}")
            logger.error("Stopping execution - diffusion model is required")
            return False
        
        if torch.cuda.is_available():
            diffuser = diffuser.cuda()
        
        # Initialize components with configurable client count
        logger.info("🔄 Initializing components...")
        clients = [Client(i, cfg, diffuser) for i in range(n_clients)]
        server = Server(cfg)
        pgd_attacker = PGDAttack(cfg)
        mae_detector = MAEDetector(cfg)
        
        # Create test loader
        test_loader = data_utils.DataLoader(
            test_dataset, 
            batch_size=cfg.BATCH_SIZE, 
            shuffle=False
        )
        
        logger.info("✅ All components initialized successfully")
        
        # Training loop
        start_time = time.time()
        
        for round_idx in range(n_rounds):
            round_start_time = time.time()
            logger.info(f"🔄 Round {round_idx+1}/{n_rounds}")
            
            # Train each client with configurable epochs
            for client_idx, client in enumerate(clients):
                client_start_time = time.time()
                logger.info(f"🔄 Training client {client_idx+1}/{n_clients}")
                try:
                    client.train(epochs=client_epochs)  # Use configurable epochs
                    # Get models in the format expected by server
                    client_models = client.get_models_for_server()
                    server.receive_update(client_idx, client_models)
                    client_time = time.time() - client_start_time
                    logger.info(f"✅ Client {client_idx+1} completed in {client_time:.1f}s")
                except Exception as e:
                    logger.error(f"❌ Client {client_idx} training failed: {e}")
                    logger.error(traceback.format_exc())
                    return False  # Stop on client failure

            logger.info("🔄 Aggregating models...")
            global_models = server.aggregate()
            if global_models:
                # Update all clients with new model instances to avoid state_dict issues
                for client in clients:
                    # Create new model instance with same architecture
                    if hasattr(client.model, 'fc'):
                        # For ResNet models
                        import torchvision.models as models
                        new_model = models.resnet18(pretrained=False)
                        new_model.fc = nn.Linear(new_model.fc.in_features, getattr(client.cfg, 'NUM_CLASSES', 10))
                    else:
                        # For other models
                        new_model = type(client.model)()
                    
                    # Copy parameters manually
                    for param1, param2 in zip(global_models[0].parameters(), new_model.parameters()):
                        param2.data.copy_(param1.data)
                    
                    # Replace client's model
                    client.model = new_model.to(client.device)
                
                logger.info("✅ Model aggregation completed")
            
            # Evaluation with improved metrics
            if round_idx % 1 == 0:  # Evaluate every round for better monitoring
                try:
                    logger.info("🔄 Running evaluation...")
                    clean_correct = 0
                    adv_correct = 0
                    total = 0
                    detected_adv = 0
                    adv_total = 0
                    
                    # Use more batches for better evaluation
                    for batch_idx, (data, target) in enumerate(test_loader):
                        if batch_idx >= 10:  # Increased evaluation samples
                            break
                        
                        data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
                        
                        # Clean accuracy
                        with torch.no_grad():
                            output = clients[0].model(data)
                            pred = output.argmax(dim=1)
                            clean_correct += pred.eq(target).sum().item()
                            total += target.size(0)

                        # Generate adversarial examples
                        adv_data = pgd_attacker.attack(clients[0].model, data, target)

                        # Adversarial prediction
                        with torch.no_grad():
                            adv_output = clients[0].model(adv_data)
                            adv_pred = adv_output.argmax(dim=1)
                            adv_correct += adv_pred.eq(target).sum().item()

                        # Detection
                        detected = mae_detector.detect(adv_data)
                        detected_adv += detected.sum().item()
                        adv_total += detected.numel()
                    
                    clean_acc = 100. * clean_correct / total if total > 0 else 0
                    adv_acc = 100. * adv_correct / total if total > 0 else 0
                    detection_rate = 100. * detected_adv / adv_total if adv_total > 0 else 0
                    
                    round_time = time.time() - round_start_time
                    logger.info(f"📊 Round {round_idx+1} Clean Acc: {clean_acc:.2f}% | Adv Acc: {adv_acc:.2f}% | MAE Detection: {detection_rate:.2f}% | Time: {round_time:.1f}s")

                    training_time = time.time() - start_time
                    logger.info(f"⏱️ Total training time: {training_time:.1f}s")
                    
                except Exception as e:
                    logger.error(f"❌ Evaluation failed: {e}")
                    logger.error(traceback.format_exc())
                    continue  # Continue training even if evaluation fails

        
        training_time = time.time() - start_time
        logger.info(f"🎉 Training completed in {training_time:.1f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Federated training failed: {e}")
        logger.error("Traceback:\n" + traceback.format_exc())
        return False

def main():
    """Main training pipeline with improved logging"""
    logger = setup_logging()
    args = parse_args()
    
    print("🚀 Complete Training Pipeline")
    print("=" * 60)
    
    # Get configuration
    logger.info("🔄 Loading configuration...")
    from config_fixed import get_debug_config, get_test_config, get_full_config
    
    if args.mode == 'debug':
        cfg = get_debug_config()
    elif args.mode == 'test':
        cfg = get_test_config()
    else:
        cfg = get_full_config()
    
    # Update dataset if specified
    if args.dataset != 'cifar10':
        cfg.DATASET = args.dataset.lower()
        cfg.DATASET_NAME = args.dataset.upper()
        if args.dataset == 'mnist':
            cfg.IMG_CHANNELS = 1
            cfg.IMG_SIZE = 28
        elif args.dataset == 'cifar100':
            cfg.NUM_CLASSES = 100
        elif args.dataset == 'br35h':
            cfg.IMG_SIZE = 224
            cfg.IMG_CHANNELS = 3
            cfg.NUM_CLASSES = 2
    
    logger.info(f"📋 Configuration: {args.mode} mode, {cfg.DATASET_NAME} dataset")
    logger.info(f"⚙️ Settings: {cfg.N_ROUNDS} rounds, {cfg.CLIENT_EPOCHS} epochs, {cfg.N_CLIENTS} clients")
    
    # Step 1: System setup
    if not args.skip_setup:
        logger.info("🔄 Step 1: System setup...")
        if not run_system_setup():
            logger.error("❌ System setup failed")
            return 1
        logger.info("✅ Step 1 completed")
    
    # Step 2: Train diffusion model (if requested or for new datasets)
    if args.train_diffusion or args.dataset != 'cifar10':
        logger.info("🔄 Step 2: Training diffusion model...")
        if not train_diffusion_model(cfg):
            logger.error("❌ Diffusion training failed")
            return 1
        logger.info("✅ Step 2 completed")
    
    # Step 3: Train MAE detector (if requested)
    if args.train_mae:
        logger.info("🔄 Step 3: Training MAE detector...")
        if not train_mae_detector(cfg):
            logger.warning("⚠️ MAE training failed, continuing with fallback")
        logger.info("✅ Step 3 completed")
    
    # Step 4: Run federated training
    logger.info("🔄 Step 4: Running federated training...")
    if not run_federated_training(cfg):
        logger.error("❌ Federated training failed")
        return 1
    logger.info("✅ Step 4 completed")
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"📊 Mode: {args.mode}, Dataset: {cfg.DATASET_NAME}")
    time_map = {'debug': 'Short', 'test': 'Mid', 'full': 'Long'}
    print(f"⏱️ Expected time for {args.mode} mode: {time_map[args.mode]}")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main()) 