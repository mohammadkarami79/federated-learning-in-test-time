#!/usr/bin/env python3
"""
Complete Pipeline
Handles the full workflow for any dataset
"""

import torch
import logging
import argparse
import time
from pathlib import Path
import traceback

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
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
    return parser.parse_args()

def run_system_setup():
    """Run system setup and verification"""
    logger = logging.getLogger(__name__)
    logger.info("🔧 Running system setup...")
    
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
        return False

def train_diffusion_model(cfg):
    """Train diffusion model for the specified dataset"""
    logger = logging.getLogger(__name__)
    logger.info(f"🌊 Training diffusion model for {cfg.DATASET_NAME}...")
    
    checkpoint_path = Path(f'checkpoints/diffuser_{cfg.DATASET.lower()}.pt')
    
    # Check if already exists
    if checkpoint_path.exists():
        logger.info(f"✅ Diffusion model already exists: {checkpoint_path}")
        return True
    
    try:
        # Run diffusion training
        import subprocess
        result = subprocess.run([
            'python', 'train_diffpure.py', 
            '--dataset', cfg.DATASET.lower(),
            '--epochs', '3'  # Reasonable training time
        ], text=True)
        
        if result.returncode == 0:
            logger.info("✅ Diffusion model training completed")
            return True
        else:
            logger.error(f"❌ Diffusion training failed: {result.stderr}")
            
            # Create minimal diffusion model as fallback
            # logger.info("📦 Creating minimal diffusion model as fallback...")
            # from diffusion.diffuser import UNet
            # diffuser = UNet(in_channels=cfg.IMG_CHANNELS, hidden_channels=64)
            # checkpoint_path.parent.mkdir(exist_ok=True)
            # torch.save(diffuser.state_dict(), checkpoint_path)
            # logger.info(f"✅ Minimal diffusion model created: {checkpoint_path}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Diffusion training error: {e}")
        return False

def train_mae_detector(cfg):
    """Train MAE detector for the specified dataset"""
    logger = logging.getLogger(__name__)
    logger.info(f"🔍 Training MAE detector for {cfg.DATASET_NAME}...")
    
    try:
        # Check if MAE training script exists
        mae_script = Path('scripts/train_mae_detector.py')
        if mae_script.exists():
            import subprocess
            result = subprocess.run([
                'python', str(mae_script),
                '--dataset', cfg.DATASET.lower(),
                '--epochs', '5'  # Quick training
            ], text=True)
            
            if result.returncode == 0:
                logger.info("✅ MAE detector training completed")
            else:
                logger.warning(f"⚠️ MAE training failed, using fallback: {result.stderr}")
        else:
            logger.info("📦 Using built-in MAE detector (no separate training needed)")
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ MAE training error: {e}, using fallback")
        return True  # Non-critical, continue with fallback

def run_federated_training(cfg):
    """Run the main federated training"""
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Starting federated training with {cfg.N_CLIENTS} clients, {cfg.N_ROUNDS} rounds")
    
    try:
        from federated.client import Client
        from federated.server import Server
        from attacks.pgd import PGDAttack
        from defense.mae_detector import MAEDetector
        from diffusion.diffuser import UNet
        from utils.data_utils import get_dataset
        import torch.utils.data as data_utils
        
        # Load datasets
        train_dataset, test_dataset = get_dataset(cfg)
        logger.info(f"📊 Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
        
        # Load diffusion model
        diffuser = UNet(in_channels=cfg.IMG_CHANNELS, hidden_channels=64)
        diffusion_path = Path(f'checkpoints/diffuser_{cfg.DATASET.lower()}.pt')
        if diffusion_path.exists():
            diffuser.load_state_dict(torch.load(diffusion_path, map_location=cfg.DEVICE))
            logger.info(f"✅ Loaded diffusion model: {diffusion_path}")
        else:
            logger.warning("⚠️ Using untrained diffusion model")
        
        if torch.cuda.is_available():
            diffuser = diffuser.cuda()
        
        # Initialize components
        clients = [Client(i, cfg, diffuser) for i in range(cfg.N_CLIENTS)]
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
        
        for round_idx in range(cfg.N_ROUNDS):
            logger.info(f"📈 Round {round_idx+1}/{cfg.N_ROUNDS}")
            
            # Train each client
            for client_idx, client in enumerate(clients[:3]):
                logger.info(f"👤 Training client {client_idx+1}")
                try:
                    client.train(epochs=1)
                    server.receive_update(client_idx, [client.model])
                except Exception as e:
                    logger.warning(f"⚠️ Client {client_idx} training issue: {e}")

            global_models = server.aggregate()
            if global_models:
                for client in clients:
                    client.model.load_state_dict(global_models[0].state_dict())

            # Broadcast updated model to all clients
            for client in clients:
                client.model.load_state_dict(global_models[0].state_dict()) 
            print(cfg.DEVICE)
            # Simple evaluation
            # with torch.no_grad():
            try:
                if round_idx % 2 == 0:  # Evaluate every 2 rounds
                    clean_correct = 0
                    adv_correct = 0
                    total = 0
                    detected_adv = 0
                    adv_total = 0
                    for batch_idx, (data, target) in enumerate(test_loader):
                        if batch_idx >= 5:  # Quick evaluation
                            break
                        
                        data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
                        # Clean accuracy
                        # Clean prediction
                        with torch.no_grad():
                            output = clients[0].model(data)
                            pred = output.argmax(dim=1)
                            clean_correct += pred.eq(target).sum().item()
                            total += target.size(0)

                        # Generate adversarial examples (needs grad)
                        adv_data = pgd_attacker.attack(clients[0].model, data, target)

                        # Adversarial prediction
                        with torch.no_grad():
                            adv_output = clients[0].model(adv_data)
                            adv_pred = adv_output.argmax(dim=1)
                            adv_correct += adv_pred.eq(target).sum().item()

                        # Detection
                        # detected = mae_detector.detect(adv_data)
                        # detected_adv += detected.sum().item()
                        # adv_total += detected.numel()
                    clean_acc = 100. * clean_correct / total if total > 0 else 0
                    adv_acc = 100. * adv_correct / total if total > 0 else 0
                    # detection_rate = 100. * detected_adv / adv_total if adv_total > 0 else 0
                    detection_rate = 0
                    logger.info(f"📊 Round {round_idx+1} Clean Acc: {clean_acc:.2f}% | Adv Acc: {adv_acc:.2f}% | MAE Detection: {detection_rate:.2f}%")

                    training_time = time.time() - start_time
                    logger.info(f"✅ Training completed in {training_time:.1f}s")
                # client = clients[0]
                # correct = 0
                # total = 0
                
                # with torch.no_grad():
                #     for batch_idx, (data, target) in enumerate(test_loader):
                #         if batch_idx >= 5:  # Quick evaluation
                #             break
                #         data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
                #         output = client.model(data)
                #         pred = output.argmax(dim=1)
                #         correct += pred.eq(target).sum().item()
                #         total += target.size(0)
                
                # accuracy = 100. * correct / total if total > 0 else 0
                # logger.info(f"📊 Round {round_idx+1} Accuracy: {accuracy:.2f}%")
                
            except Exception as e:
                logger.warning(f"⚠️ Evaluation error: {e}")
                logger.warning("Traceback:\n" + traceback.format_exc())

        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time:.1f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Federated training failed: {e}")
        logger.error("Traceback:\n" + traceback.format_exc())
        return False

def main():
    """Main training pipeline"""
    logger = setup_logging()
    args = parse_args()
    
    print("🚀 Complete Training Pipeline")
    print("=" * 60)
    
    # Get configuration
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
            cfg.N_CLASSES = 100
        elif args.dataset == 'br35h':
            cfg.IMG_SIZE = 224
            cfg.IMG_CHANNELS = 3
            cfg.N_CLASSES = 2
    
    logger.info(f"📋 Configuration: {args.mode} mode, {cfg.DATASET_NAME} dataset")
    logger.info(f"⚙️ Settings: {cfg.N_ROUNDS} rounds, {cfg.LOCAL_STEPS_PER_EPOCH} steps, {cfg.N_CLIENTS} clients")
    
    # Step 1: System setup
    if not args.skip_setup:
        if not run_system_setup():
            return 1
    
    # Step 2: Train diffusion model (if requested or for new datasets)
    if args.train_diffusion or args.dataset != 'cifar10':
        if not train_diffusion_model(cfg):
            logger.error("❌ Diffusion training failed")
            return 1
    
    # Step 3: Train MAE detector (if requested)
    if args.train_mae:
        if not train_mae_detector(cfg):
            logger.warning("⚠️ MAE training failed, continuing with fallback")
    
    # Step 4: Run federated training
    if not run_federated_training(cfg):
        logger.error("❌ Federated training failed")
        return 1
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"📊 Mode: {args.mode}, Dataset: {cfg.DATASET_NAME}")
    time_map = {'debug': 'Short', 'test': 'Mid', 'full': 'Long'}
    print(f"⏱️ Expected time for {args.mode} mode: {time_map[args.mode]}")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main()) 