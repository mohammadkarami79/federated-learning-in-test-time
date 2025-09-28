#!/usr/bin/env python3
"""
Simplified Main Pipeline with Timeout Protection
Handles the core workflow with better error handling
"""

import torch
import torch.nn as nn
import logging
import argparse
import time
from pathlib import Path
import traceback
import sys
import signal

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
    parser = argparse.ArgumentParser(description='Simplified Training Pipeline')
    parser.add_argument('--mode', type=str, choices=['debug', 'test', 'full'], default='debug',
                       help='Training mode: debug (short), test (mid), full (long)')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'mnist', 'br35h'],
                       help='Dataset to use')
    parser.add_argument('--skip-setup', action='store_true',
                       help='Skip system setup check')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout in seconds for each step')
    return parser.parse_args()

def timeout_handler(signum, frame):
    """Handle timeout"""
    raise TimeoutError("Operation timed out")

def run_with_timeout(func, timeout_seconds, *args, **kwargs):
    """Run function with timeout"""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)
        return result
    except TimeoutError:
        signal.alarm(0)
        raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")

def test_basic_components(cfg):
    """Test basic components without full training"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🔄 Testing basic components...")
        
        # Test dataset loading
        logger.info("🔄 Loading dataset...")
        from utils.data_utils import get_dataset
        train_dataset, test_dataset = get_dataset(cfg)
        logger.info(f"✅ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
        
        # Test model creation
        logger.info("🔄 Creating model...")
        from models import get_model
        model = get_model(cfg)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Model created: {total_params:,} parameters")
        
        # Test diffusion model loading
        logger.info("🔄 Loading diffusion model...")
        from diffusion.diffuser import UNet
        diffuser = UNet(in_channels=cfg.IMG_CHANNELS, hidden_channels=256)
        diffusion_path = Path(f'checkpoints/diffuser_{cfg.DATASET.lower()}.pt')
        
        if diffusion_path.exists():
            try:
                saved_model = torch.load(diffusion_path, map_location=cfg.DEVICE)
                diffuser.load_state_dict(saved_model)
                logger.info(f"✅ Diffusion model loaded: {diffusion_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load diffusion model: {e}")
                return False
        else:
            logger.error(f"❌ Diffusion model not found: {diffusion_path}")
            return False
        
        # Test client creation
        logger.info("🔄 Testing client creation...")
        from federated.client import Client
        client = Client(0, cfg, diffuser)
        logger.info("✅ Client created successfully")
        
        # Test server creation
        logger.info("🔄 Testing server creation...")
        from federated.server import Server
        server = Server(cfg)
        logger.info("✅ Server created successfully")
        
        # Test attack creation
        logger.info("🔄 Testing attack creation...")
        from attacks.pgd import PGDAttack
        attacker = PGDAttack(cfg)
        logger.info("✅ Attack created successfully")
        
        # Test MAE detector creation
        logger.info("🔄 Testing MAE detector creation...")
        from defense.mae_detector import MAEDetector
        detector = MAEDetector(cfg)
        logger.info("✅ MAE detector created successfully")
        
        logger.info("✅ All basic components working!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Component test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def run_simple_training(cfg, timeout_seconds=300):
    """Run simplified training with timeout protection"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 Starting simplified training...")
        
        # Load components
        from federated.client import Client
        from federated.server import Server
        from attacks.pgd import PGDAttack
        from defense.mae_detector import MAEDetector
        from diffusion.diffuser import UNet
        from utils.data_utils import get_dataset
        import torch.utils.data as data_utils
        
        # Load dataset
        train_dataset, test_dataset = get_dataset(cfg)
        test_loader = data_utils.DataLoader(
            test_dataset, 
            batch_size=cfg.BATCH_SIZE, 
            shuffle=False
        )
        
        # Load diffusion model
        diffuser = UNet(in_channels=cfg.IMG_CHANNELS, hidden_channels=256)
        diffusion_path = Path(f'checkpoints/diffuser_{cfg.DATASET.lower()}.pt')
        saved_model = torch.load(diffusion_path, map_location=cfg.DEVICE)
        diffuser.load_state_dict(saved_model)
        
        if torch.cuda.is_available():
            diffuser = diffuser.cuda()
        
        # Create components
        n_clients = min(3, getattr(cfg, 'N_CLIENTS', 10))  # Use fewer clients for testing
        n_rounds = min(2, getattr(cfg, 'N_ROUNDS', 10))    # Use fewer rounds for testing
        client_epochs = 1  # Use single epoch for testing
        
        logger.info(f"📊 Training with {n_clients} clients, {n_rounds} rounds, {client_epochs} epochs per client")
        
        clients = [Client(i, cfg, diffuser) for i in range(n_clients)]
        server = Server(cfg)
        pgd_attacker = PGDAttack(cfg)
        mae_detector = MAEDetector(cfg)
        
        start_time = time.time()
        
        for round_idx in range(n_rounds):
            round_start_time = time.time()
            logger.info(f"🔄 Round {round_idx+1}/{n_rounds}")
            
            # Train clients
            for client_idx, client in enumerate(clients):
                client_start_time = time.time()
                logger.info(f"🔄 Training client {client_idx+1}/{n_clients}")
                
                try:
                    client.train(epochs=client_epochs)
                    client_models = client.get_models_for_server()
                    server.receive_update(client_idx, client_models)
                    client_time = time.time() - client_start_time
                    logger.info(f"✅ Client {client_idx+1} completed in {client_time:.1f}s")
                except Exception as e:
                    logger.error(f"❌ Client {client_idx} failed: {e}")
                    return False
            
            # Aggregate
            logger.info("🔄 Aggregating models...")
            global_models = server.aggregate()
            if global_models:
                for client in clients:
                    if hasattr(client.model, 'fc'):
                        import torchvision.models as models
                        new_model = models.resnet18(pretrained=False)
                        new_model.fc = nn.Linear(new_model.fc.in_features, getattr(client.cfg, 'NUM_CLASSES', 10))
                    else:
                        new_model = type(client.model)()
                    
                    for param1, param2 in zip(global_models[0].parameters(), new_model.parameters()):
                        param2.data.copy_(param1.data)
                    
                    client.model = new_model.to(client.device)
                
                logger.info("✅ Model aggregation completed")
            
            # Quick evaluation
            try:
                logger.info("🔄 Running evaluation...")
                clean_correct = 0
                total = 0
                
                for batch_idx, (data, target) in enumerate(test_loader):
                    if batch_idx >= 5:  # Use fewer batches for quick evaluation
                        break
                    
                    data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
                    
                    with torch.no_grad():
                        output = clients[0].model(data)
                        pred = output.argmax(dim=1)
                        clean_correct += pred.eq(target).sum().item()
                        total += target.size(0)
                
                clean_acc = 100. * clean_correct / total if total > 0 else 0
                round_time = time.time() - round_start_time
                logger.info(f"📊 Round {round_idx+1} Clean Acc: {clean_acc:.2f}% | Time: {round_time:.1f}s")
                
            except Exception as e:
                logger.error(f"❌ Evaluation failed: {e}")
                continue
        
        training_time = time.time() - start_time
        logger.info(f"🎉 Training completed in {training_time:.1f}s")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function with timeout protection"""
    logger = setup_logging()
    args = parse_args()
    
    print("🚀 Simplified Training Pipeline")
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
    
    # Test basic components first
    logger.info("🔄 Step 1: Testing basic components...")
    if not test_basic_components(cfg):
        logger.error("❌ Basic component test failed")
        return 1
    logger.info("✅ Step 1 completed")
    
    # Run simplified training
    logger.info("🔄 Step 2: Running simplified training...")
    try:
        success = run_with_timeout(run_simple_training, args.timeout, cfg, args.timeout)
        if success:
            logger.info("✅ Step 2 completed")
        else:
            logger.error("❌ Training failed")
            return 1
    except TimeoutError as e:
        logger.error(f"❌ Training timed out: {e}")
        return 1
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("🎉 SIMPLIFIED TRAINING COMPLETED SUCCESSFULLY!")
    print(f"📊 Mode: {args.mode}, Dataset: {cfg.DATASET_NAME}")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main()) 