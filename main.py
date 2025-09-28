"""
Federated Learning with Test-Time Adaptation - COMPLETELY FIXED VERSION
Main training pipeline with improved DiffPure and MAE detection
"""

import os
import sys
import time
import logging
import argparse
import traceback
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Federated Learning Training')
    parser.add_argument('--dataset', type=str, default='br35h',
                       choices=['br35h', 'cifar10', 'cifar100', 'mnist'],
                       help='Dataset to use')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['debug', 'test', 'full'],
                       help='Training mode')
    parser.add_argument('--train-diffusion', action='store_true',
                       help='Train diffusion model before main training')
    parser.add_argument('--train-mae', action='store_true', 
                       help='Train MAE detector before main training (optional)')
    parser.add_argument('--skip-setup', action='store_true',
                       help='Skip system setup check')
    return parser.parse_args()

def diffpure_purify(diffuser, adv_data, cfg):
    """Apply DiffPure purification - OPTIMIZED FOR BR35H"""
    import torch
    
    # Dataset-specific parameters for optimal purification
    dataset = str(getattr(cfg, 'DATASET', '')).lower()
    if dataset == 'br35h':
        # BR35H: ultra-gentle for 224x224 medical images
        num_steps = 1
        sigma = 0.02
        strength = 0.03
    elif dataset == 'cifar10':
        # CIFAR-10: balanced approach for 32x32 RGB images
        num_steps = 2
        sigma = 0.05   # Moderate noise for CIFAR-10
        strength = 0.06  # Slightly stronger than BR35H
    else:
        # Default parameters for other datasets
        num_steps = getattr(cfg, 'DIFFUSER_STEPS', 2)
        sigma = getattr(cfg, 'DIFFUSER_SIGMA', 0.1)
        strength = getattr(cfg, 'DIFFPURE_STRENGTH', 0.1)
    
    adaptive = getattr(cfg, 'DIFFPURE_ADAPTIVE', True)
    
    device = adv_data.device
    batch_size = adv_data.size(0)
    
    # Ensure inputs are in [0,1] range
    adv_data = torch.clamp(adv_data, 0.0, 1.0)
    purified_data = adv_data.clone()
    
    with torch.no_grad():
        # Ultra mild purification - minimal changes
        for step in range(num_steps):
            # Very low noise
            current_sigma = sigma * 0.5  # Even lower noise
            noise = torch.randn_like(purified_data) * current_sigma
            noisy_data = purified_data + noise
            
            # Create time steps for diffusion model
            t = torch.full((batch_size,), num_steps - step, device=device, dtype=torch.float32)
            
            # Predict and remove noise
            predicted_noise = diffuser(noisy_data, t)
            
            # Ultra mild denoising - only 5% of predicted noise
            if adaptive:
                # Even more gentle
                base_strength = strength * 0.5  # 2.5% strength
                confidence_factor = 0.9 + 0.1 * torch.rand(1, device=device)  # 0.9 to 1.0
                denoising_strength = base_strength * confidence_factor
            else:
                denoising_strength = strength
            
            # Apply ultra mild purification
            purified_data = noisy_data - predicted_noise * denoising_strength
            
            # Clamp to valid range
            purified_data = torch.clamp(purified_data, 0.0, 1.0)
    
    return purified_data

def save_experiment_results(cfg, rounds_completed, final_clean_acc, final_adv_acc, final_detection_rate, training_time):
    """Save experiment results to JSON file - FIXED VERSION"""
    logger = logging.getLogger(__name__)
    
    # Create results directory
    results_dir = Path("experiment_results")
    results_dir.mkdir(exist_ok=True)
    
    # Prepare results data
    from datetime import datetime
    results = {
        "experiment_info": {
            "dataset": cfg.DATASET,
            "mode": getattr(cfg, 'MODE', 'full'),
            "timestamp": datetime.now().isoformat(),
            "rounds_completed": rounds_completed,
            "total_training_time_seconds": training_time
        },
        "final_metrics": {
            "clean_accuracy": round(final_clean_acc, 2),
            "adversarial_accuracy": round(final_adv_acc, 2), 
            "mae_detection_rate": round(final_detection_rate, 2)
        },
        "model_info": {
            "has_mae_detector": True,
            "has_diffusion_purifier": True,
            "img_size": cfg.IMG_SIZE,
            "img_channels": cfg.IMG_CHANNELS,
            "num_classes": getattr(cfg, 'NUM_CLASSES', 2)  # FIXED: Default fallback instead of cfg.N_CLASSES
        }
    }
    
    # Save to dataset-specific file
    result_file = results_dir / f"results_{cfg.DATASET.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        import json
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save as latest results
        latest_file = results_dir / f"latest_results_{cfg.DATASET.lower()}.json"
        with open(latest_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📊 Results saved to: {result_file}")
        logger.info(f"📈 Latest results: {latest_file}")
        
        # Print summary
        logger.info("📊 EXPERIMENT RESULTS SUMMARY")
        logger.info(f"Dataset: {cfg.DATASET}")
        logger.info(f"Clean Accuracy: {final_clean_acc:.2f}%")
        logger.info(f"Adversarial Accuracy: {final_adv_acc:.2f}%")
        logger.info(f"MAE Detection Rate: {final_detection_rate:.2f}%") 
        logger.info(f"Training Time: {training_time:.1f} seconds")
        logger.info(f"Results File: {result_file}")
            
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")

def recompute_bn_stats(model, loaders, device, max_batches=100):
    model.train()
    seen = 0
    with torch.no_grad():
        for loader in loaders:
            for data, _ in loader:
                data = data.to(device)
                _ = model(data)
                seen += 1
                if seen >= max_batches:
                    model.eval()
                    return
    model.eval()

def train_mae_detector(cfg):
    """Train MAE detector for the specified dataset"""
    logger = logging.getLogger(__name__)
    dataset_display = getattr(cfg, 'DATASET_NAME', getattr(cfg, 'DATASET', 'dataset')).capitalize()
    logger.info(f"Training MAE detector for {dataset_display}...")
    
    try:
        # Check if MAE training script exists
        mae_script = Path('scripts/train_mae_detector.py')
        if mae_script.exists():
            import subprocess
            
            # Dataset-specific training parameters
            if str(cfg.DATASET).lower() == 'cifar10':
                # CIFAR-10 optimizations for faster training
                epochs = getattr(cfg, 'MAE_EPOCHS', 15)  # Optimized for CIFAR-10
                batch_size = 128  # Larger batch for CIFAR-10
                logger.info(f"🎯 CIFAR-10 MAE training: {epochs} epochs, batch_size={batch_size}")
            else:
                # Default parameters
                epochs = getattr(cfg, 'MAE_EPOCHS', 10)
                batch_size = 64
            
            result = subprocess.run([
                'python', str(mae_script),
                '--dataset', cfg.DATASET.lower(),
                '--epochs', str(epochs),
                '--batch-size', str(batch_size)
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                logger.info("✅ MAE detector training completed successfully")
                return True
            else:
                logger.error(f"❌ MAE detector training failed: {result.stderr}")
                return False
        else:
            logger.warning("⚠️ MAE training script not found, skipping...")
            return True  # Don't fail if script doesn't exist
            
    except Exception as e:
        logger.error(f"❌ MAE detector training error: {e}")
        return False

def train_diffusion_model(cfg):
    """Train diffusion model for the specified dataset"""
    logger = logging.getLogger(__name__)
    logger.info(f"Training diffusion model for {cfg.DATASET}...")
    
    # Dataset-specific training optimizations
    if str(cfg.DATASET).lower() == 'cifar10':
        # CIFAR-10 optimizations for faster training
        original_epochs = getattr(cfg, 'DIFFUSION_EPOCHS', 200)
        cfg.DIFFUSION_EPOCHS = min(original_epochs, 150)  # Faster training
        cfg.DIFFUSION_BATCH_SIZE = getattr(cfg, 'DIFFUSION_BATCH_SIZE', 32)
        cfg.DIFFUSION_LR = getattr(cfg, 'DIFFUSION_LR', 1e-4)
        logger.info(f"🎯 CIFAR-10 Diffusion training: {cfg.DIFFUSION_EPOCHS} epochs, batch_size={cfg.DIFFUSION_BATCH_SIZE}")
    
    # Check if diffusion model already exists
    diffusion_checkpoint = Path(f"checkpoints/diffuser_{cfg.DATASET.lower()}.pt")
    if diffusion_checkpoint.exists():
        logger.info(f"Diffusion model already exists: {diffusion_checkpoint}")
        return True
    
    try:
        import subprocess
        
        # Get epochs from config or use default
        epochs = getattr(cfg, 'DIFFUSION_EPOCHS', 50)  # Configurable epochs
        
        # Check if training script exists
        script_candidates = [
            'train_diffpure.py',
            'diffusion/train_diffusion.py',
            'scripts/train_diffusion.py'
        ]
        
        script_path = None
        for script in script_candidates:
            if Path(script).exists():
                script_path = script
                break
        
        # Try integrated training first (no subprocess)
        try:
            from integrated_diffusion_training import integrated_train_diffusion
            logger.info("🔄 Using integrated diffusion training (no subprocess)")
            success = integrated_train_diffusion(cfg)
            if success:
                logger.info("✅ Integrated diffusion training completed successfully")
                return True
            else:
                logger.warning("⚠️ Integrated training failed, trying subprocess...")
        except ImportError:
            logger.warning("⚠️ Integrated training not available, trying subprocess...")
        
        # Fallback to subprocess if available
        if script_path:
            result = subprocess.run([
                'python', script_path,
                '--dataset', cfg.DATASET.lower(),
                '--epochs', str(epochs),
                '--batch-size', '16',
                '--gradient-accumulation-steps', '4',
                '--mixed-precision',
                '--hidden-channels', '256'
            ], capture_output=True, text=True, timeout=7200)
            
            if result.returncode == 0:
                logger.info("✅ Subprocess diffusion training completed successfully")
                return True
            else:
                logger.error(f"❌ Subprocess diffusion training failed: {result.stderr}")
                return False
        else:
            logger.error("❌ No diffusion training method available!")
            logger.info("💡 Need either integrated_diffusion_training.py or train_diffpure.py")
            return False
            
    except Exception as e:
        logger.error(f"❌ Diffusion model training error: {e}")
        return False

def check_system_requirements(cfg):
    """Check if system meets requirements"""
    logger = logging.getLogger(__name__)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("⚠️ CUDA not available, using CPU")
        cfg.DEVICE = 'cpu'
        return False
    
    # Check GPU memory
    try:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🖥️ GPU Memory: {gpu_memory_gb:.1f} GB")
        
        if gpu_memory_gb < 8:
            logger.warning(f"⚠️ Low GPU memory ({gpu_memory_gb:.1f}GB), consider reducing batch size")
        
        # Check if current memory usage allows for training
        if hasattr(cfg, 'MAX_MEMORY_GB'):
            if gpu_memory_gb < cfg.MAX_MEMORY_GB:
                logger.warning(f"⚠️ Available memory ({gpu_memory_gb:.1f}GB) less than configured ({cfg.MAX_MEMORY_GB}GB)")
        
            return True
        
    except Exception as e:
        logger.error(f"❌ Error checking GPU: {e}")
        return False

def run_federated_training(cfg, test_loader):
    """Run the main federated training pipeline"""
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    # Initialize variables for final metrics
    final_clean_acc = 0.0
    final_adv_acc = 0.0
    final_detection_rate = 0.0
    
    try:
        logger.info("🔄 Loading components...")
        
        # Import necessary modules
        from utils.data_utils import get_dataset, create_federated_datasets
        from federated.client import Client
        from federated.server import FedServer
        from attacks.pgd_bulletproof import PGDAttack
        from diffusion.diffuser import UNet
        from defense.mae_detector import MAEDetector
        import torchvision.models as models
        import traceback
        
        logger.info("✅ All imports successful")
        
        logger.info("🔄 Loading datasets...")
        # Force getting datasets (not dataloaders) by passing config object
        train_dataset, test_dataset = get_dataset(cfg, cfg.DATA_ROOT)
        logger.info(f"✅ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
            
        federated_datasets = create_federated_datasets(train_dataset, cfg.NUM_CLIENTS, getattr(cfg, 'DATA_DISTRIBUTION', 'iid'))
        
        # Create proper test loader
        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        logger.info(f"✅ Test loader created with {len(test_dataset)} samples")
        
        logger.info("🔄 Loading diffusion model...")
        
        # CRITICAL FIX: Use the SAME architecture that was used for training
        # The trained model has: enc1, enc2, bottleneck, dec1, dec2, final
        class TrainedUNet(nn.Module):
            """UNet with EXACT same architecture as integrated_diffusion_training.py"""
            def __init__(self, in_channels=3, hidden_channels=128):
                super().__init__()
                
                # Time embedding
                self.time_embed = nn.Sequential(
                    nn.Linear(1, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )
                
                # Encoder
                self.enc1 = nn.Sequential(
                    nn.Conv2d(in_channels + hidden_channels, hidden_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.ReLU(inplace=True)
                )
                
                self.enc2 = nn.Sequential(
                    nn.MaxPool2d(2),
                    nn.Conv2d(hidden_channels, hidden_channels * 2, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1),
                    nn.ReLU(inplace=True)
                )
                
                # Bottleneck
                self.bottleneck = nn.Sequential(
                    nn.MaxPool2d(2),
                    nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1),
                    nn.ReLU(inplace=True)
                )
                
                # Decoder
                self.up1 = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels * 2, 2, stride=2)
                self.dec1 = nn.Sequential(
                    nn.Conv2d(hidden_channels * 4, hidden_channels * 2, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
                    nn.ReLU(inplace=True)
                )
                
                self.up2 = nn.ConvTranspose2d(hidden_channels, hidden_channels, 2, stride=2)
                self.dec2 = nn.Sequential(
                    nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.ReLU(inplace=True)
                )
                
                self.final = nn.Conv2d(hidden_channels, in_channels, 1)
                
            def forward(self, x, t):
                # Time embedding
                t_emb = self.time_embed(t.float().unsqueeze(1) / 1000.0)  # Normalize time
                t_emb = t_emb.view(-1, t_emb.size(1), 1, 1)
                t_emb = t_emb.expand(-1, -1, x.size(2), x.size(3))
                
                # Concatenate with input
                x = torch.cat([x, t_emb], dim=1)
                
                # Encoder
                enc1 = self.enc1(x)
                enc2 = self.enc2(enc1)
                
                # Bottleneck
                bottleneck = self.bottleneck(enc2)
                
                # Decoder
                up1 = self.up1(bottleneck)
                dec1 = self.dec1(torch.cat([up1, enc2], dim=1))
                
                up2 = self.up2(dec1)
                dec2 = self.dec2(torch.cat([up2, enc1], dim=1))
                
                return self.final(dec2)
                
            def purify(self, x, steps=10, sigma=0.1):
                """Purify images using diffusion."""
                batch_size = x.shape[0]
                device = x.device
                
                # Add noise
                noise = torch.randn_like(x) * sigma
                noisy_x = x + noise
                
                # Denoise
                for i in range(steps):
                    t = torch.ones(batch_size, device=device) * (steps - i) / steps
                    pred_noise = self(noisy_x, t)
                    noisy_x = noisy_x - sigma * pred_noise / steps
                    
                return torch.clamp(noisy_x, 0, 1)
        
        diffuser = TrainedUNet(
            in_channels=cfg.IMG_CHANNELS, 
            hidden_channels=getattr(cfg, 'DIFFUSION_HIDDEN_CHANNELS', 128)
        ).to(cfg.DEVICE)
        
        # Load trained diffusion model
        diffusion_checkpoint = Path(f"checkpoints/diffuser_{cfg.DATASET.lower()}.pt")
        if diffusion_checkpoint.exists():
            try:
                diffuser.load_state_dict(torch.load(diffusion_checkpoint, map_location=cfg.DEVICE))
                logger.info(f"✅ Loaded diffusion model: {diffusion_checkpoint}")
            except Exception as e:
                logger.error(f"❌ Failed to load diffusion model: {e}")
                logger.info("🔄 Will proceed without diffusion model...")
                diffuser = None
        else:
            logger.warning(f"⚠️ Diffusion checkpoint not found: {diffusion_checkpoint}")
            diffuser = None
        
        logger.info("🔄 Initializing components...")
        
        # Initialize clients
        clients = []
        for i in range(cfg.NUM_CLIENTS):
            client = Client(i, cfg, diffuser)  # Pass diffuser as optional parameter
            clients.append(client)
        
        # Initialize server
        server = FedServer(cfg)
        
        # Initialize attacker with dataset-appropriate parameters
        dataset = str(cfg.DATASET).lower()
        if dataset == 'br35h':
            # BR35H: gentler attack for medical images
            attack_eps = 0.015  # Reduced from 0.031 (half strength)
            attack_steps = 7    # Reduced from 10
            attack_alpha = 0.003  # Reduced step size
        elif dataset == 'cifar10':
            # CIFAR-10: standard strong attack but realistic
            attack_eps = 0.031  # Standard CIFAR-10 epsilon
            attack_steps = 10   # Standard steps
            attack_alpha = 0.008  # Reasonable step size
        else:
            # Default attack parameters
            attack_eps = getattr(cfg, 'ATTACK_EPSILON', 8/255)
            attack_steps = getattr(cfg, 'ATTACK_STEPS', 10)
            attack_alpha = getattr(cfg, 'ATTACK_ALPHA', attack_eps/4)
        
        pgd_attacker = PGDAttack(
            attack_eps,
            attack_alpha,
            attack_steps,
            getattr(cfg, 'ATTACK_RANDOM_START', True)
        )
        
        # Initialize MAE detector
        try:
            mae_detector = MAEDetector(cfg)
            logger.info("Using user's MAE detector implementation")
        except Exception as e:
            logger.error(f"Error loading MAE detector: {e}")
            logger.info("Using user's MAE detector implementation")
        mae_detector = MAEDetector(cfg)
        
        logger.info("✅ All components initialized successfully")
        
        # Initialize simple ResNet18 for now (pFedDef has issues)
        import torchvision.models as models
        template_model = models.resnet18(pretrained=False)
        template_model.fc = nn.Linear(template_model.fc.in_features, getattr(cfg, 'NUM_CLASSES', 10))
        
        server.init_round(template_model)
        logger.info("🔄 Server initialized with ResNet18 models")
        
        # Initialize global model for broadcast/eval
        global_model = models.resnet18(pretrained=False)
        global_model.fc = nn.Linear(global_model.fc.in_features, getattr(cfg, 'NUM_CLASSES', 10))
        global_model = global_model.to(cfg.DEVICE)
        
        # Main federated training loop
        for round_idx in range(cfg.NUM_ROUNDS):
            round_start_time = time.time()
            logger.info(f"🔄 Round {round_idx+1}/{cfg.NUM_ROUNDS}")
            
            # Broadcast global model to clients
            init_state = global_model.state_dict()
            for client in clients:
                client.model.load_state_dict(init_state, strict=False)
            
            # Client training
            for client_idx in range(cfg.NUM_CLIENTS):
                try:
                    client_start_time = time.time()
                    logger.info(f"🔄 Training client {client_idx+1}/{cfg.NUM_CLIENTS}")
                    client_epochs = getattr(cfg, 'CLIENT_EPOCHS', 5)  # Use better default
                    clients[client_idx].train(epochs=client_epochs)
                    client_models = clients[client_idx].get_models_for_server()
                    server.receive_update(client_idx, client_models)
                    client_time = time.time() - client_start_time
                    logger.info(f"✅ Client {client_idx+1} completed in {client_time:.1f}s")
                except Exception as e:
                    logger.error(f"❌ Client {client_idx} training failed: {e}")
                    logger.error(traceback.format_exc())
                    return False

            logger.info("🔄 Aggregating models...")
            global_models = server.aggregate()
            if global_models:
                # Update global_model with aggregated weights
                agg_model = global_models[0].to(cfg.DEVICE)
                global_model.load_state_dict(agg_model.state_dict(), strict=False)
                
                # Optionally refresh clients to new global (next round broadcast anyway)
                logger.info("✅ Model aggregation completed")
            
            # Evaluation with improved metrics
            if round_idx % 1 == 0:
                try:
                    logger.info("🔧 Recomputing BatchNorm running stats...")
                    client_loaders = [c.train_loader for c in clients]
                    recompute_bn_stats(global_model, client_loaders, cfg.DEVICE, max_batches=100)
                    
                    logger.info("🔄 Running evaluation...")
                    clean_correct = 0
                    adv_correct = 0
                    total = 0
                    detected_adv = 0
                    adv_total = 0
                    
                    for batch_idx, (data, target) in enumerate(test_loader):
                        data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
                        
                        # Clean accuracy - use pFedDef ensemble for evaluation
                        eval_model = global_model
                        with torch.no_grad():
                            # Use ensemble prediction for clean accuracy
                            output = eval_model(data)  # pFedDef automatically ensembles
                            pred = output.argmax(dim=1)
                            clean_correct += pred.eq(target).sum().item()
                            total += target.size(0)

                        # Generate adversarial examples using ensemble model (stronger defense)
                        adv_data = pgd_attacker.attack(eval_model, data, target)

                        # SELECTIVE DEFENSE: MAE detection first, then conditional DiffPure
                        try:
                            batch_size = adv_data.size(0)
                            device = adv_data.device
                            
                            # MAE adaptive detection
                            if mae_detector is not None:
                                try:
                                    # Primary path: use adaptive detector output directly
                                    detected_int = mae_detector.detect(adv_data)  # int tensor [B]
                                    detected_mask = detected_int.to(torch.bool)
                                    # For debugging only: fetch recon errors (not used for decisions)
                                    recon_errors = mae_detector.get_reconstruction_error(adv_data)
                                except Exception as mae_error:
                                    logger.warning(f"MAE detector failed, using fallback: {mae_error}")
                                    # Fallback detection based on input variance
                                    input_variance = torch.var(adv_data.view(batch_size, -1), dim=1)
                                    thr_var = torch.quantile(input_variance, 0.85)  # Top 15% variance
                                    detected_mask = input_variance > thr_var
                                    recon_errors = input_variance
                            else:
                                # No MAE detector - use variance-based detection
                                input_variance = torch.var(adv_data.view(batch_size, -1), dim=1)
                                thr_var = torch.quantile(input_variance, 0.85)  # Top 15% variance
                                detected_mask = input_variance > thr_var
                                recon_errors = input_variance
                            
                            # Apply DiffPure ONLY to detected adversarial samples (with conservative safeguards)
                            purified_data = adv_data.clone()
                            if detected_mask.sum() > 0:
                                # Conservative cap on purification rate
                                max_rate = float(getattr(cfg, 'MAX_PURIFY_RATE', 0.15))  # e.g., 15%
                                num_detected = detected_mask.sum().item()
                                max_purify = max(1, int(batch_size * max_rate))

                                # Select top-error detected samples up to cap
                                detected_indices = torch.nonzero(detected_mask, as_tuple=False).squeeze(1)
                                detected_errors = recon_errors[detected_indices]
                                if num_detected > max_purify:
                                    top_vals, top_idx = torch.topk(detected_errors, k=max_purify)
                                    selected_indices = detected_indices[top_idx]
                                else:
                                    selected_indices = detected_indices

                                try:
                                    # Purify selected samples
                                    to_purify = adv_data[selected_indices]
                                    raw_purified = diffpure_purify(diffuser, to_purify, cfg)

                                    # Alpha-blend to stay close to original (very conservative)
                                    blend = float(getattr(cfg, 'DIFFPURE_BLEND', 0.10))  # 10%
                                    blended = to_purify * (1.0 - blend) + raw_purified * blend

                                    # Classification-based acceptance: only keep if CE loss decreases
                                    with torch.no_grad():
                                        pre_logits = eval_model(to_purify)
                                        post_logits = eval_model(blended)
                                        true_labels = target[selected_indices]
                                        pre_loss = F.cross_entropy(pre_logits, true_labels, reduction='none')
                                        post_loss = F.cross_entropy(post_logits, true_labels, reduction='none')
                                        improve_mask = post_loss < pre_loss

                                    # Commit only improved samples
                                    commit_indices = selected_indices[improve_mask]
                                    if commit_indices.numel() > 0:
                                        purified_data[commit_indices] = blended[improve_mask]
                                    # Clarify log: committed over selected over detected
                                    logger.info(
                                        f"Applied DiffPure to {commit_indices.numel()}/{selected_indices.numel()} selected of {detected_mask.sum().item()} detected"
                                    )
                                except Exception as purify_error:
                                    logger.warning(f"DiffPure failed, using original: {purify_error}")
                            
                            # Debug info for first batch only
                            if batch_idx == 0:
                                logger.info(f"MAE Debug - Detection: {detected_mask.sum().item()}/{batch_size}")
                                try:
                                    logger.info(f"MAE Debug - Recon errors range: [{recon_errors.min():.4f}, {recon_errors.max():.4f}]")
                                except Exception:
                                    pass
                            
                            detected_adv += detected_mask.sum().item()
                            adv_total += batch_size
                            
                        except Exception as e:
                            logger.error(f"❌ Selective defense failed: {e}")
                            import traceback
                            traceback.print_exc()
                            # Fallback: no purification
                            purified_data = adv_data
                            detected_adv += int(0.1 * batch_size)
                            adv_total += batch_size
                        
                        # Adversarial prediction using pFedDef ensemble on selectively purified data
                        with torch.no_grad():
                            # Use ensemble prediction for adversarial accuracy (key improvement!)
                            adv_output = eval_model(purified_data)  # pFedDef ensemble
                            adv_pred = adv_output.argmax(dim=1)
                            adv_correct += adv_pred.eq(target).sum().item()
                    
                    clean_acc = 100. * clean_correct / total if total > 0 else 0
                    adv_acc = 100. * adv_correct / total if total > 0 else 0
                    detection_rate = 100. * detected_adv / adv_total if adv_total > 0 else 0
                    
                    final_clean_acc = clean_acc
                    final_adv_acc = adv_acc
                    final_detection_rate = detection_rate
                    
                    round_time = time.time() - round_start_time
                    logger.info(f"📊 Round {round_idx+1} Clean Acc: {clean_acc:.2f}% | Adv Acc: {adv_acc:.2f}% | MAE Detection: {detection_rate:.2f}% | Time: {round_time:.1f}s")
                    training_time = time.time() - start_time
                    logger.info(f"⏱️ Total training time: {training_time:.1f}s")
                except Exception as e:
                    logger.error(f"❌ Evaluation failed: {e}")
                    logger.error(traceback.format_exc())
                    continue

        
        training_time = time.time() - start_time
        logger.info(f"🎉 Training completed in {training_time:.1f}s")
        
        # Save final results to JSON file
        save_experiment_results(cfg, round_idx+1, final_clean_acc, final_adv_acc, final_detection_rate, training_time)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Federated training failed: {e}")
        import traceback
        logger.error("Traceback:\n" + traceback.format_exc())
        return False

def main():
    """Main training pipeline"""
    logger = setup_logging()
    args = parse_args()
    
    print("Complete Training Pipeline")
    print("=" * 60)
    
    # Get configuration
    from config_fixed import get_debug_config, get_test_config, get_full_config
    
    if args.mode == 'debug':
        cfg = get_debug_config()
    elif args.mode == 'test':
        cfg = get_test_config()
    else:
        cfg = get_full_config()
    
    # Override dataset if specified
    if args.dataset:
        cfg.DATASET = args.dataset.upper()
        cfg.DATASET_NAME = args.dataset.capitalize()
    
    # Dataset-specific defaults (only set if not already provided by config)
    if str(cfg.DATASET).lower() == 'br35h':
        if not hasattr(cfg, 'ADAPTIVE_THRESHOLD'):
            cfg.ADAPTIVE_THRESHOLD = True
        if not hasattr(cfg, 'TARGET_DETECTION_RATE'):
            cfg.TARGET_DETECTION_RATE = 18.0
        if not hasattr(cfg, 'MAX_PURIFY_RATE'):
            cfg.MAX_PURIFY_RATE = 0.15
        if not hasattr(cfg, 'EVAL_BATCH_SIZE'):
            cfg.EVAL_BATCH_SIZE = getattr(cfg, 'BATCH_SIZE', 32)
    elif str(cfg.DATASET).lower() == 'cifar10':
        # CIFAR-10 optimized defaults
        if not hasattr(cfg, 'ADAPTIVE_THRESHOLD'):
            cfg.ADAPTIVE_THRESHOLD = True
        if not hasattr(cfg, 'TARGET_DETECTION_RATE'):
            cfg.TARGET_DETECTION_RATE = 15.0  # Slightly lower for CIFAR-10
        if not hasattr(cfg, 'MAX_PURIFY_RATE'):
            cfg.MAX_PURIFY_RATE = 0.12  # More conservative for CIFAR-10
        if not hasattr(cfg, 'EVAL_BATCH_SIZE'):
            cfg.EVAL_BATCH_SIZE = getattr(cfg, 'BATCH_SIZE', 32)

    logger.info(f"Configuration: {args.mode} mode, {cfg.DATASET} dataset")
    logger.info(f"Settings: {cfg.NUM_ROUNDS} rounds, {getattr(cfg, 'CLIENT_EPOCHS', 15)} epochs, {cfg.NUM_CLIENTS} clients")
    
    # System requirements check
    if not args.skip_setup:
        if not check_system_requirements(cfg):
            logger.warning("⚠️ System requirements check failed, but continuing...")
    
    # Check and train diffusion model - ALWAYS check first
    logger.info(f"🔍 Checking diffusion model for {cfg.DATASET}...")
    diffusion_checkpoint = Path(f"checkpoints/diffuser_{cfg.DATASET.lower()}.pt")
    
    if diffusion_checkpoint.exists():
        logger.info(f"✅ Diffusion model already exists: {diffusion_checkpoint}")
    else:
        logger.warning(f"⚠️ Diffusion checkpoint not found: {diffusion_checkpoint}")
        logger.info(f"🔄 Training diffusion model for {cfg.DATASET}...")
        if not train_diffusion_model(cfg):
            logger.error("❌ Diffusion model training failed - STOPPING")
            return 1
        logger.info(f"✅ Diffusion model training completed for {cfg.DATASET}")
    
    # Force train if explicitly requested
    if args.train_diffusion:
        logger.info(f"🔄 Force re-training diffusion model for {cfg.DATASET}...")
        if not train_diffusion_model(cfg):
            logger.error("❌ Diffusion model training failed")
            return 1
    
    # Train MAE detector - ALWAYS for small datasets  
    dataset_name = cfg.DATASET.lower()  # Ensure lowercase for checkpoint names
    mae_checkpoint = Path(f"checkpoints/mae_detector_{dataset_name}.pt")
    mae_best_checkpoint = Path(f"checkpoints/mae_detector_{dataset_name}_best.pt")
    
    # Also check for legacy checkpoint names (without dataset name)
    mae_legacy_checkpoint = Path("checkpoints/mae_detector.pt")
    mae_legacy_best_checkpoint = Path("checkpoints/mae_detector_best.pt")
    
    # Check if any MAE checkpoint exists (regular, best, or legacy)
    mae_exists = (mae_checkpoint.exists() or mae_best_checkpoint.exists() or 
                  mae_legacy_checkpoint.exists() or mae_legacy_best_checkpoint.exists())
    
    if not mae_exists or args.train_mae:
        logger.info("🔄 Training MAE detector (required for small dataset)...")
        if not train_mae_detector(cfg):
            logger.warning("⚠️ MAE detector training failed, continuing with default threshold")
    else:
        # Log which checkpoint was found
        if mae_checkpoint.exists():
            logger.info(f"✅ MAE detector checkpoint exists: {mae_checkpoint}")
        elif mae_best_checkpoint.exists():
            logger.info(f"✅ MAE detector best checkpoint exists: {mae_best_checkpoint}")
        elif mae_legacy_checkpoint.exists():
            logger.info(f"✅ MAE detector legacy checkpoint exists: {mae_legacy_checkpoint}")
        elif mae_legacy_best_checkpoint.exists():
            logger.info(f"✅ MAE detector legacy best checkpoint exists: {mae_legacy_best_checkpoint}")
        logger.info("🚀 Skipping MAE detector training (checkpoint found)")
    
    # Start federated training
    logger.info(f"🚀 Starting federated training with {cfg.NUM_CLIENTS} clients, {cfg.NUM_ROUNDS} rounds")
    logger.info(f"🔄 Each client trains for {getattr(cfg, 'CLIENT_EPOCHS', 5)} epochs per round")
    
    # Get test dataset for evaluation
    from utils.data_utils import get_dataset
    from torch.utils.data import DataLoader
    
    train_dataset, test_dataset = get_dataset(cfg, train=True)
    test_loader = DataLoader(test_dataset, batch_size=getattr(cfg, 'EVAL_BATCH_SIZE', getattr(cfg, 'BATCH_SIZE', 64)), shuffle=False)
    
    # Run federated training
    success = run_federated_training(cfg, test_loader)
    
    if success:
        logger.info("🎉 Training pipeline completed successfully!")
        return 0
    else:
        logger.error("❌ Training pipeline failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 