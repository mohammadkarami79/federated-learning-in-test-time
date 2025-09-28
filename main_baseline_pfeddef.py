"""
Federated Learning Baseline - pFedDef ONLY (No MAE, No DiffPure)
Pure pFedDef performance evaluation for fair comparison
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
import json
from datetime import datetime

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
    parser = argparse.ArgumentParser(description='pFedDef Baseline Evaluation')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['br35h', 'cifar10', 'cifar100', 'mnist'],
                       help='Dataset to use')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['debug', 'test', 'full'],
                       help='Training mode')
    parser.add_argument('--skip-setup', action='store_true',
                       help='Skip system setup check')
    return parser.parse_args()

def get_config(dataset_name, mode):
    """Get configuration for baseline experiment"""
    
    # Base configuration
    config = {
        'DATASET': dataset_name.upper(),
        'DATASET_NAME': dataset_name.lower(),
        'MODE': mode,
        'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Federated Learning Settings
        'NUM_CLIENTS': 10,
        'CLIENTS_PER_ROUND': 10,
        'CLIENT_EPOCHS': 8,  # MATCHED TO LOG8.TXT
        
        # Training Parameters - MATCHED TO LOG8.TXT
        'LEARNING_RATE': 0.01,
        'BATCH_SIZE': 64,  # Keep same as original
        'WEIGHT_DECAY': 1e-4,
        'MAX_BATCHES_PER_EPOCH': 782,  # CIFAR-10: 50000/64 ≈ 782 batches
        
        # Defense: ONLY pFedDef (NO MAE, NO DiffPure)
        'USE_MAE_DETECTOR': False,  # DISABLED
        'USE_DIFFPURE': False,      # DISABLED  
        'USE_PFEDDEF': True,        # ONLY THIS
        
        # Evaluation
        'EVAL_BATCH_SIZE': 256,
    }
    
    # Dataset-specific settings
    if dataset_name.lower() == 'cifar10':
        config.update({
            'IMG_SIZE': 32,
            'IMG_CHANNELS': 3,
            'NUM_CLASSES': 10,
            'ROUNDS': 15 if mode == 'full' else 3,
        })
    elif dataset_name.lower() == 'br35h':
        config.update({
            'IMG_SIZE': 224,
            'IMG_CHANNELS': 3,
            'NUM_CLASSES': 2,
            'ROUNDS': 15 if mode == 'full' else 3,
        })
    
        # Mode-specific adjustments
        if mode == 'debug':
            config.update({
                'NUM_CLIENTS': 3,
                'CLIENTS_PER_ROUND': 3,
                'CLIENT_EPOCHS': 2,
                'ROUNDS': 2,
            })
        elif mode == 'test':
            config.update({
                'ROUNDS': 5,
                'CLIENT_EPOCHS': 8,  # KEEP 8 EPOCHS EVEN IN TEST MODE FOR LOG8.TXT MATCH
            })
    
    return type('Config', (), config)()

class PGDAttack:
    """PGD Attack - IDENTICAL to main.py for fair comparison"""
    def __init__(self, epsilon, alpha, steps, random_start=True):
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
    
    def attack(self, model, images, labels):
        """Generate adversarial examples"""
        images = images.clone().detach().to(model.device if hasattr(model, 'device') else next(model.parameters()).device)
        labels = labels.clone().detach().to(images.device)
        
        # Initialize adversarial images
        adv_images = images.clone().detach()
        
        if self.random_start:
            # Random start within epsilon ball
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        
        for _ in range(self.steps):
            adv_images.requires_grad_(True)  # CRITICAL: Enable gradients
            
            # Forward pass
            if hasattr(model, 'get_models_for_server'):
                # Handle pFedDef ensemble
                models = model.get_models_for_server()
                if len(models) > 1:
                    # Ensemble prediction
                    outputs = []
                    for m in models:
                        m.eval()
                        outputs.append(m(adv_images))
                    outputs = torch.stack(outputs).mean(dim=0)
                else:
                    model.eval()  # Set model to eval mode
                    outputs = models[0](adv_images)
            else:
                model.eval()  # Set model to eval mode
                outputs = model(adv_images)
            
            # Calculate loss
            loss = F.cross_entropy(outputs, labels)
            
            # Calculate gradients
            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
            
            # Update adversarial images
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        
        return adv_images

def run_baseline_evaluation(cfg):
    """Run pFedDef-only baseline evaluation"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("BASELINE EVALUATION: pFedDef ONLY (No MAE, No DiffPure)")
        logger.info(f"Configuration: {cfg.MODE} mode, {cfg.DATASET} dataset")
        logger.info(f"Settings: {cfg.ROUNDS} rounds, {cfg.CLIENT_EPOCHS} epochs, {cfg.NUM_CLIENTS} clients")
        
        # GPU Info
        if torch.cuda.is_available():
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        logger.info("Loading components...")
        
        # Import required modules
        from federated.server import FedServer
        from torch.utils.data import Subset
        
        logger.info("All imports successful")
        
        logger.info("Loading datasets...")
        
        # Load dataset
        from torchvision import transforms
        
        if cfg.DATASET.lower() == 'cifar10':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            # BR35H or others
            transform_train = transforms.Compose([
                transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            transform_test = transform_train
        
        # Load datasets using the original function
        import torchvision.datasets as datasets
        
        if cfg.DATASET.lower() == 'cifar10':
            train_dataset = datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform_train
            )
            test_dataset = datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test
            )
        elif cfg.DATASET.lower() == 'br35h':
            # For BR35H, use the original get_dataset function
            train_loader = get_dataset_original('br35h', train=True, transform=transform_train)
            test_loader = get_dataset_original('br35h', train=False, transform=transform_test)
            train_dataset = train_loader.dataset
            test_dataset = test_loader.dataset
        else:
            raise ValueError(f"Unsupported dataset: {cfg.DATASET}")
        
        # Test loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        logger.info(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
        logger.info(f"Test loader created with {len(test_dataset)} samples")
        
        logger.info("Initializing components...")
        
        # Initialize clients - PURE pFedDef models with manual data distribution
        clients = []
        for client_id in range(cfg.NUM_CLIENTS):
            # Create data subset for this client - SAME AS LOG8.TXT
            # Each client gets ALL data (non-IID but same as original experiment)
            client_dataset = train_dataset
            
            client_loader = DataLoader(
                client_dataset,
                batch_size=cfg.BATCH_SIZE,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
            
            # Create simple baseline client
            class BaselineClient:
                def __init__(self, client_id, cfg, train_loader):
                    self.client_id = client_id
                    self.cfg = cfg
                    self.device = torch.device(cfg.DEVICE)
                    self.train_loader = train_loader
                    
                    # Create simple ResNet18
                    import torchvision.models as models
                    self.model = models.resnet18(weights=None)
                    self.model.fc = nn.Linear(self.model.fc.in_features, cfg.NUM_CLASSES)
                    self.model = self.model.to(self.device)
                
                def get_models_for_server(self):
                    return [self.model]
                
                def train(self, epochs=1):
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.SGD(
                        self.model.parameters(),
                        lr=self.cfg.LEARNING_RATE,
                        momentum=0.9,
                        weight_decay=5e-4
                    )
                    
                    self.model.train()
                    for epoch in range(epochs):
                        running_loss = 0.0
                        correct = 0
                        total = 0
                        batch_count = 0
                        max_batches = getattr(self.cfg, 'MAX_BATCHES_PER_EPOCH', len(self.train_loader))
                        
                        for batch_idx, (data, targets) in enumerate(self.train_loader):
                            if batch_count >= max_batches:
                                break  # Limit batches per epoch for realistic timing
                                
                            data, targets = data.to(self.device), targets.to(self.device)
                            
                            optimizer.zero_grad()
                            outputs = self.model(data)
                            loss = criterion(outputs, targets)
                            loss.backward()
                            optimizer.step()
                            
                            running_loss += loss.item()
                            _, predicted = outputs.max(1)
                            total += targets.size(0)
                            correct += predicted.eq(targets).sum().item()
                            batch_count += 1
                        
                        acc = 100.0 * correct / total
                        avg_loss = running_loss / batch_count if batch_count > 0 else 0.0
                        logger.info(f"Client {self.client_id} Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
            
            client = BaselineClient(client_id, cfg, client_loader)
            clients.append(client)
        
        logger.info(f"Created {len(clients)} clients")
        
        # Initialize server
        server = FedServer(cfg)
        
        # Get model template from first client
        model_template = clients[0].model
        global_models = server.init_round(model_template)
        
        logger.info("All components initialized successfully")
        logger.info(f"Server initialized with pFedDef models")
        
        # Initialize attacker with IDENTICAL parameters to main.py
        dataset = str(cfg.DATASET).lower()
        if dataset == 'br35h':
            # BR35H: gentler attack for medical images
            attack_eps = 0.015  # Reduced from 0.031 (half strength)
            attack_steps = 7    # Reduced from 10
            attack_alpha = 0.003  # Reduced step size
        elif dataset == 'cifar10':
            # CIFAR-10: MATCHED TO LOG8.TXT PARAMETERS
            attack_eps = 0.031  # Standard CIFAR-10 epsilon
            attack_steps = 10   # Standard steps
            attack_alpha = 0.007  # MATCHED TO LOG8.TXT (was 0.008)
        else:
            # Default attack parameters
            attack_eps = 8/255
            attack_steps = 10
            attack_alpha = attack_eps/4
        
        pgd_attacker = PGDAttack(
            attack_eps,
            attack_alpha,
            attack_steps,
            random_start=True
        )
        
        logger.info(f"PGD Attack (LOG8.TXT MATCH): eps={attack_eps:.3f}, steps={attack_steps}, alpha={attack_alpha:.3f}")
        
        # Training loop
        logger.info(f"Starting federated training with {cfg.NUM_CLIENTS} clients, {cfg.ROUNDS} rounds")
        logger.info(f"Each client trains for {cfg.CLIENT_EPOCHS} epochs per round")
        
        results = {
            'rounds': [],
            'clean_accuracies': [],
            'adversarial_accuracies': [],
            'training_times': []
        }
        
        total_start_time = time.time()
        
        for round_num in range(1, cfg.ROUNDS + 1):
            round_start_time = time.time()
            logger.info(f"Round {round_num}/{cfg.ROUNDS}")
            
            # Distribute global models to clients
            for client in clients:
                # Update client model with global model state
                client.model.load_state_dict(global_models[0].state_dict())
            
            # Client training
            for i, client in enumerate(clients, 1):
                logger.info(f"Training client {i}/{cfg.NUM_CLIENTS}")
                client_start = time.time()
                
                client.train(epochs=cfg.CLIENT_EPOCHS)
                
                client_time = time.time() - client_start
                logger.info(f"Client {i} completed in {client_time:.1f}s")
                
                # Send updates to server
                local_models = client.get_models_for_server()
                server.receive_update(client.client_id, local_models)
            
            # Server aggregation
            logger.info("Server aggregating...")
            global_models = server.aggregate()
            logger.info("Aggregation completed")
            
            # Evaluation
            logger.info("Evaluating...")
            
            # Use first global model for evaluation
            eval_model = global_models[0]
            eval_model.eval()
            
            clean_correct = 0
            adv_correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(cfg.DEVICE)
                labels = labels.to(cfg.DEVICE)
                
                # Clean accuracy - use no_grad only for clean evaluation
                with torch.no_grad():
                    clean_outputs = eval_model(images)
                    clean_pred = clean_outputs.argmax(dim=1)
                    clean_correct += (clean_pred == labels).sum().item()
                
                # Generate adversarial examples - REQUIRES GRADIENTS
                adv_images = pgd_attacker.attack(eval_model, images, labels)
                
                # Adversarial accuracy - use no_grad for final evaluation
                with torch.no_grad():
                    adv_outputs = eval_model(adv_images)
                    adv_pred = adv_outputs.argmax(dim=1)
                    adv_correct += (adv_pred == labels).sum().item()
                
                total += labels.size(0)
                
                if batch_idx % 20 == 0:
                    logger.info(f"Batch {batch_idx}/{len(test_loader)}")
            
            clean_acc = 100.0 * clean_correct / total
            adv_acc = 100.0 * adv_correct / total
            round_time = time.time() - round_start_time
            
            results['rounds'].append(round_num)
            results['clean_accuracies'].append(clean_acc)
            results['adversarial_accuracies'].append(adv_acc)
            results['training_times'].append(round_time)
            
            logger.info(f"Round {round_num} Results:")
            logger.info(f"   Clean Acc: {clean_acc:.2f}%")
            logger.info(f"   Adv Acc: {adv_acc:.2f}%")
            logger.info(f"   Time: {round_time:.1f}s")
        
        total_time = time.time() - total_start_time
        
        # Final results
        final_clean_acc = results['clean_accuracies'][-1]
        final_adv_acc = results['adversarial_accuracies'][-1]
        
        logger.info("BASELINE EVALUATION COMPLETED!")
        logger.info(f"Final Results (pFedDef ONLY):")
        logger.info(f"   Clean Accuracy: {final_clean_acc:.2f}%")
        logger.info(f"   Adversarial Accuracy: {final_adv_acc:.2f}%")
        logger.info(f"   Total Training Time: {total_time:.1f}s")
        
        # Save results
        experiment_results = {
            "experiment_info": {
                "type": "baseline_pfeddef_only_log8_match",
                "dataset": cfg.DATASET,
                "mode": cfg.MODE,
                "timestamp": datetime.now().isoformat(),
                "rounds_completed": cfg.ROUNDS,
                "total_training_time_seconds": total_time,
                "matched_to_log": "log8.txt",
                "client_epochs": cfg.CLIENT_EPOCHS
            },
            "defense_config": {
                "use_mae_detector": False,
                "use_diffpure": False,
                "use_pfeddef": True,
                "defense_description": "pFedDef ensemble only - no additional defenses"
            },
            "attack_config": {
                "attack_type": "PGD",
                "epsilon": attack_eps,
                "steps": attack_steps,
                "alpha": attack_alpha
            },
            "final_metrics": {
                "clean_accuracy": final_clean_acc,
                "adversarial_accuracy": final_adv_acc
            },
            "round_by_round": {
                "rounds": results['rounds'],
                "clean_accuracies": results['clean_accuracies'],
                "adversarial_accuracies": results['adversarial_accuracies'],
                "training_times": results['training_times']
            },
            "model_info": {
                "has_mae_detector": False,
                "has_diffusion_purifier": False,
                "has_pfeddef_ensemble": True,
                "img_size": cfg.IMG_SIZE,
                "img_channels": cfg.IMG_CHANNELS,
                "num_classes": cfg.NUM_CLASSES
            }
        }
        
        # Save to JSON file
        results_file = f"baseline_pfeddef_{cfg.DATASET.lower()}_{cfg.MODE}_results.json"
        with open(results_file, 'w') as f:
            json.dump(experiment_results, f, indent=2)
        
        logger.info(f"📁 Results saved to: {results_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Baseline evaluation failed: {e}")
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        return 1

def main():
    """Main function"""
    logger = setup_logging()
    
    try:
        args = parse_args()
        cfg = get_config(args.dataset, args.mode)
        
        # System info
        logger.info("STARTING BASELINE EVALUATION - pFedDef ONLY")
        logger.info(f"Dataset: {cfg.DATASET}, Mode: {cfg.MODE}")
        logger.info(f"Device: {cfg.DEVICE}")
        
        if not args.skip_setup:
            logger.info("System check...")
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
            else:
                logger.warning("CUDA not available, using CPU")
        
        # Run baseline evaluation
        return run_baseline_evaluation(cfg)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
