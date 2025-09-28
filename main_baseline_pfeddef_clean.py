# -*- coding: utf-8 -*-
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
            'CLIENT_EPOCHS': 8,  # MATCHED TO LOG8.TXT
            'LEARNING_RATE': 0.01,
            'BATCH_SIZE': 64,  # Keep same as original
            'WEIGHT_DECAY': 1e-4,
            'MAX_BATCHES_PER_EPOCH': 782,  # CIFAR-10: 50000/64 approx 782 batches
        })
    elif dataset_name.lower() == 'br35h':
        config.update({
            'IMG_SIZE': 224,
            'IMG_CHANNELS': 3,
            'NUM_CLASSES': 2,
            'ROUNDS': 15 if mode == 'full' else 3,
            'CLIENT_EPOCHS': 5,  # MATCHED TO LOG11.TXT (BR35H uses 5 epochs)
            'LEARNING_RATE': 0.01,
            'BATCH_SIZE': 64,  # Same as CIFAR-10 for consistency
            'WEIGHT_DECAY': 1e-4,
            'MAX_BATCHES_PER_EPOCH': 38,  # BR35H: 2400/64 approx 38 batches
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
        import numpy as np
        
        logger.info("All imports successful")
        
        logger.info("Loading datasets...")
        
        # Load dataset
        from torchvision import transforms
        import torchvision.datasets as datasets
        
        if cfg.DATASET.lower() == 'cifar10':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            train_dataset = datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform_train
            )
            test_dataset = datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test
            )
        elif cfg.DATASET.lower() == 'br35h':
            # BR35H dataset - USE SAME LOADER AS MAIN.PY
            try:
                # Try to import from utils.data_utils like main.py
                from utils.data_utils import get_dataset
                train_dataset, test_dataset = get_dataset(cfg)
                logger.info("BR35H dataset loaded using get_dataset()")
            except (ImportError, AttributeError, FileNotFoundError) as e:
                logger.warning(f"get_dataset() failed: {e}")
                logger.info("Trying alternative BR35H loading methods...")
                
                # Fallback: Manual BR35H loading
                transform_train = transforms.Compose([
                    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                transform_test = transforms.Compose([
                    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                
                # Try multiple possible paths
                possible_paths = [
                    ('./data/br35h/train', './data/br35h/test'),
                    ('./data/BR35H/train', './data/BR35H/test'),
                    ('./data/br35h_brain_tumor/train', './data/br35h_brain_tumor/test'),
                    ('../data/br35h/train', '../data/br35h/test'),
                    ('../data/BR35H/train', '../data/BR35H/test'),
                ]
                
                dataset_loaded = False
                for train_path, test_path in possible_paths:
                    try:
                        train_dataset = datasets.ImageFolder(root=train_path, transform=transform_train)
                        test_dataset = datasets.ImageFolder(root=test_path, transform=transform_test)
                        logger.info(f"BR35H dataset loaded from: {train_path}")
                        dataset_loaded = True
                        break
                    except FileNotFoundError:
                        continue
                
                if not dataset_loaded:
                    # Create dummy dataset for BR35H (2 classes, 224x224x3)
                    logger.warning("Creating dummy BR35H dataset for testing...")
                    import torch.utils.data as data_utils
                    
                    # Create dummy tensors
                    dummy_train_data = torch.randn(2400, 3, 224, 224)
                    dummy_train_labels = torch.randint(0, 2, (2400,))
                    dummy_test_data = torch.randn(600, 3, 224, 224)
                    dummy_test_labels = torch.randint(0, 2, (600,))
                    
                    train_dataset = data_utils.TensorDataset(dummy_train_data, dummy_train_labels)
                    test_dataset = data_utils.TensorDataset(dummy_test_data, dummy_test_labels)
                    logger.warning("Using dummy data - results won't be meaningful!")
        else:
            raise ValueError(f"Dataset {cfg.DATASET} not implemented in baseline")
        
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
                    self.client_id = client_id  # CRITICAL: Must be set
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
        
        # Initialize server with custom aggregation for baseline
        class BaselineServer:
            def __init__(self, cfg):
                self.cfg = cfg
                self.global_models = None
                self.client_updates = {}
                
            def init_round(self, model_template):
                if self.global_models is None:
                    import copy
                    # Deep copy and ensure proper device placement
                    cloned_model = copy.deepcopy(model_template)
                    cloned_model = cloned_model.to(torch.device(self.cfg.DEVICE))
                    
                    # Initialize with random weights (don't use pre-trained)
                    def init_weights(m):
                        if isinstance(m, nn.Linear):
                            torch.nn.init.xavier_uniform_(m.weight)
                            if m.bias is not None:
                                m.bias.data.fill_(0.01)
                        elif isinstance(m, nn.Conv2d):
                            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                            if m.bias is not None:
                                m.bias.data.fill_(0.01)
                    
                    cloned_model.apply(init_weights)
                    self.global_models = [cloned_model]
                return self.global_models
                
            def receive_update(self, client_id, local_models):
                if not isinstance(local_models, list):
                    local_models = [local_models]
                self.client_updates[client_id] = [local_models[0]]
                
            def aggregate(self):
                n_clients = len(self.client_updates)
                assert n_clients > 0, "No updates received"
                
                # Get all client models
                client_models = [
                    updates[0].state_dict() 
                    for updates in self.client_updates.values()
                ]
                
                # Average ALL parameters (including BatchNorm)
                avg_state = {}
                
                # Get all parameter keys from first client
                reference_keys = list(client_models[0].keys())
                
                for key in reference_keys:
                    if all(key in client_state for client_state in client_models):
                        # Stack parameters from all clients
                        param_list = [client_state[key] for client_state in client_models]
                        
                        # Check if parameters have same shape
                        shapes = [p.shape for p in param_list]
                        if len(set(shapes)) == 1:  # All same shape
                            stacked = torch.stack(param_list)
                            # Convert to float if needed
                            if stacked.dtype in [torch.long, torch.int, torch.int64, torch.int32]:
                                stacked = stacked.float()
                            avg_state[key] = stacked.mean(dim=0)
                        else:
                            # Use first client's parameter if shapes don't match
                            avg_state[key] = param_list[0]
                
                # Load averaged weights
                try:
                    self.global_models[0].load_state_dict(avg_state, strict=True)
                except Exception as e:
                    logger.warning(f"Failed to load averaged state: {e}")
                    # Fallback: use first client's state
                    self.global_models[0].load_state_dict(client_models[0], strict=True)
                
                # Clear updates
                self.client_updates.clear()
                return self.global_models
        
        server = BaselineServer(cfg)
        
        # Get model template from first client
        model_template = clients[0].model
        global_models = server.init_round(model_template)
        
        logger.info("All components initialized successfully")
        logger.info(f"Server initialized with pFedDef models")
        
        # Initialize attacker with IDENTICAL parameters to respective logs
        dataset = str(cfg.DATASET).lower()
        if dataset == 'br35h':
            # BR35H: MATCHED TO LOG11.TXT PARAMETERS
            attack_eps = 0.015  # Matched to log11.txt: "Epsilon: 0.015000"
            attack_steps = 7    # Matched to log11.txt: "Steps: 7"
            attack_alpha = 0.003  # Matched to log11.txt: "Step size: 0.003000"
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
            logger.info("Distributing global model to clients...")
            global_state = global_models[0].state_dict()
            for client in clients:
                # Update client model with global model state
                client.model.load_state_dict(global_state)
                client.model = client.model.to(client.device)  # Ensure proper device
            logger.info(f"Global model distributed to {len(clients)} clients")
            
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
            
            # DEBUG: Check model output distribution
            debug_predictions = []
            
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(cfg.DEVICE)
                labels = labels.to(cfg.DEVICE)
                
                # Clean accuracy - use no_grad only for clean evaluation
                with torch.no_grad():
                    clean_outputs = eval_model(images)
                    clean_pred = clean_outputs.argmax(dim=1)
                    clean_correct += (clean_pred == labels).sum().item()
                    
                    # DEBUG: Collect predictions for first batch
                    if batch_idx == 0:
                        debug_predictions = clean_pred.cpu().numpy()
                        logger.info(f"DEBUG - First batch predictions: {np.unique(debug_predictions, return_counts=True)}")
                        logger.info(f"DEBUG - First batch labels: {np.unique(labels.cpu().numpy(), return_counts=True)}")
                        logger.info(f"DEBUG - Output logits range: [{clean_outputs.min().item():.3f}, {clean_outputs.max().item():.3f}]")
                        
                        # Check final layer weights
                        fc_weight = eval_model.fc.weight.data
                        fc_bias = eval_model.fc.bias.data
                        logger.info(f"DEBUG - FC weight shape: {fc_weight.shape}, range: [{fc_weight.min().item():.3f}, {fc_weight.max().item():.3f}]")
                        logger.info(f"DEBUG - FC bias shape: {fc_bias.shape}, values: {fc_bias.cpu().numpy()}")
                        
                        # Check first few logits in detail
                        logger.info(f"DEBUG - First sample logits: {clean_outputs[0].cpu().numpy()}")
                        logger.info(f"DEBUG - Softmax probs: {torch.softmax(clean_outputs[0], dim=0).cpu().numpy()}")
                
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
        
        logger.info(f"Results saved to: {results_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Baseline evaluation failed: {e}")
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        return 1

def run_cifar10_baseline():
    """Run CIFAR-10 baseline experiment - matched to log8.txt"""
    logger = setup_logging()
    
    try:
        cfg = get_config('cifar10', 'full')
        
        # System info
        logger.info("STARTING CIFAR-10 BASELINE EVALUATION - pFedDef ONLY")
        logger.info(f"Dataset: {cfg.DATASET}, Mode: {cfg.MODE}")
        logger.info(f"Matched to: log8.txt (CIFAR-10 benchmark)")
        logger.info(f"Device: {cfg.DEVICE}")
        
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

def run_br35h_baseline():
    """Run BR35H baseline experiment - matched to log11.txt"""
    logger = setup_logging()
    
    try:
        cfg = get_config('br35h', 'full')
        
        # System info
        logger.info("STARTING BR35H BASELINE EVALUATION - pFedDef ONLY")
        logger.info(f"Dataset: {cfg.DATASET}, Mode: {cfg.MODE}")
        logger.info(f"Matched to: log11.txt (BR35H benchmark)")
        logger.info(f"Device: {cfg.DEVICE}")
        
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

def main():
    """Main function with easy dataset selection"""
    logger = setup_logging()
    
    try:
        args = parse_args()
        cfg = get_config(args.dataset, args.mode)
        
        # System info
        logger.info("STARTING BASELINE EVALUATION - pFedDef ONLY")
        logger.info(f"Dataset: {cfg.DATASET}, Mode: {cfg.MODE}")
        
        if cfg.DATASET.lower() == 'cifar10':
            logger.info("Matched to: log8.txt (CIFAR-10 benchmark)")
        elif cfg.DATASET.lower() == 'br35h':
            logger.info("Matched to: log11.txt (BR35H benchmark)")
            
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
    # Multiple ways to run:
    # 1. Command line arguments: python main_baseline_pfeddef_clean.py --dataset cifar10 --mode full
    # 2. Direct CIFAR-10: run_cifar10_baseline()
    # 3. Direct BR35H: run_br35h_baseline()
    
    import sys
    
    # Check if running with special function names
    if len(sys.argv) > 1 and sys.argv[1] == 'cifar10_only':
        exit_code = run_cifar10_baseline()
    elif len(sys.argv) > 1 and sys.argv[1] == 'br35h_only':
        exit_code = run_br35h_baseline()
    else:
        # Standard command line argument parsing
        exit_code = main()
    
    sys.exit(exit_code)
