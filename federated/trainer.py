"""
Federated training orchestration
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import time
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional
import logging

from config import Config, DEVICE
from .client import Client
from .server import Server
from metrics.logger import MetricsLogger
from attacks.pgd import PGDAttack
from diffusion.diffuser import UNet
from utils.model_manager import ModelManager
from attacks.internal_pgd import internal_attack

class FederatedTrainer:
    def __init__(self, model_template):
        """
        Initialize federated trainer
        
        Args:
            model_template: Template model to copy for clients
        """
        self.model_template = model_template
        
        # Load and split dataset
        self.train_dataset, self.test_dataset = self.load_cifar10()
        
        # Initialize clients
        self.clients = [
            Client(
                client_id=i,
                model=self.create_client_model(),
                train_dataset=self.train_dataset,
                test_dataset=self.test_dataset
            )
            for i in range(CONFIG['N_CLIENTS'])
        ]
        
        # Initialize server
        self.server = Server(self.model_template)
        
        # Initialize logger
        self.logger = MetricsLogger(log_dir=Path('results/metrics'))
    
    def load_cifar10(self):
        """Load and preprocess CIFAR-10 dataset"""
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                )
            ])
        )
        
        return train_dataset, test_dataset
    
    def create_client_model(self):
        """Create a new model instance for a client"""
        return type(self.model_template)(**self.model_template.get_params())
    
    def train(self):
        """Run federated training process"""
        print(f"Starting federated training with {CONFIG['N_CLIENTS']} clients")
        print(f"Using device: {DEVICE}")
        
        for round_idx in range(CONFIG['GLOBAL_ROUNDS']):
            print(f"\nRound {round_idx + 1}/{CONFIG['GLOBAL_ROUNDS']}")
            round_start = time.time()
            
            # Train round
            metrics = self.server.train_round(self.clients, round_idx)
            
            # Print round summary
            print(f"Round {round_idx + 1} Summary:")
            print(f"Clean Accuracy: {metrics['clean_acc']:.2f}%")
            print(f"Adversarial Accuracy: {metrics['adv_acc']:.2f}%")
            print(f"Grey-box Attack Success: {metrics['greybox_sr']:.2f}%")
            print(f"Round Time: {metrics['latency']:.2f}s")
            print(f"Memory Usage: {metrics['memory']:.2f}MB")
        
        # Get final summary
        summary = self.logger.get_summary()
        print("\nTraining Complete!")
        print("Final Metrics:")
        print(f"Clean Accuracy: {summary['clean_acc_mean']:.2f}%")
        print(f"Adversarial Accuracy: {summary['adv_acc_mean']:.2f}%")
        print(f"Grey-box Attack Success: {summary['greybox_sr_mean']:.2f}%")
        
        # Generate plots
        self.logger.plot_metrics()
        
        return summary 

def train_diffpure(cfg: Config) -> nn.Module:
    """Pre-train DiffPure model on CIFAR-10."""
    logging.info("Pre-training DiffPure model on CIFAR-10...")
    
    # Setup data
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    trainloader = DataLoader(
        trainset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    # Initialize model and optimizer
    diffuser = UNet(in_channels=3, hidden_channels=64).to(DEVICE)
    optimizer = torch.optim.Adam(diffuser.parameters(), lr=1e-4)
    
    # Training loop
    diffuser.train()
    for epoch in range(10):  # Quick training for debug preset
        epoch_loss = 0
        for batch_idx, (data, _) in enumerate(trainloader):
            data = data.to(DEVICE)
            batch_size = data.shape[0]
            
            # Sample random time steps
            t = torch.rand(batch_size, device=DEVICE)
            
            # Add noise
            noise = torch.randn_like(data) * cfg.DIFFUSER_SIGMA
            noisy_data = data + noise
            
            # Predict noise
            pred_noise = diffuser(noisy_data, t)
            loss = torch.nn.functional.mse_loss(pred_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logging.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')
                
        logging.info(f'Epoch {epoch} average loss: {epoch_loss/len(trainloader):.6f}')
    
    # Save model
    model_manager = ModelManager()
    save_path = model_manager.get_latest_model('diffusion', 'cifar10')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(diffuser.state_dict(), save_path)
    
    return diffuser

def is_diffuser_trained(cfg: Config) -> bool:
    """Check if DiffPure model is trained."""
    model_manager = ModelManager()
    model_path = model_manager.get_latest_model('diffusion', 'cifar10')
    return model_path.exists()

def run_federated(cfg: Config):
    """Run federated training with combined DiffPure + pFedDef defense."""
    
    # First ensure DiffPure model is trained
    if not is_diffuser_trained(cfg):
        diffuser = train_diffpure(cfg)
    else:
        logging.info("Using pre-trained DiffPure model")
        diffuser = UNet(in_channels=3, hidden_channels=64).to(DEVICE)
        model_manager = ModelManager()
        model_path = model_manager.get_latest_model('diffusion', 'cifar10')
        diffuser.load_state_dict(torch.load(model_path))
    
    # Initialize components
    server = Server(cfg)
    clients = [Client(i, cfg, diffuser) for i in range(cfg.N_CLIENTS)]
    logger = MetricsLogger(cfg.output_dir)
    
    # Training loop
    for round_idx in range(cfg.N_ROUNDS):
        logger.start_round()
        
        # Local training
        for client in clients:
            client.train(epochs=cfg.LOCAL_EPOCHS)
            
        # Server aggregation
        client_models = [client.model for client in clients]
        server.receive_update(0, client_models)  # Use client_id=0 since we're aggregating all models
        global_models = server.aggregate()
        
        # Distribute updated model
        for client in clients:
            client.model.load_state_dict(global_models[0].state_dict())  # Use first model as global
            
        # Evaluation
        metrics = evaluate_round(clients, cfg)
        metrics['round_time'] = time.time() - logger.round_start_time
        
        # Log metrics
        logger.log_metrics(round_idx, metrics)
        
        if cfg.verbose:
            logging.info(f"Round {round_idx}: clean_acc={metrics['clean_acc']:.3f}, "
                        f"adv_acc={metrics['adv_acc']:.3f}, "
                        f"grey_sr={metrics['grey_sr']:.3f}")

def evaluate_round(clients: List[Client], cfg: Config) -> Dict[str, float]:
    """Evaluate current round on all metrics."""
    metrics = {}
    
    # Clean accuracy
    clean_acc = 0
    for client in clients:
        _, acc = client.evaluate()  # Now using the client's evaluate method without arguments
        clean_acc += acc
    metrics['clean_acc'] = clean_acc / len(clients)
    
    # Adversarial accuracy (white-box PGD)
    adv_acc = 0
    attack = torch.nn.ModuleList([
        PGDAttack(
            epsilon=cfg.PGD_EPS,
            steps=cfg.PGD_STEPS
        ) for _ in range(cfg.N_CLIENTS)
    ])
    
    for client_idx, client in enumerate(clients):
        test_loader = client.test_loader  # Use the client's test_loader
        correct = 0
        total = 0
        
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # Generate adversarial examples
            adv_data = attack[client_idx].perturb(client.model, data, target)
            
            # Evaluate with defense
            with torch.no_grad():
                output = client.classifier(adv_data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        adv_acc += correct / total
    metrics['adv_acc'] = adv_acc / len(clients)
    
    # Grey-box attack success rate
    metrics['grey_sr'] = internal_attack(clients, cfg)
    
    return metrics 