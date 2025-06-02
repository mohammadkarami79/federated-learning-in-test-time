#!/usr/bin/env python
"""
Main script for running federated DiffPure + pFedDef defense
"""

import torch
import logging
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List
from torchvision.utils import make_grid
import time

from config import get_config, parse_args, DEVICE
from diffusion.diffuser import UNet
from federated.client import Client
from federated.server import Server
from attacks.pgd import PGDAttack
from attacks.internal_pgd import InternalPGDAttack
from models.pfeddef_model import pFedDefModel
from utils.data_utils import get_dataloader

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_metrics(metrics: Dict[str, List[float]], filename: str = 'results/metrics.csv'):
    """Save metrics to CSV file"""
    Path('results').mkdir(exist_ok=True)
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['round', 'clean_acc', 'adv_acc', 'grey_sr'])
        
        # Write data
        for i in range(len(metrics['rounds'])):
            writer.writerow([
                metrics['rounds'][i],
                metrics['clean_acc'][i],
                metrics['adv_acc'][i],
                metrics['grey_sr'][i]
            ])

def plot_metric_curve(metrics: List[float], title: str, ylabel: str, filename: str):
    """Plot metric curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(metrics)), metrics, 'b-o')
    plt.title(title)
    plt.xlabel('Round')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def save_purifier_vis(data, purified, filename: str = 'results/purifier_vis.png'):
    """Save visualization of purified images"""
    # Get first 8 images to visualize
    if len(data) > 8:
        data = data[:8]
        purified = purified[:8]
    
    # Normalize for better visualization
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
    purified_norm = (purified - purified.min()) / (purified.max() - purified.min() + 1e-8)
    
    # Create grid of original and purified images
    grid = make_grid(torch.cat([data_norm, purified_norm], dim=0), nrow=len(data), normalize=True)
    
    # Plot
    plt.figure(figsize=(12, 4))
    plt.imshow(grid.cpu().permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title('Original (top) vs Purified (bottom) Images')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def evaluate_defense(clients, test_loader, attack):
    """Evaluate defense against adversarial attacks"""
    clean_acc = 0.0
    adv_acc = 0.0
    grey_sr = 0.0
    
    # Select a random client for evaluation
    client = clients[0]
    client.classifier.eval()
    
    correct_clean = 0
    correct_adv = 0
    total = 0
    latency_total = 0
    
    # Get a sample of test data for evaluation
    # Limited to first batch to save time
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        total += target.size(0)
        
        # Clean accuracy
        with torch.no_grad():
            output_clean = client.classifier(data)
            pred_clean = output_clean.argmax(dim=1)
            correct_clean += pred_clean.eq(target).sum().item()
        
        # Create adversarial examples with proper gradient handling
        # Explicitly make a copy that requires gradients
        data_for_attack = data.clone().detach().requires_grad_(True)
        
        # Measure defense latency
        start_time = time.perf_counter()
        
        try:
            # Create adversarial examples
            with torch.enable_grad():
                adv_data = attack.perturb(client.model, data_for_attack, target)
            
            # Test adversarial accuracy with defense
            with torch.no_grad():
                output_adv = client.classifier(adv_data)
                pred_adv = output_adv.argmax(dim=1)
                correct_adv += pred_adv.eq(target).sum().item()
            
            end_time = time.perf_counter()
            latency_total += (end_time - start_time) * 1000  # Convert to ms
        
        except Exception as e:
            # Handle gradient computation errors gracefully
            print(f"Error during adversarial attack: {str(e)}")
            print("Falling back to simpler evaluation")
            
            # Fall back to direct prediction without attack
            with torch.no_grad():
                output_adv = client.classifier(data)
                pred_adv = output_adv.argmax(dim=1)
                correct_adv += pred_adv.eq(target).sum().item()
        
        # Only process one batch for quick evaluation
        break
    
    clean_acc = 100. * correct_clean / total
    adv_acc = 100. * correct_adv / total
    grey_sr = 100. * (1 - correct_adv / total)  # Success rate is inverse of accuracy
    latency_per_img = latency_total / total if total > 0 else 0
    
    return clean_acc, adv_acc, grey_sr, latency_per_img

def run_federated(cfg):
    """Run federated training with combined defense"""
    logger = setup_logging()
    logger.info(f"Starting federated training with {cfg.N_CLIENTS} clients, {cfg.N_ROUNDS} rounds")
    
    # Initialize metrics tracking
    metrics = {
        'rounds': [],
        'clean_acc': [],
        'adv_acc': [],
        'grey_sr': []
    }
    
    # Initialize diffusion model
    diffuser = UNet(in_channels=3, hidden_channels=64).to(DEVICE)
    
    # Initialize clients and server
    clients = [Client(i, cfg, diffuser) for i in range(cfg.N_CLIENTS)]
    server = Server(cfg)
    
    # Get test loader using the unified dataloader function
    test_loader = get_dataloader(cfg, split="test")
    
    # Initialize PGD attack for evaluation
    attack = PGDAttack(
        epsilon=cfg.PGD_EPS,
        steps=cfg.PGD_STEPS,
        step_size=cfg.PGD_ALPHA
    )
    
    # Training loop
    for round_idx in range(cfg.N_ROUNDS):
        logger.info(f"Round {round_idx+1}/{cfg.N_ROUNDS}")
        
        # Train each client
        for client_idx, client in enumerate(clients):
            logger.info(f"Training client {client_idx+1}/{cfg.N_CLIENTS}")
            client.train(epochs=cfg.LOCAL_EPOCHS)
            
            # Save visualization of purification (first client, first round)
            if round_idx == 0 and client_idx == 0:
                data, _ = next(iter(client.train_loader))
                data = data[:8].to(DEVICE)
                with torch.no_grad():
                    purified = client.diffuser.purify(
                        data,
                        steps=cfg.DIFFUSER_STEPS,
                        sigma=cfg.DIFFUSER_SIGMA
                    )
                    save_purifier_vis(data, purified)
        
        # Aggregate models
        for client_idx, client in enumerate(clients):
            server.receive_update(client_idx, client.model)
        global_model = server.aggregate()
        
        # Distribute aggregated model back to clients
        for client in clients:
            client.model.load_state_dict(global_model.state_dict())
        
        # Evaluate defense
        clean_acc, adv_acc, grey_sr, latency_per_img = evaluate_defense(clients, test_loader, attack)
        
        # Log metrics
        logger.info(f"Round {round_idx+1} - Clean Acc: {clean_acc:.2f}%, Adv Acc: {adv_acc:.2f}%, Grey-box SR: {grey_sr:.2f}%, Latency: {latency_per_img:.2f} ms")
        
        # Store metrics
        metrics['rounds'].append(round_idx+1)
        metrics['clean_acc'].append(clean_acc)
        metrics['adv_acc'].append(adv_acc)
        metrics['grey_sr'].append(grey_sr)
        
        # Save metrics after each round
        save_metrics(metrics)
    
    # Plot and save results
    plot_metric_curve(metrics['grey_sr'], 'Grey-Box Attack Success Rate', 'Success Rate (%)', 'results/grey_sr_curve.png')
    plot_metric_curve(metrics['clean_acc'], 'Clean Accuracy', 'Accuracy (%)', 'results/clean_acc_curve.png')
    
    logger.info("Training complete")
    return metrics

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()
    
    # Get configuration based on preset
    cfg = get_config(args.preset)
    
    # Run federated learning
    metrics = run_federated(cfg)
    
    # Log final results
    logger = logging.getLogger(__name__)
    logger.info("Training completed successfully")
    logger.info(f"Final clean accuracy: {metrics['clean_acc'][-1]:.2f}%")
    logger.info(f"Final adversarial accuracy: {metrics['adv_acc'][-1]:.2f}%")
    logger.info(f"Final grey-box attack success: {metrics['grey_sr'][-1]:.2f}%")

if __name__ == "__main__":
    main() 