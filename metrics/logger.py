"""
Metrics logger for tracking training and evaluation metrics
"""

import csv
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

class MetricsLogger:
    def __init__(self, log_dir):
        """
        Initialize metrics logger
        
        Args:
            log_dir: Directory to save metrics
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = {
            'round': [],
            'clean_acc': [],
            'adv_acc': [],
            'greybox_sr': [],
            'latency': [],
            'memory': []
        }
        
        # Create CSV file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = self.log_dir / f'metrics_{timestamp}.csv'
        
        # Write header
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.metrics.keys())
            
        self.round_start_time = None
    
    def start_round(self):
        """Start timing a new round."""
        self.round_start_time = time.time()
    
    def log_metrics(self, round_idx, metrics):
        """
        Log metrics for the current round
        
        Args:
            round_idx: Current round index
            metrics: Dictionary containing metrics to log
        """
        self.log_round(
            round_num=round_idx,
            clean_acc=metrics['clean_acc'],
            adv_acc=metrics['adv_acc'],
            greybox_sr=metrics.get('grey_sr', 0.0),  # Handle both grey_sr and greybox_sr keys
            latency=metrics.get('round_time', 0.0),
            memory=metrics.get('memory', 0.0)
        )
    
    def log_round(self, round_num, clean_acc, adv_acc, greybox_sr, latency, memory):
        """
        Log metrics for a round
        
        Args:
            round_num: Current round number
            clean_acc: Clean accuracy
            adv_acc: Adversarial accuracy
            greybox_sr: Grey-box attack success rate
            latency: Training latency in seconds
            memory: GPU memory usage in MB
        """
        # Store metrics
        self.metrics['round'].append(round_num)
        self.metrics['clean_acc'].append(clean_acc)
        self.metrics['adv_acc'].append(adv_acc)
        self.metrics['greybox_sr'].append(greybox_sr)
        self.metrics['latency'].append(latency)
        self.metrics['memory'].append(memory)
        
        # Write to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                round_num,
                f"{clean_acc:.2f}",
                f"{adv_acc:.2f}",
                f"{greybox_sr:.2f}",
                f"{latency:.2f}",
                f"{memory:.2f}"
            ])
    
    def get_summary(self):
        """
        Get summary statistics of logged metrics
        
        Returns:
            dict: Summary statistics
        """
        summary = {}
        for metric in ['clean_acc', 'adv_acc', 'greybox_sr', 'latency', 'memory']:
            values = np.array(self.metrics[metric])
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
            summary[f'{metric}_min'] = np.min(values)
            summary[f'{metric}_max'] = np.max(values)
        
        return summary
    
    def plot_metrics(self):
        """
        Plot logged metrics
        
        Returns:
            dict: Paths to generated plots
        """
        try:
            import matplotlib.pyplot as plt
            
            plot_paths = {}
            for metric in ['clean_acc', 'adv_acc', 'greybox_sr']:
                plt.figure(figsize=(10, 6))
                plt.plot(self.metrics['round'], self.metrics[metric], 'b-', label=metric)
                plt.xlabel('Round')
                plt.ylabel(metric)
                plt.title(f'{metric} vs Round')
                plt.grid(True)
                plt.legend()
                
                # Save plot
                plot_path = self.log_dir / f'{metric}_plot.png'
                plt.savefig(plot_path)
                plt.close()
                
                plot_paths[metric] = plot_path
            
            return plot_paths
            
        except ImportError:
            print("Matplotlib not installed. Skipping plot generation.")
            return {} 