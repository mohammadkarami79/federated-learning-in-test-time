import os
import csv
import time
from pathlib import Path
from typing import Dict

class MetricsLogger:
    def __init__(self, output_dir: str):
        """Initialize metrics logger.
        
        Args:
            output_dir: Directory to save metrics files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics file
        self.metrics_file = self.output_dir / 'metrics.csv'
        self.fieldnames = ['round', 'clean_acc', 'adv_acc', 'grey_sr', 'round_time']
        
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                
    def log_round(self, round_idx: int, metrics: Dict[str, float]):
        """Log metrics for current round.
        
        Args:
            round_idx: Current round number
            metrics: Dictionary of metrics to log
        """
        metrics['round'] = round_idx
        
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(metrics)
            
        # Also save detailed round metrics
        round_file = self.output_dir / f'metrics_round{round_idx:03d}.csv'
        with open(round_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writeheader()
            writer.writerow(metrics) 