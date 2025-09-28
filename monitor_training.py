#!/usr/bin/env python3
"""
Training Monitor - Check progress of selective defense training
"""

import os
import time
import json
from pathlib import Path
import glob

def monitor_training():
    """Monitor training progress by checking log files and results"""
    
    print("=== Selective Defense Training Monitor ===")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if training is running
    project_root = Path(__file__).parent
    
    # Check for log files
    log_files = list(project_root.glob("*.log"))
    if log_files:
        print("[LOGS] Log Files Found:")
        for log_file in log_files:
            size = log_file.stat().st_size
            modified = time.ctime(log_file.stat().st_mtime)
            print(f"  - {log_file.name}: {size} bytes, modified: {modified}")
        print()
    
    # Check experiment results
    results_dir = project_root / "experiment_results"
    if results_dir.exists():
        print("[RESULTS] Experiment Results:")
        result_files = list(results_dir.glob("*.json"))
        if result_files:
            # Get the latest result file
            latest_result = max(result_files, key=lambda f: f.stat().st_mtime)
            print(f"  Latest: {latest_result.name}")
            
            try:
                with open(latest_result, 'r') as f:
                    data = json.load(f)
                    
                if 'rounds' in data:
                    print(f"  Rounds completed: {len(data['rounds'])}")
                    
                    if data['rounds']:
                        latest_round = data['rounds'][-1]
                        print(f"  Latest round metrics:")
                        if 'clean_accuracy' in latest_round:
                            print(f"    Clean Accuracy: {latest_round['clean_accuracy']:.2f}%")
                        if 'adversarial_accuracy' in latest_round:
                            print(f"    Adversarial Accuracy: {latest_round['adversarial_accuracy']:.2f}%")
                        if 'detection_rate' in latest_round:
                            print(f"    Detection Rate: {latest_round['detection_rate']:.2f}%")
                            
            except Exception as e:
                print(f"  Error reading results: {e}")
        else:
            print("  No result files found yet")
        print()
    
    # Check for checkpoint files
    checkpoints_dir = project_root / "checkpoints"
    if checkpoints_dir.exists():
        checkpoint_files = list(checkpoints_dir.glob("*.pth"))
        if checkpoint_files:
            print("[CHECKPOINTS] Saved Models:")
            for cp in checkpoint_files[-3:]:  # Show last 3
                size = cp.stat().st_size / (1024*1024)  # MB
                modified = time.ctime(cp.stat().st_mtime)
                print(f"  - {cp.name}: {size:.1f}MB, {modified}")
            print()
    
    # Check system resources
    try:
        import psutil
        print("[SYSTEM] System Status:")
        print(f"  CPU Usage: {psutil.cpu_percent()}%")
        print(f"  Memory Usage: {psutil.virtual_memory().percent}%")
        print(f"  Available Memory: {psutil.virtual_memory().available / (1024**3):.1f}GB")
        print()
    except ImportError:
        pass
    
    # Check GPU if available
    try:
        import torch
        if torch.cuda.is_available():
            print("[GPU] GPU Status:")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_used = torch.cuda.memory_allocated(i) / (1024**3)
                memory_total = props.total_memory / (1024**3)
                print(f"  GPU {i} ({props.name}): {memory_used:.1f}GB / {memory_total:.1f}GB")
            print()
    except Exception:
        pass
    
    print("=== End Monitor ===")

if __name__ == "__main__":
    monitor_training()
