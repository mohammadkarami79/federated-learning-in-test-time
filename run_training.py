#!/usr/bin/env python3
"""
Simple Training Launcher for pFedDef + DiffPure
Routes to the main.py with appropriate arguments
"""

import sys
import subprocess
import argparse

def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description='Launch pFedDef + DiffPure Training')
    parser.add_argument('mode', nargs='?', default='debug', 
                       choices=['debug', 'test', 'full'],
                       help='Training mode: debug (2-5 min), test (10-20 min), full (20-60 min)')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100', 'mnist'],
                       help='Dataset to use')
    parser.add_argument('--train-diffusion', action='store_true',
                       help='Train diffusion model before main training')
    parser.add_argument('--train-mae', action='store_true',
                       help='Train MAE detector before main training')
    return parser.parse_args()

def main():
    """Main launcher"""
    args = parse_args()
    
    print(f"🚀 Launching pFedDef + DiffPure Training")
    print(f"📋 Mode: {args.mode}, Dataset: {args.dataset.upper()}")
    
    # Build command for main.py
    cmd = [
        sys.executable, 'main.py',
        '--mode', args.mode,
        '--dataset', args.dataset
    ]
    
    if args.train_diffusion:
        cmd.append('--train-diffusion')
    
    if args.train_mae:
        cmd.append('--train-mae')
    
    print(f"💻 Running: {' '.join(cmd)}")
    
    # Execute main.py
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 