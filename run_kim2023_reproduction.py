#!/usr/bin/env python3
"""
Quick runner for Kim et al., 2023 reproduction experiment
=========================================================

This script provides an easy way to run the Kim et al., 2023 reproduction
with different configurations and monitoring options.
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

def run_experiment(mode='full', background=False, log_suffix=''):
    """Run the Kim et al., 2023 reproduction experiment"""
    
    script_path = Path(__file__).parent / 'main_kim2023_reproduction.py'
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return 1
    
    # Create command
    cmd = ['python', str(script_path)]
    
    # Add output directory with timestamp
    import time
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = f'./kim2023_results_{mode}_{timestamp}{log_suffix}'
    cmd.extend(['--output-dir', output_dir])
    
    print("🚀 Starting Kim et al., 2023 Reproduction Experiment")
    print("=" * 70)
    print(f"📊 Mode: {mode}")
    print(f"📂 Output: {output_dir}")
    print(f"🔧 Command: {' '.join(cmd)}")
    print()
    print("📋 Experiment Configuration:")
    print("   - Architecture: MobileNetV2 (vs ResNet18 in log8.txt)")
    print("   - Attack: L2-PGD ε=4.5, α=0.01, K=10 (vs L∞-PGD ε=0.031)")
    print("   - Clients: 40 (vs 10 in log8.txt)")
    print("   - Data: Non-IID β=0.4 (vs IID in log8.txt)")
    print("   - Expected Runtime: ~4-5 hours (vs ~1.5 hours for log8.txt)")
    print("=" * 70)
    
    if background:
        # Run in background with nohup
        log_file = f'kim2023_reproduction_{mode}_{timestamp}.log'
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                ['nohup'] + cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
        
        print(f"✅ Started in background (PID: {process.pid})")
        print(f"📝 Log file: {log_file}")
        print(f"👁️  Monitor with: tail -f {log_file}")
        return 0
    else:
        # Run in foreground
        try:
            result = subprocess.run(cmd, check=True)
            print("✅ Experiment completed successfully!")
            return result.returncode
        except subprocess.CalledProcessError as e:
            print(f"❌ Experiment failed with exit code: {e.returncode}")
            return e.returncode
        except KeyboardInterrupt:
            print("\n⚠️  Experiment interrupted by user")
            return 130

def main():
    parser = argparse.ArgumentParser(description='Run Kim et al., 2023 Reproduction')
    parser.add_argument('--mode', choices=['full', 'test', 'debug'], default='full',
                       help='Experiment mode')
    parser.add_argument('--background', '-b', action='store_true',
                       help='Run in background with nohup')
    parser.add_argument('--log-suffix', default='',
                       help='Suffix for log files')
    
    args = parser.parse_args()
    
    # Check system
    if not Path('./data').exists():
        print("📁 Creating data directory...")
        Path('./data').mkdir(exist_ok=True)
    
    if not Path('./checkpoints').exists():
        print("📁 Creating checkpoints directory...")
        Path('./checkpoints').mkdir(exist_ok=True)
    
    # Check for required models
    required_models = [
        'checkpoints/diffuser_cifar10.pt',
        'checkpoints/mae_detector_cifar10.pt'
    ]
    
    missing_models = [model for model in required_models if not Path(model).exists()]
    if missing_models:
        print("⚠️  Warning: Some pre-trained models are missing:")
        for model in missing_models:
            print(f"   - {model}")
        print("The experiment will try to load them, but may skip defense components if not found.")
        print()
    
    # Run experiment
    return run_experiment(args.mode, args.background, args.log_suffix)

if __name__ == "__main__":
    sys.exit(main())
