#!/usr/bin/env python3
"""
System Check for Kim et al., 2023 Reproduction
==============================================

Quick system check before running the Kim et al., 2023 reproduction experiment.
"""

import os
import sys
import subprocess
from pathlib import Path
import torch
import torchvision.models as models

def check_gpu():
    """Check GPU availability and memory"""
    print("🔍 GPU Check:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # Check GPU memory
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    used, total = map(int, line.split(', '))
                    free = total - used
                    print(f"   GPU {i} Memory: {used}MB/{total}MB used, {free}MB free")
        except:
            print("   GPU memory info not available")
    else:
        print("   ⚠️  CUDA not available - will run on CPU (very slow!)")

def check_models():
    """Check required pre-trained models"""
    print("\n🔍 Pre-trained Models Check:")
    
    required_models = [
        'checkpoints/diffuser_cifar10.pt',
        'checkpoints/mae_detector_cifar10.pt'
    ]
    
    all_found = True
    for model_path in required_models:
        if Path(model_path).exists():
            size = Path(model_path).stat().st_size / (1024*1024)
            print(f"   ✅ {model_path} ({size:.1f}MB)")
        else:
            print(f"   ❌ {model_path} - MISSING")
            all_found = False
    
    if not all_found:
        print("   ⚠️  Some models missing - defense components may be disabled")
    
    return all_found

def check_architecture():
    """Check MobileNetV2 architecture"""
    print("\n🔍 Architecture Check:")
    
    try:
        model = models.mobilenet_v2(weights=None)
        print(f"   ✅ MobileNetV2 created successfully")
        
        # Check classifier
        in_features = model.classifier[1].in_features
        print(f"   Classifier input features: {in_features}")
        
        # Test forward pass
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model = model.to(device)
            test_input = torch.randn(2, 3, 32, 32).to(device)  # batch_size=2 for BatchNorm
        else:
            device = torch.device('cpu')
            test_input = torch.randn(2, 3, 32, 32)  # batch_size=2 for BatchNorm
        
        model.eval()  # Set to eval mode to avoid BatchNorm issues
        with torch.no_grad():
            output = model(test_input)
        
        print(f"   ✅ Forward pass successful: {output.shape}")
        
    except Exception as e:
        print(f"   ❌ MobileNetV2 error: {e}")
        return False
    
    return True

def check_disk_space():
    """Check available disk space"""
    print("\n🔍 Disk Space Check:")
    
    try:
        result = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                header = lines[0]
                data = lines[1]
                print(f"   {header}")
                print(f"   {data}")
                
                # Extract available space
                parts = data.split()
                if len(parts) >= 4:
                    available = parts[3]
                    print(f"   Available: {available}")
                    
                    # Check if enough space (rough estimate: need ~10GB)
                    if 'G' in available:
                        size = float(available.replace('G', ''))
                        if size < 10:
                            print("   ⚠️  Low disk space - may cause issues")
    except:
        print("   Disk space info not available")

def check_python_env():
    """Check Python environment"""
    print("\n🔍 Python Environment Check:")
    print(f"   Python: {sys.version}")
    print(f"   PyTorch: {torch.__version__}")
    
    try:
        import torchvision
        print(f"   TorchVision: {torchvision.__version__}")
    except:
        print("   ❌ TorchVision not available")
    
    try:
        import numpy as np
        print(f"   NumPy: {np.__version__}")
    except:
        print("   ❌ NumPy not available")

def main():
    """Main system check"""
    print("=" * 50)
    print("KIM ET AL., 2023 SYSTEM CHECK")
    print("=" * 50)
    
    # Run all checks
    check_python_env()
    check_gpu()
    models_ok = check_models()
    arch_ok = check_architecture()
    check_disk_space()
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print("✅ GPU available")
    else:
        print("⚠️  No GPU - will be very slow")
    
    if models_ok:
        print("✅ All pre-trained models found")
    else:
        print("⚠️  Some pre-trained models missing")
    
    if arch_ok:
        print("✅ MobileNetV2 architecture working")
    else:
        print("❌ Architecture issues detected")
    
    # Overall status
    if torch.cuda.is_available() and arch_ok:
        print("\n🎯 STATUS: READY TO RUN")
        print("\nNext steps:")
        print("1. python run_kim2023_reproduction.py --background")
        print("2. tail -f kim2023_reproduction_*.log")
    else:
        print("\n⚠️  STATUS: ISSUES DETECTED")
        print("Please resolve issues before running experiment")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
