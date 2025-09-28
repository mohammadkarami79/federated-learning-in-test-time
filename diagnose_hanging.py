#!/usr/bin/env python3
"""
Diagnose why diffusion training hangs
"""

import torch
import subprocess
import psutil
import logging
import os
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_system_resources():
    """Check system resources"""
    logger.info("🔍 SYSTEM RESOURCE CHECK")
    logger.info("=" * 50)
    
    # GPU
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            total = props.total_memory / (1024**3)
            
            logger.info(f"GPU {i}: {props.name}")
            logger.info(f"  Total: {total:.1f} GB")
            logger.info(f"  Allocated: {allocated:.1f} GB")
            logger.info(f"  Reserved: {reserved:.1f} GB")
            logger.info(f"  Free: {total - reserved:.1f} GB")
            
            if (total - reserved) < 4:
                logger.warning("⚠️  LOW GPU MEMORY!")
    
    # RAM
    memory = psutil.virtual_memory()
    logger.info(f"RAM: {memory.percent:.1f}% used ({memory.used // (1024**3)} GB / {memory.total // (1024**3)} GB)")
    if memory.percent > 85:
        logger.warning("⚠️  HIGH RAM USAGE!")
    
    # Disk
    disk = psutil.disk_usage('.')
    logger.info(f"Disk: {disk.percent:.1f}% used ({disk.free // (1024**3)} GB free)")
    if disk.percent > 90:
        logger.warning("⚠️  LOW DISK SPACE!")

def check_training_scripts():
    """Check which training scripts exist"""
    logger.info("\n🔍 TRAINING SCRIPTS CHECK")
    logger.info("=" * 50)
    
    scripts = [
        'train_diffpure.py',
        'diffusion/train_diffusion.py', 
        'scripts/train_diffusion.py',
        'train_diffusion_cifar10.py',
        'quick_diffusion_cifar10.py'
    ]
    
    found_scripts = []
    for script in scripts:
        path = Path(script)
        if path.exists():
            size = path.stat().st_size
            logger.info(f"✅ {script} exists ({size} bytes)")
            found_scripts.append(script)
        else:
            logger.info(f"❌ {script} not found")
    
    return found_scripts

def test_subprocess():
    """Test if subprocess calls work"""
    logger.info("\n🔍 SUBPROCESS TEST")
    logger.info("=" * 50)
    
    try:
        # Test simple command
        result = subprocess.run(['python', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info(f"✅ Python subprocess works: {result.stdout.strip()}")
        else:
            logger.error(f"❌ Python subprocess failed: {result.stderr}")
            
        # Test with current environment
        result = subprocess.run(['python', '-c', 'import torch; print(f"PyTorch: {torch.__version__}")'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info(f"✅ PyTorch subprocess works: {result.stdout.strip()}")
        else:
            logger.error(f"❌ PyTorch subprocess failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Subprocess timeout!")
    except Exception as e:
        logger.error(f"❌ Subprocess error: {e}")

def test_diffusion_import():
    """Test diffusion model imports"""
    logger.info("\n🔍 DIFFUSION IMPORTS TEST")
    logger.info("=" * 50)
    
    try:
        # Test basic imports
        import torchvision
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        logger.info("✅ Basic imports work")
        
        # Test dataset loading
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=False, transform=transform
        )
        logger.info(f"✅ CIFAR-10 dataset loaded: {len(dataset)} samples")
        
        # Test small dataloader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        data, target = next(iter(dataloader))
        logger.info(f"✅ DataLoader works: {data.shape}")
        
    except Exception as e:
        logger.error(f"❌ Diffusion imports failed: {e}")

def suggest_solutions():
    """Suggest solutions based on diagnostics"""
    logger.info("\n💡 SOLUTIONS")
    logger.info("=" * 50)
    
    logger.info("Based on the diagnosis, try these solutions:")
    logger.info("")
    logger.info("1. 🚀 QUICK SOLUTION - Use built-in diffusion training:")
    logger.info("   python main.py --dataset cifar10 --train-diffusion --mode full")
    logger.info("")
    logger.info("2. 🛠️  MANUAL SOLUTION - Train diffusion separately:")
    logger.info("   python quick_diffusion_cifar10.py")
    logger.info("   then: python main.py --dataset cifar10 --mode full --skip-setup")
    logger.info("")
    logger.info("3. 🔧 DEBUG SOLUTION - Check specific error:")
    logger.info("   Look at the last few lines of the log for specific error")
    logger.info("")
    logger.info("4. 💾 MEMORY SOLUTION - If low memory:")
    logger.info("   Reduce batch size or restart system")

if __name__ == "__main__":
    try:
        logger.info("🚀 COMPREHENSIVE DIAGNOSIS")
        logger.info("=" * 60)
        
        check_system_resources()
        found_scripts = check_training_scripts()
        test_subprocess() 
        test_diffusion_import()
        suggest_solutions()
        
        logger.info("\n🎯 DIAGNOSIS COMPLETE")
        logger.info("Check the suggestions above!")
        
    except Exception as e:
        logger.error(f"❌ Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()
