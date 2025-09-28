#!/usr/bin/env python3
"""
Debug GPU Memory Issues for CIFAR-10 Training
"""

import torch
import psutil
import GPUtil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_system():
    """Debug system resources and GPU status"""
    
    logger.info("🔍 SYSTEM DIAGNOSTICS")
    logger.info("=" * 50)
    
    # CPU and RAM
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    logger.info(f"💻 CPU Usage: {cpu_percent}%")
    logger.info(f"🧠 RAM: {memory.percent}% used ({memory.used // (1024**3)} GB / {memory.total // (1024**3)} GB)")
    
    # GPU Status
    if torch.cuda.is_available():
        logger.info(f"🚀 CUDA Available: {torch.cuda.device_count()} GPUs")
        
        for i in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            memory_total = gpu.total_memory / (1024**3)
            
            logger.info(f"GPU {i}: {gpu.name}")
            logger.info(f"  - Total Memory: {memory_total:.1f} GB")
            logger.info(f"  - Allocated: {memory_allocated:.1f} GB")
            logger.info(f"  - Reserved: {memory_reserved:.1f} GB")
            logger.info(f"  - Free: {memory_total - memory_reserved:.1f} GB")
            
        # Clear GPU cache
        torch.cuda.empty_cache()
        logger.info("🧹 GPU cache cleared")
        
    else:
        logger.warning("❌ CUDA not available")
    
    # Process Information
    logger.info("\n🔄 RUNNING PYTHON PROCESSES:")
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_percent']):
        try:
            if proc.info['name'] == 'python' and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'][-2:])  # Last 2 args
                logger.info(f"  PID {proc.info['pid']}: {cmdline} (RAM: {proc.info['memory_percent']:.1f}%)")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

def test_simple_training():
    """Test simple CIFAR-10 training to identify bottlenecks"""
    
    logger.info("\n🧪 TESTING SIMPLE TRAINING")
    logger.info("=" * 50)
    
    try:
        import torchvision
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        
        # Simple dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=False, transform=transform
        )
        
        # Test different configurations
        configs = [
            {"batch_size": 32, "num_workers": 0, "name": "Single-threaded"},
            {"batch_size": 32, "num_workers": 2, "name": "Multi-threaded"},
            {"batch_size": 64, "num_workers": 2, "name": "Larger batch"},
        ]
        
        for config in configs:
            logger.info(f"\n📊 Testing: {config['name']}")
            try:
                dataloader = DataLoader(
                    dataset, 
                    batch_size=config['batch_size'], 
                    shuffle=True, 
                    num_workers=config['num_workers'],
                    pin_memory=torch.cuda.is_available(),
                    timeout=30 if config['num_workers'] > 0 else 0
                )
                
                # Test loading a few batches
                for i, (data, target) in enumerate(dataloader):
                    if torch.cuda.is_available():
                        data, target = data.cuda(), target.cuda()
                    
                    logger.info(f"  Batch {i+1}: {data.shape} loaded successfully")
                    
                    if i >= 2:  # Test only 3 batches
                        break
                        
                logger.info(f"  ✅ {config['name']} works fine")
                
            except Exception as e:
                logger.error(f"  ❌ {config['name']} failed: {e}")
        
    except Exception as e:
        logger.error(f"❌ Training test failed: {e}")

def recommend_fixes():
    """Recommend fixes based on diagnostics"""
    
    logger.info("\n💡 RECOMMENDATIONS")
    logger.info("=" * 50)
    
    # Check GPU memory
    if torch.cuda.is_available():
        memory_free = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        memory_used = torch.cuda.memory_reserved(0) / (1024**3)
        memory_available = memory_free - memory_used
        
        if memory_available < 8:
            logger.warning("⚠️  Low GPU memory detected")
            logger.info("🔧 Try reducing batch size or num_workers")
            logger.info("🔧 Command: python main.py --dataset cifar10 --mode full --skip-setup")
        else:
            logger.info("✅ GPU memory looks good")
    
    # Check RAM
    memory = psutil.virtual_memory()
    if memory.percent > 85:
        logger.warning("⚠️  High RAM usage detected")
        logger.info("🔧 Close unnecessary programs")
        logger.info("🔧 Reduce num_workers in DataLoader")
    else:
        logger.info("✅ RAM usage looks good")
    
    logger.info("\n🚀 SUGGESTED COMMANDS:")
    logger.info("1. Kill all Python processes: pkill -f python")
    logger.info("2. Clear GPU cache: python -c 'import torch; torch.cuda.empty_cache()'")
    logger.info("3. Restart training: nohup python main.py --dataset cifar10 --mode full --skip-setup > new_log.log 2>&1 &")

if __name__ == "__main__":
    try:
        debug_system()
        test_simple_training()
        recommend_fixes()
    except Exception as e:
        logger.error(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
