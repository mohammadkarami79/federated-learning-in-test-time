#!/usr/bin/env python3
"""
Complete System Setup for pFedDef + DiffPure
This script ensures all prerequisites are met before training
"""

import sys
import time
import torch
import logging
from pathlib import Path
from typing import Tuple

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_system_requirements() -> Tuple[bool, str]:
    """Check if system meets requirements"""
    logger = logging.getLogger(__name__)
    
    # Check Python version
    if sys.version_info < (3, 8):
        return False, f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}"
    
    # Check PyTorch
    try:
        import torch
        logger.info(f"✅ PyTorch {torch.__version__}")
    except ImportError:
        return False, "PyTorch not installed. Run: pip install torch torchvision"
    
    # Check CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        if gpu_memory < 3.0:
            logger.warning(f"⚠️ GPU has {gpu_memory:.1f}GB memory. 4GB+ recommended.")
    else:
        logger.info("⚠️ CUDA not available. Training will be slower on CPU.")
    
    # Check core dependencies
    required_modules = ['torchvision', 'numpy', 'matplotlib', 'tqdm', 'PIL']
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✅ {module}")
        except ImportError:
            return False, f"Missing dependency: {module}. Run: pip install -r requirements.txt"
    
    return True, "All requirements satisfied"

def check_data_ready() -> Tuple[bool, str]:
    """Check if CIFAR-10 data is ready"""
    logger = logging.getLogger(__name__)
    
    try:
        from config_fixed import get_debug_config
        from utils.data_utils import get_dataset
        
        cfg = get_debug_config()
        train_dataset, test_dataset = get_dataset(cfg)
        
        logger.info(f"✅ CIFAR-10 dataset: {len(train_dataset)} train, {len(test_dataset)} test")
        return True, "Dataset ready"
    
    except Exception as e:
        return False, f"Dataset loading failed: {e}"

def check_models_ready() -> Tuple[bool, str]:
    """Check if models can be created"""
    logger = logging.getLogger(__name__)
    
    try:
        from config_fixed import get_debug_config
        from models import get_model
        
        cfg = get_debug_config()
        model = get_model(cfg)
        total_params = sum(p.numel() for p in model.parameters())
        
        logger.info(f"✅ Model created: {total_params:,} parameters")
        return True, "Models ready"
    
    except Exception as e:
        return False, f"Model creation failed: {e}"

def setup_diffusion_model() -> Tuple[bool, str]:
    """Setup diffusion model (check if exists or create minimal one)"""
    logger = logging.getLogger(__name__)
    
    # Check if diffusion checkpoint exists
    checkpoint_path = Path('checkpoints/diffuser.pt')
    
    if checkpoint_path.exists():
        logger.info(f"✅ Diffusion model found: {checkpoint_path}")
        return True, "Diffusion model ready"
    
    # Create minimal diffusion model if not exists
    logger.info("📦 Creating minimal diffusion model...")
    try:
        from diffusion.diffuser import UNet
        from config_fixed import get_debug_config
        
        cfg = get_debug_config()
        diffuser = UNet(in_channels=3, hidden_channels=64)
        
        # Create checkpoints directory
        checkpoint_path.parent.mkdir(exist_ok=True)
        
        # Save minimal model (will be fine-tuned during training)
        torch.save(diffuser.state_dict(), checkpoint_path)
        logger.info(f"✅ Minimal diffusion model created: {checkpoint_path}")
        
        return True, "Diffusion model created"
    
    except Exception as e:
        return False, f"Diffusion model setup failed: {e}"

def setup_mae_detector() -> Tuple[bool, str]:
    """Setup MAE detector"""
    logger = logging.getLogger(__name__)
    
    try:
        from config_fixed import get_debug_config
        from defense.mae_detector import MAEDetector
        
        cfg = get_debug_config()
        detector = MAEDetector(cfg)
        
        # Test detection
        test_images = torch.randn(2, 3, 32, 32)
        if torch.cuda.is_available():
            test_images = test_images.cuda()
        
        with torch.no_grad():
            predictions = detector.detect(test_images)
        
        logger.info("✅ MAE detector ready")
        return True, "MAE detector ready"
    
    except Exception as e:
        return False, f"MAE detector setup failed: {e}"

def check_attacks_ready() -> Tuple[bool, str]:
    """Check if adversarial attacks work"""
    logger = logging.getLogger(__name__)
    
    try:
        from config_fixed import get_debug_config
        from attacks.pgd import PGDAttack
        from models import get_model
        
        cfg = get_debug_config()
        model = get_model(cfg)
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        model.eval()  # Set model to eval mode
        
        attack = PGDAttack(cfg)
        
        # Test attack
        test_images = torch.randn(2, 3, 32, 32)
        test_labels = torch.randint(0, 10, (2,))
        
        if torch.cuda.is_available():
            test_images = test_images.cuda()
            test_labels = test_labels.cuda()
        
        test_images.requires_grad_(True)
        
        # Use the attack properly - try different calling methods
        try:
            adv_images = attack.forward(model, test_images, test_labels)
        except Exception:
            # Fallback to generate method
            adv_images = attack.generate(model, test_images, test_labels)
        
        logger.info("✅ PGD attack ready")
        return True, "Attacks ready"
    
    except Exception as e:
        return False, f"Attack setup failed: {e}"

def check_federated_ready() -> Tuple[bool, str]:
    """Check if federated components work"""
    logger = logging.getLogger(__name__)
    
    try:
        from config_fixed import get_debug_config
        from federated.client import Client
        from federated.server import Server
        from diffusion.diffuser import UNet
        
        cfg = get_debug_config()
        
        # Test diffusion model
        diffuser = UNet(in_channels=3, hidden_channels=64)
        if torch.cuda.is_available():
            diffuser = diffuser.cuda()
        
        # Test client creation
        client = Client(0, cfg, diffuser)
        
        # Test server creation
        server = Server(cfg)
        
        logger.info("✅ Federated components ready")
        return True, "Federated learning ready"
    
    except Exception as e:
        return False, f"Federated setup failed: {e}"

def run_comprehensive_check() -> bool:
    """Run comprehensive system check"""
    logger = setup_logging()
    
    print("🔍 COMPREHENSIVE SYSTEM CHECK")
    print("=" * 60)
    
    checks = [
        ("System Requirements", check_system_requirements),
        ("Data Loading", check_data_ready),
        ("Model Creation", check_models_ready),
        ("Diffusion Model", setup_diffusion_model),
        ("MAE Detector", setup_mae_detector),
        ("Adversarial Attacks", check_attacks_ready),
        ("Federated Learning", check_federated_ready),
    ]
    
    all_passed = True
    results = []
    
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}")
        print("-" * 40)
        
        try:
            start_time = time.time()
            success, message = check_func()
            duration = time.time() - start_time
            
            if success:
                print(f"✅ PASS - {message} ({duration:.2f}s)")
                results.append((check_name, "✅ PASS", message))
            else:
                print(f"❌ FAIL - {message}")
                results.append((check_name, "❌ FAIL", message))
                all_passed = False
                
        except Exception as e:
            print(f"❌ ERROR - {str(e)}")
            results.append((check_name, "❌ ERROR", str(e)))
            all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SYSTEM CHECK SUMMARY")
    print("=" * 60)
    
    for check_name, status, message in results:
        print(f"{status} {check_name}: {message}")
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 ALL CHECKS PASSED - SYSTEM READY FOR TRAINING!")
        print("\n💡 Next steps:")
        print("   1. Run quick test: python simple_test.py")
        print("   2. Start training: python run_training.py debug")
        print("   3. Or use optimized: python main1.py")
        return True
    else:
        print("❌ SOME CHECKS FAILED - PLEASE FIX ISSUES ABOVE")
        print("\n🔧 Troubleshooting:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Check GPU setup: nvidia-smi")
        print("   3. Run setup again: python setup_system.py")
        return False

def main():
    """Main function"""
    success = run_comprehensive_check()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 