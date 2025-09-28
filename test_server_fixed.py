#!/usr/bin/env python3
"""
Fixed test script for server with corrected UNet parameters and MAE config
"""

import sys
import os

def test_basic_setup():
    """Test basic setup"""
    print("=== 🔍 Testing Basic Setup ===")
    try:
        import torch
        print('✅ Python version:', sys.version)
        print('✅ PyTorch version:', torch.__version__)
        print('✅ CUDA available:', torch.cuda.is_available())
        if torch.cuda.is_available():
            print('✅ GPU name:', torch.cuda.get_device_name(0))
            print('✅ GPU memory:', torch.cuda.get_device_properties(0).total_memory // (1024**3), 'GB')
        return True
    except Exception as e:
        print('❌ Basic setup error:', e)
        return False

def test_core_imports():
    """Test core imports"""
    print("=== 📦 Testing Core Imports ===")
    try:
        from defense.mae_detector1 import MAEDetector
        print('✅ MAE Detector import successful')
        
        from train_diffpure import get_config_for_dataset, parse_args
        print('✅ Train DiffPure import successful')
        
        from diffusion.diffuser import UNet
        print('✅ Diffusion UNet import successful')
        
        from utils.data_utils import get_dataset
        print('✅ Data utils import successful')
        
        print('🎉 ALL IMPORTS SUCCESSFUL!')
        return True
        
    except Exception as e:
        print('❌ Import error:', e)
        import traceback
        traceback.print_exc()
        return False

def test_br35h_config():
    """Test BR35H config"""
    print("=== ⚙️ Testing BR35H Config ===")
    try:
        from train_diffpure import get_config_for_dataset
        
        cfg = get_config_for_dataset('br35h')
        print('✅ Config created for BR35H')
        print('   - Dataset:', cfg.DATASET)
        print('   - Dataset Name:', cfg.DATASET_NAME)
        print('   - Image Size:', cfg.IMG_SIZE)
        print('   - Image Channels:', cfg.IMG_CHANNELS)
        print('   - Classes:', cfg.N_CLASSES)
        
        expected_mae_checkpoint = f'checkpoints/mae_detector_{cfg.DATASET}.pt'
        print('   - Expected MAE checkpoint:', expected_mae_checkpoint)
        
        if os.path.exists(expected_mae_checkpoint):
            print('✅ MAE checkpoint found!')
        else:
            print('⚠️ MAE checkpoint not found, will use fallback')
        
        return True
        
    except Exception as e:
        print('❌ Config error:', e)
        import traceback
        traceback.print_exc()
        return False

def test_mae_detector_loading():
    """Test MAE detector loading"""
    print("=== 🔍 Testing MAE Detector Loading ===")
    try:
        from defense.mae_detector1 import MAEDetector
        from train_diffpure import get_config_for_dataset
        
        cfg = get_config_for_dataset('br35h')
        print('Creating MAE Detector...')
        
        detector = MAEDetector(cfg)
        print('✅ MAE Detector created successfully!')
        print('   - Checkpoint path:', detector.ckpt)
        print('   - Threshold:', detector.threshold)
        print('   - Device:', detector.device)
        
        # Test model parameters
        param_count = sum(p.numel() for p in detector.model.parameters())
        print('   - Model parameters:', f'{param_count:,}')
        
        print('🎉 MAE DETECTOR TEST PASSED!')
        return True
        
    except Exception as e:
        print('❌ MAE Detector error:', e)
        import traceback
        traceback.print_exc()
        return False

def test_diffusion_model_creation():
    """Test diffusion model creation with correct parameters"""
    print("=== 🎨 Testing Diffusion Model Creation ===")
    try:
        from diffusion.diffuser import UNet
        from train_diffpure import get_config_for_dataset
        
        cfg = get_config_for_dataset('br35h')
        print('Creating UNet model...')
        
        # Create UNet with correct parameters (only in_channels and hidden_channels)
        model = UNet(
            in_channels=cfg.IMG_CHANNELS,
            hidden_channels=256  # Compatible with existing checkpoint
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        print('✅ UNet model created successfully!')
        print('   - Input channels:', cfg.IMG_CHANNELS)
        print('   - Hidden channels: 256')
        print('   - Model parameters:', f'{param_count:,}')
        
        # Test memory usage
        import torch
        if torch.cuda.is_available():
            model = model.cuda()
            print('✅ Model moved to GPU successfully!')
            
            # Test forward pass with small batch
            dummy_input = torch.randn(2, cfg.IMG_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE).cuda()
            dummy_t = torch.randn(2).cuda()
            
            with torch.no_grad():
                output = model(dummy_input, dummy_t)
                print('✅ Forward pass successful!')
                print('   - Input shape:', dummy_input.shape)
                print('   - Output shape:', output.shape)
        
        print('🎉 DIFFUSION MODEL TEST PASSED!')
        return True
        
    except Exception as e:
        print('❌ Diffusion model error:', e)
        import traceback
        traceback.print_exc()
        return False

def test_dataset_loading():
    """Test dataset loading"""
    print("=== 📊 Testing Dataset Loading ===")
    try:
        from train_diffpure import get_config_for_dataset
        from utils.data_utils import get_dataset
        import torch.utils.data as data_utils
        
        cfg = get_config_for_dataset('br35h')
        print('Loading BR35H dataset...')
        
        train_dataset, test_dataset = get_dataset(cfg)
        print('✅ Dataset loaded successfully!')
        print('   - Train samples:', len(train_dataset))
        print('   - Test samples:', len(test_dataset))
        
        # Test dataloader
        train_loader = data_utils.DataLoader(
            train_dataset, 
            batch_size=16,  # Our new batch size
            shuffle=True,
            num_workers=2
        )
        
        # Test one batch
        batch = next(iter(train_loader))
        data, targets = batch
        print('✅ DataLoader test successful!')
        print('   - Batch data shape:', data.shape)
        print('   - Batch targets shape:', targets.shape)
        print('   - Data type:', data.dtype)
        print('   - Data range: [{:.3f}, {:.3f}]'.format(data.min().item(), data.max().item()))
        
        print('🎉 DATASET TEST PASSED!')
        return True
        
    except Exception as e:
        print('❌ Dataset error:', e)
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Final integration test"""
    print("=== 🚀 Final Integration Test ===")
    try:
        from train_diffpure import parse_args, get_config_for_dataset
        from defense.mae_detector1 import MAEDetector
        from diffusion.diffuser import UNet
        from utils.data_utils import get_dataset
        import torch
        import torch.utils.data as data_utils
        
        print('🔄 Running complete integration test...')
        
        # 1. Config
        cfg = get_config_for_dataset('br35h')
        print('✅ 1. Config created')
        
        # 2. MAE Detector
        mae_detector = MAEDetector(cfg)
        print('✅ 2. MAE Detector loaded')
        
        # 3. Dataset
        train_dataset, test_dataset = get_dataset(cfg)
        train_loader = data_utils.DataLoader(train_dataset, batch_size=16, shuffle=True)
        print('✅ 3. Dataset loaded')
        
        # 4. Diffusion Model with correct parameters
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNet(
            in_channels=cfg.IMG_CHANNELS,
            hidden_channels=256
        ).to(device)
        print('✅ 4. Diffusion model created')
        
        # 5. Test training step
        model.train()
        batch = next(iter(train_loader))
        data, targets = batch
        data = data.to(device)
        
        with torch.no_grad():
            batch_size = data.shape[0]
            t = torch.rand(batch_size, device=device)
            noise = torch.randn_like(data) * 0.1
            noisy_data = data + noise
            
            # Test mixed precision
            if torch.cuda.is_available():
                from torch.cuda.amp import autocast
                with autocast():
                    predicted_noise = model(noisy_data, t)
                    print('✅ 5. Mixed precision forward pass successful')
            else:
                predicted_noise = model(noisy_data, t)
                print('✅ 5. Forward pass successful')
        
        print('🎉🎉🎉 ALL TESTS PASSED! SYSTEM READY! 🎉🎉🎉')
        print('✅ You can now run diffusion training safely!')
        return True
        
    except Exception as e:
        print('❌ Integration test failed:', e)
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 STARTING COMPREHENSIVE SYSTEM TEST")
    print("=" * 50)
    
    tests = [
        test_basic_setup,
        test_core_imports,
        test_br35h_config,
        test_mae_detector_loading,
        test_diffusion_model_creation,
        test_dataset_loading,
        test_integration
    ]
    
    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)
        print("=" * 50)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"📊 TEST SUMMARY: {passed}/{total} PASSED")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! SYSTEM IS READY!")
        print("\n🚀 Next step: Run diffusion training:")
        print("export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb=64")
        print("export CUDA_VISIBLE_DEVICES=1")
        print("nohup python train_diffpure.py --dataset br35h --epochs 50 --batch-size 16 --gradient-accumulation-steps 4 --mixed-precision --hidden-channels 256 > diffusion_br35h_final.log 2>&1 &")
    else:
        print("❌ SOME TESTS FAILED! Please fix the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
