#!/usr/bin/env python3
"""
Integration test for the complete training pipeline
"""

import sys
import torch
import subprocess
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_training_pipeline():
    """Test a mini version of the complete training pipeline"""
    print("🚀 Starting Integration Test...")
    print("=" * 50)
    
    # Test with debug mode for speed
    cmd = [
        'python', 'main.py',
        '--dataset', 'cifar10',
        '--mode', 'debug',
        '--skip-setup'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("Expected runtime: ~2-3 minutes for debug mode")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        # Run with timeout of 5 minutes
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
            cwd=str(project_root)
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n⏱️ Test completed in {duration:.1f} seconds")
        print(f"Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("🎉 Integration test PASSED!")
            
            # Look for key metrics in output
            output = result.stdout
            if "Clean Acc" in output:
                print("✅ Found accuracy metrics")
            if "MAE Detection" in output:
                print("✅ Found MAE detection metrics")
            if "Training completed" in output:
                print("✅ Training completed successfully")
                
            # Show last few lines of output
            lines = output.strip().split('\n')
            print("\n📊 Final output:")
            for line in lines[-10:]:
                if line.strip():
                    print(f"   {line}")
            
            return True
        else:
            print("❌ Integration test FAILED!")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Integration test TIMED OUT (5 minutes)")
        print("This might be normal for full training, but debug mode should be faster")
        return False
    except Exception as e:
        print(f"❌ Integration test ERROR: {e}")
        return False

def check_requirements():
    """Check if system has basic requirements"""
    print("🔍 Checking requirements...")
    
    # Check CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ CUDA available: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️ CUDA not available, will use CPU")
    
    # Check dataset
    data_dir = Path("data")
    if data_dir.exists():
        print(f"✅ Data directory exists: {data_dir}")
    else:
        print(f"⚠️ Data directory not found: {data_dir}")
    
    return True

def main():
    """Run integration tests"""
    print("🧪 Federated Learning Integration Test")
    print("=" * 60)
    
    # Basic checks
    if not check_requirements():
        print("❌ Requirements check failed")
        return 1
    
    # Main integration test
    if test_training_pipeline():
        print("\n🎉 All integration tests PASSED!")
        print("✅ System is ready for full training")
        return 0
    else:
        print("\n❌ Integration tests FAILED!")
        print("⚠️ Check the errors above and fix issues")
        return 1

if __name__ == "__main__":
    exit(main())
