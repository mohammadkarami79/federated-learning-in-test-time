#!/usr/bin/env python3
"""Quick test for MobileNetV2 BatchNorm fix"""

import torch
import torchvision.models as models

def test_mobilenet():
    print("Testing MobileNetV2...")
    
    try:
        # Create model
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2, inplace=False),
            torch.nn.Linear(in_features, 10)
        )
        
        print(f"✅ Model created, classifier input: {in_features}")
        
        # Test with batch_size=1 (problematic case)
        model.eval()
        test_input = torch.randn(1, 3, 32, 32)
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ Batch size 1 test passed: {output.shape}")
        
        # Test with batch_size=2 
        test_input = torch.randn(2, 3, 32, 32)
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ Batch size 2 test passed: {output.shape}")
        
        # Test gradient computation
        test_input.requires_grad_(True)
        output = model(test_input)
        loss = output.sum()
        loss.backward()
        
        print("✅ Gradient computation test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_mobilenet()
    if success:
        print("\n🎯 MobileNetV2 is working correctly!")
    else:
        print("\n⚠️  MobileNetV2 has issues")
