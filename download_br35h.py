#!/usr/bin/env python3
"""
Automatic BR35H Dataset Downloader
Downloads and sets up BR35H brain tumor dataset
"""

import os
import requests
import zipfile
import shutil
from pathlib import Path

def download_br35h():
    """Download and setup BR35H dataset"""
    print("🏥 Setting up BR35H Medical Dataset...")
    
    # Create directory
    br35h_dir = Path("data/Br35H")
    br35h_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created directory: {br35h_dir}")
    
    # Check if already exists
    if (br35h_dir / "no").exists() and (br35h_dir / "yes").exists():
        print("✅ BR35H dataset already exists!")
        return True
    
    print("📥 BR35H dataset not found. Please download manually:")
    print("1. Go to: https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection")
    print("2. Download the dataset")
    print("3. Extract to: data/Br35H/")
    print("4. Should have folders: data/Br35H/no/ and data/Br35H/yes/")
    
    return False

def test_br35h():
    """Test BR35H dataset loading"""
    try:
        from utils.datasets.br35h import Br35HDataset
        
        if not os.path.exists('data/Br35H'):
            print("❌ BR35H folder not found")
            return False
            
        if not os.path.exists('data/Br35H/no') or not os.path.exists('data/Br35H/yes'):
            print("❌ Missing no/ or yes/ folders")
            return False
        
        dataset = Br35HDataset('data/Br35H')
        print(f"✅ BR35H Dataset Size: {len(dataset)}")
        print(f"✅ Classes: {dataset.classes}")
        
        # Test sample
        img, label = dataset[0]
        print(f"✅ Sample shape: {img.size if hasattr(img, 'size') else 'Processed'}")
        print("✅ BR35H Dataset Working!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing BR35H: {e}")
        return False

def main():
    """Main function"""
    print("=" * 50)
    print("🏥 BR35H Medical Dataset Setup")
    print("=" * 50)
    
    # Try to setup
    if download_br35h():
        # Test if working
        if test_br35h():
            print("\n🎉 BR35H setup complete!")
            print("Ready for experiments: python main.py --dataset br35h --mode debug")
        else:
            print("\n⚠️ BR35H setup incomplete. Please download manually.")
    else:
        print("\n📋 Manual download required.")
        print("After downloading, run: python download_br35h.py")

if __name__ == "__main__":
    main() 