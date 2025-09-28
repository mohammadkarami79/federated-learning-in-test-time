#!/usr/bin/env python3
"""
تحلیل نتایج BR35H و شناسایی مشکلات دفاع انتخابی
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_br35h_results():
    """تحلیل نتایج BR35H"""
    
    # نتایج فعلی
    results = {
        "clean_accuracy": 98.0,
        "adversarial_accuracy": 40.33,
        "mae_detection_rate": 69.67,
        "mae_threshold": 0.22,
        "recon_errors_range": [0.1591, 0.2509],
        "diffpure_steps": 3,
        "diffpure_sigma": 0.25
    }
    
    print("🔍 تحلیل نتایج BR35H:")
    print("=" * 50)
    
    # 1. تحلیل گپ دقت
    gap = results["clean_accuracy"] - results["adversarial_accuracy"]
    print(f"📊 گپ دقت: {gap:.2f}%")
    print(f"   - Clean Accuracy: {results['clean_accuracy']:.2f}%")
    print(f"   - Adversarial Accuracy: {results['adversarial_accuracy']:.2f}%")
    print(f"   - مشکل: گپ بیش از 50% نشان‌دهنده عدم کارایی دفاع است")
    
    # 2. تحلیل MAE Detection
    print(f"\n🎯 MAE Detection Analysis:")
    print(f"   - Detection Rate: {results['mae_detection_rate']:.2f}%")
    print(f"   - Threshold: {results['mae_threshold']:.3f}")
    print(f"   - Recon Errors Range: {results['recon_errors_range']}")
    print(f"   - مشکل: Detection rate خیلی بالا، threshold نامناسب")
    
    # 3. تحلیل DiffPure
    print(f"\n🛡️ DiffPure Analysis:")
    print(f"   - Steps: {results['diffpure_steps']}")
    print(f"   - Sigma: {results['diffpure_sigma']}")
    print(f"   - مشکل: Steps کم، Sigma کم - قدرت پاک‌سازی ناکافی")
    
    # 4. پیشنهادات بهبود
    print(f"\n💡 پیشنهادات بهبود:")
    print("=" * 30)
    
    print("1. تنظیم MAE Threshold:")
    print("   - کاهش threshold از 0.22 به 0.15-0.18")
    print("   - استفاده از adaptive threshold با target rate 15-20%")
    
    print("\n2. بهبود DiffPure:")
    print("   - افزایش steps از 3 به 5-8")
    print("   - افزایش sigma از 0.25 به 0.5-0.7")
    print("   - استفاده از scheduler برای steps")
    
    print("\n3. تنظیم Hyperparameters:")
    print("   - کاهش learning rate برای جلوگیری از overfitting")
    print("   - اضافه کردن weight decay")
    print("   - کاهش client epochs")
    
    print("\n4. بهبود MAE Model:")
    print("   - استفاده از MAE با architecture بهتر")
    print("   - تنظیم patch size برای 224x224")
    print("   - بهبود training strategy")
    
    # 5. مقایسه با CIFAR-10
    print(f"\n📈 مقایسه با CIFAR-10 (log7.txt):")
    print("   - CIFAR-10: Clean 87.7%, Adv 72.12%, MAE 15.63%")
    print("   - BR35H: Clean 98.0%, Adv 40.33%, MAE 69.67%")
    print("   - مشکل: BR35H MAE rate خیلی بالا، Adv accuracy خیلی پایین")
    
    return results

def suggest_config_fixes():
    """پیشنهاد تنظیمات جدید"""
    
    print("\n🔧 تنظیمات پیشنهادی برای config_br35h_selective_defense.py:")
    print("=" * 60)
    
    config_fixes = {
        "MAE_THRESHOLD": 0.15,  # کاهش از 0.22
        "ADAPTIVE_THRESHOLD": True,
        "TARGET_DETECTION_RATE": 18.0,  # کاهش از 69.67%
        
        "DIFFUSER_STEPS": 6,  # افزایش از 3
        "DIFFUSER_SIGMA": 0.6,  # افزایش از 0.25
        
        "LEARNING_RATE": 0.001,  # کاهش از 0.005
        "WEIGHT_DECAY": 1e-4,  # اضافه کردن
        "CLIENT_EPOCHS": 10,  # کاهش از 15
        
        "MAE_PATCH_SIZE": 16,  # مناسب برای 224x224
        "MAE_EMBED_DIM": 256,
        "MAE_DECODER_EMBED_DIM": 256,
        "MAE_NUM_HEADS": 8,
        "MAE_DEPTH": 6
    }
    
    for key, value in config_fixes.items():
        print(f"   {key} = {value}")
    
    return config_fixes

def create_improved_config():
    """ایجاد config بهبود یافته"""
    
    config_content = '''# BR35H Selective Defense Configuration - Improved
# Based on analysis of current results

# Dataset Configuration
DATASET = 'br35h'
IMG_SIZE = 224
NUM_CLASSES = 2
MODEL_NAME = 'resnet34'

# Training Configuration
NUM_ROUNDS = 15
CLIENT_EPOCHS = 10  # Reduced to prevent overfitting
BATCH_SIZE = 32
LEARNING_RATE = 0.001  # Reduced learning rate
WEIGHT_DECAY = 1e-4  # Added weight decay
MOMENTUM = 0.9

# Attack Configuration
ATTACK_EPSILON = 0.031
ATTACK_STEPS = 10
ATTACK_ALPHA = 0.007

# MAE Detector Configuration
MAE_THRESHOLD = 0.15  # Reduced threshold
ADAPTIVE_THRESHOLD = True
TARGET_DETECTION_RATE = 18.0  # Reduced target rate
MAE_PATCH_SIZE = 16
MAE_DEPTH = 6
MAE_NUM_HEADS = 8
MAE_EMBED_DIM = 256
MAE_DECODER_EMBED_DIM = 256

# DiffPure Configuration
DIFFUSER_STEPS = 6  # Increased steps
DIFFUSER_SIGMA = 0.6  # Increased sigma
EVAL_BATCH_SIZE = 32

# Other Configuration
NUM_CLIENTS = 5
DATA_DISTRIBUTION = 'iid'
DEVICE = 'cuda'
'''
    
    with open('config_br35h_improved.py', 'w') as f:
        f.write(config_content)
    
    print(f"\n✅ Config بهبود یافته ذخیره شد: config_br35h_improved.py")

if __name__ == "__main__":
    results = analyze_br35h_results()
    config_fixes = suggest_config_fixes()
    create_improved_config()
    
    print(f"\n🎯 خلاصه مشکلات:")
    print("1. MAE threshold خیلی بالا (0.22 → 0.15)")
    print("2. DiffPure ناکافی (3 steps → 6 steps)")
    print("3. Overfitting شدید (loss=0.0000)")
    print("4. Detection rate خیلی بالا (69.67% → 18%)")
    
    print(f"\n🚀 مراحل بعدی:")
    print("1. اعمال تنظیمات جدید")
    print("2. Retrain با config بهبود یافته")
    print("3. تست روی subset کوچک")
    print("4. مقایسه نتایج")
