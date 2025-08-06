# 🏥 راهنمای Setup BR35H برای Windows

## 📋 **وضعیت فعلی**
- ❌ Kaggle installed ولی PATH issue دارد  
- ❌ Dataset هنوز دانلود نشده
- ❌ Windows commands نیاز به تنظیم

---

## 🔧 **راه‌حل‌های Windows**

### **روش 1: دانلود دستی (ساده‌ترین)**

1. **برو به Kaggle:**
   - https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection
   - Login کن
   - Download کن

2. **Extract کن:**
   ```cmd
   # Extract downloaded file به data\Br35H\
   # باید folders 'no' و 'yes' داشته باشید
   ```

### **روش 2: PowerShell با Kaggle API**

```powershell
# 1. Kaggle API setup
$kagglePath = "$env:USERPROFILE\.kaggle"
New-Item -ItemType Directory -Force -Path $kagglePath

# 2. Download dataset
python -m kaggle datasets download -d ahmedhamada0/brain-tumor-detection

# 3. Extract
Expand-Archive brain-tumor-detection.zip -DestinationPath data\Br35H\

# 4. Cleanup
Remove-Item brain-tumor-detection.zip
```

### **روش 3: Manual Windows Commands**

```cmd
# Create directory
mkdir data\Br35H

# Download using Python
python -c "
import requests
import zipfile
import os

# Manual download and extract
print('Setting up BR35H manually...')
os.makedirs('data/Br35H', exist_ok=True)
print('Directory created: data/Br35H')
"
```

---

## 📂 **Structure مورد نیاز**

```
data\Br35H\
├── no\              # تصاویر بدون تومور (~1500 تصویر)
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── yes\             # تصاویر با تومور (~1500 تصویر)
    ├── Y1.jpg
    ├── Y2.jpg
    └── ...
```

---

## 🔍 **تست Setup**

### **تست اولیه:**
```cmd
# بررسی structure
dir data\Br35H
dir data\Br35H\no
dir data\Br35H\yes
```

### **تست Loading:**
```cmd
python -c "
from utils.datasets.br35h import Br35HDataset
import os
if os.path.exists('data/Br35H'):
    if os.path.exists('data/Br35H/no') and os.path.exists('data/Br35H/yes'):
        dataset = Br35HDataset('data/Br35H')
        print(f'BR35H Dataset Size: {len(dataset)}')
        print(f'Classes: {dataset.classes}')
        print('✅ BR35H Dataset Working!')
    else:
        print('❌ Missing no/ or yes/ folders')
else:
    print('❌ BR35H folder not found')
"
```

### **تست Complete Pipeline:**
```cmd
python main.py --dataset br35h --mode debug
```

---

## 🚀 **Alternative: ادامه بدون BR35H**

اگر BR35H setup مشکل داشت، می‌تونید با 3 dataset دیگه ادامه بدید:

```cmd
# تست سیستم با 3 datasets موجود
python main.py --dataset cifar10 --mode debug
python main.py --dataset cifar100 --mode debug  
python main.py --dataset mnist --mode debug

# Full experiments
python main.py --dataset cifar10 --mode full
python main.py --dataset cifar100 --mode full
python main.py --dataset mnist --mode full
```

### **نتایج Paper با 3 Datasets:**
- **CIFAR-10**: Computer vision benchmark
- **CIFAR-100**: Complex classification  
- **MNIST**: Simple vision task
- **Multi-domain evaluation**: Still strong for paper

---

## 📊 **تصمیم نهایی**

### **Option A: Setup BR35H (اگر وقت دارید)**
- Medical AI application
- 4 datasets complete
- Stronger paper

### **Option B: ادامه بدون BR35H (اگر عجله دارید)**  
- 3 datasets قوی
- Paper ready الان
- مدت زمان کمتر

---

## 🎯 **پیشنهاد**

**بذارید ابتدا با 3 datasets موجود شروع کنیم و نتایج بگیریم. بعداً اگر وقت داشتید BR35H اضافه کنید.**

```cmd
# شروع experiments
python setup_system.py
python main.py --dataset cifar10 --mode debug
```

**آماده برای شروع experiments! 🚀**