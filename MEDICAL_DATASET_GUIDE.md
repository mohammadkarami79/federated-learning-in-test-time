# 🏥 راهنمای کامل Medical Dataset (BR35H)

## 📋 **وضعیت فعلی BR35H Dataset**

### ✅ **موجود در کد:**
- کلاس `Br35HDataset` در `utils/datasets/br35h.py`
- Integration کامل در `utils/data_utils.py`
- پشتیبانی در `main.py` و `train_diffpure.py`

### ❌ **مفقود:**
- فایل‌های actual dataset در `data/Br35H/`

---

## 🔍 **چرا در data folder نیست؟**

احتمالاً به این دلایل:
1. **GitHub Limitations**: فایل‌های بزرگ معمولاً در GitHub نیستند
2. **License Issues**: ممکنه dataset کپی‌رایت داشته باشه
3. **Manual Download Required**: نیاز به دانلود دستی

---

## 📥 **نحوه دریافت BR35H Dataset**

### **روش 1: دانلود از Kaggle (توصیه شده)**
```bash
# 1. Install kaggle
pip install kaggle

# 2. Setup Kaggle API (نیاز به API key از kaggle.com)
mkdir ~/.kaggle
# Copy your kaggle.json to ~/.kaggle/

# 3. Download BR35H dataset
kaggle datasets download -d ahmedhamada0/brain-tumor-detection

# 4. Extract to correct location
unzip brain-tumor-detection.zip -d data/Br35H/
```

### **روش 2: از zip file شما**
اگر zip file شما حاوی BR35H هست:
```bash
# Extract BR35H من zip file
mkdir -p data/Br35H
# Copy folders 'no' and 'yes' from your zip to data/Br35H/
```

### **روش 3: دانلود دستی**
1. برو به: https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection
2. دانلود کن
3. Extract کن به `data/Br35H/`

---

## 📂 **Structure صحیح BR35H:**

```
data/Br35H/
├── no/              # 1500+ images - بدون تومور مغزی  
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── ...
│   └── N.jpg
└── yes/             # 1500+ images - با تومور مغزی
    ├── Y1.jpg
    ├── Y2.jpg
    ├── ...
    └── YN.jpg
```

---

## 🔧 **تست BR35H Integration**

### **تست اولیه:**
```bash
# Check if structure is correct
ls -la data/Br35H/
ls -la data/Br35H/no/ | wc -l
ls -la data/Br35H/yes/ | wc -l
```

### **تست Loading:**
```python
# Test dataset loading
python -c "
from utils.datasets.br35h import Br35HDataset
import os
if os.path.exists('data/Br35H'):
    dataset = Br35HDataset('data/Br35H')
    print(f'BR35H Dataset Size: {len(dataset)}')
    print(f'Classes: {dataset.classes}')
    if len(dataset) > 0:
        img, label = dataset[0]
        print(f'Sample shape: {img.size if hasattr(img, \"size\") else \"Processed\"}')
        print('✅ BR35H Dataset Working!')
    else:
        print('❌ Dataset Empty')
else:
    print('❌ BR35H folder not found')
"
```

### **تست Complete Pipeline:**
```bash
# Test full training with BR35H
python main.py --dataset br35h --mode debug
```

---

## 🎯 **مشخصات BR35H Dataset**

### **Technical Specs:**
- **Classes**: 2 (no tumor, yes tumor)
- **Images**: ~3000 total
- **Format**: JPG
- **Size**: Various (224x224 recommended)
- **Channels**: 3 (RGB)
- **Type**: Medical brain scans

### **Configuration در کد:**
```python
# From utils/datasets/br35h.py
BR35H_INFO = {
    'num_classes': 2,
    'input_channels': 3, 
    'input_size': 224,
    'mean': (0.485, 0.456, 0.406),  # ImageNet means
    'std': (0.229, 0.224, 0.225)    # ImageNet stds
}
```

---

## 🚀 **نحوه استفاده در Paper**

### **1. Training Command:**
```bash
# Debug mode (quick test)
python main.py --dataset br35h --mode debug

# Full training for paper
python main.py --dataset br35h --mode full --epochs 25
```

### **2. Expected Results:**
- **Clean Accuracy**: 85-95% (medical data معمولاً accuracy بالاتری داره)
- **Adversarial Robustness**: 60-80%
- **Training Time**: 15-30 minutes

### **3. Medical-Specific Features:**
کد ما شامل medical image optimizations هست:
- **Rician Noise**: مخصوص medical images
- **Anatomical Constraints**: برای بهتر شدن training
- **Larger Input Size**: 224x224 به جای 32x32

---

## ⚠️ **نکات مهم**

### **1. Privacy & Ethics:**
- BR35H یک public dataset هست
- برای research استفاده مجاز
- در paper ذکر کنید که از public medical data استفاده کردید

### **2. Performance Expectations:**
- Medical data معمولاً بهتر train می‌شه
- Adversarial attacks روی medical data متفاوت عمل می‌کنن
- نتایج ممکنه از computer vision datasets متفاوت باشه

### **3. Paper Benefits:**
- **Multi-domain Evaluation**: نشون می‌ده روش شما روی medical data هم کار می‌کنه
- **Real-world Application**: medical imaging یک application مهم federated learning هست
- **Robustness Testing**: medical data برای test کردن robustness عالیه

---

## 🎉 **خلاصه**

**وضعیت فعلی:** کد آماده، فقط dataset نیاز داره  
**برای Paper:** می‌تونه نتایج عالی بده  
**زمان Setup**: 10-15 دقیقه  
**تست**: `python main.py --dataset br35h --mode debug`

**بعد از setup، شما 4 dataset کامل خواهید داشت که برای paper عالیه!**