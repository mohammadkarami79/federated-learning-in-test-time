# 🚨 راهنمای نهایی اپدیت سرور - حل مشکل Dataset کوچک

## ⚠️ **مشکل اصلی شناسایی شد:**
```
Dataset: 94 samples ÷ 15 clients = 6.3 samples per client
نتیجه: Severe Overfitting + MAE Detection 100% + Clean Acc 50%
```

## 🎯 **راه حل: بهینه سازی برای Dataset کوچک**

---

## 📋 **مراحل اپدیت سرور:**

### **مرحله 1: توقف پروسه فعلی**
```bash
pkill -f "python main.py"
pkill -f "main_PUBLICATION"

# بررسی که هیچ پروسه‌ای نمانده
ps aux | grep python
```

### **مرحله 2: بکاپ فایل‌های فعلی**
```bash
cd ~/FLBrain/federeated_learning_in_test_time/federated-learning-in-test-time

# بکاپ
cp config_fixed.py config_fixed.py.backup
cp defense/mae_detector1.py defense/mae_detector1.py.backup
```

### **مرحله 3: ویرایش config_fixed.py**
```bash
nano config_fixed.py
```

**پیدا کن و تغییر بده:**

**تغییر 1: Training Parameters (حدود خط 236-241)**
```python
# پیدا کن:
    # Training parameters - SMALL DATASET OPTIMIZED
    cfg.BATCH_SIZE = 16        # SMALL for tiny dataset (94 samples)
    cfg.LEARNING_RATE = 0.01   # HIGHER for small dataset
    cfg.WEIGHT_DECAY = 1e-4    # LOWER for small dataset
    cfg.NUM_EPOCHS = 20        # MORE epochs for small dataset
    cfg.CLIENT_EPOCHS = 5      # MORE epochs per client

# تغییر بده به:
    # Training parameters - TINY DATASET OPTIMIZED (94 samples!)
    cfg.BATCH_SIZE = 8         # VERY SMALL for 94 samples
    cfg.LEARNING_RATE = 0.005  # MODERATE for small dataset
    cfg.WEIGHT_DECAY = 1e-5    # VERY LOW for small dataset
    cfg.NUM_EPOCHS = 30        # MANY epochs for small dataset
    cfg.CLIENT_EPOCHS = 8      # MANY epochs per client
```

**تغییر 2: Federated Parameters (حدود خط 254-255)**
```python
# پیدا کن:
    cfg.NUM_CLIENTS = 5  # FEWER clients for small dataset
    cfg.NUM_ROUNDS = 15  # MORE rounds for small dataset

# تغییر بده به:
    cfg.NUM_CLIENTS = 3   # VERY FEW clients (94÷3=31 samples per client)
    cfg.NUM_ROUNDS = 25   # MANY rounds for convergence
```

### **مرحله 4: ویرایش defense/mae_detector1.py**
```bash
nano defense/mae_detector1.py
```

**پیدا کن و تغییر بده (حدود خط 357-359):**
```python
# پیدا کن:
        # CRITICAL FIX: Use proper threshold calculation
        # For small datasets, use mean + std approach
        self.threshold = mean_err + 0.5 * std_err

# تغییر بده به:
        # CRITICAL FIX: Proper threshold for small datasets
        # Use 75th percentile for better balance
        self.threshold = torch.quantile(errs, 0.75).item()
        
        # Fallback if threshold is too extreme
        if self.threshold < mean_err:
            self.threshold = mean_err + 0.3 * std_err
```

### **مرحله 5: پاک کردن Checkpoints قدیمی**
```bash
# پاک کردن MAE checkpoints (حتماً!)
rm -f checkpoints/mae_detector_*.pt
rm -f checkpoints/mae_detector.pt

# لیست فایل‌های باقی‌مانده
ls -la checkpoints/
```

### **مرحله 6: شروع Training جدید**
```bash
# شروع training با نام جدید
nohup python main.py --dataset br35h --mode full --skip-setup > main_TINY_DATASET_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# گرفتن PID
echo "Training started with PID: $!"

# نظارت بر لاگ
tail -f main_TINY_DATASET_*.log
```

---

## 📊 **نتایج مورد انتظار:**

### **با تنظیمات جدید:**
```
Dataset: 94 samples ÷ 3 clients = 31 samples per client ✅
MAE Threshold: 0.4-0.7 (منطقی) ✅
MAE Detection: 20-50% (نه 100%!) ✅
Clean Accuracy: 70-85% (نه 50%!) ✅
```

### **علائم موفقیت در لاگ:**
```
Settings: 25 rounds, 8 epochs, 3 clients ✅
MAE Debug - Threshold: 0.45-0.65 ✅
MAE Debug - Detection sum: 8-16/32 ✅
Clean Acc: 75%+ ✅
Training: Stable progression ✅
```

---

## 🔍 **نظارت و بررسی:**

### **چیزهایی که باید ببینی:**
1. **Clients: 3** (نه 15!)
2. **Rounds: 25** (نه 8!)
3. **MAE Threshold: 0.4-0.7** (نه 0.001!)
4. **Clean Acc: بالای 70%** (نه 50%!)

### **اگر هنوز مشکل داری:**
```bash
# چک کردن تغییرات
grep -n "NUM_CLIENTS = 3" config_fixed.py
grep -n "BATCH_SIZE = 8" config_fixed.py
grep -n "quantile.*0.75" defense/mae_detector1.py

# باید خط‌های مربوطه رو نشون بده
```

---

## 🚨 **نکات مهم:**

### **حتماً انجام بده:**
- ✅ NUM_CLIENTS = 3 (خیلی مهم!)
- ✅ BATCH_SIZE = 8 (خیلی مهم!)
- ✅ پاک کردن mae_detector checkpoints
- ✅ نظارت بر threshold جدید

### **علائم خطر:**
- ❌ اگر هنوز MAE Detection = 100%
- ❌ اگر هنوز Clean Acc < 60%
- ❌ اگر Threshold < 0.1

---

## 📞 **تست سریع:**
```bash
# بعد از 2 round، باید ببینی:
tail -n 50 main_TINY_DATASET_*.log | grep -E "(Clean Acc|MAE Debug|Detection sum)"

# باید نتایج بهتری نشون بده
```

**این بار باید کار کنه!** 🎯

**مشکل اصلی: Dataset کوچک + Client زیاد = Overfitting**
**راه حل: Client کم + Epoch زیاد + Threshold منطقی** ✅
