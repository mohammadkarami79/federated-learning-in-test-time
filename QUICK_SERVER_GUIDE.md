# راهنمای سریع اجرا در سرور - Kim et al., 2023

## ✅ **مشکل BatchNorm برطرف شد!**

مشکل `Expected more than 1 value per channel when training` برطرف شده است.

## 🚀 **قدم‌های اجرا:**

### **1. بررسی نهایی سیستم**
```bash
# در سرور اجرا کنید:
cd ~/FLBrain/federeated_learning_in_test_time/federated-learning-in-test-time
conda activate br35h_env

# تست سریع
python quick_test_mobilenet.py

# بررسی کامل سیستم
python check_system_kim2023.py
```

### **2. اجرای آزمایش Kim et al., 2023**
```bash
# روش 1: اجرای پس‌زمینه (پیشنهادی)
nohup python run_kim2023_reproduction.py --background > kim2023_runner.log 2>&1 &

# دریافت PID برای کنترل
echo $! > kim2023_pid.txt
echo "Process ID: $(cat kim2023_pid.txt)"
```

### **3. مانیتورینگ**
```bash
# نظارت بر لاگ اصلی
tail -f kim2023_reproduction_full_*.log

# بررسی پیشرفت
grep -E "Round.*Clean Acc|Adv Acc" kim2023_reproduction_full_*.log

# نظارت بر منابع سیستم
watch -n 10 nvidia-smi
```

### **4. کنترل فرآیند**
```bash
# بررسی وضعیت
ps -p $(cat kim2023_pid.txt) -o pid,cmd,%cpu,%mem,etime

# توقف در صورت نیاز
kill $(cat kim2023_pid.txt)
```

### **5. مشاهده نتایج**
```bash
# پس از اتمام (4-5 ساعت)
find . -name "final_results.json" -newer kim2023_pid.txt -exec cat {} \; | jq '.final_metrics'

# مقایسه با log8.txt
echo "Kim2023 vs log8.txt:"
echo "Architecture: MobileNetV2 vs ResNet18"
echo "Attack: L2-PGD (ε=4.5) vs L∞-PGD (ε=0.031)"
echo "Clients: 40 vs 10"
```

---

## 🎯 **تفاوت‌های کلیدی Kim et al., 2023:**

### **Attack Parameters:**
- **Norm:** L2 (به جای L∞)
- **Epsilon:** 4.5 (به جای 0.031)
- **Alpha:** 0.01 (به جای 0.007)
- **Steps:** 10 (مشابه log8.txt)

### **Model & Data:**
- **Architecture:** MobileNetV2 (به جای ResNet18)
- **Clients:** 40 (به جای 10)
- **Data Distribution:** Non-IID β=0.4 (به جای IID)

### **Runtime Expectations:**
- **Duration:** ~4-5 hours (4x بیشتر از log8.txt)
- **Memory:** ~2-3GB GPU per client
- **Target Results:** 
  - Clean Acc: 85-90%
  - L2-Adv Acc: ~48% (بهتر از 20% baseline)

---

## ⚡ **دستورات یک‌خطی برای سرور:**

```bash
# اجرای کامل
cd ~/FLBrain/federeated_learning_in_test_time/federated-learning-in-test-time && conda activate br35h_env && nohup python run_kim2023_reproduction.py --background > kim2023_runner.log 2>&1 & echo $! > kim2023_pid.txt && echo "Started Kim2023 experiment, PID: $(cat kim2023_pid.txt)"

# مانیتورینگ
tail -f kim2023_reproduction_full_*.log

# بررسی وضعیت
ps -p $(cat kim2023_pid.txt) && echo "Experiment running..." || echo "Experiment finished"
```

**✅ همه چیز آماده است! اکنون می‌توانید آزمایش Kim et al., 2023 را اجرا کنید.**
