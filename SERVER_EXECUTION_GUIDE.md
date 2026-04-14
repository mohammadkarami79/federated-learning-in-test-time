# راهنمای اجرای Kim et al., 2023 در سرور

## 🎯 تنظیمات دقیق مطابق مقاله

### پارامترهای حمله (EXACT MATCH):
- **نوع حمله:** L2-norm PGD
- **ε (Epsilon):** 4.5
- **α (Alpha):** 0.01  
- **K (Steps):** 10
- **معماری:** MobileNetV2
- **کلاینت‌ها:** 40
- **توزیع:** Non-IID (β=0.4)

## 📋 مراحل اجرا در سرور

### مرحله 1: آماده‌سازی محیط
```bash
# رفتن به دایرکتوری پروژه
cd ~/FLBrain/federeated_learning_in_test_time/federated-learning-in-test-time

# فعال‌سازی محیط مجازی
conda activate br35h_env

# بررسی فضای دیسک
df -h .

# بررسی حافظه
free -h

# بررسی GPU
nvidia-smi
```

### مرحله 2: بررسی فایل‌های مورد نیاز
```bash
# بررسی وجود مدل‌های پیش‌آموزش داده شده
ls -la checkpoints/diffuser_cifar10.pt
ls -la checkpoints/mae_detector_cifar10.pt

# بررسی فایل‌های جدید
ls -la main_kim2023_reproduction.py
ls -la run_kim2023_reproduction.py
ls -la KIM2023_REPRODUCTION_GUIDE.md
```

### مرحله 3: تست اولیه (اختیاری)
```bash
# تست سریع برای بررسی خطاها
python main_kim2023_reproduction.py --help

# تست import کردن کتابخانه‌ها
python -c "
import torch
import torchvision.models as models
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
model = models.mobilenet_v2(weights=None)
print('MobileNetV2 created successfully')
"
```

### مرحله 4: اجرای آزمایش اصلی

#### روش 1: اجرای ساده (پیشنهادی)
```bash
# اجرا با runner script
nohup python run_kim2023_reproduction.py --background --log-suffix "_kim2023_final" > kim2023_runner.log 2>&1 &

# نمایش PID برای مانیتورینگ
echo $! > kim2023_pid.txt
cat kim2023_pid.txt
```

#### روش 2: اجرای مستقیم
```bash
# ایجاد نام یکتا برای نتایج
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./kim2023_results_${TIMESTAMP}"

# اجرای آزمایش
nohup python main_kim2023_reproduction.py --output-dir "$OUTPUT_DIR" > kim2023_main_${TIMESTAMP}.log 2>&1 &

# ذخیره PID
echo $! > kim2023_pid.txt
```

### مرحله 5: مانیتورینگ

#### نظارت بر لاگ‌ها
```bash
# مشاهده آخرین لاگ
tail -f kim2023_*.log

# جستجوی نتایج دور به دور
grep -E "Round.*Clean Acc|Adv Acc" kim2023_*.log

# بررسی پیشرفت آزمایش
grep -E "Starting|Round [0-9]+/[0-9]+" kim2023_*.log | tail -5
```

#### بررسی منابع سیستم
```bash
# نظارت بر پردازنده و حافظه
htop

# نظارت بر GPU
watch -n 5 nvidia-smi

# بررسی فضای دیسک
watch -n 30 'df -h . && ls -lah kim2023_results_*/'
```

#### کنترل فرآیند
```bash
# بررسی وضعیت فرآیند
PID=$(cat kim2023_pid.txt)
ps -p $PID -o pid,ppid,cmd,%cpu,%mem,etime

# توقف فرآیند (در صورت نیاز)
kill $PID

# توقف اجباری (فقط در موارد اضطراری)
kill -9 $PID
```

### مرحله 6: بررسی نتایج

#### مشاهده نتایج نهایی
```bash
# یافتن دایرکتوری نتایج
ls -la kim2023_results_*/

# مشاهده نتایج نهایی
cat kim2023_results_*/final_results.json | jq '.final_metrics'

# مقایسه با log8.txt
echo "=== Kim2023 Results ==="
cat kim2023_results_*/final_results.json | jq '.final_metrics'
echo "=== log8.txt Results ==="
echo "Clean Accuracy: 88.11%"
echo "Adversarial Accuracy: 77.98%"
```

#### تحلیل دقیق‌تر
```bash
# نمایش پیشرفت دور به دور
cat kim2023_results_*/final_results.json | jq '.round_by_round[] | {round: .round, clean_acc: .clean_accuracy, adv_acc: .adversarial_accuracy}'

# اطلاعات کلی آزمایش
cat kim2023_results_*/final_results.json | jq '.experiment_info'

# بررسی تنظیمات
cat kim2023_results_*/config.json | jq '{architecture: .ARCHITECTURE, attack_norm: .ATTACK_NORM, epsilon: .ATTACK_EPSILON, alpha: .ATTACK_ALPHA, clients: .NUM_CLIENTS}'
```

### مرحله 7: تمیزکاری (اختیاری)

#### حذف فایل‌های موقت
```bash
# حذف لاگ‌های قدیمی (بعد از تأیید نتایج)
# rm kim2023_*.log

# فشرده‌سازی نتایج
# tar -czf kim2023_results_$(date +%Y%m%d).tar.gz kim2023_results_*/
```

## ⚠️ نکات مهم

### پارامترهای کلیدی تفاوت:
```bash
# Kim et al., 2023 vs log8.txt
echo "Parameter | Kim2023 | log8.txt"
echo "---------|---------|----------"
echo "Architecture | MobileNetV2 | ResNet18"
echo "Attack Norm | L2 | L∞"
echo "Epsilon | 4.5 | 0.031"
echo "Alpha | 0.01 | 0.007"
echo "Clients | 40 | 10"
echo "Data Split | Non-IID (β=0.4) | IID"
```

### زمان‌بندی انتظاری:
- **هر دور:** ~15-20 دقیقه (40 کلاینت)
- **کل آزمایش:** ~4-5 ساعت (15 دور)
- **4x بیشتر از log8.txt** به دلیل 40 کلاینت

### نتایج مورد انتظار:
- **Clean Accuracy:** 85-90%
- **L2-Adversarial Accuracy:** ~48% (بهتر از L∞)
- **Detection Rate:** متغیر

## 🚨 عیب‌یابی

### مشکلات رایج:
```bash
# خطای حافظه GPU
# کاهش batch size در config

# خطای import
# بررسی محیط مجازی و کتابخانه‌ها

# خطای checkpoint
# بررسی وجود مدل‌های پیش‌آموزش داده شده
```

### تماس‌های اضطراری:
```bash
# توقف همه فرآیندهای python
# pkill -f "main_kim2023_reproduction.py"

# پاک کردن cache GPU
# python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
```

---

**✅ آماده اجرا!** 

همه چیز برای بازتولید دقیق Kim et al., 2023 آماده است.
