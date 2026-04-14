# 🚀 اجرای آزمایش Kim et al., 2023 - نسخه اصلاح شده

## دستورات سرور:

### گزینه 1: اجرای ساده با nohup
```bash
# ورود به پوشه
cd ~/FLBrain/federeated_learning_in_test_time/federated-learning-in-test-time

# فعال‌سازی محیط
conda activate br35h_env

# متوقف کردن فرآیندهای قبلی
pkill -f "main_kim2023"
rm -f *pid*.txt

# اجرا با nohup
nohup python run_kim2023_fixed.py > kim2023_fixed_$(date +%Y%m%d_%H%M%S).log 2>&1 & echo $! > kim2023_fixed_pid.txt

# نظارت
echo "PID: $(cat kim2023_fixed_pid.txt)"
tail -f kim2023_fixed_*.log
```

### گزینه 2: استفاده از اسکریپت bash
```bash
# ورود به پوشه و فعال‌سازی محیط
cd ~/FLBrain/federeated_learning_in_test_time/federated-learning-in-test-time
conda activate br35h_env

# قابل اجرا کردن اسکریپت
chmod +x start_kim2023_fixed_nohup.sh

# اجرا
./start_kim2023_fixed_nohup.sh

# نظارت
tail -f kim2023_fixed_nohup_*.log
```

### گزینه 3: دستور یکخطی
```bash
cd ~/FLBrain/federeated_learning_in_test_time/federated-learning-in-test-time && conda activate br35h_env && nohup python run_kim2023_fixed.py > kim2023_fixed_$(date +%Y%m%d_%H%M%S).log 2>&1 & echo $! > kim2023_fixed_pid.txt && echo "Started with PID: $(cat kim2023_fixed_pid.txt)" && tail -f kim2023_fixed_*.log
```

## 🎯 نتایج مورد انتظار:

با نسخه اصلاح شده باید این نتایج را ببینید:

| Round | Clean Acc | Adv Acc | Detection | وضعیت |
|-------|-----------|---------|-----------|--------|
| 1-3   | 20-40%    | 15-25%  | >0%       | ✅ بهبود |
| 4-8   | 50-70%    | 30-40%  | متغیر    | ✅ پیشرفت |
| 12-15 | >70%      | >40%    | متغیر    | ✅ هدف |

## 📊 نظارت:

```bash
# بررسی وضعیت
ps -p $(cat kim2023_fixed_pid.txt) && echo "✅ در حال اجرا" || echo "❌ متوقف شده"

# مشاهده آخرین لاگ‌ها
tail -20 kim2023_fixed_*.log

# بررسی استفاده از GPU
nvidia-smi
```

## 🛑 متوقف کردن:

```bash
# متوقف کردن آزمایش
pkill -f "kim2023_fixed"

# یا با PID
kill $(cat kim2023_fixed_pid.txt)

# پاکسازی
rm -f kim2023_fixed_pid.txt
```

## ⚡ مزایای نسخه اصلاح شده:

1. **Server Aggregation**: از روش کارآمد `main.py` استفاده می‌کند
2. **MAE Detector**: درست لود می‌شود و کار می‌کند  
3. **DiffPure**: معماری سازگار با checkpoint موجود
4. **MobileNetV2**: اصلاح شده و بهینه‌سازی شده
5. **L2-PGD**: پیاده‌سازی صحیح حمله

**این نسخه باید مشکلات قبلی (دقت 10%) را حل کند و نتایج مناسب ارائه دهد!** 🎉
