# 🚨 EMERGENCY STOP GUIDE

## فوری: متوقف کردن همه آزمایش‌ها

### گزینه 1: دستور سریع (Linux/Unix)
```bash
# متوقف کردن همه فرآیندهای FL
pkill -f "main.py"
pkill -f "main_kim2023"
pkill -f "main_baseline"
pkill -f "train_diffusion"
pkill -f "python.*main"

# پاک کردن فایل‌های PID
rm -f *pid*.txt kim2023_pid.txt

echo "✅ همه فرآیندها متوقف شدند"
```

### گزینه 2: اسکریپت Python
```bash
# اجرای اسکریپت مدیریت فرآیند
python check_running_processes.py

# یا اسکریپت متوقف کردن سریع
python stop_all_experiments.py
```

### گزینه 3: دستی
```bash
# بررسی فرآیندهای در حال اجرا
ps aux | grep python | grep -E "(main|kim2023|baseline)"

# متوقف کردن فرآیند خاص (PID را جایگزین کنید)
kill -TERM <PID>

# اگر متوقف نشد، کشتن اجباری
kill -9 <PID>
```

## بررسی وضعیت

### بررسی فرآیندهای در حال اجرا:
```bash
# بررسی فرآیندهای Python
ps aux | grep python

# بررسی فایل‌های PID
ls -la *pid*.txt

# بررسی لاگ‌های اخیر
ls -lt *.log | head -5
```

### بررسی استفاده از GPU:
```bash
# بررسی استفاده از GPU
nvidia-smi

# بررسی حافظه GPU
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### بررسی استفاده از CPU:
```bash
# بررسی بار سیستم
top -p $(pgrep -d, python)

# یا
htop -p $(pgrep -d, python)
```

## پاکسازی بعد از توقف

```bash
# پاک کردن فایل‌های موقت
rm -f nohup.out
rm -f *_$(date +%Y%m%d)*.log

# پاک کردن cache های Python
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete

# آزادسازی cache GPU
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
```

## شروع مجدد آزمایش

بعد از متوقف کردن همه فرآیندها:

```bash
# بررسی که همه چیز متوقف شده
ps aux | grep python | grep -v grep

# شروع آزمایش جدید
nohup python main_kim2023_reproduction.py > kim2023_new_$(date +%Y%m%d_%H%M%S).log 2>&1 & echo $! > kim2023_pid.txt

# نظارت
tail -f kim2023_new_*.log
```

## ⚠️ نکات مهم

1. **همیشه** قبل از شروع آزمایش جدید، همه فرآیندهای قبلی را متوقف کنید
2. فایل‌های PID را پاک کنید تا سردرگمی نشود  
3. لاگ‌های قدیمی را نگه دارید برای مرجع
4. حافظه GPU را پاک کنید اگر مشکل حافظه دارید

## 🆘 اگر هیچ کدام کار نکرد

```bash
# ری‌استارت کامل سشن
exit
# دوباره لاگین کنید

# یا ری‌استارت محیط conda
conda deactivate
conda activate br35h_env
```
