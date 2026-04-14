#!/bin/bash
"""
اجرای آزمایش Kim et al., 2023 نسخه اصلاح شده با nohup
====================================================
"""

echo "🚀 شروع آزمایش Kim et al., 2023 - نسخه اصلاح شده"
echo "=================================================="

# تاریخ و زمان فعلی
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# نام فایل لاگ
LOG_FILE="kim2023_fixed_nohup_${TIMESTAMP}.log"

echo "📝 فایل لاگ: $LOG_FILE"
echo "📊 اسکریپت: run_kim2023_fixed.py"
echo "🔧 حالت: nohup (پس‌زمینه)"

# متوقف کردن فرآیندهای قبلی
echo "🛑 متوقف کردن فرآیندهای قبلی..."
pkill -f "main_kim2023" 2>/dev/null || true
pkill -f "kim2023_fixed" 2>/dev/null || true
rm -f kim2023_fixed_pid.txt 2>/dev/null || true

echo "✅ آماده برای شروع..."
echo "=================================================="

# اجرا با nohup
nohup python run_kim2023_fixed.py > "$LOG_FILE" 2>&1 & 

# ذخیره PID
echo $! > kim2023_fixed_nohup_pid.txt

echo "✅ آزمایش در پس‌زمینه شروع شد!"
echo "📋 PID: $(cat kim2023_fixed_nohup_pid.txt)"
echo "📝 لاگ: $LOG_FILE"
echo ""
echo "👁️  نظارت با:"
echo "   tail -f $LOG_FILE"
echo ""
echo "🛑 متوقف کردن با:"
echo "   pkill -f kim2023_fixed"
echo "   # یا"
echo "   kill $(cat kim2023_fixed_nohup_pid.txt)"
echo ""
echo "📊 بررسی وضعیت:"
echo "   ps -p $(cat kim2023_fixed_nohup_pid.txt)"
