#!/bin/bash
# FINAL SERVER DEPLOYMENT SCRIPT
# ==============================

echo "Starting Final Server Deployment"
echo "=================================="

# 1. Kill existing processes
echo "1. Killing existing training processes..."
pkill -f python || true
pkill -f main.py || true

# 2. Clean broken files
echo "2. Cleaning broken checkpoints and logs..."
rm -f checkpoints/mae_detector*.pt
rm -f logs/log*.txt
rm -f *.log

# 3. Create necessary directories
echo "3. Creating directories..."
mkdir -p checkpoints
mkdir -p logs
mkdir -p defense

# 4. Set permissions
echo "4. Setting permissions..."
chmod +x run_fixed_cifar10.py
chmod +x config_ultimate.py

# 5. Install dependencies if needed
echo "5. Checking dependencies..."
python -c "import torch, torchvision" || pip install torch torchvision

echo "Deployment preparation complete!"
echo ""
echo "Next steps:"
echo "1. Run: python run_fixed_cifar10.py"
echo "2. Monitor logs for stable training"
echo "3. Check for MAE detection rate ~10-20%"
echo "4. Verify adversarial accuracy improvement"
