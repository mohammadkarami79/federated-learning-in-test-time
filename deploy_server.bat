@echo off
REM FINAL SERVER DEPLOYMENT SCRIPT (Windows)
REM ==========================================

echo Starting Final Server Deployment
echo ==================================

REM 1. Kill existing processes
echo 1. Killing existing training processes...
taskkill /F /IM python.exe >nul 2>&1

REM 2. Clean broken files
echo 2. Cleaning broken checkpoints and logs...
del /Q checkpoints\mae_detector*.pt >nul 2>&1
del /Q logs\log*.txt >nul 2>&1
del /Q *.log >nul 2>&1

REM 3. Create necessary directories
echo 3. Creating directories...
if not exist checkpoints mkdir checkpoints
if not exist logs mkdir logs
if not exist defense mkdir defense

REM 4. Check dependencies
echo 4. Checking dependencies...
python -c "import torch, torchvision" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install torch torchvision
)

echo Deployment preparation complete!
echo.
echo Next steps:
echo 1. Run: python run_fixed_cifar10.py
echo 2. Monitor logs for stable training
echo 3. Check for MAE detection rate ~10-20%%
echo 4. Verify adversarial accuracy improvement

pause
