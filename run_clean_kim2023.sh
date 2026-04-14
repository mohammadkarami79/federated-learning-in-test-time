#!/bin/bash
echo "==================================================="
echo "Kim et al., 2023 - CLEAN VERSION (No Encoding Issues)"
echo "==================================================="

# Stop any running processes
echo "Stopping previous experiments..."
pkill -f "kim2023" 2>/dev/null || true
pkill -f "main_kim2023" 2>/dev/null || true
sleep 2

# Clean PID files
rm -f *kim2023*pid*.txt 2>/dev/null || true

# Create timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="kim2023_clean_${TIMESTAMP}.log"

echo "Starting CLEAN experiment..."
echo "Log file: $LOG_FILE"

# Run the clean version
nohup python main_kim2023_clean.py > "$LOG_FILE" 2>&1 & 
echo $! > kim2023_clean_pid.txt

echo "========================================"
echo "✅ Started successfully!"
echo "PID: $(cat kim2023_clean_pid.txt)"
echo "Log: $LOG_FILE"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Stop with:"
echo "  pkill -f kim2023_clean"
echo "========================================"
