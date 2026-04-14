#!/usr/bin/env python3
"""
Direct Complete Test Runner - No Interactive Input
"""

import subprocess
import time
import os

def main():
    print("🔧 Running Complete Test Directly...")
    print("=" * 50)
    
    # Stop any existing experiments
    try:
        subprocess.run(['pkill', '-f', 'kim2023'], check=False)
        time.sleep(2)
        print("✅ Stopped existing experiments")
    except:
        print("ℹ️ No experiments to stop")
    
    # Run complete test directly
    timestamp = int(time.time())
    log_file = f"complete_test_{timestamp}.log"
    pid_file = "complete_test_pid.txt"
    
    print(f"📝 Starting complete test...")
    print(f"📋 Log file: {log_file}")
    
    # Start the test in background
    cmd = f"nohup python main_kim2023_final_complete.py > {log_file} 2>&1 & echo $! > {pid_file}"
    result = os.system(cmd)
    
    time.sleep(3)
    
    # Check if it started
    if os.path.exists(log_file):
        print("✅ Complete test started successfully!")
        print(f"👁️ Monitor with: tail -f {log_file}")
        print()
        print("📊 Expected Results:")
        print("   Round 1-2: Clean accuracy 15-35%")
        print("   Round 3-5: Clean accuracy 40-60%")
        print("   With defenses: MAE detection + optional DiffPure")
        print()
        print("🔍 Should handle DiffPure gracefully if import fails")
    else:
        print("❌ Failed to start test")
        print("📋 Try manually: python main_kim2023_final_complete.py")

if __name__ == "__main__":
    main()
