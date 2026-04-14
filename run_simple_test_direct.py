#!/usr/bin/env python3
"""
Direct Simple Test Runner - No Interactive Input
"""

import subprocess
import time
import os

def main():
    print("🚀 Running Simple Test Directly...")
    print("=" * 50)
    
    # Stop any existing experiments
    try:
        subprocess.run(['pkill', '-f', 'kim2023'], check=False)
        time.sleep(2)
        print("✅ Stopped existing experiments")
    except:
        print("ℹ️ No experiments to stop")
    
    # Run simple test directly
    timestamp = int(time.time())
    log_file = f"simple_test_{timestamp}.log"
    pid_file = "simple_test_pid.txt"
    
    print(f"📝 Starting simple test...")
    print(f"📋 Log file: {log_file}")
    
    # Start the test in background
    cmd = f"nohup python kim2023_simple_test.py > {log_file} 2>&1 & echo $! > {pid_file}"
    result = os.system(cmd)
    
    time.sleep(3)
    
    # Check if it started
    if os.path.exists(log_file):
        print("✅ Simple test started successfully!")
        print(f"👁️ Monitor with: tail -f {log_file}")
        print()
        print("📊 Expected Results:")
        print("   Round 1-2: Clean accuracy 20-40%")
        print("   Round 3-5: Clean accuracy 50-70%")
        print()
        print("🔍 If accuracy improves from ~10% → training works!")
        print("🔍 If accuracy stays ~10% → deeper issue exists")
    else:
        print("❌ Failed to start test")
        print("📋 Try manually: python kim2023_simple_test.py")

if __name__ == "__main__":
    main()
