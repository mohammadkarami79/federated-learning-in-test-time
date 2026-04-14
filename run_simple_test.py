#!/usr/bin/env python3
"""
Run Simple Test Script
"""

import subprocess
import time
import glob
import os

def main():
    print("🛑 Stopping any running Kim2023 experiments...")
    
    # Stop current experiments
    try:
        subprocess.run(['pkill', '-f', 'kim2023'], check=False)
        subprocess.run(['pkill', '-f', 'main_kim2023'], check=False)
        time.sleep(3)
        print("✅ Stopped existing experiments")
    except:
        print("ℹ️ No experiments to stop")
    
    # Run simple test
    timestamp = int(time.time())
    log_file = f"simple_test_{timestamp}.log"
    
    print(f"🚀 Starting simple test (no DiffPure)...")
    print(f"📝 Log file: {log_file}")
    
    # Start the test
    cmd = f"nohup python kim2023_simple_test.py > {log_file} 2>&1 & echo $! > simple_test_pid.txt"
    os.system(cmd)
    
    time.sleep(2)
    
    print("✅ Simple test started!")
    print(f"👁️ Monitor with: tail -f {log_file}")
    print()
    print("📊 Expected Results:")
    print("   Round 1-2: Clean ~20-40%, Adv ~15-25%")
    print("   Round 3-5: Clean ~50-70%, Adv ~30-45%")
    print()
    print("🔍 What this tells us:")
    print("   • If accuracy improves → training works, DiffPure was blocking")
    print("   • If accuracy stays ~10% → deeper aggregation/training issue")
    print()
    print(f"📋 Full command: {cmd}")

if __name__ == "__main__":
    main()
