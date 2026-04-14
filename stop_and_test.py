#!/usr/bin/env python3
"""
Stop current experiment and run simple test
"""

import os
import subprocess
import time

def stop_current_experiments():
    """Stop all Kim2023 experiments"""
    try:
        # Stop by process name
        subprocess.run(['pkill', '-f', 'kim2023'], check=False)
        subprocess.run(['pkill', '-f', 'main_kim2023'], check=False)
        time.sleep(2)
        print("✅ Stopped current experiments")
    except Exception as e:
        print(f"Note: {e}")

def run_simple_test():
    """Run simple test without DiffPure"""
    print("🚀 Starting simple test...")
    cmd = [
        'nohup', 'python', 'kim2023_simple_test.py', 
        '>', f'simple_test_{int(time.time())}.log', '2>&1', '&'
    ]
    
    # Run the command
    result = subprocess.run(' '.join(cmd), shell=True, capture_output=True, text=True)
    print(f"Command: {' '.join(cmd)}")
    print("✅ Simple test started in background")
    
    # Get latest log file
    import glob
    log_files = glob.glob('simple_test_*.log')
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        print(f"📝 Monitor with: tail -f {latest_log}")
        return latest_log
    
    return None

def main():
    print("🛑 Stopping current experiments...")
    stop_current_experiments()
    
    print("\n🧪 Running simple test (no DiffPure)...")
    log_file = run_simple_test()
    
    print("\n📊 Expected results:")
    print("- Round 1-2: Clean ~20-40%, Adv ~15-25%")
    print("- Round 3-5: Clean ~50-70%, Adv ~30-45%")
    print("- If accuracy improves → training works, DiffPure was the issue")
    print("- If accuracy stays ~10% → deeper training/aggregation issue")
    
    if log_file:
        print(f"\n👁️ Monitor: tail -f {log_file}")

if __name__ == "__main__":
    main()
