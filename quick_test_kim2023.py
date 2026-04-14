#!/usr/bin/env python3
"""
Quick Test Script for Kim2023 Issue Diagnosis
"""

import subprocess
import time
import os

def stop_experiments():
    """Stop all running experiments"""
    print("🛑 Stopping all Kim2023 experiments...")
    try:
        subprocess.run(['pkill', '-f', 'kim2023'], check=False)
        subprocess.run(['pkill', '-f', 'main_kim2023'], check=False)
        time.sleep(3)
        print("✅ Stopped existing experiments")
    except:
        print("ℹ️ No experiments to stop")

def run_simple_test():
    """Run simple test without DiffPure"""
    print("\n🧪 STEP 1: Testing Training (No DiffPure)")
    print("-" * 50)
    
    timestamp = int(time.time())
    log_file = f"simple_test_{timestamp}.log"
    
    cmd = f"nohup python kim2023_simple_test.py > {log_file} 2>&1 & echo $! > simple_test_pid.txt"
    os.system(cmd)
    
    time.sleep(2)
    print(f"✅ Simple test started")
    print(f"📝 Log: {log_file}")
    print(f"👁️ Monitor: tail -f {log_file}")
    
    return log_file

def run_complete_test():
    """Run complete version with all defenses"""
    print("\n🔧 STEP 2: Testing Complete Version (All Defenses)")
    print("-" * 50)
    
    timestamp = int(time.time())
    log_file = f"complete_test_{timestamp}.log"
    
    cmd = f"nohup python main_kim2023_final_complete.py > {log_file} 2>&1 & echo $! > complete_test_pid.txt"
    os.system(cmd)
    
    time.sleep(2)
    print(f"✅ Complete test started")
    print(f"📝 Log: {log_file}")
    print(f"👁️ Monitor: tail -f {log_file}")
    
    return log_file

def main():
    print("🚀 Kim et al. 2023 - Quick Diagnosis Test")
    print("=" * 60)
    
    # Stop existing experiments
    stop_experiments()
    
    print("\n🎯 Test Strategy:")
    print("1. Simple test → Check if basic training works")
    print("2. Complete test → Check if defenses work together")
    print("\nChoose test to run:")
    print("1. Simple test only (recommended first)")
    print("2. Complete test only")
    print("3. Both tests")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        simple_log = run_simple_test()
        print(f"\n📊 Expected: Clean accuracy should improve from ~10% to 40-70%")
        print(f"👁️ Monitor: tail -f {simple_log}")
        
    elif choice == "2":
        complete_log = run_complete_test()
        print(f"\n📊 Expected: Should handle DiffPure gracefully + good accuracy")
        print(f"👁️ Monitor: tail -f {complete_log}")
        
    elif choice == "3":
        simple_log = run_simple_test()
        print("\n⏳ Waiting 30 seconds before starting complete test...")
        time.sleep(30)
        complete_log = run_complete_test()
        
        print(f"\n📊 Compare results:")
        print(f"Simple: tail -f {simple_log}")
        print(f"Complete: tail -f {complete_log}")
    
    else:
        print("❌ Invalid choice")
        return
    
    print("\n✅ Test(s) started!")
    print("\n🔍 What to look for:")
    print("• Round 1-2: Clean accuracy > 20%")
    print("• Round 3-5: Clean accuracy > 50%")
    print("• No 'aggregation errors'")
    print("• Training time ~5-10s per client")

if __name__ == "__main__":
    main()
