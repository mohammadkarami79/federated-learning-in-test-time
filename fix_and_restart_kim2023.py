#!/usr/bin/env python3
"""
Fix and restart Kim et al., 2023 experiment
==========================================

This script stops the current experiment and restarts with fixes.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def stop_current_experiment():
    """Stop the current running experiment"""
    print("🛑 Stopping current experiment...")
    
    # Read PID if exists
    pid_file = Path("kim2023_pid.txt")
    if pid_file.exists():
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            print(f"Found PID: {pid}")
            
            # Try to kill the process
            try:
                os.kill(pid, 15)  # SIGTERM
                print(f"✅ Sent SIGTERM to process {pid}")
                time.sleep(3)
                
                # Check if still running
                try:
                    os.kill(pid, 0)  # Check if process exists
                    print("Process still running, sending SIGKILL...")
                    os.kill(pid, 9)  # SIGKILL
                    time.sleep(2)
                except OSError:
                    print("✅ Process terminated successfully")
                    
            except OSError as e:
                print(f"Process {pid} not found or already terminated: {e}")
                
        except Exception as e:
            print(f"Error reading PID file: {e}")
    
    # Kill any remaining python processes related to kim2023
    try:
        result = subprocess.run(['pkill', '-f', 'kim2023'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Killed remaining kim2023 processes")
        else:
            print("No remaining kim2023 processes found")
    except:
        print("Could not run pkill (system limitation)")

def check_fixed_files():
    """Check if the fixed files exist"""
    print("\n🔍 Checking fixed files...")
    
    required_files = [
        'main_kim2023_reproduction.py',
        'run_kim2023_reproduction.py'
    ]
    
    all_found = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_found = False
    
    return all_found

def restart_experiment():
    """Restart the experiment with fixes"""
    print("\n🚀 Restarting Kim et al., 2023 experiment...")
    
    # Generate new timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Command to restart
    cmd = [
        'nohup', 'python', 'run_kim2023_reproduction.py', '--background',
        '--log-suffix', f'_fixed_{timestamp}'
    ]
    
    # Start the process
    log_file = f'kim2023_runner_fixed_{timestamp}.log'
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    # Save new PID
    with open('kim2023_pid.txt', 'w') as f:
        f.write(str(process.pid))
    
    print(f"✅ Experiment restarted with PID: {process.pid}")
    print(f"📝 Log file: {log_file}")
    print(f"👁️  Monitor with: tail -f kim2023_reproduction_full_*fixed*.log")
    
    return process.pid

def main():
    """Main function"""
    print("=" * 60)
    print("KIM ET AL., 2023 EXPERIMENT FIX & RESTART")
    print("=" * 60)
    
    # Step 1: Stop current experiment
    stop_current_experiment()
    
    # Step 2: Check fixed files
    if not check_fixed_files():
        print("\n❌ Required files missing. Please ensure fixes are applied.")
        return 1
    
    # Step 3: Restart experiment
    new_pid = restart_experiment()
    
    print("\n" + "=" * 60)
    print("RESTART COMPLETED")
    print("=" * 60)
    print(f"New PID: {new_pid}")
    print("Next steps:")
    print("1. tail -f kim2023_reproduction_full_*fixed*.log")
    print("2. Monitor progress and check for defense component loading")
    print("3. Watch for improved error handling")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
