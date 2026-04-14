#!/usr/bin/env python3
"""
Quick script to stop all running FL experiments
==============================================
"""

import subprocess
import os
import signal
from pathlib import Path

def stop_all_fl_experiments():
    """Stop all FL experiments using pkill"""
    
    print("🛑 Stopping all FL experiments...")
    
    # List of process patterns to kill
    patterns = [
        "main.py",
        "main_kim2023_reproduction.py", 
        "main_baseline_pfeddef",
        "train_diffusion",
        "integrated_diffusion",
        "run_kim2023"
    ]
    
    stopped_count = 0
    
    for pattern in patterns:
        try:
            # Check if any processes match this pattern
            result = subprocess.run(
                ["pgrep", "-f", pattern], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:  # Found processes
                pids = result.stdout.strip().split('\n')
                print(f"🔍 Found {len(pids)} processes for pattern '{pattern}'")
                
                # Kill the processes
                subprocess.run(["pkill", "-f", pattern])
                stopped_count += len(pids)
                print(f"✅ Stopped processes matching '{pattern}'")
            else:
                print(f"⚪ No processes found for pattern '{pattern}'")
                
        except FileNotFoundError:
            # pkill/pgrep not available, try different approach
            print(f"⚠️  pkill not available, skipping pattern '{pattern}'")
        except Exception as e:
            print(f"❌ Error stopping '{pattern}': {e}")
    
    # Clean up PID files
    print("\n🧹 Cleaning up PID files...")
    pid_files_cleaned = 0
    
    for pid_file in Path('.').glob('*pid*.txt'):
        try:
            pid_file.unlink()
            pid_files_cleaned += 1
            print(f"✅ Removed {pid_file.name}")
        except Exception as e:
            print(f"❌ Could not remove {pid_file.name}: {e}")
    
    print(f"\n📊 Summary:")
    print(f"- Stopped ~{stopped_count} processes")
    print(f"- Cleaned {pid_files_cleaned} PID files")
    print(f"✅ All FL experiments should now be stopped!")

if __name__ == "__main__":
    stop_all_fl_experiments()
