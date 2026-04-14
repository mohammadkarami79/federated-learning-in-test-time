#!/usr/bin/env python3
"""
Simple runner for the FIXED Kim et al., 2023 reproduction
========================================================

This script runs the fixed version that should work properly.
"""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

def main():
    """Run the fixed Kim2023 reproduction"""
    
    print("🚀 Starting FIXED Kim et al., 2023 Reproduction")
    print("=" * 60)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Output directory
    output_dir = f"./kim2023_fixed_results_{timestamp}"
    
    # Log file
    log_file = f"kim2023_fixed_{timestamp}.log"
    
    print(f"📊 Output Directory: {output_dir}")
    print(f"📝 Log File: {log_file}")
    print(f"🔧 Script: main_kim2023_reproduction_fixed.py")
    
    print("\n⚡ Expected Improvements:")
    print("- Clean Accuracy: Should increase significantly (>70%)")
    print("- Proper server aggregation (from main.py)")
    print("- Working MAE + DiffPure integration")
    print("- Fixed MobileNetV2 architecture")
    
    print("\n" + "=" * 60)
    
    # Construct command
    cmd = [
        sys.executable,
        "main_kim2023_reproduction_fixed.py",
        "--output-dir", output_dir
    ]
    
    print(f"✅ Starting experiment...")
    print(f"📋 Command: {' '.join(cmd)}")
    
    # Run in background with logging
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=Path.cwd()
            )
            
            # Save PID
            with open('kim2023_fixed_pid.txt', 'w') as pid_file:
                pid_file.write(str(process.pid))
            
            print(f"🎯 Process started with PID: {process.pid}")
            print(f"📁 PID saved to: kim2023_fixed_pid.txt")
            print(f"\n👁️  Monitor with:")
            print(f"   tail -f {log_file}")
            print(f"\n🛑 Stop with:")
            print(f"   pkill -f main_kim2023_reproduction_fixed.py")
            print(f"   # or")
            print(f"   kill {process.pid}")
            
            return process.pid
            
    except Exception as e:
        print(f"❌ Failed to start experiment: {e}")
        return None

if __name__ == "__main__":
    pid = main()
    if pid:
        print(f"\n✅ FIXED Kim2023 experiment started successfully!")
        print(f"🎯 This version should have much better results!")
    else:
        print("❌ Failed to start experiment")
        sys.exit(1)
