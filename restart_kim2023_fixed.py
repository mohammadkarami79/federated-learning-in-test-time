#!/usr/bin/env python3
"""
Restart Kim et al., 2023 experiment with fixes
==============================================

This script stops the current experiment and restarts with the fixes.
"""

import subprocess
import sys
import os
import time
from datetime import datetime
from pathlib import Path

def stop_current_experiments():
    """Stop all current Kim2023 experiments"""
    print("🛑 Stopping current experiments...")
    
    try:
        # Kill processes
        subprocess.run(["pkill", "-f", "kim2023"], check=False)
        subprocess.run(["pkill", "-f", "main_kim2023"], check=False)
        time.sleep(2)
        print("✅ Stopped running processes")
    except Exception as e:
        print(f"⚠️  Warning stopping processes: {e}")
    
    # Clean PID files
    try:
        for pid_file in Path('.').glob('*kim2023*pid*.txt'):
            pid_file.unlink()
        print("✅ Cleaned PID files")
    except Exception as e:
        print(f"⚠️  Warning cleaning PIDs: {e}")

def start_fixed_experiment():
    """Start the fixed experiment"""
    print("\n🚀 Starting FIXED Kim et al., 2023 experiment...")
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Output directory
    output_dir = f"./kim2023_fixed_restart_{timestamp}"
    
    # Log file
    log_file = f"kim2023_fixed_restart_{timestamp}.log"
    
    print(f"📊 Output Directory: {output_dir}")
    print(f"📝 Log File: {log_file}")
    
    # Command
    cmd = [
        sys.executable,
        "main_kim2023_reproduction_fixed.py",
        "--output-dir", output_dir
    ]
    
    try:
        # Start in background
        process = subprocess.Popen(
            cmd,
            stdout=open(log_file, 'w'),
            stderr=subprocess.STDOUT,
            cwd=Path.cwd()
        )
        
        # Save PID
        with open('kim2023_fixed_restart_pid.txt', 'w') as f:
            f.write(str(process.pid))
        
        print(f"✅ Started with PID: {process.pid}")
        print(f"📁 PID file: kim2023_fixed_restart_pid.txt")
        print(f"📝 Monitor: tail -f {log_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        return False

def main():
    """Main function"""
    print("🔄 RESTARTING Kim et al., 2023 Experiment with Fixes")
    print("=" * 55)
    
    print("🔧 Applied Fixes:")
    print("- ✅ L2-PGD attack gradient issue (requires_grad)")
    print("- ✅ DiffPure architecture (hidden_channels=256)")
    print("- ✅ Server aggregation from main.py")
    print("- ✅ MAE detector integration")
    
    print("\n" + "=" * 55)
    
    # Stop current
    stop_current_experiments()
    
    # Start fixed
    if start_fixed_experiment():
        print("\n🎉 Fixed experiment started successfully!")
        print("\n⚡ Expected improvements:")
        print("- Clean accuracy should increase properly (not stuck at 10%)")
        print("- No more 'requires_grad' errors")
        print("- DiffPure should load (if checkpoint compatible)")
        print("- Better convergence overall")
        
        print(f"\n👁️  Monitor with:")
        print(f"   tail -f kim2023_fixed_restart_*.log")
        print(f"\n🛑 Stop with:")
        print(f"   pkill -f kim2023_fixed")
        
    else:
        print("\n❌ Failed to start fixed experiment")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
