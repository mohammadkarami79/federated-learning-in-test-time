#!/usr/bin/env python3
"""
Check and manage running FL processes
=====================================

This script helps identify and manage running federated learning processes
"""

import subprocess
import psutil
import os
import signal
from pathlib import Path

def check_python_processes():
    """Check all Python processes related to FL"""
    print("🔍 Checking Python processes...")
    
    fl_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                
                # Check if it's related to FL
                if any(keyword in cmdline.lower() for keyword in [
                    'main.py', 'main_kim2023', 'main_baseline', 
                    'federated', 'diffusion', 'mae', 'train_'
                ]):
                    fl_processes.append({
                        'pid': proc.info['pid'],
                        'cmdline': cmdline,
                        'create_time': proc.info['create_time']
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    return fl_processes

def check_pid_files():
    """Check PID files in the directory"""
    print("\n📁 Checking PID files...")
    
    pid_files = []
    for pid_file in Path('.').glob('*pid*.txt'):
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
                
            # Check if process is running
            try:
                process = psutil.Process(pid)
                status = "✅ Running"
                cmdline = ' '.join(process.cmdline())
            except psutil.NoSuchProcess:
                status = "❌ Not Running"
                cmdline = "N/A"
                
            pid_files.append({
                'file': pid_file.name,
                'pid': pid,
                'status': status,
                'cmdline': cmdline
            })
        except (ValueError, FileNotFoundError):
            pass
    
    return pid_files

def check_log_files():
    """Check recent log files"""
    print("\n📝 Checking recent log files...")
    
    import time
    current_time = time.time()
    recent_logs = []
    
    for log_file in Path('.').glob('*.log'):
        try:
            stat = log_file.stat()
            age_hours = (current_time - stat.st_mtime) / 3600
            
            if age_hours < 2:  # Files modified in last 2 hours
                size_mb = stat.st_size / (1024 * 1024)
                recent_logs.append({
                    'file': log_file.name,
                    'age_hours': age_hours,
                    'size_mb': size_mb
                })
        except OSError:
            pass
    
    # Sort by modification time
    recent_logs.sort(key=lambda x: x['age_hours'])
    return recent_logs

def stop_all_fl_processes():
    """Stop all FL-related processes"""
    print("\n🛑 Stopping all FL processes...")
    
    processes = check_python_processes()
    
    if not processes:
        print("✅ No FL processes found")
        return
    
    for proc_info in processes:
        try:
            pid = proc_info['pid']
            print(f"Stopping PID {pid}: {proc_info['cmdline'][:80]}...")
            
            process = psutil.Process(pid)
            process.terminate()
            
            # Wait a bit for graceful termination
            try:
                process.wait(timeout=5)
                print(f"✅ Process {pid} terminated gracefully")
            except psutil.TimeoutExpired:
                print(f"⚠️  Process {pid} didn't terminate, forcing kill...")
                process.kill()
                print(f"✅ Process {pid} killed")
                
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"❌ Could not stop PID {pid}: {e}")

def main():
    print("🔧 FL Process Manager")
    print("=" * 50)
    
    # Check running processes
    processes = check_python_processes()
    print(f"\n🐍 Found {len(processes)} FL-related Python processes:")
    for i, proc in enumerate(processes, 1):
        print(f"{i}. PID {proc['pid']}: {proc['cmdline'][:80]}...")
    
    # Check PID files
    pid_files = check_pid_files()
    print(f"\n📁 Found {len(pid_files)} PID files:")
    for pf in pid_files:
        print(f"- {pf['file']}: PID {pf['pid']} ({pf['status']})")
        if pf['status'] == "✅ Running":
            print(f"  Command: {pf['cmdline'][:80]}...")
    
    # Check recent logs
    recent_logs = check_log_files()
    print(f"\n📝 Found {len(recent_logs)} recent log files:")
    for log in recent_logs:
        print(f"- {log['file']}: {log['age_hours']:.1f}h old, {log['size_mb']:.1f} MB")
    
    # Interactive options
    print("\n🎯 Options:")
    print("1. Stop all FL processes")
    print("2. Show detailed process info")
    print("3. Clean old PID files")
    print("4. Exit")
    
    choice = input("\nChoose option (1-4): ").strip()
    
    if choice == "1":
        stop_all_fl_processes()
        print("\n✅ All processes stopped!")
        
    elif choice == "2":
        print("\n📊 Detailed Process Information:")
        for i, proc in enumerate(processes, 1):
            print(f"\n{i}. PID {proc['pid']}")
            print(f"   Command: {proc['cmdline']}")
            print(f"   Created: {proc['create_time']}")
            
    elif choice == "3":
        print("\n🧹 Cleaning old PID files...")
        for pf in pid_files:
            if pf['status'] == "❌ Not Running":
                pid_file = Path(pf['file'])
                pid_file.unlink()
                print(f"✅ Removed {pf['file']}")
        print("✅ Cleanup complete!")
        
    elif choice == "4":
        print("👋 Goodbye!")
        return
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
