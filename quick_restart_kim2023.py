#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick restart for Kim et al., 2023 experiment - NO UNICODE
===========================================================
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def main():
    print("Quick restart Kim et al., 2023 - ENCODING FIXED")
    print("=" * 50)
    
    # Stop current
    print("Stopping current processes...")
    try:
        subprocess.run(["pkill", "-f", "kim2023"], check=False)
        time.sleep(2)
        print("Stopped.")
    except:
        pass
    
    # Clean files
    try:
        os.system("rm -f *kim2023*pid*.txt")
        print("Cleaned PID files.")
    except:
        pass
    
    # Start new
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"kim2023_encoding_fix_{timestamp}.log"
    
    print(f"Starting with log: {log_file}")
    
    cmd = f"nohup python main_kim2023_reproduction_fixed.py > {log_file} 2>&1 & echo $! > kim2023_encoding_fix_pid.txt"
    
    os.system(cmd)
    
    print("Started!")
    print(f"Monitor: tail -f {log_file}")
    print("PID file: kim2023_encoding_fix_pid.txt")

if __name__ == "__main__":
    main()
