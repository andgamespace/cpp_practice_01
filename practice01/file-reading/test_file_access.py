#!/usr/bin/env python3
"""
Test script to verify file access to stock data.
"""

import os
import sys

def check_file_access(base_dir):
    """Check if stock data files are accessible."""
    print(f"Checking file access in: {base_dir}")
    
    # Check if directory exists
    if not os.path.isdir(base_dir):
        print(f"Directory does not exist: {base_dir}")
        return False
    
    # List files in directory
    files = os.listdir(base_dir)
    print(f"Found {len(files)} files in {base_dir}:")
    for file in files:
        print(f"  - {file}")
    
    # Check specific files
    tickers = ["AAPL", "MSFT", "NVDA", "AMD"]
    for ticker in tickers:
        file_path = os.path.join(base_dir, f"time-series-{ticker}-5min.csv")
        if os.path.isfile(file_path):
            print(f"File exists: {file_path}")
            # Check file size
            size = os.path.getsize(file_path)
            print(f"  - Size: {size} bytes")
            # Check if file is readable
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                    print(f"  - First line: {first_line}")
            except Exception as e:
                print(f"  - Error reading file: {e}")
        else:
            print(f"File does not exist: {file_path}")
    
    return True

if __name__ == "__main__":
    # Check current directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Check stock data in different paths
    paths = [
        "./src/stock_data",
        "../src/stock_data",
        "src/stock_data",
        "/Users/anshc/repos/cpp_practice_01/practice01/file-reading/src/stock_data"
    ]
    
    for path in paths:
        print("\n" + "="*50)
        check_file_access(path)