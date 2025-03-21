#!/usr/bin/env python3
"""
Test script for the trading environment.

This script tests the basic functionality of the trading environment,
including data loading, backtesting, and visualization.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any

# Add the current directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import the trading environment
try:
    import trading_env
    print("Successfully imported trading_env")
except ImportError as e:
    print(f"Error importing trading_env: {e}")
    print("Make sure the my_module shared library is built and in the correct location.")
    sys.exit(1)

def test_data_loading():
    """Test data loading functionality."""
    print("\n=== Testing Data Loading ===")
    
    # Create environment
    env = trading_env.create_trading_env(
        data_dir="src/stock_data",
        tickers=["AAPL", "MSFT", "NVDA", "AMD"],
        visualize=False,
        debug=True
    )
    
    # Check if tickers are loaded
    tickers = env.engine.getAvailableTickers()
    print(f"Available tickers: {tickers}")
    
    # Test getting ticker data
    try:
        df = env.get_ticker_data("AAPL")
        print(f"Successfully loaded AAPL data with {len(df)} rows")
        print("First 5 rows:")
        print(df.head())
        return True
    except Exception as e:
        print(f"Error loading ticker data: {e}")
        return False

def test_backtest():
    """Test backtesting functionality."""
    print("\n=== Testing Backtesting ===")
    
    # Create environment
    env = trading_env.create_trading_env(
        data_dir="src/stock_data",
        tickers=["AAPL", "MSFT"],
        visualize=False,
        debug=True
    )
    
    # Run backtest
    try:
        metrics = env.run_backtest()
        print("Backtest completed successfully")
        print(f"Initial Balance: ${metrics.initialBalance:.2f}")
        print(f"Final Balance: ${metrics.finalBalance:.2f}")
        print(f"Total Return: {metrics.totalReturn * 100:.2f}%")
        print(f"Sharpe Ratio: {metrics.sharpeRatio:.2f}")
        return True
    except Exception as e:
        print(f"Error running backtest: {e}")
        return False

def test_gym_interface():
    """Test Gymnasium interface."""
    print("\n=== Testing Gymnasium Interface ===")
    
    # Create environment
    env = trading_env.create_trading_env(
        data_dir="src/stock_data",
        tickers=["AAPL"],
        visualize=False,
        debug=True
    )
    
    # Test reset
    try:
        obs, info = env.reset()
        print(f"Reset successful. Observation shape: {obs.shape}")
        print(f"Info: {info}")
        
        # Test step
        action = np.array([0.5])  # Buy 50% of available cash
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step successful. Reward: {reward}")
        print(f"Observation shape: {obs.shape}")
        print(f"Info: {info}")
        
        # Close environment
        env.close()
        return True
    except Exception as e:
        print(f"Error testing gym interface: {e}")
        return False

def test_visualization():
    """Test visualization functionality."""
    print("\n=== Testing Visualization ===")
    
    # Create environment
    env = trading_env.create_trading_env(
        data_dir="src/stock_data",
        tickers=["AAPL", "MSFT", "NVDA", "AMD"],
        visualize=False,
        debug=True
    )
    
    # Test plotting
    try:
        # Run a short backtest
        env.run_backtest()
        
        # Plot performance
        fig = env.plot_performance()
        plt.savefig("test_performance.png")
        print("Performance plot saved to test_performance.png")
        
        # Plot ticker
        fig = env.plot_ticker("AAPL")
        plt.savefig("test_ticker.png")
        print("Ticker plot saved to test_ticker.png")
        
        return True
    except Exception as e:
        print(f"Error testing visualization: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Running Trading Environment Tests ===")
    
    # Run tests
    tests = [
        ("Data Loading", test_data_loading),
        ("Backtesting", test_backtest),
        ("Gym Interface", test_gym_interface),
        ("Visualization", test_visualization)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nRunning test: {name}")
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n=== Test Summary ===")
    all_passed = True
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{name}: {status}")
        if not success:
            all_passed = False
    
    # Exit with appropriate code
    if all_passed:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())