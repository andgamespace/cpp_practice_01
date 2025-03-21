#!/usr/bin/env python3
"""
Test script for technical indicators in the trading environment.

This script demonstrates how to use the technical indicators functionality
to add various indicators to ticker data and visualize the results.
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
    import my_module
    print("Successfully imported trading_env and my_module")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure the my_module shared library is built and in the correct location.")
    sys.exit(1)

def test_technical_indicators():
    """Test technical indicators functionality."""
    print("\n=== Testing Technical Indicators ===")
    
    # Create DataLoader
    loader = my_module.DataLoader()
    loader.setDebugMode(True)
    
    # Load ticker data
    data_dir = "./src/stock_data"
    ticker = "AAPL"
    file_paths = [
        f"{data_dir}/time-series-{ticker}-5min.csv",
        f"{data_dir}/time-series-{ticker}-5min(1).csv",
        f"{data_dir}/time-series-{ticker}-5min(2).csv"
    ]
    
    success = loader.loadTickerData(ticker, file_paths)
    if not success:
        print(f"Failed to load ticker data for {ticker}")
        return False
    
    # Get the Arrow table
    table = loader.getTickerData(ticker)
    if not table:
        print(f"Failed to get ticker data for {ticker}")
        return False
    
    print(f"Successfully loaded {table.num_rows()} rows for {ticker}")
    
    # Create TechnicalIndicators instance
    ti = my_module.TechnicalIndicators()
    ti.setDebugMode(True)
    
    # Calculate SMA
    print("\nCalculating SMA...")
    table_with_sma = ti.calculateSMA(table, "close", 20)
    print(f"Table now has {table_with_sma.num_columns()} columns")
    
    # Calculate EMA
    print("\nCalculating EMA...")
    table_with_ema = ti.calculateEMA(table_with_sma, "close", 50)
    print(f"Table now has {table_with_ema.num_columns()} columns")
    
    # Calculate RSI
    print("\nCalculating RSI...")
    table_with_rsi = ti.calculateRSI(table_with_ema, "close", 14)
    print(f"Table now has {table_with_rsi.num_columns()} columns")
    
    # Calculate Bollinger Bands
    print("\nCalculating Bollinger Bands...")
    table_with_bb = ti.calculateBollingerBands(table_with_rsi, "close", 20, 2.0)
    print(f"Table now has {table_with_bb.num_columns()} columns")
    
    # Calculate MACD
    print("\nCalculating MACD...")
    table_with_macd = ti.calculateMACD(table_with_bb, "close", 12, 26, 9)
    print(f"Table now has {table_with_macd.num_columns()} columns")
    
    # Update the ticker data in the loader
    loader.updateTickerData(ticker, table_with_macd)
    
    # Convert to pandas DataFrame
    print("\nConverting to pandas DataFrame...")
    df = loader.arrowToPandas(ticker)
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    
    # Plot the data
    print("\nPlotting data...")
    plot_technical_indicators(df)
    
    return True

def plot_technical_indicators(df):
    """Plot technical indicators."""
    # Create figure and subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot 1: Price and Moving Averages
    axs[0].set_title('Price and Moving Averages')
    axs[0].plot(df['datetime'], df['close'], label='Close', color='black', alpha=0.7)
    axs[0].plot(df['datetime'], df['close_sma_20'], label='SMA(20)', color='blue')
    axs[0].plot(df['datetime'], df['close_ema_50'], label='EMA(50)', color='red')
    axs[0].fill_between(df['datetime'], df['close_bband_upper'], df['close_bband_lower'], 
                       color='gray', alpha=0.2, label='Bollinger Bands')
    axs[0].set_ylabel('Price')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Plot 2: RSI
    axs[1].set_title('Relative Strength Index (RSI)')
    axs[1].plot(df['datetime'], df['close_rsi_14'], color='purple')
    axs[1].axhline(y=70, color='r', linestyle='--', alpha=0.5)
    axs[1].axhline(y=30, color='g', linestyle='--', alpha=0.5)
    axs[1].set_ylabel('RSI')
    axs[1].set_ylim(0, 100)
    axs[1].grid(True, alpha=0.3)
    
    # Plot 3: MACD
    axs[2].set_title('MACD')
    axs[2].plot(df['datetime'], df['close_macd'], label='MACD', color='blue')
    axs[2].plot(df['datetime'], df['close_macd_signal'], label='Signal', color='red')
    axs[2].bar(df['datetime'], df['close_macd_histogram'], label='Histogram', color='gray', alpha=0.5)
    axs[2].set_ylabel('MACD')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    
    # Format x-axis
    for ax in axs:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('technical_indicators.png')
    print("Plot saved to technical_indicators.png")

def test_multiple_indicators():
    """Test adding multiple indicators at once."""
    print("\n=== Testing Multiple Indicators ===")
    
    # Create DataLoader
    loader = my_module.DataLoader()
    loader.setDebugMode(True)
    
    # Load ticker data
    data_dir = "./src/stock_data"
    ticker = "MSFT"
    file_paths = [
        f"{data_dir}/time-series-{ticker}-5min.csv",
        f"{data_dir}/time-series-{ticker}-5min(1).csv",
        f"{data_dir}/time-series-{ticker}-5min(2).csv"
    ]
    
    success = loader.loadTickerData(ticker, file_paths)
    if not success:
        print(f"Failed to load ticker data for {ticker}")
        return False
    
    # Add multiple indicators at once
    indicators = ["sma", "ema", "rsi", "bollinger", "macd"]
    params = {
        "sma": {"column": "close", "period": "20"},
        "ema": {"column": "close", "period": "50"},
        "rsi": {"column": "close", "period": "14"},
        "bollinger": {"column": "close", "period": "20", "stdDev": "2.0"},
        "macd": {"column": "close", "fastPeriod": "12", "slowPeriod": "26", "signalPeriod": "9"}
    }
    
    # Add indicators to the ticker data
    table = loader.getTickerData(ticker)
    ti = my_module.TechnicalIndicators()
    
    # Convert params to the format expected by calculateMultipleIndicators
    params_map = {}
    for key, value in params.items():
        params_map[key] = {}
        for inner_key, inner_value in value.items():
            params_map[key][inner_key] = inner_value
    
    result_table = ti.calculateMultipleIndicators(table, indicators, params_map)
    
    # Update the ticker data in the loader
    loader.updateTickerData(ticker, result_table)
    
    # Convert to pandas DataFrame
    df = loader.arrowToPandas(ticker)
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    
    return True

def test_strategy_with_indicators():
    """Test using technical indicators with a strategy."""
    print("\n=== Testing Strategy with Indicators ===")
    
    # Create DataLoader
    loader = my_module.DataLoader()
    
    # Load ticker data
    data_dir = "./src/stock_data"
    ticker = "NVDA"
    file_paths = [
        f"{data_dir}/time-series-{ticker}-5min.csv",
        f"{data_dir}/time-series-{ticker}-5min(1).csv",
        f"{data_dir}/time-series-{ticker}-5min(2).csv"
    ]
    
    success = loader.loadTickerData(ticker, file_paths)
    if not success:
        print(f"Failed to load ticker data for {ticker}")
        return False
    
    # Add technical indicators
    table = loader.getTickerData(ticker)
    ti = my_module.TechnicalIndicators()
    
    # Add SMA indicators
    table_with_sma1 = ti.calculateSMA(table, "close", 20, "sma_20")
    table_with_sma2 = ti.calculateSMA(table_with_sma1, "close", 50, "sma_50")
    
    # Update the ticker data in the loader
    loader.updateTickerData(ticker, table_with_sma2)
    
    # Create a simple SMA crossover strategy
    class SMACrossoverStrategy(my_module.Strategy):
        def onTick(self, ticker, table, currentIndex, currentHolding):
            if currentIndex < 50:
                return None  # Not enough data
                
            # Get SMA values
            sma_20_col = table.schema().GetFieldIndex("sma_20")
            sma_50_col = table.schema().GetFieldIndex("sma_50")
            close_col = table.schema().GetFieldIndex("close")
            
            sma_20 = table.column(sma_20_col).chunk(0).Value(currentIndex)
            sma_50 = table.column(sma_50_col).chunk(0).Value(currentIndex)
            
            # Get current price
            current_price = table.column(close_col).chunk(0).Value(currentIndex)
            
            # Trading logic
            if sma_20 > sma_50 and currentHolding == 0:
                # Buy signal
                tx = my_module.Transaction()
                tx.action = my_module.Action.Buy
                tx.ticker = ticker
                tx.quantity = 1
                tx.price = current_price
                return tx
            elif sma_20 < sma_50 and currentHolding > 0:
                # Sell signal
                tx = my_module.Transaction()
                tx.action = my_module.Action.Sell
                tx.ticker = ticker
                tx.quantity = currentHolding
                tx.price = current_price
                return tx
                
            return None  # No action
    
    # Create BacktestEngine
    engine = my_module.BacktestEngine()
    engine.setDebugMode(True)
    
    # Set initial balance
    engine = 100000.0  # Uses overloaded assignment operator
    
    # Set ticker data
    engine.setTickerData(loader.getAllTickerData())
    
    # Register strategy
    engine.registerStrategy(ticker, SMACrossoverStrategy())
    
    # Run backtest
    print("\nRunning backtest...")
    engine.runBacktest()
    
    # Get performance metrics
    metrics = engine.getPerformanceMetrics()
    print("\nBacktest results:")
    print(f"Initial Balance: ${metrics.initialBalance:.2f}")
    print(f"Final Balance: ${metrics.finalBalance:.2f}")
    print(f"Total Return: {metrics.totalReturn * 100:.2f}%")
    print(f"Sharpe Ratio: {metrics.sharpeRatio:.2f}")
    print(f"Win Rate: {metrics.winRate * 100:.2f}%")
    print(f"Total Trades: {metrics.totalTrades}")
    
    return True

def main():
    """Run all tests."""
    print("=== Running Technical Indicators Tests ===")
    
    # Run tests
    tests = [
        ("Technical Indicators", test_technical_indicators),
        ("Multiple Indicators", test_multiple_indicators),
        ("Strategy with Indicators", test_strategy_with_indicators)
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