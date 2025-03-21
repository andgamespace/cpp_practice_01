import gymnasium as gym
from gymnasium import spaces
import numpy as np
import my_module
import os
import json
import threading
import time
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any

# Check if PyTorch is available
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not found. Some features will be disabled.")

class TradingEnv(gym.Env):
    """
    A vectorized trading environment for reinforcement learning.
    
    This environment uses a C++ backend for efficient backtesting and supports
    multiple tickers, technical indicators, and real-time visualization.
    
    Features:
    - Concurrent processing of multiple tickers
    - Real-time visualization through WebSocket
    - Integration with PyTorch for deep learning models
    - Comprehensive performance metrics
    - Easy-to-use Python interface
    """
    metadata = {'render_modes': ['human', 'json', 'dataframe', 'plot']}

    def __init__(self,
                 data_dir: str = "../src/stock_data",
                 tickers: List[str] = None,
                 initial_balance: float = 100000.0,
                 commission: float = 0.001,
                 visualization_port: int = 8080,
                 debug: bool = False):
        """
        Initialize the trading environment.
        
        Args:
            data_dir: Directory containing CSV files with ticker data
            tickers: List of ticker symbols to trade
            initial_balance: Starting cash balance
            commission: Trading commission as a fraction of trade value
            visualization_port: Port for the web visualization server
            debug: Enable debug logging
        """
        super(TradingEnv, self).__init__()
        
        # Set default tickers if none provided
        if tickers is None:
            self.tickers = ["AAPL", "MSFT", "NVDA", "AMD"]
        else:
            self.tickers = tickers
            
        self.data_dir = data_dir
        self.initial_balance = initial_balance
        self.commission = commission
        self.visualization_port = visualization_port
        self.debug = debug
        
        # Load data and initialize engine
        self._initialize_engine()
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-100, high=100,
            shape=(len(self.tickers),),
            dtype=np.float32
        )
        
        # Observation space includes price data and technical indicators
        obs_dim = self.engine.getObservationDimension() * len(self.tickers)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Visualization server
        self.server_running = False
        
    def _initialize_engine(self):
        """Initialize the C++ backtesting engine with data."""
        # Create engine
        self.engine = my_module.BacktestEngine()
        self.engine.setDebugMode(self.debug)
        
        # Set initial balance
        self.engine = self.initial_balance
        
        # Load data
        try:
            ticker_data = my_module.load_stock_data(self.data_dir, self.tickers)
            self.engine.setTickerData(ticker_data)
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
            
        if self.debug:
            print(f"Loaded data for {len(self.engine.getAvailableTickers())} tickers")
            
    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        self.engine.reset()
        result = self.engine.step()  # Get initial observation
        
        observation = self._process_observation(result)
        info = self._get_info(result)
        
        return observation, info

    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Array of actions, one per ticker
            
        Returns:
            observation: New observation
            reward: Reward for the action
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Convert numpy array to the format expected by the engine
        if isinstance(action, np.ndarray):
            try:
                # Try to use PyTorch tensor method if available
                result = self.engine.step_with_tensor(action)
            except AttributeError:
                # Fall back to dictionary method if PyTorch is not available
                external_actions = {ticker: float(a) for ticker, a in zip(self.tickers, action)}
                result = self.engine.step_with_action(external_actions)
        else:
            # Handle dictionary actions if provided
            external_actions = {ticker: float(a) for ticker, a in zip(self.tickers, action)}
            result = self.engine.step_with_action(external_actions)
        
        # Process results
        observation = self._process_observation(result)
        reward = result.reward
        terminated = result.done
        truncated = False
        info = self._get_info(result)
        
        return observation, reward, terminated, truncated, info

    def _process_observation(self, result):
        """
        Convert the step result to a numpy array observation.
        
        Args:
            result: StepResult from the engine
            
        Returns:
            numpy array of observations
        """
        # Extract features for each ticker
        all_features = []
        
        for ticker in self.tickers:
            # Get price data
            price = result.observations.get(ticker, 0.0)
            
            # Get additional features if available
            ticker_features = result.features.get(ticker, [])
            
            # Combine price with features
            if not ticker_features:
                ticker_features = [price]
            
            all_features.extend(ticker_features)
            
        return np.array(all_features, dtype=np.float32)

    def _get_info(self, result):
        """
        Get additional information about the current state.
        
        Args:
            result: StepResult from the engine
            
        Returns:
            Dictionary of information
        """
        metrics = self.engine.getPerformanceMetrics()
        
        return {
            "cash_balance": self.engine.getCashBalance(),
            "holdings": self.engine.getHoldings(),
            "total_return": metrics.totalReturn,
            "sharpe_ratio": metrics.sharpeRatio,
            "win_rate": metrics.winRate
        }

    def render(self, mode="human"):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human', 'json', 'dataframe', or 'plot')
            
        Returns:
            Rendering result based on the mode
        """
        if mode == "human":
            return self.engine.getPortfolioMetrics()
        elif mode == "json":
            return self.engine.getPortfolioMetricsJson()
        elif mode == "dataframe":
            # Convert portfolio metrics to pandas DataFrame
            metrics_json = self.engine.getPortfolioMetricsJson()
            metrics_dict = json.loads(metrics_json)
            
            # Create main metrics DataFrame
            main_metrics = {k: v for k, v in metrics_dict.items() if k != 'holdings' and k != 'performance'}
            df_metrics = pd.DataFrame([main_metrics])
            
            # Create holdings DataFrame
            if 'holdings' in metrics_dict and metrics_dict['holdings']:
                df_holdings = pd.DataFrame(metrics_dict['holdings'])
            else:
                df_holdings = pd.DataFrame()
            
            # Create performance DataFrame
            if 'performance' in metrics_dict:
                df_performance = pd.DataFrame([metrics_dict['performance']])
            else:
                df_performance = pd.DataFrame()
            
            return {
                'metrics': df_metrics,
                'holdings': df_holdings,
                'performance': df_performance
            }
        elif mode == "plot":
            # Create a simple matplotlib plot of portfolio value
            metrics_json = self.engine.getPortfolioMetricsJson()
            metrics_dict = json.loads(metrics_json)
            
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot 1: Portfolio Value
            ax[0].set_title('Portfolio Value')
            ax[0].bar(['Initial', 'Current'],
                     [metrics_dict.get('initialBalance', 0), metrics_dict.get('cash', 0)],
                     color=['blue', 'green'])
            ax[0].set_ylabel('Value ($)')
            ax[0].grid(axis='y', linestyle='--', alpha=0.7)
            
            # Plot 2: Win/Loss Ratio
            ax[1].set_title('Trading Performance')
            wins = metrics_dict.get('wins', 0)
            losses = metrics_dict.get('losses', 0)
            ax[1].pie([wins, losses], labels=['Wins', 'Losses'], autopct='%1.1f%%',
                     colors=['green', 'red'], startangle=90)
            
            plt.tight_layout()
            return fig
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
            
    def start_visualization_server(self):
        """Start the web visualization server."""
        if not self.server_running:
            my_module.start_server(self.visualization_port, self.engine)
            self.server_running = True
            print(f"Visualization server started on port {self.visualization_port}")
            print(f"Open http://localhost:{self.visualization_port} in your browser")
            
    def stop_visualization_server(self):
        """Stop the web visualization server."""
        if self.server_running:
            my_module.stop_server()
            self.server_running = False
            print("Visualization server stopped")
            
    def close(self):
        """Clean up resources."""
        self.stop_visualization_server()
        super().close()
        
    def run_backtest(self):
        """Run a full backtest using the engine."""
        self.engine.runBacktest()
        return self.engine.getPerformanceMetrics()
        
    def get_ticker_data(self, ticker):
        """
        Get the data for a specific ticker as a pandas DataFrame.
        
        Args:
            ticker: The ticker symbol
            
        Returns:
            pandas DataFrame with the ticker data
        """
        table = self.engine.getTickerData(ticker)
        if not table:
            raise ValueError(f"No data available for ticker: {ticker}")
        
        # Convert Arrow table to pandas DataFrame
        return self.engine.arrowToPandas(ticker)
    
    def plot_ticker(self, ticker, start_idx=0, end_idx=None, figsize=(12, 6)):
        """
        Plot the price chart for a specific ticker.
        
        Args:
            ticker: The ticker symbol
            start_idx: Start index for the plot
            end_idx: End index for the plot
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        df = self.get_ticker_data(ticker)
        if end_idx is None:
            end_idx = len(df)
        
        df_slice = df.iloc[start_idx:end_idx]
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df_slice['datetime'], df_slice['close'], label='Close')
        ax.plot(df_slice['datetime'], df_slice['open'], label='Open', alpha=0.7)
        ax.fill_between(df_slice['datetime'], df_slice['low'], df_slice['high'], alpha=0.2, color='gray')
        
        ax.set_title(f'{ticker} Price Chart')
        ax.set_xlabel('Date/Time')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def get_transactions(self):
        """
        Get all transactions as a pandas DataFrame.
        
        Returns:
            pandas DataFrame with all transactions
        """
        transactions = self.engine.getTransactions()
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'datetime': tx.datetime,
                'ticker': tx.ticker,
                'action': str(tx.action).split('.')[-1],  # Convert enum to string
                'quantity': tx.quantity,
                'price': tx.price,
                'value': tx.price * tx.quantity
            }
            for tx in transactions
        ])
        
        return df
    
    def plot_performance(self, figsize=(15, 10)):
        """
        Plot comprehensive performance metrics.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib figure with performance plots
        """
        metrics = self.engine.getPerformanceMetrics()
        transactions = self.get_transactions()
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Portfolio Value
        axs[0, 0].set_title('Portfolio Value')
        axs[0, 0].bar(['Initial', 'Final'],
                     [metrics.initialBalance, metrics.finalBalance],
                     color=['blue', 'green'])
        axs[0, 0].set_ylabel('Value ($)')
        axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 2: Win/Loss Ratio
        axs[0, 1].set_title('Trading Performance')
        axs[0, 1].pie([metrics.winningTrades, metrics.losingTrades],
                     labels=['Wins', 'Losses'],
                     autopct='%1.1f%%',
                     colors=['green', 'red'],
                     startangle=90)
        
        # Plot 3: Key Metrics
        metrics_data = {
            'Total Return': metrics.totalReturn * 100,
            'Annualized Return': metrics.annualizedReturn * 100,
            'Sharpe Ratio': metrics.sharpeRatio,
            'Max Drawdown': metrics.maxDrawdown * 100
        }
        
        axs[1, 0].set_title('Key Performance Metrics')
        bars = axs[1, 0].bar(metrics_data.keys(), metrics_data.values())
        axs[1, 0].set_ylabel('Value')
        axs[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axs[1, 0].annotate(f'{height:.2f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom')
        
        # Plot 4: Trade Distribution
        if not transactions.empty:
            try:
                transactions['date'] = pd.to_datetime(transactions['datetime'])
                transactions.set_index('date', inplace=True)
                
                # Group by day and count trades
                daily_trades = transactions.resample('D').size()
                
                axs[1, 1].set_title('Trade Distribution')
                axs[1, 1].plot(daily_trades.index, daily_trades.values, marker='o')
                axs[1, 1].set_ylabel('Number of Trades')
                axs[1, 1].set_xlabel('Date')
                axs[1, 1].grid(True, alpha=0.3)
                plt.setp(axs[1, 1].xaxis.get_majorticklabels(), rotation=45)
            except Exception as e:
                axs[1, 1].set_title(f'Error plotting trades: {str(e)}')
        else:
            axs[1, 1].set_title('No Transactions Available')
        
        plt.tight_layout()
        return fig


# Example usage in a Jupyter notebook
def create_trading_env(data_dir="../src/stock_data",
                      tickers=None,
                      initial_balance=100000.0,
                      commission=0.001,
                      visualize=True,
                      visualization_port=8080,
                      debug=False,
                      auto_run_backtest=False):
    """
    Create and initialize a trading environment.
    
    Args:
        data_dir: Directory containing CSV files with ticker data
        tickers: List of ticker symbols to trade
        initial_balance: Starting cash balance
        commission: Trading commission as a fraction of trade value
        visualize: Whether to start the visualization server
        visualization_port: Port for the web visualization server
        debug: Enable debug logging
        auto_run_backtest: Whether to automatically run a backtest after initialization
        
    Returns:
        Initialized TradingEnv instance
    """
    # Create environment
    env = TradingEnv(
        data_dir=data_dir,
        tickers=tickers,
        initial_balance=initial_balance,
        commission=commission,
        visualization_port=visualization_port,
        debug=debug
    )
    
    # Start visualization server if requested
    if visualize:
        env.start_visualization_server()
    
    # Run backtest if requested
    if auto_run_backtest:
        print("Running backtest...")
        metrics = env.run_backtest()
        print(f"Backtest completed with {metrics.totalReturn * 100:.2f}% return")
        
        # Print key metrics
        print(f"Initial Balance: ${metrics.initialBalance:.2f}")
        print(f"Final Balance: ${metrics.finalBalance:.2f}")
        print(f"Sharpe Ratio: {metrics.sharpeRatio:.2f}")
        print(f"Win Rate: {metrics.winRate * 100:.2f}%")
        print(f"Total Trades: {metrics.totalTrades}")
        
    return env


def create_pytorch_model(input_dim, output_dim, hidden_dims=[64, 32]):
    """
    Create a simple PyTorch model for the trading environment.
    
    Args:
        input_dim: Input dimension (observation space)
        output_dim: Output dimension (action space)
        hidden_dims: List of hidden layer dimensions
        
    Returns:
        PyTorch model
    """
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is not available. Please install PyTorch to use this function.")
    
    layers = []
    prev_dim = input_dim
    
    # Add hidden layers
    for dim in hidden_dims:
        layers.append(torch.nn.Linear(prev_dim, dim))
        layers.append(torch.nn.ReLU())
        prev_dim = dim
    
    # Add output layer
    layers.append(torch.nn.Linear(prev_dim, output_dim))
    layers.append(torch.nn.Tanh())  # Output between -1 and 1
    
    # Create model
    model = torch.nn.Sequential(*layers)
    
    print(f"Created PyTorch model with architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {output_dim}")
    return model
