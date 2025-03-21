import gymnasium as gym
from gymnasium import spaces
import numpy as np
import my_module
import os
import json
import logging
import time
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("trading_env")

# Check if PyTorch is available
try:
    import torch
    PYTORCH_AVAILABLE = True
    logger.info("PyTorch is available and will be used for tensor operations")
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch is not available. Some features will be disabled.")

class TradingEnv(gym.Env):
    """
    A vectorized trading environment for reinforcement learning.
    
    This environment uses a C++ backend for efficient backtesting and supports
    multiple tickers, technical indicators, and real-time visualization.
    
    Features:
    - Concurrent processing of multiple tickers using Taskflow
    - Real-time visualization through WebSocket
    - Integration with PyTorch for deep learning models
    - Comprehensive performance metrics
    - Technical indicators for feature engineering
    - Easy-to-use Python interface
    """
    metadata = {'render_modes': ['human', 'json', 'dataframe', 'plot', 'rgb_array']}
    
    def __init__(self,
                 data_dir: str = "../src/stock_data",
                 tickers: List[str] = None,
                 initial_balance: float = 100000.0,
                 commission: float = 0.001,
                 visualization_port: int = 8080,
                 debug: bool = False,
                 risk_aversion: float = 0.1,
                 reward_scaling: float = 1.0,
                 window_size: int = 20,
                 max_position_size: float = 1.0,
                 technical_indicators: Optional[Dict[str, Dict[str, str]]] = None):
        """
        Initialize the trading environment.
        
        Args:
            data_dir: Directory containing CSV files with ticker data
            tickers: List of ticker symbols to trade
            initial_balance: Starting cash balance
            commission: Trading commission as a fraction of trade value
            visualization_port: Port for the web visualization server
            debug: Enable debug logging
            risk_aversion: Risk aversion parameter for reward calculation
            reward_scaling: Scaling factor for rewards
            window_size: Number of past observations to include in the state
            max_position_size: Maximum position size as a fraction of portfolio value
            technical_indicators: Dictionary of technical indicators to add to the data
                Format: {"indicator_name": {"param1": "value1", "param2": "value2"}}
                Example: {"sma": {"column": "close", "period": "20"}}
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
        self.risk_aversion = risk_aversion
        self.reward_scaling = reward_scaling
        self.window_size = window_size
        self.max_position_size = max_position_size
        self.technical_indicators = technical_indicators or {}
        
        # Set up logging
        if debug:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"Initializing TradingEnv with {len(self.tickers)} tickers")
        
        # Load data and initialize engine
        self._initialize_engine()
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(len(self.tickers),),
            dtype=np.float32
        )
        
        # Observation space includes price data and technical indicators
        obs_dim = self._calculate_observation_dimension()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        logger.info(f"Observation space dimension: {obs_dim}")
        logger.info(f"Action space dimension: {len(self.tickers)}")
        
        # Visualization server
        self.server_running = False
        
        # Performance tracking
        self.episode_returns = []
        self.episode_sharpe_ratios = []
        self.episode_drawdowns = []
        self.best_return = -np.inf
        self.best_sharpe = -np.inf
        
        # Save configuration
        self.config = {
            "data_dir": data_dir,
            "tickers": self.tickers,
            "initial_balance": initial_balance,
            "commission": commission,
            "visualization_port": visualization_port,
            "debug": debug,
            "risk_aversion": risk_aversion,
            "reward_scaling": reward_scaling,
            "window_size": window_size,
            "max_position_size": max_position_size,
            "technical_indicators": self.technical_indicators
        }
    
    def _calculate_observation_dimension(self):
        """Calculate the dimension of the observation space."""
        # Base features: OHLCV for each ticker + portfolio state
        base_dim = len(self.tickers) * 5 + 2  # 5 features per ticker + cash + portfolio value
        
        # Add dimensions for technical indicators
        indicator_dim = 0
        for indicator, params in self.technical_indicators.items():
            if indicator in ["sma", "ema", "rsi", "roc", "stddev", "atr"]:
                indicator_dim += len(self.tickers)
            elif indicator == "bollinger":
                indicator_dim += 3 * len(self.tickers)  # Upper, middle, lower bands
            elif indicator == "macd":
                indicator_dim += 3 * len(self.tickers)  # MACD, signal, histogram
            elif indicator == "stochastic":
                indicator_dim += 2 * len(self.tickers)  # %K and %D
            # Add more indicators as needed
        
        return base_dim + indicator_dim
        
    def _initialize_engine(self):
        """Initialize the C++ backtesting engine with data."""
        # Create engine
        self.engine = my_module.BacktestEngine()
        self.engine.setDebugMode(self.debug)
        
        # Create data loader
        self.loader = my_module.DataLoader()
        self.loader.setDebugMode(self.debug)
        
        # Load data
        try:
            self._load_data()
            
            # Add technical indicators if specified
            if self.technical_indicators:
                self._add_technical_indicators()
            
            # Set ticker data in the engine
            self.engine.setTickerData(self.loader.getAllTickerData())
            
        except Exception as e:
            logger.error(f"Error initializing engine: {e}")
            raise
            
        if self.debug:
            logger.debug(f"Loaded data for {len(self.engine.getAvailableTickers())} tickers")
    
    def _load_data(self):
        """Load ticker data from CSV files."""
        ticker_file_paths = {}
        
        for ticker in self.tickers:
            # Look for CSV files for this ticker
            file_paths = []
            for i in range(3):  # Try with suffixes (1) and (2)
                suffix = f"({i})" if i > 0 else ""
                file_path = os.path.join(self.data_dir, f"time-series-{ticker}-5min{suffix}.csv")
                if os.path.exists(file_path):
                    file_paths.append(file_path)
            
            if file_paths:
                ticker_file_paths[ticker] = file_paths
        
        if not ticker_file_paths:
            error_msg = f"No ticker data found in {self.data_dir}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Load all tickers
        logger.info(f"Loading data for {len(ticker_file_paths)} tickers")
        loaded = self.loader.loadMultipleTickers(ticker_file_paths)
        if loaded == 0:
            error_msg = "Failed to load any ticker data"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Update tickers list to only include those that were loaded
        self.tickers = self.loader.getAvailableTickers()
        
        logger.info(f"Successfully loaded {loaded} tickers: {self.tickers}")
    
    def _add_technical_indicators(self):
        """Add technical indicators to the ticker data."""
        logger.info("Adding technical indicators")
        
        # Create TechnicalIndicators instance
        ti = my_module.TechnicalIndicators()
        ti.setDebugMode(self.debug)
        
        # Process each ticker
        for ticker in self.tickers:
            logger.info(f"Adding indicators for {ticker}")
            
            # Get the Arrow table
            table = self.loader.getTickerData(ticker)
            if not table:
                logger.warning(f"No data found for ticker {ticker}, skipping")
                continue
            
            # Convert indicators to the format expected by calculateMultipleIndicators
            indicators = list(self.technical_indicators.keys())
            params_map = {}
            for key, value in self.technical_indicators.items():
                params_map[key] = {}
                for inner_key, inner_value in value.items():
                    params_map[key][inner_key] = inner_value
            
            # Calculate indicators
            try:
                result_table = ti.calculateMultipleIndicators(table, indicators, params_map)
                
                # Update the ticker data in the loader
                self.loader.updateTickerData(ticker, result_table)
                
                logger.info(f"Successfully added indicators for {ticker}")
            except Exception as e:
                logger.error(f"Error adding indicators for {ticker}: {e}")
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        
        Args:
            seed: Random seed
            options: Additional options
                - initial_balance: Override the initial balance
                - commission: Override the commission rate
                - window_size: Override the window size
                
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Apply options if provided
        if options:
            if 'initial_balance' in options:
                self.initial_balance = options['initial_balance']
            if 'commission' in options:
                self.commission = options['commission']
            if 'window_size' in options:
                self.window_size = options['window_size']
        
        # Reset the engine
        self.engine.reset()
        
        # Get initial observation
        result = self.engine.step()
        
        # Process observation
        observation = self._process_observation(result)
        
        # Additional info
        info = self._get_info(result)
        
        logger.debug(f"Reset environment. Initial observation shape: {observation.shape}")
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Array of actions, one per ticker
                   Values between -1 and 1, where:
                   - Positive values indicate buying
                   - Negative values indicate selling
                   - Zero indicates holding
            
        Returns:
            observation: New observation
            reward: Reward for the action
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Scale actions by max position size
        scaled_action = action * self.max_position_size
        
        # Convert numpy array to the format expected by the engine
        if isinstance(action, np.ndarray):
            if PYTORCH_AVAILABLE:
                try:
                    # Try to use PyTorch tensor method if available
                    tensor_action = torch.from_numpy(scaled_action)
                    result = self.engine.step_with_tensor(tensor_action)
                except Exception:
                    # Fall back to dictionary method
                    action_map = {ticker: float(a) for ticker, a in zip(self.tickers, scaled_action)}
                    result = self.engine.step_with_action(action_map)
            else:
                # Use dictionary method
                action_map = {ticker: float(a) for ticker, a in zip(self.tickers, scaled_action)}
                result = self.engine.step_with_action(action_map)
        elif PYTORCH_AVAILABLE and isinstance(action, torch.Tensor):
            # Scale PyTorch tensor
            scaled_tensor = action * self.max_position_size
            # Take step with PyTorch tensor
            result = self.engine.step_with_tensor(scaled_tensor)
        else:
            # Handle other action types
            action_map = {ticker: float(a) for ticker, a in zip(self.tickers, scaled_action)}
            result = self.engine.step_with_action(action_map)
        
        # Process results
        observation = self._process_observation(result)
        reward = result.reward * self.reward_scaling
        terminated = result.done
        truncated = False
        
        # Get portfolio metrics
        metrics = self.engine.getPortfolioMetrics()
        
        # Calculate additional reward components
        if len(metrics) > 1:
            # Calculate return volatility
            returns = []
            for i in range(1, len(metrics)):
                prev_value = metrics[i-1]['portfolioValue']
                curr_value = metrics[i]['portfolioValue']
                if prev_value > 0:
                    returns.append((curr_value - prev_value) / prev_value)
            
            if returns:
                volatility = np.std(returns)
                # Penalize for volatility based on risk aversion
                reward -= volatility * self.risk_aversion
        
        # Additional info
        info = self._get_info(result)
        
        logger.debug(f"Step taken. Reward: {reward}, Done: {terminated}")
        
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
        
        # Add portfolio state
        all_features.append(self.engine.getCashBalance())
        
        # Get portfolio value
        portfolio_metrics = self.engine.getPortfolioMetrics()
        if portfolio_metrics:
            portfolio_value = portfolio_metrics[-1]['portfolioValue']
        else:
            portfolio_value = self.initial_balance
        
        all_features.append(portfolio_value)
        
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
        
        # Get portfolio metrics
        portfolio_metrics = self.engine.getPortfolioMetrics()
        portfolio_value = portfolio_metrics[-1]['portfolioValue'] if portfolio_metrics else self.initial_balance
        
        return {
            "cash_balance": self.engine.getCashBalance(),
            "portfolio_value": portfolio_value,
            "holdings": self.engine.getHoldings(),
            "total_return": metrics.totalReturn,
            "sharpe_ratio": metrics.sharpeRatio,
            "win_rate": metrics.winRate,
            "max_drawdown": metrics.maxDrawdown,
            "timestamp": datetime.now().isoformat()
        }
    
    def render(self, mode="human"):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human', 'json', 'dataframe', 'plot', or 'rgb_array')
            
        Returns:
            Rendering result based on the mode
        """
        if mode == "human":
            # Start visualization server if not already running
            if not self.server_running:
                self.start_visualization_server()
            return None
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
            # Create a plot of portfolio performance
            fig = self.plot_performance()
            return fig
        elif mode == "rgb_array":
            # Generate a plot and convert to RGB array
            fig = self.plot_performance()
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            return img
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
            
    def start_visualization_server(self):
        """Start the web visualization server."""
        if not self.server_running:
            logger.info(f"Starting visualization server on port {self.visualization_port}")
            my_module.start_server(self.visualization_port, self.engine)
            self.server_running = True
            logger.info(f"Visualization server started on port {self.visualization_port}")
            logger.info(f"Open http://localhost:{self.visualization_port} in your browser")
            
    def stop_visualization_server(self):
        """Stop the web visualization server."""
        if self.server_running:
            logger.info("Stopping visualization server")
            my_module.stop_server()
            self.server_running = False
            logger.info("Visualization server stopped")
            
    def close(self):
        """Clean up resources."""
        self.stop_visualization_server()
        super().close()
        
    def run_backtest(self):
        """
        Run a full backtest using the engine.
        
        Returns:
            Performance metrics from the backtest
        """
        logger.info("Running backtest")
        self.engine.runBacktest()
        metrics = self.engine.getPerformanceMetrics()
        
        # Log key metrics
        logger.info(f"Backtest completed with {metrics.totalReturn * 100:.2f}% return")
        logger.info(f"Sharpe Ratio: {metrics.sharpeRatio:.2f}")
        logger.info(f"Win Rate: {metrics.winRate * 100:.2f}%")
        logger.info(f"Total Trades: {metrics.totalTrades}")
        
        return metrics
        
    def get_ticker_data(self, ticker):
        """
        Get the data for a specific ticker as a pandas DataFrame.
        
        Args:
            ticker: The ticker symbol
            
        Returns:
            pandas DataFrame with the ticker data
        """
        if not self.loader.hasTickerData(ticker):
            raise ValueError(f"No data available for ticker: {ticker}")
        
        # Convert Arrow table to pandas DataFrame
        return self.loader.arrowToPandas(ticker)
    
    def plot_ticker(self, ticker, start_idx=0, end_idx=None, figsize=(12, 8)):
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
        
        # Check if we have any indicators
        indicator_columns = [col for col in df_slice.columns if any(
            ind in col for ind in ['sma', 'ema', 'rsi', 'macd', 'bband', 'atr'])]
        
        # Create figure with subplots
        n_plots = 3 if indicator_columns else 2
        fig, axs = plt.subplots(n_plots, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1, 2] if n_plots == 3 else [3, 1]})
        
        # Plot 1: Price and Moving Averages
        axs[0].set_title(f'{ticker} Price Chart')
        axs[0].plot(df_slice['datetime'], df_slice['close'], label='Close', color='black')
        axs[0].plot(df_slice['datetime'], df_slice['open'], label='Open', alpha=0.7, color='blue')
        axs[0].fill_between(df_slice['datetime'], df_slice['low'], df_slice['high'], alpha=0.2, color='gray', label='Range')
        
        # Add technical indicators if available
        sma_cols = [col for col in df_slice.columns if 'sma' in col]
        for col in sma_cols:
            period = col.split('_')[-1]
            axs[0].plot(df_slice['datetime'], df_slice[col], label=f'SMA({period})')
        
        ema_cols = [col for col in df_slice.columns if 'ema' in col]
        for col in ema_cols:
            period = col.split('_')[-1]
            axs[0].plot(df_slice['datetime'], df_slice[col], label=f'EMA({period})')
        
        # Add Bollinger Bands if available
        if all(col in df_slice.columns for col in ['close_bband_upper', 'close_bband_lower']):
            axs[0].fill_between(df_slice['datetime'], df_slice['close_bband_upper'], df_slice['close_bband_lower'], 
                              color='purple', alpha=0.1, label='Bollinger Bands')
        
        axs[0].set_ylabel('Price')
        axs[0].legend(loc='upper left')
        axs[0].grid(True, alpha=0.3)
        
        # Plot 2: Volume
        axs[1].set_title('Volume')
        axs[1].bar(df_slice['datetime'], df_slice['volume'], color='green', alpha=0.5)
        axs[1].set_ylabel('Volume')
        axs[1].grid(True, alpha=0.3)
        
        # Plot 3: Technical Indicators (if available)
        if n_plots == 3:
            # Check for RSI
            if 'close_rsi_14' in df_slice.columns:
                axs[2].set_title('RSI')
                axs[2].plot(df_slice['datetime'], df_slice['close_rsi_14'], color='purple')
                axs[2].axhline(y=70, color='r', linestyle='--', alpha=0.5)
                axs[2].axhline(y=30, color='g', linestyle='--', alpha=0.5)
                axs[2].set_ylabel('RSI')
                axs[2].set_ylim(0, 100)
            # Check for MACD
            elif any(col.startswith('close_macd') for col in df_slice.columns):
                axs[2].set_title('MACD')
                macd_col = next((col for col in df_slice.columns if col == 'close_macd'), None)
                signal_col = next((col for col in df_slice.columns if col == 'close_macd_signal'), None)
                hist_col = next((col for col in df_slice.columns if col == 'close_macd_histogram'), None)
                
                if macd_col:
                    axs[2].plot(df_slice['datetime'], df_slice[macd_col], label='MACD', color='blue')
                if signal_col:
                    axs[2].plot(df_slice['datetime'], df_slice[signal_col], label='Signal', color='red')
                if hist_col:
                    axs[2].bar(df_slice['datetime'], df_slice[hist_col], label='Histogram', color='gray', alpha=0.5)
                
                axs[2].set_ylabel('MACD')
                axs[2].legend()
            else:
                axs[2].set_title('Technical Indicators')
                axs[2].text(0.5, 0.5, 'No RSI or MACD data available', 
                          horizontalalignment='center', verticalalignment='center',
                          transform=axs[2].transAxes)
            
            axs[2].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axs:
            ax.tick_params(axis='x', rotation=45)
        
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
        
        # Add value labels on bars
        for i, v in enumerate([metrics.initialBalance, metrics.finalBalance]):
            axs[0, 0].text(i, v + 0.01 * v, f'${v:.2f}', ha='center')
        
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
    
    def add_technical_indicators(self, indicators):
        """
        Add technical indicators to the ticker data.
        
        Args:
            indicators: Dictionary of technical indicators to add
                Format: {"indicator_name": {"param1": "value1", "param2": "value2"}}
                Example: {"sma": {"column": "close", "period": "20"}}
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Adding technical indicators")
        
        # Update technical indicators dictionary
        self.technical_indicators.update(indicators)
        
        # Create TechnicalIndicators instance
        ti = my_module.TechnicalIndicators()
        ti.setDebugMode(self.debug)
        
        # Process each ticker
        for ticker in self.tickers:
            logger.info(f"Adding indicators for {ticker}")
            
            # Get the Arrow table
            table = self.loader.getTickerData(ticker)
            if not table:
                logger.warning(f"No data found for ticker {ticker}, skipping")
                continue
            
            # Convert indicators to the format expected by calculateMultipleIndicators
            indicator_list = list(indicators.keys())
            params_map = {}
            for key, value in indicators.items():
                params_map[key] = {}
                for inner_key, inner_value in value.items():
                    params_map[key][inner_key] = inner_value
            
            # Calculate indicators
            try:
                result_table = ti.calculateMultipleIndicators(table, indicator_list, params_map)
                
                # Update the ticker data in the loader
                self.loader.updateTickerData(ticker, result_table)
                
                logger.info(f"Successfully added indicators for {ticker}")
            except Exception as e:
                logger.error(f"Error adding indicators for {ticker}: {e}")
                return False
        
        # Update the engine with the new data
        self.engine.setTickerData(self.loader.getAllTickerData())
        
        # Update observation space dimension
        obs_dim = self._calculate_observation_dimension()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        logger.info(f"Updated observation space dimension: {obs_dim}")
        
        return True
    
    def save_model(self, model, path="trading_model.pt"):
        """
        Save a PyTorch model.
        
        Args:
            model: PyTorch model to save
            path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        if not PYTORCH_AVAILABLE:
            logger.error("PyTorch is not available. Cannot save model.")
            return False
        
        try:
            torch.save(model.state_dict(), path)
            logger.info(f"Model saved to {path}")
            
            # Save configuration alongside the model
            config_path = os.path.splitext(path)[0] + "_config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Model configuration saved to {config_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, model, path="trading_model.pt"):
        """
        Load a PyTorch model.
        
        Args:
            model: PyTorch model to load into
            path: Path to load the model from
            
        Returns:
            Loaded PyTorch model or None if failed
        """
        if not PYTORCH_AVAILABLE:
            logger.error("PyTorch is not available. Cannot load model.")
            return None
        
        try:
            model.load_state_dict(torch.load(path))
            logger.info(f"Model loaded from {path}")
            
            # Load configuration if available
            config_path = os.path.splitext(path)[0] + "_config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Model configuration loaded from {config_path}")
            
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def evaluate_model(self, model, episodes=1):
        """
        Evaluate a PyTorch model on the environment.
        
        Args:
            model: PyTorch model to evaluate
            episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not PYTORCH_AVAILABLE:
            logger.error("PyTorch is not available. Cannot evaluate model.")
            return {}
        
        logger.info(f"Evaluating model for {episodes} episodes")
        
        returns = []
        sharpe_ratios = []
        max_drawdowns = []
        win_rates = []
        
        for episode in range(episodes):
            obs, _ = self.reset()
            done = False
            episode_rewards = []
            
            while not done:
                # Convert observation to tensor
                obs_tensor = torch.FloatTensor(obs)
                
                # Get action from model
                with torch.no_grad():
                    action = model(obs_tensor).numpy()
                
                # Take step in environment
                obs, reward, done, _, info = self.step(action)
                episode_rewards.append(reward)
            
            # Run backtest to get metrics
            metrics = self.run_backtest()
            
            # Record metrics
            returns.append(metrics.totalReturn)
            sharpe_ratios.append(metrics.sharpeRatio)
            max_drawdowns.append(metrics.maxDrawdown)
            win_rates.append(metrics.winRate)
            
            logger.info(f"Episode {episode+1}/{episodes}: Return = {metrics.totalReturn * 100:.2f}%, Sharpe = {metrics.sharpeRatio:.2f}")
        
        # Calculate average metrics
        avg_return = np.mean(returns)
        avg_sharpe = np.mean(sharpe_ratios)
        avg_drawdown = np.mean(max_drawdowns)
        avg_win_rate = np.mean(win_rates)
        
        logger.info(f"Evaluation complete. Average return: {avg_return * 100:.2f}%, Average Sharpe: {avg_sharpe:.2f}")
        
        return {
            "avg_return": avg_return,
            "avg_sharpe_ratio": avg_sharpe,
            "avg_max_drawdown": avg_drawdown,
            "avg_win_rate": avg_win_rate,
            "returns": returns,
            "sharpe_ratios": sharpe_ratios,
            "max_drawdowns": max_drawdowns,
            "win_rates": win_rates
        }


# Example usage in a Jupyter notebook
def create_trading_env(data_dir="../src/stock_data",
                      tickers=None,
                      initial_balance=100000.0,
                      commission=0.001,
                      visualize=True,
                      visualization_port=8080,
                      debug=False,
                      auto_run_backtest=False,
                      risk_aversion=0.1,
                      reward_scaling=1.0,
                      window_size=20,
                      max_position_size=1.0,
                      technical_indicators=None):
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
        risk_aversion: Risk aversion parameter for reward calculation
        reward_scaling: Scaling factor for rewards
        window_size: Number of past observations to include in the state
        max_position_size: Maximum position size as a fraction of portfolio value
        technical_indicators: Dictionary of technical indicators to add to the data
            Format: {"indicator_name": {"param1": "value1", "param2": "value2"}}
            Example: {"sma": {"column": "close", "period": "20"}}
        
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
        debug=debug,
        risk_aversion=risk_aversion,
        reward_scaling=reward_scaling,
        window_size=window_size,
        max_position_size=max_position_size,
        technical_indicators=technical_indicators
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


def train_dqn_model(env, model=None, episodes=100, learning_rate=0.001, gamma=0.99, 
                   epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                   batch_size=64, target_update=10, memory_size=10000,
                   save_path="trading_model.pt"):
    """
    Train a DQN model on the trading environment.
    
    Args:
        env: Trading environment
        model: PyTorch model (if None, a new model will be created)
        episodes: Number of episodes to train
        learning_rate: Learning rate for the optimizer
        gamma: Discount factor
        epsilon_start: Starting epsilon for epsilon-greedy exploration
        epsilon_end: Minimum epsilon value
        epsilon_decay: Epsilon decay rate
        batch_size: Batch size for training
        target_update: Number of episodes between target network updates
        memory_size: Size of the replay buffer
        save_path: Path to save the trained model
        
    Returns:
        Trained PyTorch model
    """
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is not available. Please install PyTorch to use this function.")
    
    from collections import deque
    import random
    
    # Create model if not provided
    if model is None:
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.shape[0]
        model = create_pytorch_model(input_dim, output_dim)
    
    # Create target model
    target_model = create_pytorch_model(
        env.observation_space.shape[0],
        env.action_space.shape[0]
    )
    target_model.load_state_dict(model.state_dict())
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create replay buffer
    replay_buffer = deque(maxlen=memory_size)
    
    # Training metrics
    epsilon = epsilon_start
    best_return = -np.inf
    
    # Training loop
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                obs_tensor = torch.FloatTensor(obs)
                with torch.no_grad():
                    action = model(obs_tensor).numpy()
            
            # Take step in environment
            next_obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            
            # Store transition in replay buffer
            replay_buffer.append((obs, action, reward, next_obs, done))
            
            # Update observation
            obs = next_obs
            
            # Train model if enough samples
            if len(replay_buffer) >= batch_size:
                # Sample batch
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                # Convert to tensors
                states = torch.FloatTensor(np.array(states))
                actions = torch.FloatTensor(np.array(actions))
                rewards = torch.FloatTensor(np.array(rewards))
                next_states = torch.FloatTensor(np.array(next_states))
                dones = torch.FloatTensor(np.array(dones))
                
                # Compute current Q values
                current_q = torch.sum(model(states) * actions, dim=1)
                
                # Compute target Q values
                with torch.no_grad():
                    next_q = torch.max(target_model(next_states), dim=1)[0]
                    target_q = rewards + gamma * next_q * (1 - dones)
                
                # Compute loss
                loss = torch.nn.functional.mse_loss(current_q, target_q)
                
                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Update target model
        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Run backtest to get metrics
        metrics = env.run_backtest()
        
        # Save best model
        if metrics.totalReturn > best_return:
            best_return = metrics.totalReturn
            env.save_model(model, save_path)
            print(f"New best model saved with return: {best_return * 100:.2f}%")
        
        print(f"Episode {episode+1}/{episodes}: Return = {metrics.totalReturn * 100:.2f}%, Sharpe = {metrics.sharpeRatio:.2f}, Epsilon = {epsilon:.4f}")
    
    return model