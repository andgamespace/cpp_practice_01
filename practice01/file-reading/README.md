# Trading Environment for Deep Reinforcement Learning

A high-performance trading environment for developing, testing, and training deep reinforcement learning models for algorithmic trading. This project combines the speed of C++ with the ease of use of Python, and provides a React frontend for real-time visualization.

![Trading Environment Architecture](docs/architecture.png)

## Features

- **Vectorized Environment**: Efficiently process multiple tickers concurrently
- **Taskflow Integration**: Parallel processing for improved performance with automatic thread management
- **PyTorch Support**: Seamless integration with PyTorch for deep learning models
- **WebSocket Communication**: Real-time updates between backend and frontend
- **Gymnasium Interface**: Compatible with OpenAI Gym/Gymnasium for reinforcement learning
- **CSV Data Loading**: Easily load and concatenate multiple CSV files per ticker
- **Technical Indicators**: Over 20 built-in technical indicators for feature engineering
- **Performance Metrics**: Comprehensive trading metrics and visualization
- **React Frontend**: Real-time visualization of trading performance
- **Cross-Platform**: Works on both macOS (including M1/M2) and Linux with CUDA support

## Table of Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Building from Source](#building-from-source)
  - [Server Deployment](#server-deployment)
- [Usage](#usage)
  - [Python API](#python-api)
  - [Technical Indicators](#technical-indicators)
  - [Creating Custom Strategies](#creating-custom-strategies)
  - [Using with PyTorch](#using-with-pytorch)
  - [Visualization](#visualization)
  - [Deep Reinforcement Learning](#deep-reinforcement-learning)
- [Architecture](#architecture)
  - [C++ Backend](#c-backend)
  - [Python Bindings](#python-bindings)
  - [React Frontend](#react-frontend)
  - [Concurrency Model](#concurrency-model)
- [API Reference](#api-reference)
  - [Python API](#python-api-reference)
  - [C++ API](#c-api-reference)
  - [WebSocket API](#websocket-api)
- [Examples](#examples)
  - [Simple Moving Average Strategy](#simple-moving-average-strategy)
  - [Deep Q-Network (DQN)](#deep-q-network-dqn)
  - [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
- [Testing](#testing)
  - [Python Tests](#python-tests)
  - [C++ Tests](#c-tests)
  - [Performance Benchmarks](#performance-benchmarks)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- C++ compiler with C++17 support
  - GCC 9+ (Linux)
  - Clang 10+ (macOS)
  - MSVC 2019+ (Windows)
- CMake 3.15+
- Python 3.8+
- Node.js 14+ (for frontend)

#### Required Libraries

- Apache Arrow 10.0.0+
- Drogon 1.8.0+
- PyBind11 2.10.0+
- Taskflow 3.4.0+
- spdlog 1.10.0+
- nlohmann_json 3.11.0+ (optional, will be downloaded if not found)
- PyTorch 1.12.0+ (optional, for deep learning support)

### Building from Source

#### Automated Build (Recommended)

The easiest way to build the project is to use the provided build script:

```bash
# Make the build script executable
chmod +x build_and_test.sh

# Run the build script
./build_and_test.sh
```

This script will:
1. Configure the project with CMake
2. Build the C++ backend
3. Run C++ tests
4. Install Python dependencies
5. Run Python tests

#### Manual Build

1. Clone the repository:

```bash
git clone https://github.com/yourusername/trading-env.git
cd trading-env
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Build the C++ backend:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -- -j$(nproc)
```

4. Install the frontend dependencies:

```bash
cd frontend
npm install
```

5. Build the frontend:

```bash
npm run build
```

### Server Deployment

For deployment on a server, you can use the provided vcpkg manifest mode:

1. Create a `vcpkg.json` file in your project root (or copy the provided one):

```json
{
  "name": "trading-env",
  "version": "1.0.0",
  "description": "High-performance trading environment for deep reinforcement learning",
  "homepage": "https://github.com/yourusername/trading-env",
  "dependencies": [
    "arrow",
    "drogon",
    "pybind11",
    "spdlog",
    "taskflow",
    "nlohmann-json",
    "protobuf",
    {
      "name": "vcpkg-cmake",
      "host": true
    },
    {
      "name": "vcpkg-cmake-config",
      "host": true
    }
  ],
  "features": {
    "cuda": {
      "description": "Enable CUDA support",
      "dependencies": [
        {
          "name": "arrow",
          "features": ["cuda"]
        }
      ]
    },
    "pytorch": {
      "description": "Enable PyTorch support",
      "dependencies": [
        "libtorch"
      ]
    }
  },
  "overrides": [
    {
      "name": "arrow",
      "version": ">=10.0.0"
    }
  ],
  "builtin-baseline": "2023-06-15"
}
```

2. Use the server-specific CMakeLists.txt:

```bash
cp CMakeLists.txt.server CMakeLists.txt
```

3. Build with vcpkg manifest mode:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DVCPKG_MANIFEST_MODE=ON
cmake --build . -- -j$(nproc)
```

4. For CUDA support:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DVCPKG_MANIFEST_MODE=ON -DWITH_CUDA=ON
```

5. For PyTorch support:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DVCPKG_MANIFEST_MODE=ON -DWITH_PYTORCH=ON
```

### Platform-Specific Instructions

#### macOS (Intel and Apple Silicon)

On macOS, you can use Homebrew to install the required dependencies:

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake arrow pybind11 nlohmann-json

# Install Python dependencies
pip install -r requirements.txt

# Build the project
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -- -j$(sysctl -n hw.ncpu)
```

For Apple Silicon (M1/M2) Macs, the project will automatically detect the architecture and use the appropriate optimizations.

#### Linux with CUDA

For Linux systems with NVIDIA GPUs, you can enable CUDA support:

```bash
# Install CUDA Toolkit (if not already installed)
# Follow NVIDIA's instructions for your distribution

# Build with CUDA support
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON
cmake --build . -- -j$(nproc)
```

#### Docker Deployment

A Dockerfile is provided for easy deployment:

```bash
# Build the Docker image
docker build -t trading-env .

# Run the container
docker run -p 8080:8080 -v /path/to/data:/app/data trading-env
```

## Usage

### Python API

The Python API provides an easy-to-use interface for the C++ backend. Here's a basic example:

```python
import trading_env
import numpy as np

# Create environment
env = trading_env.create_trading_env(
    data_dir="path/to/data",
    tickers=["AAPL", "MSFT", "NVDA"],
    visualize=True
)

# Reset environment
obs, info = env.reset()

# Run a few steps with random actions
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        break

# Close environment
env.close()
```

### Loading Data

The environment can load data from CSV files. The files should have columns for datetime, open, high, low, close, and volume. The environment will automatically detect the column names and delimiter.

```python
import trading_env

# Load data for multiple tickers
loader = trading_env.DataLoader()
loader.loadTickerData("AAPL", [
    "path/to/data/AAPL_1.csv",
    "path/to/data/AAPL_2.csv"
])

# Load multiple tickers concurrently
loader.loadMultipleTickers({
    "AAPL": ["path/to/data/AAPL_1.csv", "path/to/data/AAPL_2.csv"],
    "MSFT": ["path/to/data/MSFT_1.csv", "path/to/data/MSFT_2.csv"]
})

# Convert to pandas DataFrame
df = loader.arrowToPandas("AAPL")
print(df.head())
```

### Technical Indicators

The environment provides a wide range of technical indicators for feature engineering:

```python
import trading_env

# Create environment
env = trading_env.create_trading_env(
    data_dir="path/to/data",
    tickers=["AAPL"]
)

# Get ticker data
loader = trading_env.DataLoader()
loader.loadTickerData("AAPL", ["path/to/data/AAPL.csv"])

# Add technical indicators
indicators = ["sma", "ema", "rsi", "bollinger", "macd"]
params = {
    "sma": {"column": "close", "period": "20"},
    "ema": {"column": "close", "period": "20"},
    "rsi": {"column": "close", "period": "14"},
    "bollinger": {"column": "close", "period": "20", "stdDev": "2.0"},
    "macd": {"column": "close", "fastPeriod": "12", "slowPeriod": "26", "signalPeriod": "9"}
}

# Add indicators to the ticker data
trading_env.add_technical_indicators(loader, "AAPL", indicators, params)

# Convert to pandas DataFrame
df = loader.arrowToPandas("AAPL")
print(df.head())
```

### Creating Custom Strategies

You can create custom trading strategies by subclassing the `Strategy` class:

```python
import trading_env
import numpy as np

class MyStrategy(trading_env.Strategy):
    def onTick(self, ticker, table, currentIndex, currentHolding):
        # Simple moving average crossover strategy
        if currentIndex < 50:
            return None  # Not enough data
            
        # Calculate short and long moving averages
        short_ma = np.mean([table.column(4).chunk(0).Value(i) for i in range(currentIndex-20, currentIndex)])
        long_ma = np.mean([table.column(4).chunk(0).Value(i) for i in range(currentIndex-50, currentIndex)])
        
        # Get current price
        current_price = table.column(4).chunk(0).Value(currentIndex)
        
        # Trading logic
        if short_ma > long_ma and currentHolding == 0:
            # Buy signal
            tx = trading_env.Transaction()
            tx.action = trading_env.Action.Buy
            tx.ticker = ticker
            tx.quantity = 1
            tx.price = current_price
            return tx
        elif short_ma < long_ma and currentHolding > 0:
            # Sell signal
            tx = trading_env.Transaction()
            tx.action = trading_env.Action.Sell
            tx.ticker = ticker
            tx.quantity = currentHolding
            tx.price = current_price
            return tx
            
        return None  # No action

# Create environment
env = trading_env.create_trading_env(
    data_dir="path/to/data",
    tickers=["AAPL"]
)

# Register strategy
engine = trading_env.BacktestEngine()
engine.registerStrategy("AAPL", MyStrategy())

# Run backtest
engine.runBacktest()

# Get performance metrics
metrics = engine.getPerformanceMetrics()
print(f"Total Return: {metrics.totalReturn * 100:.2f}%")
print(f"Sharpe Ratio: {metrics.sharpeRatio:.2f}")
```

### Using with PyTorch

The environment can be used with PyTorch for deep reinforcement learning:

```python
import trading_env
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.network(x)

# Create environment
env = trading_env.create_trading_env(
    data_dir="path/to/data",
    tickers=["AAPL", "MSFT", "NVDA"]
)

# Reset environment
obs, _ = env.reset()

# Create model
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.shape[0]
model = PolicyNetwork(input_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for episode in range(100):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(obs)
        
        # Get action from policy
        with torch.no_grad():
            action = model(obs_tensor).numpy()
        
        # Take step in environment
        next_obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        # Update observation
        obs = next_obs
    
    print(f"Episode {episode}: Total Reward = {total_reward}")

# Close environment
env.close()
```

### Visualization

The environment provides a web-based visualization interface:

```python
import trading_env

# Create environment with visualization
env = trading_env.create_trading_env(
    data_dir="path/to/data",
    tickers=["AAPL", "MSFT", "NVDA"],
    visualize=True,
    visualization_port=8080
)

# Open http://localhost:8080 in your browser to see the visualization

# Run backtest
env.run_backtest()

# Close environment (stops the visualization server)
env.close()
```

You can also use the built-in plotting functions:

```python
import trading_env
import matplotlib.pyplot as plt

# Create environment
env = trading_env.create_trading_env(
    data_dir="path/to/data",
    tickers=["AAPL"]
)

# Run backtest
env.run_backtest()

# Plot performance
fig = env.plot_performance()
plt.savefig("performance.png")

# Plot ticker data
fig = env.plot_ticker("AAPL")
plt.savefig("aapl.png")

# Render as pandas DataFrame
df = env.render(mode="dataframe")
print(df['metrics'])
```

### Deep Reinforcement Learning

The environment includes comprehensive support for deep reinforcement learning with both DQN and PPO algorithms:

```python
import trading_env_enhanced as trading_env
from test_drl import DQNAgent, PPOAgent, train_dqn, evaluate_agent

# Create environment with technical indicators
env = trading_env.create_trading_env(
    data_dir="src/stock_data",
    tickers=["AAPL", "MSFT", "NVDA", "AMD"],
    initial_balance=100000.0,
    commission=0.001,
    technical_indicators={
        "sma": {"column": "close", "period": "20"},
        "ema": {"column": "close", "period": "50"},
        "rsi": {"column": "close", "period": "14"},
        "bollinger": {"column": "close", "period": "20", "stdDev": "2.0"}
    }
)

# Create DQN agent
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

dqn_agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    hidden_size=64,
    learning_rate=0.001,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    memory_size=10000
)

# Train DQN agent
dqn_agent = train_dqn(
    env=env,
    agent=dqn_agent,
    episodes=50,
    batch_size=64,
    target_update=10,
    tensorboard_dir="./logs/dqn",
    model_dir="./models/dqn"
)

# Evaluate DQN agent
metrics = evaluate_agent(
    env=env,
    agent=dqn_agent,
    episodes=10,
    render=True
)

print(f"Average Return: {metrics['avg_return'] * 100:.2f}%")
print(f"Average Sharpe Ratio: {metrics['avg_sharpe_ratio']:.2f}")
print(f"Win Rate: {metrics['avg_win_rate'] * 100:.2f}%")
```

The environment supports both value-based methods (DQN) and policy-based methods (PPO):

```python
# Create PPO agent
ppo_agent = PPOAgent(
    state_size=state_size,
    action_size=action_size,
    hidden_size=64,
    learning_rate=0.0003,
    gamma=0.99,
    clip_ratio=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5
)

# Train PPO agent
ppo_agent = train_ppo(
    env=env,
    agent=ppo_agent,
    episodes=50,
    batch_size=64,
    epochs=10,
    tensorboard_dir="./logs/ppo",
    model_dir="./models/ppo"
)
```

You can run the included test script to see a full demonstration:

```bash
# Make the script executable
chmod +x test_drl.py

# Run the test script
./test_drl.py --test
```

For more advanced usage, you can specify training parameters:

```bash
# Train a DQN agent
./test_drl.py --train --algorithm dqn --episodes 100 --batch-size 64 --visualize

# Train a PPO agent
./test_drl.py --train --algorithm ppo --episodes 100 --batch-size 64 --visualize

# Evaluate a trained agent
./test_drl.py --evaluate --algorithm dqn --visualize
```

## Architecture

The trading environment consists of three main components:

1. **C++ Backend**: High-performance core for data processing, backtesting, and strategy execution
2. **Python Bindings**: Easy-to-use Python interface for model development and training
3. **React Frontend**: Real-time visualization of trading performance and metrics

### C++ Backend

The C++ backend is responsible for:

- Loading and processing CSV data
- Calculating technical indicators
- Running backtests
- Executing trading strategies
- Calculating performance metrics

### Python Bindings

The Python bindings provide:

- Gymnasium-compatible interface for reinforcement learning
- Easy-to-use API for data loading and processing
- Integration with PyTorch for deep learning models
- Visualization tools for performance analysis

### React Frontend

The React frontend provides:

- Real-time visualization of trading performance
- Interactive charts for price data
- Performance metrics dashboard
- WebSocket communication with the backend

## API Reference

### Python API

#### `trading_env.create_trading_env`

Create and initialize a trading environment.

```python
def create_trading_env(
    data_dir="../src/stock_data", 
    tickers=None, 
    initial_balance=100000.0,
    commission=0.001,
    visualize=True,
    visualization_port=8080,
    debug=False,
    auto_run_backtest=False
)
```

#### `trading_env.DataLoader`

Load and process CSV data.

```python
class DataLoader:
    def __init__(self)
    def loadTickerData(self, ticker, filePaths)
    def loadMultipleTickers(self, tickerFilePaths)
    def getTickerData(self, ticker)
    def getAllTickerData(self)
    def hasTickerData(self, ticker)
    def getAvailableTickers(self)
    def setDebugMode(self, debug)
    def printTableHead(self, ticker, n=5)
    def arrowToPandas(self, ticker)
    def updateTickerData(self, ticker, table)
```

#### `trading_env.BacktestEngine`

Run backtests and execute trading strategies.

```python
class BacktestEngine:
    def __init__(self)
    def reset(self)
    def step(self)
    def step_with_action(self, actions)
    def step_with_tensor(self, tensor)
    def getPortfolioMetrics(self)
    def getPortfolioMetricsJson(self)
    def getPerformanceMetrics(self)
    def getCashBalance(self)
    def getHoldings(self)
    def getTransactions(self)
    def getObservationDimension(self)
    def getActionDimension(self)
    def getAvailableTickers(self)
    def setDebugMode(self, debug)
    def addTickerData(self, ticker, table)
    def setTickerData(self, data)
    def registerStrategy(self, ticker, strategy)
    def runBacktest(self)
    def setBroadcastCallback(self, callback)
    def setJsonBroadcastCallback(self, callback)
```

#### `trading_env.TechnicalIndicators`

Calculate technical indicators for feature engineering.

```python
class TechnicalIndicators:
    def __init__(self)
    def setDebugMode(self, debug)
    def calculateSMA(self, table, column, period, newColumnName=None)
    def calculateEMA(self, table, column, period, newColumnName=None)
    def calculateRSI(self, table, column, period, newColumnName=None)
    def calculateBollingerBands(self, table, column, period, stdDev=2.0, upperBandName=None, middleBandName=None, lowerBandName=None)
    def calculateMACD(self, table, column, fastPeriod=12, slowPeriod=26, signalPeriod=9, macdName=None, signalName=None, histogramName=None)
    def calculateATR(self, table, highColumn, lowColumn, closeColumn, period=14, newColumnName=None)
    def calculateROC(self, table, column, period, newColumnName=None)
    def calculateStdDev(self, table, column, period, newColumnName=None)
```

#### `trading_env.Strategy`

Base class for custom trading strategies.

```python
class Strategy:
    def __init__(self)
    def onTick(self, ticker, table, currentIndex, currentHolding)
```

#### `trading_env.Transaction`

Represents a trading transaction.

```python
class Transaction:
    def __init__(self)
    # Properties
    action  # Buy, Sell, or Hold
    ticker  # Ticker symbol
    quantity  # Number of shares
    price  # Price per share
    datetime  # Transaction datetime
```

#### `trading_env.PerformanceMetrics`

Represents performance metrics for a backtest.

```python
class PerformanceMetrics:
    def __init__(self)
    # Properties
    initialBalance  # Initial cash balance
    finalBalance  # Final cash balance
    totalReturn  # Total return (decimal)
    annualizedReturn  # Annualized return (decimal)
    sharpeRatio  # Sharpe ratio
    maxDrawdown  # Maximum drawdown (decimal)
    totalTrades  # Total number of trades
    winningTrades  # Number of winning trades
    losingTrades  # Number of losing trades
    winRate  # Win rate (decimal)
    profitFactor  # Profit factor
    averageWin  # Average winning trade
    averageLoss  # Average losing trade
    expectancy  # Expectancy
```

## Examples

### Simple Moving Average Crossover Strategy

```python
import trading_env

# Create environment
env = trading_env.create_trading_env(
    data_dir="path/to/data",
    tickers=["AAPL"]
)

# Add technical indicators
loader = trading_env.DataLoader()
loader.loadTickerData("AAPL", ["path/to/data/AAPL.csv"])

# Add SMA indicators
indicators = ["sma", "sma"]
params = {
    "sma": {"column": "close", "period": "20", "newColumnName": "sma_20"},
    "sma2": {"column": "close", "period": "50", "newColumnName": "sma_50"}
}

trading_env.add_technical_indicators(loader, "AAPL", indicators, params)

# Create strategy
class SMACrossoverStrategy(trading_env.Strategy):
    def onTick(self, ticker, table, currentIndex, currentHolding):
        if currentIndex < 50:
            return None  # Not enough data
            
        # Get SMA values
        sma_20 = table.column(6).chunk(0).Value(currentIndex)  # Assuming column 6 is sma_20
        sma_50 = table.column(7).chunk(0).Value(currentIndex)  # Assuming column 7 is sma_50
        
        # Get current price
        current_price = table.column(4).chunk(0).Value(currentIndex)  # Assuming column 4 is close
        
        # Trading logic
        if sma_20 > sma_50 and currentHolding == 0:
            # Buy signal
            tx = trading_env.Transaction()
            tx.action = trading_env.Action.Buy
            tx.ticker = ticker
            tx.quantity = 1
            tx.price = current_price
            return tx
        elif sma_20 < sma_50 and currentHolding > 0:
            # Sell signal
            tx = trading_env.Transaction()
            tx.action = trading_env.Action.Sell
            tx.ticker = ticker
            tx.quantity = currentHolding
            tx.price = current_price
            return tx
            
        return None  # No action

# Register strategy
engine = trading_env.BacktestEngine()
engine.setTickerData(loader.getAllTickerData())
engine.registerStrategy("AAPL", SMACrossoverStrategy())

# Run backtest
engine.runBacktest()

# Get performance metrics
metrics = engine.getPerformanceMetrics()
print(f"Total Return: {metrics.totalReturn * 100:.2f}%")
print(f"Sharpe Ratio: {metrics.sharpeRatio:.2f}")
print(f"Win Rate: {metrics.winRate * 100:.2f}%")
```

### Deep Reinforcement Learning with PyTorch

```python
import trading_env
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Define a simple DQN model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Create environment
env = trading_env.create_trading_env(
    data_dir="path/to/data",
    tickers=["AAPL", "MSFT", "NVDA"],
    initial_balance=100000.0,
    commission=0.001
)

# Add technical indicators
for ticker in ["AAPL", "MSFT", "NVDA"]:
    indicators = ["sma", "ema", "rsi", "bollinger", "macd"]
    params = {
        "sma": {"column": "close", "period": "20"},
        "ema": {"column": "close", "period": "20"},
        "rsi": {"column": "close", "period": "14"},
        "bollinger": {"column": "close", "period": "20", "stdDev": "2.0"},
        "macd": {"column": "close", "fastPeriod": "12", "slowPeriod": "26", "signalPeriod": "9"}
    }
    trading_env.add_technical_indicators(env.loader, ticker, indicators, params)

# Reset environment
obs, _ = env.reset()

# Create model
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.shape[0]
model = DQN(input_dim, output_dim)
target_model = DQN(input_dim, output_dim)
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Replay buffer
replay_buffer = deque(maxlen=10000)
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

# Training loop
for episode in range(1000):
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
                q_values = model(obs_tensor)
            action = q_values.numpy()
        
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
            
            # Compute Q values
            q_values = model(states)
            next_q_values = target_model(next_states)
            
            # Compute target Q values
            target_q_values = rewards + gamma * torch.max(next_q_values, dim=1)[0] * (1 - dones)
            
            # Compute loss
            loss = criterion(q_values, target_q_values.unsqueeze(1))
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Update target model
    if episode % 10 == 0:
        target_model.load_state_dict(model.state_dict())
    
    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    print(f"Episode {episode}: Total Reward = {total_reward}, Epsilon = {epsilon:.4f}")

# Save model
torch.save(model.state_dict(), "dqn_model.pt")

# Close environment
env.close()
```

## Testing

Run the comprehensive test suite:

```bash
./build_and_test.sh
```

Or run individual tests:

```bash
# C++ tests
./build/file_reading --test

# Python tests
python test.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.