#!/usr/bin/env python3
"""
Test script for deep reinforcement learning with the trading environment.

This script demonstrates how to use the trading environment with PyTorch
for training and testing deep reinforcement learning models.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
import argparse
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_drl")

# Add the current directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    PYTORCH_AVAILABLE = True
    logger.info("PyTorch is available and will be used for training")
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch is not available. Training functions will be disabled.")

# Try to import the trading environment
try:
    from trading_env_enhanced import TradingEnv, create_trading_env, create_pytorch_model
    logger.info("Successfully imported trading environment")
except ImportError as e:
    logger.error(f"Error importing trading environment: {e}")
    logger.error("Make sure trading_env_enhanced.py is in the current directory")
    sys.exit(1)

# Try to import the C++ module
try:
    import my_module
    logger.info("Successfully imported C++ backend module")
except ImportError as e:
    logger.error(f"Error importing C++ module: {e}")
    logger.error("Make sure the my_module shared library is built and in the correct location")
    sys.exit(1)


class DQNAgent:
    """Deep Q-Network agent for trading."""
    
    def __init__(self, state_size, action_size, hidden_size=64, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, memory_size=10000):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space
            hidden_size: Size of the hidden layers
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor
            epsilon_start: Starting epsilon for epsilon-greedy exploration
            epsilon_end: Minimum epsilon value
            epsilon_decay: Epsilon decay rate
            memory_size: Size of the replay buffer
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Cannot create DQN agent.")
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        
        # Create Q-network and target network
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Create optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Create replay buffer
        self.memory = []
        
        # Training metrics
        self.loss_history = []
    
    def _build_model(self):
        """Build a neural network model for DQN."""
        model = nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_size),
            nn.Tanh()  # Output between -1 and 1 for trading actions
        )
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose an action using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(-1, 1, self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.q_network(state_tensor)
        return action_values.numpy()[0]
    
    def replay(self, batch_size):
        """Train the network using experience replay."""
        if len(self.memory) < batch_size:
            return 0
        
        # Sample batch from memory
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for i in minibatch:
            states.append(self.memory[i][0])
            actions.append(self.memory[i][1])
            rewards.append(self.memory[i][2])
            next_states.append(self.memory[i][3])
            dones.append(self.memory[i][4])
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))
        
        # Get Q values for current states
        q_values = self.q_network(states)
        
        # Get Q values for next states from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q = torch.max(next_q_values, dim=1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Update Q values for the actions taken
        q_values_for_actions = torch.sum(q_values * actions, dim=1)
        
        # Compute loss
        loss = nn.MSELoss()(q_values_for_actions, target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Store loss
        self.loss_history.append(loss.item())
        
        return loss.item()
    
    def update_target_network(self):
        """Update the target network with weights from the Q-network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path):
        """Save the model to a file."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'loss_history': self.loss_history
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path):
        """Load the model from a file."""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.loss_history = checkpoint['loss_history']
        logger.info(f"Model loaded from {path}")


class PPOAgent:
    """Proximal Policy Optimization agent for trading."""
    
    def __init__(self, state_size, action_size, hidden_size=64, learning_rate=0.0003, gamma=0.99,
                 clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        """
        Initialize the PPO agent.
        
        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space
            hidden_size: Size of the hidden layers
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor
            clip_ratio: PPO clipping parameter
            value_coef: Value function coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm for clipping
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Cannot create PPO agent.")
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Create actor-critic network
        self.actor_critic = self._build_model()
        
        # Create optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # Training metrics
        self.loss_history = []
    
    def _build_model(self):
        """Build an actor-critic neural network model for PPO."""
        class ActorCritic(nn.Module):
            def __init__(self, state_size, action_size, hidden_size):
                super(ActorCritic, self).__init__()
                
                # Shared layers
                self.shared = nn.Sequential(
                    nn.Linear(state_size, hidden_size),
                    nn.ReLU()
                )
                
                # Actor (policy) layers
                self.actor = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, action_size),
                    nn.Tanh()  # Output between -1 and 1 for trading actions
                )
                
                # Critic (value) layers
                self.critic = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                )
                
                # Action log standard deviation (learnable)
                self.log_std = nn.Parameter(torch.zeros(action_size))
            
            def forward(self, state):
                """Forward pass through the network."""
                x = self.shared(state)
                
                # Actor: mean of action distribution
                action_mean = self.actor(x)
                
                # Critic: state value
                value = self.critic(x)
                
                return action_mean, self.log_std, value
            
            def act(self, state):
                """Sample an action from the policy."""
                action_mean, log_std, value = self.forward(state)
                std = torch.exp(log_std)
                
                # Sample from normal distribution
                normal = torch.distributions.Normal(action_mean, std)
                action = normal.sample()
                
                # Clip action to [-1, 1]
                action = torch.clamp(action, -1.0, 1.0)
                
                # Calculate log probability
                log_prob = normal.log_prob(action).sum(dim=-1)
                
                return action, log_prob, value
            
            def evaluate(self, state, action):
                """Evaluate an action."""
                action_mean, log_std, value = self.forward(state)
                std = torch.exp(log_std)
                
                # Create normal distribution
                normal = torch.distributions.Normal(action_mean, std)
                
                # Calculate log probability
                log_prob = normal.log_prob(action).sum(dim=-1)
                
                # Calculate entropy
                entropy = normal.entropy().sum(dim=-1)
                
                return log_prob, entropy, value
        
        return ActorCritic(self.state_size, self.action_size, self.hidden_size)
    
    def act(self, state):
        """Choose an action using the policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.actor_critic.act(state_tensor)
        return action.numpy()[0]
    
    def update(self, states, actions, old_log_probs, returns, advantages, batch_size=64, epochs=10):
        """Update the policy using PPO."""
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        old_log_probs = torch.FloatTensor(np.array(old_log_probs))
        returns = torch.FloatTensor(np.array(returns))
        advantages = torch.FloatTensor(np.array(advantages))
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy for multiple epochs
        for _ in range(epochs):
            # Generate random indices
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            # Update in batches
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                idx = indices[start:end]
                
                # Get batch data
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                
                # Evaluate actions
                log_probs, entropy, values = self.actor_critic.evaluate(batch_states, batch_actions)
                
                # Calculate ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Calculate surrogate losses
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                
                # Calculate actor loss
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Calculate critic loss
                critic_loss = nn.MSELoss()(values.squeeze(-1), batch_returns)
                
                # Calculate entropy loss
                entropy_loss = -entropy.mean()
                
                # Calculate total loss
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
                
                # Optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Store loss
                self.loss_history.append(loss.item())
    
    def save(self, path):
        """Save the model to a file."""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path):
        """Load the model from a file."""
        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_history = checkpoint['loss_history']
        logger.info(f"Model loaded from {path}")


def train_dqn(env, agent, episodes=100, batch_size=64, target_update=10, 
             tensorboard_dir="./logs", model_dir="./models"):
    """
    Train a DQN agent on the trading environment.
    
    Args:
        env: Trading environment
        agent: DQN agent
        episodes: Number of episodes to train
        batch_size: Batch size for training
        target_update: Number of episodes between target network updates
        tensorboard_dir: Directory for TensorBoard logs
        model_dir: Directory for saving models
        
    Returns:
        Trained DQN agent
    """
    if not PYTORCH_AVAILABLE:
        logger.error("PyTorch is not available. Cannot train DQN agent.")
        return agent
    
    # Create directories if they don't exist
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create TensorBoard writer
    writer = SummaryWriter(tensorboard_dir)
    
    # Training metrics
    best_return = -np.inf
    episode_rewards = []
    episode_returns = []
    episode_sharpe_ratios = []
    
    # Training loop
    for episode in range(episodes):
        # Reset environment
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        # Episode loop
        while not done:
            # Choose action
            action = agent.act(state)
            
            # Take step in environment
            next_state, reward, done, _, _ = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            # Train agent
            loss = agent.replay(batch_size)
            
            # Log loss to TensorBoard
            if loss > 0:
                writer.add_scalar('Loss/train', loss, episode)
        
        # Update target network
        if episode % target_update == 0:
            agent.update_target_network()
        
        # Run backtest to get metrics
        metrics = env.run_backtest()
        
        # Record metrics
        episode_rewards.append(total_reward)
        episode_returns.append(metrics.totalReturn)
        episode_sharpe_ratios.append(metrics.sharpeRatio)
        
        # Log metrics to TensorBoard
        writer.add_scalar('Reward/train', total_reward, episode)
        writer.add_scalar('Return/train', metrics.totalReturn, episode)
        writer.add_scalar('Sharpe/train', metrics.sharpeRatio, episode)
        writer.add_scalar('Epsilon', agent.epsilon, episode)
        
        # Save best model
        if metrics.totalReturn > best_return:
            best_return = metrics.totalReturn
            agent.save(os.path.join(model_dir, 'dqn_best.pt'))
            logger.info(f"New best model saved with return: {best_return * 100:.2f}%")
        
        # Save checkpoint
        if episode % 10 == 0:
            agent.save(os.path.join(model_dir, f'dqn_episode_{episode}.pt'))
        
        # Print progress
        logger.info(f"Episode {episode+1}/{episodes}: Reward = {total_reward:.2f}, Return = {metrics.totalReturn * 100:.2f}%, Sharpe = {metrics.sharpeRatio:.2f}, Epsilon = {agent.epsilon:.4f}")
    
    # Save final model
    agent.save(os.path.join(model_dir, 'dqn_final.pt'))
    
    # Close TensorBoard writer
    writer.close()
    
    # Plot training metrics
    plot_training_metrics(episode_rewards, episode_returns, episode_sharpe_ratios, 'DQN')
    
    return agent


def train_ppo(env, agent, episodes=100, batch_size=64, epochs=10, 
             tensorboard_dir="./logs", model_dir="./models"):
    """
    Train a PPO agent on the trading environment.
    
    Args:
        env: Trading environment
        agent: PPO agent
        episodes: Number of episodes to train
        batch_size: Batch size for training
        epochs: Number of epochs to train on each batch
        tensorboard_dir: Directory for TensorBoard logs
        model_dir: Directory for saving models
        
    Returns:
        Trained PPO agent
    """
    if not PYTORCH_AVAILABLE:
        logger.error("PyTorch is not available. Cannot train PPO agent.")
        return agent
    
    # Create directories if they don't exist
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create TensorBoard writer
    writer = SummaryWriter(tensorboard_dir)
    
    # Training metrics
    best_return = -np.inf
    episode_rewards = []
    episode_returns = []
    episode_sharpe_ratios = []
    
    # Training loop
    for episode in range(episodes):
        # Reset environment
        state, _ = env.reset()
        done = False
        
        # Storage for episode data
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        
        # Episode loop
        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action, log probability, and value
            with torch.no_grad():
                action, log_prob, value = agent.actor_critic.act(state_tensor)
            
            # Take step in environment
            next_state, reward, done, _, _ = env.step(action.numpy()[0])
            
            # Store data
            states.append(state)
            actions.append(action.numpy()[0])
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob.item())
            
            # Update state
            state = next_state
        
        # Calculate returns and advantages
        returns = []
        advantages = []
        
        # Calculate returns (discounted sum of rewards)
        R = 0
        for r in reversed(rewards):
            R = r + agent.gamma * R
            returns.insert(0, R)
        
        # Convert to numpy arrays
        returns = np.array(returns)
        values = np.array(values)
        
        # Calculate advantages
        advantages = returns - values
        
        # Update policy
        agent.update(states, actions, log_probs, returns, advantages, batch_size, epochs)
        
        # Run backtest to get metrics
        metrics = env.run_backtest()
        
        # Record metrics
        episode_rewards.append(sum(rewards))
        episode_returns.append(metrics.totalReturn)
        episode_sharpe_ratios.append(metrics.sharpeRatio)
        
        # Log metrics to TensorBoard
        writer.add_scalar('Reward/train', sum(rewards), episode)
        writer.add_scalar('Return/train', metrics.totalReturn, episode)
        writer.add_scalar('Sharpe/train', metrics.sharpeRatio, episode)
        
        # Save best model
        if metrics.totalReturn > best_return:
            best_return = metrics.totalReturn
            agent.save(os.path.join(model_dir, 'ppo_best.pt'))
            logger.info(f"New best model saved with return: {best_return * 100:.2f}%")
        
        # Save checkpoint
        if episode % 10 == 0:
            agent.save(os.path.join(model_dir, f'ppo_episode_{episode}.pt'))
        
        # Print progress
        logger.info(f"Episode {episode+1}/{episodes}: Reward = {sum(rewards):.2f}, Return = {metrics.totalReturn * 100:.2f}%, Sharpe = {metrics.sharpeRatio:.2f}")
    
    # Save final model
    agent.save(os.path.join(model_dir, 'ppo_final.pt'))
    
    # Close TensorBoard writer
    writer.close()
    
    # Plot training metrics
    plot_training_metrics(episode_rewards, episode_returns, episode_sharpe_ratios, 'PPO')
    
    return agent


def evaluate_agent(env, agent, episodes=10, render=True):
    """
    Evaluate an agent on the trading environment.
    
    Args:
        env: Trading environment
        agent: Agent to evaluate
        episodes: Number of episodes to evaluate
        render: Whether to render the environment
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Evaluation metrics
    returns = []
    sharpe_ratios = []
    max_drawdowns = []
    win_rates = []
    
    for episode in range(episodes):
        # Reset environment
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        # Episode loop
        while not done:
            # Choose action
            action = agent.act(state)
            
            # Take step in environment
            next_state, reward, done, _, _ = env.step(action)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            # Render environment
            if render:
                env.render()
        
        # Run backtest to get metrics
        metrics = env.run_backtest()
        
        # Record metrics
        returns.append(metrics.totalReturn)
        sharpe_ratios.append(metrics.sharpeRatio)
        max_drawdowns.append(metrics.maxDrawdown)
        win_rates.append(metrics.winRate)
        
        # Print progress
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


def plot_training_metrics(rewards, returns, sharpe_ratios, algorithm_name):
    """
    Plot training metrics.
    
    Args:
        rewards: List of episode rewards
        returns: List of episode returns
        sharpe_ratios: List of episode Sharpe ratios
        algorithm_name: Name of the algorithm
    """
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(3, 1, 1)
    plt.plot(rewards)
    plt.title(f'{algorithm_name} Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # Plot returns
    plt.subplot(3, 1, 2)
    plt.plot(returns)
    plt.title(f'{algorithm_name} Training Returns')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.grid(True)
    
    # Plot Sharpe ratios
    plt.subplot(3, 1, 3)
    plt.plot(sharpe_ratios)
    plt.title(f'{algorithm_name} Training Sharpe Ratios')
    plt.xlabel('Episode')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{algorithm_name}_training_metrics.png')
    plt.close()


def test_drl_trading():
    """Test deep reinforcement learning for trading."""
    logger.info("=== Testing Deep Reinforcement Learning for Trading ===")
    
    # Check if PyTorch is available
    if not PYTORCH_AVAILABLE:
        logger.error("PyTorch is not available. Cannot test DRL trading.")
        return False
    
    # Create trading environment
    env = create_trading_env(
        data_dir="./src/stock_data",
        tickers=["AAPL", "MSFT", "NVDA", "AMD"],
        initial_balance=100000.0,
        commission=0.001,
        visualize=True,
        visualization_port=8080,
        debug=True,
        technical_indicators={
            "sma": {"column": "close", "period": "20"},
            "ema": {"column": "close", "period": "50"},
            "rsi": {"column": "close", "period": "14"},
            "bollinger": {"column": "close", "period": "20", "stdDev": "2.0"},
            "macd": {"column": "close", "fastPeriod": "12", "slowPeriod": "26", "signalPeriod": "9"}
        }
    )
    
    # Create DQN agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    logger.info(f"State size: {state_size}")
    logger.info(f"Action size: {action_size}")
    
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
    logger.info("Training DQN agent...")
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
    logger.info("Evaluating DQN agent...")
    dqn_metrics = evaluate_agent(
        env=env,
        agent=dqn_agent,
        episodes=10,
        render=True
    )
    
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
    logger.info("Training PPO agent...")
    ppo_agent = train_ppo(
        env=env,
        agent=ppo_agent,
        episodes=50,
        batch_size=64,
        epochs=10,
        tensorboard_dir="./logs/ppo",
        model_dir="./models/ppo"
    )
    
    # Evaluate PPO agent
    logger.info("Evaluating PPO agent...")
    ppo_metrics = evaluate_agent(
        env=env,
        agent=ppo_agent,
        episodes=10,
        render=True
    )
    
    # Compare DQN and PPO
    logger.info("=== Comparison of DQN and PPO ===")
    logger.info(f"DQN Average Return: {dqn_metrics['avg_return'] * 100:.2f}%")
    logger.info(f"PPO Average Return: {ppo_metrics['avg_return'] * 100:.2f}%")
    logger.info(f"DQN Average Sharpe: {dqn_metrics['avg_sharpe_ratio']:.2f}")
    logger.info(f"PPO Average Sharpe: {ppo_metrics['avg_sharpe_ratio']:.2f}")
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Plot returns
    plt.subplot(2, 1, 1)
    plt.plot(dqn_metrics['returns'], label='DQN')
    plt.plot(ppo_metrics['returns'], label='PPO')
    plt.title('Returns Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    
    # Plot Sharpe ratios
    plt.subplot(2, 1, 2)
    plt.plot(dqn_metrics['sharpe_ratios'], label='DQN')
    plt.plot(ppo_metrics['sharpe_ratios'], label='PPO')
    plt.title('Sharpe Ratio Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('drl_comparison.png')
    plt.close()
    
    # Close environment
    env.close()
    
    return True


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description='Test deep reinforcement learning for trading')
    parser.add_argument('--test', action='store_true', help='Run tests')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'ppo', 'both'], help='Algorithm to use')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--visualize', action='store_true', help='Visualize training')
    parser.add_argument('--port', type=int, default=8080, help='Visualization port')
    parser.add_argument('--model-dir', type=str, default='./models', help='Model directory')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')
    
    args = parser.parse_args()
    
    if args.test:
        test_drl_trading()
    elif args.train:
        # Create trading environment
        env = create_trading_env(
            data_dir="./src/stock_data",
            tickers=["AAPL", "MSFT", "NVDA", "AMD"],
            initial_balance=100000.0,
            commission=0.001,
            visualize=args.visualize,
            visualization_port=args.port,
            debug=True,
            technical_indicators={
                "sma": {"column": "close", "period": "20"},
                "ema": {"column": "close", "period": "50"},
                "rsi": {"column": "close", "period": "14"},
                "bollinger": {"column": "close", "period": "20", "stdDev": "2.0"},
                "macd": {"column": "close", "fastPeriod": "12", "slowPeriod": "26", "signalPeriod": "9"}
            }
        )
        
        # Create agent
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        
        if args.algorithm == 'dqn' or args.algorithm == 'both':
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
            logger.info("Training DQN agent...")
            dqn_agent = train_dqn(
                env=env,
                agent=dqn_agent,
                episodes=args.episodes,
                batch_size=args.batch_size,
                target_update=10,
                tensorboard_dir=f"{args.log_dir}/dqn",
                model_dir=f"{args.model_dir}/dqn"
            )
        
        if args.algorithm == 'ppo' or args.algorithm == 'both':
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
            logger.info("Training PPO agent...")
            ppo_agent = train_ppo(
                env=env,
                agent=ppo_agent,
                episodes=args.episodes,
                batch_size=args.batch_size,
                epochs=10,
                tensorboard_dir=f"{args.log_dir}/ppo",
                model_dir=f"{args.model_dir}/ppo"
            )
        
        # Close environment
        env.close()
    elif args.evaluate:
        # Create trading environment
        env = create_trading_env(
            data_dir="./src/stock_data",
            tickers=["AAPL", "MSFT", "NVDA", "AMD"],
            initial_balance=100000.0,
            commission=0.001,
            visualize=args.visualize,
            visualization_port=args.port,
            debug=True,
            technical_indicators={
                "sma": {"column": "close", "period": "20"},
                "ema": {"column": "close", "period": "50"},
                "rsi": {"column": "close", "period": "14"},
                "bollinger": {"column": "close", "period": "20", "stdDev": "2.0"},
                "macd": {"column": "close", "fastPeriod": "12", "slowPeriod": "26", "signalPeriod": "9"}
            }
        )
        
        # Create agent
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        
        if args.algorithm == 'dqn' or args.algorithm == 'both':
            dqn_agent = DQNAgent(
                state_size=state_size,
                action_size=action_size,
                hidden_size=64,
                learning_rate=0.001,
                gamma=0.99,
                epsilon_start=0.01,  # Low epsilon for evaluation
                epsilon_end=0.01,
                epsilon_decay=1.0,
                memory_size=10000
            )
            
            # Load DQN agent
            dqn_agent.load(f"{args.model_dir}/dqn/dqn_best.pt")
            
            # Evaluate DQN agent
            logger.info("Evaluating DQN agent...")
            dqn_metrics = evaluate_agent(
                env=env,
                agent=dqn_agent,
                episodes=10,
                render=True
            )
        
        if args.algorithm == 'ppo' or args.algorithm == 'both':
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
            
            # Load PPO agent
            ppo_agent.load(f"{args.model_dir}/ppo/ppo_best.pt")
            
            # Evaluate PPO agent
            logger.info("Evaluating PPO agent...")
            ppo_metrics = evaluate_agent(
                env=env,
                agent=ppo_agent,
                episodes=10,
                render=True
            )
        
        # Close environment
        env.close()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()