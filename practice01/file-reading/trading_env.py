import gym
from gym import spaces
import numpy as np
import my_module

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TradingEnv, self).__init__()
        # Instantiate an independent backtesting engine.
        self.engine = my_module.BacktestEngine()
        # Define action space: one continuous action per ticker.
        # Action: a value indicating desired number of shares to trade (positive: buy, negative: sell).
        self.tickers = ["AAPL", "MSFT", "NVDA", "AMD"]
        self.action_space = spaces.Box(low=-100, high=100, shape=(len(self.tickers),), dtype=np.float32)
        # Observation space: an array of current close prices.
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(self.tickers),), dtype=np.float32)

    def reset(self):
        self.engine.reset()
        result = self.engine.step()  # Use default strategy for initial observation.
        return self._obs_to_array(result.observations)

    def step(self, action):
        # 'action' is an array of floats, one per ticker.
        external_actions = {ticker: float(a) for ticker, a in zip(self.tickers, action)}
        result = self.engine.step_with_action(external_actions)
        obs = self._obs_to_array(result.observations)
        reward = result.reward
        done = result.done
        info = {}
        return obs, reward, done, info

    def _obs_to_array(self, obs_dict):
        return np.array([obs_dict.get(ticker, 0.0) for ticker in self.tickers], dtype=np.float32)

    def render(self, mode="human"):
        print(self.engine.getPortfolioMetrics())
