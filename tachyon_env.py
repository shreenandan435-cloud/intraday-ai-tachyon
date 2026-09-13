import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TachyonL2Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.max_steps = len(self.df) - 1
        self.initial_balance = initial_balance
        
        # 0 = Flat/Hold, 1 = Long, 2 = Close
        self.action_space = spaces.Discrete(3)
        
        # Feature Vector: [mid_price, spread, obi_l1, obi_deep, micro_price, position, unrealized_pnl_bps]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0: flat, 1: long
        self.entry_price = 0.0
        self.trades = 0
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        mid = float(row['mid_price'])
        
        if self.position == 1 and self.entry_price > 0:
            unrealized_pnl_bps = ((mid - self.entry_price) / self.entry_price) * 10000.0
        else:
            unrealized_pnl_bps = 0.0

        obs = np.array([
            row['mid_price'],
            row['spread'],
            row['obi_l1'],
            row['obi_deep'],
            row['micro_price'],
            float(self.position),
            unrealized_pnl_bps
        ], dtype=np.float32)
        return obs

    def step(self, action):
        row = self.df.iloc[self.current_step]
        ask = float(row['ask_price_0'])
        bid = float(row['bid_price_0'])
        reward = 0.0
        
        # Indian Intraday Equities: ₹20 brokerage cap + ~0.03% turnover taxes/slippage
        TRANSACTION_COST_PCT = 0.0005  # 5 bps conservative round-trip slippage/fee

        # Action 1: Open Long
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = ask
            # Penalize entry with fee immediately
            reward -= (self.entry_price * TRANSACTION_COST_PCT)

        # Action 2: Close Position
        elif action == 2 and self.position == 1:
            gross_pnl = bid - self.entry_price
            net_pnl = gross_pnl - (bid * TRANSACTION_COST_PCT)
            reward += net_pnl
            self.balance += net_pnl
            self.position = 0
            self.entry_price = 0.0
            self.trades += 1

        # Holding penalty to discourage idle capital lockup during dead order books
        elif action == 0 and self.position == 1:
            reward -= 0.001

        self.current_step += 1
        terminated = self.current_step >= self.max_steps or self.balance <= (self.initial_balance * 0.90)
        truncated = False

        return self._get_obs(), float(reward), terminated, truncated, {
            "balance": self.balance,
            "trades": self.trades
        }