import os
import glob
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class StationaryL2Env(gym.Env):
    """Zero-price-dependency order book gym using relative basis points."""
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.max_steps = len(self.df) - 1
        self.initial_balance = initial_balance
        
        # Actions: 0 = Flat/Pass, 1 = Buy Long, 2 = Sell/Exit
        self.action_space = spaces.Discrete(3)
        
        # State Vector: [spread_bps, micro_delta_bps, obi_l1, obi_deep, position, unrealized_pnl_bps]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.trades = 0
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        mid = float(row['mid_price'])
        
        unrealized_bps = 0.0
        if self.position == 1 and self.entry_price > 0:
            unrealized_bps = ((mid - self.entry_price) / self.entry_price) * 10000.0

        return np.array([
            row['spread_bps'],
            row['micro_delta_bps'],
            row['obi_l1'],
            row['obi_deep'],
            float(self.position),
            unrealized_bps
        ], dtype=np.float32)

    def step(self, action):
        row = self.df.iloc[self.current_step]
        ask = float(row['ask_price_0'])
        bid = float(row['bid_price_0'])
        reward = 0.0
        FEE_BPS = 2.0  # Conservative 2 bps per side

        # Action 1: Enter Long
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = ask
            reward = 0.001  # Micro-incentive to break zero-action freeze

        # Action 2: Exit
        elif action == 2 and self.position == 1:
            gross_return_bps = ((bid - self.entry_price) / self.entry_price) * 10000.0
            net_return_bps = gross_return_bps - (2 * FEE_BPS)
            
            # Direct normalized reward scaled to trade efficiency
            reward = float(net_return_bps / 10.0)
            
            pnl_cash = (bid - self.entry_price) - (bid * 0.0004)
            self.balance += pnl_cash
            self.position = 0
            self.entry_price = 0.0
            self.trades += 1

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        return self._get_obs(), reward, terminated, False, {
            "balance": self.balance,
            "trades": self.trades
        }

def prepare_l2_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df['spread'] = df['ask_price_0'] - df['bid_price_0']
    df['mid_price'] = (df['ask_price_0'] + df['bid_price_0']) / 2.0
    
    # Strictly stationary relative features
    df['spread_bps'] = (df['spread'] / df['mid_price']) * 10000.0
    df['obi_l1'] = (df['bid_qty_0'] - df['ask_qty_0']) / (df['bid_qty_0'] + df['ask_qty_0'] + 1e-6)
    
    bid_vols = df[['bid_qty_0', 'bid_qty_1', 'bid_qty_2', 'bid_qty_3', 'bid_qty_4']].sum(axis=1)
    ask_vols = df[['ask_qty_0', 'ask_qty_1', 'ask_qty_2', 'ask_qty_3', 'ask_qty_4']].sum(axis=1)
    df['obi_deep'] = (bid_vols - ask_vols) / (bid_vols + ask_vols + 1e-6)
    
    micro = (df['bid_price_0'] * df['ask_qty_0'] + df['ask_price_0'] * df['bid_qty_0']) / (df['bid_qty_0'] + df['ask_qty_0'] + 1e-6)
    df['micro_delta_bps'] = ((micro - df['mid_price']) / df['mid_price']) * 10000.0
    
    return df.dropna().reset_index(drop=True)

if __name__ == "__main__":
    train_path = r"C:\Users\Shree\intraday-ai-tachyon\data\ticks\depth\date=2026-08-26\symbol=RAMBHAJO\1300.parquet"
    test_path  = r"C:\Users\Shree\intraday-ai-tachyon\data\ticks\depth\date=2026-08-26\symbol=RAMBHAJO\1400.parquet"
    
    print("[*] Processing Stationary L2 Features...")
    df_train = prepare_l2_dataframe(train_path)
    df_test = prepare_l2_dataframe(test_path)
    
    env_train = DummyVecEnv([lambda: StationaryL2Env(df_train)])
    
    # Running on CPU to maximize throughput for small vector policies
    model = PPO(
        "MlpPolicy",
        env_train,
        learning_rate=5e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.05,  # Higher entropy prevents the do-nothing collapse
        gamma=0.95,
        device="cpu",
        verbose=0
    )
    
    print("[*] Training PPO with Inaction Prevention (60,000 steps)...")
    model.learn(total_timesteps=60_000)
    
    print("[*] Backtesting on Unseen 14:00 Tape (Deterministic)...")
    env_test = StationaryL2Env(df_test)
    obs, info = env_test.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(action)
        done = terminated or truncated
        
    pnl = info['balance'] - 10000.0
    print("\n=== RE-VALIDATION RESULTS ===")
    print(f"Total Trades Executed : {info['trades']}")
    print(f"Final Balance         : ₹{info['balance']:.2f}")
    print(f"Net Realized PnL      : ₹{pnl:.2f}")
