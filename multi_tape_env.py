import os
import glob
import random
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class MultiTapeL2Env(gym.Env):
    """Target-Position L2 Environment: 0 = Flat (Cash), 1 = Long (Equity)."""
    def __init__(self, file_list, max_episode_steps=2048):
        super().__init__()
        self.file_list = list(file_list)
        self.max_episode_steps = max_episode_steps
        
        # Binary target position eliminates invalid action pollution
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.reset()

    def _load_random_tape(self):
        while len(self.file_list) > 0:
            target_file = random.choice(self.file_list)
            try:
                df = pd.read_parquet(target_file)
                df = df[(df['bid_price_0'] > 0) & (df['ask_price_0'] > 0)].copy()
                df = df[df['ask_price_0'] > df['bid_price_0']].copy()
                
                if len(df) < 100:
                    self.file_list.remove(target_file)
                    continue

                # Stationary features
                df['spread'] = df['ask_price_0'] - df['bid_price_0']
                df['mid_price'] = (df['ask_price_0'] + df['bid_price_0']) / 2.0
                df['spread_bps'] = (df['spread'] / df['mid_price']) * 10000.0
                df['obi_l1'] = (df['bid_qty_0'] - df['ask_qty_0']) / (df['bid_qty_0'] + df['ask_qty_0'] + 1e-6)
                
                bid_vols = df[['bid_qty_0', 'bid_qty_1', 'bid_qty_2', 'bid_qty_3', 'bid_qty_4']].sum(axis=1)
                ask_vols = df[['ask_qty_0', 'ask_qty_1', 'ask_qty_2', 'ask_qty_3', 'ask_qty_4']].sum(axis=1)
                df['obi_deep'] = (bid_vols - ask_vols) / (bid_vols + ask_vols + 1e-6)
                
                micro = (df['bid_price_0'] * df['ask_qty_0'] + df['ask_price_0'] * df['bid_qty_0']) / (df['bid_qty_0'] + df['ask_qty_0'] + 1e-6)
                df['micro_delta_bps'] = ((micro - df['mid_price']) / df['mid_price']) * 10000.0
                
                cleaned = df.dropna().reset_index(drop=True)
                if len(cleaned) < 100:
                    self.file_list.remove(target_file)
                    continue

                self.current_df = cleaned
                max_start = max(0, len(self.current_df) - self.max_episode_steps - 1)
                self.start_step = random.randint(0, max_start) if max_start > 0 else 0
                self.step_idx = self.start_step
                self.prev_mid = float(self.current_df.iloc[self.step_idx]['mid_price'])
                return
            except Exception:
                self.file_list.remove(target_file)
        
        raise RuntimeError("No readable tapes remaining.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._load_random_tape()
        self.position = 0
        self.entry_price = 0.0
        self.episode_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        safe_idx = min(self.step_idx, len(self.current_df) - 1)
        row = self.current_df.iloc[safe_idx]
        mid = float(row['mid_price'])
        unrealized_bps = 0.0
        
        if self.position == 1 and self.entry_price > 0:
            unrealized_bps = ((mid - self.entry_price) / max(self.entry_price, 1e-6)) * 10000.0

        return np.array([
            row['spread_bps'],
            row['micro_delta_bps'],
            row['obi_l1'],
            row['obi_deep'],
            float(self.position),
            unrealized_bps
        ], dtype=np.float32)

    def step(self, action):
        safe_idx = min(self.step_idx, len(self.current_df) - 1)
        row = self.current_df.iloc[safe_idx]
        ask = float(row['ask_price_0'])
        bid = float(row['bid_price_0'])
        mid = float(row['mid_price'])
        reward = 0.0
        ROUNDTRIP_FEE_BPS = 3.0  # Conservative friction

        target_position = action  # 0 = Target Flat, 1 = Target Long

        # Transition 1: Open Long
        if self.position == 0 and target_position == 1 and ask > 0:
            self.position = 1
            self.entry_price = ask
            reward = 0.0

        # Transition 2: Close Long (Take Profit / Manual Exit)
        elif self.position == 1 and target_position == 0 and self.entry_price > 0:
            net_bps = (((bid - self.entry_price) / self.entry_price) * 10000.0) - ROUNDTRIP_FEE_BPS
            reward = float(net_bps / 10.0)
            self.position = 0
            self.entry_price = 0.0

        # State Maintenance: Holding Long
        elif self.position == 1 and target_position == 1:
            unrealized_bps = ((bid - self.entry_price) / max(self.entry_price, 1e-6)) * 10000.0
            
            # Risk Daemon: Hard Stop Ejection (-20 bps)
            if unrealized_bps <= -20.0:
                reward = float((unrealized_bps - ROUNDTRIP_FEE_BPS) / 10.0)
                self.position = 0
                self.entry_price = 0.0
            else:
                # Mark-to-market incremental reward
                step_bps = ((mid - self.prev_mid) / max(self.prev_mid, 1e-6)) * 10000.0
                reward = (step_bps / 10.0) - 0.001

        # State Maintenance: Holding Flat (Cash)
        elif self.position == 0 and target_position == 0:
            reward = 0.0

        self.prev_mid = mid
        self.step_idx += 1
        self.episode_step += 1
        terminated = (self.episode_step >= self.max_episode_steps) or (self.step_idx >= len(self.current_df) - 1)
        return self._get_obs(), float(reward), terminated, False, {}
