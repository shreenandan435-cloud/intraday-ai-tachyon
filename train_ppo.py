import os
import torch
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tachyon_env import TachyonL2Env

def load_and_preprocess(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    
    # Mathematical L2 features
    df['spread'] = df['ask_price_0'] - df['bid_price_0']
    df['mid_price'] = (df['ask_price_0'] + df['bid_price_0']) / 2.0
    
    # OBI (Level 1)
    df['obi_l1'] = (df['bid_qty_0'] - df['ask_qty_0']) / (df['bid_qty_0'] + df['ask_qty_0'] + 1e-6)
    
    # OBI (5-Level Weighted)
    bid_vols = df[['bid_qty_0', 'bid_qty_1', 'bid_qty_2', 'bid_qty_3', 'bid_qty_4']].sum(axis=1)
    ask_vols = df[['ask_qty_0', 'ask_qty_1', 'ask_qty_2', 'ask_qty_3', 'ask_qty_4']].sum(axis=1)
    df['obi_deep'] = (bid_vols - ask_vols) / (bid_vols + ask_vols + 1e-6)
    
    # Micro-price
    df['micro_price'] = (df['bid_price_0'] * df['ask_qty_0'] + df['ask_price_0'] * df['bid_qty_0']) / (df['bid_qty_0'] + df['ask_qty_0'] + 1e-6)
    
    return df.dropna().reset_index(drop=True)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Training on compute device: {device.upper()}")

    data_path = r"data\ticks\depth\date=2026-08-26\symbol=RAMBHAJO\1300.parquet"
    print(f"[*] Ingesting depth snapshot: {data_path}")
    df = load_and_preprocess(data_path)

    # Vectorized Environment
    env = DummyVecEnv([lambda: TachyonL2Env(df)])

    # SOTA PPO Policy with hardware acceleration
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourages strategic exploration
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        device=device
    )

    print("[*] Commencing training across simulated order book steps...")
    model.learn(total_timesteps=50_000)

    os.makedirs("models", exist_ok=True)
    model.save("models/tachyon_ppo_l2")
    print("[✓] Model weights serialized to models/tachyon_ppo_l2.zip")