import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from tachyon_env import TachyonL2Env

def load_and_preprocess(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df['spread'] = df['ask_price_0'] - df['bid_price_0']
    df['mid_price'] = (df['ask_price_0'] + df['bid_price_0']) / 2.0
    df['obi_l1'] = (df['bid_qty_0'] - df['ask_qty_0']) / (df['bid_qty_0'] + df['ask_qty_0'] + 1e-6)
    bid_vols = df[['bid_qty_0', 'bid_qty_1', 'bid_qty_2', 'bid_qty_3', 'bid_qty_4']].sum(axis=1)
    ask_vols = df[['ask_qty_0', 'ask_qty_1', 'ask_qty_2', 'ask_qty_3', 'ask_qty_4']].sum(axis=1)
    df['obi_deep'] = (bid_vols - ask_vols) / (bid_vols + ask_vols + 1e-6)
    df['micro_price'] = (df['bid_price_0'] * df['ask_qty_0'] + df['ask_price_0'] * df['bid_qty_0']) / (df['bid_qty_0'] + df['ask_qty_0'] + 1e-6)
    return df.dropna().reset_index(drop=True)

if __name__ == "__main__":
    # Load unseen out-of-sample data (the 14:00 hour block)
    test_file = r"C:\Users\Shree\intraday-ai-tachyon\data\ticks\depth\date=2026-08-26\symbol=RAMBHAJO\1400.parquet"
    print(f"[*] Loading Out-of-Sample Tape: {test_file}")
    df_test = load_and_preprocess(test_file)
    
    # Initialize Sandbox
    env = TachyonL2Env(df_test, initial_balance=10000.0)
    
    # Load Serialized Model
    model = PPO.load("models/tachyon_ppo_l2")
    
    obs, info = env.reset()
    done = False
    
    print("[*] Commencing Deterministic Backtest...")
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    pnl = info['balance'] - 10000.0
    
    print("\n=== OUT-OF-SAMPLE BACKTEST RESULTS ===")
    print(f"Total Trades Executed : {info['trades']}")
    print(f"Final Balance         : ₹{info['balance']:.2f}")
    print(f"Net Profit/Loss       : ₹{pnl:.2f}")
    
    if pnl > 0:
        print("\n[✓] ENGINE PROFITABLE ON UNSEEN DATA. READY FOR LIVE SHADOW MODE.")
    else:
        print("\n[X] OVERFITTING DETECTED. AGENT BLED TO FEES. REQUIRES WIDER TRAINING SET.")
