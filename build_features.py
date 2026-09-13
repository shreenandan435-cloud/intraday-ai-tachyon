import pandas as pd
import numpy as np

target = r"C:\Users\Shree\intraday-ai-tachyon\data\ticks\depth\date=2026-08-26\symbol=RAMBHAJO\1300.parquet"
df = pd.read_parquet(target)

# 1. Core Physics
df['spread'] = df['ask_price_0'] - df['bid_price_0']
df['mid_price'] = (df['ask_price_0'] + df['bid_price_0']) / 2.0

# 2. Order Book Imbalance (OBI) - Level 1
df['obi_l1'] = (df['bid_qty_0'] - df['ask_qty_0']) / (df['bid_qty_0'] + df['ask_qty_0'])

# 3. Weighted Volume Imbalance (All 5 Levels)
df['total_bid_vol'] = df[['bid_qty_0', 'bid_qty_1', 'bid_qty_2', 'bid_qty_3', 'bid_qty_4']].sum(axis=1)
df['total_ask_vol'] = df[['ask_qty_0', 'ask_qty_1', 'ask_qty_2', 'ask_qty_3', 'ask_qty_4']].sum(axis=1)
df['obi_deep'] = (df['total_bid_vol'] - df['total_ask_vol']) / (df['total_bid_vol'] + df['total_ask_vol'])

# 4. Micro-Price (Volume-weighted fair value)
df['micro_price'] = (df['bid_price_0'] * df['ask_qty_0'] + df['ask_price_0'] * df['bid_qty_0']) / (df['bid_qty_0'] + df['ask_qty_0'])

print("=== RL STATE VECTOR (FEATURES) ===")
print(df[['mid_price', 'spread', 'obi_l1', 'obi_deep', 'micro_price']].head(5).to_string())
