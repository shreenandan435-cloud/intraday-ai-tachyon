import pandas as pd

# Targeting a successfully flushed ~943 KB NSE order book file
target = r"C:\Users\Shree\intraday-ai-tachyon\data\ticks\depth\date=2026-08-26\symbol=RAMBHAJO\1300.parquet"

try:
    df = pd.read_parquet(target)
    print("=== DATASET SHAPE ===")
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}\n")
    print("=== SCHEMA ===")
    print(df.dtypes, "\n")
    print("=== FIRST 3 ROWS ===")
    print(df.head(3).to_string())
except Exception as e:
    print(f"Failed to parse: {e}")