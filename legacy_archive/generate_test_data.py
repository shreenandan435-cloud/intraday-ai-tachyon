import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

print("[+] Generating 10,000 synthetic L3 snapshots for plumbing verification...")
data = {}
for i in range(20):
    data[f'bid_p_{i}'] = 2500.0 - (i * 0.5) - np.random.uniform(0, 0.2, 10000)
    data[f'bid_q_{i}'] = np.random.randint(100, 5000, 10000)
    data[f'ask_p_{i}'] = 2500.5 + (i * 0.5) + np.random.uniform(0, 0.2, 10000)
    data[f'ask_q_{i}'] = np.random.randint(100, 5000, 10000)

df = pd.DataFrame(data)
df['target_spread'] = np.random.uniform(1.0, 6.0, 10000)

table = pa.Table.from_pandas(df)
pq.write_table(table, 'synthetic_test.parquet', compression='zstd')
print("[✓] synthetic_test.parquet created successfully.")
