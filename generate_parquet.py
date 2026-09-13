import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

cols = (
    [f"bid_price_{i}" for i in range(5)]
    + [f"bid_qty_{i}" for i in range(5)]
    + [f"ask_price_{i}" for i in range(5)]
    + [f"ask_qty_{i}" for i in range(5)]
    + ["spread_tick", "obi_l1", "vwap_bid_bps", "vwap_ask_bps", "total_qty_log"]
)
arr = np.random.rand(10000, 25).astype(np.float32)
df = pd.DataFrame(arr, columns=cols)
pq.write_table(pa.Table.from_pandas(df), 'mock_binance_l2.parquet')
print('Parquet file generated')
