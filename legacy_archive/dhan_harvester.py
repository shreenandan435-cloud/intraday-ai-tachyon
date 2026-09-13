import struct
import time
import os
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict

# 20-Level Microstructure Schema
L3_SCHEMA = pa.schema([
    ("recv_ts_ns", pa.int64()),
    ("security_id", pa.int32()),
    ("bid_prices", pa.list_(pa.float64(), 20)),
    ("bid_quantities", pa.list_(pa.int32(), 20)),
    ("bid_orders", pa.list_(pa.int32(), 20)),
    ("ask_prices", pa.list_(pa.float64(), 20)),
    ("ask_quantities", pa.list_(pa.int32(), 20)),
    ("ask_orders", pa.list_(pa.int32(), 20)),
    ("spread_bps", pa.float32()),
    ("micro_price", pa.float64())
])

class DhanParquetHarvester:
    def __init__(self, output_dir="./market_lake/raw_l3"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer = []
        self.batch_size = 5000
        
        # State tracker to merge staggered Bid and Ask packets
        self.books = defaultdict(lambda: {
            "bid_prices": [0.0]*20, "bid_quantities": [0]*20, "bid_orders": [0]*20,
            "ask_prices": [0.0]*20, "ask_quantities": [0]*20, "ask_orders": [0]*20,
        })

    def ingest(self, raw_bytes: bytes):
        recv_ns = time.time_ns()
        offset = 0
        total_len = len(raw_bytes)
        updated_secs = set()
        
        # Parse stacked binary payload (332 bytes per depth side)
        while offset < total_len:
            if offset + 12 > total_len:
                break
                
            msg_len = struct.unpack_from("<h", raw_bytes, offset)[0]
            if msg_len <= 0 or offset + msg_len > total_len:
                break
                
            feed_code = raw_bytes[offset + 2]
            sec_id = struct.unpack_from("<I", raw_bytes, offset + 4)[0]
            
            # Feed Codes: 41 = Bid (Buy), 51 = Ask (Sell)
            if feed_code in (41, 51):
                data_offset = offset + 12
                prices, quantities, orders = [], [], []
                
                # Unpack 20 levels: Price (double), Qty (uint32), Orders (uint32)
                for _ in range(20):
                    p, q, o = struct.unpack_from("<dII", raw_bytes, data_offset)
                    prices.append(p)
                    quantities.append(q)
                    orders.append(o)
                    data_offset += 16
                    
                if feed_code == 41:
                    self.books[sec_id]["bid_prices"] = prices
                    self.books[sec_id]["bid_quantities"] = quantities
                    self.books[sec_id]["bid_orders"] = orders
                elif feed_code == 51:
                    self.books[sec_id]["ask_prices"] = prices
                    self.books[sec_id]["ask_quantities"] = quantities
                    self.books[sec_id]["ask_orders"] = orders
                    
                updated_secs.add(sec_id)
                
            offset += msg_len
            
        # Snapshot the combined L3 book for the ML tensor
        for sec_id in updated_secs:
            book = self.books[sec_id]
            bp0, ap0 = book["bid_prices"][0], book["ask_prices"][0]
            bq0, aq0 = book["bid_quantities"][0], book["ask_quantities"][0]
            
            mid = (bp0 + ap0) / 2.0 if (bp0 + ap0) > 0 else 1.0
            spread_bps = float(((ap0 - bp0) / mid) * 10_000) if mid > 1.0 else 0.0
            
            total_top_qty = bq0 + aq0
            micro_price = (bp0 * aq0 + ap0 * bq0) / total_top_qty if total_top_qty > 0 else mid
            
            self.buffer.append({
                "recv_ts_ns": recv_ns,
                "security_id": sec_id,
                "bid_prices": book["bid_prices"],
                "bid_quantities": book["bid_quantities"],
                "bid_orders": book["bid_orders"],
                "ask_prices": book["ask_prices"],
                "ask_quantities": book["ask_quantities"],
                "ask_orders": book["ask_orders"],
                "spread_bps": spread_bps,
                "micro_price": float(micro_price)
            })
            
        if len(self.buffer) >= self.batch_size:
            self.flush_to_disk()

    def flush_to_disk(self):
        if not self.buffer:
            return
            
        date_str = time.strftime("%Y-%m-%d")
        dest_path = self.output_dir / f"date={date_str}"
        dest_path.mkdir(parents=True, exist_ok=True)
        
        file_name = dest_path / f"depth_batch_{int(time.time())}.parquet"
        
        table = pa.Table.from_pylist(self.buffer, schema=L3_SCHEMA)
        pq.write_table(table, file_name, compression="zstd", compression_level=7)
        self.buffer.clear()
        print(f"[{time.strftime('%H:%M:%S')}] Flushed batch to {file_name}")
