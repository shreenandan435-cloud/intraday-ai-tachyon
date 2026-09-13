import time
import queue
import threading
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

class TachyonLiveEngine:
    def __init__(self, model_path: str, initial_capital: float = 10000.0):
        print(f"[*] Loading model weights from: {model_path}")
        self.model = PPO.load(model_path, device="cpu")
        self.capital = initial_capital
        
        # Position state
        self.position = 0          # 0 = Flat, 1 = Long
        self.entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.ROUNDTRIP_FEE_BPS = 3.0
        
        # Last known book state for graceful shutdown
        self.last_bid = 0.0
        self.last_ts = ""
        self.last_symbol = ""
        
        # Ingestion queue
        self.tick_queue = queue.Queue(maxsize=50000)
        self.is_running = True
        self.tick_count = 0

    def enqueue_l2_snapshot(self, symbol: str, timestamp, bid_prices, bid_qtys, ask_prices, ask_qtys):
        try:
            self.tick_queue.put_nowait((symbol, timestamp, bid_prices, bid_qtys, ask_prices, ask_qtys))
        except queue.Full:
            pass

    def start_inference_loop(self):
        print("[✓] Inference worker thread active. Processing depth queue...")
        
        while self.is_running or not self.tick_queue.empty():
            try:
                symbol, ts, bid_prices, bid_qtys, ask_prices, ask_qtys = self.tick_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            bid_0 = float(bid_prices[0])
            ask_0 = float(ask_prices[0])

            if bid_0 <= 0 or ask_0 <= 0 or ask_0 <= bid_0:
                self.tick_queue.task_done()
                continue

            self.last_bid = bid_0
            self.last_ts = ts
            self.last_symbol = symbol
            self.tick_count += 1

            # 1. Feature Engineering
            spread = ask_0 - bid_0
            mid = (ask_0 + bid_0) / 2.0
            spread_bps = (spread / mid) * 10000.0
            obi_l1 = (bid_qtys[0] - ask_qtys[0]) / (bid_qtys[0] + ask_qtys[0] + 1e-6)

            total_bid_vol = sum(bid_qtys[:5])
            total_ask_vol = sum(ask_qtys[:5])
            obi_deep = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-6)

            micro_price = (bid_0 * ask_qtys[0] + ask_0 * bid_qtys[0]) / (bid_qtys[0] + ask_qtys[0] + 1e-6)
            micro_delta_bps = ((micro_price - mid) / mid) * 10000.0

            # 2. Risk Daemon & Unrealized PnL
            unrealized_bps = 0.0
            if self.position == 1 and self.entry_price > 0:
                unrealized_bps = ((mid - self.entry_price) / self.entry_price) * 10000.0
                
                # Periodic Heartbeat Monitor
                if self.tick_count % 1000 == 0:
                    paper_pnl = bid_0 - self.entry_price
                    print(f"[{ts}] [HEARTBEAT] Position Active | Mid: ₹{mid:.2f} | Unrealized: {unrealized_bps:+.1f} bps (₹{paper_pnl:+.2f})")

                # Hard Stop-Loss Ejection (-20 bps)
                if unrealized_bps <= -20.0:
                    self._execute_exit(symbol, bid_0, ts, reason="RISK DAEMON STOP-LOSS (-20 bps)")
                    self.tick_queue.task_done()
                    continue

                # EOD Auto Square-Off Check (15:15 IST)
                if "15:15" in str(ts):
                    self._execute_exit(symbol, bid_0, ts, reason="INTRADAY 15:15 EOD SQUARE-OFF")
                    self.tick_queue.task_done()
                    continue

            # 3. Model Inference
            obs = np.array([
                spread_bps,
                micro_delta_bps,
                obi_l1,
                obi_deep,
                float(self.position),
                unrealized_bps
            ], dtype=np.float32)

            target_pos, _ = self.model.predict(obs, deterministic=True)

            # 4. State Transitions
            if self.position == 0 and target_pos == 1:
                self._execute_entry(symbol, ask_0, ts, obi_l1, spread_bps)
            elif self.position == 1 and target_pos == 0:
                self._execute_exit(symbol, bid_0, ts, reason="TACHYON ALPHA EXIT")

            self.tick_queue.task_done()

        # Final Stream Shutdown Flush
        if self.position == 1 and self.entry_price > 0 and self.last_bid > 0:
            self._execute_exit(self.last_symbol, self.last_bid, self.last_ts, reason="STREAM END FLUSH")

    def _execute_entry(self, symbol: str, ask_price: float, ts, obi: float, spread_bps: float):
        self.position = 1
        self.entry_price = ask_price
        print(f"\n[{ts}] [ENTRY LONG] {symbol} @ ₹{ask_price:.2f} | OBI-L1: {obi:+.2f} | Spread: {spread_bps:.1f} bps")

    def _execute_exit(self, symbol: str, bid_price: float, ts, reason: str):
        gross_pnl = bid_price - self.entry_price
        fee = self.entry_price * (self.ROUNDTRIP_FEE_BPS / 10000.0)
        net_pnl = gross_pnl - fee

        self.capital += net_pnl
        self.total_trades += 1
        if net_pnl > 0:
            self.winning_trades += 1

        win_rate = (self.winning_trades / self.total_trades) * 100.0
        print(f"[{ts}] [EXIT LONG]  {symbol} @ ₹{bid_price:.2f} | Net: ₹{net_pnl:+.2f} | Reason: {reason}")
        print(f"    --> Capital: ₹{self.capital:.2f} | Total Trades: {self.total_trades} | WinRate: {win_rate:.1f}%\n")

        self.position = 0
        self.entry_price = 0.0

def run_weekend_replay(engine: TachyonLiveEngine, parquet_path: str, playback_speed: float = 0.0001):
    print(f"[*] Initiating Weekend Replay from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    symbol = parquet_path.split("symbol=")[-1].split("\\")[0]
    
    bid_p_cols = [f'bid_price_{i}' for i in range(5)]
    bid_q_cols = [f'bid_qty_{i}' for i in range(5)]
    ask_p_cols = [f'ask_price_{i}' for i in range(5)]
    ask_q_cols = [f'ask_qty_{i}' for i in range(5)]

    print(f"[*] Streaming {len(df)} L2 updates through ingestion socket...")
    for idx, row in df.iterrows():
        b_prices = row[bid_p_cols].values.astype(float)
        b_qtys = row[bid_q_cols].values.astype(float)
        a_prices = row[ask_p_cols].values.astype(float)
        a_qtys = row[ask_q_cols].values.astype(float)
        ts = row.get('timestamp', f"TICK_{idx:05d}")

        engine.enqueue_l2_snapshot(symbol, ts, b_prices, b_qtys, a_prices, a_qtys)
        
        if playback_speed > 0:
            time.sleep(playback_speed)

    while not engine.tick_queue.empty():
        time.sleep(0.01)
        
    engine.is_running = False

if __name__ == "__main__":
    engine = TachyonLiveEngine(model_path="models/tachyon_brain_generalized")

    infer_thread = threading.Thread(target=engine.start_inference_loop)
    infer_thread.start()

    sample_tape = r"data\ticks\depth\date=2026-08-26\symbol=RAMBHAJO\1300.parquet"
    run_weekend_replay(engine, sample_tape, playback_speed=0.0001)

    infer_thread.join()
    print("[✓] Weekend Replay complete. All positions closed.")
