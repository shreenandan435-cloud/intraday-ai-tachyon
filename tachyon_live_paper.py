import time
import numpy as np
from stable_baselines3 import PPO

class TachyonLivePaperExecutor:
    def __init__(self, model_path: str, initial_capital: float = 10000.0):
        print(f"[*] Initializing Tachyon SOTA Paper Engine...")
        self.model = PPO.load(model_path, device="cpu")
        self.capital = initial_capital
        self.position = 0          # 0 = Flat, 1 = Long
        self.entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.ROUNDTRIP_FEE_BPS = 3.0
        print(f"[✓] Model loaded. Capital: ₹{self.capital:.2f} | Risk Daemon: Active (-20 bps stop)")

    def process_l2_tick(self, symbol: str, bid_prices, bid_qtys, ask_prices, ask_qtys):
        """Processes a single incoming Level 2 order book snapshot in real-time."""
        bid_0, ask_0 = float(bid_prices[0]), float(ask_prices[0])
        
        # Guard against crossed or unpopulated books
        if bid_0 <= 0 or ask_0 <= 0 or ask_0 <= bid_0:
            return

        # 1. Stationary Microstructure Features
        spread = ask_0 - bid_0
        mid = (ask_0 + bid_0) / 2.0
        spread_bps = (spread / mid) * 10000.0
        obi_l1 = (bid_qtys[0] - ask_qtys[0]) / (bid_qtys[0] + ask_qtys[0] + 1e-6)
        
        total_bid_vol = sum(bid_qtys[:5])
        total_ask_vol = sum(ask_qtys[:5])
        obi_deep = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-6)
        
        micro_price = (bid_0 * ask_qtys[0] + ask_0 * bid_qtys[0]) / (bid_qtys[0] + ask_qtys[0] + 1e-6)
        micro_delta_bps = ((micro_price - mid) / mid) * 10000.0

        # 2. Risk Daemon Evaluation (Stop-Loss Override)
        unrealized_bps = 0.0
        if self.position == 1 and self.entry_price > 0:
            unrealized_bps = ((mid - self.entry_price) / self.entry_price) * 10000.0
            
            # Emergency Stop Ejection
            if unrealized_bps <= -20.0:
                self._execute_exit(symbol, bid_0, reason="RISK DAEMON STOP-LOSS (-20 bps)")
                return

        # 3. Neural Network State Vector
        obs = np.array([
            spread_bps,
            micro_delta_bps,
            obi_l1,
            obi_deep,
            float(self.position),
            unrealized_bps
        ], dtype=np.float32)

        # 4. Model Decision
        target_pos, _ = self.model.predict(obs, deterministic=True)

        # 5. Order State Machine
        if self.position == 0 and target_pos == 1:
            self._execute_entry(symbol, ask_0, obi_l1, spread_bps)
        elif self.position == 1 and target_pos == 0:
            self._execute_exit(symbol, bid_0, reason="TACHYON ALPHA EXIT")

    def _execute_entry(self, symbol: str, ask_price: float, obi: float, spread_bps: float):
        self.position = 1
        self.entry_price = ask_price
        print(f"\n[ENTRY LONG] {symbol} @ ₹{ask_price:.2f} | OBI-L1: {obi:+.2f} | Spread: {spread_bps:.1f} bps")

    def _execute_exit(self, symbol: str, bid_price: float, reason: str):
        gross_pnl = bid_price - self.entry_price
        fee = self.entry_price * (self.ROUNDTRIP_FEE_BPS / 10000.0)
        net_pnl = gross_pnl - fee
        
        self.capital += net_pnl
        self.total_trades += 1
        if net_pnl > 0:
            self.winning_trades += 1

        win_rate = (self.winning_trades / self.total_trades) * 100.0
        print(f"[EXIT LONG]  {symbol} @ ₹{bid_price:.2f} | Net PnL: ₹{net_pnl:+.2f} | Reason: {reason}")
        print(f"[*] Capital: ₹{self.capital:.2f} | Total Trades: {self.total_trades} | WinRate: {win_rate:.1f}%\n")
        
        self.position = 0
        self.entry_price = 0.0

if __name__ == "__main__":
    runner = TachyonLivePaperExecutor("models/tachyon_brain_generalized")
    print("\n[✓] Tachyon Paper Execution Layer is operational and awaiting WebSocket ticks.")
