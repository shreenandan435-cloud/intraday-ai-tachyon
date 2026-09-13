import time
from collections import defaultdict, deque
import pandas as pd
import numpy as np

class CandleAccumulator:
    def __init__(self, candle_minutes=5, max_history=100):
        self.candle_interval_sec = candle_minutes * 60
        self.max_history = max_history
        
        # Current active bar state: {token: {open, high, low, close, volume, ticks, vwap_pv, start_ts}}
        self.active_bars = {}
        # Completed historical bars: {token: deque([bar_dict, ...], maxlen=max_history)}
        self.completed_bars = defaultdict(lambda: deque(maxlen=self.max_history))

    def on_tick(self, token: str, ltp: float, total_vol: int):
        now = time.time()
        
        if token not in self.active_bars:
            self._init_new_bar(token, ltp, total_vol, now)
            return None

        bar = self.active_bars[token]
        
        # Check if the active candle duration has elapsed
        if now - bar["start_ts"] >= self.candle_interval_sec:
            completed = self._finalize_bar(token)
            self._init_new_bar(token, ltp, total_vol, now)
            return completed

        # Update active bar metrics
        bar["high"] = max(bar["high"], ltp)
        bar["low"] = min(bar["low"], ltp)
        bar["close"] = ltp
        bar["ticks"] += 1
        
        # Calculate volume delta from exchange cumulative volume counter
        vol_delta = max(0, total_vol - bar["last_cum_vol"])
        bar["volume"] += vol_delta
        bar["last_cum_vol"] = total_vol
        bar["pv_sum"] += (ltp * vol_delta)
        
        return None

    def _init_new_bar(self, token, ltp, total_vol, start_ts):
        self.active_bars[token] = {
            "token": token,
            "start_ts": start_ts,
            "open": ltp,
            "high": ltp,
            "low": ltp,
            "close": ltp,
            "volume": 0,
            "ticks": 1,
            "pv_sum": 0.0,
            "last_cum_vol": total_vol
        }

    def _finalize_bar(self, token):
        bar = self.active_bars[token]
        vwap = (bar["pv_sum"] / bar["volume"]) if bar["volume"] > 0 else bar["close"]
        
        completed_record = {
            "timestamp": bar["start_ts"],
            "open": bar["open"],
            "high": bar["high"],
            "low": bar["low"],
            "close": bar["close"],
            "volume": bar["volume"],
            "ticks": bar["ticks"],
            "vwap": vwap
        }
        self.completed_bars[token].append(completed_record)
        return completed_record

    def get_dataframe(self, token: str) -> pd.DataFrame:
        bars = list(self.completed_bars[token])
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        
        # Calculate Average True Range (ATR 14)
        if len(df) >= 2:
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=min(14, len(df)), min_periods=1).mean()
        else:
            df['atr'] = df['high'] - df['low']
            
        return df
