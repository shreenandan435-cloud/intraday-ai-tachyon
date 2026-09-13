import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

class GeometricBreakoutEngine:
    def __init__(self, rvol_threshold=2.0, tick_ratio_threshold=1.5, min_extrema_order=3):
        self.rvol_threshold = rvol_threshold
        self.tick_ratio_threshold = tick_ratio_threshold
        self.order = min_extrema_order

    def evaluate_breakout(self, df: pd.DataFrame):
        """
        Evaluates whether the latest closed bar represents a valid geometric breakout.
        Returns: (is_valid: bool, setup_info: dict)
        """
        if len(df) < 20:
            return False, {"reason": "Insufficient candle history (<20)"}

        # 1. Detect structural swing highs across historical bars (excluding latest)
        highs = df['high'].iloc[:-1].values
        swing_high_indices = argrelextrema(highs, np.greater, order=self.order)[0]
        
        if len(swing_high_indices) == 0:
            return False, {"reason": "No structural swing high identified"}

        recent_swing_high_idx = swing_high_indices[-1]
        resistance_level = highs[recent_swing_high_idx]
        
        # Structural Stop Loss: Lowest point between the resistance formation and breakout
        structural_support = df['low'].iloc[recent_swing_high_idx:-1].min()

        latest_bar = df.iloc[-1]
        atr = latest_bar['atr'] if latest_bar['atr'] > 0 else 1.0

        # 2. Breakout Geometry Verification (Clean acceptance, not a wick sweep)
        breakout_barrier = resistance_level + (0.15 * atr)
        if latest_bar['close'] <= breakout_barrier:
            return False, {"reason": "Close below resistance + ATR buffer"}

        # 3. Relative Volume (RVOL) Confirmation (20-period baseline)
        avg_volume = df['volume'].iloc[-21:-1].mean()
        rvol = (latest_bar['volume'] / avg_volume) if avg_volume > 0 else 1.0
        if rvol < self.rvol_threshold:
            return False, {"reason": f"RVOL {rvol:.2f} below target {self.rvol_threshold}x"}

        # 4. Trade Count Confirmation (Prevents single-trade block spikes)
        avg_ticks = df['ticks'].iloc[-21:-1].mean()
        tick_ratio = (latest_bar['ticks'] / avg_ticks) if avg_ticks > 0 else 1.0
        if tick_ratio < self.tick_ratio_threshold:
            return False, {"reason": f"Tick count ratio {tick_ratio:.2f} below {self.tick_ratio_threshold}x"}

        # 5. VWAP Confluence
        if latest_bar['close'] < latest_bar['vwap']:
            return False, {"reason": "Price trading below intraday VWAP"}

        setup = {
            "entry_price": float(latest_bar['close']),
            "resistance": float(resistance_level),
            "stop_loss": float(structural_support),
            "atr": float(atr),
            "rvol": float(rvol),
            "tick_ratio": float(tick_ratio)
        }
        return True, setup
