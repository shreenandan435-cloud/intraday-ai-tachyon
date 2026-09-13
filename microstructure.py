from collections import deque
import logging

class RollingMicrostructure:
    def __init__(self, atr_period=14, vol_lookback=20):
        self.atr_period = atr_period
        self.vol_lookback = vol_lookback
        
        self.prev_close = None
        self.tr_history = deque(maxlen=atr_period)
        self.vol_history = deque(maxlen=vol_lookback)
        self.tick_history = deque(maxlen=vol_lookback)
        
    def update_and_calculate(self, high: float, low: float, close: float, volume: int, ticks: int):
        """
        Calculates ATR, RVOL, and Tick Ratio strictly without look-ahead bias.
        Compares the current completed bar against the historically closed baseline.
        """
        # 1. True Range Calculation
        if self.prev_close is not None:
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        else:
            tr = high - low
            
        self.tr_history.append(tr)
        self.prev_close = close
        
        atr = sum(self.tr_history) / len(self.tr_history) if self.tr_history else 0.0
        
        # 2. RVOL (Relative Volume) Calculation BEFORE appending current volume
        avg_vol = sum(self.vol_history) / len(self.vol_history) if self.vol_history else volume
        rvol = (volume / avg_vol) if avg_vol > 0 else 1.0
        
        # 3. Tick Activity Expansion Calculation BEFORE appending current ticks
        avg_ticks = sum(self.tick_history) / len(self.tick_history) if self.tick_history else ticks
        tick_ratio = (ticks / avg_ticks) if avg_ticks > 0 else 1.0
        
        # 4. Store current values for the NEXT bar's baseline
        self.vol_history.append(volume)
        self.tick_history.append(ticks)
        
        return atr, rvol, tick_ratio
