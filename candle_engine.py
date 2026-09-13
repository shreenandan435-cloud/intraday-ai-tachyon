import logging
from typing import Dict, Optional
from models import MarketTick, Candle
from microstructure import RollingMicrostructure

class CandleEngine:
    def __init__(self, interval_minutes: int = 5, atr_period: int = 14, vol_lookback: int = 20):
        self.interval_sec = interval_minutes * 60
        self.active_bars = {}  
        self.microstructures: Dict[str, RollingMicrostructure] = {}
        
        self.atr_period = atr_period
        self.vol_lookback = vol_lookback
        self.logger = logging.getLogger("Tachyon.CandleEngine")

    def process_tick(self, tick: MarketTick) -> Optional[Candle]:
        """
        Ingests a verified MarketTick and returns a completed Candle object if a bin boundary is crossed.
        """
        token = tick.token
        
        # Align timestamp strictly to the exchange epoch to prevent local drift
        exchange_sec = tick.exchange_timestamp / 1000.0
        if exchange_sec <= 0:
            exchange_sec = tick.local_receive_timestamp
            
        bin_timestamp = (int(exchange_sec) // self.interval_sec) * self.interval_sec
        
        # Initialize new tracker if first time seeing token
        if token not in self.active_bars:
            self._init_new_bar(token, tick, bin_timestamp)
            if token not in self.microstructures:
                self.microstructures[token] = RollingMicrostructure(self.atr_period, self.vol_lookback)
            return None
        
        bar = self.active_bars[token]
        
        # Time threshold crossed -> Finalize current bar and roll over
        if bin_timestamp > bar['bin_timestamp']:
            completed_candle = self._finalize_bar(token)
            self._init_new_bar(token, tick, bin_timestamp)
            return completed_candle
        
        # Update active bar state
        bar['high'] = max(bar['high'], tick.ltp)
        bar['low'] = min(bar['low'], tick.ltp)
        bar['close'] = tick.ltp
        bar['ticks'] += 1
        
        vol_delta = max(0, tick.cumulative_volume - bar['last_cum_vol'])
        bar['volume'] += vol_delta
        bar['last_cum_vol'] = tick.cumulative_volume
        bar['pv_sum'] += (tick.ltp * vol_delta)
        
        return None

    def _init_new_bar(self, token: str, tick: MarketTick, bin_timestamp: int):
        self.active_bars[token] = {
            'bin_timestamp': bin_timestamp,
            'open': tick.ltp,
            'high': tick.ltp,
            'low': tick.ltp,
            'close': tick.ltp,
            'volume': 0,
            'ticks': 1,
            'pv_sum': 0.0,
            'last_cum_vol': tick.cumulative_volume
        }

    def _finalize_bar(self, token: str) -> Candle:
        bar = self.active_bars[token]
        vwap = (bar['pv_sum'] / bar['volume']) if bar['volume'] > 0 else bar['close']
        
        ms = self.microstructures[token]
        atr, rvol, tick_ratio = ms.update_and_calculate(
            high=bar['high'],
            low=bar['low'],
            close=bar['close'],
            volume=bar['volume'],
            ticks=bar['ticks']
        )
        
        candle = Candle(
            token=token,
            timestamp=bar['bin_timestamp'],
            open=bar['open'],
            high=bar['high'],
            low=bar['low'],
            close=bar['close'],
            volume=bar['volume'],
            ticks=bar['ticks'],
            vwap=round(vwap, 2),
            atr=round(atr, 2),
            rvol=round(rvol, 2),
            tick_ratio=round(tick_ratio, 2)
        )
        self.logger.info(f"Candle Closed | {token} | C:{candle.close} | V:{candle.volume} | RVOL:{candle.rvol}x | ATR:{candle.atr}")
        return candle
