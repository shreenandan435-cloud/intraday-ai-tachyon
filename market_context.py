import logging
from models import MarketRegime, Candle

class MarketContextEngine:
    def __init__(self, benchmark_token: str = "99926000", trend_threshold_pct: float = 0.20):
        # "99926000" is a standard placeholder for Nifty 50 Spot in Angel One. 
        # You can adjust this to Nifty Futures ("NIFTY") token based on your feed preference.
        self.benchmark_token = benchmark_token
        self.trend_threshold = trend_threshold_pct
        self.logger = logging.getLogger("Tachyon.MarketContext")

    def classify_regime(self, benchmark_candle: Candle, day_open_price: float) -> MarketRegime:
        """
        Deterministically classifies the intraday market regime based on 
        the benchmark's open-to-current return and VWAP relationship.
        """
        if day_open_price <= 0:
            return MarketRegime.UNKNOWN
            
        day_return_pct = ((benchmark_candle.close - day_open_price) / day_open_price) * 100.0
        
        # Rule 1: Must be up more than the threshold AND trading above intraday VWAP
        if day_return_pct > self.trend_threshold and benchmark_candle.close > benchmark_candle.vwap:
            return MarketRegime.TRENDING_UP
            
        # Rule 2: Must be down more than the threshold AND trading below intraday VWAP
        elif day_return_pct < -self.trend_threshold and benchmark_candle.close < benchmark_candle.vwap:
            return MarketRegime.TRENDING_DOWN
            
        # Rule 3: Choppy / No clear directional conviction
        else:
            return MarketRegime.RANGING

    def calculate_relative_strength(self, stock_candle: Candle, stock_day_open: float, 
                                    benchmark_candle: Candle, benchmark_day_open: float) -> float:
        """
        Calculates intraday Relative Strength (RS) percentage spread.
        Positive value means the stock is outperforming the benchmark.
        """
        if stock_day_open <= 0 or benchmark_day_open <= 0:
            return 0.0
            
        stock_return = ((stock_candle.close - stock_day_open) / stock_day_open) * 100.0
        benchmark_return = ((benchmark_candle.close - benchmark_day_open) / benchmark_day_open) * 100.0
        
        rs_spread = stock_return - benchmark_return
        return round(rs_spread, 3)
