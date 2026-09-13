from models import Candle, MarketRegime
from market_context import MarketContextEngine
from logger import setup_logger

def test_phase_4():
    log = setup_logger("Tachyon.Core")
    log.info("Starting Phase 4 Verification Test...")

    engine = MarketContextEngine()

    # --- SCENARIO 1: Bullish Market, Strong Stock ---
    log.info("--- SCENARIO 1: Bullish Market ---")
    nifty_open = 24000.0
    nifty_candle_up = Candle(
        token="99926000", timestamp=1690000000,
        open=24050.0, high=24150.0, low=24040.0, close=24120.0, # Up 0.5% from open
        volume=100000, ticks=5000, vwap=24080.0, atr=30.0
    )
    
    tcs_open = 4000.0
    tcs_candle_strong = Candle(
        token="11536", timestamp=1690000000,
        open=4020.0, high=4070.0, low=4010.0, close=4060.0, # Up 1.5% from open
        volume=50000, ticks=2000, vwap=4040.0, atr=8.0
    )

    regime_up = engine.classify_regime(nifty_candle_up, nifty_open)
    rs_strong = engine.calculate_relative_strength(tcs_candle_strong, tcs_open, nifty_candle_up, nifty_open)
    
    log.info(f"Market Regime: {regime_up.name}")
    log.info(f"TCS Relative Strength Spread: +{rs_strong}% vs Benchmark")


    # --- SCENARIO 2: Bearish Market Flush ---
    log.info("--- SCENARIO 2: Bearish Market Flush ---")
    nifty_candle_down = Candle(
        token="99926000", timestamp=1690003000,
        open=23950.0, high=23980.0, low=23850.0, close=23880.0, # Down -0.5% from open
        volume=120000, ticks=6000, vwap=23920.0, atr=35.0
    )
    
    regime_down = engine.classify_regime(nifty_candle_down, nifty_open)
    log.info(f"Market Regime: {regime_down.name}")
    if regime_down == MarketRegime.TRENDING_DOWN:
        log.info("[SAFETY LOCK] System should block new LONG breakouts.")

    log.info("Phase 4 Verification Complete.")

if __name__ == "__main__":
    test_phase_4()
