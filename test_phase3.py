import time
from models import MarketTick
from candle_engine import CandleEngine
from logger import setup_logger

def test_phase_3():
    log = setup_logger("Tachyon.Core")
    log.info("Starting Phase 3 Verification Test...")

    engine = CandleEngine(interval_minutes=5)
    token = "11536"
    
    # Base timestamp (e.g., 09:15:00 epoch equivalent)
    base_ts = 1690000000000  

    ticks = [
        # --- Candle 1 (09:15 Bin) ---
        MarketTick(token, base_ts + 10000, time.time(), 1, 100.0, 10, 1000),
        MarketTick(token, base_ts + 60000, time.time(), 2, 102.0, 10, 1500), # High: 102, Vol: 500
        # --- Candle 2 (09:20 Bin) --- Triggers Candle 1 close
        MarketTick(token, base_ts + 310000, time.time(), 3, 101.5, 10, 1500), # Initial tick for Bar 2
        MarketTick(token, base_ts + 450000, time.time(), 4, 103.0, 10, 3000), # Vol delta: 1500
        # --- Candle 3 (09:25 Bin) --- Triggers Candle 2 close
        MarketTick(token, base_ts + 610000, time.time(), 5, 104.0, 10, 3000), # Initial tick for Bar 3
    ]

    for i, tick in enumerate(ticks):
        log.info(f"Ingesting Tick {i+1} @ LTP: {tick.ltp}")
        candle = engine.process_tick(tick)
        
        if candle:
            log.info(f"    --> VALIDATED CANDLE EMITTED: Type={type(candle).__name__}")
            log.info(f"        O:{candle.open} H:{candle.high} L:{candle.low} C:{candle.close}")
            log.info(f"        VWAP:{candle.vwap} | ATR:{candle.atr} | RVOL:{candle.rvol}x | Ticks:{candle.ticks}")

    log.info("Phase 3 Verification Complete.")

if __name__ == "__main__":
    test_phase_3()
