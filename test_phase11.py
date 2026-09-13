import time
from models import MarketTick
from candle_engine import CandleEngine
from paper_execution import PaperExecutionHarness
from config import SystemConfig
from replay_engine import ReplayEngine
from logger import setup_logger

def synthetic_tick_generator():
    """Generates 120 ticks (representing 2 minutes of 1-tick-per-second data)"""
    # 09:15:00 AM Epoch
    base_ts = 1690000000000 
    
    for i in range(1, 121):
        yield MarketTick(
            token="11536",
            exchange_timestamp=base_ts + (i * 1000), 
            local_receive_timestamp=time.time(), # Ignored by replay, uses exchange_ts
            sequence_number=i,
            ltp=4000.0 + (i * 0.1),
            last_traded_quantity=10,
            cumulative_volume=i * 10
        )

def test_phase11():
    # Fix from Phase 10: Initialize the ROOT "Tachyon" logger so all sub-modules inherit formats!
    log = setup_logger("Tachyon")
    log.info("Starting Phase 11 Verification Test...")

    sys_config = SystemConfig(trading_mode="PAPER")
    harness = PaperExecutionHarness(sys_config)
    
    # 1-minute candles for rapid testing (should yield exactly 2 candles from 120 seconds of ticks)
    candle_engine = CandleEngine(interval_minutes=1) 

    replay = ReplayEngine(candle_engine, harness)
    
    # Run the offline replay pump
    start_time = time.time()
    replay.run_replay(synthetic_tick_generator())
    elapsed = (time.time() - start_time) * 1000
    
    log.info(f"Phase 11 Verification Complete. Replay speed: {elapsed:.2f} ms")

if __name__ == "__main__":
    test_phase11()
