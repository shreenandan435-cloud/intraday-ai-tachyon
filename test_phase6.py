from models import Candle
from structure_engine import ConsolidationZone
from breakout_engine import BreakoutEngine, BreakoutResult
from logger import setup_logger

def test_phase_6():
    log = setup_logger("Tachyon.Core")
    log.info("Starting Phase 6 Verification Test...")

    engine = BreakoutEngine(atr_buffer=0.15, min_rvol=1.8, min_tick_ratio=1.3, max_extension_atr=1.0)
    
    # Established Zone from Phase 5 (Res: 103, Sup: 100, ATR: 2.0)
    zone = ConsolidationZone(is_valid=True, resistance_top=103.0, resistance_bottom=102.5, support_base=100.0, width_atr=1.5)
    
    # The Buffer required to clear = 103.0 + (0.15 * 2.0) = 103.30

    test_candles = [
        ("Candle 1: Weak Wick Sweep", Candle(
            "11536", 1690000000, open=102.0, high=104.0, low=101.5, close=103.1, # Closes below 103.30 buffer
            volume=50000, ticks=2000, vwap=102.5, atr=2.0, rvol=2.5, tick_ratio=2.0
        )),
        ("Candle 2: Low Volume Fakeout", Candle(
            "11536", 1690000000, open=102.0, high=104.0, low=101.5, close=103.5, # Good close
            volume=10000, ticks=500, vwap=102.5, atr=2.0, rvol=1.1, tick_ratio=1.0 # Fails RVOL 1.8x
        )),
        ("Candle 3: Overextended Chase", Candle(
            "11536", 1690000000, open=102.0, high=106.0, low=101.5, close=105.5, # Closes way past 103.0
            volume=50000, ticks=2000, vwap=102.5, atr=2.0, rvol=3.0, tick_ratio=2.5
        )),
        ("Candle 4: Perfect Institutional Breakout", Candle(
            "11536", 1690000000, open=102.0, high=104.0, low=101.5, close=103.5, # Closes above buffer, not overextended
            volume=50000, ticks=2000, vwap=102.5, atr=2.0, rvol=2.1, tick_ratio=1.5
        ))
    ]

    for name, candle in test_candles:
        log.info(f"--- Evaluating {name} ---")
        result = engine.evaluate_breakout(candle, zone)
        log.info(f"Validity: {result.is_valid} | Reason: {result.reason}")
        if result.is_valid:
            log.info(f"Entry: ₹{result.entry_price:.2f} | Stop: ₹{result.structural_stop:.2f}")

    log.info("Phase 6 Verification Complete.")

if __name__ == "__main__":
    test_phase_6()
