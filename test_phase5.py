from models import Candle
from structure_engine import StructureEngine
from logger import setup_logger

def generate_simulated_chart(is_choppy: bool) -> list:
    candles = []
    for i in range(25):
        # Create peaks every 5 candles to trigger argrelextrema
        is_peak = (i % 5 == 0) and i > 0
        is_trough = (i % 5 == 2)
        
        if is_choppy:
            # Wild swings: Highs hit 110, Lows hit 90 (ATR=2.0)
            high = 110.0 if is_peak else 105.0
            low = 90.0 if is_trough else 95.0
            atr = 2.0
        else:
            # Tight compression: Highs cluster exactly at 103.0, Lows hold 100.0 (ATR=1.5)
            high = 103.0 if is_peak else 102.0
            low = 100.0 if is_trough else 101.0
            atr = 1.5
            
        candles.append(Candle(
            token="11536", timestamp=i*300, 
            open=101.5, high=high, low=low, close=101.5, 
            volume=1000, ticks=100, vwap=101.5, atr=atr
        ))
    return candles

def test_phase_5():
    log = setup_logger("Tachyon.Core")
    log.info("Starting Phase 5 Verification Test...")

    engine = StructureEngine(lookback=24, max_width_atr=3.5, min_touches=2)

    # --- SCENARIO 1: Choppy, Volatile Range ---
    log.info("--- SCENARIO 1: Choppy Range ---")
    choppy_candles = generate_simulated_chart(is_choppy=True)
    choppy_zone = engine.evaluate_consolidation(choppy_candles)
    
    log.info(f"Validity: {choppy_zone.is_valid}")
    log.info(f"Reason: {choppy_zone.reason}")
    log.info(f"Zone Width: {choppy_zone.width_atr:.2f} ATRs")

    # --- SCENARIO 2: Tight, Clean Consolidation ---
    log.info("--- SCENARIO 2: Tight Consolidation ---")
    tight_candles = generate_simulated_chart(is_choppy=False)
    tight_zone = engine.evaluate_consolidation(tight_candles)

    log.info(f"Validity: {tight_zone.is_valid}")
    log.info(f"Reason: {tight_zone.reason}")
    if tight_zone.is_valid:
        log.info(f"Resistance Band: ₹{tight_zone.resistance_bottom:.2f} - ₹{tight_zone.resistance_top:.2f}")
        log.info(f"Support Base (Stop Loss): ₹{tight_zone.support_base:.2f}")
        log.info(f"Compression Width: {tight_zone.width_atr:.2f} ATRs")

    log.info("Phase 5 Verification Complete.")

if __name__ == "__main__":
    test_phase_5()
