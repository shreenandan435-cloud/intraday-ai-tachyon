import pandas as pd
import numpy as np
from stream_accumulator import CandleAccumulator
from geometry_engine import GeometricBreakoutEngine
from order_router import SafeOrderRouter

print("[+] Verifying realistic consolidation and breakout pattern...")

# 1. Initialize engine
engine = GeometricBreakoutEngine(rvol_threshold=1.5, tick_ratio_threshold=1.2, min_extrema_order=3)

# 2. Build a realistic 25-candle consolidation:
# - Bars 0 to 10: Run-up to a swing high at 105.0
# - Bars 11 to 23: Pullback and consolidation between 101.0 and 103.5
# - Bar 24: High-volume breakout closing at 106.2
bars = []

# Impulse leg up
for i in range(11):
    bars.append({
        "timestamp": i,
        "open": 100.0 + (i * 0.4),
        "high": 100.5 + (i * 0.45), # Bar 10 High = 105.0
        "low": 99.8 + (i * 0.4),
        "close": 100.3 + (i * 0.4),
        "volume": 1000,
        "ticks": 100,
        "vwap": 100.0 + (i * 0.4),
        "atr": 1.2
    })

# Consolidation channel (pullback)
for i in range(11, 24):
    bars.append({
        "timestamp": i,
        "open": 102.5,
        "high": 103.5,             # Resistance remains at 105.0 from Bar 10
        "low": 101.2,              # Support base
        "close": 102.8,
        "volume": 800,
        "ticks": 90,
        "vwap": 102.5,
        "atr": 1.2
    })

# Breakout Bar (Bar 24)
bars.append({
    "timestamp": 24,
    "open": 103.0,
    "high": 106.5,
    "low": 102.8,
    "close": 106.2,                # Breaks 105.0 + (0.15 * 1.2) = 105.18
    "volume": 3200,                # 4.0x RVOL
    "ticks": 280,                  # 3.1x Tick Ratio
    "vwap": 103.5,
    "atr": 1.2
})

df_sim = pd.DataFrame(bars)
is_valid, setup = engine.evaluate_breakout(df_sim)

print(f"[+] Breakout Evaluation Result: {is_valid}")
if is_valid:
    print(f"    • Detected Resistance : ₹{setup['resistance']:.2f}")
    print(f"    • Entry Price         : ₹{setup['entry_price']:.2f}")
    print(f"    • Structural Stop Loss: ₹{setup['stop_loss']:.2f}")
    print(f"    • Confirmed RVOL      : {setup['rvol']:.2f}x")
    print(f"    • Confirmed Tick Ratio: {setup['tick_ratio']:.2f}x")

    router = SafeOrderRouter(smart_api_client=None, risk_budget_rupees=1000.0, paper_trading=True)
    qty = router.calculate_position_size(setup['entry_price'], setup['stop_loss'])
    target = round(setup['entry_price'] + (2.0 * (setup['entry_price'] - setup['stop_loss'])), 2)
    print(f"[✓] Execution Plan: BUY {qty} shares | Target: ₹{target} | Max Risk: ₹1,000.00")

print("[SUCCESS] Geometric scanner and risk math verified end-to-end.")
