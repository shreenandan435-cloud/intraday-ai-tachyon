from orderbook_engine import OrderbookAnalyzer
from logger import setup_logger

def test_phase_7():
    log = setup_logger("Tachyon.Core")
    log.info("Starting Phase 7 Verification Test...")

    analyzer = OrderbookAnalyzer(max_spread_pct=0.10)
    
    ltp = 2200.80

    # --- SCENARIO 1: Tight, Liquid Book ---
    # Spread: 2201.20 (Ask) - 2200.40 (Bid) = 0.80 (0.036%)
    liquid_packet = {
        'best_5_buy_data': [
            {'price': 220040, 'quantity': 1500},
            {'price': 220000, 'quantity': 1200}
        ],
        'best_5_sell_data': [
            {'price': 220120, 'quantity': 1800},
            {'price': 220150, 'quantity': 2000}
        ]
    }
    
    log.info("--- SCENARIO 1: Liquid Order Book ---")
    state_liquid = analyzer.evaluate_liquidity(liquid_packet, ltp)
    log.info(f"Bid: ₹{state_liquid.best_bid:.2f} | Ask: ₹{state_liquid.best_ask:.2f} | Spread: ₹{state_liquid.spread_rupees:.2f} ({state_liquid.spread_pct:.3f}%)")
    log.info(f"Validity: {state_liquid.is_tradable} | Reason: {state_liquid.reason}")


    # --- SCENARIO 2: Illiquid / Wide Spread Book ---
    # Spread: 2202.80 (Ask) - 2198.80 (Bid) = 4.00 (0.181%)
    illiquid_packet = {
        'best_5_buy_data': [
            {'price': 219880, 'quantity': 100},
            {'price': 219800, 'quantity': 50}
        ],
        'best_5_sell_data': [
            {'price': 220280, 'quantity': 150},
            {'price': 220300, 'quantity': 80}
        ]
    }

    log.info("--- SCENARIO 2: Wide Spread ---")
    state_illiquid = analyzer.evaluate_liquidity(illiquid_packet, ltp)
    log.info(f"Bid: ₹{state_illiquid.best_bid:.2f} | Ask: ₹{state_illiquid.best_ask:.2f} | Spread: ₹{state_illiquid.spread_rupees:.2f} ({state_illiquid.spread_pct:.3f}%)")
    log.info(f"Validity: {state_illiquid.is_tradable} | Reason: {state_illiquid.reason}")

    log.info("Phase 7 Verification Complete.")

if __name__ == "__main__":
    test_phase_7()
