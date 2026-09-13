from config import RiskConfig
from risk_manager import RiskManager
from logger import setup_logger

def test_phase_9():
    log = setup_logger("Tachyon.Core")
    log.info("Starting Phase 9 Verification Test...")

    # Initialize with default ₹1000 risk, max 2 positions, -₹3000 daily loss limit
    config = RiskConfig()
    risk_engine = RiskManager(config)

    # --- SCENARIO 1: Expensive Stock (e.g., TCS @ ₹4050) ---
    ltp_tcs = 4050.0
    sl_tcs = 4020.0
    atr_tcs = 35.0
    
    is_valid, qty1, limit1, target1, reason1 = risk_engine.calculate_sizing_and_levels(ltp_tcs, sl_tcs, atr_tcs)
    
    log.info("--- SCENARIO 1: Sizing an Expensive Stock ---")
    log.info(f"LTP: ₹{ltp_tcs} | Stop: ₹{sl_tcs}")
    log.info(f"Quantity: {qty1} shares | Limit: ₹{limit1} | Target: ₹{target1}")
    total_risk1 = qty1 * (limit1 - sl_tcs)
    log.info(f"Total Projected Risk: ₹{total_risk1:.2f} (Budget: ₹{config.risk_per_trade_rupees})")


    # --- SCENARIO 2: Cheaper Stock (e.g., SBIN @ ₹800) ---
    ltp_sbin = 800.0
    sl_sbin = 792.0
    atr_sbin = 12.0
    
    is_valid, qty2, limit2, target2, reason2 = risk_engine.calculate_sizing_and_levels(ltp_sbin, sl_sbin, atr_sbin)

    log.info("--- SCENARIO 2: Sizing a Cheaper Stock ---")
    log.info(f"LTP: ₹{ltp_sbin} | Stop: ₹{sl_sbin}")
    log.info(f"Quantity: {qty2} shares | Limit: ₹{limit2} | Target: ₹{target2}")
    total_risk2 = qty2 * (limit2 - sl_sbin)
    log.info(f"Total Projected Risk: ₹{total_risk2:.2f} (Budget: ₹{config.risk_per_trade_rupees})")


    # --- SCENARIO 3: Portfolio Governor (Loss Limit Hit) ---
    log.info("--- SCENARIO 3: Portfolio Governor Check ---")
    is_locked, lock_reason = risk_engine.check_portfolio_locks(open_positions_count=1, realized_pnl=-3100.0)
    
    log.info(f"Is Engine Locked? {is_locked}")
    if is_locked:
        log.info(f"Lock Reason: {lock_reason}")

    log.info("Phase 9 Verification Complete.")

if __name__ == "__main__":
    test_phase_9()
