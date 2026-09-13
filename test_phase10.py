import time
from config import SystemConfig
from models import OrderExecution, MarketTick
from paper_execution import PaperExecutionHarness
from logger import setup_logger

def test_phase_10():
    log = setup_logger("Tachyon.Core")
    log.info("Starting Phase 10 Verification Test...")

    sys_config = SystemConfig(trading_mode="PAPER")
    
    # Fast timeouts for testing
    harness = PaperExecutionHarness(sys_config, simulated_latency_ms=150, order_timeout_sec=2)
    token = "11536"
    
    # Create Marketable Limit Order for 50 shares
    order = OrderExecution(
        internal_id="SIG_001", token=token, symbol="TCS-EQ", quantity=50, 
        limit_price=4055.0, stop_loss=4020.0, target=4125.0
    )
    
    log.info("--- SCENARIO 1: Latency & Partial to Full Fill ---")
    order_id = harness.submit_order(order)
    
    # TICK 1: Arrives instantly. Blocked by Latency simulation.
    tick1 = MarketTick(token, 0, time.time(), 1, 4052.0, 20, 1000)
    harness.process_tick_for_fills(tick1)
    
    time.sleep(0.2) # Clear network latency
    
    # TICK 2: Market prints 20 shares. Harness will capture a subset causing a partial fill.
    tick2 = MarketTick(token, 0, time.time(), 2, 4053.0, 20, 1020)
    harness.process_tick_for_fills(tick2)
    
    # TICK 3: Large trade prints. Should consume the remainder.
    tick3 = MarketTick(token, 0, time.time(), 3, 4054.0, 200, 1220)
    harness.process_tick_for_fills(tick3)
    
    log.info("--- SCENARIO 2: Order Timeout ---")
    order_stale = OrderExecution(
        internal_id="SIG_002", token=token, symbol="TCS-EQ", quantity=20, 
        limit_price=4055.0, stop_loss=4020.0, target=4125.0
    )
    harness.submit_order(order_stale)
    
    log.info("Simulating 2.5s price movement away from limit...")
    time.sleep(2.5) # Breach the 2-second timeout
    
    # TICK 4: Market returns, but the order should be flagged stale and cancelled.
    tick4 = MarketTick(token, 0, time.time(), 4, 4050.0, 100, 1220)
    harness.process_tick_for_fills(tick4)

    log.info("Phase 10 Verification Complete.")

if __name__ == "__main__":
    test_phase_10()
