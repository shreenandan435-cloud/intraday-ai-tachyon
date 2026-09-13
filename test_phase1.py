import time
from config import TachyonConfig
from models import MarketTick, OrderState, OrderExecution, SignalScore
from state_machine import TachyonStateMachine, SystemState
from logger import setup_logger

def test_phase_1():
    # 1. Initialize Logger
    log = setup_logger("Tachyon.Core")
    log.info("Starting Phase 1 Verification Test...")

    # 2. Test Configuration
    config = TachyonConfig()
    log.info(f"Config Loaded - Trading Mode: {config.system.trading_mode}, Risk Budget: ₹{config.risk.risk_per_trade_rupees}")
    
    # 3. Test Models
    tick = MarketTick(
        token="11536",
        exchange_timestamp=1690000000,
        local_receive_timestamp=time.time(),
        sequence_number=1,
        ltp=2200.80,
        last_traded_quantity=14,
        cumulative_volume=10000
    )
    log.info(f"Model Verification - Tick Created: {tick.token} @ {tick.ltp}")

    order = OrderExecution(
        internal_id="ORD_001",
        token="11536",
        symbol="TCS-EQ",
        quantity=50,
        limit_price=2205.00,
        stop_loss=2190.00,
        target=2235.00
    )
    log.info(f"Model Verification - Order State: {order.state.name}")

    # 4. Test State Machine
    sm = TachyonStateMachine()
    sm.transition_to(SystemState.PRE_MARKET, reason="Schedule reached")
    sm.transition_to(SystemState.AUTHENTICATING, reason="TOTP Generated")
    sm.transition_to(SystemState.SCANNING, reason="WebSocket Connected")

    if sm.get_state() == SystemState.SCANNING:
        log.info("Phase 1 Verification Complete. All systems nominal.")

if __name__ == "__main__":
    test_phase_1()
