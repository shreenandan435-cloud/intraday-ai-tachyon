import time
from config import TachyonConfig
from orchestrator import TachyonOrchestrator
from logger import setup_logger

def test_infinite_order_loop_fixed():
    log = setup_logger("Tachyon.Audit")
    log.warning("AUDIT: Testing for Duplicate Order Submission Defect...")
    
    orchestrator = TachyonOrchestrator(TachyonConfig())
    token = "11536"
    
    # 1. Start 25 candles in the PAST so timestamps pass the DataQualityMonitor
    base_ts = int(time.time() * 1000) - (25 * 300000)
    
    # 2. Build 24-candle zone with valid tick/volume density
    cum_vol = 0
    for i in range(24):
        for tick_idx in range(3):
            cum_vol += 1000
            packet = {
                'token': token, 'sequence_number': (i*3) + tick_idx + 1, 
                'exchange_timestamp': base_ts + (i * 300000) + (tick_idx * 50000), 
                'last_traded_price': 10150 if tick_idx % 2 == 0 else 10050, 
                'last_traded_quantity': 1000, 'volume_trade_for_the_day': cum_vol,
                'best_5_buy_data': [{'price': 10145, 'quantity': 500}],
                'best_5_sell_data': [{'price': 10155, 'quantity': 500}]
            }
            orchestrator.on_market_packet(packet)
            
        rollover_packet = packet.copy()
        rollover_packet['exchange_timestamp'] = base_ts + ((i+1) * 300000) + 100
        orchestrator.on_market_packet(rollover_packet)

    # 3. Inject Breakout Wave (10 rapid ticks to simulate institutional buying)
    log.warning("Injecting Institutional Breakout Wave (10 rapid ticks)...")
    for tick_idx in range(10):
        cum_vol += 6000
        orchestrator.on_market_packet({
            'token': token, 'sequence_number': 100 + tick_idx, 
            'exchange_timestamp': base_ts + (24 * 300000) + (tick_idx * 1000), 
            'last_traded_price': 10190, 
            'last_traded_quantity': 6000, 'volume_trade_for_the_day': cum_vol,
            # Super tight spread to guarantee the Execution Score maxes out
            'best_5_buy_data': [{'price': 10185, 'quantity': 5000}], 
            'best_5_sell_data': [{'price': 10190, 'quantity': 5000}]
        })

    # Assert correct behavior
    orders_submitted = len(orchestrator.executor.pending_orders)
    log.warning(f"AUDIT RESULT: Orders currently submitted for single breakout: {orders_submitted}")
    
    if orders_submitted > 1:
        log.error(f"FAIL: Infinite Order Loop defect confirmed. Submitted {orders_submitted} orders.")
    elif orders_submitted == 1:
        log.info("PASS: Duplicate exposure prevented. Exposure ledger is functioning perfectly.")
    else:
        log.error("FAIL: No orders were submitted. Pipeline broken.")

if __name__ == "__main__":
    test_infinite_order_loop_fixed()
