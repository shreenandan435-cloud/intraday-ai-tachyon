import time
from config import TachyonConfig
from orchestrator import TachyonOrchestrator
from logger import setup_logger

def test_infinite_order_loop_fixed():
    log = setup_logger("Tachyon.Audit")
    log.warning("AUDIT: Testing for Duplicate Order Submission Defect...")
    
    orchestrator = TachyonOrchestrator(TachyonConfig())
    token = "11536"
    
    # 1. Start 25 candles in the PAST so timestamps are valid
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

    # 3. Inject A+ Breakout (Tick 1) - This MUST trigger an order
    log.warning("Injecting Breakout Tick 1 (Should Trigger Order)...")
    cum_vol += 6000
    orchestrator.on_market_packet({
        'token': token, 'sequence_number': 100, 
        'exchange_timestamp': base_ts + (24 * 300000) + 1000, 
        'last_traded_price': 10190, 
        'last_traded_quantity': 6000, 'volume_trade_for_the_day': cum_vol,
        'best_5_buy_data': [{'price': 10185, 'quantity': 1500}],
        'best_5_sell_data': [{'price': 10195, 'quantity': 2000}]
    })
    
    # 4. Inject Duplicate Breakout (Tick 2) - This MUST BE BLOCKED by Fix 1
    log.warning("Injecting Breakout Tick 2 (Must be blocked)...")
    cum_vol += 6000
    orchestrator.on_market_packet({
        'token': token, 'sequence_number': 101, 
        'exchange_timestamp': base_ts + (24 * 300000) + 2000, 
        'last_traded_price': 10195, 
        'last_traded_quantity': 6000, 'volume_trade_for_the_day': cum_vol,
        'best_5_buy_data': [{'price': 10185, 'quantity': 1500}],
        'best_5_sell_data': [{'price': 10195, 'quantity': 2000}]
    })

    # Assert correct behavior
    orders_submitted = len(orchestrator.executor.pending_orders)
    log.warning(f"AUDIT RESULT: Orders currently submitted for single breakout: {orders_submitted}")
    
    if orders_submitted > 1:
        log.error("FAIL: Infinite Order Loop defect confirmed.")
    elif orders_submitted == 1:
        log.info("PASS: Duplicate exposure prevented. Exposure ledger is functioning.")
    else:
        log.error("FAIL: No orders were submitted. Pipeline broken.")

if __name__ == "__main__":
    test_infinite_order_loop_fixed()
