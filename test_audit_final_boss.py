import time
from config import TachyonConfig
from orchestrator import TachyonOrchestrator
from logger import setup_logger

def test_audit_final_boss():
    log = setup_logger("Tachyon")
    log.warning("AUDIT: Final Boss - The Perfect Breakout...")
    
    orchestrator = TachyonOrchestrator(TachyonConfig())
    token = "11536"
    
    base_ts = int(time.time() * 1000) - (25 * 300000)
    
    cum_vol = 0
    # 1. Build the 24-Candle Consolidation Zone
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

    # 2. Inject an EXPANDING Bullish Breakout (Walking price from 101.60 to 102.50)
    log.warning("Injecting EXPANDING Institutional Breakout Wave...")
    for tick_idx in range(10):
        cum_vol += 6000
        current_price = 10160 + (tick_idx * 10)  # Price walks up 10 paisa per tick
        orchestrator.on_market_packet({
            'token': token, 'sequence_number': 100 + tick_idx, 
            'exchange_timestamp': base_ts + (24 * 300000) + (tick_idx * 1000), 
            'last_traded_price': current_price, 
            'last_traded_quantity': 6000, 'volume_trade_for_the_day': cum_vol,
            'best_5_buy_data': [{'price': current_price - 5, 'quantity': 5000}], 
            'best_5_sell_data': [{'price': current_price + 5, 'quantity': 5000}]
        })

    # 3. Close the breakout candle to trigger the evaluation
    log.warning("Pushing clock forward to close the breakout candle...")
    orchestrator.on_market_packet({
        'token': token, 'sequence_number': 120, 
        'exchange_timestamp': base_ts + (25 * 300000) + 100, 
        'last_traded_price': 10250, 
        'last_traded_quantity': 100, 'volume_trade_for_the_day': cum_vol + 100,
        'best_5_buy_data': [{'price': 10245, 'quantity': 5000}], 
        'best_5_sell_data': [{'price': 10255, 'quantity': 5000}]
    })
    
    # 4. Inject a DUPLICATE tick in the new candle to test the Exposure Ledger Block
    log.warning("Injecting duplicate tick to test exposure ledger block...")
    orchestrator.on_market_packet({
        'token': token, 'sequence_number': 121, 
        'exchange_timestamp': base_ts + (25 * 300000) + 200, 
        'last_traded_price': 10260, 
        'last_traded_quantity': 100, 'volume_trade_for_the_day': cum_vol + 200,
        'best_5_buy_data': [{'price': 10255, 'quantity': 5000}], 
        'best_5_sell_data': [{'price': 10265, 'quantity': 5000}]
    })

    # 5. Final Assertion
    orders_submitted = len(orchestrator.executor.pending_orders)
    log.warning(f"AUDIT RESULT: Orders currently submitted for single breakout: {orders_submitted}")
    
    if orders_submitted > 1:
        log.error(f"FAIL: Infinite Order Loop defect confirmed. Submitted {orders_submitted} orders.")
    elif orders_submitted == 1:
        log.info("PASS: Duplicate exposure prevented. Exposure ledger is functioning perfectly.")
    else:
        log.error("FAIL: No orders were submitted. Pipeline broken.")

if __name__ == "__main__":
    test_audit_final_boss()
