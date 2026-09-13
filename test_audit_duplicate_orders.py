import time
from config import TachyonConfig
from orchestrator import TachyonOrchestrator
from logger import setup_logger

def test_infinite_order_loop():
    log = setup_logger("Tachyon.Audit")
    log.warning("AUDIT: Testing for Duplicate Order Submission Defect...")
    
    orchestrator = TachyonOrchestrator(TachyonConfig())
    token = "11536"
    base_ts = int(time.time() * 1000)
    
    # 1. Build Zone (24 candles)
    for i in range(24):
        orchestrator.on_market_packet({
            'token': token, 'sequence_number': i, 'exchange_timestamp': base_ts + (i * 300000), 
            'last_traded_price': 10150, 'last_traded_quantity': 1000, 'volume_trade_for_the_day': 1000*(i+1),
            'best_5_buy_data': [{'price': 10145, 'quantity': 500}],
            'best_5_sell_data': [{'price': 10155, 'quantity': 500}]
        })
    
    # 2. Trigger Breakout (Tick 1)
    log.warning("Injecting Breakout Tick 1...")
    orchestrator.on_market_packet({
        'token': token, 'sequence_number': 100, 'exchange_timestamp': base_ts + (24 * 300000) + 1000, 
        'last_traded_price': 10190, 'last_traded_quantity': 6000, 'volume_trade_for_the_day': 50000,
        'best_5_buy_data': [{'price': 10185, 'quantity': 1500}],
        'best_5_sell_data': [{'price': 10195, 'quantity': 2000}]
    })
    
    # 3. Next tick arrives 1 second later (Price holds above breakout)
    log.warning("Injecting Breakout Tick 2 (Same Candle)...")
    orchestrator.on_market_packet({
        'token': token, 'sequence_number': 101, 'exchange_timestamp': base_ts + (24 * 300000) + 2000, 
        'last_traded_price': 10195, 'last_traded_quantity': 1000, 'volume_trade_for_the_day': 51000,
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
    test_infinite_order_loop()
