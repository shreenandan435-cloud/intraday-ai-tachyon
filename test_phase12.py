import time
from config import TachyonConfig
from orchestrator import TachyonOrchestrator
from logger import setup_logger

def test_phase12():
    log = setup_logger("Tachyon")
    log.info("Starting Phase 12 Verification Test: Full Pipeline Integration...")

    config = TachyonConfig()
    orchestrator = TachyonOrchestrator(config)
    
    token = "11536"
    base_ts = 1690000000000
    
    log.info("Simulating 24 completed 5-minute candles to build consolidation zone...")
    for i in range(24):
        # 24 candles representing a tight range (100.0 - 102.0)
        # Using 5-min intervals (300 sec * 1000 ms)
        packet = {
            'token': token, 'sequence_number': i+1, 'exchange_timestamp': base_ts + (i * 300000) + 299900, 
            'last_traded_price': 10150, 'last_traded_quantity': 100, 'volume_trade_for_the_day': 10000 * (i+1),
            'best_5_buy_data': [{'price': 10145, 'quantity': 500}],
            'best_5_sell_data': [{'price': 10155, 'quantity': 500}]
        }
        orchestrator.on_market_packet(packet)
        # Force candle engine to roll over by sending the next bin's first tick
        packet['exchange_timestamp'] += 200
        orchestrator.on_market_packet(packet)

    log.info("Simulating high-volume breakout tick...")
    breakout_packet = {
        'token': token, 'sequence_number': 100, 'exchange_timestamp': base_ts + (25 * 300000) + 299900, 
        'last_traded_price': 10350, # 103.50 breaks the 102.00 resistance
        'last_traded_quantity': 500, 'volume_trade_for_the_day': 300000, # Massive volume expansion
        'best_5_buy_data': [{'price': 10345, 'quantity': 1500}],
        'best_5_sell_data': [{'price': 10355, 'quantity': 2000}]
    }
    
    orchestrator.on_market_packet(breakout_packet)
    
    # Trigger final rollover
    breakout_packet['exchange_timestamp'] += 200
    orchestrator.on_market_packet(breakout_packet)

    log.info("Phase 12 Verification Complete.")

if __name__ == "__main__":
    test_phase12()
