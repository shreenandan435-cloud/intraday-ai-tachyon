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
    # Sync to current real-world time to pass the Data Quality Monitor
    base_ts = int(time.time() * 1000) - (25 * 300000)
    
    log.info("Simulating 24 completed 5-minute candles to build consolidation zone...")
    cum_vol = 0
    for i in range(24):
        # Inject 3 ticks per candle to establish a healthy baseline tick_ratio
        for tick_idx in range(3):
            cum_vol += 1000
            packet = {
                'token': token, 'sequence_number': (i*3) + tick_idx + 1, 
                'exchange_timestamp': base_ts + (i * 300000) + (tick_idx * 50000), 
                'last_traded_price': 10150 if tick_idx % 2 == 0 else 10000, # Create a 1.5 ATR
                'last_traded_quantity': 1000, 'volume_trade_for_the_day': cum_vol,
                'best_5_buy_data': [{'price': 10145, 'quantity': 500}],
                'best_5_sell_data': [{'price': 10155, 'quantity': 500}]
            }
            orchestrator.on_market_packet(packet)
            
        # Force candle engine to roll over by sending the next bin's first tick
        rollover_packet = packet.copy()
        rollover_packet['exchange_timestamp'] = base_ts + ((i+1) * 300000) + 100
        orchestrator.on_market_packet(rollover_packet)

    log.info("Simulating high-volume breakout tick...")
    # Inject 8 ticks rapidly to trigger >2.0x tick_ratio and >2.0x RVOL
    for tick_idx in range(8):
        cum_vol += 5000
        breakout_packet = {
            'token': token, 'sequence_number': 100 + tick_idx, 
            'exchange_timestamp': base_ts + (24 * 300000) + (tick_idx * 1000), 
            'last_traded_price': 10350, # 103.50 breaks the resistance
            'last_traded_quantity': 5000, 'volume_trade_for_the_day': cum_vol,
            'best_5_buy_data': [{'price': 10345, 'quantity': 1500}],
            'best_5_sell_data': [{'price': 10355, 'quantity': 2000}]
        }
        orchestrator.on_market_packet(breakout_packet)
    
    # Trigger final rollover to complete the breakout candle
    breakout_packet['exchange_timestamp'] += 300000
    orchestrator.on_market_packet(breakout_packet)

    log.info("Phase 12 Verification Complete.")

if __name__ == "__main__":
    test_phase12()
