import time
from config import TachyonConfig
from orchestrator import TachyonOrchestrator
from logger import setup_logger

def test_phase12_final():
    log = setup_logger("Tachyon")
    log.info("Starting Phase 12 Final Verification: Executing Paper Order...")

    config = TachyonConfig()
    orchestrator = TachyonOrchestrator(config)
    
    token = "11536"
    # Sync timestamps cleanly
    base_ts = int(time.time() * 1000) - (25 * 300000)
    
    log.info("Building 24-candle consolidation base (Resistance capped near 101.50)...")
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

    log.info("Injecting clean institutional breakout (Price: 101.90, High RVOL)...")
    for tick_idx in range(8):
        cum_vol += 6000
        breakout_packet = {
            'token': token, 'sequence_number': 100 + tick_idx, 
            'exchange_timestamp': base_ts + (24 * 300000) + (tick_idx * 1000), 
            'last_traded_price': 10190, # Clean breakout above 101.50 resistance without overextending
            'last_traded_quantity': 6000, 'volume_trade_for_the_day': cum_vol,
            'best_5_buy_data': [{'price': 10185, 'quantity': 1500}],
            'best_5_sell_data': [{'price': 10195, 'quantity': 2000}]
        }
        orchestrator.on_market_packet(breakout_packet)
    
    # Trigger final rollover
    breakout_packet['exchange_timestamp'] += 300000
    orchestrator.on_market_packet(breakout_packet)

    log.info("Phase 12 Pipeline Complete.")

if __name__ == "__main__":
    test_phase12_final()
