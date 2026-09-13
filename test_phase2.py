import time
from feed_adapter import AngelFeedAdapter
from logger import setup_logger

def test_phase_2():
    log = setup_logger("Tachyon.Core")
    log.info("Starting Phase 2 Verification Test...")

    adapter = AngelFeedAdapter()

    # Raw packet sample from your previous Sunday test
    raw_packet_1 = {
        'subscription_mode': 3, 'exchange_type': 1, 'token': '11536', 
        'sequence_number': 29185, 
        'exchange_timestamp': int(time.time() * 1000) - 50,  # Simulate 50ms latency
        'last_traded_price': 220080, 'last_traded_quantity': 14, 
        'volume_trade_for_the_day': 2634124
    }
    
    # Packet with a sequence gap (Expected 29186, got 29188)
    raw_packet_2 = raw_packet_1.copy()
    raw_packet_2['sequence_number'] = 29188 
    raw_packet_2['exchange_timestamp'] = int(time.time() * 1000) - 20
    
    # Packet with high latency (Simulate 3000ms delay)
    raw_packet_3 = raw_packet_1.copy()
    raw_packet_3['sequence_number'] = 29189
    raw_packet_3['exchange_timestamp'] = int(time.time() * 1000) - 3000 

    log.info("Ingesting Packet 1 (Normal)...")
    tick1 = adapter.parse_snap_quote(raw_packet_1)
    if tick1:
        log.info(f"Normalized Tick: Token={tick1.token}, LTP={tick1.ltp}, CumVol={tick1.cumulative_volume}")

    log.info("Ingesting Packet 2 (Sequence Gap)...")
    tick2 = adapter.parse_snap_quote(raw_packet_2)

    log.info("Ingesting Packet 3 (High Latency)...")
    tick3 = adapter.parse_snap_quote(raw_packet_3)

    log.info("Phase 2 Verification Complete.")

if __name__ == "__main__":
    test_phase_2()
