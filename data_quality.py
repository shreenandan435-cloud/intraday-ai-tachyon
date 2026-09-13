import time
import logging
from typing import Dict, Tuple
from models import MarketTick

class DataQualityMonitor:
    def __init__(self, max_latency_ms: int = 1500):
        self.max_latency_ms = max_latency_ms
        self.last_sequence: Dict[str, int] = {}
        self.logger = logging.getLogger("Tachyon.DataQuality")

    def evaluate_tick(self, tick: MarketTick) -> Tuple[bool, str]:
        """
        Validates tick freshness and sequence integrity.
        Returns (is_valid: bool, reason_code: str)
        """
        # 1. Latency Check (Angel One exchange_timestamp is in epoch milliseconds)
        exchange_sec = tick.exchange_timestamp / 1000.0
        latency_ms = (tick.local_receive_timestamp - exchange_sec) * 1000.0
        
        if latency_ms > self.max_latency_ms:
            # Note: During testing/weekends, exchange timestamps might be stale.
            self.logger.warning(f"[DATA_DEGRADED] High Latency on {tick.token}: {latency_ms:.2f}ms")
            return False, "HIGH_LATENCY"
        
        # Negative latency usually means local clock is slightly behind exchange clock
        if latency_ms < -1000: 
            self.logger.warning(f"[CLOCK_DESYNC] Local clock behind exchange on {tick.token}: {latency_ms:.2f}ms")

        # 2. Sequence Gap Check
        expected_seq = self.last_sequence.get(tick.token, tick.sequence_number - 1) + 1
        
        if tick.sequence_number < expected_seq:
            self.logger.warning(f"[STALE_PACKET] Out-of-order tick on {tick.token}. Expected >= {expected_seq}, got {tick.sequence_number}")
            return False, "OUT_OF_ORDER"
            
        if tick.sequence_number > expected_seq:
            gap = tick.sequence_number - expected_seq
            self.logger.debug(f"[GAP_DETECTED] Missed {gap} packets on {tick.token}. Expected {expected_seq}, got {tick.sequence_number}")

        # Update sequence state
        self.last_sequence[tick.token] = tick.sequence_number
        return True, "OK"
