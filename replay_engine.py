import logging
from typing import Generator
from models import MarketTick
from candle_engine import CandleEngine
from paper_execution import PaperExecutionHarness

class ReplayEngine:
    def __init__(self, candle_engine: CandleEngine, execution_harness: PaperExecutionHarness):
        """
        The Replay Engine uses the exact same production engines as the live system,
        preventing 'simulation bias' or look-ahead errors.
        """
        self.candle_engine = candle_engine
        self.execution_harness = execution_harness
        self.logger = logging.getLogger("Tachyon.Replay")

    def run_replay(self, tick_stream: Generator[MarketTick, None, None]):
        """Pumps historical ticks through the system at maximum computational speed."""
        self.logger.info("Starting historical replay stream...")
        ticks_processed = 0
        candles_formed = 0
        
        for tick in tick_stream:
            # 1. Process Executions (Simulate limit order fills on historical data)
            self.execution_harness.process_tick_for_fills(tick)
            
            # 2. Build Candles
            candle = self.candle_engine.process_tick(tick)
            
            # 3. Strategy Hook
            if candle:
                candles_formed += 1
                # In Phase 12, this hook will trigger the StructureEngine and BreakoutEngine
                self.logger.debug(f"Replay formed historical candle @ {candle.timestamp}")

            ticks_processed += 1
            
        self.logger.info(f"Replay Complete. Processed {ticks_processed} ticks, formed {candles_formed} candles.")
