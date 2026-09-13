import logging
from dataclasses import dataclass
from models import Candle
from structure_engine import ConsolidationZone

@dataclass
class BreakoutResult:
    is_valid: bool
    reason: str
    entry_price: float = 0.0
    structural_stop: float = 0.0
    extension_atr: float = 0.0

class BreakoutEngine:
    def __init__(self, atr_buffer: float = 0.15, min_rvol: float = 1.8, 
                 min_tick_ratio: float = 1.3, max_extension_atr: float = 1.0):
        self.atr_buffer = atr_buffer
        self.min_rvol = min_rvol
        self.min_tick_ratio = min_tick_ratio
        self.max_extension_atr = max_extension_atr
        self.logger = logging.getLogger("Tachyon.Breakout")

    def evaluate_breakout(self, candle: Candle, zone: ConsolidationZone) -> BreakoutResult:
        """
        Deterministically evaluates if the current candle legitimately breaks the structural zone.
        """
        if not zone.is_valid:
            return BreakoutResult(False, "INVALID_ZONE")

        atr = candle.atr if candle.atr > 0 else 1.0
        clearance_threshold = zone.resistance_top + (self.atr_buffer * atr)

        # 1. Price Acceptance Gate
        if candle.close <= zone.resistance_top:
            return BreakoutResult(False, "NO_BREAKOUT")
            
        if candle.close < clearance_threshold:
            return BreakoutResult(False, "WEAK_CLOSE_OR_WICK_SWEEP")

        # 2. VWAP Confluence Gate
        if candle.close <= candle.vwap:
            return BreakoutResult(False, "BELOW_VWAP_RESISTANCE")

        # 3. Institutional Participation Gates
        if candle.rvol < self.min_rvol:
            return BreakoutResult(False, f"LOW_VOLUME_FAKEOUT ({candle.rvol}x)")
            
        if candle.tick_ratio < self.min_tick_ratio:
            return BreakoutResult(False, f"LOW_ACTIVITY ({candle.tick_ratio}x)")

        # 4. Don't Chase Gate (Check extension from resistance)
        extension = (candle.close - zone.resistance_top) / atr
        if extension > self.max_extension_atr:
            return BreakoutResult(False, f"OVEREXTENDED_CHASE ({extension:.2f} ATRs)", extension_atr=extension)

        # Passes all gates
        return BreakoutResult(
            is_valid=True,
            reason="CONFIRMED_BREAKOUT",
            entry_price=candle.close,
            structural_stop=zone.support_base,
            extension_atr=extension
        )
