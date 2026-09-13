import logging
from typing import List
from models import Candle

class ConsolidationZone:
    def __init__(self, is_valid: bool, resistance: float, support: float, reason: str = "", width_atr: float = 0.0):
        self.is_valid = is_valid
        self.resistance = resistance
        self.resistance_top = resistance  
        self.support = support
        self.support_bottom = support     
        self.support_base = support       
        self.reason = reason
        self.width_atr = width_atr        # Fix 7: Expose width in ATR units for OpportunityScorer

class StructureEngine:
    def __init__(self, lookback: int = 24, max_width_atr: float = 3.0):
        self.lookback = lookback
        self.max_width_atr = max_width_atr
        self.logger = logging.getLogger("Tachyon.Structure")

    def evaluate_consolidation(self, history: List[Candle]) -> ConsolidationZone:
        if len(history) < self.lookback:
            return ConsolidationZone(False, 0.0, 0.0, "INSUFFICIENT_DATA")

        window = history[-self.lookback:]
        
        highs = [getattr(c, 'high', c.close) for c in window]
        lows = [getattr(c, 'low', c.close) for c in window]
        atrs = [c.atr for c in window if c.atr > 0]

        if not atrs:
            return ConsolidationZone(False, 0.0, 0.0, "ZERO_ATR")

        avg_atr = sum(atrs) / len(atrs)
        resistance_line = max(highs)
        support_line = min(lows)
        zone_width = resistance_line - support_line
        
        # Calculate width in terms of ATR for the signal scorer
        width_atr = zone_width / avg_atr if avg_atr > 0 else 0.0

        max_allowed_width = self.max_width_atr * avg_atr
        
        if zone_width > max_allowed_width:
             return ConsolidationZone(False, resistance_line, support_line, f"TOO_WIDE (Width:{zone_width:.2f} > Max:{max_allowed_width:.2f})", width_atr)

        return ConsolidationZone(True, resistance_line, support_line, "VALID_BASE", width_atr)
