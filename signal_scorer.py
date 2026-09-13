import logging
from models import Candle, SignalScore
from structure_engine import ConsolidationZone
from orderbook_engine import OrderBookState

class OpportunityScorer:
    def __init__(self):
        self.logger = logging.getLogger("Tachyon.Scorer")

    def generate_score(self, token: str, zone: ConsolidationZone, candle: Candle, 
                       rs_spread: float, ob_state: OrderBookState) -> SignalScore:
        """
        Synthesizes setup parameters into a normalized 0-100 opportunity score.
        """
        score = 0.0
        details = []

        # 1. Structure Quality (Max 30 pts)
        # Tighter width = higher score (Range: 3.5 to 0.5 ATRs)
        struct_pts = max(0, min(30, 30 * (3.5 - zone.width_atr) / (3.5 - 0.5)))
        score += struct_pts
        details.append(f"Struct:{struct_pts:.1f}")

        # 2. Breakout Power (Max 30 pts)
        # RVOL (Max 20 pts, Range: 1.8x to 4.0x)
        rvol_pts = max(0, min(20, 20 * (candle.rvol - 1.8) / (4.0 - 1.8)))
        # Tick Ratio (Max 10 pts, Range: 1.3x to 3.0x)
        tick_pts = max(0, min(10, 10 * (candle.tick_ratio - 1.3) / (3.0 - 1.3)))
        score += (rvol_pts + tick_pts)
        details.append(f"Power:{(rvol_pts + tick_pts):.1f}")

        # 3. Relative Strength (Max 20 pts)
        # Range: 0.0% to 2.0% outperformance vs benchmark
        rs_pts = max(0, min(20, 20 * (rs_spread - 0.0) / (2.0 - 0.0)))
        score += rs_pts
        details.append(f"RS:{rs_pts:.1f}")

        # 4. Execution Quality (Max 20 pts)
        # Tighter spread = higher score (Range: 0.10% down to 0.02%)
        exec_pts = max(0, min(20, 20 * (0.10 - ob_state.spread_pct) / (0.10 - 0.02)))
        score += exec_pts
        details.append(f"Exec:{exec_pts:.1f}")

        return SignalScore(
            token=token,
            total_score=round(score, 1),
            passed_conditions=details,
            is_valid=True
        )

    def rank_opportunities(self, scores: list[SignalScore]) -> list[SignalScore]:
        """Sorts a list of SignalScores descending by total score."""
        return sorted(scores, key=lambda x: x.total_score, reverse=True)
