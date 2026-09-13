from models import Candle
from structure_engine import ConsolidationZone
from orderbook_engine import OrderBookState
from signal_scorer import OpportunityScorer
from logger import setup_logger

def test_phase_8():
    log = setup_logger("Tachyon.Core")
    log.info("Starting Phase 8 Verification Test...")

    scorer = OpportunityScorer()

    # --- SETUP A: Average / Weak Breakout (Passes minimums, but barely) ---
    zone_a = ConsolidationZone(True, 100, 95, 90, 2.5) # Wide 2.5 ATR base
    candle_a = Candle("STOCK_A", 0, 0, 0, 0, 101, 1000, 100, 100, 2.0, rvol=1.9, tick_ratio=1.4)
    rs_a = 0.20 # Outperforming by only 0.2%
    ob_a = OrderBookState(0, 0, 0, 0.08, 0, 0, True, "") # 0.08% spread (wide but legal)
    
    score_a = scorer.generate_score("STOCK_A (Average)", zone_a, candle_a, rs_a, ob_a)

    # --- SETUP B: Exceptional / A+ Breakout ---
    zone_b = ConsolidationZone(True, 100, 99, 98, 1.0) # Tight 1.0 ATR base
    candle_b = Candle("STOCK_B", 0, 0, 0, 0, 101, 1000, 100, 100, 1.0, rvol=3.8, tick_ratio=2.9)
    rs_b = 1.50 # Outperforming by massive 1.5%
    ob_b = OrderBookState(0, 0, 0, 0.03, 0, 0, True, "") # Crisp 0.03% spread
    
    score_b = scorer.generate_score("STOCK_B (Exceptional)", zone_b, candle_b, rs_b, ob_b)

    # --- RANKING ---
    ranked = scorer.rank_opportunities([score_a, score_b])
    
    log.info("--- OPPORTUNITY RANKING BOARD ---")
    for rank, s in enumerate(ranked, 1):
        log.info(f"#{rank} -> {s.token} | Score: {s.total_score}/100")
        log.info(f"       Breakdown: {', '.join(s.passed_conditions)}")

    log.info("Phase 8 Verification Complete.")

if __name__ == "__main__":
    test_phase_8()
