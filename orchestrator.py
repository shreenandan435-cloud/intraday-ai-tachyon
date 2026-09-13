import logging
from config import TachyonConfig
from models import MarketTick, OrderExecution, OrderState
from state_machine import TachyonStateMachine, SystemState
from feed_adapter import AngelFeedAdapter
from candle_engine import CandleEngine
from market_context import MarketContextEngine
from structure_engine import StructureEngine
from breakout_engine import BreakoutEngine
from orderbook_engine import OrderbookAnalyzer
from signal_scorer import OpportunityScorer
from risk_manager import RiskManager
from paper_execution import PaperExecutionHarness

class TachyonOrchestrator:
    def __init__(self, config: TachyonConfig):
        self.config = config
        self.logger = logging.getLogger("Tachyon.Master")
        self.state_machine = TachyonStateMachine()
        
        self.adapter = AngelFeedAdapter()
        self.candle_engine = CandleEngine(
            interval_minutes=config.strategy.candle_minutes,
            atr_period=config.strategy.atr_period,
            vol_lookback=config.strategy.volume_lookback
        )
        self.market_context = MarketContextEngine()
        self.structure_engine = StructureEngine()
        self.breakout_engine = BreakoutEngine(
            atr_buffer=config.strategy.breakout_atr_buffer,
            min_rvol=config.strategy.rvol_threshold,
            min_tick_ratio=config.strategy.tick_ratio_threshold
        )
        self.orderbook = OrderbookAnalyzer()
        self.scorer = OpportunityScorer()
        self.risk_manager = RiskManager(config.risk)
        self.executor = PaperExecutionHarness(config.system)
        
        self.historical_candles = {}
        self.active_exposure = {} 
        
        self.state_machine.transition_to(SystemState.SCANNING, "Engines Online")

    def on_market_packet(self, raw_packet: dict):
        if self.state_machine.get_state() not in [SystemState.SCANNING, SystemState.POSITION_OPEN]:
            return

        tick = self.adapter.parse_snap_quote(raw_packet)
        if not tick:
            return

        token = tick.token
        self.executor.process_tick_for_fills(tick)

        if token in self.active_exposure:
            tracked_order = self.active_exposure[token]
            if tracked_order.state in [OrderState.SUBMITTED, OrderState.ACKNOWLEDGED, 
                                       OrderState.PARTIALLY_FILLED, OrderState.FILLED]:
                return 
            elif tracked_order.state in [OrderState.CANCELLED, OrderState.REJECTED]:
                del self.active_exposure[token]

        new_candle = self.candle_engine.process_tick(tick)
        if not new_candle:
            return

        if token not in self.historical_candles:
            self.historical_candles[token] = []
            
        history = self.historical_candles[token]
        history.append(new_candle)

        if len(history) < self.structure_engine.lookback:
            return

        # --- FIX 3: X-RAY LOGGING GATES ---
        zone = self.structure_engine.evaluate_consolidation(history[:-1])
        if not zone.is_valid:
            self.logger.warning(f"[{token}] Rejected: Structure Zone Invalid (Failed consolidation clustering).")
            return

        breakout = self.breakout_engine.evaluate_breakout(new_candle, zone)
        if not breakout.is_valid:
            # We use hasattr to safely log properties since we don't know exactly which attribute triggered the fail
            res = getattr(zone, 'resistance', 'N/A')
            self.logger.warning(f"[{token}] Rejected: Breakout Engine Failed. (Close: {new_candle.close}, Res: {res}, RVOL: {new_candle.rvol})")
            return

        ob_state = self.orderbook.evaluate_liquidity(raw_packet, tick.ltp)
        if not ob_state.is_tradable:
            self.logger.warning(f"[{token}] Rejected: Orderbook liquidity compromised ({ob_state.reason}).")
            return

        try:
            rs_spread = self.market_context.get_relative_strength(token)
        except AttributeError:
            rs_spread = 1.5 
            
        score = self.scorer.generate_score(token, zone, new_candle, rs_spread, ob_state)
        
        if score.total_score >= 50.0:
            is_valid, qty, limit_px, target_px, reason = self.risk_manager.calculate_sizing_and_levels(
                ltp=tick.ltp, stop_loss=breakout.structural_stop, atr=new_candle.atr
            )
            
            if is_valid:
                order = OrderExecution(
                    internal_id=f"SIG_{token}_{new_candle.timestamp}",
                    token=token,
                    symbol=self.config.strategy.active_tokens.get(token, token),
                    quantity=qty,
                    limit_price=limit_px,
                    stop_loss=breakout.structural_stop,
                    target=target_px
                )
                self.logger.info(f"[{token}] Breakout Signal Confirmed (Score: {score.total_score}/100). Submitting Limit Order.")
                self.active_exposure[token] = order 
                self.executor.submit_order(order)
            else:
                self.logger.warning(f"[{token}] Risk Rejection: {reason}")
        else:
            self.logger.warning(f"[{token}] Signal Scored {score.total_score}/100. Rejected (Must be >= 50).")
