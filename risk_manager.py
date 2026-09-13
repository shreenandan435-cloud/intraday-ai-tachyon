import math
import logging
from typing import Tuple, Dict
from config import RiskConfig

class RiskManager:
    def __init__(self, config: RiskConfig):
        self.config = config
        self.logger = logging.getLogger("Tachyon.Risk")

    def calculate_sizing_and_levels(self, ltp: float, stop_loss: float, atr: float) -> Tuple[bool, int, float, float, str]:
        """
        Calculates exact share sizing based on structural risk.
        Returns: (is_valid, quantity, limit_price, target_price, reason)
        """
        if ltp <= stop_loss:
            return False, 0, 0.0, 0.0, "STOP_LOSS_ABOVE_LTP"

        # 1. Calculate Limit Entry Price (Allowable Slippage)
        # Limit price = LTP + max(5 paise, 5% of ATR)
        slippage_buffer = max(0.05, self.config.slippage_atr_ratio * atr)
        limit_price = round(ltp + slippage_buffer, 2)

        # 2. Calculate Risk Per Share (Using worst-case Limit Price)
        risk_per_share = limit_price - stop_loss
        if risk_per_share <= 0:
            return False, 0, 0.0, 0.0, "INVALID_RISK_PER_SHARE"

        # 3. Calculate Position Size (Floor to prevent exceeding budget)
        quantity = math.floor(self.config.risk_per_trade_rupees / risk_per_share)
        
        if quantity < 1:
            return False, 0, 0.0, 0.0, "RISK_BUDGET_TOO_SMALL_FOR_STOCK"

        # 4. Calculate Target (Default 1:2 R:R based on actual fill risk)
        target_price = round(limit_price + (risk_per_share * self.config.target_risk_reward_ratio), 2)

        return True, quantity, limit_price, target_price, "SIZING_OK"

    def check_portfolio_locks(self, open_positions_count: int, realized_pnl: float) -> Tuple[bool, str]:
        """
        Validates system-wide hard limits. Returns (is_locked, reason).
        """
        if open_positions_count >= self.config.max_open_positions:
            return True, f"MAX_POSITIONS_REACHED ({open_positions_count})"

        if realized_pnl <= -self.config.max_daily_loss_rupees:
            return True, f"DAILY_LOSS_LIMIT_BREACHED (₹{realized_pnl:.2f})"

        return False, "PORTFOLIO_CLEAR"
