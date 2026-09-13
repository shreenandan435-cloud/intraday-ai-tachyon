import time
import random
import logging
from typing import Dict
from models import OrderExecution, OrderState, MarketTick
from config import SystemConfig

class PaperExecutionHarness:
    def __init__(self, system_config: SystemConfig, simulated_latency_ms: int = 150, order_timeout_sec: int = 15):
        if system_config.trading_mode == "LIVE":
            raise PermissionError("Paper harness initialized while configuration is set to LIVE mode!")
            
        self.simulated_latency_ms = simulated_latency_ms
        self.order_timeout_sec = order_timeout_sec
        self.pending_orders: Dict[str, OrderExecution] = {}
        self.order_submission_time: Dict[str, float] = {}
        self.logger = logging.getLogger("Tachyon.PaperExec")

    def submit_order(self, order: OrderExecution) -> str:
        """Simulates sending an order. Requires explicit opt-in via config."""
        order.broker_order_id = f"PAPER_{int(time.time()*1000)}"
        order.state = OrderState.SUBMITTED
        self.pending_orders[order.broker_order_id] = order
        self.order_submission_time[order.broker_order_id] = time.time()
        self.logger.info(f"[{order.symbol}] Paper Order Submitted: {order.quantity} @ Limit ₹{order.limit_price}")
        return order.broker_order_id

    def process_tick_for_fills(self, tick: MarketTick):
        """Evaluates pending paper orders against incoming market data."""
        current_time = time.time()
        
        for order_id, order in list(self.pending_orders.items()):
            if order.token != tick.token:
                continue

            sub_time = self.order_submission_time[order_id]

            # 1. Simulate Latency (Blocks instant fills)
            if (current_time - sub_time) * 1000 < self.simulated_latency_ms:
                if order.state == OrderState.SUBMITTED:
                    order.state = OrderState.ACKNOWLEDGED
                    self.logger.debug(f"[{order.symbol}] Order Acknowledged (Latency simulation)")
                continue

            # 2. Simulate Order Timeout (Prevent stale fills)
            if (current_time - sub_time) > self.order_timeout_sec:
                order.state = OrderState.CANCELLED
                self.logger.warning(f"[{order.symbol}] Paper Order CANCELLED (Timeout exceeded {self.order_timeout_sec}s)")
                del self.pending_orders[order_id]
                del self.order_submission_time[order_id]
                continue

            # 3. Fill Evaluation (Market trades at or below limit)
            if tick.ltp <= order.limit_price:
                # Simulate realistic partial fills based on actual tick liquidity
                available = max(1, int(tick.last_traded_quantity * random.uniform(0.2, 1.0)))
                fill_qty = min(order.quantity - order.filled_quantity, available)

                # Simulate slippage penalty (never exceeding limit price)
                max_slip = min(order.limit_price - tick.ltp, order.limit_price * 0.0005)
                slippage = random.uniform(0.0, max_slip)
                fill_price = round(tick.ltp + slippage, 2)

                old_filled = order.filled_quantity
                order.filled_quantity += fill_qty

                # Weighted average execution price
                if order.filled_quantity > 0:
                    total_value = (order.average_fill_price * old_filled) + (fill_price * fill_qty)
                    order.average_fill_price = round(total_value / order.filled_quantity, 2)

                if order.filled_quantity >= order.quantity:
                    order.state = OrderState.FILLED
                    self.logger.info(f"[{order.symbol}] Paper FULL FILL | Qty: {order.quantity} | Avg Price: ₹{order.average_fill_price}")
                    del self.pending_orders[order_id]
                    del self.order_submission_time[order_id]
                else:
                    order.state = OrderState.PARTIALLY_FILLED
                    self.logger.info(f"[{order.symbol}] Paper PARTIAL FILL | {fill_qty} shares @ ₹{fill_price} | Progress: {order.filled_quantity}/{order.quantity}")
