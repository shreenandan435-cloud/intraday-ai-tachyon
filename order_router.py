import math
import logging
import time

class SafeOrderRouter:
    def __init__(self, smart_api_client, risk_budget_rupees=1000.0, max_slippage_atr_ratio=0.05, paper_trading=True):
        self.client = smart_api_client
        self.risk_budget = risk_budget_rupees
        self.max_slippage_ratio = max_slippage_atr_ratio
        self.paper_trading = paper_trading

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0
        shares = math.floor(self.risk_budget / risk_per_share)
        return max(1, shares)

    def place_marketable_limit_buy(self, symbol: str, token: str, ltp: float, atr: float, stop_loss: float):
        qty = self.calculate_position_size(ltp, stop_loss)
        if qty <= 0:
            logging.error(f"[!] Invalid position size computed for {symbol}")
            return None

        slippage_buffer = max(0.05, self.max_slippage_ratio * atr)
        limit_price = round(round((ltp + slippage_buffer) / 0.05) * 0.05, 2)
        target_price = round(ltp + (2.0 * abs(ltp - stop_loss)), 2)

        # PAPER TRADING SIMULATION
        if self.paper_trading:
            simulated_id = f"PAPER_{int(time.time())}"
            logging.info(f"[PAPER TRADE] Virtual Limit BUY: {symbol} | Qty: {qty} | Price: {limit_price} | SL: {stop_loss}")
            return {
                "order_id": simulated_id,
                "quantity": qty,
                "limit_price": limit_price,
                "stop_loss": stop_loss,
                "target": target_price,
                "is_paper": True
            }

        # LIVE EXECUTION ON ANGEL ONE OMS
        order_params = {
            "variety": "NORMAL",
            "tradingsymbol": symbol,
            "symboltoken": token,
            "transactiontype": "BUY",
            "exchange": "NSE",
            "ordertype": "LIMIT",
            "producttype": "INTRADAY",
            "duration": "DAY",
            "price": str(limit_price),
            "quantity": str(qty)
        }
        
        try:
            order_id = self.client.placeOrder(order_params)
            logging.info(f"[✓] Live Order placed: {symbol} | ID: {order_id}")
            return {
                "order_id": order_id,
                "quantity": qty,
                "limit_price": limit_price,
                "stop_loss": stop_loss,
                "target": target_price,
                "is_paper": False
            }
        except Exception as e:
            logging.error(f"[!] Order placement error: {str(e)}")
            return None
