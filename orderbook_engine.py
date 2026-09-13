import logging

class OrderbookState:
    def __init__(self, is_tradable: bool, reason: str, spread: float = 0.0, bid_qty: int = 0, ask_qty: int = 0):
        self.is_tradable = is_tradable
        self.reason = reason
        self.spread = spread
        self.bid_qty = bid_qty
        self.ask_qty = ask_qty

class OrderbookAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger("Tachyon.Orderbook")

    def evaluate_liquidity(self, raw_packet: dict, ltp: float) -> OrderbookState:
        try:
            bids = raw_packet.get('best_5_buy_data', [])
            asks = raw_packet.get('best_5_sell_data', [])

            if not isinstance(bids, list) or not isinstance(asks, list):
                return OrderbookState(False, "INVALID_BOOK (Not lists)")

            if not bids or not asks:
                return OrderbookState(False, "INSUFFICIENT_DEPTH (Empty arrays)")

            # FIX: Explicitly check for None instead of relying on Python truthiness,
            # otherwise a quantity of `0` is treated as a missing key and dropped.
            parsed_bids = [b for b in bids if isinstance(b, dict) and b.get('price') is not None and b.get('quantity') is not None]
            parsed_asks = [a for a in asks if isinstance(a, dict) and a.get('price') is not None and a.get('quantity') is not None]

            if not parsed_bids or not parsed_asks:
                return OrderbookState(False, "INSUFFICIENT_DEPTH (No valid populated levels)")

            best_bid = float(parsed_bids[0]['price'])
            best_ask = float(parsed_asks[0]['price'])

            if best_bid >= best_ask:
                return OrderbookState(False, "INVALID_BOOK (Crossed book detected)")

            total_bid_qty = sum(int(b['quantity']) for b in parsed_bids)
            total_ask_qty = sum(int(a['quantity']) for a in parsed_asks)

            if total_bid_qty == 0 or total_ask_qty == 0:
                return OrderbookState(False, "INSUFFICIENT_DEPTH (Zero total quantity side)")

            spread = best_ask - best_bid
            
            # Max allowable spread check (e.g. 0.2% of LTP to prevent illusionary fills)
            if spread / ltp > 0.002:
                 return OrderbookState(False, f"SPREAD_TOO_WIDE ({spread:.2f})")

            return OrderbookState(True, "VALID_BOOK", spread, total_bid_qty, total_ask_qty)

        except Exception as e:
            self.logger.error(f"L2 Parser caught malformed payload: {e}")
            return OrderbookState(False, "MALFORMED_PACKET")
