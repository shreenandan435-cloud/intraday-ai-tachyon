import time
import logging
import threading
import queue
from enum import Enum
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
from models import MarketTick

class ConnectionState(Enum):
    DISCONNECTED = 1
    CONNECTING = 2
    CONNECTED = 3
    SUBSCRIBING = 4
    LIVE = 5
    RECONNECTING = 6
    HALTED = 7
    STOPPED = 8

class AngelFeedAdapter:
    def __init__(self, api_key: str, client_code: str, auth_token: str, feed_token: str, tokens: list):
        self.logger = logging.getLogger("Tachyon.Feed")
        self.state = ConnectionState.DISCONNECTED
        self.tokens = tokens
        
        self.ws = SmartWebSocketV2(auth_token, api_key, client_code, feed_token)
        self.ws.on_data = self._on_data
        self.ws.on_open = self._on_open
        self.ws.on_error = self._on_error
        self.ws.on_close = self._on_close
        
        # Thread-safe queue prevents websocket blocking from quant engine lag
        self.packet_queue = queue.Queue()
        self.last_packet_time = 0
        self.reconnect_count = 0
        self.ws_thread = None
        
    def start(self):
        self.state = ConnectionState.CONNECTING
        self.logger.info(f"Initializing SmartWebSocketV2 connection for {len(self.tokens)} tokens...")
        self.ws_thread = threading.Thread(target=self.ws.connect, daemon=True)
        self.ws_thread.start()

    def _on_open(self, ws):
        self.state = ConnectionState.CONNECTED
        self.logger.info("WebSocket Connected. Initiating Subscriptions...")
        self.state = ConnectionState.SUBSCRIBING
        
        correlation_id = "tachyon_shadow_sub"
        action = 1  # 1 = Subscribe
        mode = 3    # 3 = SNAP_QUOTE (L1 + L2 Depth)
        token_list = [{"exchangeType": 1, "tokens": self.tokens}]
        
        self.ws.subscribe(correlation_id, mode, token_list)
        self.state = ConnectionState.LIVE
        self.logger.info(f"Subscription confirmed. Feed State: LIVE")

    def _on_data(self, ws, message):
        if self.state != ConnectionState.LIVE:
            return
        self.last_packet_time = time.perf_counter()
        self.packet_queue.put((self.last_packet_time, message))

    def _on_error(self, ws, error):
        self.logger.error(f"WebSocket Feed Error: {error}")
        self.state = ConnectionState.HALTED

    def _on_close(self, ws, code, reason):
        self.logger.warning(f"WebSocket Closed: Code {code} - Reason: {reason}")
        self.state = ConnectionState.DISCONNECTED
        self._handle_reconnect()

    def _handle_reconnect(self):
        self.reconnect_count += 1
        self.state = ConnectionState.RECONNECTING
        self.logger.warning(f"Connection lost. Attempting Feed Reconnect #{self.reconnect_count}...")
        # Clear stale packets from queue to prevent processing old data after reconnect
        with self.packet_queue.mutex:
            self.packet_queue.queue.clear()
        time.sleep(2)  # Backoff
        self.start()

    def parse_snap_quote(self, raw_packet: dict):
        try:
            return MarketTick(
                token=str(raw_packet.get('token', '')),
                timestamp=int(raw_packet.get('exchange_timestamp', 0)),
                ltp=float(raw_packet.get('last_traded_price', 0.0)),
                volume=int(raw_packet.get('volume_trade_for_the_day', 0))
            )
        except Exception:
            return None
