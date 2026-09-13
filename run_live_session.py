import os
import sys
import time
import json
import threading
import urllib.request
import numpy as np
import pyotp
from dotenv import load_dotenv
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
from memory_broker import init_ledger, reset_stale_locks, log_trade_exit, check_trade_gate

load_dotenv()

# Institutional Risk Parameters
STOP_LOSS_BPS = -20.0
BREAKEVEN_TRIGGER = 15.0
BREAKEVEN_LOCK = 1.0
TRAIL_TRIGGER = 35.0
TRAIL_DISTANCE = 15.0
MAX_HOLD_SECONDS = 120

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ALERTS_ENABLED = os.getenv("TELEGRAM_ALERTS_ENABLED", "false").lower() == "true"

def dispatch_telegram(message: str):
    if TELEGRAM_ALERTS_ENABLED and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=4):
                pass
        except Exception as e:
            print(f"[!] Telegram Alert Dispatch Failed: {e}")

class TachyonLiveEngine:
    def __init__(self, tokens):
        self.tokens = tokens
        self.token_lookup = {str(t["token"]).strip(): t for t in tokens}
        self.positions = {}
        
        # Microstructure tracking
        self.l2_metrics = {str(t["token"]).strip(): {"ltp": 0.0, "spread_bps": 0.0, "bid_ratio": 0.5, "gate": "READY"} for t in tokens}
        self.last_tbq_tsq = {str(t["token"]).strip(): {"tbq": 0, "tsq": 0} for t in tokens}
        
        # Telemetry
        self.tick_counter = 0
        self.session_capital = 10000.0
        self.total_trades = 0
        self.winning_trades = 0
        self.running = True

        # Start live CLI heartbeat monitor
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

    def _heartbeat_loop(self):
        while self.running:
            time.sleep(3)
            current_time = time.strftime("%H:%M:%S")
            print(f"\n[⚡ TACHYON TELEMETRY {current_time}] Ingested: {self.tick_counter} ticks | Capital: ₹{self.session_capital:.2f} | Open Pos: {len(self.positions)}")
            for token, meta in self.token_lookup.items():
                m = self.l2_metrics[token]
                _, gate_status = check_trade_gate(meta["symbol"])
                m["gate"] = gate_status
                print(f"  • {meta['symbol']:<15} | LTP: ₹{m['ltp']:>7.2f} | Spread: {m['spread_bps']:>4.1f} bps | Book: {m['bid_ratio']*100:>4.1f}% Bids | Gate: {m['gate']}")

    def process_tick(self, tick):
        self.tick_counter += 1
        
        # Clean null-byte padding from binary C-structures
        raw_token = str(tick.get("token", "")).strip().replace("\x00", "")
        symbol_meta = self.token_lookup.get(raw_token)
        if not symbol_meta:
            return

        symbol = symbol_meta["symbol"]
        
        # Extract LTP (convert paise to rupees)
        ltp_raw = tick.get("last_traded_price", 0)
        ltp = float(ltp_raw) / 100.0 if ltp_raw > 0 else 0.0
        if ltp <= 0:
            return

        self.l2_metrics[raw_token]["ltp"] = ltp

        # 1. Active Position Management (Risk Daemon)
        if symbol in self.positions:
            pos = self.positions[symbol]
            entry_price = pos["entry"]
            hold_time = time.time() - pos["entry_time"]
            pnl_bps = ((ltp - entry_price) / entry_price) * 10000

            # Dynamic Trailing Logic
            if pnl_bps >= TRAIL_TRIGGER:
                new_stop = pnl_bps - TRAIL_DISTANCE
                if new_stop > pos["stop_bps"]:
                    pos["stop_bps"] = new_stop
            elif pnl_bps >= BREAKEVEN_TRIGGER and pos["stop_bps"] < BREAKEVEN_LOCK:
                pos["stop_bps"] = BREAKEVEN_LOCK

            exit_reason = None
            if pnl_bps <= pos["stop_bps"]:
                exit_reason = "STOP LOSS / TRAIL"
            elif hold_time >= MAX_HOLD_SECONDS:
                exit_reason = "TACHYON ALPHA EXIT"

            if exit_reason:
                net_pnl = (ltp - entry_price) * pos["qty"]
                self.session_capital += net_pnl
                self.total_trades += 1
                if net_pnl > 0:
                    self.winning_trades += 1

                win_rate = (self.winning_trades / self.total_trades) * 100.0
                print(f"\n[🔴] {exit_reason}: {symbol} | Exit: ₹{ltp:.2f} | PnL: ₹{net_pnl:.2f} | Hold: {int(hold_time)}s")

                msg = (f"🎯 *TACHYON EXIT LONG PAPER*\n"
                       f"• Symbol: `{symbol}` ({pos['qty']} sh)\n"
                       f"• Exit: `₹{ltp:.2f}` (Entry: `₹{entry_price:.2f}`)\n"
                       f"• Net PnL: `{'+' if net_pnl>0 else ''}₹{net_pnl:.2f}` (Hold: {int(hold_time)}s)\n"
                       f"• Reason: `{exit_reason}`\n"
                       f"• Capital: `₹{self.session_capital:.2f}` | WinRate: `{win_rate:.1f}%`")
                dispatch_telegram(msg)

                log_trade_exit(symbol, net_pnl, exit_reason)
                del self.positions[symbol]
            return

        # 2. Extract Accurate L2 Market Depth
        buy_data = tick.get("best_5_buy_data", [])
        sell_data = tick.get("best_5_sell_data", [])
        
        if not buy_data or not sell_data:
            return

        best_bid = float(buy_data[0].get("price", 0)) / 100.0
        best_ask = float(sell_data[0].get("price", 0)) / 100.0

        if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
            return

        spread_bps = ((best_ask - best_bid) / best_bid) * 10000
        self.l2_metrics[raw_token]["spread_bps"] = spread_bps

        # Aggregate liquidity ratio
        tbq = float(tick.get("total_buy_quantity", 0))
        tsq = float(tick.get("total_sell_quantity", 0))
        if (tbq + tsq) > 0:
            bid_ratio = tbq / (tbq + tsq)
            self.l2_metrics[raw_token]["bid_ratio"] = bid_ratio
        else:
            bid_ratio = 0.5

        # 3. Memory Gate Check
        gate_ok, _ = check_trade_gate(symbol)
        if not gate_ok:
            return

        # 4. Institutional Entry Trigger
        # Condition: Tight spread (< 4.5 bps), buyer dominance (> 58% book), and positive delta absorption
        prev_data = self.last_tbq_tsq[raw_token]
        delta_tbq = tbq - prev_data["tbq"]
        self.last_tbq_tsq[raw_token] = {"tbq": tbq, "tsq": tsq}

        if spread_bps <= 4.5 and bid_ratio >= 0.58 and delta_tbq > 0:
            qty = max(1, int((self.session_capital / 3) / ltp))
            self.positions[symbol] = {
                "entry": ltp,
                "qty": qty,
                "entry_time": time.time(),
                "stop_bps": STOP_LOSS_BPS
            }

            print(f"\n[🟢] TACHYON ENTRY LONG: {symbol} | Entry: ₹{ltp:.2f} | Spread: {spread_bps:.1f} bps | Bids: {bid_ratio*100:.1f}%")

            msg = (f"🟢 *TACHYON ENTRY LONG PAPER*\n"
                   f"• Symbol: `{symbol}` ({qty} shares)\n"
                   f"• Entry: `₹{ltp:.2f}`\n"
                   f"• Book Pressure: `{bid_ratio*100:.1f}% Bids`\n"
                   f"• Spread: `{spread_bps:.1f} bps`")
            dispatch_telegram(msg)

def main():
    print("[*] Initializing Tachyon Institutional Engine...")
    init_ledger()
    reset_stale_locks()

    try:
        with open("resolved_tokens.json", "r") as f:
            tokens = json.load(f)
    except FileNotFoundError:
        tokens = []

    if not tokens:
        print("[!] resolved_tokens.json missing or empty. Please run screener.py first.")
        sys.exit(1)

    print(f"[✓] Active Targets Loaded ({len(tokens)}): {[t['symbol'] for t in tokens]}")
    print("[*] Authenticating with Angel One SmartAPI...")

    CLIENT_CODE = os.getenv("SMARTAPI_CLIENT_CODE")
    PASSWORD    = os.getenv("SMARTAPI_PASSWORD")
    TOTP_SECRET = os.getenv("SMARTAPI_TOTP_SECRET")
    API_KEY     = os.getenv("SMARTAPI_API_KEY")

    smart_api = SmartConnect(api_key=API_KEY)
    totp = pyotp.TOTP(TOTP_SECRET).now()
    session = smart_api.generateSession(CLIENT_CODE, PASSWORD, totp)

    if not session or not session.get("status"):
        print("[!] Session authentication failed.")
        sys.exit(1)

    auth_token = session.get("data", {}).get("jwtToken")
    feed_token = smart_api.getfeedToken()
    print("[✓] Session authenticated. Connecting to live WebSocket stream...")

    engine = TachyonLiveEngine(tokens)
    ws = SmartWebSocketV2(auth_token, API_KEY, CLIENT_CODE, feed_token)

    def on_data(wsapp, message):
        try:
            engine.process_tick(message)
        except Exception as e:
            print(f"[!] Tick Error: {e}")

    def on_open(wsapp):
        print("[✓] WebSocket V2 connected. Subscribing to Mode-3 Snap Quote telemetry...")
        token_list = [{"exchangeType": 1, "tokens": [str(t["token"]).strip() for t in tokens]}]
        ws.subscribe("tachyon_session", 3, token_list)
        dispatch_telegram(f"🚀 *TACHYON TRADING ENGINE ONLINE*\n• Targets: `{', '.join([t['symbol'] for t in tokens])}`\n• Pipeline: Mode 3 L2 Depth Active")

    def on_error(wsapp, error):
        print(f"\n[!] WebSocket Error: {error}")

    def on_close(wsapp):
        print("\n[*] WebSocket stream terminated.")
        engine.running = False

    ws.on_data = on_data
    ws.on_open = on_open
    ws.on_error = on_error
    ws.on_close = on_close

    try:
        ws.connect()
    except KeyboardInterrupt:
        print("\n[!] Engine shutting down gracefully...")
        engine.running = False

if __name__ == "__main__":
    main()
