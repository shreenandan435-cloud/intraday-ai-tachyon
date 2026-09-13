import os
import sys
import time
import json
import threading
import urllib.request
import numpy as np
import pyotp
from datetime import datetime
from dotenv import load_dotenv
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
from memory_broker import init_ledger, reset_stale_locks, log_trade_exit, check_trade_gate

load_dotenv()

CLIENT_CODE = os.getenv("SMARTAPI_CLIENT_CODE")
PASSWORD    = os.getenv("SMARTAPI_PASSWORD")
TOTP_SECRET = os.getenv("SMARTAPI_TOTP_SECRET")
API_KEY     = os.getenv("SMARTAPI_API_KEY")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ALERTS_ENABLED = os.getenv("TELEGRAM_ALERTS_ENABLED", "false").lower() == "true"

# Quantitative Risk Constants
STOP_LOSS_BPS = -20.0       # -0.20%
BREAKEVEN_TRIGGER = 15.0    # +0.15%
BREAKEVEN_LOCK = 1.0        # +0.01%
TRAIL_TRIGGER = 35.0        # +0.35% Target / Trail Activator
TRAIL_DISTANCE = 15.0       # 0.15% Trail Gap
MAX_HOLD_SECONDS = 120      # 2-minute momentum window

def dispatch_telegram(message: str):
    if TELEGRAM_ALERTS_ENABLED and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=4): pass
        except Exception: pass

class TachyonOrchestrator:
    def __init__(self):
        self.smart_api = None
        self.ws = None
        self.active_targets = {}
        self.positions = {}
        self.price_history = {}
        self.last_tbq_tsq = {}
        self.session_capital = 10000.0
        self.winning_trades = 0
        self.total_trades = 0
        self.tick_counter = 0
        self.running = True

    def authenticate(self):
        self.smart_api = SmartConnect(api_key=API_KEY)
        self.smart_api.timeout = 15
        totp = pyotp.TOTP(TOTP_SECRET).now()
        session = self.smart_api.generateSession(CLIENT_CODE, PASSWORD, totp)
        if not session or not session.get("status"):
            print("[!] Authentication failed.")
            sys.exit(1)
        auth_token = session.get("data", {}).get("jwtToken")
        feed_token = self.smart_api.getfeedToken()
        return auth_token, feed_token

    def scan_relative_strength(self):
        if not os.path.exists("fno_universe.json"):
            return []

        with open("fno_universe.json", "r") as f:
            universe = json.load(f)

        tokens = [str(item["token"]) for item in universe]
        meta_lookup = {str(item["token"]): item for item in universe}

        try:
            quotes = self.smart_api.getMarketData(mode="FULL", exchangeTokens={"NSE": tokens})
            if not quotes or not quotes.get("status") or "fetched" not in quotes.get("data", {}):
                return []
            
            fetched = quotes["data"]["fetched"]
            valid_assets, changes = [], []

            for q in fetched:
                ltp = float(q.get("ltp", 0.0))
                close = float(q.get("close", 0.0))
                upper_circuit = float(q.get("upperCircuit", 0.0))
                if close <= 0 or ltp <= 0: continue

                pct_change = ((ltp - close) / close) * 100.0
                changes.append(pct_change)
                runway = ((upper_circuit - ltp) / ltp) * 100.0 if upper_circuit > 0 else 10.0
                token_str = str(q.get("symbolToken", q.get("token", "")))
                
                meta = meta_lookup.get(token_str)
                if meta:
                    valid_assets.append({
                        "symbol": meta["symbol"], "token": token_str,
                        "ltp": ltp, "pct_change": pct_change, "runway": runway
                    })

            if not changes: return []
            market_median = float(np.median(changes))

            candidates = []
            for a in valid_assets:
                gate_ok, _ = check_trade_gate(a["symbol"])
                if a["pct_change"] > 0.3 and a["runway"] >= 2.0 and gate_ok:
                    a["rs_score"] = a["pct_change"] - market_median
                    candidates.append(a)

            candidates.sort(key=lambda x: x["rs_score"], reverse=True)
            return candidates[:3]
        except Exception as e:
            return []

    def start_heartbeat(self):
        def loop():
            while self.running:
                time.sleep(3)
                now_str = datetime.now().strftime("%H:%M:%S")
                print(f"\n[⚡ TACHYON RUNTIME {now_str}] Ingested: {self.tick_counter} | Cap: ₹{self.session_capital:.2f} | Pos: {len(self.positions)}")
                for token, meta in list(self.active_targets.items()):
                    hist = self.price_history.get(token, [])
                    ltp = hist[-1] if hist else 0.0
                    _, gate = check_trade_gate(meta["symbol"])
                    print(f"  • {meta['symbol']:<15} | LTP: ₹{ltp:>7.2f} | State: {gate}")
        threading.Thread(target=loop, daemon=True).start()

    def process_tick(self, tick):
        self.tick_counter += 1
        raw_token = str(tick.get("token", "")).strip().replace("\x00", "")
        meta = self.active_targets.get(raw_token)
        if not meta: return

        symbol = meta["symbol"]
        ltp_raw = tick.get("last_traded_price", 0)
        ltp = float(ltp_raw) / 100.0 if ltp_raw > 0 else 0.0
        if ltp <= 0: return

        if raw_token not in self.price_history: self.price_history[raw_token] = []
        p_hist = self.price_history[raw_token]
        p_hist.append(ltp)
        if len(p_hist) > 100: p_hist.pop(0)

        # 1. Active Position Management
        if symbol in self.positions:
            pos = self.positions[symbol]
            hold_time = time.time() - pos["entry_time"]
            pnl_bps = ((ltp - pos["entry"]) / pos["entry"]) * 10000
            pnl_pct = ((ltp - pos["entry"]) / pos["entry"]) * 100.0

            if pnl_bps >= TRAIL_TRIGGER:
                new_stop = pnl_bps - TRAIL_DISTANCE
                if new_stop > pos["stop_bps"]: pos["stop_bps"] = new_stop
            elif pnl_bps >= BREAKEVEN_TRIGGER and pos["stop_bps"] < BREAKEVEN_LOCK:
                pos["stop_bps"] = BREAKEVEN_LOCK

            exit_reason, readable_reason = None, None
            if pnl_bps <= pos["stop_bps"]:
                if pos["stop_bps"] > 0:
                    exit_reason = "PROFIT TARGET / TRAILING STOP"
                    readable_reason = "🎯 TARGET REACHED (Profit Locked)"
                else:
                    exit_reason = "STOP LOSS HIT"
                    readable_reason = "🛑 STOP LOSS TRIGGERED"
            elif hold_time >= MAX_HOLD_SECONDS:
                exit_reason = "ALPHA TIME EXIT"
                readable_reason = "⏱️ 2-MIN TIMEOUT (Momentum Stalled)"

            if exit_reason:
                net_pnl = (ltp - pos["entry"]) * pos["qty"]
                self.session_capital += net_pnl
                self.total_trades += 1
                if net_pnl > 0: self.winning_trades += 1
                winrate = (self.winning_trades / self.total_trades) * 100
                pnl_prefix = "+" if net_pnl >= 0 else ""
                pnl_icon = "📈" if net_pnl >= 0 else "📉"

                msg = (
                    f"🔴 *EXIT: {symbol}*\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"• Exit Price: `₹{ltp:.2f}` (Entry: `₹{pos['entry']:.2f}`)\n"
                    f"• Net PnL: `{pnl_icon} {pnl_prefix}₹{net_pnl:.2f}` (`{pnl_prefix}{pnl_pct:.2f}%`)\n"
                    f"• Outcome: `{readable_reason}`\n"
                    f"• Hold Time: `{int(hold_time)}s`\n"
                    f"• Total Capital: `₹{self.session_capital:.2f}` (`Win Rate: {winrate:.1f}%`)"
                )
                print(f"\n{msg}\n")
                dispatch_telegram(msg)
                log_trade_exit(symbol, net_pnl, exit_reason)
                del self.positions[symbol]
            return

        # 2. Gate Verification
        gate_ok, _ = check_trade_gate(symbol)
        if not gate_ok: return

        # 3. Microstructure & Order Book Depth
        buy_data = tick.get("best_5_buy_data", [])
        sell_data = tick.get("best_5_sell_data", [])
        if not buy_data or not sell_data: return

        best_bid = float(buy_data[0].get("price", 0)) / 100.0
        best_ask = float(sell_data[0].get("price", 0)) / 100.0
        if best_bid <= 0 or best_ask <= best_bid: return

        spread_bps = ((best_ask - best_bid) / best_bid) * 10000
        tbq = float(tick.get("total_buy_quantity", 0))
        tsq = float(tick.get("total_sell_quantity", 0))
        if (tbq + tsq) <= 0: return

        bid_ratio = tbq / (tbq + tsq)
        prev_tbq = self.last_tbq_tsq.get(raw_token, {}).get("tbq", tbq)
        self.last_tbq_tsq[raw_token] = {"tbq": tbq, "tsq": tsq}
        delta_tbq = tbq - prev_tbq

        # 4. Momentum Confirmation (Ticking up & at local high)
        is_ticking_up = len(p_hist) >= 2 and ltp >= p_hist[-2]
        is_at_local_high = ltp >= max(p_hist)

        if spread_bps <= 4.0 and bid_ratio >= 0.58 and delta_tbq > 0 and is_ticking_up and is_at_local_high:
            qty = max(1, int((self.session_capital / 3) / ltp))
            target_price = round(ltp * (1 + (TRAIL_TRIGGER / 10000)), 2)
            stop_price = round(ltp * (1 + (STOP_LOSS_BPS / 10000)), 2)
            entry_time_str = datetime.now().strftime("%I:%M:%S %p")

            self.positions[symbol] = {
                "entry": ltp,
                "qty": qty,
                "entry_time": time.time(),
                "stop_bps": STOP_LOSS_BPS
            }

            msg = (
                f"🟢 *BUY: {symbol}*\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"• Quantity: `{qty} shares`\n"
                f"• Entry Price: `₹{ltp:.2f}`\n"
                f"• Target (+{TRAIL_TRIGGER/100:.2f}%): `₹{target_price:.2f}`\n"
                f"• Stop Loss ({STOP_LOSS_BPS/100:.2f}%): `₹{stop_price:.2f}`\n"
                f"• Time: `{entry_time_str}`"
            )
            print(f"\n{msg}\n")
            dispatch_telegram(msg)

def run():
    print("[*] Launching Tachyon Autonomous Master Pipeline...")
    init_ledger()
    reset_stale_locks()
    
    orch = TachyonOrchestrator()
    auth_token, feed_token = orch.authenticate()
    orch.start_heartbeat()

    def scanner_worker():
        while orch.running:
            now = datetime.now().time()
            if now >= datetime.strptime("09:20", "%H:%M").time() and now < datetime.strptime("15:15", "%H:%M").time():
                leaders = orch.scan_relative_strength()
                if leaders:
                    orch.active_targets = {str(l["token"]): l for l in leaders}
                    with open("resolved_tokens.json", "w") as f:
                        json.dump(leaders, f, indent=2)
                    
                    if orch.ws:
                        sub_list = [{"exchangeType": 1, "tokens": [str(l["token"]) for l in leaders]}]
                        orch.ws.subscribe("tachyon_session", 3, sub_list)
                    print(f"[✓] Dynamic Re-Scan: Stream shifted to {[l['symbol'] for l in leaders]}")
            time.sleep(600)

    threading.Thread(target=scanner_worker, daemon=True).start()

    leaders = orch.scan_relative_strength()
    if not leaders and os.path.exists("resolved_tokens.json"):
        with open("resolved_tokens.json", "r") as f:
            leaders = json.load(f)

    orch.active_targets = {str(l["token"]): l for l in leaders}
    orch.ws = SmartWebSocketV2(auth_token, API_KEY, CLIENT_CODE, feed_token)

    def on_data(wsapp, msg): orch.process_tick(msg)
    
    # CRITICAL FIX: Absorb arbitrary arguments to bypass Angel One SDK bugs
    def on_error(*args, **kwargs): pass
    def on_close(*args, **kwargs): print("\n[*] WebSocket cleanly disconnected.")
    
    def on_open(wsapp):
        if orch.active_targets:
            sub_list = [{"exchangeType": 1, "tokens": list(orch.active_targets.keys())}]
            orch.ws.subscribe("tachyon_session", 3, sub_list)
        
        target_names = ", ".join([t['symbol'] for t in orch.active_targets.values()])
        dispatch_telegram(
            f"🚀 *TACHYON ENGINE ONLINE*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"• Monitoring: `{target_names}`\n"
            f"• Strategy: Relative Strength + Upward Momentum\n"
            f"• Engine: Autonomous Live Tracking"
        )

    orch.ws.on_data = on_data
    orch.ws.on_open = on_open
    orch.ws.on_error = on_error
    orch.ws.on_close = on_close

    try:
        orch.ws.connect()
    except KeyboardInterrupt:
        print("\n[!] User interrupted. Engine shutting down gracefully...")
        orch.running = False
        sys.exit(0)

if __name__ == "__main__":
    run()
