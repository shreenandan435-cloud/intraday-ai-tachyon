import os
import time
import signal
import sys
from datetime import datetime
from dotenv import load_dotenv
import pyotp
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

from stream_accumulator import CandleAccumulator
from geometry_engine import GeometricBreakoutEngine
from order_router import SafeOrderRouter
from telegram_notifier import TelegramNotifier

load_dotenv()

# Match exact .env variable names
API_KEY = os.getenv("SMARTAPI_API_KEY")
CLIENT_CODE = os.getenv("SMARTAPI_CLIENT_CODE")
PASSWORD = os.getenv("SMARTAPI_PASSWORD")
TOTP_SECRET = os.getenv("SMARTAPI_TOTP_SECRET")
TRADING_MODE = os.getenv("TRADING_MODE", "PAPER").upper()

# Target liquid F&O equities for morning momentum tracking
TRACKED_TOKENS = {
    "11536": "TCS-EQ",
    "1594": "INFY-EQ",
    "1333": "HDFCBANK-EQ",
    "3045": "SBIN-EQ",
    "11532": "RELIANCE-EQ"
}

notifier = TelegramNotifier()
accumulator = CandleAccumulator(candle_minutes=5)
geometry = GeometricBreakoutEngine(rvol_threshold=1.8, tick_ratio_threshold=1.3)

active_positions = {}  # token: {entry_price, sl, target, qty, order_id}

def handle_shutdown(sig, frame):
    print("\n[!] Emergency shutdown triggered. Sending alert...")
    notifier.send("⚠️ *EMERGENCY SHUTDOWN:* Trading engine stopped manually.")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

def wait_until_market_prep(target_hour=9, target_minute=14, target_second=0):
    """
    Blocks execution with a live countdown timer until target_hour:target_minute:target_second.
    Allows running the script early in the morning before heading to college.
    """
    now = datetime.now()
    target_time = now.replace(hour=target_hour, minute=target_minute, second=target_second, microsecond=0)
    
    if now < target_time:
        total_wait_sec = int((target_time - now).total_seconds())
        print(f"[+] Early boot detected at {now.strftime('%H:%M:%S')}.")
        print(f"[+] Engine entering standby until {target_time.strftime('%H:%M:%S')} IST.\n")
        
        while total_wait_sec > 0:
            hrs, rem = divmod(total_wait_sec, 3600)
            mins, secs = divmod(rem, 60)
            sys.stdout.write(f"\r⏳ Standby Countdown: {hrs:02d}h {mins:02d}m {secs:02d}s until market initialization... ")
            sys.stdout.flush()
            time.sleep(1)
            total_wait_sec -= 1
            
        print("\n\n[✓] 09:14:00 AM reached. Initializing trading stack...\n")
    else:
        print(f"[+] Current time {now.strftime('%H:%M:%S')} is past 09:14:00 AM. Skipping standby.\n")

def initialize_angel_session():
    print("[+] Authenticating with Angel One SmartAPI...")
    smart_api = SmartConnect(api_key=API_KEY)
    totp = pyotp.TOTP(TOTP_SECRET.strip()).now()
    session = smart_api.generateSession(CLIENT_CODE, PASSWORD, totp)
    
    if not session.get('status'):
        raise ConnectionError(f"SmartAPI login failed: {session.get('message')}")
        
    auth_token = session['data']['jwtToken']
    feed_token = smart_api.getfeedToken()
    print("[✓] SmartAPI session authenticated successfully.")
    return smart_api, auth_token, feed_token

def main():
    # 1. Market Open Countdown Guard
    wait_until_market_prep(target_hour=9, target_minute=14, target_second=0)

    # 2. Authentication & Setup
    smart_api, auth_token, feed_token = initialize_angel_session()
    is_paper = (TRADING_MODE != "LIVE")
    router = SafeOrderRouter(smart_api_client=smart_api, risk_budget_rupees=1000.0, paper_trading=is_paper)
    
    mode_label = "VIRTUAL / PAPER" if is_paper else "LIVE CAPITAL"
    notifier.send(
        f"🟢 *TACHYON ENGINE ONLINE*\n"
        f"• *Mode:* `{mode_label}`\n"
        f"• *Strategy:* Geometric Breakout\n"
        f"• *Tokens Loaded:* `{len(TRACKED_TOKENS)} symbols`\n"
        f"• *Risk Budget:* `₹1000/trade`"
    )

    sws = SmartWebSocketV2(auth_token, API_KEY, CLIENT_CODE, feed_token)

    def on_data(wsapp, message):
        token = str(message.get('token'))
        ltp = float(message.get('last_traded_price', 0)) / 100.0
        cum_vol = int(message.get('volume_trade_for_the_day', 0))

        if token not in TRACKED_TOKENS or ltp <= 0:
            return

        now = datetime.now()
        
        # 1. Real-time Risk Exit Monitoring
        if token in active_positions:
            pos = active_positions[token]
            # Stop-Loss Hit
            if ltp <= pos["sl"]:
                pnl = (ltp - pos["entry_price"]) * pos["qty"]
                notifier.notify_exit(TRACKED_TOKENS[token], ltp, pnl, "Stop-Loss Hit")
                del active_positions[token]
            # Target Hit (1:2 R:R)
            elif ltp >= pos["target"]:
                pnl = (ltp - pos["entry_price"]) * pos["qty"]
                notifier.notify_exit(TRACKED_TOKENS[token], ltp, pnl, "1:2 Target Hit")
                del active_positions[token]

        # 2. Ingest tick into Candle Accumulator
        closed_bar = accumulator.on_tick(token, ltp, cum_vol)
        
        # 3. Process Breakout Evaluation upon 5-min candle completion
        if closed_bar:
            df = accumulator.get_dataframe(token)
            
            # Entry Window Filter (09:25 to 11:45 AM only; skip 12:00-1:30 PM lull)
            if not ((now.hour == 9 and now.minute >= 25) or (now.hour in (10, 11) and now.minute <= 45)):
                return

            if token not in active_positions and len(active_positions) < 2:
                is_valid, setup = geometry.evaluate_breakout(df)
                if is_valid:
                    order = router.place_marketable_limit_buy(
                        symbol=TRACKED_TOKENS[token],
                        token=token,
                        ltp=setup["entry_price"],
                        atr=setup["atr"],
                        stop_loss=setup["stop_loss"]
                    )
                    if order:
                        active_positions[token] = {
                            "entry_price": setup["entry_price"],
                            "sl": setup["stop_loss"],
                            "target": order["target"],
                            "qty": order["quantity"],
                            "order_id": order["order_id"]
                        }
                        notifier.notify_breakout_entry(
                            symbol=TRACKED_TOKENS[token],
                            entry_price=setup["entry_price"],
                            sl=setup["stop_loss"],
                            target=order["target"],
                            qty=order["quantity"],
                            rvol=setup["rvol"]
                        )

        # 4. Auto Square-Off at 03:15 PM IST
        if now.hour == 15 and now.minute >= 15:
            if active_positions:
                for t, p in list(active_positions.items()):
                    pnl = (ltp - p["entry_price"]) * p["qty"]
                    notifier.notify_exit(TRACKED_TOKENS[t], ltp, pnl, "3:15 PM Auto Square-Off")
                active_positions.clear()
            notifier.send("🛑 *MARKET CLOSE:* All positions squared off. Shutting down.")
            sws.close_connection()
            sys.exit(0)

    def on_open(wsapp):
        print(f"[✓] Connected to SmartAPI Mode-3 WebSocket ({mode_label}). Subscribing to tokens...")
        token_list = [{"exchangeType": 1, "tokens": list(TRACKED_TOKENS.keys())}]
        sws.subscribe("smart_stream", 3, token_list)

    def on_error(wsapp, error):
        print(f"[!] WebSocket Error: {error}")

    sws.on_open = on_open
    sws.on_data = on_data
    sws.on_error = on_error

    print(f"[+] Starting WebSocket stream listener in {mode_label} mode...")
    sws.connect()

if __name__ == "__main__":
    main()
