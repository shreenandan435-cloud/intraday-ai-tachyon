import os
import sys
import time
import json
import urllib.request
import pyotp
import numpy as np
from dotenv import load_dotenv
from SmartApi import SmartConnect

load_dotenv()

CLIENT_CODE = os.getenv("SMARTAPI_CLIENT_CODE")
PASSWORD    = os.getenv("SMARTAPI_PASSWORD")
TOTP_SECRET = os.getenv("SMARTAPI_TOTP_SECRET")
API_KEY     = os.getenv("SMARTAPI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ALERTS_ENABLED = os.getenv("TELEGRAM_ALERTS_ENABLED", "false").lower() == "true"

def dispatch_telegram(message: str):
    if TELEGRAM_ALERTS_ENABLED and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try: urllib.request.urlopen(req, timeout=5)
        except Exception: pass

def main():
    print("[*] Initializing Tachyon Relative Strength (RS) Screener...")
    
    if not os.path.exists("fno_universe.json"):
        print("[!] CRITICAL: fno_universe.json missing. Run 'python build_universe_local.py' first.")
        sys.exit(1)

    session, smart_api = None, None
    for attempt in range(1, 4):
        try:
            smart_api = SmartConnect(api_key=API_KEY)
            smart_api.timeout = 15
            totp = pyotp.TOTP(TOTP_SECRET).now()
            session = smart_api.generateSession(CLIENT_CODE, PASSWORD, totp)
            if session and session.get("status"):
                print(f"[✓] Authenticated on attempt {attempt}.")
                break
        except Exception as e:
            print(f"[!] Login attempt {attempt} failed: {e}")
            if attempt < 3: time.sleep(2)

    if not session or not session.get("status"):
        print("[!] Failed to establish session with Angel One.")
        sys.exit(1)

    with open("fno_universe.json", "r") as f:
        universe = json.load(f)

    tokens = [str(item["token"]) for item in universe]
    token_to_meta = {str(item["token"]): item for item in universe}

    print(f"[*] Querying market data for {len(tokens)} assets...")
    
    quote_data = smart_api.getMarketData(mode="FULL", exchangeTokens={"NSE": tokens})
    if not quote_data or not quote_data.get("status") or "fetched" not in quote_data.get("data", {}):
        print("[!] Failed to fetch market quotes.")
        sys.exit(1)

    fetched = quote_data["data"]["fetched"]
    valid_assets, changes = [], []

    for q in fetched:
        ltp = float(q.get("ltp", 0.0))
        close = float(q.get("close", 0.0))
        upper_circuit = float(q.get("upperCircuit", 0.0))
        
        if close <= 0 or ltp <= 0: continue

        pct_change = ((ltp - close) / close) * 100.0
        changes.append(pct_change)
        runway_pct = ((upper_circuit - ltp) / ltp) * 100.0 if upper_circuit > 0 else 10.0
        
        # FIX: Catch both key structures returned by the Angel One REST API
        sym_token = str(q.get("symbolToken", q.get("token", "")))
        meta = token_to_meta.get(sym_token)

        if meta:
            valid_assets.append({
                "symbol": meta["symbol"], 
                "token": sym_token, 
                "ltp": ltp, 
                "pct_change": pct_change, 
                "runway": runway_pct
            })

    advances = sum(1 for c in changes if c > 0)
    declines = sum(1 for c in changes if c < 0)
    
    if not changes:
        print("[!] No market changes calculated. Exiting.")
        sys.exit(1)
        
    market_median_pct = float(np.median(changes))

    print(f"\n[✓] Market Breadth: {advances} Adv / {declines} Dec")
    print(f"[*] Market Median Return: {market_median_pct:+.2f}%")

    candidates = []
    for a in valid_assets:
        if a["pct_change"] > 0.3 and a["runway"] >= 2.0:
            a["rs_score"] = a["pct_change"] - market_median_pct
            candidates.append(a)

    candidates.sort(key=lambda x: x["rs_score"], reverse=True)
    selected = candidates[:3]

    with open("resolved_tokens.json", "w") as f:
        json.dump(selected, f, indent=2)

    print(f"\n[✓] Selected {len(selected)} Relative Strength Leaders:")
    msg_lines = [
        f"🎯 *TACHYON RS BREAKOUT SELECTION*",
        f"• Breadth: `{advances} Adv / {declines} Dec`",
        f"• Baseline: `{market_median_pct:+.2f}%`"
    ]
    
    for idx, s in enumerate(selected, 1):
        print(f"  {idx}. {s['symbol']} | LTP: ₹{s['ltp']:.2f} | Chg: {s['pct_change']:+.2f}% | RS: {s['rs_score']:+.2f}%")
        msg_lines.append(f"*{idx}. {s['symbol']}*\n• Chg: `{s['pct_change']:+.2f}%` | RS: `+{s['rs_score']:.2f}%`")

    if not selected:
        print("  [!] Zero assets qualified. Capital stays in cash.")
        msg_lines.append("⚠️ *NO ASSETS QUALIFIED.*")

    dispatch_telegram("\n".join(msg_lines))
    print("[✓] resolved_tokens.json refreshed successfully.")

if __name__ == "__main__":
    main()
