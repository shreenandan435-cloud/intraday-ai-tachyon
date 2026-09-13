import os
import json
import urllib.request
import pyotp
from dotenv import load_dotenv
from SmartApi import SmartConnect

load_dotenv()

CLIENT_CODE = os.getenv("SMARTAPI_CLIENT_CODE")
PASSWORD    = os.getenv("SMARTAPI_PASSWORD")
TOTP_SECRET = os.getenv("SMARTAPI_TOTP_SECRET")
API_KEY     = os.getenv("SMARTAPI_API_KEY")

TELEGRAM_BOT_TOKEN      = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID        = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ALERTS_ENABLED = os.getenv("TELEGRAM_ALERTS_ENABLED", "false").lower() == "true"
TRADING_MODE            = os.getenv("TRADING_MODE", "PAPER").upper()

def send_telegram_ping(message: str):
    if not TELEGRAM_ALERTS_ENABLED or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[!] Telegram alerts disabled or credentials missing.")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = json.dumps({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }).encode("utf-8")
    
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                print("[✓] Telegram operator alert verified & sent.")
    except Exception as e:
        print(f"[X] Telegram dispatch error: {e}")

def verify_handshake():
    print("=" * 60)
    print(f"[*] INITIATING TACHYON PRE-FLIGHT HANDSHAKE [MODE: {TRADING_MODE}]")
    print("=" * 60)
    
    if not all([CLIENT_CODE, PASSWORD, TOTP_SECRET, API_KEY]):
        print("[X] Missing critical SmartAPI credentials in .env!")
        return

    try:
        totp = pyotp.TOTP(TOTP_SECRET).now()
        smart_api = SmartConnect(api_key=API_KEY)
        session_data = smart_api.generateSession(CLIENT_CODE, PASSWORD, totp)
        
        if not session_data.get("status"):
            print(f"[X] Authentication failed: {session_data.get('message')}")
            return

        jwt_token = session_data["data"]["jwtToken"]
        feed_token = smart_api.getfeedToken()
        
        print("[✓] Angel One SmartAPI Authentication: SUCCESS")
        print(f"    - Client Code : {CLIENT_CODE}")
        print(f"    - JWT Token   : {jwt_token[:16]}... [ACTIVE]")
        print(f"    - Feed Token  : {feed_token[:16]}... [ACTIVE]")
        
        msg = (
            f"⚡ *TACHYON PRE-FLIGHT VERIFIED*\n"
            f"• Mode: `{TRADING_MODE}`\n"
            f"• Client: `{CLIENT_CODE}`\n"
            f"• SmartAPI & Feed Tokens: `ONLINE`\n"
            f"• Model: `tachyon_brain_generalized.zip`\n"
            f"• Stop Daemon: `-20 bps`"
        )
        send_telegram_ping(msg)

    except Exception as e:
        print(f"[X] Fatal handshake exception: {e}")

if __name__ == "__main__":
    verify_handshake()
