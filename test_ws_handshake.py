import os
import sys
import time
import threading
from dotenv import load_dotenv
import pyotp
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

load_dotenv()

API_KEY = os.getenv("SMARTAPI_API_KEY")
CLIENT_CODE = os.getenv("SMARTAPI_CLIENT_CODE")
PASSWORD = os.getenv("SMARTAPI_PASSWORD")
TOTP_SECRET = os.getenv("SMARTAPI_TOTP_SECRET")

print("[+] Authenticating with Angel One...")
smart_api = SmartConnect(api_key=API_KEY)
totp = pyotp.TOTP(TOTP_SECRET.strip()).now()
session = smart_api.generateSession(CLIENT_CODE, PASSWORD, totp)

if not session.get("status"):
    print(f"[!] Authentication failed: {session.get('message')}")
    sys.exit(1)

auth_token = session["data"]["jwtToken"]
feed_token = smart_api.getfeedToken()
print("[✓] Session generated. Initializing SmartWebSocketV2...")

sws = SmartWebSocketV2(auth_token, API_KEY, CLIENT_CODE, feed_token)
handshake_confirmed = False

def auto_close_timer():
    time.sleep(8)
    print("\n[+] 8-second handshake window completed.")
    if handshake_confirmed:
        print("[SUCCESS] Mode-3 WebSocket handshake and subscription verified successfully.")
    else:
        print("[!] Socket did not trigger on_open within timeout.")
    try:
        sws.close_connection()
    except Exception:
        pass

def on_open(wsapp):
    global handshake_confirmed
    handshake_confirmed = True
    print("[✓] WebSocket connection established (on_open triggered)!")
    print("[+] Subscribing to TCS-EQ (Token: 11536) in Mode 3 (Snap Quote)...")
    token_list = [{"exchangeType": 1, "tokens": ["11536"]}]
    sws.subscribe("smart_stream", 3, token_list)
    print("[✓] Subscription packet successfully dispatched to Angel One.")

def on_data(wsapp, message):
    print(f"[+] Received socket stream frame: {message}")

def on_error(wsapp, error):
    print(f"[!] WebSocket Error: {error}")

def on_close(wsapp):
    print("[✓] WebSocket closed cleanly.")

sws.on_open = on_open
sws.on_data = on_data
sws.on_error = on_error
sws.on_close = on_close

timer_thread = threading.Thread(target=auto_close_timer, daemon=True)
timer_thread.start()

print("[+] Connecting to Angel One streaming server...")
try:
    sws.connect()
except Exception as e:
    print(f"[!] Connection exception: {e}")
