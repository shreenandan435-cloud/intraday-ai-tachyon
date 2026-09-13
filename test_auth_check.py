import os
import pyotp
from SmartApi import SmartConnect

api_key = os.environ.get("SMARTAPI_API_KEY")
client_code = os.environ.get("SMARTAPI_CLIENT_CODE")
pwd = os.environ.get("SMARTAPI_PASSWORD")
totp_secret = os.environ.get("SMARTAPI_TOTP_SECRET")

try:
    smart_api = SmartConnect(api_key=api_key)
    totp = pyotp.TOTP(totp_secret).now()
    data = smart_api.generateSession(client_code, pwd, totp)
    if data['status']:
        print("[PASS] Angel One Authentication Successful.")
        print(f"[OK] Feed Token generated: {data['data']['feedToken'][:8]}...")
    else:
        print(f"[FAIL] Auth rejected: {data['message']}")
except Exception as e:
    print(f"[ERROR] Session generation crashed: {e}")
