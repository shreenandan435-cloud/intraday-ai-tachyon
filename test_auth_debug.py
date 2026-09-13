import os
import pyotp
from dotenv import load_dotenv
from SmartApi import SmartConnect

# Explicitly load variables from the local .env file
load_dotenv()

print("--- SMARTAPI CREDENTIAL CHECK (.env) ---")
api_key = os.environ.get("SMARTAPI_API_KEY", "").strip()
client_code = os.environ.get("SMARTAPI_CLIENT_CODE", "").strip()
pwd = os.environ.get("SMARTAPI_PASSWORD", "").strip()
totp_secret = os.environ.get("SMARTAPI_TOTP_SECRET", "").strip()
trading_mode = os.environ.get("TRADING_MODE", "").strip()

print(f"Trading Mode:    {trading_mode}")
print(f"API Key Length: {len(api_key)}")
print(f"Client Code:    {client_code}")
print(f"Password set:   {'YES' if pwd else 'NO'}")
print(f"TOTP Secret:    {'YES (' + totp_secret[:4] + '...)' if totp_secret else 'NO'}")

try:
    smart_api = SmartConnect(api_key=api_key)
    totp_code = pyotp.TOTP(totp_secret).now()
    print(f"Generated TOTP: {totp_code}")
    
    data = smart_api.generateSession(client_code, pwd, totp_code)
    print(f"API Raw Response: {data}")
    
    if data and isinstance(data, dict) and data.get('status'):
        print("[PASS] Session generated successfully from .env file!")
    else:
        print(f"[FAIL] Server rejection message: {data}")
except Exception as e:
    print(f"[ERROR] Session generation crashed with exception: {e}")
