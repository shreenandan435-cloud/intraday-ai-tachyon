import os
from dotenv import load_dotenv
import pyotp
from SmartApi import SmartConnect

load_dotenv()

API_KEY = os.getenv("SMARTAPI_API_KEY")
CLIENT_CODE = os.getenv("SMARTAPI_CLIENT_CODE")
PASSWORD = os.getenv("SMARTAPI_PASSWORD")
TOTP_SECRET = os.getenv("SMARTAPI_TOTP_SECRET")

print("[+] Reading credentials from .env...")
print(f"    • Client Code : {CLIENT_CODE}")
print(f"    • API Key     : {API_KEY[:6]}..." if API_KEY else "    • API Key     : Missing")

try:
    totp = pyotp.TOTP(TOTP_SECRET.strip()).now()
    print(f"    • Generated TOTP: {totp}")
    
    smart_api = SmartConnect(api_key=API_KEY)
    session = smart_api.generateSession(CLIENT_CODE, PASSWORD, totp)
    
    if session.get("status"):
        feed_token = smart_api.getfeedToken()
        print("[✓] Authentication SUCCESSFUL!")
        print(f"    • Feed Token  : {feed_token[:10]}... (active)")
    else:
        print(f"[!] Login Rejected by Angel One: {session.get('message')}")
except Exception as e:
    print(f"[!] Error: {e}")
