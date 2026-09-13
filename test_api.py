import os, pyotp, requests
from dotenv import load_dotenv
from SmartApi import SmartConnect

load_dotenv()
api_key = os.getenv("SMARTAPI_API_KEY")
client = SmartConnect(api_key=api_key)

print("Logging in...")
session = client.generateSession(
    os.getenv("SMARTAPI_CLIENT_CODE"), 
    os.getenv("SMARTAPI_PASSWORD"), 
    pyotp.TOTP(os.getenv("SMARTAPI_TOTP_SECRET")).now()
)

if not session.get("status"):
    print("Login Failed:", session)
    exit()

print("Login Success. Testing Data API...")
res = requests.post(
    "https://apiconnect.angelone.in/rest/secure/angelbroking/historical/v1/getCandleData",
    json={"exchange": "NSE", "symboltoken": "1190", "interval": "FIVE_MINUTE", "fromdate": "2026-09-01 09:15", "todate": "2026-09-04 09:15"},
    headers={
        "Authorization": f"Bearer {session['data']['jwtToken']}",
        "Content-Type": "application/json", "Accept": "application/json",
        "X-PrivateKey": api_key, "X-ClientLocalIP": "127.0.0.1", "X-ClientPublicIP": "106.193.147.98",
        "X-MACAddress": "50:bb:b5:cd:6a:db", "X-UserType": "USER", "X-SourceID": "WEB"
    }
).json()

print(f"Data Response: {res}")