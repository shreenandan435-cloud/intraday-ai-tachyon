import os
import asyncio
import json
import websockets
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")

# DhanHQ 20-Level Depth WebSocket Endpoint
WSS_URL = f"wss://depth-api-feed.dhan.co/twentydepth?token={ACCESS_TOKEN}&clientId={CLIENT_ID}&authType=2"

async def test_handshake():
    print("[+] Connecting to DhanHQ 20-Depth WebSocket...")
    try:
        async with websockets.connect(WSS_URL) as ws:
            print("[✓] Connected successfully! Authentication passed.")
            
            # Subscribe to TCS (Security ID: 11536) on NSE Equity
            sub_request = {
                "RequestCode": 21,
                "InstrumentCount": 1,
                "InstrumentList": [
                    {
                        "ExchangeSegment": "NSE_EQ",
                        "SecurityId": "11536"
                    }
                ]
            }
            await ws.send(json.dumps(sub_request))
            print("[+] Sent subscription packet for TCS (11536). Waiting for binary response...")

            # Listen for the first 3 binary packets
            for i in range(3):
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    print(f"  -> Packet {i+1} received! Size: {len(msg)} bytes (Raw Binary Depth)")
                else:
                    print(f"  -> Text response: {msg}")

            print("\n[SUCCESS] DhanHQ API handshake and depth feed verified.")

    except Exception as e:
        print(f"\n[!] Connection failed: {e}")
        print("Check if the Data API subscription is active and the token hasn't expired.")

if __name__ == "__main__":
    asyncio.run(test_handshake())