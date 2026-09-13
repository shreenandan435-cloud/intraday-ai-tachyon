import os
import sys
import time
import signal
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
from dhan_harvester import DhanParquetHarvester
import websockets
import json

load_dotenv()

CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")

if not CLIENT_ID or not ACCESS_TOKEN:
    print("[!] ERROR: DHAN_CLIENT_ID or DHAN_ACCESS_TOKEN missing from .env")
    sys.exit(1)

WSS_URL = f"wss://depth-api-feed.dhan.co/twentydepth?token={ACCESS_TOKEN}&clientId={CLIENT_ID}&authType=2"

SYMBOLS = [
    {"ExchangeSegment": "NSE_EQ", "SecurityId": "11536"}, # TCS
    {"ExchangeSegment": "NSE_EQ", "SecurityId": "1594"},  # INFOSYS
    {"ExchangeSegment": "NSE_EQ", "SecurityId": "1333"}   # HDFCBANK
]

harvester = DhanParquetHarvester(output_dir="./market_lake/raw_l3")

def handle_exit(sig, frame):
    print("\n[!] Shutdown signal received. Flushing buffers to disk...")
    harvester.flush_to_disk()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

def wait_until_market_open():
    nse_holidays = {
        "2026-09-14", # Ganesh Chaturthi
        "2026-10-02", # Mahatma Gandhi Jayanti
        "2026-10-20", # Dussehra
        "2026-11-10", # Diwali-Balipratipada
        "2026-11-24", # Prakash Gurpurb
        "2026-12-25"  # Christmas
    }
    
    while True:
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        
        market_start = now.replace(hour=9, minute=14, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=35, second=0, microsecond=0)

        is_weekend = now.weekday() >= 5
        is_holiday = today_str in nse_holidays
        is_past_close = now >= market_close
        
        if is_weekend or is_holiday or is_past_close:
            tomorrow = now + timedelta(days=1)
            next_start = tomorrow.replace(hour=9, minute=14, second=0, microsecond=0)
            wait_seconds = int((next_start - now).total_seconds())
            
            status = "Weekend" if is_weekend else ("Holiday" if is_holiday else "Market Closed")
            print(f"[*] {status}. Hibernating until next check at {next_start.strftime('%Y-%m-%d %H:%M:%S')}...", end="\r")
            time.sleep(min(3600, max(1, wait_seconds)))
            continue
            
        if now >= market_start:
            print(f"\n[+] Market window active ({now.strftime('%H:%M:%S')}). Starting ingestion.")
            break
        else:
            wait_seconds = int((market_start - now).total_seconds())
            print(f"[*] Sleeping until 09:14 AM IST ({wait_seconds}s remaining)...", end="\r")
            time.sleep(min(30, max(1, wait_seconds)))

async def run_collector():
    # To test TONIGHT, comment out the next line by adding a '#' in front of it:
    wait_until_market_open()
    
    print(f"\n[+] Connecting to DhanHQ 20-Depth feed at: {WSS_URL[:45]}...")
    async with websockets.connect(WSS_URL, ping_interval=20, ping_timeout=20) as ws:
        print("[✓] WebSocket connected. Subscribing to target instruments...")
        sub_packet = {
            "RequestCode": 21,
            "InstrumentCount": len(SYMBOLS),
            "InstrumentList": SYMBOLS
        }
        await ws.send(json.dumps(sub_packet))
        print(f"[✓] Subscribed to {len(SYMBOLS)} symbols. Recording live depth...")

        last_status = time.time()
        packet_count = 0

        while True:
            now = datetime.now()
            if now.hour == 15 and now.minute >= 35:
                print("\n[+] Market closed (15:35). Initiating graceful shutdown...")
                break

            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                if isinstance(msg, bytes) and len(msg) >= 567:
                    harvester.ingest(msg)
                    packet_count += 1
            except asyncio.TimeoutError:
                pass # Expected timeout during weekends; loops safely back
                
            if time.time() - last_status >= 60:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Ingested {packet_count} packets. Buffer size: {len(harvester.buffer)}")
                last_status = time.time()

    harvester.flush_to_disk()
    print("[SUCCESS] All market depth data flushed to Parquet.")

if __name__ == "__main__":
    asyncio.run(run_collector())
